// See GpuElecForces.h for the math and design rationale. Kernels here are
// unchanged from gpu/gemm/elecforces_test.cu, which validated this exact
// design against real fElec values from an actual nnp-train stage-2 run
// on temp/H2O_4G (max abs diff 1.1e-19, max rel diff 8.9e-14).

#include "GpuElecForces.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <cuda_runtime.h>

namespace
{

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Persistent per-structure device state -- the topology (dChidGOffset,
// ownerAtom, dGdrSelf, the edge list) is purely geometric and uploaded
// exactly once; dChidG/dAdrQ/pEelecpr/lambdaTotal/lambdaElec are
// re-uploaded on every gpuElecForcesCompute() call (see GpuElecForces.h's
// header comment for why).
struct GpuElecForcesState
{
    int numAtoms = 0;
    int numValues = 0;  // dChidGOffset[numAtoms]
    int numEdges = 0;
    int* d_dChidGOffset = nullptr;
    int* d_ownerAtom = nullptr;
    double* d_dGdrSelf = nullptr;
    int* d_edgeTarget = nullptr;
    int* d_edgeOwnerIndex = nullptr;
    double* d_edgeDGdr = nullptr;
    double* d_dChidG = nullptr;
    double* d_dAdrQ = nullptr;
    double* d_pEelecpr = nullptr;
    double* d_lambdaTotal = nullptr;
    double* d_lambdaElec = nullptr;
    double* d_weightedTotal = nullptr;
    double* d_weightedElec = nullptr;
    double* d_force = nullptr;
    double* d_forceElec = nullptr;
};

std::map<int, GpuElecForcesState>& getStates()
{
    static std::map<int, GpuElecForcesState> states;
    return states;
}

// weighted{Total,Elec}[v] = lambda{Total,Elec}[ownerAtom[v]] * dChidG[v].
__global__ void weightKernel(int numValues, int const* ownerAtom,
                              double const* dChidG,
                              double const* lambdaTotal,
                              double const* lambdaElec,
                              double* weightedTotal, double* weightedElec)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numValues) return;
    int const owner = ownerAtom[v];
    double const g = dChidG[v];
    weightedTotal[v] = lambdaTotal[owner] * g;
    weightedElec[v] = lambdaElec[owner] * g;
}

// One thread per atom: initializes force[i] = -pEelecpr[i] - self term
// (no cross-atom writes, no atomics needed).
__global__ void selfKernel(int numAtoms, int const* dChidGOffset,
                            double const* weightedTotal,
                            double const* weightedElec,
                            double const* dGdrSelf, double const* pEelecpr,
                            double* force, double* forceElec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    int const begin = dChidGOffset[i];
    int const end = dChidGOffset[i + 1];
    double tx = 0.0, ty = 0.0, tz = 0.0;
    double ex = 0.0, ey = 0.0, ez = 0.0;
    for (int k = begin; k < end; ++k)
    {
        double const dx = dGdrSelf[3 * k + 0];
        double const dy = dGdrSelf[3 * k + 1];
        double const dz = dGdrSelf[3 * k + 2];
        double const wt = weightedTotal[k];
        double const we = weightedElec[k];
        tx += wt * dx; ty += wt * dy; tz += wt * dz;
        ex += we * dx; ey += we * dy; ez += we * dz;
    }
    force[3 * i + 0] = -pEelecpr[3 * i + 0] - tx;
    force[3 * i + 1] = -pEelecpr[3 * i + 1] - ty;
    force[3 * i + 2] = -pEelecpr[3 * i + 2] - tz;
    forceElec[3 * i + 0] = -pEelecpr[3 * i + 0] - ex;
    forceElec[3 * i + 1] = -pEelecpr[3 * i + 1] - ey;
    forceElec[3 * i + 2] = -pEelecpr[3 * i + 2] - ez;
}

// One thread per atom: adds the dense -sum_j lambda(j)*dAdrQ[i][j] term.
// Runs after selfKernel (sequential launches on the default stream), no
// atomics needed -- disjoint i across threads.
__global__ void denseDAdrQKernel(int numAtoms, double const* dAdrQ,
                                  double const* lambdaTotal,
                                  double const* lambdaElec,
                                  double* force, double* forceElec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    double tx = 0.0, ty = 0.0, tz = 0.0;
    double ex = 0.0, ey = 0.0, ez = 0.0;
    for (int j = 0; j < numAtoms; ++j)
    {
        size_t const idx = 3 * ((size_t)i * numAtoms + j);
        double const dx = dAdrQ[idx + 0];
        double const dy = dAdrQ[idx + 1];
        double const dz = dAdrQ[idx + 2];
        double const lt = lambdaTotal[j];
        double const le = lambdaElec[j];
        tx += lt * dx; ty += lt * dy; tz += lt * dz;
        ex += le * dx; ey += le * dy; ez += le * dz;
    }
    force[3 * i + 0] -= tx;
    force[3 * i + 1] -= ty;
    force[3 * i + 2] -= tz;
    forceElec[3 * i + 0] -= ex;
    forceElec[3 * i + 1] -= ey;
    forceElec[3 * i + 2] -= ez;
}

// One thread per edge: scatter-add into the TARGET atom's accumulators.
// Many edges can target the same atom concurrently, hence atomicAdd.
__global__ void edgeKernel(int numEdges, int const* edgeTarget,
                            int const* edgeOwnerIndex, double const* edgeDGdr,
                            double const* weightedTotal,
                            double const* weightedElec,
                            double* force, double* forceElec)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numEdges) return;

    int const target = edgeTarget[e];
    int const owner = edgeOwnerIndex[e];
    double const dx = edgeDGdr[3 * e + 0];
    double const dy = edgeDGdr[3 * e + 1];
    double const dz = edgeDGdr[3 * e + 2];
    double const wt = weightedTotal[owner];
    double const we = weightedElec[owner];
    atomicAdd(&force[3 * target + 0], -wt * dx);
    atomicAdd(&force[3 * target + 1], -wt * dy);
    atomicAdd(&force[3 * target + 2], -wt * dz);
    atomicAdd(&forceElec[3 * target + 0], -we * dx);
    atomicAdd(&forceElec[3 * target + 1], -we * dy);
    atomicAdd(&forceElec[3 * target + 2], -we * dz);
}

} // anonymous namespace

namespace nnp
{

void gpuElecForcesUploadTopology(int structureId,
                                 int numAtoms,
                                 int const* dChidGOffset,
                                 double const* dGdrSelf,
                                 int numEdges,
                                 int const* edgeTarget,
                                 int const* edgeOwnerIndex,
                                 double const* edgeDGdr)
{
    GpuElecForcesState& s = getStates()[structureId];
    s.numAtoms = numAtoms;
    s.numValues = dChidGOffset[numAtoms];
    s.numEdges = numEdges;

    // Build the per-value owner-atom map once here (purely derived from
    // dChidGOffset, same lifetime as the rest of the topology).
    std::vector<int> ownerAtom(s.numValues);
    for (int i = 0; i < numAtoms; ++i)
    {
        for (int v = dChidGOffset[i]; v < dChidGOffset[i + 1]; ++v)
        {
            ownerAtom[v] = i;
        }
    }

    CUDA_CHECK(cudaMalloc(&s.d_dChidGOffset, (size_t)(numAtoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_ownerAtom, (size_t)s.numValues * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_dGdrSelf, (size_t)s.numValues * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_edgeTarget, (size_t)numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_edgeOwnerIndex, (size_t)numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_edgeDGdr, (size_t)numEdges * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dChidG, (size_t)s.numValues * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dAdrQ,
                         (size_t)numAtoms * numAtoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_pEelecpr, (size_t)numAtoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_lambdaTotal, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_lambdaElec, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_weightedTotal, (size_t)s.numValues * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_weightedElec, (size_t)s.numValues * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_force, (size_t)numAtoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_forceElec, (size_t)numAtoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(s.d_dChidGOffset, dChidGOffset,
                         (size_t)(numAtoms + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_ownerAtom, ownerAtom.data(),
                         (size_t)s.numValues * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_dGdrSelf, dGdrSelf,
                         (size_t)s.numValues * 3 * sizeof(double),
                         cudaMemcpyHostToDevice));
    if (numEdges > 0)
    {
        CUDA_CHECK(cudaMemcpy(s.d_edgeTarget, edgeTarget,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeOwnerIndex, edgeOwnerIndex,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeDGdr, edgeDGdr,
                             (size_t)numEdges * 3 * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
}

void gpuElecForcesCompute(int structureId,
                          double const* dChidG,
                          double const* dAdrQ,
                          double const* pEelecpr,
                          double const* lambdaTotal,
                          double const* lambdaElec,
                          double* force,
                          double* forceElec)
{
    GpuElecForcesState& s = getStates().at(structureId);

    CUDA_CHECK(cudaMemcpy(s.d_dChidG, dChidG,
                         (size_t)s.numValues * sizeof(double),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_dAdrQ, dAdrQ,
                         (size_t)s.numAtoms * s.numAtoms * 3 * sizeof(double),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_pEelecpr, pEelecpr,
                         (size_t)s.numAtoms * 3 * sizeof(double),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_lambdaTotal, lambdaTotal,
                         (size_t)s.numAtoms * sizeof(double),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_lambdaElec, lambdaElec,
                         (size_t)s.numAtoms * sizeof(double),
                         cudaMemcpyHostToDevice));

    int const block = 128;

    weightKernel<<<(s.numValues + block - 1) / block, block>>>(
        s.numValues, s.d_ownerAtom, s.d_dChidG, s.d_lambdaTotal,
        s.d_lambdaElec, s.d_weightedTotal, s.d_weightedElec);
    CUDA_CHECK(cudaGetLastError());

    selfKernel<<<(s.numAtoms + block - 1) / block, block>>>(
        s.numAtoms, s.d_dChidGOffset, s.d_weightedTotal, s.d_weightedElec,
        s.d_dGdrSelf, s.d_pEelecpr, s.d_force, s.d_forceElec);
    CUDA_CHECK(cudaGetLastError());

    denseDAdrQKernel<<<(s.numAtoms + block - 1) / block, block>>>(
        s.numAtoms, s.d_dAdrQ, s.d_lambdaTotal, s.d_lambdaElec, s.d_force,
        s.d_forceElec);
    CUDA_CHECK(cudaGetLastError());

    if (s.numEdges > 0)
    {
        edgeKernel<<<(s.numEdges + block - 1) / block, block>>>(
            s.numEdges, s.d_edgeTarget, s.d_edgeOwnerIndex, s.d_edgeDGdr,
            s.d_weightedTotal, s.d_weightedElec, s.d_force, s.d_forceElec);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(force, s.d_force,
                         (size_t)s.numAtoms * 3 * sizeof(double),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forceElec, s.d_forceElec,
                         (size_t)s.numAtoms * 3 * sizeof(double),
                         cudaMemcpyDeviceToHost));
}

} // namespace nnp
