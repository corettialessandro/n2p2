// See GpuForces.h for the math and design rationale.

#include "GpuForces.h"

#include <cstdio>
#include <cstdlib>
#include <map>
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

// Persistent per-structure device state -- the topology (dEdGOffset,
// dGdrSelf, the whole edge list) is purely geometric and uploaded
// exactly once; only d_dEdG is re-uploaded on every gpuForcesCompute()
// call (see GpuForces.h's header comment for why).
struct GpuForcesState
{
    int numAtoms = 0;
    int numValues = 0;  // dEdGOffset[numAtoms]
    int numEdges = 0;
    int* d_dEdGOffset = nullptr;
    double* d_dGdrSelf = nullptr;
    int* d_edgeTarget = nullptr;
    int* d_edgeOwnerDEdGIndex = nullptr;
    double* d_edgeDGdr = nullptr;
    double* d_dEdG = nullptr;
    double* d_force = nullptr;
};

std::map<int, GpuForcesState>& getStates()
{
    static std::map<int, GpuForcesState> states;
    return states;
}

// One thread per atom: F_i = -sum_k dEdG_i[k] * dGdrSelf_i[k].
// No cross-atom writes, so no atomics needed.
__global__ void selfForceKernel(int numAtoms,
                                 int const* dEdGOffset,
                                 double const* dEdG,
                                 double const* dGdrSelf,
                                 double* force)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    int const begin = dEdGOffset[i];
    int const end = dEdGOffset[i + 1];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int k = begin; k < end; ++k)
    {
        double const d = dEdG[k];
        fx -= d * dGdrSelf[3 * k + 0];
        fy -= d * dGdrSelf[3 * k + 1];
        fz -= d * dGdrSelf[3 * k + 2];
    }
    force[3 * i + 0] = fx;
    force[3 * i + 1] = fy;
    force[3 * i + 2] = fz;
}

// One thread per edge: scatter-add -dEdG[owner index] * dGdr into the
// TARGET atom's force accumulator. Many edges can target the same atom
// concurrently (every neighbor of that atom contributes one edge each),
// hence atomicAdd.
__global__ void pairForceKernel(int numEdges,
                                 int const* edgeTarget,
                                 int const* edgeOwnerDEdGIndex,
                                 double const* edgeDGdr,
                                 double const* dEdG,
                                 double* force)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numEdges) return;

    int const target = edgeTarget[e];
    double const d = dEdG[edgeOwnerDEdGIndex[e]];
    atomicAdd(&force[3 * target + 0], -d * edgeDGdr[3 * e + 0]);
    atomicAdd(&force[3 * target + 1], -d * edgeDGdr[3 * e + 1]);
    atomicAdd(&force[3 * target + 2], -d * edgeDGdr[3 * e + 2]);
}

} // anonymous namespace

namespace nnp
{

void gpuForcesUploadTopology(int structureId,
                             int numAtoms,
                             int const* dEdGOffset,
                             double const* dGdrSelf,
                             int numEdges,
                             int const* edgeTarget,
                             int const* edgeOwnerDEdGIndex,
                             double const* edgeDGdr)
{
    GpuForcesState& s = getStates()[structureId];
    s.numAtoms = numAtoms;
    s.numValues = dEdGOffset[numAtoms];
    s.numEdges = numEdges;

    CUDA_CHECK(cudaMalloc(&s.d_dEdGOffset, (size_t)(numAtoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_dGdrSelf, (size_t)s.numValues * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_edgeTarget, (size_t)numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_edgeOwnerDEdGIndex,
                         (size_t)numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_edgeDGdr, (size_t)numEdges * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dEdG, (size_t)s.numValues * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_force, (size_t)numAtoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(s.d_dEdGOffset, dEdGOffset,
                         (size_t)(numAtoms + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_dGdrSelf, dGdrSelf,
                         (size_t)s.numValues * 3 * sizeof(double),
                         cudaMemcpyHostToDevice));
    if (numEdges > 0)
    {
        CUDA_CHECK(cudaMemcpy(s.d_edgeTarget, edgeTarget,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeOwnerDEdGIndex, edgeOwnerDEdGIndex,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeDGdr, edgeDGdr,
                             (size_t)numEdges * 3 * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
}

void gpuForcesCompute(int structureId,
                      double const* dEdG,
                      double* force)
{
    GpuForcesState& s = getStates().at(structureId);

    CUDA_CHECK(cudaMemcpy(s.d_dEdG, dEdG,
                         (size_t)s.numValues * sizeof(double),
                         cudaMemcpyHostToDevice));

    int const block = 128;
    selfForceKernel<<<(s.numAtoms + block - 1) / block, block>>>(
        s.numAtoms, s.d_dEdGOffset, s.d_dEdG, s.d_dGdrSelf, s.d_force);
    CUDA_CHECK(cudaGetLastError());

    if (s.numEdges > 0)
    {
        pairForceKernel<<<(s.numEdges + block - 1) / block, block>>>(
            s.numEdges, s.d_edgeTarget, s.d_edgeOwnerDEdGIndex, s.d_edgeDGdr,
            s.d_dEdG, s.d_force);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(force, s.d_force,
                         (size_t)s.numAtoms * 3 * sizeof(double),
                         cudaMemcpyDeviceToHost));
}

} // namespace nnp
