// See GpuEwaldReal.h for the math and design rationale.

#include "GpuEwaldReal.h"

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

// Persistent per-structure device state -- the edge list is purely
// geometric and uploaded exactly once; only d_gammaSqrt2 (tiny) is
// re-uploaded on every gpuEwaldRealCompute() call (see GpuEwaldReal.h's
// header comment for why).
struct GpuEwaldRealState
{
    int numAtoms = 0;
    int numElements = 0;
    int numEdges = 0;
    int* d_edgeOwner = nullptr;
    int* d_edgeTarget = nullptr;
    double* d_edgeRij = nullptr;
    int* d_edgeElemI = nullptr;
    int* d_edgeElemJ = nullptr;
    double* d_gammaSqrt2 = nullptr;
    double* d_AUpperAdd = nullptr;
};

std::map<int, GpuEwaldRealState>& getStates()
{
    static std::map<int, GpuEwaldRealState> states;
    return states;
}

// One thread per edge: scatter-add the real-space erfc contribution
// into the (owner, target) slot of the dense output buffer. Multiple
// edges (periodic images of the same neighbor) can target the same
// (i, j) slot, hence atomicAdd.
__global__ void ewaldRealKernel(int numEdges,
                                 int const* edgeOwner,
                                 int const* edgeTarget,
                                 double const* edgeRij,
                                 int const* edgeElemI,
                                 int const* edgeElemJ,
                                 double const* gammaSqrt2,
                                 int numElements,
                                 double sqrt2eta,
                                 double fourPiEps,
                                 int numAtoms,
                                 double* AUpperAdd)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numEdges) return;

    int const i = edgeOwner[e];
    int const j = edgeTarget[e];
    double const rij = edgeRij[e];
    int const ei = edgeElemI[e];
    int const ej = edgeElemJ[e];

    double const erfcSqrt2Eta = erfc(rij / sqrt2eta);
    double const erfcGammaSqrt2 = erfc(rij / gammaSqrt2[ei * numElements + ej]);
    double const contrib = (erfcSqrt2Eta - erfcGammaSqrt2) / (rij * fourPiEps);

    atomicAdd(&AUpperAdd[(size_t)i * numAtoms + j], contrib);
}

} // anonymous namespace

namespace nnp
{

void gpuEwaldRealUploadTopology(int structureId,
                                int numAtoms,
                                int numElements,
                                int numEdges,
                                int const* edgeOwner,
                                int const* edgeTarget,
                                double const* edgeRij,
                                int const* edgeElemI,
                                int const* edgeElemJ)
{
    GpuEwaldRealState& s = getStates()[structureId];
    s.numAtoms = numAtoms;
    s.numElements = numElements;
    s.numEdges = numEdges;

    CUDA_CHECK(cudaMalloc(&s.d_AUpperAdd,
                         (size_t)numAtoms * numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_gammaSqrt2,
                         (size_t)numElements * numElements * sizeof(double)));

    if (numEdges > 0)
    {
        CUDA_CHECK(cudaMalloc(&s.d_edgeOwner, (size_t)numEdges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.d_edgeTarget, (size_t)numEdges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.d_edgeRij, (size_t)numEdges * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.d_edgeElemI, (size_t)numEdges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.d_edgeElemJ, (size_t)numEdges * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(s.d_edgeOwner, edgeOwner,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeTarget, edgeTarget,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeRij, edgeRij,
                             (size_t)numEdges * sizeof(double),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeElemI, edgeElemI,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_edgeElemJ, edgeElemJ,
                             (size_t)numEdges * sizeof(int),
                             cudaMemcpyHostToDevice));
    }
}

void gpuEwaldRealCompute(int structureId,
                         double const* gammaSqrt2,
                         double sqrt2eta,
                         double fourPiEps,
                         double* AUpperAdd)
{
    GpuEwaldRealState& s = getStates().at(structureId);

    CUDA_CHECK(cudaMemset(s.d_AUpperAdd, 0,
                         (size_t)s.numAtoms * s.numAtoms * sizeof(double)));

    if (s.numEdges > 0)
    {
        CUDA_CHECK(cudaMemcpy(s.d_gammaSqrt2, gammaSqrt2,
                             (size_t)s.numElements * s.numElements
                                 * sizeof(double),
                             cudaMemcpyHostToDevice));

        int const block = 128;
        ewaldRealKernel<<<(s.numEdges + block - 1) / block, block>>>(
            s.numEdges, s.d_edgeOwner, s.d_edgeTarget, s.d_edgeRij,
            s.d_edgeElemI, s.d_edgeElemJ, s.d_gammaSqrt2, s.numElements,
            sqrt2eta, fourPiEps, s.numAtoms, s.d_AUpperAdd);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(AUpperAdd, s.d_AUpperAdd,
                         (size_t)s.numAtoms * s.numAtoms * sizeof(double),
                         cudaMemcpyDeviceToHost));
}

} // namespace nnp
