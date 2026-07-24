// Phase 1 step 3: persistent per-atom SF storage (AtomBatch::G) written by a
// "grouped" GPU kernel, mirroring src/libnnp/SymGrpExpRad.cpp's actual
// optimization -- multiple symmetry functions that share the same element
// filter/cutoff radius evaluate the cutoff function ONCE per neighbor and
// reuse it for every group member (SymGrpExpRad::calculate(), read closely
// for this), rather than each SymFnc re-walking the neighbor list and
// recomputing the cutoff independently (which is what step 2's kernel did,
// one symmetry function at a time).
//
// This is also the first kernel to WRITE into AtomBatch rather than just
// read it: previous steps returned results via throwaway per-call output
// buffers. Here two ExpRad instances (H central atom, e1=H neighbor filter,
// the same two real eta/rs pairs from temp/H2O_2G/input.nn used in step 2)
// are evaluated together and written into AtomBatch::G's per-element block
// layout (allocateSfStorage()), at the flat index AtomBatch::gIndex(s, k)
// would compute (recomputed manually device-side since gIndex() is a plain
// host method, not a device function -- see the kernel for the arithmetic).
//
// Validates GPU-written G values against a CPU reference computed with the
// same grouped structure, and (implicitly, since the math and inputs are
// identical to step 2's) against ../smoke/symfnc_family_test.cu's original
// per-type validation.

#include "AtomBatch.h"
#include "ElementMap.h"
#include "Structure.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

using namespace nnp;
using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__host__ __device__ inline void cutoffTANHU(double r, double rcinv,
                                             double& fc, double& dfc)
{
    double t  = tanh(1.0 - r * rcinv);
    double t2 = t * t;
    fc  = t * t2;
    dfc = 3.0 * t2 * (t2 - 1.0) * rcinv;
}

// Grouped evaluation: one neighbor loop, cutoff computed once per neighbor,
// reused across all `numMembers` (eta[k], rs[k]) pairs -- SymGrpExpRad.cpp's
// actual structure (result[k] accumulator per member, shared pfc/pdfc).
__host__ __device__ inline void symFncExpRadGroup(
    int numNeighbors, const int* neighElem,
    const double* neighDist, int e1, double rc,
    int numMembers, const double* eta, const double* rs,
    double* result /* size numMembers, pre-zeroed by caller */)
{
    double rcinv = 1.0 / rc;
    for (int j = 0; j < numNeighbors; ++j)
    {
        if (neighElem[j] != e1) continue;
        double rij = neighDist[j];
        if (rij >= rc) continue;
        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);
        for (int k = 0; k < numMembers; ++k)
        {
            double diff = rij - rs[k];
            double pexp = exp(-eta[k] * diff * diff);
            result[k] += pexp * pfc;
        }
    }
}

__global__ void sfGroupKernel(
    int begin, int numSelected, int e1, double rc,
    int numMembers, const double* eta, const double* rs,
    const int* neighOffsetInt, const int* neighElem, const double* neighDist,
    double* G, size_t gBlockOffsetE, size_t sfCountE)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    int i = begin + t;
    int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;

    double result[8]; // numMembers is small (2 here); plenty of headroom
    for (int k = 0; k < numMembers; ++k) result[k] = 0.0;

    symFncExpRadGroup(n, &neighElem[off], &neighDist[off], e1, rc,
                       numMembers, eta, rs, result);

    size_t base = gBlockOffsetE + (size_t)t * sfCountE;
    for (int k = 0; k < numMembers; ++k) G[base + k] = result[k];
}

int main(int argc, char** argv)
{
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    printf("CUDA devices visible: %d\n", devCount);
    if (devCount == 0) { fprintf(stderr, "No CUDA device.\n"); return 1; }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device 0: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    string inputData = (argc > 1) ? argv[1] : "input.data";
    double const rc = 12.0;

    ElementMap elementMap;
    elementMap.registerElements("H O");
    Structure structure;
    structure.setElementMap(elementMap);
    structure.readFromFile(inputData);
    structure.calculateNeighborList(rc);

    AtomBatch batch = buildAtomBatch(structure, rc);
    int const H = 0, O = 1;

    // H atoms get 2 symmetry functions (the two real ExpRad instances from
    // input.nn); O atoms get 0 -- mirrors how input.nn assigns a separate,
    // independently-sized symmetry function list per central element.
    allocateSfStorage(batch, {2, 0});
    printf("AtomBatch: %zu atoms, G storage: %zu values (H block: %zu atoms x %zu SFs)\n\n",
           batch.numAtoms, batch.G.size(),
           batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.sfCountPerElement[H]);

    vector<double> eta = {0.001, 0.15};
    vector<double> rs  = {0.0,   1.9124};
    int const numMembers = (int)eta.size();

    int begin = (int)batch.elementOffset[H];
    int end   = (int)batch.elementOffset[H + 1];
    int numSelected = end - begin;

    vector<int> neighOffsetInt(batch.neighborOffset.begin(), batch.neighborOffset.end());
    vector<int> neighElemInt(batch.neighborElement.begin(), batch.neighborElement.end());

    // --- CPU reference, same grouped structure -----------------------------
    vector<double> Gcpu(batch.G.size(), 0.0);
    for (int t = 0; t < numSelected; ++t)
    {
        int i = begin + t;
        int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
        double result[8] = {0.0, 0.0};
        symFncExpRadGroup(n, &neighElemInt[off], &batch.neighborD[off], H, rc,
                           numMembers, eta.data(), rs.data(), result);
        size_t base = batch.gBlockOffset[H] + (size_t)t * batch.sfCountPerElement[H];
        for (int k = 0; k < numMembers; ++k) Gcpu[base + k] = result[k];
        // Cross-check against AtomBatch::gIndex()'s own index arithmetic.
        for (int k = 0; k < numMembers; ++k)
        {
            if (base + k != batch.gIndex(i, k))
            {
                fprintf(stderr, "gIndex mismatch at atom %d, sf %d\n", i, k);
                return 1;
            }
        }
    }

    // --- GPU: write directly into batch.G's device mirror ------------------
    int total = (int)batch.neighborD.size();
    int *d_neighOffset, *d_neighElem;
    double *d_neighDist, *d_eta, *d_rs, *d_G;

    CUDA_CHECK(cudaMalloc(&d_neighOffset, neighOffsetInt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighElem, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighDist, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eta, numMembers * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rs, numMembers * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, batch.G.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_neighOffset, neighOffsetInt.data(), neighOffsetInt.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighElem, neighElemInt.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDist, batch.neighborD.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta, eta.data(), numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rs, rs.data(), numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_G, 0, batch.G.size() * sizeof(double)));

    int blockSize = 128;
    int gridSize = (numSelected + blockSize - 1) / blockSize;
    sfGroupKernel<<<gridSize, blockSize>>>(begin, numSelected, H, rc,
        numMembers, d_eta, d_rs, d_neighOffset, d_neighElem, d_neighDist,
        d_G, batch.gBlockOffset[H], batch.sfCountPerElement[H]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> Ggpu(batch.G.size());
    CUDA_CHECK(cudaMemcpy(Ggpu.data(), d_G, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_neighOffset); cudaFree(d_neighElem); cudaFree(d_neighDist);
    cudaFree(d_eta); cudaFree(d_rs); cudaFree(d_G);

    double maxAbsErr = 0.0;
    for (size_t k = 0; k < Gcpu.size(); ++k)
        maxAbsErr = max(maxAbsErr, fabs(Ggpu[k] - Gcpu[k]));

    printf("G[H atom 0] = (%.15E, %.15E)  [cpu: (%.15E, %.15E)]\n",
           Ggpu[batch.gIndex(begin, 0)], Ggpu[batch.gIndex(begin, 1)],
           Gcpu[batch.gIndex(begin, 0)], Gcpu[batch.gIndex(begin, 1)]);
    printf("max|G_gpu-G_cpu| over all %zu stored values = %.3E\n",
           Gcpu.size(), maxAbsErr);

    bool pass = maxAbsErr < 1e-9;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
