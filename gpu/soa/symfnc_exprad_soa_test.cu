// Phase 1 step 2: prove AtomBatch (AtomBatch.h/.cpp) is directly consumable
// by a GPU kernel, not just a correct host-side re-encoding (build_batch_
// test.cpp proved that part already).
//
// Wires SymFncExpRad's CUDA kernel (first proven in ../smoke/symfnc_family_
// test.cu, same math, copied verbatim from there -- cutoffTANHU +
// symFncExpRad) directly to a real AtomBatch instead of gpu/smoke's
// loadRealSystem() (which parses a dumped text file and has to linearly
// scan for atoms of the right element). Two things the SoA/CSR layout
// buys for free, demonstrated here:
//   - Central-atom selection is a contiguous slice
//     [elementOffset[ec], elementOffset[ec+1]) of already-element-sorted
//     atoms -- no scan, no selectedAtoms index array, unlike
//     symfnc_family_test.cu's runCase().
//   - The kernel launch operates directly on that sorted range; thread t
//     maps to sorted atom index (begin + t).
//
// Loads the real H2O_2G structure via libnnp.a's own ElementMap/Structure
// (same as dump_real_neighbors.cpp), builds one AtomBatch, and validates
// GPU vs. a CPU reference computed from the SAME AtomBatch arrays (so this
// specifically checks kernel-vs-CSR-layout correctness; build_batch_test.cpp
// already checked CSR-layout-vs-Structure correctness -- together the two
// tests chain to GPU-vs-Structure).

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

// --- Copied verbatim from ../smoke/symfnc_family_test.cu (already validated
// there against SymFncExpRad::calculate()/CutoffFunction.cpp) -----------

__host__ __device__ inline void cutoffTANHU(double r, double rcinv,
                                             double& fc, double& dfc)
{
    double t  = tanh(1.0 - r * rcinv);
    double t2 = t * t;
    fc  = t * t2;
    dfc = 3.0 * t2 * (t2 - 1.0) * rcinv;
}

__host__ __device__ inline void symFncExpRad(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, double eta, double rs, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double rcinv = 1.0 / rc;
    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < numNeighbors; ++j)
    {
        if (neighElem[j] != e1) continue;
        double rij = neighDist[j];
        if (rij >= rc) continue;
        double diff = rij - rs;
        double pexp = exp(-eta * diff * diff);
        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);
        result += pexp * pfc;
        double p1 = (pdfc - 2.0 * eta * diff * pfc) * pexp / rij;
        fx += p1 * neighDx[j]; fy += p1 * neighDy[j]; fz += p1 * neighDz[j];
    }
    G = result; dGdx = fx; dGdy = fy; dGdz = fz;
}

__global__ void sfKernel(
    int begin, int numSelected, int e1, double eta, double rs, double rc,
    const int* neighOffsetInt, const int* neighElem, const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double* Gout, double* dGdxOut, double* dGdyOut, double* dGdzOut)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    int i = begin + t; // sorted atom index -- contiguous, no gather needed
    int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
    symFncExpRad(n, &neighElem[off], &neighDist[off], &neighDx[off],
                 &neighDy[off], &neighDz[off], e1, eta, rs, rc,
                 Gout[t], dGdxOut[t], dGdyOut[t], dGdzOut[t]);
}

bool runCase(char const* label, AtomBatch const& batch, int ec, int e1,
             double eta, double rs, double rc,
             vector<int> const& neighOffsetInt, vector<int> const& neighElemInt,
             vector<double> const& neighDist, vector<double> const& neighDx,
             vector<double> const& neighDy, vector<double> const& neighDz)
{
    printf("--- %s ---\n", label);

    int begin = (int)batch.elementOffset[ec];
    int end   = (int)batch.elementOffset[ec + 1];
    int numSelected = end - begin;

    vector<double> Gcpu(numSelected), dxCpu(numSelected), dyCpu(numSelected),
                   dzCpu(numSelected);
    for (int t = 0; t < numSelected; ++t)
    {
        int i = begin + t;
        int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
        symFncExpRad(n, &neighElemInt[off], &neighDist[off], &neighDx[off],
                     &neighDy[off], &neighDz[off], e1, eta, rs, rc,
                     Gcpu[t], dxCpu[t], dyCpu[t], dzCpu[t]);
    }

    int total = (int)neighDist.size();
    int *d_neighOffset, *d_neighElem;
    double *d_neighDist, *d_neighDx, *d_neighDy, *d_neighDz;
    double *d_G, *d_dx, *d_dy, *d_dz;

    CUDA_CHECK(cudaMalloc(&d_neighOffset, neighOffsetInt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighElem, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighDist, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDx, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDy, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDz, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, numSelected * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dx, numSelected * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dy, numSelected * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dz, numSelected * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_neighOffset, neighOffsetInt.data(),
        neighOffsetInt.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighElem, neighElemInt.data(),
        total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDist, neighDist.data(),
        total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDx, neighDx.data(),
        total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDy, neighDy.data(),
        total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDz, neighDz.data(),
        total * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int gridSize = (numSelected + blockSize - 1) / blockSize;
    sfKernel<<<gridSize, blockSize>>>(begin, numSelected, e1, eta, rs, rc,
        d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy,
        d_neighDz, d_G, d_dx, d_dy, d_dz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> Ggpu(numSelected), dxGpu(numSelected), dyGpu(numSelected),
                   dzGpu(numSelected);
    CUDA_CHECK(cudaMemcpy(Ggpu.data(), d_G, numSelected * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dxGpu.data(), d_dx, numSelected * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dyGpu.data(), d_dy, numSelected * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dzGpu.data(), d_dz, numSelected * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_neighOffset); cudaFree(d_neighElem); cudaFree(d_neighDist);
    cudaFree(d_neighDx); cudaFree(d_neighDy); cudaFree(d_neighDz);
    cudaFree(d_G); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_dz);

    double maxAbsErrG = 0.0, maxAbsErrD = 0.0;
    for (int t = 0; t < numSelected; ++t)
    {
        maxAbsErrG = max(maxAbsErrG, fabs(Ggpu[t] - Gcpu[t]));
        maxAbsErrD = max({maxAbsErrD, fabs(dxGpu[t] - dxCpu[t]),
                           fabs(dyGpu[t] - dyCpu[t]), fabs(dzGpu[t] - dzCpu[t])});
    }

    printf("  selected atoms=%d (elementOffset[%d..%d), no scan needed)\n",
           numSelected, begin, end);
    printf("  max|G_gpu-G_cpu|=%.3E  max|dG_gpu-dG_cpu|=%.3E\n",
           maxAbsErrG, maxAbsErrD);

    bool pass = (maxAbsErrG < 1e-9) && (maxAbsErrD < 1e-9);
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
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
    printf("AtomBatch: %zu atoms, %zu elements, %zu neighbor entries\n\n",
           batch.numAtoms, batch.numElements, batch.totalNeighbors());

    // Device buffers want int, AtomBatch stores size_t -- narrow once here.
    vector<int> neighOffsetInt(batch.neighborOffset.begin(), batch.neighborOffset.end());
    vector<int> neighElemInt(batch.neighborElement.begin(), batch.neighborElement.end());

    int const H = 0;
    bool ok = true;
    ok &= runCase("ExpRad, real params (H 2 H 0.001 0.0 12.00), via AtomBatch",
                  batch, H, H, 0.001, 0.0, rc, neighOffsetInt, neighElemInt,
                  batch.neighborD, batch.neighborDx, batch.neighborDy, batch.neighborDz);
    ok &= runCase("ExpRad, real params (H 2 H 0.15 1.9124 12.00), via AtomBatch",
                  batch, H, H, 0.15, 1.9124, rc, neighOffsetInt, neighElemInt,
                  batch.neighborD, batch.neighborDx, batch.neighborDy, batch.neighborDz);

    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
