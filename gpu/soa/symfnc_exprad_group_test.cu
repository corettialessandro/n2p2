// Phase 1 steps 3-4: persistent per-atom SF storage (AtomBatch::G/dGdx,Dy,Dz
// and AtomBatch::neighborDGdx,Dy,Dz) written by a "grouped" GPU kernel,
// mirroring src/libnnp/SymGrpExpRad.cpp's actual optimization -- multiple
// symmetry functions that share the same element filter/cutoff radius
// evaluate the cutoff function ONCE per neighbor and reuse it for every
// group member (SymGrpExpRad::calculate(), read closely for this), rather
// than each SymFnc re-walking the neighbor list and recomputing the cutoff
// independently (what step 2's kernel did, one symmetry function at a time).
//
// Step 3 (G only) is superseded in place here by step 4, which adds both
// derivative storages SymFncExpRad.cpp actually produces per matching
// neighbor j (`double const p1 = ...; Vec3D dij = p1 * n.dr;`):
//   - `atom.dGdr[index] += dij`  -- summed over all matching neighbors, this
//     atom's own contribution to its own force. Stored in AtomBatch::dGdx,
//     Dy,Dz, same per-element block layout/indexing as G (gIndex()).
//   - `n.dGdr[...] -= dij`       -- one entry PER NEIGHBOR SLOT, not summed;
//     this is what a later force-assembly kernel scatter-adds onto that
//     neighbor atom's own force (Training::collectDGdxia /
//     calculatePairForceShort). Stored in AtomBatch::neighborDGdx,Dy,Dz,
//     addressed via neighborSfIndex(). This is the piece gpu/smoke's
//     original per-type tests explicitly left out of scope
//     ("Neighbor-side derivative bookkeeping is out of scope") -- closing
//     that gap is the point of this step.
//
// Two ExpRad instances (H central atom, e1=H neighbor filter, the same two
// real eta/rs pairs from temp/H2O_2G/input.nn used in steps 1-2) are
// evaluated together per H atom. Validated against a CPU reference built
// with the same grouped structure, for G, the owner-atom derivative, AND
// every individual neighbor-slot derivative.

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
// actual structure. Writes both derivative storages described above:
// dResultX/Y/Z accumulate the owner atom's own derivative (summed over
// neighbors), neighDGdx/Dy/Dz get one entry per (neighbor slot j, member k)
// -- row-major [j*numMembers+k], not summed -- equal to -dij, matching
// SymFncExpRad.cpp's `n.dGdr[...] -= dij`. Non-matching neighbor slots are
// left untouched (assumed pre-zeroed by the caller), since that neighbor
// contributes nothing to this SF and its derivative is exactly zero.
__host__ __device__ inline void symFncExpRadGroup(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, double rc,
    int numMembers, const double* eta, const double* rs,
    double* result, double* dResultX, double* dResultY, double* dResultZ,
    double* neighDGdx, double* neighDGdy, double* neighDGdz)
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
            double p1 = (pdfc - 2.0 * eta[k] * diff * pfc) * pexp / rij;
            double dijx = p1 * neighDx[j];
            double dijy = p1 * neighDy[j];
            double dijz = p1 * neighDz[j];
            dResultX[k] += dijx; dResultY[k] += dijy; dResultZ[k] += dijz;
            int idx = j * numMembers + k;
            neighDGdx[idx] = -dijx;
            neighDGdy[idx] = -dijy;
            neighDGdz[idx] = -dijz;
        }
    }
}

__global__ void sfGroupKernel(
    int begin, int numSelected, int e1, double rc,
    int numMembers, const double* eta, const double* rs,
    const int* neighOffsetInt, const int* neighElem, const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double* G, double* dGdx, double* dGdy, double* dGdz,
    size_t gBlockOffsetE, size_t sfCountE,
    const size_t* neighborSfOffset,
    double* neighborDGdx, double* neighborDGdy, double* neighborDGdz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    int i = begin + t;
    int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;

    double result[8] = {0}, dResultX[8] = {0}, dResultY[8] = {0}, dResultZ[8] = {0};

    symFncExpRadGroup(n, &neighElem[off], &neighDist[off], &neighDx[off],
                       &neighDy[off], &neighDz[off], e1, rc, numMembers,
                       eta, rs, result, dResultX, dResultY, dResultZ,
                       &neighborDGdx[neighborSfOffset[i]],
                       &neighborDGdy[neighborSfOffset[i]],
                       &neighborDGdz[neighborSfOffset[i]]);

    size_t base = gBlockOffsetE + (size_t)t * sfCountE;
    for (int k = 0; k < numMembers; ++k)
    {
        G[base + k]    = result[k];
        dGdx[base + k] = dResultX[k];
        dGdy[base + k] = dResultY[k];
        dGdz[base + k] = dResultZ[k];
    }
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
    printf("AtomBatch: %zu atoms, G storage: %zu values (H block: %zu atoms x %zu SFs)\n"
           "neighborDGdx storage: %zu values\n\n",
           batch.numAtoms, batch.G.size(),
           batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.sfCountPerElement[H], batch.neighborDGdx.size());

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
    vector<double> dGdxCpu(batch.G.size(), 0.0), dGdyCpu(batch.G.size(), 0.0), dGdzCpu(batch.G.size(), 0.0);
    vector<double> nDGdxCpu(batch.neighborDGdx.size(), 0.0);
    vector<double> nDGdyCpu(batch.neighborDGdx.size(), 0.0);
    vector<double> nDGdzCpu(batch.neighborDGdx.size(), 0.0);
    for (int t = 0; t < numSelected; ++t)
    {
        int i = begin + t;
        int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
        double result[8] = {0}, dResultX[8] = {0}, dResultY[8] = {0}, dResultZ[8] = {0};
        symFncExpRadGroup(n, &neighElemInt[off], &batch.neighborD[off],
                           &batch.neighborDx[off], &batch.neighborDy[off], &batch.neighborDz[off],
                           H, rc, numMembers, eta.data(), rs.data(),
                           result, dResultX, dResultY, dResultZ,
                           &nDGdxCpu[batch.neighborSfOffset[i]],
                           &nDGdyCpu[batch.neighborSfOffset[i]],
                           &nDGdzCpu[batch.neighborSfOffset[i]]);
        size_t base = batch.gBlockOffset[H] + (size_t)t * batch.sfCountPerElement[H];
        for (int k = 0; k < numMembers; ++k)
        {
            Gcpu[base + k] = result[k];
            dGdxCpu[base + k] = dResultX[k];
            dGdyCpu[base + k] = dResultY[k];
            dGdzCpu[base + k] = dResultZ[k];
            // Cross-check against AtomBatch::gIndex()'s own index arithmetic.
            if (base + k != batch.gIndex(i, k))
            {
                fprintf(stderr, "gIndex mismatch at atom %d, sf %d\n", i, k);
                return 1;
            }
        }
        // Cross-check neighborSfIndex() for every local neighbor slot too.
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < numMembers; ++k)
                if (batch.neighborSfOffset[i] + (size_t)(j * numMembers + k)
                    != batch.neighborSfIndex(i, j, k))
                {
                    fprintf(stderr, "neighborSfIndex mismatch at atom %d, neighbor %d, sf %d\n", i, j, k);
                    return 1;
                }
    }

    // --- GPU: write directly into batch's device mirrors --------------------
    int total = (int)batch.neighborD.size();
    int *d_neighOffset, *d_neighElem;
    double *d_neighDist, *d_neighDx, *d_neighDy, *d_neighDz, *d_eta, *d_rs;
    double *d_G, *d_dGdx, *d_dGdy, *d_dGdz;
    size_t* d_neighborSfOffset;
    double *d_nDGdx, *d_nDGdy, *d_nDGdz;

    CUDA_CHECK(cudaMalloc(&d_neighOffset, neighOffsetInt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighElem, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighDist, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDx, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDy, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDz, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eta, numMembers * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rs, numMembers * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdx, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdy, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdz, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighborSfOffset, batch.neighborSfOffset.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_nDGdx, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdy, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdz, batch.neighborDGdx.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_neighOffset, neighOffsetInt.data(), neighOffsetInt.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighElem, neighElemInt.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDist, batch.neighborD.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDx, batch.neighborDx.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDy, batch.neighborDy.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDz, batch.neighborDz.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta, eta.data(), numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rs, rs.data(), numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighborSfOffset, batch.neighborSfOffset.data(), batch.neighborSfOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_G, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdx, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdy, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdz, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdx, 0, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdy, 0, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdz, 0, batch.neighborDGdx.size() * sizeof(double)));

    int blockSize = 128;
    int gridSize = (numSelected + blockSize - 1) / blockSize;
    sfGroupKernel<<<gridSize, blockSize>>>(begin, numSelected, H, rc,
        numMembers, d_eta, d_rs, d_neighOffset, d_neighElem, d_neighDist,
        d_neighDx, d_neighDy, d_neighDz,
        d_G, d_dGdx, d_dGdy, d_dGdz, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> Ggpu(batch.G.size()), dGdxGpu(batch.G.size()), dGdyGpu(batch.G.size()), dGdzGpu(batch.G.size());
    vector<double> nDGdxGpu(batch.neighborDGdx.size()), nDGdyGpu(batch.neighborDGdx.size()), nDGdzGpu(batch.neighborDGdx.size());
    CUDA_CHECK(cudaMemcpy(Ggpu.data(), d_G, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdxGpu.data(), d_dGdx, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdyGpu.data(), d_dGdy, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdzGpu.data(), d_dGdz, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nDGdxGpu.data(), d_nDGdx, batch.neighborDGdx.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nDGdyGpu.data(), d_nDGdy, batch.neighborDGdx.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nDGdzGpu.data(), d_nDGdz, batch.neighborDGdx.size() * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_neighOffset); cudaFree(d_neighElem); cudaFree(d_neighDist);
    cudaFree(d_neighDx); cudaFree(d_neighDy); cudaFree(d_neighDz);
    cudaFree(d_eta); cudaFree(d_rs);
    cudaFree(d_G); cudaFree(d_dGdx); cudaFree(d_dGdy); cudaFree(d_dGdz);
    cudaFree(d_neighborSfOffset); cudaFree(d_nDGdx); cudaFree(d_nDGdy); cudaFree(d_nDGdz);

    double maxAbsErrG = 0.0, maxAbsErrDGown = 0.0, maxAbsErrDGneigh = 0.0;
    for (size_t k = 0; k < Gcpu.size(); ++k)
    {
        maxAbsErrG = max(maxAbsErrG, fabs(Ggpu[k] - Gcpu[k]));
        maxAbsErrDGown = max({maxAbsErrDGown, fabs(dGdxGpu[k] - dGdxCpu[k]),
                               fabs(dGdyGpu[k] - dGdyCpu[k]), fabs(dGdzGpu[k] - dGdzCpu[k])});
    }
    for (size_t k = 0; k < nDGdxCpu.size(); ++k)
    {
        maxAbsErrDGneigh = max({maxAbsErrDGneigh, fabs(nDGdxGpu[k] - nDGdxCpu[k]),
                                 fabs(nDGdyGpu[k] - nDGdyCpu[k]), fabs(nDGdzGpu[k] - nDGdzCpu[k])});
    }

    printf("G[H atom 0] = (%.15E, %.15E)  [cpu: (%.15E, %.15E)]\n",
           Ggpu[batch.gIndex(begin, 0)], Ggpu[batch.gIndex(begin, 1)],
           Gcpu[batch.gIndex(begin, 0)], Gcpu[batch.gIndex(begin, 1)]);
    printf("max|G_gpu-G_cpu|                       = %.3E  (%zu values)\n", maxAbsErrG, Gcpu.size());
    printf("max|dG_own_gpu-dG_own_cpu|              = %.3E  (%zu values)\n", maxAbsErrDGown, Gcpu.size());
    printf("max|dG_neighbor_gpu-dG_neighbor_cpu|    = %.3E  (%zu values)\n", maxAbsErrDGneigh, nDGdxCpu.size());

    bool pass = (maxAbsErrG < 1e-9) && (maxAbsErrDGown < 1e-9) && (maxAbsErrDGneigh < 1e-9);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
