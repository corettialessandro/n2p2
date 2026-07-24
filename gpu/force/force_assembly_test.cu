// Phase 3, force assembly: the single biggest cost center Phase 0's
// profiling identified (~46.5% of wall time, Mode::calculateForces() /
// Atom::calculatePairForceShort()) -- untouched until now. Combines Phase
// 3's dEdG (steps 1-2, ../nn/) with Phase 1's per-neighbor derivative
// storage (../soa/AtomBatch.h step 4, dGdx,Dy,Dz + neighborDGdx,Dy,Dz) into
// actual per-atom forces.
//
// The exact formula (src/libnnp/Atom.cpp:372-402, N2P2_FULL_SFD_MEMORY
// branch -- the one matching AtomBatch's storage shape, one dGdr entry per
// (neighbor slot, symmetry function), no per-element filter table needed):
//   atom i's self-force:      F_self_i  = -sum_k dEdG_i[k] * dGdr_i[k]
//   atom j's contribution to
//   neighbor i's force:       F_pair_i += -sum_k dEdG_j[k] * (j's
//                                          neighbor-entry-for-i).dGdr[k]
// Mode::calculateForces() computes this from atom i's perspective: loop
// over i's unique neighbors j, then RE-SCAN all of j's neighbors looking
// for i (O(k^2) per atom, mitigated on CPU by the compact per-element
// symmetry-function table this port's AtomBatch doesn't use). AtomBatch's
// layout makes that re-scan unnecessary: neighborDGdx,Dy,Dz is already
// addressed by (owner atom j, its neighbor slot, symmetry function), so the
// natural GPU formulation is a SCATTER: one thread per atom j adds its own
// self-force directly, then walks its OWN neighbor list once, atomically
// adding each pair contribution straight onto that neighbor's force
// accumulator -- no search, and it's the same total work as the CPU
// version's inner loop without the O(k^2) part.
//
// Ground truth is NOT the real Atom::calculateSelfForceShort()/
// calculatePairForceShort() here (unlike NeuralNetwork in ../nn/): those
// need either the compact per-element table (Mode/Element/Settings setup,
// far heavier than this project's other smoke tests) or rebuilding libnnp.a
// with -DN2P2_FULL_SFD_MEMORY, which would silently change Atom's struct
// layout (Atom.h conditionally adds a member under that macro) relative to
// the rest of the already-built library -- an ABI mismatch, not a safe
// option. Instead this validates the scatter-add ALGORITHM independently:
// an explicit, differently-ordered "gather" CPU reference (for each target
// atom i, self term plus a scan over every OTHER atom j's neighbor slots
// looking for i) computes the same sums via a genuinely different traversal
// than the GPU's scatter, so an addressing bug in either one would very
// likely disagree with the other. Symmetry-function/NN values themselves
// are synthetic (random dEdG, dGdx,Dy,Dz, neighborDGdx,Dy,Dz on the real
// H2O_2G AtomBatch) -- this step validates the assembly formula in
// isolation, same incremental philosophy as every step before it. Real
// values are NOT expected to sum to zero net force here (that Newton's-
// third-law property depends on the dGdr/-dGdr sign relationship real
// SymFnc code enforces, which this synthetic data doesn't reproduce) --
// only GPU-vs-CPU agreement is checked.

#include "../soa/AtomBatch.h"
#include "ElementMap.h"
#include "Structure.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
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

// One thread per atom j: adds j's own self-force, then scatters j's pair
// contributions onto every neighbor i in j's neighbor list. Matches
// AtomBatch::gIndex(j,k)/neighborSfIndex(j,slot,k)'s addressing exactly
// (gBaseOf[j]==gIndex(j,0), neighborSfOffset[j]+slot*sfCountOf[j]+k ==
// neighborSfIndex(j,slot,k)).
__global__ void forceAssemblyKernel(
    int numAtoms,
    const size_t* neighborOffset, const size_t* neighborAtomSorted,
    const size_t* neighborSfOffset,
    const size_t* gBaseOf, const size_t* sfCountOf,
    const double* dEdG, const double* dGdx, const double* dGdy, const double* dGdz,
    const double* neighborDGdx, const double* neighborDGdy, const double* neighborDGdz,
    double* forceX, double* forceY, double* forceZ)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numAtoms) return;

    size_t gBase = gBaseOf[j];
    size_t sfCount = sfCountOf[j];

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (size_t k = 0; k < sfCount; ++k)
    {
        double dedg = dEdG[gBase + k];
        fx -= dedg * dGdx[gBase + k];
        fy -= dedg * dGdy[gBase + k];
        fz -= dedg * dGdz[gBase + k];
    }
    atomicAdd(&forceX[j], fx);
    atomicAdd(&forceY[j], fy);
    atomicAdd(&forceZ[j], fz);

    size_t nBegin = neighborOffset[j], nEnd = neighborOffset[j + 1];
    size_t sfBase = neighborSfOffset[j];
    for (size_t slot = 0; slot < nEnd - nBegin; ++slot)
    {
        size_t i = neighborAtomSorted[nBegin + slot];
        size_t base = sfBase + slot * sfCount;
        double px = 0.0, py = 0.0, pz = 0.0;
        for (size_t k = 0; k < sfCount; ++k)
        {
            double dedg = dEdG[gBase + k];
            px -= dedg * neighborDGdx[base + k];
            py -= dedg * neighborDGdy[base + k];
            pz -= dedg * neighborDGdz[base + k];
        }
        atomicAdd(&forceX[i], px);
        atomicAdd(&forceY[i], py);
        atomicAdd(&forceZ[i], pz);
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

    // Real per-element symmetry-function counts from temp/H2O_2G/input.nn,
    // same as ../nn/'s tests -- for realism, though the values themselves
    // are synthetic here.
    allocateSfStorage(batch, {35, 42});
    printf("AtomBatch: %zu atoms (%zu H, %zu O), %zu neighbor entries\n\n",
           batch.numAtoms, batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.elementOffset[O + 1] - batch.elementOffset[O],
           batch.totalNeighbors());

    // Per-atom convenience arrays (avoids needing element[]/elementOffset[]
    // on device at all -- the kernel only needs each atom's own gIndex(.,0)
    // base and its element's symmetry-function count).
    size_t numAtoms = batch.numAtoms;
    vector<size_t> gBaseOf(numAtoms), sfCountOf(numAtoms);
    for (size_t s = 0; s < numAtoms; ++s)
    {
        gBaseOf[s] = batch.gIndex(s, 0);
        sfCountOf[s] = batch.sfCountPerElement[batch.element[s]];
    }

    // Synthetic dEdG, dGdx,Dy,Dz (own-atom) and neighborDGdx,Dy,Dz --
    // this step validates the assembly formula/addressing in isolation,
    // decoupled from real symmetry-function/NN kernels (same incremental
    // philosophy as every previous step).
    mt19937 rng(7);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : batch.dEdG) v = dist(rng);
    for (auto& v : batch.dGdx) v = dist(rng);
    for (auto& v : batch.dGdy) v = dist(rng);
    for (auto& v : batch.dGdz) v = dist(rng);
    for (auto& v : batch.neighborDGdx) v = dist(rng);
    for (auto& v : batch.neighborDGdy) v = dist(rng);
    for (auto& v : batch.neighborDGdz) v = dist(rng);

    // --- CPU reference: independent "gather" traversal ----------------------
    // For each target atom i: self term, plus a scan over every atom j's
    // neighbor slots looking for i -- a different traversal order than the
    // GPU's scatter, so an addressing bug in either wouldn't be masked by
    // the other computing the same wrong thing the same way.
    vector<double> forceXCpu(numAtoms, 0.0), forceYCpu(numAtoms, 0.0), forceZCpu(numAtoms, 0.0);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        size_t gBase = gBaseOf[i], sfCount = sfCountOf[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (size_t k = 0; k < sfCount; ++k)
        {
            double dedg = batch.dEdG[gBase + k];
            fx -= dedg * batch.dGdx[gBase + k];
            fy -= dedg * batch.dGdy[gBase + k];
            fz -= dedg * batch.dGdz[gBase + k];
        }
        for (size_t j = 0; j < numAtoms; ++j)
        {
            size_t nBegin = batch.neighborOffset[j], nEnd = batch.neighborOffset[j + 1];
            size_t sfBaseJ = batch.neighborSfOffset[j], sfCountJ = sfCountOf[j];
            for (size_t slot = 0; slot < nEnd - nBegin; ++slot)
            {
                if (batch.neighborAtomSorted[nBegin + slot] != i) continue;
                size_t base = sfBaseJ + slot * sfCountJ;
                size_t gBaseJ = gBaseOf[j];
                for (size_t k = 0; k < sfCountJ; ++k)
                {
                    double dedg = batch.dEdG[gBaseJ + k];
                    fx -= dedg * batch.neighborDGdx[base + k];
                    fy -= dedg * batch.neighborDGdy[base + k];
                    fz -= dedg * batch.neighborDGdz[base + k];
                }
            }
        }
        forceXCpu[i] = fx; forceYCpu[i] = fy; forceZCpu[i] = fz;
    }

    // --- GPU: scatter-add ---------------------------------------------------
    size_t* d_neighborOffset; size_t* d_neighborAtomSorted; size_t* d_neighborSfOffset;
    size_t* d_gBaseOf; size_t* d_sfCountOf;
    double *d_dEdG, *d_dGdx, *d_dGdy, *d_dGdz;
    double *d_nDGdx, *d_nDGdy, *d_nDGdz;
    double *d_forceX, *d_forceY, *d_forceZ;

    CUDA_CHECK(cudaMalloc(&d_neighborOffset, batch.neighborOffset.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_neighborAtomSorted, batch.neighborAtomSorted.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_neighborSfOffset, batch.neighborSfOffset.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_gBaseOf, numAtoms * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_sfCountOf, numAtoms * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_dEdG, batch.dEdG.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdx, batch.dGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdy, batch.dGdy.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdz, batch.dGdz.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdx, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdy, batch.neighborDGdy.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdz, batch.neighborDGdz.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forceX, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forceY, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forceZ, numAtoms * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_neighborOffset, batch.neighborOffset.data(), batch.neighborOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighborAtomSorted, batch.neighborAtomSorted.data(), batch.neighborAtomSorted.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighborSfOffset, batch.neighborSfOffset.data(), batch.neighborSfOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gBaseOf, gBaseOf.data(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfCountOf, sfCountOf.data(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dEdG, batch.dEdG.data(), batch.dEdG.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dGdx, batch.dGdx.data(), batch.dGdx.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dGdy, batch.dGdy.data(), batch.dGdy.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dGdz, batch.dGdz.data(), batch.dGdz.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nDGdx, batch.neighborDGdx.data(), batch.neighborDGdx.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nDGdy, batch.neighborDGdy.data(), batch.neighborDGdy.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nDGdz, batch.neighborDGdz.data(), batch.neighborDGdz.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_forceX, 0, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceY, 0, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceZ, 0, numAtoms * sizeof(double)));

    int blockSize = 128;
    int gridSize = ((int)numAtoms + blockSize - 1) / blockSize;
    forceAssemblyKernel<<<gridSize, blockSize>>>((int)numAtoms,
        d_neighborOffset, d_neighborAtomSorted, d_neighborSfOffset,
        d_gBaseOf, d_sfCountOf, d_dEdG, d_dGdx, d_dGdy, d_dGdz,
        d_nDGdx, d_nDGdy, d_nDGdz, d_forceX, d_forceY, d_forceZ);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(batch.forceX.data(), d_forceX, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(batch.forceY.data(), d_forceY, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(batch.forceZ.data(), d_forceZ, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_neighborOffset); cudaFree(d_neighborAtomSorted); cudaFree(d_neighborSfOffset);
    cudaFree(d_gBaseOf); cudaFree(d_sfCountOf);
    cudaFree(d_dEdG); cudaFree(d_dGdx); cudaFree(d_dGdy); cudaFree(d_dGdz);
    cudaFree(d_nDGdx); cudaFree(d_nDGdy); cudaFree(d_nDGdz);
    cudaFree(d_forceX); cudaFree(d_forceY); cudaFree(d_forceZ);

    double maxAbsErr = 0.0;
    for (size_t i = 0; i < numAtoms; ++i)
    {
        maxAbsErr = max({maxAbsErr, fabs(batch.forceX[i] - forceXCpu[i]),
                          fabs(batch.forceY[i] - forceYCpu[i]),
                          fabs(batch.forceZ[i] - forceZCpu[i])});
    }

    printf("force[0] = (%.15E, %.15E, %.15E)\n", batch.forceX[0], batch.forceY[0], batch.forceZ[0]);
    printf("cpu[0]   = (%.15E, %.15E, %.15E)\n", forceXCpu[0], forceYCpu[0], forceZCpu[0]);
    printf("max|F_gpu-F_cpu| over %zu atoms = %.3E\n", numAtoms, maxAbsErr);

    bool pass = maxAbsErr < 1e-9;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
