// Phase 1/2 extension: neighbor-side derivatives for the narrow angular
// (SymFncExpAngn, type 3) family -- the piece ../smoke/symfnc_family_test.cu
// explicitly left out of scope ("Neighbor-side derivative bookkeeping is
// out of scope... the rjk force term... is provably irrelevant to what we
// validate here"). Needed to cover the real 35/42-wide H2O_2G production
// network end to end (../e2e/ so far only used the 16-wide radial subset,
// since this piece didn't exist yet).
//
// Derivation, re-read line by line from src/libnnp/SymFncExpAngn.cpp:79-225
// (the same care taken here as for ../nn/nn_backward_dfdc_test.cu, after
// that step's dropped-factor bug -- every new derivation in this port gets
// checked against the real CPU class, not trusted on inspection alone):
// for an angular triple (central atom i, neighbors j and k, j<k in i's
// neighbor list), the CPU source computes three force-vector terms
//   drij = p1 * nj.dr,  drik = p2 * nk.dr,  drjk = p3 * (nk.dr - nj.dr)
// and adds them as:
//   atom.dGdr[index] += drij + drik            (central atom's own
//                                                derivative -- already
//                                                ported and validated in
//                                                ../smoke/symfnc_family_test.cu)
//   nj.dGdr[index]    -= drij + drjk           (NEW: neighbor slot j)
//   nk.dGdr[index]    -= drik - drjk           (NEW: neighbor slot k)
// p1/p2/p3 (the exact formulas, re-derived from source, not re-guessed):
//   p1 = fgF*(pfc*(rinvijikF - costijkF/r2ij - p2etapl*rijs/rij)
//            + pfcik*pfcjk*pdfcij*plambda/rij)
//   p2 = fgF*(pfc*(rinvijikF - costijkF/r2ik - p2etapl*riks/rik)
//            + pfcij*pfcjk*pdfcik*plambda/rik)
//   p3 = fgF*(pfc*(rinvijikF + p2etapl*rjks/rjk)
//            - pfcij*pfcik*pdfcjk*plambda/rjk)
// where fgF/rinvijikF/costijkF are fg/rinvijik/costijk AFTER the CPU
// source's `fg *= pnorm; rinvijik *= pzl; costijk *= pzl;` rescaling (done
// once, shared by all three p1/p2/p3 -- easy to mix up with the unscaled
// versions used for the energy accumulator, which is exactly the kind of
// mistake the earlier dFdc bug was).
//
// Since a given neighbor slot (say slot m of central atom i) can appear as
// "j" in one iteration of the (j,k) pair loop and as "k" in another, its
// neighbor-derivative entry must ACCUMULATE (+=) across the whole
// double loop, unlike the radial family's neighborDGdx (Phase 1 step 4),
// where each neighbor slot got exactly one direct write. No atomics are
// needed even so: neighborDGdx's block for center atom i is only ever
// touched by the one thread computing atom i's own symmetry functions
// (ownership is per-central-atom, not per-neighbor-atom), so plain += into
// the pre-zeroed global array is safe.
//
// Grouped over all of one element's real ExpAngn instances at once (19 for
// H, 26 for O, parsed from temp/H2O_2G/input.nn -- rs is always 0 for this
// file's 4-parameter type-3 form, `eta lambda zeta rc`, no rs column):
// the expensive (j,k) pair enumeration and the pfcij/pfcik/pfcjk/costijk
// geometry are computed ONCE per pair and shared across every member,
// mirroring SymGrpExpAngn.cpp's real optimization -- generalized here to a
// per-member (e1,e2) filter (like ../e2e/'s radial generalization) since
// real H/O instances mix multiple (e1,e2) pairs that wouldn't fit a single
// shared-filter SymGrp-style group.

#include "AtomBatch.h"
#include "ElementMap.h"
#include "Structure.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
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

__host__ __device__ inline double pow_int(double x, int n)
{
    unsigned int m;
    if (n < 0) { x = 1.0 / x; m = (unsigned int)(-n); }
    else m = (unsigned int)n;
    double result = 1.0;
    do
    {
        if (m & 1) result *= x;
        m >>= 1;
        x *= x;
    } while (m);
    return result;
}

// See file header for the derivation. result/dResultX,Y,Z (size numMembers)
// and neighDGdx,Dy,Dz (size numNeighbors*numMembers, row-major
// [neighborSlot*numMembers+m]) must be pre-zeroed by the caller.
__host__ __device__ inline void symFncExpAngnGroupReal(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double rc,
    int numMembers, const int* e1, const int* e2, const double* eta,
    const double* lambda, const double* zeta,
    double* result, double* dResultX, double* dResultY, double* dResultZ,
    double* neighDGdx, double* neighDGdy, double* neighDGdz)
{
    double rc2 = rc * rc;
    double rcinv = 1.0 / rc;

    for (int j = 0; j < numNeighbors - 1; ++j)
    {
        double rij = neighDist[j];
        if (!(rij < rc)) continue;
        int nej = neighElem[j];
        double pfcij, pdfcij;
        cutoffTANHU(rij, rcinv, pfcij, pdfcij);

        for (int k = j + 1; k < numNeighbors; ++k)
        {
            double rik = neighDist[k];
            if (!(rik < rc)) continue;
            int nek = neighElem[k];

            double djkx = neighDx[k] - neighDx[j];
            double djky = neighDy[k] - neighDy[j];
            double djkz = neighDz[k] - neighDz[j];
            double rjk2 = djkx * djkx + djky * djky + djkz * djkz;
            if (!(rjk2 < rc2)) continue;
            double rjk = sqrt(rjk2);

            double pfcik, pdfcik; cutoffTANHU(rik, rcinv, pfcik, pdfcik);
            double pfcjk, pdfcjk; cutoffTANHU(rjk, rcinv, pfcjk, pdfcjk);

            double dijx = neighDx[j], dijy = neighDy[j], dijz = neighDz[j];
            double dikx = neighDx[k], diky = neighDy[k], dikz = neighDz[k];
            double costijk0 = (dijx * dikx + dijy * diky + dijz * dikz) / (rij * rik);

            double pfc = pfcij * pfcik * pfcjk;
            double r2ij = rij * rij, r2ik = rik * rik;

            for (int m = 0; m < numMembers; ++m)
            {
                bool matches = (nej == e1[m] && nek == e2[m]) ||
                               (nej == e2[m] && nek == e1[m]);
                if (!matches) continue;

                // rs == 0 for this file's 4-parameter type-3 form.
                double rijs = rij, riks = rik, rjks = rjk;
                double pexp = exp(-eta[m] * (rijs * rijs + riks * riks + rjks * rjks));
                double plambda = 1.0 + lambda[m] * costijk0;

                double pnorm = pow(2.0, 1.0 - zeta[m]);
                int zetaInt = (int)llround(zeta[m]);
                bool useIntegerPow = (fabs(zeta[m] - zetaInt) <= 1e-12);

                double fg = pexp;
                if (plambda <= 0.0) fg = 0.0;
                else fg *= useIntegerPow ? pow_int(plambda, zetaInt - 1)
                                         : pow(plambda, zeta[m] - 1.0);

                // pnorm deferred to after the whole (j,k) loop -- matches
                // the CPU source applying it once to the final sum.
                result[m] += fg * plambda * pfc;

                double fgF = fg * pnorm;
                double pzl = zeta[m] * lambda[m];
                double rinvijikF = pzl / (rij * rik);
                double costijkF  = costijk0 * pzl;
                double p2etapl   = 2.0 * eta[m] * plambda;

                double p1 = fgF * (pfc * (rinvijikF - costijkF / r2ij - p2etapl * rijs / rij)
                                   + pfcik * pfcjk * pdfcij * plambda / rij);
                double p2 = fgF * (pfc * (rinvijikF - costijkF / r2ik - p2etapl * riks / rik)
                                   + pfcij * pfcjk * pdfcik * plambda / rik);
                double p3 = fgF * (pfc * (rinvijikF + p2etapl * rjks / rjk)
                                   - pfcij * pfcik * pdfcjk * plambda / rjk);

                double drijx = p1 * dijx, drijy = p1 * dijy, drijz = p1 * dijz;
                double drikx = p2 * dikx, driky = p2 * diky, drikz = p2 * dikz;
                double drjkx = p3 * djkx, drjky = p3 * djky, drjkz = p3 * djkz;

                dResultX[m] += drijx + drikx;
                dResultY[m] += drijy + driky;
                dResultZ[m] += drijz + drikz;

                int idxJ = j * numMembers + m;
                int idxK = k * numMembers + m;
                neighDGdx[idxJ] -= drijx + drjkx;
                neighDGdy[idxJ] -= drijy + drjky;
                neighDGdz[idxJ] -= drijz + drjkz;
                neighDGdx[idxK] -= drikx - drjkx;
                neighDGdy[idxK] -= driky - drjky;
                neighDGdz[idxK] -= drikz - drjkz;
            }
        }
    }
    for (int m = 0; m < numMembers; ++m)
    {
        result[m] *= pow(2.0, 1.0 - zeta[m]);
    }
}

__global__ void sfAngnGroupKernel(
    int begin, int numSelected, double rc, int numMembers,
    const int* e1, const int* e2, const double* eta, const double* lambda,
    const double* zeta,
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

    double result[32] = {0}, dResultX[32] = {0}, dResultY[32] = {0}, dResultZ[32] = {0};

    symFncExpAngnGroupReal(n, &neighElem[off], &neighDist[off], &neighDx[off],
                           &neighDy[off], &neighDz[off], rc, numMembers,
                           e1, e2, eta, lambda, zeta,
                           result, dResultX, dResultY, dResultZ,
                           &neighborDGdx[neighborSfOffset[i]],
                           &neighborDGdy[neighborSfOffset[i]],
                           &neighborDGdz[neighborSfOffset[i]]);

    size_t base = gBlockOffsetE + (size_t)t * sfCountE;
    for (int k = 0; k < numMembers; ++k)
    {
        G[base + k] = result[k];
        dGdx[base + k] = dResultX[k];
        dGdy[base + k] = dResultY[k];
        dGdz[base + k] = dResultZ[k];
    }
}

///////////////////////////////////////////////////////////////////////////
// input.nn parsing: "symfunction_short <ec> 3 <e1> <e2> <eta> <lambda>
// <zeta> <rc>" lines only, in file order.
///////////////////////////////////////////////////////////////////////////

struct AngMember { int e1, e2; double eta, lambda, zeta, rc; };

vector<AngMember> parseExpAngnMembers(string const& path, string const& ec,
                                       ElementMap const& elementMap)
{
    vector<AngMember> members;
    ifstream in(path);
    string line;
    while (getline(in, line))
    {
        istringstream iss(line);
        string tag;
        if (!(iss >> tag) || tag != "symfunction_short") continue;
        string ecStr; int type;
        if (!(iss >> ecStr >> type)) continue;
        if (ecStr != ec || type != 3) continue;
        string e1Str, e2Str; double eta, lambda, zeta, rc;
        iss >> e1Str >> e2Str >> eta >> lambda >> zeta >> rc;
        AngMember m;
        m.e1 = (int)elementMap[e1Str]; m.e2 = (int)elementMap[e2Str];
        m.eta = eta; m.lambda = lambda; m.zeta = zeta; m.rc = rc;
        members.push_back(m);
    }
    return members;
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
    string inputNn   = (argc > 2) ? argv[2] : "input.nn";
    double const rc = 12.0;

    ElementMap elementMap;
    elementMap.registerElements("H O");
    Structure structure;
    structure.setElementMap(elementMap);
    structure.readFromFile(inputData);
    structure.calculateNeighborList(rc);

    AtomBatch batch = buildAtomBatch(structure, rc);
    int const H = 0, O = 1;

    vector<AngMember> membersH = parseExpAngnMembers(inputNn, "H", elementMap);
    vector<AngMember> membersO = parseExpAngnMembers(inputNn, "O", elementMap);
    int numH = (int)membersH.size(), numO = (int)membersO.size();
    printf("Parsed %d real ExpAngn instances for H, %d for O (from %s)\n",
           numH, numO, inputNn.c_str());

    allocateSfStorage(batch, {(size_t)numH, (size_t)numO});
    printf("AtomBatch: %zu atoms (%zu H, %zu O), %zu neighbor entries\n\n",
           batch.numAtoms, batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.elementOffset[O + 1] - batch.elementOffset[O],
           batch.totalNeighbors());

    vector<int> neighOffsetInt(batch.neighborOffset.begin(), batch.neighborOffset.end());
    vector<int> neighElemInt(batch.neighborElement.begin(), batch.neighborElement.end());

    auto runCase = [&](char const* label, int begin, int end,
                       vector<AngMember> const& members, size_t gBlockOffsetE,
                       size_t sfCountE) -> bool
    {
        printf("--- %s (%zu instances) ---\n", label, members.size());
        int numSelected = end - begin;
        int numMembers = (int)members.size();
        vector<int> e1(numMembers), e2(numMembers);
        vector<double> eta(numMembers), lambda(numMembers), zeta(numMembers);
        for (int m = 0; m < numMembers; ++m)
        {
            e1[m] = members[m].e1; e2[m] = members[m].e2;
            eta[m] = members[m].eta; lambda[m] = members[m].lambda; zeta[m] = members[m].zeta;
        }

        // --- CPU reference ---------------------------------------------------
        vector<double> gCpu(batch.G.size(), 0.0);
        vector<double> dGdxCpu(batch.G.size(), 0.0), dGdyCpu(batch.G.size(), 0.0), dGdzCpu(batch.G.size(), 0.0);
        vector<double> nDGdxCpu(batch.neighborDGdx.size(), 0.0);
        vector<double> nDGdyCpu(batch.neighborDGdx.size(), 0.0);
        vector<double> nDGdzCpu(batch.neighborDGdx.size(), 0.0);
        for (int i = begin; i < end; ++i)
        {
            int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
            vector<double> result(numMembers, 0.0), dResultX(numMembers, 0.0), dResultY(numMembers, 0.0), dResultZ(numMembers, 0.0);
            symFncExpAngnGroupReal(n, &neighElemInt[off], &batch.neighborD[off],
                                   &batch.neighborDx[off], &batch.neighborDy[off], &batch.neighborDz[off],
                                   rc, numMembers, e1.data(), e2.data(), eta.data(), lambda.data(), zeta.data(),
                                   result.data(), dResultX.data(), dResultY.data(), dResultZ.data(),
                                   &nDGdxCpu[batch.neighborSfOffset[i]], &nDGdyCpu[batch.neighborSfOffset[i]], &nDGdzCpu[batch.neighborSfOffset[i]]);
            size_t base = batch.gIndex(i, 0);
            for (int m = 0; m < numMembers; ++m)
            {
                gCpu[base + m] = result[m];
                dGdxCpu[base + m] = dResultX[m];
                dGdyCpu[base + m] = dResultY[m];
                dGdzCpu[base + m] = dResultZ[m];
            }
        }

        // --- GPU ---------------------------------------------------------------
        int *d_e1, *d_e2; double *d_eta, *d_lambda, *d_zeta;
        CUDA_CHECK(cudaMalloc(&d_e1, numMembers * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_e2, numMembers * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_eta, numMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_lambda, numMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_zeta, numMembers * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_e1, e1.data(), numMembers * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_e2, e2.data(), numMembers * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_eta, eta.data(), numMembers * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), numMembers * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zeta, zeta.data(), numMembers * sizeof(double), cudaMemcpyHostToDevice));

        int total = (int)batch.neighborD.size();
        int *d_neighOffset, *d_neighElem;
        double *d_neighDist, *d_neighDx, *d_neighDy, *d_neighDz;
        CUDA_CHECK(cudaMalloc(&d_neighOffset, neighOffsetInt.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighElem, total * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighDist, total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDx, total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDy, total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDz, total * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_neighOffset, neighOffsetInt.data(), neighOffsetInt.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neighElem, neighElemInt.data(), total * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neighDist, batch.neighborD.data(), total * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neighDx, batch.neighborDx.data(), total * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neighDy, batch.neighborDy.data(), total * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neighDz, batch.neighborDz.data(), total * sizeof(double), cudaMemcpyHostToDevice));

        size_t* d_neighborSfOffset;
        CUDA_CHECK(cudaMalloc(&d_neighborSfOffset, batch.neighborSfOffset.size() * sizeof(size_t)));
        CUDA_CHECK(cudaMemcpy(d_neighborSfOffset, batch.neighborSfOffset.data(), batch.neighborSfOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice));

        double *d_G, *d_dGdx, *d_dGdy, *d_dGdz, *d_nDGdx, *d_nDGdy, *d_nDGdz;
        CUDA_CHECK(cudaMalloc(&d_G, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdx, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdy, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdz, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_nDGdx, batch.neighborDGdx.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_nDGdy, batch.neighborDGdx.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_nDGdz, batch.neighborDGdx.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_G, 0, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_dGdx, 0, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_dGdy, 0, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_dGdz, 0, batch.G.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_nDGdx, 0, batch.neighborDGdx.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_nDGdy, 0, batch.neighborDGdx.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_nDGdz, 0, batch.neighborDGdx.size() * sizeof(double)));

        int blockSize = 128;
        sfAngnGroupKernel<<<(numSelected + blockSize - 1) / blockSize, blockSize>>>(
            begin, numSelected, rc, numMembers, d_e1, d_e2, d_eta, d_lambda, d_zeta,
            d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy, d_neighDz,
            d_G, d_dGdx, d_dGdy, d_dGdz, gBlockOffsetE, sfCountE,
            d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> gGpu(batch.G.size()), dGdxGpu(batch.G.size()), dGdyGpu(batch.G.size()), dGdzGpu(batch.G.size());
        vector<double> nDGdxGpu(batch.neighborDGdx.size()), nDGdyGpu(batch.neighborDGdx.size()), nDGdzGpu(batch.neighborDGdx.size());
        CUDA_CHECK(cudaMemcpy(gGpu.data(), d_G, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dGdxGpu.data(), d_dGdx, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dGdyGpu.data(), d_dGdy, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dGdzGpu.data(), d_dGdz, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nDGdxGpu.data(), d_nDGdx, batch.neighborDGdx.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nDGdyGpu.data(), d_nDGdy, batch.neighborDGdx.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nDGdzGpu.data(), d_nDGdz, batch.neighborDGdx.size() * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(d_e1); cudaFree(d_e2); cudaFree(d_eta); cudaFree(d_lambda); cudaFree(d_zeta);
        cudaFree(d_neighOffset); cudaFree(d_neighElem); cudaFree(d_neighDist);
        cudaFree(d_neighDx); cudaFree(d_neighDy); cudaFree(d_neighDz);
        cudaFree(d_neighborSfOffset);
        cudaFree(d_G); cudaFree(d_dGdx); cudaFree(d_dGdy); cudaFree(d_dGdz);
        cudaFree(d_nDGdx); cudaFree(d_nDGdy); cudaFree(d_nDGdz);

        double maxErrG = 0.0, maxErrDGown = 0.0, maxErrDGneigh = 0.0;
        for (size_t k = 0; k < gCpu.size(); ++k)
        {
            maxErrG = max(maxErrG, fabs(gGpu[k] - gCpu[k]));
            maxErrDGown = max({maxErrDGown, fabs(dGdxGpu[k] - dGdxCpu[k]),
                                fabs(dGdyGpu[k] - dGdyCpu[k]), fabs(dGdzGpu[k] - dGdzCpu[k])});
        }
        for (size_t k = 0; k < nDGdxCpu.size(); ++k)
        {
            maxErrDGneigh = max({maxErrDGneigh, fabs(nDGdxGpu[k] - nDGdxCpu[k]),
                                  fabs(nDGdyGpu[k] - nDGdyCpu[k]), fabs(nDGdzGpu[k] - nDGdzCpu[k])});
        }

        printf("  max|G_gpu-G_cpu|=%.3E  max|dG_own|=%.3E  max|dG_neighbor|=%.3E (%zu neighbor-slot values)\n",
               maxErrG, maxErrDGown, maxErrDGneigh, nDGdxCpu.size());
        bool pass = (maxErrG < 1e-9) && (maxErrDGown < 1e-9) && (maxErrDGneigh < 1e-9);
        printf("  %s\n", pass ? "PASS" : "FAIL");
        return pass;
    };

    bool ok = true;
    ok &= runCase("H ExpAngn", (int)batch.elementOffset[H], (int)batch.elementOffset[H + 1],
                  membersH, batch.gBlockOffset[H], batch.sfCountPerElement[H]);
    ok &= runCase("O ExpAngn", (int)batch.elementOffset[O], (int)batch.elementOffset[O + 1],
                  membersO, batch.gBlockOffset[O], batch.sfCountPerElement[O]);

    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
