// End-to-end integration test: chain every piece validated so far (Phase 2's
// radial AND narrow-angular symmetry function kernels, Phase 3's NN
// forward/backward, and force assembly) into one real GPU pipeline for the
// real H2O_2G structure, using REAL geometry and REAL symmetry-function
// parameters parsed straight out of temp/H2O_2G/input.nn -- not synthetic
// data at each stage, unlike every step before this file first existed.
//
// This now covers the FULL real 35/42-wide production network (16 radial +
// 19 angular = 35 for H, 16 radial + 26 angular = 42 for O), not just the
// 16-wide radial subset this file started with -- the angular family's
// neighbor-side derivatives (../soa/symfnc_expangn_group_test.cu) were the
// missing piece, ported and validated separately first, then wired in here.
// Symmetry function INDEX ORDER matters and is NOT arbitrary: input.nn
// defines, for a given central element, all of that element's type-2
// (radial) instances contiguously, followed by all of its type-3 (angular)
// instances (verified directly against the file, no interleaving) -- so
// concatenating [radial members][angular members] in parse order reproduces
// the real production G-vector layout exactly, not just "a" 35/42-wide
// network.
//
// GPU pipeline: all data stays resident on device across the four kernel
// launches (radial symmetry functions -> angular symmetry functions -> NN
// forward+dEdG -> force assembly) -- only the initial geometry/weights
// upload and the final energy/force download cross the host/device
// boundary, mirroring what a real batched training step would do.
//
// CPU reference: fully independent computation using the SAME real
// nnp::NeuralNetwork class (linked from lib/libnnp.a) and the same
// symFncExpRadGroupReal()/symFncExpAngnGroupReal() host/device functions
// called directly on the host -- this specifically re-checks the
// WIRING/indexing (including the two symmetry-function families sharing
// one G vector via sfIndexOffset), not the core per-neighbor math again,
// which earlier phases already validated bit-exact against
// SymFnc{ExpRad,ExpAngn}::calculate(). Force assembly's CPU side reuses
// the same independent "gather" traversal ../force/force_assembly_test.cu
// introduced.
//
// Stage 5 (new): wires ../kalman/'s Kalman filter into this pipeline with a
// REAL per-structure energy Jacobian instead of synthetic H/xi -- closing
// the loop from "symmetry functions -> NN -> forces" to "-> weight update".
// See the Stage 5 kernels' own header comment further down for the
// NeuralNetwork::calculateDEdc() derivation this needed that didn't exist
// in this port before now.

#include "../soa/AtomBatch.h"
#include "ElementMap.h"
#include "Structure.h"
#include "NeuralNetwork.h"
#include "KalmanFilter.h"

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

///////////////////////////////////////////////////////////////////////////
// Shared math building blocks
///////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////
// Stage 1: real radial (ExpRad) symmetry functions, per-member e1 filter
// (../soa/symfnc_exprad_group_test.cu's pattern). sfIndexOffset lets this
// family write into a sub-range of a wider, shared G block (radial members
// occupy [0, numRadMembers), matching input.nn's real ordering).
///////////////////////////////////////////////////////////////////////////

// neighDGdx,Dy,Dz here is the GLOBAL per-atom neighbor-derivative block
// (already offset to this central atom's own neighborSfOffset), addressed
// with the atom's FULL symmetry-function stride (sfStride, e.g. 35/42) and
// this family's sfIndexOffset within that stride -- NOT numMembers, since
// several families (radial, angular) share one wider per-atom block here.
// Written directly, no local per-neighbor-slot staging buffer (that would
// need sizing by the largest possible neighbor count, which isn't a
// compile-time constant).
__host__ __device__ inline void symFncExpRadGroupReal(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double rc,
    int numMembers, const int* e1, const double* eta, const double* rs,
    double* result, double* dResultX, double* dResultY, double* dResultZ,
    double* neighDGdx, double* neighDGdy, double* neighDGdz,
    size_t sfStride, size_t sfIndexOffset)
{
    double rcinv = 1.0 / rc;
    for (int j = 0; j < numNeighbors; ++j)
    {
        double rij = neighDist[j];
        if (rij >= rc) continue;
        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);
        for (int k = 0; k < numMembers; ++k)
        {
            if (neighElem[j] != e1[k]) continue;
            double diff = rij - rs[k];
            double pexp = exp(-eta[k] * diff * diff);
            result[k] += pexp * pfc;
            double p1 = (pdfc - 2.0 * eta[k] * diff * pfc) * pexp / rij;
            double dijx = p1 * neighDx[j];
            double dijy = p1 * neighDy[j];
            double dijz = p1 * neighDz[j];
            dResultX[k] += dijx; dResultY[k] += dijy; dResultZ[k] += dijz;
            size_t idx = (size_t)j * sfStride + sfIndexOffset + k;
            neighDGdx[idx] = -dijx;
            neighDGdy[idx] = -dijy;
            neighDGdz[idx] = -dijz;
        }
    }
}

__global__ void sfGroupKernelReal(
    int begin, int numSelected, double rc, int numMembers, int sfIndexOffset,
    const int* e1, const double* eta, const double* rs,
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

    symFncExpRadGroupReal(n, &neighElem[off], &neighDist[off], &neighDx[off],
                          &neighDy[off], &neighDz[off], rc, numMembers,
                          e1, eta, rs, result, dResultX, dResultY, dResultZ,
                          &neighborDGdx[neighborSfOffset[i]],
                          &neighborDGdy[neighborSfOffset[i]],
                          &neighborDGdz[neighborSfOffset[i]],
                          sfCountE, (size_t)sfIndexOffset);

    size_t base = gBlockOffsetE + (size_t)t * sfCountE + sfIndexOffset;
    for (int k = 0; k < numMembers; ++k)
    {
        G[base + k] = result[k];
        dGdx[base + k] = dResultX[k];
        dGdy[base + k] = dResultY[k];
        dGdz[base + k] = dResultZ[k];
    }
}

///////////////////////////////////////////////////////////////////////////
// Stage 2: real narrow angular (ExpAngn) symmetry functions, per-member
// (e1,e2) filter, WITH neighbor-side derivatives
// (../soa/symfnc_expangn_group_test.cu, validated there). sfIndexOffset
// places these members after the radial ones in the same shared G block.
///////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void symFncExpAngnGroupReal(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double rc,
    int numMembers, const int* e1, const int* e2, const double* eta,
    const double* lambda, const double* zeta,
    double* result, double* dResultX, double* dResultY, double* dResultZ,
    double* neighDGdx, double* neighDGdy, double* neighDGdz,
    size_t sfStride, size_t sfIndexOffset)
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

                double rijs = rij, riks = rik, rjks = rjk; // rs == 0
                double pexp = exp(-eta[m] * (rijs * rijs + riks * riks + rjks * rjks));
                double plambda = 1.0 + lambda[m] * costijk0;

                double pnorm = pow(2.0, 1.0 - zeta[m]);
                int zetaInt = (int)llround(zeta[m]);
                bool useIntegerPow = (fabs(zeta[m] - zetaInt) <= 1e-12);

                double fg = pexp;
                if (plambda <= 0.0) fg = 0.0;
                else fg *= useIntegerPow ? pow_int(plambda, zetaInt - 1)
                                         : pow(plambda, zeta[m] - 1.0);

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

                size_t idxJ = (size_t)j * sfStride + sfIndexOffset + m;
                size_t idxK = (size_t)k * sfStride + sfIndexOffset + m;
                neighDGdx[idxJ] -= drijx + drjkx;
                neighDGdy[idxJ] -= drijy + drjky;
                neighDGdz[idxJ] -= drijz + drjkz;
                neighDGdx[idxK] -= drikx - drjkx;
                neighDGdy[idxK] -= driky - drjky;
                neighDGdz[idxK] -= drikz - drjkz;
            }
        }
    }
    for (int m = 0; m < numMembers; ++m) result[m] *= pow(2.0, 1.0 - zeta[m]);
}

__global__ void sfAngnGroupKernel(
    int begin, int numSelected, double rc, int numMembers, int sfIndexOffset,
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
                           &neighborDGdz[neighborSfOffset[i]],
                           sfCountE, (size_t)sfIndexOffset);

    size_t base = gBlockOffsetE + (size_t)t * sfCountE + sfIndexOffset;
    for (int k = 0; k < numMembers; ++k)
    {
        G[base + k] = result[k];
        dGdx[base + k] = dResultX[k];
        dGdy[base + k] = dResultY[k];
        dGdz[base + k] = dResultZ[k];
    }
}

///////////////////////////////////////////////////////////////////////////
// Stage 3: NN forward + dEdG (copied from ../nn/nn_forward_test.cu, already
// validated there against the real NeuralNetwork class).
///////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void nnForwardAndDEdG(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    double* energyOut, double* dEdGOut)
{
    double h1[64], dfdx1[64];
    for (int k = 0; k < numHidden1; ++k)
    {
        double s = b1[k];
        for (int j = 0; j < numIn; ++j) s += W1[j * numHidden1 + k] * G[j];
        h1[k] = tanh(s);
        dfdx1[k] = 1.0 - h1[k] * h1[k];
    }
    double h2[64], dfdx2[64];
    for (int k = 0; k < numHidden2; ++k)
    {
        double s = b2[k];
        for (int j = 0; j < numHidden1; ++j) s += W2[j * numHidden2 + k] * h1[j];
        h2[k] = tanh(s);
        dfdx2[k] = 1.0 - h2[k] * h2[k];
    }
    double out = b3[0];
    for (int j = 0; j < numHidden2; ++j) out += W3[j] * h2[j];
    energyOut[0] = out;

    for (int k = 0; k < numIn; ++k)
    {
        double inner0[64];
        for (int i = 0; i < numHidden1; ++i)
            inner0[i] = W1[k * numHidden1 + i] * dfdx1[i];
        double outer0[64];
        for (int i2 = 0; i2 < numHidden2; ++i2)
        {
            double s = 0.0;
            for (int i = 0; i < numHidden1; ++i) s += W2[i * numHidden2 + i2] * inner0[i];
            outer0[i2] = s * dfdx2[i2];
        }
        double s = 0.0;
        for (int i2 = 0; i2 < numHidden2; ++i2) s += W3[i2] * outer0[i2];
        dEdGOut[k] = s;
    }
}

__global__ void nnForwardKernel(
    int begin, int numSelected, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    const double* G, size_t gBlockOffsetE, size_t sfCountE,
    double* energyOut, double* dEdG)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    int s = begin + t;
    size_t base = gBlockOffsetE + (size_t)t * sfCountE;
    double out[1];
    nnForwardAndDEdG(&G[base], numIn, W1, b1, numHidden1, W2, b2, numHidden2,
                      W3, b3, numOut, out, &dEdG[base]);
    energyOut[s] = out[0];
}

///////////////////////////////////////////////////////////////////////////
// Stage 4: force assembly (copied verbatim from
// ../force/force_assembly_test.cu, already validated there).
///////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////
// Stage 5: wiring Phase 4's Kalman filter into this pipeline with a REAL
// per-structure Jacobian, closing the loop from "symmetry functions -> NN ->
// forces" to "-> weight update". This is a genuine ENERGY update (the
// dominant update type in temp/H2O_2G/input.nn: short_energy_fraction 1.0
// vs. short_force_fraction 0.0041), i.e. exactly what
// Training::update("energy") does once per structure.
//
// The piece this needed that didn't exist yet: NeuralNetwork::calculateDEdc()
// (src/libnnp/NeuralNetwork.cpp:444), the derivative of the atomic energy
// output w.r.t. every connection (weight+bias) -- NOT calculateDFdc (that's
// the force-Jacobian, already ported in ../nn/nn_backward_dfdc_test.cu, not
// needed for an energy update). Re-derived by hand from the real source and
// confirmed to reduce to exactly the standard backprop deltas
// (dEdbHidden2/dEdbHidden1) calculateDFdc already computes as intermediates:
//   dE/db3        = 1                                (identity output, dfdx=1)
//   dE/dW3[j]     = h2[j]
//   dE/db2[j]     = W3[j]*dfdx2[j]                    (== dEdbHidden2[j])
//   dE/dW2[i][j]  = dEdbHidden2[j] * h1[i]
//   dE/db1[i]     = dfdx1[i] * sum_j W2[i][j]*dEdbHidden2[j]   (== dEdbHidden1[i])
//   dE/dW1[jin][i]= dEdbHidden1[i] * G[jin]
// Same [W1,b1,W2,b2,W3,b3] flat connection order as calculateDFdc/
// getConnections(). Since all atoms of one element share that element's
// weights, the STRUCTURE's total energy Jacobian is the per-atom dEdc
// values summed over that element's atoms -- exactly mirroring how the total
// energy itself is a per-atom sum.
//
// State vector for the Kalman update is the COMBINED H+O weight vector
// (update_strategy 0 = US_COMBINED, same as ../kalman/kalman_test.cu), built
// from the REAL current connH/connO weight arrays (not zero, unlike
// kalman_test.cu's proof-of-concept state) since a real update moves
// EXISTING weights, not a from-scratch delta. Observation is m=1: this one
// structure's energy residual (structure.energyRef - the already-computed
// total energy). The recursion kernels themselves are copied verbatim from
// ../kalman/kalman_test.cu (already validated there against both an
// independent CPU reference and the real nnp::KalmanFilter class at this
// exact N=3327 production size); only the Jacobian/observation feeding them
// is new here.
///////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void nnCalculateDEdc(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut, // numOut == 1
    double* dEdc)
{
    double h1[64], dfdx1[64];
    for (int k = 0; k < numHidden1; ++k)
    {
        double s = b1[k];
        for (int j = 0; j < numIn; ++j) s += W1[j * numHidden1 + k] * G[j];
        h1[k] = tanh(s);
        dfdx1[k] = 1.0 - h1[k] * h1[k];
    }
    double h2[64], dfdx2[64];
    for (int k = 0; k < numHidden2; ++k)
    {
        double s = b2[k];
        for (int j = 0; j < numHidden1; ++j) s += W2[j * numHidden2 + k] * h1[j];
        h2[k] = tanh(s);
        dfdx2[k] = 1.0 - h2[k] * h2[k];
    }

    size_t const offW1 = 0;
    size_t const offB1 = offW1 + (size_t)numIn * numHidden1;
    size_t const offW2 = offB1 + numHidden1;
    size_t const offB2 = offW2 + (size_t)numHidden1 * numHidden2;
    size_t const offW3 = offB2 + numHidden2;
    size_t const offB3 = offW3 + (size_t)numHidden2 * numOut;

    dEdc[offB3] = 1.0; // output layer is identity, dfdx == 1

    double dEdbHidden2[64];
    for (int j = 0; j < numHidden2; ++j)
    {
        dEdc[offW3 + j] = h2[j];
        dEdbHidden2[j] = W3[j] * dfdx2[j];
        dEdc[offB2 + j] = dEdbHidden2[j];
    }

    double dEdbHidden1[64];
    for (int i = 0; i < numHidden1; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < numHidden2; ++j)
        {
            s += W2[i * numHidden2 + j] * dEdbHidden2[j];
            dEdc[offW2 + (size_t)i * numHidden2 + j] = dEdbHidden2[j] * h1[i];
        }
        dEdbHidden1[i] = dfdx1[i] * s;
        dEdc[offB1 + i] = dEdbHidden1[i];
    }

    for (int jin = 0; jin < numIn; ++jin)
        for (int i = 0; i < numHidden1; ++i)
            dEdc[offW1 + (size_t)jin * numHidden1 + i] = dEdbHidden1[i] * G[jin];
}

__global__ void nnDEdcKernel(
    int begin, int numSelected, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    const double* G, size_t gBlockOffsetE, size_t sfCountE,
    double* dEdc, size_t dEdcBlockOffsetE, size_t connCountE)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    size_t gBase = gBlockOffsetE + (size_t)t * sfCountE;
    size_t dEdcBase = dEdcBlockOffsetE + (size_t)t * connCountE;
    nnCalculateDEdc(&G[gBase], numIn, W1, b1, numHidden1, W2, b2, numHidden2,
                     W3, b3, numOut, &dEdc[dEdcBase]);
}

// --- Kalman recursion kernels, copied verbatim from ../kalman/kalman_test.cu
// (KalmanFilter::update(), KT_STANDARD -- see that file's header comment for
// the full derivation/citation). P row-major N x N; H, X, K column-major
// N x m; A, Ainv column-major m x m.

__global__ void kalmanComputeX(const double* P, const double* H,
                                double* X, int N, int m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    if (row < N && col < m)
    {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) sum += P[row * N + k] * H[k + col * N];
        X[row + col * N] = sum;
    }
}

__global__ void kalmanComputeA(const double* H, const double* X,
                                double* A, int N, int m)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    if (i < m && j < m)
    {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) sum += H[k + i * N] * X[k + j * N];
        A[i + j * m] = sum;
    }
}

__global__ void kalmanAddDiag(double* A, int m, double value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) A[i + i * m] += value;
}

__global__ void kalmanInvertGaussJordan(const double* A, double* Ainv,
                                         double* augmented, int m)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int w = 2 * m;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j) augmented[i * w + j] = A[i + j * m];
        for (int j = 0; j < m; ++j)
            augmented[i * w + m + j] = (i == j) ? 1.0 : 0.0;
    }

    for (int col = 0; col < m; ++col)
    {
        int pivotRow = col;
        double best = fabs(augmented[col * w + col]);
        for (int r = col + 1; r < m; ++r)
        {
            double v = fabs(augmented[r * w + col]);
            if (v > best) { best = v; pivotRow = r; }
        }
        if (pivotRow != col)
        {
            for (int j = 0; j < w; ++j)
            {
                double tmp = augmented[col * w + j];
                augmented[col * w + j] = augmented[pivotRow * w + j];
                augmented[pivotRow * w + j] = tmp;
            }
        }

        double piv = augmented[col * w + col];
        for (int j = 0; j < w; ++j) augmented[col * w + j] /= piv;

        for (int r = 0; r < m; ++r)
        {
            if (r == col) continue;
            double factor = augmented[r * w + col];
            if (factor == 0.0) continue;
            for (int j = 0; j < w; ++j)
                augmented[r * w + j] -= factor * augmented[col * w + j];
        }
    }

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            Ainv[i + j * m] = augmented[i * w + m + j];
}

__global__ void kalmanComputeK(const double* X, const double* Ainv,
                                double* K, int N, int m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    if (row < N && col < m)
    {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) sum += X[row + k * N] * Ainv[k + col * m];
        K[row + col * N] = sum;
    }
}

__global__ void kalmanUpdateP(double* P, const double* K, const double* X,
                               int N, int m, double q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) sum += K[i + k * N] * X[j + k * N];
        double val = P[i * N + j] - sum;
        if (i == j) val += q;
        P[i * N + j] = val;
    }
}

__global__ void kalmanUpdateW(double* w, const double* K, const double* xi,
                               int N, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) sum += K[i + k * N] * xi[k];
        w[i] += sum;
    }
}

// --- Independent CPU reference for the Kalman recursion, also copied from
// ../kalman/kalman_test.cu (own hand-rolled Gauss-Jordan, separate code from
// the GPU kernel's).

static void cpuGaussJordanInvert(const vector<double>& A, vector<double>& Ainv,
                                  int m)
{
    vector<double> aug(m * 2 * m);
    int w = 2 * m;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j) aug[i * w + j] = A[i + j * m];
        for (int j = 0; j < m; ++j) aug[i * w + m + j] = (i == j) ? 1.0 : 0.0;
    }
    for (int col = 0; col < m; ++col)
    {
        int pivotRow = col;
        double best = fabs(aug[col * w + col]);
        for (int r = col + 1; r < m; ++r)
        {
            double v = fabs(aug[r * w + col]);
            if (v > best) { best = v; pivotRow = r; }
        }
        if (pivotRow != col)
            for (int j = 0; j < w; ++j)
                swap(aug[col * w + j], aug[pivotRow * w + j]);

        double piv = aug[col * w + col];
        for (int j = 0; j < w; ++j) aug[col * w + j] /= piv;

        for (int r = 0; r < m; ++r)
        {
            if (r == col) continue;
            double factor = aug[r * w + col];
            if (factor == 0.0) continue;
            for (int j = 0; j < w; ++j) aug[r * w + j] -= factor * aug[col * w + j];
        }
    }
    Ainv.assign(m * m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            Ainv[i + j * m] = aug[i * w + m + j];
}

static void cpuKalmanUpdate(vector<double>& P, vector<double>& w,
                             const vector<double>& H, const vector<double>& xi,
                             int N, int m, double eta, double q)
{
    vector<double> X(N * m, 0.0), A(m * m, 0.0), Ainv, K(N * m, 0.0);

    for (int col = 0; col < m; ++col)
        for (int row = 0; row < N; ++row)
        {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) sum += P[row * N + k] * H[k + col * N];
            X[row + col * N] = sum;
        }

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) sum += H[k + i * N] * X[k + j * N];
            A[i + j * m] = sum;
        }
    for (int i = 0; i < m; ++i) A[i + i * m] += 1.0 / eta;

    cpuGaussJordanInvert(A, Ainv, m);

    for (int col = 0; col < m; ++col)
        for (int row = 0; row < N; ++row)
        {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) sum += X[row + k * N] * Ainv[k + col * m];
            K[row + col * N] = sum;
        }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) sum += K[i + k * N] * X[j + k * N];
            double val = P[i * N + j] - sum;
            if (i == j) val += q;
            P[i * N + j] = val;
        }

    for (int i = 0; i < N; ++i)
    {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) sum += K[i + k * N] * xi[k];
        w[i] += sum;
    }
}

///////////////////////////////////////////////////////////////////////////
// Stage 6: a force-fit Kalman update -- same wiring as Stage 5, but for a
// single force COMPONENT of a single atom instead of the structure's total
// energy (this project's config also does force updates, just far less
// often: short_force_fraction 0.0041 vs. short_energy_fraction 1.0).
//
// The Jacobian this needs is calculateDFdc's output (../nn/
// nn_backward_dfdc_test.cu, already ported and validated -- copied
// verbatim below), not calculateDEdc's (Stage 5's, energy only). The
// subtlety: a force component depends on MULTIPLE atoms' networks, not just
// the target atom's own -- exactly the same "self + every neighbor that
// lists the target" structure forceAssemblyKernel already uses for the
// scalar energy->force chain rule, just contracting each contributing
// atom's OWN calculateDFdc against ITS OWN dGdxyz (its own dGdx for the
// self term, or the matching neighborDGdx slot for each neighbor-listing
// atom) instead of a scalar dEdG*dGdx product:
//   dF_T,x/dc = -sum_k d(dEdG_T[k])/dc * dGdx_T[k]                  (self,
//               only atom T itself)
//             - sum_{j : T in j's neighbor list} sum_k d(dEdG_j[k])/dc
//                                                * neighborDGdx[j,slot(T),k]
// and d(dEdG_atom[k])/dc is exactly what calculateDFdc's internal
// calculateD2EdGdc already computes per input k, before contracting with
// whatever dGdxyz array is handed to it -- so calling nnCalculateDFdc on
// contributing atom j's own G/weights with dGdxyz = neighborDGdx[j,slot(T),:]
// (instead of j's own dGdx) gives exactly j's contribution to T's force
// Jacobian, matching forceAssemblyKernel's scatter pattern but accumulating
// a full per-weight vector instead of a scalar.
///////////////////////////////////////////////////////////////////////////

// Copied verbatim from ../nn/nn_backward_dfdc_test.cu (see that file's
// header comment for the full derivation/citation). dFdc must be
// caller-zeroed; accumulated via -=.
__host__ __device__ inline void nnCalculateDFdc(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut, // numOut == 1
    const double* dGdxyz, double* dFdc)
{
    double h1[64], dfdx1[64], d2fdx2_1[64];
    for (int k = 0; k < numHidden1; ++k)
    {
        double s = b1[k];
        for (int j = 0; j < numIn; ++j) s += W1[j * numHidden1 + k] * G[j];
        h1[k] = tanh(s);
        dfdx1[k] = 1.0 - h1[k] * h1[k];
        d2fdx2_1[k] = -2.0 * h1[k] * dfdx1[k];
    }
    double h2[64], dfdx2[64], d2fdx2_2[64];
    for (int k = 0; k < numHidden2; ++k)
    {
        double s = b2[k];
        for (int j = 0; j < numHidden1; ++j) s += W2[j * numHidden2 + k] * h1[j];
        h2[k] = tanh(s);
        dfdx2[k] = 1.0 - h2[k] * h2[k];
        d2fdx2_2[k] = -2.0 * h2[k] * dfdx2[k];
    }

    double dEdbHidden2[64];
    for (int j = 0; j < numHidden2; ++j) dEdbHidden2[j] = W3[j] * dfdx2[j];

    double S1[64], dEdbHidden1[64];
    for (int i = 0; i < numHidden1; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < numHidden2; ++j) s += W2[i * numHidden2 + j] * dEdbHidden2[j];
        S1[i] = s;
        dEdbHidden1[i] = dfdx1[i] * s;
    }

    size_t const offW1 = 0;
    size_t const offB1 = offW1 + (size_t)numIn * numHidden1;
    size_t const offW2 = offB1 + numHidden1;
    size_t const offB2 = offW2 + (size_t)numHidden1 * numHidden2;
    size_t const offW3 = offB2 + numHidden2;

    for (int k0 = 0; k0 < numIn; ++k0)
    {
        double dGk0 = dGdxyz[k0];

        double dxdG1[64];
        for (int i = 0; i < numHidden1; ++i) dxdG1[i] = W1[k0 * numHidden1 + i];

        double dxdG2[64];
        for (int j = 0; j < numHidden2; ++j)
        {
            double s = 0.0;
            for (int i = 0; i < numHidden1; ++i)
                s += W2[i * numHidden2 + j] * dfdx1[i] * dxdG1[i];
            dxdG2[j] = s;
        }

        double jacBiasHidden2[64], jacW3[64];
        for (int j = 0; j < numHidden2; ++j)
        {
            jacBiasHidden2[j] = W3[j] * d2fdx2_2[j] * dxdG2[j];
            jacW3[j] = dfdx2[j] * dxdG2[j];
        }

        double jacBiasHidden1[64];
        for (int i = 0; i < numHidden1; ++i)
        {
            double s = 0.0;
            for (int j = 0; j < numHidden2; ++j) s += W2[i * numHidden2 + j] * jacBiasHidden2[j];
            jacBiasHidden1[i] = dfdx1[i] * s + d2fdx2_1[i] * dxdG1[i] * S1[i];
        }

        for (int j = 0; j < numHidden2; ++j)
        {
            dFdc[offW3 + j] -= jacW3[j] * dGk0;
            dFdc[offB2 + j] -= jacBiasHidden2[j] * dGk0;
        }
        for (int i = 0; i < numHidden1; ++i)
        {
            dFdc[offB1 + i] -= jacBiasHidden1[i] * dGk0;
            for (int j = 0; j < numHidden2; ++j)
            {
                double jacW2 = jacBiasHidden2[j] * h1[i] + dEdbHidden2[j] * dfdx1[i] * dxdG1[i];
                dFdc[offW2 + (size_t)i * numHidden2 + j] -= jacW2 * dGk0;
            }
        }
        for (int jin = 0; jin < numIn; ++jin)
        {
            double deltaTerm = (jin == k0) ? 1.0 : 0.0;
            for (int i = 0; i < numHidden1; ++i)
            {
                double jacW1 = jacBiasHidden1[i] * G[jin] + deltaTerm * dEdbHidden1[i];
                dFdc[offW1 + (size_t)jin * numHidden1 + i] -= jacW1 * dGk0;
            }
        }
    }
}

// One thread per atom j (same "self + scan own neighbor list" structure as
// forceAssemblyKernel), fixed to a single target atom/component (X) and
// contributing a full per-weight Jacobian vector via atomicAdd instead of a
// scalar force. numConnMax is a generous, compile-time-known bound (this
// file's fixed architecture caps real connection counts at 1751 for O) --
// not a size guessed from unbounded runtime data like the earlier
// neighbor-count buffer-overflow bug (see ../soa/'s and this file's own
// Stage 1/2 history) -- so a fixed local buffer here is safe.
__global__ void forceJacobianKernel(
    int numAtoms, int targetAtomSorted,
    const size_t* elementArr, const size_t* gBaseOf, const size_t* sfCountOf,
    const size_t* neighborOffset, const size_t* neighborAtomSorted,
    const size_t* neighborSfOffset,
    const double* G, const double* dGdx, const double* neighborDGdx,
    int numH, const double* W1H, const double* b1H, const double* W2H,
    const double* b2H, const double* W3H, const double* b3H,
    int numO, const double* W1O, const double* b1O, const double* W2O,
    const double* b2O, const double* W3O, const double* b3O,
    int numHidden1, int numHidden2, int numOut,
    size_t connCountH, size_t connCountO, double* jac)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numAtoms) return;

    size_t ej = elementArr[j];
    bool isH = (ej == 0);
    int numIn = isH ? numH : numO;
    const double *W1 = isH ? W1H : W1O, *b1 = isH ? b1H : b1O;
    const double *W2 = isH ? W2H : W2O, *b2 = isH ? b2H : b2O;
    const double *W3 = isH ? W3H : W3O, *b3 = isH ? b3H : b3O;
    size_t connCount = isH ? connCountH : connCountO;
    size_t jacOffset = isH ? 0 : connCountH;

    size_t gBase = gBaseOf[j];
    double dFdcLocal[2048];

    if (j == targetAtomSorted)
    {
        for (size_t c = 0; c < connCount; ++c) dFdcLocal[c] = 0.0;
        nnCalculateDFdc(&G[gBase], numIn, W1, b1, numHidden1, W2, b2, numHidden2,
                        W3, b3, numOut, &dGdx[gBase], dFdcLocal);
        for (size_t c = 0; c < connCount; ++c) atomicAdd(&jac[jacOffset + c], dFdcLocal[c]);
    }

    size_t nBegin = neighborOffset[j], nEnd = neighborOffset[j + 1];
    size_t sfBase = neighborSfOffset[j];
    size_t sfCountJ = sfCountOf[j];
    for (size_t slot = 0; slot < nEnd - nBegin; ++slot)
    {
        if (neighborAtomSorted[nBegin + slot] != (size_t)targetAtomSorted) continue;
        size_t base = sfBase + slot * sfCountJ;
        for (size_t c = 0; c < connCount; ++c) dFdcLocal[c] = 0.0;
        nnCalculateDFdc(&G[gBase], numIn, W1, b1, numHidden1, W2, b2, numHidden2,
                        W3, b3, numOut, &neighborDGdx[base], dFdcLocal);
        for (size_t c = 0; c < connCount; ++c) atomicAdd(&jac[jacOffset + c], dFdcLocal[c]);
    }
}

///////////////////////////////////////////////////////////////////////////
// input.nn parsing: real "symfunction_short" lines only, in file order.
///////////////////////////////////////////////////////////////////////////

struct RadMember { int e1; double eta, rs, rc; };
struct AngMember { int e1, e2; double eta, lambda, zeta, rc; };

vector<RadMember> parseExpRadMembers(string const& path, string const& ec,
                                      ElementMap const& elementMap)
{
    vector<RadMember> members;
    ifstream in(path);
    string line;
    while (getline(in, line))
    {
        istringstream iss(line);
        string tag;
        if (!(iss >> tag) || tag != "symfunction_short") continue;
        string ecStr; int type;
        if (!(iss >> ecStr >> type)) continue;
        if (ecStr != ec || type != 2) continue;
        string e1Str; double eta, rs, rc;
        iss >> e1Str >> eta >> rs >> rc;
        RadMember m;
        m.e1 = (int)elementMap[e1Str]; m.eta = eta; m.rs = rs; m.rc = rc;
        members.push_back(m);
    }
    return members;
}

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

    vector<RadMember> radH = parseExpRadMembers(inputNn, "H", elementMap);
    vector<RadMember> radO = parseExpRadMembers(inputNn, "O", elementMap);
    vector<AngMember> angH = parseExpAngnMembers(inputNn, "H", elementMap);
    vector<AngMember> angO = parseExpAngnMembers(inputNn, "O", elementMap);
    int numRadH = (int)radH.size(), numAngH = (int)angH.size();
    int numRadO = (int)radO.size(), numAngO = (int)angO.size();
    int numH = numRadH + numAngH, numO = numRadO + numAngO;
    printf("Parsed real instances -- H: %d radial + %d angular = %d; O: %d radial + %d angular = %d\n",
           numRadH, numAngH, numH, numRadO, numAngO, numO);

    allocateSfStorage(batch, {(size_t)numH, (size_t)numO});
    printf("AtomBatch: %zu atoms (%zu H, %zu O), %zu neighbor entries\n\n",
           batch.numAtoms, batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.elementOffset[O + 1] - batch.elementOffset[O],
           batch.totalNeighbors());

    vector<int> neighOffsetInt(batch.neighborOffset.begin(), batch.neighborOffset.end());
    vector<int> neighElemInt(batch.neighborElement.begin(), batch.neighborElement.end());

    // Two real NeuralNetworks, one per element, architecture sized to the
    // FULL real instance count (35/42 -> 25 -> 25 -> 1), random weights (no
    // trained weights available).
    int const numHidden1 = 25, numHidden2 = 25, numOut = 1, numLayers = 4;
    NeuralNetwork::ActivationFunction af[4] = {
        NeuralNetwork::AF_IDENTITY, NeuralNetwork::AF_TANH,
        NeuralNetwork::AF_TANH, NeuralNetwork::AF_IDENTITY};

    int nnH_layers[4] = {numH, numHidden1, numHidden2, numOut};
    NeuralNetwork nnH(numLayers, nnH_layers, af);
    nnH.initializeConnectionsRandomUniform(42);
    vector<double> connH(nnH.getNumConnections());
    nnH.getConnections(connH.data());

    int nnO_layers[4] = {numO, numHidden1, numHidden2, numOut};
    NeuralNetwork nnO(numLayers, nnO_layers, af);
    nnO.initializeConnectionsRandomUniform(43);
    vector<double> connO(nnO.getNumConnections());
    nnO.getConnections(connO.data());

    auto sliceConn = [&](vector<double>& conn, int numIn,
                          double const*& W1, double const*& b1,
                          double const*& W2, double const*& b2,
                          double const*& W3, double const*& b3)
    {
        size_t off = 0;
        W1 = conn.data() + off; off += (size_t)numIn * numHidden1;
        b1 = conn.data() + off; off += numHidden1;
        W2 = conn.data() + off; off += (size_t)numHidden1 * numHidden2;
        b2 = conn.data() + off; off += numHidden2;
        W3 = conn.data() + off; off += (size_t)numHidden2 * numOut;
        b3 = conn.data() + off; off += numOut;
    };
    double const *W1H, *b1H, *W2H, *b2H, *W3H, *b3H;
    double const *W1O, *b1O, *W2O, *b2O, *W3O, *b3O;
    sliceConn(connH, numH, W1H, b1H, W2H, b2H, W3H, b3H);
    sliceConn(connO, numO, W1O, b1O, W2O, b2O, W3O, b3O);

    // --- Per-atom convenience arrays for the force-assembly stage -----------
    size_t numAtoms = batch.numAtoms;
    vector<size_t> gBaseOf(numAtoms), sfCountOf(numAtoms);
    for (size_t s = 0; s < numAtoms; ++s)
    {
        gBaseOf[s] = batch.gIndex(s, 0);
        sfCountOf[s] = batch.sfCountPerElement[batch.element[s]];
    }

    ///////////////////////////////////////////////////////////////////////
    // GPU pipeline: everything stays on device across the four stages.
    ///////////////////////////////////////////////////////////////////////
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

    double *d_G, *d_dGdx, *d_dGdy, *d_dGdz, *d_dEdG, *d_energy;
    double *d_nDGdx, *d_nDGdy, *d_nDGdz;
    CUDA_CHECK(cudaMalloc(&d_G, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdx, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdy, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdz, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dEdG, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdx, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdy, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdz, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_G, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdx, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdy, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdz, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dEdG, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdx, 0, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdy, 0, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdz, 0, batch.neighborDGdx.size() * sizeof(double)));

    double *d_forceX, *d_forceY, *d_forceZ;
    CUDA_CHECK(cudaMalloc(&d_forceX, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forceY, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forceZ, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceX, 0, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceY, 0, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceZ, 0, numAtoms * sizeof(double)));

    size_t* d_gBaseOf; size_t* d_sfCountOf;
    CUDA_CHECK(cudaMalloc(&d_gBaseOf, numAtoms * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_sfCountOf, numAtoms * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_gBaseOf, gBaseOf.data(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfCountOf, sfCountOf.data(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t* d_neighborAtomSorted;
    CUDA_CHECK(cudaMalloc(&d_neighborAtomSorted, batch.neighborAtomSorted.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_neighborAtomSorted, batch.neighborAtomSorted.data(), batch.neighborAtomSorted.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t* d_neighborOffsetSizeT;
    CUDA_CHECK(cudaMalloc(&d_neighborOffsetSizeT, batch.neighborOffset.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_neighborOffsetSizeT, batch.neighborOffset.data(), batch.neighborOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    // Per-element member arrays.
    auto uploadRad = [](vector<RadMember> const& members,
                         int** d_e1, double** d_eta, double** d_rs)
    {
        int n = (int)members.size();
        vector<int> e1(n); vector<double> eta(n), rs(n);
        for (int k = 0; k < n; ++k) { e1[k] = members[k].e1; eta[k] = members[k].eta; rs[k] = members[k].rs; }
        CUDA_CHECK(cudaMalloc(d_e1, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(d_eta, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(d_rs, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(*d_e1, e1.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_eta, eta.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_rs, rs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    };
    auto uploadAng = [](vector<AngMember> const& members,
                         int** d_e1, int** d_e2, double** d_eta, double** d_lambda, double** d_zeta)
    {
        int n = (int)members.size();
        vector<int> e1(n), e2(n); vector<double> eta(n), lambda(n), zeta(n);
        for (int k = 0; k < n; ++k)
        {
            e1[k] = members[k].e1; e2[k] = members[k].e2;
            eta[k] = members[k].eta; lambda[k] = members[k].lambda; zeta[k] = members[k].zeta;
        }
        CUDA_CHECK(cudaMalloc(d_e1, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(d_e2, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(d_eta, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(d_lambda, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(d_zeta, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(*d_e1, e1.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_e2, e2.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_eta, eta.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_lambda, lambda.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_zeta, zeta.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    };
    int *d_e1RadH, *d_e1RadO; double *d_etaRadH, *d_rsRadH, *d_etaRadO, *d_rsRadO;
    uploadRad(radH, &d_e1RadH, &d_etaRadH, &d_rsRadH);
    uploadRad(radO, &d_e1RadO, &d_etaRadO, &d_rsRadO);
    int *d_e1AngH, *d_e2AngH; double *d_etaAngH, *d_lambdaAngH, *d_zetaAngH;
    int *d_e1AngO, *d_e2AngO; double *d_etaAngO, *d_lambdaAngO, *d_zetaAngO;
    uploadAng(angH, &d_e1AngH, &d_e2AngH, &d_etaAngH, &d_lambdaAngH, &d_zetaAngH);
    uploadAng(angO, &d_e1AngO, &d_e2AngO, &d_etaAngO, &d_lambdaAngO, &d_zetaAngO);

    int blockSize = 128;
    int beginH = (int)batch.elementOffset[H], endH = (int)batch.elementOffset[H + 1];
    int beginO = (int)batch.elementOffset[O], endO = (int)batch.elementOffset[O + 1];
    int numSelH = endH - beginH, numSelO = endO - beginO;

    // --- Stage 1: radial symmetry functions ----------------------------------
    sfGroupKernelReal<<<(numSelH + blockSize - 1) / blockSize, blockSize>>>(
        beginH, numSelH, rc, numRadH, 0, d_e1RadH, d_etaRadH, d_rsRadH,
        d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy, d_neighDz,
        d_G, d_dGdx, d_dGdy, d_dGdz, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
    CUDA_CHECK(cudaGetLastError());
    sfGroupKernelReal<<<(numSelO + blockSize - 1) / blockSize, blockSize>>>(
        beginO, numSelO, rc, numRadO, 0, d_e1RadO, d_etaRadO, d_rsRadO,
        d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy, d_neighDz,
        d_G, d_dGdx, d_dGdy, d_dGdz, batch.gBlockOffset[O], batch.sfCountPerElement[O],
        d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
    CUDA_CHECK(cudaGetLastError());

    // --- Stage 2: angular symmetry functions ---------------------------------
    sfAngnGroupKernel<<<(numSelH + blockSize - 1) / blockSize, blockSize>>>(
        beginH, numSelH, rc, numAngH, numRadH, d_e1AngH, d_e2AngH, d_etaAngH, d_lambdaAngH, d_zetaAngH,
        d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy, d_neighDz,
        d_G, d_dGdx, d_dGdy, d_dGdz, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
    CUDA_CHECK(cudaGetLastError());
    sfAngnGroupKernel<<<(numSelO + blockSize - 1) / blockSize, blockSize>>>(
        beginO, numSelO, rc, numAngO, numRadO, d_e1AngO, d_e2AngO, d_etaAngO, d_lambdaAngO, d_zetaAngO,
        d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy, d_neighDz,
        d_G, d_dGdx, d_dGdy, d_dGdz, batch.gBlockOffset[O], batch.sfCountPerElement[O],
        d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Stage 3: NN forward + dEdG ------------------------------------------
    double *d_W1H, *d_b1H, *d_W2H, *d_b2H, *d_W3H, *d_b3H;
    CUDA_CHECK(cudaMalloc(&d_W1H, numH * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1H, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2H, numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2H, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3H, numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b3H, numOut * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_W1H, W1H, numH * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1H, b1H, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2H, W2H, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2H, b2H, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3H, W3H, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3H, b3H, numOut * sizeof(double), cudaMemcpyHostToDevice));

    double *d_W1O, *d_b1O, *d_W2O, *d_b2O, *d_W3O, *d_b3O;
    CUDA_CHECK(cudaMalloc(&d_W1O, numO * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1O, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2O, numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2O, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3O, numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b3O, numOut * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_W1O, W1O, numO * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1O, b1O, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2O, W2O, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2O, b2O, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3O, W3O, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3O, b3O, numOut * sizeof(double), cudaMemcpyHostToDevice));

    nnForwardKernel<<<(numSelH + blockSize - 1) / blockSize, blockSize>>>(
        beginH, numSelH, numH, d_W1H, d_b1H, numHidden1, d_W2H, d_b2H, numHidden2,
        d_W3H, d_b3H, numOut, d_G, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_energy, d_dEdG);
    CUDA_CHECK(cudaGetLastError());

    nnForwardKernel<<<(numSelO + blockSize - 1) / blockSize, blockSize>>>(
        beginO, numSelO, numO, d_W1O, d_b1O, numHidden1, d_W2O, d_b2O, numHidden2,
        d_W3O, d_b3O, numOut, d_G, batch.gBlockOffset[O], batch.sfCountPerElement[O],
        d_energy, d_dEdG);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Stage 4: force assembly ---------------------------------------------
    forceAssemblyKernel<<<((int)numAtoms + blockSize - 1) / blockSize, blockSize>>>(
        (int)numAtoms, d_neighborOffsetSizeT, d_neighborAtomSorted, d_neighborSfOffset,
        d_gBaseOf, d_sfCountOf, d_dEdG, d_dGdx, d_dGdy, d_dGdz,
        d_nDGdx, d_nDGdy, d_nDGdz, d_forceX, d_forceY, d_forceZ);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Download final results ----------------------------------------------
    vector<double> energyGpu(numAtoms), forceXGpu(numAtoms), forceYGpu(numAtoms), forceZGpu(numAtoms);
    CUDA_CHECK(cudaMemcpy(energyGpu.data(), d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forceXGpu.data(), d_forceX, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forceYGpu.data(), d_forceY, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forceZGpu.data(), d_forceZ, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    vector<double> gGpu(batch.G.size());
    CUDA_CHECK(cudaMemcpy(gGpu.data(), d_G, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));

    ///////////////////////////////////////////////////////////////////////
    // Independent CPU reference: same real geometry/parameters/weights,
    // computed with no device involvement at all.
    ///////////////////////////////////////////////////////////////////////
    vector<double> gCpu(batch.G.size(), 0.0);
    vector<double> dGdxCpu(batch.G.size(), 0.0), dGdyCpu(batch.G.size(), 0.0), dGdzCpu(batch.G.size(), 0.0);
    vector<double> nDGdxCpu(batch.neighborDGdx.size(), 0.0);
    vector<double> nDGdyCpu(batch.neighborDGdx.size(), 0.0);
    vector<double> nDGdzCpu(batch.neighborDGdx.size(), 0.0);

    auto runRadCpu = [&](int begin, int end, vector<RadMember> const& members, int sfIndexOffset)
    {
        int numMembers = (int)members.size();
        if (numMembers == 0) return;
        vector<int> e1(numMembers); vector<double> eta(numMembers), rsv(numMembers);
        for (int k = 0; k < numMembers; ++k) { e1[k] = members[k].e1; eta[k] = members[k].eta; rsv[k] = members[k].rs; }
        for (int i = begin; i < end; ++i)
        {
            int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
            vector<double> result(numMembers, 0.0), dResultX(numMembers, 0.0), dResultY(numMembers, 0.0), dResultZ(numMembers, 0.0);
            size_t sfCountE = batch.sfCountPerElement[batch.element[i]];
            symFncExpRadGroupReal(n, &neighElemInt[off], &batch.neighborD[off],
                                  &batch.neighborDx[off], &batch.neighborDy[off], &batch.neighborDz[off],
                                  rc, numMembers, e1.data(), eta.data(), rsv.data(),
                                  result.data(), dResultX.data(), dResultY.data(), dResultZ.data(),
                                  &nDGdxCpu[batch.neighborSfOffset[i]], &nDGdyCpu[batch.neighborSfOffset[i]],
                                  &nDGdzCpu[batch.neighborSfOffset[i]], sfCountE, (size_t)sfIndexOffset);
            size_t base = batch.gIndex(i, 0) + sfIndexOffset;
            for (int k = 0; k < numMembers; ++k)
            {
                gCpu[base + k] = result[k];
                dGdxCpu[base + k] = dResultX[k];
                dGdyCpu[base + k] = dResultY[k];
                dGdzCpu[base + k] = dResultZ[k];
            }
        }
    };
    auto runAngCpu = [&](int begin, int end, vector<AngMember> const& members, int sfIndexOffset)
    {
        int numMembers = (int)members.size();
        if (numMembers == 0) return;
        vector<int> e1(numMembers), e2(numMembers);
        vector<double> eta(numMembers), lambda(numMembers), zeta(numMembers);
        for (int k = 0; k < numMembers; ++k)
        {
            e1[k] = members[k].e1; e2[k] = members[k].e2;
            eta[k] = members[k].eta; lambda[k] = members[k].lambda; zeta[k] = members[k].zeta;
        }
        for (int i = begin; i < end; ++i)
        {
            int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
            vector<double> result(numMembers, 0.0), dResultX(numMembers, 0.0), dResultY(numMembers, 0.0), dResultZ(numMembers, 0.0);
            size_t sfCountE = batch.sfCountPerElement[batch.element[i]];
            symFncExpAngnGroupReal(n, &neighElemInt[off], &batch.neighborD[off],
                                   &batch.neighborDx[off], &batch.neighborDy[off], &batch.neighborDz[off],
                                   rc, numMembers, e1.data(), e2.data(), eta.data(), lambda.data(), zeta.data(),
                                   result.data(), dResultX.data(), dResultY.data(), dResultZ.data(),
                                   &nDGdxCpu[batch.neighborSfOffset[i]], &nDGdyCpu[batch.neighborSfOffset[i]],
                                   &nDGdzCpu[batch.neighborSfOffset[i]], sfCountE, (size_t)sfIndexOffset);
            size_t base = batch.gIndex(i, 0) + sfIndexOffset;
            for (int k = 0; k < numMembers; ++k)
            {
                gCpu[base + k] = result[k];
                dGdxCpu[base + k] = dResultX[k];
                dGdyCpu[base + k] = dResultY[k];
                dGdzCpu[base + k] = dResultZ[k];
            }
        }
    };
    runRadCpu(beginH, endH, radH, 0);
    runRadCpu(beginO, endO, radO, 0);
    runAngCpu(beginH, endH, angH, numRadH);
    runAngCpu(beginO, endO, angO, numRadO);

    vector<double> energyCpu(numAtoms), dEdGCpu(batch.G.size(), 0.0);
    // Stage 5's real energy-weight Jacobian, accumulated per element while
    // we're already looping over atoms with a fresh propagate() (calculateDEdc
    // reads the same layer activations calculateDEdG does -- neither mutates
    // them, so calling both after one propagate() is safe).
    vector<double> jacHCpu(connH.size(), 0.0), jacOCpu(connO.size(), 0.0);
    vector<double> dEdcAtomBuf;
    for (int t = 0; t < numSelH; ++t)
    {
        int s = beginH + t;
        nnH.setInput(&gCpu[batch.gIndex(s, 0)]);
        nnH.propagate();
        nnH.calculateDEdG(&dEdGCpu[batch.gIndex(s, 0)]);
        nnH.getOutput(&energyCpu[s]);
        dEdcAtomBuf.assign(connH.size(), 0.0);
        nnH.calculateDEdc(dEdcAtomBuf.data());
        for (size_t c = 0; c < connH.size(); ++c) jacHCpu[c] += dEdcAtomBuf[c];
    }
    for (int t = 0; t < numSelO; ++t)
    {
        int s = beginO + t;
        nnO.setInput(&gCpu[batch.gIndex(s, 0)]);
        nnO.propagate();
        nnO.calculateDEdG(&dEdGCpu[batch.gIndex(s, 0)]);
        nnO.getOutput(&energyCpu[s]);
        dEdcAtomBuf.assign(connO.size(), 0.0);
        nnO.calculateDEdc(dEdcAtomBuf.data());
        for (size_t c = 0; c < connO.size(); ++c) jacOCpu[c] += dEdcAtomBuf[c];
    }

    // Force assembly, CPU "gather" (independent traversal order from the
    // GPU's scatter, same as ../force/force_assembly_test.cu).
    vector<double> forceXCpu(numAtoms, 0.0), forceYCpu(numAtoms, 0.0), forceZCpu(numAtoms, 0.0);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        size_t gBase = gBaseOf[i], sfCount = sfCountOf[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (size_t k = 0; k < sfCount; ++k)
        {
            double dedg = dEdGCpu[gBase + k];
            fx -= dedg * dGdxCpu[gBase + k];
            fy -= dedg * dGdyCpu[gBase + k];
            fz -= dedg * dGdzCpu[gBase + k];
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
                    double dedg = dEdGCpu[gBaseJ + k];
                    fx -= dedg * nDGdxCpu[base + k];
                    fy -= dedg * nDGdyCpu[base + k];
                    fz -= dedg * nDGdzCpu[base + k];
                }
            }
        }
        forceXCpu[i] = fx; forceYCpu[i] = fy; forceZCpu[i] = fz;
    }

    ///////////////////////////////////////////////////////////////////////
    // Compare
    ///////////////////////////////////////////////////////////////////////
    double maxAbsErrG = 0.0, maxAbsErrE = 0.0, maxAbsErrF = 0.0;
    for (size_t k = 0; k < batch.G.size(); ++k)
        maxAbsErrG = max(maxAbsErrG, fabs(gGpu[k] - gCpu[k]));

    double sumEnergyGpu = 0.0, sumEnergyCpu = 0.0;
    for (size_t i = 0; i < numAtoms; ++i)
    {
        maxAbsErrE = max(maxAbsErrE, fabs(energyGpu[i] - energyCpu[i]));
        maxAbsErrF = max({maxAbsErrF, fabs(forceXGpu[i] - forceXCpu[i]),
                           fabs(forceYGpu[i] - forceYCpu[i]), fabs(forceZGpu[i] - forceZCpu[i])});
        sumEnergyGpu += energyGpu[i];
        sumEnergyCpu += energyCpu[i];
    }

    printf("Total energy: gpu=%.15E cpu=%.15E\n", sumEnergyGpu, sumEnergyCpu);
    printf("force[0]: gpu=(%.15E,%.15E,%.15E) cpu=(%.15E,%.15E,%.15E)\n",
           forceXGpu[0], forceYGpu[0], forceZGpu[0], forceXCpu[0], forceYCpu[0], forceZCpu[0]);
    printf("max|G_gpu-G_cpu|      = %.3E  (%zu values)\n", maxAbsErrG, batch.G.size());
    printf("max|E_gpu-E_cpu|      = %.3E  (%zu atoms)\n", maxAbsErrE, numAtoms);
    printf("max|F_gpu-F_cpu|      = %.3E  (%zu atoms)\n", maxAbsErrF, numAtoms);

    bool pass = (maxAbsErrG < 1e-9) && (maxAbsErrE < 1e-9) && (maxAbsErrF < 1e-9);
    printf("%s\n", pass ? "PASS" : "FAIL");

    ///////////////////////////////////////////////////////////////////////
    // Stage 5: real per-structure Kalman ENERGY update (see the kernels'
    // header comment above for the full derivation).
    ///////////////////////////////////////////////////////////////////////
    size_t connCountH = connH.size(), connCountO = connO.size();
    allocateWeightJacobianStorage(batch, {connCountH, connCountO});

    double* d_dEdc;
    CUDA_CHECK(cudaMalloc(&d_dEdc, batch.dFdc.size() * sizeof(double)));

    nnDEdcKernel<<<(numSelH + blockSize - 1) / blockSize, blockSize>>>(
        beginH, numSelH, numH, d_W1H, d_b1H, numHidden1, d_W2H, d_b2H, numHidden2,
        d_W3H, d_b3H, numOut, d_G, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_dEdc, batch.dFdcBlockOffset[H], batch.connCountPerElement[H]);
    CUDA_CHECK(cudaGetLastError());
    nnDEdcKernel<<<(numSelO + blockSize - 1) / blockSize, blockSize>>>(
        beginO, numSelO, numO, d_W1O, d_b1O, numHidden1, d_W2O, d_b2O, numHidden2,
        d_W3O, d_b3O, numOut, d_G, batch.gBlockOffset[O], batch.sfCountPerElement[O],
        d_dEdc, batch.dFdcBlockOffset[O], batch.connCountPerElement[O]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(batch.dFdc.data(), d_dEdc, batch.dFdc.size() * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_dEdc);

    // Total structure energy Jacobian = per-atom dEdc summed over that
    // element's atoms (every atom of an element shares that element's
    // weights, same reasoning as the total energy itself being a per-atom sum).
    vector<double> jacH(connCountH, 0.0), jacO(connCountO, 0.0);
    for (int t = 0; t < numSelH; ++t)
    {
        size_t s = beginH + t;
        for (size_t c = 0; c < connCountH; ++c) jacH[c] += batch.dFdc[batch.dFdcIndex(s, c)];
    }
    for (int t = 0; t < numSelO; ++t)
    {
        size_t s = beginO + t;
        for (size_t c = 0; c < connCountO; ++c) jacO[c] += batch.dFdc[batch.dFdcIndex(s, c)];
    }

    double maxAbsErrJac = 0.0;
    for (size_t c = 0; c < connCountH; ++c) maxAbsErrJac = max(maxAbsErrJac, fabs(jacH[c] - jacHCpu[c]));
    for (size_t c = 0; c < connCountO; ++c) maxAbsErrJac = max(maxAbsErrJac, fabs(jacO[c] - jacOCpu[c]));

    int N = (int)(connCountH + connCountO), m = 1;
    vector<double> Hcol(N), w0(N);
    for (size_t c = 0; c < connCountH; ++c) { Hcol[c] = jacH[c]; w0[c] = connH[c]; }
    for (size_t c = 0; c < connCountO; ++c) { Hcol[connCountH + c] = jacO[c]; w0[connCountH + c] = connO[c]; }

    double energyRef = structure.energyRef;
    vector<double> xi(1, energyRef - sumEnergyGpu);

    // Production KalmanFilter parameters (temp/H2O_2G/input.nn:79-86,
    // kalman_type 0 = KT_STANDARD), same as ../kalman/kalman_test.cu.
    double const epsilon = 1.0e-2, q0 = 0.01, qtau = 2.302, qmin = 1.0e-6;
    double const eta0 = 0.01, etatau = 2.302, etamax = 1.0;
    double eta = eta0, q = q0;
    if (eta < etamax) eta *= exp(etatau); // first-call schedule, see kalman_test.cu

    vector<double> P0(N * N, 0.0);
    for (int i = 0; i < N; ++i) P0[i * N + i] = 1.0 / epsilon;

    // --- GPU Kalman update ---------------------------------------------------
    double *d_P, *d_Hk, *d_xi, *d_X, *d_A, *d_Ainv, *d_Aug, *d_K, *d_w;
    CUDA_CHECK(cudaMalloc(&d_P, sizeof(double) * N * N));
    CUDA_CHECK(cudaMalloc(&d_Hk, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_xi, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_Ainv, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_Aug, sizeof(double) * m * 2 * m));
    CUDA_CHECK(cudaMalloc(&d_K, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(double) * N));
    CUDA_CHECK(cudaMemcpy(d_P, P0.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Hk, Hcol.data(), sizeof(double) * N * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xi, xi.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w0.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    dim3 kBlockX(256), kGridX((N + 255) / 256, m);
    dim3 kBlockA(m, m);
    dim3 kBlockP(16, 16), kGridP((N + 15) / 16, (N + 15) / 16);
    dim3 kBlockW(256), kGridW((N + 255) / 256);

    kalmanComputeX<<<kGridX, kBlockX>>>(d_P, d_Hk, d_X, N, m);
    kalmanComputeA<<<1, kBlockA>>>(d_Hk, d_X, d_A, N, m);
    kalmanAddDiag<<<(m + 31) / 32, 32>>>(d_A, m, 1.0 / eta);
    kalmanInvertGaussJordan<<<1, 1>>>(d_A, d_Ainv, d_Aug, m);
    kalmanComputeK<<<kGridX, kBlockX>>>(d_X, d_Ainv, d_K, N, m);
    kalmanUpdateP<<<kGridP, kBlockP>>>(d_P, d_K, d_X, N, m, q);
    kalmanUpdateW<<<kGridW, kBlockW>>>(d_w, d_K, d_xi, N, m);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> wNewGpu(N);
    CUDA_CHECK(cudaMemcpy(wNewGpu.data(), d_w, sizeof(double) * N, cudaMemcpyDeviceToHost));

    cudaFree(d_P); cudaFree(d_Hk); cudaFree(d_xi); cudaFree(d_X);
    cudaFree(d_A); cudaFree(d_Ainv); cudaFree(d_Aug); cudaFree(d_K); cudaFree(d_w);

    // --- Independent CPU Kalman update (own Gauss-Jordan, own loops) --------
    vector<double> HcolCpu(N), w0Cpu(N);
    for (size_t c = 0; c < connCountH; ++c) { HcolCpu[c] = jacHCpu[c]; w0Cpu[c] = connH[c]; }
    for (size_t c = 0; c < connCountO; ++c) { HcolCpu[connCountH + c] = jacOCpu[c]; w0Cpu[connCountH + c] = connO[c]; }
    vector<double> xiCpu(1, energyRef - sumEnergyCpu);

    vector<double> Pcpu = P0;
    vector<double> wCpu = w0Cpu;
    cpuKalmanUpdate(Pcpu, wCpu, HcolCpu, xiCpu, N, m, eta, q);

    // --- Real nnp::KalmanFilter cross-check (linked from lib/libnnptrain.a,
    // debug info stripped at link time -- see run.slurm) ---------------------
    vector<double> wReal = w0Cpu;
    KalmanFilter kf(N, KalmanFilter::KT_STANDARD);
    kf.setState(wReal.data());
    kf.setParametersStandard(epsilon, q0, qtau, qmin, eta0, etatau, etamax);
    kf.setJacobian(HcolCpu.data(), m);
    kf.setError(xiCpu.data(), m);
    kf.update(m);

    double maxAbsErrWGpuCpu = 0.0, maxAbsErrWGpuReal = 0.0;
    for (int i = 0; i < N; ++i)
    {
        maxAbsErrWGpuCpu = max(maxAbsErrWGpuCpu, fabs(wNewGpu[i] - wCpu[i]));
        maxAbsErrWGpuReal = max(maxAbsErrWGpuReal, fabs(wNewGpu[i] - wReal[i]));
    }

    printf("\n--- Stage 5: Kalman energy update (N=%d combined weights) ---\n", N);
    printf("energyRef=%.10E  energy(before)=gpu:%.10E cpu:%.10E\n",
           energyRef, sumEnergyGpu, sumEnergyCpu);
    printf("max|dEdc_gpu-dEdc_cpu| = %.3E\n", maxAbsErrJac);
    printf("max|w_gpu-w_cpu|       = %.3E\n", maxAbsErrWGpuCpu);
    printf("max|w_gpu-w_real|      = %.3E\n", maxAbsErrWGpuReal);

    // --- Apply the updated weights and re-run the forward pass, purely
    // informationally: a single Kalman step from this random initial
    // weights/P/eta is not guaranteed to monotonically reduce the energy
    // error, so this is NOT a pass/fail criterion -- just a demonstration
    // that the update composes end-to-end into a new, usable network.
    vector<double> connHNew(wNewGpu.begin(), wNewGpu.begin() + connCountH);
    vector<double> connONew(wNewGpu.begin() + connCountH, wNewGpu.end());

    double const *W1Hn, *b1Hn, *W2Hn, *b2Hn, *W3Hn, *b3Hn;
    double const *W1On, *b1On, *W2On, *b2On, *W3On, *b3On;
    sliceConn(connHNew, numH, W1Hn, b1Hn, W2Hn, b2Hn, W3Hn, b3Hn);
    sliceConn(connONew, numO, W1On, b1On, W2On, b2On, W3On, b3On);
    CUDA_CHECK(cudaMemcpy(d_W1H, W1Hn, numH * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1H, b1Hn, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2H, W2Hn, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2H, b2Hn, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3H, W3Hn, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3H, b3Hn, numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1O, W1On, numO * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1O, b1On, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2O, W2On, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2O, b2On, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3O, W3On, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3O, b3On, numOut * sizeof(double), cudaMemcpyHostToDevice));

    nnForwardKernel<<<(numSelH + blockSize - 1) / blockSize, blockSize>>>(
        beginH, numSelH, numH, d_W1H, d_b1H, numHidden1, d_W2H, d_b2H, numHidden2,
        d_W3H, d_b3H, numOut, d_G, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_energy, d_dEdG);
    CUDA_CHECK(cudaGetLastError());
    nnForwardKernel<<<(numSelO + blockSize - 1) / blockSize, blockSize>>>(
        beginO, numSelO, numO, d_W1O, d_b1O, numHidden1, d_W2O, d_b2O, numHidden2,
        d_W3O, d_b3O, numOut, d_G, batch.gBlockOffset[O], batch.sfCountPerElement[O],
        d_energy, d_dEdG);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> energyGpuNew(numAtoms);
    CUDA_CHECK(cudaMemcpy(energyGpuNew.data(), d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    double sumEnergyGpuNew = 0.0;
    for (size_t i = 0; i < numAtoms; ++i) sumEnergyGpuNew += energyGpuNew[i];

    nnH.setConnections(connHNew.data());
    nnO.setConnections(connONew.data());
    double sumEnergyCpuNew = 0.0;
    for (int t = 0; t < numSelH; ++t)
    {
        int s = beginH + t;
        nnH.setInput(&gCpu[batch.gIndex(s, 0)]);
        nnH.propagate();
        double e; nnH.getOutput(&e);
        sumEnergyCpuNew += e;
    }
    for (int t = 0; t < numSelO; ++t)
    {
        int s = beginO + t;
        nnO.setInput(&gCpu[batch.gIndex(s, 0)]);
        nnO.propagate();
        double e; nnO.getOutput(&e);
        sumEnergyCpuNew += e;
    }

    printf("energy(after) =gpu:%.10E cpu:%.10E  (informational only, not a pass/fail check)\n",
           sumEnergyGpuNew, sumEnergyCpuNew);

    bool pass5 = (maxAbsErrJac < 1e-9) && (maxAbsErrWGpuCpu < 1e-8) &&
                 (maxAbsErrWGpuReal < 1e-8);
    printf("Stage 5: %s\n", pass5 ? "PASS" : "FAIL");

    ///////////////////////////////////////////////////////////////////////
    // Stage 6: force-fit Kalman update, same wiring as Stage 5, using
    // calculateDFdc instead of calculateDEdc. Runs independently of Stage 5
    // (not chained after it) -- first reset the weights Stage 5 mutated
    // (nnH/nnO's connections and the GPU weight buffers) back to the
    // original random weights, so Stage 6 starts from the same baseline
    // Stage 5 did, using connH/connO (never mutated -- Stage 5 only ever
    // wrote its updated weights into separate connHNew/connONew vectors).
    ///////////////////////////////////////////////////////////////////////
    nnH.setConnections(connH.data());
    nnO.setConnections(connO.data());
    CUDA_CHECK(cudaMemcpy(d_W1H, W1H, numH * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1H, b1H, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2H, W2H, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2H, b2H, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3H, W3H, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3H, b3H, numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1O, W1O, numO * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1O, b1O, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2O, W2O, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2O, b2O, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3O, W3O, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3O, b3O, numOut * sizeof(double), cudaMemcpyHostToDevice));

    // Target: sorted atom 0's x-component -- mirrors task_batch_size_force 1
    // (one force component per scheduled update).
    int const targetAtom = 0;
    double fRefX = structure.atoms[batch.sortedToOriginal[targetAtom]].fRef[0];

    vector<size_t> elementArr(numAtoms);
    for (size_t s = 0; s < numAtoms; ++s) elementArr[s] = batch.element[s];
    size_t* d_elementArr;
    CUDA_CHECK(cudaMalloc(&d_elementArr, numAtoms * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_elementArr, elementArr.data(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice));

    double* d_jacForce;
    CUDA_CHECK(cudaMalloc(&d_jacForce, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_jacForce, 0, N * sizeof(double)));

    forceJacobianKernel<<<((int)numAtoms + blockSize - 1) / blockSize, blockSize>>>(
        (int)numAtoms, targetAtom, d_elementArr, d_gBaseOf, d_sfCountOf,
        d_neighborOffsetSizeT, d_neighborAtomSorted, d_neighborSfOffset,
        d_G, d_dGdx, d_nDGdx,
        numH, d_W1H, d_b1H, d_W2H, d_b2H, d_W3H, d_b3H,
        numO, d_W1O, d_b1O, d_W2O, d_b2O, d_W3O, d_b3O,
        numHidden1, numHidden2, numOut, connCountH, connCountO, d_jacForce);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> jacForceGpu(N);
    CUDA_CHECK(cudaMemcpy(jacForceGpu.data(), d_jacForce, N * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_jacForce);
    cudaFree(d_elementArr);

    // --- Independent CPU Jacobian: real nnH/nnO.calculateDFdc(), same
    // self+neighbor-scan gather already used for forceXCpu above. ---------
    vector<double> jacForceCpu(N, 0.0);
    {
        size_t s = targetAtom;
        size_t e = batch.element[s];
        size_t connCountS = (e == (size_t)H) ? connCountH : connCountO;
        size_t offS = (e == (size_t)H) ? 0 : connCountH;
        vector<double> dFdcLocal(connCountS, 0.0);
        NeuralNetwork& nn = (e == (size_t)H) ? nnH : nnO;
        nn.setInput(&gCpu[batch.gIndex(s, 0)]);
        nn.propagate();
        nn.calculateDFdc(dFdcLocal.data(), &dGdxCpu[batch.gIndex(s, 0)]);
        for (size_t c = 0; c < connCountS; ++c) jacForceCpu[offS + c] += dFdcLocal[c];
    }
    for (size_t j = 0; j < numAtoms; ++j)
    {
        size_t nBegin = batch.neighborOffset[j], nEnd = batch.neighborOffset[j + 1];
        size_t sfBaseJ = batch.neighborSfOffset[j], sfCountJ = sfCountOf[j];
        size_t ej = batch.element[j];
        size_t connCountJ = (ej == (size_t)H) ? connCountH : connCountO;
        size_t offJ = (ej == (size_t)H) ? 0 : connCountH;
        NeuralNetwork& nnJ = (ej == (size_t)H) ? nnH : nnO;
        for (size_t slot = 0; slot < nEnd - nBegin; ++slot)
        {
            if (batch.neighborAtomSorted[nBegin + slot] != (size_t)targetAtom) continue;
            size_t base = sfBaseJ + slot * sfCountJ;
            vector<double> dFdcLocal(connCountJ, 0.0);
            nnJ.setInput(&gCpu[batch.gIndex(j, 0)]);
            nnJ.propagate();
            nnJ.calculateDFdc(dFdcLocal.data(), &nDGdxCpu[base]);
            for (size_t c = 0; c < connCountJ; ++c) jacForceCpu[offJ + c] += dFdcLocal[c];
        }
    }

    double maxAbsErrJacForce = 0.0;
    for (int c = 0; c < N; ++c)
        maxAbsErrJacForce = max(maxAbsErrJacForce, fabs(jacForceGpu[c] - jacForceCpu[c]));

    // --- Kalman force update: GPU, independent CPU, and the real
    // nnp::KalmanFilter class, exactly like Stage 5 but with this Jacobian. --
    vector<double> xiForce(1, fRefX - forceXGpu[targetAtom]);
    double etaF = eta0, qF = q0;
    if (etaF < etamax) etaF *= exp(etatau);

    double *d_P6, *d_Hk6, *d_xi6, *d_X6, *d_A6, *d_Ainv6, *d_Aug6, *d_K6, *d_w6;
    CUDA_CHECK(cudaMalloc(&d_P6, sizeof(double) * N * N));
    CUDA_CHECK(cudaMalloc(&d_Hk6, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_xi6, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_X6, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_A6, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_Ainv6, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_Aug6, sizeof(double) * m * 2 * m));
    CUDA_CHECK(cudaMalloc(&d_K6, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_w6, sizeof(double) * N));
    CUDA_CHECK(cudaMemcpy(d_P6, P0.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Hk6, jacForceGpu.data(), sizeof(double) * N * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xi6, xiForce.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w6, w0.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    kalmanComputeX<<<kGridX, kBlockX>>>(d_P6, d_Hk6, d_X6, N, m);
    kalmanComputeA<<<1, kBlockA>>>(d_Hk6, d_X6, d_A6, N, m);
    kalmanAddDiag<<<(m + 31) / 32, 32>>>(d_A6, m, 1.0 / etaF);
    kalmanInvertGaussJordan<<<1, 1>>>(d_A6, d_Ainv6, d_Aug6, m);
    kalmanComputeK<<<kGridX, kBlockX>>>(d_X6, d_Ainv6, d_K6, N, m);
    kalmanUpdateP<<<kGridP, kBlockP>>>(d_P6, d_K6, d_X6, N, m, qF);
    kalmanUpdateW<<<kGridW, kBlockW>>>(d_w6, d_K6, d_xi6, N, m);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> wNewForceGpu(N);
    CUDA_CHECK(cudaMemcpy(wNewForceGpu.data(), d_w6, sizeof(double) * N, cudaMemcpyDeviceToHost));
    cudaFree(d_P6); cudaFree(d_Hk6); cudaFree(d_xi6); cudaFree(d_X6);
    cudaFree(d_A6); cudaFree(d_Ainv6); cudaFree(d_Aug6); cudaFree(d_K6); cudaFree(d_w6);

    vector<double> xiForceCpu(1, fRefX - forceXCpu[targetAtom]);
    vector<double> Pcpu6 = P0;
    vector<double> wCpu6 = w0Cpu;
    cpuKalmanUpdate(Pcpu6, wCpu6, jacForceCpu, xiForceCpu, N, m, etaF, qF);

    vector<double> wReal6 = w0Cpu;
    KalmanFilter kf6(N, KalmanFilter::KT_STANDARD);
    kf6.setState(wReal6.data());
    kf6.setParametersStandard(epsilon, q0, qtau, qmin, eta0, etatau, etamax);
    kf6.setJacobian(jacForceCpu.data(), m);
    kf6.setError(xiForceCpu.data(), m);
    kf6.update(m);

    double maxAbsErrWGpuCpu6 = 0.0, maxAbsErrWGpuReal6 = 0.0;
    for (int i = 0; i < N; ++i)
    {
        maxAbsErrWGpuCpu6 = max(maxAbsErrWGpuCpu6, fabs(wNewForceGpu[i] - wCpu6[i]));
        maxAbsErrWGpuReal6 = max(maxAbsErrWGpuReal6, fabs(wNewForceGpu[i] - wReal6[i]));
    }

    printf("\n--- Stage 6: Kalman force-fit update (atom %d, x-component) ---\n", targetAtom);
    printf("fRef=%.10E  force(before)=gpu:%.10E cpu:%.10E\n",
           fRefX, forceXGpu[targetAtom], forceXCpu[targetAtom]);
    printf("max|dFdc_gpu-dFdc_cpu| = %.3E (%d combined connections)\n", maxAbsErrJacForce, N);
    printf("max|w_gpu-w_cpu|       = %.3E\n", maxAbsErrWGpuCpu6);
    printf("max|w_gpu-w_real|      = %.3E\n", maxAbsErrWGpuReal6);

    bool pass6 = (maxAbsErrJacForce < 1e-9) && (maxAbsErrWGpuCpu6 < 1e-8) &&
                 (maxAbsErrWGpuReal6 < 1e-8);
    printf("Stage 6: %s\n", pass6 ? "PASS" : "FAIL");

    bool passAll = pass && pass5 && pass6;
    printf("\n%s\n", passAll ? "ALL PASS" : "SOME FAILED");
    return passAll ? 0 : 1;
}
