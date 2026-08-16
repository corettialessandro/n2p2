// See GpuSymmetryFunction.h. Device math ported verbatim from
// gpu/soa/symfnc_exprad_group_test.cu / symfnc_expangn_group_test.cu
// (already validated there to ~1e-16 on real H2O_2G neighbor geometry --
// see those files' header comments for the line-by-line derivation
// cross-check against src/libnnp/SymFncExpRad.cpp/SymFncExpAngn.cpp).
// This file adds: (1) generalization from the prototypes' fixed
// test-harness call shape to the plain flat-array parameters declared in
// GpuSymmetryFunction.h, (2) persistent, grow-on-demand device buffers
// (same pattern as GpuNeuralNetwork.cu/GpuKalmanFilter.cu -- avoids
// cudaMalloc/cudaFree every call, the same fixed-overhead mistake found
// and fixed there), since this is called every single MD timestep.

#include "GpuSymmetryFunction.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

namespace
{

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

// Per-thread scratch limits: real H2O_2G has at most 2 ExpRad members and
// 26 ExpAngn members per element (see soft-percolating-jellyfish.md's
// prototype survey) -- 64 gives headroom without the register pressure of
// a much larger fixed size. gpuSfExpAngnGroup's kernel enforces this.
int const MAX_MEMBERS = 64;

__device__ inline void symFncExpRadGroup(
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

__global__ void sfExpRadGroupKernel(
    int numAtoms, int e1, double rc, int numMembers,
    const double* eta, const double* rs,
    const int* neighOffset, const int* neighElem, const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double* G, double* dGdx, double* dGdy, double* dGdz,
    double* neighborDGdx, double* neighborDGdy, double* neighborDGdz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numAtoms) return;
    int off = neighOffset[t], n = neighOffset[t + 1] - off;

    double result[MAX_MEMBERS] = {0}, dResultX[MAX_MEMBERS] = {0},
           dResultY[MAX_MEMBERS] = {0}, dResultZ[MAX_MEMBERS] = {0};

    symFncExpRadGroup(n, &neighElem[off], &neighDist[off], &neighDx[off],
                       &neighDy[off], &neighDz[off], e1, rc, numMembers,
                       eta, rs, result, dResultX, dResultY, dResultZ,
                       &neighborDGdx[(size_t)off * numMembers],
                       &neighborDGdy[(size_t)off * numMembers],
                       &neighborDGdz[(size_t)off * numMembers]);

    size_t base = (size_t)t * numMembers;
    for (int k = 0; k < numMembers; ++k)
    {
        G[base + k]    = result[k];
        dGdx[base + k] = dResultX[k];
        dGdy[base + k] = dResultY[k];
        dGdz[base + k] = dResultZ[k];
    }
}

__device__ inline void symFncExpAngnGroupReal(
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

__global__ void sfExpAngnGroupKernel(
    int numAtoms, double rc, int numMembers,
    const int* e1, const int* e2, const double* eta, const double* lambda,
    const double* zeta,
    const int* neighOffset, const int* neighElem, const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double* G, double* dGdx, double* dGdy, double* dGdz,
    double* neighborDGdx, double* neighborDGdy, double* neighborDGdz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numAtoms) return;
    int off = neighOffset[t], n = neighOffset[t + 1] - off;

    double result[MAX_MEMBERS] = {0}, dResultX[MAX_MEMBERS] = {0},
           dResultY[MAX_MEMBERS] = {0}, dResultZ[MAX_MEMBERS] = {0};

    symFncExpAngnGroupReal(n, &neighElem[off], &neighDist[off], &neighDx[off],
                           &neighDy[off], &neighDz[off], rc, numMembers,
                           e1, e2, eta, lambda, zeta,
                           result, dResultX, dResultY, dResultZ,
                           &neighborDGdx[(size_t)off * numMembers],
                           &neighborDGdy[(size_t)off * numMembers],
                           &neighborDGdz[(size_t)off * numMembers]);

    size_t base = (size_t)t * numMembers;
    for (int k = 0; k < numMembers; ++k)
    {
        G[base + k] = result[k];
        dGdx[base + k] = dResultX[k];
        dGdy[base + k] = dResultY[k];
        dGdz[base + k] = dResultZ[k];
    }
}

// --- Persistent, grow-on-demand device buffers -----------------------
// Same rationale as GpuNeuralNetwork.cu: cudaMalloc/cudaFree carry fixed
// driver overhead independent of size, and this is called every MD
// timestep, so allocating fresh every call would repeat the exact mistake
// found and fixed there. One static, never-shrinking buffer set per
// function (not per element) -- sequential calls for different elements
// within the same timestep just reuse/grow the same buffers, which is
// fine since this is single-threaded per MPI rank and buffers are tiny
// relative to the NN weight/G buffers already cached this way.
struct SfBuffers
{
    int    *d_neighOffset = nullptr, *d_neighElem = nullptr;
    double *d_neighDist = nullptr, *d_neighDx = nullptr, *d_neighDy = nullptr,
           *d_neighDz = nullptr;
    double *d_G = nullptr, *d_dGdx = nullptr, *d_dGdy = nullptr, *d_dGdz = nullptr;
    double *d_neighborDGdx = nullptr, *d_neighborDGdy = nullptr,
           *d_neighborDGdz = nullptr;
    size_t capAtoms = 0, capNeigh = 0, capG = 0, capNeighG = 0;
    // Per-function parameter buffers (numMembers-sized, tiny, but still
    // cached rather than malloc'd every call for the same reason).
    int    *d_e1 = nullptr, *d_e2 = nullptr;
    double *d_eta = nullptr, *d_rs = nullptr, *d_lambda = nullptr, *d_zeta = nullptr;
    size_t capMembers = 0;

    void ensureAtoms(int numAtoms)
    {
        if ((size_t)(numAtoms + 1) <= capAtoms) return;
        if (d_neighOffset) cudaFree(d_neighOffset);
        capAtoms = numAtoms + 1;
        CUDA_CHECK(cudaMalloc(&d_neighOffset, capAtoms * sizeof(int)));
    }

    void ensureNeigh(size_t totalNeigh)
    {
        if (totalNeigh <= capNeigh) return;
        if (d_neighElem) { cudaFree(d_neighElem); cudaFree(d_neighDist);
            cudaFree(d_neighDx); cudaFree(d_neighDy); cudaFree(d_neighDz); }
        capNeigh = totalNeigh;
        CUDA_CHECK(cudaMalloc(&d_neighElem, capNeigh * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighDist, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDx, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDy, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDz, capNeigh * sizeof(double)));
    }

    void ensureG(size_t gSize)
    {
        if (gSize <= capG) return;
        if (d_G) { cudaFree(d_G); cudaFree(d_dGdx); cudaFree(d_dGdy); cudaFree(d_dGdz); }
        capG = gSize;
        CUDA_CHECK(cudaMalloc(&d_G, capG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdx, capG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdy, capG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdz, capG * sizeof(double)));
    }

    void ensureNeighG(size_t neighGSize)
    {
        if (neighGSize <= capNeighG) return;
        if (d_neighborDGdx) { cudaFree(d_neighborDGdx); cudaFree(d_neighborDGdy);
            cudaFree(d_neighborDGdz); }
        capNeighG = neighGSize;
        CUDA_CHECK(cudaMalloc(&d_neighborDGdx, capNeighG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighborDGdy, capNeighG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighborDGdz, capNeighG * sizeof(double)));
    }

    void ensureMembersRad(int numMembers)
    {
        if ((size_t)numMembers <= capMembers) return;
        if (d_eta) { cudaFree(d_eta); cudaFree(d_rs); }
        capMembers = numMembers;
        CUDA_CHECK(cudaMalloc(&d_eta, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rs, capMembers * sizeof(double)));
    }

    void ensureMembersAngn(int numMembers)
    {
        if ((size_t)numMembers <= capMembers) return;
        if (d_e1) { cudaFree(d_e1); cudaFree(d_e2); cudaFree(d_eta);
            cudaFree(d_lambda); cudaFree(d_zeta); }
        capMembers = numMembers;
        CUDA_CHECK(cudaMalloc(&d_e1, capMembers * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_e2, capMembers * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_eta, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_lambda, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_zeta, capMembers * sizeof(double)));
    }
};

SfBuffers& radBuffers()
{
    static SfBuffers b;
    return b;
}

SfBuffers& angnBuffers()
{
    static SfBuffers b;
    return b;
}

} // anonymous namespace

namespace nnp
{

void gpuSfExpRadGroup(int numAtoms, int const* neighOffset,
                      int const* neighElement, double const* neighDist,
                      double const* neighDx, double const* neighDy,
                      double const* neighDz,
                      int e1, double rc, int numMembers,
                      double const* eta, double const* rs,
                      double* G, double* dGdx, double* dGdy, double* dGdz,
                      double* neighborDGdx, double* neighborDGdy,
                      double* neighborDGdz)
{
    if (numMembers > MAX_MEMBERS)
    {
        fprintf(stderr, "gpuSfExpRadGroup: numMembers=%d exceeds MAX_MEMBERS=%d\n",
                numMembers, MAX_MEMBERS);
        exit(1);
    }
    SfBuffers& b = radBuffers();
    int const totalNeigh = neighOffset[numAtoms];
    size_t const gSize = (size_t)numAtoms * numMembers;
    size_t const neighGSize = (size_t)totalNeigh * numMembers;

    b.ensureAtoms(numAtoms);
    b.ensureNeigh(totalNeigh);
    b.ensureG(gSize);
    b.ensureNeighG(neighGSize);
    b.ensureMembersRad(numMembers);

    CUDA_CHECK(cudaMemcpy(b.d_neighOffset, neighOffset, (numAtoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighElem, neighElement, totalNeigh * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDist, neighDist, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDx, neighDx, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDy, neighDy, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDz, neighDz, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_eta, eta, numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_rs, rs, numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b.d_neighborDGdx, 0, neighGSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(b.d_neighborDGdy, 0, neighGSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(b.d_neighborDGdz, 0, neighGSize * sizeof(double)));

    int const blockSize = 128;
    int const gridSize = (numAtoms + blockSize - 1) / blockSize;
    sfExpRadGroupKernel<<<gridSize, blockSize>>>(
        numAtoms, e1, rc, numMembers, b.d_eta, b.d_rs,
        b.d_neighOffset, b.d_neighElem, b.d_neighDist,
        b.d_neighDx, b.d_neighDy, b.d_neighDz,
        b.d_G, b.d_dGdx, b.d_dGdy, b.d_dGdz,
        b.d_neighborDGdx, b.d_neighborDGdy, b.d_neighborDGdz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(G, b.d_G, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdx, b.d_dGdx, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdy, b.d_dGdy, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdz, b.d_dGdz, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(neighborDGdx, b.d_neighborDGdx, neighGSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(neighborDGdy, b.d_neighborDGdy, neighGSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(neighborDGdz, b.d_neighborDGdz, neighGSize * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpuSfExpAngnGroup(int numAtoms, int const* neighOffset,
                       int const* neighElement, double const* neighDist,
                       double const* neighDx, double const* neighDy,
                       double const* neighDz,
                       double rc, int numMembers,
                       int const* e1, int const* e2, double const* eta,
                       double const* lambda, double const* zeta,
                       double* G, double* dGdx, double* dGdy, double* dGdz,
                       double* neighborDGdx, double* neighborDGdy,
                       double* neighborDGdz)
{
    if (numMembers > MAX_MEMBERS)
    {
        fprintf(stderr, "gpuSfExpAngnGroup: numMembers=%d exceeds MAX_MEMBERS=%d\n",
                numMembers, MAX_MEMBERS);
        exit(1);
    }
    SfBuffers& b = angnBuffers();
    int const totalNeigh = neighOffset[numAtoms];
    size_t const gSize = (size_t)numAtoms * numMembers;
    size_t const neighGSize = (size_t)totalNeigh * numMembers;

    b.ensureAtoms(numAtoms);
    b.ensureNeigh(totalNeigh);
    b.ensureG(gSize);
    b.ensureNeighG(neighGSize);
    b.ensureMembersAngn(numMembers);

    CUDA_CHECK(cudaMemcpy(b.d_neighOffset, neighOffset, (numAtoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighElem, neighElement, totalNeigh * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDist, neighDist, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDx, neighDx, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDy, neighDy, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_neighDz, neighDz, totalNeigh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_e1, e1, numMembers * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_e2, e2, numMembers * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_eta, eta, numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_lambda, lambda, numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_zeta, zeta, numMembers * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b.d_neighborDGdx, 0, neighGSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(b.d_neighborDGdy, 0, neighGSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(b.d_neighborDGdz, 0, neighGSize * sizeof(double)));

    int const blockSize = 128;
    int const gridSize = (numAtoms + blockSize - 1) / blockSize;
    sfExpAngnGroupKernel<<<gridSize, blockSize>>>(
        numAtoms, rc, numMembers, b.d_e1, b.d_e2, b.d_eta, b.d_lambda, b.d_zeta,
        b.d_neighOffset, b.d_neighElem, b.d_neighDist,
        b.d_neighDx, b.d_neighDy, b.d_neighDz,
        b.d_G, b.d_dGdx, b.d_dGdy, b.d_dGdz,
        b.d_neighborDGdx, b.d_neighborDGdy, b.d_neighborDGdz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(G, b.d_G, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdx, b.d_dGdx, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdy, b.d_dGdy, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdz, b.d_dGdz, gSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(neighborDGdx, b.d_neighborDGdx, neighGSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(neighborDGdy, b.d_neighborDGdy, neighGSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(neighborDGdz, b.d_neighborDGdz, neighGSize * sizeof(double), cudaMemcpyDeviceToHost));
}

} // namespace nnp
