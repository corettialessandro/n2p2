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
#include <cstring>
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
// prototype survey) -- 32 gives headroom without the register pressure of
// a much larger fixed size. gpuSfExpAngnGroup's kernel enforces this.
int const MAX_MEMBERS = 32;

__device__ inline double warpReduceSum(double val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-per-atom kernels (soft-percolating-jellyfish.md, Phase 5 follow-up):
// the original one-thread-per-atom kernels below launched gridSize =
// ceil(numAtoms/blockSize) blocks -- at the atom counts a real MPI rank
// actually sees (measured as low as ~20-270 atoms/rank at realistic
// production rank counts), that is only 1-4 thread blocks against an
// A100's 108 SMs, and the GPU dispatch cost was measured to stay nearly
// FLAT regardless of atoms/call across a 14x range (630 vs 8640 atoms,
// single rank) -- strong evidence of severe occupancy-bound underuse, not
// a compute-bound cost. Fix: assign one WARP (32 threads) per atom
// instead of one thread. Each lane processes a disjoint stride of that
// atom's neighbors (ExpRad) or outer pair-index j (ExpAngn), then the
// owner-atom accumulators (result/dResultX/Y/Z) are combined via a
// shuffle-based warp reduction -- no atomics needed for that part, since
// the reduction stays within one warp. This multiplies the number of
// concurrently active parallel units by up to 32x at the same atom
// count, directly targeting the diagnosed bottleneck.
//
// ExpAngn's neighbor-side derivative writes (neighDGdx/y/z) are the one
// place this needs real synchronization: a given neighbor slot's
// accumulator receives contributions from MULTIPLE (j,k) pairs, and with
// pair work spread across lanes by outer index j, a neighbor k can
// receive writes from whichever lane owns each j<k -- generally a
// DIFFERENT lane than whichever owns k itself as an outer index later.
// That's an inherent many-writers-one-slot pattern, so both the idxJ and
// idxK writes use atomicAdd (double atomicAdd is native on this
// project's target compute capability, sm_80). ExpRad has no such
// problem -- each neighbor slot's derivative is written by exactly one
// lane (whichever lane's stride includes that neighbor), so it keeps the
// original direct (non-atomic, single-write) form.
//
// Every arithmetic expression below is copied verbatim from the
// validated original (not algebraically simplified, e.g. the seemingly-
// redundant rijs/rij term in ExpAngn's p1) to avoid any risk of a subtle
// floating-point reassociation changing results at the ~1e-16 level the
// existing correctness harness checks against.

__global__ void sfExpRadGroupKernel(
    int numAtoms, int e1, double rc, int numMembers,
    const double* eta, const double* rs,
    const int* neighOffset, const int* neighElem, const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double* G, double* dGdx, double* dGdy, double* dGdz,
    double* neighborDGdx, double* neighborDGdy, double* neighborDGdz)
{
    int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    int t = globalThread / 32;
    int lane = globalThread % 32;
    if (t >= numAtoms) return;
    int off = neighOffset[t], n = neighOffset[t + 1] - off;

    double result[MAX_MEMBERS] = {0}, dResultX[MAX_MEMBERS] = {0},
           dResultY[MAX_MEMBERS] = {0}, dResultZ[MAX_MEMBERS] = {0};

    double rcinv = 1.0 / rc;
    for (int j = lane; j < n; j += 32)
    {
        int idxN = off + j;
        if (neighElem[idxN] != e1) continue;
        double rij = neighDist[idxN];
        if (rij >= rc) continue;
        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);
        for (int k = 0; k < numMembers; ++k)
        {
            double diff = rij - rs[k];
            double pexp = exp(-eta[k] * diff * diff);
            result[k] += pexp * pfc;
            double p1 = (pdfc - 2.0 * eta[k] * diff * pfc) * pexp / rij;
            double dijx = p1 * neighDx[idxN];
            double dijy = p1 * neighDy[idxN];
            double dijz = p1 * neighDz[idxN];
            dResultX[k] += dijx; dResultY[k] += dijy; dResultZ[k] += dijz;
            size_t idx = (size_t)idxN * numMembers + k;
            neighborDGdx[idx] = -dijx;
            neighborDGdy[idx] = -dijy;
            neighborDGdz[idx] = -dijz;
        }
    }

    size_t base = (size_t)t * numMembers;
    for (int k = 0; k < numMembers; ++k)
    {
        double r  = warpReduceSum(result[k]);
        double dx = warpReduceSum(dResultX[k]);
        double dy = warpReduceSum(dResultY[k]);
        double dz = warpReduceSum(dResultZ[k]);
        if (lane == 0)
        {
            G[base + k]    = r;
            dGdx[base + k] = dx;
            dGdy[base + k] = dy;
            dGdz[base + k] = dz;
        }
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
    int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    int t = globalThread / 32;
    int lane = globalThread % 32;
    if (t >= numAtoms) return;
    int off = neighOffset[t], n = neighOffset[t + 1] - off;

    double result[MAX_MEMBERS] = {0}, dResultX[MAX_MEMBERS] = {0},
           dResultY[MAX_MEMBERS] = {0}, dResultZ[MAX_MEMBERS] = {0};

    double rc2 = rc * rc;
    double rcinv = 1.0 / rc;

    for (int j = lane; j < n - 1; j += 32)
    {
        int idxNJ = off + j;
        double rij = neighDist[idxNJ];
        if (!(rij < rc)) continue;
        int nej = neighElem[idxNJ];
        double pfcij, pdfcij;
        cutoffTANHU(rij, rcinv, pfcij, pdfcij);
        double dijx = neighDx[idxNJ], dijy = neighDy[idxNJ], dijz = neighDz[idxNJ];

        for (int k = j + 1; k < n; ++k)
        {
            int idxNK = off + k;
            double rik = neighDist[idxNK];
            if (!(rik < rc)) continue;
            int nek = neighElem[idxNK];

            double dikx = neighDx[idxNK], diky = neighDy[idxNK], dikz = neighDz[idxNK];
            double djkx = dikx - dijx;
            double djky = diky - dijy;
            double djkz = dikz - dijz;
            double rjk2 = djkx * djkx + djky * djky + djkz * djkz;
            if (!(rjk2 < rc2)) continue;
            double rjk = sqrt(rjk2);

            double pfcik, pdfcik; cutoffTANHU(rik, rcinv, pfcik, pdfcik);
            double pfcjk, pdfcjk; cutoffTANHU(rjk, rcinv, pfcjk, pdfcjk);

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

                size_t idxJ = (size_t)idxNJ * numMembers + m;
                size_t idxK = (size_t)idxNK * numMembers + m;
                atomicAdd(&neighborDGdx[idxJ], -(drijx + drjkx));
                atomicAdd(&neighborDGdy[idxJ], -(drijy + drjky));
                atomicAdd(&neighborDGdz[idxJ], -(drijz + drjkz));
                atomicAdd(&neighborDGdx[idxK], -(drikx - drjkx));
                atomicAdd(&neighborDGdy[idxK], -(driky - drjky));
                atomicAdd(&neighborDGdz[idxK], -(drikz - drjkz));
            }
        }
    }

    size_t base = (size_t)t * numMembers;
    for (int m = 0; m < numMembers; ++m)
    {
        double r  = warpReduceSum(result[m]);
        double dx = warpReduceSum(dResultX[m]);
        double dy = warpReduceSum(dResultY[m]);
        double dz = warpReduceSum(dResultZ[m]);
        if (lane == 0)
        {
            // Only G gets this normalization -- the derivative
            // accumulators already have it baked in via fgF = fg * pnorm
            // inside the pair loop above (matches the original serial
            // code exactly: result[m] *= pow(...) post-loop, dResultX/Y/Z
            // untouched).
            G[base + m]    = r * pow(2.0, 1.0 - zeta[m]);
            dGdx[base + m] = dx;
            dGdy[base + m] = dy;
            dGdz[base + m] = dz;
        }
    }
}

// --- Persistent, grow-on-demand device buffers, PLUS pinned host mirror
// buffers on a dedicated stream -----------------------------------------
// Same cudaMalloc/cudaFree-avoidance rationale as GpuNeuralNetwork.cu
// (fixed driver overhead independent of size, called every MD timestep).
//
// Added after profiling the first working version end to end (Phase 5
// step 8, soft-percolating-jellyfish.md): correct, but ~10x SLOWER than
// CPU-only, even at a single MPI rank with zero GPU contention. Root
// cause: each group call issued ~16 *synchronous* cudaMemcpy/
// cudaDeviceSynchronize calls in a row -- and cudaMemcpy on ordinary
// pageable host memory (plain heap/std::vector, what Mode.cpp's CSR
// arrays are) blocks the calling host thread until the transfer AND any
// prior work on the device complete, with each block/wake potentially
// paying OS-level scheduling latency, not just the transfer time itself
// (this compounds badly at ~9 group calls/timestep x ~16 blocking calls
// each x 2000 timesteps). cudaMemcpyAsync only actually avoids that
// per-call blocking round trip when the HOST side is pinned
// (page-locked) memory -- on pageable memory the driver silently
// degrades it to synchronous behavior anyway, so both changes are
// required together, not either alone.
//
// Fix: every device buffer now has a same-shape pinned host mirror
// (cudaMallocHost, not malloc). Mode.cpp's ordinary pageable arrays are
// staged into these mirrors with a plain memcpy (fast, no driver
// round-trip), then every H2D/D2H transfer and the memsets run as
// *Async on one dedicated stream, with a single cudaStreamSynchronize()
// as the only blocking point per call -- collapsing ~16 blocking
// round-trips into 1. Mode.cpp's calling convention (plain pointers, no
// CUDA types) is unchanged; this is entirely internal to this file.
struct SfBuffers
{
    cudaStream_t stream = nullptr;

    int    *d_neighOffset = nullptr, *d_neighElem = nullptr;
    double *d_neighDist = nullptr, *d_neighDx = nullptr, *d_neighDy = nullptr,
           *d_neighDz = nullptr;
    double *d_G = nullptr, *d_dGdx = nullptr, *d_dGdy = nullptr, *d_dGdz = nullptr;
    double *d_neighborDGdx = nullptr, *d_neighborDGdy = nullptr,
           *d_neighborDGdz = nullptr;
    int    *d_e1 = nullptr, *d_e2 = nullptr;
    double *d_eta = nullptr, *d_rs = nullptr, *d_lambda = nullptr, *d_zeta = nullptr;

    int    *h_neighOffset = nullptr, *h_neighElem = nullptr;
    double *h_neighDist = nullptr, *h_neighDx = nullptr, *h_neighDy = nullptr,
           *h_neighDz = nullptr;
    double *h_G = nullptr, *h_dGdx = nullptr, *h_dGdy = nullptr, *h_dGdz = nullptr;
    double *h_neighborDGdx = nullptr, *h_neighborDGdy = nullptr,
           *h_neighborDGdz = nullptr;
    int    *h_e1 = nullptr, *h_e2 = nullptr;
    double *h_eta = nullptr, *h_rs = nullptr, *h_lambda = nullptr, *h_zeta = nullptr;

    size_t capAtoms = 0, capNeigh = 0, capG = 0, capNeighG = 0, capMembers = 0;

    void ensureStream()
    {
        if (!stream) CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void ensureAtoms(int numAtoms)
    {
        if ((size_t)(numAtoms + 1) <= capAtoms) return;
        if (d_neighOffset) { cudaFree(d_neighOffset); cudaFreeHost(h_neighOffset); }
        capAtoms = numAtoms + 1;
        CUDA_CHECK(cudaMalloc(&d_neighOffset, capAtoms * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_neighOffset, capAtoms * sizeof(int)));
    }

    void ensureNeigh(size_t totalNeigh)
    {
        if (totalNeigh <= capNeigh) return;
        if (d_neighElem) {
            cudaFree(d_neighElem); cudaFree(d_neighDist);
            cudaFree(d_neighDx); cudaFree(d_neighDy); cudaFree(d_neighDz);
            cudaFreeHost(h_neighElem); cudaFreeHost(h_neighDist);
            cudaFreeHost(h_neighDx); cudaFreeHost(h_neighDy); cudaFreeHost(h_neighDz);
        }
        capNeigh = totalNeigh;
        CUDA_CHECK(cudaMalloc(&d_neighElem, capNeigh * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighDist, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDx, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDy, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighDz, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_neighElem, capNeigh * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_neighDist, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_neighDx, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_neighDy, capNeigh * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_neighDz, capNeigh * sizeof(double)));
    }

    void ensureG(size_t gSize)
    {
        if (gSize <= capG) return;
        if (d_G) {
            cudaFree(d_G); cudaFree(d_dGdx); cudaFree(d_dGdy); cudaFree(d_dGdz);
            cudaFreeHost(h_G); cudaFreeHost(h_dGdx); cudaFreeHost(h_dGdy); cudaFreeHost(h_dGdz);
        }
        capG = gSize;
        CUDA_CHECK(cudaMalloc(&d_G, capG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdx, capG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdy, capG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdz, capG * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_G, capG * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_dGdx, capG * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_dGdy, capG * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_dGdz, capG * sizeof(double)));
    }

    void ensureNeighG(size_t neighGSize)
    {
        if (neighGSize <= capNeighG) return;
        if (d_neighborDGdx) {
            cudaFree(d_neighborDGdx); cudaFree(d_neighborDGdy); cudaFree(d_neighborDGdz);
            cudaFreeHost(h_neighborDGdx); cudaFreeHost(h_neighborDGdy); cudaFreeHost(h_neighborDGdz);
        }
        capNeighG = neighGSize;
        CUDA_CHECK(cudaMalloc(&d_neighborDGdx, capNeighG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighborDGdy, capNeighG * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighborDGdz, capNeighG * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_neighborDGdx, capNeighG * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_neighborDGdy, capNeighG * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_neighborDGdz, capNeighG * sizeof(double)));
    }

    void ensureMembersRad(int numMembers)
    {
        if ((size_t)numMembers <= capMembers) return;
        if (d_eta) { cudaFree(d_eta); cudaFree(d_rs); cudaFreeHost(h_eta); cudaFreeHost(h_rs); }
        capMembers = numMembers;
        CUDA_CHECK(cudaMalloc(&d_eta, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rs, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_eta, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_rs, capMembers * sizeof(double)));
    }

    void ensureMembersAngn(int numMembers)
    {
        if ((size_t)numMembers <= capMembers) return;
        if (d_e1) {
            cudaFree(d_e1); cudaFree(d_e2); cudaFree(d_eta);
            cudaFree(d_lambda); cudaFree(d_zeta);
            cudaFreeHost(h_e1); cudaFreeHost(h_e2); cudaFreeHost(h_eta);
            cudaFreeHost(h_lambda); cudaFreeHost(h_zeta);
        }
        capMembers = numMembers;
        CUDA_CHECK(cudaMalloc(&d_e1, capMembers * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_e2, capMembers * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_eta, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_lambda, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_zeta, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_e1, capMembers * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_e2, capMembers * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_eta, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_lambda, capMembers * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_zeta, capMembers * sizeof(double)));
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
    b.ensureStream();
    int const totalNeigh = neighOffset[numAtoms];
    size_t const gSize = (size_t)numAtoms * numMembers;
    size_t const neighGSize = (size_t)totalNeigh * numMembers;

    b.ensureAtoms(numAtoms);
    b.ensureNeigh(totalNeigh);
    b.ensureG(gSize);
    b.ensureNeighG(neighGSize);
    b.ensureMembersRad(numMembers);

    memcpy(b.h_neighOffset, neighOffset, (numAtoms + 1) * sizeof(int));
    memcpy(b.h_neighElem, neighElement, totalNeigh * sizeof(int));
    memcpy(b.h_neighDist, neighDist, totalNeigh * sizeof(double));
    memcpy(b.h_neighDx, neighDx, totalNeigh * sizeof(double));
    memcpy(b.h_neighDy, neighDy, totalNeigh * sizeof(double));
    memcpy(b.h_neighDz, neighDz, totalNeigh * sizeof(double));
    memcpy(b.h_eta, eta, numMembers * sizeof(double));
    memcpy(b.h_rs, rs, numMembers * sizeof(double));

    cudaStream_t s = b.stream;
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighOffset, b.h_neighOffset, (numAtoms + 1) * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighElem, b.h_neighElem, totalNeigh * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDist, b.h_neighDist, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDx, b.h_neighDx, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDy, b.h_neighDy, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDz, b.h_neighDz, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_eta, b.h_eta, numMembers * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_rs, b.h_rs, numMembers * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemsetAsync(b.d_neighborDGdx, 0, neighGSize * sizeof(double), s));
    CUDA_CHECK(cudaMemsetAsync(b.d_neighborDGdy, 0, neighGSize * sizeof(double), s));
    CUDA_CHECK(cudaMemsetAsync(b.d_neighborDGdz, 0, neighGSize * sizeof(double), s));

    // Warp-per-atom: each atom needs 32 threads (one warp), not one.
    int const blockSize = 128;                 // 4 warps/block
    int const warpsPerBlock = blockSize / 32;
    int const gridSize = (numAtoms + warpsPerBlock - 1) / warpsPerBlock;
    sfExpRadGroupKernel<<<gridSize, blockSize, 0, s>>>(
        numAtoms, e1, rc, numMembers, b.d_eta, b.d_rs,
        b.d_neighOffset, b.d_neighElem, b.d_neighDist,
        b.d_neighDx, b.d_neighDy, b.d_neighDz,
        b.d_G, b.d_dGdx, b.d_dGdy, b.d_dGdz,
        b.d_neighborDGdx, b.d_neighborDGdy, b.d_neighborDGdz);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(b.h_G, b.d_G, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_dGdx, b.d_dGdx, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_dGdy, b.d_dGdy, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_dGdz, b.d_dGdz, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_neighborDGdx, b.d_neighborDGdx, neighGSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_neighborDGdy, b.d_neighborDGdy, neighGSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_neighborDGdz, b.d_neighborDGdz, neighGSize * sizeof(double), cudaMemcpyDeviceToHost, s));

    CUDA_CHECK(cudaStreamSynchronize(s));

    memcpy(G, b.h_G, gSize * sizeof(double));
    memcpy(dGdx, b.h_dGdx, gSize * sizeof(double));
    memcpy(dGdy, b.h_dGdy, gSize * sizeof(double));
    memcpy(dGdz, b.h_dGdz, gSize * sizeof(double));
    memcpy(neighborDGdx, b.h_neighborDGdx, neighGSize * sizeof(double));
    memcpy(neighborDGdy, b.h_neighborDGdy, neighGSize * sizeof(double));
    memcpy(neighborDGdz, b.h_neighborDGdz, neighGSize * sizeof(double));
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
    b.ensureStream();
    int const totalNeigh = neighOffset[numAtoms];
    size_t const gSize = (size_t)numAtoms * numMembers;
    size_t const neighGSize = (size_t)totalNeigh * numMembers;

    b.ensureAtoms(numAtoms);
    b.ensureNeigh(totalNeigh);
    b.ensureG(gSize);
    b.ensureNeighG(neighGSize);
    b.ensureMembersAngn(numMembers);

    memcpy(b.h_neighOffset, neighOffset, (numAtoms + 1) * sizeof(int));
    memcpy(b.h_neighElem, neighElement, totalNeigh * sizeof(int));
    memcpy(b.h_neighDist, neighDist, totalNeigh * sizeof(double));
    memcpy(b.h_neighDx, neighDx, totalNeigh * sizeof(double));
    memcpy(b.h_neighDy, neighDy, totalNeigh * sizeof(double));
    memcpy(b.h_neighDz, neighDz, totalNeigh * sizeof(double));
    memcpy(b.h_e1, e1, numMembers * sizeof(int));
    memcpy(b.h_e2, e2, numMembers * sizeof(int));
    memcpy(b.h_eta, eta, numMembers * sizeof(double));
    memcpy(b.h_lambda, lambda, numMembers * sizeof(double));
    memcpy(b.h_zeta, zeta, numMembers * sizeof(double));

    cudaStream_t s = b.stream;
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighOffset, b.h_neighOffset, (numAtoms + 1) * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighElem, b.h_neighElem, totalNeigh * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDist, b.h_neighDist, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDx, b.h_neighDx, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDy, b.h_neighDy, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_neighDz, b.h_neighDz, totalNeigh * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_e1, b.h_e1, numMembers * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_e2, b.h_e2, numMembers * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_eta, b.h_eta, numMembers * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_lambda, b.h_lambda, numMembers * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.d_zeta, b.h_zeta, numMembers * sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemsetAsync(b.d_neighborDGdx, 0, neighGSize * sizeof(double), s));
    CUDA_CHECK(cudaMemsetAsync(b.d_neighborDGdy, 0, neighGSize * sizeof(double), s));
    CUDA_CHECK(cudaMemsetAsync(b.d_neighborDGdz, 0, neighGSize * sizeof(double), s));

    // Warp-per-atom: each atom needs 32 threads (one warp), not one.
    int const blockSize = 128;                 // 4 warps/block
    int const warpsPerBlock = blockSize / 32;
    int const gridSize = (numAtoms + warpsPerBlock - 1) / warpsPerBlock;
    sfExpAngnGroupKernel<<<gridSize, blockSize, 0, s>>>(
        numAtoms, rc, numMembers, b.d_e1, b.d_e2, b.d_eta, b.d_lambda, b.d_zeta,
        b.d_neighOffset, b.d_neighElem, b.d_neighDist,
        b.d_neighDx, b.d_neighDy, b.d_neighDz,
        b.d_G, b.d_dGdx, b.d_dGdy, b.d_dGdz,
        b.d_neighborDGdx, b.d_neighborDGdy, b.d_neighborDGdz);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(b.h_G, b.d_G, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_dGdx, b.d_dGdx, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_dGdy, b.d_dGdy, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_dGdz, b.d_dGdz, gSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_neighborDGdx, b.d_neighborDGdx, neighGSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_neighborDGdy, b.d_neighborDGdy, neighGSize * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(b.h_neighborDGdz, b.d_neighborDGdz, neighGSize * sizeof(double), cudaMemcpyDeviceToHost, s));

    CUDA_CHECK(cudaStreamSynchronize(s));

    memcpy(G, b.h_G, gSize * sizeof(double));
    memcpy(dGdx, b.h_dGdx, gSize * sizeof(double));
    memcpy(dGdy, b.h_dGdy, gSize * sizeof(double));
    memcpy(dGdz, b.h_dGdz, gSize * sizeof(double));
    memcpy(neighborDGdx, b.h_neighborDGdx, neighGSize * sizeof(double));
    memcpy(neighborDGdy, b.h_neighborDGdy, neighGSize * sizeof(double));
    memcpy(neighborDGdz, b.h_neighborDGdz, neighGSize * sizeof(double));
}

} // namespace nnp
