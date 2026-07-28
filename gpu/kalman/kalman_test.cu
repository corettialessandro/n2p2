// Phase 4 step 1: standalone CUDA port of KalmanFilter::update()
// (src/libnnptrain/KalmanFilter.cpp:126-197), the weight-update algorithm
// nnp-train actually uses (kalman_type 0 = KT_STANDARD in this project's
// temp/H2O_2G/input.nn:79). Only KT_STANDARD is ported here -- that's the
// only mode this project's production config uses (KT_FADINGMEMORY exists
// in the CPU code but isn't exercised by input.nn, same "port what's
// actually used" choice already made for symmetry functions in ../soa and
// ../e2e).
//
// The algorithm (state w = full weight vector, both elements combined,
// since input.nn:42 sets update_strategy 0 = US_COMBINED -- one filter over
// all N = 3327 weights for H2O_2G, not one per element):
//
//   X = P . H                (N x N) . (N x m) -> N x m   [KalmanFilter.cpp:134]
//   A = H^T . X + R          (m x N) . (N x m) -> m x m, R = I/eta   [:138,142-148]
//   K = X . A^-1             (N x m) . (m x m) -> N x m              [:158]
//   P -= K . X^T             covariance downdate                     [:162]
//   P += Q                   Q = I*q (process noise)                [:171]
//   w += K . xi              state/weight update                    [:178]
//
// with the scalar schedule eta *= exp(etatau) (capped at etamax, applied
// *before* building R) and q *= exp(-qtau) (floored at qmin, applied
// *after* P is updated) -- this exact ordering is replicated below since
// getting it backwards would silently use next iteration's eta/q instead
// of this iteration's.
//
// N, m for the real H2O_2G production run (US_COMBINED, MPI_Gather over
// 32 ranks with task_batch_size 1): N = 3327 (1576 H-connections + 1751
// O-connections), m = 32 (one Jacobian column gathered per MPI rank).
// P is dense N x N (~89 MB) -- not block-diagonal, since it's one combined
// filter. Both the small smoke-test size and this real production size are
// exercised below.
//
// Validation, three independent implementations of the same math:
//  1. These CUDA kernels (the thing being ported).
//  2. A from-scratch nested-loop C++ CPU reference below (own hand-rolled
//     Gauss-Jordan inversion, NOT shared code with the GPU kernel's) --
//     catches CUDA indexing/parallelization bugs.
//  3. The REAL nnp::KalmanFilter class (linked from lib/libnnptrain.a,
//     Eigen-based, LU-decomposition inverse) run in lockstep on the exact
//     same H/xi/parameter sequence. Its P/K are private with no getters,
//     so only its externally-visible state vector w can be cross-checked --
//     but since w = w + K.xi depends on every step of the recursion, a bug
//     in P or K anywhere in the chain would show up in w within a step or
//     two. This is the strongest available ground truth: genuine n2p2
//     production code, not a reimplementation.
//     (KalmanFilter.o/Updater.o have zero undefined GSL or MPI runtime
//     symbols -- confirmed via `nm -u` -- so linking lib/libnnptrain.a here
//     needs no MPI/GSL libraries at link time, only mpi.h and Eigen headers
//     at compile time for KalmanFilter.h's includes.)
//
// The m x m inverse uses a single-thread (<<<1,1>>>) Gauss-Jordan kernel --
// deliberately not batched/parallel, since m is small (32 in production)
// and this step is about proving the recursion's numerics correct, not
// performance (same "correctness first" staging as every other phase). The
// dominant-cost kernel is updateP (O(N^2 * m) per call), the real target
// for a later cuBLAS dsyrk/dgemm-based rewrite.

#include "KalmanFilter.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
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

// ---------------------------------------------------------------------
// GPU kernels. P is row-major N x N (symmetric, both halves kept live).
// H, X, K are column-major N x m (matches Eigen::Map<MatrixXd>(ptr, N, m)
// layout used by the real KalmanFilter::setJacobian(), so the same host
// buffer could later be handed to either implementation unchanged).
// A, Ainv are column-major m x m.
// ---------------------------------------------------------------------

__global__ void kernelComputeX(const double* P, const double* H,
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

__global__ void kernelComputeA(const double* H, const double* X,
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

__global__ void kernelAddDiag(double* A, int m, double value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) A[i + i * m] += value;
}

// Single-thread Gauss-Jordan inversion of an m x m matrix. `augmented` is a
// caller-allocated device buffer of size m * 2m (row-major [A | I]) -- no
// per-thread fixed-size local array (a fixed-size local buffer sized by
// guesswork caused a real bug earlier in this project; this avoids that
// class of mistake entirely by taking the buffer size from the actual m).
__global__ void kernelInvertGaussJordan(const double* A, double* Ainv,
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

__global__ void kernelComputeK(const double* X, const double* Ainv,
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

__global__ void kernelUpdateP(double* P, const double* K, const double* X,
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

__global__ void kernelUpdateW(double* w, const double* K, const double* xi,
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

// ---------------------------------------------------------------------
// Independent CPU reference: plain nested loops, own Gauss-Jordan
// inversion (separate code from the GPU kernel's, though same textbook
// algorithm -- the real KalmanFilter class below provides the genuinely
// different-algorithm (LU-based) cross-check).
// ---------------------------------------------------------------------

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

// One cpuUpdate() call == one KalmanFilter::update() call, replicated by
// hand with plain loops over the caller-owned P/w arrays.
static void cpuUpdate(vector<double>& P, vector<double>& w,
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

// ---------------------------------------------------------------------
// One test case: N (state size), m (observation size), T sequential
// updates with fresh random H/xi each time (mirroring nnp-train calling
// KalmanFilter::update() once per scheduled structure/force-component).
// ---------------------------------------------------------------------

static bool runCase(const char* label, int N, int m, int T,
                     double epsilon, double q0, double qtau, double qmin,
                     double eta0, double etatau, double etamax,
                     unsigned seed)
{
    printf("\n=== %s: N=%d, m=%d, T=%d updates ===\n", label, N, m, T);

    // Shared initial P = I/epsilon (matches KalmanFilter::setParametersStandard).
    vector<double> P0(N * N, 0.0);
    for (int i = 0; i < N; ++i) P0[i * N + i] = 1.0 / epsilon;

    vector<double> Pcpu = P0;
    vector<double> wCpu(N, 0.0), wGpu(N, 0.0), wReal(N, 0.0);

    double *dP, *dH, *dxi, *dX, *dA, *dAinv, *dAug, *dK, *dw;
    CUDA_CHECK(cudaMalloc(&dP, sizeof(double) * N * N));
    CUDA_CHECK(cudaMalloc(&dH, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&dxi, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&dX, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&dA, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&dAinv, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&dAug, sizeof(double) * m * 2 * m));
    CUDA_CHECK(cudaMalloc(&dK, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&dw, sizeof(double) * N));

    CUDA_CHECK(cudaMemcpy(dP, P0.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dw, 0, sizeof(double) * N));

    KalmanFilter kf(N, KalmanFilter::KT_STANDARD);
    kf.setState(wReal.data());
    kf.setParametersStandard(epsilon, q0, qtau, qmin, eta0, etatau, etamax);

    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    double eta = eta0, q = q0;
    double maxDiffGpuCpuP = 0.0, maxDiffGpuCpuW = 0.0, maxDiffGpuRealW = 0.0;

    dim3 blockX(256), gridX((N + 255) / 256, m);
    dim3 blockA(m, m);
    dim3 blockP(16, 16), gridP((N + 15) / 16, (N + 15) / 16);
    dim3 blockW(256), gridW((N + 255) / 256);

    for (int t = 0; t < T; ++t)
    {
        vector<double> H(N * m), xi(m);
        for (auto& v : H) v = dist(rng);
        for (auto& v : xi) v = dist(rng);

        // Scalar schedule: eta grows *before* being used to build R this
        // step (KalmanFilter.cpp:142); q decays *after* being used to
        // build Q this step (KalmanFilter.cpp:182). Same `eta`/`q` values
        // feed both the GPU and CPU-reference paths below.
        if (eta < etamax) eta *= exp(etatau);
        double etaUsed = eta;
        double qUsed = q;

        CUDA_CHECK(cudaMemcpy(dH, H.data(), sizeof(double) * N * m, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dxi, xi.data(), sizeof(double) * m, cudaMemcpyHostToDevice));

        kernelComputeX<<<gridX, blockX>>>(dP, dH, dX, N, m);
        kernelComputeA<<<1, blockA>>>(dH, dX, dA, N, m);
        kernelAddDiag<<<(m + 31) / 32, 32>>>(dA, m, 1.0 / etaUsed);
        kernelInvertGaussJordan<<<1, 1>>>(dA, dAinv, dAug, m);
        kernelComputeK<<<gridX, blockX>>>(dX, dAinv, dK, N, m);
        kernelUpdateP<<<gridP, blockP>>>(dP, dK, dX, N, m, qUsed);
        kernelUpdateW<<<gridW, blockW>>>(dw, dK, dxi, N, m);
        CUDA_CHECK(cudaDeviceSynchronize());

        cpuUpdate(Pcpu, wCpu, H, xi, N, m, etaUsed, qUsed);

        kf.setJacobian(H.data(), m);
        kf.setError(xi.data(), m);
        kf.update(m);

        if (q > qmin) q *= exp(-qtau);

        vector<double> Pgpu(N * N), wGpuStep(N);
        CUDA_CHECK(cudaMemcpy(Pgpu.data(), dP, sizeof(double) * N * N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(wGpuStep.data(), dw, sizeof(double) * N, cudaMemcpyDeviceToHost));

        double diffP = 0.0, diffW = 0.0, diffRealW = 0.0;
        for (int i = 0; i < N * N; ++i)
            diffP = max(diffP, fabs(Pgpu[i] - Pcpu[i]));
        for (int i = 0; i < N; ++i)
        {
            diffW = max(diffW, fabs(wGpuStep[i] - wCpu[i]));
            diffRealW = max(diffRealW, fabs(wGpuStep[i] - wReal[i]));
        }
        maxDiffGpuCpuP = max(maxDiffGpuCpuP, diffP);
        maxDiffGpuCpuW = max(maxDiffGpuCpuW, diffW);
        maxDiffGpuRealW = max(maxDiffGpuRealW, diffRealW);

        printf("  update %d: |P_gpu-P_cpu|=%.3E  |w_gpu-w_cpu|=%.3E  "
               "|w_gpu-w_real|=%.3E  eta=%.4f q=%.3E\n",
               t, diffP, diffW, diffRealW, etaUsed, qUsed);
    }

    cudaFree(dP); cudaFree(dH); cudaFree(dxi); cudaFree(dX);
    cudaFree(dA); cudaFree(dAinv); cudaFree(dAug); cudaFree(dK); cudaFree(dw);

    printf("  max over all updates: |P_gpu-P_cpu|=%.3E  |w_gpu-w_cpu|=%.3E  "
           "|w_gpu-w_real|=%.3E\n",
           maxDiffGpuCpuP, maxDiffGpuCpuW, maxDiffGpuRealW);

    double tol = 1e-8;
    bool pass = maxDiffGpuCpuP < tol && maxDiffGpuCpuW < tol &&
                maxDiffGpuRealW < tol;
    printf("  %s: %s\n", label, pass ? "PASS" : "FAIL");
    return pass;
}

int main()
{
    // Production KalmanFilter parameters, straight out of
    // temp/H2O_2G/input.nn:79-86 (kalman_type 0 = KT_STANDARD).
    const double epsilon = 1.0e-2, q0 = 0.01, qtau = 2.302, qmin = 1.0e-6;
    const double eta0 = 0.01, etatau = 2.302, etamax = 1.0;

    bool ok = true;

    // Small smoke-test size, fast iteration during development.
    ok &= runCase("small", 50, 5, 5, epsilon, q0, qtau, qmin,
                  eta0, etatau, etamax, 12345);

    // Real H2O_2G production size: N = 3327 combined weights (1576 H +
    // 1751 O, update_strategy 0 = US_COMBINED), m = 32 (MPI_Gather over
    // 32 ranks, task_batch_size 1).
    ok &= runCase("production-sized (H2O_2G, N=3327, m=32)", 3327, 32, 5,
                  epsilon, q0, qtau, qmin, eta0, etatau, etamax, 67890);

    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
