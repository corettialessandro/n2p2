// Performance work, part 4 (last of three requested): replace ../kalman/
// kalman_test.cu's naive one-thread-per-output-element kernels for the two
// dominant O(N^2*m) steps of KalmanFilter::update() -- X = P.H (computeX)
// and the K.X^T term of the covariance downdate P -= K.X^T (updateP) --
// with cuBLAS dgemm. At the real H2O_2G production size (N=3327 combined
// weights, m=32), these two steps are ~354M multiply-adds each; every other
// step (A = H^T.X, K = X.Ainv, the m x m inverse, w += K.xi) is
// O(N*m^2) or smaller (~3.4M ops or less) and is left as the existing naive
// kernel from kalman_test.cu -- not worth the extra code for <1% of the
// total flops, an explicit, honest scoping decision (same "batch what
// actually dominates" reasoning used to leave Kalman's A/inverse/K
// untouched here and calculateDFdc/calculateDEdc's own small per-k0 GEMMs
// as ordinary, non-batched dgemm in ../gemm/nn_dfdc_gemm_test.cu).
//
// Unlike every other file in this directory, P/H/X/K here are NOT row-major
// like AtomBatch/G -- kalman_test.cu deliberately stores H/X/K
// column-major, matching Eigen::Map<MatrixXd>'s layout, so the same buffers
// could be handed to the real KalmanFilter class unchanged (P is row-major,
// but symmetric, so its buffer reads identically either way). That means
// none of this file's GEMMs need the row-major-via-column-major trick used
// everywhere else in ../gemm/ -- cuBLAS's native column-major form applies
// directly: X = P.H and K = X.Ainv are plain cublasDgemm(OP_N, OP_N) calls,
// and P -= K.X^T just needs CUBLAS_OP_T on X, not a materialized transpose.
// An earlier draft of this file reused the row-major trick out of habit
// (copied from ../gemm/nn_forward_gemm_test.cu without checking the layout
// actually matched here) and silently computed the wrong matrix product --
// caught immediately by the 3-way validation below (P/w diverging from both
// the naive kernel and the independent CPU references, worsening every
// update), not by any subtler failure mode. Worth remembering: a helper
// that's correct in one file isn't automatically correct in another one
// with a different data-layout convention, even within the same directory.
//
// Ground truth: the same dual reference kalman_test.cu already established
// -- an independent CPU implementation (own nested loops, own Gauss-Jordan)
// and the real nnp::KalmanFilter class (linked from lib/libnnptrain.a, same
// debug-info-stripping workaround) -- plus the existing naive-kernel GPU
// pipeline itself, copied verbatim, so this is genuinely a three-way check
// (new GEMM path vs. old naive-kernel path vs. two independent CPU
// references), and a direct timing comparison between the two GPU paths.

#include "KalmanFilter.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, (int)st); \
        exit(1); \
    } \
} while (0)

// P, H, X, K here are ALL column-major (P row-major too, but symmetric, so
// its buffer is identical either way) -- this project's other GEMM helpers
// (gemmRowMajor, used for AtomBatch/G's genuinely row-major arrays) do NOT
// apply here; H/X/K are deliberately Eigen-compatible column-major (see
// ../kalman/kalman_test.cu's header comment), so cuBLAS can be called in
// its native column-major form directly, with plain transpose flags where
// needed -- no row-major-via-column-major trick, no manual transpose
// kernel. (An earlier draft of this file reused the row-major trick out of
// habit; it silently computed the wrong matrix product since H/X/K aren't
// row-major here, caught immediately by the 3-way validation below -- kept
// as a cautionary note, not a comment worth losing.)

// X (N x m, col-major) = P (N x N, col-major) . H (N x m, col-major).
void computeX(cublasHandle_t handle, int N, int m,
              const double* P, const double* H, double* X)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, m, N, &alpha, P, N, H, N, &beta, X, N));
}

// K (N x m, col-major) = X (N x m, col-major) . Ainv (m x m, col-major).
void computeK(cublasHandle_t handle, int N, int m,
              const double* X, const double* Ainv, double* K)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, m, m, &alpha, X, N, Ainv, m, &beta, K, N));
}

// P (N x N) -= K (N x m, col-major) . X^T (m x N) -- X^T via CUBLAS_OP_T
// directly on X's own buffer, no materialized transpose needed.
void updatePGemmTerm(cublasHandle_t handle, int N, int m,
                      const double* K, const double* X, double* P)
{
    double const alpha = -1.0, beta = 1.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N, m, &alpha, K, N, X, N, &beta, P, N));
}

// --- Naive kernels, copied verbatim from ../kalman/kalman_test.cu (already
// validated there): the one-thread-per-output-element reference this GEMM
// path is compared against. ------------------------------------------------

__global__ void kalmanComputeXNaive(const double* P, const double* H,
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
        for (int j = 0; j < m; ++j) augmented[i * w + m + j] = (i == j) ? 1.0 : 0.0;
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
            for (int j = 0; j < w; ++j)
            {
                double tmp = augmented[col * w + j];
                augmented[col * w + j] = augmented[pivotRow * w + j];
                augmented[pivotRow * w + j] = tmp;
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

__global__ void kalmanComputeKNaive(const double* X, const double* Ainv,
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

__global__ void kalmanUpdatePNaive(double* P, const double* K, const double* X,
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

// --- Independent CPU reference (own nested loops, own Gauss-Jordan),
// copied verbatim from ../kalman/kalman_test.cu. ---------------------------

static void cpuGaussJordanInvert(const vector<double>& A, vector<double>& Ainv, int m)
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
            for (int j = 0; j < w; ++j) swap(aug[col * w + j], aug[pivotRow * w + j]);
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

static bool runCase(cublasHandle_t handle, const char* label, int N, int m, int T,
                     double epsilon, double q0, double qtau, double qmin,
                     double eta0, double etatau, double etamax, unsigned seed)
{
    printf("\n=== %s: N=%d, m=%d, T=%d updates ===\n", label, N, m, T);

    vector<double> P0(N * N, 0.0);
    for (int i = 0; i < N; ++i) P0[i * N + i] = 1.0 / epsilon;

    vector<double> Pcpu = P0;
    vector<double> wCpu(N, 0.0), wGemm(N, 0.0), wNaive(N, 0.0), wReal(N, 0.0);

    double *d_P_gemm, *d_P_naive, *d_H, *d_xi, *d_X, *d_A, *d_Ainv, *d_Aug, *d_K, *d_w;
    CUDA_CHECK(cudaMalloc(&d_P_gemm, sizeof(double) * N * N));
    CUDA_CHECK(cudaMalloc(&d_P_naive, sizeof(double) * N * N));
    CUDA_CHECK(cudaMalloc(&d_H, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_xi, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_Ainv, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_Aug, sizeof(double) * m * 2 * m));
    CUDA_CHECK(cudaMalloc(&d_K, sizeof(double) * N * m));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(double) * N));

    CUDA_CHECK(cudaMemcpy(d_P_gemm, P0.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P_naive, P0.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_w, 0, sizeof(double) * N));

    KalmanFilter kf(N, KalmanFilter::KT_STANDARD);
    kf.setState(wReal.data());
    kf.setParametersStandard(epsilon, q0, qtau, qmin, eta0, etatau, etamax);

    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    double eta = eta0, q = q0;
    double maxDiffGemmNaiveP = 0.0, maxDiffGemmCpuW = 0.0, maxDiffGemmRealW = 0.0;

    dim3 blockX(256), gridX((N + 255) / 256, m);
    dim3 blockA(m, m);
    dim3 blockP(16, 16), gridP((N + 15) / 16, (N + 15) / 16);
    dim3 blockW(256), gridW((N + 255) / 256);

    for (int t = 0; t < T; ++t)
    {
        vector<double> H(N * m), xi(m);
        for (auto& v : H) v = dist(rng);
        for (auto& v : xi) v = dist(rng);

        if (eta < etamax) eta *= exp(etatau);
        double etaUsed = eta, qUsed = q;

        CUDA_CHECK(cudaMemcpy(d_H, H.data(), sizeof(double) * N * m, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xi, xi.data(), sizeof(double) * m, cudaMemcpyHostToDevice));

        // --- GEMM path: cuBLAS for X=P.H and the K.X^T term of updateP ---
        computeX(handle, N, m, d_P_gemm, d_H, d_X);
        kalmanComputeA<<<1, blockA>>>(d_H, d_X, d_A, N, m);
        kalmanAddDiag<<<(m + 31) / 32, 32>>>(d_A, m, 1.0 / etaUsed);
        kalmanInvertGaussJordan<<<1, 1>>>(d_A, d_Ainv, d_Aug, m);
        computeK(handle, N, m, d_X, d_Ainv, d_K);
        updatePGemmTerm(handle, N, m, d_K, d_X, d_P_gemm);
        kalmanAddDiag<<<(N + 255) / 256, 256>>>(d_P_gemm, N, qUsed);
        kalmanUpdateW<<<gridW, blockW>>>(d_w, d_K, d_xi, N, m);

        // --- Naive path (unchanged from ../kalman/kalman_test.cu) --------
        double *d_Xn, *d_An, *d_Ainvn, *d_Augn, *d_Kn;
        CUDA_CHECK(cudaMalloc(&d_Xn, sizeof(double) * N * m));
        CUDA_CHECK(cudaMalloc(&d_An, sizeof(double) * m * m));
        CUDA_CHECK(cudaMalloc(&d_Ainvn, sizeof(double) * m * m));
        CUDA_CHECK(cudaMalloc(&d_Augn, sizeof(double) * m * 2 * m));
        CUDA_CHECK(cudaMalloc(&d_Kn, sizeof(double) * N * m));
        kalmanComputeXNaive<<<gridX, blockX>>>(d_P_naive, d_H, d_Xn, N, m);
        kalmanComputeA<<<1, blockA>>>(d_H, d_Xn, d_An, N, m);
        kalmanAddDiag<<<(m + 31) / 32, 32>>>(d_An, m, 1.0 / etaUsed);
        kalmanInvertGaussJordan<<<1, 1>>>(d_An, d_Ainvn, d_Augn, m);
        kalmanComputeKNaive<<<gridX, blockX>>>(d_Xn, d_Ainvn, d_Kn, N, m);
        kalmanUpdatePNaive<<<gridP, blockP>>>(d_P_naive, d_Kn, d_Xn, N, m, qUsed);
        cudaFree(d_Xn); cudaFree(d_An); cudaFree(d_Ainvn); cudaFree(d_Augn); cudaFree(d_Kn);

        CUDA_CHECK(cudaDeviceSynchronize());

        cpuUpdate(Pcpu, wCpu, H, xi, N, m, etaUsed, qUsed);
        kf.setJacobian(H.data(), m);
        kf.setError(xi.data(), m);
        kf.update(m);

        if (q > qmin) q *= exp(-qtau);

        vector<double> Pgemm(N * N), Pnaive(N * N), wGemmStep(N);
        CUDA_CHECK(cudaMemcpy(Pgemm.data(), d_P_gemm, sizeof(double) * N * N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Pnaive.data(), d_P_naive, sizeof(double) * N * N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(wGemmStep.data(), d_w, sizeof(double) * N, cudaMemcpyDeviceToHost));

        double diffPn = 0.0, diffWc = 0.0, diffWr = 0.0;
        for (int i = 0; i < N * N; ++i) diffPn = max(diffPn, fabs(Pgemm[i] - Pnaive[i]));
        for (int i = 0; i < N; ++i)
        {
            diffWc = max(diffWc, fabs(wGemmStep[i] - wCpu[i]));
            diffWr = max(diffWr, fabs(wGemmStep[i] - wReal[i]));
        }
        maxDiffGemmNaiveP = max(maxDiffGemmNaiveP, diffPn);
        maxDiffGemmCpuW = max(maxDiffGemmCpuW, diffWc);
        maxDiffGemmRealW = max(maxDiffGemmRealW, diffWr);

        printf("  update %d: |P_gemm-P_naive|=%.3E  |w_gemm-w_cpu|=%.3E  "
               "|w_gemm-w_real|=%.3E\n", t, diffPn, diffWc, diffWr);
    }

    printf("  max over all updates: |P_gemm-P_naive|=%.3E  |w_gemm-w_cpu|=%.3E  "
           "|w_gemm-w_real|=%.3E\n", maxDiffGemmNaiveP, maxDiffGemmCpuW, maxDiffGemmRealW);

    // --- Timing: one more update, repeated, GEMM path vs. naive path -------
    vector<double> Hf(N * m), xif(m);
    for (auto& v : Hf) v = dist(rng);
    for (auto& v : xif) v = dist(rng);
    CUDA_CHECK(cudaMemcpy(d_H, Hf.data(), sizeof(double) * N * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xi, xif.data(), sizeof(double) * m, cudaMemcpyHostToDevice));

    int const reps = (N > 1000) ? 50 : 500;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r)
    {
        computeX(handle, N, m, d_P_gemm, d_H, d_X);
        kalmanComputeA<<<1, blockA>>>(d_H, d_X, d_A, N, m);
        kalmanAddDiag<<<(m + 31) / 32, 32>>>(d_A, m, 1.0 / eta);
        kalmanInvertGaussJordan<<<1, 1>>>(d_A, d_Ainv, d_Aug, m);
        computeK(handle, N, m, d_X, d_Ainv, d_K);
        updatePGemmTerm(handle, N, m, d_K, d_X, d_P_gemm);
        kalmanAddDiag<<<(N + 255) / 256, 256>>>(d_P_gemm, N, q);
        kalmanUpdateW<<<gridW, blockW>>>(d_w, d_K, d_xi, N, m);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float msGemm = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msGemm, t0, t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r)
    {
        double *d_Xn, *d_An, *d_Ainvn, *d_Augn, *d_Kn;
        CUDA_CHECK(cudaMalloc(&d_Xn, sizeof(double) * N * m));
        CUDA_CHECK(cudaMalloc(&d_An, sizeof(double) * m * m));
        CUDA_CHECK(cudaMalloc(&d_Ainvn, sizeof(double) * m * m));
        CUDA_CHECK(cudaMalloc(&d_Augn, sizeof(double) * m * 2 * m));
        CUDA_CHECK(cudaMalloc(&d_Kn, sizeof(double) * N * m));
        kalmanComputeXNaive<<<gridX, blockX>>>(d_P_naive, d_H, d_Xn, N, m);
        kalmanComputeA<<<1, blockA>>>(d_H, d_Xn, d_An, N, m);
        kalmanAddDiag<<<(m + 31) / 32, 32>>>(d_An, m, 1.0 / eta);
        kalmanInvertGaussJordan<<<1, 1>>>(d_An, d_Ainvn, d_Augn, m);
        kalmanComputeKNaive<<<gridX, blockX>>>(d_Xn, d_Ainvn, d_Kn, N, m);
        kalmanUpdatePNaive<<<gridP, blockP>>>(d_P_naive, d_Kn, d_Xn, N, m, q);
        cudaFree(d_Xn); cudaFree(d_An); cudaFree(d_Ainvn); cudaFree(d_Augn); cudaFree(d_Kn);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float msNaive = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msNaive, t0, t1));

    printf("  timing (%d reps): gemm=%.4f ms/update, naive=%.4f ms/update (%.2fx)\n",
           reps, msGemm / reps, msNaive / reps, msNaive / msGemm);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_P_gemm); cudaFree(d_P_naive); cudaFree(d_H); cudaFree(d_xi);
    cudaFree(d_X); cudaFree(d_A); cudaFree(d_Ainv); cudaFree(d_Aug);
    cudaFree(d_K); cudaFree(d_w);

    double tol = 1e-8;
    bool pass = maxDiffGemmNaiveP < tol && maxDiffGemmCpuW < tol && maxDiffGemmRealW < tol;
    printf("  %s: %s\n", label, pass ? "PASS" : "FAIL");
    return pass;
}

int main()
{
    double const epsilon = 1.0e-2, q0 = 0.01, qtau = 2.302, qmin = 1.0e-6;
    double const eta0 = 0.01, etatau = 2.302, etamax = 1.0;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    bool ok = true;
    ok &= runCase(handle, "small", 50, 5, 5, epsilon, q0, qtau, qmin,
                  eta0, etatau, etamax, 12345);
    ok &= runCase(handle, "production-sized (H2O_2G, N=3327, m=32)", 3327, 32, 5,
                  epsilon, q0, qtau, qmin, eta0, etatau, etamax, 67890);

    cublasDestroy(handle);
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
