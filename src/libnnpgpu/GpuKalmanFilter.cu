// See GpuKalmanFilter.h. Implementation ported from
// gpu/gemm/kalman_gemm_test.cu (cuBLAS for X=P.H and the K.X^T term of
// P -= K.X^T, already validated there against gpu/kalman/kalman_test.cu's
// naive kernels, an independent CPU reference, and the real
// nnp::KalmanFilter class, over multiple update() calls at both a small
// size and the real H2O_2G production size N=3327/m=32).
//
// Unlike every other file in src/libnnpgpu, P/H/X/K here are NOT row-major
// -- they match Eigen::Map<MatrixXd>'s natural column-major layout (see
// gpu/kalman/kalman_test.cu's header comment for why: KalmanFilter's real
// H/xi/w buffers are handed in with that layout already, and P is always
// symmetric in practice so row-major vs column-major makes no difference
// for it). cuBLAS's native column-major form is used directly throughout,
// no row-major-via-column-major trick.

#include "GpuKalmanFilter.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace
{

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__, \
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

// X (N x m, col-major) = P (N x N, col-major) . H (N x m, col-major).
// cublasDsymm, NOT cublasDgemm: the real KalmanFilter::update() computes
// this step as X = P.selfadjointView<Lower>() * H, i.e. it deliberately
// reads ONLY P's lower triangle (mirroring it for the upper half) rather
// than trusting whatever is literally stored in the full matrix. This
// matters because the *other* update (P -= K.X^T, below) is a plain dense
// subtraction with no symmetry enforcement, so P can accumulate genuine
// floating-point asymmetry over many update() calls (the class's own
// commented-out diagnostic, "Max. deviation of symmetric form of P",
// suggests this was known); the CPU path "self-heals" every iteration by
// only ever reading the lower triangle here, while an earlier version of
// this function used a plain full-matrix cublasDgemm, silently reading
// stale/inconsistent upper-triangle values too -- caught by a real
// end-to-end nnp-train run whose learning-curve.out diverged by ~4e4 after
// one training epoch (mostly fixed by an earlier correction to this file,
// a hand-rolled-inverse-vs-Eigen-inverse mismatch, but this residual
// asymmetry bug remained and needed its own fix). cublasDsymm treats P as
// symmetric via CUBLAS_FILL_MODE_LOWER, reading only its lower triangle --
// the direct GPU equivalent of Eigen's selfadjointView<Lower>().
void computeXGemm(cublasHandle_t handle, int N, int m,
                   double const* P, double const* H, double* X)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                             N, m, &alpha, P, N, H, N, &beta, X, N));
}

// P (N x N) -= K (N x m, col-major) . X^T (m x N) -- X^T via CUBLAS_OP_T
// directly on X's own buffer, no materialized transpose needed.
void updatePGemmTerm(cublasHandle_t handle, int N, int m,
                      double const* K, double const* X, double* P)
{
    double const alpha = -1.0, beta = 1.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N, m, &alpha, K, N, X, N, &beta, P, N));
}

__global__ void kalmanAddDiag(double* A, int m, double value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) A[i + i * m] += value;
}

} // anonymous namespace

namespace nnp
{

struct GpuKalmanFilterState
{
    int N = 0;
    int mCapacity = 0; // current allocated size of the m-dependent buffers

    double* dP = nullptr; // N x N, persistent across calls
    double* dH = nullptr; // N x mCapacity
    double* dX = nullptr; // N x mCapacity, persistent between the two calls
                          // of one update step (computeX writes it, updateP
                          // reads it back without re-upload)
    double* dK = nullptr; // N x mCapacity

    cublasHandle_t handle = nullptr;
};

namespace
{

void ensureMCapacity(GpuKalmanFilterState* s, int m)
{
    if (m <= s->mCapacity) return;

    if (s->dH) cudaFree(s->dH);
    if (s->dX) cudaFree(s->dX);
    if (s->dK) cudaFree(s->dK);

    CUDA_CHECK(cudaMalloc(&s->dH, (size_t)s->N * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->dX, (size_t)s->N * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->dK, (size_t)s->N * m * sizeof(double)));

    s->mCapacity = m;
}

} // anonymous namespace

GpuKalmanFilterState* gpuKalmanCreate(int N, double const* P0)
{
    GpuKalmanFilterState* s = new GpuKalmanFilterState();
    s->N = N;
    CUBLAS_CHECK(cublasCreate(&s->handle));

    CUDA_CHECK(cudaMalloc(&s->dP, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(s->dP, P0, (size_t)N * N * sizeof(double),
                          cudaMemcpyHostToDevice));

    return s;
}

void gpuKalmanComputeX(GpuKalmanFilterState* state, int m,
                       double const* H, double* X)
{
    ensureMCapacity(state, m);
    int const N = state->N;

    CUDA_CHECK(cudaMemcpy(state->dH, H, (size_t)N * m * sizeof(double),
                          cudaMemcpyHostToDevice));
    computeXGemm(state->handle, N, m, state->dP, state->dH, state->dX);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(X, state->dX, (size_t)N * m * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void gpuKalmanUpdateP(GpuKalmanFilterState* state, int m,
                      double const* K, double q)
{
    int const N = state->N;

    CUDA_CHECK(cudaMemcpy(state->dK, K, (size_t)N * m * sizeof(double),
                          cudaMemcpyHostToDevice));
    updatePGemmTerm(state->handle, N, m, state->dK, state->dX, state->dP);
    kalmanAddDiag<<<(N + 255) / 256, 256>>>(state->dP, N, q);
    CUDA_CHECK(cudaGetLastError());
}

void gpuKalmanGetP(GpuKalmanFilterState* state, double* P)
{
    CUDA_CHECK(cudaMemcpy(P, state->dP,
                          (size_t)state->N * state->N * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void gpuKalmanDestroy(GpuKalmanFilterState* state)
{
    if (!state) return;
    cudaFree(state->dP); cudaFree(state->dH);
    cudaFree(state->dX); cudaFree(state->dK);
    cublasDestroy(state->handle);
    delete state;
}

} // namespace nnp
