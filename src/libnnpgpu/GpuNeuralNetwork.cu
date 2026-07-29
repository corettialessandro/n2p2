// See GpuNeuralNetwork.h. Implementation ported from
// gpu/gemm/nn_forward_gemm_test.cu (already validated there to ~6e-15
// against both the one-thread-per-atom GPU kernel and the real
// nnp::NeuralNetwork class, and timed at 25.6x/27.9x over the per-atom
// kernel for this project's real H2O_2G element sizes), generalized here
// to run-time hidden-layer sizes -- that test file hardcoded numHidden1=
// numHidden2=25 since it only ever exercised H2O_2G's particular input.nn;
// this library is called from the real Mode::calculateAtomicNeuralNetworks()
// on whatever dataset is loaded, so the sizes must be real parameters, not
// compile-time constants (see NeuralNetwork::hasGpuCompatibleArchitecture(),
// which callers use to confirm the *shape* of the network -- exactly two
// tanh hidden layers, single identity output neuron -- is one this code
// supports, before calling in with the actual sizes).
//
// Persistent per-architecture device state (added after the first real
// end-to-end nnp-train timing run showed why it mattered): each of the
// three functions below used to cudaMalloc every buffer it needed --
// roughly three dozen for gpuNnForceDFdcSum() -- and cudaFree all of them
// again before returning, on EVERY call. That's the same mistake
// GpuKalmanFilter.h/.cu was deliberately designed to avoid for P (see its
// own header comment), just not noticed here at first because these
// functions were validated for correctness in isolation, not benchmarked
// at the real calling frequency (once per update candidate, ~300+ times
// per epoch) until a full end-to-end run with GPU contention resolved
// (via MPS -- see gpu/README.md) exposed the remaining gap between
// "kernel is 10x faster in isolation" and "call site is 1.3x faster
// end to end": cudaMalloc/cudaFree carry real fixed driver overhead
// independent of matrix size, and at this call frequency that overhead,
// not the actual compute, was dominating the wall-clock cost.
//
// Fix: weight-dependent buffers (sized by numIn/numHidden1/numHidden2,
// fixed for a given element's architecture, only their VALUES change
// call to call since Kalman updates weights every time) and atom-
// dependent buffers (sized by numAtoms, which varies call to call) are
// now cached in a process-lifetime state per (numIn, numHidden1,
// numHidden2) triple -- in practice one entry per distinct element
// architecture (H, O, ...). Weight buffers are allocated once and
// re-uploaded (cudaMemcpy, not cudaMalloc) every call; atom-dependent
// buffers grow on demand (never shrink), the same pattern
// GpuKalmanFilter.cu's ensureMCapacity() already uses for the analogous
// problem. The three functions' public signatures are unchanged -- this
// is purely an internal caching layer, transparent to every caller
// (Mode.cpp, Training.cpp).

#include "GpuNeuralNetwork.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <tuple>
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

// Lazily-initialized, process-lifetime cuBLAS handle -- avoids paying
// handle-creation cost on every call (this function is expected to be
// called once per element per structure, i.e. often over a training run).
// Not thread-safe; fine as long as callers don't invoke this concurrently
// from multiple host threads (n2p2 parallelizes across MPI ranks, not
// OpenMP, for this call path -- OpenMP is disabled by default, see
// src/makefile.gnu's PROJECT_CFLAGS comment).
cublasHandle_t& gpuHandle()
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) CUBLAS_CHECK(cublasCreate(&handle));
    return handle;
}

// Row-major C(m x n) = A(m x k) . B(k x n) -- see gpu/gemm/'s identical
// helper (nn_forward_gemm_test.cu, nn_dedc_gemm_test.cu,
// nn_dfdc_gemm_test.cu) for the row-major-via-column-major derivation,
// already validated there many times over.
void gemmRowMajor(cublasHandle_t handle, int m, int n, int k,
                   double const* A, double const* B, double* C)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k, &alpha, B, n, A, k, &beta, C, n));
}

__global__ void biasTanhKernel(double const* pre, double const* b,
                                int numAtoms, int width, double* H, double* dfdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int col = idx % width;
    double h = tanh(pre[idx] + b[col]);
    H[idx] = h;
    dfdx[idx] = 1.0 - h * h;
}

__global__ void addOutputBiasKernel(double const* pre, double b3,
                                     int numAtoms, double* energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;
    energy[i] = pre[i] + b3;
}

__global__ void scaleByRowKernel(double const* A, double const* row,
                                  int numAtoms, int width, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    out[idx] = A[idx] * row[idx % width];
}

__global__ void elementwiseMulKernel(double const* a, double const* b, int n, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

__global__ void fillOnesKernel(double* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = 1.0;
}

// Row-major C(m x n) = alpha * A^T(m x k) . B(k x n) + beta * C, where A is
// stored row-major (k x m) (so A^T is the (m x k) matrix actually
// multiplied) and B is row-major (k x n) -- i.e. the same
// row-major-via-column-major trick as gemmRowMajor() above, but contracting
// over A's FIRST (row) axis instead of its second. Used to reduce/sum over
// the atom (batch) axis directly via GEMM (e.g. dE/dW1 summed over atoms is
// G^T . v1s, with the atom axis as the contracted dimension k) instead of a
// separate reduction kernel plus host-side summation. The explicit
// alpha/beta (mirroring gpu/gemm/nn_dfdc_gemm_test.cu's
// outerProductBatchedAcc) let callers fold in calculateDFdc's "-="
// convention and accumulate across repeated calls (e.g. every iteration of
// a k0 loop) directly in the GEMM itself, beta=1 accumulating into
// whatever C already holds.
void gemmRowMajorATransBAcc(cublasHandle_t handle, int m, int n, int k,
                             double alpha, double const* A, double const* B,
                             double beta, double* C)
{
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             n, m, k, &alpha, B, n, A, m, &beta, C, n));
}

void gemmRowMajorATransB(cublasHandle_t handle, int m, int n, int k,
                          double const* A, double const* B, double* C)
{
    gemmRowMajorATransBAcc(handle, m, n, k, 1.0, A, B, 0.0, C);
}

__global__ void biasTanhD2Kernel(double const* pre, double const* b,
                                  int numAtoms, int width,
                                  double* H, double* dfdx, double* d2fdx2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int col = idx % width;
    double h = tanh(pre[idx] + b[col]);
    H[idx] = h;
    double dh = 1.0 - h * h;
    dfdx[idx] = dh;
    d2fdx2[idx] = -2.0 * h * dh;
}

__global__ void elementwiseAddKernel(double const* a, double const* b, int n, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

// out[atom,col] = A[atom,col] * colVec[atom] -- broadcasts a per-atom
// SCALAR across the width dimension (the opposite broadcast direction from
// scaleByRowKernel, which broadcasts a per-column value across atoms).
// Requires colVec to be contiguous (length numAtoms) -- see transposeKernel
// below for why dGdxyz's k0-th "column" needs pre-transposing first.
__global__ void scaleByColKernel(double const* A, double const* colVec,
                                  int numAtoms, int width, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int atom = idx / width;
    out[idx] = A[idx] * colVec[atom];
}

// out(width x numAtoms row-major) = in(numAtoms x width row-major)^T --
// used once to transpose dGdxyz so that, inside the k0 loop below, the k0-th
// "column" of the original (numAtoms x numIn) array is a CONTIGUOUS
// (length-numAtoms) row of the transposed array -- required because the
// atom-axis-contracting GEMM helper above needs its A operand contiguous,
// and dGdxyz's per-k0 column is strided (stride numIn) in its original
// atom-major layout.
__global__ void transposeKernel(double const* in, int numAtoms, int width, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int atom = idx / width, col = idx % width;
    out[(size_t)col * numAtoms + atom] = in[idx];
}

// Architecture key: a given element's (numIn, numHidden1, numHidden2)
// never changes during a run, so it uniquely identifies which persistent
// state a call should reuse. If two different elements happened to share
// the exact same triple, they'd correctly share one state too (every call
// fully overwrites the weight buffers before use, and n2p2 doesn't call
// these functions concurrently across elements), just without a
// performance-irrelevant separate allocation.
using ArchKey = std::tuple<int, int, int>;

void hostTransposeW1(double const* W1, int numIn, int numHidden1, double* W1T)
{
    for (int j = 0; j < numIn; ++j)
        for (int i = 0; i < numHidden1; ++i)
            W1T[(size_t)i * numIn + j] = W1[(size_t)j * numHidden1 + i];
}

void hostTransposeW2(double const* W2, int numHidden1, int numHidden2, double* W2T)
{
    for (int i = 0; i < numHidden1; ++i)
        for (int j = 0; j < numHidden2; ++j)
            W2T[(size_t)j * numHidden1 + i] = W2[(size_t)i * numHidden2 + j];
}

// --- gpuNnForwardDEdG's persistent state -----------------------------------

struct GpuNnForwardState
{
    int numIn = 0, numHidden1 = 0, numHidden2 = 0;
    int atomCapacity = 0;

    // Weight-dependent (fixed size, alloc once; re-uploaded every call).
    double *d_W1 = nullptr, *d_b1 = nullptr, *d_W2 = nullptr, *d_b2 = nullptr;
    double *d_W3 = nullptr, *d_W1T = nullptr, *d_W2T = nullptr;
    std::vector<double> W1Thost, W2Thost;

    // Atom-dependent (grow on demand).
    double *d_G = nullptr, *d_H1pre = nullptr, *d_H1 = nullptr, *d_dfdx1 = nullptr;
    double *d_H2pre = nullptr, *d_H2 = nullptr, *d_dfdx2 = nullptr;
    double *d_outPre = nullptr, *d_energy = nullptr;
    double *d_v2 = nullptr, *d_v1 = nullptr, *d_v1s = nullptr, *d_dEdG = nullptr;
};

void ensureAtomCapacity(GpuNnForwardState& s, int numAtoms)
{
    if (numAtoms <= s.atomCapacity) return;
    cudaFree(s.d_G); cudaFree(s.d_H1pre); cudaFree(s.d_H1); cudaFree(s.d_dfdx1);
    cudaFree(s.d_H2pre); cudaFree(s.d_H2); cudaFree(s.d_dfdx2);
    cudaFree(s.d_outPre); cudaFree(s.d_energy);
    cudaFree(s.d_v2); cudaFree(s.d_v1); cudaFree(s.d_v1s); cudaFree(s.d_dEdG);

    int const numIn = s.numIn, numHidden1 = s.numHidden1, numHidden2 = s.numHidden2;
    CUDA_CHECK(cudaMalloc(&s.d_G, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_outPre, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_energy, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dEdG, (size_t)numAtoms * numIn * sizeof(double)));
    s.atomCapacity = numAtoms;
}

GpuNnForwardState& getForwardState(int numIn, int numHidden1, int numHidden2)
{
    static std::map<ArchKey, GpuNnForwardState*> cache;
    ArchKey key(numIn, numHidden1, numHidden2);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;

    GpuNnForwardState* s = new GpuNnForwardState();
    s->numIn = numIn; s->numHidden1 = numHidden1; s->numHidden2 = numHidden2;
    s->W1Thost.resize((size_t)numHidden1 * numIn);
    s->W2Thost.resize((size_t)numHidden2 * numHidden1);
    CUDA_CHECK(cudaMalloc(&s->d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_b1, (size_t)numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_b2, (size_t)numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W3, (size_t)numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W1T, (size_t)numHidden1 * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
    cache[key] = s;
    return *s;
}

// --- gpuNnEnergyDEdcSum's persistent state ----------------------------------

struct GpuNnEnergyState
{
    int numIn = 0, numHidden1 = 0, numHidden2 = 0;
    int atomCapacity = 0;

    double *d_W1 = nullptr, *d_b1 = nullptr, *d_W2 = nullptr, *d_b2 = nullptr;
    double *d_W3 = nullptr, *d_W2T = nullptr;
    std::vector<double> W2Thost;
    double* d_dEdcSum = nullptr; // connCount-sized, fixed per architecture

    double *d_G = nullptr, *d_H1pre = nullptr, *d_H1 = nullptr, *d_dfdx1 = nullptr;
    double *d_H2pre = nullptr, *d_H2 = nullptr, *d_dfdx2 = nullptr;
    double *d_outPre = nullptr, *d_energy = nullptr;
    double *d_v2 = nullptr, *d_v1 = nullptr, *d_v1s = nullptr, *d_ones = nullptr;
};

void ensureAtomCapacity(GpuNnEnergyState& s, int numAtoms)
{
    if (numAtoms <= s.atomCapacity) return;
    cudaFree(s.d_G); cudaFree(s.d_H1pre); cudaFree(s.d_H1); cudaFree(s.d_dfdx1);
    cudaFree(s.d_H2pre); cudaFree(s.d_H2); cudaFree(s.d_dfdx2);
    cudaFree(s.d_outPre); cudaFree(s.d_energy);
    cudaFree(s.d_v2); cudaFree(s.d_v1); cudaFree(s.d_v1s); cudaFree(s.d_ones);

    int const numIn = s.numIn, numHidden1 = s.numHidden1, numHidden2 = s.numHidden2;
    CUDA_CHECK(cudaMalloc(&s.d_G, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_outPre, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_energy, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_ones, (size_t)numAtoms * sizeof(double)));
    s.atomCapacity = numAtoms;
}

GpuNnEnergyState& getEnergyState(int numIn, int numHidden1, int numHidden2)
{
    static std::map<ArchKey, GpuNnEnergyState*> cache;
    ArchKey key(numIn, numHidden1, numHidden2);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;

    int const numOut = 1;
    size_t const connCount = (size_t)numIn * numHidden1 + numHidden1
                            + (size_t)numHidden1 * numHidden2 + numHidden2
                            + (size_t)numHidden2 * numOut + numOut;

    GpuNnEnergyState* s = new GpuNnEnergyState();
    s->numIn = numIn; s->numHidden1 = numHidden1; s->numHidden2 = numHidden2;
    s->W2Thost.resize((size_t)numHidden2 * numHidden1);
    CUDA_CHECK(cudaMalloc(&s->d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_b1, (size_t)numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_b2, (size_t)numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W3, (size_t)numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_dEdcSum, connCount * sizeof(double)));
    cache[key] = s;
    return *s;
}

// --- gpuNnForceDFdcSum's persistent state ------------------------------------

struct GpuNnForceState
{
    int numIn = 0, numHidden1 = 0, numHidden2 = 0;
    int atomCapacity = 0;

    double *d_W1 = nullptr, *d_b1 = nullptr, *d_W2 = nullptr, *d_b2 = nullptr;
    double *d_W3 = nullptr, *d_W1T = nullptr, *d_W2T = nullptr;
    std::vector<double> W1Thost, W2Thost;
    double* d_dFdcSum = nullptr; // connCount-sized, fixed per architecture

    double *d_G = nullptr, *d_dGdxyz = nullptr, *d_dGdxyzT = nullptr;
    double *d_H1pre = nullptr, *d_H1 = nullptr, *d_dfdx1 = nullptr, *d_d2fdx2_1 = nullptr;
    double *d_H2pre = nullptr, *d_H2 = nullptr, *d_dfdx2 = nullptr, *d_d2fdx2_2 = nullptr;
    double *d_outPre = nullptr, *d_energy = nullptr;
    double *d_v2 = nullptr, *d_v1 = nullptr, *d_v1s = nullptr, *d_d2S1 = nullptr, *d_dEdG = nullptr;
    double *d_u = nullptr, *d_dxdG2 = nullptr, *d_tmp = nullptr, *d_jacBH2 = nullptr, *d_jacW3 = nullptr;
    double *d_T = nullptr, *d_term1 = nullptr, *d_term2 = nullptr, *d_jacBH1 = nullptr;
    double *d_P = nullptr, *d_Q2 = nullptr, *d_Gscaled = nullptr;
};

void ensureAtomCapacity(GpuNnForceState& s, int numAtoms)
{
    if (numAtoms <= s.atomCapacity) return;
    cudaFree(s.d_G); cudaFree(s.d_dGdxyz); cudaFree(s.d_dGdxyzT);
    cudaFree(s.d_H1pre); cudaFree(s.d_H1); cudaFree(s.d_dfdx1); cudaFree(s.d_d2fdx2_1);
    cudaFree(s.d_H2pre); cudaFree(s.d_H2); cudaFree(s.d_dfdx2); cudaFree(s.d_d2fdx2_2);
    cudaFree(s.d_outPre); cudaFree(s.d_energy);
    cudaFree(s.d_v2); cudaFree(s.d_v1); cudaFree(s.d_v1s); cudaFree(s.d_d2S1); cudaFree(s.d_dEdG);
    cudaFree(s.d_u); cudaFree(s.d_dxdG2); cudaFree(s.d_tmp); cudaFree(s.d_jacBH2); cudaFree(s.d_jacW3);
    cudaFree(s.d_T); cudaFree(s.d_term1); cudaFree(s.d_term2); cudaFree(s.d_jacBH1);
    cudaFree(s.d_P); cudaFree(s.d_Q2); cudaFree(s.d_Gscaled);

    int const numIn = s.numIn, numHidden1 = s.numHidden1, numHidden2 = s.numHidden2;
    CUDA_CHECK(cudaMalloc(&s.d_G, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dGdxyz, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dGdxyzT, (size_t)numIn * numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_d2fdx2_1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_d2fdx2_2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_outPre, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_energy, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_d2S1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dEdG, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_u, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_dxdG2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_tmp, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_jacBH2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_jacW3, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_T, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_term1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_term2, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_jacBH1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_P, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_Q2, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s.d_Gscaled, (size_t)numAtoms * numIn * sizeof(double)));
    s.atomCapacity = numAtoms;
}

GpuNnForceState& getForceState(int numIn, int numHidden1, int numHidden2)
{
    static std::map<ArchKey, GpuNnForceState*> cache;
    ArchKey key(numIn, numHidden1, numHidden2);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;

    int const numOut = 1;
    size_t const connCount = (size_t)numIn * numHidden1 + numHidden1
                            + (size_t)numHidden1 * numHidden2 + numHidden2
                            + (size_t)numHidden2 * numOut + numOut;

    GpuNnForceState* s = new GpuNnForceState();
    s->numIn = numIn; s->numHidden1 = numHidden1; s->numHidden2 = numHidden2;
    s->W1Thost.resize((size_t)numHidden1 * numIn);
    s->W2Thost.resize((size_t)numHidden2 * numHidden1);
    CUDA_CHECK(cudaMalloc(&s->d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_b1, (size_t)numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_b2, (size_t)numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W3, (size_t)numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W1T, (size_t)numHidden1 * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_dFdcSum, connCount * sizeof(double)));
    cache[key] = s;
    return *s;
}

} // anonymous namespace

namespace nnp
{

void gpuNnForwardDEdG(int numAtoms, int numIn, int numHidden1, int numHidden2,
                      double const* connections,
                      double const* G, double* energyOut, double* dEdGOut)
{
    int const numOut = 1;
    cublasHandle_t handle = gpuHandle();
    GpuNnForwardState& s = getForwardState(numIn, numHidden1, numHidden2);
    ensureAtomCapacity(s, numAtoms);

    size_t off = 0;
    double const* W1 = connections + off; off += (size_t)numIn * numHidden1;
    double const* b1 = connections + off; off += numHidden1;
    double const* W2 = connections + off; off += (size_t)numHidden1 * numHidden2;
    double const* b2 = connections + off; off += numHidden2;
    double const* W3 = connections + off; off += (size_t)numHidden2 * numOut;
    double const  b3 = connections[off];

    hostTransposeW1(W1, numIn, numHidden1, s.W1Thost.data());
    hostTransposeW2(W2, numHidden1, numHidden2, s.W2Thost.data());

    CUDA_CHECK(cudaMemcpy(s.d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W1T, s.W1Thost.data(), (size_t)numHidden1 * numIn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W2T, s.W2Thost.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_G, G, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

    int const blk = 256;

    gemmRowMajor(handle, numAtoms, numHidden1, numIn, s.d_G, s.d_W1, s.d_H1pre);
    biasTanhKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        s.d_H1pre, s.d_b1, numAtoms, numHidden1, s.d_H1, s.d_dfdx1);

    gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, s.d_H1, s.d_W2, s.d_H2pre);
    biasTanhKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        s.d_H2pre, s.d_b2, numAtoms, numHidden2, s.d_H2, s.d_dfdx2);

    gemmRowMajor(handle, numAtoms, numOut, numHidden2, s.d_H2, s.d_W3, s.d_outPre);
    addOutputBiasKernel<<<(numAtoms + blk - 1) / blk, blk>>>(s.d_outPre, b3, numAtoms, s.d_energy);

    scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        s.d_dfdx2, s.d_W3, numAtoms, numHidden2, s.d_v2);
    gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, s.d_v2, s.d_W2T, s.d_v1);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        s.d_dfdx1, s.d_v1, numAtoms * numHidden1, s.d_v1s);
    gemmRowMajor(handle, numAtoms, numIn, numHidden1, s.d_v1s, s.d_W1T, s.d_dEdG);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(energyOut, s.d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dEdGOut, s.d_dEdG, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpuNnEnergyDEdcSum(int numAtoms, int numIn, int numHidden1, int numHidden2,
                        double const* connections,
                        double const* G, double* energyOut, double* dEdcSumOut)
{
    int const numOut = 1;
    cublasHandle_t handle = gpuHandle();
    GpuNnEnergyState& s = getEnergyState(numIn, numHidden1, numHidden2);
    ensureAtomCapacity(s, numAtoms);

    size_t off = 0;
    double const* W1 = connections + off; off += (size_t)numIn * numHidden1;
    double const* b1 = connections + off; off += numHidden1;
    double const* W2 = connections + off; off += (size_t)numHidden1 * numHidden2;
    double const* b2 = connections + off; off += numHidden2;
    double const* W3 = connections + off; off += (size_t)numHidden2 * numOut;
    double const  b3 = connections[off];

    // Offsets into the flat dEdc/connections layout ([W1,b1,W2,b2,W3,b3]),
    // matching NeuralNetwork::calculateDEdc()'s own offset bookkeeping.
    size_t const offW1 = 0;
    size_t const offB1 = offW1 + (size_t)numIn * numHidden1;
    size_t const offW2 = offB1 + numHidden1;
    size_t const offB2 = offW2 + (size_t)numHidden1 * numHidden2;
    size_t const offW3 = offB2 + numHidden2;
    size_t const offB3 = offW3 + (size_t)numHidden2 * numOut;

    hostTransposeW2(W2, numHidden1, numHidden2, s.W2Thost.data());

    CUDA_CHECK(cudaMemcpy(s.d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W2T, s.W2Thost.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_G, G, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

    int const blk = 256;

    fillOnesKernel<<<(numAtoms + blk - 1) / blk, blk>>>(s.d_ones, numAtoms);

    gemmRowMajor(handle, numAtoms, numHidden1, numIn, s.d_G, s.d_W1, s.d_H1pre);
    biasTanhKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        s.d_H1pre, s.d_b1, numAtoms, numHidden1, s.d_H1, s.d_dfdx1);

    gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, s.d_H1, s.d_W2, s.d_H2pre);
    biasTanhKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        s.d_H2pre, s.d_b2, numAtoms, numHidden2, s.d_H2, s.d_dfdx2);

    gemmRowMajor(handle, numAtoms, numOut, numHidden2, s.d_H2, s.d_W3, s.d_outPre);
    addOutputBiasKernel<<<(numAtoms + blk - 1) / blk, blk>>>(s.d_outPre, b3, numAtoms, s.d_energy);

    // v2 = dfdx2 .* W3 (broadcast), v1s = dfdx1 .* (v2 . W2^T) -- identical
    // to gpuNnForwardDEdG()'s dEdG pipeline (both dEdG and dEdc reuse these
    // same intermediates, see gpu/gemm/nn_dedc_gemm_test.cu's header
    // comment), just consumed differently below.
    scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        s.d_dfdx2, s.d_W3, numAtoms, numHidden2, s.d_v2);
    gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, s.d_v2, s.d_W2T, s.d_v1);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        s.d_dfdx1, s.d_v1, numAtoms * numHidden1, s.d_v1s);

    // Sum over the atom axis, straight into the packed dEdc-sum output --
    // each of these contracts numAtoms via a GEMM instead of a per-atom
    // outer product plus a separate reduction.
    gemmRowMajorATransB(handle, numIn, numHidden1, numAtoms,
                         s.d_G, s.d_v1s, s.d_dEdcSum + offW1);
    gemmRowMajorATransB(handle, 1, numHidden1, numAtoms,
                         s.d_ones, s.d_v1s, s.d_dEdcSum + offB1);
    gemmRowMajorATransB(handle, numHidden1, numHidden2, numAtoms,
                         s.d_H1, s.d_v2, s.d_dEdcSum + offW2);
    gemmRowMajorATransB(handle, 1, numHidden2, numAtoms,
                         s.d_ones, s.d_v2, s.d_dEdcSum + offB2);
    gemmRowMajorATransB(handle, 1, numHidden2, numAtoms,
                         s.d_ones, s.d_H2, s.d_dEdcSum + offW3);
    // dE/db3 summed over atoms is exactly numAtoms (each atom contributes a
    // constant 1) -- set directly on the host below, no kernel needed.

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(energyOut, s.d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dEdcSumOut, s.d_dEdcSum, offB3 * sizeof(double), cudaMemcpyDeviceToHost));
    dEdcSumOut[offB3] = (double)numAtoms;
}

void gpuNnForceDFdcSum(int numAtoms, int numIn, int numHidden1, int numHidden2,
                       double const* connections,
                       double const* G, double const* dGdxyz,
                       double* energyOut, double* dEdGOut, double* dFdcSumOut)
{
    int const numOut = 1;
    cublasHandle_t handle = gpuHandle();
    GpuNnForceState& s = getForceState(numIn, numHidden1, numHidden2);
    ensureAtomCapacity(s, numAtoms);

    size_t off = 0;
    double const* W1 = connections + off; off += (size_t)numIn * numHidden1;
    double const* b1 = connections + off; off += numHidden1;
    double const* W2 = connections + off; off += (size_t)numHidden1 * numHidden2;
    double const* b2 = connections + off; off += numHidden2;
    double const* W3 = connections + off; off += (size_t)numHidden2 * numOut;
    double const  b3 = connections[off];

    size_t const offW1 = 0;
    size_t const offB1 = offW1 + (size_t)numIn * numHidden1;
    size_t const offW2 = offB1 + numHidden1;
    size_t const offB2 = offW2 + (size_t)numHidden1 * numHidden2;
    size_t const offW3 = offB2 + numHidden2;
    size_t const offB3 = offW3 + (size_t)numHidden2 * numOut;
    size_t const connCount = offB3 + 1;

    hostTransposeW1(W1, numIn, numHidden1, s.W1Thost.data());
    hostTransposeW2(W2, numHidden1, numHidden2, s.W2Thost.data());

    CUDA_CHECK(cudaMemcpy(s.d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W1T, s.W1Thost.data(), (size_t)numHidden1 * numIn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_W2T, s.W2Thost.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_G, G, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_dGdxyz, dGdxyz, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(s.d_dFdcSum, 0, connCount * sizeof(double)));

    int const blk = 256;

    // --- Precompute once, batched over atoms, no k0 dependence yet -------
    gemmRowMajor(handle, numAtoms, numHidden1, numIn, s.d_G, s.d_W1, s.d_H1pre);
    biasTanhD2Kernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        s.d_H1pre, s.d_b1, numAtoms, numHidden1, s.d_H1, s.d_dfdx1, s.d_d2fdx2_1);

    gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, s.d_H1, s.d_W2, s.d_H2pre);
    biasTanhD2Kernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        s.d_H2pre, s.d_b2, numAtoms, numHidden2, s.d_H2, s.d_dfdx2, s.d_d2fdx2_2);

    gemmRowMajor(handle, numAtoms, numOut, numHidden2, s.d_H2, s.d_W3, s.d_outPre);
    addOutputBiasKernel<<<(numAtoms + blk - 1) / blk, blk>>>(s.d_outPre, b3, numAtoms, s.d_energy);

    // v2/v1/v1s/dEdG: identical to gpuNnForwardDEdG()'s pipeline (this
    // function needs it->energy/it->dEdG populated correctly too, since
    // Mode::calculateForces() reads them afterward).
    scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        s.d_dfdx2, s.d_W3, numAtoms, numHidden2, s.d_v2);
    gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, s.d_v2, s.d_W2T, s.d_v1);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        s.d_dfdx1, s.d_v1, numAtoms * numHidden1, s.d_v1s);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        s.d_d2fdx2_1, s.d_v1, numAtoms * numHidden1, s.d_d2S1);
    gemmRowMajor(handle, numAtoms, numIn, numHidden1, s.d_v1s, s.d_W1T, s.d_dEdG);

    transposeKernel<<<(numAtoms * numIn + blk - 1) / blk, blk>>>(
        s.d_dGdxyz, numAtoms, numIn, s.d_dGdxyzT);

    // --- Host-side loop over each input (cheap: numIn is at most a few
    // dozen for any real dataset) -- see nn_dfdc_gemm_test.cu's header
    // comment for the full derivation this mirrors, minus the per-atom
    // batched-rank-1 outer products it needed (replaced here by ordinary
    // atom-axis-contracting GEMMs, since only the atom-summed result is
    // ever needed -- see GpuNeuralNetwork.h's doc comment).
    for (int k0 = 0; k0 < numIn; ++k0)
    {
        double const* W1row = s.d_W1 + (size_t)k0 * numHidden1;
        double const* dGk0 = s.d_dGdxyzT + (size_t)k0 * numAtoms;

        scaleByRowKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            s.d_dfdx1, W1row, numAtoms, numHidden1, s.d_u);
        gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, s.d_u, s.d_W2, s.d_dxdG2);
        elementwiseMulKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
            s.d_d2fdx2_2, s.d_dxdG2, numAtoms * numHidden2, s.d_tmp);
        scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
            s.d_tmp, s.d_W3, numAtoms, numHidden2, s.d_jacBH2);
        elementwiseMulKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
            s.d_dfdx2, s.d_dxdG2, numAtoms * numHidden2, s.d_jacW3);

        gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, s.d_jacBH2, s.d_W2T, s.d_T);
        elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            s.d_dfdx1, s.d_T, numAtoms * numHidden1, s.d_term1);
        scaleByRowKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            s.d_d2S1, W1row, numAtoms, numHidden1, s.d_term2);
        elementwiseAddKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            s.d_term1, s.d_term2, numAtoms * numHidden1, s.d_jacBH1);

        // Sum-reductions over the atom axis, accumulating (beta=1) across
        // every k0 iteration directly into the packed dFdc-sum output.
        gemmRowMajorATransBAcc(handle, 1, numHidden2, numAtoms, -1.0,
                                dGk0, s.d_jacW3, 1.0, s.d_dFdcSum + offW3);
        gemmRowMajorATransBAcc(handle, 1, numHidden2, numAtoms, -1.0,
                                dGk0, s.d_jacBH2, 1.0, s.d_dFdcSum + offB2);
        gemmRowMajorATransBAcc(handle, 1, numHidden1, numAtoms, -1.0,
                                dGk0, s.d_jacBH1, 1.0, s.d_dFdcSum + offB1);

        scaleByColKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            s.d_H1, dGk0, numAtoms, numHidden1, s.d_P);
        scaleByColKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            s.d_u, dGk0, numAtoms, numHidden1, s.d_Q2);
        gemmRowMajorATransBAcc(handle, numHidden1, numHidden2, numAtoms, -1.0,
                                s.d_P, s.d_jacBH2, 1.0, s.d_dFdcSum + offW2);
        gemmRowMajorATransBAcc(handle, numHidden1, numHidden2, numAtoms, -1.0,
                                s.d_Q2, s.d_v2, 1.0, s.d_dFdcSum + offW2);

        scaleByColKernel<<<(numAtoms * numIn + blk - 1) / blk, blk>>>(
            s.d_G, dGk0, numAtoms, numIn, s.d_Gscaled);
        gemmRowMajorATransBAcc(handle, numIn, numHidden1, numAtoms, -1.0,
                                s.d_Gscaled, s.d_jacBH1, 1.0, s.d_dFdcSum + offW1);

        // calculateDFdc's "deltaTerm": row k0 (only) of dFdc's W1 block
        // additionally gets -= sum_atom v1s[atom,:]*dGk0[atom], on top of
        // whatever the accumulation just above already wrote there.
        gemmRowMajorATransBAcc(handle, 1, numHidden1, numAtoms, -1.0,
                                dGk0, s.d_v1s, 1.0,
                                s.d_dFdcSum + offW1 + (size_t)k0 * numHidden1);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(energyOut, s.d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dEdGOut, s.d_dEdG, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dFdcSumOut, s.d_dFdcSum, connCount * sizeof(double), cudaMemcpyDeviceToHost));
}

} // namespace nnp
