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

#include "GpuNeuralNetwork.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
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

} // anonymous namespace

namespace nnp
{

void gpuNnForwardDEdG(int numAtoms, int numIn, int numHidden1, int numHidden2,
                      double const* connections,
                      double const* G, double* energyOut, double* dEdGOut)
{
    int const numOut = 1;
    cublasHandle_t handle = gpuHandle();

    size_t off = 0;
    double const* W1 = connections + off; off += (size_t)numIn * numHidden1;
    double const* b1 = connections + off; off += numHidden1;
    double const* W2 = connections + off; off += (size_t)numHidden1 * numHidden2;
    double const* b2 = connections + off; off += numHidden2;
    double const* W3 = connections + off; off += (size_t)numHidden2 * numOut;
    double const  b3 = connections[off];

    std::vector<double> W1T((size_t)numHidden1 * numIn);
    for (int j = 0; j < numIn; ++j)
        for (int i = 0; i < numHidden1; ++i)
            W1T[(size_t)i * numIn + j] = W1[(size_t)j * numHidden1 + i];
    std::vector<double> W2T((size_t)numHidden2 * numHidden1);
    for (int i = 0; i < numHidden1; ++i)
        for (int j = 0; j < numHidden2; ++j)
            W2T[(size_t)j * numHidden1 + i] = W2[(size_t)i * numHidden2 + j];

    double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_W1T, *d_W2T, *d_G;
    CUDA_CHECK(cudaMalloc(&d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3, (size_t)numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W1T, (size_t)numHidden1 * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1T, W1T.data(), (size_t)numHidden1 * numIn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2T, W2T.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

    double *d_H1pre, *d_H1, *d_dfdx1, *d_H2pre, *d_H2, *d_dfdx2;
    double *d_outPre, *d_energy, *d_v2, *d_v1, *d_v1s, *d_dEdG;
    CUDA_CHECK(cudaMalloc(&d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_outPre, (size_t)numAtoms * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dEdG, (size_t)numAtoms * numIn * sizeof(double)));

    int const blk = 256;

    gemmRowMajor(handle, numAtoms, numHidden1, numIn, d_G, d_W1, d_H1pre);
    biasTanhKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        d_H1pre, d_b1, numAtoms, numHidden1, d_H1, d_dfdx1);

    gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, d_H1, d_W2, d_H2pre);
    biasTanhKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        d_H2pre, d_b2, numAtoms, numHidden2, d_H2, d_dfdx2);

    gemmRowMajor(handle, numAtoms, numOut, numHidden2, d_H2, d_W3, d_outPre);
    addOutputBiasKernel<<<(numAtoms + blk - 1) / blk, blk>>>(d_outPre, b3, numAtoms, d_energy);

    scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        d_dfdx2, d_W3, numAtoms, numHidden2, d_v2);
    gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, d_v2, d_W2T, d_v1);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        d_dfdx1, d_v1, numAtoms * numHidden1, d_v1s);
    gemmRowMajor(handle, numAtoms, numIn, numHidden1, d_v1s, d_W1T, d_dEdG);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(energyOut, d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dEdGOut, d_dEdG, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_W1T); cudaFree(d_W2T); cudaFree(d_G);
    cudaFree(d_H1pre); cudaFree(d_H1); cudaFree(d_dfdx1);
    cudaFree(d_H2pre); cudaFree(d_H2); cudaFree(d_dfdx2);
    cudaFree(d_outPre); cudaFree(d_energy);
    cudaFree(d_v2); cudaFree(d_v1); cudaFree(d_v1s); cudaFree(d_dEdG);
}

void gpuNnEnergyDEdcSum(int numAtoms, int numIn, int numHidden1, int numHidden2,
                        double const* connections,
                        double const* G, double* energyOut, double* dEdcSumOut)
{
    int const numOut = 1;
    cublasHandle_t handle = gpuHandle();

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

    std::vector<double> W2T((size_t)numHidden2 * numHidden1);
    for (int i = 0; i < numHidden1; ++i)
        for (int j = 0; j < numHidden2; ++j)
            W2T[(size_t)j * numHidden1 + i] = W2[(size_t)i * numHidden2 + j];

    double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_W2T, *d_G;
    CUDA_CHECK(cudaMalloc(&d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3, (size_t)numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2T, W2T.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

    double *d_H1pre, *d_H1, *d_dfdx1, *d_H2pre, *d_H2, *d_dfdx2;
    double *d_outPre, *d_energy, *d_v2, *d_v1, *d_v1s, *d_ones, *d_dEdcSum;
    CUDA_CHECK(cudaMalloc(&d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_outPre, (size_t)numAtoms * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ones, (size_t)numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dEdcSum, (offB3 + 1) * sizeof(double)));

    int const blk = 256;

    fillOnesKernel<<<(numAtoms + blk - 1) / blk, blk>>>(d_ones, numAtoms);

    gemmRowMajor(handle, numAtoms, numHidden1, numIn, d_G, d_W1, d_H1pre);
    biasTanhKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        d_H1pre, d_b1, numAtoms, numHidden1, d_H1, d_dfdx1);

    gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, d_H1, d_W2, d_H2pre);
    biasTanhKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        d_H2pre, d_b2, numAtoms, numHidden2, d_H2, d_dfdx2);

    gemmRowMajor(handle, numAtoms, numOut, numHidden2, d_H2, d_W3, d_outPre);
    addOutputBiasKernel<<<(numAtoms + blk - 1) / blk, blk>>>(d_outPre, b3, numAtoms, d_energy);

    // v2 = dfdx2 .* W3 (broadcast), v1s = dfdx1 .* (v2 . W2^T) -- identical
    // to gpuNnForwardDEdG()'s dEdG pipeline (both dEdG and dEdc reuse these
    // same intermediates, see gpu/gemm/nn_dedc_gemm_test.cu's header
    // comment), just consumed differently below.
    scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        d_dfdx2, d_W3, numAtoms, numHidden2, d_v2);
    gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, d_v2, d_W2T, d_v1);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        d_dfdx1, d_v1, numAtoms * numHidden1, d_v1s);

    // Sum over the atom axis, straight into the packed dEdc-sum output --
    // each of these contracts numAtoms via a GEMM instead of a per-atom
    // outer product plus a separate reduction.
    gemmRowMajorATransB(handle, numIn, numHidden1, numAtoms,
                         d_G, d_v1s, d_dEdcSum + offW1);
    gemmRowMajorATransB(handle, 1, numHidden1, numAtoms,
                         d_ones, d_v1s, d_dEdcSum + offB1);
    gemmRowMajorATransB(handle, numHidden1, numHidden2, numAtoms,
                         d_H1, d_v2, d_dEdcSum + offW2);
    gemmRowMajorATransB(handle, 1, numHidden2, numAtoms,
                         d_ones, d_v2, d_dEdcSum + offB2);
    gemmRowMajorATransB(handle, 1, numHidden2, numAtoms,
                         d_ones, d_H2, d_dEdcSum + offW3);
    // dE/db3 summed over atoms is exactly numAtoms (each atom contributes a
    // constant 1) -- set directly on the host below, no kernel needed.

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(energyOut, d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dEdcSumOut, d_dEdcSum, offB3 * sizeof(double), cudaMemcpyDeviceToHost));
    dEdcSumOut[offB3] = (double)numAtoms;

    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_W2T); cudaFree(d_G);
    cudaFree(d_H1pre); cudaFree(d_H1); cudaFree(d_dfdx1);
    cudaFree(d_H2pre); cudaFree(d_H2); cudaFree(d_dfdx2);
    cudaFree(d_outPre); cudaFree(d_energy);
    cudaFree(d_v2); cudaFree(d_v1); cudaFree(d_v1s);
    cudaFree(d_ones); cudaFree(d_dEdcSum);
}

void gpuNnForceDFdcSum(int numAtoms, int numIn, int numHidden1, int numHidden2,
                       double const* connections,
                       double const* G, double const* dGdxyz,
                       double* energyOut, double* dEdGOut, double* dFdcSumOut)
{
    int const numOut = 1;
    cublasHandle_t handle = gpuHandle();

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

    std::vector<double> W1T((size_t)numHidden1 * numIn);
    for (int j = 0; j < numIn; ++j)
        for (int i = 0; i < numHidden1; ++i)
            W1T[(size_t)i * numIn + j] = W1[(size_t)j * numHidden1 + i];
    std::vector<double> W2T((size_t)numHidden2 * numHidden1);
    for (int i = 0; i < numHidden1; ++i)
        for (int j = 0; j < numHidden2; ++j)
            W2T[(size_t)j * numHidden1 + i] = W2[(size_t)i * numHidden2 + j];

    double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_W1T, *d_W2T, *d_G, *d_dGdxyz;
    CUDA_CHECK(cudaMalloc(&d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3, (size_t)numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W1T, (size_t)numHidden1 * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdxyz, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1T, W1T.data(), (size_t)numHidden1 * numIn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2T, W2T.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dGdxyz, dGdxyz, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

    double *d_H1pre, *d_H1, *d_dfdx1, *d_d2fdx2_1;
    double *d_H2pre, *d_H2, *d_dfdx2, *d_d2fdx2_2;
    double *d_outPre, *d_energy, *d_v2, *d_v1, *d_v1s, *d_d2S1, *d_dEdG, *d_dGdxyzT;
    CUDA_CHECK(cudaMalloc(&d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d2fdx2_1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d2fdx2_2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_outPre, (size_t)numAtoms * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d2S1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dEdG, (size_t)numAtoms * numIn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdxyzT, (size_t)numIn * numAtoms * sizeof(double)));

    // Per-k0 scratch, reused every loop iteration.
    double *d_u, *d_dxdG2, *d_tmp, *d_jacBH2, *d_jacW3, *d_T;
    double *d_term1, *d_term2, *d_jacBH1, *d_P, *d_Q2, *d_Gscaled;
    CUDA_CHECK(cudaMalloc(&d_u, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dxdG2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jacBH2, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jacW3, (size_t)numAtoms * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_T, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_term1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_term2, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jacBH1, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q2, (size_t)numAtoms * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Gscaled, (size_t)numAtoms * numIn * sizeof(double)));

    double* d_dFdcSum;
    CUDA_CHECK(cudaMalloc(&d_dFdcSum, connCount * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dFdcSum, 0, connCount * sizeof(double)));

    int const blk = 256;

    // --- Precompute once, batched over atoms, no k0 dependence yet -------
    gemmRowMajor(handle, numAtoms, numHidden1, numIn, d_G, d_W1, d_H1pre);
    biasTanhD2Kernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        d_H1pre, d_b1, numAtoms, numHidden1, d_H1, d_dfdx1, d_d2fdx2_1);

    gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, d_H1, d_W2, d_H2pre);
    biasTanhD2Kernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        d_H2pre, d_b2, numAtoms, numHidden2, d_H2, d_dfdx2, d_d2fdx2_2);

    gemmRowMajor(handle, numAtoms, numOut, numHidden2, d_H2, d_W3, d_outPre);
    addOutputBiasKernel<<<(numAtoms + blk - 1) / blk, blk>>>(d_outPre, b3, numAtoms, d_energy);

    // v2/v1/v1s/dEdG: identical to gpuNnForwardDEdG()'s pipeline (this
    // function needs it->energy/it->dEdG populated correctly too, since
    // Mode::calculateForces() reads them afterward).
    scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
        d_dfdx2, d_W3, numAtoms, numHidden2, d_v2);
    gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, d_v2, d_W2T, d_v1);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        d_dfdx1, d_v1, numAtoms * numHidden1, d_v1s);
    elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
        d_d2fdx2_1, d_v1, numAtoms * numHidden1, d_d2S1);
    gemmRowMajor(handle, numAtoms, numIn, numHidden1, d_v1s, d_W1T, d_dEdG);

    transposeKernel<<<(numAtoms * numIn + blk - 1) / blk, blk>>>(
        d_dGdxyz, numAtoms, numIn, d_dGdxyzT);

    // --- Host-side loop over each input (cheap: numIn is at most a few
    // dozen for any real dataset) -- see nn_dfdc_gemm_test.cu's header
    // comment for the full derivation this mirrors, minus the per-atom
    // batched-rank-1 outer products it needed (replaced here by ordinary
    // atom-axis-contracting GEMMs, since only the atom-summed result is
    // ever needed -- see GpuNeuralNetwork.h's doc comment).
    for (int k0 = 0; k0 < numIn; ++k0)
    {
        double const* W1row = d_W1 + (size_t)k0 * numHidden1;
        double const* dGk0 = d_dGdxyzT + (size_t)k0 * numAtoms;

        scaleByRowKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            d_dfdx1, W1row, numAtoms, numHidden1, d_u);
        gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, d_u, d_W2, d_dxdG2);
        elementwiseMulKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
            d_d2fdx2_2, d_dxdG2, numAtoms * numHidden2, d_tmp);
        scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
            d_tmp, d_W3, numAtoms, numHidden2, d_jacBH2);
        elementwiseMulKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
            d_dfdx2, d_dxdG2, numAtoms * numHidden2, d_jacW3);

        gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, d_jacBH2, d_W2T, d_T);
        elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            d_dfdx1, d_T, numAtoms * numHidden1, d_term1);
        scaleByRowKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            d_d2S1, W1row, numAtoms, numHidden1, d_term2);
        elementwiseAddKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            d_term1, d_term2, numAtoms * numHidden1, d_jacBH1);

        // Sum-reductions over the atom axis, accumulating (beta=1) across
        // every k0 iteration directly into the packed dFdc-sum output.
        gemmRowMajorATransBAcc(handle, 1, numHidden2, numAtoms, -1.0,
                                dGk0, d_jacW3, 1.0, d_dFdcSum + offW3);
        gemmRowMajorATransBAcc(handle, 1, numHidden2, numAtoms, -1.0,
                                dGk0, d_jacBH2, 1.0, d_dFdcSum + offB2);
        gemmRowMajorATransBAcc(handle, 1, numHidden1, numAtoms, -1.0,
                                dGk0, d_jacBH1, 1.0, d_dFdcSum + offB1);

        scaleByColKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            d_H1, dGk0, numAtoms, numHidden1, d_P);
        scaleByColKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
            d_u, dGk0, numAtoms, numHidden1, d_Q2);
        gemmRowMajorATransBAcc(handle, numHidden1, numHidden2, numAtoms, -1.0,
                                d_P, d_jacBH2, 1.0, d_dFdcSum + offW2);
        gemmRowMajorATransBAcc(handle, numHidden1, numHidden2, numAtoms, -1.0,
                                d_Q2, d_v2, 1.0, d_dFdcSum + offW2);

        scaleByColKernel<<<(numAtoms * numIn + blk - 1) / blk, blk>>>(
            d_G, dGk0, numAtoms, numIn, d_Gscaled);
        gemmRowMajorATransBAcc(handle, numIn, numHidden1, numAtoms, -1.0,
                                d_Gscaled, d_jacBH1, 1.0, d_dFdcSum + offW1);

        // calculateDFdc's "deltaTerm": row k0 (only) of dFdc's W1 block
        // additionally gets -= sum_atom v1s[atom,:]*dGk0[atom], on top of
        // whatever the accumulation just above already wrote there.
        gemmRowMajorATransBAcc(handle, 1, numHidden1, numAtoms, -1.0,
                                dGk0, d_v1s, 1.0,
                                d_dFdcSum + offW1 + (size_t)k0 * numHidden1);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(energyOut, d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dEdGOut, d_dEdG, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dFdcSumOut, d_dFdcSum, connCount * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_W1T); cudaFree(d_W2T); cudaFree(d_G); cudaFree(d_dGdxyz);
    cudaFree(d_H1pre); cudaFree(d_H1); cudaFree(d_dfdx1); cudaFree(d_d2fdx2_1);
    cudaFree(d_H2pre); cudaFree(d_H2); cudaFree(d_dfdx2); cudaFree(d_d2fdx2_2);
    cudaFree(d_outPre); cudaFree(d_energy);
    cudaFree(d_v2); cudaFree(d_v1); cudaFree(d_v1s); cudaFree(d_d2S1);
    cudaFree(d_dEdG); cudaFree(d_dGdxyzT);
    cudaFree(d_u); cudaFree(d_dxdG2); cudaFree(d_tmp); cudaFree(d_jacBH2);
    cudaFree(d_jacW3); cudaFree(d_T); cudaFree(d_term1); cudaFree(d_term2);
    cudaFree(d_jacBH1); cudaFree(d_P); cudaFree(d_Q2); cudaFree(d_Gscaled);
    cudaFree(d_dFdcSum);
}

} // namespace nnp
