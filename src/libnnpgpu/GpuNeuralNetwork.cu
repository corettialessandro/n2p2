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

// Row-major C(m x n) = A^T(m x k) . B(k x n), where A is stored row-major
// (k x m) (so A^T is the (m x k) matrix actually multiplied) and B is
// row-major (k x n) -- i.e. the same row-major-via-column-major trick as
// gemmRowMajor() above, but contracting over A's FIRST (row) axis instead
// of its second. Used to reduce/sum over the atom (batch) axis directly via
// GEMM (e.g. dE/dW1 summed over atoms is G^T . v1s, with the atom axis as
// the contracted dimension k) instead of a separate reduction kernel plus
// host-side summation.
void gemmRowMajorATransB(cublasHandle_t handle, int m, int n, int k,
                          double const* A, double const* B, double* C)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             n, m, k, &alpha, B, n, A, m, &beta, C, n));
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

} // namespace nnp
