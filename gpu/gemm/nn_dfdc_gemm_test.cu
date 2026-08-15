// Performance work, part 3: batch NeuralNetwork::calculateDFdc() (the
// force-weight Jacobian, ../nn/nn_backward_dfdc_test.cu's nnCalculateDFdc)
// across every atom of one element, using cuBLAS. The hardest of the three
// batching targets: unlike calculateDEdc (nn_dedc_gemm_test.cu, no loop
// over inputs), calculateDFdc loops over every input k0 and, for each,
// touches every connection -- so this keeps a host-side loop over k0
// (cheap: at most 42 iterations) and, on each iteration, batches the
// atom dimension via a mix of ordinary shared-weight GEMMs (same trick as
// nn_forward_gemm_test.cu, since W1/W2/W3 don't depend on the atom) and
// batched RANK-1 outer products (cublasDgemmStridedBatched, since the
// per-atom activations/sensitivities genuinely differ per atom) -- the
// latter is the same tool nn_dedc_gemm_test.cu introduced.
//
// Re-deriving the serial per-atom algorithm (nnCalculateDFdc,
// ../nn/nn_backward_dfdc_test.cu) with an atom-row dimension added
// everywhere, and factoring out what's k0-INDEPENDENT (computed once) from
// what varies per k0 (recomputed each loop iteration):
//
// Once, before the k0 loop (batched over atoms, no k0 dependence):
//   H1,dfdx1,d2fdx2_1, H2,dfdx2,d2fdx2_2         (forward pass, as usual)
//   v2   = dfdx2 .* W3        (broadcast)         =: dEdbHidden2
//   S1   = v2 . W2^T                              (GEMM, shared W2^T)
//   v1s  = dfdx1 .* S1                            =: dEdbHidden1
//   d2S1 = d2fdx2_1 .* S1
//
// Per k0 (W1row = W1's k0-th row, shared across atoms, NOT atom-specific):
//   u                = dfdx1 .* W1row                    (broadcast)
//   dxdG2            = u . W2                             (GEMM, shared W2)
//   jacBiasHidden2    = (d2fdx2_2 .* dxdG2) .* W3          (broadcast)
//   jacW3             = dfdx2 .* dxdG2
//   T                 = jacBiasHidden2 . W2^T              (GEMM, shared)
//   jacBiasHidden1    = dfdx1 .* T + d2S1 .* W1row         (broadcast)
//   dFdc_W3  -= jacW3 * dGk0                       (dGk0 = dGdxyz[:,k0])
//   dFdc_b2  -= jacBiasHidden2 * dGk0
//   dFdc_b1  -= jacBiasHidden1 * dGk0
//   dFdc_W2  -= outer(h1*dGk0, jacBiasHidden2) + outer(u*dGk0, v2)   -- two
//                                    batched rank-1 GEMMs, ACCUMULATED
//                                    (beta=1) across every k0 iteration
//   dFdc_W1[:, k0-th input row not special here; see below]
//     -= outer(G*dGk0, jacBiasHidden1)             -- batched rank-1 GEMM,
//                                                     accumulated across k0
//     row k0 ALSO gets -= v1s*dGk0 added directly (the real algorithm's
//     `deltaTerm` -- see ../nn/nn_backward_dfdc_test.cu's header comment:
//     dE/db_hidden1 contributes to dFdc's W1 ONLY at the row matching the
//     input actually being differentiated).
// dFdc_b3 is always exactly 0 (identity output layer's d2fdx2 is 0) and is
// simply never written (buffer pre-zeroed).
//
// Ground truth: the real nnp::NeuralNetwork::calculateDFdc(), and the
// already-validated serial nnCalculateDFdc (copied verbatim from
// ../nn/nn_backward_dfdc_test.cu, itself already validated there to
// ~2.5e-14 against the real class).

#include "../soa/AtomBatch.h"
#include "ElementMap.h"
#include "Structure.h"
#include "NeuralNetwork.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <functional>
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

void gemmRowMajor(cublasHandle_t handle, int m, int n, int k,
                   const double* A, const double* B, double* C)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k, &alpha, B, n, A, k, &beta, C, n));
}

// Batched rank-1 outer product with explicit alpha/beta (beta=1 accumulates
// into whatever C already holds -- used here to sum contributions across
// every k0 iteration and, within one iteration, across the two terms that
// both land in dFdc_W2). See ../gemm/nn_dedc_gemm_test.cu's identical
// (alpha=1,beta=0) helper for the row-major-via-column-major derivation.
void outerProductBatchedAcc(cublasHandle_t handle, int m, int n,
                             double alpha,
                             const double* A, int strideA,
                             const double* B, int strideB,
                             double beta,
                             double* C, int strideC, int batch)
{
    CUBLAS_CHECK(cublasDgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, 1, &alpha,
        B, n, strideB,
        A, 1, strideA,
        &beta, C, n, strideC, batch));
}

// --- Copied verbatim from ../nn/nn_backward_dfdc_test.cu (already
// validated there to ~2.5e-14): the serial per-atom reference. -----------

__host__ __device__ inline void nnCalculateDFdc(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
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

__global__ void nnDFdcKernel(
    int numAtoms, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    const double* G, const double* dGdxyz, double* dFdc, size_t connCount)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numAtoms) return;
    nnCalculateDFdc(&G[(size_t)t * numIn], numIn, W1, b1, numHidden1,
                     W2, b2, numHidden2, W3, b3, numOut,
                     &dGdxyz[(size_t)t * numIn], &dFdc[(size_t)t * connCount]);
}

// --- Elementwise kernels ---------------------------------------------------

// --- Activation-function generalization (gpu-portability follow-up): see
// nn_forward_gemm_test.cu's identical comment for the full rationale.
// This is the D2 (value+dfdx+d2fdx2) counterpart needed by the
// force-Jacobian pipeline below -- d2fdx2 is the piece the plain forward
// pass (nn_forward_gemm_test.cu) never needed. Formulas ported verbatim
// from NeuralNetwork::propagateLayer() (NeuralNetwork.cpp:785-919),
// EXP_LIMIT=35.0 clamp on LOGISTIC/SOFTPLUS only. Ordinals match
// NeuralNetwork::ActivationFunction's implicit declaration order. ---

__device__ __forceinline__ void activationForwardD2(double x, int af,
                                                      double& value,
                                                      double& dfdx,
                                                      double& d2fdx2)
{
    switch (af)
    {
        case 1: // AF_IDENTITY
            value = x; dfdx = 1.0; d2fdx2 = 0.0;
            break;
        case 2: // AF_TANH
        {
            double h = tanh(x);
            double dh = 1.0 - h * h;
            value = h; dfdx = dh; d2fdx2 = -2.0 * h * dh;
            break;
        }
        case 3: // AF_LOGISTIC
            if (x > 35.0) { value = 1.0; dfdx = 0.0; d2fdx2 = 0.0; }
            else if (x < -35.0) { value = 0.0; dfdx = 0.0; d2fdx2 = 0.0; }
            else
            {
                double s = 1.0 / (1.0 + exp(-x));
                value = s; dfdx = s * (1.0 - s);
                d2fdx2 = s * (1.0 - s) * (1.0 - 2.0 * s);
            }
            break;
        case 4: // AF_SOFTPLUS
            if (x > 35.0) { value = x; dfdx = 1.0; d2fdx2 = 0.0; }
            else if (x < -35.0) { value = 0.0; dfdx = 0.0; d2fdx2 = 0.0; }
            else
            {
                double s = 1.0 / (1.0 + exp(-x));
                value = log(1.0 + exp(x)); dfdx = s; d2fdx2 = s * (1.0 - s);
            }
            break;
        case 5: // AF_RELU
            if (x > 0.0) { value = x; dfdx = 1.0; d2fdx2 = 0.0; }
            else { value = 0.0; dfdx = 0.0; d2fdx2 = 0.0; }
            break;
        case 6: // AF_GAUSSIAN
        {
            double e = exp(-0.5 * x * x);
            value = e; dfdx = -x * e; d2fdx2 = (x * x - 1.0) * e;
            break;
        }
        case 7: // AF_COS
        {
            double c = cos(x);
            value = c; dfdx = -sin(x); d2fdx2 = -c;
            break;
        }
        case 8: // AF_REVLOGISTIC
        {
            double s = 1.0 / (1.0 + exp(-x));
            value = 1.0 - s; dfdx = s * (s - 1.0);
            d2fdx2 = s * (s - 1.0) * (1.0 - 2.0 * s);
            break;
        }
        case 9: // AF_EXP
        {
            double e = exp(-x);
            value = e; dfdx = -e; d2fdx2 = e;
            break;
        }
        case 10: // AF_HARMONIC
            value = x * x; dfdx = 2.0 * x; d2fdx2 = 2.0;
            break;
        default:
            value = x; dfdx = 1.0; d2fdx2 = 0.0;
            break;
    }
}

__global__ void biasActivationD2Kernel(const double* pre, const double* b,
                                        int af, int numAtoms, int width,
                                        double* H, double* dfdx, double* d2fdx2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int col = idx % width;
    activationForwardD2(pre[idx] + b[col], af, H[idx], dfdx[idx], d2fdx2[idx]);
}

// out[atom,col] = A[atom,col] * row[col]  (row vector broadcast across atoms)
__global__ void scaleByRowKernel(const double* A, const double* row,
                                  int numAtoms, int width, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    out[idx] = A[idx] * row[idx % width];
}

__global__ void elementwiseMulKernel(const double* a, const double* b, int n, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

__global__ void elementwiseAddKernel(const double* a, const double* b, int n, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

// out[atom,col] = A[atom,col] * dGdxyz[atom,k0]  (per-atom scalar broadcast
// across the width dimension -- the opposite broadcast direction from
// scaleByRowKernel).
__global__ void scaleByDGk0Kernel(const double* A, const double* dGdxyz,
                                   int numIn, int k0, int numAtoms, int width,
                                   double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int atom = idx / width;
    out[idx] = A[idx] * dGdxyz[(size_t)atom * numIn + k0];
}

// dst[atom*dstStride + dstOffset + col] -= A[atom,col] * dGdxyz[atom,k0],
// for col in [0,width) -- accumulates one k0 iteration's contribution
// directly into the packed [W1,b1,W2,b2,W3,b3] dFdc buffer at a fixed
// per-atom offset.
__global__ void accumulateNegScaledKernel(const double* A, const double* dGdxyz,
                                           int numIn, int k0, int numAtoms,
                                           int width, double* dst,
                                           size_t dstStride, size_t dstOffset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int atom = idx / width, col = idx % width;
    double dGk0 = dGdxyz[(size_t)atom * numIn + k0];
    dst[(size_t)atom * dstStride + dstOffset + col] -= A[idx] * dGk0;
}

// The calculateDFdc "deltaTerm": dFdc_W1's row k0 (only) additionally gets
// -= v1s[atom,:] * dGdxyz[atom,k0], on top of whatever the outer-product
// accumulation already wrote there.
__global__ void addDeltaRowKernel(const double* v1s, const double* dGdxyz,
                                   int numIn, int k0, int numAtoms,
                                   int numHidden1, double* dFdc,
                                   size_t connCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * numHidden1) return;
    int atom = idx / numHidden1, i = idx % numHidden1;
    double dGk0 = dGdxyz[(size_t)atom * numIn + k0];
    size_t off = (size_t)atom * connCount + (size_t)k0 * numHidden1 + i;
    dFdc[off] -= v1s[idx] * dGk0;
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
    double const rc = 12.0;

    ElementMap elementMap;
    elementMap.registerElements("H O");
    Structure structure;
    structure.setElementMap(elementMap);
    structure.readFromFile(inputData);
    structure.calculateNeighborList(rc);

    AtomBatch batch = buildAtomBatch(structure, rc);
    int const H = 0, O = 1;
    allocateSfStorage(batch, {35, 42});

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    bool allOk = true;

    auto runElement = [&](char const* label, int e, int numIn, unsigned seed,
                          NeuralNetwork::ActivationFunction hiddenAf,
                          int hiddenAfOrdinal)
    {
        printf("--- %s (numIn=%d) ---\n", label, numIn);
        int const numHidden1 = 25, numHidden2 = 25, numOut = 1, numLayers = 4;
        NeuralNetwork::ActivationFunction af[4] = {
            NeuralNetwork::AF_IDENTITY, hiddenAf,
            hiddenAf, NeuralNetwork::AF_IDENTITY};
        int layers[4] = {numIn, numHidden1, numHidden2, numOut};
        NeuralNetwork nn(numLayers, layers, af);
        nn.initializeConnectionsRandomUniform(seed);
        vector<double> conn(nn.getNumConnections());
        nn.getConnections(conn.data());
        // See nn_forward_gemm_test.cu's identical comment: AF_EXP is
        // unclamped on the CPU side too, and [-1,1] random weights overflow
        // it through two chained exp() layers -- scale down so the test
        // reflects a numerically realistic (trainable) regime.
        if (hiddenAfOrdinal == 9) // AF_EXP
        {
            for (double& w : conn) w *= 0.05;
            nn.setConnections(conn.data());
        }
        size_t connCount = (size_t)nn.getNumConnections();

        size_t off = 0;
        double const* W1 = conn.data() + off; off += (size_t)numIn * numHidden1;
        double const* b1 = conn.data() + off; off += numHidden1;
        double const* W2 = conn.data() + off; off += (size_t)numHidden1 * numHidden2;
        double const* b2 = conn.data() + off; off += numHidden2;
        double const* W3 = conn.data() + off; off += (size_t)numHidden2 * numOut;
        double const* b3 = conn.data() + off; off += numOut;

        vector<double> W2T((size_t)numHidden2 * numHidden1);
        for (int i = 0; i < numHidden1; ++i)
            for (int j = 0; j < numHidden2; ++j)
                W2T[(size_t)j * numHidden1 + i] = W2[(size_t)i * numHidden2 + j];

        int begin = (int)batch.elementOffset[e];
        int end   = (int)batch.elementOffset[e + 1];
        int numAtoms = end - begin;

        mt19937 rng(seed + 1000);
        uniform_real_distribution<double> dist(-1.0, 1.0);
        vector<double> Ghost((size_t)numAtoms * numIn), dGdxyzHost((size_t)numAtoms * numIn);
        for (int t = 0; t < numAtoms; ++t)
            for (int k = 0; k < numIn; ++k)
            {
                Ghost[(size_t)t * numIn + k] = dist(rng);
                dGdxyzHost[(size_t)t * numIn + k] = dist(rng);
            }

        // --- CPU reference: real NeuralNetwork::calculateDFdc() -----------
        vector<vector<double>> dFdcCpu(numAtoms, vector<double>(connCount, 0.0));
        for (int t = 0; t < numAtoms; ++t)
        {
            nn.setInput(&Ghost[(size_t)t * numIn]);
            nn.propagate();
            nn.calculateDFdc(dFdcCpu[t].data(), &dGdxyzHost[(size_t)t * numIn]);
        }

        // --- Device setup ----------------------------------------------------
        double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_W2T, *d_G, *d_dGdxyz;
        CUDA_CHECK(cudaMalloc(&d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W3, (size_t)numHidden2 * numOut * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b3, numOut * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_G, (size_t)numAtoms * numIn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dGdxyz, (size_t)numAtoms * numIn * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3, b3, numOut * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2T, W2T.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_G, Ghost.data(), (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dGdxyz, dGdxyzHost.data(), (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

        double *d_H1pre, *d_H1, *d_dfdx1, *d_d2fdx2_1;
        double *d_H2pre, *d_H2, *d_dfdx2, *d_d2fdx2_2;
        double *d_v2, *d_S1, *d_v1s, *d_d2S1;
        CUDA_CHECK(cudaMalloc(&d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_d2fdx2_1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_d2fdx2_2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_S1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_d2S1, (size_t)numAtoms * numHidden1 * sizeof(double)));

        // Per-k0 scratch (reused every iteration).
        double *d_u, *d_dxdG2, *d_tmp, *d_jacBH2, *d_jacW3, *d_T, *d_term1, *d_term2, *d_jacBH1;
        double *d_P, *d_Q2, *d_Gscaled;
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

        double* d_dFdcGemm;
        CUDA_CHECK(cudaMalloc(&d_dFdcGemm, (size_t)numAtoms * connCount * sizeof(double)));

        size_t offW1 = 0, offB1 = (size_t)numIn * numHidden1;
        size_t offW2 = offB1 + numHidden1;
        size_t offB2 = offW2 + (size_t)numHidden1 * numHidden2;
        size_t offW3 = offB2 + numHidden2;

        int blk = 256;
        auto runGemmPipeline = [&]()
        {
            CUDA_CHECK(cudaMemset(d_dFdcGemm, 0, (size_t)numAtoms * connCount * sizeof(double)));

            gemmRowMajor(handle, numAtoms, numHidden1, numIn, d_G, d_W1, d_H1pre);
            biasActivationD2Kernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                d_H1pre, d_b1, hiddenAfOrdinal, numAtoms, numHidden1, d_H1, d_dfdx1, d_d2fdx2_1);
            gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, d_H1, d_W2, d_H2pre);
            biasActivationD2Kernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                d_H2pre, d_b2, hiddenAfOrdinal, numAtoms, numHidden2, d_H2, d_dfdx2, d_d2fdx2_2);

            scaleByRowKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                d_dfdx2, d_W3, numAtoms, numHidden2, d_v2);
            gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, d_v2, d_W2T, d_S1);
            elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                d_dfdx1, d_S1, numAtoms * numHidden1, d_v1s);
            elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                d_d2fdx2_1, d_S1, numAtoms * numHidden1, d_d2S1);

            for (int k0 = 0; k0 < numIn; ++k0)
            {
                const double* W1row = d_W1 + (size_t)k0 * numHidden1;

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

                accumulateNegScaledKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                    d_jacW3, d_dGdxyz, numIn, k0, numAtoms, numHidden2, d_dFdcGemm, connCount, offW3);
                accumulateNegScaledKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                    d_jacBH2, d_dGdxyz, numIn, k0, numAtoms, numHidden2, d_dFdcGemm, connCount, offB2);
                accumulateNegScaledKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                    d_jacBH1, d_dGdxyz, numIn, k0, numAtoms, numHidden1, d_dFdcGemm, connCount, offB1);

                scaleByDGk0Kernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                    d_H1, d_dGdxyz, numIn, k0, numAtoms, numHidden1, d_P);
                scaleByDGk0Kernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                    d_u, d_dGdxyz, numIn, k0, numAtoms, numHidden1, d_Q2);
                outerProductBatchedAcc(handle, numHidden1, numHidden2, -1.0,
                                        d_P, numHidden1, d_jacBH2, numHidden2,
                                        1.0, d_dFdcGemm + offW2, (int)connCount, numAtoms);
                outerProductBatchedAcc(handle, numHidden1, numHidden2, -1.0,
                                        d_Q2, numHidden1, d_v2, numHidden2,
                                        1.0, d_dFdcGemm + offW2, (int)connCount, numAtoms);

                scaleByDGk0Kernel<<<(numAtoms * numIn + blk - 1) / blk, blk>>>(
                    d_G, d_dGdxyz, numIn, k0, numAtoms, numIn, d_Gscaled);
                outerProductBatchedAcc(handle, numIn, numHidden1, -1.0,
                                        d_Gscaled, numIn, d_jacBH1, numHidden1,
                                        1.0, d_dFdcGemm + offW1, (int)connCount, numAtoms);

                addDeltaRowKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                    d_v1s, d_dGdxyz, numIn, k0, numAtoms, numHidden1, d_dFdcGemm, connCount);
            }
        };
        runGemmPipeline();
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> dFdcGemm((size_t)numAtoms * connCount);
        CUDA_CHECK(cudaMemcpy(dFdcGemm.data(), d_dFdcGemm, (size_t)numAtoms * connCount * sizeof(double), cudaMemcpyDeviceToHost));

        // --- One-thread-per-atom GPU reference ------------------------------
        // Hardcodes tanh() (nnCalculateDFdc/nnDFdcKernel above) -- only a
        // valid second ground truth for AF_TANH, same reasoning as
        // nn_forward_gemm_test.cu. The real NeuralNetwork::calculateDFdc()
        // (already compared above) is the sole ground truth for every other
        // activation.
        bool const isTanh = (hiddenAfOrdinal == 2);
        double maxErrPerAtom = 0.0;
        float msGemm = 0.0f, msPerAtom = 0.0f;
        double* d_dFdcPerAtom = nullptr;
        std::function<void()> runPerAtomKernel;
        if (isTanh)
        {
            CUDA_CHECK(cudaMalloc(&d_dFdcPerAtom, (size_t)numAtoms * connCount * sizeof(double)));
            runPerAtomKernel = [&]()
            {
                CUDA_CHECK(cudaMemset(d_dFdcPerAtom, 0, (size_t)numAtoms * connCount * sizeof(double)));
                nnDFdcKernel<<<(numAtoms + blk - 1) / blk, blk>>>(
                    numAtoms, numIn, d_W1, d_b1, numHidden1, d_W2, d_b2, numHidden2,
                    d_W3, d_b3, numOut, d_G, d_dGdxyz, d_dFdcPerAtom, connCount);
            };
            runPerAtomKernel();
            CUDA_CHECK(cudaDeviceSynchronize());

            vector<double> dFdcPerAtom((size_t)numAtoms * connCount);
            CUDA_CHECK(cudaMemcpy(dFdcPerAtom.data(), d_dFdcPerAtom, (size_t)numAtoms * connCount * sizeof(double), cudaMemcpyDeviceToHost));
            for (int t = 0; t < numAtoms; ++t)
                for (size_t c = 0; c < connCount; ++c)
                {
                    size_t idx = (size_t)t * connCount + c;
                    maxErrPerAtom = max(maxErrPerAtom, fabs(dFdcGemm[idx] - dFdcPerAtom[idx]));
                }
        }

        double maxErrCpu = 0.0;
        for (int t = 0; t < numAtoms; ++t)
            for (size_t c = 0; c < connCount; ++c)
            {
                size_t idx = (size_t)t * connCount + c;
                maxErrCpu = max(maxErrCpu, fabs(dFdcGemm[idx] - dFdcCpu[t][c]));
            }
        printf("  atoms=%d  connCount=%zu\n", numAtoms, connCount);
        printf("  max|dFdc_gemm-dFdc_cpu|=%.3E\n", maxErrCpu);
        if (isTanh) printf("  max|dFdc_gemm-dFdc_perAtom|=%.3E\n", maxErrPerAtom);

        int const reps = 100;
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < reps; ++r) runGemmPipeline();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&msGemm, t0, t1));

        if (isTanh)
        {
            CUDA_CHECK(cudaEventRecord(t0));
            for (int r = 0; r < reps; ++r) runPerAtomKernel();
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            CUDA_CHECK(cudaEventElapsedTime(&msPerAtom, t0, t1));
            printf("  timing (%d reps): gemm=%.4f ms/call, per-atom=%.4f ms/call (%.2fx)\n",
                   reps, msGemm / reps, msPerAtom / reps, msPerAtom / msGemm);
        }
        else
        {
            printf("  timing (%d reps): gemm=%.4f ms/call\n", reps, msGemm / reps);
        }

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
        cudaFree(d_W3); cudaFree(d_b3); cudaFree(d_W2T); cudaFree(d_G); cudaFree(d_dGdxyz);
        cudaFree(d_H1pre); cudaFree(d_H1); cudaFree(d_dfdx1); cudaFree(d_d2fdx2_1);
        cudaFree(d_H2pre); cudaFree(d_H2); cudaFree(d_dfdx2); cudaFree(d_d2fdx2_2);
        cudaFree(d_v2); cudaFree(d_S1); cudaFree(d_v1s); cudaFree(d_d2S1);
        cudaFree(d_u); cudaFree(d_dxdG2); cudaFree(d_tmp); cudaFree(d_jacBH2);
        cudaFree(d_jacW3); cudaFree(d_T); cudaFree(d_term1); cudaFree(d_term2);
        cudaFree(d_jacBH1); cudaFree(d_P); cudaFree(d_Q2); cudaFree(d_Gscaled);
        cudaFree(d_dFdcGemm);
        if (isTanh) cudaFree(d_dFdcPerAtom);

        bool pass = maxErrCpu < 1e-9;
        if (isTanh) pass = pass && maxErrPerAtom < 1e-9;
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        return pass;
    };

    struct AfCase { char const* name; NeuralNetwork::ActivationFunction af; int ordinal; };
    AfCase const afCases[] = {
        {"identity",    NeuralNetwork::AF_IDENTITY,    1},
        {"tanh",        NeuralNetwork::AF_TANH,        2},
        {"logistic",    NeuralNetwork::AF_LOGISTIC,    3},
        {"softplus",    NeuralNetwork::AF_SOFTPLUS,    4},
        {"relu",        NeuralNetwork::AF_RELU,        5},
        {"gaussian",    NeuralNetwork::AF_GAUSSIAN,    6},
        {"cos",         NeuralNetwork::AF_COS,         7},
        {"revlogistic", NeuralNetwork::AF_REVLOGISTIC, 8},
        {"exp",         NeuralNetwork::AF_EXP,         9},
        {"harmonic",    NeuralNetwork::AF_HARMONIC,   10},
    };
    for (auto const& c : afCases)
    {
        char label[64];
        snprintf(label, sizeof(label), "H short-range NN [%s]", c.name);
        allOk &= runElement(label, H, 35, 44, c.af, c.ordinal);
    }
    allOk &= runElement("O short-range NN [softplus]", O, 42, 45,
                         NeuralNetwork::AF_SOFTPLUS, 4);

    cublasDestroy(handle);
    printf("%s\n", allOk ? "ALL PASS" : "SOME FAILED");
    return allOk ? 0 : 1;
}
