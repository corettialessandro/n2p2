// Performance work: replace Phase 3's one-thread-per-atom NN forward+dEdG
// kernel (../nn/nn_forward_test.cu) with cuBLAS GEMMs batched across every
// atom of one element at once -- the actual "many tiny per-atom MLPs -> a
// handful of per-element batched evaluations" opportunity
// GPU_PORTING_PLAN.md's Phase 3 section names, not yet exploited by any
// prior step (every kernel so far has been one CUDA thread doing one
// atom's tiny sequential MLP).
//
// The key structural fact this exploits: every atom of one element shares
// the SAME weights. So this isn't "batched GEMM" in the cuBLAS
// gemmStridedBatched sense (many independent small matrix pairs) -- it's a
// single ordinary GEMM per layer per element, since the weight operand is
// identical across the whole atom batch and only the data (G) operand
// varies row-by-row:
//   G (numAtoms_e x numIn)   . W1 (numIn x 25)     -> H1pre (numAtoms_e x 25)
//   H1 = tanh(H1pre + b1)                             (elementwise, broadcast bias)
//   H1 (numAtoms_e x 25)     . W2 (25 x 25)        -> H2pre (numAtoms_e x 25)
//   H2 = tanh(H2pre + b2)
//   H2 (numAtoms_e x 25)     . W3 (25 x 1)         -> energy (+ b3)
// dEdG (NeuralNetwork::calculateDEdG()'s per-input forward-sensitivity
// sweep, already ported in ../nn/nn_forward_test.cu) batches the same way,
// re-expressed as two more GEMMs plus cheap elementwise ops:
//   v2[atom,j]  = dfdx2[atom,j] * W3[j]                (elementwise, W3 broadcast)
//   v1          = v2 (numAtoms_e x 25) . W2^T (25 x 25)  -> GEMM
//   v1s[atom,i] = dfdx1[atom,i] * v1[atom,i]           (elementwise)
//   dEdG        = v1s (numAtoms_e x 25) . W1^T (25 x numIn) -> GEMM
// W1^T/W2^T are precomputed once on the host (tiny matrices, <=42x25) and
// uploaded alongside W1/W2 -- this keeps every cuBLAS call a plain
// CUBLAS_OP_N call via the standard row-major-via-column-major trick
// (see gemmRowMajor() below), rather than relying on cuBLAS transpose flags
// interacting with that trick, a well-known source of sign/axis mistakes.
//
// Ground truth: THREE independent computations of the same energy/dEdG:
//   1. cuBLAS GEMM-batched (this file's new code, what's being validated).
//   2. The already-validated one-thread-per-atom GPU kernel, copied
//      verbatim from ../nn/nn_forward_test.cu.
//   3. The real nnp::NeuralNetwork class (propagate/calculateDEdG),
//      exactly as ../nn/nn_forward_test.cu used it.
// Synthetic random G (not real symmetry-function values) -- this validates
// the NN math in isolation, same incremental philosophy ../nn/
// nn_forward_test.cu used; the real per-atom kernel and real NeuralNetwork
// class are both already proven correct against real symmetry-function
// data elsewhere (../e2e/), so re-deriving that chain here would just be
// redundant, not stronger validation.
//
// Also times both GPU paths (repeated launches, cudaEvent-based) -- at only
// 420/210 atoms (this project's real H2O_2G structure), GEMM call overhead
// (multiple cuBLAS launches, handle/stream setup) may not show a large win
// over the simple per-atom kernel; the real payoff is expected during an
// actual multi-structure training loop with larger per-element batches
// and/or when this replaces calculateDFdc/calculateDEdc's much heavier
// per-atom loops (a separate, harder batching problem -- see this file's
// closing comment). Timings are reported honestly, not used as a pass/fail
// criterion.

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

// C(m x n, row-major) = A(m x k, row-major) . B(k x n, row-major), via the
// standard trick of treating each row-major matrix as the transpose of a
// column-major one: C^T = B^T . A^T computed in column-major IS exactly
// C = A.B in row-major, with no transpose flags needed on the cuBLAS call
// itself (just swapped operand order/leading dimensions).
void gemmRowMajor(cublasHandle_t handle, int m, int n, int k,
                   const double* A, const double* B, double* C)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k, &alpha, B, n, A, k, &beta, C, n));
}

// --- Copied verbatim from ../nn/nn_forward_test.cu (already validated
// there): the one-thread-per-atom reference this GEMM path is compared
// against. ---------------------------------------------------------------

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

// --- Activation-function generalization (gpu-portability follow-up): the
// GEMM pipeline itself (gemmRowMajor calls, bias broadcast, elementwise
// products) never depended on which activation function is used -- only
// this one bias+activation kernel did, via its hardcoded tanh(). Ported
// verbatim from NeuralNetwork::propagateLayer()'s per-activation formulas
// (NeuralNetwork.cpp:785-919), including the EXP_LIMIT=35.0 overflow
// clamp on LOGISTIC/SOFTPLUS only (NeuralNetwork.cpp:25) -- matching the
// CPU reference bit-for-bit rather than adding clamps the CPU doesn't
// have. Ordinals below match NeuralNetwork::ActivationFunction's implicit
// declaration order (NeuralNetwork.h:32-55); AF_UNSET=0 is never passed.
// This is the exact code destined for src/libnnpgpu/GpuNeuralNetwork.cu
// once validated here against the real NeuralNetwork class for all 10
// activations, not just the tanh case this file originally covered. ---

__device__ __forceinline__ void activationForward(double x, int af,
                                                    double& value,
                                                    double& dfdx)
{
    switch (af)
    {
        case 1: // AF_IDENTITY
            value = x; dfdx = 1.0;
            break;
        case 2: // AF_TANH
        {
            double h = tanh(x);
            value = h; dfdx = 1.0 - h * h;
            break;
        }
        case 3: // AF_LOGISTIC
            if (x > 35.0) { value = 1.0; dfdx = 0.0; }
            else if (x < -35.0) { value = 0.0; dfdx = 0.0; }
            else
            {
                double s = 1.0 / (1.0 + exp(-x));
                value = s; dfdx = s * (1.0 - s);
            }
            break;
        case 4: // AF_SOFTPLUS
            if (x > 35.0) { value = x; dfdx = 1.0; }
            else if (x < -35.0) { value = 0.0; dfdx = 0.0; }
            else
            {
                value = log(1.0 + exp(x));
                dfdx = 1.0 / (1.0 + exp(-x));
            }
            break;
        case 5: // AF_RELU
            if (x > 0.0) { value = x; dfdx = 1.0; }
            else { value = 0.0; dfdx = 0.0; }
            break;
        case 6: // AF_GAUSSIAN
        {
            double e = exp(-0.5 * x * x);
            value = e; dfdx = -x * e;
            break;
        }
        case 7: // AF_COS
            value = cos(x); dfdx = -sin(x);
            break;
        case 8: // AF_REVLOGISTIC
        {
            double s = 1.0 / (1.0 + exp(-x));
            value = 1.0 - s; dfdx = s * (s - 1.0);
            break;
        }
        case 9: // AF_EXP
        {
            double e = exp(-x);
            value = e; dfdx = -e;
            break;
        }
        case 10: // AF_HARMONIC
            value = x * x; dfdx = 2.0 * x;
            break;
        default:
            value = x; dfdx = 1.0;
            break;
    }
}

// H = activation(pre + b), pre/H both (numAtoms x width) row-major, b
// broadcast across atoms. Also returns dfdx (needed by the dEdG GEMMs).
__global__ void biasActivationKernel(const double* pre, const double* b,
                                      int af, int numAtoms, int width,
                                      double* H, double* dfdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int col = idx % width;
    activationForward(pre[idx] + b[col], af, H[idx], dfdx[idx]);
}

__global__ void addOutputBiasKernel(const double* pre, double b3,
                                     int numAtoms, double* energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;
    energy[i] = pre[i] + b3;
}

// v2[atom,j] = dfdx2[atom,j] * W3[j] (W3 broadcast across atoms).
__global__ void scaleByW3Kernel(const double* dfdx2, const double* W3,
                                 int numAtoms, int width, double* v2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int col = idx % width;
    v2[idx] = dfdx2[idx] * W3[col];
}

// v1s[atom,i] = dfdx1[atom,i] * v1[atom,i] (both same shape, elementwise).
__global__ void elementwiseMulKernel(const double* a, const double* b,
                                      int n, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
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
        // AF_EXP has no overflow clamp on the CPU side either (unlike
        // LOGISTIC/SOFTPLUS's EXP_LIMIT) -- with [-1,1] random weights,
        // two chained unclamped exp() layers overflow toward double's
        // range limit, where GPU vs. CPU exp() diverge at the ULP level
        // and that gets amplified exponentially. No real trained network
        // would survive to such weights (the Kalman update would diverge
        // to NaN long before). Scale weights down for this one activation
        // so the test reflects a numerically realistic regime instead of
        // an untrained-random-weight overflow artifact; re-sync into `nn`
        // so the CPU reference below uses the identical scaled weights.
        if (hiddenAfOrdinal == 9) // AF_EXP
        {
            for (double& w : conn) w *= 0.05;
            nn.setConnections(conn.data());
        }

        size_t off = 0;
        double const* W1 = conn.data() + off; off += (size_t)numIn * numHidden1;
        double const* b1 = conn.data() + off; off += numHidden1;
        double const* W2 = conn.data() + off; off += (size_t)numHidden1 * numHidden2;
        double const* b2 = conn.data() + off; off += numHidden2;
        double const* W3 = conn.data() + off; off += (size_t)numHidden2 * numOut;
        double const* b3 = conn.data() + off; off += numOut;

        // Precompute transposes on the host (tiny matrices) so every GEMM
        // call below is a plain CUBLAS_OP_N (see gemmRowMajor()'s comment).
        vector<double> W1T((size_t)numHidden1 * numIn);
        for (int j = 0; j < numIn; ++j)
            for (int i = 0; i < numHidden1; ++i)
                W1T[(size_t)i * numIn + j] = W1[(size_t)j * numHidden1 + i];
        vector<double> W2T((size_t)numHidden2 * numHidden1);
        for (int i = 0; i < numHidden1; ++i)
            for (int j = 0; j < numHidden2; ++j)
                W2T[(size_t)j * numHidden1 + i] = W2[(size_t)i * numHidden2 + j];

        int begin = (int)batch.elementOffset[e];
        int end   = (int)batch.elementOffset[e + 1];
        int numAtoms = end - begin;

        mt19937 rng(seed + 1000);
        uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int t = 0; t < numAtoms; ++t)
            for (int k = 0; k < numIn; ++k)
                batch.G[batch.gIndex(begin + t, k)] = dist(rng);

        // --- CPU reference: real NeuralNetwork::propagate/calculateDEdG ---
        vector<double> energyCpu(numAtoms), dEdGCpu((size_t)numAtoms * numIn);
        for (int t = 0; t < numAtoms; ++t)
        {
            nn.setInput(&batch.G[batch.gIndex(begin + t, 0)]);
            nn.propagate();
            nn.calculateDEdG(&dEdGCpu[(size_t)t * numIn]);
            nn.getOutput(&energyCpu[t]);
        }

        // --- Device buffers, shared by both GPU paths --------------------
        double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_W1T, *d_W2T, *d_G;
        CUDA_CHECK(cudaMalloc(&d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W3, (size_t)numHidden2 * numOut * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b3, numOut * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W1T, (size_t)numHidden1 * numIn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_G, (size_t)numAtoms * numIn * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3, b3, numOut * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W1T, W1T.data(), (size_t)numHidden1 * numIn * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2T, W2T.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));

        vector<double> Ghost((size_t)numAtoms * numIn);
        for (int t = 0; t < numAtoms; ++t)
            for (int k = 0; k < numIn; ++k)
                Ghost[(size_t)t * numIn + k] = batch.G[batch.gIndex(begin + t, k)];
        CUDA_CHECK(cudaMemcpy(d_G, Ghost.data(), (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

        // --- GEMM-batched forward + dEdG ---------------------------------
        double *d_H1pre, *d_H1, *d_dfdx1, *d_H2pre, *d_H2, *d_dfdx2;
        double *d_outPre, *d_energyGemm, *d_v2, *d_v1, *d_v1s, *d_dEdGGemm;
        CUDA_CHECK(cudaMalloc(&d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_outPre, (size_t)numAtoms * numOut * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_energyGemm, numAtoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dEdGGemm, (size_t)numAtoms * numIn * sizeof(double)));

        int blk = 256;
        auto runGemmPipeline = [&]()
        {
            gemmRowMajor(handle, numAtoms, numHidden1, numIn, d_G, d_W1, d_H1pre);
            biasActivationKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                d_H1pre, d_b1, hiddenAfOrdinal, numAtoms, numHidden1, d_H1, d_dfdx1);

            gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, d_H1, d_W2, d_H2pre);
            biasActivationKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                d_H2pre, d_b2, hiddenAfOrdinal, numAtoms, numHidden2, d_H2, d_dfdx2);

            gemmRowMajor(handle, numAtoms, numOut, numHidden2, d_H2, d_W3, d_outPre);
            addOutputBiasKernel<<<(numAtoms + blk - 1) / blk, blk>>>(d_outPre, b3[0], numAtoms, d_energyGemm);

            scaleByW3Kernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                d_dfdx2, d_W3, numAtoms, numHidden2, d_v2);
            gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, d_v2, d_W2T, d_v1);
            elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                d_dfdx1, d_v1, numAtoms * numHidden1, d_v1s);
            gemmRowMajor(handle, numAtoms, numIn, numHidden1, d_v1s, d_W1T, d_dEdGGemm);
        };
        runGemmPipeline();
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> energyGemm(numAtoms), dEdGGemm((size_t)numAtoms * numIn);
        CUDA_CHECK(cudaMemcpy(energyGemm.data(), d_energyGemm, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dEdGGemm.data(), d_dEdGGemm, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyDeviceToHost));

        // --- One-thread-per-atom GPU reference (../nn/nn_forward_test.cu) --
        // Hardcodes tanh() (nnForwardAndDEdG above) -- only a valid second
        // ground truth for AF_TANH. For every other activation this project
        // now supports, the real NeuralNetwork CPU class (already compared
        // above) is the sole ground truth -- generalizing this redundant
        // per-atom kernel too would just be re-deriving the same check.
        bool const isTanh = (hiddenAfOrdinal == 2);
        double maxErrEGemmPerAtom = 0.0, maxErrDEdGGemmPerAtom = 0.0;
        float msGemm = 0.0f, msPerAtom = 0.0f;
        double *d_energyPerAtom = nullptr, *d_dEdGPerAtom = nullptr;
        std::function<void()> runPerAtomKernel;
        if (isTanh)
        {
            CUDA_CHECK(cudaMalloc(&d_energyPerAtom, numAtoms * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_dEdGPerAtom, (size_t)numAtoms * numIn * sizeof(double)));
            runPerAtomKernel = [&]()
            {
                nnForwardKernel<<<(numAtoms + blk - 1) / blk, blk>>>(
                    0, numAtoms, numIn, d_W1, d_b1, numHidden1, d_W2, d_b2, numHidden2,
                    d_W3, d_b3, numOut, d_G, 0, numIn, d_energyPerAtom, d_dEdGPerAtom);
            };
            runPerAtomKernel();
            CUDA_CHECK(cudaDeviceSynchronize());

            vector<double> energyPerAtom(numAtoms), dEdGPerAtom((size_t)numAtoms * numIn);
            CUDA_CHECK(cudaMemcpy(energyPerAtom.data(), d_energyPerAtom, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dEdGPerAtom.data(), d_dEdGPerAtom, (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyDeviceToHost));
            for (int t = 0; t < numAtoms; ++t)
            {
                maxErrEGemmPerAtom = max(maxErrEGemmPerAtom, fabs(energyGemm[t] - energyPerAtom[t]));
                for (int k = 0; k < numIn; ++k)
                {
                    size_t idx = (size_t)t * numIn + k;
                    maxErrDEdGGemmPerAtom = max(maxErrDEdGGemmPerAtom, fabs(dEdGGemm[idx] - dEdGPerAtom[idx]));
                }
            }
        }

        // --- Compare against the real NeuralNetwork CPU class --------------
        double maxErrEGemmCpu = 0.0, maxErrDEdGGemmCpu = 0.0;
        for (int t = 0; t < numAtoms; ++t)
        {
            maxErrEGemmCpu = max(maxErrEGemmCpu, fabs(energyGemm[t] - energyCpu[t]));
            for (int k = 0; k < numIn; ++k)
            {
                size_t idx = (size_t)t * numIn + k;
                maxErrDEdGGemmCpu = max(maxErrDEdGGemmCpu, fabs(dEdGGemm[idx] - dEdGCpu[idx]));
            }
        }
        printf("  atoms=%d\n", numAtoms);
        printf("  max|E_gemm-E_cpu|=%.3E max|dEdG_gemm-dEdG_cpu|=%.3E\n",
               maxErrEGemmCpu, maxErrDEdGGemmCpu);
        if (isTanh)
        {
            printf("  max|E_gemm-E_perAtom|=%.3E max|dEdG_gemm-dEdG_perAtom|=%.3E\n",
                   maxErrEGemmPerAtom, maxErrDEdGGemmPerAtom);
        }

        // --- Timing (repeated launches, steady state) -----------------------
        int const reps = 500;
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
        cudaFree(d_W3); cudaFree(d_b3); cudaFree(d_W1T); cudaFree(d_W2T); cudaFree(d_G);
        cudaFree(d_H1pre); cudaFree(d_H1); cudaFree(d_dfdx1);
        cudaFree(d_H2pre); cudaFree(d_H2); cudaFree(d_dfdx2);
        cudaFree(d_outPre); cudaFree(d_energyGemm);
        cudaFree(d_v2); cudaFree(d_v1); cudaFree(d_v1s); cudaFree(d_dEdGGemm);
        if (isTanh) { cudaFree(d_energyPerAtom); cudaFree(d_dEdGPerAtom); }

        bool pass = maxErrEGemmCpu < 1e-9 && maxErrDEdGGemmCpu < 1e-9;
        if (isTanh) pass = pass && maxErrEGemmPerAtom < 1e-9 && maxErrDEdGGemmPerAtom < 1e-9;
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
    // Softplus also gets real second-element coverage (O, numIn=42) since
    // it's the one activation with real in-repo production usage
    // (examples/nnp-train/Cu2S_PBE, QM9 both use "p p l").
    allOk &= runElement("O short-range NN [softplus]", O, 42, 45,
                         NeuralNetwork::AF_SOFTPLUS, 4);

    cublasDestroy(handle);

    printf("%s\n", allOk ? "ALL PASS" : "SOME FAILED");
    return allOk ? 0 : 1;
}
