// Performance work, part 2: batch NeuralNetwork::calculateDEdc() (the
// energy-weight Jacobian, ../e2e/'s Stage 5, ported serially as
// nnCalculateDEdc) across every atom of one element, using cuBLAS.
//
// Unlike the forward/dEdG pass (nn_forward_gemm_test.cu), calculateDEdc's
// output per atom is a full per-connection vector -- genuinely atom-
// specific in a way that isn't just "the same shared-weight GEMM applied to
// a batch of rows". But it turns out almost the entire computation is
// EITHER already produced by the forward+dEdG GEMM pipeline, OR reduces to
// a small number of per-atom RANK-1 outer products, which is exactly what
// cublasDgemmStridedBatched (m x k times k x n, batched over independent
// matrix pairs, k=1 here) is for -- this is the first genuine use of
// "batched" GEMM in the cuBLAS sense in this port (nn_forward_gemm_test.cu
// used ordinary, non-batched dgemm throughout, since every atom there
// shared the exact same weight matrix).
//
// Recall (../e2e/e2e_single_structure_test.cu's Stage 5 header comment) the
// serial per-atom algorithm:
//   dE/db3        = 1
//   dE/dW3[j]     = h2[j]
//   dE/db2[j]     = W3[j]*dfdx2[j]                     =: dEdbHidden2[j]
//   dE/dW2[i][j]  = dEdbHidden2[j] * h1[i]
//   dE/db1[i]     = dfdx1[i] * sum_j W2[i][j]*dEdbHidden2[j]  =: dEdbHidden1[i]
//   dE/dW1[jin][i]= dEdbHidden1[i] * G[jin]
// Batched across atoms (each array below now has an atom row dimension):
//   dE/dW3  = H2                              (alias -- the forward pass's
//                                               own H2 matrix, no new work)
//   dE/db2  = v2 := dfdx2 .* W3 (broadcast)    (alias -- literally the same
//                                               v2 nn_forward_gemm_test.cu
//                                               already computes for dEdG)
//   dE/db1  = v1s := dfdx1 .* (v2 . W2^T)      (alias -- also already
//                                               computed for dEdG)
//   dE/dW2[atom] = outer(h1[atom,:], v2[atom,:])     -- batched rank-1 GEMM
//   dE/dW1[atom] = outer(G[atom,:], v1s[atom,:])     -- batched rank-1 GEMM
//   dE/db3  = 1                                (constant, no computation)
// So the ENTIRE batched calculateDEdc costs exactly two
// cublasDgemmStridedBatched calls (k=1) plus a small packing kernel that
// copies the four alias arrays into the right offsets of a combined
// per-atom [W1,b1,W2,b2,W3,b3] buffer (matching AtomBatch::dFdcIndex's flat
// order, so this could feed ../e2e/'s Stage 5 directly) -- everything else
// is already sitting in device memory from the forward pass.
//
// Ground truth: the real nnp::NeuralNetwork::calculateDEdc(), and the
// already-validated serial nnCalculateDEdc (copied verbatim from
// ../e2e/e2e_single_structure_test.cu's Stage 5, itself already validated
// there to 2.3e-13 against the real class on real data).

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

// C(m x n, row-major) = A(m x k, row-major) . B(k x n, row-major) -- see
// ../gemm/nn_forward_gemm_test.cu's identical helper for the derivation.
void gemmRowMajor(cublasHandle_t handle, int m, int n, int k,
                   const double* A, const double* B, double* C)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k, &alpha, B, n, A, k, &beta, C, n));
}

// Batched rank-1 outer product: for each of `batch` independent atoms,
// C_atom(m x n, row-major) = A_atom(m x 1) . B_atom(1 x n) -- a direct
// strided-batched generalization of gemmRowMajor with k=1. strideC lets the
// caller write each atom's (m x n) block directly at an offset inside a
// wider per-atom buffer (e.g. dEdc's packed [W1,b1,W2,b2,W3,b3] layout),
// rather than into a tightly-packed standalone array.
void outerProductBatched(cublasHandle_t handle, int m, int n,
                          const double* A, int strideA,
                          const double* B, int strideB,
                          double* C, int strideC, int batch)
{
    double const alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, 1, &alpha,
        B, n, strideB,
        A, 1, strideA,
        &beta, C, n, strideC, batch));
}

// --- Copied verbatim from ../e2e/e2e_single_structure_test.cu's Stage 5
// (already validated there): the serial per-atom reference. -------------

__host__ __device__ inline void nnCalculateDEdc(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut, // numOut == 1
    double* dEdc)
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

    size_t const offW1 = 0;
    size_t const offB1 = offW1 + (size_t)numIn * numHidden1;
    size_t const offW2 = offB1 + numHidden1;
    size_t const offB2 = offW2 + (size_t)numHidden1 * numHidden2;
    size_t const offW3 = offB2 + numHidden2;
    size_t const offB3 = offW3 + (size_t)numHidden2 * numOut;

    dEdc[offB3] = 1.0;

    double dEdbHidden2[64];
    for (int j = 0; j < numHidden2; ++j)
    {
        dEdc[offW3 + j] = h2[j];
        dEdbHidden2[j] = W3[j] * dfdx2[j];
        dEdc[offB2 + j] = dEdbHidden2[j];
    }

    double dEdbHidden1[64];
    for (int i = 0; i < numHidden1; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < numHidden2; ++j)
        {
            s += W2[i * numHidden2 + j] * dEdbHidden2[j];
            dEdc[offW2 + (size_t)i * numHidden2 + j] = dEdbHidden2[j] * h1[i];
        }
        dEdbHidden1[i] = dfdx1[i] * s;
        dEdc[offB1 + i] = dEdbHidden1[i];
    }

    for (int jin = 0; jin < numIn; ++jin)
        for (int i = 0; i < numHidden1; ++i)
            dEdc[offW1 + (size_t)jin * numHidden1 + i] = dEdbHidden1[i] * G[jin];
}

__global__ void nnDEdcKernel(
    int numAtoms, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    const double* G, double* dEdc, size_t connCount)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numAtoms) return;
    nnCalculateDEdc(&G[(size_t)t * numIn], numIn, W1, b1, numHidden1,
                     W2, b2, numHidden2, W3, b3, numOut,
                     &dEdc[(size_t)t * connCount]);
}

// --- New: forward-pass elementwise kernels (same as nn_forward_gemm_test.cu)

__global__ void biasTanhKernel(const double* pre, const double* b,
                                int numAtoms, int width, double* H, double* dfdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int col = idx % width;
    double h = tanh(pre[idx] + b[col]);
    H[idx] = h;
    dfdx[idx] = 1.0 - h * h;
}

__global__ void scaleByW3Kernel(const double* dfdx2, const double* W3,
                                 int numAtoms, int width, double* v2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms * width) return;
    int col = idx % width;
    v2[idx] = dfdx2[idx] * W3[col];
}

__global__ void elementwiseMulKernel(const double* a, const double* b, int n, double* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

// Packs the four "alias" pieces (dE/db1=v1s, dE/db2=v2, dE/dW3=H2, dE/db3=1)
// into their offsets of the combined per-atom [W1,b1,W2,b2,W3,b3] buffer.
// The W1/W2 blocks are written directly by outerProductBatched() and are
// NOT touched here.
__global__ void packDEdcKernel(int numAtoms, int numIn, int numHidden1,
                                int numHidden2, const double* v1s,
                                const double* v2, const double* H2,
                                double* dEdc, size_t connCount)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numAtoms) return;
    size_t offB1 = (size_t)numIn * numHidden1;
    size_t offB2 = offB1 + numHidden1 + (size_t)numHidden1 * numHidden2;
    size_t offW3 = offB2 + numHidden2;
    size_t offB3 = offW3 + numHidden2; // numOut == 1
    double* row = &dEdc[(size_t)t * connCount];
    for (int i = 0; i < numHidden1; ++i) row[offB1 + i] = v1s[(size_t)t * numHidden1 + i];
    for (int j = 0; j < numHidden2; ++j) row[offB2 + j] = v2[(size_t)t * numHidden2 + j];
    for (int j = 0; j < numHidden2; ++j) row[offW3 + j] = H2[(size_t)t * numHidden2 + j];
    row[offB3] = 1.0;
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

    auto runElement = [&](char const* label, int e, int numIn, unsigned seed)
    {
        printf("--- %s (numIn=%d) ---\n", label, numIn);
        int const numHidden1 = 25, numHidden2 = 25, numOut = 1, numLayers = 4;
        NeuralNetwork::ActivationFunction af[4] = {
            NeuralNetwork::AF_IDENTITY, NeuralNetwork::AF_TANH,
            NeuralNetwork::AF_TANH, NeuralNetwork::AF_IDENTITY};
        int layers[4] = {numIn, numHidden1, numHidden2, numOut};
        NeuralNetwork nn(numLayers, layers, af);
        nn.initializeConnectionsRandomUniform(seed);
        vector<double> conn(nn.getNumConnections());
        nn.getConnections(conn.data());
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
        for (int t = 0; t < numAtoms; ++t)
            for (int k = 0; k < numIn; ++k)
                batch.G[batch.gIndex(begin + t, k)] = dist(rng);

        // --- CPU reference: real NeuralNetwork::calculateDEdc() -----------
        vector<vector<double>> dEdcCpu(numAtoms, vector<double>(connCount));
        for (int t = 0; t < numAtoms; ++t)
        {
            nn.setInput(&batch.G[batch.gIndex(begin + t, 0)]);
            nn.propagate();
            nn.calculateDEdc(dEdcCpu[t].data());
        }

        // --- Device setup --------------------------------------------------
        double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_W2T, *d_G;
        CUDA_CHECK(cudaMalloc(&d_W1, (size_t)numIn * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W2, (size_t)numHidden1 * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W3, (size_t)numHidden2 * numOut * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b3, numOut * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_W2T, (size_t)numHidden2 * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_G, (size_t)numAtoms * numIn * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_W1, W1, (size_t)numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2, W2, (size_t)numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3, W3, (size_t)numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3, b3, numOut * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2T, W2T.data(), (size_t)numHidden2 * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));

        vector<double> Ghost((size_t)numAtoms * numIn);
        for (int t = 0; t < numAtoms; ++t)
            for (int k = 0; k < numIn; ++k)
                Ghost[(size_t)t * numIn + k] = batch.G[batch.gIndex(begin + t, k)];
        CUDA_CHECK(cudaMemcpy(d_G, Ghost.data(), (size_t)numAtoms * numIn * sizeof(double), cudaMemcpyHostToDevice));

        double *d_H1pre, *d_H1, *d_dfdx1, *d_H2pre, *d_H2, *d_dfdx2, *d_v2, *d_v1, *d_v1s;
        CUDA_CHECK(cudaMalloc(&d_H1pre, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dfdx1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H2pre, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dfdx2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v2, (size_t)numAtoms * numHidden2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v1, (size_t)numAtoms * numHidden1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v1s, (size_t)numAtoms * numHidden1 * sizeof(double)));

        double* d_dEdcGemm;
        CUDA_CHECK(cudaMalloc(&d_dEdcGemm, (size_t)numAtoms * connCount * sizeof(double)));

        int blk = 256;
        auto runGemmPipeline = [&]()
        {
            gemmRowMajor(handle, numAtoms, numHidden1, numIn, d_G, d_W1, d_H1pre);
            biasTanhKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                d_H1pre, d_b1, numAtoms, numHidden1, d_H1, d_dfdx1);
            gemmRowMajor(handle, numAtoms, numHidden2, numHidden1, d_H1, d_W2, d_H2pre);
            biasTanhKernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                d_H2pre, d_b2, numAtoms, numHidden2, d_H2, d_dfdx2);

            scaleByW3Kernel<<<(numAtoms * numHidden2 + blk - 1) / blk, blk>>>(
                d_dfdx2, d_W3, numAtoms, numHidden2, d_v2);
            gemmRowMajor(handle, numAtoms, numHidden1, numHidden2, d_v2, d_W2T, d_v1);
            elementwiseMulKernel<<<(numAtoms * numHidden1 + blk - 1) / blk, blk>>>(
                d_dfdx1, d_v1, numAtoms * numHidden1, d_v1s);

            // dE/dW2[atom] = outer(h1[atom,:], v2[atom,:]), written directly
            // into the packed dEdc buffer's W2 sub-block.
            size_t offW2 = (size_t)numIn * numHidden1 + numHidden1;
            outerProductBatched(handle, numHidden1, numHidden2,
                                 d_H1, numHidden1, d_v2, numHidden2,
                                 d_dEdcGemm + offW2, (int)connCount, numAtoms);
            // dE/dW1[atom] = outer(G[atom,:], v1s[atom,:]).
            outerProductBatched(handle, numIn, numHidden1,
                                 d_G, numIn, d_v1s, numHidden1,
                                 d_dEdcGemm, (int)connCount, numAtoms);

            packDEdcKernel<<<(numAtoms + blk - 1) / blk, blk>>>(
                numAtoms, numIn, numHidden1, numHidden2, d_v1s, d_v2, d_H2,
                d_dEdcGemm, connCount);
        };
        runGemmPipeline();
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> dEdcGemm((size_t)numAtoms * connCount);
        CUDA_CHECK(cudaMemcpy(dEdcGemm.data(), d_dEdcGemm, (size_t)numAtoms * connCount * sizeof(double), cudaMemcpyDeviceToHost));

        // --- One-thread-per-atom GPU reference ------------------------------
        double* d_dEdcPerAtom;
        CUDA_CHECK(cudaMalloc(&d_dEdcPerAtom, (size_t)numAtoms * connCount * sizeof(double)));
        auto runPerAtomKernel = [&]()
        {
            nnDEdcKernel<<<(numAtoms + blk - 1) / blk, blk>>>(
                numAtoms, numIn, d_W1, d_b1, numHidden1, d_W2, d_b2, numHidden2,
                d_W3, d_b3, numOut, d_G, d_dEdcPerAtom, connCount);
        };
        runPerAtomKernel();
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> dEdcPerAtom((size_t)numAtoms * connCount);
        CUDA_CHECK(cudaMemcpy(dEdcPerAtom.data(), d_dEdcPerAtom, (size_t)numAtoms * connCount * sizeof(double), cudaMemcpyDeviceToHost));

        double maxErrCpu = 0.0, maxErrPerAtom = 0.0;
        for (int t = 0; t < numAtoms; ++t)
            for (size_t c = 0; c < connCount; ++c)
            {
                size_t idx = (size_t)t * connCount + c;
                maxErrCpu = max(maxErrCpu, fabs(dEdcGemm[idx] - dEdcCpu[t][c]));
                maxErrPerAtom = max(maxErrPerAtom, fabs(dEdcGemm[idx] - dEdcPerAtom[idx]));
            }
        printf("  atoms=%d  connCount=%zu\n", numAtoms, connCount);
        printf("  max|dEdc_gemm-dEdc_cpu|=%.3E max|dEdc_gemm-dEdc_perAtom|=%.3E\n",
               maxErrCpu, maxErrPerAtom);

        int const reps = 500;
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < reps; ++r) runGemmPipeline();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float msGemm = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msGemm, t0, t1));

        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < reps; ++r) runPerAtomKernel();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float msPerAtom = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msPerAtom, t0, t1));
        printf("  timing (%d reps): gemm=%.4f ms/call, per-atom=%.4f ms/call (%.2fx)\n",
               reps, msGemm / reps, msPerAtom / reps, msPerAtom / msGemm);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
        cudaFree(d_W3); cudaFree(d_b3); cudaFree(d_W2T); cudaFree(d_G);
        cudaFree(d_H1pre); cudaFree(d_H1); cudaFree(d_dfdx1);
        cudaFree(d_H2pre); cudaFree(d_H2); cudaFree(d_dfdx2);
        cudaFree(d_v2); cudaFree(d_v1); cudaFree(d_v1s);
        cudaFree(d_dEdcGemm); cudaFree(d_dEdcPerAtom);

        bool pass = maxErrCpu < 1e-9 && maxErrPerAtom < 1e-9;
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        return pass;
    };

    allOk &= runElement("H short-range NN", H, 35, 44);
    allOk &= runElement("O short-range NN", O, 42, 45);

    cublasDestroy(handle);
    printf("%s\n", allOk ? "ALL PASS" : "SOME FAILED");
    return allOk ? 0 : 1;
}
