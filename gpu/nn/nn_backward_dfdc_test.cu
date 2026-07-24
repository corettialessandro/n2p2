// Phase 3 step 3: the weight-Jacobian pass, calculateDFdc/calculateD2EdGdc,
// on GPU. This is the OTHER half of the ~30.8%-of-wall-time NN backward
// cost §3a's profiling identified (dEdG, step 2, is the first half) -- it's
// what actually gets consumed by Training::update()'s Jacobian assembly for
// the weight updater (Kalman filter / gradient descent both fit forces,
// hence need dF/dc, not just dE/dc).
//
// NeuralNetwork::calculateDFdc() (src/libnnp/NeuralNetwork.cpp:490) computes
//   dFdc[c] = -sum_k (d^2E / dc dG_k) * dGdxyz[k]
// where dGdxyz[k] = dG_k/dx_{l,gamma} for some atom l and coordinate
// component gamma (Phase 1's AtomBatch::dGdx/neighborDGdx would supply this
// in a full integration -- out of scope here, this step validates the NN
// math in isolation with synthetic dGdxyz, same incremental philosophy as
// nn_forward_test.cu's synthetic G). It gets there via calculateD2EdGdc(),
// called once per input k -- ported here as one specialized, unrolled
// device function for our fixed architecture (2 hidden tanh(25) layers +
// linear output(1)), rather than a generic per-layer loop, since porting
// the *exact* CPU algorithm (not a from-scratch re-derivation) is this
// project's established approach.
//
// Derivation actually used below (re-derived by hand from
// NeuralNetwork.cpp:444-719, specialized to numLayers=4, numOut=1):
//   Forward:   h1[i]=tanh(x1_i), dfdx1=1-h1^2, d2fdx2_1=-2 h1 dfdx1  (i=0..24)
//              h2[j]=tanh(x2_j), dfdx2=1-h2^2, d2fdx2_2=-2 h2 dfdx2  (j=0..24)
//              output layer is identity: dfdx3=1, d2fdx2_3=0 always.
//   calculateDEdb (bias sensitivities, computed ONCE per atom, independent
//   of which input k we differentiate against -- exactly mirroring how the
//   real code calls it once, outside computeDFdc's loop over inputs):
//     dEdb_output = dfdx3 = 1                     (single output neuron)
//     dEdb_hidden2[j] = dEdb_output * W3[j] * dfdx2[j] = W3[j] * dfdx2[j]
//       (the dfdx2[j] factor here is layers[i].neurons[j].dfdx in
//       calculateDEdb's recursion -- the *destination* layer's own dfdx, not
//       the source's -- easy to drop by mistake; a first pass at this file
//       did exactly that and every dFdc entry involving W2/b1/W1 came out
//       wrong by an O(1) amount as a result, caught by comparing against
//       the real NeuralNetwork::calculateDFdc())
//     S1[i] = sum_j W2[i][j] * dEdb_hidden2[j]
//     dEdb_hidden1[i] = dfdx1[i] * S1[i]
//   calculateDxdG(k) + calculateD2EdGdc(k, ...) fused, per input k:
//     dxdG1[i] = W1[k][i]
//     dxdG2[j] = sum_i W2[i][j] * dfdx1[i] * dxdG1[i]
//     jacBiasHidden2[j] = W3[j] * d2fdx2_2[j] * dxdG2[j]   -> d2EdGdc for b2[j]
//       (uses dEdb_output==1 directly, not dEdb_hidden2 -- a different
//       term in calculateD2EdGdc's i==2 branch than the one above)
//     jacW3[j]          = dfdx2[j] * dxdG2[j]              -> d2EdGdc for W3[j]
//     T[i] = sum_j W2[i][j] * jacBiasHidden2[j]
//     jacBiasHidden1[i] = dfdx1[i]*T[i] + d2fdx2_1[i]*dxdG1[i]*S1[i]  -> d2EdGdc for b1[i]
//     jacW2[i][j] = jacBiasHidden2[j]*h1[i] + dEdb_hidden2[j]*dfdx1[i]*dxdG1[i] -> d2EdGdc for W2[i][j]
//     jacW1[jin][i] = jacBiasHidden1[i]*G[jin] + (jin==k ? dEdb_hidden1[i] : 0)
//                                                          -> d2EdGdc for W1[jin][i]
//                     (touches ALL jin, not just k -- the real code's i==0
//                     branch loops over every input-layer neuron)
//     d2EdGdc for b3 (output bias) is always exactly 0, since d2fdx2 of an
//     identity activation is 0 -- a real, not-a-bug consequence: dE/db3 is
//     the constant 1 regardless of any input, so its derivative wrt any G
//     is trivially 0. dFdc's b3 entry is therefore just left at 0.
//   dFdc[c] -= (that connection's d2EdGdc value) * dGdxyz[k], accumulated
//   over all k.

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

// See the file header for the derivation. dFdc must be pre-zeroed by the
// caller (accumulated via -=). Offsets within dFdc match
// NeuralNetwork::getConnections()'s flat order: [W1, b1, W2, b2, W3, b3].
__host__ __device__ inline void nnCalculateDFdc(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut, // numOut == 1
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

    // dEdb_hidden2[j] = dEdb_output * W3[j] * dfdx2[j], dEdb_output == 1
    // (identity output layer) -- the dfdx2[j] factor here is
    // layers[i].neurons[j].dfdx in calculateDEdb's recursion (i.e. the
    // *destination* layer's own dfdx, not the source's); easy to drop by
    // mistake, and doing so is exactly the bug this comment now guards
    // against having reintroduced.
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
    // offB3 (output bias) always accumulates zero -- see file header.

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
    int begin, int numSelected, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    const double* G, size_t gBlockOffsetE, size_t sfCountE,
    const double* dGdxyz,
    double* dFdc, size_t dFdcBlockOffsetE, size_t connCountE)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    size_t gBase = gBlockOffsetE + (size_t)t * sfCountE;
    size_t dFdcBase = dFdcBlockOffsetE + (size_t)t * connCountE;
    nnCalculateDFdc(&G[gBase], numIn, W1, b1, numHidden1, W2, b2, numHidden2,
                     W3, b3, numOut, &dGdxyz[gBase], &dFdc[dFdcBase]);
}

bool runElement(char const* label, AtomBatch& batch, int e, int numIn,
                unsigned int seed)
{
    printf("--- %s (numIn=%d) ---\n", label, numIn);

    int const numHidden1 = 25, numHidden2 = 25, numOut = 1;
    int numLayers = 4;
    int numNeuronsPerLayer[4] = {numIn, numHidden1, numHidden2, numOut};
    NeuralNetwork::ActivationFunction af[4] = {
        NeuralNetwork::AF_IDENTITY, NeuralNetwork::AF_TANH,
        NeuralNetwork::AF_TANH, NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(seed);

    int const numConnections = nn.getNumConnections();
    vector<double> conn(numConnections);
    nn.getConnections(conn.data());

    size_t off = 0;
    double const* W1 = conn.data() + off; off += (size_t)numIn * numHidden1;
    double const* b1 = conn.data() + off; off += numHidden1;
    double const* W2 = conn.data() + off; off += (size_t)numHidden1 * numHidden2;
    double const* b2 = conn.data() + off; off += numHidden2;
    double const* W3 = conn.data() + off; off += (size_t)numHidden2 * numOut;
    double const* b3 = conn.data() + off; off += numOut;

    int begin = (int)batch.elementOffset[e];
    int end   = (int)batch.elementOffset[e + 1];
    int numSelected = end - begin;

    mt19937 rng(seed + 1000);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int t = 0; t < numSelected; ++t)
        for (int k = 0; k < numIn; ++k)
            batch.G[batch.gIndex(begin + t, k)] = dist(rng);

    // Synthetic per-atom dGdxyz (stand-in for AtomBatch::dGdx/neighborDGdx
    // in a full integration), same block layout as G.
    vector<double> dGdxyz(batch.G.size());
    for (int t = 0; t < numSelected; ++t)
        for (int k = 0; k < numIn; ++k)
            dGdxyz[batch.gIndex(begin + t, k)] = dist(rng);

    // --- CPU reference: real NeuralNetwork::propagate()/calculateDFdc() ---
    vector<vector<double>> dFdcCpu(numSelected, vector<double>(numConnections));
    for (int t = 0; t < numSelected; ++t)
    {
        int s = begin + t;
        nn.setInput(&batch.G[batch.gIndex(s, 0)]);
        nn.propagate();
        nn.calculateDFdc(dFdcCpu[t].data(), &dGdxyz[batch.gIndex(s, 0)]);
    }

    // --- GPU ----------------------------------------------------------------
    double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_G, *d_dGdxyz, *d_dFdc;
    CUDA_CHECK(cudaMalloc(&d_W1, numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3, numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b3, numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdxyz, dGdxyz.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dFdc, batch.dFdc.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_W1, W1, numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, W3, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, b3, numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, batch.G.data(), batch.G.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dGdxyz, dGdxyz.data(), dGdxyz.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dFdc, 0, batch.dFdc.size() * sizeof(double)));

    int blockSize = 128;
    int gridSize = (numSelected + blockSize - 1) / blockSize;
    nnDFdcKernel<<<gridSize, blockSize>>>(begin, numSelected, numIn,
        d_W1, d_b1, numHidden1, d_W2, d_b2, numHidden2, d_W3, d_b3, numOut,
        d_G, batch.gBlockOffset[e], batch.sfCountPerElement[e], d_dGdxyz,
        d_dFdc, batch.dFdcBlockOffset[e], batch.connCountPerElement[e]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(batch.dFdc.data(), d_dFdc, batch.dFdc.size() * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_b3); cudaFree(d_G); cudaFree(d_dGdxyz);
    cudaFree(d_dFdc);

    double maxAbsErr = 0.0;
    for (int t = 0; t < numSelected; ++t)
        for (int c = 0; c < numConnections; ++c)
            maxAbsErr = max(maxAbsErr,
                fabs(batch.dFdc[batch.dFdcIndex(begin + t, c)] - dFdcCpu[t][c]));

    printf("  selected atoms=%d  numConnections=%d\n", numSelected, numConnections);
    printf("  max|dFdc_gpu-dFdc_cpu|=%.3E\n", maxAbsErr);

    bool pass = maxAbsErr < 1e-9;
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
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
    // numConnections = numIn*25+25 (W1,b1) + 25*25+25 (W2,b2) + 25*1+1 (W3,b3)
    size_t connH = 35 * 25 + 25 + 25 * 25 + 25 + 25 * 1 + 1;
    size_t connO = 42 * 25 + 25 + 25 * 25 + 25 + 25 * 1 + 1;
    allocateWeightJacobianStorage(batch, {connH, connO});
    printf("AtomBatch: %zu atoms (%zu H, %zu O), dFdc storage: %zu values\n\n",
           batch.numAtoms, batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.elementOffset[O + 1] - batch.elementOffset[O], batch.dFdc.size());

    bool ok = true;
    ok &= runElement("H short-range NN", batch, H, 35, 44);
    ok &= runElement("O short-range NN", batch, O, 42, 45);

    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
