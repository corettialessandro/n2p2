// Phase 3 steps 1-2: NN forward pass AND backward pass (dEdG) on GPU,
// batched by element, reading from AtomBatch::G (../soa/AtomBatch.h) and
// writing into AtomBatch::energy/dEdG. Per-atom energy + dEdG in n2p2 comes
// from Mode::calculateAtomicNeuralNetworks() (src/libnnp/Mode.cpp:1666):
// for each atom, `nn.setInput(&atom.G.front()); nn.propagate();
// nn.calculateDEdG(&atom.dEdG.front()); nn.getOutput(&atom.energy);`, where
// `nn` is looked up per element -- i.e. all atoms of a given element share
// the same weights, run through the same small dense MLP. That's the
// batching opportunity Phase 3 targets (GPU_PORTING_PLAN.md Phase 3): "many
// tiny per-atom MLPs" -> a handful of per-element batched evaluations.
//
// H2O_2G's actual architecture (temp/H2O_2G/input.nn: global_hidden_layers_
// short 2, global_nodes_short 25 25, global_activation_short t t l):
// input (35 for H, 42 for O -- real symfunction_short counts in that file)
// -> 25 (tanh) -> 25 (tanh) -> 1 (identity). One thread per atom (same
// pattern as ../smoke's symmetry function kernels), looping sequentially
// over the tiny per-atom MLP -- not yet the batched-GEMM (cuBLAS
// gemmStridedBatched) version the plan eventually wants; that's a follow-up
// once this establishes correctness. Central-atom selection reuses step 2's
// trick: elementOffset[e]..[e+1] is already a contiguous, sorted range, no
// gather needed.
//
// dEdG (step 2) mirrors NeuralNetwork::calculateDEdG()'s exact algorithm
// (src/libnnp/NeuralNetwork.cpp:396): for each input k, forward-propagate a
// unit sensitivity through the layers (multiply by weights and dfdx at each
// stage) to get dE/dG_k -- NOT a from-scratch reverse-mode backprop, which
// would give the same numbers via a mathematically equivalent but
// differently-ordered computation; this ports the CPU's specific algorithm,
// same "exact port" philosophy used for the symmetry functions. Since every
// hidden layer here is tanh, dfdx = 1 - value^2 is recovered directly from
// the already-computed forward-pass activations (no separate storage of
// pre-activation x needed); the output layer is identity, so its dfdx is
// exactly 1.
//
// Ground truth: a REAL nnp::NeuralNetwork (linked from lib/libnnp.a, not a
// reimplementation) with the same architecture, random weights via its own
// initializeConnectionsRandomUniform(), propagated the standard way
// (setInput/propagate/getOutput/calculateDEdG) -- same "reuse real n2p2
// classes as the CPU reference" approach used for ElementMap/Structure in
// dump_real_neighbors.cpp. G values are synthetic random numbers (this step
// validates the forward+backward pass in isolation, decoupled from Phase
// 2's symmetry-function kernels, same incremental philosophy used
// throughout); atom counts per element (420 H, 210 O) come from the real
// H2O_2G structure via AtomBatch.

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

// Layer math mirrors NeuralNetwork::propagateLayer() exactly for this
// architecture (normalizeNeurons == false, the default and what input.nn
// leaves it at -- no "normalize_nodes" keyword there): dtmp = sum_j W[j][k]
// * prevValue[j] + bias[k], activation applied per layer. Weight layout
// matches NeuralNetwork::getConnections()'s documented order: for layer i,
// W[j*numCur+k] is the weight from previous-layer neuron j to current-layer
// neuron k (j-major, k-minor), followed by numCur biases -- i.e. exactly
// the "X (numAtoms x numPrev) times W (numPrev x numCur)" shape a future
// batched-GEMM version would want, X being the per-element G/hidden matrix.
//
// Computes both the forward pass (energy) and, immediately after (reusing
// the tanh activations for dfdx = 1 - value^2), the backward pass (dEdG),
// mirroring how Mode.cpp calls propagate() then calculateDEdG() using the
// neuron state propagate() just left behind.
__host__ __device__ inline void nnForwardAndDEdG(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut, // numOut == 1 here
    double* energyOut, double* dEdGOut /* size numIn */)
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
    for (int j = 0; j < numHidden2; ++j) out += W3[j] * h2[j]; // numOut==1
    energyOut[0] = out; // identity activation (output layer, "l"), dfdx3 = 1

    // Backward: for each input k, forward-propagate its unit sensitivity
    // through the layers -- NeuralNetwork::calculateDEdG()'s exact
    // algorithm, specialized to 2 hidden layers + linear output.
    for (int k = 0; k < numIn; ++k)
    {
        double inner0[64];
        for (int i = 0; i < numHidden1; ++i)
            inner0[i] = W1[k * numHidden1 + i] * dfdx1[i];

        double outer0[64];
        for (int i2 = 0; i2 < numHidden2; ++i2)
        {
            double s = 0.0;
            for (int i = 0; i < numHidden1; ++i)
                s += W2[i * numHidden2 + i2] * inner0[i];
            outer0[i2] = s * dfdx2[i2];
        }

        double s = 0.0;
        for (int i2 = 0; i2 < numHidden2; ++i2) s += W3[i2] * outer0[i2];
        dEdGOut[k] = s; // dfdx3 == 1 (identity)
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

// Runs one element's worth of atoms through both the real CPU NeuralNetwork
// and the GPU kernel above, using the SAME weights and G values, and
// compares energy and dEdG.
bool runElement(char const* label, AtomBatch& batch, int e, int numIn,
                unsigned int seed)
{
    printf("--- %s (numIn=%d) ---\n", label, numIn);

    int const numHidden1 = 25, numHidden2 = 25, numOut = 1;
    int numLayers = 4;
    int numNeuronsPerLayer[4] = {numIn, numHidden1, numHidden2, numOut};
    NeuralNetwork::ActivationFunction af[4] = {
        NeuralNetwork::AF_IDENTITY, // input layer, unused but mandatory
        NeuralNetwork::AF_TANH, NeuralNetwork::AF_TANH,
        NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(seed);

    vector<double> conn(nn.getNumConnections());
    nn.getConnections(conn.data());

    // Slice the flat connections array into per-layer W/b, matching
    // getConnections()'s documented per-layer order.
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

    // Synthetic G values for this element's block (random, in [-1, 1] like
    // typical scaled/centered symmetry function output).
    mt19937 rng(seed + 1000);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int t = 0; t < numSelected; ++t)
        for (int k = 0; k < numIn; ++k)
            batch.G[batch.gIndex(begin + t, k)] = dist(rng);

    // --- CPU reference: real NeuralNetwork::propagate()/calculateDEdG() ---
    vector<double> energyCpu(numSelected);
    vector<vector<double>> dEdGCpu(numSelected, vector<double>(numIn));
    for (int t = 0; t < numSelected; ++t)
    {
        int s = begin + t;
        nn.setInput(&batch.G[batch.gIndex(s, 0)]);
        nn.propagate();
        nn.calculateDEdG(dEdGCpu[t].data());
        nn.getOutput(&energyCpu[t]);
    }

    // --- GPU: one thread per atom, reading straight from AtomBatch::G -----
    double *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_G, *d_energy, *d_dEdG;
    CUDA_CHECK(cudaMalloc(&d_W1, numIn * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2, numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3, numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b3, numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, batch.numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dEdG, batch.dEdG.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_W1, W1, numIn * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, W3, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, b3, numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, batch.G.data(), batch.G.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_energy, 0, batch.numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dEdG, 0, batch.dEdG.size() * sizeof(double)));

    int blockSize = 128;
    int gridSize = (numSelected + blockSize - 1) / blockSize;
    nnForwardKernel<<<gridSize, blockSize>>>(begin, numSelected, numIn,
        d_W1, d_b1, numHidden1, d_W2, d_b2, numHidden2, d_W3, d_b3, numOut,
        d_G, batch.gBlockOffset[e], batch.sfCountPerElement[e],
        d_energy, d_dEdG);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(batch.energy.data(), d_energy, batch.numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(batch.dEdG.data(), d_dEdG, batch.dEdG.size() * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_b3); cudaFree(d_G); cudaFree(d_energy);
    cudaFree(d_dEdG);

    double maxAbsErrE = 0.0, maxAbsErrDEdG = 0.0;
    for (int t = 0; t < numSelected; ++t)
    {
        maxAbsErrE = max(maxAbsErrE, fabs(batch.energy[begin + t] - energyCpu[t]));
        for (int k = 0; k < numIn; ++k)
            maxAbsErrDEdG = max(maxAbsErrDEdG,
                fabs(batch.dEdG[batch.gIndex(begin + t, k)] - dEdGCpu[t][k]));
    }

    printf("  selected atoms=%d  energy[0]: cpu=%.15E gpu=%.15E\n",
           numSelected, energyCpu[0], batch.energy[begin]);
    printf("  max|E_gpu-E_cpu|=%.3E  max|dEdG_gpu-dEdG_cpu|=%.3E\n",
           maxAbsErrE, maxAbsErrDEdG);

    bool pass = (maxAbsErrE < 1e-9) && (maxAbsErrDEdG < 1e-9);
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

    // Real per-element symmetry-function counts from temp/H2O_2G/input.nn
    // (35 "symfunction_short H ..." lines, 42 "... O ..." lines).
    allocateSfStorage(batch, {35, 42});
    printf("AtomBatch: %zu atoms (%zu H, %zu O)\n\n", batch.numAtoms,
           batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.elementOffset[O + 1] - batch.elementOffset[O]);

    bool ok = true;
    ok &= runElement("H short-range NN", batch, H, 35, 42);
    ok &= runElement("O short-range NN", batch, O, 42, 43);

    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
