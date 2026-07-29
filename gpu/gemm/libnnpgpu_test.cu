// Validates src/libnnpgpu/GpuNeuralNetwork.cu's gpuNnForwardDEdG() -- the
// generalized (run-time hidden-layer-size) version of
// nn_forward_gemm_test.cu's forward+dEdG pipeline that Phase 6 wires into
// Mode::calculateAtomicNeuralNetworks(). nn_forward_gemm_test.cu only ever
// exercised H2O_2G's 25/25 hidden sizes (hardcoded in that file); this test
// deliberately ALSO exercises a different architecture (30/20, matching
// examples/nnp-predict/Anisole_SCAN/input.nn's global_nodes_short) to prove
// the generalization is genuinely correct, not just re-lucky on the same
// numbers every prior test used -- the research behind this integration
// found the "2 hidden tanh(25) + linear(1)" shape is a run-time property
// read from input.nn, not a fixed constant (see
// NeuralNetwork::hasGpuCompatibleArchitecture()'s doc comment), so a GPU
// implementation that only worked for 25/25 would be a real, silent bug for
// most of this repo's other bundled example datasets.
//
// Each case now calls gpuNnForwardDEdG() repeatedly, with FRESH random
// weights and a VARYING atom count each iteration (mirroring real training:
// weights change every Kalman update, and different structures have
// different atom counts per element) -- exercising the persistent,
// per-architecture device state added after this function stopped
// cudaMalloc/cudaFree-ing everything on every call (see GpuNeuralNetwork.cu's
// header comment). A single-shot test can't catch a stale-state-reuse bug
// or a capacity-growth-logic bug; calling repeatedly with shrinking and
// growing atom counts, and independently re-checking against a fresh CPU
// reference every time, can.

#include "../../src/libnnpgpu/GpuNeuralNetwork.h"
#include "NeuralNetwork.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

using namespace nnp;
using namespace std;

static bool runCase(char const* label, vector<int> const& atomsSchedule,
                     int numIn, int numHidden1, int numHidden2, unsigned seed)
{
    printf("--- %s (numIn=%d, hidden=%d/%d, %zu calls) ---\n",
           label, numIn, numHidden1, numHidden2, atomsSchedule.size());

    int const numOut = 1, numLayers = 4;
    NeuralNetwork::ActivationFunction af[4] = {
        NeuralNetwork::AF_IDENTITY, NeuralNetwork::AF_TANH,
        NeuralNetwork::AF_TANH, NeuralNetwork::AF_IDENTITY};
    int layers[4] = {numIn, numHidden1, numHidden2, numOut};

    mt19937 rng(seed + 1000);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    double maxErrE = 0.0, maxErrDEdG = 0.0;
    for (size_t call = 0; call < atomsSchedule.size(); ++call)
    {
        int const numAtoms = atomsSchedule[call];

        // Fresh random weights every call -- real training updates them
        // after every single Kalman update, so nothing about the weight
        // buffers should be assumed to persist in VALUE, only in size.
        NeuralNetwork nn(numLayers, layers, af);
        nn.initializeConnectionsRandomUniform(seed + (unsigned)call);
        vector<double> conn(nn.getNumConnections());
        nn.getConnections(conn.data());

        if (!nn.hasGpuCompatibleArchitecture())
        {
            printf("  hasGpuCompatibleArchitecture() returned false unexpectedly\n");
            return false;
        }

        vector<double> G((size_t)numAtoms * numIn);
        for (auto& v : G) v = dist(rng);

        // --- CPU reference: real NeuralNetwork::propagate()/calculateDEdG() ---
        vector<double> energyCpu(numAtoms), dEdGCpu((size_t)numAtoms * numIn);
        for (int t = 0; t < numAtoms; ++t)
        {
            nn.setInput(&G[(size_t)t * numIn]);
            nn.propagate();
            nn.calculateDEdG(&dEdGCpu[(size_t)t * numIn]);
            nn.getOutput(&energyCpu[t]);
        }

        // --- Library function (persistent per-architecture GPU state) -----
        vector<double> energyGpu(numAtoms), dEdGGpu((size_t)numAtoms * numIn);
        gpuNnForwardDEdG(numAtoms, numIn, numHidden1, numHidden2,
                         conn.data(), G.data(), energyGpu.data(), dEdGGpu.data());

        for (int t = 0; t < numAtoms; ++t)
        {
            maxErrE = max(maxErrE, fabs(energyGpu[t] - energyCpu[t]));
            for (int k = 0; k < numIn; ++k)
            {
                size_t idx = (size_t)t * numIn + k;
                maxErrDEdG = max(maxErrDEdG, fabs(dEdGGpu[idx] - dEdGCpu[idx]));
            }
        }
        printf("  call %2zu: atoms=%4d  max|E_gpu-E_cpu|=%.3E  max|dEdG_gpu-dEdG_cpu|=%.3E\n",
               call, numAtoms, maxErrE, maxErrDEdG);
    }

    printf("  max over all calls: max|E|=%.3E  max|dEdG|=%.3E\n", maxErrE, maxErrDEdG);
    bool pass = maxErrE < 1e-9 && maxErrDEdG < 1e-9;
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main()
{
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    printf("CUDA devices visible: %d\n", devCount);
    if (devCount == 0) { fprintf(stderr, "No CUDA device.\n"); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device 0: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    bool ok = true;
    // H2O_2G's real architecture (35/42 inputs, 25/25 hidden), called
    // repeatedly with a varying (growing/shrinking) atom count, exercising
    // the persistent state's capacity growth logic.
    ok &= runCase("H2O_2G-like H", {420, 420, 100, 420, 500, 420, 420},
                  35, 25, 25, 44);
    ok &= runCase("H2O_2G-like O", {210, 50, 210, 210, 300, 210},
                  42, 25, 25, 45);
    // A genuinely different architecture (Anisole_SCAN's 30/20 hidden,
    // asymmetric hidden-layer sizes) -- proves this isn't hardcoded 25/25,
    // and that its persistent state (a separate cache entry, keyed by
    // architecture) doesn't collide with the H2O_2G-like cases above.
    ok &= runCase("Anisole_SCAN-like", {300, 300, 150, 300}, 50, 30, 20, 46);

    printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
