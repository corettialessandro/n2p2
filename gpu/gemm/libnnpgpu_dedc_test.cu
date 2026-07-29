// Validates src/libnnpgpu/GpuNeuralNetwork.cu's gpuNnEnergyDEdcSum() --
// the function Training::update()'s "energy" Jacobian branch (HDNNP_2G)
// needs: per-atom energy plus the SUM over atoms of calculateDEdc()'s
// per-connection output (that branch immediately accumulates each atom's
// dEdc into the same Jacobian row, so only the sum is ever needed -- see
// Training.cpp's "if (k == \"energy\")" block).
//
// Ground truth: the real nnp::NeuralNetwork::propagate()/calculateDEdc(),
// summed by hand over all atoms. Also exercises a non-25/25 architecture
// (30/20, as libnnpgpu_test.cu does for gpuNnForwardDEdG) to confirm the
// generalization to run-time hidden-layer sizes holds here too.
//
// Each case calls gpuNnEnergyDEdcSum() repeatedly, with FRESH random
// weights and a VARYING atom count each iteration (mirroring real
// training), exercising the persistent, per-architecture device state
// added after this function stopped cudaMalloc/cudaFree-ing everything on
// every call (see GpuNeuralNetwork.cu's header comment).

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

    double maxErrE = 0.0, maxErrDEdc = 0.0;
    for (size_t call = 0; call < atomsSchedule.size(); ++call)
    {
        int const numAtoms = atomsSchedule[call];

        NeuralNetwork nn(numLayers, layers, af);
        nn.initializeConnectionsRandomUniform(seed + (unsigned)call);
        vector<double> conn(nn.getNumConnections());
        nn.getConnections(conn.data());
        size_t const connCount = (size_t)nn.getNumConnections();

        if (!nn.hasGpuCompatibleArchitecture())
        {
            printf("  hasGpuCompatibleArchitecture() returned false unexpectedly\n");
            return false;
        }

        vector<double> G((size_t)numAtoms * numIn);
        for (auto& v : G) v = dist(rng);

        // --- CPU reference: real NeuralNetwork::propagate()/calculateDEdc(),
        // summed by hand over all atoms (mirrors Training::update()'s own
        // accumulation of dXdc.at(element) across the atom loop). ------------
        vector<double> energyCpu(numAtoms);
        vector<double> dEdcSumCpu(connCount, 0.0);
        for (int t = 0; t < numAtoms; ++t)
        {
            nn.setInput(&G[(size_t)t * numIn]);
            nn.propagate();
            nn.getOutput(&energyCpu[t]);
            vector<double> dEdc(connCount);
            nn.calculateDEdc(dEdc.data());
            for (size_t j = 0; j < connCount; ++j) dEdcSumCpu[j] += dEdc[j];
        }

        // --- Library function (persistent per-architecture GPU state) -----
        vector<double> energyGpu(numAtoms);
        vector<double> dEdcSumGpu(connCount);
        gpuNnEnergyDEdcSum(numAtoms, numIn, numHidden1, numHidden2,
                           conn.data(), G.data(), energyGpu.data(), dEdcSumGpu.data());

        for (int t = 0; t < numAtoms; ++t)
            maxErrE = max(maxErrE, fabs(energyGpu[t] - energyCpu[t]));
        for (size_t j = 0; j < connCount; ++j)
            maxErrDEdc = max(maxErrDEdc, fabs(dEdcSumGpu[j] - dEdcSumCpu[j]));

        printf("  call %2zu: atoms=%4d  connCount=%zu  max|E_gpu-E_cpu|=%.3E  max|dEdcSum_gpu-dEdcSum_cpu|=%.3E\n",
               call, numAtoms, connCount, maxErrE, maxErrDEdc);
    }

    printf("  max over all calls: max|E|=%.3E  max|dEdcSum|=%.3E\n", maxErrE, maxErrDEdc);
    bool pass = maxErrE < 1e-9 && maxErrDEdc < 1e-9;
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
    // and that its persistent state doesn't collide with the H2O_2G-like
    // cases above.
    ok &= runCase("Anisole_SCAN-like", {300, 300, 150, 300}, 50, 30, 20, 46);

    printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
