// Validates src/libnnpgpu/GpuNeuralNetwork.cu's gpuNnForceDFdcSum() -- the
// function Training::update()'s "force" Jacobian branch (HDNNP_2G) needs:
// per-atom energy and dEdG (so Mode::calculateForces() still gets correct
// inputs afterward), plus the SUM over atoms of calculateDFdc()'s
// per-connection output, each atom contributing via its OWN dGdxyz row
// (Training::collectDGdxia()'s per-atom result for one fixed (a,c) update
// candidate) -- see Training.cpp's "if (k == \"force\")" block.
//
// Ground truth: the real nnp::NeuralNetwork::propagate()/calculateDEdG()/
// calculateDFdc(), the latter summed by hand over all atoms. Also exercises
// a non-25/25 architecture (30/20) as every other libnnpgpu_*_test.cu does.

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

static bool runCase(char const* label, int numAtoms, int numIn,
                     int numHidden1, int numHidden2, unsigned seed)
{
    printf("--- %s (numIn=%d, hidden=%d/%d) ---\n", label, numIn, numHidden1, numHidden2);

    int const numOut = 1, numLayers = 4;
    NeuralNetwork::ActivationFunction af[4] = {
        NeuralNetwork::AF_IDENTITY, NeuralNetwork::AF_TANH,
        NeuralNetwork::AF_TANH, NeuralNetwork::AF_IDENTITY};
    int layers[4] = {numIn, numHidden1, numHidden2, numOut};
    NeuralNetwork nn(numLayers, layers, af);
    nn.initializeConnectionsRandomUniform(seed);
    vector<double> conn(nn.getNumConnections());
    nn.getConnections(conn.data());
    size_t const connCount = (size_t)nn.getNumConnections();

    if (!nn.hasGpuCompatibleArchitecture())
    {
        printf("  hasGpuCompatibleArchitecture() returned false unexpectedly\n");
        return false;
    }

    mt19937 rng(seed + 1000);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> G((size_t)numAtoms * numIn), dGdxyz((size_t)numAtoms * numIn);
    for (auto& v : G) v = dist(rng);
    for (auto& v : dGdxyz) v = dist(rng);

    // --- CPU reference: real NeuralNetwork::propagate()/calculateDEdG()/
    // calculateDFdc(), the latter summed by hand over all atoms. ----------
    vector<double> energyCpu(numAtoms), dEdGCpu((size_t)numAtoms * numIn);
    vector<double> dFdcSumCpu(connCount, 0.0);
    for (int t = 0; t < numAtoms; ++t)
    {
        nn.setInput(&G[(size_t)t * numIn]);
        nn.propagate();
        nn.calculateDEdG(&dEdGCpu[(size_t)t * numIn]);
        nn.getOutput(&energyCpu[t]);
        vector<double> dFdc(connCount, 0.0);
        nn.calculateDFdc(dFdc.data(), &dGdxyz[(size_t)t * numIn]);
        for (size_t j = 0; j < connCount; ++j) dFdcSumCpu[j] += dFdc[j];
    }

    // --- New library function -------------------------------------------
    vector<double> energyGpu(numAtoms), dEdGGpu((size_t)numAtoms * numIn);
    vector<double> dFdcSumGpu(connCount);
    gpuNnForceDFdcSum(numAtoms, numIn, numHidden1, numHidden2,
                      conn.data(), G.data(), dGdxyz.data(),
                      energyGpu.data(), dEdGGpu.data(), dFdcSumGpu.data());

    double maxErrE = 0.0, maxErrDEdG = 0.0, maxErrDFdc = 0.0;
    for (int t = 0; t < numAtoms; ++t)
    {
        maxErrE = max(maxErrE, fabs(energyGpu[t] - energyCpu[t]));
        for (int k = 0; k < numIn; ++k)
        {
            size_t idx = (size_t)t * numIn + k;
            maxErrDEdG = max(maxErrDEdG, fabs(dEdGGpu[idx] - dEdGCpu[idx]));
        }
    }
    for (size_t j = 0; j < connCount; ++j)
        maxErrDFdc = max(maxErrDFdc, fabs(dFdcSumGpu[j] - dFdcSumCpu[j]));

    printf("  atoms=%d  connCount=%zu\n", numAtoms, connCount);
    printf("  max|E_gpu-E_cpu|=%.3E  max|dEdG_gpu-dEdG_cpu|=%.3E  "
           "max|dFdcSum_gpu-dFdcSum_cpu|=%.3E\n",
           maxErrE, maxErrDEdG, maxErrDFdc);

    bool pass = maxErrE < 1e-9 && maxErrDEdG < 1e-9 && maxErrDFdc < 1e-6;
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
    // H2O_2G's real architecture (35/42 inputs, 25/25 hidden).
    ok &= runCase("H2O_2G-like H", 420, 35, 25, 25, 44);
    ok &= runCase("H2O_2G-like O", 210, 42, 25, 25, 45);
    // A genuinely different architecture (Anisole_SCAN's 30/20 hidden,
    // asymmetric hidden-layer sizes) -- proves this isn't hardcoded 25/25.
    ok &= runCase("Anisole_SCAN-like", 300, 50, 30, 20, 46);

    printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
