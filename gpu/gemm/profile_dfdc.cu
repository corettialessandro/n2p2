// Minimal profiling harness for gpuNnForceDFdcSum() -- NOT a correctness
// test (see libnnpgpu_dfdc_test.cu for that). Purpose: isolate exactly ONE
// representative call, after the persistent per-architecture GPU state
// (GpuNeuralNetwork.cu) is already warmed up (weight buffers allocated,
// atom-capacity grown, cuBLAS handle created, CUDA context/JIT warm), so
// that nsys/ncu see only the steady-state cost of a single call -- the
// thing that shows up as F_err in a real training run's timing.out.
//
// Bracketed with cudaProfilerStart()/Stop() so `nsys profile
// --capture-range=cudaProfilerApi` and `ncu --profile-from-start off`
// capture only the one call in between, not the warm-up calls or process
// startup/teardown.

#include "../../src/libnnpgpu/GpuNeuralNetwork.h"
#include "NeuralNetwork.h"

#include <cstdio>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using namespace nnp;
using namespace std;

int main()
{
    // H2O_2G's real "H" architecture and a representative per-candidate
    // atom count (matches libnnpgpu_dfdc_test.cu's H2O_2G-like H case).
    int const numIn = 35, numHidden1 = 25, numHidden2 = 25, numOut = 1, numLayers = 4;
    int const numAtoms = 420;

    NeuralNetwork::ActivationFunction af[4] = {
        NeuralNetwork::AF_IDENTITY, NeuralNetwork::AF_TANH,
        NeuralNetwork::AF_TANH, NeuralNetwork::AF_IDENTITY};
    int layers[4] = {numIn, numHidden1, numHidden2, numOut};

    NeuralNetwork nn(numLayers, layers, af);
    nn.initializeConnectionsRandomUniform(1);
    vector<double> conn(nn.getNumConnections());
    nn.getConnections(conn.data());

    mt19937 rng(7);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> G((size_t)numAtoms * numIn), dGdxyz((size_t)numAtoms * numIn);
    for (auto& v : G) v = dist(rng);
    for (auto& v : dGdxyz) v = dist(rng);

    vector<double> energy(numAtoms), dEdG((size_t)numAtoms * numIn);
    vector<double> dFdcSum(nn.getNumConnections());

    // Warm-up: allocate persistent state, grow atom capacity to numAtoms,
    // create/warm the cuBLAS handle, let the driver JIT-compile everything.
    for (int i = 0; i < 5; ++i)
    {
        gpuNnForceDFdcSum(numAtoms, numIn, numHidden1, numHidden2,
                          conn.data(), G.data(), dGdxyz.data(),
                          energy.data(), dEdG.data(), dFdcSum.data());
    }
    cudaDeviceSynchronize();

    printf("Warm-up done. Profiling one steady-state call...\n");
    fflush(stdout);

    cudaProfilerStart();
    gpuNnForceDFdcSum(numAtoms, numIn, numHidden1, numHidden2,
                      conn.data(), G.data(), dGdxyz.data(),
                      energy.data(), dEdG.data(), dFdcSum.data());
    cudaDeviceSynchronize();
    cudaProfilerStop();

    printf("Done.\n");
    return 0;
}
