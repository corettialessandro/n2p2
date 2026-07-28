// n2p2 - A neural network potential package
//
// Phase 6 (GPU_PORTING_PLAN.md): the first real call site connecting the
// GPU work validated standalone under gpu/ to the actual training/
// prediction binaries. Kept as its own library (built only when N2P2_GPU
// is defined) so the default CPU-only build is completely unaffected --
// see src/makefile.gnu's N2P2_GPU option and Mode::calculateAtomicNeuralNetworks().
//
// This header has no CUDA-specific types in it (plain doubles/ints/bool)
// so it can be included from ordinary C++ translation units (Mode.cpp)
// without needing nvcc.

#ifndef GPU_NEURAL_NETWORK_H
#define GPU_NEURAL_NETWORK_H

namespace nnp
{

/** Batched cuBLAS forward pass + calculateDEdG for every atom of ONE
 *  element at once (all such atoms share the same network connections).
 *  Ported from gpu/gemm/nn_forward_gemm_test.cu, generalized to run-time
 *  hidden layer sizes (that file hardcoded 25/25 for the H2O_2G example;
 *  real datasets configure hidden layer sizes via input.nn, see
 *  NeuralNetwork::hasGpuCompatibleArchitecture()). Callers MUST check
 *  hasGpuCompatibleArchitecture() on the network first -- this function
 *  assumes exactly two tanh hidden layers and a single-neuron identity
 *  output layer without checking again.
 *
 * @param[in]  numAtoms Number of atoms (all of the same element).
 * @param[in]  numIn Number of symmetry functions / input neurons.
 * @param[in]  numHidden1 Size of the first hidden layer.
 * @param[in]  numHidden2 Size of the second hidden layer.
 * @param[in]  connections Flat connections array, same order as
 *             NeuralNetwork::getConnections() ([W1,b1,W2,b2,W3,b3]).
 * @param[in]  G Symmetry function values, (numAtoms x numIn) row-major
 *             (i.e. atom-major, matching Atom::G laid out one atom after
 *             another).
 * @param[out] energyOut Atomic energies, length numAtoms.
 * @param[out] dEdGOut Derivative of atomic energy w.r.t. each symmetry
 *             function, (numAtoms x numIn) row-major, same layout as G.
 */
void gpuNnForwardDEdG(int numAtoms, int numIn, int numHidden1, int numHidden2,
                      double const* connections,
                      double const* G, double* energyOut, double* dEdGOut);

}

#endif
