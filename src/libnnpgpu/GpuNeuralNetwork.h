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
 *  assumes exactly two hidden layers and a single-neuron identity output
 *  layer without checking again (any of NeuralNetwork::ActivationFunction's
 *  10 activations is supported on the two hidden layers, see activation1/
 *  activation2 below; depth beyond two hidden layers is not supported).
 *
 * @param[in]  numAtoms Number of atoms (all of the same element).
 * @param[in]  numIn Number of symmetry functions / input neurons.
 * @param[in]  numHidden1 Size of the first hidden layer.
 * @param[in]  numHidden2 Size of the second hidden layer.
 * @param[in]  activation1 First hidden layer's activation function --
 *             MUST match NeuralNetwork::ActivationFunction's enum ordinal
 *             (implicit declaration order, NeuralNetwork.h: AF_IDENTITY=1,
 *             AF_TANH=2, AF_LOGISTIC=3, AF_SOFTPLUS=4, AF_RELU=5,
 *             AF_GAUSSIAN=6, AF_COS=7, AF_REVLOGISTIC=8, AF_EXP=9,
 *             AF_HARMONIC=10; AF_UNSET=0 is never valid here). This header
 *             deliberately doesn't include NeuralNetwork.h (see file
 *             header), so callers pass e.g.
 *             (int)nn.getActivationFunctionOfLayer(1).
 * @param[in]  activation2 Second hidden layer's activation function, same
 *             ordinal convention as activation1 -- independent of it (a
 *             network may mix activations across its two hidden layers).
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
                      int activation1, int activation2,
                      double const* connections,
                      double const* G, double* energyOut, double* dEdGOut);

/** Batched cuBLAS forward pass + calculateDEdc for every atom of ONE
 *  element at once, SUMMED across atoms (the energy-weight Jacobian
 *  contribution Training::update()'s "energy" branch needs: one Jacobian
 *  row per structure, accumulated atom-by-atom on the CPU side via
 *  NeuralNetwork::calculateDEdc()). Ported from
 *  gpu/gemm/nn_dedc_gemm_test.cu's per-atom batching, but the atom-axis
 *  reduction is folded directly into two extra GEMMs (contracting the atom
 *  dimension) instead of materializing a per-atom dEdc array and summing on
 *  the host -- cheaper here since the caller only ever wants the sum.
 *  Same architecture restriction as gpuNnForwardDEdG(): callers MUST check
 *  hasGpuCompatibleArchitecture() first.
 *
 * @param[in]  numAtoms Number of atoms (all of the same element).
 * @param[in]  numIn Number of symmetry functions / input neurons.
 * @param[in]  numHidden1 Size of the first hidden layer.
 * @param[in]  numHidden2 Size of the second hidden layer.
 * @param[in]  activation1 First hidden layer's activation function -- see
 *             gpuNnForwardDEdG()'s doc comment for the ordinal convention.
 * @param[in]  activation2 Second hidden layer's activation function, same
 *             convention as activation1.
 * @param[in]  connections Flat connections array, same order as
 *             NeuralNetwork::getConnections() ([W1,b1,W2,b2,W3,b3]).
 * @param[in]  G Symmetry function values, (numAtoms x numIn) row-major.
 * @param[out] energyOut Atomic energies, length numAtoms.
 * @param[out] dEdcSumOut Sum over all numAtoms atoms of calculateDEdc()'s
 *             per-atom output, same flat [W1,b1,W2,b2,W3,b3] layout and
 *             length as NeuralNetwork::getNumConnections().
 */
void gpuNnEnergyDEdcSum(int numAtoms, int numIn, int numHidden1, int numHidden2,
                        int activation1, int activation2,
                        double const* connections,
                        double const* G, double* energyOut, double* dEdcSumOut);

/** Batched cuBLAS forward pass + calculateDEdG + calculateDFdc for every
 *  atom of ONE element at once, with the force-weight Jacobian SUMMED
 *  across atoms -- the contribution Training::update()'s "force" branch
 *  needs: for one fixed (atom, coordinate) update candidate, every atom in
 *  the structure contributes calculateDFdc(dGdxyz_atom) to the SAME
 *  Jacobian row, accumulated on the CPU side today via a per-atom loop.
 *  Ported from gpu/gemm/nn_dfdc_gemm_test.cu's per-atom batching (itself
 *  the hardest of the three Jacobian passes -- a host-side loop over each
 *  of the numIn inputs, since calculateDFdc's inner algorithm touches every
 *  connection once per input). Unlike that file, this function only ever
 *  needs the atom-axis SUM (same insight as gpuNnEnergyDEdcSum()), which
 *  turns every one of nn_dfdc_gemm_test.cu's batched RANK-1
 *  cublasDgemmStridedBatched outer products into a single ordinary
 *  cublasDgemm contracting the atom dimension directly -- simpler *and*
 *  cheaper than the per-atom-preserving version, not just cheaper to
 *  transfer off the device.
 *  Same architecture restriction as gpuNnForwardDEdG(): callers MUST check
 *  hasGpuCompatibleArchitecture() first.
 *
 * @param[in]  numAtoms Number of atoms (all of the same element).
 * @param[in]  numIn Number of symmetry functions / input neurons.
 * @param[in]  numHidden1 Size of the first hidden layer.
 * @param[in]  numHidden2 Size of the second hidden layer.
 * @param[in]  activation1 First hidden layer's activation function -- see
 *             gpuNnForwardDEdG()'s doc comment for the ordinal convention.
 * @param[in]  activation2 Second hidden layer's activation function, same
 *             convention as activation1.
 * @param[in]  connections Flat connections array, same order as
 *             NeuralNetwork::getConnections() ([W1,b1,W2,b2,W3,b3]).
 * @param[in]  G Symmetry function values, (numAtoms x numIn) row-major.
 * @param[in]  dGdxyz Derivative of each atom's own symmetry functions with
 *             respect to the ONE external coordinate degree of freedom
 *             this call's Jacobian row is for (i.e. what
 *             Training::collectDGdxia() computes per atom), (numAtoms x
 *             numIn) row-major, same layout as G.
 * @param[out] energyOut Atomic energies, length numAtoms.
 * @param[out] dEdGOut Derivative of atomic energy w.r.t. each symmetry
 *             function, (numAtoms x numIn) row-major, same layout as G.
 * @param[out] dFdcSumOut Sum over all numAtoms atoms of calculateDFdc()'s
 *             per-atom output (each atom's own dGdxyz row used for its own
 *             contribution), same flat [W1,b1,W2,b2,W3,b3] layout and
 *             length as NeuralNetwork::getNumConnections().
 */
void gpuNnForceDFdcSum(int numAtoms, int numIn, int numHidden1, int numHidden2,
                       int activation1, int activation2,
                       double const* connections,
                       double const* G, double const* dGdxyz,
                       double* energyOut, double* dEdGOut, double* dFdcSumOut);

}

#endif
