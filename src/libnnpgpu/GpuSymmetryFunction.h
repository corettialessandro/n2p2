// n2p2 - A neural network potential package
//
// Phase 5 (soft-percolating-jellyfish.md, LAMMPS/ML-HDNNP GPU port):
// symmetry-function evaluation on GPU. Motivated by Stopwatch profiling of
// a real LAMMPS MD run showing symmetry-function evaluation at ~95% of
// per-timestep cost (vs. ~3.3% for the already-GPU-ported NN forward pass)
// -- unlike nnp-train, where `memorize_symfunc_results` caches symmetry
// functions across epochs on a fixed dataset, LAMMPS recomputes them from
// scratch every single timestep since atoms move.
//
// Ported from gpu/soa/symfnc_exprad_group_test.cu and
// symfnc_expangn_group_test.cu (already validated there against a
// host/device-shared reference on real H2O_2G neighbor geometry, to
// ~1e-16), generalized from those files' "one element, hardcoded/parsed
// members" test-harness shape to plain flat-array parameters a caller
// builds per call, matching GpuNeuralNetwork.h's calling convention.
//
// Scope: only the two symmetry-function types actually used by this
// project's real datasets so far (SymGrpExpRad, type 2; SymGrpExpAngn,
// type 3 -- H2O_2G's input.nn uses exclusively these two, confirmed by
// grep) and only the CT_TANHU cutoff function (the only one this
// project's real datasets have used) -- the other 9 SymFnc/SymGrp type
// pairs and other cutoff types are explicitly out of scope until a real
// dataset needs them, same "port what's used" precedent as elsewhere in
// this GPU port (2G-before-4G, activation functions before depth).
// Callers MUST gate on this before dispatching -- see
// Element::hasGpuCompatibleSymmetryFunctions() (Mode.cpp's production
// call site) for the actual compatibility check.
//
// This header has no CUDA-specific types in it (plain doubles/ints) so it
// can be included from ordinary C++ translation units (Mode.cpp) without
// needing nvcc, same convention as GpuNeuralNetwork.h.

#ifndef GPU_SYMMETRY_FUNCTION_H
#define GPU_SYMMETRY_FUNCTION_H

namespace nnp
{

/** GPU-accelerated evaluation of one SymGrpExpRad-equivalent radial
 *  symmetry function group (type 2) for every atom of ONE element at
 *  once. Computes G (energy input) and both derivative sides needed for
 *  the force chain rule: the owner atom's own derivative (summed over
 *  all matching neighbors) and each individual neighbor slot's
 *  derivative (mirrors Atom::Neighbor::dGdr, one entry per slot, not
 *  summed -- what LAMMPS's InterfaceLammps::getForces() scatter-adds
 *  onto that neighbor's own force).
 *
 * Values written are UNSCALED -- i.e. this is SymFncExpRad::calculate()'s
 * raw accumulator, before its final `atom.G[index] = scale(result)` and
 * before its derivative's `scalingFactor` multiply. The caller applies
 * SymFnc::scale()/getScalingFactor() afterward using the real per-member
 * production scaling parameters (loaded from scaling.data) -- this
 * project's established practice of reusing validated production math
 * for numerically fragile logic (scaling/normalization) rather than
 * re-deriving it, same reasoning as the LAMMPS force-assembly code path
 * staying untouched/CPU-only in this same port.
 *
 * Assumes the CT_TANHU cutoff function
 * (@f$f_c(r) = \tanh^3(1 - r/r_c)@f$, no inner-cutoff/alpha dependence)
 * -- the only cutoff type this project's real datasets have configured
 * so far. Matches CutoffFunction::fdfTANHU() exactly (confirmed by
 * direct comparison against src/libnnp/CutoffFunction.cpp).
 *
 * @param[in]  numAtoms Number of atoms of this element.
 * @param[in]  neighOffset CSR offsets into the neigh* arrays, length
 *             numAtoms+1: atom t's neighbors are at flat indices
 *             [neighOffset[t], neighOffset[t+1]). Must preserve the same
 *             order as that atom's real Atom::neighbors vector, so the
 *             caller can scatter neighborDGdx/Dy/Dz results back onto
 *             the matching Atom::Neighbor::dGdr by position.
 * @param[in]  neighElement Flat neighbor element-index array, length
 *             neighOffset[numAtoms].
 * @param[in]  neighDist Flat neighbor distance array (Atom::Neighbor::d),
 *             same length.
 * @param[in]  neighDx/neighDy/neighDz Flat neighbor displacement vector
 *             arrays (Atom::Neighbor::dr), same length.
 * @param[in]  e1 Neighbor-element filter (this group's common feature,
 *             SymGrpExpRad::getEc()'s neighbor-side counterpart).
 * @param[in]  rc Cutoff radius (SymGrpBaseCutoff::getRc()).
 * @param[in]  numMembers Number of (eta,rs) pairs sharing this group's
 *             element filter and cutoff.
 * @param[in]  eta/rs Per-member Gaussian parameters, length numMembers
 *             (SymFncExpRad::getEta()/getRs() for each member).
 * @param[out] G Flat, unscaled symmetry-function output, length
 *             numAtoms*numMembers, row-major (atom-major, i.e. atom t's
 *             members are G[t*numMembers .. t*numMembers+numMembers)).
 * @param[out] dGdx/dGdy/dGdz Owner-atom derivative (summed over all
 *             matching neighbors), same shape/length as G.
 * @param[out] neighborDGdx/neighborDGdy/neighborDGdz Per-neighbor-slot
 *             derivative, length neighOffset[numAtoms]*numMembers,
 *             row-major [neighborSlot*numMembers+member]. Caller MUST
 *             zero this buffer before the call -- non-matching neighbor
 *             slots (element != e1, or d >= rc) are left untouched by
 *             design, since zero is exactly the correct derivative for a
 *             neighbor this symmetry function doesn't depend on.
 */
void gpuSfExpRadGroup(int numAtoms, int const* neighOffset,
                      int const* neighElement, double const* neighDist,
                      double const* neighDx, double const* neighDy,
                      double const* neighDz,
                      int e1, double rc, int numMembers,
                      double const* eta, double const* rs,
                      double* G, double* dGdx, double* dGdy, double* dGdz,
                      double* neighborDGdx, double* neighborDGdy,
                      double* neighborDGdz);

/** Same shape and scope as gpuSfExpRadGroup() but for one
 *  SymGrpExpAngn-equivalent narrow angular symmetry function group
 *  (type 3): per-member (e1,e2) element-pair filters -- real H2O_2G
 *  instances mix multiple element pairs within one element's group, so
 *  this is generalized beyond ExpRad's single shared filter, matching
 *  gpu/soa/symfnc_expangn_group_test.cu's real-`input.nn`-driven
 *  generalization rather than SymGrpExpAngn's own single-filter grouping
 *  assumption. `rs` is always 0 for this type's 4-parameter
 *  `symfunction_short <ec> 3 <e1> <e2> <eta> <lambda> <zeta> <rc>`
 *  input.nn form, so it isn't a parameter here.
 *
 * Neighbor-slot derivatives ACCUMULATE (+=) rather than being written
 * once, unlike the radial case -- a given neighbor slot can appear as
 * "j" in one (j,k) pair and "k" in another within the same atom's
 * angular sum (SymFncExpAngn.cpp:79-225's structure). Caller must still
 * zero the buffer first; accumulation happens only across this one
 * call's internal (j,k) loop, not across separate calls.
 *
 * @param[in]  numAtoms,neighOffset,neighElement,neighDist,
 *             neighDx,neighDy,neighDz,rc As gpuSfExpRadGroup().
 * @param[in]  numMembers Number of (e1,e2,eta,lambda,zeta) instances
 *             sharing this cutoff (all of one element's real ExpAngn
 *             instances at once -- e.g. 19 for H, 26 for O in H2O_2G).
 * @param[in]  e1/e2 Per-member element-pair filter (order-independent --
 *             a neighbor pair (j,k) matches member m if
 *             {elem(j),elem(k)} == {e1[m],e2[m]}), length numMembers.
 * @param[in]  eta/lambda/zeta Per-member parameters
 *             (SymFncBaseExpAng::getEta()/getLambda()/getZeta()), length
 *             numMembers.
 * @param[out] G,dGdx,dGdy,dGdz As gpuSfExpRadGroup(), unscaled.
 * @param[out] neighborDGdx/neighborDGdy/neighborDGdz As
 *             gpuSfExpRadGroup(), caller-zeroed, accumulated in place.
 */
void gpuSfExpAngnGroup(int numAtoms, int const* neighOffset,
                       int const* neighElement, double const* neighDist,
                       double const* neighDx, double const* neighDy,
                       double const* neighDz,
                       double rc, int numMembers,
                       int const* e1, int const* e2, double const* eta,
                       double const* lambda, double const* zeta,
                       double* G, double* dGdx, double* dGdy, double* dGdz,
                       double* neighborDGdx, double* neighborDGdy,
                       double* neighborDGdz);

}

#endif
