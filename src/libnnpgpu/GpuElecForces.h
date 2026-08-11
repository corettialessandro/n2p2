// n2p2 - A neural network potential package
//
// GPU-accelerated HDNNP_4G electrostatics-force block of
// Mode::calculateForces() -- see Mode.cpp's call site, right after the
// existing HDNNP_2G-only GpuForces.h dispatch (which this is deliberately
// independent of and orthogonal to, same as GpuForces.h is to
// GpuNeuralNetwork.h).
//
// Fine-grained profiling (temporary Stopwatch instrumentation, see
// gpu/README.md's 4G electrostatics section) found that within this
// block, Structure::calculateForceLambdaTotal()/Elec() (two dense
// (numAtoms+1)x(numAtoms+1) solves against an already-cached
// factorization, see Structure::AConstrainedQr) cost under 1ms/call
// combined -- the earlier factorize-once CPU fix already took care of
// the dense linear algebra here. The real cost (~148ms/call, a 346x
// gap on a real production run: calculateDQdr() 0.05s/epoch vs the
// whole calculateForces() call 17.3s/epoch) is the O(numAtoms^2 x
// avgNeighbors) double loop that calls Atom::calculateDChidr() for
// every atom pair -- which internally does a linear neighbor-list
// search per pair. This header ports THAT loop, not the solves (those
// stay on the CPU, exactly as validated correct and cheap already).
//
// Math: lambdaTotal(j)/lambdaElec(j) are constant per OWNER atom j, so
// they can be pre-multiplied into dChidG_j[k] once per call
// (weightKernel below). That turns the whole per-pair computation into
// exactly the same self-term + owner-centric-edge-list shape
// GpuForces.cu already uses for the 2G short-range force port -- see
// that header's comment for the edge-list derivation, which applies
// here unchanged (only the per-value array being scattered differs:
// weighted dChidG instead of dEdG). The one addition is a separate,
// embarrassingly-parallel dense O(numAtoms^2) reduction for the
// ai.dAdrQ[j] term (not neighbor-list-limited, unlike dChidr) -- one
// thread per atom, no cross-atom writes, no atomics.
//
// No factorization anywhere in this port (unlike the electrostatics
// solve itself, which deliberately stays CPU/Eigen) -- this is a pure
// reduction/scatter, so none of the Kalman-filter m x m inverse's
// numerical-fragility risk applies. Still validated against REAL
// production data (gpu/gemm/elecforces_test.cu, against fElec values
// dumped from an actual nnp-train stage-2 run on temp/H2O_4G, not
// synthetic), matching this project's "verify against real data"
// discipline regardless.
//
// Persistent per-structure topology state: dChidGOffset, dGdrSelf, the
// edge list, and the per-value owner-atom map are purely geometric
// (weight-independent) and uploaded ONCE per structure (keyed by
// Structure::index, exactly like GpuForces.h), mirroring the exact
// lesson that module's header documents (a first, stateless
// rebuild-and-reupload-every-call pass measured WORSE end-to-end for
// the analogous short-range edge list). dChidG, dAdrQ, pEelecpr,
// lambdaTotal, and lambdaElec all change every call (they depend on
// the elec-NN's current weights / current charges) and are re-uploaded
// every gpuElecForcesCompute() call.
//
// This header has no CUDA-specific types in it (plain doubles/ints) so
// it can be included from ordinary C++ translation units (Mode.cpp)
// without needing nvcc.

#ifndef GPU_ELEC_FORCES_H
#define GPU_ELEC_FORCES_H

namespace nnp
{

/** Upload a structure's electrostatics-force topology to the GPU ONCE.
 *  Call this exactly once per structureId, before any
 *  gpuElecForcesCompute() call for that same structureId.
 *
 * @param[in] structureId Stable identifier for the structure (e.g.
 *            Structure::index) -- used as the persistent-state cache
 *            key, same convention as GpuForces.h.
 * @param[in] numAtoms Number of atoms in the structure.
 * @param[in] dChidGOffset CSR-style offsets into dChidG/dGdrSelf,
 *            length numAtoms + 1 (same layout as GpuForces.h's
 *            dEdGOffset, but for the elec-NN's dChidG rather than the
 *            short-NN's dEdG).
 * @param[in] dGdrSelf Flattened per-atom self-term dGdr (Vec3D, i.e. 3
 *            doubles per entry, same indexing as dChidG), length
 *            dChidGOffset[numAtoms] * 3.
 * @param[in] numEdges Number of (owner, target) pair-term
 *            contributions -- one per (owner atom, neighbor,
 *            symmetry function) triple, same granularity as
 *            GpuForces.h's edge list.
 * @param[in] edgeTarget Target atom index (the one being pushed on),
 *            length numEdges.
 * @param[in] edgeOwnerIndex Flat index into this structure's dChidG
 *            array for this edge's contribution, length numEdges.
 * @param[in] edgeDGdr Flattened Vec3D dGdr value for this edge, length
 *            numEdges * 3.
 */
void gpuElecForcesUploadTopology(int structureId,
                                 int numAtoms,
                                 int const* dChidGOffset,
                                 double const* dGdrSelf,
                                 int numEdges,
                                 int const* edgeTarget,
                                 int const* edgeOwnerIndex,
                                 double const* edgeDGdr);

/** Compute the HDNNP_4G electrostatics force contribution (both the
 *  total force and the electrostatics-only fElec) on GPU for a
 *  structure whose topology has already been uploaded via
 *  gpuElecForcesUploadTopology(). Re-uploads dChidG, dAdrQ, pEelecpr,
 *  lambdaTotal, lambdaElec every call (all five depend on the elec-NN's
 *  current weights and/or current charges).
 *
 * @param[in]  structureId Same identifier passed to
 *             gpuElecForcesUploadTopology() for this structure.
 * @param[in]  dChidG Flattened per-atom dChidG, length matching what
 *             was implied by dChidGOffset at upload time.
 * @param[in]  dAdrQ Flattened dA/dr * Q, length numAtoms * numAtoms *
 *             3 (row i = atom i, column j = atom j, Vec3D each).
 * @param[in]  pEelecpr Flattened per-atom electrostatic position
 *             derivative (Atom::pEelecpr), length numAtoms * 3.
 * @param[in]  lambdaTotal Per-atom adjoint vector from
 *             Structure::calculateForceLambdaTotal(), length numAtoms
 *             (the trailing Lagrange-multiplier entry is not needed
 *             here).
 * @param[in]  lambdaElec Per-atom adjoint vector from
 *             Structure::calculateForceLambdaElec(), length numAtoms.
 * @param[out] force Overwritten (not accumulated) with the computed
 *             electrostatics force contribution on each atom -- ADD
 *             this to the short-range force already in Atom::f,
 *             mirroring the CPU code's `ai.f -= ai.pEelecpr; ...`
 *             in-place accumulation (this function returns the
 *             would-be-subtracted total directly, already negated, so
 *             callers should do `ai.f += force[...]`; see Mode.cpp's
 *             call site for the exact convention used), length
 *             numAtoms * 3.
 * @param[out] forceElec Overwritten with the computed fElec (purely
 *             electrostatic force, same convention as force), length
 *             numAtoms * 3.
 */
void gpuElecForcesCompute(int structureId,
                          double const* dChidG,
                          double const* dAdrQ,
                          double const* pEelecpr,
                          double const* lambdaTotal,
                          double const* lambdaElec,
                          double* force,
                          double* forceElec);

}

#endif
