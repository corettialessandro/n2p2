// n2p2 - A neural network potential package
//
// GPU-accelerated real-space part of
// Structure::calculateElectrostaticEnergy()'s Ewald matrix assembly (see
// gpu/README.md's "Ewald matrix assembly" follow-ups). Unlike the
// reciprocal-space sum (already reformulated as two GEMMs on CPU, and
// cheap for this project's datasets), the real-space erfc double sum
// measured out as the actual dominant cost: ~6s/epoch/rank, ~202M
// neighbor-pairs/rank/epoch, genuinely neighbor-list-bounded (not
// O(numAtoms^2)) transcendental-function-bound compute.
//
// Math (mirrors the CPU loop in Structure.cpp exactly): for every atom i
// and every neighbor j (tag >= i, i.e. only the upper triangle, within
// the Ewald real-space cutoff rCut -- possibly several periodic images
// of the same physical atom, each a separate neighbor-list entry with
// its own distance):
//   A(i, j) += (erfc(rij / sqrt2eta) - erfc(rij / gammaSqrt2(ei, ej)))
//              / (rij * fourPiEps)
// This is naturally an EDGE LIST (owner atom i, target atom j, distance
// rij, element pair (ei, ej)) -- iterating it one-thread-per-edge turns
// the whole sum into a parallel scatter-add, atomicAdd-ing into a dense
// (numAtoms x numAtoms) output buffer (multiple periodic images of the
// same (i, j) pair are multiple edges targeting the same output slot,
// hence atomicAdd -- not a per-edge-unique write).
//
// Persistent per-structure topology state, exactly mirroring
// GpuForces.h's rationale: the edge list (edgeOwner/edgeTarget/edgeRij/
// edgeElemI/edgeElemJ) is purely geometric -- atom positions and the
// real-space cutoff never change during training -- so it is uploaded
// exactly ONCE per structure (gpuEwaldRealUploadTopology(), keyed by
// structureId, i.e. Structure::index) and reused for every subsequent
// call. Only gammaSqrt2 (depends on the trainable per-element Qsigma)
// and sqrt2eta/fourPiEps (tiny scalars) change per call, re-uploaded
// every time by gpuEwaldRealCompute().
//
// This header has no CUDA-specific types in it (plain doubles/ints) so
// it can be included from ordinary C++ translation units (Structure.cpp)
// without needing nvcc.

#ifndef GPU_EWALD_REAL_H
#define GPU_EWALD_REAL_H

namespace nnp
{

/** Upload a structure's real-space Ewald edge list to the GPU ONCE.
 *  Call this exactly once per structureId, before any
 *  gpuEwaldRealCompute() call for that same structureId -- the topology
 *  (purely geometric) is cached and reused for every subsequent call.
 *
 * @param[in] structureId Stable identifier for the structure (e.g.
 *            Structure::index) -- used as the persistent-state cache
 *            key.
 * @param[in] numAtoms Number of atoms in the structure.
 * @param[in] numElements Number of elements in this dataset (fixed for
 *            the whole training run) -- used to size the small
 *            gammaSqrt2 buffer allocated here and re-filled every
 *            gpuEwaldRealCompute() call.
 * @param[in] numEdges Number of (owner, target) real-space
 *            contributions (see this file's header comment).
 * @param[in] edgeOwner Owner atom index i, length numEdges.
 * @param[in] edgeTarget Target atom index j (>= edgeOwner[e]), length
 *            numEdges.
 * @param[in] edgeRij Distance between owner and target for this
 *            (possibly periodic-image) edge, length numEdges.
 * @param[in] edgeElemI Owner atom's element index, length numEdges.
 * @param[in] edgeElemJ Target atom's element index, length numEdges.
 */
void gpuEwaldRealUploadTopology(int structureId,
                                int numAtoms,
                                int numElements,
                                int numEdges,
                                int const* edgeOwner,
                                int const* edgeTarget,
                                double const* edgeRij,
                                int const* edgeElemI,
                                int const* edgeElemJ);

/** Compute the real-space Ewald matrix contribution on GPU for a
 *  structure whose topology has already been uploaded via
 *  gpuEwaldRealUploadTopology(). Only re-uploads gammaSqrt2 (the one
 *  input that changes every call, since it depends on the current
 *  per-element Qsigma weights) plus the tiny sqrt2eta/fourPiEps
 *  scalars.
 *
 * @param[in]  structureId Same identifier passed to
 *             gpuEwaldRealUploadTopology() for this structure.
 * @param[in]  gammaSqrt2 Flattened (row-major) numElements x
 *             numElements matrix, matching numElements passed to
 *             gpuEwaldRealUploadTopology().
 * @param[in]  sqrt2eta sqrt(2) * Ewald eta parameter.
 * @param[in]  fourPiEps Unit-system prefactor (see caller).
 * @param[out] AUpperAdd Overwritten (not accumulated) with the
 *             computed real-space contribution, dense row-major
 *             numAtoms x numAtoms (numAtoms as passed to
 *             gpuEwaldRealUploadTopology()) -- only entries with
 *             row <= col are ever written by any edge; the caller adds
 *             this into the upper triangle of its own A matrix.
 */
void gpuEwaldRealCompute(int structureId,
                         double const* gammaSqrt2,
                         double sqrt2eta,
                         double fourPiEps,
                         double* AUpperAdd);

}

#endif
