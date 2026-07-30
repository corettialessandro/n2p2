// n2p2 - A neural network potential package
//
// GPU-accelerated Mode::calculateForces() (HDNNP_2G, !N2P2_FULL_SFD_MEMORY
// only -- see Mode.cpp's call site). This is the "calculateForces() is
// ~84% of F_err" follow-up in gpu/README.md: unlike the three dispatch
// functions in GpuNeuralNetwork.h, this one has nothing to do with the NN
// forward/backward pass or its architecture -- it only needs each atom's
// dEdG (already computed elsewhere by the time this runs) and the
// structure's neighbor-list geometry (dGdr values), so it has no
// hasGpuCompatibleArchitecture()-style restriction.
//
// Math (mirrors Atom::calculateSelfForceShort()/calculatePairForceShort()
// exactly): for every atom i,
//   F_i = -sum_k dEdG_i[k] * dGdrSelf_i[k]                      (self term)
//         - sum_{(j,k): n=j.neighbors[k], n.index==i}
//               dEdG_j[table_j[n.element][k]] * n.dGdr[k]       (pair term)
// The self term has no cross-atom dependency. The pair term is naturally
// an EDGE LIST (owner atom j, target atom i, and j's own dEdG index for
// this contribution) -- iterating it owner-centric (this header's
// caller's job to build, see Mode.cpp's call site and
// gpu/gemm/libnnpgpu_forces_test.cu) avoids the CPU code's O(neighbors)
// inner search for the back-reference entry entirely, and turns the pair
// term into a simple parallel scatter-add, one thread per edge,
// atomicAdd-ing into the target atom's force accumulator.
//
// Persistent per-structure topology state: a first-pass version of this
// (rebuild-and-reupload-everything-every-call, matching the phased
// approach that worked fine for GpuNeuralNetwork.cu's three dispatch
// functions) was measured end-to-end and made F_err WORSE, not better
// (104-110s -> 136.3s -- see gpu/README.md). Root cause: the edge list
// here is one to two orders of magnitude bigger than anything the three
// NN dispatch functions ever moved (up to ~1.7M entries for one real
// H2O_2G structure, tens of MB per call), called up to 4x per force
// update (SM_THRESHOLD's 3 trials + 1 final PART 2 call) under the same
// 32-way MPS contention that makes cudaMalloc/cudaMemcpy/cudaFree
// themselves become multi-millisecond-to-second blocking stalls (see
// gpu/README.md's rank-0 profiling section). Unlike the NN dispatch
// functions' weight/G/dGdxyz inputs, which genuinely change every call,
// this function's topology (dEdGOffset, dGdrSelf, the whole edge list)
// is PURELY GEOMETRIC and never changes during training -- atom
// positions are fixed once a structure is loaded. Only dEdG changes
// (since it depends on the NN's current weights). So the topology is
// uploaded exactly ONCE per structure (gpuForcesUploadTopology(),
// keyed by structureId -- Structure::index, stable for the structure's
// whole lifetime) and reused for every subsequent call
// (gpuForcesCompute(), which only re-uploads the small dEdG array).
//
// This header has no CUDA-specific types in it (plain doubles/ints)
// so it can be included from ordinary C++ translation units (Mode.cpp)
// without needing nvcc.

#ifndef GPU_FORCES_H
#define GPU_FORCES_H

namespace nnp
{

/** Upload a structure's force-computation topology to the GPU ONCE.
 *  Call this exactly once per structureId, before any
 *  gpuForcesCompute() call for that same structureId -- the topology
 *  (purely geometric, weight-independent) is cached and reused for
 *  every subsequent call.
 *
 * @param[in] structureId Stable identifier for the structure (e.g.
 *            Structure::index) -- used as the persistent-state cache
 *            key.
 * @param[in] numAtoms Number of atoms in the structure.
 * @param[in] dEdGOffset CSR-style offsets into dEdG/dGdrSelf, length
 *            numAtoms + 1: atom i's own data spans
 *            [dEdGOffset[i], dEdGOffset[i+1]), i.e.
 *            dEdGOffset[i+1] - dEdGOffset[i] == atom i's
 *            numSymmetryFunctions.
 * @param[in] dGdrSelf Flattened per-atom self-term dGdr (Vec3D, i.e. 3
 *            doubles per entry, same indexing as dEdG), length
 *            dEdGOffset[numAtoms] * 3.
 * @param[in] numEdges Number of (owner, target) pair-term contributions
 *            (see this file's header comment for the edge-list
 *            derivation).
 * @param[in] edgeTarget Target atom index (the one being pushed on),
 *            length numEdges.
 * @param[in] edgeOwnerDEdGIndex Flat index into this structure's dEdG
 *            array (i.e. dEdGOffset[owner] +
 *            table_owner[neighborElement][k]) for this edge's
 *            contribution, length numEdges.
 * @param[in] edgeDGdr Flattened Vec3D dGdr value for this edge, length
 *            numEdges * 3.
 */
void gpuForcesUploadTopology(int structureId,
                             int numAtoms,
                             int const* dEdGOffset,
                             double const* dGdrSelf,
                             int numEdges,
                             int const* edgeTarget,
                             int const* edgeOwnerDEdGIndex,
                             double const* edgeDGdr);

/** Compute atomic forces on GPU for a structure whose topology has
 *  already been uploaded via gpuForcesUploadTopology(). Only re-uploads
 *  dEdG (the one input that changes every call, since it depends on the
 *  NN's current weights).
 *
 * @param[in]  structureId Same identifier passed to
 *             gpuForcesUploadTopology() for this structure.
 * @param[in]  dEdG Flattened per-atom dEdG, length matching what was
 *             implied by dEdGOffset at upload time.
 * @param[out] force Overwritten (not accumulated) with the computed
 *             force on each atom, length numAtoms * 3 (numAtoms as
 *             passed to gpuForcesUploadTopology()).
 */
void gpuForcesCompute(int structureId,
                      double const* dEdG,
                      double* force);

}

#endif
