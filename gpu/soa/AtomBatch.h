// Phase 1 (GPU_PORTING_PLAN.md) step 1: struct-of-arrays, element-grouped,
// contiguous flattening of a real n2p2 Structure's atoms and neighbor lists.
//
// Today's layout (libnnp/Structure.h, Atom.h) is array-of-structs: one
// std::vector<Atom> per structure, each Atom owning its own
// std::vector<Atom::Neighbor>, each Neighbor its own std::vector<double>
// cache / std::vector<Vec3D> dGdr. That's pointer-chasing and non-coalesced
// on GPU. AtomBatch replaces it with:
//
//   - Atoms reordered so all atoms of element 0 come first, then element 1,
//     etc. (elementOffset gives the boundaries) -- this is what lets a later
//     phase batch the per-element NN forward/backward pass into one GEMM
//     per element instead of one tiny matmul per atom.
//   - Positions stored as separate contiguous x/y/z arrays (SoA), indexed by
//     sorted position.
//   - Neighbor lists flattened into CSR (compressed sparse row): a single
//     neighborOffset[] prefix-sum array plus flat neighborElement/neighborD/
//     neighborDx/Dy/Dz arrays, addressed via [neighborOffset[s],
//     neighborOffset[s+1]) for sorted atom s. This is the same shape as
//     gpu/smoke's real_neighbors_full.txt dump, generalized: element-sorted,
//     built directly from Structure in memory (no text round-trip), and with
//     explicit offsets instead of a linear scan to find each atom's slice.
//
// This header has no CUDA dependency -- it's a plain host-side data
// structure. It is deliberately validated against Atom::neighbors on real
// data (see build_batch_test.cpp) before any kernel is written to consume
// it, per the step-by-step approach used for the Phase 2 symmetry function
// kernels.
#pragma once

#include <cstddef>
#include <vector>

namespace nnp { struct Structure; }

struct AtomBatch
{
    std::size_t numAtoms    = 0;
    std::size_t numElements = 0;

    // Atom-level SoA, indexed by SORTED position s in [0, numAtoms).
    std::vector<std::size_t> elementOffset;    // size numElements+1
    std::vector<std::size_t> sortedToOriginal; // size numAtoms
    std::vector<std::size_t> originalToSorted; // size numAtoms (inverse perm)
    std::vector<std::size_t> element;          // size numAtoms
    std::vector<double>      x, y, z;          // size numAtoms

    // Neighbor CSR, indexed by SORTED position s in [0, numAtoms).
    // Neighbor j of sorted atom s lives at flat index
    // [neighborOffset[s] + j], j in [0, neighborOffset[s+1]-neighborOffset[s]).
    std::vector<std::size_t> neighborOffset;      // size numAtoms+1
    std::vector<std::size_t> neighborAtomSorted;  // size totalNeighbors
    std::vector<std::size_t> neighborElement;     // size totalNeighbors
    std::vector<double>      neighborD;           // size totalNeighbors
    std::vector<double>      neighborDx;          // size totalNeighbors
    std::vector<double>      neighborDy;          // size totalNeighbors
    std::vector<double>      neighborDz;          // size totalNeighbors

    std::size_t totalNeighbors() const { return neighborD.size(); }

    // --- Symmetry function output storage (step 3) -----------------------
    // Persistent per-atom storage for symmetry function values, added once a
    // kernel needs to write results somewhere durable instead of returning
    // them through throwaway per-call buffers (as step 2's kernel did). Each
    // element has its own fixed symmetry function count (mirroring how
    // input.nn defines a separate list of symmetry functions per central
    // element), so the storage is one flat array with one contiguous block
    // per element: block e holds (atoms of element e) x (sfCountPerElement[e])
    // values, row-major (all of one atom's SF values are contiguous). Call
    // allocateSfStorage() to size this after buildAtomBatch().
    std::vector<std::size_t> sfCountPerElement; // size numElements
    std::vector<std::size_t> gBlockOffset;      // size numElements+1
    std::vector<double>      G;                 // size gBlockOffset[numElements]

    // Flat index into #G for sorted atom s's sfIndex-th symmetry function.
    std::size_t gIndex(std::size_t s, std::size_t sfIndex) const
    {
        std::size_t e = element[s];
        std::size_t localRow = s - elementOffset[e];
        return gBlockOffset[e] + localRow * sfCountPerElement[e] + sfIndex;
    }
};

// Build an AtomBatch from a Structure whose neighbor list has already been
// computed (Structure::calculateNeighborList(rc) must have been called with
// rc >= the cutoff used here). Only neighbors with d < rc are kept, mirroring
// the filter SymFnc::calculate() implementations apply via the neighbor
// cutoff bookkeeping.
AtomBatch buildAtomBatch(nnp::Structure const& structure, double rc);

// Size and zero-initialize an AtomBatch's #G storage, one block per element
// sized (atoms of that element) x (sfCountPerElement[e]). Must be called
// after buildAtomBatch() has already set elementOffset/numElements.
void allocateSfStorage(AtomBatch& batch,
                       std::vector<std::size_t> const& sfCountPerElement);
