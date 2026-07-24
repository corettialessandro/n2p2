// Phase 1 step 1 validation: build an AtomBatch from the real H2O_2G
// structure(s) and check, atom by atom and neighbor by neighbor, that it
// reproduces exactly what Structure::calculateNeighborList() + Atom::neighbors
// already store -- pure host-side check, no CUDA/GPU involved yet. Proves
// the new SoA/CSR layout is a faithful, lossless re-encoding of the existing
// AoS data before any kernel is written to consume it.
//
// Linked against lib/libnnp.a, same as gpu/smoke/dump_real_neighbors.cpp (no
// GSL/BLAS needed -- see that file's header comment). Not part of any build
// target, see run.slurm.

#include "AtomBatch.h"
#include "ElementMap.h"
#include "Structure.h"
#include <iostream>
#include <cmath>

using namespace nnp;
using namespace std;

namespace
{

int    failures = 0;
size_t checks    = 0;

void expectEq(size_t a, size_t b, char const* what)
{
    ++checks;
    if (a != b)
    {
        cerr << "FAIL " << what << ": " << a << " != " << b << "\n";
        ++failures;
    }
}

void expectExact(double a, double b, char const* what)
{
    ++checks;
    if (a != b)
    {
        cerr << "FAIL " << what << ": " << a << " != " << b << "\n";
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv)
{
    string inputData = (argc > 1) ? argv[1] : "input.data";
    double const rc = 12.0; // matches input.nn cutoffs, same as dump_real_neighbors.cpp

    ElementMap elementMap;
    elementMap.registerElements("H O");

    Structure structure;
    structure.setElementMap(elementMap);
    structure.readFromFile(inputData);
    structure.calculateNeighborList(rc);

    cout << "Structure has " << structure.numAtoms << " atoms, "
         << structure.numElements << " elements.\n";

    AtomBatch batch = buildAtomBatch(structure, rc);

    expectEq(batch.numAtoms, structure.numAtoms, "numAtoms");
    expectEq(batch.numElements, structure.numElements, "numElements");

    // --- Permutation is a bijection --------------------------------------
    for (size_t i = 0; i < batch.numAtoms; ++i)
    {
        size_t s = batch.originalToSorted[i];
        expectEq(batch.sortedToOriginal[s], i, "sortedToOriginal(originalToSorted(i)) == i");
    }

    // --- Element grouping + positions -------------------------------------
    size_t totalNeighborsCheck = 0;
    for (size_t e = 0; e < batch.numElements; ++e)
    {
        for (size_t s = batch.elementOffset[e]; s < batch.elementOffset[e + 1]; ++s)
        {
            expectEq(batch.element[s], e, "element[s] within its elementOffset block");
        }
    }

    for (size_t i = 0; i < structure.numAtoms; ++i)
    {
        Atom const& atom = structure.atoms[i];
        size_t s = batch.originalToSorted[i];

        expectEq(batch.element[s], atom.element, "element[s] == atom.element");
        expectExact(batch.x[s], atom.r[0], "x[s] == atom.r[0]");
        expectExact(batch.y[s], atom.r[1], "y[s] == atom.r[1]");
        expectExact(batch.z[s], atom.r[2], "z[s] == atom.r[2]");

        // --- Neighbor CSR slice for this atom -----------------------------
        size_t begin = batch.neighborOffset[s];
        size_t end   = batch.neighborOffset[s + 1];

        size_t refCount = 0;
        for (auto const& n : atom.neighbors) if (n.d < rc) refCount++;
        expectEq(end - begin, refCount, "neighbor count for atom");

        size_t k = begin;
        for (auto const& n : atom.neighbors)
        {
            if (n.d >= rc) continue;
            expectEq(batch.sortedToOriginal[batch.neighborAtomSorted[k]], n.index,
                      "neighborAtomSorted maps back to original neighbor index");
            expectEq(batch.neighborElement[k], n.element, "neighborElement");
            expectExact(batch.neighborD[k], n.d, "neighborD");
            expectExact(batch.neighborDx[k], n.dr[0], "neighborDx");
            expectExact(batch.neighborDy[k], n.dr[1], "neighborDy");
            expectExact(batch.neighborDz[k], n.dr[2], "neighborDz");
            ++k;
        }
        totalNeighborsCheck += refCount;
    }
    expectEq(batch.totalNeighbors(), totalNeighborsCheck, "grand total neighbor count");

    cout << "Batch has " << batch.totalNeighbors() << " total neighbor entries.\n";
    cout << checks << " checks run, " << failures << " failures.\n";
    cout << (failures == 0 ? "PASS" : "FAIL") << "\n";

    return failures == 0 ? 0 : 1;
}
