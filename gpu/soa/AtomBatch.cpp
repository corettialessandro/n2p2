#include "AtomBatch.h"
#include "Structure.h"

using namespace std;
using namespace nnp;

AtomBatch buildAtomBatch(Structure const& structure, double rc)
{
    AtomBatch b;
    b.numAtoms    = structure.numAtoms;
    b.numElements = structure.numElements;

    // --- Element-sorted atom order -----------------------------------
    // elementOffset[e] = number of atoms with element < e (prefix sum of
    // per-element counts), so atoms of element e occupy sorted positions
    // [elementOffset[e], elementOffset[e+1]).
    b.elementOffset.assign(b.numElements + 1, 0);
    for (auto const& atom : structure.atoms) b.elementOffset[atom.element + 1]++;
    for (size_t e = 0; e < b.numElements; ++e)
    {
        b.elementOffset[e + 1] += b.elementOffset[e];
    }

    b.sortedToOriginal.assign(b.numAtoms, 0);
    b.originalToSorted.assign(b.numAtoms, 0);
    b.element.assign(b.numAtoms, 0);
    b.x.assign(b.numAtoms, 0.0);
    b.y.assign(b.numAtoms, 0.0);
    b.z.assign(b.numAtoms, 0.0);

    // Running write cursor per element, seeded at that element's block start.
    vector<size_t> cursor(b.elementOffset.begin(), b.elementOffset.end() - 1);
    for (size_t i = 0; i < b.numAtoms; ++i)
    {
        Atom const& atom = structure.atoms[i];
        size_t s = cursor[atom.element]++;
        b.sortedToOriginal[s] = i;
        b.originalToSorted[i] = s;
        b.element[s]           = atom.element;
        b.x[s] = atom.r[0];
        b.y[s] = atom.r[1];
        b.z[s] = atom.r[2];
    }

    // --- Neighbor CSR ---------------------------------------------------
    b.neighborOffset.assign(b.numAtoms + 1, 0);
    for (size_t s = 0; s < b.numAtoms; ++s)
    {
        Atom const& atom = structure.atoms[b.sortedToOriginal[s]];
        size_t count = 0;
        for (auto const& n : atom.neighbors) if (n.d < rc) count++;
        b.neighborOffset[s + 1] = count;
    }
    for (size_t s = 0; s < b.numAtoms; ++s)
    {
        b.neighborOffset[s + 1] += b.neighborOffset[s];
    }

    size_t const totalNeighbors = b.neighborOffset[b.numAtoms];
    b.neighborAtomSorted.assign(totalNeighbors, 0);
    b.neighborElement.assign(totalNeighbors, 0);
    b.neighborD.assign(totalNeighbors, 0.0);
    b.neighborDx.assign(totalNeighbors, 0.0);
    b.neighborDy.assign(totalNeighbors, 0.0);
    b.neighborDz.assign(totalNeighbors, 0.0);

    for (size_t s = 0; s < b.numAtoms; ++s)
    {
        Atom const& atom = structure.atoms[b.sortedToOriginal[s]];
        size_t k = b.neighborOffset[s];
        for (auto const& n : atom.neighbors)
        {
            if (n.d < rc)
            {
                b.neighborAtomSorted[k] = b.originalToSorted[n.index];
                b.neighborElement[k]    = n.element;
                b.neighborD[k]          = n.d;
                b.neighborDx[k]         = n.dr[0];
                b.neighborDy[k]         = n.dr[1];
                b.neighborDz[k]         = n.dr[2];
                ++k;
            }
        }
    }

    return b;
}

void allocateSfStorage(AtomBatch& b, vector<size_t> const& sfCountPerElement)
{
    b.sfCountPerElement = sfCountPerElement;
    b.gBlockOffset.assign(b.numElements + 1, 0);
    for (size_t e = 0; e < b.numElements; ++e)
    {
        size_t atomsInElement = b.elementOffset[e + 1] - b.elementOffset[e];
        b.gBlockOffset[e + 1] = b.gBlockOffset[e]
                              + atomsInElement * sfCountPerElement[e];
    }
    b.G.assign(b.gBlockOffset[b.numElements], 0.0);
}
