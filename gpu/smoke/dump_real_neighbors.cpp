// One-off diagnostic tool: use n2p2's own ElementMap/Structure classes
// (linked from the already-built lib/libnnp.a -- not reimplemented) to load
// a real structure from temp/H2O_2G/input.data and build its real neighbor
// list via the real Structure::calculateNeighborList(). Dumps EVERY atom's
// complete, unfiltered neighbor list (element, distance, displacement
// vector) -- exactly what n2p2 itself stores in Atom::neighbors -- so any
// symmetry function's test code can apply its own element-selection rule
// (ec/e1/e2 filters, or none for the *Weighted variants) itself, mirroring
// the corresponding SymFnc::calculate() exactly. One dump feeds every
// symmetry function type's smoke test.
//
// Not part of any build target -- compiled standalone, see run.slurm.

#include "ElementMap.h"
#include "Structure.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace nnp;
using namespace std;

int main(int argc, char** argv)
{
    string inputData = (argc > 1) ? argv[1] : "input.data";
    string outFile   = (argc > 2) ? argv[2] : "real_neighbors_full.txt";
    double const rc = 12.0; // matches all radial/angular cutoffs in input.nn

    ElementMap elementMap;
    elementMap.registerElements("H O");

    Structure structure;
    structure.setElementMap(elementMap);
    structure.readFromFile(inputData);
    structure.calculateNeighborList(rc);

    ofstream out(outFile);
    out << setprecision(17);
    out << "# atomLocalIndex atomElement neighborElement r dx dy dz\n";
    out << "# rc = " << rc << ", elements: 0=H 1=O\n";

    size_t numPairs = 0;
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        Atom const& atom = structure.atoms[i];
        for (auto const& n : atom.neighbors)
        {
            if (n.d < rc)
            {
                out << i << " " << atom.element << " " << n.element << " "
                    << n.d << " "
                    << n.dr[0] << " " << n.dr[1] << " " << n.dr[2] << "\n";
                ++numPairs;
            }
        }
    }

    cerr << "Structure has " << structure.numAtoms << " atoms total.\n";
    cerr << "Dumped " << numPairs << " neighbor pairs (all elements) to "
         << outFile << "\n";

    return 0;
}
