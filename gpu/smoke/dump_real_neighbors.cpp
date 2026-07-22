// One-off diagnostic tool: use n2p2's own ElementMap/Structure classes
// (linked from the already-built lib/libnnp.a -- not reimplemented) to load
// a real structure from temp/H2O_2G/input.data, build its real neighbor
// list via the real Structure::calculateNeighborList(), and dump the H-H
// pairs (central atom element H, neighbor element H, within rc) to a flat
// text file. This is what feeds real neighbor geometry into the CUDA
// smoke test (symfnc_exprad_test.cu) instead of synthetic random data.
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
    string outFile   = (argc > 2) ? argv[2] : "real_neighbors_HH.txt";
    double const rc = 12.0; // matches all H-H radial functions in input.nn

    ElementMap elementMap;
    elementMap.registerElements("H O");

    Structure structure;
    structure.setElementMap(elementMap);
    structure.readFromFile(inputData);
    structure.calculateNeighborList(rc);

    size_t const elementH = elementMap["H"];

    ofstream out(outFile);
    out << setprecision(17);
    out << "# centralAtomLocalIndex r dx dy dz\n";
    out << "# central atom element = H, neighbor element = H, rc = " << rc << "\n";

    size_t numCentralAtoms = 0;
    size_t numPairs = 0;
    for (auto const& atom : structure.atoms)
    {
        if (atom.element != elementH) continue;

        for (auto const& n : atom.neighbors)
        {
            if (n.element == elementH && n.d < rc)
            {
                out << numCentralAtoms << " "
                    << n.d << " "
                    << n.dr[0] << " " << n.dr[1] << " " << n.dr[2] << "\n";
                ++numPairs;
            }
        }
        ++numCentralAtoms;
    }

    cerr << "Structure has " << structure.numAtoms << " atoms total.\n";
    cerr << "Dumped " << numPairs << " H-H neighbor pairs for "
         << numCentralAtoms << " central H atoms to " << outFile << "\n";

    return 0;
}
