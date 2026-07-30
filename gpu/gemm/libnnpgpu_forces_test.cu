// Validates src/libnnpgpu/GpuForces.cu's gpuForcesUploadTopology()/
// gpuForcesCompute() -- the GPU port of Mode::calculateForces()
// (gpu/README.md's "F_err is ~84% calculateForces(), not GPU dispatch"
// follow-up, and its persistent-per-structure-topology follow-up after
// the first, stateless pass measured WORSE end-to-end).
//
// Ground truth: the REAL nnp::Mode::calculateForces() (public method),
// called directly -- not a hand-rolled reimplementation. Runs against
// REAL H2O_2G structures (read directly from temp/H2O_2G/input.data,
// several in a row, so atom count and neighbor-list topology genuinely
// vary structure to structure, matching this project's established "vary
// conditions across repeated calls" test philosophy -- see
// libnnpgpu_test.cu/_dedc_test.cu/_dfdc_test.cu).
//
// Crucially, this also tests the PERSISTENT-STATE aspect specifically:
// per structure, uploads topology ONCE (gpuForcesUploadTopology()), then
// calls gpuForcesCompute() MULTIPLE times with DIFFERENT, manually
// perturbed dEdG values each time (simulating repeated
// Mode::calculateForces() calls across different NN weights during
// training, which is exactly the real calling pattern -- topology fixed,
// dEdG changes) -- each compared against a FRESH real
// Mode::calculateForces() CPU reference computed for that exact
// perturbed dEdG (calculateForces() is public and reads whatever is
// currently in structure.atoms[i].dEdG/.dGdr/.neighbors, so perturbing
// dEdG directly and re-calling it gives an authoritative, real reference
// without needing to re-run the NN).

#include "../../src/libnnpgpu/GpuForces.h"
#include "Prediction.h"
#include "Structure.h"
#include "Element.h"
#include "Atom.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

using namespace nnp;
using namespace std;

// Prediction.h keeps elements/maxCutoffRadius protected (Mode.h) and only
// supports reading ONE structure per call by reopening the file from the
// start -- this subclass exposes what's needed to (a) read several DIFFERENT
// real structures in a row from one open stream, and (b) access the
// symmetry function element-pair table needed to build the edge list.
class TestPrediction : public Prediction
{
public:
    vector<Element> const& getElements() const { return elements; }
    double getMaxCutoffRadiusProtected() const { return maxCutoffRadius; }

    // Mirrors Prediction::readStructureFromFile()'s body exactly, but
    // reads from an already-open stream instead of reopening the file,
    // so repeated calls advance through successive structures.
    bool readNextStructure(ifstream& file)
    {
        if (!file.good() ||
            file.peek() == ifstream::traits_type::eof()) return false;
        structure.reset();
        structure.setElementMap(elementMap);
        structure.readFromFile(file);
        removeEnergyOffset(structure);
        if (normalize)
        {
            structure.toNormalizedUnits(meanEnergy, convEnergy, convLength,
                                        convCharge);
        }
        return true;
    }

    // Deliberately calls evaluateNNP() directly instead of predict() --
    // predict() follows evaluateNNP() with toPhysicalUnits(), which has a
    // real, pre-existing bug in Atom::toPhysicalUnits()/toNormalizedUnits()
    // (src/libnnp/Atom.cpp's post-neighbor-list conversion loop indexes
    // the atom's OWN dGdr instead of the neighbor's `it->dGdr`, so it
    // re-multiplies dGdr by convLength once per neighbor entry --
    // convLength^numNeighbors). It's dormant in normal nnp-predict usage
    // since nothing reads dGdr after that conversion runs, but this test
    // DOES need dGdr afterward, so it sidesteps the bug by never calling
    // toPhysicalUnits() at all -- comparing everything in the NNP's
    // native normalized units instead, which is all this test needs.
    void evaluateOnly() { evaluateNNP(structure); }
};

int main()
{
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    printf("CUDA devices visible: %d\n", devCount);
    if (devCount == 0) { fprintf(stderr, "No CUDA device.\n"); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device 0: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    ofstream logFile("nnp-forces-test.log");
    TestPrediction prediction;
    prediction.log.registerStreamPointer(&logFile);
    prediction.setup();

    double const maxCutoffRadius = prediction.getMaxCutoffRadiusProtected();
    vector<Element> const& elements = prediction.getElements();

    ifstream dataFile("input.data");
    if (!dataFile.is_open())
    {
        fprintf(stderr, "Cannot open input.data\n");
        return 1;
    }

    mt19937 rng(99);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    int const numStructuresToTest = 5;
    int const callsPerStructure = 4; // upload topology once, compute 4x with varying dEdG
    double maxErr = 0.0;
    bool ok = true;
    int tested = 0;
    for (int s = 0; s < numStructuresToTest; ++s)
    {
        if (!prediction.readNextStructure(dataFile)) break;
        Structure& structure = prediction.structure;
        // NOT predict() -- see TestPrediction::evaluateOnly()'s comment.
        // Gets us a real geometry/neighbor list/dGdr and an initial dEdG.
        prediction.evaluateOnly();

        int const numAtoms = (int)structure.numAtoms;

        // --- Build CSR dEdGOffset/dGdrSelf (self term topology) --------
        vector<int> dEdGOffset(numAtoms + 1, 0);
        for (int i = 0; i < numAtoms; ++i)
        {
            dEdGOffset[i + 1] = dEdGOffset[i]
                + (int)structure.atoms.at(i).numSymmetryFunctions;
        }
        int const numValues = dEdGOffset[numAtoms];
        vector<double> dGdrSelf((size_t)numValues * 3);
        for (int i = 0; i < numAtoms; ++i)
        {
            Atom const& a = structure.atoms.at(i);
            int const off = dEdGOffset[i];
            for (size_t k = 0; k < a.numSymmetryFunctions; ++k)
            {
                dGdrSelf[3 * (off + k) + 0] = a.dGdr.at(k).r[0];
                dGdrSelf[3 * (off + k) + 1] = a.dGdr.at(k).r[1];
                dGdrSelf[3 * (off + k) + 2] = a.dGdr.at(k).r[2];
            }
        }

        // --- Build owner-centric edge list topology (pair term) --------
        // Same set of (owner, target, dEdG-index, dGdr) contributions
        // Mode::calculateForces() computes, just traversed owner-centric
        // (one direct pass over each atom's own neighbor list) instead of
        // target-centric-with-a-search -- see GpuForces.h's header
        // comment for why these are equivalent.
        vector<int> edgeTarget, edgeOwnerDEdGIndex;
        vector<double> edgeDGdr;
        for (int j = 0; j < numAtoms; ++j)
        {
            Atom const& aj = structure.atoms.at(j);
            vector<vector<size_t>> const& tableFull =
                elements.at(aj.element).getSymmetryFunctionTable();
            size_t const numNeighbors =
                aj.getStoredMinNumNeighbors(maxCutoffRadius);
            for (size_t k = 0; k < numNeighbors; ++k)
            {
                Atom::Neighbor const& n = aj.neighbors.at(k);
                vector<size_t> const& table = tableFull.at(n.element);
                for (size_t m = 0; m < n.dGdr.size(); ++m)
                {
                    edgeTarget.push_back((int)n.index);
                    edgeOwnerDEdGIndex.push_back(dEdGOffset[j]
                                                + (int)table.at(m));
                    edgeDGdr.push_back(n.dGdr.at(m).r[0]);
                    edgeDGdr.push_back(n.dGdr.at(m).r[1]);
                    edgeDGdr.push_back(n.dGdr.at(m).r[2]);
                }
            }
        }
        int const numEdges = (int)edgeTarget.size();

        // Topology uploaded ONCE for this structure.
        gpuForcesUploadTopology(s, numAtoms, dEdGOffset.data(),
                                dGdrSelf.data(), numEdges, edgeTarget.data(),
                                edgeOwnerDEdGIndex.data(), edgeDGdr.data());

        double localMaxErr = 0.0;
        for (int call = 0; call < callsPerStructure; ++call)
        {
            // Perturb dEdG (simulating a weight update between calls,
            // real geometry/topology unchanged) and get a fresh, real CPU
            // reference for THIS exact dEdG via the public
            // calculateForces() -- not a re-derivation.
            if (call > 0)
            {
                for (auto& a : structure.atoms)
                {
                    for (auto& d : a.dEdG) d = dist(rng);
                }
            }
            prediction.calculateForces(structure);

            vector<double> dEdG(numValues);
            for (int i = 0; i < numAtoms; ++i)
            {
                Atom const& a = structure.atoms.at(i);
                int const off = dEdGOffset[i];
                for (size_t k = 0; k < a.numSymmetryFunctions; ++k)
                {
                    dEdG[off + k] = a.dEdG.at(k);
                }
            }

            vector<double> forceGpu((size_t)numAtoms * 3);
            gpuForcesCompute(s, dEdG.data(), forceGpu.data());

            double callMaxErr = 0.0;
            for (int i = 0; i < numAtoms; ++i)
            {
                Vec3D const& fRef = structure.atoms.at(i).f;
                for (int c = 0; c < 3; ++c)
                {
                    callMaxErr = max(callMaxErr,
                                    fabs(forceGpu[3 * i + c] - fRef.r[c]));
                }
            }
            localMaxErr = max(localMaxErr, callMaxErr);
            printf("structure %d call %d: atoms=%4d  edges=%6d  "
                   "max|F diff|=%.3E\n",
                   s, call, numAtoms, numEdges, callMaxErr);
        }

        maxErr = max(maxErr, localMaxErr);
        if (localMaxErr > 1e-6) ok = false;
        tested++;
    }

    printf("\ntested %d real H2O_2G structures, %d calls each "
           "(1 topology upload + %d dEdG-only recomputes)\n",
           tested, callsPerStructure, callsPerStructure - 1);
    printf("max over all structures/calls: %.3E\n", maxErr);
    printf("%s\n", (ok && tested > 0) ? "ALL PASS" : "SOME FAILED");
    return (ok && tested > 0) ? 0 : 1;
}
