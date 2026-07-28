// Validates the real Phase 6 GPU call site -- Mode::calculateAtomicNeuralNetworks()
// -- by calling it directly (it's public API, see Mode.h's own class-level
// doc comment: "mode.calculateAtomicNeuralNetworks(structure, true);") on a
// real Prediction object built from the real H2O_2G input.nn/weights.*.data,
// bypassing Mode::calculateSymmetryFunctionGroups() (a pre-existing crash in
// this branch unrelated to this port -- see gpu/e2e_predict_check/README
// or the commit message for the gdb backtrace: it segfaults inside
// SymGrpExpRad::calculate(), called well before calculateAtomicNeuralNetworks
// is ever reached, and nothing about that path was touched by this GPU
// integration). Since calculateAtomicNeuralNetworks only reads Atom::G and
// writes Atom::energy/dEdG (confirmed by reading its source), synthetic
// random G values are just as valid a test of THIS call site as real
// symmetry-function output would be -- same "isolate the piece actually
// being changed" approach used throughout gpu/'s other validation code.
//
// This binary is compiled and linked TWICE by run_mode_call_site.slurm:
// once against a plain libnnp.a (CPU path), once against a libnnp.a built
// with GPU=1 plus libnnpgpu.a and the CUDA runtime libraries (GPU path) --
// both builds share this exact same source file, only the library/link
// flags differ, exactly mirroring how nnp-predict itself is built for the
// two cases.

#include "Prediction.h"
#include "NeuralNetwork.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace nnp;
using namespace std;

// Prediction::elements is inherited from Mode as `protected` -- exposing it
// via `using` in a test-only subclass is the standard, safe way to reach a
// protected member for testing without changing its access in the real
// class (which several methods rely on staying non-public).
class TestPrediction : public Prediction
{
public:
    using Mode::elements;
};

int main(int argc, char** argv)
{
    string dir = (argc > 1) ? argv[1] : ".";

    TestPrediction p;
    p.setup();
    p.readStructureFromFile((dir + "/input.data").c_str());

    string const id = "short";
    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto& a : p.structure.atoms)
    {
        int numIn = p.elements.at(a.element).neuralNetworks.at(id)
                    .getNumNeuronsInLayer(0);
        a.G.assign(numIn, 0.0);
        a.dEdG.assign(numIn, 0.0);
        for (auto& g : a.G) g = dist(rng);
    }

    p.calculateAtomicNeuralNetworks(p.structure, true, id);

    printf("atoms=%zu\n", p.structure.atoms.size());
    double sumEnergy = 0.0;
    for (auto const& a : p.structure.atoms) sumEnergy += a.energy;
    printf("sumEnergy=%.15E\n", sumEnergy);
    for (size_t i = 0; i < p.structure.atoms.size(); ++i)
    {
        Atom const& a = p.structure.atoms.at(i);
        printf("atom %5zu energy=%.15E dEdG[0]=%.15E dEdG[last]=%.15E\n",
               i, a.energy, a.dEdG.front(), a.dEdG.back());
    }

    return 0;
}
