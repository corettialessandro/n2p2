// Kernel-level validation (Phase 5 step 6, soft-percolating-jellyfish.md):
// GPU-vs-CPU symmetry-function agreement against the REAL production
// pipeline (Prediction::predict() -> Mode::evaluateNNP() ->
// calculateSymmetryFunctionGroups()) -- NOT the gpu/soa prototypes' own
// self-referential host/device check, which never independently verified
// against the actual SymGrpExpRad/SymGrpExpAngn classes and never applied
// the scale()/getScalingFactor() step those classes' real output needs.
//
// Ground truth: Structure::atoms[i].G/dGdr (owner) and
// .atoms[i].neighbors[j].dGdr (neighbor-side), read straight off a real
// predict() call on the real H2O_2G structure + converged 100-epoch
// weights. Compared against src/libnnpgpu/GpuSymmetryFunction.h's
// gpuSfExpRadGroup()/gpuSfExpAngnGroup(), with SymFnc::scale()/
// getScalingFactor() applied to the kernels' raw output afterward (see
// that header's doc comment for why this project chose to reuse the real
// scaling code rather than reimplement it).
//
// Requires -DN2P2_FULL_SFD_MEMORY (this port's chosen GPU-path memory
// layout, see soft-percolating-jellyfish.md's Phase 5 writeup) so that
// Atom::Neighbor::dGdr is indexed by GLOBAL symmetry-function index
// directly, matching gpuSf*Group()'s neighborDGd[x,y,z] output layout --
// libnnp itself must also have been built with this flag (see this
// directory's slurm script).
//
// The stale-prebuilt-library segfault documented in
// gpu/e2e_predict_check/mode_gpu_call_site_test.cpp's header comment
// (predict()'s calculateSymmetryFunctionGroups() path crashing) was
// root-caused there (commit 1477941) to a non-portable -march=native
// library built on a different node than it ran on -- not a source bug.
// A fresh clean rebuild on the actual execution node (this file's slurm
// script does this) does not hit it; nnp-predict itself was already
// validated end-to-end through this exact path to ~1e-16/1e-17.

#include "Prediction.h"
#include "Element.h"
#include "SymGrp.h"
#include "SymGrpBaseCutoff.h"
#include "SymFncExpRad.h"
#include "SymFncBaseExpAng.h"
#include "CutoffFunction.h"
#include "GpuSymmetryFunction.h"

#ifndef N2P2_FULL_SFD_MEMORY
#error "This validation harness requires -DN2P2_FULL_SFD_MEMORY " \
       "(matches the GPU symmetry-function dispatch path's chosen " \
       "memory layout -- see GpuSymmetryFunction.h)."
#endif

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace nnp;
using namespace std;

// Prediction::elements is `protected` (inherited from Mode) -- exposing it
// via `using` in a test-only subclass, same pattern as
// gpu/e2e_predict_check/mode_gpu_call_site_test.cpp's TestPrediction.
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

    // Deliberately NOT calling p.predict(): this input.nn has unit
    // normalization active (mean_energy/conv_energy/conv_length all set,
    // confirmed by grep), and predict() converts back to physical units
    // (structure.toPhysicalUnits()) *after* evaluateNNP() -- calling it
    // would leave a window where re-reading Atom::neighbors afterward
    // risks a units mismatch against the G/dGdr that were actually
    // computed in normalized-unit space. Instead, replicate
    // Mode::evaluateNNP()'s own sequence up to (and stopping at) symmetry
    // functions -- calculateNeighborList() then
    // calculateSymmetryFunctionGroups() -- so Atom::G/dGdr/neighbors are
    // read in exactly the coordinate space they were computed in, no
    // conversion round-trip involved at all.
    Structure& structure = p.structure;

    // rc per SymGrp is already in whatever units this model was trained
    // in (read directly from input.nn, never itself re-normalized) --
    // take the max across every group this harness will test to size one
    // neighbor list covering all of them, same as evaluateNNP() does with
    // its own maxCutoffRadius.
    double maxRc = 0.0;
    for (size_t e = 0; e < p.elements.size(); ++e)
    {
        for (SymGrp* g : p.elements.at(e).getSymmetryFunctionGroups())
        {
            if (g->getType() != 2 && g->getType() != 3) continue;
            SymGrpBaseCutoff* gc = dynamic_cast<SymGrpBaseCutoff*>(g);
            maxRc = max(maxRc, gc->getRc());
        }
    }
    structure.calculateNeighborList(maxRc, false);
    p.calculateSymmetryFunctionGroups(structure, true);
    printf("Structure: %zu atoms, %zu elements\n", structure.atoms.size(),
           p.elements.size());

    double maxErrG = 0.0, maxErrDGown = 0.0, maxErrDGneigh = 0.0;
    long numG = 0, numNeighG = 0;
    bool anyChecked = false;

    for (size_t e = 0; e < p.elements.size(); ++e)
    {
        Element& elem = p.elements.at(e);

        // Atoms of this element, original structure order preserved (so
        // neighbor-slot results scatter back by position).
        vector<size_t> atomIdx;
        for (size_t i = 0; i < structure.atoms.size(); ++i)
        {
            if (structure.atoms.at(i).element == e) atomIdx.push_back(i);
        }
        if (atomIdx.empty()) continue;

        for (SymGrp* g : elem.getSymmetryFunctionGroups())
        {
            size_t const type = g->getType();
            if (type != 2 && type != 3)
            {
                printf("Element %zu: skipping group type %zu (out of scope this pass)\n",
                       e, type);
                continue;
            }

            SymGrpBaseCutoff* gc = dynamic_cast<SymGrpBaseCutoff*>(g);
            if (gc->getCutoffType() != CutoffFunction::CT_TANHU)
            {
                printf("Element %zu: skipping group type %zu (cutoff not CT_TANHU)\n",
                       e, type);
                continue;
            }

            double const rc = gc->getRc();
            vector<size_t> const& memberIdx = g->getMemberIndices();
            int const numMembers = (int)memberIdx.size();
            int const numAtoms = (int)atomIdx.size();

            // Flat neighbor CSR for this element's atoms, filtered to this
            // group's cutoff, preserving each atom's own neighbor order.
            vector<int> neighOffset(numAtoms + 1, 0);
            for (int t = 0; t < numAtoms; ++t)
            {
                Atom const& a = structure.atoms.at(atomIdx.at(t));
                int cnt = 0;
                for (auto const& n : a.neighbors) if (n.d < rc) ++cnt;
                neighOffset.at(t + 1) = neighOffset.at(t) + cnt;
            }
            int const totalNeigh = neighOffset.at(numAtoms);
            vector<int> neighElem(totalNeigh);
            vector<double> neighDist(totalNeigh), neighDx(totalNeigh),
                           neighDy(totalNeigh), neighDz(totalNeigh);
            // Real Atom::neighbors index each kept CSR slot came from, to
            // scatter neighbor-slot results back onto the right
            // Atom::Neighbor::dGdr.
            vector<size_t> neighRealIdx(totalNeigh);
            for (int t = 0; t < numAtoms; ++t)
            {
                Atom const& a = structure.atoms.at(atomIdx.at(t));
                int k = neighOffset.at(t);
                for (size_t j = 0; j < a.neighbors.size(); ++j)
                {
                    Atom::Neighbor const& n = a.neighbors.at(j);
                    if (n.d < rc)
                    {
                        neighElem.at(k)  = (int)n.element;
                        neighDist.at(k)  = n.d;
                        neighDx.at(k)    = n.dr[0];
                        neighDy.at(k)    = n.dr[1];
                        neighDz.at(k)    = n.dr[2];
                        neighRealIdx.at(k) = j;
                        ++k;
                    }
                }
            }

            vector<double> G((size_t)numAtoms * numMembers, 0.0);
            vector<double> dGdx(G.size(), 0.0), dGdy(G.size(), 0.0), dGdz(G.size(), 0.0);
            vector<double> neighborDGdx((size_t)totalNeigh * numMembers, 0.0);
            vector<double> neighborDGdy(neighborDGdx.size(), 0.0);
            vector<double> neighborDGdz(neighborDGdx.size(), 0.0);

            if (type == 2)
            {
                SymFncExpRad const& first = dynamic_cast<SymFncExpRad const&>(
                    elem.getSymmetryFunction(memberIdx.at(0)));
                int const e1 = (int)first.getE1();
                vector<double> eta(numMembers), rs(numMembers);
                for (int m = 0; m < numMembers; ++m)
                {
                    SymFncExpRad const& sf = dynamic_cast<SymFncExpRad const&>(
                        elem.getSymmetryFunction(memberIdx.at(m)));
                    eta.at(m) = sf.getEta();
                    rs.at(m)  = sf.getRs();
                }
                gpuSfExpRadGroup(numAtoms, neighOffset.data(), neighElem.data(),
                                 neighDist.data(), neighDx.data(), neighDy.data(),
                                 neighDz.data(), e1, rc, numMembers,
                                 eta.data(), rs.data(),
                                 G.data(), dGdx.data(), dGdy.data(), dGdz.data(),
                                 neighborDGdx.data(), neighborDGdy.data(),
                                 neighborDGdz.data());
            }
            else // type == 3
            {
                vector<int> e1(numMembers), e2(numMembers);
                vector<double> eta(numMembers), lambda(numMembers), zeta(numMembers);
                for (int m = 0; m < numMembers; ++m)
                {
                    SymFncBaseExpAng const& sf = dynamic_cast<SymFncBaseExpAng const&>(
                        elem.getSymmetryFunction(memberIdx.at(m)));
                    e1.at(m) = (int)sf.getE1();
                    e2.at(m) = (int)sf.getE2();
                    eta.at(m) = sf.getEta();
                    lambda.at(m) = sf.getLambda();
                    zeta.at(m) = sf.getZeta();
                }
                gpuSfExpAngnGroup(numAtoms, neighOffset.data(), neighElem.data(),
                                  neighDist.data(), neighDx.data(), neighDy.data(),
                                  neighDz.data(), rc, numMembers,
                                  e1.data(), e2.data(), eta.data(), lambda.data(),
                                  zeta.data(),
                                  G.data(), dGdx.data(), dGdy.data(), dGdz.data(),
                                  neighborDGdx.data(), neighborDGdy.data(),
                                  neighborDGdz.data());
            }

            // Apply the real production scale()/getScalingFactor() per
            // member, then compare against the real CPU-computed values.
            for (int t = 0; t < numAtoms; ++t)
            {
                Atom const& a = structure.atoms.at(atomIdx.at(t));
                for (int m = 0; m < numMembers; ++m)
                {
                    SymFnc const& sf = elem.getSymmetryFunction(memberIdx.at(m));
                    size_t const gIdx = (size_t)t * numMembers + m;
                    size_t const globalIdx = memberIdx.at(m);
                    double const sfac = sf.getScalingFactor();

                    double const gScaled = sf.scale(G.at(gIdx));
                    double const errG  = fabs(gScaled - a.G.at(globalIdx));
                    double const errDx = fabs(sfac * dGdx.at(gIdx) - a.dGdr.at(globalIdx)[0]);
                    double const errDy = fabs(sfac * dGdy.at(gIdx) - a.dGdr.at(globalIdx)[1]);
                    double const errDz = fabs(sfac * dGdz.at(gIdx) - a.dGdr.at(globalIdx)[2]);
                    maxErrG = max(maxErrG, errG);
                    maxErrDGown = max({maxErrDGown, errDx, errDy, errDz});
                    ++numG;
                }

                int const off = neighOffset.at(t);
                int const nCount = neighOffset.at(t + 1) - off;
                for (int jLocal = 0; jLocal < nCount; ++jLocal)
                {
                    size_t const realJ = neighRealIdx.at(off + jLocal);
                    Atom::Neighbor const& n = a.neighbors.at(realJ);
                    for (int m = 0; m < numMembers; ++m)
                    {
                        SymFnc const& sf = elem.getSymmetryFunction(memberIdx.at(m));
                        double const sfac = sf.getScalingFactor();
                        size_t const idx = (size_t)(off + jLocal) * numMembers + m;
                        size_t const globalIdx = memberIdx.at(m);
                        double const errX = fabs(sfac * neighborDGdx.at(idx) - n.dGdr.at(globalIdx)[0]);
                        double const errY = fabs(sfac * neighborDGdy.at(idx) - n.dGdr.at(globalIdx)[1]);
                        double const errZ = fabs(sfac * neighborDGdz.at(idx) - n.dGdr.at(globalIdx)[2]);
                        maxErrDGneigh = max({maxErrDGneigh, errX, errY, errZ});
                        ++numNeighG;
                    }
                }
            }
            anyChecked = true;
            printf("Element %zu, group type %zu: %d members, %d atoms, %d neighbor entries -- checked\n",
                   e, type, numMembers, numAtoms, totalNeigh);
        }
    }

    printf("\nmax|G_gpu-G_cpu|                     = %.3E  (%ld values)\n", maxErrG, numG);
    printf("max|dG_own_gpu-dG_own_cpu|           = %.3E  (%ld values)\n", maxErrDGown, numG);
    printf("max|dG_neighbor_gpu-dG_neighbor_cpu| = %.3E  (%ld values)\n", maxErrDGneigh, numNeighG);

    bool const pass = anyChecked && maxErrG < 1e-9 && maxErrDGown < 1e-9 && maxErrDGneigh < 1e-9;
    printf("%s\n", pass ? "ALL PASS" : "FAIL");
    return pass ? 0 : 1;
}
