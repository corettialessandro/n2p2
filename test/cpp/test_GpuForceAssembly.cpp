#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GpuForceAssembly
#include <boost/test/unit_test.hpp>

#include "GpuForceAssembly.h"
#include "tolerance.h"
#include <cmath>     // std::sin, std::cos
#include <cstddef>   // std::size_t
#include <stdexcept> // std::runtime_error
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

// Irregular test topology: numAtoms atoms, numValues symmetry function
// values, a varying number of terms per atom (including atoms without
// terms).
struct TestTopology
{
    size_t         numAtoms;
    size_t         numValues;
    vector<int>    termOffset;
    vector<int>    termValue;
    vector<double> termDGdr;

    TestTopology(size_t numAtoms, size_t numValues, size_t seed) :
        numAtoms(numAtoms),
        numValues(numValues),
        termOffset(numAtoms + 1, 0)
    {
        for (size_t i = 0; i < numAtoms; ++i)
        {
            size_t const numTerms = (i % 7 == 3) ? 0 : 5 + (i * 13 + seed) % 90;
            termOffset.at(i + 1) = termOffset.at(i) + numTerms;
            for (size_t t = 0; t < numTerms; ++t)
            {
                termValue.push_back((i * 31 + t * 17 + seed) % numValues);
                for (size_t c = 0; c < 3; ++c)
                {
                    termDGdr.push_back(sin(0.3 * termDGdr.size() + seed));
                }
            }
        }
    }

    // Sums on the CPU.
    vector<double> apply(size_t numWeightVectors,
                         vector<double> const& weights) const
    {
        vector<double> result(numWeightVectors * 3 * numAtoms, 0.0);
        for (size_t w = 0; w < numWeightVectors; ++w)
        {
            for (size_t i = 0; i < numAtoms; ++i)
            {
                for (int t = termOffset.at(i); t < termOffset.at(i + 1); ++t)
                {
                    double const weight =
                        weights.at(w * numValues + termValue.at(t));
                    for (size_t c = 0; c < 3; ++c)
                    {
                        result.at((w * numAtoms + i) * 3 + c) +=
                            weight * termDGdr.at(3 * t + c);
                    }
                }
            }
        }
        return result;
    }

};

vector<double> makeWeights(size_t numWeightVectors, size_t numValues)
{
    vector<double> w(numWeightVectors * numValues);
    for (size_t i = 0; i < w.size(); ++i) w.at(i) = cos(0.7 * i + 0.2);
    return w;
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Apply_WeightVectors_MatchesCpuSums)
{
    TestTopology const t(150, 400, 1);
    GpuForceTopology topology(t.numValues, t.termOffset, t.termValue,
                              t.termDGdr);
    for (size_t nw = 1; nw <= GpuForceTopology::maxWeightVectors; ++nw)
    {
        BOOST_TEST_INFO("weight vectors: " << nw);
        vector<double> const weights = makeWeights(nw, t.numValues);
        vector<double> const cpu = t.apply(nw, weights);
        vector<double> gpu(cpu.size());
        topology.apply(nw, weights.data(), gpu.data());
        Tolerance const tolerance(gpuAbsoluteEvaluation, gpuRelativeEvaluation);
        for (size_t i = 0; i < cpu.size(); ++i)
        {
            BOOST_TEST_INFO("entry " << i);
            REQUIRE_CLOSE_TO_REFERENCE(cpu.at(i), gpu.at(i), tolerance);
        }
    }
}

BOOST_AUTO_TEST_CASE(Apply_Repeated_Deterministic)
{
    TestTopology const t(200, 300, 2);
    GpuForceTopology topology(t.numValues, t.termOffset, t.termValue,
                              t.termDGdr);
    vector<double> const weights = makeWeights(2, t.numValues);
    vector<double> first(2 * 3 * t.numAtoms);
    vector<double> again(first.size());
    topology.apply(2, weights.data(), first.data());
    for (size_t r = 0; r < 5; ++r)
    {
        topology.apply(2, weights.data(), again.data());
        BOOST_REQUIRE(first == again);
    }
}

BOOST_AUTO_TEST_CASE(MemoryUsage_CreateAndDestroy_Tracked)
{
    size_t const before = GpuForceTopology::getTotalMemoryUsage();
    TestTopology const a(100, 200, 3);
    TestTopology const b(50, 80, 4);
    {
        GpuForceTopology ta(a.numValues, a.termOffset, a.termValue,
                            a.termDGdr);
        BOOST_REQUIRE_EQUAL(ta.getMemoryUsage(),
                            GpuForceTopology::memoryUsage(
                                a.numAtoms, a.numValues, a.termValue.size()));
        {
            GpuForceTopology tb(b.numValues, b.termOffset, b.termValue,
                                b.termDGdr);
            BOOST_REQUIRE_EQUAL(GpuForceTopology::getTotalMemoryUsage(),
                                before + ta.getMemoryUsage()
                                + tb.getMemoryUsage());
        }
        BOOST_REQUIRE_EQUAL(GpuForceTopology::getTotalMemoryUsage(),
                            before + ta.getMemoryUsage());
    }
    BOOST_REQUIRE_EQUAL(GpuForceTopology::getTotalMemoryUsage(), before);
}

BOOST_AUTO_TEST_CASE(Constructor_InconsistentInput_Throws)
{
    size_t const before = GpuForceTopology::getTotalMemoryUsage();
    TestTopology t(10, 20, 6);
    t.termOffset.back() += 1;
    BOOST_REQUIRE_THROW(GpuForceTopology(t.numValues, t.termOffset,
                                         t.termValue, t.termDGdr),
                        runtime_error);
    BOOST_REQUIRE_EQUAL(GpuForceTopology::getTotalMemoryUsage(), before);
}

BOOST_AUTO_TEST_CASE(Apply_WrongNumberOfWeightVectors_Throws)
{
    TestTopology const t(10, 20, 7);
    GpuForceTopology topology(t.numValues, t.termOffset, t.termValue,
                              t.termDGdr);
    vector<double> weights(4 * t.numValues, 1.0);
    vector<double> result(4 * 3 * t.numAtoms);
    BOOST_REQUIRE_THROW(topology.apply(0, weights.data(), result.data()),
                        runtime_error);
    BOOST_REQUIRE_THROW(topology.apply(4, weights.data(), result.data()),
                        runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
