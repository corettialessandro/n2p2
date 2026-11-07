#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GpuEwaldOperator
#include <boost/test/unit_test.hpp>

#include "GpuEwaldOperator.h"
#include <algorithm> // std::max
#include <cmath>     // std::cos, std::sin, std::fabs
#include <cstddef>   // std::size_t
#include <stdexcept> // std::runtime_error
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

// A structure of numGlobal atoms split into two "processes" (atoms with even
// and odd index), with tables of numK reciprocal vectors, a diagonal and
// sparse real-space rows; the matrix rows of each part are applied on the GPU
// and compared with the formula evaluated on the CPU.
struct Split
{
    size_t                 numGlobal = 37;
    size_t                 numK = 23;
    vector<double>         position;
    vector<double>         kvector;
    vector<double>         kcoeff;
    vector<double>         diagonal;
    // Real-space terms of all atoms (row i: columns and values).
    vector<vector<int>>    column;
    vector<vector<double>> value;
    vector<double>         v;

    Split()
    {
        for (size_t i = 0; i < numGlobal; ++i)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                position.push_back(10.0 * (0.5 + 0.45 * sin(1.7 * i
                                                            + 2.3 * c)));
            }
            diagonal.push_back(1.0 + 0.1 * (i % 5));
            v.push_back(0.3 * sin(0.9 * i) - 0.05);
            column.emplace_back();
            value.emplace_back();
            for (size_t j = 0; j < numGlobal; ++j)
            {
                if (j == i || (i + 2 * j) % 7 != 0) continue;
                column.back().push_back(j);
                value.back().push_back(0.2 / (1.0 + (i + j) % 11));
            }
        }
        for (size_t k = 0; k < numK; ++k)
        {
            kvector.push_back(0.3 * (k % 3 + 1));
            kvector.push_back(-0.2 * (k % 4));
            kvector.push_back(0.1 * (k % 5) + 0.05);
            kcoeff.push_back(0.01 / (1.0 + k));
        }
    }

    double phase(size_t k, size_t i) const
    {
        return kvector[3 * k + 0] * position[3 * i + 0]
             + kvector[3 * k + 1] * position[3 * i + 1]
             + kvector[3 * k + 2] * position[3 * i + 2];
    }

    // Matrix rows of the atoms of one part on a GPU operator, tables with
    // row stride `stride`.
    void setup(GpuEwaldOperator& op, vector<size_t> const& atoms,
               size_t stride, vector<double>& cosTable,
               vector<double>& sinTable) const
    {
        size_t const n = atoms.size();
        cosTable.assign(numK * stride, 99.0);
        sinTable.assign(numK * stride, 99.0);
        vector<int> localIndex;
        vector<double> diag;
        vector<int> rowOffset(1, 0);
        vector<int> col;
        vector<double> val;
        for (size_t l = 0; l < n; ++l)
        {
            size_t const i = atoms[l];
            for (size_t k = 0; k < numK; ++k)
            {
                cosTable[k * stride + l] = cos(phase(k, i));
                sinTable[k * stride + l] = sin(phase(k, i));
            }
            localIndex.push_back(i);
            diag.push_back(diagonal[i]);
            col.insert(col.end(), column[i].begin(), column[i].end());
            val.insert(val.end(), value[i].begin(), value[i].end());
            rowOffset.push_back(col.size());
        }
        op.setup(numK, n, numGlobal, cosTable.data(), sinTable.data(),
                 stride, kcoeff.data(), localIndex, diag, rowOffset, col,
                 val);
    }

    // (A v)_i on the CPU.
    double reference(size_t i) const
    {
        double r = diagonal[i] * v[i];
        for (size_t t = 0; t < column[i].size(); ++t)
        {
            r += value[i][t] * v[column[i][t]];
        }
        for (size_t k = 0; k < numK; ++k)
        {
            double sr = 0.0;
            double si = 0.0;
            for (size_t j = 0; j < numGlobal; ++j)
            {
                sr += v[j] * cos(phase(k, j));
                si += v[j] * sin(phase(k, j));
            }
            r += 2.0 * kcoeff[k] * (sr * cos(phase(k, i))
                                    + si * sin(phase(k, i)));
        }
        return r;
    }
};

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Apply_TwoParts_MatchesCpuFormula)
{
    Split const sys;
    vector<size_t> part[2];
    for (size_t i = 0; i < sys.numGlobal; ++i) part[i % 2].push_back(i);
    GpuEwaldOperator op[2];
    vector<double> cosTable[2];
    vector<double> sinTable[2];
    for (int p = 0; p < 2; ++p)
    {
        // Row stride larger than the number of local atoms, as in LAMMPS.
        sys.setup(op[p], part[p], part[p].size() + 5, cosTable[p],
                  sinTable[p]);
    }
    vector<double> s(2 * sys.numK, 0.0);
    for (int p = 0; p < 2; ++p)
    {
        vector<double> sp(2 * sys.numK);
        op[p].structureFactors(sys.v.data(), sp.data());
        for (size_t k = 0; k < 2 * sys.numK; ++k) s[k] += sp[k];
    }
    double scale = 0.0;
    for (size_t i = 0; i < sys.numGlobal; ++i)
    {
        scale = max(scale, fabs(sys.reference(i)));
    }
    for (int p = 0; p < 2; ++p)
    {
        vector<double> result(part[p].size());
        op[p].apply(s.data(), result.data());
        for (size_t l = 0; l < part[p].size(); ++l)
        {
            BOOST_REQUIRE_SMALL(result[l] - sys.reference(part[p][l]),
                                1e-12 * scale);
        }
    }
}

BOOST_AUTO_TEST_CASE(Apply_RepeatedSetup_ReusesMemory)
{
    // A second, smaller setup on the same object (memory kept) gives the
    // same result as a fresh object.
    Split const sys;
    vector<size_t> all;
    vector<size_t> few;
    for (size_t i = 0; i < sys.numGlobal; ++i)
    {
        all.push_back(i);
        if (i % 3 == 0) few.push_back(i);
    }
    GpuEwaldOperator reused;
    GpuEwaldOperator fresh;
    vector<double> ct;
    vector<double> st;
    sys.setup(reused, all, all.size(), ct, st);
    sys.setup(reused, few, few.size(), ct, st);
    sys.setup(fresh, few, few.size(), ct, st);
    vector<double> s1(2 * sys.numK);
    vector<double> s2(2 * sys.numK);
    reused.structureFactors(sys.v.data(), s1.data());
    fresh.structureFactors(sys.v.data(), s2.data());
    BOOST_REQUIRE(s1 == s2);
    vector<double> r1(few.size());
    vector<double> r2(few.size());
    reused.apply(s1.data(), r1.data());
    fresh.apply(s2.data(), r2.data());
    BOOST_REQUIRE(r1 == r2);
}

BOOST_AUTO_TEST_CASE(Setup_NoLocalAtoms_ZeroStructureFactors)
{
    Split const sys;
    GpuEwaldOperator op;
    vector<double> ct;
    vector<double> st;
    sys.setup(op, vector<size_t>(), 1, ct, st);
    vector<double> s(2 * sys.numK, 1.0);
    op.structureFactors(sys.v.data(), s.data());
    for (double x : s) BOOST_REQUIRE_EQUAL(x, 0.0);
    op.apply(s.data(), nullptr);
}

BOOST_AUTO_TEST_CASE(Setup_InconsistentInput_Throws)
{
    GpuEwaldOperator op;
    vector<double> table(4, 0.0);
    double const kcoeff[2] = {1.0, 1.0};
    // Neighbor index beyond the structure.
    BOOST_REQUIRE_THROW(op.setup(2, 1, 3, table.data(), table.data(), 2,
                                 kcoeff, {0}, {1.0}, {0, 1}, {3}, {0.5}),
                        runtime_error);
    // Row offsets inconsistent with the terms.
    BOOST_REQUIRE_THROW(op.setup(2, 1, 3, table.data(), table.data(), 2,
                                 kcoeff, {0}, {1.0}, {0, 2}, {1}, {0.5}),
                        runtime_error);
    // Use before a successful setup.
    vector<double> s(4);
    BOOST_REQUIRE_THROW(op.structureFactors(table.data(), s.data()),
                        runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
