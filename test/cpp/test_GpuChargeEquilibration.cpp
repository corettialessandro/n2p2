#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GpuChargeEquilibration
#include <boost/test/unit_test.hpp>

#include "GpuChargeEquilibration.h"
#include "tolerance.h"
#include <Eigen/Dense> // MatrixXd, VectorXd, PartialPivLU
#include <cmath>       // std::sqrt, std::erf, std::erfc, std::exp, std::sin
#include <cstddef>     // std::size_t
#include <stdexcept>   // std::runtime_error
#include <vector>      // std::vector

using namespace std;
using namespace nnp;

// Small test system: two types, irregular positions in an orthorhombic box
// whose shortest side is below the real-space cutoff, so that several
// periodic images contribute.
struct TestSystem
{
    vector<int>    type;
    vector<double> position;
    vector<double> hardness = {0.9, 0.4};
    vector<double> sigma = {0.7, 1.3};
    double         box[3] = {7.0, 8.5, 9.5};
    double         eta = 1.6;
    double         realCutoff = 8.0;
    vector<double> kvector;
    vector<double> kcoeff;

    TestSystem(size_t numAtoms)
    {
        for (size_t i = 0; i < numAtoms; ++i)
        {
            type.push_back(i % 3 == 0 ? 0 : 1);
            for (size_t c = 0; c < 3; ++c)
            {
                position.push_back(box[c] * (0.5 + 0.45 * sin(1.7 * i
                                                              + 2.3 * c)));
            }
        }
        double const pi = 3.14159265358979323846;
        double const volume = box[0] * box[1] * box[2];
        for (int kx = 0; kx <= 3; ++kx)
        {
            for (int ky = -3; ky <= 3; ++ky)
            {
                for (int kz = -3; kz <= 3; ++kz)
                {
                    if (kx == 0 && (ky < 0 || (ky == 0 && kz <= 0))) continue;
                    double const k[3] = {2 * pi * kx / box[0],
                                         2 * pi * ky / box[1],
                                         2 * pi * kz / box[2]};
                    double const k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                    kvector.insert(kvector.end(), k, k + 3);
                    kcoeff.push_back(4 * pi / volume
                                     * exp(-0.5 * eta * eta * k2) / k2);
                }
            }
        }
    }

    // The documented matrix [A 1; 1^T 0] on the CPU.
    Eigen::MatrixXd matrix(bool periodic) const
    {
        double const pi = 3.14159265358979323846;
        size_t const n = type.size();
        Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n + 1, n + 1);
        for (size_t i = 0; i < n; ++i)
        {
            int const ei = type.at(i);
            for (size_t j = 0; j < n; ++j)
            {
                int const ej = type.at(j);
                double const gamma = sqrt(2.0 * (sigma.at(ei) * sigma.at(ei)
                                                 + sigma.at(ej) * sigma.at(ej)));
                double d[3];
                for (size_t c = 0; c < 3; ++c)
                {
                    d[c] = position.at(3 * i + c) - position.at(3 * j + c);
                }
                if (!periodic)
                {
                    if (i == j)
                    {
                        a(i, i) = hardness.at(ei)
                                + 1.0 / (sigma.at(ei) * sqrt(pi));
                    }
                    else
                    {
                        double const r = sqrt(d[0] * d[0] + d[1] * d[1]
                                              + d[2] * d[2]);
                        a(i, j) = erf(r / gamma) / r;
                    }
                    continue;
                }
                double sum = 0.0;
                for (int nx = -3; nx <= 3; ++nx)
                {
                    for (int ny = -3; ny <= 3; ++ny)
                    {
                        for (int nz = -3; nz <= 3; ++nz)
                        {
                            double const x = d[0] + nx * box[0];
                            double const y = d[1] + ny * box[1];
                            double const z = d[2] + nz * box[2];
                            double const r = sqrt(x * x + y * y + z * z);
                            if (r >= realCutoff || r == 0.0) continue;
                            sum += (erfc(r / (sqrt(2.0) * eta))
                                    - erfc(r / gamma)) / r;
                        }
                    }
                }
                for (size_t k = 0; k < kcoeff.size(); ++k)
                {
                    sum += 2.0 * kcoeff.at(k)
                         * cos(kvector.at(3 * k + 0) * d[0]
                               + kvector.at(3 * k + 1) * d[1]
                               + kvector.at(3 * k + 2) * d[2]);
                }
                if (i == j)
                {
                    sum += hardness.at(ei) + 1.0 / (sigma.at(ei) * sqrt(pi))
                         - 2.0 / (eta * sqrt(2.0 * pi));
                }
                a(i, j) = sum;
            }
            a(i, n) = 1.0;
            a(n, i) = 1.0;
        }
        return a;
    }
};

void requireSolutionsMatch(bool periodic)
{
    TestSystem const s(37);
    size_t const n = s.type.size();
    GpuChargeEquilibration q;
    q.factorize(s.type, s.position, s.hardness, s.sigma, periodic, s.box,
                s.eta, s.realCutoff, s.kvector, s.kcoeff);
    BOOST_REQUIRE_EQUAL(q.getNumAtoms(), n);
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(s.matrix(periodic));
    // Two right-hand sides with the same factorization (charges, Lagrange
    // multipliers).
    for (double c : {0.3, 0.0})
    {
        vector<double> b(n);
        Eigen::VectorXd rhs(n + 1);
        for (size_t i = 0; i < n; ++i)
        {
            b.at(i) = cos(0.9 * i + 10 * c);
            rhs(i) = b.at(i);
        }
        rhs(n) = c;
        Eigen::VectorXd const reference = lu.solve(rhs);
        vector<double> x;
        q.solve(b, c, x);
        BOOST_REQUIRE_EQUAL(x.size(), n);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            BOOST_TEST_INFO("atom " << i << " constraint " << c);
            REQUIRE_CLOSE_TO_REFERENCE(reference(i), x.at(i),
                                       Tolerance(1.0E-10, 1.0E-10));
            sum += x.at(i);
        }
        REQUIRE_CLOSE_TO_REFERENCE(c, sum, Tolerance(1.0E-10));
    }
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Periodic_MatchesCpu)
{
    requireSolutionsMatch(true);
}

BOOST_AUTO_TEST_CASE(NonPeriodic_MatchesCpu)
{
    requireSolutionsMatch(false);
}

BOOST_AUTO_TEST_CASE(FactorizeMatrix_SeveralRhs_MatchesCpu)
{
    TestSystem const s(41);
    size_t const n = s.type.size();
    Eigen::MatrixXd const a = s.matrix(true);
    GpuChargeEquilibration q;
    BOOST_REQUIRE_EQUAL(q.getFactorizationId(), 0);
    q.factorize(n, a.data());
    BOOST_REQUIRE_EQUAL(q.getNumAtoms(), n);
    size_t const id = q.getFactorizationId();
    BOOST_REQUIRE_NE(id, 0);
    Eigen::MatrixXd b(n + 1, 3);
    for (size_t i = 0; i <= n; ++i)
    {
        for (size_t k = 0; k < 3; ++k) b(i, k) = cos(0.7 * i + 1.9 * k);
    }
    Eigen::MatrixXd const reference = Eigen::PartialPivLU<Eigen::MatrixXd>(a)
                                      .solve(b);
    Eigen::MatrixXd x = b;
    q.solve(3, x.data());
    for (size_t k = 0; k < 3; ++k)
    {
        for (size_t i = 0; i <= n; ++i)
        {
            BOOST_TEST_INFO("row " << i << " right-hand side " << k);
            REQUIRE_CLOSE_TO_REFERENCE(reference(i, k), x(i, k),
                                       Tolerance(1.0E-10, 1.0E-10));
        }
    }
    // Each factorization has a new identifier, also in other objects.
    GpuChargeEquilibration other;
    other.factorize(n, a.data());
    BOOST_REQUIRE_NE(other.getFactorizationId(), id);
    q.factorize(n, a.data());
    BOOST_REQUIRE_NE(q.getFactorizationId(), id);
    BOOST_REQUIRE_NE(q.getFactorizationId(), other.getFactorizationId());
}

BOOST_AUTO_TEST_CASE(InconsistentInput_Throws)
{
    TestSystem s(5);
    GpuChargeEquilibration q;
    vector<double> x;
    BOOST_REQUIRE_THROW(q.solve(vector<double>(5, 0.0), 0.0, x),
                        runtime_error);
    s.type.at(2) = 7;
    BOOST_REQUIRE_THROW(q.factorize(s.type, s.position, s.hardness, s.sigma,
                                    true, s.box, s.eta, s.realCutoff,
                                    s.kvector, s.kcoeff),
                        runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
