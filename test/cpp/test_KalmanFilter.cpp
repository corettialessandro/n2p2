#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE KalmanFilter
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "KalmanFilter.h"
#include "tolerance.h"
#include "utility.h"
#ifdef N2P2_GPU
#include "gpuTest.h"
#endif
#include <algorithm> // std::max
#include <cmath>     // std::exp, std::fabs, std::sin
#include <cstddef>   // std::size_t
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_ZeroSizeState_Throws)
{
    BOOST_REQUIRE_THROW(KalmanFilter(0, KalmanFilter::KT_STANDARD),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(Constructor_UnknownType_Throws)
{
    BOOST_REQUIRE_THROW(
        KalmanFilter(1, static_cast<KalmanFilter::KalmanType>(2)),
        runtime_error);
}

BOOST_AUTO_TEST_CASE(Standard_Update_CorrectValues)
{
    // Scalar case (sizeState = sizeObservation = 1): reference values
    // obtained from an independent Python re-implementation of the exact
    // update formula (see KalmanFilter::update(), KT_STANDARD branch),
    // replicating the same operation order/precision.
    KalmanFilter kf(1, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(2.0, 0.01, 0.1, 0.001, 1.0, 0.1, 10.0);

    vector<double> state = {1.0};
    double const error = 0.4;
    vector<double> const jacobian = {0.5};

    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());

    kf.update();
    BOOST_REQUIRE_SMALL(state.at(0) - 1.0971027059695633, accuracy);

    // Second consecutive update: also exercises the learning-rate increase
    // and process-noise annealing recurrences between calls.
    kf.update();
    BOOST_REQUIRE_SMALL(state.at(0) - 1.193618839027742, accuracy);

    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 2);
}

BOOST_AUTO_TEST_CASE(FadingMemory_Update_CorrectValues)
{
    // Same scalar-case approach as Standard_Update_CorrectValues, for the
    // KT_FADINGMEMORY branch.
    KalmanFilter kf(1, KalmanFilter::KT_FADINGMEMORY);
    kf.setParametersFadingMemory(2.0, 0.01, 0.1, 0.001, 0.99, 0.9);

    vector<double> state = {1.0};
    double const error = 0.4;
    vector<double> const jacobian = {0.5};

    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());

    kf.update();

    BOOST_REQUIRE_SMALL(state.at(0) - 1.0896860986547086, accuracy);
    BOOST_REQUIRE_SMALL(kf.getLambda() - 0.991, accuracy);
    BOOST_REQUIRE_SMALL(kf.getGamma() - 0.5022601707684581, accuracy);
}

BOOST_AUTO_TEST_CASE(Standard_InfoAndStatus_ReportCorrectType)
{
    KalmanFilter kf(2, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(1.0, 0.01, 0.1, 0.001, 1.0, 0.1, 10.0);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const jacobian = {0.3, -0.2};
    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());
    kf.update();

    vector<string> const info = kf.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("KT_STANDARD") != string::npos);

    string const s = kf.status(3);
    BOOST_REQUIRE(s.find("3") != string::npos);

    vector<string> const header = kf.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_CASE(FadingMemory_InfoAndStatus_ReportCorrectType)
{
    KalmanFilter kf(2, KalmanFilter::KT_FADINGMEMORY);
    kf.setParametersFadingMemory(1.0, 0.01, 0.1, 0.001, 0.99, 0.9);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const jacobian = {0.3, -0.2};
    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());
    kf.update();

    vector<string> const info = kf.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("KT_FADINGMEMORY") != string::npos);

    string const s = kf.status(1);
    BOOST_REQUIRE(s.find("1") != string::npos);

    vector<string> const header = kf.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_CASE(Getters_ReflectSetParameters)
{
    KalmanFilter kf(2, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(2.0, 0.02, 0.05, 0.001, 0.5, 0.2, 5.0);

    BOOST_REQUIRE(kf.getType() == KalmanFilter::KT_STANDARD);
    BOOST_REQUIRE_SMALL(kf.getEpsilon() - 2.0, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQ0() - 0.02, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQtau() - 0.05, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQmin() - 0.001, accuracy);
    BOOST_REQUIRE_SMALL(kf.getEta() - 0.5, accuracy);
    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 0);

    kf.setSizeObservation(3);
    BOOST_REQUIRE_EQUAL(kf.getSizeObservation(), 3);
}


// Independent reference implementation of KalmanFilter::update() with plain
// loops and a dense, symmetric covariance matrix P (no Eigen), for a state of
// size n and an observation of size m. Jacobian layout as in KalmanFilter:
// column-major n x m, H(i, j) = jacobian[j * n + i].
struct ReferenceKalman
{
    size_t                 n;
    bool                   fading;
    double                 q, qtau, qmin, eta, etatau, etamax, lambda, nu;
    vector<vector<double>> P;

    ReferenceKalman(size_t n, bool fading, double epsilon) :
        n(n), fading(fading), q(0), qtau(0), qmin(0), eta(0), etatau(0),
        etamax(0), lambda(0), nu(0),
        P(n, vector<double>(n, 0.0))
    {
        for (size_t i = 0; i < n; ++i) P[i][i] = 1.0 / epsilon;
    }

    void update(vector<double>&       w,
                vector<double> const& xi,
                vector<double> const& jacobian)
    {
        size_t const m = xi.size();
        auto H = [&](size_t i, size_t j) { return jacobian[j * n + i]; };
        // X = P . H (n x m), A = H^T . X + R (m x m).
        vector<vector<double>> X(n, vector<double>(m, 0.0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                for (size_t k = 0; k < n; ++k) X[i][j] += P[i][k] * H(k, j);
        if (!fading && eta < etamax) eta *= exp(etatau);
        vector<vector<double>> A(m, vector<double>(m, 0.0));
        for (size_t a = 0; a < m; ++a)
        {
            for (size_t b = 0; b < m; ++b)
                for (size_t k = 0; k < n; ++k) A[a][b] += H(k, a) * X[k][b];
            A[a][a] += fading ? lambda : 1.0 / eta;
        }
        // Ainv by Gauss-Jordan elimination with partial pivoting.
        vector<vector<double>> Ainv(m, vector<double>(m, 0.0));
        for (size_t a = 0; a < m; ++a) Ainv[a][a] = 1.0;
        for (size_t c = 0; c < m; ++c)
        {
            size_t piv = c;
            for (size_t r = c + 1; r < m; ++r)
                if (fabs(A[r][c]) > fabs(A[piv][c])) piv = r;
            swap(A[c], A[piv]);
            swap(Ainv[c], Ainv[piv]);
            double const d = A[c][c];
            for (size_t k = 0; k < m; ++k) { A[c][k] /= d; Ainv[c][k] /= d; }
            for (size_t r = 0; r < m; ++r)
            {
                if (r == c) continue;
                double const f = A[r][c];
                for (size_t k = 0; k < m; ++k)
                {
                    A[r][k] -= f * A[c][k];
                    Ainv[r][k] -= f * Ainv[c][k];
                }
            }
        }
        // K = X . Ainv, P = P - K . X^T (then forgetting factor), P += q I.
        vector<vector<double>> K(n, vector<double>(m, 0.0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                for (size_t k = 0; k < m; ++k) K[i][j] += X[i][k] * Ainv[k][j];
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                for (size_t k = 0; k < m; ++k) P[i][j] -= K[i][k] * X[j][k];
                if (fading) P[i][j] /= lambda;
            }
        for (size_t i = 0; i < n; ++i) P[i][i] += q;
        // w = w + K . xi
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < m; ++k) w[i] += K[i][k] * xi[k];
        if (q > qmin) q *= exp(-qtau);
        if (fading) lambda = nu * lambda + 1.0 - nu;
    }
};

// Deterministic, irregular test inputs for update number u.
void fillInputs(size_t                u,
                vector<double>&       xi,
                vector<double>&       jacobian)
{
    for (size_t j = 0; j < xi.size(); ++j)
        xi[j] = 0.3 * sin(1.7 * (j + 1) + 0.9 * u);
    for (size_t k = 0; k < jacobian.size(); ++k)
        jacobian[k] = sin(0.37 * (k + 1) * (u + 2)) + 0.1 * (k % 3);
}

// Multi-dimensional updates (state 7, observation 3, three consecutive
// updates) against the reference above. Unlike the scalar tests, these see
// errors in matrix layout, transposes and in the use of the lower triangle
// of P (selfadjointView), all of which vanish at size 1.
BOOST_DATA_TEST_CASE(Update_MultiDimensional_MatchesReference,
                     bdata::make({false, true}),
                     fading)
{
    size_t const n = 7;
    size_t const m = 3;
    KalmanFilter kf(n, fading ? KalmanFilter::KT_FADINGMEMORY
                              : KalmanFilter::KT_STANDARD);
    ReferenceKalman ref(n, fading, 0.5);
    ref.q = 0.02;
    ref.qtau = 0.1;
    ref.qmin = 0.001;
    if (fading)
    {
        kf.setParametersFadingMemory(0.5, 0.02, 0.1, 0.001, 0.98, 0.9);
        ref.lambda = 0.98;
        ref.nu = 0.9;
    }
    else
    {
        kf.setParametersStandard(0.5, 0.02, 0.1, 0.001, 0.8, 0.1, 10.0);
        ref.eta = 0.8;
        ref.etatau = 0.1;
        ref.etamax = 10.0;
    }

    vector<double> state(n), stateRef(n);
    for (size_t i = 0; i < n; ++i) state[i] = stateRef[i] = 0.1 * i - 0.2;
    vector<double> xi(m);
    vector<double> jacobian(n * m);
    kf.setState(state.data());
    kf.setSizeObservation(m);
    for (size_t u = 0; u < 3; ++u)
    {
        fillInputs(u, xi, jacobian);
        kf.setError(xi.data(), m);
        kf.setJacobian(jacobian.data(), m);
        kf.update(m);
        ref.update(stateRef, xi, jacobian);
        for (size_t i = 0; i < n; ++i)
        {
            BOOST_TEST_INFO("update " << u << ", state component " << i);
            BOOST_REQUIRE_SMALL(state[i] - stateRef[i],
                                1.0E-12 * max(1.0, fabs(stateRef[i])));
        }
    }
    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 3);
}

#ifndef N2P2_GPU
BOOST_AUTO_TEST_CASE(SetGpuEnabled_CpuBuild_OnlyOffAllowed)
{
    KalmanFilter kf(3, KalmanFilter::KT_STANDARD);
    BOOST_REQUIRE_NO_THROW(kf.setGpuEnabled(false));
    BOOST_REQUIRE_THROW(kf.setGpuEnabled(true), runtime_error);
}
#endif

BOOST_AUTO_TEST_SUITE_END()

#ifdef N2P2_GPU
// Tests of the GPU implementation of the covariance update against the CPU
// implementation, in the same binary.

// Kalman filter of state size n with test parameters and its own state.
struct TestFilter
{
    KalmanFilter   kf;
    vector<double> state;

    TestFilter(size_t n, bool fading, bool gpu) :
        kf(n, fading ? KalmanFilter::KT_FADINGMEMORY
                     : KalmanFilter::KT_STANDARD),
        state(n)
    {
        kf.setGpuEnabled(gpu);
        if (fading)
        {
            kf.setParametersFadingMemory(0.5, 0.02, 0.1, 0.001, 0.98, 0.9);
        }
        else
        {
            kf.setParametersStandard(0.5, 0.02, 0.1, 0.001, 0.8, 0.1, 10.0);
        }
        for (size_t i = 0; i < n; ++i) state[i] = 0.01 * i - 0.2;
        kf.setState(state.data());
    }

    void update(vector<double> const& xi, vector<double> const& jacobian)
    {
        kf.setError(xi.data(), xi.size());
        kf.setJacobian(jacobian.data(), xi.size());
        kf.update(xi.size());
    }
};

// Compares states and the P and K statistics of status() (Pasym, a
// round-off quantity, excluded; status() prints 9 significant digits).
void requireSameFilter(TestFilter& gpu, TestFilter& cpu)
{
    Tolerance const tolerance = buildTolerance(Tolerance(0.0));
    Tolerance const tolerancePrinted = buildTolerance(Tolerance(0.0),
                                                      gpuRelativePrinted);
    for (size_t i = 0; i < cpu.state.size(); ++i)
    {
        BOOST_TEST_INFO("state component " << i);
        REQUIRE_CLOSE_TO_REFERENCE(cpu.state[i], gpu.state[i], tolerance);
    }
    vector<string> const statusGpu = split(reduce(gpu.kf.status(0)));
    vector<string> const statusCpu = split(reduce(cpu.kf.status(0)));
    BOOST_REQUIRE_EQUAL(statusGpu.at(1), statusCpu.at(1));
    for (size_t column : {2, 3, 5})
    {
        BOOST_TEST_INFO("status column " << column);
        REQUIRE_CLOSE_TO_REFERENCE(stod(statusCpu.at(column)),
                                   stod(statusGpu.at(column)),
                                   tolerancePrinted);
    }
}

BOOST_AUTO_TEST_SUITE(GpuTests)

BOOST_DATA_TEST_CASE(Update_Gpu_MatchesCpu, bdata::make({false, true}), fading)
{
    size_t const n = 300;
    size_t const m = 8;
    TestFilter gpu(n, fading, true);
    TestFilter cpu(n, fading, false);
    vector<double> xi(m);
    vector<double> jacobian(n * m);
    for (size_t u = 0; u < 5; ++u)
    {
        BOOST_TEST_INFO("update " << u);
        UnitTests::fillInputs(u, xi, jacobian);
        gpu.update(xi, jacobian);
        cpu.update(xi, jacobian);
        requireSameFilter(gpu, cpu);
    }
}

BOOST_AUTO_TEST_CASE(Update_ObservationSizeChanges_MatchesCpu)
{
    // The device buffers for H, X and K grow with the observation size.
    size_t const n = 50;
    TestFilter gpu(n, false, true);
    TestFilter cpu(n, false, false);
    size_t u = 0;
    for (size_t m : {2, 9, 1, 12})
    {
        BOOST_TEST_INFO("observation size " << m);
        vector<double> xi(m);
        vector<double> jacobian(n * m);
        UnitTests::fillInputs(u++, xi, jacobian);
        gpu.update(xi, jacobian);
        cpu.update(xi, jacobian);
        requireSameFilter(gpu, cpu);
    }
}

BOOST_AUTO_TEST_CASE(SetGpuEnabled_SwitchDuringTraining_KeepsCovariance)
{
    // P moves between host and device when switching: GPU, CPU, GPU again
    // must continue exactly like a filter that stays on the CPU.
    size_t const n = 40;
    size_t const m = 3;
    TestFilter switching(n, true, true);
    TestFilter cpu(n, true, false);
    vector<double> xi(m);
    vector<double> jacobian(n * m);
    for (size_t u = 0; u < 6; ++u)
    {
        BOOST_TEST_INFO("update " << u);
        if (u == 2) switching.kf.setGpuEnabled(false);
        if (u == 4) switching.kf.setGpuEnabled(true);
        UnitTests::fillInputs(u, xi, jacobian);
        switching.update(xi, jacobian);
        cpu.update(xi, jacobian);
        requireSameFilter(switching, cpu);
    }
}

BOOST_AUTO_TEST_CASE(Info_GpuOrCpu_Reported)
{
    KalmanFilter kf(4, KalmanFilter::KT_STANDARD);
    vector<string> info = kf.info();
    BOOST_REQUIRE(info.back().find("Covariance matrix P on GPU")
                  != string::npos);
    kf.setGpuEnabled(false);
    info = kf.info();
    BOOST_REQUIRE(info.back().find("Covariance matrix P on CPU")
                  != string::npos);
}

BOOST_AUTO_TEST_CASE(Constructor_NoVisibleDevice_Throws)
{
    // The device check belongs to the setup of the filter.
    BOOST_REQUIRE_EQUAL(runWithoutGpu("WithoutGpu"), 0);
}

BOOST_AUTO_TEST_SUITE_END()

// Run only by runWithoutGpu() above, in a process without visible devices.
BOOST_AUTO_TEST_SUITE(WithoutGpu, *boost::unit_test::disabled())

BOOST_AUTO_TEST_CASE(Constructor_NoVisibleDevice_Throws)
{
    BOOST_REQUIRE_THROW(KalmanFilter(10, KalmanFilter::KT_STANDARD),
                        runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
#endif
