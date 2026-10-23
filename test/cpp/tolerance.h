#ifndef TOLERANCE_H
#define TOLERANCE_H

#include <boost/test/unit_test.hpp>
#include <algorithm> // std::max
#include <cmath>     // std::fabs
#include <iomanip>   // std::setprecision

// Tolerance for comparing a computed value against a stored reference value.
//
// A value passes if |value - reference| <= max(absolute, relative * |reference|).
// The relative part defaults to 0: the tests keep their historical, purely
// absolute tolerances.
//
// The reference values were recorded from CPU runs. GPU builds (compiled with
// -DN2P2_GPU, i.e. "make GPU=1") sum in a different order (atomic scatters,
// cuBLAS reductions) and differ in the last digits, so tests comparing against
// CPU references wrap their tolerance in buildTolerance(), which widens it on
// GPU builds only.
struct Tolerance
{
    double absolute;
    double relative;

    Tolerance(double absolute, double relative = 0.0) :
        absolute(absolute),
        relative(relative)
    {
    }

    double bound(double reference) const
    {
        return std::max(absolute, relative * std::fabs(reference));
    }
};

// GPU profile for quantities computed in a single evaluation and compared in
// full double precision (energies, forces, charges).
double const gpuRelativeEvaluation = 1.0E-10;
double const gpuAbsoluteEvaluation = 1.0E-12;

// GPU profile for quantities read back from text output. Most output files
// print 8 to 10 significant digits, so a last-digit difference in the
// computation can change the printed value by up to about 1E-8 relative.
double const gpuRelativePrinted = 1.0E-7;

/** Tolerance for the current build.
 *
 * @param[in] cpu Tolerance on CPU builds, returned unchanged there.
 * @param[in] relativeGpu Relative tolerance on GPU builds (the larger of this
 *                        and cpu.relative is used).
 * @param[in] absoluteGpu Absolute tolerance on GPU builds (the larger of this
 *                        and cpu.absolute is used).
 */
inline Tolerance buildTolerance(Tolerance const& cpu,
                                double relativeGpu = gpuRelativeEvaluation,
                                double absoluteGpu = gpuAbsoluteEvaluation)
{
#ifdef N2P2_GPU
    return Tolerance(std::max(cpu.absolute, absoluteGpu),
                     std::max(cpu.relative, relativeGpu));
#else
    (void)relativeGpu;
    (void)absoluteGpu;
    return cpu;
#endif
}

// Like BOOST_REQUIRE_SMALL(value - reference, ...), but with a Tolerance and
// a message stating reference, value, deviation and allowed bound.
#define REQUIRE_CLOSE_TO_REFERENCE(reference, value, tolerance)               \
    BOOST_REQUIRE_MESSAGE(                                                     \
        std::fabs((value) - (reference)) <= (tolerance).bound(reference),      \
        std::setprecision(17) << "deviation |" #value " - " #reference "| = " \
        << std::fabs((value) - (reference)) << " exceeds "                     \
        << (tolerance).bound(reference) << " (reference "                      \
        << (reference) << ", value " << (value) << ")")

#endif
