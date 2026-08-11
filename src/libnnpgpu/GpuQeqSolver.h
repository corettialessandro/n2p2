// n2p2 - A neural network potential package
//
// GPU port of Structure::AConstrainedQr (Eigen::ColPivHouseholderQR),
// the dense (numAtoms+1)x(numAtoms+1) charge-equilibration solve family
// that -- after `memorize_symfunc_results` eliminated stage 1's
// symmetry-function recompute cost -- became 44.3% of stage-1 epoch time
// (see gpu/README.md). Backs six call sites in Structure.cpp:
// calculateElectrostaticEnergy() (the factorize + first solve),
// calculateDQdChi(), calculateDQdJ(), calculateDQdr(),
// calculateForceLambdaTotal(), calculateForceLambdaElec() -- the first
// three are stage-1 (charge training), the last three stage-2 (force
// training), so this port benefits both.
//
// Numerical safety was checked against real production data BEFORE this
// was written, not assumed from theory -- the same discipline that would
// have caught GpuKalmanFilter's ~10-order-of-magnitude divergence (a
// small m x m Eigen .inverse() that was NOT numerically interchangeable
// with a hand-rolled GPU inverse on real, sometimes ill-conditioned,
// Jacobian data). Here: real AConstrained/bConstrained dumped from
// temp/H2O_4G (60 real 631x631 structures) gave cond ~ 3.27e5, and a
// direct QR-vs-LU solve comparison (Eigen ColPivHouseholderQR vs
// PartialPivLU, then cusolverDnDgetrf/getrs itself via
// gpu/gemm/qeq_solver_test.cu) agreed to ~1e-15 relative error, worst
// case ~7.6e-14 -- float64 machine precision, no measurable accuracy
// loss from switching to LU.
//
// Unlike GpuForces.h's topology (purely geometric, cached forever),
// AConstrained depends on the current per-element hardness (a trainable
// weight) as well as geometry, so it CANNOT be cached across calls --
// gpuQeqFactorize() must be called (and refactorizes unconditionally)
// every time Structure::calculateElectrostaticEnergy() runs, exactly
// like the CPU code's unconditional AConstrainedQr.compute() today. What
// IS persistent, keyed by structureId (Structure::index) like
// GpuForces.h/GpuElecForces.h, is the device buffer allocation itself
// (avoiding repeat cudaMalloc/cudaFree churn across calls for the same
// structure) and the LU factors, reused by every gpuQeqSolve() call
// until the next gpuQeqFactorize() for that same structureId.
//
// Batching is NOT optional for calculateDQdChi()/calculateDQdJ() --
// those currently call .solve() numAtoms/numElements times in a loop;
// doing that as hundreds of separate small GPU round trips would very
// plausibly repeat GpuForces.cu's documented first-pass MPS-contention
// regression (many small calls, each paying full cudaMemcpy latency,
// measured to make things WORSE until batched -- see gpu/README.md).
// gpuQeqSolve()'s nrhs parameter supports exactly this: callers should
// build one (n x nrhs) matrix and issue ONE call, not nrhs separate
// ones. calculateDQdr()/calculateForceLambdaTotal()/calculateForceLambdaElec()
// are called with nrhs=1 from their actual Training.cpp/Mode.cpp call
// sites -- GpuElecForces already validated that small unbatched per-call
// GPU round trips are an acceptable, real (if modest) net win in this
// exact codebase, so nrhs=1 calls here are not preemptively
// over-engineered into a batch that doesn't exist at the call site.
//
// This header has no CUDA-specific types (plain doubles/ints/size_t) so
// it can be included from ordinary C++ translation units (Structure.cpp)
// without needing nvcc.

#ifndef GPU_QEQ_SOLVER_H
#define GPU_QEQ_SOLVER_H

#include <cstddef>

namespace nnp
{

/** Factorize AConstrained (n x n, column-major -- Eigen::MatrixXd's
 *  native layout, no transpose needed) via cusolverDnDgetrf, into
 *  persistent per-structureId device buffers (reused/grown on demand
 *  across calls for the same structureId, not reallocated every time).
 *  MUST be called once per Structure::calculateElectrostaticEnergy()
 *  call, before any gpuQeqSolve() call for that structureId -- unlike
 *  GpuForces.h's topology cache, this factorization is NOT reused across
 *  calculateElectrostaticEnergy() calls (AConstrained changes: hardness
 *  is a trainable weight), only within the window until the next
 *  gpuQeqFactorize() for the same structureId.
 *
 * @param[in] structureId Stable identifier for the structure (e.g.
 *            Structure::index) -- used as the persistent-state cache
 *            key.
 * @param[in] n Matrix dimension (numAtoms + 1, including the Lagrange
 *            total-charge-constraint row/column).
 * @param[in] AConstrained Flat n x n matrix, column-major.
 *
 * @throw std::runtime_error if cusolverDnDgetrf reports a singular or
 *        numerically exact-zero-pivot matrix (info != 0) -- should not
 *        happen for real, well-conditioned production data (see this
 *        file's header comment), but checked rather than assumed.
 */
void gpuQeqFactorize(std::size_t structureId, int n,
                      double const* AConstrained);

/** Solve using the LU factors from the most recent gpuQeqFactorize()
 *  call for this structureId (cusolverDnDgetrs). Supports nrhs > 1 for
 *  a single batched multi-right-hand-side solve -- callers with several
 *  right-hand sides against the same factorization (calculateDQdChi(),
 *  calculateDQdJ()) MUST batch into one call rather than looping
 *  nrhs=1 calls (see this file's header comment on why).
 *
 * @param[in]  structureId Stable identifier, must match a preceding
 *             gpuQeqFactorize() call.
 * @param[in]  n Matrix dimension, must match the preceding
 *             gpuQeqFactorize() call for this structureId.
 * @param[in]  nrhs Number of right-hand sides (columns of B/X).
 * @param[in]  B Flat n x nrhs matrix, column-major.
 * @param[out] X Flat n x nrhs matrix, column-major, into a caller-owned
 *             buffer.
 *
 * @throw std::runtime_error if no matching gpuQeqFactorize() call was
 *        made for this structureId, or if cusolverDnDgetrs reports an
 *        error.
 */
void gpuQeqSolve(std::size_t structureId, int n, int nrhs,
                  double const* B, double* X);

}

#endif
