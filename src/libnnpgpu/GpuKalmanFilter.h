// n2p2 - A neural network potential package
//
// Phase 6 follow-up (GPU_PORTING_PLAN.md): the third and last of this
// port's real call sites, after Mode::calculateAtomicNeuralNetworks()
// (nnp-predict) and Training::update()'s two Jacobian branches
// (nnp-train's energy/force weight derivatives). This one accelerates
// KalmanFilter::update() itself (src/libnnptrain/KalmanFilter.cpp:126-197),
// KT_STANDARD only -- the only mode this project's production input.nn
// actually uses (same "port what's used" scoping as every earlier phase).
//
// ONLY the two dominant O(N^2*m) steps are ported here: X = P.H and the
// K.X^T term of the covariance downdate P -= K.X^T (both cuBLAS dgemm,
// ported from gpu/gemm/kalman_gemm_test.cu). Everything else -- A = H^T.X
// + R, the m x m inverse, K = X.Ainv, w += K.xi -- stays on the HOST, using
// the exact same Eigen expressions KalmanFilter::update()'s original code
// already uses (see KalmanFilter.cpp's dispatch). This is a deliberate
// correction from an earlier version of this file, which also ported the
// m x m inverse to a hand-rolled GPU Gauss-Jordan kernel: that step is
// O(N*m^2), under 1% of the flops, so porting it bought no real speedup,
// and it turned out NOT to be numerically interchangeable with Eigen's
// LU-based .inverse() on real (correlated, sometimes ill-conditioned)
// production Jacobian data -- caught by a real end-to-end nnp-train run
// diverging by ~10 orders of magnitude after one epoch, even though a
// synthetic-random-data standalone test (which never produces an
// ill-conditioned m x m matrix by construction) passed cleanly. Keeping
// the small linear-algebra steps on the host, byte-for-byte identical to
// the proven CPU path, removes that entire class of risk for free.
//
// Unlike GpuNeuralNetwork's functions (stateless, one call = one complete
// computation), this one needs PERSISTENT device-resident state: P is
// ~89 MB at the real production size (N=3327), and KalmanFilter::update()
// is called far more often per epoch than Mode::calculateAtomicNeuralNetworks()
// or Training::update()'s Jacobian branches -- re-uploading/downloading all
// of P on every single call would very plausibly erase the GEMM speedup
// entirely (the underlying win here is only 1.22x-1.39x to begin with, see
// gemm/kalman_gemm_test.cu). So P (and X, transiently, between the two
// calls below) live on the GPU for the lifetime of one KalmanFilter
// object; only H (uploaded), X (downloaded once, after computeX), and K
// (uploaded once, before updateP) cross the PCIe bus, all small relative
// to P.
//
// This header has no CUDA-specific types (an opaque forward-declared
// struct pointer plus plain doubles/ints) so it can be included from
// ordinary C++ translation units (KalmanFilter.cpp) without needing nvcc.

#ifndef GPU_KALMAN_FILTER_H
#define GPU_KALMAN_FILTER_H

namespace nnp
{

struct GpuKalmanFilterState;

/** Create persistent GPU-resident Kalman filter state and upload the
 *  initial error covariance matrix P.
 *
 * @param[in] N State vector size (number of weights).
 * @param[in] P0 Initial P, N x N. Only ever used symmetric in practice (P
 *            starts as a multiple of the identity and KalmanFilter::update()
 *            keeps it that way analytically), so row-major vs column-major
 *            doesn't matter -- same reasoning gemm/kalman_gemm_test.cu's
 *            header comment already relies on.
 * @return Opaque state handle, owned by the caller; free with
 *         gpuKalmanDestroy().
 */
GpuKalmanFilterState* gpuKalmanCreate(int N, double const* P0);

/** X = P . H (the first of the two dominant O(N^2*m) steps), using the
 *  persistent device-resident P. Uploads H, computes on the GPU, downloads
 *  the result to the caller's host buffer -- the caller then does A =
 *  H^T.X + R, the m x m inverse, and K = X.Ainv on the host with Eigen
 *  (see KalmanFilter.cpp), exactly as the original CPU-only code always
 *  did, before calling gpuKalmanUpdateP() below with the resulting K.
 *
 * @param[in]  state GPU state from gpuKalmanCreate().
 * @param[in]  m Observation vector size for THIS call (may vary call to
 *             call, e.g. a partial final batch -- internal scratch buffers
 *             grow on demand, mirroring Eigen's own automatic resize in
 *             the original code).
 * @param[in]  H Jacobian, N x m, column-major (matches
 *             KalmanFilter::setJacobian()'s documented layout, i.e.
 *             Eigen::Map<MatrixXd const>'s natural storage).
 * @param[out] X Result, N x m, column-major, into a caller-owned buffer
 *             (e.g. KalmanFilter::X's own storage).
 */
void gpuKalmanComputeX(GpuKalmanFilterState* state, int m,
                       double const* H, double* X);

/** P -= K . X^T (the second of the two dominant O(N^2*m) steps), then
 *  P += q*I -- updates the persistent device-resident P in place. Reuses
 *  the device-resident X already sitting there from the immediately
 *  preceding gpuKalmanComputeX() call (same m), no re-upload needed; only
 *  K is uploaded.
 *
 * @param[in] state GPU state, with X from the matching gpuKalmanComputeX()
 *            call for this same update step.
 * @param[in] m Observation vector size (must match the preceding
 *            gpuKalmanComputeX() call).
 * @param[in] K Kalman gain matrix, N x m, column-major (the caller's
 *            K = X.Ainv result, computed on the host).
 * @param[in] q Process noise to add to P's diagonal after the downdate.
 */
void gpuKalmanUpdateP(GpuKalmanFilterState* state, int m,
                      double const* K, double q);

/** Download the current device-resident P (N x N) to a caller-owned host
 *  buffer -- needed only when host code (KalmanFilter::status()) reads P
 *  for diagnostics; P otherwise never leaves the device between update()
 *  calls.
 */
void gpuKalmanGetP(GpuKalmanFilterState* state, double* P);

/** Free GPU state created by gpuKalmanCreate(). */
void gpuKalmanDestroy(GpuKalmanFilterState* state);

}

#endif
