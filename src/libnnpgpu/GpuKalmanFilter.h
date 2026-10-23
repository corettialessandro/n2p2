// n2p2 - A neural network potential package
// Copyright (C) 2026 Alessandro Coretti (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef GPUKALMANFILTER_H
#define GPUKALMANFILTER_H

// Error covariance matrix of KalmanFilter on the GPU. Plain C++ interface,
// included from code compiled without nvcc.

#include <cstddef> // std::size_t

namespace nnp
{

/** GPU part of KalmanFilter::update(): keeps the error covariance matrix P
 * on the device and runs the two operations scaling with sizeState^2, X = P . H
 * and P = P - K . X^T. The small remaining steps stay in KalmanFilter.
 *
 * All matrices are column-major, as Eigen stores them.
 */
class GpuKalmanFilter
{
public:
    /** Check the device and allocate P.
     *
     * @param[in] sizeState Size of the state vector (P is sizeState^2).
     *
     * Throws std::runtime_error if no usable device is available (see
     * gpuRequireDevice()) or the memory cannot be allocated.
     */
    explicit GpuKalmanFilter(std::size_t sizeState);
    /** Free all device memory.
     */
    ~GpuKalmanFilter();
    GpuKalmanFilter(GpuKalmanFilter const&) = delete;
    GpuKalmanFilter& operator=(GpuKalmanFilter const&) = delete;
    /** Copy P from the host to the device.
     *
     * @param[in] P Matrix of size sizeState x sizeState.
     */
    void        setP(double const* P);
    /** Copy P from the device to the host.
     *
     * @param[out] P Matrix of size sizeState x sizeState.
     */
    void        getP(double* P) const;
    /** Calculate X = P . H, using only the lower triangle of P.
     *
     * @param[in] sizeObservation Number of columns of H and X.
     * @param[in] H Matrix of size sizeState x sizeObservation.
     * @param[out] X Matrix of size sizeState x sizeObservation.
     *
     * X is also kept on the device for the next updateP().
     */
    void        computeX(std::size_t   sizeObservation,
                         double const* H,
                         double*       X);
    /** Update P = scale * (P - K . X^T) + q * I.
     *
     * @param[in] sizeObservation Number of columns of K, as in the
     *                            preceding computeX().
     * @param[in] K Matrix of size sizeState x sizeObservation.
     * @param[in] scale Factor applied after the downdate.
     * @param[in] q Value added to the diagonal.
     *
     * Uses X of the preceding computeX().
     */
    void        updateP(std::size_t   sizeObservation,
                        double const* K,
                        double        scale,
                        double        q);
    /** Index of the device holding P.
     */
    int         getDevice() const;

private:
    struct Data;
    Data* data;

    /// Make sure the buffers for H, X and K hold sizeObservation columns.
    void        reserve(std::size_t sizeObservation);
    /// Free all device memory and the cuBLAS handle.
    void        release();
};

}

#endif
