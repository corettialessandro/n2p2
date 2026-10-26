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

#ifndef GPUCHARGEEQUILIBRATION_H
#define GPUCHARGEEQUILIBRATION_H

// Direct solution of the 4G-HDNNP charge equilibration on the GPU. Plain C++
// interface, included from code compiled without nvcc.

#include <cstddef> // std::size_t
#include <vector>  // std::vector

namespace nnp
{

/** Charge equilibration matrix of a whole structure on the GPU.
 *
 * Builds and factorizes the matrix of the linear system
 * @f[
 *   \begin{pmatrix} A & 1 \\ 1^T & 0 \end{pmatrix}
 *   \begin{pmatrix} x \\ \mu \end{pmatrix} =
 *   \begin{pmatrix} b \\ c \end{pmatrix}
 * @f]
 * with the Coulomb matrix of Gaussian charges A (4 pi epsilon = 1):
 * - periodic (Ewald summation with width eta): diagonal
 *   J_i + 1/(sigma_i sqrt(pi)) - 2/(eta sqrt(2 pi)), real-space terms
 *   [erfc(r/(sqrt(2) eta)) - erfc(r/gamma_ij)]/r summed over all periodic
 *   images closer than the real-space cutoff, and reciprocal-space terms
 *   2 sum_k c_k cos(k (r_i - r_j));
 * - non-periodic: diagonal J_i + 1/(sigma_i sqrt(pi)), off-diagonal
 *   erf(r/gamma_ij)/r;
 * where gamma_ij = sqrt(2 (sigma_i^2 + sigma_j^2)). The charges follow with
 * b = -chi and c = total charge, the Lagrange multipliers of the 4G forces
 * with b = -dE/dQ and c = 0, both from the same factorization.
 *
 * Alternatively, factorizes a complete matrix of the linear system built
 * elsewhere (Structure::calculateElectrostaticEnergy()).
 */
class GpuChargeEquilibration
{
public:
    GpuChargeEquilibration();
    /** Free all device memory.
     */
    ~GpuChargeEquilibration();
    GpuChargeEquilibration(GpuChargeEquilibration const&) = delete;
    GpuChargeEquilibration& operator=(GpuChargeEquilibration const&) = delete;
    /** Build and factorize the matrix of a structure.
     *
     * @param[in] type Type index of each atom (0, 1, ...).
     * @param[in] position Positions (3 per atom).
     * @param[in] hardness Hardness J of each type.
     * @param[in] sigma Gaussian width of each type.
     * @param[in] periodic Periodic (orthorhombic) structure.
     * @param[in] box Box lengths (periodic only, 3 values).
     * @param[in] eta Ewald width (periodic only).
     * @param[in] realCutoff Real-space cutoff (periodic only).
     * @param[in] kvector Reciprocal vectors (periodic only, 3 per vector).
     * @param[in] kcoeff Coefficient c_k of each reciprocal vector.
     *
     * Throws std::runtime_error for inconsistent input, if the memory cannot
     * be allocated or if the matrix is singular.
     */
    void        factorize(std::vector<int> const&    type,
                          std::vector<double> const& position,
                          std::vector<double> const& hardness,
                          std::vector<double> const& sigma,
                          bool                       periodic,
                          double const*              box,
                          double                     eta,
                          double                     realCutoff,
                          std::vector<double> const& kvector,
                          std::vector<double> const& kcoeff);
    /** Factorize a complete matrix of the linear system.
     *
     * @param[in] numAtoms Number of atoms N.
     * @param[in] matrix Matrix including the constraint row and column
     *                   ((N + 1) x (N + 1), column-major).
     *
     * Throws std::runtime_error if the memory cannot be allocated or if the
     * matrix is singular.
     */
    void        factorize(std::size_t numAtoms, double const* matrix);
    /** Solve with the last factorization.
     *
     * @param[in] b Right-hand side (one value per atom).
     * @param[in] c Constraint value (sum of the solution).
     * @param[out] x Solution (one value per atom).
     */
    void        solve(std::vector<double> const& b,
                      double                     c,
                      std::vector<double>&       x);
    /** Solve for several right-hand sides with the last factorization.
     *
     * @param[in] numRhs Number of right-hand sides.
     * @param[in,out] rhs Right-hand sides (b, c) as columns of length N + 1
     *                    (column-major), overwritten with the solutions
     *                    (x, mu).
     */
    void        solve(std::size_t numRhs, double* rhs);
    /** Number of atoms of the last factorization (0 if none).
     */
    std::size_t getNumAtoms() const;
    /** Identifies the last factorization, unique among all objects (0 if
     *  none).
     */
    std::size_t getFactorizationId() const;
    /** Device memory needed for a structure.
     *
     * @param[in] numAtoms Number of atoms.
     * @param[in] numK Number of reciprocal vectors.
     * @param[in] numRhs Number of right-hand sides solved at once.
     *
     * @return Bytes.
     */
    static std::size_t memoryUsage(std::size_t numAtoms,
                                   std::size_t numK,
                                   std::size_t numRhs = 1);

private:
    struct Data;
    Data* data;

    /** LU factorization of the matrix on the device.
     */
    void        factorizeDeviceMatrix(std::size_t numAtoms);
};

}

#endif
