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

#ifndef GPUEWALDOPERATOR_H
#define GPUEWALDOPERATOR_H

// Charge equilibration matrix (Ewald summation) applied to vectors on the
// GPU, for the iterative solvers of a domain-decomposed structure. Plain C++
// interface, included from code compiled without nvcc.

#include <cstddef> // std::size_t
#include <vector>  // std::vector

namespace nnp
{

/** Rows of the charge equilibration matrix of the local atoms of one MPI
 *  process on its GPU (periodic structure, Ewald summation).
 *
 * The matrix A of GpuChargeEquilibration (periodic case) is applied to a
 * vector v of all atoms of the structure in two steps, between which the
 * caller sums the structure factors over all processes:
 * 1. structureFactors(): S_k = sum_i v_i exp(i k r_i) over the local atoms
 *    (real parts, then imaginary parts);
 * 2. apply(): (A v)_i for the local atoms i, from the diagonal term, the
 *    real-space terms of the neighbors (any atom of the structure) and the
 *    reciprocal-space terms 2 sum_k c_k Re(S_k exp(-i k r_i)) with the summed
 *    structure factors.
 * The tables exp(i k r_i) of the local atoms, the coefficients c_k, the
 * diagonal and the real-space terms (sparse rows) are given once per
 * configuration by setup(). The device memory is kept and reused.
 */
class GpuEwaldOperator
{
public:
    GpuEwaldOperator();
    /** Free all device memory.
     */
    ~GpuEwaldOperator();
    GpuEwaldOperator(GpuEwaldOperator const&) = delete;
    GpuEwaldOperator& operator=(GpuEwaldOperator const&) = delete;
    /** Set the matrix rows of the local atoms for a configuration.
     *
     * @param[in] numK Number of reciprocal vectors.
     * @param[in] numLocal Number of local atoms.
     * @param[in] numGlobal Number of atoms of the structure (length of the
     *                      vectors v).
     * @param[in] cosTable cos(k r_i): numK rows of numLocal values.
     * @param[in] sinTable sin(k r_i), same layout.
     * @param[in] rowStride Distance between rows of the tables (values).
     * @param[in] kcoeff Coefficient c_k of each reciprocal vector.
     * @param[in] localIndex Index of each local atom in the vectors v.
     * @param[in] diagonal Diagonal term of each local atom.
     * @param[in] rowOffset Start of the real-space terms of each local atom
     *                      (numLocal + 1 entries).
     * @param[in] column Index in the vectors v of each real-space term.
     * @param[in] value Value of each real-space term.
     *
     * Throws std::runtime_error for inconsistent input or if the memory
     * cannot be allocated.
     */
    void        setup(std::size_t                numK,
                      std::size_t                numLocal,
                      std::size_t                numGlobal,
                      double const*              cosTable,
                      double const*              sinTable,
                      std::size_t                rowStride,
                      double const*              kcoeff,
                      std::vector<int> const&    localIndex,
                      std::vector<double> const& diagonal,
                      std::vector<int> const&    rowOffset,
                      std::vector<int> const&    column,
                      std::vector<double> const& value);
    /** Structure factors of the local atoms (step 1).
     *
     * @param[in] v Vector of all atoms (numGlobal values), kept for apply().
     * @param[out] s Real parts of S_k, then imaginary parts (2 numK values).
     */
    void        structureFactors(double const* v, double* s);
    /** Rows of the local atoms of A v (step 2), for the v of the last
     *  structureFactors() call.
     *
     * @param[in] s Structure factors summed over all processes (2 numK
     *              values, as from structureFactors()).
     * @param[out] result (A v)_i of the local atoms (numLocal values).
     */
    void        apply(double const* s, double* result);
    /** Device memory needed.
     *
     * @param[in] numK Number of reciprocal vectors.
     * @param[in] numLocal Number of local atoms.
     * @param[in] numGlobal Number of atoms of the structure.
     * @param[in] numTerms Number of real-space terms.
     *
     * @return Bytes.
     */
    static std::size_t memoryUsage(std::size_t numK,
                                   std::size_t numLocal,
                                   std::size_t numGlobal,
                                   std::size_t numTerms);

private:
    struct Data;
    Data* data;
};

}

#endif
