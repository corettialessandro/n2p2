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

#ifndef GPUFORCEASSEMBLY_H
#define GPUFORCEASSEMBLY_H

// Assembly of forces from symmetry function derivatives on the GPU. Plain
// C++ interface, included from code compiled without nvcc.

#include <cstddef> // std::size_t
#include <vector>  // std::vector

namespace nnp
{

/** Force assembly topology of one structure on the GPU.
 *
 * Holds, for each atom i, the terms (index of a symmetry function value,
 * dG/dr_i) of all symmetry functions depending on the position of atom i:
 * its own and those of its neighbors. Applied to weight vectors w (one entry
 * per symmetry function value of the structure) it gives the sum over the
 * terms of w * dG/dr_i for every atom, e.g. -F_i for w = dE/dG. The topology
 * only depends on the geometry, so it is kept with its structure and reused
 * for new weights.
 *
 * The sums are deterministic: the terms of each atom are always added in
 * the same order.
 */
class GpuForceTopology
{
public:
    /// Maximum number of weight vectors per apply() call.
    static std::size_t const maxWeightVectors = 3;

    /** Copy a topology to the current device.
     *
     * @param[in] numValues Number of symmetry function values (length of a
     *                      weight vector).
     * @param[in] termOffset Start of the terms of each atom (number of
     *                       atoms + 1 entries).
     * @param[in] termValue Symmetry function value index of each term.
     * @param[in] termDGdr dG/dr of each term (3 per term).
     *
     * Throws std::runtime_error for inconsistent input or if the memory
     * cannot be allocated.
     */
    GpuForceTopology(std::size_t                numValues,
                     std::vector<int> const&    termOffset,
                     std::vector<int> const&    termValue,
                     std::vector<double> const& termDGdr);
    /** Free all device memory.
     */
    ~GpuForceTopology();
    GpuForceTopology(GpuForceTopology const&) = delete;
    GpuForceTopology& operator=(GpuForceTopology const&) = delete;
    /** Apply the topology to weight vectors.
     *
     * @param[in] numWeightVectors Number of weight vectors (at most
     *                             maxWeightVectors).
     * @param[in] weights Weight vectors, one after the other (numValues
     *                    each).
     * @param[out] result For each weight vector the sums for all atoms, one
     *                    after the other (3 per atom).
     */
    void               apply(std::size_t   numWeightVectors,
                             double const* weights,
                             double*       result);
    /** Device memory used by this topology in bytes.
     */
    std::size_t        getMemoryUsage() const;
    /** Device memory a topology with the given size would use in bytes.
     *
     * @param[in] numAtoms Number of atoms.
     * @param[in] numValues Number of symmetry function values.
     * @param[in] numTerms Number of terms.
     */
    static std::size_t memoryUsage(std::size_t numAtoms,
                                   std::size_t numValues,
                                   std::size_t numTerms);
    /** Device memory used by all topologies of this process in bytes.
     */
    static std::size_t getTotalMemoryUsage();

private:
    struct Data;
    Data* data;
};

}

#endif
