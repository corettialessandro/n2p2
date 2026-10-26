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

#ifndef GPUSYMMETRYFUNCTIONS_H
#define GPUSYMMETRYFUNCTIONS_H

// Symmetry functions on the GPU. Plain C++ interface, included from code
// compiled without nvcc.

#include <cstddef> // std::size_t
#include <vector>  // std::vector

namespace nnp
{

/** Parameters of one symmetry function, as given to the GPU.
 *
 * Copies of the values of the corresponding SymFnc object (see there for
 * the meaning). Element indices refer to the ElementMap.
 */
struct GpuSymmetryFunction
{
    /// Symmetry function type (see isSupported()).
    int              type;
    /// Index among the symmetry functions of the central element.
    int              index;
    /// %Element of the (first) neighbor.
    int              e1;
    /// %Element of the second neighbor (angular types only).
    int              e2;
    /// Cutoff radius.
    double           rc;
    /// Cutoff function type (CutoffFunction::CutoffType).
    int              cutoffType;
    /// Inner cutoff parameter of the cutoff function.
    double           cutoffAlpha;
    double           eta;
    double           rs;
    /// Angular types only.
    double           lambda;
    /// Angular types only.
    double           zeta;
    /// Angular types only.
    bool             useIntegerPow;
    /// Angular types only.
    int              zetaInt;
    /// Scaling type (SymFnc::ScalingType).
    int              scalingType;
    double           Smin;
    double           Gmin;
    double           Gmean;
    double           scalingFactor;
    /// Index of this symmetry function among the derivatives stored for a
    /// neighbor of each element, -1 if it does not depend on that element.
    std::vector<int> indexPerElement;
};

/** Symmetry functions of all elements of a model on the GPU.
 *
 * Holds the parameters; the results for a structure are kept in a
 * GpuSymmetryFunctionData object.
 */
class GpuSymmetryFunctionSet
{
public:
    /** Check if a symmetry function type has a GPU implementation.
     *
     * @param[in] type Symmetry function type.
     *
     * @return True for types 2 (radial), 3 (narrow angular) and 9 (wide
     *         angular).
     */
    static bool isSupported(int type);
    /// Maximum number of elements of a model.
    static std::size_t const maxElements = 32;
    /** Copy the parameters of all symmetry functions to the current device.
     *
     * @param[in] symmetryFunctions Symmetry functions of each element.
     *
     * Throws std::runtime_error for unsupported types, more than maxElements
     * elements or inconsistent input.
     */
    GpuSymmetryFunctionSet(
        std::vector<std::vector<GpuSymmetryFunction>> const&
        symmetryFunctions);
    /** Free all device memory.
     */
    ~GpuSymmetryFunctionSet();
    GpuSymmetryFunctionSet(GpuSymmetryFunctionSet const&) = delete;
    GpuSymmetryFunctionSet& operator=(GpuSymmetryFunctionSet const&) = delete;

private:
    struct Data;
    Data* data;

    friend class GpuSymmetryFunctionData;
};

/** Symmetry functions and their derivatives for one structure on the GPU.
 *
 * calculate() computes the (scaled) symmetry function values of all atoms,
 * which are returned to the host, and their derivatives with respect to the
 * positions of the atoms and their neighbors, which stay on the GPU.
 * apply() contracts the derivatives with weight vectors (e.g. dE/dG for
 * forces).
 *
 * Positions are identified by targets: each atom and each neighbor entry has
 * a target index (e.g. the index of the atom in the structure, or the index
 * of the local or ghost atom in LAMMPS). Periodic images of an atom share its
 * target.
 *
 * All results are deterministic: sums are always taken in the same order.
 */
class GpuSymmetryFunctionData
{
public:
    /// Maximum number of weight vectors per apply() call.
    static std::size_t const maxWeightVectors = 3;
    /// Maximum number of neighbors of an atom.
    static std::size_t const maxNeighbors = 2048;

    GpuSymmetryFunctionData();
    /** Free all device memory.
     */
    ~GpuSymmetryFunctionData();
    GpuSymmetryFunctionData(GpuSymmetryFunctionData const&) = delete;
    GpuSymmetryFunctionData& operator=(GpuSymmetryFunctionData const&)
        = delete;
    /** Calculate symmetry functions (and derivatives) of a structure.
     *
     * @param[in] set Symmetry functions of the model.
     * @param[in] atomElement %Element of each atom.
     * @param[in] atomTarget Target index of each atom.
     * @param[in] neighborOffset Start of the neighbors of each atom (number
     *                           of atoms + 1 entries).
     * @param[in] neighborElement %Element of each neighbor.
     * @param[in] neighborTarget Target index of each neighbor.
     * @param[in] neighborGeometry Distance and distance vector (4 per
     *                             neighbor: d, dx, dy, dz).
     * @param[in] derivatives Also calculate derivatives (for apply()).
     * @param[out] values Symmetry function values of all atoms, one atom
     *                    after the other.
     *
     * Throws std::runtime_error for inconsistent input, more than
     * maxNeighbors neighbors of an atom, or if memory cannot be allocated.
     * Device memory is kept and reused by later calls.
     */
    void        calculate(GpuSymmetryFunctionSet const& set,
                          std::vector<int> const&       atomElement,
                          std::vector<int> const&       atomTarget,
                          std::vector<int> const&       neighborOffset,
                          std::vector<int> const&       neighborElement,
                          std::vector<int> const&       neighborTarget,
                          std::vector<double> const&    neighborGeometry,
                          bool                          derivatives,
                          double*                       values);
    /** Contract the derivatives with weight vectors.
     *
     * @param[in] set Symmetry functions of the model (same as for
     *                calculate()).
     * @param[in] numWeightVectors Number of weight vectors (at most
     *                             maxWeightVectors).
     * @param[in] weights Weight vectors w, one after the other (one entry
     *                    per symmetry function value, same order as the
     *                    values of calculate()).
     * @param[out] result For each weight vector and each target i the sum
     *                    of w * dG/dr_i over all symmetry function values,
     *                    3 per target (getNumTargets() targets), one weight
     *                    vector after the other.
     *
     * Requires derivatives from the last calculate() call. Throws
     * std::runtime_error otherwise.
     */
    void        apply(GpuSymmetryFunctionSet const& set,
                      std::size_t                   numWeightVectors,
                      double const*                 weights,
                      double*                       result);
    /** Derivatives of all values with respect to one coordinate of one
     *  target.
     *
     * @param[in] set Symmetry functions of the model (same as for
     *                calculate()).
     * @param[in] target Target index.
     * @param[in] component Coordinate (0, 1 or 2).
     * @param[out] result dG/dr_target of all symmetry function values
     *                    (getNumValues(), same order as the values of
     *                    calculate()).
     *
     * Requires derivatives from the last calculate() call. Throws
     * std::runtime_error otherwise.
     */
    void        derivativesOfTarget(GpuSymmetryFunctionSet const& set,
                                    std::size_t                   target,
                                    std::size_t                   component,
                                    double*                       result);
    /** Number of targets of the last calculation (largest target index + 1).
     */
    std::size_t getNumTargets() const;
    /** Number of symmetry function values of the last calculation.
     */
    std::size_t getNumValues() const;
    /** Device memory used in bytes.
     */
    std::size_t getMemoryUsage() const;

private:
    struct Data;
    Data* data;
};

}

#endif
