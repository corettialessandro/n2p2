// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
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

#ifndef MODE_H
#define MODE_H

#include "CutoffFunction.h"
#include "Element.h"
#include "ElementMap.h"
#include "ErfcBuf.h"
#include "EwaldSetup.h"
#include "Kspace.h"
#include "Log.h"
#include "ScreeningFunction.h"
#include "Settings.h"
#include "Structure.h"
#include "SymFnc.h"
#include <cstddef> // std::size_t
#include <map>     // std::map
#include <memory>  // std::shared_ptr
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

class GpuChargeEquilibration;
class GpuSymmetryFunctionSet;

/** Base class for all NNP applications.
 *
 * This top-level class is the anchor point for existing and future
 * applications. It contains functions to set up an existing neural network
 * potential and calculate energies and forces for configurations given as
 * Structure. A minimal setup requires some consecutive functions calls as this
 * minimal example shows:
 *
 * ```
 * Mode mode;
 * mode.initialize();
 * mode.loadSettingsFile();
 * mode.setupElementMap();
 * mode.setupElements();
 * mode.setupCutoff();
 * mode.setupSymmetryFunctions();
 * mode.setupSymmetryFunctionGroups();
 * mode.setupSymmetryFunctionStatistics(false, false, true, false);
 * mode.setupNeuralNetwork();
 * ```
 * To load weights and scaling information from files add these lines:
 * ```
 * mode.setupSymmetryFunctionScaling();
 * mode.setupNeuralNetworkWeights();
 * ```
 * The NNP is now ready! If we load a structure from a data file:
 * ```
 * Structure structure;
 * ifstream file;
 * file.open("input.data");
 * structure.setupElementMap(mode.elementMap);
 * structure.readFromFile(file);
 * file.close();
 * ```
 * we can finally predict the energy and forces from the neural network
 * potential:
 * ```
 * structure.calculateNeighborList(mode.getMaxCutoffRadius());
 * mode.calculateSymmetryFunctionGroups(structure, true);
 * mode.calculateAtomicNeuralNetworks(structure, true);
 * mode.calculateEnergy(structure);
 * mode.calculateForces(structure);
 * cout << structure.energy << '\n';
 * ```
 * The resulting potential energy is stored in Structure::energy, the forces
 * on individual atoms are located within the Structure::atoms vector in
 * Atom::f.
 */
class Mode
{
public:
    enum class NNPType
    {
        /// Short range NNP (2G-HDNNP).
        HDNNP_2G = 2,
        /// NNP with electrostatics and non-local charge transfer (4G-HDNNP).
        HDNNP_4G = 4,
        /** Short range NNP with charge NN, no electrostatics/Qeq (M. Bircher).
         *
         * This is a simplified version of a 4G-HDNNP. Two neural networks are
         * used: the first one predicts atomic charges, which will be used for
         * the second NN as additional input neuron. There is no electrostatic
         * energy and no global charge equilibration as in 4G-HDNNP.
         */
        HDNNP_Q = 10
    };

    /// Describes how multiple NNs in a committee should be used.
    enum class CommitteeMode
    {
        /// No committee functionality used, only single NN prediction.
        DISABLED,
        /// Only a single NN predicts, all toghether are used for validation.
        VALIDATION,
        /// All committee NNs are averaged for predicition.
        PREDICTION
    };

    Mode();
    /** Write welcome message with version information.
     */
    void                     initialize();
    /** Open settings file and load all keywords into memory.
     *
     * @param[in] fileName Settings file name.
     */
    void                     loadSettingsFile(std::string const& fileName
                                                                 = "input.nn");
    /** Switch off the committee (keywords committee_mode and
     *  committee_data) with a warning in the log.
     *
     * Committee training is not implemented: training uses only the HDNNP
     * in the main directory, each member is trained separately. Settings
     * files that also define the committee for prediction can thus be used
     * for training.
     */
    void                     disableCommittee();
    /** Check that all committee members use the data set normalization,
     *  atomic energy offsets and symmetry function scaling of the main
     *  settings.
     *
     * @param[in] directoryPrefix Directory of the weight files (the member
     *                            directories are in it).
     *
     * The committee uses input.nn and scaling.data of the main directory for
     * all members. If a member directory also contains input.nn or
     * scaling.data (e.g. from training the member), the keywords defining
     * the model (symmetry functions, neural network topology, scaling type),
     * the normalization keywords (mean_energy, conv_energy, conv_length,
     * conv_charge), the atomic energy offsets (atom_energy) and the scaling
     * data must agree, otherwise the members would be evaluated with
     * another model definition or combined in different units: throws if
     * they differ. Called by setupNeuralNetworkWeights() with a
     * committee.
     */
    void                     checkCommitteeMembers(
                                 std::string const& directoryPrefix);
    /** Combine multiple setup routines and provide a basic NNP setup.
     *
     * @param[in] nnpDir Optional directory where NNP files reside.
     * @param[in] skipNormalize Whether to skip normalization setup.
     * @param[in] initialHardness Signalizes to use initial hardness in NN
     *                  settings.
     *
     * Sets up elements, symmetry functions, symmetry function groups, neural
     * networks. No symmetry function scaling data is read, no weights are set.
     */
    void                     setupGeneric(std::string const& nnpDir = "",
                                          bool               skipNormalize = false,
                                          bool               initialHardness = false);
    /** Set up normalization.
     *
     * @param[in] standalone Whether to write section header and footer.
     *
     * If the keywords `mean_energy`, `conv_length` and
     * `conv_length` are present, the provided conversion factors are used to
     * internally use a different unit system.
     */
    void                     setupNormalization(bool standalone = true);
    /** Set up the element map.
     *
     * Uses keyword `elements`. This function should follow immediately after
     * settings are loaded via loadSettingsFile().
     */
    virtual void             setupElementMap();
    /** Set up all Element instances.
     *
     * Uses keywords `number_of_elements` and `atom_energy`. This function
     * should follow immediately after setupElementMap().
     */
    virtual void             setupElements();
    /** Set up cutoff function for all symmetry functions.
     *
     * Uses keyword `cutoff_type`. Cutoff parameters are read from settings
     * keywords and stored internally. As soon as setupSymmetryFunctions() is
     * called the settings are restored and used for all symmetry functions.
     * Thus, this function must be called before setupSymmetryFunctions().
     */
    void                     setupCutoff();
    /** Set up all symmetry functions.
     *
     * Uses keyword `symfunction_short`. Reads all symmetry functions from
     * settings and automatically assigns them to the correct element.
     */
    virtual void             setupSymmetryFunctions();
    /** Set up "empy" symmetry function scaling.
     *
     * Does not use any keywords. Sets no scaling for all symmetry functions.
     * Call after setupSymmetryFunctions(). Alternatively set scaling via
     * setupSymmetryFunctionScaling().
     */
    void                     setupSymmetryFunctionScalingNone();
    /** Set up symmetry function scaling from file.
     *
     * @param[in] fileName Scaling file name.
     *
     * Uses keywords `scale_symmetry_functions`, `center_symmetry_functions`,
     * `scale_symmetry_functions_sigma`, `scale_min_short` and
     * `scale_max_short`. Reads in scaling information and sets correct scaling
     * behavior for all symmetry functions. Call after
     * setupSymmetryFunctions().
     */
    virtual void             setupSymmetryFunctionScaling(
                                 std::string const& fileName = "scaling.data");
    /** Set up symmetry function groups.
     *
     * Does not use any keywords. Call after setupSymmetryFunctions() and
     * ensure that correct scaling behavior has already been set.
     */
    virtual void             setupSymmetryFunctionGroups();
#ifndef N2P2_NO_SF_CACHE
    /** Set up symmetry function cache.
     *
     * @param[in] verbose If true, print more cache information.
     *
     * Searches symmetry functions for identical cutoff functions or compact
     * function (i.e. all cachable stuff) and sets up a caching index.
     */
    virtual void             setupSymmetryFunctionCache(bool verbose = false);
#endif
    /** Extract required memory dimensions for symmetry function derivatives.
     *
     * @param[in] verbose If true, print all symmetry function lines.
     *
     * Call after symmetry functions have been set up and sorted.
     */
    void                     setupSymmetryFunctionMemory(bool verbose = false);
    /** Set up symmetry function statistics collection.
     *
     * @param[in] collectStatistics Whether statistics (min, max, mean, sigma)
     *                              is collected.
     * @param[in] collectExtrapolationWarnings Whether extrapolation warnings
     *                                         are logged.
     * @param[in] writeExtrapolationWarnings Write extrapolation warnings
     *                                       immediately when they occur.
     * @param[in] stopOnExtrapolationWarnings Throw error immediately when
     *                                        an extrapolation warning occurs.
     *
     * Does not use any keywords. Calling this setup function is not required,
     * by default no statistics collection is enabled (all arguments `false`).
     * Call after setupElements().
     */
    void                     setupSymmetryFunctionStatistics(
                                             bool collectStatistics,
                                             bool collectExtrapolationWarnings,
                                             bool writeExtrapolationWarnings,
                                             bool stopOnExtrapolationWarnings);
    /** Setup matrix storing all symmetry function cut-offs for each element.
     */
    void                     setupCutoffMatrix();
    /** Set up neural networks for all elements.
     *
     * Uses keywords `global_hidden_layers_short`, `global_nodes_short`,
     * `global_activation_short`, `normalize_nodes`. Call after
     * setupSymmetryFunctions(), only then the number of input layer neurons is
     * known.
     */
    virtual void             setupNeuralNetwork();
    /** Set up neural network weights from files with given name format.
     *
     * @param[in] fileNameFormats Map of NN ids to format for weight file
     *                            names. Must contain a placeholder "%zu" for
     *                            the element atomic number. Map keys are
     *                            "short", "elec". Map argument may be
     *                            ommitted, then default name formats are used.
     *
     * Does not use any keywords. The weight files should contain one weight
     * per line, see NeuralNetwork::setConnections() for the correct order.
     */
    virtual void             setupNeuralNetworkWeights(
                                 std::map<std::string,
                                          std::string> fileNameFormats =
                                 std::map<std::string,
                                          std::string>());
    /** Set up neural network weights from files with given name format.
     *
     * @param[in] directoryPrefix Directory prefix which is applied to all
     *                            fileNameFormats.
     * @param[in] fileNameFormats Map of NN ids to format for weight file
     *                            names. Must contain a placeholder "%zu" for
     *                            the element atomic number. Map keys are
     *                            "short", "elec". Map argument may be
     *                            ommitted, then default name formats are used.
     *
     * Does not use any keywords. The weight files should contain one weight
     * per line, see NeuralNetwork::setConnections() for the correct order.
     */
    virtual void             setupNeuralNetworkWeights(
                                 std::string           directoryPrefix,
                                 std::map<std::string,
                                          std::string> fileNameFormats =
                                 std::map<std::string,
                                          std::string>());
    /** Read in weights for a specific type of neural network only, skipping
     *  any others -- unlike setupNeuralNetworkWeights(), which loads every
     *  configured network kind unconditionally. Lets a caller load a
     *  subset (e.g. a 4G-HDNNP's "elec" network alone, without requiring
     *  "short" network weight files to exist on disk at all).
     *
     * @param[in] id Actual network type to initialize ("short" or "elec").
     * @param[in] fileNameFormat Weights file name format.
     * @param[in] fileDir Directory of the weight files (prefix of
     *                    fileNameFormat); the weights of committee member c
     *                    > 0 are read from fileDir + committee prefix + "-c/"
     *                    + fileNameFormat.
     */
    void                     readNeuralNetworkWeights(
                                 std::string const& id,
                                 std::string const& fileNameFormat,
                                 std::string const& fileDir = "");
    /** Switch evaluation of one neural network type on the GPU on or off.
     *
     * @param[in] id Neural network type ("short" or "elec").
     * @param[in] enabled Evaluate on the GPU if true.
     *
     * GPU builds switch it on when the weights of a network type are set up
     * (readNeuralNetworkWeights(), random initialization in training): this
     * is where the device check happens (see src/libnnpgpu/README.md).
     * Switching it off is internal, for tests comparing the GPU and CPU
     * code paths in one binary. Throws std::runtime_error when enabling the
     * GPU in a CPU build.
     */
    void                     setNeuralNetworkGpuEnabled(
                                 std::string const& id,
                                 bool               enabled);
    /** Switch the assembly of forces on the GPU on or off.
     *
     * @param[in] enabled Assemble forces on the GPU if true.
     *
     * Switched together with the GPU evaluation of the "short" neural
     * networks (see setNeuralNetworkGpuEnabled()), i.e. on in GPU builds
     * once their weights are set up. Switching it separately is internal,
     * for tests. Throws std::runtime_error when enabling the GPU in a CPU
     * build.
     *
     * The GPU assembles the forces of a structure from its second force
     * calculation with the same geometry on, if its topology fits into the
     * memory budget (see assembleForcesGpu()); otherwise the CPU does.
     */
    void                     setForceAssemblyGpuEnabled(bool enabled);
    /** Check if forces are assembled on the GPU.
     */
    bool                     isForceAssemblyGpuEnabled() const;
    /** Number of force calculations assembled on the GPU and on the CPU.
     *
     * @param[out] gpu Force calculations assembled on the GPU.
     * @param[out] cpu Force calculations assembled on the CPU.
     *
     * Counted since the GPU force assembly was enabled (GPU builds).
     */
    void                     getForceAssemblyCounts(std::size_t& gpu,
                                                    std::size_t& cpu) const;
    /** Switch the calculation of symmetry functions on the GPU on or off.
     *
     * @param[in] enabled Calculate symmetry functions on the GPU if true.
     *
     * Switched on while the "short" or the "elec" neural networks are
     * evaluated on the GPU (see setNeuralNetworkGpuEnabled()), i.e. in GPU
     * builds once the weights of either type are set up. Switching it separately is internal,
     * for tests. Throws std::runtime_error when enabling the GPU in a CPU
     * build.
     *
     * The GPU calculates symmetry functions where their derivatives are
     * used right away by forces assembled on the GPU: evaluateNNP(), the
     * LAMMPS interface and the error calculations and update candidates of
     * the training without memorized results (see
     * calculateSymmetryFunctionsGpu() and
     * Training::calculateSymmetryFunctionsTraining()). Other callers of
     * calculateSymmetryFunctions() and calculateSymmetryFunctionGroups()
     * (force update Jacobians, tools) get them from the CPU, with
     * derivatives on the host. If a symmetry function type of the model has no GPU
     * implementation, all symmetry functions are calculated on the CPU,
     * with a warning in the log.
     */
    void                     setSymmetryFunctionGpuEnabled(bool enabled);
    /** Check if symmetry functions are calculated on the GPU.
     *
     * @return False if switched off or if a symmetry function type of the
     *         model has no GPU implementation.
     */
    bool                     isSymmetryFunctionGpuEnabled() const;
    /** Switch the solution of the charge equilibration on the GPU on or off
     *  (4G-HDNNP).
     *
     * @param[in] enabled Solve on the GPU if true.
     *
     * Switched together with the GPU evaluation of the "elec" neural
     * networks (see setNeuralNetworkGpuEnabled()), i.e. on in GPU builds
     * once their weights are set up. Switching it separately is internal,
     * for tests. Throws std::runtime_error when enabling the GPU in a CPU
     * build.
     *
     * chargeEquilibration() builds the A matrix on the CPU; the GPU
     * factorizes it (LU instead of QR) and solves all linear systems with it
     * (Structure::solveA()), for structures up to a size given by the memory
     * budget (see getChargeEquilibrationGpuMaxAtoms()). Larger structures
     * are solved on the CPU.
     */
    void                     setChargeEquilibrationGpuEnabled(bool enabled);
    /** Largest structure whose charge equilibration is solved on the GPU.
     *
     * @return Number of atoms, 0 if switched off.
     *
     * The device memory of a structure with N atoms (matrix and N right-hand
     * sides, see GpuChargeEquilibration::memoryUsage()) must fit into a
     * quarter of the device memory per process (gpuDefaultCacheBudget() / 2).
     */
    std::size_t              getChargeEquilibrationGpuMaxAtoms() const;
    /** Read in atomic hardness of all elements from files (4G-HDNNP).
     *
     * @param[in] fileNameFormat Name format of file containing atomic
     *                           hardness data.
     *
     * Throws if a file is missing or does not contain exactly one value.
     */
    void                     readHardness(std::string const& fileNameFormat =
                                          "hardness.%03zu.data");
    /** Set up electrostatics related stuff (hardness, screening, ...).
     *
     * @param[in] initialHardness Use initial hardness from keyword in settings
     *                            file (useful for training).
     * @param[in] directoryPrefix Directory prefix which is applied to
     *                            fileNameFormat.
     * @param[in] fileNameFormat Name format of file containing atomic
     *                           hardness data.
     */
    virtual void             setupElectrostatics(bool initialHardness =
                                                 false,
                                                 std::string directoryPrefix =
                                                 "",
                                                 std::string fileNameFormat =
                                                 "hardness.%03zu.data");
    /** Calculate all symmetry functions for all atoms in given structure.
     *
     * @param[in] structure Input structure.
     * @param[in] derivatives If `true` calculate also derivatives of symmetry
     *                        functions.
     *
     * This function should be replaced by calculateSymmetryFunctionGroups()
     * whenever possible. Results are stored in Atom::G. If derivatives are
     * calculated, additional results are stored in Atom::dGdr and
     * Atom::Neighbor::dGdr.
     */
    void                     calculateSymmetryFunctions(
                                                       Structure& structure,
                                                       bool const derivatives);
    /** Calculate all symmetry function groups for all atoms in given
     * structure.
     *
     * @param[in] structure Input structure.
     * @param[in] derivatives If `true` calculate also derivatives of symmetry
     *                        functions.
     *
     * This function replaces calculateSymmetryFunctions() when symmetry
     * function groups are enabled (faster, default behavior). Results are
     * stored in Atom::G. If derivatives are calculated, additional results are
     * stored in Atom::dGdr and Atom::Neighbor::dGdr.
     */
    void                     calculateSymmetryFunctionGroups(
                                                       Structure& structure,
                                                       bool const derivatives);
    // /** Calculate a single atomic neural network for a given atom and nn type.
    // *
    // * @param[in] nnId Neural network identifier, e.g. "short", "charge".
    // * @param[in] atom Input atom.
    // * @param[in] derivatives If `true` calculate also derivatives of neural
    // *                        networks with respect to input layer neurons
    // *                        (required for force calculation).
    // *
    // * The atomic energy and charge is stored in Atom::energy and Atom::charge,
    // * respectively. If derivatives are calculated the results are stored in
    // * Atom::dEdG or Atom::dQdG.
    // */
    //void                     calculateAtomicNeuralNetwork(
    //                                           std::string const& nnId,
    //                                           Atom&              atom,
    //                                           bool const         derivatives);
    /** Calculate atomic neural networks for all atoms in given structure.
     *
     * @param[in] structure Input structure.
     * @param[in] derivatives If `true` calculate also derivatives of neural
     *                        networks with respect to input layer neurons
     *                        (required for force calculation).
     * @param[in] id Neural network ID to use. If empty, the first entry
     *                        nnk.front() is used.
     */
    void                     calculateAtomicNeuralNetworks(
                                           Structure&  structure,
                                           bool const  derivatives,
                                           std::string id = "");
    /** Perform global charge equilibration method.
     *
     * @param[in] structure Input structure.
     * @param[in] derivativesElec Turn on/off calculation of dElecdQ and
     *                          pElecpr (Typically needed for elecstrosttic
     *                          forces).
     */
    void                     chargeEquilibration(
                                                Structure& structure,
                                                bool const derivativesElec);
    /** Calculate potential energy for a given structure.
     *
     * @param[in] structure Input structure.
     *
     * Sum up potential energy from atomic energy contributions. Result is
     * stored in Structure::energy.
     */
    void                     calculateEnergy(Structure& structure) const;
    /** Calculate total charge for a given structure.
     *
     * @param[in] structure Input structure.
     *
     * Sum up charge from atomic charge contributions. Result is
     * stored in Structure::charge.
     */
    void                     calculateCharge(Structure& structure) const;
    /** Calculate forces for all atoms in given structure.
     *
     * @param[in] structure Input structure.
     *
     * Combine intermediate results from symmetry function and neural network
     * computation to atomic forces. Results are stored in Atom::f.
     */
    void                     calculateForces(Structure& structure) const;
    /** Evaluate neural network potential (includes total energy, optionally
     *  forces and in some cases charges.
     *  @param[in] structure Input structure.
     *  @param[in] useForces If true, calculate forces too.
     *  @param[in] useDEdG If true, calculate dE/dG too.
     *  @param[in] chargesOnly If true (4G-HDNNP only), stop after charge
     *             equilibration: skip the short-range NN forward pass,
     *             energy and forces entirely (implies useForces = false).
     *             Lets callers evaluate/compare atomic charges without the
     *             short-range NN's weights being loaded or trained at all
     *             (e.g. right after 4G-HDNNP stage-1-only training).
     */
    void                     evaluateNNP(Structure& structure,
                                         bool useForces = true,
                                         bool useDEdG = true,
                                         bool chargesOnly = false);
    /** Add atomic energy offsets to reference energy.
     *
     * @param[in] structure Input structure.
     * @param[in] ref If true, use reference energy, otherwise use NN energy.
     */
    void                     addEnergyOffset(Structure& structure,
                                             bool       ref = true);
    /** Remove atomic energy offsets from reference energy.
     *
     * @param[in] structure Input structure.
     * @param[in] ref If true, use reference energy, otherwise use NN energy.
     *
     * This function should be called immediately after structures are read in.
     */
    void                     removeEnergyOffset(Structure& structure,
                                                bool       ref = true);
    /** Get atomic energy offset for given structure.
     *
     * @param[in] structure Input structure.
     *
     * @return Summed atomic energy offsets for structure.
     */
    double                   getEnergyOffset(Structure const& structure) const;
    /** Add atomic energy offsets and return energy.
     *
     * @param[in] structure Input structure.
     * @param[in] ref If true, use reference energy, otherwise use NN energy.
     *
     * @return Reference or NNP energy with energy offsets added.
     *
     * @note If normalization is used, ensure that structure energy is already
     * in physical units.
     */
    double                   getEnergyWithOffset(
                                            Structure const& structure,
                                            bool             ref = true) const;
    /** Apply normalization to given property.
     *
     * @param[in] property One of "energy", "force", "charge", "hardness"
     *                     "negativity".
     * @param[in] value Input property value in physical units.
     *
     * @return Property in normalized units.
     */
    double                   normalized(std::string const& property,
                                        double             value) const;
    /** Apply normalization to given energy of structure.
     *
     * @param[in] structure Input structure with energy in physical units.
     * @param[in] ref If true, use reference energy, otherwise use NN energy.
     *
     * @return Energy in normalized units.
     */
    double                   normalizedEnergy(
                                            Structure const& structure,
                                            bool             ref = true) const;
    /** Undo normalization for a given property.
     *
     * @param[in] property One of "energy", "force", "charge", "hardness",
     *                     "negativity".
     * @param[in] value Input property value in normalized units.
     *
     * @return Property in physical units.
     */
    double                   physical(std::string const& property,
                                      double             value) const;
    /** Undo normalization for a given energy of structure.
     *
     * @param[in] structure Input structure with energy in normalized units.
     * @param[in] ref If true, use reference energy, otherwise use NN energy.
     *
     * @return Energy in physical units.
     */
    double                   physicalEnergy(Structure const& structure,
                                            bool             ref = true) const;
    /** Convert one structure to normalized units.
     *
     * @param[in,out] structure Input structure.
     */
    void                     convertToNormalizedUnits(
                                                   Structure& structure) const;
    /** Convert one structure to physical units.
     *
     * @param[in,out] structure Input structure.
     */
    void                     convertToPhysicalUnits(
                                                   Structure& structure) const;
    /** Logs Ewald params whenever they change.
     *
     */
    void                     logEwaldCutoffs();
    /** Count total number of extrapolation warnings encountered for all
     * elements and symmetry functions.
     *
     * @return Number of extrapolation warnings.
     */
    std::size_t              getNumExtrapolationWarnings() const;
    /** Erase all extrapolation warnings and reset counters.
     */
    void                     resetExtrapolationWarnings();
    /** Getter for Mode::nnpType.
     *
     * @return HDNNP type (2G, 4G,..) that was set up.
     */
    NNPType                  getNnpType() const;
    /** Getter for Mode::committeeMode.
     *
     * @return Committee mode.
     */
    CommitteeMode            getCommitteeMode() const;
    /** Getter for Mode::meanEnergy.
     *
     * @return Mean energy per atom.
     */
    double                   getMeanEnergy() const;
    /** Getter for Mode::convEnergy.
     *
     * @return Energy unit conversion factor.
     */
    double                   getConvEnergy() const;
    /** Getter for Mode::convLength.
     *
     * @return Length unit conversion factor.
     */
    double                   getConvLength() const;
    /** Getter for Mode::convCharge.
     *
     * @return Charge unit conversion factor.
     */
    double                   getConvCharge() const;
    /** Getter for Mode::ewaldSetup.precision.
     *
     * @return Ewald precision in 4G-HDNNPs.
     *
     */
    double                   getEwaldPrecision() const;
    /** Getter for Mode::ewaldSetup.maxCharge.
     *
     * @return Ewald max charge if specified in 4G-HDNNPs.
     *
     */
    double                   getEwaldMaxCharge() const;
    /** Getter for Mode::ewaldSetup.maxQsigma.
     *
     * @return Ewald max sigma parameter in 4G-HDNNPs.
     *
     */
    double                   getEwaldMaxSigma() const;
    /** Getter for Mode::ewaldSetup.truncMethod.
     *
     * @return Ewald truncation method in 4G-HDNNPs.
     *
     */
    EWALDTruncMethod         getEwaldTruncationMethod() const;
    /** Getter for Mode::kspaceSolver.
     *
     * @return K-space solver to be used in 4G-HDNNPs.
     *
     */
    KSPACESolver             kspaceSolver() const;
    /** Getter for Mode::maxCutoffRadius.
     *
     * @return Maximum cutoff radius of all symmetry functions.
     *
     * The maximum cutoff radius is determined by setupSymmetryFunctions().
     */
    double                   getMaxCutoffRadius() const;
    /** Getter for Mode::numElements.
     *
     * @return Number of elements defined.
     *
     * The number of elements is determined by setupElements().
     */
    std::size_t              getNumElements() const;
    /** Getter for Mode::screeningFunction.
     *
     * @return Copy of screening function instance.
     */
    ScreeningFunction        getScreeningFunction() const;
    /** Get number of symmetry functions per element.
     *
     * @return Vector with number of symmetry functions for each element.
     */
    std::vector<std::size_t> getNumSymmetryFunctions() const;
    /** Check if normalization is enabled.
     *
     * @return Value of #normalize.
     */
    bool                     useNormalization() const;
    /** Check if keyword was found in settings file.
     *
     * @param[in] keyword Keyword for which value is requested.
     *
     * @return `true` if keyword exists, `false` otherwise.
     */
    bool                     settingsKeywordExists(
                                             std::string const& keyword) const;
    /** Get value for given keyword in Settings instance.
     *
     * @param[in] keyword Keyword for which value is requested.
     *
     * @return Value string corresponding to keyword.
     */
    std::string              settingsGetValue(
                                             std::string const& keyword) const;
    /** Prune symmetry functions according to their range and write settings
     * file.
     *
     * @param[in] threshold Symmetry functions with range (max - min) smaller
     *                      than this threshold will be pruned.
     *
     * @return List of line numbers with symmetry function to be removed.
     */
    std::vector<std::size_t> pruneSymmetryFunctionsRange(double threshold);
    /** Prune symmetry functions with sensitivity analysis data.
     *
     * @param[in] threshold Symmetry functions with sensitivity lower than this
     *                      threshold will be pruned.
     * @param[in] sensitivity Sensitivity data for each element and symmetry
     *                        function.
     *
     * @return List of line numbers with symmetry function to be removed.
     */
    std::vector<std::size_t> pruneSymmetryFunctionsSensitivity(
                                            double threshold,
                                            std::vector<
                                            std::vector<double> > sensitivity);
    /** Copy settings file but comment out lines provided.
     *
     * @param[in] prune List of line numbers to comment out.
     * @param[in] fileName Output file name.
     */
    void                     writePrunedSettingsFile(
                                              std::vector<std::size_t> prune,
                                              std::string              fileName
                                                          = "output.nn") const;
    /** Write complete settings file.
     *
     * @param[in,out] file Settings file.
     */
    void                     writeSettingsFile(
                                             std::ofstream* const& file) const;

    /// Global element map, populated by setupElementMap().
    ElementMap elementMap;
    /// Global log file.
    Log        log;

protected:
#ifdef N2P2_GPU
    /** Evaluate the atomic neural networks on the GPU, if enabled.
     *
     * Same arguments as calculateAtomicNeuralNetworks().
     *
     * @return False if the networks are not evaluated on the GPU.
     */
    bool                     calculateAtomicNeuralNetworksGpu(
                                           Structure&         structure,
                                           bool const         derivatives,
                                           std::string const& id);
    /** Short-range forces and 4G electrostatic dChi/dr sums from the GPU
     * force assembly, if worthwhile.
     *
     * @param[in] structure Input structure (with symmetry function
     *                      derivatives, dEdG and, for 4G, dChidG).
     * @param[in] lambdaTotal Lagrange multipliers for the total forces (4G).
     * @param[in] lambdaElec Lagrange multipliers for the electrostatic
     *                       forces (4G).
     * @param[out] shortForces Short-range force on each atom.
     * @param[out] sumTotal As calculateDChidrSums() (4G).
     * @param[out] sumElec As calculateDChidrSums() (4G).
     *
     * @return False if the forces are to be assembled on the CPU.
     *
     * The topology (symmetry function derivatives) is copied to the GPU and
     * kept with the structure (Structure::gpuData) at the second call for
     * the same geometry (Structure::geometryStamp), so later calls only copy
     * the new weights. Building it costs more than a force calculation on
     * the CPU, so a geometry used once is left to the CPU, as is a structure
     * whose topology would exceed the memory budget of this process.
     */
    bool                     assembleForcesGpu(
                                   Structure&             structure,
                                   Eigen::VectorXd const& lambdaTotal,
                                   Eigen::VectorXd const& lambdaElec,
                                   std::vector<Vec3D>&    shortForces,
                                   std::vector<Vec3D>&    sumTotal,
                                   std::vector<Vec3D>&    sumElec) const;
    /** Calculate the symmetry functions of a structure on the GPU, if
     * enabled.
     *
     * @param[in] structure Input structure (with neighbor list).
     * @param[in] derivatives Also calculate the derivatives.
     *
     * @return False if the caller has to calculate them on the CPU.
     *
     * Same as calculateSymmetryFunctionGroups(), except that the
     * derivatives stay on the GPU (Structure::gpuData, under a new
     * Structure::geometryStamp): Atom::dGdr and Atom::Neighbor::dGdr are
     * not filled and Atom::hasSymmetryFunctionDerivatives stays false.
     * calculateForces() uses them, as does code calling
     * applySymmetryFunctionDerivativesGpu().
     */
    bool                     calculateSymmetryFunctionsGpu(
                                                   Structure& structure,
                                                   bool       derivatives);
    /** Check if a structure has symmetry function derivatives on the GPU.
     *
     * @param[in] structure Input structure.
     *
     * @return True if calculateSymmetryFunctionsGpu() calculated them for
     *         the current geometry.
     */
    bool                     hasSymmetryFunctionDerivativesGpu(
                                       Structure const& structure) const;
    /** Contract the symmetry function derivatives on the GPU with weights.
     *
     * @param[in] structure Input structure (see
     *                      hasSymmetryFunctionDerivativesGpu()).
     * @param[in] numWeightVectors Number of weight vectors (at most 3).
     * @param[in] weights Weight vectors, one entry per symmetry function
     *                    value (atoms one after the other).
     * @param[out] result For each weight vector and each target the sum of
     *                    w * dG/dr (3 per target, see
     *                    GpuSymmetryFunctionData::apply()).
     *
     * @return Number of targets (largest neighbor or atom index + 1).
     */
    std::size_t              applySymmetryFunctionDerivativesGpu(
                                       Structure const&     structure,
                                       std::size_t          numWeightVectors,
                                       double const*        weights,
                                       std::vector<double>& result) const;
    /** Derivatives of all symmetry functions of a structure with respect to
     * one coordinate of one atom, from the GPU.
     *
     * @param[in] structure Input structure (see
     *                      hasSymmetryFunctionDerivativesGpu()).
     * @param[in] indexAtom Index of the atom.
     * @param[in] indexComponent Coordinate (0, 1 or 2).
     * @param[out] result Derivatives of the symmetry functions of all atoms,
     *                    one atom after the other (as Atom::G).
     */
    void                     getSymmetryFunctionDerivativesGpu(
                                       Structure const&     structure,
                                       std::size_t          indexAtom,
                                       std::size_t          indexComponent,
                                       std::vector<double>& result) const;
    /** Propagate all atoms of a structure through their networks on the
     * GPU, one batch per element.
     *
     * @param[in] structure Input structure.
     * @param[in] id Neural network type.
     * @param[in] chargeInput Append the atomic charge to the inputs.
     * @param[in] output Atom member receiving the network output.
     * @param[in] derivative Atom member receiving the derivatives with
     *                       respect to the inputs, not calculated if
     *                       nullptr.
     */
    void                     propagateAtomsGpu(
                                   Structure&                structure,
                                   std::string const&        id,
                                   bool                      chargeInput,
                                   double Atom::*            output,
                                   std::vector<double> Atom::* derivative);
#endif
    /// Setup data for one neural network.
    struct NNSetup
    {
        struct Topology
        {
            /// Number of NN layers (including input and output layer).
            int                                numLayers;
            /// Number of neurons per layer.
            std::vector<int>                   numNeuronsPerLayer;
            /// Activation function type per layer.
            std::vector<
            NeuralNetwork::ActivationFunction> activationFunctionsPerLayer;

            /// Constructor.
            Topology() : numLayers(0) {};
        };

        /// NN identifier, e.g. "short", "charge",...
        std::string           id;
        /// Description string for log output, e.g. "electronegativity".
        std::string           name;
        /// Format for weight files.
        std::string           weightFileFormat;
        /// Suffix for keywords (NN topology related).
        std::string           keywordSuffix;
        /// Suffix for some other keywords (weight file loading related).
        std::string           keywordSuffix2;
        /// Per-element NN topology.
        std::vector<Topology> topology;
    };


    NNPType                    nnpType;
    CommitteeMode              committeeMode;
    bool                       normalize;
    bool                       checkExtrapolationWarnings;
    std::size_t                numElements;
    std::size_t                committeeSize;
    std::string                committeePrefix;
    /// Symmetry function scaling file (see setupSymmetryFunctionScaling()).
    std::string                scalingFileName;
    std::vector<std::size_t>   minNeighbors;
    std::vector<std::string>   committeeIds;
    std::vector<double>        minCutoffRadius;
    double                     maxCutoffRadius;
    double                     cutoffAlpha;
    double                     meanEnergy;
    double                     convEnergy;
    double                     convLength;
    double                     convCharge;
    double                     fourPiEps;
    EwaldSetup                 ewaldSetup;
    KspaceGrid                 kspaceGrid;
    settings::Settings         settings;
    SymFnc::ScalingType        scalingType;
    CutoffFunction::CutoffType cutoffType;
    ScreeningFunction          screeningFunction;
    std::vector<Element>       elements;
    std::vector<std::string>   nnk;
    std::map<
    std::string, NNSetup>      nns;
    /// Matrix storing all symmetry function cut-offs for all elements.
    std::vector<
    std::vector<double>>       cutoffs;
    ErfcBuf                    erfcBuf;
    /// If forces are assembled on the GPU (GPU builds).
    bool                       gpuForceAssembly;
    /// Device memory budget for force assembly topologies (GPU builds).
    std::size_t                gpuForceBudget;
    /// Force calculations assembled on the GPU and on the CPU.
    mutable std::size_t        gpuForceCountGpu;
    mutable std::size_t        gpuForceCountCpu;
    /// If symmetry functions are calculated on the GPU (GPU builds).
    bool                       gpuSymmetryFunctionsEnabled;
    /// Symmetry functions on the GPU, set up at the first use.
    std::shared_ptr<
    GpuSymmetryFunctionSet>    gpuSymmetryFunctions;
    /// Largest structure whose charge equilibration is solved on the GPU, 0
    /// if none (GPU builds).
    std::size_t                gpuChargeEquilibrationMaxAtoms;
    /// Charge equilibration solver on the GPU, set up at the first use.
    std::shared_ptr<
    GpuChargeEquilibration>    gpuChargeEquilibration;

private:
    /** Short-range force contributions of the neighbors of each atom.
     *
     * @param[in] structure Input structure.
     * @param[out] pairForces For each atom i the sum over its neighbors j
     *                        (including periodic images of i itself) of
     *                        -dE_j/dr_i.
     * @param[in] committeeMember Use the derivatives Atom::dEdGCom of this
     *                            committee member instead of Atom::dEdG
     *                            (default -1: Atom::dEdG).
     *
     * Collected from the side of atom j in one pass over its neighbors
     * within the symmetry function cutoff.
     */
    void                       calculatePairForcesShort(
                                     Structure const&       structure,
                                     std::vector<Vec3D>&    pairForces,
                                     int const              committeeMember
                                                                = -1) const;
    /** Sum over all atoms j of the derivatives of their electronegativities
     *  with respect to the coordinates of each atom i, weighted with the
     *  force Lagrange multipliers (4G-HDNNP forces).
     *
     * @param[in] structure Input structure.
     * @param[in] lambdaTotal Lagrange multipliers for the total forces.
     * @param[in] lambdaElec Lagrange multipliers for the electrostatic
     *                       forces.
     * @param[out] sumTotal For each atom i the sum over j of lambdaTotal(j)
     *                      * dChi_j/dr_i.
     * @param[out] sumElec Same with lambdaElec.
     *
     * Gives the same contributions as Atom::calculateDChidr() for every
     * pair of atoms, collected from the side of atom j: only atom j itself
     * and its neighbors within the symmetry function cutoff contribute, so
     * one pass over the neighbor lists replaces a neighbor list search for
     * each pair of atoms.
     */
    void                       calculateDChidrSums(
                                     Structure const&       structure,
                                     Eigen::VectorXd const& lambdaTotal,
                                     Eigen::VectorXd const& lambdaElec,
                                     std::vector<Vec3D>&    sumTotal,
                                     std::vector<Vec3D>&    sumElec) const;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline Mode::NNPType Mode::getNnpType() const
{
    return nnpType;
}

inline Mode::CommitteeMode Mode::getCommitteeMode() const
{
    return committeeMode;
}

inline double Mode::getMeanEnergy() const
{
    return meanEnergy;
}

inline double Mode::getConvEnergy() const
{
    return convEnergy;
}

inline double Mode::getConvLength() const
{
    return convLength;
}

inline double Mode::getConvCharge() const
{
    return convCharge;
}

inline double Mode::getMaxCutoffRadius() const
{
    return maxCutoffRadius;
}

inline std::size_t Mode::getNumElements() const
{
    return numElements;
}

inline double Mode::getEwaldPrecision() const
{
    return ewaldSetup.getPrecision();
}

inline double Mode::getEwaldMaxCharge() const
{
    return ewaldSetup.getMaxCharge();
}

inline double Mode::getEwaldMaxSigma() const
{
    return ewaldSetup.getMaxQSigma();
}

inline EWALDTruncMethod Mode::getEwaldTruncationMethod() const
{
    return ewaldSetup.getTruncMethod();
}

inline KSPACESolver Mode::kspaceSolver() const
{
    return kspaceGrid.kspaceSolver;
}

inline bool Mode::useNormalization() const
{
    return normalize;
}

}

#endif
