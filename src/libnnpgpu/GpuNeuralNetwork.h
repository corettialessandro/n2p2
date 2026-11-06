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

#ifndef GPUNEURALNETWORK_H
#define GPUNEURALNETWORK_H

// Feed-forward neural network evaluated on the GPU for many inputs at once.
// Plain C++ interface, included from code compiled without nvcc.

#include <cstddef> // std::size_t
#include <vector>  // std::vector

namespace nnp
{

/** GPU twin of a NeuralNetwork: same architecture and connections, but
 * propagates a whole batch of inputs (e.g. all atoms of one element) with
 * one matrix product per layer.
 *
 * Supports everything NeuralNetwork supports: any number of layers, all
 * activation functions and normalized neurons. The output layer must have a
 * single neuron (as all networks in n2p2).
 *
 * Batches of inputs and derivatives are column-major matrices with one
 * column per input, i.e. one input vector after the other.
 */
class GpuNeuralNetwork
{
public:
    /** Check the device and allocate the connections.
     *
     * @param[in] numNeuronsPerLayer Number of neurons per layer, including
     *                               input and output layer.
     * @param[in] activationFunctionsPerLayer Activation function per layer
     *            (NeuralNetwork::ActivationFunction, entry 0 unused).
     * @param[in] normalizeNeurons If neurons are normalized.
     *
     * Throws std::runtime_error if no usable device is available (see
     * gpuRequireDevice()) or the memory cannot be allocated.
     */
    GpuNeuralNetwork(std::vector<int> const& numNeuronsPerLayer,
                     std::vector<int> const& activationFunctionsPerLayer,
                     bool                    normalizeNeurons);
    /** Free all device memory.
     */
    ~GpuNeuralNetwork();
    GpuNeuralNetwork(GpuNeuralNetwork const&) = delete;
    GpuNeuralNetwork& operator=(GpuNeuralNetwork const&) = delete;
    /** Copy connections (weights and biases) to the device.
     *
     * @param[in] connections All connections, in the order of
     *                        NeuralNetwork::getConnections().
     */
    void        setConnections(double const* connections);
    /** Propagate a batch of inputs.
     *
     * @param[in] numInputs Number of input vectors.
     * @param[in] input Input vectors (numNeuronsPerLayer[0] x numInputs).
     * @param[out] output Output neuron values (numInputs).
     * @param[out] dOutputdInput Derivatives of the output with respect to
     *                           the inputs, same layout as input. Not
     *                           calculated if nullptr.
     * @param[out] statistics Neuron statistics of this batch: minimum,
     *             maximum, sum and sum of squares, each over all neurons
     *             layer by layer (input values for the input layer, values
     *             before activation otherwise), i.e. 4 x total number of
     *             neurons.
     */
    void        propagate(std::size_t   numInputs,
                          double const* input,
                          double*       output,
                          double*       dOutputdInput,
                          double*       statistics);
    /** Index of the device holding the network.
     */
    int         getDevice() const;

private:
    struct Data;
    Data* data;

    /// Make sure the per-input buffers hold numInputs columns.
    void        reserve(std::size_t numInputs);
    /// Free all device memory and the cuBLAS handle.
    void        release();
};

}

#endif
