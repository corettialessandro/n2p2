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

#include "GpuNeuralNetwork.h"
#include "GpuDevice.h"
#include "GpuError.h"
#include "NeuralNetwork.h"
#include <algorithm> // std::max
#include <cfloat>    // DBL_MAX
#include <stdexcept> // std::runtime_error

using namespace std;

namespace
{

using nnp::NeuralNetwork;

int const threadsPerBlock = 256;

/// Same limit as in NeuralNetwork.cpp.
__device__ double const expLimit = 35.0;

size_t numBlocks(size_t numThreads)
{
    return (numThreads + threadsPerBlock - 1) / threadsPerBlock;
}

// Activation function value and first two derivatives, with the same
// formulas and branches as NeuralNetwork::propagateLayer().
__device__ void activate(int     f,
                         double  x,
                         double& value,
                         double& dfdx,
                         double& d2fdx2)
{
    double t = 0.0;
    switch (f)
    {
    case NeuralNetwork::AF_IDENTITY:
        value = x;
        dfdx = 1.0;
        d2fdx2 = 0.0;
        break;
    case NeuralNetwork::AF_TANH:
        t = tanh(x);
        value = t;
        dfdx = 1.0 - t * t;
        d2fdx2 = -2.0 * t * (1.0 - t * t);
        break;
    case NeuralNetwork::AF_LOGISTIC:
        if (x > expLimit)
        {
            value = 1.0;
            dfdx = 0.0;
            d2fdx2 = 0.0;
        }
        else if (x < -expLimit)
        {
            value = 0.0;
            dfdx = 0.0;
            d2fdx2 = 0.0;
        }
        else
        {
            t = 1.0 / (1.0 + exp(-x));
            value = t;
            dfdx = t * (1.0 - t);
            d2fdx2 = t * (1.0 - t) * (1.0 - 2.0 * t);
        }
        break;
    case NeuralNetwork::AF_SOFTPLUS:
        if (x > expLimit)
        {
            value = x;
            dfdx = 1.0;
            d2fdx2 = 0.0;
        }
        else if (x < -expLimit)
        {
            value = 0.0;
            dfdx = 0.0;
            d2fdx2 = 0.0;
        }
        else
        {
            t = exp(x);
            value = log(1.0 + t);
            t = 1.0 / (1.0 + 1.0 / t);
            dfdx = t;
            d2fdx2 = t * (1.0 - t);
        }
        break;
    case NeuralNetwork::AF_RELU:
        if (x > 0.0)
        {
            value = x;
            dfdx = 1.0;
            d2fdx2 = 0.0;
        }
        else
        {
            value = 0.0;
            dfdx = 0.0;
            d2fdx2 = 0.0;
        }
        break;
    case NeuralNetwork::AF_GAUSSIAN:
        t = exp(-0.5 * x * x);
        value = t;
        dfdx = -x * t;
        d2fdx2 = (x * x - 1.0) * t;
        break;
    case NeuralNetwork::AF_COS:
        t = cos(x);
        value = t;
        dfdx = -sin(x);
        d2fdx2 = -t;
        break;
    case NeuralNetwork::AF_REVLOGISTIC:
        t = 1.0 / (1.0 + exp(-x));
        value = 1.0 - t;
        dfdx = t * (t - 1.0);
        d2fdx2 = t * (t - 1.0) * (1.0 - 2.0 * t);
        break;
    case NeuralNetwork::AF_EXP:
        t = exp(-x);
        value = t;
        dfdx = -t;
        d2fdx2 = t;
        break;
    case NeuralNetwork::AF_HARMONIC:
        value = x * x;
        dfdx = 2.0 * x;
        d2fdx2 = 2.0;
        break;
    default:
        // Unset or unknown: rejected by the constructor.
        value = 0.0;
        dfdx = 0.0;
        d2fdx2 = 0.0;
    }
}

// x = (x + bias) / scale, then value and derivatives of the activation
// function, for all neurons (rows) and inputs (columns) of a layer.
__global__ void activateLayer(int           f,
                              size_t        numNeurons,
                              size_t        numInputs,
                              double const* bias,
                              double        scale,
                              double*       x,
                              double*       value,
                              double*       dfdx,
                              double*       d2fdx2)
{
    size_t const i = blockIdx.x * static_cast<size_t>(blockDim.x)
                   + threadIdx.x;
    if (i >= numNeurons * numInputs) return;
    double const v = (x[i] + bias[i % numNeurons]) / scale;
    x[i] = v;
    activate(f, v, value[i], dfdx[i], d2fdx2[i]);
}

// ydot = dfdx * xdot (tangent of the activation).
__global__ void multiplyDerivative(size_t        numEntries,
                                   double const* dfdx,
                                   double const* xdot,
                                   double*       ydot)
{
    size_t const i = blockIdx.x * static_cast<size_t>(blockDim.x)
                   + threadIdx.x;
    if (i < numEntries) ydot[i] = dfdx[i] * xdot[i];
}

// gdot = (h * dfdx + hPrime * d2fdx2 * xdot) / scale, the tangent of
// g = h * dfdx / scale with h = W^T g_next and hPrime = W^T gdot_next.
__global__ void tangentBackward(size_t        numEntries,
                                double const* hPrime,
                                double const* h,
                                double const* dfdx,
                                double const* d2fdx2,
                                double const* xdot,
                                double        scale,
                                double*       gdot)
{
    size_t const i = blockIdx.x * static_cast<size_t>(blockDim.x)
                   + threadIdx.x;
    if (i < numEntries)
    {
        gdot[i] = (hPrime[i] * dfdx[i] + h[i] * d2fdx2[i] * xdot[i]) / scale;
    }
}

// a = b * c / scale.
__global__ void multiplyScaled(size_t        numEntries,
                               double const* b,
                               double const* c,
                               double        scale,
                               double*       a)
{
    size_t const i = blockIdx.x * static_cast<size_t>(blockDim.x)
                   + threadIdx.x;
    if (i < numEntries) a[i] = b[i] * c[i] / scale;
}

__global__ void fill(size_t numEntries, double value, double* a)
{
    size_t const i = blockIdx.x * static_cast<size_t>(blockDim.x)
                   + threadIdx.x;
    if (i < numEntries) a[i] = value;
}

// Minimum, maximum, sum and sum of squares of each row of a numRows x
// numColumns matrix, one block per row. Results go to statistics[k *
// stride + offset + row], k = 0 (minimum) to 3 (sum of squares).
__global__ void rowStatistics(size_t        numRows,
                              size_t        numColumns,
                              double const* matrix,
                              size_t        offset,
                              size_t        stride,
                              double*       statistics)
{
    __shared__ double sMin[threadsPerBlock];
    __shared__ double sMax[threadsPerBlock];
    __shared__ double sSum[threadsPerBlock];
    __shared__ double sSum2[threadsPerBlock];
    size_t const row = blockIdx.x;
    double vMin = DBL_MAX;
    double vMax = -DBL_MAX;
    double vSum = 0.0;
    double vSum2 = 0.0;
    for (size_t j = threadIdx.x; j < numColumns; j += blockDim.x)
    {
        double const v = matrix[row + j * numRows];
        vMin = fmin(vMin, v);
        vMax = fmax(vMax, v);
        vSum += v;
        vSum2 += v * v;
    }
    sMin[threadIdx.x] = vMin;
    sMax[threadIdx.x] = vMax;
    sSum[threadIdx.x] = vSum;
    sSum2[threadIdx.x] = vSum2;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
        {
            sMin[threadIdx.x] = fmin(sMin[threadIdx.x],
                                     sMin[threadIdx.x + s]);
            sMax[threadIdx.x] = fmax(sMax[threadIdx.x],
                                     sMax[threadIdx.x + s]);
            sSum[threadIdx.x] += sSum[threadIdx.x + s];
            sSum2[threadIdx.x] += sSum2[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        statistics[0 * stride + offset + row] = sMin[0];
        statistics[1 * stride + offset + row] = sMax[0];
        statistics[2 * stride + offset + row] = sSum[0];
        statistics[3 * stride + offset + row] = sSum2[0];
    }
}

}

namespace nnp
{

struct GpuNeuralNetwork::Data
{
    int             device;
    size_t          numLayers;
    vector<int>     numNeurons;
    vector<int>     activation;
    bool            normalize;
    /// Offsets of weights and biases of layer l (l >= 1) in connections.
    vector<size_t>  weightOffset;
    vector<size_t>  biasOffset;
    size_t          numConnections;
    /// Offset of layer l in the neuron statistics.
    vector<size_t>  neuronOffset;
    size_t          numNeuronsTotal;
    size_t          capacity;
    double*         connections;
    /// Per layer, numNeurons[l] x capacity: x (before activation), value,
    /// dfdx, d2fdx2, the backward pass g (derivatives with respect to the
    /// biases), the tangents xdot, ydot (value) and gdot along an input
    /// direction and the scratch matrix h. Layer 0 uses value and ydot
    /// only.
    vector<double*> x;
    vector<double*> value;
    vector<double*> dfdx;
    vector<double*> d2fdx2;
    vector<double*> g;
    vector<double*> xdot;
    vector<double*> ydot;
    vector<double*> gdot;
    vector<double*> h;
    /// Derivatives of the output with respect to the inputs.
    double*         dOutputdInput;
    /// Vector of ones (capacity), sums over the batch.
    double*         ones;
    /// Summed derivatives with respect to the connections.
    double*         dConnections;
    double*         statistics;
    cublasHandle_t  handle;
};

GpuNeuralNetwork::GpuNeuralNetwork(
                                vector<int> const& numNeuronsPerLayer,
                                vector<int> const& activationFunctionsPerLayer,
                                bool               normalizeNeurons) :
    data(new Data())
{
    Data& d = *data;
    d.numLayers = numNeuronsPerLayer.size();
    d.numNeurons = numNeuronsPerLayer;
    d.activation = activationFunctionsPerLayer;
    d.normalize = normalizeNeurons;
    d.capacity = 0;
    d.connections = nullptr;
    d.dOutputdInput = nullptr;
    d.ones = nullptr;
    d.dConnections = nullptr;
    d.statistics = nullptr;
    d.handle = nullptr;
    for (vector<double*>* b : {&d.x, &d.value, &d.dfdx, &d.d2fdx2, &d.g,
                               &d.xdot, &d.ydot, &d.gdot, &d.h})
    {
        b->assign(d.numLayers, nullptr);
    }
    try
    {
        if (d.numLayers < 3 || d.activation.size() != d.numLayers
            || d.numNeurons.back() != 1)
        {
            throw runtime_error("ERROR: GPU neural network needs at least "
                                "three layers and a single output "
                                "neuron.\n");
        }
        for (size_t l = 1; l < d.numLayers; ++l)
        {
            if (d.activation.at(l) < NeuralNetwork::AF_IDENTITY ||
                d.activation.at(l) > NeuralNetwork::AF_HARMONIC)
            {
                throw runtime_error("ERROR: Unknown activation function for "
                                    "GPU neural network.\n");
            }
        }
        d.weightOffset.assign(d.numLayers, 0);
        d.biasOffset.assign(d.numLayers, 0);
        size_t count = 0;
        for (size_t l = 1; l < d.numLayers; ++l)
        {
            d.weightOffset.at(l) = count;
            count += static_cast<size_t>(d.numNeurons.at(l))
                   * d.numNeurons.at(l - 1);
            d.biasOffset.at(l) = count;
            count += d.numNeurons.at(l);
        }
        d.numConnections = count;
        d.neuronOffset.assign(d.numLayers, 0);
        count = 0;
        for (size_t l = 0; l < d.numLayers; ++l)
        {
            d.neuronOffset.at(l) = count;
            count += d.numNeurons.at(l);
        }
        d.numNeuronsTotal = count;

        d.device = gpuRequireDevice();
        NNP_CUDA_CHECK(cudaMalloc(&d.connections,
                                  d.numConnections * sizeof(double)));
        NNP_CUDA_CHECK(cudaMalloc(&d.dConnections,
                                  d.numConnections * sizeof(double)));
        NNP_CUDA_CHECK(cudaMalloc(&d.statistics,
                                  4 * d.numNeuronsTotal * sizeof(double)));
        NNP_CUBLAS_CHECK(cublasCreate(&d.handle));
    }
    catch (...)
    {
        release();
        throw;
    }
}

GpuNeuralNetwork::~GpuNeuralNetwork()
{
    release();
}

void GpuNeuralNetwork::release()
{
    // No checks here: called from the destructor, which must not throw.
    Data& d = *data;
    cudaSetDevice(d.device);
    if (d.handle != nullptr) cublasDestroy(d.handle);
    cudaFree(d.connections);
    cudaFree(d.dConnections);
    cudaFree(d.statistics);
    cudaFree(d.dOutputdInput);
    cudaFree(d.ones);
    for (vector<double*>* b : {&d.x, &d.value, &d.dfdx, &d.d2fdx2, &d.g,
                               &d.xdot, &d.ydot, &d.gdot, &d.h})
    {
        for (double* p : *b) cudaFree(p);
    }
    delete data;
    data = nullptr;
}

void GpuNeuralNetwork::reserve(size_t numInputs)
{
    Data& d = *data;
    if (numInputs <= d.capacity) return;
    // Grow by at least 50 % to avoid frequent reallocation.
    size_t const capacity = max(numInputs, d.capacity + d.capacity / 2);
    d.capacity = 0;
    NNP_CUDA_CHECK(cudaFree(d.dOutputdInput));
    NNP_CUDA_CHECK(cudaFree(d.ones));
    d.dOutputdInput = nullptr;
    d.ones = nullptr;
    NNP_CUDA_CHECK(cudaMalloc(&d.dOutputdInput,
                              d.numNeurons.at(0) * capacity
                              * sizeof(double)));
    NNP_CUDA_CHECK(cudaMalloc(&d.ones, capacity * sizeof(double)));
    fill<<<numBlocks(capacity), threadsPerBlock>>>(capacity, 1.0, d.ones);
    NNP_CUDA_CHECK(cudaGetLastError());
    for (size_t l = 0; l < d.numLayers; ++l)
    {
        size_t const bytes = d.numNeurons.at(l) * capacity * sizeof(double);
        for (vector<double*>* b : {&d.x, &d.value, &d.dfdx, &d.d2fdx2, &d.g,
                                   &d.xdot, &d.ydot, &d.gdot, &d.h})
        {
            NNP_CUDA_CHECK(cudaFree(b->at(l)));
            b->at(l) = nullptr;
            // Input layer: only values and their tangents are needed.
            if (l == 0 && b != &d.value && b != &d.ydot) continue;
            NNP_CUDA_CHECK(cudaMalloc(&(b->at(l)), bytes));
        }
    }
    d.capacity = capacity;
}

void GpuNeuralNetwork::setConnections(double const* connections)
{
    NNP_CUDA_CHECK(cudaSetDevice(data->device));
    NNP_CUDA_CHECK(cudaMemcpy(data->connections,
                              connections,
                              data->numConnections * sizeof(double),
                              cudaMemcpyHostToDevice));
}

void GpuNeuralNetwork::propagate(size_t        numInputs,
                                 double const* input,
                                 double*       output,
                                 double*       dOutputdInput,
                                 double*       statistics)
{
    if (numInputs == 0) return;
    forward(numInputs, input);
    if (dOutputdInput != nullptr)
    {
        backward(numInputs);
        inputDerivatives(numInputs, dOutputdInput);
    }
    results(numInputs, output, statistics);
}

void GpuNeuralNetwork::propagateDEdc(size_t        numInputs,
                                     double const* input,
                                     double*       output,
                                     double*       dOutputdConnectionsSum,
                                     double*       statistics)
{
    if (numInputs == 0) return;
    forward(numInputs, input);
    backward(numInputs);
    // As NeuralNetwork::calculateDEdc(): weights delta . y^T, biases delta.
    connectionDerivatives(numInputs,
                          data->g,
                          data->value,
                          {},
                          {},
                          1.0,
                          dOutputdConnectionsSum);
    results(numInputs, output, statistics);
}

void GpuNeuralNetwork::propagateDFdc(size_t        numInputs,
                                     double const* input,
                                     double const* dInputdx,
                                     double*       output,
                                     double*       dOutputdInput,
                                     double*       dForcedConnectionsSum,
                                     double*       statistics)
{
    if (numInputs == 0) return;
    forward(numInputs, input);
    backward(numInputs);
    if (dOutputdInput != nullptr) inputDerivatives(numInputs, dOutputdInput);
    tangent(numInputs, dInputdx);
    // Directional derivative of calculateDEdc() along dInputdx, as
    // NeuralNetwork::calculateD2EdGdc() summed over the inputs: weights
    // gdot . y^T + g . ydot^T, biases gdot. F = -dE/dx gives the sign.
    connectionDerivatives(numInputs,
                          data->gdot,
                          data->value,
                          data->g,
                          data->ydot,
                          -1.0,
                          dForcedConnectionsSum);
    results(numInputs, output, statistics);
}

void GpuNeuralNetwork::forward(size_t numInputs, double const* input)
{
    Data& d = *data;
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    reserve(numInputs);
    int const numColumns = static_cast<int>(numInputs);
    double const one = 1.0;
    double const zero = 0.0;

    NNP_CUDA_CHECK(cudaMemcpy(d.value.at(0),
                              input,
                              d.numNeurons.at(0) * numInputs * sizeof(double),
                              cudaMemcpyHostToDevice));

    // Forward pass: x = (W . value_prev + b) / scale, value = f(x).
    for (size_t l = 1; l < d.numLayers; ++l)
    {
        int const n = d.numNeurons.at(l);
        int const nPrev = d.numNeurons.at(l - 1);
        NNP_CUBLAS_CHECK(cublasDgemm(d.handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, numColumns, nPrev,
                                     &one,
                                     d.connections + d.weightOffset.at(l), n,
                                     d.value.at(l - 1), nPrev,
                                     &zero, d.x.at(l), n));
        size_t const numEntries = n * numInputs;
        activateLayer<<<numBlocks(numEntries), threadsPerBlock>>>(
            d.activation.at(l),
            n,
            numInputs,
            d.connections + d.biasOffset.at(l),
            d.normalize ? static_cast<double>(nPrev) : 1.0,
            d.x.at(l),
            d.value.at(l),
            d.dfdx.at(l),
            d.d2fdx2.at(l));
        NNP_CUDA_CHECK(cudaGetLastError());
    }

    // Neuron statistics: input values, values before activation otherwise.
    for (size_t l = 0; l < d.numLayers; ++l)
    {
        rowStatistics<<<d.numNeurons.at(l), threadsPerBlock>>>(
            d.numNeurons.at(l),
            numInputs,
            l == 0 ? d.value.at(0) : d.x.at(l),
            d.neuronOffset.at(l),
            d.numNeuronsTotal,
            d.statistics);
        NNP_CUDA_CHECK(cudaGetLastError());
    }
}

void GpuNeuralNetwork::backward(size_t numInputs)
{
    // As NeuralNetwork::calculateDEdb(): g = dfdx / scale for the output
    // layer, then g_l = (W_{l+1}^T . g_{l+1}) * dfdx_l / scale_l.
    Data& d = *data;
    int const numColumns = static_cast<int>(numInputs);
    size_t const last = d.numLayers - 1;
    double const one = 1.0;
    double const zero = 0.0;
    for (size_t l = last; l >= 1; --l)
    {
        int const n = d.numNeurons.at(l);
        double const scale = d.normalize ? d.numNeurons.at(l - 1) : 1.0;
        size_t const numEntries = n * numInputs;
        if (l == last)
        {
            fill<<<numBlocks(numEntries), threadsPerBlock>>>(numEntries,
                                                             1.0,
                                                             d.h.at(l));
        }
        else
        {
            int const nNext = d.numNeurons.at(l + 1);
            NNP_CUBLAS_CHECK(cublasDgemm(
                d.handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, numColumns, nNext,
                &one,
                d.connections + d.weightOffset.at(l + 1), nNext,
                d.g.at(l + 1), nNext,
                &zero, d.h.at(l), n));
        }
        multiplyScaled<<<numBlocks(numEntries), threadsPerBlock>>>(
            numEntries, d.h.at(l), d.dfdx.at(l), scale, d.g.at(l));
        NNP_CUDA_CHECK(cudaGetLastError());
    }
}

void GpuNeuralNetwork::inputDerivatives(size_t numInputs,
                                        double* dOutputdInput)
{
    // As NeuralNetwork::calculateDEdG(): W_1^T . g_1.
    Data& d = *data;
    int const numColumns = static_cast<int>(numInputs);
    int const nIn = d.numNeurons.at(0);
    int const n1 = d.numNeurons.at(1);
    double const one = 1.0;
    double const zero = 0.0;
    NNP_CUBLAS_CHECK(cublasDgemm(d.handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 nIn, numColumns, n1,
                                 &one,
                                 d.connections + d.weightOffset.at(1), n1,
                                 d.g.at(1), n1,
                                 &zero, d.dOutputdInput, nIn));
    NNP_CUDA_CHECK(cudaMemcpy(dOutputdInput,
                              d.dOutputdInput,
                              nIn * numInputs * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

void GpuNeuralNetwork::tangent(size_t numInputs, double const* dInputdx)
{
    // Forward mode along the input direction, as NeuralNetwork::
    // calculateDxdG(): xdot = W . ydot_prev / scale, ydot = dfdx * xdot.
    // Then the derivative of backward() along it: for the output layer
    // gdot = d2fdx2 * xdot / scale, otherwise gdot_l = (W^T . gdot_{l+1}
    // * dfdx_l + W^T . g_{l+1} * d2fdx2_l * xdot_l) / scale_l.
    Data& d = *data;
    int const numColumns = static_cast<int>(numInputs);
    size_t const last = d.numLayers - 1;
    double const zero = 0.0;
    double const one = 1.0;
    NNP_CUDA_CHECK(cudaMemcpy(d.ydot.at(0),
                              dInputdx,
                              d.numNeurons.at(0) * numInputs * sizeof(double),
                              cudaMemcpyHostToDevice));
    for (size_t l = 1; l < d.numLayers; ++l)
    {
        int const n = d.numNeurons.at(l);
        int const nPrev = d.numNeurons.at(l - 1);
        double const scaleInv = d.normalize ? 1.0 / nPrev : 1.0;
        NNP_CUBLAS_CHECK(cublasDgemm(d.handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, numColumns, nPrev,
                                     &scaleInv,
                                     d.connections + d.weightOffset.at(l), n,
                                     d.ydot.at(l - 1), nPrev,
                                     &zero, d.xdot.at(l), n));
        size_t const numEntries = n * numInputs;
        multiplyDerivative<<<numBlocks(numEntries), threadsPerBlock>>>(
            numEntries, d.dfdx.at(l), d.xdot.at(l), d.ydot.at(l));
        NNP_CUDA_CHECK(cudaGetLastError());
    }
    for (size_t l = last; l >= 1; --l)
    {
        int const n = d.numNeurons.at(l);
        double const scale = d.normalize ? d.numNeurons.at(l - 1) : 1.0;
        size_t const numEntries = n * numInputs;
        if (l == last)
        {
            // h (ones) from backward(), hPrime = 0.
            multiplyScaled<<<numBlocks(numEntries), threadsPerBlock>>>(
                numEntries, d.d2fdx2.at(l), d.xdot.at(l), scale, d.gdot.at(l));
        }
        else
        {
            int const nNext = d.numNeurons.at(l + 1);
            // h_l = W^T . g_{l+1} is still there from backward(); the
            // scratch for W^T . gdot_{l+1} is gdot_l itself.
            NNP_CUBLAS_CHECK(cublasDgemm(
                d.handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, numColumns, nNext,
                &one,
                d.connections + d.weightOffset.at(l + 1), nNext,
                d.gdot.at(l + 1), nNext,
                &zero, d.gdot.at(l), n));
            tangentBackward<<<numBlocks(numEntries), threadsPerBlock>>>(
                numEntries,
                d.gdot.at(l),
                d.h.at(l),
                d.dfdx.at(l),
                d.d2fdx2.at(l),
                d.xdot.at(l),
                scale,
                d.gdot.at(l));
        }
        NNP_CUDA_CHECK(cudaGetLastError());
    }
}

void GpuNeuralNetwork::connectionDerivatives(
                                        size_t                 numInputs,
                                        vector<double*> const& delta,
                                        vector<double*> const& y,
                                        vector<double*> const& deltaExtra,
                                        vector<double*> const& yExtra,
                                        double                 factor,
                                        double*                result)
{
    Data& d = *data;
    int const numColumns = static_cast<int>(numInputs);
    double const zero = 0.0;
    double const one = 1.0;
    for (size_t l = 1; l < d.numLayers; ++l)
    {
        int const n = d.numNeurons.at(l);
        int const nPrev = d.numNeurons.at(l - 1);
        double* weights = d.dConnections + d.weightOffset.at(l);
        // Weights (n x nPrev, same layout as in the connections).
        NNP_CUBLAS_CHECK(cublasDgemm(d.handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T,
                                     n, nPrev, numColumns,
                                     &factor,
                                     delta.at(l), n,
                                     y.at(l - 1), nPrev,
                                     &zero, weights, n));
        if (!deltaExtra.empty())
        {
            NNP_CUBLAS_CHECK(cublasDgemm(d.handle,
                                         CUBLAS_OP_N, CUBLAS_OP_T,
                                         n, nPrev, numColumns,
                                         &factor,
                                         deltaExtra.at(l), n,
                                         yExtra.at(l - 1), nPrev,
                                         &one, weights, n));
        }
        // Biases: sum of delta over the batch.
        NNP_CUBLAS_CHECK(cublasDgemv(d.handle,
                                     CUBLAS_OP_N,
                                     n, numColumns,
                                     &factor,
                                     delta.at(l), n,
                                     d.ones, 1,
                                     &zero,
                                     d.dConnections + d.biasOffset.at(l), 1));
    }
    NNP_CUDA_CHECK(cudaMemcpy(result,
                              d.dConnections,
                              d.numConnections * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

void GpuNeuralNetwork::results(size_t  numInputs,
                               double* output,
                               double* statistics)
{
    Data& d = *data;
    NNP_CUDA_CHECK(cudaMemcpy(output,
                              d.value.at(d.numLayers - 1),
                              numInputs * sizeof(double),
                              cudaMemcpyDeviceToHost));
    NNP_CUDA_CHECK(cudaMemcpy(statistics,
                              d.statistics,
                              4 * d.numNeuronsTotal * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

int GpuNeuralNetwork::getDevice() const
{
    return data->device;
}

}
