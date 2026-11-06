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

#include "GpuForceAssembly.h"
#include "GpuDevice.h"
#include "GpuError.h"
#include <atomic>    // std::atomic
#include <climits>   // INT_MAX
#include <stdexcept> // std::runtime_error

using namespace std;

namespace
{

int const threadsPerAtom = 32;
int const threadsPerBlock = 256;

/// Device memory of all topologies of this process.
atomic<size_t> totalMemoryUsage(0);

// One warp per atom: sums of weight * dG/dr over the terms of the atom, in a
// fixed order (lane strides, then a fixed reduction tree).
__global__ void assemble(size_t        numAtoms,
                         int           numWeightVectors,
                         size_t        numValues,
                         int const*    termOffset,
                         int const*    termValue,
                         double const* termDGdr,
                         double const* weights,
                         double*       result)
{
    size_t const atom = (blockIdx.x * static_cast<size_t>(blockDim.x)
                         + threadIdx.x) / threadsPerAtom;
    int const lane = threadIdx.x % threadsPerAtom;
    if (atom >= numAtoms) return;

    double sum[3 * nnp::GpuForceTopology::maxWeightVectors] = {0.0};
    for (int t = termOffset[atom] + lane; t < termOffset[atom + 1];
         t += threadsPerAtom)
    {
        size_t const v = termValue[t];
        double const* d = termDGdr + 3 * static_cast<size_t>(t);
        for (int w = 0; w < numWeightVectors; ++w)
        {
            double const weight = weights[w * numValues + v];
            sum[3 * w + 0] += weight * d[0];
            sum[3 * w + 1] += weight * d[1];
            sum[3 * w + 2] += weight * d[2];
        }
    }
    for (int k = 0; k < 3 * numWeightVectors; ++k)
    {
        for (int offset = threadsPerAtom / 2; offset > 0; offset /= 2)
        {
            sum[k] += __shfl_down_sync(0xffffffff, sum[k], offset);
        }
    }
    if (lane == 0)
    {
        for (int w = 0; w < numWeightVectors; ++w)
        {
            for (int c = 0; c < 3; ++c)
            {
                result[(w * numAtoms + atom) * 3 + c] = sum[3 * w + c];
            }
        }
    }
}

}

namespace nnp
{

struct GpuForceTopology::Data
{
    int     device;
    size_t  numAtoms;
    size_t  numValues;
    size_t  bytes;
    int*    termOffset;
    int*    termValue;
    double* termDGdr;
    double* weights;
    double* result;
};

size_t GpuForceTopology::memoryUsage(size_t numAtoms,
                                     size_t numValues,
                                     size_t numTerms)
{
    return (numAtoms + 1) * sizeof(int) + numTerms * sizeof(int)
         + 3 * numTerms * sizeof(double)
         + maxWeightVectors * (numValues + 3 * numAtoms) * sizeof(double);
}

size_t GpuForceTopology::getTotalMemoryUsage()
{
    return totalMemoryUsage;
}

GpuForceTopology::GpuForceTopology(size_t                numValues,
                                   vector<int> const&    termOffset,
                                   vector<int> const&    termValue,
                                   vector<double> const& termDGdr) :
    data(new Data())
{
    Data& d = *data;
    d.numAtoms = termOffset.size() - 1;
    d.numValues = numValues;
    d.bytes = 0;
    d.termOffset = nullptr;
    d.termValue = nullptr;
    d.termDGdr = nullptr;
    d.weights = nullptr;
    d.result = nullptr;
    size_t const numTerms = termValue.size();
    try
    {
        if (termOffset.empty() || numTerms > static_cast<size_t>(INT_MAX) ||
            termOffset.back() != static_cast<int>(numTerms) ||
            termDGdr.size() != 3 * numTerms)
        {
            throw runtime_error("ERROR: Inconsistent or too large GPU force "
                                "assembly topology.\n");
        }
        d.device = gpuCurrentDevice();
        auto allocate = [](void** p, size_t bytes)
        {
            NNP_CUDA_CHECK(cudaMalloc(p, bytes > 0 ? bytes : 1));
        };
        allocate(reinterpret_cast<void**>(&d.termOffset),
                 termOffset.size() * sizeof(int));
        allocate(reinterpret_cast<void**>(&d.termValue),
                 numTerms * sizeof(int));
        allocate(reinterpret_cast<void**>(&d.termDGdr),
                 termDGdr.size() * sizeof(double));
        allocate(reinterpret_cast<void**>(&d.weights),
                 maxWeightVectors * numValues * sizeof(double));
        allocate(reinterpret_cast<void**>(&d.result),
                 maxWeightVectors * 3 * d.numAtoms * sizeof(double));
        NNP_CUDA_CHECK(cudaMemcpy(d.termOffset,
                                  termOffset.data(),
                                  termOffset.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        NNP_CUDA_CHECK(cudaMemcpy(d.termValue,
                                  termValue.data(),
                                  numTerms * sizeof(int),
                                  cudaMemcpyHostToDevice));
        NNP_CUDA_CHECK(cudaMemcpy(d.termDGdr,
                                  termDGdr.data(),
                                  termDGdr.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));
    }
    catch (...)
    {
        cudaFree(d.termOffset);
        cudaFree(d.termValue);
        cudaFree(d.termDGdr);
        cudaFree(d.weights);
        cudaFree(d.result);
        delete data;
        throw;
    }
    d.bytes = memoryUsage(d.numAtoms, numValues, numTerms);
    totalMemoryUsage += d.bytes;
}

GpuForceTopology::~GpuForceTopology()
{
    // No checks: the destructor must not throw.
    Data& d = *data;
    cudaSetDevice(d.device);
    cudaFree(d.termOffset);
    cudaFree(d.termValue);
    cudaFree(d.termDGdr);
    cudaFree(d.weights);
    cudaFree(d.result);
    totalMemoryUsage -= d.bytes;
    delete data;
}

void GpuForceTopology::apply(size_t        numWeightVectors,
                             double const* weights,
                             double*       result)
{
    Data& d = *data;
    if (numWeightVectors < 1 || numWeightVectors > maxWeightVectors)
    {
        throw runtime_error("ERROR: Wrong number of weight vectors for GPU "
                            "force assembly.\n");
    }
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    NNP_CUDA_CHECK(cudaMemcpy(d.weights,
                              weights,
                              numWeightVectors * d.numValues * sizeof(double),
                              cudaMemcpyHostToDevice));
    size_t const numThreads = d.numAtoms * threadsPerAtom;
    size_t const blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 0)
    {
        assemble<<<blocks, threadsPerBlock>>>(
            d.numAtoms,
            static_cast<int>(numWeightVectors),
            d.numValues,
            d.termOffset,
            d.termValue,
            d.termDGdr,
            d.weights,
            d.result);
        NNP_CUDA_CHECK(cudaGetLastError());
    }
    NNP_CUDA_CHECK(cudaMemcpy(result,
                              d.result,
                              numWeightVectors * 3 * d.numAtoms
                              * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

size_t GpuForceTopology::getMemoryUsage() const
{
    return data->bytes;
}

}
