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

#include "GpuKalmanFilter.h"
#include "GpuDevice.h"
#include "GpuError.h"

using namespace std;

namespace
{

// P = scale * P, grid-stride loop over all n * n entries.
__global__ void scaleMatrix(double* P, size_t numEntries, double scale)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numEntries;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        P[i] *= scale;
    }
}

// P = P + q * I.
__global__ void addToDiagonal(double* P, size_t n, double q)
{
    size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) P[i * (n + 1)] += q;
}

int const threadsPerBlock = 256;

}

namespace nnp
{

struct GpuKalmanFilter::Data
{
    int            device;
    size_t         sizeState;
    size_t         capacity;
    double*        P;
    double*        H;
    double*        X;
    double*        K;
    cublasHandle_t handle;
};

GpuKalmanFilter::GpuKalmanFilter(size_t sizeState) : data(new Data())
{
    // Data() is zero-initialized: release() works at any point below.
    data->sizeState = sizeState;
    try
    {
        data->device = gpuRequireDevice();
        NNP_CUDA_CHECK(cudaMalloc(&data->P,
                                  sizeState * sizeState * sizeof(double)));
        NNP_CUBLAS_CHECK(cublasCreate(&data->handle));
        reserve(1);
    }
    catch (...)
    {
        release();
        throw;
    }
}

GpuKalmanFilter::~GpuKalmanFilter()
{
    release();
}

void GpuKalmanFilter::release()
{
    // No checks here: called from the destructor, which must not throw.
    cudaSetDevice(data->device);
    if (data->handle != nullptr) cublasDestroy(data->handle);
    cudaFree(data->P);
    cudaFree(data->H);
    cudaFree(data->X);
    cudaFree(data->K);
    delete data;
    data = nullptr;
}

void GpuKalmanFilter::reserve(size_t sizeObservation)
{
    if (sizeObservation <= data->capacity) return;
    NNP_CUDA_CHECK(cudaFree(data->H));
    NNP_CUDA_CHECK(cudaFree(data->X));
    NNP_CUDA_CHECK(cudaFree(data->K));
    data->H = nullptr;
    data->X = nullptr;
    data->K = nullptr;
    data->capacity = 0;
    size_t const bytes = data->sizeState * sizeObservation * sizeof(double);
    NNP_CUDA_CHECK(cudaMalloc(&data->H, bytes));
    NNP_CUDA_CHECK(cudaMalloc(&data->X, bytes));
    NNP_CUDA_CHECK(cudaMalloc(&data->K, bytes));
    data->capacity = sizeObservation;
}

void GpuKalmanFilter::setP(double const* P)
{
    NNP_CUDA_CHECK(cudaSetDevice(data->device));
    size_t const n = data->sizeState;
    NNP_CUDA_CHECK(cudaMemcpy(data->P,
                              P,
                              n * n * sizeof(double),
                              cudaMemcpyHostToDevice));
}

void GpuKalmanFilter::getP(double* P) const
{
    NNP_CUDA_CHECK(cudaSetDevice(data->device));
    size_t const n = data->sizeState;
    NNP_CUDA_CHECK(cudaMemcpy(P,
                              data->P,
                              n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

void GpuKalmanFilter::computeX(size_t        sizeObservation,
                               double const* H,
                               double*       X)
{
    NNP_CUDA_CHECK(cudaSetDevice(data->device));
    reserve(sizeObservation);
    int const n = static_cast<int>(data->sizeState);
    int const m = static_cast<int>(sizeObservation);
    size_t const bytes = data->sizeState * sizeObservation * sizeof(double);
    NNP_CUDA_CHECK(cudaMemcpy(data->H, H, bytes, cudaMemcpyHostToDevice));
    double const one = 1.0;
    double const zero = 0.0;
    NNP_CUBLAS_CHECK(cublasDsymm(data->handle,
                                 CUBLAS_SIDE_LEFT,
                                 CUBLAS_FILL_MODE_LOWER,
                                 n, m,
                                 &one, data->P, n,
                                 data->H, n,
                                 &zero, data->X, n));
    NNP_CUDA_CHECK(cudaMemcpy(X, data->X, bytes, cudaMemcpyDeviceToHost));
}

void GpuKalmanFilter::updateP(size_t        sizeObservation,
                              double const* K,
                              double        scale,
                              double        q)
{
    NNP_CUDA_CHECK(cudaSetDevice(data->device));
    size_t const n = data->sizeState;
    int const ni = static_cast<int>(n);
    int const m = static_cast<int>(sizeObservation);
    NNP_CUDA_CHECK(cudaMemcpy(data->K,
                              K,
                              n * sizeObservation * sizeof(double),
                              cudaMemcpyHostToDevice));
    double const minusOne = -1.0;
    double const one = 1.0;
    NNP_CUBLAS_CHECK(cublasDgemm(data->handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 ni, ni, m,
                                 &minusOne, data->K, ni,
                                 data->X, ni,
                                 &one, data->P, ni));
    if (scale != 1.0)
    {
        size_t const numEntries = n * n;
        size_t const maxBlocks = 65535;
        size_t blocks = (numEntries + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > maxBlocks) blocks = maxBlocks;
        scaleMatrix<<<blocks, threadsPerBlock>>>(data->P, numEntries, scale);
        NNP_CUDA_CHECK(cudaGetLastError());
    }
    addToDiagonal<<<(n + threadsPerBlock - 1) / threadsPerBlock,
                    threadsPerBlock>>>(data->P, n, q);
    NNP_CUDA_CHECK(cudaGetLastError());
}

int GpuKalmanFilter::getDevice() const
{
    return data->device;
}

}
