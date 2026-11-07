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

#include "GpuEwaldOperator.h"
#include "GpuDevice.h"
#include "GpuError.h"
#include <cublas_v2.h>
#include <stdexcept> // std::runtime_error

using namespace std;

namespace
{

// Values of the local atoms: vl_i = v[index_i].
__global__ void gatherLocal(int           n,
                            int const*    index,
                            double const* v,
                            double*       vl)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vl[i] = v[index[i]];
}

// w = 2 c_k s_k for the real and the imaginary parts.
__global__ void scaleStructureFactors(int           numK,
                                      double const* kcoeff,
                                      double const* s,
                                      double*       w)
{
    int const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numK) return;
    w[k] = 2.0 * kcoeff[k] * s[k];
    w[numK + k] = 2.0 * kcoeff[k] * s[numK + k];
}

/* One warp per local atom i: result_i += diagonal_i vl_i + sum of the
 * real-space terms of row i (lanes strided over the terms, then summed in a
 * fixed order: deterministic).
 */
__global__ void addRealSpace(int           n,
                             int const*    rowOffset,
                             int const*    column,
                             double const* value,
                             double const* diagonal,
                             double const* v,
                             double const* vl,
                             double*       result)
{
    int const i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    if (i >= n) return;
    double sum = 0.0;
    for (int t = rowOffset[i] + lane; t < rowOffset[i + 1]; t += 32)
    {
        sum += value[t] * v[column[t]];
    }
    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) result[i] += diagonal[i] * vl[i] + sum;
}

/// Device buffer that grows when needed.
struct Buffer
{
    void*  p;
    size_t bytes;

    Buffer() : p(nullptr), bytes(0) {}
    void ensure(size_t required)
    {
        if (required <= bytes) return;
        cudaFree(p);
        p = nullptr;
        bytes = 0;
        NNP_CUDA_CHECK(cudaMalloc(&p, required));
        bytes = required;
    }
    template<typename T> T* get() const { return static_cast<T*>(p); }
};

}

namespace nnp
{

struct GpuEwaldOperator::Data
{
    int            device;
    cublasHandle_t cublas;
    size_t         numK;
    size_t         numLocal;
    size_t         numGlobal;
    bool           ready;
    Buffer         cosTable;
    Buffer         sinTable;
    Buffer         kcoeff;
    Buffer         localIndex;
    Buffer         diagonal;
    Buffer         rowOffset;
    Buffer         column;
    Buffer         value;
    Buffer         v;
    Buffer         vl;
    Buffer         s;
    Buffer         w;
    Buffer         result;
};

GpuEwaldOperator::GpuEwaldOperator() : data(new Data())
{
    Data& d = *data;
    d.device = gpuCurrentDevice();
    d.numK = 0;
    d.numLocal = 0;
    d.numGlobal = 0;
    d.ready = false;
    try
    {
        NNP_CUBLAS_CHECK(cublasCreate(&d.cublas));
    }
    catch (...)
    {
        delete data;
        throw;
    }
}

GpuEwaldOperator::~GpuEwaldOperator()
{
    // No checks: the destructor must not throw.
    Data& d = *data;
    cudaSetDevice(d.device);
    cublasDestroy(d.cublas);
    for (Buffer* b : {&d.cosTable, &d.sinTable, &d.kcoeff, &d.localIndex,
                      &d.diagonal, &d.rowOffset, &d.column, &d.value, &d.v,
                      &d.vl, &d.s, &d.w, &d.result})
    {
        cudaFree(b->p);
    }
    delete data;
}

void GpuEwaldOperator::setup(size_t                numK,
                             size_t                numLocal,
                             size_t                numGlobal,
                             double const*         cosTable,
                             double const*         sinTable,
                             size_t                rowStride,
                             double const*         kcoeff,
                             vector<int> const&    localIndex,
                             vector<double> const& diagonal,
                             vector<int> const&    rowOffset,
                             vector<int> const&    column,
                             vector<double> const& value)
{
    Data& d = *data;
    d.ready = false;
    if (localIndex.size() != numLocal ||
        diagonal.size() != numLocal ||
        rowOffset.size() != numLocal + 1 ||
        rowOffset.front() != 0 ||
        static_cast<size_t>(rowOffset.back()) != column.size() ||
        column.size() != value.size() ||
        (numK > 0 && numLocal > rowStride))
    {
        throw runtime_error("ERROR: Inconsistent input for the GPU Ewald "
                            "operator.\n");
    }
    for (int i : localIndex)
    {
        if (i < 0 || static_cast<size_t>(i) >= numGlobal)
        {
            throw runtime_error("ERROR: Local atom index out of range for "
                                "the GPU Ewald operator.\n");
        }
    }
    for (int j : column)
    {
        if (j < 0 || static_cast<size_t>(j) >= numGlobal)
        {
            throw runtime_error("ERROR: Neighbor index out of range for the "
                                "GPU Ewald operator.\n");
        }
    }
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    size_t const nt = numK * numLocal;
    d.cosTable.ensure(nt * sizeof(double) + 1);
    d.sinTable.ensure(nt * sizeof(double) + 1);
    d.kcoeff.ensure(numK * sizeof(double) + 1);
    d.localIndex.ensure(numLocal * sizeof(int) + 1);
    d.diagonal.ensure(numLocal * sizeof(double) + 1);
    d.rowOffset.ensure((numLocal + 1) * sizeof(int));
    d.column.ensure(column.size() * sizeof(int) + 1);
    d.value.ensure(value.size() * sizeof(double) + 1);
    d.v.ensure(numGlobal * sizeof(double) + 1);
    d.vl.ensure(numLocal * sizeof(double) + 1);
    d.s.ensure(2 * numK * sizeof(double) + 1);
    d.w.ensure(2 * numK * sizeof(double) + 1);
    d.result.ensure(numLocal * sizeof(double) + 1);
    if (nt > 0)
    {
        // Column k of the device matrices (numLocal x numK, column-major) is
        // row k of the tables.
        NNP_CUDA_CHECK(cudaMemcpy2D(d.cosTable.p, numLocal * sizeof(double),
                                    cosTable, rowStride * sizeof(double),
                                    numLocal * sizeof(double), numK,
                                    cudaMemcpyHostToDevice));
        NNP_CUDA_CHECK(cudaMemcpy2D(d.sinTable.p, numLocal * sizeof(double),
                                    sinTable, rowStride * sizeof(double),
                                    numLocal * sizeof(double), numK,
                                    cudaMemcpyHostToDevice));
    }
    auto copy = [](Buffer& b, void const* host, size_t bytes)
    {
        if (bytes > 0)
        {
            NNP_CUDA_CHECK(cudaMemcpy(b.p, host, bytes,
                                      cudaMemcpyHostToDevice));
        }
    };
    copy(d.kcoeff, kcoeff, numK * sizeof(double));
    copy(d.localIndex, localIndex.data(), numLocal * sizeof(int));
    copy(d.diagonal, diagonal.data(), numLocal * sizeof(double));
    copy(d.rowOffset, rowOffset.data(), (numLocal + 1) * sizeof(int));
    copy(d.column, column.data(), column.size() * sizeof(int));
    copy(d.value, value.data(), value.size() * sizeof(double));
    d.numK = numK;
    d.numLocal = numLocal;
    d.numGlobal = numGlobal;
    d.ready = true;
}

void GpuEwaldOperator::structureFactors(double const* v, double* s)
{
    Data& d = *data;
    if (!d.ready)
    {
        throw runtime_error("ERROR: GPU Ewald operator used before setup.\n");
    }
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    if (d.numGlobal > 0)
    {
        NNP_CUDA_CHECK(cudaMemcpy(d.v.p, v, d.numGlobal * sizeof(double),
                                  cudaMemcpyHostToDevice));
    }
    if (d.numLocal > 0)
    {
        gatherLocal<<<(d.numLocal + 127) / 128, 128>>>(d.numLocal,
                                                       d.localIndex.get<int>(),
                                                       d.v.get<double>(),
                                                       d.vl.get<double>());
        NNP_CUDA_CHECK(cudaGetLastError());
    }
    if (d.numK == 0) return;
    if (d.numLocal == 0)
    {
        for (size_t k = 0; k < 2 * d.numK; ++k) s[k] = 0.0;
        return;
    }
    double const one = 1.0;
    double const zero = 0.0;
    int const m = d.numLocal;
    int const n = d.numK;
    NNP_CUBLAS_CHECK(cublasDgemv(d.cublas, CUBLAS_OP_T, m, n, &one,
                                 d.cosTable.get<double>(), m,
                                 d.vl.get<double>(), 1, &zero,
                                 d.s.get<double>(), 1));
    NNP_CUBLAS_CHECK(cublasDgemv(d.cublas, CUBLAS_OP_T, m, n, &one,
                                 d.sinTable.get<double>(), m,
                                 d.vl.get<double>(), 1, &zero,
                                 d.s.get<double>() + n, 1));
    NNP_CUDA_CHECK(cudaMemcpy(s, d.s.p, 2 * d.numK * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

void GpuEwaldOperator::apply(double const* s, double* result)
{
    Data& d = *data;
    if (!d.ready)
    {
        throw runtime_error("ERROR: GPU Ewald operator used before setup.\n");
    }
    if (d.numLocal == 0) return;
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    int const m = d.numLocal;
    int const n = d.numK;
    NNP_CUDA_CHECK(cudaMemset(d.result.p, 0, m * sizeof(double)));
    if (n > 0)
    {
        NNP_CUDA_CHECK(cudaMemcpy(d.s.p, s, 2 * n * sizeof(double),
                                  cudaMemcpyHostToDevice));
        scaleStructureFactors<<<(n + 127) / 128, 128>>>(n,
                                                        d.kcoeff.get<double>(),
                                                        d.s.get<double>(),
                                                        d.w.get<double>());
        NNP_CUDA_CHECK(cudaGetLastError());
        double const one = 1.0;
        NNP_CUBLAS_CHECK(cublasDgemv(d.cublas, CUBLAS_OP_N, m, n, &one,
                                     d.cosTable.get<double>(), m,
                                     d.w.get<double>(), 1, &one,
                                     d.result.get<double>(), 1));
        NNP_CUBLAS_CHECK(cublasDgemv(d.cublas, CUBLAS_OP_N, m, n, &one,
                                     d.sinTable.get<double>(), m,
                                     d.w.get<double>() + n, 1, &one,
                                     d.result.get<double>(), 1));
    }
    addRealSpace<<<(m * 32 + 127) / 128, 128>>>(m,
                                                d.rowOffset.get<int>(),
                                                d.column.get<int>(),
                                                d.value.get<double>(),
                                                d.diagonal.get<double>(),
                                                d.v.get<double>(),
                                                d.vl.get<double>(),
                                                d.result.get<double>());
    NNP_CUDA_CHECK(cudaGetLastError());
    NNP_CUDA_CHECK(cudaMemcpy(result, d.result.p, m * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

size_t GpuEwaldOperator::memoryUsage(size_t numK,
                                     size_t numLocal,
                                     size_t numGlobal,
                                     size_t numTerms)
{
    return (2 * numK * numLocal + 5 * numK + 3 * numLocal + numGlobal
            + numTerms) * sizeof(double)
           + (2 * numLocal + 1 + numTerms) * sizeof(int);
}

}
