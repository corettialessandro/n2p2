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

#include "GpuChargeEquilibration.h"
#include "GpuDevice.h"
#include "GpuError.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <atomic>    // std::atomic
#include <cmath>     // std::sqrt, std::ceil
#include <stdexcept> // std::runtime_error
#include <string>    // std::to_string

using namespace std;

namespace
{

int const tile = 16;

/* One thread per matrix element (i, j), column-major with leading dimension
 * n + 1; row and column n hold the constraint. Real-space part only, the
 * reciprocal-space part is added by a matrix product.
 */
__global__ void buildMatrix(int           n,
                            int const*    type,
                            double const* position,
                            double const* hardness,
                            double const* sigma,
                            bool          periodic,
                            double        boxX,
                            double        boxY,
                            double        boxZ,
                            int           imagesX,
                            int           imagesY,
                            int           imagesZ,
                            double        eta,
                            double        realCutoff,
                            double*       matrix)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    int const j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > n || j > n) return;
    size_t const ld = n + 1;
    double* a = matrix + i + j * ld;
    if (i == n || j == n)
    {
        *a = (i == n && j == n) ? 0.0 : 1.0;
        return;
    }
    double const pi = 3.14159265358979323846;
    int const ei = type[i];
    int const ej = type[j];
    double const gamma = sqrt(2.0 * (sigma[ei] * sigma[ei]
                                     + sigma[ej] * sigma[ej]));
    double dx = position[3 * i + 0] - position[3 * j + 0];
    double dy = position[3 * i + 1] - position[3 * j + 1];
    double dz = position[3 * i + 2] - position[3 * j + 2];
    double value = 0.0;
    if (!periodic)
    {
        if (i == j) value = hardness[ei] + 1.0 / (sigma[ei] * sqrt(pi));
        else
        {
            double const r = sqrt(dx * dx + dy * dy + dz * dz);
            value = erf(r / gamma) / r;
        }
        *a = value;
        return;
    }
    // Minimum image, then all images closer than the cutoff.
    dx -= boxX * rint(dx / boxX);
    dy -= boxY * rint(dy / boxY);
    dz -= boxZ * rint(dz / boxZ);
    double const sqrt2eta = sqrt(2.0) * eta;
    double const rc2 = realCutoff * realCutoff;
    for (int nx = -imagesX; nx <= imagesX; ++nx)
    {
        double const x = dx + nx * boxX;
        for (int ny = -imagesY; ny <= imagesY; ++ny)
        {
            double const y = dy + ny * boxY;
            for (int nz = -imagesZ; nz <= imagesZ; ++nz)
            {
                double const z = dz + nz * boxZ;
                double const r2 = x * x + y * y + z * z;
                if (r2 >= rc2 || r2 == 0.0) continue;
                double const r = sqrt(r2);
                value += (erfc(r / sqrt2eta) - erfc(r / gamma)) / r;
            }
        }
    }
    if (i == j)
    {
        value += hardness[ei] + 1.0 / (sigma[ei] * sqrt(pi))
               - 2.0 / (eta * sqrt(2.0 * pi));
    }
    *a = value;
}

/* Reciprocal-space factors: m(i, k) = sqrt(2 c_k) cos(k r_i) and
 * m(i, numK + k) = sqrt(2 c_k) sin(k r_i), column-major (n rows), so that
 * m m^T is the reciprocal-space part of A.
 */
__global__ void buildReciprocal(int           n,
                                int           numK,
                                double const* position,
                                double const* kvector,
                                double const* kcoeff,
                                double*       m)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = blockIdx.y; k < numK; k += gridDim.y)
    {
        double const kr = kvector[3 * k + 0] * position[3 * i + 0]
                        + kvector[3 * k + 1] * position[3 * i + 1]
                        + kvector[3 * k + 2] * position[3 * i + 2];
        double const w = sqrt(2.0 * kcoeff[k]);
        m[i + static_cast<size_t>(k) * n] = w * cos(kr);
        m[i + static_cast<size_t>(numK + k) * n] = w * sin(kr);
    }
}

/// Device memory that grows when needed.
struct Buffer
{
    void*  p;
    size_t bytes;

    Buffer() : p(nullptr), bytes(0) {}
    void ensure(size_t required)
    {
        if (required <= bytes) return;
        if (p != nullptr) NNP_CUDA_CHECK(cudaFree(p));
        p = nullptr;
        bytes = 0;
        NNP_CUDA_CHECK(cudaMalloc(&p, required));
        bytes = required;
    }
    template<typename T> T* get() const { return static_cast<T*>(p); }
};

/// Last factorization identifier handed out (all objects).
atomic<size_t> lastFactorizationId(0);

void upload(Buffer& b, void const* host, size_t bytes)
{
    b.ensure(bytes > 0 ? bytes : 1);
    if (bytes > 0)
    {
        NNP_CUDA_CHECK(cudaMemcpy(b.p, host, bytes, cudaMemcpyHostToDevice));
    }
}

}

namespace nnp
{

struct GpuChargeEquilibration::Data
{
    int                device;
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
    size_t             numAtoms;
    size_t             factorizationId;
    Buffer             type;
    Buffer             position;
    Buffer             hardness;
    Buffer             sigma;
    Buffer             kvector;
    Buffer             kcoeff;
    Buffer             reciprocal;
    Buffer             matrix;
    Buffer             pivot;
    Buffer             work;
    Buffer             info;
    Buffer             rhs;
};

GpuChargeEquilibration::GpuChargeEquilibration() : data(new Data())
{
    Data& d = *data;
    d.device = gpuCurrentDevice();
    d.numAtoms = 0;
    d.factorizationId = 0;
    d.cublas = nullptr;
    d.cusolver = nullptr;
    try
    {
        NNP_CUBLAS_CHECK(cublasCreate(&d.cublas));
        NNP_CUSOLVER_CHECK(cusolverDnCreate(&d.cusolver));
    }
    catch (...)
    {
        if (d.cublas != nullptr) cublasDestroy(d.cublas);
        delete data;
        throw;
    }
}

GpuChargeEquilibration::~GpuChargeEquilibration()
{
    // No checks: the destructor must not throw.
    Data& d = *data;
    cudaSetDevice(d.device);
    for (Buffer* b : {&d.type, &d.position, &d.hardness, &d.sigma,
                      &d.kvector, &d.kcoeff, &d.reciprocal, &d.matrix,
                      &d.pivot, &d.work, &d.info, &d.rhs})
    {
        cudaFree(b->p);
    }
    cusolverDnDestroy(d.cusolver);
    cublasDestroy(d.cublas);
    delete data;
}

size_t GpuChargeEquilibration::memoryUsage(size_t numAtoms,
                                           size_t numK,
                                           size_t numRhs)
{
    return (numAtoms + 1) * (numAtoms + 1) * sizeof(double)
         + 2 * numAtoms * numK * sizeof(double)
         + numAtoms * (3 * sizeof(double) + sizeof(int))
         + (numAtoms + 1) * numRhs * sizeof(double);
}

void GpuChargeEquilibration::factorize(vector<int> const&    type,
                                       vector<double> const& position,
                                       vector<double> const& hardness,
                                       vector<double> const& sigma,
                                       bool                  periodic,
                                       double const*         box,
                                       double                eta,
                                       double                realCutoff,
                                       vector<double> const& kvector,
                                       vector<double> const& kcoeff)
{
    Data& d = *data;
    d.numAtoms = 0;
    d.factorizationId = 0;
    size_t const n = type.size();
    size_t const numK = kcoeff.size();
    if (n == 0 || position.size() != 3 * n || hardness.size() != sigma.size()
        || kvector.size() != 3 * numK)
    {
        throw runtime_error("ERROR: Inconsistent input for the GPU charge "
                            "equilibration.\n");
    }
    for (int t : type)
    {
        if (t < 0 || t >= static_cast<int>(hardness.size()))
        {
            throw runtime_error("ERROR: Unknown type in the GPU charge "
                                "equilibration.\n");
        }
    }
    int imagesX = 0;
    int imagesY = 0;
    int imagesZ = 0;
    if (periodic)
    {
        if (box == nullptr || !(box[0] > 0.0 && box[1] > 0.0 && box[2] > 0.0)
            || !(eta > 0.0) || !(realCutoff > 0.0))
        {
            throw runtime_error("ERROR: Inconsistent periodic input for the "
                                "GPU charge equilibration.\n");
        }
        imagesX = static_cast<int>(ceil(realCutoff / box[0]));
        imagesY = static_cast<int>(ceil(realCutoff / box[1]));
        imagesZ = static_cast<int>(ceil(realCutoff / box[2]));
    }

    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    upload(d.type, type.data(), n * sizeof(int));
    upload(d.position, position.data(), 3 * n * sizeof(double));
    upload(d.hardness, hardness.data(), hardness.size() * sizeof(double));
    upload(d.sigma, sigma.data(), sigma.size() * sizeof(double));
    size_t const ld = n + 1;
    d.matrix.ensure(ld * ld * sizeof(double));
    dim3 const threads(tile, tile);
    dim3 const blocks((ld + tile - 1) / tile, (ld + tile - 1) / tile);
    buildMatrix<<<blocks, threads>>>(n,
                                     d.type.get<int>(),
                                     d.position.get<double>(),
                                     d.hardness.get<double>(),
                                     d.sigma.get<double>(),
                                     periodic,
                                     periodic ? box[0] : 0.0,
                                     periodic ? box[1] : 0.0,
                                     periodic ? box[2] : 0.0,
                                     imagesX,
                                     imagesY,
                                     imagesZ,
                                     eta,
                                     realCutoff,
                                     d.matrix.get<double>());
    NNP_CUDA_CHECK(cudaGetLastError());
    if (periodic && numK > 0)
    {
        upload(d.kvector, kvector.data(), 3 * numK * sizeof(double));
        upload(d.kcoeff, kcoeff.data(), numK * sizeof(double));
        d.reciprocal.ensure(2 * n * numK * sizeof(double));
        dim3 const kblocks((n + 255) / 256, numK < 65535 ? numK : 65535);
        buildReciprocal<<<kblocks, 256>>>(n,
                                          numK,
                                          d.position.get<double>(),
                                          d.kvector.get<double>(),
                                          d.kcoeff.get<double>(),
                                          d.reciprocal.get<double>());
        NNP_CUDA_CHECK(cudaGetLastError());
        double const one = 1.0;
        NNP_CUBLAS_CHECK(cublasDgemm(d.cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_T,
                                     n,
                                     n,
                                     2 * numK,
                                     &one,
                                     d.reciprocal.get<double>(),
                                     n,
                                     d.reciprocal.get<double>(),
                                     n,
                                     &one,
                                     d.matrix.get<double>(),
                                     ld));
    }

    factorizeDeviceMatrix(n);
}

void GpuChargeEquilibration::factorize(size_t numAtoms, double const* matrix)
{
    Data& d = *data;
    d.numAtoms = 0;
    d.factorizationId = 0;
    if (numAtoms == 0 || matrix == nullptr)
    {
        throw runtime_error("ERROR: Inconsistent input for the GPU charge "
                            "equilibration.\n");
    }
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    size_t const ld = numAtoms + 1;
    upload(d.matrix, matrix, ld * ld * sizeof(double));
    factorizeDeviceMatrix(numAtoms);
}

void GpuChargeEquilibration::factorizeDeviceMatrix(size_t numAtoms)
{
    Data& d = *data;
    size_t const ld = numAtoms + 1;
    // LU factorization with partial pivoting (the matrix is symmetric but
    // indefinite because of the constraint).
    int workSize = 0;
    NNP_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(d.cusolver,
                                                   ld,
                                                   ld,
                                                   d.matrix.get<double>(),
                                                   ld,
                                                   &workSize));
    d.work.ensure(static_cast<size_t>(workSize) * sizeof(double) + 1);
    d.pivot.ensure(ld * sizeof(int));
    d.info.ensure(sizeof(int));
    NNP_CUSOLVER_CHECK(cusolverDnDgetrf(d.cusolver,
                                        ld,
                                        ld,
                                        d.matrix.get<double>(),
                                        ld,
                                        d.work.get<double>(),
                                        d.pivot.get<int>(),
                                        d.info.get<int>()));
    int info = 0;
    NNP_CUDA_CHECK(cudaMemcpy(&info, d.info.p, sizeof(int),
                              cudaMemcpyDeviceToHost));
    if (info != 0)
    {
        throw runtime_error("ERROR: GPU charge equilibration matrix is "
                            "singular (getrf info " + to_string(info)
                            + ").\n");
    }
    d.numAtoms = numAtoms;
    d.factorizationId = ++lastFactorizationId;
}

void GpuChargeEquilibration::solve(vector<double> const& b,
                                   double                c,
                                   vector<double>&       x)
{
    Data& d = *data;
    size_t const n = d.numAtoms;
    if (n == 0 || b.size() != n)
    {
        throw runtime_error("ERROR: No GPU charge equilibration matrix for "
                            "this right-hand side.\n");
    }
    vector<double> rhs(b);
    rhs.push_back(c);
    solve(1, rhs.data());
    x.assign(rhs.begin(), rhs.begin() + n);
}

void GpuChargeEquilibration::solve(size_t numRhs, double* rhs)
{
    Data& d = *data;
    size_t const n = d.numAtoms;
    if (n == 0)
    {
        throw runtime_error("ERROR: No GPU charge equilibration matrix for "
                            "this right-hand side.\n");
    }
    if (numRhs == 0) return;
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    size_t const bytes = (n + 1) * numRhs * sizeof(double);
    upload(d.rhs, rhs, bytes);
    NNP_CUSOLVER_CHECK(cusolverDnDgetrs(d.cusolver,
                                        CUBLAS_OP_N,
                                        n + 1,
                                        numRhs,
                                        d.matrix.get<double>(),
                                        n + 1,
                                        d.pivot.get<int>(),
                                        d.rhs.get<double>(),
                                        n + 1,
                                        d.info.get<int>()));
    NNP_CUDA_CHECK(cudaMemcpy(rhs, d.rhs.p, bytes, cudaMemcpyDeviceToHost));
}

size_t GpuChargeEquilibration::getNumAtoms() const
{
    return data->numAtoms;
}

size_t GpuChargeEquilibration::getFactorizationId() const
{
    return data->factorizationId;
}

}
