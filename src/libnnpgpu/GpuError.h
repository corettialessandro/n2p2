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

#ifndef GPUERROR_H
#define GPUERROR_H

// Error checks for CUDA, cuBLAS and cuSOLVER calls, for use in .cu files
// only. A failed call throws std::runtime_error, like every other error in
// n2p2, instead of terminating the process from inside the library: callers
// (applications, the LAMMPS interface, the tests) handle GPU errors the same
// way as all others.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept> // std::runtime_error
#include <string>    // std::string, std::to_string

namespace nnp
{

/** Throw a std::runtime_error describing a failed GPU library call.
 *
 * @param[in] call Source text of the call.
 * @param[in] file Source file of the call.
 * @param[in] line Source line of the call.
 * @param[in] error Error description of the library.
 */
inline void throwGpuError(char const*        call,
                          char const*        file,
                          int                line,
                          std::string const& error)
{
    throw std::runtime_error("ERROR: GPU call " + std::string(call)
                             + " failed at " + std::string(file) + ":"
                             + std::to_string(line) + ": " + error + "\n");
}

}

#define NNP_CUDA_CHECK(call)                                                  \
    do                                                                        \
    {                                                                         \
        cudaError_t const nnpCudaError = (call);                              \
        if (nnpCudaError != cudaSuccess)                                      \
        {                                                                     \
            nnp::throwGpuError(#call, __FILE__, __LINE__,                     \
                               cudaGetErrorString(nnpCudaError));             \
        }                                                                     \
    } while (0)

#define NNP_CUBLAS_CHECK(call)                                                \
    do                                                                        \
    {                                                                         \
        cublasStatus_t const nnpCublasStatus = (call);                        \
        if (nnpCublasStatus != CUBLAS_STATUS_SUCCESS)                         \
        {                                                                     \
            nnp::throwGpuError(#call, __FILE__, __LINE__,                     \
                               cublasGetStatusString(nnpCublasStatus));       \
        }                                                                     \
    } while (0)

#define NNP_CUSOLVER_CHECK(call)                                              \
    do                                                                        \
    {                                                                         \
        cusolverStatus_t const nnpCusolverStatus = (call);                    \
        if (nnpCusolverStatus != CUSOLVER_STATUS_SUCCESS)                     \
        {                                                                     \
            nnp::throwGpuError(#call, __FILE__, __LINE__,                     \
                               "cuSOLVER status "                             \
                               + std::to_string(nnpCusolverStatus));          \
        }                                                                     \
    } while (0)

#endif
