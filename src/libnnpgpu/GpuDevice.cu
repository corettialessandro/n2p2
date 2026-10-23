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

#include "GpuDevice.h"
#include "GpuError.h"
#include <cstdlib>   // std::getenv, std::atoi
#include <stdexcept> // std::runtime_error
#include <string>    // std::string, std::to_string

using namespace std;

namespace
{

// Empty kernel: loading it shows whether the device can run this build.
__global__ void probeKernel()
{
}

}

namespace nnp
{

int gpuDeviceCount()
{
    int count = 0;
    // No device or no driver: report zero devices instead of an error.
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        cudaGetLastError();
        return 0;
    }
    return count;
}

int gpuLocalRank()
{
    char const* const names[] = {"OMPI_COMM_WORLD_LOCAL_RANK",
                                 "MPI_LOCALRANKID",
                                 "MV2_COMM_WORLD_LOCAL_RANK",
                                 "SLURM_LOCALID"};
    for (char const* name : names)
    {
        char const* value = getenv(name);
        if (value != nullptr) return atoi(value);
    }
    return 0;
}

int gpuSelectDevice()
{
    int const count = gpuDeviceCount();
    if (count == 0)
    {
        throw runtime_error("ERROR: No CUDA device available.\n");
    }
    int const device = gpuLocalRank() % count;
    NNP_CUDA_CHECK(cudaSetDevice(device));
    return device;
}

int gpuRequireDevice()
{
    if (gpuDeviceCount() == 0)
    {
        throw runtime_error("ERROR: This n2p2 was built with GPU support "
                            "(GPU=1), but no CUDA device is visible to this "
                            "process. Run it on a node with a GPU, or rebuild "
                            "n2p2 without GPU=1.\n");
    }
    int const device = gpuSelectDevice();
    cudaFuncAttributes attributes;
    cudaError_t const error = cudaFuncGetAttributes(&attributes, probeKernel);
    if (error != cudaSuccess)
    {
        cudaGetLastError();
        cudaDeviceProp properties;
        NNP_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        throw runtime_error("ERROR: GPU " + to_string(device) + " ("
                            + string(properties.name)
                            + ", compute capability "
                            + to_string(properties.major) + "."
                            + to_string(properties.minor)
                            + ") cannot run the GPU code of this n2p2 build ("
                            + string(cudaGetErrorString(error))
                            + "). Rebuild n2p2 with the matching -arch in "
                            "PROJECT_NVCCFLAGS.\n");
    }
    return device;
}

int gpuCurrentDevice()
{
    int device = 0;
    NNP_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

string gpuDeviceName(int device)
{
    cudaDeviceProp properties;
    NNP_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return string(properties.name);
}

}
