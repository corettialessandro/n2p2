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

using namespace std;

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
