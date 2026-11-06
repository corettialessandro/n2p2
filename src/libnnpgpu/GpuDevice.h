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

#ifndef GPUDEVICE_H
#define GPUDEVICE_H

// Selection of the CUDA device used by this process. Plain C++ interface,
// included from code compiled without nvcc.

#include <cstddef> // std::size_t
#include <string>  // std::string

namespace nnp
{

/** Number of CUDA devices visible to this process.
 *
 * @return Device count, 0 if there is no device or no CUDA driver.
 */
int         gpuDeviceCount();
/** Rank of this process among the processes on its node.
 *
 * Read from the environment set by the MPI launcher, checking in this order:
 * OMPI_COMM_WORLD_LOCAL_RANK (Open MPI), MPI_LOCALRANKID (MPICH, Intel MPI),
 * MV2_COMM_WORLD_LOCAL_RANK (MVAPICH2) and SLURM_LOCALID (srun).
 *
 * @return Local rank, 0 if none of these variables is set.
 */
int         gpuLocalRank();
/** Number of processes on the node of this process.
 *
 * Read from the environment set by the MPI launcher, checking in this order:
 * OMPI_COMM_WORLD_LOCAL_SIZE (Open MPI), MPI_LOCALNRANKS (MPICH, Intel MPI),
 * MV2_COMM_WORLD_LOCAL_SIZE (MVAPICH2) and SLURM_NTASKS_PER_NODE (srun).
 *
 * @return Local size, 1 if none of these variables is set.
 */
int         gpuLocalSize();
/** Select the CUDA device of this process.
 *
 * Processes on a node are spread over its devices: the device index is the
 * local rank modulo the device count.
 *
 * @return Index of the selected device.
 *
 * Throws std::runtime_error if no device is available.
 */
int         gpuSelectDevice();
/** Make sure this process can run GPU code and select its device.
 *
 * Every component running on the GPU calls this when it is set up, before
 * any expensive work (see src/libnnpgpu/README.md, rule 3). Selects the
 * device with gpuSelectDevice() and checks that it can run the code of this
 * build.
 *
 * @return Index of the selected device.
 *
 * Throws std::runtime_error with an explanation if no device is visible or
 * the device cannot run the compiled code (architecture mismatch).
 */
int         gpuRequireDevice();
/** Index of the CUDA device currently used by this process.
 */
int         gpuCurrentDevice();
/** Device memory a process should use for caches.
 *
 * Half of the memory of its device, shared by the processes of the
 * node using this device (gpuLocalSize() divided by the number of devices,
 * rounded up).
 *
 * @return Memory in bytes.
 */
std::size_t gpuDefaultCacheBudget();
/** Name of a CUDA device, e.g. "NVIDIA A100-SXM-64GB".
 *
 * @param[in] device Device index.
 *
 * Throws std::runtime_error for an invalid index.
 */
std::string gpuDeviceName(int device);

}

#endif
