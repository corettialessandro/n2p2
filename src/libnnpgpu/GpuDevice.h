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

#include <string> // std::string

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
/** Index of the CUDA device currently used by this process.
 */
int         gpuCurrentDevice();
/** Name of a CUDA device, e.g. "NVIDIA A100-SXM-64GB".
 *
 * @param[in] device Device index.
 *
 * Throws std::runtime_error for an invalid index.
 */
std::string gpuDeviceName(int device);

}

#endif
