// Sanity check for the multi-GPU rank-binding scheme used by
// run_qeq_multigpu_wallclock.slurm: prints which physical GPU (PCI bus
// ID) this process's CUDA runtime actually sees as "device 0" after
// CUDA_VISIBLE_DEVICES has been set per-rank by the mpirun wrapper, along
// with the MPI local rank env var that drove the assignment. Confirms
// the binding before trusting it in a real (expensive) training run.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main()
{
    char const* localRank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    char const* visible = getenv("CUDA_VISIBLE_DEVICES");
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        printf("localRank=%s CUDA_VISIBLE_DEVICES=%s cudaGetDeviceProperties FAILED: %s\n",
               localRank ? localRank : "(unset)",
               visible ? visible : "(unset)",
               cudaGetErrorString(err));
        return 1;
    }
    printf("localRank=%s CUDA_VISIBLE_DEVICES=%s -> device0 name=%s "
           "pciBusID=%d pciDeviceID=%d pciDomainID=%d\n",
           localRank ? localRank : "(unset)",
           visible ? visible : "(unset)",
           prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
    return 0;
}
