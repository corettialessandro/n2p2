// Diagnoses the "no CUDA-capable device is detected" failure seen in
// run_fullscale_gpu4.slurm's real nnp-train run (which uses 4 separate
// per-GPU MPS daemons + a CUDA_VISIBLE_DEVICES rank-binding wrapper).
// The earlier gpu_bind_check.cu sanity check passed WITHOUT MPS
// involved -- this program repeats that check but also tries the
// specific things GpuQeqSolver.cu does (cudaMalloc, cusolverDnCreate)
// to isolate whether MPS itself, not just CUDA_VISIBLE_DEVICES binding,
// is the problem.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>

int main()
{
    char const* localRank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    char const* visible = getenv("CUDA_VISIBLE_DEVICES");
    char const* pipeDir = getenv("CUDA_MPS_PIPE_DIRECTORY");
    char const* logDir = getenv("CUDA_MPS_LOG_DIRECTORY");

    printf("localRank=%s CUDA_VISIBLE_DEVICES=%s CUDA_MPS_PIPE_DIRECTORY=%s "
           "CUDA_MPS_LOG_DIRECTORY=%s\n",
           localRank ? localRank : "(unset)",
           visible ? visible : "(unset)",
           pipeDir ? pipeDir : "(unset)",
           logDir ? logDir : "(unset)");
    fflush(stdout);

    int count = -1;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("localRank=%s cudaGetDeviceCount -> err=%s count=%d\n",
           localRank ? localRank : "(unset)", cudaGetErrorString(err), count);
    fflush(stdout);
    if (err != cudaSuccess) return 1;

    int device = -1;
    err = cudaGetDevice(&device);
    printf("localRank=%s cudaGetDevice -> err=%s device=%d\n",
           localRank ? localRank : "(unset)", cudaGetErrorString(err), device);
    fflush(stdout);

    void* p = nullptr;
    err = cudaMalloc(&p, sizeof(int));
    printf("localRank=%s cudaMalloc -> err=%s\n",
           localRank ? localRank : "(unset)", cudaGetErrorString(err));
    fflush(stdout);
    if (err != cudaSuccess) return 1;
    cudaFree(p);

    cusolverDnHandle_t h;
    cusolverStatus_t st = cusolverDnCreate(&h);
    printf("localRank=%s cusolverDnCreate -> status=%d\n",
           localRank ? localRank : "(unset)", (int)st);
    fflush(stdout);
    if (st == CUSOLVER_STATUS_SUCCESS) cusolverDnDestroy(h);

    return 0;
}
