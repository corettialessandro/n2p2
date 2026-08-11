// n2p2 - A neural network potential package
//
// Implementation of GpuQeqSolver.h -- see that file's header comment for
// the full design rationale (why a per-structureId persistent device
// buffer cache rather than a Structure-owned handle, why refactorization
// is unconditional every call unlike GpuForces' topology cache, why
// batching is required for calculateDQdChi()/calculateDQdJ()).

#include "GpuQeqSolver.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nnp
{

namespace
{

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        throw std::runtime_error(std::string("ERROR: CUDA error in ") \
            + __FILE__ + ":" + std::to_string(__LINE__) + ": " \
            + cudaGetErrorString(err_) + "\n"); \
    } \
} while (0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t st_ = (call); \
    if (st_ != CUSOLVER_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("ERROR: cuSOLVER error in ") \
            + __FILE__ + ":" + std::to_string(__LINE__) \
            + ": status=" + std::to_string((int)st_) + "\n"); \
    } \
} while (0)

struct QeqState
{
    int n = 0;
    int nrhsCap = 0;
    int lwork = 0;
    double* d_A = nullptr;
    int* d_Ipiv = nullptr;
    int* d_info = nullptr;
    double* d_work = nullptr;
    double* d_B = nullptr;
    bool factorized = false;
};

cusolverDnHandle_t& handle()
{
    static cusolverDnHandle_t h = nullptr;
    if (h == nullptr)
    {
        CUSOLVER_CHECK(cusolverDnCreate(&h));
    }
    return h;
}

std::unordered_map<std::size_t, QeqState>& states()
{
    static std::unordered_map<std::size_t, QeqState> s;
    return s;
}

void ensureFactorBuffers(QeqState& st, int n)
{
    if (st.n != n)
    {
        if (st.d_A != nullptr) cudaFree(st.d_A);
        if (st.d_Ipiv != nullptr) cudaFree(st.d_Ipiv);
        if (st.d_info == nullptr) CUDA_CHECK(cudaMalloc(&st.d_info, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&st.d_A, (std::size_t)n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&st.d_Ipiv, (std::size_t)n * sizeof(int)));

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle(), n, n, st.d_A,
                                                    n, &lwork));
        if (st.d_work != nullptr) cudaFree(st.d_work);
        CUDA_CHECK(cudaMalloc(&st.d_work, (std::size_t)lwork * sizeof(double)));
        st.lwork = lwork;
        st.n = n;
        st.nrhsCap = 0;
        if (st.d_B != nullptr) { cudaFree(st.d_B); st.d_B = nullptr; }
    }
}

void ensureSolveBuffer(QeqState& st, int nrhs)
{
    if (st.nrhsCap < nrhs)
    {
        if (st.d_B != nullptr) cudaFree(st.d_B);
        CUDA_CHECK(cudaMalloc(&st.d_B,
                               (std::size_t)st.n * nrhs * sizeof(double)));
        st.nrhsCap = nrhs;
    }
}

} // anonymous namespace

void gpuQeqFactorize(std::size_t structureId, int n,
                      double const* AConstrained)
{
    QeqState& st = states()[structureId];
    ensureFactorBuffers(st, n);

    CUDA_CHECK(cudaMemcpy(st.d_A, AConstrained,
                           (std::size_t)n * n * sizeof(double),
                           cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnDgetrf(handle(), n, n, st.d_A, n, st.d_work,
                                     st.d_Ipiv, st.d_info));
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, st.d_info, sizeof(int),
                           cudaMemcpyDeviceToHost));
    if (info != 0)
    {
        throw std::runtime_error(
            "ERROR: gpuQeqFactorize: cusolverDnDgetrf reported a singular "
            "or invalid AConstrained matrix (info=" + std::to_string(info)
            + ") for structureId=" + std::to_string(structureId) + ".\n");
    }
    st.factorized = true;
}

void gpuQeqSolve(std::size_t structureId, int n, int nrhs,
                  double const* B, double* X)
{
    auto it = states().find(structureId);
    if (it == states().end() || !it->second.factorized || it->second.n != n)
    {
        throw std::runtime_error(
            "ERROR: gpuQeqSolve: no matching gpuQeqFactorize() call for "
            "structureId=" + std::to_string(structureId) + ".\n");
    }
    QeqState& st = it->second;
    ensureSolveBuffer(st, nrhs);

    CUDA_CHECK(cudaMemcpy(st.d_B, B, (std::size_t)n * nrhs * sizeof(double),
                           cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnDgetrs(handle(), CUBLAS_OP_N, n, nrhs, st.d_A,
                                     n, st.d_Ipiv, st.d_B, n, st.d_info));
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, st.d_info, sizeof(int),
                           cudaMemcpyDeviceToHost));
    if (info != 0)
    {
        throw std::runtime_error(
            "ERROR: gpuQeqSolve: cusolverDnDgetrs failed (info="
            + std::to_string(info) + ") for structureId="
            + std::to_string(structureId) + ".\n");
    }

    CUDA_CHECK(cudaMemcpy(X, st.d_B, (std::size_t)n * nrhs * sizeof(double),
                           cudaMemcpyDeviceToHost));
}

}
