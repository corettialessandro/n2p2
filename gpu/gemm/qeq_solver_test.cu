// Validates the planned GpuQeqSolver.h/.cu design: replacing
// Structure::AConstrainedQr (Eigen::ColPivHouseholderQR) with a
// cuSOLVER LU factorize (cusolverDnDgetrf) + solve (cusolverDnDgetrs),
// against real AConstrained/bConstrained matrices dumped from an actual
// nnp::Structure::calculateElectrostaticEnergy() call on temp/H2O_4G (real
// periodic 630-atom structures, real hardness/geometry from an actual
// nnp-train stage-1 run) -- see src/libnnp/Structure.cpp's TEMPORARY
// N2P2_DUMP_QEQ dump (reverted after use) and
// temp/4g_profile/run_qeq_dump.slurm. Not synthetic data.
//
// This mirrors this session's earlier CPU-only QR-vs-LU check (Eigen
// ColPivHouseholderQR vs PartialPivLU, relative error ~1e-15 on the same
// 60 real structures) but through the ACTUAL cuSOLVER API the production
// port will use, not just an Eigen stand-in.
//
// AConstrained is dumped column-major (Eigen::MatrixXd's native layout),
// matching cusolverDnDgetrf's expected layout directly -- no transpose
// needed.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t st = (call); \
    if (st != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error at %s:%d: status=%d\n", __FILE__, \
                __LINE__, (int)st); \
        exit(1); \
    } \
} while (0)

namespace
{

vector<char> readFile(string const& path)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Could not open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long const sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<char> buf(sz);
    if (sz > 0) { size_t const nread = fread(buf.data(), 1, sz, f); (void)nread; }
    fclose(f);
    return buf;
}

template <typename T>
vector<T> readTyped(string const& path)
{
    vector<char> raw = readFile(path);
    size_t const n = raw.size() / sizeof(T);
    vector<T> out(n);
    memcpy(out.data(), raw.data(), raw.size());
    return out;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "USAGE: %s <fixture_root_dir>\n", argv[0]);
        return 1;
    }
    string const root = argv[1];

    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    double worstRelErr = 0.0;
    double worstResidQr = 0.0;
    double worstResidLu = 0.0;

    for (int fixtureIdx = 0; fixtureIdx < 5; ++fixtureIdx)
    {
        string const dir = root + "/" + to_string(fixtureIdx);
        size_t n, structureIndex;
        {
            FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
            if (!f) { fprintf(stderr, "Could not open %s/meta.txt\n", dir.c_str()); return 1; }
            if (fscanf(f, "%zu %zu", &n, &structureIndex) != 2)
            {
                fprintf(stderr, "Could not parse meta.txt\n");
                return 1;
            }
            fclose(f);
        }

        vector<double> A = readTyped<double>(dir + "/AConstrained.bin");
        vector<double> b = readTyped<double>(dir + "/bConstrained.bin");
        vector<double> Qref = readTyped<double>(dir + "/Q_ref.bin");

        double* d_A;
        double* d_B;
        int* d_Ipiv;
        int* d_info;
        CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ipiv, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, b.data(), n * sizeof(double),
                             cudaMemcpyHostToDevice));

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, (int)n, (int)n,
                                                    d_A, (int)n, &lwork));
        double* d_work;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

        CUSOLVER_CHECK(cusolverDnDgetrf(handle, (int)n, (int)n, d_A, (int)n,
                                         d_work, d_Ipiv, d_info));
        int info = 0;
        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                             cudaMemcpyDeviceToHost));
        if (info != 0)
        {
            fprintf(stderr, "getrf failed for fixture %d: info=%d\n",
                    fixtureIdx, info);
            return 1;
        }

        CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, (int)n, 1,
                                         d_A, (int)n, d_Ipiv, d_B, (int)n,
                                         d_info));
        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                             cudaMemcpyDeviceToHost));
        if (info != 0)
        {
            fprintf(stderr, "getrs failed for fixture %d: info=%d\n",
                    fixtureIdx, info);
            return 1;
        }

        vector<double> Qlu(n);
        CUDA_CHECK(cudaMemcpy(Qlu.data(), d_B, n * sizeof(double),
                             cudaMemcpyDeviceToHost));

        // Relative error vs Eigen ColPivHouseholderQR's Q_ref, and
        // residuals of both against the original (pre-factorize) A/b --
        // A was overwritten by getrf, so re-read it for the residual.
        vector<double> Aorig = readTyped<double>(dir + "/AConstrained.bin");
        double diffNorm = 0.0, refNorm = 0.0;
        double residLuNorm = 0.0, bNorm = 0.0;
        vector<double> residLu(n, 0.0);
        for (size_t i = 0; i < n; ++i)
        {
            double const d = Qlu[i] - Qref[i];
            diffNorm += d * d;
            refNorm += Qref[i] * Qref[i];
            double rowSum = 0.0;
            for (size_t j = 0; j < n; ++j)
            {
                // Aorig is column-major: A(i,j) = Aorig[i + j*n]
                rowSum += Aorig[i + j * n] * Qlu[j];
            }
            residLu[i] = rowSum - b[i];
            residLuNorm += residLu[i] * residLu[i];
            bNorm += b[i] * b[i];
        }
        double residQrNorm = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double rowSum = 0.0;
            for (size_t j = 0; j < n; ++j)
            {
                rowSum += Aorig[i + j * n] * Qref[j];
            }
            double const r = rowSum - b[i];
            residQrNorm += r * r;
        }

        double const relErr = sqrt(diffNorm) / sqrt(refNorm);
        double const residQr = sqrt(residQrNorm) / sqrt(bNorm);
        double const residLuRel = sqrt(residLuNorm) / sqrt(bNorm);

        printf("fixture=%d structureIndex=%zu n=%zu relErrQ=%.6e "
               "residQr=%.6e residLu=%.6e\n",
               fixtureIdx, structureIndex, n, relErr, residQr, residLuRel);

        worstRelErr = max(worstRelErr, relErr);
        worstResidQr = max(worstResidQr, residQr);
        worstResidLu = max(worstResidLu, residLuRel);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Ipiv);
        cudaFree(d_info);
        cudaFree(d_work);
    }

    printf("\nWorst across all fixtures: relErrQ=%.6e residQr=%.6e "
           "residLu=%.6e\n", worstRelErr, worstResidQr, worstResidLu);

    cusolverDnDestroy(handle);
    return 0;
}
