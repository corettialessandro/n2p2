// Validates the planned GPU port of Mode::calculateForces()'s HDNNP_4G
// block (the O(numAtoms^2 * avgNeighbors) dChidr double loop, confirmed by
// direct profiling to be the dominant stage-2 cost -- ~148ms/call vs the
// two calculateForceLambdaTotal()/Elec() solves at <1ms/call combined, see
// gpu/README.md's 4G electrostatics section).
//
// Design: lambdaElec(j) (constant per owner atom j) is pre-multiplied into
// dChidG_j[k] once, turning the whole computation into the exact same
// self-term + owner-centric-edge-list shape src/libnnpgpu/GpuForces.cu
// already uses for the 2G short-range force port -- see this file's
// weightKernel/selfKernel/edgeKernel below. The remaining dAdrQ term is a
// separate dense O(numAtoms^2) reduction, embarrassingly parallel
// (one thread per atom, no cross-atom writes).
//
// Ground truth: real fElec values dumped directly from a real
// nnp::Mode::calculateForces() HDNNP_4G run on temp/H2O_4G (real periodic
// 630-atom structure, real charges/weights/hardness from an actual
// nnp-train stage-2 run) -- see src/libnnp/Mode.cpp's TEMPORARY
// N2P2_DUMP_4G_ELECFORCES dump (reverted after use) and
// temp/4g_profile/run_elecforces_dump.slurm. Not synthetic data.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

namespace
{

__global__ void weightKernel(int numValues, int const* ownerAtom,
                              double const* dChidG, double const* lambda,
                              double* weighted)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numValues) return;
    weighted[v] = lambda[ownerAtom[v]] * dChidG[v];
}

// One thread per atom: initializes force[i] = -pEelecpr[i] - self term.
__global__ void selfKernel(int numAtoms, int const* dChidGOffset,
                            double const* weighted, double const* dGdrSelf,
                            double const* pEelecpr, double* force)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    int const begin = dChidGOffset[i];
    int const end = dChidGOffset[i + 1];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int k = begin; k < end; ++k)
    {
        double const w = weighted[k];
        fx += w * dGdrSelf[3 * k + 0];
        fy += w * dGdrSelf[3 * k + 1];
        fz += w * dGdrSelf[3 * k + 2];
    }
    force[3 * i + 0] = -pEelecpr[3 * i + 0] - fx;
    force[3 * i + 1] = -pEelecpr[3 * i + 1] - fy;
    force[3 * i + 2] = -pEelecpr[3 * i + 2] - fz;
}

// One thread per atom: adds the dense -sum_j lambda(j)*dAdrQ[i][j] term.
// Runs after selfKernel (sequential kernel launches on the default
// stream), no atomics needed -- disjoint i across threads.
__global__ void denseDAdrQKernel(int numAtoms, double const* dAdrQ,
                                  double const* lambda, double* force)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < numAtoms; ++j)
    {
        double const l = lambda[j];
        size_t const idx = 3 * ((size_t)i * numAtoms + j);
        fx += l * dAdrQ[idx + 0];
        fy += l * dAdrQ[idx + 1];
        fz += l * dAdrQ[idx + 2];
    }
    force[3 * i + 0] -= fx;
    force[3 * i + 1] -= fy;
    force[3 * i + 2] -= fz;
}

// One thread per edge: scatter-add -weighted[edgeOwnerIndex[e]]*edgeDGdr[e]
// into the TARGET atom's force accumulator (atomicAdd, many edges can
// target the same atom).
__global__ void edgeKernel(int numEdges, int const* edgeTarget,
                            int const* edgeOwnerIndex, double const* edgeDGdr,
                            double const* weighted, double* force)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numEdges) return;

    int const target = edgeTarget[e];
    double const w = weighted[edgeOwnerIndex[e]];
    atomicAdd(&force[3 * target + 0], -w * edgeDGdr[3 * e + 0]);
    atomicAdd(&force[3 * target + 1], -w * edgeDGdr[3 * e + 1]);
    atomicAdd(&force[3 * target + 2], -w * edgeDGdr[3 * e + 2]);
}

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
        fprintf(stderr, "USAGE: %s <fixture_dir>\n", argv[0]);
        return 1;
    }
    string const dir = argv[1];

    size_t numAtoms;
    int numValues, numEdges;
    {
        FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
        if (!f) { fprintf(stderr, "Could not open meta.txt\n"); return 1; }
        if (fscanf(f, "%zu %d %d", &numAtoms, &numValues, &numEdges) != 3)
        {
            fprintf(stderr, "Could not parse meta.txt\n");
            return 1;
        }
        fclose(f);
    }
    printf("Fixture: numAtoms=%zu numValues=%d numEdges=%d\n",
           numAtoms, numValues, numEdges);

    vector<int> dChidGOffset = readTyped<int>(dir + "/dChidGOffset.bin");
    vector<double> dChidG = readTyped<double>(dir + "/dChidG.bin");
    vector<double> dGdrSelf = readTyped<double>(dir + "/dGdrSelf.bin");
    vector<int> edgeTarget = readTyped<int>(dir + "/edgeTarget.bin");
    vector<int> edgeOwnerIndex = readTyped<int>(dir + "/edgeOwnerIndex.bin");
    vector<double> edgeDGdr = readTyped<double>(dir + "/edgeDGdr.bin");
    vector<double> dAdrQ = readTyped<double>(dir + "/dAdrQ.bin");
    vector<double> pEelecpr = readTyped<double>(dir + "/pEelecpr.bin");
    vector<double> lambdaElec = readTyped<double>(dir + "/lambdaElec.bin");
    vector<double> fElecRef = readTyped<double>(dir + "/fElec_ref.bin");

    // Build ownerAtom[v]: which atom's CSR range flat index v falls into.
    vector<int> ownerAtom(numValues);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        for (int v = dChidGOffset[i]; v < dChidGOffset[i + 1]; ++v)
        {
            ownerAtom[v] = (int)i;
        }
    }

    int *d_dChidGOffset, *d_ownerAtom, *d_edgeTarget, *d_edgeOwnerIndex;
    double *d_dChidG, *d_dGdrSelf, *d_edgeDGdr, *d_dAdrQ, *d_pEelecpr;
    double *d_lambdaElec, *d_weighted, *d_force;

    CUDA_CHECK(cudaMalloc(&d_dChidGOffset, (numAtoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ownerAtom, numValues * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dChidG, numValues * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdrSelf, (size_t)numValues * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_edgeTarget, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edgeOwnerIndex, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edgeDGdr, (size_t)numEdges * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dAdrQ, numAtoms * numAtoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pEelecpr, numAtoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lambdaElec, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weighted, numValues * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_force, numAtoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_dChidGOffset, dChidGOffset.data(),
                         (numAtoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ownerAtom, ownerAtom.data(),
                         numValues * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dChidG, dChidG.data(),
                         numValues * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dGdrSelf, dGdrSelf.data(),
                         (size_t)numValues * 3 * sizeof(double),
                         cudaMemcpyHostToDevice));
    if (numEdges > 0)
    {
        CUDA_CHECK(cudaMemcpy(d_edgeTarget, edgeTarget.data(),
                             numEdges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeOwnerIndex, edgeOwnerIndex.data(),
                             numEdges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeDGdr, edgeDGdr.data(),
                             (size_t)numEdges * 3 * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_dAdrQ, dAdrQ.data(),
                         numAtoms * numAtoms * 3 * sizeof(double),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pEelecpr, pEelecpr.data(),
                         numAtoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lambdaElec, lambdaElec.data(),
                         numAtoms * sizeof(double), cudaMemcpyHostToDevice));

    int const block = 128;
    weightKernel<<<(numValues + block - 1) / block, block>>>(
        numValues, d_ownerAtom, d_dChidG, d_lambdaElec, d_weighted);
    CUDA_CHECK(cudaGetLastError());

    selfKernel<<<(numAtoms + block - 1) / block, block>>>(
        (int)numAtoms, d_dChidGOffset, d_weighted, d_dGdrSelf, d_pEelecpr,
        d_force);
    CUDA_CHECK(cudaGetLastError());

    denseDAdrQKernel<<<(numAtoms + block - 1) / block, block>>>(
        (int)numAtoms, d_dAdrQ, d_lambdaElec, d_force);
    CUDA_CHECK(cudaGetLastError());

    if (numEdges > 0)
    {
        edgeKernel<<<(numEdges + block - 1) / block, block>>>(
            numEdges, d_edgeTarget, d_edgeOwnerIndex, d_edgeDGdr, d_weighted,
            d_force);
        CUDA_CHECK(cudaGetLastError());
    }

    vector<double> force(numAtoms * 3);
    CUDA_CHECK(cudaMemcpy(force.data(), d_force, numAtoms * 3 * sizeof(double),
                         cudaMemcpyDeviceToHost));

    double maxAbsDiff = 0.0, maxRelDiff = 0.0, refNorm = 0.0;
    for (size_t i = 0; i < numAtoms * 3; ++i)
    {
        double const diff = fabs(force[i] - fElecRef[i]);
        maxAbsDiff = max(maxAbsDiff, diff);
        if (fabs(fElecRef[i]) > 1e-12)
        {
            maxRelDiff = max(maxRelDiff, diff / fabs(fElecRef[i]));
        }
        refNorm += fElecRef[i] * fElecRef[i];
    }
    refNorm = sqrt(refNorm);

    printf("fElec reference norm: %.6e\n", refNorm);
    printf("Max abs diff (GPU vs CPU real production data): %.6e\n",
           maxAbsDiff);
    printf("Max rel diff (where |ref| > 1e-12): %.6e\n", maxRelDiff);

    bool const pass = maxAbsDiff < 1e-8;
    printf("%s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
