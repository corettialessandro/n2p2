// Standalone GPU smoke test for the gpu-portability effort.
//
// Goal: prove the CUDA toolchain compiles and runs on a Booster A100, and
// that a from-scratch CUDA port of one symmetry function type produces
// bit-close results to the existing CPU implementation. This intentionally
// does not touch n2p2's build system or link against libnnp -- it is a
// standalone reimplementation of the exact math in
// src/libnnp/SymFncExpRad.cpp (radial symmetry function, type 2) and
// src/libnnp/CutoffFunction.cpp's fdfTANHU (cutoff_type 2, the type our
// H2O_2G/input.nn actually uses), for one-thread-per-atom GPU execution --
// the kernel granularity Phase 2 of GPU_PORTING_PLAN.md calls for.
//
// Deliberately out of scope: the final scale()/center() affine transform
// (a trivial post-hoc step, not part of the neighbor-loop math being
// ported) and the full Atom/Structure/Settings machinery (unnecessary
// scaffolding for validating just the numerical kernel).
//
// Validates: unscaled G accumulation ("result" in the CPU source) and its
// derivative w.r.t. the central atom's position (accumulated dGdr), for two
// parameter sets taken directly from temp/H2O_2G/input.nn (H-H radial,
// rs=0 and rs!=0 cases).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

// TANHU cutoff (n2p2 CutoffType::CT_TANHU = 2):
//   fc(r)  = tanh(1 - r/rc)^3
//   dfc(r) = 3*tanh(1-r/rc)^2 * (tanh(1-r/rc)^2 - 1) / rc
// (src/libnnp/CutoffFunction.cpp:156, fdfTANHU)
__host__ __device__ inline void cutoffTANHU(double r, double rcinv,
                                             double& fc, double& dfc)
{
    double t  = tanh(1.0 - r * rcinv);
    double t2 = t * t;
    fc  = t * t2;
    dfc = 3.0 * t2 * (t2 - 1.0) * rcinv;
}

// Radial symmetry function (type 2), one atom's contribution, matching
// SymFncExpRad::calculate() (src/libnnp/SymFncExpRad.cpp:124) up to (not
// including) the final scale() call.
__host__ __device__ inline void symFncExpRad(
    int numNeighbors,
    const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double eta, double rs, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double rcinv = 1.0 / rc;
    double result = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < numNeighbors; ++j)
    {
        double rij = neighDist[j];
        if (rij >= rc) continue;

        double diff = rij - rs;
        double pexp = exp(-eta * diff * diff);

        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);

        result += pexp * pfc;

        // Force/derivative contribution (SymFncExpRad.cpp:167-172).
        double p1 = (pdfc - 2.0 * eta * diff * pfc) * pexp / rij;
        fx += p1 * neighDx[j];
        fy += p1 * neighDy[j];
        fz += p1 * neighDz[j];
    }

    G    = result;
    dGdx = fx;
    dGdy = fy;
    dGdz = fz;
}

__global__ void symFncExpRadKernel(
    int numAtoms,
    const int* neighCount, const int* neighOffset,
    const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double eta, double rs, double rc,
    double* Gout, double* dGdxOut, double* dGdyOut, double* dGdzOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    int off = neighOffset[i];
    int n   = neighCount[i];
    symFncExpRad(n, &neighDist[off], &neighDx[off], &neighDy[off], &neighDz[off],
                 eta, rs, rc,
                 Gout[i], dGdxOut[i], dGdyOut[i], dGdzOut[i]);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

struct TestSystem
{
    int numAtoms;
    std::vector<int> neighCount, neighOffset;
    std::vector<double> neighDist, neighDx, neighDy, neighDz;
};

// Synthetic water-like system: ~630 atoms, ~106 neighbors/atom on average
// (matching the real neighbor statistics measured on H2O_2G, see
// nnp-scaling's "NEIGHBOR STATISTICS" output: min 81, mean 106.4, max 132),
// random distances uniform in [0, rc) and random directions.
TestSystem makeTestSystem(int numAtoms, double rc, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> distR(0.5, rc); // avoid r=0 singularity
    std::uniform_int_distribution<int> distN(81, 132);
    std::normal_distribution<double> distDir(0.0, 1.0);

    TestSystem sys;
    sys.numAtoms = numAtoms;
    sys.neighCount.resize(numAtoms);
    sys.neighOffset.resize(numAtoms);

    int total = 0;
    for (int i = 0; i < numAtoms; ++i)
    {
        int n = distN(rng);
        sys.neighCount[i] = n;
        sys.neighOffset[i] = total;
        total += n;
    }

    sys.neighDist.resize(total);
    sys.neighDx.resize(total);
    sys.neighDy.resize(total);
    sys.neighDz.resize(total);

    for (int k = 0; k < total; ++k)
    {
        double r = distR(rng);
        double dx = distDir(rng), dy = distDir(rng), dz = distDir(rng);
        double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        sys.neighDist[k] = r;
        sys.neighDx[k] = dx / norm * r;
        sys.neighDy[k] = dy / norm * r;
        sys.neighDz[k] = dz / norm * r;
    }

    return sys;
}

// Loads real H-H neighbor pairs dumped by dump_real_neighbors.cpp (which
// uses n2p2's own Structure::calculateNeighborList() on the actual
// temp/H2O_2G/input.data system) -- format: lines of
// "centralAtomLocalIndex r dx dy dz", grouped by (already-sorted, since the
// dumper writes them in atom order) central atom index. Real neighbor
// geometry, not synthetic. Assumes the last H atom in the structure has at
// least one H neighbor within rc (true for this dense liquid-water system,
// ~420 H atoms in a ~35A box) so numAtoms can be derived from the highest
// index seen in the file.
TestSystem loadRealTestSystem(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in)
    {
        fprintf(stderr, "Could not open %s\n", filename.c_str());
        exit(1);
    }

    std::vector<int> centralIdx;
    std::vector<double> dist, dx, dy, dz;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int idx; double r, x, y, z;
        iss >> idx >> r >> x >> y >> z;
        centralIdx.push_back(idx);
        dist.push_back(r);
        dx.push_back(x);
        dy.push_back(y);
        dz.push_back(z);
    }

    if (centralIdx.empty())
    {
        fprintf(stderr, "No neighbor pairs found in %s\n", filename.c_str());
        exit(1);
    }

    int numAtoms = centralIdx.back() + 1;

    TestSystem sys;
    sys.numAtoms = numAtoms;
    sys.neighCount.assign(numAtoms, 0);
    sys.neighOffset.assign(numAtoms, 0);
    for (int idx : centralIdx) sys.neighCount[idx]++;

    int total = 0;
    for (int i = 0; i < numAtoms; ++i)
    {
        sys.neighOffset[i] = total;
        total += sys.neighCount[i];
    }

    sys.neighDist.resize(total);
    sys.neighDx.resize(total);
    sys.neighDy.resize(total);
    sys.neighDz.resize(total);

    std::vector<int> cursor = sys.neighOffset;
    for (size_t k = 0; k < centralIdx.size(); ++k)
    {
        int idx = centralIdx[k];
        int pos = cursor[idx]++;
        sys.neighDist[pos] = dist[k];
        sys.neighDx[pos] = dx[k];
        sys.neighDy[pos] = dy[k];
        sys.neighDz[pos] = dz[k];
    }

    return sys;
}

void cpuReference(const TestSystem& sys, double eta, double rs, double rc,
                   std::vector<double>& G, std::vector<double>& dGdx,
                   std::vector<double>& dGdy, std::vector<double>& dGdz)
{
    G.resize(sys.numAtoms);
    dGdx.resize(sys.numAtoms);
    dGdy.resize(sys.numAtoms);
    dGdz.resize(sys.numAtoms);

    for (int i = 0; i < sys.numAtoms; ++i)
    {
        int off = sys.neighOffset[i];
        int n = sys.neighCount[i];
        symFncExpRad(n, &sys.neighDist[off], &sys.neighDx[off],
                     &sys.neighDy[off], &sys.neighDz[off],
                     eta, rs, rc, G[i], dGdx[i], dGdy[i], dGdz[i]);
    }
}

bool runCase(const char* label, double eta, double rs, double rc,
             const TestSystem& sys)
{
    printf("--- %s (eta=%.6g rs=%.6g rc=%.6g) ---\n", label, eta, rs, rc);

    std::vector<double> Gcpu, dGdxCpu, dGdyCpu, dGdzCpu;
    cpuReference(sys, eta, rs, rc, Gcpu, dGdxCpu, dGdyCpu, dGdzCpu);

    int total = sys.neighOffset.back() + sys.neighCount.back();

    int *d_neighCount, *d_neighOffset;
    double *d_neighDist, *d_neighDx, *d_neighDy, *d_neighDz;
    double *d_G, *d_dGdx, *d_dGdy, *d_dGdz;

    CUDA_CHECK(cudaMalloc(&d_neighCount, sys.numAtoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighOffset, sys.numAtoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighDist, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDx, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDy, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDz, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, sys.numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdx, sys.numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdy, sys.numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdz, sys.numAtoms * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_neighCount, sys.neighCount.data(),
                          sys.numAtoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighOffset, sys.neighOffset.data(),
                          sys.numAtoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDist, sys.neighDist.data(),
                          total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDx, sys.neighDx.data(),
                          total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDy, sys.neighDy.data(),
                          total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDz, sys.neighDz.data(),
                          total * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int gridSize = (sys.numAtoms + blockSize - 1) / blockSize;
    symFncExpRadKernel<<<gridSize, blockSize>>>(
        sys.numAtoms, d_neighCount, d_neighOffset,
        d_neighDist, d_neighDx, d_neighDy, d_neighDz,
        eta, rs, rc, d_G, d_dGdx, d_dGdy, d_dGdz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> Ggpu(sys.numAtoms), dGdxGpu(sys.numAtoms),
                         dGdyGpu(sys.numAtoms), dGdzGpu(sys.numAtoms);
    CUDA_CHECK(cudaMemcpy(Ggpu.data(), d_G, sys.numAtoms * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdxGpu.data(), d_dGdx, sys.numAtoms * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdyGpu.data(), d_dGdy, sys.numAtoms * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dGdzGpu.data(), d_dGdz, sys.numAtoms * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_neighCount); cudaFree(d_neighOffset);
    cudaFree(d_neighDist); cudaFree(d_neighDx); cudaFree(d_neighDy); cudaFree(d_neighDz);
    cudaFree(d_G); cudaFree(d_dGdx); cudaFree(d_dGdy); cudaFree(d_dGdz);

    double maxAbsErrG = 0.0, maxRelErrG = 0.0, maxAbsErrD = 0.0;
    for (int i = 0; i < sys.numAtoms; ++i)
    {
        double absErrG = std::fabs(Ggpu[i] - Gcpu[i]);
        double relErrG = absErrG / std::max(1e-300, std::fabs(Gcpu[i]));
        maxAbsErrG = std::max(maxAbsErrG, absErrG);
        maxRelErrG = std::max(maxRelErrG, relErrG);

        double dEx = std::fabs(dGdxGpu[i] - dGdxCpu[i]);
        double dEy = std::fabs(dGdyGpu[i] - dGdyCpu[i]);
        double dEz = std::fabs(dGdzGpu[i] - dGdzCpu[i]);
        maxAbsErrD = std::max({maxAbsErrD, dEx, dEy, dEz});
    }

    printf("  atoms=%d  G[0]: cpu=%.15E gpu=%.15E\n", sys.numAtoms, Gcpu[0], Ggpu[0]);
    printf("  max|G_gpu-G_cpu|=%.3E  max relerr=%.3E  max|dG_gpu-dG_cpu|=%.3E\n",
           maxAbsErrG, maxRelErrG, maxAbsErrD);

    bool pass = (maxAbsErrG < 1e-10) && (maxAbsErrD < 1e-10);
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char** argv)
{
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    printf("CUDA devices visible: %d\n", devCount);
    if (devCount == 0)
    {
        fprintf(stderr, "No CUDA device found -- aborting.\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device 0: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const double rc = 12.00; // common cutoff for all H-H radial functions in input.nn

    bool ok = true;

    printf("=== Synthetic random test system ===\n");
    TestSystem synth = makeTestSystem(630, rc, /*seed=*/12345);
    // symfunction_short H 2 H 0.001  0.0    12.00  (input.nn line 116)
    ok &= runCase("synthetic, H-H rs=0", 0.001, 0.0, rc, synth);
    // symfunction_short H 2 H 0.15   1.9124 12.00  (input.nn line 120)
    ok &= runCase("synthetic, H-H rs!=0", 0.15, 1.9124, rc, synth);

    // Real neighbor geometry, from an actual H2O_2G structure, dumped by
    // dump_real_neighbors.cpp (n2p2's own Structure::calculateNeighborList()
    // on temp/H2O_2G/input.data) -- pass its output path as argv[1].
    if (argc > 1)
    {
        printf("\n=== Real H2O_2G neighbor data (%s) ===\n", argv[1]);
        TestSystem real = loadRealTestSystem(argv[1]);
        ok &= runCase("real data, H-H rs=0", 0.001, 0.0, rc, real);
        ok &= runCase("real data, H-H rs!=0", 0.15, 1.9124, rc, real);
    }
    else
    {
        printf("\n(no real-data file given as argv[1] -- skipping real-data cases)\n");
    }

    printf("\n%s\n", ok ? "ALL CASES PASSED" : "SOME CASES FAILED");
    return ok ? 0 : 1;
}
