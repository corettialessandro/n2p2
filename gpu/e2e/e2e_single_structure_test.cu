// End-to-end integration test: chain every piece validated so far (Phase 2's
// radial symmetry function kernel, Phase 3's NN forward/backward, and force
// assembly) into one real GPU pipeline for the real H2O_2G structure, using
// REAL geometry and REAL symmetry-function parameters parsed straight out of
// temp/H2O_2G/input.nn -- not synthetic data at each stage, unlike every
// previous step. This is the natural next move recommended after Phase 3
// closed out: every piece so far was validated in isolation (dEdG used
// random G; force assembly used random dEdG/derivatives), so nothing had
// exercised the WIRING between phases with real, physically-connected
// numbers. This test is where integration bugs (element/index mismatches,
// wrong cutoffs, parsing mistakes) would actually surface.
//
// Scope: only the type-2 (SymFncExpRad, radial) instances from input.nn are
// used (16 for H, 16 for O -- see the two `awk`/`grep` counts against that
// file), not the real production 35/42-wide network (which also includes
// type-3/angular instances). Reason: Phase 1 step 4's neighbor-side
// derivative storage (AtomBatch::neighborDGdx,Dy,Dz) was only ever
// exercised for the radial family; gpu/smoke's angular kernels explicitly
// left neighbor-side derivatives (and the "rjk" term) out of scope. Adding
// full angular neighbor derivatives (two neighbors per term, not one, plus
// that extra rjk contribution) is a real, separate derivation -- a natural
// follow-up to THIS step, not bundled into it. What's tested here is
// therefore a smaller, self-consistent 16-input network per element (real
// geometry, real eta/rs/e1/rc values, real neighbor lists, real
// NeuralNetwork class with random weights since no trained weights exist)
// rather than the full production architecture -- but every number in it is
// real, not synthetic, all the way through symmetry functions -> NN
// forward -> dEdG -> force assembly.
//
// GPU pipeline: all data stays resident on device across the three kernel
// launches (symmetry functions -> NN forward+dEdG -> force assembly) --
// only the initial geometry/weights upload and the final energy/force
// download cross the host/device boundary, mirroring what a real batched
// training step would do.
//
// CPU reference: fully independent computation using the SAME real
// nnp::NeuralNetwork class (linked from lib/libnnp.a) and the same
// symFncExpRadGroupReal() host/device function (called directly on the
// host, since it's __host__ __device__ -- this specifically re-checks the
// WIRING/indexing, not the core per-neighbor math again, which earlier
// phases already validated bit-exact against SymFncExpRad::calculate()).
// Force assembly's CPU side reuses the same independent "gather" traversal
// ../force/force_assembly_test.cu introduced.

#include "../soa/AtomBatch.h"
#include "ElementMap.h"
#include "Structure.h"
#include "NeuralNetwork.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>

using namespace nnp;
using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

///////////////////////////////////////////////////////////////////////////
// Stage 1: real radial (ExpRad) symmetry functions, per-member e1 (a group
// can mix e1=H and e1=O members here, unlike SymGrpExpRad's real grouping
// rule of "same e1" -- harmless generalization since the cutoff computation
// doesn't depend on e1 at all, only which members' sums a given neighbor
// contributes to).
///////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void cutoffTANHU(double r, double rcinv,
                                             double& fc, double& dfc)
{
    double t  = tanh(1.0 - r * rcinv);
    double t2 = t * t;
    fc  = t * t2;
    dfc = 3.0 * t2 * (t2 - 1.0) * rcinv;
}

__host__ __device__ inline void symFncExpRadGroupReal(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double rc,
    int numMembers, const int* e1, const double* eta, const double* rs,
    double* result, double* dResultX, double* dResultY, double* dResultZ,
    double* neighDGdx, double* neighDGdy, double* neighDGdz)
{
    double rcinv = 1.0 / rc;
    for (int j = 0; j < numNeighbors; ++j)
    {
        double rij = neighDist[j];
        if (rij >= rc) continue;
        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);
        for (int k = 0; k < numMembers; ++k)
        {
            if (neighElem[j] != e1[k]) continue;
            double diff = rij - rs[k];
            double pexp = exp(-eta[k] * diff * diff);
            result[k] += pexp * pfc;
            double p1 = (pdfc - 2.0 * eta[k] * diff * pfc) * pexp / rij;
            double dijx = p1 * neighDx[j];
            double dijy = p1 * neighDy[j];
            double dijz = p1 * neighDz[j];
            dResultX[k] += dijx; dResultY[k] += dijy; dResultZ[k] += dijz;
            int idx = j * numMembers + k;
            neighDGdx[idx] = -dijx;
            neighDGdy[idx] = -dijy;
            neighDGdz[idx] = -dijz;
        }
    }
}

__global__ void sfGroupKernelReal(
    int begin, int numSelected, double rc, int numMembers,
    const int* e1, const double* eta, const double* rs,
    const int* neighOffsetInt, const int* neighElem, const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double* G, double* dGdx, double* dGdy, double* dGdz,
    size_t gBlockOffsetE, size_t sfCountE,
    const size_t* neighborSfOffset,
    double* neighborDGdx, double* neighborDGdy, double* neighborDGdz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    int i = begin + t;
    int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;

    double result[32] = {0}, dResultX[32] = {0}, dResultY[32] = {0}, dResultZ[32] = {0};

    symFncExpRadGroupReal(n, &neighElem[off], &neighDist[off], &neighDx[off],
                          &neighDy[off], &neighDz[off], rc, numMembers,
                          e1, eta, rs, result, dResultX, dResultY, dResultZ,
                          &neighborDGdx[neighborSfOffset[i]],
                          &neighborDGdy[neighborSfOffset[i]],
                          &neighborDGdz[neighborSfOffset[i]]);

    size_t base = gBlockOffsetE + (size_t)t * sfCountE;
    for (int k = 0; k < numMembers; ++k)
    {
        G[base + k] = result[k];
        dGdx[base + k] = dResultX[k];
        dGdy[base + k] = dResultY[k];
        dGdz[base + k] = dResultZ[k];
    }
}

///////////////////////////////////////////////////////////////////////////
// Stage 2: NN forward + dEdG (copied from ../nn/nn_forward_test.cu, already
// validated there against the real NeuralNetwork class).
///////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void nnForwardAndDEdG(
    const double* G, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    double* energyOut, double* dEdGOut)
{
    double h1[64], dfdx1[64];
    for (int k = 0; k < numHidden1; ++k)
    {
        double s = b1[k];
        for (int j = 0; j < numIn; ++j) s += W1[j * numHidden1 + k] * G[j];
        h1[k] = tanh(s);
        dfdx1[k] = 1.0 - h1[k] * h1[k];
    }
    double h2[64], dfdx2[64];
    for (int k = 0; k < numHidden2; ++k)
    {
        double s = b2[k];
        for (int j = 0; j < numHidden1; ++j) s += W2[j * numHidden2 + k] * h1[j];
        h2[k] = tanh(s);
        dfdx2[k] = 1.0 - h2[k] * h2[k];
    }
    double out = b3[0];
    for (int j = 0; j < numHidden2; ++j) out += W3[j] * h2[j];
    energyOut[0] = out;

    for (int k = 0; k < numIn; ++k)
    {
        double inner0[64];
        for (int i = 0; i < numHidden1; ++i)
            inner0[i] = W1[k * numHidden1 + i] * dfdx1[i];
        double outer0[64];
        for (int i2 = 0; i2 < numHidden2; ++i2)
        {
            double s = 0.0;
            for (int i = 0; i < numHidden1; ++i) s += W2[i * numHidden2 + i2] * inner0[i];
            outer0[i2] = s * dfdx2[i2];
        }
        double s = 0.0;
        for (int i2 = 0; i2 < numHidden2; ++i2) s += W3[i2] * outer0[i2];
        dEdGOut[k] = s;
    }
}

__global__ void nnForwardKernel(
    int begin, int numSelected, int numIn,
    const double* W1, const double* b1, int numHidden1,
    const double* W2, const double* b2, int numHidden2,
    const double* W3, const double* b3, int numOut,
    const double* G, size_t gBlockOffsetE, size_t sfCountE,
    double* energyOut, double* dEdG)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    int s = begin + t;
    size_t base = gBlockOffsetE + (size_t)t * sfCountE;
    double out[1];
    nnForwardAndDEdG(&G[base], numIn, W1, b1, numHidden1, W2, b2, numHidden2,
                      W3, b3, numOut, out, &dEdG[base]);
    energyOut[s] = out[0];
}

///////////////////////////////////////////////////////////////////////////
// Stage 3: force assembly (copied verbatim from
// ../force/force_assembly_test.cu, already validated there).
///////////////////////////////////////////////////////////////////////////

__global__ void forceAssemblyKernel(
    int numAtoms,
    const size_t* neighborOffset, const size_t* neighborAtomSorted,
    const size_t* neighborSfOffset,
    const size_t* gBaseOf, const size_t* sfCountOf,
    const double* dEdG, const double* dGdx, const double* dGdy, const double* dGdz,
    const double* neighborDGdx, const double* neighborDGdy, const double* neighborDGdz,
    double* forceX, double* forceY, double* forceZ)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numAtoms) return;

    size_t gBase = gBaseOf[j];
    size_t sfCount = sfCountOf[j];

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (size_t k = 0; k < sfCount; ++k)
    {
        double dedg = dEdG[gBase + k];
        fx -= dedg * dGdx[gBase + k];
        fy -= dedg * dGdy[gBase + k];
        fz -= dedg * dGdz[gBase + k];
    }
    atomicAdd(&forceX[j], fx);
    atomicAdd(&forceY[j], fy);
    atomicAdd(&forceZ[j], fz);

    size_t nBegin = neighborOffset[j], nEnd = neighborOffset[j + 1];
    size_t sfBase = neighborSfOffset[j];
    for (size_t slot = 0; slot < nEnd - nBegin; ++slot)
    {
        size_t i = neighborAtomSorted[nBegin + slot];
        size_t base = sfBase + slot * sfCount;
        double px = 0.0, py = 0.0, pz = 0.0;
        for (size_t k = 0; k < sfCount; ++k)
        {
            double dedg = dEdG[gBase + k];
            px -= dedg * neighborDGdx[base + k];
            py -= dedg * neighborDGdy[base + k];
            pz -= dedg * neighborDGdz[base + k];
        }
        atomicAdd(&forceX[i], px);
        atomicAdd(&forceY[i], py);
        atomicAdd(&forceZ[i], pz);
    }
}

///////////////////////////////////////////////////////////////////////////
// input.nn parsing: real "symfunction_short <ec> 2 <e1> <eta> <rs> <rc>"
// lines only, in file order.
///////////////////////////////////////////////////////////////////////////

struct RadMember { int e1; double eta, rs, rc; };

vector<RadMember> parseExpRadMembers(string const& path, string const& ec,
                                      ElementMap const& elementMap)
{
    vector<RadMember> members;
    ifstream in(path);
    string line;
    while (getline(in, line))
    {
        istringstream iss(line);
        string tag;
        if (!(iss >> tag) || tag != "symfunction_short") continue;
        string ecStr; int type;
        if (!(iss >> ecStr >> type)) continue;
        if (ecStr != ec || type != 2) continue;
        string e1Str; double eta, rs, rc;
        iss >> e1Str >> eta >> rs >> rc;
        RadMember m;
        m.e1 = (int)elementMap[e1Str];
        m.eta = eta; m.rs = rs; m.rc = rc;
        members.push_back(m);
    }
    return members;
}

int main(int argc, char** argv)
{
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    printf("CUDA devices visible: %d\n", devCount);
    if (devCount == 0) { fprintf(stderr, "No CUDA device.\n"); return 1; }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device 0: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    string inputData = (argc > 1) ? argv[1] : "input.data";
    string inputNn   = (argc > 2) ? argv[2] : "input.nn";
    double const rc = 12.0;

    ElementMap elementMap;
    elementMap.registerElements("H O");
    Structure structure;
    structure.setElementMap(elementMap);
    structure.readFromFile(inputData);
    structure.calculateNeighborList(rc);

    AtomBatch batch = buildAtomBatch(structure, rc);
    int const H = 0, O = 1;

    vector<RadMember> membersH = parseExpRadMembers(inputNn, "H", elementMap);
    vector<RadMember> membersO = parseExpRadMembers(inputNn, "O", elementMap);
    int numH = (int)membersH.size(), numO = (int)membersO.size();
    printf("Parsed %d real ExpRad instances for H, %d for O (from %s)\n",
           numH, numO, inputNn.c_str());

    allocateSfStorage(batch, {(size_t)numH, (size_t)numO});
    printf("AtomBatch: %zu atoms (%zu H, %zu O), %zu neighbor entries\n\n",
           batch.numAtoms, batch.elementOffset[H + 1] - batch.elementOffset[H],
           batch.elementOffset[O + 1] - batch.elementOffset[O],
           batch.totalNeighbors());

    vector<int> neighOffsetInt(batch.neighborOffset.begin(), batch.neighborOffset.end());
    vector<int> neighElemInt(batch.neighborElement.begin(), batch.neighborElement.end());

    // Two real NeuralNetworks, one per element, architecture sized to this
    // reduced real instance count (16 -> 25 -> 25 -> 1), random weights
    // (no trained weights available).
    int const numHidden1 = 25, numHidden2 = 25, numOut = 1, numLayers = 4;
    NeuralNetwork::ActivationFunction af[4] = {
        NeuralNetwork::AF_IDENTITY, NeuralNetwork::AF_TANH,
        NeuralNetwork::AF_TANH, NeuralNetwork::AF_IDENTITY};

    int nnH_layers[4] = {numH, numHidden1, numHidden2, numOut};
    NeuralNetwork nnH(numLayers, nnH_layers, af);
    nnH.initializeConnectionsRandomUniform(42);
    vector<double> connH(nnH.getNumConnections());
    nnH.getConnections(connH.data());

    int nnO_layers[4] = {numO, numHidden1, numHidden2, numOut};
    NeuralNetwork nnO(numLayers, nnO_layers, af);
    nnO.initializeConnectionsRandomUniform(43);
    vector<double> connO(nnO.getNumConnections());
    nnO.getConnections(connO.data());

    auto sliceConn = [&](vector<double>& conn, int numIn,
                          double const*& W1, double const*& b1,
                          double const*& W2, double const*& b2,
                          double const*& W3, double const*& b3)
    {
        size_t off = 0;
        W1 = conn.data() + off; off += (size_t)numIn * numHidden1;
        b1 = conn.data() + off; off += numHidden1;
        W2 = conn.data() + off; off += (size_t)numHidden1 * numHidden2;
        b2 = conn.data() + off; off += numHidden2;
        W3 = conn.data() + off; off += (size_t)numHidden2 * numOut;
        b3 = conn.data() + off; off += numOut;
    };
    double const *W1H, *b1H, *W2H, *b2H, *W3H, *b3H;
    double const *W1O, *b1O, *W2O, *b2O, *W3O, *b3O;
    sliceConn(connH, numH, W1H, b1H, W2H, b2H, W3H, b3H);
    sliceConn(connO, numO, W1O, b1O, W2O, b2O, W3O, b3O);

    // --- Per-atom convenience arrays for the force-assembly stage -----------
    size_t numAtoms = batch.numAtoms;
    vector<size_t> gBaseOf(numAtoms), sfCountOf(numAtoms);
    for (size_t s = 0; s < numAtoms; ++s)
    {
        gBaseOf[s] = batch.gIndex(s, 0);
        sfCountOf[s] = batch.sfCountPerElement[batch.element[s]];
    }

    ///////////////////////////////////////////////////////////////////////
    // GPU pipeline: everything stays on device across the three stages.
    ///////////////////////////////////////////////////////////////////////
    int total = (int)batch.neighborD.size();

    int *d_neighOffset, *d_neighElem;
    double *d_neighDist, *d_neighDx, *d_neighDy, *d_neighDz;
    CUDA_CHECK(cudaMalloc(&d_neighOffset, neighOffsetInt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighElem, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighDist, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDx, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDy, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDz, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_neighOffset, neighOffsetInt.data(), neighOffsetInt.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighElem, neighElemInt.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDist, batch.neighborD.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDx, batch.neighborDx.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDy, batch.neighborDy.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDz, batch.neighborDz.data(), total * sizeof(double), cudaMemcpyHostToDevice));

    size_t* d_neighborSfOffset;
    CUDA_CHECK(cudaMalloc(&d_neighborSfOffset, batch.neighborSfOffset.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_neighborSfOffset, batch.neighborSfOffset.data(), batch.neighborSfOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    double *d_G, *d_dGdx, *d_dGdy, *d_dGdz, *d_dEdG, *d_energy;
    double *d_nDGdx, *d_nDGdy, *d_nDGdz;
    CUDA_CHECK(cudaMalloc(&d_G, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdx, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdy, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dGdz, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dEdG, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdx, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdy, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nDGdz, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_G, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdx, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdy, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dGdz, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dEdG, 0, batch.G.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdx, 0, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdy, 0, batch.neighborDGdx.size() * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nDGdz, 0, batch.neighborDGdx.size() * sizeof(double)));

    double *d_forceX, *d_forceY, *d_forceZ;
    CUDA_CHECK(cudaMalloc(&d_forceX, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forceY, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forceZ, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceX, 0, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceY, 0, numAtoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forceZ, 0, numAtoms * sizeof(double)));

    size_t* d_gBaseOf; size_t* d_sfCountOf;
    CUDA_CHECK(cudaMalloc(&d_gBaseOf, numAtoms * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_sfCountOf, numAtoms * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_gBaseOf, gBaseOf.data(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfCountOf, sfCountOf.data(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t* d_neighborAtomSorted;
    CUDA_CHECK(cudaMalloc(&d_neighborAtomSorted, batch.neighborAtomSorted.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_neighborAtomSorted, batch.neighborAtomSorted.data(), batch.neighborAtomSorted.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t* d_neighborOffsetSizeT;
    CUDA_CHECK(cudaMalloc(&d_neighborOffsetSizeT, batch.neighborOffset.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_neighborOffsetSizeT, batch.neighborOffset.data(), batch.neighborOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    // Per-element member arrays for stage 1.
    auto uploadMembers = [](vector<RadMember> const& members,
                             int** d_e1, double** d_eta, double** d_rs)
    {
        int n = (int)members.size();
        vector<int> e1(n); vector<double> eta(n), rs(n);
        for (int k = 0; k < n; ++k) { e1[k] = members[k].e1; eta[k] = members[k].eta; rs[k] = members[k].rs; }
        CUDA_CHECK(cudaMalloc(d_e1, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(d_eta, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(d_rs, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(*d_e1, e1.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_eta, eta.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_rs, rs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    };
    int *d_e1H, *d_e1O; double *d_etaH, *d_rsH, *d_etaO, *d_rsO;
    uploadMembers(membersH, &d_e1H, &d_etaH, &d_rsH);
    uploadMembers(membersO, &d_e1O, &d_etaO, &d_rsO);

    // --- Stage 1: symmetry functions -----------------------------------------
    int blockSize = 128;
    int beginH = (int)batch.elementOffset[H], endH = (int)batch.elementOffset[H + 1];
    int beginO = (int)batch.elementOffset[O], endO = (int)batch.elementOffset[O + 1];
    int numSelH = endH - beginH, numSelO = endO - beginO;

    sfGroupKernelReal<<<(numSelH + blockSize - 1) / blockSize, blockSize>>>(
        beginH, numSelH, rc, numH, d_e1H, d_etaH, d_rsH,
        d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy, d_neighDz,
        d_G, d_dGdx, d_dGdy, d_dGdz, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
    CUDA_CHECK(cudaGetLastError());

    sfGroupKernelReal<<<(numSelO + blockSize - 1) / blockSize, blockSize>>>(
        beginO, numSelO, rc, numO, d_e1O, d_etaO, d_rsO,
        d_neighOffset, d_neighElem, d_neighDist, d_neighDx, d_neighDy, d_neighDz,
        d_G, d_dGdx, d_dGdy, d_dGdz, batch.gBlockOffset[O], batch.sfCountPerElement[O],
        d_neighborSfOffset, d_nDGdx, d_nDGdy, d_nDGdz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Stage 2: NN forward + dEdG ------------------------------------------
    double *d_W1H, *d_b1H, *d_W2H, *d_b2H, *d_W3H, *d_b3H;
    CUDA_CHECK(cudaMalloc(&d_W1H, numH * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1H, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2H, numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2H, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3H, numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b3H, numOut * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_W1H, W1H, numH * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1H, b1H, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2H, W2H, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2H, b2H, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3H, W3H, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3H, b3H, numOut * sizeof(double), cudaMemcpyHostToDevice));

    double *d_W1O, *d_b1O, *d_W2O, *d_b2O, *d_W3O, *d_b3O;
    CUDA_CHECK(cudaMalloc(&d_W1O, numO * numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b1O, numHidden1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W2O, numHidden1 * numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b2O, numHidden2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W3O, numHidden2 * numOut * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b3O, numOut * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_W1O, W1O, numO * numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1O, b1O, numHidden1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2O, W2O, numHidden1 * numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2O, b2O, numHidden2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3O, W3O, numHidden2 * numOut * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3O, b3O, numOut * sizeof(double), cudaMemcpyHostToDevice));

    nnForwardKernel<<<(numSelH + blockSize - 1) / blockSize, blockSize>>>(
        beginH, numSelH, numH, d_W1H, d_b1H, numHidden1, d_W2H, d_b2H, numHidden2,
        d_W3H, d_b3H, numOut, d_G, batch.gBlockOffset[H], batch.sfCountPerElement[H],
        d_energy, d_dEdG);
    CUDA_CHECK(cudaGetLastError());

    nnForwardKernel<<<(numSelO + blockSize - 1) / blockSize, blockSize>>>(
        beginO, numSelO, numO, d_W1O, d_b1O, numHidden1, d_W2O, d_b2O, numHidden2,
        d_W3O, d_b3O, numOut, d_G, batch.gBlockOffset[O], batch.sfCountPerElement[O],
        d_energy, d_dEdG);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Stage 3: force assembly ---------------------------------------------
    forceAssemblyKernel<<<((int)numAtoms + blockSize - 1) / blockSize, blockSize>>>(
        (int)numAtoms, d_neighborOffsetSizeT, d_neighborAtomSorted, d_neighborSfOffset,
        d_gBaseOf, d_sfCountOf, d_dEdG, d_dGdx, d_dGdy, d_dGdz,
        d_nDGdx, d_nDGdy, d_nDGdz, d_forceX, d_forceY, d_forceZ);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Download final results ----------------------------------------------
    vector<double> energyGpu(numAtoms), forceXGpu(numAtoms), forceYGpu(numAtoms), forceZGpu(numAtoms);
    CUDA_CHECK(cudaMemcpy(energyGpu.data(), d_energy, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forceXGpu.data(), d_forceX, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forceYGpu.data(), d_forceY, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forceZGpu.data(), d_forceZ, numAtoms * sizeof(double), cudaMemcpyDeviceToHost));
    vector<double> gGpu(batch.G.size());
    CUDA_CHECK(cudaMemcpy(gGpu.data(), d_G, batch.G.size() * sizeof(double), cudaMemcpyDeviceToHost));

    ///////////////////////////////////////////////////////////////////////
    // Independent CPU reference: same real geometry/parameters/weights,
    // computed with no device involvement at all.
    ///////////////////////////////////////////////////////////////////////
    vector<double> gCpu(batch.G.size(), 0.0);
    vector<double> dGdxCpu(batch.G.size(), 0.0), dGdyCpu(batch.G.size(), 0.0), dGdzCpu(batch.G.size(), 0.0);
    vector<double> nDGdxCpu(batch.neighborDGdx.size(), 0.0);
    vector<double> nDGdyCpu(batch.neighborDGdx.size(), 0.0);
    vector<double> nDGdzCpu(batch.neighborDGdx.size(), 0.0);

    auto runSfCpu = [&](int begin, int end, vector<RadMember> const& members)
    {
        int numMembers = (int)members.size();
        vector<int> e1(numMembers); vector<double> eta(numMembers), rsv(numMembers);
        for (int k = 0; k < numMembers; ++k) { e1[k] = members[k].e1; eta[k] = members[k].eta; rsv[k] = members[k].rs; }
        for (int i = begin; i < end; ++i)
        {
            int off = neighOffsetInt[i], n = neighOffsetInt[i + 1] - off;
            vector<double> result(numMembers, 0.0), dResultX(numMembers, 0.0), dResultY(numMembers, 0.0), dResultZ(numMembers, 0.0);
            symFncExpRadGroupReal(n, &neighElemInt[off], &batch.neighborD[off],
                                  &batch.neighborDx[off], &batch.neighborDy[off], &batch.neighborDz[off],
                                  rc, numMembers, e1.data(), eta.data(), rsv.data(),
                                  result.data(), dResultX.data(), dResultY.data(), dResultZ.data(),
                                  &nDGdxCpu[batch.neighborSfOffset[i]], &nDGdyCpu[batch.neighborSfOffset[i]], &nDGdzCpu[batch.neighborSfOffset[i]]);
            size_t base = batch.gIndex(i, 0);
            for (int k = 0; k < numMembers; ++k)
            {
                gCpu[base + k] = result[k];
                dGdxCpu[base + k] = dResultX[k];
                dGdyCpu[base + k] = dResultY[k];
                dGdzCpu[base + k] = dResultZ[k];
            }
        }
    };
    runSfCpu(beginH, endH, membersH);
    runSfCpu(beginO, endO, membersO);

    vector<double> energyCpu(numAtoms), dEdGCpu(batch.G.size(), 0.0);
    for (int t = 0; t < numSelH; ++t)
    {
        int s = beginH + t;
        nnH.setInput(&gCpu[batch.gIndex(s, 0)]);
        nnH.propagate();
        nnH.calculateDEdG(&dEdGCpu[batch.gIndex(s, 0)]);
        nnH.getOutput(&energyCpu[s]);
    }
    for (int t = 0; t < numSelO; ++t)
    {
        int s = beginO + t;
        nnO.setInput(&gCpu[batch.gIndex(s, 0)]);
        nnO.propagate();
        nnO.calculateDEdG(&dEdGCpu[batch.gIndex(s, 0)]);
        nnO.getOutput(&energyCpu[s]);
    }

    // Force assembly, CPU "gather" (independent traversal order from the
    // GPU's scatter, same as ../force/force_assembly_test.cu).
    vector<double> forceXCpu(numAtoms, 0.0), forceYCpu(numAtoms, 0.0), forceZCpu(numAtoms, 0.0);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        size_t gBase = gBaseOf[i], sfCount = sfCountOf[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (size_t k = 0; k < sfCount; ++k)
        {
            double dedg = dEdGCpu[gBase + k];
            fx -= dedg * dGdxCpu[gBase + k];
            fy -= dedg * dGdyCpu[gBase + k];
            fz -= dedg * dGdzCpu[gBase + k];
        }
        for (size_t j = 0; j < numAtoms; ++j)
        {
            size_t nBegin = batch.neighborOffset[j], nEnd = batch.neighborOffset[j + 1];
            size_t sfBaseJ = batch.neighborSfOffset[j], sfCountJ = sfCountOf[j];
            for (size_t slot = 0; slot < nEnd - nBegin; ++slot)
            {
                if (batch.neighborAtomSorted[nBegin + slot] != i) continue;
                size_t base = sfBaseJ + slot * sfCountJ;
                size_t gBaseJ = gBaseOf[j];
                for (size_t k = 0; k < sfCountJ; ++k)
                {
                    double dedg = dEdGCpu[gBaseJ + k];
                    fx -= dedg * nDGdxCpu[base + k];
                    fy -= dedg * nDGdyCpu[base + k];
                    fz -= dedg * nDGdzCpu[base + k];
                }
            }
        }
        forceXCpu[i] = fx; forceYCpu[i] = fy; forceZCpu[i] = fz;
    }

    ///////////////////////////////////////////////////////////////////////
    // Compare
    ///////////////////////////////////////////////////////////////////////
    double maxAbsErrG = 0.0, maxAbsErrE = 0.0, maxAbsErrF = 0.0;
    for (size_t k = 0; k < batch.G.size(); ++k)
        maxAbsErrG = max(maxAbsErrG, fabs(gGpu[k] - gCpu[k]));
    double sumEnergyGpu = 0.0, sumEnergyCpu = 0.0;
    for (size_t i = 0; i < numAtoms; ++i)
    {
        maxAbsErrE = max(maxAbsErrE, fabs(energyGpu[i] - energyCpu[i]));
        maxAbsErrF = max({maxAbsErrF, fabs(forceXGpu[i] - forceXCpu[i]),
                           fabs(forceYGpu[i] - forceYCpu[i]), fabs(forceZGpu[i] - forceZCpu[i])});
        sumEnergyGpu += energyGpu[i];
        sumEnergyCpu += energyCpu[i];
    }

    printf("Total energy: gpu=%.15E cpu=%.15E\n", sumEnergyGpu, sumEnergyCpu);
    printf("force[0]: gpu=(%.15E,%.15E,%.15E) cpu=(%.15E,%.15E,%.15E)\n",
           forceXGpu[0], forceYGpu[0], forceZGpu[0], forceXCpu[0], forceYCpu[0], forceZCpu[0]);
    printf("max|G_gpu-G_cpu|      = %.3E  (%zu values)\n", maxAbsErrG, batch.G.size());
    printf("max|E_gpu-E_cpu|      = %.3E  (%zu atoms)\n", maxAbsErrE, numAtoms);
    printf("max|F_gpu-F_cpu|      = %.3E  (%zu atoms)\n", maxAbsErrF, numAtoms);

    bool pass = (maxAbsErrG < 1e-9) && (maxAbsErrE < 1e-9) && (maxAbsErrF < 1e-9);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
