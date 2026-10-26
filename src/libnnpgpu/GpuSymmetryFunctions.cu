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

#include "GpuSymmetryFunctions.h"
#include "GpuDevice.h"
#include "GpuError.h"
#include <climits>   // INT_MAX
#include <cmath>     // std::exp, std::pow
#include <stdexcept> // std::runtime_error
#include <string>    // std::to_string

using namespace std;

namespace
{

/// Symmetry functions evaluated together (same type, neighbor elements and
/// cutoff), see Chunk.
int const maxMembers = 8;
int const threadsPerBlock = 128;
int const warpsPerBlock = threadsPerBlock / 32;
int const threadsPerTarget = 32;
int const maxWeightVectors = 3;

enum Kind
{
    RADIAL,
    ANGULAR_NARROW,
    ANGULAR_WIDE
};

// Same order as CutoffFunction::CutoffType.
enum CutoffType
{
    CT_HARD,
    CT_COS,
    CT_TANHU,
    CT_TANH,
    CT_EXP,
    CT_POLY1,
    CT_POLY2,
    CT_POLY3,
    CT_POLY4
};

struct Cutoff
{
    int    type;
    double rc;
    double rcinv;
    double rci;
    double iw;
    /// Prefactor of CT_TANH.
    double pre;
};

/** Symmetry functions of one element evaluated together.
 *
 * Same type, neighbor elements and cutoff function, as in the symmetry
 * function groups of the CPU code, at most maxMembers of them.
 */
struct Chunk
{
    int    kind;
    int    e1;
    /// Equal to e1 for radial symmetry functions.
    int    e2;
    int    numMembers;
    Cutoff cutoff;
    /// Index of the value among the symmetry functions of the element.
    int    index[maxMembers];
    /// Index of the derivative for a neighbor of element e1 and e2.
    int    deriv1[maxMembers];
    int    deriv2[maxMembers];
    /// If the exponential differs from that of the previous member.
    int    newExp[maxMembers];
    int    useIntegerPow[maxMembers];
    int    zetaInt[maxMembers];
    double eta[maxMembers];
    double rs[maxMembers];
    double lambda[maxMembers];
    double zeta[maxMembers];
    /// 2^(1 - zeta).
    double pnorm[maxMembers];
    /// zeta * lambda.
    double pzl[maxMembers];
    /// Scaled value: offset + factor * (G - shift); derivatives * factor.
    double scaleOffset[maxMembers];
    double scaleFactor[maxMembers];
    double scaleShift[maxMembers];
};

// Same as CutoffFunction::fdf().
__device__ void cutoffFunction(Cutoff const& c,
                               double        r,
                               double&       fc,
                               double&       dfc)
{
    double const pi = 3.14159265358979323846;
    double const e = 2.71828182845904523536;
    if (c.type == CT_HARD)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    if (c.type == CT_TANHU || c.type == CT_TANH)
    {
        double const temp = tanh(1.0 - r * c.rcinv);
        double const temp2 = temp * temp;
        fc = temp * temp2;
        dfc = 3.0 * temp2 * (temp2 - 1.0) * c.rcinv;
        if (c.type == CT_TANH)
        {
            fc *= c.pre;
            dfc *= c.pre;
        }
        return;
    }
    if (r < c.rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - c.rci) * c.iw;
    if (c.type == CT_COS)
    {
        double const temp = cos(pi * x);
        fc = 0.5 * (temp + 1.0);
        dfc = -0.5 * c.iw * pi * sqrt(1.0 - temp * temp);
        return;
    }
    // Core functions (CoreFunction::fdf()).
    if (c.type == CT_POLY1)
    {
        fc = (2.0 * x - 3.0) * x * x + 1.0;
        dfc = x * (6.0 * x - 6.0);
    }
    else if (c.type == CT_POLY2)
    {
        double const x2 = x * x;
        fc = ((15.0 - 6.0 * x) * x - 10.0) * x * x2 + 1.0;
        dfc = x2 * ((60.0 - 30.0 * x) * x - 30.0);
    }
    else if (c.type == CT_POLY3)
    {
        double const x2 = x * x;
        fc = (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
        dfc = x2 * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
    }
    else if (c.type == CT_POLY4)
    {
        double x4 = x * x;
        x4 *= x4;
        fc = (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0)
           * x * x4 + 1.0;
        dfc = x4 * (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0)
                    - 630.0);
    }
    else // CT_EXP
    {
        double const temp = 1.0 / (x * x - 1.0);
        double const temp2 = exp(temp);
        fc = e * temp2;
        dfc = -2.0 * e * x * temp * temp * temp2;
    }
    dfc *= c.iw;
}

// Same as pow_int() (utility.h).
__device__ double powInt(double x, int n)
{
    unsigned int m;
    if (n < 0)
    {
        x = 1.0 / x;
        m = -n;
    }
    else m = n;
    double result = 1.0;
    do
    {
        if (m & 1) result *= x;
        m >>= 1;
        x *= x;
    }
    while (m);
    return result;
}

/* One block per atom, one thread per neighbor j (strided). For each chunk the
 * thread computes all contributions to the symmetry functions involving
 * neighbor j and, as it is the only thread writing them, the derivatives
 * with respect to the position of neighbor j. Angular terms of each pair of
 * neighbors are therefore computed twice (once by each thread), values are
 * halved. The derivatives with respect to the position of the atom itself
 * follow from translation invariance (see contract()).
 */
__global__ void calculateKernel(Chunk const*  chunks,
                                int const*    chunkOffset,
                                int const*    atomElement,
                                int const*    valueOffset,
                                int const*    neighborOffset,
                                int const*    neighborElement,
                                double const* neighborGeometry,
                                int const*    derivOffset,
                                bool          derivatives,
                                double*       values,
                                double*       deriv)
{
    // Cutoff function and derivative of each neighbor for the current chunk.
    extern __shared__ double cache[];
    __shared__ Chunk ch;
    __shared__ double partial[warpsPerBlock][maxMembers];

    int const atom = blockIdx.x;
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const warp = tid / 32;
    int const element = atomElement[atom];
    int const n0 = neighborOffset[atom];
    int const numNeighbors = neighborOffset[atom + 1] - n0;
    double const* geometry = neighborGeometry + 4 * static_cast<size_t>(n0);
    int const* elements = neighborElement + n0;

    for (int c = chunkOffset[element]; c < chunkOffset[element + 1]; ++c)
    {
        __syncthreads();
        int const* src = reinterpret_cast<int const*>(chunks + c);
        int* dst = reinterpret_cast<int*>(&ch);
        for (int i = tid; i < static_cast<int>(sizeof(Chunk) / sizeof(int));
             i += blockDim.x)
        {
            dst[i] = src[i];
        }
        __syncthreads();
        double const rc = ch.cutoff.rc;
        for (int j = tid; j < numNeighbors; j += blockDim.x)
        {
            double const r = geometry[4 * j];
            int const ej = elements[j];
            double fc = 0.0;
            double dfc = 0.0;
            if (r < rc && (ej == ch.e1 || ej == ch.e2))
            {
                cutoffFunction(ch.cutoff, r, fc, dfc);
            }
            cache[2 * j] = fc;
            cache[2 * j + 1] = dfc;
        }
        __syncthreads();

        double g[maxMembers];
        for (int m = 0; m < maxMembers; ++m) g[m] = 0.0;
        for (int j = tid; j < numNeighbors; j += blockDim.x)
        {
            int const ej = elements[j];
            double const rij = geometry[4 * j];
            if (!(rij < rc) || (ej != ch.e1 && ej != ch.e2)) continue;
            double const* drij = geometry + 4 * j + 1;
            double const pfcij = cache[2 * j];
            double const pdfcij = cache[2 * j + 1];
            double* dj = derivatives
                       ? deriv + 3 * static_cast<size_t>(derivOffset[n0 + j])
                       : nullptr;
            if (ch.kind == RADIAL)
            {
                #pragma unroll
                for (int m = 0; m < maxMembers; ++m)
                {
                    if (m >= ch.numMembers) continue;
                    double const rijs = rij - ch.rs[m];
                    double const pexp = exp(-ch.eta[m] * rijs * rijs);
                    g[m] += pexp * pfcij;
                    if (!derivatives) continue;
                    double const p1 = ch.scaleFactor[m]
                                    * (pdfcij - 2.0 * ch.eta[m] * rijs
                                       * pfcij) * pexp / rij;
                    double* d = dj + 3 * ch.deriv1[m];
                    d[0] = -p1 * drij[0];
                    d[1] = -p1 * drij[1];
                    d[2] = -p1 * drij[2];
                }
                continue;
            }

            bool const narrow = (ch.kind == ANGULAR_NARROW);
            double const r2ij = rij * rij;
            double dg[maxMembers][3];
            for (int m = 0; m < maxMembers; ++m)
            {
                dg[m][0] = 0.0;
                dg[m][1] = 0.0;
                dg[m][2] = 0.0;
            }
            for (int k = 0; k < numNeighbors; ++k)
            {
                if (k == j) continue;
                int const ek = elements[k];
                if (!((ch.e1 == ej && ch.e2 == ek) ||
                      (ch.e2 == ej && ch.e1 == ek))) continue;
                double const rik = geometry[4 * k];
                if (!(rik < rc)) continue;
                double const* drik = geometry + 4 * k + 1;
                double const drjk[3] = {drik[0] - drij[0],
                                        drik[1] - drij[1],
                                        drik[2] - drij[2]};
                double rjk = 0.0;
                double pfcjk = 1.0;
                double pdfcjk = 0.0;
                if (narrow)
                {
                    double const r2jk = drjk[0] * drjk[0]
                                      + drjk[1] * drjk[1]
                                      + drjk[2] * drjk[2];
                    if (!(r2jk < rc * rc)) continue;
                    rjk = sqrt(r2jk);
                    cutoffFunction(ch.cutoff, rjk, pfcjk, pdfcjk);
                }
                double const pfcik = cache[2 * k];
                double const rinvijik = 1.0 / rij / rik;
                double const costijk = (drij[0] * drik[0] +
                                        drij[1] * drik[1] +
                                        drij[2] * drik[2]) * rinvijik;
                double const pfc = narrow ? pfcij * pfcik * pfcjk
                                          : pfcij * pfcik;
                double rijs = 0.0;
                double riks = 0.0;
                double rjks = 0.0;
                double pexp = 0.0;
                #pragma unroll
                for (int m = 0; m < maxMembers; ++m)
                {
                    if (m >= ch.numMembers) continue;
                    if (ch.newExp[m])
                    {
                        rijs = rij - ch.rs[m];
                        riks = rik - ch.rs[m];
                        if (narrow)
                        {
                            rjks = rjk - ch.rs[m];
                            pexp = exp(-ch.eta[m] * (rijs * rijs +
                                                     riks * riks +
                                                     rjks * rjks));
                        }
                        else
                        {
                            pexp = exp(-ch.eta[m] * (rijs * rijs +
                                                     riks * riks));
                        }
                    }
                    double const plambda = 1.0 + ch.lambda[m] * costijk;
                    double fg = pexp;
                    if (plambda <= 0.0) fg = 0.0;
                    else if (ch.useIntegerPow[m])
                    {
                        fg *= powInt(plambda, ch.zetaInt[m] - 1);
                    }
                    else fg *= pow(plambda, ch.zeta[m] - 1.0);
                    g[m] += fg * plambda * pfc;
                    if (!derivatives) continue;
                    fg *= ch.pnorm[m];
                    double const rinv = rinvijik * ch.pzl[m];
                    double const cost = costijk * ch.pzl[m];
                    double const p2etapl = 2.0 * ch.eta[m] * plambda;
                    double p1;
                    double p3;
                    if (narrow)
                    {
                        p1 = fg * (pfc * (rinv - cost / r2ij
                                          - p2etapl * rijs / rij)
                                   + pfcik * pfcjk * pdfcij * plambda / rij);
                        p3 = fg * (pfc * (rinv + p2etapl * rjks / rjk)
                                   - pfcij * pfcik * pdfcjk * plambda / rjk);
                    }
                    else
                    {
                        p1 = fg * (pfc * (rinv - cost / r2ij
                                          - p2etapl * rijs / rij)
                                   + pfcik * pdfcij * plambda / rij);
                        p3 = fg * pfc * rinv;
                    }
                    dg[m][0] -= p1 * drij[0] + p3 * drjk[0];
                    dg[m][1] -= p1 * drij[1] + p3 * drjk[1];
                    dg[m][2] -= p1 * drij[2] + p3 * drjk[2];
                }
            }
            if (!derivatives) continue;
            #pragma unroll
            for (int m = 0; m < maxMembers; ++m)
            {
                if (m >= ch.numMembers) continue;
                int const index = (ej == ch.e1) ? ch.deriv1[m] : ch.deriv2[m];
                double* d = dj + 3 * index;
                d[0] = dg[m][0] * ch.scaleFactor[m];
                d[1] = dg[m][1] * ch.scaleFactor[m];
                d[2] = dg[m][2] * ch.scaleFactor[m];
            }
        }

        // Sum over all threads in a fixed order.
        for (int m = 0; m < maxMembers; ++m)
        {
            double v = g[m];
            for (int offset = 16; offset > 0; offset /= 2)
            {
                v += __shfl_down_sync(0xffffffff, v, offset);
            }
            if (lane == 0) partial[warp][m] = v;
        }
        __syncthreads();
        if (tid < ch.numMembers)
        {
            double sum = 0.0;
            for (int w = 0; w < warpsPerBlock; ++w) sum += partial[w][tid];
            if (ch.kind != RADIAL) sum = 0.5 * sum * ch.pnorm[tid];
            values[valueOffset[atom] + ch.index[tid]] =
                ch.scaleOffset[tid]
                + ch.scaleFactor[tid] * (sum - ch.scaleShift[tid]);
        }
    }
}

/* One block per atom, one thread per neighbor j (strided): contraction of
 * the derivatives with respect to the position of neighbor j with the
 * weights, and minus their sum for the position of the atom itself.
 * Contributions: one per neighbor entry, then one per atom, 3 per weight
 * vector each.
 */
__global__ void contractKernel(int           numWeightVectors,
                               int           numValues,
                               int           numSlots,
                               int           numElements,
                               int const*    atomElement,
                               int const*    valueOffset,
                               int const*    neighborOffset,
                               int const*    neighborElement,
                               int const*    derivOffset,
                               int const*    tableOffset,
                               int const*    tableCount,
                               int const*    table,
                               double const* deriv,
                               double const* weights,
                               double*       contrib)
{
    __shared__ double partial[warpsPerBlock][3 * maxWeightVectors];

    int const atom = blockIdx.x;
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const warp = tid / 32;
    int const element = atomElement[atom];
    int const n0 = neighborOffset[atom];
    int const numNeighbors = neighborOffset[atom + 1] - n0;
    int const v0 = valueOffset[atom];

    double self[3 * maxWeightVectors];
    for (int i = 0; i < 3 * maxWeightVectors; ++i) self[i] = 0.0;
    for (int j = tid; j < numNeighbors; j += blockDim.x)
    {
        int const slot = n0 + j;
        int const t = element * numElements + neighborElement[slot];
        int const count = tableCount[t];
        int const* tbl = table + tableOffset[t];
        double const* d = deriv + 3 * static_cast<size_t>(derivOffset[slot]);
        double s[3 * maxWeightVectors];
        for (int i = 0; i < 3 * maxWeightVectors; ++i) s[i] = 0.0;
        for (int k = 0; k < count; ++k)
        {
            int const v = v0 + tbl[k];
            #pragma unroll
            for (int w = 0; w < maxWeightVectors; ++w)
            {
                if (w >= numWeightVectors) continue;
                double const weight = weights[w * numValues + v];
                s[3 * w + 0] += weight * d[3 * k + 0];
                s[3 * w + 1] += weight * d[3 * k + 1];
                s[3 * w + 2] += weight * d[3 * k + 2];
            }
        }
        double* out = contrib + 3 * static_cast<size_t>(numWeightVectors)
                                  * slot;
        for (int i = 0; i < 3 * numWeightVectors; ++i)
        {
            out[i] = s[i];
            self[i] -= s[i];
        }
    }
    for (int i = 0; i < 3 * maxWeightVectors; ++i)
    {
        double v = self[i];
        for (int offset = 16; offset > 0; offset /= 2)
        {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane == 0) partial[warp][i] = v;
    }
    __syncthreads();
    if (tid < 3 * numWeightVectors)
    {
        double sum = 0.0;
        for (int w = 0; w < warpsPerBlock; ++w) sum += partial[w][tid];
        contrib[3 * static_cast<size_t>(numWeightVectors)
                * (numSlots + atom) + tid] = sum;
    }
}

/* One thread per atom: derivatives of its values with respect to one
 * coordinate of one target, from its neighbor entries pointing to the target
 * and, if the atom itself is the target, minus all of its entries (entries of
 * periodic images of the target itself cancel). Each atom's values are only
 * written by its own thread, in neighbor order.
 */
__global__ void targetDerivativeKernel(int           numAtoms,
                                       int           numElements,
                                       int           target,
                                       int           component,
                                       int const*    atomElement,
                                       int const*    atomTarget,
                                       int const*    valueOffset,
                                       int const*    neighborOffset,
                                       int const*    neighborElement,
                                       int const*    neighborTarget,
                                       int const*    derivOffset,
                                       int const*    tableOffset,
                                       int const*    tableCount,
                                       int const*    table,
                                       double const* deriv,
                                       double*       result)
{
    int const atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= numAtoms) return;
    int const element = atomElement[atom];
    int const self = atomTarget[atom] == target ? 1 : 0;
    double* r = result + valueOffset[atom];
    for (int slot = neighborOffset[atom]; slot < neighborOffset[atom + 1];
         ++slot)
    {
        int const factor = (neighborTarget[slot] == target ? 1 : 0) - self;
        if (factor == 0) continue;
        int const t = element * numElements + neighborElement[slot];
        int const count = tableCount[t];
        int const* tbl = table + tableOffset[t];
        double const* d = deriv + 3 * static_cast<size_t>(derivOffset[slot]);
        for (int k = 0; k < count; ++k)
        {
            r[tbl[k]] += factor * d[3 * k + component];
        }
    }
}

// One warp per target: sum of its contributions in a fixed order.
__global__ void gatherKernel(int           numTargets,
                             int           numWeightVectors,
                             int const*    targetOffset,
                             int const*    targetContrib,
                             double const* contrib,
                             double*       result)
{
    int const target = (blockIdx.x * blockDim.x + threadIdx.x)
                     / threadsPerTarget;
    int const lane = threadIdx.x % threadsPerTarget;
    if (target >= numTargets) return;

    double sum[3 * maxWeightVectors];
    for (int i = 0; i < 3 * maxWeightVectors; ++i) sum[i] = 0.0;
    for (int i = targetOffset[target] + lane; i < targetOffset[target + 1];
         i += threadsPerTarget)
    {
        double const* c = contrib + 3 * static_cast<size_t>(numWeightVectors)
                                      * targetContrib[i];
        for (int k = 0; k < 3 * numWeightVectors; ++k) sum[k] += c[k];
    }
    for (int k = 0; k < 3 * maxWeightVectors; ++k)
    {
        for (int offset = threadsPerTarget / 2; offset > 0; offset /= 2)
        {
            sum[k] += __shfl_down_sync(0xffffffff, sum[k], offset);
        }
    }
    if (lane == 0)
    {
        for (int w = 0; w < numWeightVectors; ++w)
        {
            for (int c = 0; c < 3; ++c)
            {
                result[(static_cast<size_t>(w) * numTargets + target) * 3 + c]
                    = sum[3 * w + c];
            }
        }
    }
}

/// Device memory that grows when needed and is kept for later use.
struct Buffer
{
    void*  p;
    size_t bytes;

    Buffer() : p(nullptr), bytes(0) {}
    void ensure(size_t required, size_t& total)
    {
        if (required <= bytes) return;
        if (p != nullptr) NNP_CUDA_CHECK(cudaFree(p));
        p = nullptr;
        total -= bytes;
        bytes = 0;
        size_t const allocate = required + required / 4;
        NNP_CUDA_CHECK(cudaMalloc(&p, allocate));
        bytes = allocate;
        total += bytes;
    }
    template<typename T> T* get() const { return static_cast<T*>(p); }
};

void upload(Buffer& b, void const* host, size_t bytes, size_t& total)
{
    b.ensure(bytes > 0 ? bytes : 1, total);
    if (bytes > 0)
    {
        NNP_CUDA_CHECK(cudaMemcpy(b.p, host, bytes, cudaMemcpyHostToDevice));
    }
}

}

namespace nnp
{

struct GpuSymmetryFunctionSet::Data
{
    int         device;
    int         numElements;
    /// Number of symmetry functions of each element.
    vector<int> numValues;
    /// Number of derivatives stored per neighbor, for each pair (element,
    /// neighbor element).
    vector<int> tableCount;
    Chunk*      chunks;
    int*        chunkOffset;
    int*        tableOffset;
    int*        tableCountDevice;
    int*        table;
};

bool GpuSymmetryFunctionSet::isSupported(int type)
{
    return type == 2 || type == 3 || type == 9;
}

GpuSymmetryFunctionSet::GpuSymmetryFunctionSet(
    vector<vector<GpuSymmetryFunction>> const& symmetryFunctions) :
    data(new Data())
{
    Data& d = *data;
    d.chunks = nullptr;
    d.chunkOffset = nullptr;
    d.tableOffset = nullptr;
    d.tableCountDevice = nullptr;
    d.table = nullptr;
    try
    {
        int const numElements = symmetryFunctions.size();
        d.numElements = numElements;
        vector<Chunk> chunks;
        vector<int> chunkOffset(1, 0);
        vector<int> tableOffset;
        vector<int> table;
        double const e = exp(1.0);
        double const tanhPre = pow((e + 1 / e) / (e - 1 / e), 3);
        for (int ec = 0; ec < numElements; ++ec)
        {
            vector<GpuSymmetryFunction> const& sfs = symmetryFunctions.at(ec);
            d.numValues.push_back(sfs.size());
            for (size_t i = 0; i < sfs.size(); ++i)
            {
                GpuSymmetryFunction const& s = sfs.at(i);
                if (!isSupported(s.type))
                {
                    throw runtime_error("ERROR: Symmetry function type "
                                        + to_string(s.type) + " is not "
                                        "available on the GPU.\n");
                }
                if (s.index != static_cast<int>(i) ||
                    s.indexPerElement.size() !=
                    static_cast<size_t>(numElements))
                {
                    throw runtime_error("ERROR: Inconsistent symmetry "
                                        "functions for the GPU.\n");
                }
            }
            // Derivatives stored per neighbor of each element: the
            // symmetry functions depending on that element.
            for (int ne = 0; ne < numElements; ++ne)
            {
                vector<int> t;
                for (auto const& s : sfs)
                {
                    int const k = s.indexPerElement.at(ne);
                    if (k < 0) continue;
                    if (k >= static_cast<int>(t.size())) t.resize(k + 1, -1);
                    t.at(k) = s.index;
                }
                for (int v : t)
                {
                    if (v < 0)
                    {
                        throw runtime_error("ERROR: Inconsistent symmetry "
                                            "function derivative indices "
                                            "for the GPU.\n");
                    }
                }
                tableOffset.push_back(table.size());
                d.tableCount.push_back(t.size());
                table.insert(table.end(), t.begin(), t.end());
            }
            // Chunks: symmetry functions sharing type, neighbor elements and
            // cutoff function, in the order of their index.
            vector<bool> done(sfs.size(), false);
            for (size_t i = 0; i < sfs.size(); ++i)
            {
                if (done.at(i)) continue;
                GpuSymmetryFunction const& first = sfs.at(i);
                Chunk c = {};
                c.kind = first.type == 2 ? RADIAL
                       : first.type == 3 ? ANGULAR_NARROW : ANGULAR_WIDE;
                c.e1 = first.e1;
                c.e2 = first.type == 2 ? first.e1 : first.e2;
                c.numMembers = 0;
                c.cutoff.type = first.cutoffType;
                c.cutoff.rc = first.rc;
                c.cutoff.rcinv = 1.0 / first.rc;
                c.cutoff.rci = first.rc * first.cutoffAlpha;
                c.cutoff.iw = 1.0 / (first.rc - c.cutoff.rci);
                c.cutoff.pre = tanhPre;
                for (size_t j = i; j < sfs.size(); ++j)
                {
                    GpuSymmetryFunction const& s = sfs.at(j);
                    if (done.at(j) ||
                        s.type != first.type ||
                        s.e1 != first.e1 ||
                        (s.type != 2 && s.e2 != first.e2) ||
                        s.rc != first.rc ||
                        s.cutoffType != first.cutoffType ||
                        s.cutoffAlpha != first.cutoffAlpha) continue;
                    int const m = c.numMembers++;
                    done.at(j) = true;
                    c.index[m] = s.index;
                    c.deriv1[m] = s.indexPerElement.at(c.e1);
                    c.deriv2[m] = s.indexPerElement.at(c.e2);
                    c.newExp[m] = (m == 0 ||
                                   s.eta != c.eta[m - 1] ||
                                   s.rs != c.rs[m - 1]);
                    c.useIntegerPow[m] = s.useIntegerPow;
                    c.zetaInt[m] = s.zetaInt;
                    c.eta[m] = s.eta;
                    c.rs[m] = s.rs;
                    c.lambda[m] = s.lambda;
                    c.zeta[m] = s.zeta;
                    c.pnorm[m] = pow(2.0, 1.0 - s.zeta);
                    c.pzl[m] = s.zeta * s.lambda;
                    // SymFnc::scale(): ST_NONE, ST_SCALE, ST_CENTER,
                    // ST_SCALECENTER, ST_SCALESIGMA.
                    if (s.scalingType == 0)
                    {
                        c.scaleOffset[m] = 0.0;
                        c.scaleFactor[m] = 1.0;
                        c.scaleShift[m] = 0.0;
                    }
                    else if (s.scalingType == 1)
                    {
                        c.scaleOffset[m] = s.Smin;
                        c.scaleFactor[m] = s.scalingFactor;
                        c.scaleShift[m] = s.Gmin;
                    }
                    else if (s.scalingType == 2)
                    {
                        c.scaleOffset[m] = 0.0;
                        c.scaleFactor[m] = 1.0;
                        c.scaleShift[m] = s.Gmean;
                    }
                    else if (s.scalingType == 3 || s.scalingType == 4)
                    {
                        c.scaleOffset[m] = s.Smin;
                        c.scaleFactor[m] = s.scalingFactor;
                        c.scaleShift[m] = s.Gmean;
                    }
                    else
                    {
                        throw runtime_error("ERROR: Unknown symmetry "
                                            "function scaling type.\n");
                    }
                    if (c.deriv1[m] < 0 || c.deriv2[m] < 0 ||
                        c.cutoff.type < CT_HARD || c.cutoff.type > CT_POLY4)
                    {
                        throw runtime_error("ERROR: Inconsistent symmetry "
                                            "functions for the GPU.\n");
                    }
                    if (c.numMembers == maxMembers) break;
                }
                chunks.push_back(c);
            }
            chunkOffset.push_back(chunks.size());
        }

        d.device = gpuCurrentDevice();
        size_t total = 0;
        auto copy = [&total](void** p, void const* host, size_t bytes)
        {
            Buffer b;
            upload(b, host, bytes, total);
            *p = b.p;
        };
        copy(reinterpret_cast<void**>(&d.chunks), chunks.data(),
             chunks.size() * sizeof(Chunk));
        copy(reinterpret_cast<void**>(&d.chunkOffset), chunkOffset.data(),
             chunkOffset.size() * sizeof(int));
        copy(reinterpret_cast<void**>(&d.tableOffset), tableOffset.data(),
             tableOffset.size() * sizeof(int));
        copy(reinterpret_cast<void**>(&d.tableCountDevice),
             d.tableCount.data(), d.tableCount.size() * sizeof(int));
        copy(reinterpret_cast<void**>(&d.table), table.data(),
             table.size() * sizeof(int));
    }
    catch (...)
    {
        cudaFree(d.chunks);
        cudaFree(d.chunkOffset);
        cudaFree(d.tableOffset);
        cudaFree(d.tableCountDevice);
        cudaFree(d.table);
        delete data;
        throw;
    }
}

GpuSymmetryFunctionSet::~GpuSymmetryFunctionSet()
{
    // No checks: the destructor must not throw.
    Data& d = *data;
    cudaSetDevice(d.device);
    cudaFree(d.chunks);
    cudaFree(d.chunkOffset);
    cudaFree(d.tableOffset);
    cudaFree(d.tableCountDevice);
    cudaFree(d.table);
    delete data;
}

struct GpuSymmetryFunctionData::Data
{
    int                                   device;
    size_t                                total;
    /// Set and sizes of the last calculation.
    GpuSymmetryFunctionSet::Data const*   set;
    bool                                  hasDerivatives;
    size_t                                numAtoms;
    size_t                                numSlots;
    size_t                                numValues;
    size_t                                numTargets;
    Buffer                                atomElement;
    Buffer                                valueOffset;
    Buffer                                neighborOffset;
    Buffer                                neighborElement;
    Buffer                                atomTarget;
    Buffer                                neighborTarget;
    Buffer                                neighborGeometry;
    Buffer                                derivOffset;
    Buffer                                deriv;
    Buffer                                values;
    Buffer                                targetOffset;
    Buffer                                targetContrib;
    Buffer                                weights;
    Buffer                                contrib;
    Buffer                                result;
};

GpuSymmetryFunctionData::GpuSymmetryFunctionData() : data(new Data())
{
    Data& d = *data;
    d.device = gpuCurrentDevice();
    d.total = 0;
    d.set = nullptr;
    d.hasDerivatives = false;
    d.numAtoms = 0;
    d.numSlots = 0;
    d.numValues = 0;
    d.numTargets = 0;
}

GpuSymmetryFunctionData::~GpuSymmetryFunctionData()
{
    // No checks: the destructor must not throw.
    Data& d = *data;
    cudaSetDevice(d.device);
    for (Buffer* b : {&d.atomElement, &d.valueOffset, &d.neighborOffset,
                      &d.neighborElement, &d.atomTarget, &d.neighborTarget,
                      &d.neighborGeometry, &d.derivOffset,
                      &d.deriv, &d.values, &d.targetOffset, &d.targetContrib,
                      &d.weights, &d.contrib, &d.result})
    {
        cudaFree(b->p);
    }
    delete data;
}

void GpuSymmetryFunctionData::calculate(GpuSymmetryFunctionSet const& set,
                                        vector<int> const& atomElement,
                                        vector<int> const& atomTarget,
                                        vector<int> const& neighborOffset,
                                        vector<int> const& neighborElement,
                                        vector<int> const& neighborTarget,
                                        vector<double> const& neighborGeometry,
                                        bool                  derivatives,
                                        double*               values)
{
    Data& d = *data;
    GpuSymmetryFunctionSet::Data const& s = *set.data;
    d.set = nullptr;
    d.hasDerivatives = false;
    size_t const numAtoms = atomElement.size();
    if (neighborOffset.size() != numAtoms + 1 ||
        atomTarget.size() != numAtoms ||
        neighborOffset.front() != 0)
    {
        throw runtime_error("ERROR: Inconsistent structure for GPU "
                            "symmetry functions.\n");
    }
    size_t const numSlots = neighborOffset.back();
    if (neighborElement.size() != numSlots ||
        neighborTarget.size() != numSlots ||
        neighborGeometry.size() != 4 * numSlots ||
        numSlots + numAtoms > static_cast<size_t>(INT_MAX))
    {
        throw runtime_error("ERROR: Inconsistent or too large structure for "
                            "GPU symmetry functions.\n");
    }
    // Offsets of the values of each atom and of the derivatives of each
    // neighbor entry.
    vector<int> valueOffset(numAtoms + 1, 0);
    vector<int> derivOffset(numSlots, 0);
    size_t numDerivs = 0;
    for (size_t i = 0; i < numAtoms; ++i)
    {
        int const e = atomElement.at(i);
        if (e < 0 || e >= s.numElements)
        {
            throw runtime_error("ERROR: Unknown element for GPU symmetry "
                                "functions.\n");
        }
        int const n = neighborOffset.at(i + 1) - neighborOffset.at(i);
        if (n < 0 || n > static_cast<int>(maxNeighbors))
        {
            throw runtime_error("ERROR: Too many neighbors for GPU symmetry "
                                "functions (" + to_string(n) + ").\n");
        }
        valueOffset.at(i + 1) = valueOffset.at(i) + s.numValues.at(e);
        for (int j = neighborOffset.at(i); j < neighborOffset.at(i + 1); ++j)
        {
            int const ne = neighborElement.at(j);
            if (ne < 0 || ne >= s.numElements)
            {
                throw runtime_error("ERROR: Unknown element for GPU "
                                    "symmetry functions.\n");
            }
            derivOffset.at(j) = numDerivs;
            numDerivs += s.tableCount.at(e * s.numElements + ne);
        }
    }
    if (numDerivs > static_cast<size_t>(INT_MAX))
    {
        throw runtime_error("ERROR: Too many GPU symmetry function "
                            "derivatives.\n");
    }
    size_t const numValues = valueOffset.back();

    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    upload(d.atomElement, atomElement.data(), numAtoms * sizeof(int),
           d.total);
    upload(d.valueOffset, valueOffset.data(), (numAtoms + 1) * sizeof(int),
           d.total);
    upload(d.neighborOffset, neighborOffset.data(),
           (numAtoms + 1) * sizeof(int), d.total);
    upload(d.neighborElement, neighborElement.data(), numSlots * sizeof(int),
           d.total);
    upload(d.neighborGeometry, neighborGeometry.data(),
           4 * numSlots * sizeof(double), d.total);
    d.values.ensure(numValues * sizeof(double) + 1, d.total);
    if (derivatives)
    {
        upload(d.derivOffset, derivOffset.data(), numSlots * sizeof(int),
               d.total);
        upload(d.atomTarget, atomTarget.data(), numAtoms * sizeof(int),
               d.total);
        upload(d.neighborTarget, neighborTarget.data(),
               numSlots * sizeof(int), d.total);
        d.deriv.ensure(3 * numDerivs * sizeof(double) + 1, d.total);
        NNP_CUDA_CHECK(cudaMemset(d.deriv.p, 0,
                                  3 * numDerivs * sizeof(double)));

        // Contributions to each target: the neighbor entries pointing to it
        // and the atom itself, in the order of their index.
        int numTargets = 0;
        for (int t : atomTarget) numTargets = max(numTargets, t + 1);
        for (int t : neighborTarget) numTargets = max(numTargets, t + 1);
        vector<int> targetOffset(numTargets + 1, 0);
        auto target = [&](size_t c) -> int
        {
            return c < numSlots ? neighborTarget[c]
                                : atomTarget[c - numSlots];
        };
        size_t const numContrib = numSlots + numAtoms;
        for (size_t c = 0; c < numContrib; ++c)
        {
            int const t = target(c);
            if (t < 0)
            {
                throw runtime_error("ERROR: Negative target index for GPU "
                                    "symmetry functions.\n");
            }
            targetOffset[t + 1]++;
        }
        for (int t = 0; t < numTargets; ++t)
        {
            targetOffset[t + 1] += targetOffset[t];
        }
        vector<int> targetContrib(numContrib);
        vector<int> position(targetOffset.begin(), targetOffset.end() - 1);
        for (size_t c = 0; c < numContrib; ++c)
        {
            targetContrib[position[target(c)]++] = c;
        }
        upload(d.targetOffset, targetOffset.data(),
               (numTargets + 1) * sizeof(int), d.total);
        upload(d.targetContrib, targetContrib.data(),
               numContrib * sizeof(int), d.total);
        d.numTargets = numTargets;
    }

    if (numAtoms > 0)
    {
        size_t maxNeighborsStructure = 0;
        for (size_t i = 0; i < numAtoms; ++i)
        {
            maxNeighborsStructure = max<size_t>(maxNeighborsStructure,
                                                neighborOffset[i + 1]
                                                - neighborOffset[i]);
        }
        calculateKernel<<<numAtoms,
                          threadsPerBlock,
                          2 * maxNeighborsStructure * sizeof(double)>>>(
            s.chunks,
            s.chunkOffset,
            d.atomElement.get<int>(),
            d.valueOffset.get<int>(),
            d.neighborOffset.get<int>(),
            d.neighborElement.get<int>(),
            d.neighborGeometry.get<double>(),
            d.derivOffset.get<int>(),
            derivatives,
            d.values.get<double>(),
            d.deriv.get<double>());
        NNP_CUDA_CHECK(cudaGetLastError());
    }
    if (numValues > 0)
    {
        NNP_CUDA_CHECK(cudaMemcpy(values, d.values.p,
                                  numValues * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    }

    d.set = set.data;
    d.hasDerivatives = derivatives;
    d.numAtoms = numAtoms;
    d.numSlots = numSlots;
    d.numValues = numValues;
    if (!derivatives) d.numTargets = 0;
}

void GpuSymmetryFunctionData::apply(GpuSymmetryFunctionSet const& set,
                                    size_t                        numWeightVectors,
                                    double const*                 weights,
                                    double*                       result)
{
    Data& d = *data;
    GpuSymmetryFunctionSet::Data const& s = *set.data;
    if (!d.hasDerivatives || d.set != set.data)
    {
        throw runtime_error("ERROR: No GPU symmetry function derivatives for "
                            "these symmetry functions.\n");
    }
    if (numWeightVectors < 1 || numWeightVectors > maxWeightVectors)
    {
        throw runtime_error("ERROR: Wrong number of weight vectors for GPU "
                            "symmetry functions.\n");
    }
    int const nw = numWeightVectors;
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    upload(d.weights, weights, nw * d.numValues * sizeof(double), d.total);
    d.contrib.ensure(3 * nw * (d.numSlots + d.numAtoms) * sizeof(double) + 1,
                     d.total);
    d.result.ensure(3 * nw * d.numTargets * sizeof(double) + 1, d.total);
    if (d.numAtoms > 0)
    {
        contractKernel<<<d.numAtoms, threadsPerBlock>>>(
            nw,
            d.numValues,
            d.numSlots,
            s.numElements,
            d.atomElement.get<int>(),
            d.valueOffset.get<int>(),
            d.neighborOffset.get<int>(),
            d.neighborElement.get<int>(),
            d.derivOffset.get<int>(),
            s.tableOffset,
            s.tableCountDevice,
            s.table,
            d.deriv.get<double>(),
            d.weights.get<double>(),
            d.contrib.get<double>());
        NNP_CUDA_CHECK(cudaGetLastError());
    }
    if (d.numTargets > 0)
    {
        size_t const threads = d.numTargets * threadsPerTarget;
        size_t const blocks = (threads + 255) / 256;
        gatherKernel<<<blocks, 256>>>(d.numTargets,
                                      nw,
                                      d.targetOffset.get<int>(),
                                      d.targetContrib.get<int>(),
                                      d.contrib.get<double>(),
                                      d.result.get<double>());
        NNP_CUDA_CHECK(cudaGetLastError());
        NNP_CUDA_CHECK(cudaMemcpy(result, d.result.p,
                                  3 * nw * d.numTargets * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    }
}

void GpuSymmetryFunctionData::derivativesOfTarget(
                                        GpuSymmetryFunctionSet const& set,
                                        size_t                        target,
                                        size_t                        component,
                                        double*                       result)
{
    Data& d = *data;
    GpuSymmetryFunctionSet::Data const& s = *set.data;
    if (!d.hasDerivatives || d.set != set.data)
    {
        throw runtime_error("ERROR: No GPU symmetry function derivatives for "
                            "these symmetry functions.\n");
    }
    if (target >= d.numTargets || component > 2)
    {
        throw runtime_error("ERROR: Wrong target or component for GPU "
                            "symmetry function derivatives.\n");
    }
    if (d.numValues == 0) return;
    NNP_CUDA_CHECK(cudaSetDevice(d.device));
    d.result.ensure(d.numValues * sizeof(double), d.total);
    NNP_CUDA_CHECK(cudaMemset(d.result.p, 0, d.numValues * sizeof(double)));
    targetDerivativeKernel<<<(d.numAtoms + 127) / 128, 128>>>(
        d.numAtoms,
        s.numElements,
        target,
        component,
        d.atomElement.get<int>(),
        d.atomTarget.get<int>(),
        d.valueOffset.get<int>(),
        d.neighborOffset.get<int>(),
        d.neighborElement.get<int>(),
        d.neighborTarget.get<int>(),
        d.derivOffset.get<int>(),
        s.tableOffset,
        s.tableCountDevice,
        s.table,
        d.deriv.get<double>(),
        d.result.get<double>());
    NNP_CUDA_CHECK(cudaGetLastError());
    NNP_CUDA_CHECK(cudaMemcpy(result, d.result.p,
                              d.numValues * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

size_t GpuSymmetryFunctionData::getNumTargets() const
{
    return data->numTargets;
}

size_t GpuSymmetryFunctionData::getNumValues() const
{
    return data->numValues;
}

size_t GpuSymmetryFunctionData::getMemoryUsage() const
{
    return data->total;
}

}
