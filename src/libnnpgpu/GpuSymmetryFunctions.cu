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
#include <iterator>  // std::prev
#include <algorithm> // std::stable_partition
#include <map>       // std::multimap
#include <mutex>     // std::mutex, std::lock_guard
#include <stdexcept> // std::runtime_error
#include <string>    // std::to_string

using namespace std;

namespace
{

/* Symmetry functions evaluated together (same type, neighbor elements and
 * cutoff), see Chunk. Results do not depend on it. Measured on A100 (2G
 * water and two oxides): 4 is 6-12% faster than 8 (more, smaller work items
 * and fewer registers for the per-lane sums; spills nearly vanish), 3 and 2
 * are slower (the geometry of each pair is computed once per chunk).
 */
int const maxMembers = 4;
int const threadsPerBlock = 128;
int const warpsPerBlock = threadsPerBlock / 32;
/* calculateKernel launch: 64 threads per atom (8 blocks per SM, i.e. at
 * most 65536 / (64 * 8) = 128 registers per thread) from largeLaunchAtoms
 * atoms per launch on, else 128 threads (4 blocks per SM, same register
 * limit). The results do not
 * depend on the block size (see calculateKernel), so this is a pure
 * performance choice. Measured on A100 with 2G water (6.4 A cutoff, 32 ranks
 * on 4 GPUs with MPS): 128 threads for 270 atoms per launch 25% faster than
 * 64 in MD; 64 threads slightly faster from about 2000 atoms per launch on.
 */
int const largeLaunchAtoms = 1024;
/// Shared memory per block of calculateKernel in the normal case (up to 4
/// blocks per SM on A100).
size_t const sharedBudget = 40 * 1024;
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
    /// Index of the cutoff among the distinct cutoffs of the element.
    int    cutoffSlot;
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

/* Contributions of neighbor w (local index among the neighbors of the chunk's
 * elements: those of e1, then those of e2) to the symmetry functions of the
 * chunk, added to g, and the derivatives with respect to its position, which
 * only this call writes. Angular terms of each pair of neighbors are computed
 * twice (once for each neighbor), values are halved later. cache: cutoff
 * function and derivative of the chunk's cutoff for each neighbor j.
 */
__device__ __forceinline__ void neighborTerms(Chunk const&  ch,
                                              int           w,
                                              int           n1,
                                              int           b1,
                                              int           b2,
                                              int const*    order,
                                              int const*    elementOffset,
                                              int const*    elements,
                                              double const* geometry,
                                              double const* cache,
                                              int           n0,
                                              int const*    derivOffset,
                                              bool          derivatives,
                                              double*       deriv,
                                              double        (&g)[maxMembers])
{
    double const rc = ch.cutoff.rc;
    int const j = order[w < n1 ? b1 + w : b2 + w - n1];
    int const ej = elements[j];
    double const rij = geometry[4 * j];
    if (!(rij < rc)) return;
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
        return;
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
    // Neighbors of the partner element.
    int const ep = (ej == ch.e1) ? ch.e2 : ch.e1;
    for (int q = elementOffset[ep]; q < elementOffset[ep + 1]; ++q)
    {
        int const k = order[q];
        if (k == j) continue;
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
    if (!derivatives) return;
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

/* One block per atom. Work items: slices of 32 neighbors of a chunk (the
 * neighbors of the chunk's elements, those of e1 then those of e2, grouped by
 * element once per atom in their order). The warps of the block take the
 * items of all chunks of the atom one after the other, whichever warp is
 * free next, so items of different chunks run in parallel: many small chunks
 * (e.g. rare element pairs) do not leave most threads idle, and no block-wide
 * synchronization is needed per chunk. Each neighbor of an item is done by
 * one lane (neighborTerms()), which writes its derivatives; the derivatives
 * with respect to the position of the atom itself follow from translation
 * invariance (see contract()). The cutoff function of each neighbor is
 * calculated once per atom for each distinct cutoff of the element.
 *
 * Values: each item is summed over the 32 lanes of its warp in a fixed
 * order, and the items of a chunk are added in their order (after the
 * items of a wave of at most `capacity` items are done). The results
 * therefore depend neither on the block size, nor on the capacity, nor on
 * which warp does which item: these are pure performance choices.
 *
 * Dynamic shared memory: cutoff function and derivative of each neighbor for
 * each distinct cutoff (maxCutoffs * 2 * neighborStride doubles), item sums
 * (capacity * maxMembers doubles), chunk sums (maxChunks * maxMembers
 * doubles), neighbors grouped by element (neighborStride ints), first item of
 * each chunk (maxChunks + 1 ints).
 */
template<int threads, int minBlocks>
__global__ void __launch_bounds__(threads, minBlocks)
calculateKernel(Chunk const*  chunks,
                int const*    chunkOffset,
                Cutoff const* cutoffs,
                int const*    cutoffOffset,
                int           numElements,
                int           neighborStride,
                int           maxCutoffs,
                int           maxChunks,
                int           capacity,
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
    extern __shared__ double shared[];
    __shared__ int elementOffset[nnp::GpuSymmetryFunctionSet::maxElements + 1];
    __shared__ int position[nnp::GpuSymmetryFunctionSet::maxElements];
    __shared__ Chunk warpChunk[threads / 32];
    __shared__ int nextItem;
    __shared__ int numItems;

    int const atom = blockIdx.x;
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const warp = tid / 32;
    int const element = atomElement[atom];
    int const n0 = neighborOffset[atom];
    int const numNeighbors = neighborOffset[atom + 1] - n0;
    double const* geometry = neighborGeometry + 4 * static_cast<size_t>(n0);
    int const* elements = neighborElement + n0;
    size_t const stride = 2 * static_cast<size_t>(neighborStride);
    double* cache = shared;
    double* partial = cache + stride * maxCutoffs;
    double* accum = partial + static_cast<size_t>(capacity) * maxMembers;
    int* order = reinterpret_cast<int*>(accum
                                        + static_cast<size_t>(maxChunks)
                                        * maxMembers);
    int* itemOffset = order + neighborStride;
    Chunk const* chunksElement = chunks + chunkOffset[element];
    int const numChunks = chunkOffset[element + 1] - chunkOffset[element];
    Cutoff const* cutoffsElement = cutoffs + cutoffOffset[element];
    int const numCutoffs = cutoffOffset[element + 1] - cutoffOffset[element];

    // Number of neighbors of the elements of a chunk.
    auto chunkNeighbors = [&](Chunk const& c)
    {
        int const n = elementOffset[c.e1 + 1] - elementOffset[c.e1];
        return (c.e2 != c.e1) ? n + elementOffset[c.e2 + 1]
                                  - elementOffset[c.e2] : n;
    };

    if (tid == 0)
    {
        for (int e = 0; e <= numElements; ++e) elementOffset[e] = 0;
        for (int j = 0; j < numNeighbors; ++j) elementOffset[elements[j] + 1]++;
        for (int e = 0; e < numElements; ++e)
        {
            elementOffset[e + 1] += elementOffset[e];
            position[e] = elementOffset[e];
        }
        for (int j = 0; j < numNeighbors; ++j) order[position[elements[j]]++] = j;
        itemOffset[0] = 0;
        for (int c = 0; c < numChunks; ++c)
        {
            itemOffset[c + 1] = itemOffset[c]
                              + (chunkNeighbors(chunksElement[c]) + 31) / 32;
        }
        numItems = itemOffset[numChunks];
    }
    for (int i = tid; i < numCutoffs * numNeighbors; i += blockDim.x)
    {
        int const k = i / numNeighbors;
        int const j = i % numNeighbors;
        Cutoff const& cut = cutoffsElement[k];
        double const r = geometry[4 * j];
        double fc = 0.0;
        double dfc = 0.0;
        if (r < cut.rc) cutoffFunction(cut, r, fc, dfc);
        cache[stride * k + 2 * j] = fc;
        cache[stride * k + 2 * j + 1] = dfc;
    }
    for (int i = tid; i < numChunks * maxMembers; i += blockDim.x)
    {
        accum[i] = 0.0;
    }

    int current = -1;
    Chunk& ch = warpChunk[warp];
    for (int wave = 0; true; wave += capacity)
    {
        if (tid == 0) nextItem = wave;
        __syncthreads();
        if (wave >= numItems) break;
        int const waveEnd = min(numItems, wave + capacity);
        while (true)
        {
            int item = 0;
            if (lane == 0) item = atomicAdd(&nextItem, 1);
            item = __shfl_sync(0xffffffff, item, 0);
            if (item >= waveEnd) break;
            int c = 0;
            while (itemOffset[c + 1] <= item) ++c;
            if (c != current)
            {
                int const* src = reinterpret_cast<int const*>(chunksElement
                                                              + c);
                int* dst = reinterpret_cast<int*>(&ch);
                for (int i = lane;
                     i < static_cast<int>(sizeof(Chunk) / sizeof(int));
                     i += 32)
                {
                    dst[i] = src[i];
                }
                current = c;
                __syncwarp();
            }
            int const b1 = elementOffset[ch.e1];
            int const n1 = elementOffset[ch.e1 + 1] - b1;
            int const b2 = elementOffset[ch.e2];
            int const n = chunkNeighbors(ch);
            int const w = 32 * (item - itemOffset[c]) + lane;
            double g[maxMembers];
            for (int m = 0; m < maxMembers; ++m) g[m] = 0.0;
            if (w < n)
            {
                neighborTerms(ch, w, n1, b1, b2, order, elementOffset,
                              elements, geometry,
                              cache + stride * ch.cutoffSlot, n0,
                              derivOffset, derivatives, deriv, g);
            }
            for (int m = 0; m < maxMembers; ++m)
            {
                double v = g[m];
                for (int offset = 16; offset > 0; offset /= 2)
                {
                    v += __shfl_down_sync(0xffffffff, v, offset);
                }
                if (lane == 0)
                {
                    partial[(item - wave) * maxMembers + m] = v;
                }
            }
        }
        __syncthreads();
        // Item sums of this wave added to their chunks, in item order.
        for (int i = tid; i < numChunks * maxMembers; i += blockDim.x)
        {
            int const c = i / maxMembers;
            int const m = i % maxMembers;
            int const begin = max(itemOffset[c], wave);
            int const end = min(itemOffset[c + 1], waveEnd);
            for (int item = begin; item < end; ++item)
            {
                accum[i] += partial[(item - wave) * maxMembers + m];
            }
        }
        __syncthreads();
    }

    // Scaled values.
    for (int i = tid; i < numChunks * maxMembers; i += blockDim.x)
    {
        int const m = i % maxMembers;
        Chunk const& c = chunksElement[i / maxMembers];
        if (m >= c.numMembers) continue;
        double sum = accum[i];
        if (c.kind != RADIAL) sum = 0.5 * sum * c.pnorm[m];
        values[valueOffset[atom] + c.index[m]] =
            c.scaleOffset[m] + c.scaleFactor[m] * (sum - c.scaleShift[m]);
    }
}

typedef void (*CalculateKernel)(Chunk const*, int const*, Cutoff const*,
                                int const*, int, int, int, int, int,
                                int const*, int const*, int const*,
                                int const*, double const*, int const*, bool,
                                double*, double*);

// calculateKernel for a block size (64 or 128 threads), both limited to
// 128 registers per thread.
CalculateKernel calculateKernelFor(int threads)
{
    if (threads == 64) return calculateKernel<64, 8>;
    return calculateKernel<128, 4>;
}

// Static shared memory of a kernel.
size_t staticShared(CalculateKernel kernel)
{
    cudaFuncAttributes a;
    NNP_CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
    return a.sharedSizeBytes;
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

/* Device memory released by GpuSymmetryFunctionData objects, kept for reuse:
 * in training, the symmetry functions of the structures are freed and
 * calculated again over and over, and cudaMalloc/cudaFree are slow and
 * synchronize the device. A request takes the smallest cached block that fits
 * and is at most twice as large, otherwise new memory (25% headroom). Cached
 * blocks beyond a quarter of the cache budget of the process
 * (gpuDefaultCacheBudget()) are freed, largest first. One pool per process,
 * which uses one device.
 */
class DevicePool
{
public:
    DevicePool() : cached(0), limit(nnp::gpuDefaultCacheBudget() / 4) {}
    void* get(size_t required, size_t& bytes)
    {
        lock_guard<mutex> lock(m);
        auto it = blocks.lower_bound(required);
        if (it != blocks.end() && it->first <= 2 * required)
        {
            bytes = it->first;
            void* p = it->second;
            cached -= bytes;
            blocks.erase(it);
            return p;
        }
        bytes = required + required / 4;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess)
        {
            // Out of memory: give the cached blocks back and try once more.
            (void)cudaGetLastError();
            for (auto const& b : blocks) cudaFree(b.second);
            blocks.clear();
            cached = 0;
            NNP_CUDA_CHECK(cudaMalloc(&p, bytes));
        }
        return p;
    }
    // No checks: also called from destructors, which must not throw.
    void put(void* p, size_t bytes)
    {
        if (p == nullptr) return;
        lock_guard<mutex> lock(m);
        blocks.emplace(bytes, p);
        cached += bytes;
        while (cached > limit && !blocks.empty())
        {
            auto it = prev(blocks.end());
            cudaFree(it->second);
            cached -= it->first;
            blocks.erase(it);
        }
    }

private:
    mutex                     m;
    multimap<size_t, void*>   blocks;
    size_t                    cached;
    size_t                    limit;
};

// Never destroyed: at exit the CUDA context may already be gone.
DevicePool& devicePool()
{
    static DevicePool* pool = new DevicePool();
    return *pool;
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
        total -= bytes;
        devicePool().put(p, bytes);
        p = nullptr;
        bytes = 0;
        p = devicePool().get(required, bytes);
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
    /// Distinct cutoffs of each element (Chunk::cutoffSlot).
    Cutoff*     cutoffs;
    int*        cutoffOffset;
    /// Largest number of distinct cutoffs and of chunks of an element.
    int         maxCutoffs;
    int         maxChunks;
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
    d.cutoffs = nullptr;
    d.cutoffOffset = nullptr;
    d.maxCutoffs = 0;
    d.maxChunks = 0;
    d.tableOffset = nullptr;
    d.tableCountDevice = nullptr;
    d.table = nullptr;
    try
    {
        int const numElements = symmetryFunctions.size();
        if (numElements > static_cast<int>(maxElements))
        {
            throw runtime_error("ERROR: Too many elements for GPU symmetry "
                                "functions.\n");
        }
        d.numElements = numElements;
        vector<Chunk> chunks;
        vector<int> chunkOffset(1, 0);
        vector<Cutoff> cutoffs;
        vector<int> cutoffOffset(1, 0);
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
                c.cutoffSlot = -1;
                for (int k = cutoffOffset.back();
                     k < static_cast<int>(cutoffs.size()); ++k)
                {
                    Cutoff const& o = cutoffs.at(k);
                    if (o.type == c.cutoff.type && o.rc == c.cutoff.rc &&
                        o.rci == c.cutoff.rci)
                    {
                        c.cutoffSlot = k - cutoffOffset.back();
                    }
                }
                if (c.cutoffSlot < 0)
                {
                    c.cutoffSlot = cutoffs.size() - cutoffOffset.back();
                    cutoffs.push_back(c.cutoff);
                }
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
            // Angular chunks first: the most expensive work items of an atom
            // start first, the cheap radial ones fill in at the end (the
            // order of the chunks does not change any result).
            stable_partition(chunks.begin() + chunkOffset.back(), chunks.end(),
                             [](Chunk const& c) { return c.kind != RADIAL; });
            chunkOffset.push_back(chunks.size());
            cutoffOffset.push_back(cutoffs.size());
            d.maxChunks = max(d.maxChunks,
                              chunkOffset.back() - *(chunkOffset.end() - 2));
            d.maxCutoffs = max(d.maxCutoffs,
                               cutoffOffset.back()
                               - *(cutoffOffset.end() - 2));
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
        copy(reinterpret_cast<void**>(&d.cutoffs), cutoffs.data(),
             cutoffs.size() * sizeof(Cutoff));
        copy(reinterpret_cast<void**>(&d.cutoffOffset), cutoffOffset.data(),
             cutoffOffset.size() * sizeof(int));
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
        cudaFree(d.cutoffs);
        cudaFree(d.cutoffOffset);
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
    cudaFree(d.cutoffs);
    cudaFree(d.cutoffOffset);
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
        devicePool().put(b->p, b->bytes);
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
        // See largeLaunchAtoms.
        int threads = 128;
        if (numAtoms >= static_cast<size_t>(largeLaunchAtoms)) threads = 64;
        CalculateKernel kernel = calculateKernelFor(threads);
        // Shared memory: fixed part, plus the item sums of one wave
        // (capacity items). Results do not depend on the capacity: as many
        // items as all chunks can have, within sharedBudget per block.
        size_t const stride = maxNeighborsStructure;
        size_t const fixed = s.maxCutoffs * 2 * stride * sizeof(double)
                           + s.maxChunks * maxMembers * sizeof(double)
                           + stride * sizeof(int)
                           + (s.maxChunks + 1) * sizeof(int);
        size_t const perItem = maxMembers * sizeof(double);
        size_t const maxItems = s.maxChunks * ((stride + 31) / 32);
        size_t const used = fixed + staticShared(kernel);
        size_t capacity = used + perItem < sharedBudget
                        ? (sharedBudget - used) / perItem : 1;
        capacity = max<size_t>(1, min(capacity, maxItems));
        size_t const sharedBytes = fixed + capacity * perItem;
        if (sharedBytes + staticShared(kernel) > 48 * 1024)
        {
            // Very many neighbors or cutoffs: opt in to more shared memory.
            int optin = 0;
            NNP_CUDA_CHECK(cudaDeviceGetAttribute(
                               &optin,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               d.device));
            if (sharedBytes + staticShared(kernel)
                > static_cast<size_t>(optin))
            {
                throw runtime_error("ERROR: Too many neighbors for GPU "
                                    "symmetry functions on this device.\n");
            }
            NNP_CUDA_CHECK(cudaFuncSetAttribute(
                               kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(sharedBytes)));
        }
        kernel<<<numAtoms, threads, sharedBytes>>>(s.chunks,
                                                   s.chunkOffset,
                                                   s.cutoffs,
                                                   s.cutoffOffset,
                                                   s.numElements,
                                                   maxNeighborsStructure,
                                                   s.maxCutoffs,
                                                   s.maxChunks,
                                                   capacity,
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
