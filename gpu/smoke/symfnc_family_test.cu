// GPU smoke test, all 11 of n2p2's leaf symmetry function types.
//
// Each type is a from-scratch CUDA device function mirroring the exact CPU
// math in the corresponding src/libnnp/SymFnc*.cpp, one thread per
// (selected) central atom:
//   Radial family:
//   - SymFncExpRad          (type 2,  radial, TANHU cutoff)
//   - SymFncExpRadWeighted  (type 12, radial, no element filter, Z-weighted)
//   - SymFncCompRad         (type 20, radial, compact-support core POLY2)
//   - SymFncCompRadWeighted (type 23, radial, compact core, Z-weighted)
//   Exp-angular family (classic (1+lambda*cos theta)^zeta form):
//   - SymFncExpAngn         (type 3,  narrow angular, 3-distance dependent)
//   - SymFncExpAngnWeighted (type 13, narrow angular, no filter, Z-weighted)
//   - SymFncExpAngw         (type 9,  wide angular, 2-distance dependent)
//   Compact-angular family (angle-space CompactFunction of acos(cos theta)):
//   - SymFncCompAngn         (type 21, narrow angular, 3-distance dependent)
//   - SymFncCompAngnWeighted (type 24, narrow angular, no filter, Z-weighted)
//   - SymFncCompAngw         (type 22, wide angular, 2-distance dependent)
//   - SymFncCompAngwWeighted (type 25, wide angular, no filter, Z-weighted)
//
// Scope, consistent with symfnc_exprad_test.cu: validates the unscaled
// energy accumulator ("result" in the CPU source, before the final scale()
// affine transform) and the CENTRAL atom's own accumulated derivative
// (atom.dGdr[index] in the CPU source) with scalingFactor treated as 1.
// Neighbor-side derivative bookkeeping (n.dGdr) is out of scope, same as
// before. For angular functions this also means the "rjk" force term
// (which only ever contributes to neighbor-side derivatives, never to
// atom.dGdr[index] -- see SymFncExpAngn.cpp/SymFncExpAngw.cpp) is provably
// irrelevant to what we validate here, so it isn't computed at all.
//
// Real neighbor geometry: loaded from dump_real_neighbors.cpp's output,
// n2p2's own Structure::calculateNeighborList() on temp/H2O_2G/input.data
// (real_neighbors_full.txt: every atom's complete, unfiltered neighbor
// list). Real production parameters (from temp/H2O_2G/input.nn) are used
// for ExpRad and ExpAngn, the two types the real input.nn actually uses;
// the other 5 types use representative parameter values on the same real
// geometry, since input.nn doesn't define instances of them.

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

///////////////////////////////////////////////////////////////////////////
// Shared math building blocks
///////////////////////////////////////////////////////////////////////////

// TANHU cutoff (n2p2 CutoffType::CT_TANHU = 2, CutoffFunction.cpp:156).
__host__ __device__ inline void cutoffTANHU(double r, double rcinv,
                                             double& fc, double& dfc)
{
    double t  = tanh(1.0 - r * rcinv);
    double t2 = t * t;
    fc  = t * t2;
    dfc = 3.0 * t2 * (t2 - 1.0) * rcinv;
}

// CompactFunction with POLY2 core (CompactFunction.h/CoreFunction.h's
// default: Type::POLY2, asymmetric=false), given left/right bounds.
// fPOLY2(x) = ((15-6x)x-10)x^3 + 1, dfPOLY2(x) = x^2*((60-30x)x-30).
__host__ __device__ inline void compactPoly2(double r, double left,
                                              double right,
                                              double& f, double& df)
{
    double center = 0.5 * (left + right);
    double width   = 0.5 * (right - left);
    double scale   = 1.0 / width;
    double a       = (r - center) * scale;
    double x       = fabs(a);
    double x2      = x * x;
    double fx  = ((15.0 - 6.0 * x) * x - 10.0) * x * x2 + 1.0;
    double dfx = x2 * ((60.0 - 30.0 * x) * x - 30.0);
    f  = fx;
    df = copysign(scale * dfx, a);
}

// Exact port of utility.cpp's pow_int (exponentiation by squaring), used
// whenever zeta is an integer -- matches n2p2's useIntegerPow path exactly.
__host__ __device__ inline double pow_int(double x, int n)
{
    unsigned int m;
    if (n < 0) { x = 1.0 / x; m = (unsigned int)(-n); }
    else m = (unsigned int)n;
    double result = 1.0;
    do
    {
        if (m & 1) result *= x;
        m >>= 1;
        x *= x;
    } while (m);
    return result;
}

// Atomic number lookup for this test's ElementMap ("H O" -> sorted by
// atomic number, H=0 (Z=1), O=1 (Z=8); see ElementMap::registerElements).
__host__ __device__ inline double atomicNumberOf(int elementIndex)
{
    return elementIndex == 0 ? 1.0 : 8.0;
}

///////////////////////////////////////////////////////////////////////////
// Radial family (SymFnc{Exp,Comp}Rad{,Weighted}.cpp)
///////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void symFncExpRad(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, double eta, double rs, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double rcinv = 1.0 / rc;
    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < numNeighbors; ++j)
    {
        if (neighElem[j] != e1) continue;
        double rij = neighDist[j];
        if (rij >= rc) continue;
        double diff = rij - rs;
        double pexp = exp(-eta * diff * diff);
        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);
        result += pexp * pfc;
        double p1 = (pdfc - 2.0 * eta * diff * pfc) * pexp / rij;
        fx += p1 * neighDx[j]; fy += p1 * neighDy[j]; fz += p1 * neighDz[j];
    }
    G = result; dGdx = fx; dGdy = fy; dGdz = fz;
}

__host__ __device__ inline void symFncExpRadWeighted(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double eta, double rs, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double rcinv = 1.0 / rc;
    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < numNeighbors; ++j)
    {
        double rij = neighDist[j];
        if (rij >= rc) continue;
        double w = atomicNumberOf(neighElem[j]);
        double diff = rij - rs;
        double pexp = w * exp(-eta * diff * diff);
        double pfc, pdfc;
        cutoffTANHU(rij, rcinv, pfc, pdfc);
        result += pexp * pfc;
        double p1 = (pdfc - 2.0 * eta * diff * pfc) * pexp / rij;
        fx += p1 * neighDx[j]; fy += p1 * neighDy[j]; fz += p1 * neighDz[j];
    }
    G = result; dGdx = fx; dGdy = fy; dGdz = fz;
}

__host__ __device__ inline void symFncCompRad(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, double rl, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < numNeighbors; ++j)
    {
        if (neighElem[j] != e1) continue;
        double rij = neighDist[j];
        if (!(rij > rl && rij < rc)) continue;
        double rad, drad;
        compactPoly2(rij, rl, rc, rad, drad);
        result += rad;
        double p1 = drad / rij;
        fx += p1 * neighDx[j]; fy += p1 * neighDy[j]; fz += p1 * neighDz[j];
    }
    G = result; dGdx = fx; dGdy = fy; dGdz = fz;
}

__host__ __device__ inline void symFncCompRadWeighted(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double rl, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < numNeighbors; ++j)
    {
        double rij = neighDist[j];
        if (!(rij > rl && rij < rc)) continue;
        double w = atomicNumberOf(neighElem[j]);
        double rad, drad;
        compactPoly2(rij, rl, rc, rad, drad);
        result += rad * w;
        double p1 = w * drad / rij;
        fx += p1 * neighDx[j]; fy += p1 * neighDy[j]; fz += p1 * neighDz[j];
    }
    G = result; dGdx = fx; dGdy = fy; dGdz = fz;
}

///////////////////////////////////////////////////////////////////////////
// Exp-angular family (SymFncExpAng{n,w}{,Weighted}.cpp)
///////////////////////////////////////////////////////////////////////////

// Narrow angular (type 3): 3-distance dependent (rij, rik, AND rjk both in
// the exponential and as a third cutoff factor). e1==-1 means "no element
// filter" (the Weighted variant: sum over all neighbor-pair elements,
// weighted by Z(nej)*Z(nek) instead).
__host__ __device__ inline void symFncExpAngnCore(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, int e2, bool weighted,
    double eta, double rs, double lambda, double zeta, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double const pnorm = pow(2.0, 1.0 - zeta);
    double const pzl   = zeta * lambda;
    double const rc2   = rc * rc;
    int const zetaInt  = (int)llround(zeta);
    bool const useIntegerPow = (fabs(zeta - zetaInt) <= 1e-12);

    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < numNeighbors - 1; ++j)
    {
        int nej = neighElem[j];
        double rij = neighDist[j];
        if (!weighted && !(nej == e1 || nej == e2)) continue;
        if (!(rij < rc)) continue;

        double pfcij, pdfcij;
        cutoffTANHU(rij, 1.0 / rc, pfcij, pdfcij);

        for (int k = j + 1; k < numNeighbors; ++k)
        {
            int nek = neighElem[k];
            if (!weighted)
            {
                if (!((nej == e1 && nek == e2) || (nej == e2 && nek == e1)))
                    continue;
            }
            double rik = neighDist[k];
            if (!(rik < rc)) continue;

            double djkx = neighDx[k] - neighDx[j];
            double djky = neighDy[k] - neighDy[j];
            double djkz = neighDz[k] - neighDz[j];
            double rjk2 = djkx * djkx + djky * djky + djkz * djkz;
            if (!(rjk2 < rc2)) continue;
            double rjk = sqrt(rjk2);

            double pfcik, pdfcik;
            cutoffTANHU(rik, 1.0 / rc, pfcik, pdfcik);
            double pfcjk, pdfcjk;
            cutoffTANHU(rjk, 1.0 / rc, pfcjk, pdfcjk);

            double dijx = neighDx[j], dijy = neighDy[j], dijz = neighDz[j];
            double dikx = neighDx[k], diky = neighDy[k], dikz = neighDz[k];
            double costijk = dijx * dikx + dijy * diky + dijz * dikz;
            double rinvijik = 1.0 / rij / rik;
            costijk *= rinvijik;

            double pfc  = pfcij * pfcik * pfcjk;
            double r2ij = rij * rij, r2ik = rik * rik;
            double rijs = rij - rs, riks = rik - rs, rjks = rjk - rs;
            double pexp = exp(-eta * (rijs * rijs + riks * riks + rjks * rjks));
            if (weighted) pexp *= atomicNumberOf(nej) * atomicNumberOf(nek);

            double plambda = 1.0 + lambda * costijk;
            double fg = pexp;
            if (plambda <= 0.0) fg = 0.0;
            else fg *= useIntegerPow ? pow_int(plambda, zetaInt - 1)
                                     : pow(plambda, zeta - 1.0);

            result += fg * plambda * pfc;

            // Force: only the central atom's own contribution
            // (atom.dGdr[index] += drij + drik) is in scope here -- the
            // rjk/p3 term only ever affects neighbor-side bookkeeping
            // (nj.dGdr/nk.dGdr in the CPU source), never atom.dGdr[index].
            double fgF = fg * pnorm;
            double rinvijikF = rinvijik * pzl;
            double costijkF  = costijk * pzl;
            double p2etapl = 2.0 * eta * plambda;
            double p1 = fgF * (pfc * (rinvijikF - costijkF / r2ij
                        - p2etapl * rijs / rij)
                        + pfcik * pfcjk * pdfcij * plambda / rij);
            double p2 = fgF * (pfc * (rinvijikF - costijkF / r2ik
                        - p2etapl * riks / rik)
                        + pfcij * pfcjk * pdfcik * plambda / rik);

            fx += p1 * dijx + p2 * dikx;
            fy += p1 * dijy + p2 * diky;
            fz += p1 * dijz + p2 * dikz;
        }
    }
    G = result * pnorm; dGdx = fx; dGdy = fy; dGdz = fz;
}

__host__ __device__ inline void symFncExpAngn(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, int e2,
    double eta, double rs, double lambda, double zeta, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    symFncExpAngnCore(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                       neighDz, e1, e2, /*weighted=*/false, eta, rs, lambda,
                       zeta, rc, G, dGdx, dGdy, dGdz);
}

__host__ __device__ inline void symFncExpAngnWeighted(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz,
    double eta, double rs, double lambda, double zeta, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    symFncExpAngnCore(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                       neighDz, -1, -1, /*weighted=*/true, eta, rs, lambda,
                       zeta, rc, G, dGdx, dGdy, dGdz);
}

// Wide angular (type 9): only rij, rik (no rjk in the exponential or as a
// cutoff factor -- SymFncExpAngw.cpp has no rjk<rc check and pfc=pfcij*pfcik
// only). e1/e2 element filter (this type has no Weighted variant in n2p2).
__host__ __device__ inline void symFncExpAngw(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, int e2,
    double eta, double rs, double lambda, double zeta, double rc,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double const pnorm = pow(2.0, 1.0 - zeta);
    double const pzl   = zeta * lambda;
    int const zetaInt  = (int)llround(zeta);
    bool const useIntegerPow = (fabs(zeta - zetaInt) <= 1e-12);

    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < numNeighbors - 1; ++j)
    {
        int nej = neighElem[j];
        double rij = neighDist[j];
        if (!(nej == e1 || nej == e2)) continue;
        if (!(rij < rc)) continue;

        double pfcij, pdfcij;
        cutoffTANHU(rij, 1.0 / rc, pfcij, pdfcij);

        for (int k = j + 1; k < numNeighbors; ++k)
        {
            int nek = neighElem[k];
            if (!((nej == e1 && nek == e2) || (nej == e2 && nek == e1)))
                continue;
            double rik = neighDist[k];
            if (!(rik < rc)) continue;

            double pfcik, pdfcik;
            cutoffTANHU(rik, 1.0 / rc, pfcik, pdfcik);

            double dijx = neighDx[j], dijy = neighDy[j], dijz = neighDz[j];
            double dikx = neighDx[k], diky = neighDy[k], dikz = neighDz[k];
            double costijk = dijx * dikx + dijy * diky + dijz * dikz;
            double rinvijik = 1.0 / rij / rik;
            costijk *= rinvijik;

            double pfc  = pfcij * pfcik;
            double r2ij = rij * rij, r2ik = rik * rik;
            double rijs = rij - rs, riks = rik - rs;
            double pexp = exp(-eta * (rijs * rijs + riks * riks));

            double plambda = 1.0 + lambda * costijk;
            double fg = pexp;
            if (plambda <= 0.0) fg = 0.0;
            else fg *= useIntegerPow ? pow_int(plambda, zetaInt - 1)
                                     : pow(plambda, zeta - 1.0);

            result += fg * plambda * pfc;

            double fgF = fg * pnorm;
            double rinvijikF = rinvijik * pzl;
            double costijkF  = costijk * pzl;
            double p2etapl = 2.0 * eta * plambda;
            double p1 = fgF * (pfc * (rinvijikF - costijkF / r2ij
                        - p2etapl * rijs / rij) + pfcik * pdfcij * plambda / rij);
            double p2 = fgF * (pfc * (rinvijikF - costijkF / r2ik
                        - p2etapl * riks / rik) + pfcij * pdfcik * plambda / rik);

            fx += p1 * dijx + p2 * dikx;
            fy += p1 * dijy + p2 * diky;
            fz += p1 * dijz + p2 * dikz;
        }
    }
    G = result * pnorm; dGdx = fx; dGdy = fy; dGdz = fz;
}

///////////////////////////////////////////////////////////////////////////
// Compact-angular family (SymFncCompAng{n,w}{,Weighted}.cpp)
//
// A completely different parameterization from the Exp-angular family
// above: the angular part is a CompactFunction (same POLY2 core as the
// radial part) evaluated at acos(cos theta) directly -- compact support in
// angle-space (angleLeftRadians/angleRightRadians), not (1+lambda*cos)^zeta.
// weighted=true folds Z(nej)*Z(nek) into ang/dang (SymFncCompAngnWeighted.cpp
// multiplies "ang *=" and "dang *=" by the atomic-number product, not pexp
// as in the Exp-angular weighted case) -- verified this still satisfies the
// product rule correctly (both the rad*phi and ang*chi force terms end up
// weighted, since dang/ang carry the weight into every term they appear in).
///////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void symFncCompAngnCore(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, int e2, bool weighted,
    double rl, double rc, double angleLeftRad, double angleRightRad,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double const r2l = (rl > 0.0) ? rl * rl : 0.0;
    double const r2c = rc * rc;
    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < numNeighbors - 1; ++j)
    {
        int nej = neighElem[j];
        double rij = neighDist[j];
        if (!weighted && !(nej == e1 || nej == e2)) continue;
        if (!(rij < rc && rij > rl)) continue;

        double radij, dradij;
        compactPoly2(rij, rl, rc, radij, dradij);

        for (int k = j + 1; k < numNeighbors; ++k)
        {
            int nek = neighElem[k];
            if (!weighted)
            {
                if (!((nej == e1 && nek == e2) || (nej == e2 && nek == e1)))
                    continue;
            }
            double rik = neighDist[k];
            if (!(rik < rc && rik > rl)) continue;

            double djkx = neighDx[k] - neighDx[j];
            double djky = neighDy[k] - neighDy[j];
            double djkz = neighDz[k] - neighDz[j];
            double rjk2 = djkx * djkx + djky * djky + djkz * djkz;
            if (!(rjk2 < r2c && rjk2 > r2l)) continue;
            double rjk = sqrt(rjk2);

            double radik, dradik;
            compactPoly2(rik, rl, rc, radik, dradik);
            double radjk, dradjk;
            compactPoly2(rjk, rl, rc, radjk, dradjk);

            double dijx = neighDx[j], dijy = neighDy[j], dijz = neighDz[j];
            double dikx = neighDx[k], diky = neighDy[k], dikz = neighDz[k];
            double costijk = dijx * dikx + dijy * diky + dijz * dikz;
            double rinvijik = 1.0 / rij / rik;
            costijk *= rinvijik;

            if (costijk <= -1.0 || costijk >= 1.0) continue;
            double acostijk = acos(costijk);
            if (acostijk < angleLeftRad || acostijk > angleRightRad) continue;

            double ang, dang;
            compactPoly2(acostijk, angleLeftRad, angleRightRad, ang, dang);

            double weight = weighted ? atomicNumberOf(nej) * atomicNumberOf(nek) : 1.0;
            double angW = ang * weight;
            double rad = radij * radik * radjk;
            result += rad * angW;

            double dacostijk = -1.0 / sqrt(1.0 - costijk * costijk);
            double dangW = dang * dacostijk * weight;

            double rinvij = rinvijik * rik;
            double rinvik = rinvijik * rij;
            double phiijik = rinvij * (rinvik - rinvij * costijk);
            double phiikij = rinvik * (rinvij - rinvik * costijk);
            phiijik *= dangW;
            phiikij *= dangW;

            double chiij = rinvij * dradij * radik * radjk;
            double chiik = rinvik * radij * dradik * radjk;

            double p1 = rad * phiijik + angW * chiij;
            double p2 = rad * phiikij + angW * chiik;

            fx += p1 * dijx + p2 * dikx;
            fy += p1 * dijy + p2 * diky;
            fz += p1 * dijz + p2 * dikz;
        }
    }
    G = result; dGdx = fx; dGdy = fy; dGdz = fz;
}

__host__ __device__ inline void symFncCompAngn(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, int e2, double rl, double rc,
    double angleLeftRad, double angleRightRad,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    symFncCompAngnCore(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                        neighDz, e1, e2, /*weighted=*/false, rl, rc,
                        angleLeftRad, angleRightRad, G, dGdx, dGdy, dGdz);
}

__host__ __device__ inline void symFncCompAngnWeighted(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double rl, double rc,
    double angleLeftRad, double angleRightRad,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    symFncCompAngnCore(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                        neighDz, -1, -1, /*weighted=*/true, rl, rc,
                        angleLeftRad, angleRightRad, G, dGdx, dGdy, dGdz);
}

// Wide compact angular: only rij, rik (no rjk/radjk term -- mirrors how
// ExpAngw drops rjk relative to ExpAngn).
__host__ __device__ inline void symFncCompAngwCore(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, int e2, bool weighted,
    double rl, double rc, double angleLeftRad, double angleRightRad,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    double result = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < numNeighbors - 1; ++j)
    {
        int nej = neighElem[j];
        double rij = neighDist[j];
        if (!weighted && !(nej == e1 || nej == e2)) continue;
        if (!(rij < rc && rij > rl)) continue;

        double radij, dradij;
        compactPoly2(rij, rl, rc, radij, dradij);

        for (int k = j + 1; k < numNeighbors; ++k)
        {
            int nek = neighElem[k];
            if (!weighted)
            {
                if (!((nej == e1 && nek == e2) || (nej == e2 && nek == e1)))
                    continue;
            }
            double rik = neighDist[k];
            if (!(rik < rc && rik > rl)) continue;

            double radik, dradik;
            compactPoly2(rik, rl, rc, radik, dradik);

            double dijx = neighDx[j], dijy = neighDy[j], dijz = neighDz[j];
            double dikx = neighDx[k], diky = neighDy[k], dikz = neighDz[k];
            double costijk = dijx * dikx + dijy * diky + dijz * dikz;
            double rinvijik = 1.0 / rij / rik;
            costijk *= rinvijik;

            if (costijk <= -1.0 || costijk >= 1.0) continue;
            double acostijk = acos(costijk);
            if (acostijk < angleLeftRad || acostijk > angleRightRad) continue;

            double ang, dang;
            compactPoly2(acostijk, angleLeftRad, angleRightRad, ang, dang);

            double weight = weighted ? atomicNumberOf(nej) * atomicNumberOf(nek) : 1.0;
            double angW = ang * weight;
            double rad = radij * radik;
            result += rad * angW;

            double dacostijk = -1.0 / sqrt(1.0 - costijk * costijk);
            double dangW = dang * dacostijk * weight;

            double rinvij = rinvijik * rik;
            double rinvik = rinvijik * rij;
            double phiijik = rinvij * (rinvik - rinvij * costijk);
            double phiikij = rinvik * (rinvij - rinvik * costijk);
            phiijik *= dangW;
            phiikij *= dangW;

            double chiij = rinvij * radik * dradij;
            double chiik = rinvik * radij * dradik;

            double p1 = rad * phiijik + angW * chiij;
            double p2 = rad * phiikij + angW * chiik;

            fx += p1 * dijx + p2 * dikx;
            fy += p1 * dijy + p2 * diky;
            fz += p1 * dijz + p2 * dikz;
        }
    }
    G = result; dGdx = fx; dGdy = fy; dGdz = fz;
}

__host__ __device__ inline void symFncCompAngw(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, int e1, int e2, double rl, double rc,
    double angleLeftRad, double angleRightRad,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    symFncCompAngwCore(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                        neighDz, e1, e2, /*weighted=*/false, rl, rc,
                        angleLeftRad, angleRightRad, G, dGdx, dGdy, dGdz);
}

__host__ __device__ inline void symFncCompAngwWeighted(
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz, double rl, double rc,
    double angleLeftRad, double angleRightRad,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    symFncCompAngwCore(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                        neighDz, -1, -1, /*weighted=*/true, rl, rc,
                        angleLeftRad, angleRightRad, G, dGdx, dGdy, dGdz);
}

///////////////////////////////////////////////////////////////////////////
// Real data loading (dump_real_neighbors.cpp's output)
///////////////////////////////////////////////////////////////////////////

struct RealSystem
{
    int numAtoms;
    std::vector<int> atomElement;
    std::vector<int> neighCount, neighOffset;
    std::vector<int> neighElement;
    std::vector<double> neighDist, neighDx, neighDy, neighDz;
};

RealSystem loadRealSystem(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in) { fprintf(stderr, "Could not open %s\n", filename.c_str()); exit(1); }

    std::vector<int> idx, elemA, elemN;
    std::vector<double> dist, dx, dy, dz;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int ai, ae, ne; double r, x, y, z;
        iss >> ai >> ae >> ne >> r >> x >> y >> z;
        idx.push_back(ai); elemA.push_back(ae); elemN.push_back(ne);
        dist.push_back(r); dx.push_back(x); dy.push_back(y); dz.push_back(z);
    }
    if (idx.empty()) { fprintf(stderr, "No data in %s\n", filename.c_str()); exit(1); }

    int numAtoms = idx.back() + 1;

    RealSystem sys;
    sys.numAtoms = numAtoms;
    sys.atomElement.assign(numAtoms, -1);
    sys.neighCount.assign(numAtoms, 0);
    sys.neighOffset.assign(numAtoms, 0);
    for (size_t k = 0; k < idx.size(); ++k)
    {
        sys.atomElement[idx[k]] = elemA[k];
        sys.neighCount[idx[k]]++;
    }

    int total = 0;
    for (int i = 0; i < numAtoms; ++i) { sys.neighOffset[i] = total; total += sys.neighCount[i]; }

    sys.neighElement.resize(total);
    sys.neighDist.resize(total);
    sys.neighDx.resize(total);
    sys.neighDy.resize(total);
    sys.neighDz.resize(total);

    std::vector<int> cursor = sys.neighOffset;
    for (size_t k = 0; k < idx.size(); ++k)
    {
        int pos = cursor[idx[k]]++;
        sys.neighElement[pos] = elemN[k];
        sys.neighDist[pos] = dist[k];
        sys.neighDx[pos] = dx[k];
        sys.neighDy[pos] = dy[k];
        sys.neighDz[pos] = dz[k];
    }
    return sys;
}

///////////////////////////////////////////////////////////////////////////
// CPU reference + GPU kernel + comparison, generic over "which symfunc"
///////////////////////////////////////////////////////////////////////////

enum class SfKind { ExpRad, ExpRadWeighted, CompRad, CompRadWeighted,
                     ExpAngn, ExpAngnWeighted, ExpAngw,
                     CompAngn, CompAngnWeighted, CompAngw, CompAngwWeighted };

struct SfParams
{
    SfKind kind;
    int centralElement;   // ec: which atoms to treat as central
    int e1 = -1, e2 = -1; // neighbor element filter (unused if weighted)
    double eta = 0, rs = 0, rl = 0, lambda = 0, zeta = 0, rc = 12.0;
    double angleLeftRad = 0.0, angleRightRad = M_PI; // compact-angular only
};

__host__ __device__ inline void evalSf(
    const SfParams& p,
    int numNeighbors, const int* neighElem,
    const double* neighDist, const double* neighDx, const double* neighDy,
    const double* neighDz,
    double& G, double& dGdx, double& dGdy, double& dGdz)
{
    switch (p.kind)
    {
        case SfKind::ExpRad:
            symFncExpRad(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                         neighDz, p.e1, p.eta, p.rs, p.rc, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::ExpRadWeighted:
            symFncExpRadWeighted(numNeighbors, neighElem, neighDist, neighDx,
                         neighDy, neighDz, p.eta, p.rs, p.rc, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::CompRad:
            symFncCompRad(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                         neighDz, p.e1, p.rl, p.rc, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::CompRadWeighted:
            symFncCompRadWeighted(numNeighbors, neighElem, neighDist, neighDx,
                         neighDy, neighDz, p.rl, p.rc, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::ExpAngn:
            symFncExpAngn(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                         neighDz, p.e1, p.e2, p.eta, p.rs, p.lambda, p.zeta,
                         p.rc, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::ExpAngnWeighted:
            symFncExpAngnWeighted(numNeighbors, neighElem, neighDist, neighDx,
                         neighDy, neighDz, p.eta, p.rs, p.lambda, p.zeta, p.rc,
                         G, dGdx, dGdy, dGdz);
            break;
        case SfKind::ExpAngw:
            symFncExpAngw(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                         neighDz, p.e1, p.e2, p.eta, p.rs, p.lambda, p.zeta,
                         p.rc, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::CompAngn:
            symFncCompAngn(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                         neighDz, p.e1, p.e2, p.rl, p.rc, p.angleLeftRad,
                         p.angleRightRad, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::CompAngnWeighted:
            symFncCompAngnWeighted(numNeighbors, neighElem, neighDist, neighDx,
                         neighDy, neighDz, p.rl, p.rc, p.angleLeftRad,
                         p.angleRightRad, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::CompAngw:
            symFncCompAngw(numNeighbors, neighElem, neighDist, neighDx, neighDy,
                         neighDz, p.e1, p.e2, p.rl, p.rc, p.angleLeftRad,
                         p.angleRightRad, G, dGdx, dGdy, dGdz);
            break;
        case SfKind::CompAngwWeighted:
            symFncCompAngwWeighted(numNeighbors, neighElem, neighDist, neighDx,
                         neighDy, neighDz, p.rl, p.rc, p.angleLeftRad,
                         p.angleRightRad, G, dGdx, dGdy, dGdz);
            break;
    }
}

__global__ void sfKernel(
    SfParams p, int numSelected, const int* selectedAtoms,
    const int* neighCount, const int* neighOffset,
    const int* neighElem, const double* neighDist,
    const double* neighDx, const double* neighDy, const double* neighDz,
    double* Gout, double* dGdxOut, double* dGdyOut, double* dGdzOut)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numSelected) return;
    int i = selectedAtoms[t];
    int off = neighOffset[i], n = neighCount[i];
    evalSf(p, n, &neighElem[off], &neighDist[off], &neighDx[off],
           &neighDy[off], &neighDz[off],
           Gout[t], dGdxOut[t], dGdyOut[t], dGdzOut[t]);
}

bool runCase(const char* label, const SfParams& p, const RealSystem& sys)
{
    printf("--- %s ---\n", label);

    std::vector<int> selected;
    for (int i = 0; i < sys.numAtoms; ++i)
        if (sys.atomElement[i] == p.centralElement) selected.push_back(i);
    int numSelected = (int)selected.size();

    std::vector<double> Gcpu(numSelected), dxCpu(numSelected),
                         dyCpu(numSelected), dzCpu(numSelected);
    for (int t = 0; t < numSelected; ++t)
    {
        int i = selected[t];
        int off = sys.neighOffset[i], n = sys.neighCount[i];
        evalSf(p, n, &sys.neighElement[off], &sys.neighDist[off],
               &sys.neighDx[off], &sys.neighDy[off], &sys.neighDz[off],
               Gcpu[t], dxCpu[t], dyCpu[t], dzCpu[t]);
    }

    int *d_selected, *d_neighCount, *d_neighOffset, *d_neighElem;
    double *d_neighDist, *d_neighDx, *d_neighDy, *d_neighDz;
    double *d_G, *d_dx, *d_dy, *d_dz;
    int total = (int)sys.neighDist.size();

    CUDA_CHECK(cudaMalloc(&d_selected, numSelected * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighCount, sys.numAtoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighOffset, sys.numAtoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighElem, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighDist, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDx, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDy, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighDz, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, numSelected * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dx, numSelected * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dy, numSelected * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dz, numSelected * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_selected, selected.data(), numSelected * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighCount, sys.neighCount.data(), sys.numAtoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighOffset, sys.neighOffset.data(), sys.numAtoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighElem, sys.neighElement.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDist, sys.neighDist.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDx, sys.neighDx.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDy, sys.neighDy.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighDz, sys.neighDz.data(), total * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int gridSize = (numSelected + blockSize - 1) / blockSize;
    sfKernel<<<gridSize, blockSize>>>(p, numSelected, d_selected,
        d_neighCount, d_neighOffset, d_neighElem, d_neighDist,
        d_neighDx, d_neighDy, d_neighDz, d_G, d_dx, d_dy, d_dz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> Ggpu(numSelected), dxGpu(numSelected),
                         dyGpu(numSelected), dzGpu(numSelected);
    CUDA_CHECK(cudaMemcpy(Ggpu.data(), d_G, numSelected * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dxGpu.data(), d_dx, numSelected * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dyGpu.data(), d_dy, numSelected * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dzGpu.data(), d_dz, numSelected * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_selected); cudaFree(d_neighCount); cudaFree(d_neighOffset);
    cudaFree(d_neighElem); cudaFree(d_neighDist); cudaFree(d_neighDx);
    cudaFree(d_neighDy); cudaFree(d_neighDz);
    cudaFree(d_G); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_dz);

    double maxAbsErrG = 0.0, maxRelErrG = 0.0, maxAbsErrD = 0.0;
    for (int t = 0; t < numSelected; ++t)
    {
        double a = fabs(Ggpu[t] - Gcpu[t]);
        maxAbsErrG = std::max(maxAbsErrG, a);
        maxRelErrG = std::max(maxRelErrG, a / std::max(1e-300, fabs(Gcpu[t])));
        maxAbsErrD = std::max({maxAbsErrD, fabs(dxGpu[t] - dxCpu[t]),
                                fabs(dyGpu[t] - dyCpu[t]), fabs(dzGpu[t] - dzCpu[t])});
    }

    printf("  selected atoms=%d  G[0]: cpu=%.15E gpu=%.15E\n",
           numSelected, Gcpu[0], Ggpu[0]);
    printf("  max|G_gpu-G_cpu|=%.3E  max relerr=%.3E  max|dG_gpu-dG_cpu|=%.3E\n",
           maxAbsErrG, maxRelErrG, maxAbsErrD);

    bool pass = (maxAbsErrG < 1e-9) && (maxAbsErrD < 1e-9);
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
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

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <real_neighbors_full.txt>\n", argv[0]);
        return 1;
    }
    RealSystem sys = loadRealSystem(argv[1]);
    printf("Loaded real system: %d atoms\n\n", sys.numAtoms);

    const int H = 0, O = 1;
    bool ok = true;

    // --- Radial family, real production parameters where they exist ---
    {
        SfParams p{SfKind::ExpRad, H}; p.e1 = H; p.eta = 0.001; p.rs = 0.0; p.rc = 12.0;
        ok &= runCase("ExpRad, real params (H 2 H 0.001 0.0 12.00)", p, sys);
    }
    {
        SfParams p{SfKind::ExpRad, H}; p.e1 = H; p.eta = 0.15; p.rs = 1.9124; p.rc = 12.0;
        ok &= runCase("ExpRad, real params (H 2 H 0.15 1.9124 12.00)", p, sys);
    }
    {
        SfParams p{SfKind::ExpRadWeighted, H}; p.eta = 0.01; p.rs = 0.0; p.rc = 12.0;
        ok &= runCase("ExpRadWeighted, representative params", p, sys);
    }
    {
        SfParams p{SfKind::CompRad, H}; p.e1 = H; p.rl = 0.5; p.rc = 12.0;
        ok &= runCase("CompRad, representative params", p, sys);
    }
    {
        SfParams p{SfKind::CompRadWeighted, O}; p.rl = 0.5; p.rc = 12.0;
        ok &= runCase("CompRadWeighted, representative params", p, sys);
    }

    // --- Exp-angular family, real production parameters where they exist ---
    {
        SfParams p{SfKind::ExpAngn, O}; p.e1 = H; p.e2 = H;
        p.eta = 0.001; p.lambda = 1.0; p.zeta = 1.0; p.rc = 12.0;
        ok &= runCase("ExpAngn, real params (O 3 H H 0.001 1.0 1.0 12.00)", p, sys);
    }
    {
        SfParams p{SfKind::ExpAngn, H}; p.e1 = O; p.e2 = H;
        p.eta = 0.07; p.lambda = 1.0; p.zeta = 4.0; p.rc = 12.0;
        ok &= runCase("ExpAngn, real params (H 3 O H 0.07 1.0 4.0 12.00)", p, sys);
    }
    {
        SfParams p{SfKind::ExpAngnWeighted, O};
        p.eta = 0.01; p.lambda = 1.0; p.zeta = 2.0; p.rc = 12.0;
        ok &= runCase("ExpAngnWeighted, representative params", p, sys);
    }
    {
        SfParams p{SfKind::ExpAngw, O}; p.e1 = H; p.e2 = H;
        p.eta = 0.01; p.lambda = -1.0; p.zeta = 2.0; p.rc = 12.0;
        ok &= runCase("ExpAngw, representative params", p, sys);
    }

    // --- Compact-angular family, representative params (not in real
    // input.nn; angleLeft/Right span the full [0,180] degree range) ---
    {
        SfParams p{SfKind::CompAngn, O}; p.e1 = H; p.e2 = H;
        p.rl = 0.5; p.rc = 12.0; p.angleLeftRad = 0.0; p.angleRightRad = M_PI;
        ok &= runCase("CompAngn, representative params", p, sys);
    }
    {
        SfParams p{SfKind::CompAngnWeighted, O};
        p.rl = 0.5; p.rc = 12.0; p.angleLeftRad = 0.0; p.angleRightRad = M_PI;
        ok &= runCase("CompAngnWeighted, representative params", p, sys);
    }
    {
        SfParams p{SfKind::CompAngw, O}; p.e1 = H; p.e2 = H;
        p.rl = 0.5; p.rc = 12.0; p.angleLeftRad = 0.0; p.angleRightRad = M_PI;
        ok &= runCase("CompAngw, representative params", p, sys);
    }
    {
        SfParams p{SfKind::CompAngwWeighted, O};
        p.rl = 0.5; p.rc = 12.0; p.angleLeftRad = 0.0; p.angleRightRad = M_PI;
        ok &= runCase("CompAngwWeighted, representative params", p, sys);
    }

    printf("\n%s\n", ok ? "ALL CASES PASSED" : "SOME CASES FAILED");
    return ok ? 0 : 1;
}
