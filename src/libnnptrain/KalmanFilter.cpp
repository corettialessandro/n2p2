// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
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

#include "KalmanFilter.h"
#include "utility.h"
#ifdef N2P2_GPU
#include "GpuKalmanFilter.h"
#endif
#include <Eigen/LU>
#include <iostream>
#include <stdexcept>

using namespace Eigen;
using namespace std;
using namespace nnp;

KalmanFilter::KalmanFilter(size_t const sizeState,
                           KalmanType const type) :
    Updater(sizeState),
    sizeObservation(0   ),
    numUpdates     (0   ),
    epsilon        (0.0 ),
    q              (0.0 ),
    q0             (0.0 ),
    qtau           (0.0 ),
    qmin           (0.0 ),
    eta            (0.0 ),
    eta0           (0.0 ),
    etatau         (0.0 ),
    etamax         (0.0 ),
    lambda         (0.0 ),
    nu             (0.0 ),
    gamma          (0.0 ),
    w              (NULL),
    xi             (NULL),
    H              (NULL)
#ifdef N2P2_GPU
    , gpuState     (NULL)
#endif
{
    if (!(type == KT_STANDARD ||
          type == KT_FADINGMEMORY))
    {
        throw runtime_error("ERROR: Unknown Kalman filter type.\n");
    }

    if (sizeState < 1)
    {
        throw runtime_error("ERROR: Wrong Kalman filter dimensions.\n");
    }

    this->type            = type;
    sizeObservation = 1;

    w  = new Map<VectorXd      >(0, sizeState);
    xi = new Map<VectorXd const>(0, sizeObservation);
    H  = new Map<MatrixXd const>(0, sizeState, sizeObservation);
    P.resize(sizeState, sizeState);
    P.setIdentity();
    // Prevent problems with unallocated K when log starts.
    K.resize(sizeState, sizeObservation);
    K.setZero();
}

KalmanFilter::~KalmanFilter()
{
#ifdef N2P2_GPU
    if (gpuState) gpuKalmanDestroy(gpuState);
#endif
}

void KalmanFilter::setSizeObservation(size_t const size)
{
    sizeObservation = size;

    return;
}

void KalmanFilter::setState(double* state)
{
    new (w) Map<VectorXd>(state, sizeState);

    return;
}

void KalmanFilter::setError(double const* const error)
{
    setError(error, sizeObservation);

    return;
}

void KalmanFilter::setError(double const* const error, size_t const size)
{
    new (xi) Map<VectorXd const>(error, size);

    return;
}

void KalmanFilter::setJacobian(double const* const jacobian)
{
    setJacobian(jacobian, sizeObservation);

    return;
}

void KalmanFilter::setJacobian(double const* const jacobian,
                               size_t const columns)
{
    new (H) Map<MatrixXd const>(jacobian, sizeState, columns);

    return;
}

void KalmanFilter::update()
{
    update(sizeObservation);

    return;
}

void KalmanFilter::update(size_t const sizeObservation)
{
    sw[prefix].start(timingReset);

#ifdef N2P2_GPU
    // Phase 6 follow-up (GPU_PORTING_PLAN.md): KT_STANDARD only -- the
    // only mode this project's production input.nn actually uses (same
    // scoping gpu/kalman/kalman_test.cu already chose). Only the two
    // dominant O(N^2*m) steps (X=P.H, the K.X^T term of P-=K.X^T) run on
    // the GPU, via a lazily-created persistent GpuKalmanFilterState
    // (src/libnnpgpu/GpuKalmanFilter.h) that keeps P device-resident
    // across every call -- re-uploading/downloading all of P (N x N,
    // ~89 MB at production size) every single call would very plausibly
    // erase the GEMM speedup entirely, since this update() is called far
    // more often per epoch than the Jacobian call sites above. Everything
    // else (A=H^T.X+R, the m x m inverse, K=X.Ainv, w+=K.xi) stays exactly
    // as the original CPU code below, using #X/#K's normal Eigen storage
    // -- deliberately NOT reimplemented on the GPU: it's O(N*m^2), under
    // 1% of the flops, and an earlier version of this code that DID port
    // the inverse to a hand-rolled GPU Gauss-Jordan kernel diverged
    // catastrophically (~1e10) on real production data despite passing a
    // synthetic-random-data standalone test, because Gauss-Jordan isn't
    // numerically interchangeable with Eigen's LU-based .inverse() on a
    // real, sometimes ill-conditioned m x m matrix. status() below
    // downloads P from this state on demand, since #P goes stale once
    // it's active; #K never goes stale since it's computed on the host
    // below exactly as always.
    if (type == KT_STANDARD)
    {
        if (!gpuState)
        {
            gpuState = gpuKalmanCreate((int)sizeState, P.data());
        }

        X.resize(sizeState, sizeObservation);
        gpuKalmanComputeX(gpuState, (int)sizeObservation, H->data(), X.data());

        // Calculate scaling matrix.
        // A = H^T . X
        MatrixXd A = H->transpose() * X;

        // Increase learning rate.
        // eta(n) = eta(0) * exp(n * tau)
        if (eta < etamax) eta *= exp(etatau);

        // Add measurement noise.
        // A = A + R
        A.diagonal() += VectorXd::Constant(sizeObservation, 1.0 / eta);

        // Calculate Kalman gain matrix.
        // K = X . A^-1
        K.resize(sizeState, sizeObservation);
        K = X * A.inverse();

        // Update error covariance matrix.
        // P = P - K . X^T; P = P + Q
        gpuKalmanUpdateP(gpuState, (int)sizeObservation, K.data(), q);

        // Update state vector.
        // w = w + K . xi
        (*w) += K * (*xi);

        // Anneal process noise.
        // q(n) = q(0) * exp(-n * tau)
        if (q > qmin) q *= exp(-qtau);

        numUpdates++;

        if (timingReset) timingReset = false;
        sw[prefix].stop();

        return;
    }
#endif

    X.resize(sizeState, sizeObservation);

    // Calculate temporary result.
    // X = P . H
    X = P.selfadjointView<Lower>() * (*H);

    // Calculate scaling matrix.
    // A = H^T . X
    MatrixXd A = H->transpose() * X;

    // Increase learning rate.
    // eta(n) = eta(0) * exp(n * tau)
    if (type == KT_STANDARD && eta < etamax) eta *= exp(etatau);

    // Add measurement noise.
    // A = A + R
    if (type == KT_STANDARD)
    {
        A.diagonal() += VectorXd::Constant(sizeObservation, 1.0 / eta);
    }
    else if (type == KT_FADINGMEMORY)
    {
        A.diagonal() += VectorXd::Constant(sizeObservation, lambda);
    }

    // Calculate Kalman gain matrix.
    // K = X . A^-1
    K.resize(sizeState, sizeObservation);
    K = X * A.inverse();

    // Update error covariance matrix.
    // P = P - K . X^T
    P.noalias() -= K * X.transpose();

    // Apply forgetting factor.
    if (type == KT_FADINGMEMORY)
    {
        P *= 1.0 / lambda;
    }
    // Add process noise.
    // P = P + Q
    P.diagonal() += VectorXd::Constant(sizeState, q);

    //double numAcc = (P - P.transpose()).cwiseAbs().maxCoeff();
    //cout << "Max. deviation of symmetric form of P: " << numAcc << endl;

    // Update state vector.
    // w =  w + K . xi
    (*w) += K * (*xi);

    // Anneal process noise.
    // q(n) = q(0) * exp(-n * tau)
    if (q > qmin) q *= exp(-qtau);

    // Update forgetting factor.
    if (type == KT_FADINGMEMORY)
    {
        lambda = nu * lambda + 1.0 - nu;
        gamma = 1.0 / (1.0 + lambda / gamma);
    }

    numUpdates++;

    if (timingReset) timingReset = false;
    sw[prefix].stop();

    return;
}

void KalmanFilter::setParametersStandard(double const epsilon,
                                         double const q0,
                                         double const qtau,
                                         double const qmin,
                                         double const eta0,
                                         double const etatau,
                                         double const etamax)
{
    this->epsilon = epsilon;
    this->q0      = q0     ;
    this->qtau    = qtau   ;
    this->qmin    = qmin   ;
    this->eta0    = eta0   ;
    this->etatau  = etatau ;
    this->etamax  = etamax ;

    q = q0;
    eta = eta0;
    P /= epsilon;

    return;
}

void KalmanFilter::setParametersFadingMemory(double const epsilon,
                                             double const q0,
                                             double const qtau,
                                             double const qmin,
                                             double const lambda,
                                             double const nu)
{
    this->epsilon = epsilon;
    this->q0      = q0     ;
    this->qtau    = qtau   ;
    this->qmin    = qmin   ;
    this->lambda  = lambda ;
    this->nu      = nu     ;

    q = q0;
    P /= epsilon;
    gamma = 1.0;

    return;
}

string KalmanFilter::status(size_t epoch) const
{
#ifdef N2P2_GPU
    // #P is stale (never touched on the host) once GPU-accelerated updates
    // are active -- sync the current device-resident value in before using
    // it for diagnostics below. #K needs no such sync: it's always
    // computed on the host in update() regardless of GPU acceleration.
    if (gpuState)
    {
        gpuKalmanGetP(gpuState, P.data());
    }
#endif

    double Pasym = 0.5 * (P - P.transpose()).array().abs().mean();
    double Pdiag = P.diagonal().array().abs().sum();
    double Poffdiag = (P.array().abs().sum() - Pdiag)
                    / (sizeState * (sizeState - 1));
    Pdiag /= sizeState;
    double Kmean = K.array().abs().mean();

    string s = strpr("%10zu %10zu %16.8E %16.8E %16.8E %16.8E %16.8E",
                     epoch, numUpdates, Pdiag, Poffdiag, Pasym, Kmean, q);
    if (type == KT_STANDARD)
    {
        s += strpr(" %16.8E", eta);
    }
    else if (type == KT_FADINGMEMORY)
    {
        s += strpr(" %16.8E %16.8E", lambda, numUpdates * gamma);
    }
    s += '\n';

    return s;
}

vector<string> KalmanFilter::statusHeader() const
{
    vector<string> header;

    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Kalman filter status report.");
    colSize.push_back(10);
    colName.push_back("epoch");
    colInfo.push_back("Training epoch.");
    colSize.push_back(10);
    colName.push_back("nupdates");
    colInfo.push_back("Number of updates performed.");
    colSize.push_back(16);
    colName.push_back("Pdiag");
    colInfo.push_back("Mean of absolute diagonal values of error covariance "
                      "matrix P.");
    colSize.push_back(16);
    colName.push_back("Poffdiag");
    colInfo.push_back("Mean of absolute off-diagonal values of error "
                      "covariance matrix P.");
    colSize.push_back(16);
    colName.push_back("Pasym");
    colInfo.push_back("Asymmetry of P, i.e. mean of absolute values of "
                      "asymmetric part 0.5*(P - P^T).");
    colSize.push_back(16);
    colName.push_back("Kmean");
    colInfo.push_back("Mean of abolute compontents of Kalman gain matrix K.");
    colSize.push_back(16);
    colName.push_back("q");
    colInfo.push_back("Magnitude of process noise (= diagonal entries of Q).");
    if (type == KT_STANDARD)
    {
        colSize.push_back(16);
        colName.push_back("eta");
        colInfo.push_back("Learning rate.");
    }
    else if (type == KT_FADINGMEMORY)
    {
        colSize.push_back(16);
        colName.push_back("lambda");
        colInfo.push_back("Forgetting factor for fading memory KF.");
        colSize.push_back(16);
        colName.push_back("kgamma");
        colInfo.push_back("Forgetting gain k * gamma(k).");
    }
    header = createFileHeader(title, colSize, colName, colInfo);

    return header;
}

vector<string> KalmanFilter::info() const
{
    vector<string> v;

    if (type == KT_STANDARD)
    {
        v.push_back(strpr("KalmanType::KT_STANDARD (%d)\n", type));
        v.push_back(strpr("sizeState       = %zu\n", sizeState));
        v.push_back(strpr("sizeObservation = %zu\n", sizeObservation));
        v.push_back(strpr("epsilon         = %12.4E\n", epsilon));
        v.push_back(strpr("q0              = %12.4E\n", q0     ));
        v.push_back(strpr("qtau            = %12.4E\n", qtau   ));
        v.push_back(strpr("qmin            = %12.4E\n", qmin   ));
        v.push_back(strpr("eta0            = %12.4E\n", eta0   ));
        v.push_back(strpr("etatau          = %12.4E\n", etatau ));
        v.push_back(strpr("etamax          = %12.4E\n", etamax ));
    }
    else if (type == KT_FADINGMEMORY)
    {
        v.push_back(strpr("KalmanType::KT_FADINGMEMORY (%d)\n", type));
        v.push_back(strpr("sizeState       = %zu\n", sizeState));
        v.push_back(strpr("sizeObservation = %zu\n", sizeObservation));
        v.push_back(strpr("epsilon         = %12.4E\n", epsilon));
        v.push_back(strpr("q0              = %12.4E\n", q0     ));
        v.push_back(strpr("qtau            = %12.4E\n", qtau   ));
        v.push_back(strpr("qmin            = %12.4E\n", qmin   ));
        v.push_back(strpr("lambda          = %12.4E\n", lambda));
        v.push_back(strpr("nu              = %12.4E\n", nu    ));
    }
    v.push_back(strpr("OpenMP threads used: %d\n", nbThreads()));

    return v;
}
