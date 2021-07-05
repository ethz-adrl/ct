/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
**********************************************************************************************************************/

#pragma once

namespace ct {
namespace optcon {

template <typename _MatrixType, int _UpLo>
Cholesky<_MatrixType, _UpLo>::Cholesky() : Eigen::LLT<_MatrixType, _UpLo>()
{
}

template <typename _MatrixType, int _UpLo>
Cholesky<_MatrixType, _UpLo>::Cholesky(const _MatrixType& m) : Eigen::LLT<_MatrixType, _UpLo>(m)
{
}

template <typename _MatrixType, int _UpLo>
Cholesky<_MatrixType, _UpLo>& Cholesky<_MatrixType, _UpLo>::setIdentity()
{
    this->m_matrix.setIdentity();
    this->m_isInitialized = true;
    return *this;
}

template <typename _MatrixType, int _UpLo>
bool Cholesky<_MatrixType, _UpLo>::isIdentity() const
{
    eigen_assert(this->m_isInitialized && "LLT is not initialized.");
    return this->m_matrix.isIdentity();
}

template <typename _MatrixType, int _UpLo>
template <typename Derived>
Cholesky<_MatrixType, _UpLo>& Cholesky<_MatrixType, _UpLo>::setL(const Eigen::MatrixBase<Derived>& matrix)
{
    this->m_matrix = matrix.template triangularView<Eigen::Lower>();
    this->m_isInitialized = true;
    return *this;
}

template <typename _MatrixType, int _UpLo>
template <typename Derived>
Cholesky<_MatrixType, _UpLo>& Cholesky<_MatrixType, _UpLo>::setU(const Eigen::MatrixBase<Derived>& matrix)
{
    this->m_matrix = matrix.template triangularView<Eigen::Upper>().adjoint();
    this->m_isInitialized = true;
    return *this;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::UnscentedKalmanFilter(
    std::shared_ptr<SystemModelBase<STATE_DIM, CONTROL_DIM, SCALAR>> f,
    std::shared_ptr<LinearMeasurementModel<OUTPUT_DIM, STATE_DIM, SCALAR>> h,
    const state_vector_t& x0,
    SCALAR alpha,
    SCALAR beta,
    SCALAR kappa,
    const ct::core::StateMatrix<STATE_DIM, SCALAR>& P0)
    : Base(f, h, x0), alpha_(alpha), beta_(beta), kappa_(kappa), P_(P0)
{
    computeWeights();
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::UnscentedKalmanFilter(
    std::shared_ptr<SystemModelBase<STATE_DIM, CONTROL_DIM, SCALAR>> f,
    std::shared_ptr<LinearMeasurementModel<OUTPUT_DIM, STATE_DIM, SCALAR>> h,
    const UnscentedKalmanFilterSettings<STATE_DIM, SCALAR>& ukf_settings)
    : Base(f, h, ukf_settings.x0),
      alpha_(ukf_settings.alpha),
      beta_(ukf_settings.beta),
      kappa_(ukf_settings.kappa),
      P_(ukf_settings.P0)
{
    computeWeights();
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
auto UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::predict(const control_vector_t& u,
    const ct::core::Time& dt,
    const ct::core::Time& t) -> const state_vector_t&
{
    if (!computeSigmaPoints())
        throw std::runtime_error("UnscentedKalmanFilter : Numerical error.");

    this->x_est_ = computeStatePrediction(u, dt, t);

    computeCovarianceFromSigmaPoints<STATE_DIM>(
        this->x_est_, sigmaStatePoints_, this->f_->computeDerivativeNoise(this->x_est_, u, dt, t), P_);

    return this->x_est_;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
auto UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::update(const output_vector_t& z,
    const ct::core::Time& dt,
    const ct::core::Time& t) -> const state_vector_t&
{
    SigmaPoints<OUTPUT_DIM> sigmaMeasurementPoints;

    // Predict measurement (and corresponding sigma points)
    ct::core::OutputVector<OUTPUT_DIM, SCALAR> y = this->computeMeasurementPrediction(sigmaMeasurementPoints, t);

    // Compute innovation covariance
    Covariance<OUTPUT_DIM> P;
    computeCovarianceFromSigmaPoints<OUTPUT_DIM>(
        y, sigmaMeasurementPoints, this->h_->computeDerivativeNoise(this->x_est_, t), P);

    Eigen::Matrix<SCALAR, STATE_DIM, OUTPUT_DIM> K;
    computeKalmanGain(y, sigmaMeasurementPoints, P, K);

    // Update state
    this->x_est_ += K * (z - y);

    // Update state covariance
    updateStateCovariance(K, P);

    return this->x_est_;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
bool UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeSigmaPoints()
{
    // Get square root of covariance
    CovarianceSquareRoot<STATE_DIM> llt;
    llt.compute(P_);
    if (llt.info() != Eigen::Success)
        return false;

    Eigen::Matrix<SCALAR, STATE_DIM, STATE_DIM> _S = llt.matrixL().toDenseMatrix();

    // Set left "block" (first column)
    this->sigmaStatePoints_.template leftCols<1>() = this->x_est_;
    // Set center block with x + gamma_ * S
    this->sigmaStatePoints_.template block<STATE_DIM, STATE_DIM>(0, 1) = (gamma_ * _S).colwise() + this->x_est_;
    // Set right block with x - gamma_ * S
    this->sigmaStatePoints_.template rightCols<STATE_DIM>() = (-gamma_ * _S).colwise() + this->x_est_;

    return true;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
template <size_t SIZE>
bool UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeCovarianceFromSigmaPoints(
    const Eigen::Matrix<SCALAR, SIZE, 1>& mean,
    const SigmaPoints<SIZE>& sigmaPoints,
    const Covariance<SIZE>& noiseCov,
    Covariance<SIZE>& cov)
{
    SigmaPoints<SIZE> W = this->sigmaWeights_c_.transpose().template replicate<SIZE, 1>();
    SigmaPoints<SIZE> tmp = (sigmaPoints.colwise() - mean);
    cov = tmp.cwiseProduct(W) * tmp.transpose() + noiseCov;

    return true;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
bool UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeKalmanGain(
    const ct::core::OutputVector<OUTPUT_DIM, SCALAR>& y,
    const SigmaPoints<OUTPUT_DIM>& sigmaMeasurementPoints,
    const Covariance<OUTPUT_DIM>& P_yy,
    Eigen::Matrix<SCALAR, STATE_DIM, OUTPUT_DIM>& K)
{
    SigmaPoints<STATE_DIM> W = this->sigmaWeights_c_.transpose().template replicate<STATE_DIM, 1>();
    Eigen::Matrix<SCALAR, STATE_DIM, OUTPUT_DIM> P_xy =
        (sigmaStatePoints_.colwise() - this->x_est_).cwiseProduct(W).eval() *
        (sigmaMeasurementPoints.colwise() - y).transpose();

    K = P_xy * P_yy.inverse();
    return true;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
bool UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::updateStateCovariance(
    const Eigen::Matrix<SCALAR, STATE_DIM, OUTPUT_DIM>& K,
    const Covariance<OUTPUT_DIM>& P)
{
    P_ -= K * P * K.transpose();
    return true;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
auto UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeStatePrediction(
    const ct::core::ControlVector<CONTROL_DIM, SCALAR>& u,
    const ct::core::Time& dt,
    const ct::core::Time& t) -> const state_vector_t
{
    // Pass each sigma point through non-linear state transition function
    computeSigmaPointTransition(u, dt, t);

    // Compute predicted state from predicted sigma points
    return computePredictionFromSigmaPoints<STATE_DIM>(sigmaStatePoints_);
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
ct::core::OutputVector<OUTPUT_DIM, SCALAR>
UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeMeasurementPrediction(
    SigmaPoints<OUTPUT_DIM>& sigmaMeasurementPoints,
    const ct::core::Time& t)
{
    // Predict measurements for each sigma point
    computeSigmaPointMeasurements(sigmaMeasurementPoints, t);

    // Predict measurement from sigma measurement points
    return computePredictionFromSigmaPoints<OUTPUT_DIM>(sigmaMeasurementPoints);
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
void UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeWeights()

{
    SCALAR L = SCALAR(STATE_DIM);
    lambda_ = alpha_ * alpha_ * (L + kappa_) - L;
    gamma_ = std::sqrt(L + lambda_);

    // Make sure L != -lambda_ to avoid division by zero
    assert(std::abs(L + lambda_) > 1e-6);

    // Make sure L != -kappa_ to avoid division by zero
    assert(std::abs(L + kappa_) > 1e-6);

    SCALAR W_m_0 = lambda_ / (L + lambda_);
    SCALAR W_c_0 = W_m_0 + (SCALAR(1) - alpha_ * alpha_ + beta_);
    SCALAR W_i = SCALAR(1) / (SCALAR(2) * alpha_ * alpha_ * (L + kappa_));

    // Make sure W_i > 0 to avoid square-root of negative number
    assert(W_i > SCALAR(0));

    sigmaWeights_m_[0] = W_m_0;
    sigmaWeights_c_[0] = W_c_0;

    for (int i = 1; i < SigmaPointCount; ++i)
    {
        sigmaWeights_m_[i] = W_i;
        sigmaWeights_c_[i] = W_i;
    }
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
void UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeSigmaPointTransition(
    const ct::core::ControlVector<CONTROL_DIM, SCALAR>& u,
    const ct::core::Time& dt,
    const ct::core::Time& t)
{
    for (size_t i = 0; i < SigmaPointCount; ++i)
    {
        sigmaStatePoints_.col(i) = this->f_->computeDynamics(sigmaStatePoints_.col(i), u, dt, t);
    }
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
void UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computeSigmaPointMeasurements(
    SigmaPoints<OUTPUT_DIM>& sigmaMeasurementPoints,
    const ct::core::Time& t)
{
    for (size_t i = 0; i < SigmaPointCount; ++i)
    {
        sigmaMeasurementPoints.col(i) = this->h_->computeMeasurement(sigmaStatePoints_.col(i), t);
    }
}

template <size_t STATE_DIM, size_t CONTROL_DIM, size_t OUTPUT_DIM, typename SCALAR>
template <size_t DIM>
Eigen::Matrix<SCALAR, DIM, 1>
UnscentedKalmanFilter<STATE_DIM, CONTROL_DIM, OUTPUT_DIM, SCALAR>::computePredictionFromSigmaPoints(
    const SigmaPoints<DIM>& sigmaPoints)
{
    // Use efficient matrix x vector computation
    return sigmaPoints * sigmaWeights_m_;
}

}  // namespace optcon
}  // namespace ct
