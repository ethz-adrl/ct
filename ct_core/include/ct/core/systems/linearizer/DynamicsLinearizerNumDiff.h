/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
**********************************************************************************************************************/
#pragma once

namespace ct {
namespace core {

//! Computes the linearization of a system dynamics function through numerical finite differencing
/*!
 * This class takes a function handle representing system dynamics of the form
 * \f$ f(x(t),t,u(t),\dot{x(t)}) \f$ or \f$ f(x[n],n,u[n],x[n+1]) \f$ where the
 * last argument is the result of the evaluation in each case. It then computes
 * the linearization around a given point \f$ x = x_s \f$, \f$ u = u_s \f$.
 *
 * Finite differencing is used to calculate partial derivatives
 * \f[
 * \begin{aligned}
 * A &= \frac{df}{dx} |_{x=x_s, u=u_s} \\
 * B &= \frac{df}{du} |_{x=x_s, u=u_s}
 * \end{aligned}
 * \f]
 */
template <typename MANIFOLD, size_t CONTROL_DIM, typename SCALAR = typename MANIFOLD::Scalar, typename TIME = SCALAR>
class DynamicsLinearizerNumDiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static const size_t STATE_DIM = MANIFOLD::TangentDim;
    using Tangent = typename MANIFOLD::Tangent;
    using control_vector_t = ControlVector<CONTROL_DIM, SCALAR>;                        //!< control vector type
    using state_matrix_t = StateMatrix<STATE_DIM, SCALAR>;                              //!< state Jacobian type (A)
    using state_control_matrix_t = StateControlMatrix<STATE_DIM, CONTROL_DIM, SCALAR>;  //!< control Jacobian type (B)

    //!< dynamics function signature
    using dynamics_fct_t = std::function<void(const MANIFOLD&, const TIME&, const control_vector_t&, Tangent&)>;

    //! default constructor
    /*!
     * Initializes the linearizer with a dynamics function object
     *
     * @param nonlinearSystem non-linear system to linearize
     * @param doubleSidedDerivative if true, double sided numerical differentiation is used
     */
    DynamicsLinearizerNumDiff(dynamics_fct_t dyn, bool doubleSidedDerivative = true)
        : dynamics_fct_(dyn), doubleSidedDerivative_(doubleSidedDerivative)
    {
        dFdx_.setZero();
        dFdu_.setZero();
        eps_ = sqrt(Eigen::NumTraits<SCALAR>::epsilon());
    }

    //! copy constructor
    DynamicsLinearizerNumDiff(const DynamicsLinearizerNumDiff& rhs)
        : dynamics_fct_(rhs.dynamics_fct_),
          doubleSidedDerivative_(rhs.doubleSidedDerivative_),
          dFdx_(rhs.dFdx_),
          dFdu_(rhs.dFdu_),
          eps_(rhs.eps_)
    {
    }

    /*!
     * @brief get the Jacobian with respect to the state
     * This computes the linearization of the dynamics with respect to the state
     * at a given point \f$ x=x_s \f$, \f$ u=u_s \f$, i.e. it computes
     *
     * \f[
     * B = \frac{df}{dx} |_{x=x_s, u=u_s}
     * \f]
     *
     * @param x state to linearize at
     * @param u control to linearize at
     * @param t time
     * @return Jacobian wrt state
     */
    //! implementation for the euclidean manifold case
    template <typename M = MANIFOLD, typename std::enable_if<ct::core::is_euclidean<M>::value, bool>::type = true>
    const state_matrix_t& getDerivativeState(const MANIFOLD& x, const control_vector_t& u, const TIME t = TIME(0))
    {
        if (!doubleSidedDerivative_)
            dynamics_fct_(x, t, u, res_ref_);

        for (size_t i = 0; i < STATE_DIM; i++)
        {
            // inspired from http://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic
            SCALAR h = eps_ * std::max(std::abs<SCALAR>(x(i)), SCALAR(1.0));
            SCALAR x_ph = x(i) + h;
            SCALAR dxp = x_ph - x(i);

            MANIFOLD x_perturbed = x;
            x_perturbed(i) = x_ph;

            // evaluate dynamics at perturbed state
            Tangent res_plus;
            dynamics_fct_(x_perturbed, t, u, res_plus);

            if (doubleSidedDerivative_)
            {
                SCALAR x_mh = x(i) - h;
                SCALAR dxm = x(i) - x_mh;

                x_perturbed = x;
                x_perturbed(i) = x_mh;

                Tangent res_minus;
                dynamics_fct_(x_perturbed, t, u, res_minus);

                dFdx_.col(i) = (res_plus - res_minus) / (dxp + dxm);
            }
            else
            {
                dFdx_.col(i) = (res_plus - res_ref_) / dxp;
            }
        }

        return dFdx_;
    }

    // implementation for the non-euclidean case
    template <typename M = MANIFOLD, typename std::enable_if<!(ct::core::is_euclidean<M>::value), bool>::type = true>
    const state_matrix_t& getDerivativeState(const MANIFOLD& m, const control_vector_t& u, const TIME t = TIME(0))
    {
        if (!doubleSidedDerivative_)
            dynamics_fct_(m, t, u, res_ref_);

        Tangent h_vec;
        Tangent t_perturbed;
        Tangent res_plus;

        for (size_t i = 0; i < STATE_DIM; i++)
        {
            // inspired from http://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic
            SCALAR h = eps_ * std::max(std::abs<SCALAR>(m.log().coeffs()(i)), SCALAR(1.0));

            h_vec.setZero();
            h_vec.coeffs()(i) = h;

            t_perturbed = m.log() + h_vec;
            SCALAR dxp = (t_perturbed - m.log()).coeffs()(i);

            // evaluate dynamics at perturbed state
            dynamics_fct_(t_perturbed.exp(), t, u, res_plus);

            if (doubleSidedDerivative_)
            {
                t_perturbed = m.log() - h_vec;
                SCALAR dxm = (m.log() - t_perturbed).coeffs()(i);

                Tangent res_minus;
                dynamics_fct_(t_perturbed.exp(), t, u, res_minus);

                dFdx_.col(i) = (res_plus - res_minus).coeffs() / (dxp + dxm);
            }
            else
            {
                dFdx_.col(i) = ((res_plus - res_ref_).coeffs()) / dxp;
            }
        }

        return dFdx_;
    }

    //! get the Jacobian with respect to the input
    /*!
     * This computes the linearization of the dynamics with respect to the input
     * at a given point \f$ x=x_s \f$, \f$ u=u_s \f$, i.e. it computes
     *
     * \f[
     * B = \frac{df}{du} |_{x=x_s, u=u_s}
     * \f]
     *
     * @param x state to linearize at
     * @param u control to linearize at
     * @param t time
     * @return Jacobian wrt input
     */
    // implementation for euclidean case
    template <typename M = MANIFOLD, typename std::enable_if<ct::core::is_euclidean<M>::value, bool>::type = true>
    const state_control_matrix_t& getDerivativeControl(const MANIFOLD& x,
        const control_vector_t& u,
        const TIME t = TIME(0))
    {
        if (!doubleSidedDerivative_)
            dynamics_fct_(x, t, u, res_ref_);

        for (size_t i = 0; i < CONTROL_DIM; i++)
        {
            // inspired from http://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic
            SCALAR h = eps_ * std::max(std::abs<SCALAR>(u(i)), SCALAR(1.0));
            SCALAR u_ph = u(i) + h;
            SCALAR dup = u_ph - u(i);

            control_vector_t u_perturbed = u;
            u_perturbed(i) = u_ph;

            // evaluate dynamics at perturbed state
            Tangent res_plus;
            dynamics_fct_(x, t, u_perturbed, res_plus);

            if (doubleSidedDerivative_)
            {
                SCALAR u_mh = u(i) - h;
                SCALAR dum = u(i) - u_mh;

                u_perturbed = u;
                u_perturbed(i) = u_mh;

                Tangent res_minus;
                dynamics_fct_(x, t, u_perturbed, res_minus);

                dFdu_.col(i) = (res_plus - res_minus) / (dup + dum);
            }
            else
            {
                dFdu_.col(i) = (res_plus - res_ref_) / dup;
            }
        }

        return dFdu_;
    }

    // implementation for non-euclidean case
    template <typename M = MANIFOLD, typename std::enable_if<!(ct::core::is_euclidean<M>::value), bool>::type = true>
    const state_control_matrix_t& getDerivativeControl(const MANIFOLD& x,
        const control_vector_t& u,
        const TIME t = TIME(0))
    {
        if (!doubleSidedDerivative_)
            dynamics_fct_(x, t, u, res_ref_);

        for (size_t i = 0; i < CONTROL_DIM; i++)
        {
            // inspired from http://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic
            SCALAR h = eps_ * std::max(std::abs<SCALAR>(u(i)), SCALAR(1.0));
            SCALAR u_ph = u(i) + h;
            SCALAR dup = u_ph - u(i);

            control_vector_t u_perturbed = u;
            u_perturbed(i) = u_ph;

            // evaluate dynamics at perturbed state
            Tangent res_plus;
            dynamics_fct_(x, t, u_perturbed, res_plus);

            if (doubleSidedDerivative_)
            {
                SCALAR u_mh = u(i) - h;
                SCALAR dum = u(i) - u_mh;

                u_perturbed = u;
                u_perturbed(i) = u_mh;

                Tangent res_minus;
                dynamics_fct_(x, t, u_perturbed, res_minus);

                dFdu_.col(i) = (res_plus - res_minus).coeffs() / (dup + dum);
            }
            else
            {
                dFdu_.col(i) = (res_plus - res_ref_).coeffs() / dup;
            }
        }

        return dFdu_;
    }

    /**
     * @brief Get the Double Sided Derivative Flag
     */
    bool getDoubleSidedDerivativeFlag() const { return doubleSidedDerivative_; }
protected:
    dynamics_fct_t dynamics_fct_;  //!< function handle to system dynamics

    bool doubleSidedDerivative_;  //!< flag if double sided numerical differentiation should be used

    SCALAR eps_;  //!< perturbation for numerical differentiation


    // internally used variables
    state_matrix_t dFdx_;          //!< Jacobian wrt state
    state_control_matrix_t dFdu_;  //!< Jacobian wrt input

    Tangent res_ref_;  //!< reference result for numerical differentiation
};

}  // namespace core
}  // namespace ct