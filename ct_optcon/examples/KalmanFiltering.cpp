/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
**********************************************************************************************************************/

/*!
 * \example KalmanFiltering.cpp
 *
 * This example shows how to use the Kalman Filter to estimate the state of a simple oscillator.
 *
 */

#include <ct/optcon/optcon.h>
#include <ct/core/examples/CustomController.h>  // Using the custom controller from ct_core examples.
#include "exampleDir.h"

int main(int argc, char** argv)
{
    // file with kalman weights
    std::string settingsFile = ct::optcon::exampleDir + "/kalmanFilterWeights.info";

    // a damped oscillator has two states, position and velocity
    const size_t state_dim = ct::core::SecondOrderSystem::STATE_DIM;      // = 2
    const size_t control_dim = ct::core::SecondOrderSystem::CONTROL_DIM;  // = 1
    const size_t output_dim = 2;  // we assume we observe the full state (however with noise)

    // create an initial state: we initialize it at a point with unit deflection and zero velocity
    ct::core::StateVector<state_dim> x;
    x(0) = 1.0;
    x(1) = 0.0;

    // create an oscillator
    double w_n = 50;
    std::shared_ptr<ct::core::SecondOrderSystem> oscillator(new ct::core::SecondOrderSystem(w_n));

    // create a simple PD controller
    double kp = 10;
    double kd = 1;
    ct::core::ControlVector<control_dim> uff;
    uff << 2.0;
    std::shared_ptr<CustomController> controller(new CustomController(uff, kp, kd));

    // assign the controller
    oscillator->setController(controller);

    // create an integrator for "simulating" the measured data
    ct::core::Integrator<state_dim> integrator(oscillator, ct::core::IntegrationType::RK4);

    ct::core::StateVectorArray<state_dim> states;
    ct::core::ControlVectorArray<control_dim> controls;
    ct::core::tpl::TimeArray<double> times;

    ct::core::StateMatrix<state_dim> process_var;
    ct::core::loadMatrix(settingsFile, "process_noise.process_var", process_var);

    ct::core::GaussianNoise position_process_noise(0.0, process_var(0, 0));
    ct::core::GaussianNoise velocity_process_noise(0.0, process_var(1, 1));

    // simulate 100 steps
    double dt = 0.001;
    size_t nSteps = 100;
    states.push_back(x);
    for (size_t i = 0; i < nSteps; i++)
    {
        // compute control (needed for filter later)
        ct::core::ControlVector<control_dim> u_temp;
        controller->computeControl(x, i * dt, u_temp);
        controls.push_back(u_temp);

        integrator.integrate_n_steps(x, i * dt, 1, dt);

        position_process_noise.noisify(x(0));  // Position noise.
        velocity_process_noise.noisify(x(1));  // Velocity noise.

        states.push_back(x);
        times.push_back(i * dt);
    }

    // create system observation matrix C: we measure both position and velocity
    ct::core::OutputStateMatrix<output_dim, state_dim> C;
    C.setIdentity();

    // load Kalman Filter weighting matrices from file
    ct::core::StateMatrix<state_dim> Q, dFdv;
    ct::core::OutputMatrix<output_dim> R;
    dFdv.setIdentity();
    ct::core::loadMatrix(settingsFile, "kalman_weights.Q", Q);
    ct::core::loadMatrix(settingsFile, "kalman_weights.R", R);
    std::cout << "Loaded Kalman R as " << std::endl << R << std::endl;
    std::cout << "Loaded Kalman Q as " << std::endl << Q << std::endl;

    // create a sensitivity approximator to compute A and B matrices
    std::shared_ptr<ct::core::SystemLinearizer<state_dim, control_dim>> linearizer(
        new ct::core::SystemLinearizer<state_dim, control_dim>(oscillator));

    std::shared_ptr<ct::core::SensitivityApproximation<state_dim, control_dim>> sensApprox(
        new ct::core::SensitivityApproximation<state_dim, control_dim>(dt, linearizer));


    // the observer is supplied with a dynamic model identical to the one used above for data generation
    std::shared_ptr<ct::core::SecondOrderSystem> oscillator_observer_model(new ct::core::SecondOrderSystem(w_n));

    // set up the system model
    std::shared_ptr<ct::optcon::CTSystemModel<state_dim, control_dim>> sysModel(
        new ct::optcon::CTSystemModel<state_dim, control_dim>(oscillator_observer_model, sensApprox, dFdv));

    // set up the measurement model
    ct::core::OutputMatrix<output_dim> dHdw;
    dHdw.setIdentity();
    std::shared_ptr<ct::optcon::LinearMeasurementModel<output_dim, state_dim>> measModel(
        new ct::optcon::LTIMeasurementModel<output_dim, state_dim>(C, dHdw));

    // set up Filter, e.g. extended Kalman or Unscented Kalman filter
    ct::optcon::ExtendedKalmanFilter<state_dim, control_dim, output_dim> filter(
        sysModel, measModel, Q, R, states[0], Q);
    // ct::optcon::UnscentedKalmanFilter<state_dim, control_dim, output_dim> filter(sysModel, measModel, states[0]);


    ct::core::StateMatrix<state_dim> meas_var;
    ct::core::loadMatrix(settingsFile, "measurement_noise.measurement_var", meas_var);

    ct::core::GaussianNoise position_measurement_noise(0.0, meas_var(0, 0));
    ct::core::GaussianNoise velocity_measurement_noise(0.0, meas_var(1, 1));

    ct::core::StateVectorArray<state_dim> states_est(states.size());
    ct::core::StateVectorArray<state_dim> states_meas(states.size());
    states_est[0] = states[0];
    states_meas[0] = states[0];

    // run the filter over the simulated data
    for (size_t i = 1; i < states.size(); ++i)
    {
        // compute an observation
        states_meas[i] = states[i];

        // note that this is technically not correct, the noise enters not on the state but on the output!!!
        position_measurement_noise.noisify(states_meas[i](0));  // Position noise.
        velocity_measurement_noise.noisify(states_meas[i](1));  // Velocity noise.
        ct::core::OutputVector<output_dim> y = C * states_meas[i];

        // Kalman filter prediction step
        filter.predict(controls[i], dt, dt * i);

        // Kalman filter estimation step
        ct::core::StateVector<state_dim> x_est = filter.update(y, dt, dt * i);

        // and log for printing
        states_est[i] = x_est;
    }


// plot if plotting library built.
#ifdef PLOTTING_ENABLED

    std::vector<double> time_plot, pos_est_plot, vel_est_plot, pos_plot, vel_plot, pos_meas_plot, vel_meas_plot;
    for (size_t i = 0; i < times.size(); i++)
    {
        time_plot.push_back(times[i]);
        pos_est_plot.push_back(states_est[i](0));
        vel_est_plot.push_back(states_est[i](1));
        pos_plot.push_back(states[i](0));
        vel_plot.push_back(states[i](1));
        pos_meas_plot.push_back(states_meas[i](0));
        vel_meas_plot.push_back(states_meas[i](1));
    }

    // plot position
    ct::core::plot::subplot(2, 1, 1);
    ct::core::plot::labelPlot("pos est", time_plot, pos_est_plot);
    ct::core::plot::labelPlot("ground truth", time_plot, pos_plot, "r--");
    ct::core::plot::labelPlot("pos meas", time_plot, pos_meas_plot, "k--");
    ct::core::plot::legend();
    ct::core::plot::ylabel("pos [m]");

    // plot velocity
    ct::core::plot::subplot(2, 1, 2);
    ct::core::plot::labelPlot("vel est", time_plot, vel_est_plot);
    ct::core::plot::labelPlot("ground truth", time_plot, vel_plot, "r--");
    ct::core::plot::labelPlot("vel meas", time_plot, vel_meas_plot, "k--");
    ct::core::plot::ylabel("vel [m/sec]");
    ct::core::plot::xlabel("time [sec]");
    ct::core::plot::legend();

    ct::core::plot::show();
#else  // print results to command line
    for (size_t i = 0; i < states_est.size(); ++i)
        std::cout << "State\t\tState_est\n"
                  << std::fixed << std::setprecision(6) << states[i][0] << "\t" << states_est[i][0] << std::endl
                  << states[i][1] << "\t" << states_est[i][1] << std::endl
                  << std::endl;
#endif

    return 0;
}
