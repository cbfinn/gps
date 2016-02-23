/*
Controller that executes a trial using a time-varying linear-Gaussian
control law.
*/
#pragma once

// Headers.
#include <vector>
#include <Eigen/Dense>

// Superclass.
#include "gps_agent_pkg/trialcontroller.h"

namespace gps_control
{

class LinearGaussianController : public TrialController
{
private:
    // Linear feedbacks.
    std::vector<Eigen::MatrixXd> K_;

    // Bias.
    std::vector<Eigen::VectorXd> k_;
public:
    // Constructor.
    LinearGaussianController();
    // Destructor.
    virtual ~LinearGaussianController();
    // Compute the action at the current time step.
    virtual void get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U);
    // Configure the controller.
    virtual void configure_controller(OptionsMap &options);
};

}
