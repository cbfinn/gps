/*
Controller that executes a trial using a neural network policy using Caffe.
*/
#pragma once

// Headers.
#include <vector>
#include <Eigen/Dense>

#include "gps_agent_pkg/neuralnetworkcaffe.h"

// Superclass.
#include "gps_agent_pkg/trialcontroller.h"

namespace gps_control
{

class CaffeNNController : public TrialController
{
private:
    // Pointer to Caffe network
    boost::scoped_ptr<NeuralNetworkCaffe> net_;
public:
    // Constructor.
    CaffeNNController();
    // Destructor.
    virtual ~CaffeNNController();
    // Compute the action at the current time step.
    virtual void get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U);
    // Configure the controller.
    virtual void configure_controller(OptionsMap &options);
};

}
