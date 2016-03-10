/*
Controller that executes a trial using a neural network policy using tf.
*/
#pragma once

// Headers.
#include <vector>
#include <Eigen/Dense>

// Superclass.
#include "gps_agent_pkg/trialcontroller.h"

namespace gps_control
{

    class TensorflowController : public TrialController
    {
    private:
        int last_command_id_received = 0;
        int last_command_id_acted_upon = 0;
        Eigen::VectorXd last_action_command_received;
    public:
        // Constructor.
        TensorflowController();
        // Destructor.
        virtual ~TensorflowController();
        // Compute the action at the current time step.
        virtual void get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U);
        // Configure the controller.
        virtual void configure_controller(OptionsMap &options);
        // receive new actions from subscriber.
        virtual void update_action_command(const gps_agent_pkg::TfActionCommand::ConstPtr& msg);
    };

}
