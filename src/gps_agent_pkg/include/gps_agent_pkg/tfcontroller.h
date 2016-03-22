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

    class TfController : public TrialController
    {

    public:
        // Constructor.
        TfController();
        // Destructor.
        virtual ~TfController();
        // Compute the action at the current time step.
        virtual void get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U);
        // Configure the controller.
        virtual void configure_controller(OptionsMap &options);
        // receive new actions from subscriber.
        virtual void update_action_command(int id, const Eigen::VectorXd &command);
        //publish the observations as we use them to act.
        virtual void publish_obs(Eigen::VectorXd obs, RobotPlugin *plugin);
        int last_command_id_received = 0;
        int last_command_id_acted_upon = 0;
        int failed_attempts = 0;
        Eigen::VectorXd last_action_command_received;
    };

}
