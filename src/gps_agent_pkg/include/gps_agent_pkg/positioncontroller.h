/*
Controller that moves the arm to a position, either in joint space or in task
space.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "gps_agent_pkg/controller.h"
#include "gps/proto/gps.pb.h"

namespace gps_control
{

class PositionController : public Controller
{
private:
    // P gains.
    Eigen::VectorXd pd_gains_p_;
    // D gains.
    Eigen::VectorXd pd_gains_d_;
    // I gains.
    Eigen::VectorXd pd_gains_i_;
    // Integral terms.
    Eigen::VectorXd pd_integral_;
    Eigen::VectorXd i_clamp_;
    // Maximum joint velocities.
    Eigen::VectorXd max_velocities_;
    // Temporary storage for Jacobian.
    Eigen::MatrixXd temp_jacobian_;
    // Temporary storage for joint angle offset.
    Eigen::VectorXd temp_angles_;
    // Current target (joint space).
    Eigen::VectorXd target_angles_;
    // Current target (task space).
    Eigen::VectorXd target_pose_;
    // Latest joint angles.
    Eigen::VectorXd current_angles_;
    // Latest joint angle velocities.
    Eigen::VectorXd current_angle_velocities_;
    // Latest pose.
    Eigen::VectorXd current_pose_;

    //Eigen::VectorXd torques_;

    // Current mode.
    gps::PositionControlMode mode_;
    // Current arm.
    gps::ActuatorType arm_;
    // Time since motion start.
    ros::Time start_time_;
    // Time of last update.
    ros::Time last_update_time_;
public:
    // Constructor.
    PositionController(ros::NodeHandle& n, gps::ActuatorType arm, int size);
    // Destructor.
    virtual ~PositionController();
    // Update the controller (take an action).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques);
    // Configure the controller.
    virtual void configure_controller(OptionsMap &options);
    // Check if controller is finished with its current task.
    virtual bool is_finished() const;
    // Reset the controller -- this is typically called when the controller is turned on.
    virtual void reset(ros::Time update_time);
    // Should this report when position achieved?
    bool report_waiting;
};

}

/*

note that this thing may be running at the same time as the trial controller in order to control the left arm

would be good to have the following functionality:

1. go to a position (with maximum speed limit and high gains)

2. go to a position task space (Via J^T control)

3. hold position with variable stiffness

4. signal when at position with a timeout and variable stiffness (integral term?)

*/
