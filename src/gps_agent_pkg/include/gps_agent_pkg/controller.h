/*
Base class for a controller. Controllers take in sensor readings and choose the action.
*/
#pragma once

// Headers.
#include <boost/scoped_ptr.hpp>
#include <ros/ros.h>
#include <time.h>
#include <ros/time.h>
#include <Eigen/Dense>

// This allows us to use options.
#include "gps_agent_pkg/options.h"
#include "gps/proto/gps.pb.h"

namespace gps_control
{

// Forward declarations.
class Sample;
class RobotPlugin;

class Controller
{
private:

public:
    // Constructor.
    Controller(ros::NodeHandle& n, gps::ActuatorType arm, int size);
    Controller();
    // Destructor.
    virtual ~Controller();
    // Update the controller (take an action).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques) = 0;
    // Configure the controller.
    virtual void configure_controller(OptionsMap &options);
    // Set update delay on the controller.
    virtual void set_update_delay(double new_step_length);
    // Get update delay on the controller.
    virtual double get_update_delay();
    // Check if controller is finished with its current task.
    virtual bool is_finished() const = 0;
    // Reset the controller -- this is typically called when the controller is turned on.
    virtual void reset(ros::Time update_time);
};

}
