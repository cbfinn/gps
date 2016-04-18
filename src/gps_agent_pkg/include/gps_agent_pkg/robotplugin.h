/*
This is the base class for the robot plugin, which takes care of interfacing
with the robot.
*/
#pragma once

// Headers.
#include <vector>
#include <Eigen/Dense>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <realtime_tools/realtime_publisher.h>

#include "gps_agent_pkg/PositionCommand.h"
#include "gps_agent_pkg/TrialCommand.h"
#include "gps_agent_pkg/RelaxCommand.h"
#include "gps_agent_pkg/SampleResult.h"
#include "gps_agent_pkg/DataRequest.h"
#include "gps_agent_pkg/TfActionCommand.h"
#include "gps_agent_pkg/TfObsData.h"
#include "gps_agent_pkg/TfParams.h"
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/controller.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps/proto/gps.pb.h"

// Convenience defines.
#define ros_publisher_ptr(X) boost::scoped_ptr<realtime_tools::RealtimePublisher<X> >
#define MAX_TRIAL_LENGTH 2000

namespace gps_control
{

// Forward declarations.
// Controllers.
class PositionController;
class TrialController;
// Sensors.
class Sensor;
// Sample.
class Sample;
// Custom ROS messages.
class SampleResult;
class PositionCommand;
class TrialCommand;
class RelaxCommand;


class RobotPlugin
{
protected:
    ros::Time last_update_time_;
    // Temporary storage for active arm torques to be applied at each step.
    Eigen::VectorXd active_arm_torques_;
    // Temporary storage for passive arm torques to be applied at each step.
    Eigen::VectorXd  passive_arm_torques_;
    // Position controller for passive arm.
    boost::scoped_ptr<PositionController> passive_arm_controller_;
    // Position controller for active arm.
    boost::scoped_ptr<PositionController> active_arm_controller_;
    // Current trial controller (if any).
    boost::scoped_ptr<TrialController> trial_controller_;
    // Sensor data for the current time step.
    boost::scoped_ptr<Sample> current_time_step_sample_;
    // Auxiliary Sensor data for the current time step.
    boost::scoped_ptr<Sample> aux_current_time_step_sample_;
    // Sensors.
    std::vector<boost::shared_ptr<Sensor> > sensors_;
    // Auxiliary Sensors.
    std::vector<boost::shared_ptr<Sensor> > aux_sensors_;
    // KDL chains for the end-effectors.
    KDL::Chain passive_arm_fk_chain_, active_arm_fk_chain_;
    // KDL solvers for the end-effectors.
    boost::shared_ptr<KDL::ChainFkSolverPos> passive_arm_fk_solver_, active_arm_fk_solver_;
    // KDL solvers for end-effector Jacobians.
    boost::shared_ptr<KDL::ChainJntToJacSolver> passive_arm_jac_solver_, active_arm_jac_solver_;
    // Subscribers.
    // Subscriber for position control commands.
    ros::Subscriber position_subscriber_;
    // Subscriber trial commands.
    ros::Subscriber trial_subscriber_;
    ros::Subscriber test_sub_;
    // Subscriber for relax commands.
    ros::Subscriber relax_subscriber_;
    // Subscriber for current state report request.
    ros::Subscriber data_request_subscriber_;
    // Publishers.
    // Publish result of a trial, completion of position command, or just a report.
    ros_publisher_ptr(gps_agent_pkg::SampleResult) report_publisher_;
    // Is a trial arm data request pending?
    bool trial_data_request_waiting_;
    // Is a auxiliary data request pending?
    bool aux_data_request_waiting_;
    // Are the sensors initialized?
    bool sensors_initialized_;
    // Is everything initialized for the trial controller?
    bool controller_initialized_;
    //tf publisher
    ros_publisher_ptr(gps_agent_pkg::TfObsData) tf_publisher_;
    //tf action subscriber
    ros::Subscriber action_subscriber_tf_;
public:
    // Constructor (this should do nothing).
    RobotPlugin();
    // Destructor.
    virtual ~RobotPlugin();
    // Initialize everything.
    virtual void initialize(ros::NodeHandle& n);
    // Initialize all of the ROS subscribers and publishers.
    virtual void initialize_ros(ros::NodeHandle& n);
    // Initialize all of the position controllers.
    virtual void initialize_position_controllers(ros::NodeHandle& n);
    // Initialize all of the sensors (this also includes FK computation objects).
    virtual void initialize_sensors(ros::NodeHandle& n);
    // TODO: Comment
    virtual void initialize_sample(boost::scoped_ptr<Sample>& sample, gps::ActuatorType actuator_type);

    //Helper method to configure all sensors
    virtual void configure_sensors(OptionsMap &opts);

    // Report publishers
    // Publish a sample with data from up to T timesteps
    virtual void publish_sample_report(boost::scoped_ptr<Sample>& sample, int T=1);

    // Subscriber callbacks.
    // Position command callback.
    virtual void position_subscriber_callback(const gps_agent_pkg::PositionCommand::ConstPtr& msg);
    // Trial command callback.
    virtual void trial_subscriber_callback(const gps_agent_pkg::TrialCommand::ConstPtr& msg);
    virtual void test_callback(const std_msgs::Empty::ConstPtr& msg);
    // Relax command callback.
    virtual void relax_subscriber_callback(const gps_agent_pkg::RelaxCommand::ConstPtr& msg);
    // Data request callback.
    virtual void data_request_subscriber_callback(const gps_agent_pkg::DataRequest::ConstPtr& msg);
    //tf callback
    virtual void tf_robot_action_command_callback(const gps_agent_pkg::TfActionCommand::ConstPtr& msg);

    // Update functions.
    // Update the sensors at each time step.
    virtual void update_sensors(ros::Time current_time, bool is_controller_step);
    // Update the controllers at each time step.
    virtual void update_controllers(ros::Time current_time, bool is_controller_step);
    // Accessors.
    // Get current time.
    virtual ros::Time get_current_time() const = 0;
    // Get sensor
    virtual Sensor *get_sensor(SensorType sensor, gps::ActuatorType actuator_type);
    // Get current encoder readings (robot-dependent).
    virtual void get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const = 0;
    // Get forward kinematics solver.
    virtual void get_fk_solver(boost::shared_ptr<KDL::ChainFkSolverPos> &fk_solver, boost::shared_ptr<KDL::ChainJntToJacSolver> &jac_solver, gps::ActuatorType arm);

    //tf controller commands.
    //tf publish observation command.
    virtual void tf_publish_obs(Eigen::VectorXd obs);

};

}
