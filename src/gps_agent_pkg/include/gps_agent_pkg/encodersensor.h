/*
Joint encoder sensor: returns joint angles and, optionally, their velocities.
*/
#pragma once

#include <kdl/jntarray.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include "gps/proto/gps.pb.h"

// Superclass.
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"
#include "gps_agent_pkg/encoderfilter.h"

// This sensor writes to the following data types:
// JointAngle
// JointVelocity
// EndEffectorPoint
// EndEffectorPointVelocity
// EndEffectorPosition
// EndEffectorRotation
// EndEffectorJacobian

namespace gps_control
{

class EncoderSensor: public Sensor
{
private:
    // Previous joint angles.
    Eigen::VectorXd previous_angles_;
    // Previous joint velocities.
    Eigen::VectorXd previous_velocities_;
    // Temporary storage for joint angles.
    Eigen::VectorXd temp_joint_angles_;
    // Temporary storage for KDL joint angle array.
    KDL::JntArray temp_joint_array_;
    // Temporary storage for KDL tip pose.
    KDL::Frame temp_tip_pose_;
    // Temporary storage for KDL Jacobian.
    KDL::Jacobian temp_jacobian_;
    // End-effector site/point offsets
    int n_points_;
    // EE Point jacobian
    Eigen::MatrixXd point_jacobians_;
    Eigen::MatrixXd point_jacobians_rot_;

    boost::shared_ptr<KDL::ChainFkSolverPos> fk_solver_;
    boost::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;

    boost::scoped_ptr<EncoderFilter> joint_filter_;

    // End-effector points in the space of the end-effector.
    Eigen::MatrixXd end_effector_points_;
    // Previous end-effector points.
    Eigen::MatrixXd previous_end_effector_points_;
    // End-effector points target.
    Eigen::MatrixXd end_effector_points_target_;
    // Velocities of points.
    Eigen::MatrixXd previous_end_effector_point_velocities_;
    // Temporary storage.
    Eigen::MatrixXd temp_end_effector_points_;
    // Previous end-effector position.
    Eigen::Vector3d previous_position_;
    // Previous end-effector rotation.
    Eigen::Matrix3d previous_rotation_;
    // Previous end-effector Jacobian.
    Eigen::MatrixXd previous_jacobian_;
    // Time from last update when the previous angles were recorded (necessary to compute velocities).
    ros::Time previous_angles_time_;

    // which arm is this EncoderSensor for?
    gps::ActuatorType actuator_type_;
public:
    // Constructor.
    EncoderSensor(ros::NodeHandle& n, RobotPlugin *plugin, gps::ActuatorType actuator_type);
    // Destructor.
    virtual ~EncoderSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    virtual void configure_sensor(OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual void set_sample_data_format(boost::scoped_ptr<Sample>& sample);
    // Set data on the provided sample.
    virtual void set_sample_data(boost::scoped_ptr<Sample>& sample, int t);
};

}
