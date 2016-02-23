/*
Kalman filter for encoder data. Each filter filters one degree of freedom at
a time.
*/
#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"

namespace gps_control
{

class EncoderFilter
{
private:
    Eigen::MatrixXd time_matrix_;
    Eigen::VectorXd observation_vector_;
    Eigen::MatrixXd filtered_state_;

    int num_joints_;
    bool is_configured_;
public:
    // Constructor.
    EncoderFilter(ros::NodeHandle& n, const Eigen::VectorXd &initial_state);
    // Destructor.
    virtual ~EncoderFilter();
    // Update the Kalman filter.
    virtual void update(double sec_elapsed, Eigen::VectorXd &state);
    // Configure the Kalman filter.
    virtual void configure(const std::string &params);

    // Return filtered state.
    virtual void get_state(Eigen::VectorXd &state) const;
    // Return filtered velocity.
    virtual void get_velocity(Eigen::VectorXd &state) const;
};

}
