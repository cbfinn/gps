#include "gps_agent_pkg/encoderfilter.h"
#include "gps_agent_pkg/util.h"
#include <stdlib.h>

using namespace gps_control;

EncoderFilter::EncoderFilter(ros::NodeHandle& n, const Eigen::VectorXd &initial_state)
{
    // Set initial state.
    num_joints_ = initial_state.size();

    is_configured_ = false;
    std::string params;
    if (!n.getParam("encoder_filter_params", params)) {
        ROS_ERROR("Failed to receive joint kalman filter params.");
        return;
    }

    configure(params);

    for (int i = 0; i < num_joints_; ++i) {
        filtered_state_(0,i) = initial_state(i);
    }
}

// Destructor.
EncoderFilter::~EncoderFilter()
{
    // Nothing to do here.
}

void EncoderFilter::configure(const std::string& params)
{
    ROS_INFO("Configuring encoder Kalman filter.");
    std::vector<std::string> matrices;
    util::split(params, '\n', matrices);

    // First line is time matrix
    std::vector<std::string> time_values;
    util::split(matrices[0], ' ', time_values);
    // Second line is observation vector
    std::vector<std::string> obs_values;
    util::split(matrices[1], ' ', obs_values);

    int filter_order = obs_values.size();
    time_matrix_.resize(filter_order, filter_order);
    observation_vector_.resize(filter_order);

    filtered_state_.resize(filter_order, num_joints_);
    filtered_state_.fill(0.0);

    for (int i = 0; i < filter_order; ++i) {
        for (int j = 0; j < filter_order; ++j) {
            time_matrix_(i,j) = (double) atof(time_values[i + j*filter_order].c_str());
        }
        observation_vector_(i) = (double) atof(obs_values[i].c_str());
    }
    is_configured_ = true;
    ROS_INFO("Joint kalman filter configured.");
}

void EncoderFilter::update(double sec_elapsed, Eigen::VectorXd &state)
{
    if (is_configured_) {
        filtered_state_ = time_matrix_ * filtered_state_ + observation_vector_ * state.transpose();
    } else {
        ROS_FATAL("Not implemented if not configured");
    }
}

void EncoderFilter::get_state(Eigen::VectorXd &state) const
{
    state = filtered_state_.row(0);
}

void EncoderFilter::get_velocity(Eigen::VectorXd &velocity) const
{
    velocity = filtered_state_.row(1);
}
