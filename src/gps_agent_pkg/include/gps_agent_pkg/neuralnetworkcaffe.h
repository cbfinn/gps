/*
Helper class for running caffe neural networks on the robot.
*/
#pragma once

// Headers
#include <Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include "caffe/caffe.hpp"
#include "google/protobuf/text_format.h"
#include "gps_agent_pkg/neuralnetwork.h"

using namespace caffe;

namespace gps_control
{

class NeuralNetworkCaffe : public NeuralNetwork {
protected:
    // The network
    shared_ptr<Net<float> > net_;

public:
    // Constructs caffe network using the specified model file
    NeuralNetworkCaffe(const char *model_file, Phase phase);
    // Constructs caffe network using the specified NetParameter
    NeuralNetworkCaffe(NetParameter& model_param);

    virtual ~NeuralNetworkCaffe();

    // Function that takes in an input state and outputs the neural network output action.
    virtual void forward(const Eigen::VectorXd &input, Eigen::VectorXd &output);
    // Function that takes in an input state and other features and outputs the neural network output action.
    virtual void forward(const Eigen::VectorXd &input, std::vector<float> &feat_input, Eigen::VectorXd &output);

    // Set the weights of the neural network.
    virtual void set_weights(void *weights_ptr);
    virtual void set_weights(NetParameter& net_param);
};

}
