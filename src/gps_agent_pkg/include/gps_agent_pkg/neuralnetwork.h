/*
This is the base class for the neural network object. The neural network object
is a helper class for running neural networks on the robot.
*/
#pragma once

// Headers
#include <Eigen/Dense>
#include <vector>
#include <ros/ros.h>

namespace gps_control
{

class NeuralNetwork {
protected:
    // Internal scales and biases of network input
    Eigen::MatrixXd scale_;
    Eigen::VectorXd bias_;

public:
    // Pre-allocated scaled input data
    Eigen::VectorXd input_scaled_;

    virtual ~NeuralNetwork();

    // Function that takes in an input state and outputs the neural network output action.
    // Should be implemented in the subclass.
    virtual void forward(const Eigen::VectorXd &input, Eigen::VectorXd &output);

    // Set the scales and biases, also preallocate any internal temporaries for fast evaluation.
    // This is implemented in neuralnetwork.cpp.
    virtual void set_scalebias(const Eigen::MatrixXd& scale, const Eigen::VectorXd& bias);

    // Set the weights of the neural network.
    // Should be implemented in the subclass.
    virtual void set_weights(void *weights_ptr);
};

}
