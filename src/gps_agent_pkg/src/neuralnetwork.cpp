#include "gps_agent_pkg/neuralnetwork.h"

using namespace gps_control;

// Destruct all objects.
NeuralNetwork::~NeuralNetwork()
{
    // Nothing to do here.
}

void NeuralNetwork::forward(const Eigen::VectorXd &input, Eigen::VectorXd &output)
{
    // Nothing to do here.
}

// Set the scales and biases, also preallocate any internal temporaries for fast evaluation.
void NeuralNetwork::set_scalebias(const Eigen::MatrixXd& scale, const Eigen::VectorXd& bias)
{
    scale_ = scale;
    bias_ = bias;

    int dim_bias = bias.size();

    // Preallocate temporaries
    input_scaled_.resize(dim_bias);
    ROS_INFO("Scale and bias set successfully");
}

void NeuralNetwork::set_weights(void *weights_ptr)
{
    // Nothing to do here.
}
