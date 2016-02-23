#include "gps_agent_pkg/neuralnetworkcaffe.h"

using namespace gps_control;

// Construct network from prototxt file
NeuralNetworkCaffe::NeuralNetworkCaffe(const char *model_file, Phase phase)
{
    ROS_INFO("Constructing Caffe net from file %f", model_file);
    net_.reset(new Net<float>(model_file, phase));

    // If we're not in CPU_ONLY mode, use the GPU
#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
#endif
    ROS_INFO("Net constructed");
}

// Construct network from model specs.
NeuralNetworkCaffe::NeuralNetworkCaffe(NetParameter& model_param)
{
    ROS_INFO("Constructing Caffe net from net param");
    net_.reset(new Net<float>(model_param));
    // If we're not in CPU_ONLY mode, use the GPU.
#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
#endif
    ROS_INFO("Net constructed");
}

// Destructor -- free up memory here.
NeuralNetworkCaffe::~NeuralNetworkCaffe()
{
}

// This function computes the action from rgb features and joint states.
void NeuralNetworkCaffe::forward(const Eigen::VectorXd &input, std::vector<float> &feat_input, Eigen::VectorXd &output)
{
    // Transform the input by scale and bias.
    // Note that this assumes that all state information that we don't want to feed to the network is stored at the end of the state vector.
    assert(x.rows() >= input_scaled_.rows());
    input_scaled_ = scale_*input.segment(0, input_scaled_.rows()) + bias_;

    ROS_FATAL("Forward with >1 input not implemented!");
    // TODO implement, will be very similar to forward with one input
}

// Run forward pass of network with passed in input, and fill in output.
void NeuralNetworkCaffe::forward(const Eigen::VectorXd &input, Eigen::VectorXd &output)
{
    // Transform the input by scale and bias.
    // Note that this assumes that all state information that we don't want to feed to the network is stored at the end of the state vector.
    assert(x.rows() >= input_scaled_.rows());
    input_scaled_ = scale_ * input.segment(0, input_scaled_.rows()) + bias_;

    Blob<float>* input_blob = net_->bottom_vecs()[1][0];

    float* blob_data = input_blob->mutable_cpu_data();
    for (int i = 0; i < input_scaled_.rows(); ++i) {
      blob_data[i] = (float) input_scaled_(i);
    }

    // Call net forward.
    net_->ForwardFrom(1);

    const vector<Blob<float>*>& output_blobs = net_->output_blobs();

    // Copy output blob to u.
    for (int i = 0; i < output.rows(); ++i) {
        output[i] = (double) output_blobs[0]->cpu_data()[i];
    }

}

// Set the weights on the network from protobuffer string
void NeuralNetworkCaffe::set_weights(void *weights_ptr)
{
    ROS_INFO("Reading model weights");
    NetParameter net_param;
    std::string *weights = static_cast<std::string*>(weights_ptr);
    const std::string weights_string = *weights;  // Make a copy
    delete weights;

    google::protobuf::TextFormat::ParseFromString(weights_string, &net_param);
    this->set_weights(&net_param);
}

// Set the weights on the network from protobuffer string
void NeuralNetworkCaffe::set_weights(NetParameter& net_param)
{
    net_->CopyTrainedLayersFrom(net_param);
    ROS_INFO("NN weights set successfully");
}
