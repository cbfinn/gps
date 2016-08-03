#include "gps_agent_pkg/rostopicsensor.h"
#include <math.h>

using namespace gps_control;
// Constructor.
ROSTopicSensor::ROSTopicSensor(ros::NodeHandle& n, RobotPlugin *plugin): Sensor(n, plugin)
{
    // Initialize subscribers
    if (!n.getParam("feat_topic",topic_name_))
	topic_name_ = "/caffe_features_publisher";
    // Initialize data vector.
    ROS_INFO("init rostopic sensor, topic is %s", topic_name_.c_str());
    data_size_ = 64;
    latest_data_.resize(data_size_);
    latest_data_eigen_.resize(data_size_);
    subscriber_ = n.subscribe(topic_name_, 1, &ROSTopicSensor::update_data_vector, this);
}
// Destructor.
ROSTopicSensor::~ROSTopicSensor()
{
    // Nothing to do here.
}
// Callback from ros topic sensor
void ROSTopicSensor::update_data_vector(const std_msgs::Float64MultiArray::ConstPtr& msg) {
    if (latest_data_.empty()) {
	ROS_INFO("latest data empty, dim size %d", msg->layout.dim[0].size);
	data_size_ = msg->layout.dim[0].size;
	latest_data_.resize(data_size_);
	latest_data_eigen_.resize(data_size_);
    } else { // better way to make this assertion? (error message?)
	assert(latest_data_.size() == data_size_);
	assert(msg->layout.dim[0].size == data_size_);
    }
    for (int i = 0; i < data_size_; i++)
	{
	    if (isnan(msg->data[i])) {
		    ROS_ERROR("data %d is nan %e", i, msg->data[i]);
		}
	    latest_data_[i] = msg->data[i];
	    latest_data_eigen_[i] = msg->data[i];
	}
}
// Update the sensor (called every tick).
void ROSTopicSensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
    // Not needed for this sensor, since callback function is used.
}
// The settings include the configuration for the Kalman filter.
void ROSTopicSensor::configure_sensor(OptionsMap &options)
{
    ROS_INFO("configuring rostopicsensor");
    //data_size_ = boost::get<int>(options["data_size"]); // Maybe just set this my size of first sample?
}
// Set data format and meta data on the provided sample.
void ROSTopicSensor::set_sample_data_format(boost::scoped_ptr<Sample>& sample) 
{
    // Set image size and format.
    OptionsMap data_metadata;
    ROS_INFO("Setting ROS_TOPIC_SENSOR meta data to %d", data_size_);
    sample->set_meta_data(gps::IMAGE_FEAT,data_size_,SampleDataFormatEigenVector,data_metadata);
}
// Set data on the provided sample.
void ROSTopicSensor::set_sample_data(boost::scoped_ptr<Sample>& sample, int t) 
{
    sample->set_data_vector(t,gps::IMAGE_FEAT,latest_data_.data(),latest_data_.size(),SampleDataFormatEigenVector);
    // sample->set_data_vector(t,gps::JOINT_ANGLES,previous_angles_.data(),previous_angles_.size(),SampleDataFormatEigenVector);
    // sample->set_data_vector(t,gps::IMAGE_FEAT,latest_data_eigen_.data(),latest_data_eigen_.size(),SampleDataFormatEigenVector);

}
