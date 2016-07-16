/*
  ROS topic sensor: records readings published to a particluar ros topic.
*/
#pragma once
#include <std_msgs/Float64MultiArray.h>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
// Superclass.
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"
// This sensor writes to the following data types:
// IMAGE_FEAT
namespace gps_control
{
    class ROSTopicSensor: public Sensor
    {
    private:
	// Latest data vector.
	std::vector<double> latest_data_;
	Eigen::VectorXd latest_data_eigen_;
	// Subscribers
	ros::Subscriber subscriber_;
	// Vector dimension
	int data_size_;
	std::string topic_name_;
    public:
	// Constructor.
	ROSTopicSensor(ros::NodeHandle& n, RobotPlugin *plugin);
	// Destructor.
	virtual ~ROSTopicSensor();
	// Update the sensor (called every tick).
	virtual void update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step);
	void update_data_vector(const std_msgs::Float64MultiArray::ConstPtr& msg);
	// Configure the sensor (for sensor-specific trial settings).
	virtual void configure_sensor(OptionsMap &options);
	// Set data format and meta data on the provided sample.
	virtual void set_sample_data_format(boost::scoped_ptr<Sample>& sample);
	// Set data on the provided sample.
	virtual void set_sample_data(boost::scoped_ptr<Sample>& sample, int t);
    };
}
