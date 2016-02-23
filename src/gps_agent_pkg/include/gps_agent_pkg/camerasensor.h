/*
Camera sensor: records latest images from camera.
*/
#pragma once

#include <sensor_msgs/Image.h>

// Superclass.
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"

// This sensor writes to the following data types:
// RGBImage
// DepthImage

// Default values for image dimensions
#define IMAGE_WIDTH_INIT 320
#define IMAGE_HEIGHT_INIT 240
#define IMAGE_WIDTH 240
#define IMAGE_HEIGHT 240

namespace gps_control
{

class CameraSensor: public Sensor
{
private:
    // Latest image.
    std::vector<uint8_t> latest_rgb_image_;
    std::vector<uint16_t> latest_depth_image_;

    // Time at which the image was first published.
    ros::Time latest_rgb_time_, latest_depth_time_;

    // Image subscribers
    ros::Subscriber depth_subscriber_, rgb_subscriber_;

    // Image dimensions, before and after cropping, for both rgb and depth images
    int image_width_init_, image_height_init_, image_width_, image_height_, image_size_;

    std::string rgb_topic_name_, depth_topic_name_;

public:
    // Constructor.
    CameraSensor(ros::NodeHandle& n, RobotPlugin *plugin);
    // Destructor.
    virtual ~CameraSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step);
    void update_rgb_image(const sensor_msgs::Image::ConstPtr& msg);
    void update_depth_image(const sensor_msgs::Image::ConstPtr& msg);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to set resolution, cropping, topic to listen to...
    virtual void configure_sensor(const OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual void set_sample_data_format(boost::scoped_ptr<Sample> sample) const;
    // Set data on the provided sample.
    virtual void set_sample_data(boost::scoped_ptr<Sample> sample) const;
};

}
