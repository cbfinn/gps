#include "gps_agent_pkg/camerasensor.h"

using namespace gps_control;

// Constructor.
CameraSensor::CameraSensor(ros::NodeHandle& n, RobotPlugin *plugin): Sensor(n, plugin)
{
    // Initialize subscribers
    if (!n.getParam("rgb_topic",rgb_topic_name_))
        rgb_topic_name_ = "/camera/rgb/image_color";
    if (!n.getParam("depth_topic",depth_topic_name_))
        depth_topic_name_ = "/camera/depth_registered/image_raw";

    if (!rgb_topic_name_.empty())
      rgb_subscriber_ = n.subscribe(rgb_topic_name_, 1, &CameraSensor::update_rgb_image, this);
    if (!depth_topic_name_.empty())
      depth_subscriber_ = n.subscribe(depth_topic_name_, 1, &CameraSensor::update_depth_image, this);

    // Initialize image config specs - image_width_init_, image_width_, etc.
    if (!n.getParam("image_width",image_width_))
      image_width_ = IMAGE_WIDTH;
    if (!n.getParam("image_height",image_height_))
      image_height_ = IMAGE_HEIGHT;
    if (!n.getParam("image_width_init",image_width_init_))
      image_width_init_ = IMAGE_WIDTH_INIT;
    if (!n.getParam("image_height_init",image_height_init_))
      image_height_init_ = IMAGE_HEIGHT_INIT;
    image_size_ = image_width_*image_height_;

    // Initialize rgb image.
    latest_rgb_image_.resize(image_size_*3,0);

    // Initialize depth image.
    latest_depth_image_.resize(image_size_,0);

    // Set time.
    latest_rgb_time_ = ros::Time(0.0);
    latest_depth_time_ = ros::Time(0.0);
}

// Destructor.
CameraSensor::~CameraSensor()
{
    // Nothing to do here.
}

// Callback from camera sensor. Crops and updates the stored rgb image
void CameraSensor::update_rgb_image(const sensor_msgs::Image::ConstPtr& msg) {
    latest_rgb_time_ = msg->header.stamp;
    if (latest_rgb_image_.empty()) {
        latest_rgb_image_.resize(3*image_size_);
    } else { // better way to make this assertion? (error message?)
        assert(latest_rgb_image_.size() == 3*image_size_);
    }
    // Check message dimensions.
    assert(msg->width == image_width_init_ && msg->height == image_height_init_);

    int x_start = (image_width_init_ - image_width_) / 2;
    int y_start = (image_height_init_ - image_height_) / 2;

    // Store the image, cropping middle region according to image width and image height
    /* TODO - this could be done more efficiently. */
    for (int y = 0; y < image_height_init_; y++)
    {
        for (int x = 0; x < image_width_init_; x++)
        {
            if (x >= x_start && x < x_start+image_width_)
            {
                if (y >= y_start && y < y_start+image_height_)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        latest_rgb_image_[(y-y_start)*image_width_*3 + (x-x_start)*3 + c] = msg->data[y*image_width_init_*3 + x*3 + c];
                    }
                }
            }
        }
    }
}

// Callback from camera sensor. Crops and updates the stored depth image
void CameraSensor::update_depth_image(const sensor_msgs::Image::ConstPtr& msg) {
    latest_depth_time_ = msg->header.stamp;
    if (latest_depth_image_.empty()) {
        latest_depth_image_.resize(image_size_);
    } else { // better way to make this assertion? (error message?)
        assert(latest_depth_image_.size() == 3*image_size_);
    }
    // Check message dimensions.
    assert(msg->width == image_width_init_ && msg->height == image_height_init_);

    int x_start = (image_width_init_ - image_width_) / 2;
    int y_start = (image_height_init_ - image_height_) / 2;

    // Store the image, cropping middle region according to image width and image height
    /* TODO - this could be done more efficiently. */
    for (int y = 0; y < image_height_init_; y++)
    {
        for (int x = 0; x < image_width_init_; x++)
        {
            if (x >= x_start && x < x_start+image_width_)
            {
                if (y >= y_start && y < y_start+image_height_)
                {
                    latest_depth_image_[(y-y_start)*image_width_ + (x-x_start)] = msg->data[y*image_width_init_ + x];
                }
            }
        }
    }
}

// Update the sensor (called every tick).
void CameraSensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
  // Not needed for camera, since callback function is used.
}

// The settings include the configuration for the Kalman filter.
void CameraSensor::configure_sensor(const OptionsMap &options)
{
    // not used for camera sensor, though maybe in the future for image specs.
}

// Set data format and meta data on the provided sample.
void CameraSensor::set_sample_data_format(boost::scoped_ptr<Sample> sample) const
{
    // Set image size and format.
    OptionsMap rgb_metadata;
    sample->set_meta_data(gps::RGB_IMAGE,image_size_*3,SampleDataFormatUInt8,rgb_metadata);

    // Set joint velocities size and format.
    OptionsMap depth_metadata;
    sample->set_meta_data(gps::DEPTH_IMAGE,image_size_*2,SampleDataFormatUInt16,depth_metadata);
}

// Set data on the provided sample.
void CameraSensor::set_sample_data(boost::scoped_ptr<Sample> sample) const
{
    // Set rgb image.
    sample->set_data(0,gps::RGB_IMAGE,&latest_rgb_image_[0],latest_rgb_image_.size(),SampleDataFormatUInt8);

    // Set depth image.
    sample->set_data(0,gps::DEPTH_IMAGE,&latest_depth_image_[0],latest_depth_image_.size(),SampleDataFormatUInt16);
}
