/*
This is the base class for the sensor object. The sensor object encapsulates
anything that produces state (X) or observation (phi) information.
*/
#pragma once

// Headers.
#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>

// This header defines the main enum that lists the available sensors, which
// is also used by the state assembler.
#include "gps/proto/gps.pb.h"

// This header contains additional defines for communicating with the sample object.
#include "gps_agent_pkg/sample.h"

namespace gps_control
{

// List of sensor types.
// Note that each sensor might produce multiple data types!
enum SensorType
{
    EncoderSensorType = 0,
    CameraSensorType,
    TotalSensorTypes
};

// Forward declarations.
class Sample;
class RobotPlugin;

class Sensor
{
protected:
    // Current sensor update delay, in seconds (should match controller step length).
    double sensor_step_length_;
public:
    // Factory function.
    static Sensor* create_sensor(SensorType type, ros::NodeHandle& n, RobotPlugin *plugin, gps::ActuatorType);
    // Constructor.
    Sensor(ros::NodeHandle& n, RobotPlugin *plugin);
    // Destructor.
    virtual ~Sensor();
    // Reset the sensor, clearing any previous state and setting it to the current state.
    virtual void reset(RobotPlugin *plugin, ros::Time current_time);
    // Update the sensor (called every tick).
    // plugin is used to query robot-specific information (such as forward kinematics).
    // sec_elapsed is used to estimate time -- this specifies how much time elapsed since the last update.
    // is_controller_step -- this specifies whether this update coincides with a linear-Gaussian or neural network controller update.
    virtual void update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step);
    // Set Sensor update delay.
    virtual void set_update(double new_sensor_step_length);
    // Configure the Sensor (for Sensor-specific trial settings).
    virtual void configure_sensor(OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual void set_sample_data_format(boost::scoped_ptr<Sample>& sample);
    // Set data on the provided sample.
    virtual void set_sample_data(boost::scoped_ptr<Sample>& sample, int t);
};

}

/*
How the sensor objects are supposed to work:
- each sensor object may include subscribers
- each sensor object has an update function
- this update function includes the robot's state
- a joint angle sensor will just store the latest input
- a vision sensor will store the latest image from the subscriber
- the vision sensor does pretty much nothing during the update
- this is also true for other subscriber-based sensor, such as F/T sensor
- be sure to add warnings if a subscriber-based sensor has not updated recently
- this is done by storing time stamps, and checking how old the newest message is
- note that updates will happen at up to 1000 Hz, so we won't get messages for each update
- however, we can have the RobotPlugin tell us the frequency of the current controller, and check we have at least that frequency
*/
