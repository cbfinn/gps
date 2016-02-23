#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/encodersensor.h"

using namespace gps_control;

// Factory function.
Sensor* Sensor::create_sensor(SensorType type, ros::NodeHandle& n, RobotPlugin *plugin, gps::ActuatorType actuator_type)
{
    switch (type)
    {
    case EncoderSensorType:
        return (Sensor *) (new EncoderSensor(n,plugin,actuator_type));
    /*
    case CameraSensorType:
        return CameraSensor(n,plugin);
    */
    default:
        ROS_ERROR("Unknown sensor type %i requested from sensor constructor!",type);
        return NULL;
    }
}

// Constructor.
Sensor::Sensor(ros::NodeHandle& n, RobotPlugin *plugin)
{
    // Nothing to do.
}

// Destructor.
Sensor::~Sensor()
{
    // Nothing to do.
}

// Reset the sensor, clearing any previous state and setting it to the current state.
void Sensor::reset(RobotPlugin *plugin, ros::Time current_time)
{
    // Nothing to do.
}

// Update the sensor (called every tick).
void Sensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
    // Nothing to do.
}

// Set sensor update delay.
void Sensor::set_update(double new_sensor_step_length)
{
    sensor_step_length_ = new_sensor_step_length;
}

// Configure the sensor (for sensor-specific trial settings).
void Sensor::configure_sensor(OptionsMap &options)
{
    // Nothing to do.
}

void Sensor::set_sample_data_format(boost::scoped_ptr<Sample>& sample)
{

}

void Sensor::set_sample_data(boost::scoped_ptr<Sample>& sample, int t)
{

}
