#include "gps_agent_pkg/controller.h"

using namespace gps_control;

// Constructors.
Controller::Controller(ros::NodeHandle& n, gps::ActuatorType arm, int size)
{
}

Controller::Controller()
{
}

// Destructor.
Controller::~Controller()
{
}

void Controller::configure_controller(OptionsMap &options)
{
}

void Controller::set_update_delay(double new_step_length)
{
}

double Controller::get_update_delay()
{
    return 1.0;
}

void Controller::reset(ros::Time update_time)
{
}

