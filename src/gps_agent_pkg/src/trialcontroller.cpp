#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/trialcontroller.h"

using namespace gps_control;

// Constructor.
TrialController::TrialController() : Controller()
{
    // Set initial time.
    last_update_time_ = ros::Time(0.0);
    step_counter_ = 0;
    trial_end_step_ = 1;
}

// Destructor.
TrialController::~TrialController()
{
}

// Update the controller (take an action).
void TrialController::update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques)
{
    if (is_finished()){
        ROS_ERROR("Updating when controller is finished. May seg fault.");
    }
    Eigen::VectorXd X, obs;
    sample->get_data(step_counter_, X, state_datatypes_);
    sample->get_data(step_counter_, obs, obs_datatypes_);

    // Ask subclass to fill in torques
    get_action(step_counter_, X, obs, torques);

    // Set the torques for the sample
    sample->set_data(step_counter_,gps::ACTION,torques,torques.size(),SampleDataFormatDouble);

    // Update last update time.
    last_update_time_ = current_time;
    step_counter_++;
    ROS_INFO("Step counter: %d", step_counter_);
}

void TrialController::configure_controller(OptionsMap &options)
{
    ROS_INFO_STREAM(">TrialController::configure_controller");
    if(!is_finished()){
        // TODO(chelsea/sergey/zoe) This error happens every time...
        ROS_ERROR("Cannot configure controller while a trial is in progress");
    }
    std::vector<int> datatypes;

    int T = boost::get<int>(options["T"]);
    step_counter_ = 0;
    trial_end_step_ = T;

    datatypes = boost::get<std::vector<int> >(options["state_datatypes"]);
    state_datatypes_.resize(datatypes.size());
    for(int i=0; i<datatypes.size(); i++){
        state_datatypes_[i] = (gps::SampleType) datatypes[i];
    }

    datatypes = boost::get<std::vector<int> >(options["obs_datatypes"]);
    obs_datatypes_.resize(datatypes.size());
    for(int i=0; i<datatypes.size(); i++){
        obs_datatypes_[i] = (gps::SampleType) datatypes[i];
    }

}

// Check if controller is finished with its current task.
bool TrialController::is_finished() const
{
    return step_counter_ >= trial_end_step_;
}

int TrialController::get_step_counter(){
    return step_counter_;
}

int TrialController::get_trial_length(){
    return trial_end_step_;
}

// Reset the controller -- this is typically called when the controller is turned on.
void TrialController::reset(ros::Time time)
{
    last_update_time_ = time;
    step_counter_ = 0;
    trial_end_step_ = 1;
}

