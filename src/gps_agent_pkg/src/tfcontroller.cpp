
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/util.h"
#include "gps_agent_pkg/tfcontroller.h"

using namespace gps_control;

// Constructor.
TfController::TfController()
: TrialController()
{
    is_configured_ = false;

}

// Destructor.
TfController::~TfController() {
}

void TfController::update_action_command(int id, const Eigen::VectorXd &command) {
    last_command_id_received = id;
    last_action_command_received = command;
}

void TfController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
    if (is_configured_) {
        if(last_command_id_acted_upon < last_command_id_received){
            last_command_id_acted_upon = last_command_id_received;
            failed_attempts = 0;
            U = last_action_command_received;
        }
        else if(failed_attempts < 2){ //this would allow acting on stale actions...maybe a bad idea?
            U = last_action_command_received;
            failed_attempts++;
        }
        else{
            ROS_FATAL("no new action command received. Can not act on stale actions.");
        }
    }
}

// Configure the controller.
void TfController::configure_controller(OptionsMap &options)
{
    last_command_id_received = 0;
    last_command_id_acted_upon = 0;
    failed_attempts = 0;
    int dU = boost::get<int>(options["dU"]);
    last_action_command_received.resize(dU);
    for (int i = 0; i < dU; ++i)
    {
        last_action_command_received(i) = 0;
    }
    //Call superclass
    TrialController::configure_controller(options);
    ROS_INFO_STREAM("Set Tensorflow network parameters");
    is_configured_ = true;
}

void TfController::publish_obs(Eigen::VectorXd obs, RobotPlugin *plugin){
    plugin ->tf_publish_obs(obs);
}
