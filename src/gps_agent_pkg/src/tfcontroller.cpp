
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/util.h"

using namespace gps_control;

// Constructor.
TensorflowController::TensorflowController()
: TrialController()
{
    is_configured_ = false;

}


// Destructor.
TensorflowController::~TensorflowController() {
}

void TensorflowController::update_action_command(int id, const Eigen::VectorXd &command) {
    last_command_id_received = id;
    last_action_command_received = command;
}

void TensorflowController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
    if (is_configured_) {
        if(last_command_id_acted_upon < last_command_id_received){
            last_command_id_acted_upon = last_command_id_received;
            U = last_action_command_received;
        }
        else{
            ROS_FATAL("no new action command received. Can not act on stale actions.");
        }
    }
}

// Configure the controller.
void TensorflowController::configure_controller(OptionsMap &options)
{
    //ros::NodeHandle& n;
    //action_topic_name_ = "robot_action";
    //action_subscriber_ = n.subscribe(action_topic_name, 1, &TensorflowController::update_action_command, this);

    //Call superclass
    TrialController::configure_controller(options);
    ROS_INFO_STREAM("Set Tensorflow network parameters");
    is_configured_ = true;
}
