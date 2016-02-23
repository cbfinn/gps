#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/lingausscontroller.h"
#include "gps_agent_pkg/util.h"

using namespace gps_control;

// Constructor.
LinearGaussianController::LinearGaussianController()
: TrialController()
{
    is_configured_ = false;
}

// Destructor.
LinearGaussianController::~LinearGaussianController()
{
}


void LinearGaussianController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
    U = K_[t]*X+k_[t];
}

// Configure the controller.
void LinearGaussianController::configure_controller(OptionsMap &options)
{
    //Call superclass
    TrialController::configure_controller(options);

    // TODO: Update K_
    int T = boost::get<int>(options["T"]);

    //TODO Don't do this hacky string indexing
    K_.resize(T);
    for(int i=0; i<T; i++){
        K_[i] = boost::get<Eigen::MatrixXd>(options["K_"+to_string(i)]);
    }

    k_.resize(T);
    for(int i=0; i<T; i++){
        k_[i] = boost::get<Eigen::VectorXd>(options["k_"+to_string(i)]);
    }
    ROS_INFO_STREAM("Set LG parameters");
    is_configured_ = true;
}
