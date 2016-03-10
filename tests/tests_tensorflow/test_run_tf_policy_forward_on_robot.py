import os
import sys
import numpy as np
import rospy

# Add gps/python to path so that imports work.
gps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'python'))
sys.path.append(gps_path)

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.config_tf import POLICY_OPT_TF

from gps.agent.config import AGENT_ROS
from gps.agent.ros.run_tf_policy_forward_on_robot import ForwardTfAgent
from gps.agent.ros.run_tf_policy_forward_on_robot import policy_to_msg
from gps.agent.ros.run_tf_policy_forward_on_robot import msg_to_sample
from gps_agent_pkg.msg import TfActionCommand


def test_init():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    policy_opt.policy.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    policy_opt.policy.bias = -np.mean(obs_reshaped.dot(policy_opt.policy.scale), axis=0)
    ForwardTfAgent(policy_opt.policy)


def test_init_from_checkpoint():
    # this part is just for creating the file that we will then load.
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    check_path = gps_path + '/gps/algorithm/policy_opt/tf_checkpoint/policy_checkpoint'
    policy_opt.policy.pickle_policy(deg_obs, deg_action, check_path)

    # actual loading test is here.
    tf_map = POLICY_OPT_TF['network_model']
    check_path = gps_path + '/gps/algorithm/policy_opt/tf_checkpoint/policy_checkpoint'
    ForwardTfAgent.init_from_saved_policy(check_path, tf_map)


def test_run_service():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    policy_opt.policy.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    policy_opt.policy.bias = -np.mean(obs_reshaped.dot(policy_opt.policy.scale), axis=0)
    agent = ForwardTfAgent(policy_opt.policy)

    fake_sample_topic = rospy.Publisher('robot_action', TfActionCommand)
    rospy.init_node('gps_agent_ros_node')
    for iter_step in range(0, 5):
        msg = np.zeros((7,))
        fake_sample_topic.publish(policy_to_msg(7, msg, 0))
        agent.run_service()
        print iter_step
    #    publish and subscribe


def main():
    #test_init()
    #test_init_from_checkpoint()
    test_run_service()

if __name__ == '__main__':
    main()



