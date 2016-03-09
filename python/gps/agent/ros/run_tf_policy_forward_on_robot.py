import tensorflow as tf
import time
import pickle
import numpy as np

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.config_tf import POLICY_OPT_TF
#from gps_agent_pkg.msg import TfActionCommand
#from gps_agent_pkg.msg import SampleResult
#from gps.agent.ros.ros_utils import ServiceEmulator
#from gps.sample.sample import Sample


class ForwardTfAgent:
    def __init__(self, policy):
        # default, you pass the policy and the noise in.
        self.policy = policy
        self.current_action_id = 0
        self.sess = policy.sess
        #self.ross_service = ServiceEmulator('robot_action', TfActionCommand,
        #                                    'robot_observation', )

    @classmethod
    def init_from_saved_policy(cls, path_to_policy_checkpoint, tf_generator):
        """
        Load policy from policy checkpoint. For running forward without initializing policy_opt.
        See tf_policy's load_policy class method.
        """
        pol = TfPolicy.load_policy(path_to_policy_checkpoint, tf_generator=tf_generator)
        return cls(pol)

    def run_service(self, time_to_run=None):
        should_stop = False
        action = policy_to_msg(np.zeros(self.dU))
        start_time = time.time()
        while should_stop is False:
            ros_sample = msg_to_sample(self.ross_service.publish_and_wait(action, timeout=time_to_run))
            current_time = time.time()
            if current_time - start_time > time_to_run:
                should_stop = True
            else:
                action = policy_to_msg(self._get_new_action(ros_sample), self.current_action_id)

    def _get_new_action(self, obs):
        self.current_action_id += 1
        return self.policy.act(None, obs, None, None)


def policy_to_msg(deg_action, action, action_id):
    """
    Convert an action to a TFActionCommand message.
    """
    msg = TfActionCommand()
    msg.action = action
    msg.dU = deg_action
    msg.id = action_id
    return msg


def msg_to_sample(ros_msg, agent):
    """
    Convert a SampleResult ROS message into a Sample Python object.
    """
    sample = Sample(agent)
    for sensor in ros_msg.sensor_data:
        sensor_id = sensor.data_type
        shape = np.array(sensor.shape)
        data = np.array(sensor.data).reshape(shape)
        sample.set(sensor_id, data)
    return sample
