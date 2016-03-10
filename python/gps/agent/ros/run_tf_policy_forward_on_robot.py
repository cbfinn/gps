import tensorflow as tf
import time
import pickle
import numpy as np
import rospy

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.config_tf import POLICY_OPT_TF
from gps_agent_pkg.msg import TfActionCommand
from gps_agent_pkg.msg import SampleResult
#from gps.agent.ros.ros_utils import ServiceEmulator
from gps.sample.sample import Sample


class ForwardTfAgent:
    def __init__(self, policy, dU):
        """
        Default. You pass the policy in.
        """
        self.policy = policy
        self.current_action_id = 0
        self.dU = dU
        self.sess = policy.sess
        #self.ross_service = ServiceEmulator('robot_action', TfActionCommand,
        #                                    'robot_observation', SampleResult)

        self._pub = rospy.Publisher('robot_action', TfActionCommand)
        self._sub = rospy.Subscriber('robot_observation', SampleResult, self._callback)

        self._subscriber_msg = None
        self.observations_stale = True

    @classmethod
    def init_from_saved_policy(cls, path_to_policy_checkpoint, tf_generator):
        """
        Load policy from policy checkpoint. For running forward without initializing policy_opt.
        See tf_policy's load_policy class method.
        """
        pol = TfPolicy.load_policy(path_to_policy_checkpoint, tf_generator=tf_generator)
        return cls(pol)

    def _callback(self, message):
            self._subscriber_msg = message
            self.observations_stale = False

    def publish(self, pub_msg):
        """ Publish a message without waiting for response. """
        self._pub.publish(pub_msg)

    def run_service(self, time_to_run=5):
        should_stop = False
        start_time = time.time()
        action = policy_to_msg(self.dU, np.zeros(self.dU), self.current_action_id)
        self.current_action_id += 1
        self.publish(policy_to_msg(self.dU, action, self.current_action_id))
        while should_stop is False:
            current_time = time.time()
            if current_time - start_time > time_to_run:
                    should_stop = True
            elif self.observations_stale is False:
                last_obs = msg_to_sample(self._subscriber_msg, None)
                action = policy_to_msg(self.dU, self._get_new_action(last_obs), self.current_action_id)
                self.publish(policy_to_msg(self.dU, action, self.current_action_id))
                self.observations_stale = True
                self.current_action_id += 1
            else:
                rospy.sleep(0.005)

    def _get_new_action(self, obs):
        self.current_action_id += 1
        return self.policy.act(None, obs, None, None)


def policy_to_msg(deg_action, action, action_id):
    """
    Convert an action to a TFActionCommand message.
    """
    msg = TfActionCommand()
    msg.action = action.tolist()
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
