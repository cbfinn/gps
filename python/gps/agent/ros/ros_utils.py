""" This file defines utilities for the ROS agents. """
import numpy as np

import rospy

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps_agent_pkg.msg import ControllerParams, LinGaussParams, TfParams, CaffeParams, TfActionCommand
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import LIN_GAUSS_CONTROLLER, CAFFE_CONTROLLER, TF_CONTROLLER
import logging
LOGGER = logging.getLogger(__name__)
try:
    from gps.algorithm.policy.caffe_policy import CaffePolicy
    NO_CAFFE = False
except ImportError as e:
    NO_CAFFE = True
    LOGGER.info("Caffe not imported")
try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:
    TfPolicy = None


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


def policy_to_msg(policy, noise):
    """
    Convert a policy object to a ROS ControllerParams message.
    """
    msg = ControllerParams()
    if isinstance(policy, LinearGaussianPolicy):
        msg.controller_to_execute = LIN_GAUSS_CONTROLLER
        msg.lingauss = LinGaussParams()
        msg.lingauss.dX = policy.dX
        msg.lingauss.dU = policy.dU
        msg.lingauss.K_t = \
                policy.K.reshape(policy.T * policy.dX * policy.dU).tolist()
        msg.lingauss.k_t = \
                policy.fold_k(noise).reshape(policy.T * policy.dU).tolist()
    elif NO_CAFFE is False and isinstance(policy, CaffePolicy):
        msg.controller_to_execute = CAFFE_CONTROLLER
        msg.caffe = CaffeParams()
        msg.caffe.net_param = policy.get_net_param()
        msg.caffe.bias = policy.bias.tolist()
        msg.caffe.dU = policy.dU
        scale_shape = policy.scale.shape
        msg.caffe.scale = policy.scale.reshape(scale_shape[0] * scale_shape[1]).tolist()
        msg.caffe.dim_bias = scale_shape[0]
        scaled_noise = np.zeros_like(noise)
        for i in range(noise.shape[0]):
            scaled_noise[i] = policy.chol_pol_covar.T.dot(noise[i])
        msg.caffe.noise = scaled_noise.reshape(-1).tolist()
    elif isinstance(policy, TfPolicy):
        msg.controller_to_execute = TF_CONTROLLER
        msg.tf = TfParams()
        msg.tf.dU = policy.dU
    else:
        raise NotImplementedError("Caffe not imported or Unknown policy object: %s" % policy)
    return msg


def tf_policy_to_action_msg(deg_action, action, action_id):
        """
        Convert an action to a TFActionCommand message.
        """
        msg = TfActionCommand()
        msg.action = action.tolist()
        msg.dU = deg_action
        msg.id = action_id
        return msg


def tf_obs_msg_to_numpy(obs_message):
    # ToDo: Reshape this if needed.
    return np.array(obs_message.data)


class TimeoutException(Exception):
    """ Exception thrown on timeouts. """
    def __init__(self, sec_waited):
        Exception.__init__(self, "Timed out after %f seconds", sec_waited)


class ServiceEmulator(object):
    """
    Emulates a ROS service (request-response) from a
    publisher-subscriber pair.
    Args:
        pub_topic: Publisher topic.
        pub_type: Publisher message type.
        sub_topic: Subscriber topic.
        sub_type: Subscriber message type.
    """
    def __init__(self, pub_topic, pub_type, sub_topic, sub_type):
        self._pub = rospy.Publisher(pub_topic, pub_type)
        self._sub = rospy.Subscriber(sub_topic, sub_type, self._callback)

        self._waiting = False
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg):
        """ Publish a message without waiting for response. """
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg, timeout=5.0, poll_delay=0.01,
                         check_id=False):
        """
        Publish a message and wait for the response.
        Args:
            pub_msg: Message to publish.
            timeout: Timeout in seconds.
            poll_delay: Speed of polling for the subscriber message in
                seconds.
            check_id: If enabled, will only return messages with a
                matching id field.
        Returns:
            sub_msg: Subscriber message.
        """
        if check_id:  # This is not yet implemented in C++.
            raise NotImplementedError()

        self._waiting = True
        self.publish(pub_msg)

        time_waited = 0
        while self._waiting:
            rospy.sleep(poll_delay)
            time_waited += 0.01
            if time_waited > timeout:
                raise TimeoutException(time_waited)
        return self._subscriber_msg
