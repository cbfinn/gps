import tempfile

import numpy as np
import tensorflow as tf

from gps.algorithm.policy.policy import Policy


class TfPolicy(Policy):
    """
    A neural network policy implemented in Caffe. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        test_net: Initialized tf network that can run forward.
        var: Du-dimensional noise variance vector.
    """
    def __init__(self, obs_tensor, act_op, var, sess, device_string):
        Policy.__init__(self)
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.sess = sess
        self.device_string = device_string  # is it even worth running this on the gpu? Won't comm time dominate?
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # Normalize obs.
        obs = obs.dot(self.scale) + self.bias
        with tf.device(self.device_string):
            action_mean = self.sess.run(self.act_op, feed_dict={self.obs_tensor: np.expand_dims(obs, 0)})
        u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u
