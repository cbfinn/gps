import pickle
import os
import uuid

import numpy as np
import tensorflow as tf

from gps.algorithm.policy.policy import Policy


class TfPolicy(Policy):
    """
    A neural network policy implemented in tensor flow. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        obs_tensor: tensor representing tf observation. Used in feed dict for forward pass.
        act_op: tf op to execute the forward pass. Use sess.run on this op.
        var: Du-dimensional noise variance vector.
        sess: tf session.
        device_string: tf device string for running on either gpu or cpu.
    """
    def __init__(self, dU, obs_tensor, act_op, feat_op, var, sess, device_string, copy_param_scope=None):
        Policy.__init__(self)
        self.dU = dU
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.feat_op = feat_op
        self.sess = sess
        self.device_string = device_string
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None

        if copy_param_scope:
            self.copy_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=copy_param_scope)
            self.copy_params_assign_placeholders = [tf.placeholder(tf.float32, shape=param.get_shape()) for
                                                      param in self.copy_params]

            self.copy_params_assign_ops = [tf.assign(self.copy_params[i],
                                                     self.copy_params_assign_placeholders[i])
                                             for i in range(len(self.copy_params))]

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
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        with tf.device(self.device_string):
            action_mean = self.sess.run(self.act_op, feed_dict={self.obs_tensor: obs})
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u[0]  # the DAG computations are batched by default, but we use batch size 1.

    def get_features(self, obs):
        """
        Return the image features for an observation.
        Args:
            obs: Observation vector.
        """
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        # Assume that features don't depend on the robot config, so don't normalize by scale and bias.
        with tf.device(self.device_string):
            feat = self.sess.run(self.feat_op, feed_dict={self.obs_tensor: obs})
        return feat[0]  # the DAG computations are batched by default, but we use batch size 1.

    def get_copy_params(self):
        param_values = self.sess.run(self.copy_params)
        return {self.copy_params[i].name:param_values[i] for i in range(len(self.copy_params))}

    def set_copy_params(self, param_values):
        value_list = [param_values[self.copy_params[i].name] for i in range(len(self.copy_params))]
        feeds = {self.copy_params_assign_placeholders[i]:value_list[i] for i in range(len(self.copy_params))}
        self.sess.run(self.copy_params_assign_ops, feed_dict=feeds)


    def pickle_policy(self, deg_obs, deg_action, checkpoint_path, goal_state=None, should_hash=False):
        """
        We can save just the policy if we are only interested in running forward at a later point
        without needing a policy optimization class. Useful for debugging and deploying.
        """
        if should_hash is True:
            hash_str = str(uuid.uuid4())
            checkpoint_path += hash_str
        os.mkdir(checkpoint_path + '/')
        checkpoint_path += '/_pol'
        pickled_pol = {'deg_obs': deg_obs, 'deg_action': deg_action, 'chol_pol_covar': self.chol_pol_covar,
                       'checkpoint_path_tf': checkpoint_path + '_tf_data', 'scale': self.scale, 'bias': self.bias,
                       'device_string': self.device_string, 'goal_state': goal_state, 'x_idx': self.x_idx}
        pickle.dump(pickled_pol, open(checkpoint_path, "wb"))
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path + '_tf_data')

    @classmethod
    def load_policy(cls, policy_dict_path, tf_generator, network_config=None):
        """
        For when we only need to load a policy for the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        pol_dict = pickle.load(open(policy_dict_path, "rb"))
        tf_map = tf_generator(dim_input=pol_dict['deg_obs'], dim_output=pol_dict['deg_action'],
                              batch_size=1, network_config=network_config)

        sess = tf.Session()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        saver = tf.train.Saver()
        check_file = pol_dict['checkpoint_path_tf']
        saver.restore(sess, check_file)

        device_string = pol_dict['device_string']

        cls_init = cls(pol_dict['deg_action'], tf_map.get_input_tensor(), tf_map.get_output_op(), np.zeros((1,)),
                       sess, device_string)
        cls_init.chol_pol_covar = pol_dict['chol_pol_covar']
        cls_init.scale = pol_dict['scale']
        cls_init.bias = pol_dict['bias']
        cls_init.x_idx = pol_dict['x_idx']
        return cls_init

