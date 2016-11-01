""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver


LOGGER = logging.getLogger(__name__)


class PolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.batch_size = self._hyperparams['batch_size']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            self.gpu_device = self._hyperparams['gpu_id']
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_op = None  # mu_hat
        self.feat_op = None # features
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.feat_vals = None
        self.init_network()
        self.init_solver()
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        self.sess = tf.Session()
        self.policy = TfPolicy(dU, self.obs_tensor, self.act_op, self.feat_op,
                               np.zeros(dU), self.sess, self.device_string, copy_param_scope=self._hyperparams['copy_param_scope'])
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        if 'obs_image_data' not in self._hyperparams['network_params']:
            self._hyperparams['network_params'].update({'obs_image_data': []})
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
            i += dim
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
                                  network_config=self._hyperparams['network_params'])
        self.obs_tensor = tf_map.get_input_tensor()
        self.precision_tensor = tf_map.get_precision_tensor()
        self.action_tensor = tf_map.get_target_output_tensor()
        self.act_op = tf_map.get_output_op()
        self.feat_op = tf_map.get_feature_op()
        self.loss_scalar = tf_map.get_loss_op()
        self.fc_vars = fc_vars
        self.last_conv_vars = last_conv_vars

        # Setup the gradients
        self.grads = [tf.gradients(self.act_op[:,u], self.obs_tensor)[0]
                for u in range(self._dU)]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver = TfSolver(loss_scalar=self.loss_scalar,
                               solver_name=self._hyperparams['solver_type'],
                               base_lr=self._hyperparams['lr'],
                               lr_policy=self._hyperparams['lr_policy'],
                               momentum=self._hyperparams['momentum'],
                               weight_decay=self._hyperparams['weight_decay'],
                               fc_vars=self.fc_vars,
                               last_conv_vars=self.last_conv_vars)
        self.saver = tf.train.Saver()

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.solver.get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.last_conv_vars: conv_values[idx_i],
                             self.action_tensor: tgt_mu[idx_i],
                             self.precision_tensor: tgt_prc[idx_i]}
                train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                if (i+1) % 500 == 0:
                    LOGGER.info('tensorflow iteration %d, average loss %f',
                                    i+1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.obs_tensor: obs[idx_i],
                         self.action_tensor: tgt_mu[idx_i],
                         self.precision_tensor: tgt_prc[idx_i]}
            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f',
                             i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.feat_op is not None:
            self.feat_vals = self.solver.get_var_values(self.sess, self.feat_op, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))

        return self.policy

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if self.policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, self.x_idx] = (obs[n, :, self.x_idx].T.dot(self.policy.scale)
                                         + self.policy.bias).T

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                feed_dict = {self.obs_tensor: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(self.act_op, feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    def save_model(self, fname):
        LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname, write_meta_graph=False)

    def restore_model(self, fname):
        self.saver.restore(self.sess, fname)
        LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save_model(f.name) # TODO - is this implemented.
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'tf_iter': self.tf_iter,
            'x_idx': self.policy.x_idx,
            'chol_pol_covar': self.policy.chol_pol_covar,
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.policy.x_idx = state['x_idx']
        self.policy.chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']

        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)

