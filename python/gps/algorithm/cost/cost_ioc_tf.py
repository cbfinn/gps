
""" This file defines neural network cost function. """
import copy
import logging
import numpy as np
import tempfile

import tensorflow as tf
from tf_cost_utils import jacobian_t, construct_nn_cost_net_tf
from google.protobuf.text_format import MessageToString

from gps.algorithm.cost.config import COST_IOC_NN
from gps.algorithm.cost.cost import Cost
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES,\
    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_JACOBIANS

LOGGER = logging.getLogger(__name__)


class CostIOCTF(Cost):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_IOC_NN) # Set this up in the config?
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._dO = self._hyperparams['dO']
        self._T = self._hyperparams['T']
        self.use_jacobian = self._hyperparams['use_jacobian']

        self.demo_batch_size = self._hyperparams['demo_batch_size']
        self.sample_batch_size = self._hyperparams['sample_batch_size']

        self._iteration_count = 1

        self._init_solver()

    def copy(self):
      raise NotImplementedError()

  # TODO - cache dfdx / add option to not compute it when necessary
    def compute_dfdx(self, obs):
      """
      Evaluate the jacobian of the features w.r.t. the input.
      Args:
        obs: The vector of observations of a single sample
      Returns:
        feat: The features (byproduct of computing dfdx)
        dydx: The jacobians dfdx
      """
      feat, dfdx = self.run([self.test_feat, self.dfdx], test_obs=obs)
      return feat, dfdx


    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        # TODO - right now, we're going to assume that Obs = X
        T = sample.T
        obs = sample.get_obs()
        sample_u = sample.get_U()

        dO = self._dO
        dU = sample.dU
        dX = sample.dX
        # Initialize terms.
        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        tq_norm = np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1, keepdims=True)
        l, A, b, wu = self.run([self.outputs['test_loss'], 
                                self.outputs['A'], 
                                self.outputs['b']
                                self.outputs['wu']], test_obs=obs, test_torque_norm=tq_norm)
        weighted_array = np.c_[A, b]  # weighted_array = np.c_[params['Ax'][0].data, np.array([params['Ax'][1].data]).T]
        A = weighted_array.T.dot(weighted_array)

        # get intermediate features
        feat, dfdx = self.compute_dfdx(obs)

        #l += 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)  # already computed by the network
        if self._hyperparams['learn_wu']:
            wu_mult = wu
        else:
            wu_mult = 1.0
        lu = wu_mult * self._hyperparams['wu'] * sample_u
        luu = wu_mult * np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])

        dldf = A.dot(np.vstack((feat.T, np.ones((1, T))))) # Assuming A is a (df + 1) x (df + 1) matrix
        dF = feat.shape[1]
        for t in xrange(T):
          lx[t, :] = dfdx[t, :, :].T.dot(dldf[:dF, t])
          lxx[t, :, :] = dfdx[t, :, :].T.dot(A[:dF,:dF]).dot(dfdx[t, :, :])

        if self.use_jacobian and END_EFFECTOR_POINT_JACOBIANS in sample._data:
            jnt_idx = sample.agent.get_idx_x(JOINT_ANGLES)
            vel_idx = sample.agent.get_idx_x(JOINT_VELOCITIES)

            jx = sample.get(END_EFFECTOR_POINT_JACOBIANS)
            dl_dee = sample.agent.unpack_data_x(lx, [END_EFFECTOR_POINTS])
            dl_dev = sample.agent.unpack_data_x(lx, [END_EFFECTOR_POINT_VELOCITIES])

            for t in xrange(T):
                lx[t, jnt_idx] += jx[t].T.dot(dl_dee[t])
                lx[t, vel_idx] += jx[t].T.dot(dl_dev[t])

        return l, lx, lu, lxx, luu, lux

    def update(self, demoU, demoX, demoO, d_log_iw, sampleU, sampleX, sampleO, s_log_iw, itr=-1):
        """
        Learn cost function with generic function representation.
        Args:
            demoU: the actions of demonstrations.
            demoX: the states of demonstrations.
            demoO: the observations of demonstrations.
            d_log_iw: log importance weights for demos.
            sampleU: the actions of samples.
            sampleX: the states of samples.
            sampleO: the observations of samples.
            s_log_iw: log importance weights for samples.
        """
        raise NotImplementedError()
        # Call self.ioc_optimizer


    def _init_solver(self, sample_batch_size=None):
        """ Helper method to initialize the solver. """
        """
        solver_param.display = 0  # Don't display anything.
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']
        solver_param.random_seed = self._hyperparams['random_seed']
        """

        # Pass in net parameter by protostring (could add option to input prototxt file).
        network_arch_params = self._hyperparams['network_arch_params']

        network_arch_params['dim_input'] = self._dO
        network_arch_params['demo_batch_size'] = self._hyperparams['demo_batch_size']
        if sample_batch_size is None:
            network_arch_params['sample_batch_size'] = self._hyperparams['sample_batch_size']
        else:
            network_arch_params['sample_batch_size'] = sample_batch_size
        network_arch_params['T'] = self._T
        network_arch_params['phase'] = TRAIN
        network_arch_params['ioc_loss'] = self._hyperparams['ioc_loss']
        network_arch_params['Nq'] = self._iteration_count
        network_arch_params['smooth_reg_weight'] = self._hyperparams['smooth_reg_weight']
        network_arch_params['mono_reg_weight'] = self._hyperparams['mono_reg_weight']
        network_arch_params['gp_reg_weight'] = self._hyperparams['gp_reg_weight']
        network_arch_params['learn_wu'] = self._hyperparams['learn_wu']
        inputs, outputs = self._hyperparams['network_model'](**network_arch_params)

        self.input_dict = inputs
        self.sup_loss = outputs['sup_loss']
        self.ioc_loss = outputs['ioc_loss']
        self.test_feats = outputs['test_feats']

        optimizer = tf.train.AdamOptimizer(learning_rate=self._hyperparams['lr'])
        self.ioc_optimizer = optimizer.optimize(self.ioc_loss)
        self.sup_optimizer = optimizer.optimize(self.sup_loss)

        # Set up gradients
        test_obs = inputs['test_obs']
        self.dfdx = jacobian_t(test_feats, test_obs)
        
        self.session = tf.Session()

    def run(targets, **feeds):
        feed_dict = {self.input_dict[k]:v for (k,v) in feeds.iteritems()}
        self.session.run(targets, feed_dict=feed_dict)


    # For pickling.
    def __getstate__(self):
        #self.solver.snapshot()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'T': self._T,
        }

    # For unpickling.
    def __setstate__(self, state):
        # TODO - finalize this once __init__ is finalized (setting dO and T)
        self.__init__(state['hyperparams'])

        # TODO: Load from snapshot

