""" This file defines neural network cost function. """
import copy
import logging
import numpy as np
import tempfile
import uuid
from itertools import izip

import tensorflow as tf
from tf_cost_utils import jacobian, multimodal_nn_cost_net_tf

from gps.utility.general_utils import BatchSampler
from gps.algorithm.cost.config import COST_IOC_NN
from gps.algorithm.cost.cost import Cost
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES,\
    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_JACOBIANS

LOGGER = logging.getLogger(__name__)


class CostIOCVisionTF(Cost):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_IOC_NN) # Set this up in the config?
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._T = self._hyperparams['T']
        self.use_jacobian = self._hyperparams['use_jacobian']

        self.demo_batch_size = self._hyperparams['demo_batch_size']
        self.sample_batch_size = self._hyperparams['sample_batch_size']

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._init_solver()

    def copy(self):
        new_cost = CostIOCVisionTF(self._hyperparams)
        with tempfile.NamedTemporaryFile('w+b', suffix='.wts', delete=True) as f:
            self.save_model(f.name)
            f.seek(0)
            new_cost.restore_model(f.name)
        return new_cost

    def compute_lx_lxx(self, obs, dX):
        T, dO = obs.shape
        lx = np.zeros((T, dX))
        lxx = np.zeros((T, dX, dX))
        for t in range(T):
            lx[t], lxx[t] = self.run([self.dldx, self.dldxx], test_obs_single=obs[t])
        return lx, lxx

    def get_features(self, obs):
        """ Get the image features corresponding to the rgb image.
        """
        return self.run([self.outputs['test_feat']], test_obs=obs)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
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
        l[:] = np.squeeze(self.run([self.outputs['test_loss']], test_obs=obs, test_torque_norm=tq_norm)[0])
        lx, lxx = self.compute_lx_lxx(obs, dX)

        if self._hyperparams['learn_wu']:
            raise NotImplementedError()
        else:
            wu_mult = 1.0
        lu = wu_mult * self._hyperparams['wu'] * sample_u
        luu = wu_mult * np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])

        import pdb; pdb.set_trace()
        if self.use_jacobian and (END_EFFECTOR_POINT_JACOBIANS in sample._data):
            jnt_idx = sample.agent.get_idx_x(JOINT_ANGLES)
            vel_idx = sample.agent.get_idx_x(JOINT_VELOCITIES)

            jx = sample.get(END_EFFECTOR_POINT_JACOBIANS)
            # TODO - support end_effector_points_no_target here.
            import pdb; pdb.set_trace()
            dl_dee = sample.agent.unpack_data_x(lx, [END_EFFECTOR_POINTS])
            dl_dev = sample.agent.unpack_data_x(lx, [END_EFFECTOR_POINT_VELOCITIES])

            for t in xrange(T):
                lx[t, jnt_idx] += jx[t].T.dot(dl_dee[t])
                lx[t, vel_idx] += jx[t].T.dot(dl_dev[t])

        return l, lx, lu, lxx, luu, lux

    def update(self, demoU, demoO, d_log_iw, sampleU, sampleO, s_log_iw, itr=-1):
        """
        Learn cost function with generic function representation.
        Args:
            demoU: the actions of demonstrations.
            demoO: the observations of demonstrations.
            d_log_iw: log importance weights for demos.
            sampleU: the actions of samples.
            sampleO: the observations of samples.
            s_log_iw: log importance weights for samples.
        """
        print 'Updating cost'
        demo_torque_norm = np.sum(demoU **2, axis=2, keepdims=True)
        sample_torque_norm = np.sum(sampleU **2, axis=2, keepdims=True)

        num_samp = sampleU.shape[0]
        s_log_iw = s_log_iw[-num_samp:,:]
        d_sampler = BatchSampler([demoO, demo_torque_norm, d_log_iw])
        s_sampler = BatchSampler([sampleO, sample_torque_norm, s_log_iw])

        demo_batch = self._hyperparams['demo_batch_size']
        samp_batch = self._hyperparams['sample_batch_size']
        # TODO - make sure this is on GPU.
        for i, (d_batch, s_batch) in enumerate(
                izip(d_sampler.with_replacement(batch_size=self.demo_batch_size), \
                    s_sampler.with_replacement(batch_size=self.sample_batch_size))):
            ioc_loss, grad = self.run([self.ioc_loss, self.ioc_optimizer],
                                      demo_obs=d_batch[0],
                                      demo_torque_norm=d_batch[1],
                                      demo_iw = d_batch[2],
                                      sample_obs = s_batch[0],
                                      sample_torque_norm = s_batch[1],
                                      sample_iw = s_batch[2])
            if i % 200 == 0:
                LOGGER.debug("Iteration %d loss: %f", i, ioc_loss)

            if i > self._hyperparams['iterations']:
                break


    def _init_solver(self, sample_batch_size=None):
        """ Helper method to initialize the solver. """

        # Pass in net parameter by protostring (could add option to input prototxt file).
        network_arch_params = self._hyperparams['network_arch_params']

        network_arch_params['demo_batch_size'] = self._hyperparams['demo_batch_size']
        if sample_batch_size is None:
            network_arch_params['sample_batch_size'] = self._hyperparams['sample_batch_size']
        else:
            network_arch_params['sample_batch_size'] = sample_batch_size
        network_arch_params['T'] = self._T
        network_arch_params['ioc_loss'] = self._hyperparams['ioc_loss']
        #network_arch_params['Nq'] = self._iteration_count
        network_arch_params['smooth_reg_weight'] = self._hyperparams['smooth_reg_weight']
        network_arch_params['mono_reg_weight'] = self._hyperparams['mono_reg_weight']
        network_arch_params['gp_reg_weight'] = self._hyperparams['gp_reg_weight']
        network_arch_params['learn_wu'] = self._hyperparams['learn_wu']


        x_idx, img_idx, i = [], [], 0
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                img_idx = img_idx + list(range(i, i+dim))
            else:
                x_idx = x_idx + list(range(i, i+dim))
            i += dim
        network_arch_params['dim_input'] = len(img_idx) + len(x_idx)
        self._dO = network_arch_params['dim_input']

        network_arch_params['x_idx'] = x_idx
        network_arch_params['img_idx'] = img_idx
        inputs, outputs = multimodal_nn_cost_net_tf(**network_arch_params)
        self.outputs = outputs

        self.input_dict = inputs
        self.sup_loss = outputs['sup_loss']
        self.ioc_loss = outputs['ioc_loss']

        optimizer = tf.train.AdamOptimizer(learning_rate=self._hyperparams['lr'])
        self.ioc_optimizer = optimizer.minimize(self.ioc_loss)
        self.sup_optimizer = optimizer.minimize(self.sup_loss)

        # Set up gradients
        l_single, obs_single = outputs['test_loss_single'], inputs['test_obs_single']
        # NOTE - we are assuming that the features here are the same
        # as the state X
        feat_single = outputs['test_feat_single']

        self.dldx =  tf.gradients(l_single, feat_single)[0]
        self.dldxx = jacobian(self.dldx, feat_single)

        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def run(self, targets, **feeds):
        with self.graph.as_default():
            feed_dict = {self.input_dict[k]:v for (k,v) in feeds.iteritems()}
            result = self.session.run(targets, feed_dict=feed_dict)
        return result

    def save_model(self, fname):
        LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.session, fname)

    def restore_model(self, fname):
        self.saver.restore(self.session, fname)
        LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        checkpoint_fname = self._hyperparams['weights_file_prefix']+str(uuid.uuid4())+'.wts'
        self.save_model(checkpoint_fname)
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'T': self._T,
            'checkpoint_file': checkpoint_fname,
        }

    # For unpickling.
    def __setstate__(self, state):
        # TODO - finalize this once __init__ is finalized (setting dO and T)
        self.__init__(state['hyperparams'])
        self.restore_model(state['checkpoint_file'])

