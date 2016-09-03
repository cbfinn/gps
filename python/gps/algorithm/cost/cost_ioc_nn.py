""" This file defines neural network cost function. """
import copy
import logging
import numpy as np
import tempfile

import caffe
from caffe.proto.caffe_pb2 import SolverParameter, TRAIN, TEST
from google.protobuf.text_format import MessageToString

from gps.algorithm.cost.config import COST_IOC_NN
from gps.algorithm.cost.cost import Cost

LOGGER = logging.getLogger(__name__)


class CostIOCNN(Cost):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_IOC_NN) # Set this up in the config?
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._dO = self._hyperparams['dO']
        self._T = self._hyperparams['T']

        # By default using caffe
        if self._hyperparams['use_gpu']:
            caffe.set_device(self._hyperparams['gpu_id'])
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.demo_batch_size = self._hyperparams['demo_batch_size']
        self.sample_batch_size = self._hyperparams['sample_batch_size']

        self._iteration_count = 1

        self._init_solver()

    def copy(self):
      new_cost = CostIOCNN(self._hyperparams)
      self.solver.snapshot()
      new_cost.caffe_iter = self.caffe_iter
      new_cost.solver.restore(
              self._hyperparams['weights_file_prefix'] + '_iter_' +
              str(self.caffe_iter) + '.solverstate'
      )
      new_cost.solver.test_nets[0].copy_from(
              self._hyperparams['weights_file_prefix'] + '_iter_' +
              str(self.caffe_iter) + '.caffemodel'
      )
      new_cost.solver.test_nets[1].copy_from(
              self._hyperparams['weights_file_prefix'] + '_iter_' +
              str(self.caffe_iter) + '.caffemodel'
      )
      return new_cost

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
      blob_names = self.solver.test_nets[1].blobs.keys()
      feat_shape = self.solver.test_nets[1].blobs[blob_names[-1]].data[:].shape
      dF = feat_shape[-1]
      out_diff = np.zeros((feat_shape))

      T, dX = obs.shape
      dfdx = np.zeros((T, dF, dX))

      for i in range(dF):
          self.solver.test_nets[1].blobs[blob_names[0]].data[:] = obs
          # run forward pass before running backward pass
          feat = self.solver.test_nets[1].forward().values()[0][0]
          # construct output diff
          out_diff = out_diff * 0.0
          out_diff[:, :, i] = 1.0
          top_diff = {blob_names[-1]: out_diff}
          dfdx[:, i, :] = self.solver.test_nets[1].backward(diffs=[blob_names[0]], **top_diff).values()[0]

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

        blob_names = self.solver.test_nets[0].blobs.keys()
        self.solver.test_nets[0].blobs[blob_names[0]].data[:] = obs
        self.solver.test_nets[0].blobs[blob_names[1]].data[:] = np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1, keepdims=True)
        l = 0.5*self.solver.test_nets[0].forward().values()[0][0].reshape(T)

        # Get weights array from caffe (M in the old codebase)
        params = self.solver.test_nets[0].params
        weighted_array = np.c_[params['Ax'][0].data, np.array([params['Ax'][1].data]).T]
        A = weighted_array.T.dot(weighted_array)

        # get intermediate features
        feat, dfdx = self.compute_dfdx(obs)

        #l += 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)  # already computed by the network
        if self._hyperparams['learn_wu']:
            wu_mult = params['all_u'][0].data[0][0]
        else:
            wu_mult = 1.0
        lu = wu_mult * self._hyperparams['wu'] * sample_u
        luu = wu_mult * np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])

        dldf = A.dot(np.vstack((feat.T, np.ones((1, T))))) # Assuming A is a (df + 1) x (df + 1) matrix
        dF = feat.shape[1]
        for t in xrange(T):
          lx[t, :] = dfdx[t, :, :].T.dot(dldf[:dF, t])
          lxx[t, :, :] = dfdx[t, :, :].T.dot(A[:dF,:dF]).dot(dfdx[t, :, :])
        return l, lx, lu, lxx, luu, lux

    # TODO - we might want to make the demos and samples input as SampleList objects, rather than arrays.
    # TODO - also we might want to exclude demoU/sampleU since we generally don't use them
    def update(self, demoU, demoX, demoO, d_log_iw, sampleU, sampleX, sampleO, s_log_iw):
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

        Nd = demoO.shape[0]
        Ns = sampleO.shape[0]

        if Ns <= self.sample_batch_size:
            sample_batch_size = Ns
            old_net = self.solver.net
            self._init_solver(sample_batch_size)
            self.solver.net.share_with(old_net)
        else:
            sample_batch_size = self.sample_batch_size

        blob_names = self.solver.net.blobs.keys()
        dbatches_per_epoch = np.floor(Nd / self.demo_batch_size)
        sbatches_per_epoch = np.floor(Ns / sample_batch_size)

        demo_idx = range(Nd)
        sample_idx = range(Ns)
        average_loss = 0

        # Compute the variance in each dimension of the observation.
        stacked_obs = np.vstack((demoO, sampleO))
        dO = demoO.shape[2]
        T = demoO.shape[1]
        var_obs = np.zeros((T, dO))
        for i in xrange(dO):
        	var_obs[:, i] = np.var(stacked_obs[:, :, i], axis=0)

        for i in range(self._hyperparams['iterations']):
          # Randomly sample batches
          np.random.shuffle(demo_idx)
          np.random.shuffle(sample_idx)

          # Load in data for this batch.
          d_start_idx = int(i * self.demo_batch_size %
              (dbatches_per_epoch * self.demo_batch_size))
          s_start_idx = int(i * sample_batch_size %
              (sbatches_per_epoch * sample_batch_size))
          d_idx_i = demo_idx[d_start_idx:d_start_idx+self.demo_batch_size]
          s_idx_i = sample_idx[s_start_idx:s_start_idx+sample_batch_size]
          self.solver.net.blobs[blob_names[0]].data[:] = demoO[d_idx_i]
          self.solver.net.blobs[blob_names[1]].data[:] = np.sum(self._hyperparams['wu']*demoU[d_idx_i]**2, axis=2, keepdims=True)
          self.solver.net.blobs[blob_names[2]].data[:] = d_log_iw[d_idx_i]
          self.solver.net.blobs[blob_names[3]].data[:] = sampleO[s_idx_i]
          self.solver.net.blobs[blob_names[4]].data[:] = np.sum(self._hyperparams['wu']*sampleU[s_idx_i]**2, axis=2, keepdims=True)
          self.solver.net.blobs[blob_names[5]].data[:] = s_log_iw[s_idx_i]
          self.solver.net.blobs[blob_names[6]].data[:] = var_obs
          self.solver.net.blobs[blob_names[7]].data[:] = np.vstack((demoO[d_idx_i], sampleO[s_idx_i]))
          self.solver.step(1)
          train_loss = self.solver.net.blobs[blob_names[-1]].data
          average_loss += train_loss
          if i % 500 == 0 and i != 0:
            LOGGER.debug('Caffe iteration %d, average loss %f',
                         i, average_loss / 500)
            average_loss = 0


        # Keep track of Caffe iterations for loading solver states.
        self.caffe_iter += self._hyperparams['iterations']
        self.solver.test_nets[0].share_with(self.solver.net)
        self.solver.test_nets[1].share_with(self.solver.net)
        # DEBUGGING
        #import pdb; pdb.set_trace()
        #debug=False
        #if debug:
        #    old_net = self.solver.net  # test_nets[0]
        #    self._hyperparams['ioc_loss'] = 'MPF'
        #    self._init_solver()
        #    self.solver.net.share_with(old_net)
        #    # TODO = also need to change algorithm._hyperparams['ioc'] to MPF
        # END DEBUGGING



    def _init_solver(self, sample_batch_size=None):
        """ Helper method to initialize the solver. """
        solver_param = SolverParameter()
        solver_param.display = 0  # Don't display anything.
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']
        solver_param.random_seed = self._hyperparams['random_seed']

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
        solver_param.train_net_param.CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        # For running forward in python.
        network_arch_params['phase'] = TEST
        solver_param.test_net_param.add().CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        network_arch_params['phase'] = 'forward_feat'
        solver_param.test_net_param.add().CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        # These are required by Caffe to be set, but not used.
        solver_param.test_iter.append(1)
        solver_param.test_iter.append(1)
        solver_param.test_interval = 1000000

        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write(MessageToString(solver_param))
        f.close()
        self.solver = caffe.get_solver(f.name)
        self.caffe_iter = 0

    # For pickling.
    def __getstate__(self):
        self.solver.snapshot()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'T': self._T,
            'caffe_iter': self.caffe_iter,
        }

    # For unpickling.
    def __setstate__(self, state):
        # TODO - finalize this once __init__ is finalized (setting dO and T)
        self.__init__(state['hyperparams'])
        self.caffe_iter = state['caffe_iter']
        self.solver.restore(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.solverstate'
        )
        self.solver.test_nets[0].copy_from(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.caffemodel'
        )
        self.solver.test_nets[1].copy_from(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.caffemodel'
        )



    # TODO - we might want to make the demos and samples input as SampleList objects, rather than arrays.
    # TODO - also we might want to exclude demoU/sampleU since we generally don't use them
    def update_supervised(self, sampleU, sampleX, sampleO, sample_cost):
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

        Ns = sampleO.shape[0]

        sample_batch_size = self.sample_batch_size + self.demo_batch_size

        blob_names = self.solver.net.blobs.keys()
        sbatches_per_epoch = np.floor(Ns / sample_batch_size)

        sample_idx = range(Ns)
        average_loss = 0

        for i in range(self._hyperparams['iterations']):
          # Randomly sample batches
          np.random.shuffle(sample_idx)

          # Load in data for this batch.
          s_start_idx = int(i * sample_batch_size %
              (sbatches_per_epoch * sample_batch_size))
          s_idx_i = sample_idx[s_start_idx:s_start_idx+sample_batch_size]
          self.solver.net.blobs[blob_names[0]].data[:] = sampleO[s_idx_i]
          self.solver.net.blobs[blob_names[1]].data[:] = np.sum(self._hyperparams['wu']*sampleU[s_idx_i]**2, axis=2, keepdims=True)
          self.solver.net.blobs[blob_names[2]].data[:] = sample_cost[s_idx_i]
          self.solver.step(1)
          train_loss = self.solver.net.blobs[blob_names[-1]].data
          average_loss += train_loss
          if i % 500 == 0 and i != 0:
            LOGGER.debug('Caffe iteration %d, average loss %f',
                         i, average_loss / 500)
            average_loss = 0


        # Keep track of Caffe iterations for loading solver states.
        self.caffe_iter += self._hyperparams['iterations']
        self.solver.test_nets[0].share_with(self.solver.net)
        self.solver.test_nets[1].share_with(self.solver.net)


