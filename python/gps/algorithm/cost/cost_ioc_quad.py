""" This file defines quadratic cost function. """
import copy
import logging
import numpy as np
import tempfile

import caffe
from caffe.proto.caffe_pb2 import SolverParameter, TRAIN, TEST
from google.protobuf.text_format import MessageToString

from gps.algorithm.cost.config import COST_IOC_QUADRATIC
from gps.algorithm.cost.cost import Cost

LOGGER = logging.getLogger(__name__)


class CostIOCQuadratic(Cost):
    """ Set up weighted quadratic norm loss with learned parameters. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_IOC_QUADRATIC) # Set this up in the config?
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
        self.caffe_iter = 0

        self._init_solver()

    def copy(self):
      new_cost = CostIOCQuadratic(self._hyperparams)
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
      return new_cost


    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        # TODO - right now, we're going to assume that Obs = X
        T = sample.T
        obs = sample.get_obs()
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
        l = 0.5*self.solver.test_nets[0].forward().values()[0][0].reshape(T)

        # Get weights array from caffe (M in the old codebase)
        params = self.solver.test_nets[0].params.values()
        weighted_array = np.c_[params[0][0].data, np.array([params[0][1].data]).T]
        A = weighted_array.T.dot(weighted_array)

        sample_u = sample.get_U()
        l += 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
        lu = self._hyperparams['wu'] * sample_u
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])

        dldx = A.dot(np.vstack((obs.T, np.ones((1, T))))) # Assuming A is a (dX + 1) x (dO + 1) matrix
        lx = dldx.T[:, :dO]
        for t in xrange(T):
          lxx[t, :, :] = A[:dO,:dO]
        return l, lx, lu, lxx, luu, lux


    # TODO - we might want to make the demos and samples input as SampleList objects, rather than arrays.
    # TODO - also we might want to exclude demoU/sampleU since we generally don't use them
    # TODO - change name of dlogis/slogis to d_log_iw and s_log_iw.
    def update(self, demoU, demoO, dlogis, sampleU, sampleO, slogis, itr=None):
        """
        Learn cost function with generic function representation.
        Args:
            demoU: the actions of demonstrations.
            demoO: the observations of demonstrations.
            dlogis: importance weights for demos.
            sampleU: the actions of samples.
            sampleO: the observations of samples.
            slogis: importance weights for samples.
        """
        Nd = demoO.shape[0]
        Ns = sampleO.shape[0]
        blob_names = self.solver.net.blobs.keys()
        dbatches_per_epoch = np.floor(Nd / self.demo_batch_size)
        sbatches_per_epoch = np.floor(Ns / self.sample_batch_size)

        demo_idx = range(Nd)
        sample_idx = range(Ns)
        average_loss = 0

        for i in range(self._hyperparams['iterations']):
            # Randomly sample batches
            np.random.shuffle(demo_idx)
            np.random.shuffle(sample_idx)

            # Load in data for this batch.
            d_start_idx = int(i * self.demo_batch_size %
                              (dbatches_per_epoch * self.demo_batch_size))
            s_start_idx = int(i * self.sample_batch_size %
                              (sbatches_per_epoch * self.sample_batch_size))
            d_idx_i = demo_idx[d_start_idx:d_start_idx+self.demo_batch_size]
            s_idx_i = sample_idx[s_start_idx:s_start_idx+self.sample_batch_size]
            self.solver.net.blobs[blob_names[0]].data[:] = demoO[d_idx_i]
            self.solver.net.blobs[blob_names[1]].data[:] = dlogis[d_idx_i]
            self.solver.net.blobs[blob_names[2]].data[:] = sampleO[s_idx_i]
            self.solver.net.blobs[blob_names[3]].data[:] = slogis[s_idx_i]
            #if i == 5:
            #  import pdb; pdb.set_trace()
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

    def _init_solver(self):
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
        network_arch_params['sample_batch_size'] = self._hyperparams['sample_batch_size']
        network_arch_params['T'] = self._T
        network_arch_params['phase'] = TRAIN
        network_arch_params['ioc_loss'] = self._hyperparams['ioc_loss']
        solver_param.train_net_param.CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        # For running forward in python.
        network_arch_params['phase'] = TEST
        solver_param.test_net_param.add().CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        # These are required by Caffe to be set, but not used.
        solver_param.test_iter.append(1)
        # solver_param.test_iter.append(1)
        solver_param.test_interval = 1000000

        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write(MessageToString(solver_param))
        f.close()
        self.solver = caffe.get_solver(f.name)

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
