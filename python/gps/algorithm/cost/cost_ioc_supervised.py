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
from gps.algorithm.cost.cost_ioc_nn import CostIOCNN
from gps.utility.demo_utils import xu_to_sample_list, extract_demos

LOGGER = logging.getLogger(__name__)


class CostIOCSupervised(CostIOCNN):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams):
        super(CostIOCSupervised, self).__init__(hyperparams)
        self.gt_cost = hyperparams['gt_cost']  # Ground truth cost
        self.agent = hyperparams['agent']  # Required for sample packing
        if hyperparams.get('init_from_demos', False):
            self.init_supervised_demos(hyperparams['demo_file'])

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
        pass # Do nothing

    def init_supervised_demos(self, demo_file):
        X, U, O = extract_demos(demo_file)
        self.init_supervised(U, X, O)

    def init_supervised(self, sampleU, sampleX, sampleO):
        """
        """

        Ns = sampleO.shape[0]  # Num samples

        sample_batch_size = self.sample_batch_size + self.demo_batch_size

        blob_names = self.solver.net.blobs.keys()
        sbatches_per_epoch = np.floor(Ns / sample_batch_size)

        sample_idx = range(Ns)
        average_loss = 0

        sample_costs = []
        sample_list = xu_to_sample_list(self.agent, sampleX, sampleU)
        for n in range(Ns):
            l, _, _, _, _, _ = self.gt_cost.eval(sample_list[n])
            sample_costs.append(l)

        for i in range(self._hyperparams['iterations']):
          # Randomly sample batches
          np.random.shuffle(sample_idx)

          # Load in data for this batch.
          s_start_idx = int(i * sample_batch_size %
              (sbatches_per_epoch * sample_batch_size))
          s_idx_i = sample_idx[s_start_idx:s_start_idx+sample_batch_size]
          self.solver.net.blobs[blob_names[0]].data[:] = sampleO[s_idx_i]
          self.solver.net.blobs[blob_names[1]].data[:] = np.sum(self._hyperparams['wu']*sampleU[s_idx_i]**2, axis=2, keepdims=True)
          self.solver.net.blobs[blob_names[2]].data[:] = sample_costs[s_idx_i]
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

