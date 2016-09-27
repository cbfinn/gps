""" This file defines neural network cost function. """
import copy
import logging
import numpy as np
import tempfile
import uuid
from itertools import izip

import tensorflow as tf

from gps.utility.demo_utils import xu_to_sample_list, extract_demos
from gps.utility.general_utils import BatchSampler
from gps.algorithm.cost.config import COST_IOC_NN
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_JACOBIANS

LOGGER = logging.getLogger(__name__)


class CostIOCSupervised(CostIOCTF):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams):
        super(CostIOCSupervised, self).__init__(hyperparams)
        self.gt_cost = hyperparams['gt_cost']  # Ground truth cost
        self.gt_cost = self.gt_cost['type'](self.gt_cost)

        self.eval_gt = hyperparams.get('eval_gt', False)
        self.multi_objective_wt = hyperparams.get('multi_objective', 0.0)

        self.update_after = hyperparams.get('update_after', 0)

        self.demo_agent = hyperparams['agent']  # Required for sample packing
        self.demo_agent = self.demo_agent['type'](self.demo_agent)
        self.weights_dir = hyperparams['weight_dir']

        demo_file, traj_file = hyperparams['demo_file'], hyperparams.get('traj_samples', [])
        X, U, tX, tU = self.extract_supervised_data(demo_file, traj_file)
        self.init_supervised(X, U, None, tX, tU)

    def extract_supervised_data(self, demo_file, traj_files):
        X, U, O, cond = extract_demos(demo_file)

        import pickle

        for traj_file in traj_files:
            with open(traj_file, 'r') as f:
                sample_lists = pickle.load(f)
                for sample_list in sample_lists:
                    X = np.r_[sample_list.get_X(), X]
                    U = np.r_[sample_list.get_U(), U]
                    # O = np.r_[sample_list.get_obs(), O]
        n_test = 5
        testX = X[-n_test:]
        testU = U[-n_test:]
        X = X[:-n_test]
        U = U[:-n_test]
        return X, U, testX, testU

    #TODO:  move this to a supervised cost
    def init_supervised(self, sampleU, sampleX, sampleO, testX, testU, heartbeat=100):
        demo_torque_norm = np.sum(sampleU **2, axis=2, keepdims=True)
        sample_torque_norm = np.sum(testU **2, axis=2, keepdims=True)

        num_samp = sampleU.shape[0]
        sample_costs = []
        sample_list = xu_to_sample_list(self.demo_agent, sampleX, sampleU)
        for n in range(num_samp):
            l, _, _, _, _, _ = self.gt_cost.eval(sample_list[n])
            sample_costs.append(l)
        sample_costs = np.array(sample_costs)
        sample_costs = np.expand_dims(sample_costs, -1)

        sampler = BatchSampler([sampleO, sample_torque_norm, sample_costs])

        for i, s_batch in enumerate(sampler.with_replacement(batch_size=5)):
            loss, grad = self.run([self.sup_loss, self.sup_optimizer],
                                  sup_obs=s_batch[0],
                                  sup_torque_norm=s_batch[1],
                                  sup_cost_labels = s_batch[2])
            if i%200 == 0:
                LOGGER.debug("Iteration %d loss: %f", i, loss)

            if i > self._hyperparams['init_iterations']:
                break
