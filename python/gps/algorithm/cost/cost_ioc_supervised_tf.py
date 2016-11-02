""" This file defines neural network cost function. """
import logging
import numpy as np
import pickle
from itertools import izip
import copy
import os


import tensorflow as tf

from gps.utility.demo_utils import xu_to_sample_list, extract_demos
from gps.utility.general_utils import BatchSampler
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF

LOGGER = logging.getLogger(__name__)


class CostIOCSupervised(CostIOCTF):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams, train_samples=None, test_samples=None, train_costs=None):
        super(CostIOCSupervised, self).__init__(hyperparams)
        self.gt_cost = hyperparams['gt_cost']  # Ground truth cost
        self.gt_cost = self.gt_cost['type'](self.gt_cost)

        self.eval_gt = hyperparams.get('eval_gt', False)
        self.multi_objective_wt = hyperparams.get('multi_objective', 0.0)
        self.finetune = hyperparams.get('finetune', False)
        self.multiobj = hyperparams.get('multiobj', False)

        self.update_after = hyperparams.get('update_after', 0)

        if hyperparams.get('agent', False):
            demo_agent = hyperparams['agent']  # Required for sample packing
            demo_agent = demo_agent['type'](demo_agent)
        # self.weights_dir = hyperparams['weight_dir']

        demo_file, traj_file = hyperparams['demo_file'], hyperparams.get('traj_samples', [])
        if hyperparams.get('agent', False):
            train_samples, test_samples, train_costs = self.extract_supervised_data(demo_agent, demo_file, traj_file) 
            self.init_supervised(train_samples, test_samples, train_costs)
        self.sup_samples = train_samples
        self.sup_costs = train_costs
        self.sup_test_samples = test_samples
        self.demo_agent = None  # don't pickle agent
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']

    def copy(self):
        new_cost = CostIOCSupervised(self._hyperparams, train_samples=copy.copy(self.sup_samples), \
                        test_samples=copy.copy(self.sup_test_samples), \
                        train_costs=copy.copy(self.sup_costs))
        with open('tmp.wts', 'w+b') as f:
            self.save_model(f.name)
            f.seek(0)
            new_cost.restore_model(f.name)
            os.remove(f.name+'.meta')
        os.remove('tmp.wts')
        return new_cost

    def update(self, demoU, demoO, d_log_iw, sampleU, sampleO, s_log_iw, itr=-1):
        if self.finetune:
            if self.multiobj:
                self.update_multiobj(demoU, demoO, d_log_iw, sampleU, sampleO, s_log_iw, self.sup_samples,
                                     self.sup_costs, itr=itr)
            else:
                return super(CostIOCSupervised, self).update(demoU, demoO, d_log_iw, sampleU, sampleO, s_log_iw, itr=itr)
        else:
            pass

    def extract_supervised_data(self, demo_agent, demo_file, traj_files):
        X, U, O, cond = extract_demos(demo_file)


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
        train_samples = xu_to_sample_list(demo_agent, X, U)
        test_samples = xu_to_sample_list(demo_agent, testX, testU)


        num_samp = U.shape[0]
        sample_costs = []
        for n in range(num_samp):
            l, _, _, _, _, _ = self.gt_cost.eval(train_samples[n])
            sample_costs.append(l)
        sample_costs = np.array(sample_costs)
        sample_costs = np.expand_dims(sample_costs, -1)

        return train_samples, test_samples, sample_costs

    def init_supervised(self, train_samples, test_samples, train_costs, heartbeat=100):
        sample_torque_norm = np.sum(self._hyperparams['wu']* (train_samples.get_U() **2), axis=2, keepdims=True)

        sampler = BatchSampler([train_samples.get_X(), sample_torque_norm, train_costs])
        batch_size = self._hyperparams['demo_batch_size']+self._hyperparams['sample_batch_size']

        for i, s_batch in enumerate(sampler.with_replacement(batch_size=batch_size)):
            loss, grad = self.run([self.sup_loss, self.sup_optimizer],
                                  sup_obs=s_batch[0],
                                  sup_torque_norm=s_batch[1],
                                  sup_cost_labels = s_batch[2])
            if i%heartbeat == 0:
                LOGGER.debug("Iteration %d loss: %f", i, loss)

            if i > self._hyperparams['init_iterations']:
                break


    def update_multiobj(self, demoU, demoO, d_log_iw, sampleU, sampleO, s_log_iw,
                        sup_samples, sup_cost_labels, itr=-1):
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
        demo_torque_norm = np.sum(self._hyperparams['wu']*(demoU **2), axis=2, keepdims=True)
        sample_torque_norm = np.sum(self._hyperparams['wu']*(sampleU **2), axis=2, keepdims=True)
        sup_torque_norm = np.sum(self._hyperparams['wu']*(sup_samples.get_U() **2), axis=2, keepdims=True)

        num_samp = sampleU.shape[0]
        s_log_iw = s_log_iw[-num_samp:,:]
        d_sampler = BatchSampler([demoO, demo_torque_norm, d_log_iw])
        s_sampler = BatchSampler([sampleO, sample_torque_norm, s_log_iw])
        sup_sampler = BatchSampler([sup_samples.get_X(), sup_torque_norm, sup_cost_labels])

        demo_batch = self._hyperparams['demo_batch_size']
        samp_batch = self._hyperparams['sample_batch_size']
        sup_batch_size = demo_batch+samp_batch
        for i, (d_batch, s_batch, sup_batch) in enumerate(
                izip(d_sampler.with_replacement(batch_size=demo_batch), s_sampler.with_replacement(batch_size=samp_batch),
                     sup_sampler.with_replacement(batch_size=sup_batch_size))):
            ioc_loss, grad = self.run([self.ioc_loss, self.ioc_optimizer],
                                      demo_obs=d_batch[0],
                                      demo_torque_norm=d_batch[1],
                                      demo_iw = d_batch[2],
                                      sample_obs = s_batch[0],
                                      sample_torque_norm = s_batch[1],
                                      sample_iw = s_batch[2],
                                      sup_obs=sup_batch[0],
                                      sup_torque_norm=sup_batch[1],
                                      sup_cost_labels = sup_batch[2])
            if i%200 == 0:
                LOGGER.debug("Iteration %d loss: %f", i, ioc_loss)

            if i > self._hyperparams['iterations']:
                break
