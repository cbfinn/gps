""" This file defines the iLQG-based trajectory optimization method. """
import logging

import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.sample.sample_list import SampleList
from gps.utility import ColorLogger
from gps.utility.general_utils import compute_distance
from gps.utility.demo_utils import extract_samples, get_target_end_effector
from gps.algorithm.traj_opt.traj_opt_utils import traj_distr_kl

LOGGER = ColorLogger(__name__)


class AlgorithmTrajOpt(Algorithm):
    """ Sample-based trajectory optimization. """
    def __init__(self, hyperparams):
        Algorithm.__init__(self, hyperparams)

    def iteration(self, sample_lists):
        """
        Run iteration of LQR.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        if self._hyperparams['ioc']:
            self.N = sum(len(self.sample_list[i]) for i in self.sample_list.keys())
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            if self._hyperparams['ioc']:
                prev_samples = self.sample_list[m].get_samples()
                prev_samples.extend(sample_lists[m].get_samples())
                self.sample_list[m] = SampleList(prev_samples)
                self.N += len(sample_lists[m])

        # Update dynamics model using all samples.
        self._update_dynamics()

        # Update the cost during learning if we use IOC.
        if self._hyperparams['ioc']:
            if self._hyperparams['ioc_maxent_iter'] == -1 or self.iteration_count < self._hyperparams['ioc_maxent_iter']:
                self._update_cost()

        self._update_step_size()  # KL Divergence step size.

        # Run inner loop to compute new policies.
        for _ in range(self._hyperparams['inner_iterations']):
            self._update_trajectories()

        # Computing KL-divergence between sample distribution and demo distribution
        itr = self.iteration_count
        if self._hyperparams['ioc']:
            for i in xrange(self.M):
                mu, sigma = self.traj_opt.forward(self.traj_distr[itr][i], self.traj_info[itr][i])
                # KL divergence between current traj. distribution and gt distribution
                if len(self.demo_traj) != 0:
                    self.kl_div[itr].append(traj_distr_kl(mu, sigma, self.traj_distr[itr][i], self.demo_traj[0])) # Assuming Md == 1
                else:
                    self.kl_div[itr].append(-999)

        if 'target_end_effector' in self._hyperparams:
            for i in xrange(self.M):
                target_position = get_target_end_effector(self._hyperparams, i)
                dists = compute_distance(target_position, sample_lists[i])
                self.dists_to_target[itr].append(sum(dists) / sample_lists[i].num_samples())
        self._advance_iteration_variables()

    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt.estimate_cost(
            self.prev[m].traj_distr, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = self.traj_opt.estimate_cost(
            self.cur[m].traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        if self._hyperparams['ioc']:
          # use prevous cost to estimate cost of current traj distr.
          self._eval_cost(m, prev_cost=True)
          new_actual_laplace_obj = self.traj_opt.estimate_cost(
              self.cur[m].traj_distr, self.cur[m].prevcost_traj_info
          )
        else:
          new_actual_laplace_obj = self.traj_opt.estimate_cost(
              self.cur[m].traj_distr, self.cur[m].traj_info
          )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        if(self._hyperparams['ioc']):
            new_mc_obj = np.mean(np.sum(self.cur[m].prevcost_cs, axis=1), axis=0)
        else:
            new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        # TODO generate synethic samples here if desired? (or somewhere else)?
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if self._hyperparams['ioc_maxent_iter'] == -1 or self.iteration_count < self._hyperparams['ioc_maxent_iter']:
            multiplier = self._hyperparams["max_ent_traj"]
        else:
            multiplier = 0.0
        fCm, fcv = traj_info.Cm / (eta + multiplier), traj_info.cv / (eta + multiplier)
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k

        # Add in the trajectory divergence term.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += eta / (eta + multiplier) * np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += eta / (eta + multiplier) * np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        return fCm, fcv
