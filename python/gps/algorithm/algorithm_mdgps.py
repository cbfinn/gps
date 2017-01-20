""" This file defines the MD-based GPS algorithm. """
import copy
import logging
import pickle

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList
from gps.utility import ColorLogger
from gps.utility.demo_utils import get_target_end_effector
from gps.utility.general_utils import Timer, compute_distance

LOGGER = ColorLogger(__name__)


class AlgorithmMDGPS(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MDGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        with Timer('build policy prior'):
            policy_prior = self._hyperparams['policy_prior']
            for m in range(self.M):
                self.cur[m].pol_info = PolicyInfo(self._hyperparams)
                self.cur[m].pol_info.policy_prior = \
                        policy_prior['type'](policy_prior)

        with Timer('build policy opt'):
            self.policy_opt = self._hyperparams['policy_opt']['type'](
                self._hyperparams['policy_opt'], self.dO, self.dU
            )

        with Timer('init cost params'):
            # initialize cost params
            if self._hyperparams['init_cost_params']:
                with open(self._hyperparams['init_cost_params'], 'r') as f:
                    init_algorithm = pickle.load(f)
                conv_params = init_algorithm.policy_opt.policy.get_copy_params()
                self.cost.set_vision_params(conv_params)

            if self._hyperparams['ioc'] and 'get_vision_params' in dir(self.cost):
                # Make cost and policy conv params consistent here.
                self.policy_opt.policy.set_copy_params(conv_params)

    def iteration(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        itr = self.iteration_count

        with Timer('compute_dist_to_target'):
            # Store the samples.
            if self._hyperparams['ioc']:
                self.N = sum(len(self.sample_list[i]) for i in self.sample_list.keys())
                self.num_samples = [len(self.sample_list[i]) for i in self.sample_list.keys()]
            for m in range(self.M):
                self.cur[m].sample_list = sample_lists[m]
                if self._hyperparams['ioc']:
                    prev_samples = self.sample_list[m].get_samples()
                    prev_samples.extend(sample_lists[m].get_samples())
                    self.sample_list[m] = SampleList(prev_samples)
                    self.N += len(sample_lists[m])
                # Compute mean distance to target. For peg experiment only.
                if 'target_end_effector' in self._hyperparams:
                    for i in xrange(self.M):
                        target_position = get_target_end_effector(self._hyperparams, i)
                        dists = compute_distance(target_position, sample_lists[i])
                        self.dists_to_target[itr].append(sum(dists) / sample_lists[i].num_samples())

        # On the first iteration we need to make sure that the policy somewhat
        # matches the init controller. Otherwise the LQR backpass starts with
        # a bad linearization, and things don't work out well.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            # Only update fc layers.
            with Timer('copy_policy - UpdatePolicy'):
                self._update_policy(fc_only=True)

        # Update dynamics linearizations.
        with Timer('UpdateDynamics'):
            self._update_dynamics()

        if self._hyperparams['ioc'] and (self._hyperparams['ioc_maxent_iter'] == -1 or \
                                        itr < self._hyperparams['ioc_maxent_iter']):
                with Timer('UpdateCost'):
                    self._update_cost()

        # Update policy linearizations.
        for m in range(self.M):
            with Timer('EvalCost'):
                self._eval_cost(m)
            with Timer('PolicyFit'):
                self._update_policy_fit(m)

        # C-step
        if self.iteration_count > 0:
            try:
                with Timer('stepadjust'):
                    self._stepadjust()
            except OverflowError:
                import pdb; pdb.set_trace()
        with Timer('UpdateTrajectories'):
            self._update_trajectories()

        # S-step
        if self._hyperparams['ioc'] and 'get_vision_params' in dir(self.cost):
            # copy conv layers from cost to policy here.
            conv_params = self.cost.get_vision_params()
            self.policy_opt.policy.set_copy_params(conv_params)

        with Timer('UpdatePolicy'):
            if self._hyperparams['ioc']:
                self._update_policy(fc_only=True)
            else:
                self._update_policy(fc_only=False)


        # Prepare for next iteration
        with Timer('AdvanceIteration'):
            self._advance_iteration_variables()

    def _update_policy(self, fc_only=False):
        """ Compute the new policy
        Args:
            fc_only: whether or not to only train the fc layers (no e2e)
        """
        LOGGER.info('Updating policy.')
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
                wt[:, t].fill(pol_info.pol_wt[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt, fc_only=fc_only)

    def _update_policy_fit(self, m):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
            init: Whether this is the initial fitting of the policy.
        """
        LOGGER.info('Updating policy fit.')
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_X()
        obs = samples.get_obs().copy()
        with Timer('compute probs'):
            pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
            pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        with Timer('update prior'):
            policy_prior = pol_info.policy_prior
            samples = SampleList(self.cur[m].sample_list)
            mode = self._hyperparams['policy_sample_mode']
            policy_prior.update(samples, self.policy_opt, mode)

        with Timer('compute posterior'):
            # Fit linearization and store in pol_info.
            pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                    policy_prior.fit(X, pol_mu, pol_sig)
            for t in range(T):
                pol_info.chol_pol_S[t, :, :] = \
                        sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        """
        Algorithm._advance_iteration_variables(self)
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = \
                    self.prev[m].traj_info.last_kl_step
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def _stepadjust(self):
        """
        Calculate new step sizes. This version uses the same step size
        for all conditions.
        """
        LOGGER.info('Adjusting step')
        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev) # May be different in future.
        prev_laplace = np.arange(prev_M).astype(np.float32)
        prev_mc = np.arange(prev_M).astype(np.float32)
        prev_predicted = np.arange(prev_M).astype(np.float32)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            try:
                prev_laplace[m] = self.traj_opt.estimate_cost(
                        prev_nn, self.prev[m].traj_info
                ).sum()
            except OverflowError:
                import pdb; pdb.set_trace()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = self.traj_opt.estimate_cost(
                    prev_lg, self.prev[m].traj_info
            ).sum()

        # Compute current cost.
        cur_laplace = np.arange(self.M).astype(np.float32)
        cur_mc = np.arange(self.M).astype(np.float32)
        for m in range(self.M):
            if self._hyperparams['ioc']:
                self._eval_cost(m, prev_cost=True)
                traj_info = self.cur[m].prevcost_traj_info
            else:
                traj_info = self.cur[m].traj_info
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            try:
                cur_laplace[m] = self.traj_opt.estimate_cost(
                        cur_nn, traj_info
                ).sum()
            except OverflowError:
                import pdb; pdb.set_trace()
            if self._hyperparams['ioc']:
                cur_mc[m] = self.cur[m].prevcost_cs.mean(axis=0).sum()
            else:
                cur_mc[m] = self.cur[m].cs.mean(axis=0).sum()

        # Compute predicted and actual improvement.
        prev_laplace = prev_laplace.mean()
        prev_mc = prev_mc.mean()
        prev_predicted = prev_predicted.mean()
        cur_laplace = cur_laplace.mean()
        cur_mc = cur_mc.mean()
        if self._hyperparams['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        else:
            raise NotImplementedError()
        LOGGER.debug('Previous cost: Laplace: %f, MC: %f',
                     prev_laplace, prev_mc)
        LOGGER.debug('Predicted cost: Laplace: %f', prev_predicted)
        LOGGER.debug('Actual cost: Laplace: %f, MC: %f',
                     cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if self._hyperparams['ioc_maxent_iter'] == -1 or self.iteration_count < self._hyperparams['ioc_maxent_iter']:
            multiplier = self._hyperparams['max_ent_traj']
        else:
            multiplier = 0.0
        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB, kB = pol_info.pol_K[t, :, :], pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv
