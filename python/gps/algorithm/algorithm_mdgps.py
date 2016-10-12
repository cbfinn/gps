""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList
from gps.algorithm.traj_opt.traj_opt_utils import traj_distr_kl
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE

LOGGER = logging.getLogger(__name__)


class AlgorithmMDGPS(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MDGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        policy_prior = self._hyperparams['policy_prior']
        for m in range(self.M):
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

        self.num_policies = self._hyperparams['num_policies']
        if not self._hyperparams['multiple_policy']:
            self.policy_opt = self._hyperparams['policy_opt']['type'](
                self._hyperparams['policy_opt'], self.dO, self.dU
            )
            self.num_policies = 1
        else:
            self.policy_opts = [self._hyperparams['policy_opt'][i]['type'](
                self._hyperparams['policy_opt'][i], self.dO, self.dU
            ) for i in xrange(self.num_policies)]

    def iteration(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples.
        itr = self.iteration_count
        self.N = sum(len(self.sample_list[i]) for i in self.sample_list.keys())
        self.num_samples = [len(self.sample_list[i]) for i in self.sample_list.keys()]
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            prev_samples = self.sample_list[m].get_samples()
            prev_samples.extend(sample_lists[m].get_samples())
            self.sample_list[m] = SampleList(prev_samples)
            self.N += len(sample_lists[m])
            # Compute mean distance to target. For peg experiment only.
            if 'target_end_effector' in self._hyperparams:
                if type(self._hyperparams['target_end_effector']) is list:
                    target_position = self._hyperparams['target_end_effector'][m][:3]
                else:
                    target_position = self._hyperparams['target_end_effector'][:3]
                cur_samples = sample_lists[m].get_samples()

                sample_end_effectors = [cur_samples[i].get(END_EFFECTOR_POINTS) for i in xrange(len(cur_samples))]
                dists = [np.nanmin(np.sqrt(np.sum((sample_end_effectors[i][:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0) \
                         for i in xrange(len(cur_samples))]
                self.dists_to_target[itr].append(sum(dists) / len(cur_samples))

        # Comment this when use random policy initialization and add after line 78
        if self.iteration_count == 0 and self._hyperparams['policy_eval']:
            self.policy_opts[self.iteration_count] = self.policy_opt.copy()

        # On the first iteration we need to make sure that the policy somewhat
        # matches the init controller. Otherwise the LQR backpass starts with
        # a bad linearization, and things don't work out well.
        elif self.iteration_count == 0 and not self._hyperparams['init_demo_policy']:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            # Only update fc layers.
            self._update_policy(fc_only=True)

        # Update dynamics linearizations.
        self._update_dynamics()

        # Move this after line 78 if using random initializarion.
        if self._hyperparams['ioc'] and self._hyperparams['init_demo_policy']:
            raise ValueError("haven't supported this with vision and dynamics fit has moved.")
        if not self._hyperparams['global_cost']:
            raise NotImplementedError('no support for multiple costs with vision.')

        if self._hyperparams['ioc'] and not self._hyperparams['init_demo_policy']:
            if self._hyperparams['ioc_maxent_iter'] == -1 or itr < self._hyperparams['ioc_maxent_iter']:
                # TODO - copy conv layers from policy to cost here, at all iterations.
                conv_params = self.policy_opt.policy.get_copy_params()
                self.cost.set_vision_params(conv_params)

                self._update_cost()
                for m in range(self.M):
                    for sample in self.cur[m].sample_list:
                        sample.update_features(self.cost) # assumes a single cost.
                if self.cur[0].traj_info.dynamics.prior._max_samples > len(self.cur[0].sample_list):
                    print LOGGER.warn('refitting dynamics -- updating prior with the same set of samples')
                self._update_dynamics()  # recompute dynamics with new state space.

        # Update policy linearizations.
        for m in range(self.M):
            self._eval_cost(m)
            self._update_policy_fit(m)

        # C-step
        if self.iteration_count > 0:
            try:
                self._stepadjust()
            except OverflowError:
                import pdb; pdb.set_trace()
        self._update_trajectories()

        # S-step
        # TODO - copy conv layers from cost to policy here.
        if self._hyperparams['ioc']: # TODO and if using vision
            conv_params = self.cost.get_vision_params()
            self.policy_opt.policy.set_copy_params(conv_params)
        self._update_policy()

        # Computing KL-divergence between sample distribution and demo distribution
        #if self._hyperparams['ioc'] and not self._hyperparams['learning_from_prior']:
        #    for i in xrange(self.M):
        #        mu, sigma = self.traj_opt.forward(self.traj_distr[itr][i], self.traj_info[itr][i])
        #        # KL divergence between current traj. distribution and gt distribution
        #        self.kl_div[itr].append(traj_distr_kl(mu, sigma, self.traj_distr[itr][i], self.demo_traj[0])) # Assuming Md == 1

        # Prepare for next iteration
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
        if not self._hyperparams['multiple_policy']:
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
        else:
            for i in range(self.num_policies):
                for m in range(self.M / self.num_policies * i, self.M / self.num_policies * (i + 1)):
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
                        for j in range(N):
                            mu[j, t, :] = (traj.K[t, :, :].dot(X[j, t, :]) + traj.k[t, :])
                        wt[:, t].fill(pol_info.pol_wt[t])
                    tgt_mu = np.concatenate((tgt_mu, mu))
                    tgt_prc = np.concatenate((tgt_prc, prc))
                    tgt_wt = np.concatenate((tgt_wt, wt))
                    obs_data = np.concatenate((obs_data, samples.get_obs()))
                    self.policy_opts[i].update(obs_data, tgt_mu, tgt_prc, tgt_wt, fc_only=fc_only)

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
        if not self._hyperparams['multiple_policy']:
            pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        else:
            pol_mu, pol_sig = self.policy_opts[m/self.num_policies].prob(obs)[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[m].sample_list)
        mode = self._hyperparams['policy_sample_mode']
        if not self._hyperparams['multiple_policy']:
            policy_prior.update(samples, self.policy_opt, mode)
        else:
            policy_prior.update(samples, self.policy_opts[m/self.num_policies], mode)

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

    def _update_cost(self):
        """ Update the cost objective in each iteration. """

        # Estimate the importance weights for fusion distributions.
        # For the ICML version of the objective, this uses the dynamics to fit a controller to the demo.

        # TODO - fusion distribution for importance weights is incorrect with changing state space (IOC w/ vision).
        # Shouldn't be catastropic, importance weights are still mostly right without fusion distribution.
        # Correct thing to do is to get rgb image from the observation, and calculate the corresponding X for each controller.

        # TODO - What is catastrophic is if an empirical demo controller is fit using dynamics fit to
        # a different state space. To handle, this the demoX are recalculated every iteration, BUT we need
        # the policy conv layers to be copied over to the cost.
        demos_logiw, samples_logiw = self.importance_weights()

        # Update the learned cost
        # Transform all the dictionaries to arrays
        M = len(self.prev)
        Md = self._hyperparams['demo_M']
        sampleU_arr = np.vstack((self.sample_list[i].get_U() for i in xrange(M)))
        sampleO_arr = np.vstack((self.sample_list[i].get_obs() for i in xrange(M)))
        samples_logiw = {i: samples_logiw[i].reshape((-1, 1)) for i in xrange(M)}
        demos_logiw = {i: demos_logiw[i].reshape((-1, 1)) for i in xrange(Md)}
        demos_logiw_arr = np.hstack([demos_logiw[i] for i in xrange(Md)]).reshape((-1, 1))
        samples_logiw_arr = np.hstack([samples_logiw[i] for i in xrange(M)]).reshape((-1, 1))
        if not self._hyperparams['global_cost']:
            for i in xrange(M):
                self.cost[i].update(self.demoU, self.demoO, demos_logiw_arr, self.sample_list[i].get_U(),
                                self.sample_list[i].get_obs(), samples_logiw[i], itr=self.iteration_count)
        else:
            self.cost.update(self.demoU, self.demoO, demos_logiw_arr, sampleU_arr,
                                                        sampleO_arr, samples_logiw_arr, itr=self.iteration_count)

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if self._hyperparams['ioc_maxent_iter'] == -1 or self.iteration_count < self._hyperparams['ioc_maxent_iter']:
            multiplier = self._hyperparams['max_ent_traj']
        else:
            multiplier = 0.0
        pol_info = self.cur[m].pol_info
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
