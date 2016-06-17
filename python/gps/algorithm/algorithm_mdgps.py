""" This file defines the MD-based GPS algorithm. """
import copy
import logging
from IPython.core.debugger import Tracer; debug_here = Tracer()

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo, gauss_fit_joint_prior
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList


LOGGER = logging.getLogger(__name__)


class AlgorithmMDGPS(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    MD-based guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MDGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        for m in range(self.M):
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            policy_prior = self._hyperparams['policy_prior']
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )

        # extra stuff to pass parameters to sampling
        # TODO: this is sloppy, handle better
        if not self._hyperparams['agent_use_nn_policy']:
            del self._hyperparams['agent_use_nn_policy']

    def iteration(self, sample_lists):
        """
        Run iteration of BADMM-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        self._update_dynamics()  # Update dynamics model using all sample.
        self._update_policy_samples()  # Choose samples to use with the policy.
        self._update_step_size()  # KL Divergence step size (also fits policy).

        for m in range(self.M):
            self._update_policy_fit(m)  # Update policy priors.

        self._update_trajectories()
        self._update_policy(self.iteration_count, 1) # HACK: need to set inner_itr=1

        for m in range(self.M):
            self._update_policy_fit(m)  # Update policy priors.

        # TODO: not sure about this, copied from previous
        for m in range(self.M):
            kl_m = self._policy_kl(m)[0]
            self.cur[m].pol_info.prev_kl = kl_m

#        # Run inner loop to compute new policies.
#        for inner_itr in range(self._hyperparams['inner_iterations']):
#            #TODO: Could start from init controller.
#            if self.iteration_count > 0 or inner_itr > 0:
#                # Update the policy.
#                self._update_policy(self.iteration_count, inner_itr)
#            for m in range(self.M):
#                self._update_policy_fit(m)  # Update policy priors.
#
#            # TODO: not sure about this, copied from previous
#            if (self.iteration_count > 0 or inner_itr > 0) and \
#                    (inner_itr == self._hyperparams['inner_iterations'] - 1):
#                for m in range(self.M):
#                    kl_m = self._policy_kl(m)[0]
#                    self.cur[m].pol_info.prev_kl = kl_m
#
#            self._update_trajectories()

        self._advance_iteration_variables()

    def _update_policy_samples(self):
        """ Update the list of samples to use with the policy. """
        #TODO: Handle synthetic samples.
        max_policy_samples = self._hyperparams['max_policy_samples']
        if self._hyperparams['policy_sample_mode'] == 'add':
            for m in range(self.M):
                samples = self.cur[m].pol_info.policy_samples
                samples.extend(self.cur[m].sample_list)
                if len(samples) > max_policy_samples:
                    start = len(samples) - max_policy_samples
                    self.cur[m].pol_info.policy_samples = samples[start:]
        else:
            for m in range(self.M):
                self.cur[m].pol_info.policy_samples = self.cur[m].sample_list

    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._update_policy_fit(m, init=True)
            self._eval_cost(m)
            # Adjust step size relative to the previous iteration.
            if self.iteration_count > 0:
                self._stepadjust(m)

    def _update_policy(self, itr, inner_itr):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            traj, pol_info = self.cur[m].traj_distr, self.cur[m].pol_info
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
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt,
                               itr, inner_itr)

    def _update_policy_fit(self, m, init=False):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
            init: Whether this is the initial fitting of the policy.
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_X()
        pol_mu, pol_sig = self.policy_opt.prob(samples.get_obs().copy())[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig
        # Update policy prior.
        if init:
            self.cur[m].pol_info.policy_prior.update(
                samples, self.policy_opt,
                SampleList(self.cur[m].pol_info.policy_samples)
            )
        else:
            self.cur[m].pol_info.policy_prior.update(
                SampleList([]), self.policy_opt,
                SampleList(self.cur[m].pol_info.policy_samples)
            )
        # Collapse policy covariances. This is not really correct, but
        # it works fine so long as the policy covariance doesn't depend
        # on state.
        pol_sig = np.mean(pol_sig, axis=0)
        # Estimate the policy linearization at each time step.
        for t in range(T):
            # Assemble diagonal weights matrix and data.
            dwts = (1.0 / N) * np.ones(N)
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate((Ts, Ps), axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.cur[m].pol_info.policy_prior.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # On the first time step, always slightly regularize covariance.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8 * np.eye(dX)
            # Perform computation.
            pol_K, pol_k, pol_S = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0,
                                                        dwts, dX, dU, sig_reg)
            pol_S += pol_sig[t, :, :]
            pol_info.pol_K[t, :, :], pol_info.pol_k[t, :] = pol_K, pol_k
            pol_info.pol_S[t, :, :], pol_info.chol_pol_S[t, :, :] = \
                    pol_S, sp.linalg.cholesky(pol_S)

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

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # NOTE: we only copy in K/k/pol_covar, only things needed for cost
        prev_pol_info = self.prev[m].pol_info
        prev_nn = self.prev[m].traj_distr.nans_like()
        prev_nn.K = prev_pol_info.pol_K
        prev_nn.k = prev_pol_info.pol_k
        prev_nn.pol_covar = self.prev[m].traj_distr.pol_covar

        cur_pol_info = self.cur[m].pol_info
        cur_nn = self.cur[m].traj_distr.nans_like()
        cur_nn.K = cur_pol_info.pol_K
        cur_nn.k = cur_pol_info.pol_k
        cur_nn.pol_covar = self.cur[m].traj_distr.pol_covar

        cur_traj_distr = self.cur[m].traj_distr

        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        prev_laplace_cost = self.traj_opt.estimate_cost(
            prev_nn, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the prev samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_cost = self.traj_opt.estimate_cost(
            cur_traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        if (self._hyperparams['step_rule'] == 'classic'):
            new_actual_laplace_cost = self.traj_opt.estimate_cost(
                cur_traj_distr, self.cur[m].traj_info
            )
        elif (self._hyperparams['step_rule'] == 'global'):
            new_actual_laplace_cost = self.traj_opt.estimate_cost(
                cur_nn, self.cur[m].traj_info
            )
        else:
            raise NotImplementedError

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual costective values based on the samples.
        prev_mc_cost = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_cost = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, prev_mc_cost, new_mc_cost)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(prev_laplace_cost) - \
                np.sum(new_predicted_laplace_cost)
        actual_impr = np.sum(prev_laplace_cost) - \
                np.sum(new_actual_laplace_cost)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(prev_laplace_cost), prev_mc_cost)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_cost), new_mc_cost)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_cost), new_mc_cost)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def _policy_kl(self, m, prev=False):
        """
        Monte-Carlo estimate of KL divergence between policy and
        trajectory.
        """
        dU, T = self.dU, self.T
        if prev:
            traj, pol_info = self.prev[m].traj_distr, self.cur[m].pol_info
            samples = self.prev[m].sample_list
        else:
            traj, pol_info = self.cur[m].traj_distr, self.cur[m].pol_info
            samples = self.cur[m].sample_list
        N = len(samples)
        X, obs = samples.get_X(), samples.get_obs()
        kl, kl_m = np.zeros((N, T)), np.zeros(T)
        # Compute policy mean and covariance at each sample.
        pol_mu, _, pol_prec, pol_det_sigma = self.policy_opt.prob(obs.copy())
        # Compute KL divergence.
        for t in range(T):
            # Compute trajectory action at sample.
            traj_mu = np.zeros((N, dU))
            for i in range(N):
                traj_mu[i, :] = traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :]
            diff = pol_mu[:, t, :] - traj_mu
            tr_pp_ct = pol_prec[:, t, :, :] * traj.pol_covar[t, :, :]
            k_ln_det_ct = 0.5 * dU + np.sum(
                np.log(np.diag(traj.chol_pol_covar[t, :, :]))
            )
            ln_det_cp = np.log(pol_det_sigma[:, t])
            # IMPORTANT: Note that this assumes that pol_prec does not
            #            depend on state!!!!
            #            (Only the last term makes this assumption.)
            d_pp_d = np.sum(diff * (diff.dot(pol_prec[1, t, :, :])), axis=1)
            kl[:, t] = 0.5 * np.sum(np.sum(tr_pp_ct, axis=1), axis=1) - \
                    k_ln_det_ct + 0.5 * ln_det_cp + 0.5 * d_pp_d
            tr_pp_ct_m = np.mean(tr_pp_ct, axis=0)
            kl_m[t] = 0.5 * np.sum(np.sum(tr_pp_ct_m, axis=0), axis=0) - \
                    k_ln_det_ct + 0.5 * np.mean(ln_det_cp) + \
                    0.5 * np.mean(d_pp_d)
        return kl_m, kl

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
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
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta)

        return fCm, fcv
