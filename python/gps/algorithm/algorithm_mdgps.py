""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo, gauss_fit_joint_prior
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

        for m in range(self.M):
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            policy_prior = self._hyperparams['policy_prior']
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )

    def iteration(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples.
        self.N = sum(len(self.sample_list[i]) for i in self.sample_list.keys())
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            prev_samples = self.sample_list[m].get_samples()
            prev_samples.extend(sample_lists[m].get_samples())
            self.sample_list[m] = SampleList(prev_samples)
            self.N += len(sample_lists[m])

        if self.iteration_count == 0:
            self.policy_opts[self.iteration_count] = self.policy_opt.copy()

        self._update_dynamics()  # Update dynamics model using all sample.
        self._update_policy_samples()  # Choose samples to use with the policy.

        if self._hyperparams['ioc']:
            self._update_cost()

        # On the first iteration we need to make sure that the policy somewhat
        # matches the init controller. Otherwise the LQR backpass starts with
        # a bad linearization, and things don't work out well.
        if self.iteration_count == 0 and not self._hyperparams['init_demo_policy']:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            self._update_policy()

        # Step adjustment
        self._update_step_size()  # KL Divergence step size (also fits policy).
        for m in range(self.M):
            pol_info = self.cur[m].pol_info
            # Explicitly store the current pol_info, need for step size calc
            self.cur[m].init_pol_info = copy.deepcopy(pol_info)

        # C-step
        self._update_trajectories()
        for m in range(self.M):
            # Save initial kl for debugging / visualization.
            self.cur[m].pol_info.init_kl = self._policy_kl(m)[0]

        # S-step
        self._update_policy()
        for m in range(self.M):
            self._update_policy_fit(m)  # Update policy priors.
            # Save final kl for debugging / visualization.
            kl_m = self._policy_kl(m)[0]
            self.cur[m].pol_info.prev_kl = kl_m

        # Computing KL-divergence between sample distribution and demo distribution
        itr = self.iteration_count
        if self._hyperparams['ioc'] and not self._hyperparams['init_demo_policy']:
            for i in xrange(self.M):
                mu, sigma = self.traj_opt.forward(self.traj_distr[itr][i], self.traj_info[itr][i])
                # KL divergence between current traj. distribution and gt distribution
                self.kl_div[itr].append(traj_distr_kl(mu, sigma, self.traj_distr[itr][i], self.demo_traj[0])) # Assuming Md == 1
        # Compute mean distance to target. For peg experiment only.
        if self._hyperparams['learning_from_prior']:
            for i in xrange(self.M):
                target_position = self._hyperparams['target_end_effector'][:3]
                cur_samples = sample_lists[i].get_samples()
                sample_end_effectors = [cur_samples[i].get(END_EFFECTOR_POINTS) for i in xrange(len(cur_samples))]
                dists = [np.nanmin(np.sqrt(np.sum((sample_end_effectors[i][:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0) \
                         for i in xrange(len(cur_samples))]
                self.dists_to_target[itr].append(sum(dists) / len(cur_samples))
        self._advance_iteration_variables()

    def _update_policy_samples(self):
        """ Update the list of samples to use with the policy. """
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

    def _update_policy(self):
        """ Compute the new policy. """
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
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

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
        # Get the necessary linearizations
        # TODO: right now the moving parts are complicated, and we need to
        # use this hacky 'init_size_pol_info'. Need to refactor.
        prev_nn = self.prev[m].init_pol_info.traj_distr()
        cur_nn = self.cur[m].pol_info.traj_distr()
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
        if self._hyperparams['ioc']:
            self._eval_cost(m, prev_cost=True)
            traj_info = self.cur[m].prevcost_traj_info
        else:
            traj_info = self.cur[m].traj_info
        if (self._hyperparams['step_rule'] == 'classic'):
            new_actual_laplace_cost = self.traj_opt.estimate_cost(
                cur_traj_distr, traj_info
            )
        elif (self._hyperparams['step_rule'] == 'global'):
            new_actual_laplace_cost = self.traj_opt.estimate_cost(
                cur_nn, traj_info
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
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
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

    def _update_cost(self):
        """ Update the cost objective in each iteration. """
        # Estimate the importance weights for fusion distributions.
        demos_logiw, samples_logiw = self.importance_weights()

        # Update the learned cost
        # Transform all the dictionaries to arrays for global cost
        M = len(self.prev)
        Md = self._hyperparams['demo_M']
        sampleU_arr = np.vstack((self.sample_list[i].get_U() for i in xrange(M)))
        sampleX_arr = np.vstack((self.sample_list[i].get_X() for i in xrange(M)))
        sampleO_arr = np.vstack((self.sample_list[i].get_obs() for i in xrange(M)))
        demos_logiw_arr = np.hstack((demos_logiw[i] for i in xrange(Md))).reshape((-1, 1))
        samples_logiw_arr = np.hstack([samples_logiw[i] for i in xrange(M)]).reshape((-1, 1))
        demos_logiw = {i: demos_logiw[i].reshape((-1, 1)) for i in xrange(Md)}
        samples_logiw = {i: samples_logiw[i].reshape((-1, 1)) for i in xrange(M)}
        # TODO - not sure if we want one cost function per condition...
        if not self._hyperparams['global_cost']:
            for i in xrange(M):
                cost_ioc = self.cost[i]
                cost_ioc.update(self.demoU, self.demoX, self.demoO, demos_logiw_arr, self.sample_list[i].get_U(), \
                                    self.sample_list[i].get_X(), self.sample_list[i].get_obs(), samples_logiw[i])
        else:
            cost_ioc = self.cost
            cost_ioc.update(self.demoU, self.demoX, self.demoO, demos_logiw_arr, sampleU_arr, sampleX_arr, \
                                                        sampleO_arr, samples_logiw_arr)

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
