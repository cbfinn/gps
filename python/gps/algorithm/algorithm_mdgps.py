""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList
from sklearn.cluster import KMeans

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

        # Store single policy prior.
        policy_prior = self._hyperparams['policy_prior']

        # Store pol_info per condition.
        for m in range(self.M):
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

        # Store pol per cluster.
        if self._hyperparams['num_clusters']:
            self.policy_prior = policy_prior['type'](policy_prior)
            self.cluster_pol_info = [PolicyInfo(self._hyperparams) \
                    for k in range(self._hyperparams['num_clusters'])
            ]

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
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # On the first iteration, need to train on init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [self.cur[m].traj_distr \
                    for m in range(self.M)]
            self._update_policy()

        # If clustering, set self.cluster_idx.
        if self._hyperparams['num_clusters']:
            self._cluster_samples()

        # Update dynamics/policy linearizations.
        # NOTE: Skip for 'traj_em', since fits while clustering.
        if self._hyperparams['cluster_method'] != 'traj_em':
            self._update_dynamics()
            self._update_policy_fit()

        # Update quadratic cost expansions for all conditions.
        for m in range(self.M):
            self._eval_cost(m)

        # C-step
        if self.iteration_count > 0:
            self._stepadjust()
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

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
        if 'fc_only_iterations' in self.policy_opt._hyperparams:
            self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt, self.iteration_count)
        else:
            self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def _update_policy_fit(self, update_prior=True):
        """
        Update dynamics linearizations (dispatch for conditions/clusters).
        """
        if self._hyperparams['num_clusters']:
            K = self._hyperparams['num_clusters']
            if update_prior:
                all_samples = self._get_samples()
                self.policy_prior.update(all_samples, self.policy_opt,
                    mode=self._hyperparams['policy_sample_mode'])
            for k in range(K):
                self._update_cluster_policy_fit(k)
        else:
            for m in range(self.M):
                self._update_condition_policy_fit(m, update_prior)

    def _update_condition_policy_fit(self, m, update_prior):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory for condition m.
        """
        samples = self.cur[m].sample_list
        X = samples.get_X()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]

        pol_info = self.cur[m].pol_info
        if update_prior:
            pol_info.policy_prior.update(samples, self.policy_opt,
                    mode=self._hyperparams['policy_sample_mode'])

        # Fit policy linearization and store.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                pol_info.policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(X.shape[1]):
            pol_info.chol_pol_S[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def _update_cluster_policy_fit(self, k):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory for condition m.
        """
        idxs = np.where(self.cluster_idx == k)[0]
        LOGGER.debug("Updating policy fit for cluster %d, idxs %s", k, idxs)
        if idxs.size == 0:
            return

        samples = self._get_samples(idxs)
        X = samples.get_X()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]

        # Fit policy linearization and store.
        pol_info = self.cluster_pol_info[k]
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                self.policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(X.shape[1]):
            pol_info.chol_pol_S[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_S[t, :, :])

        # Copy to trajectory-level info.
        for i in idxs:
            self.cur[i].pol_info.pol_K = pol_info.pol_K
            self.cur[i].pol_info.pol_k = pol_info.pol_k
            self.cur[i].pol_info.pol_S = pol_info.pol_S
            self.cur[i].pol_info.chol_pol_S = pol_info.chol_pol_S

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
        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev) # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            prev_laplace[m] = self.traj_opt.estimate_cost(
                    prev_nn, self.prev[m].traj_info
            ).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = self.traj_opt.estimate_cost(
                    prev_lg, self.prev[m].traj_info
            ).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            cur_laplace[m] = self.traj_opt.estimate_cost(
                    cur_nn, self.cur[m].traj_info
            ).sum()
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

    def _cluster_samples(self, mode=None):
        """
        Divide samples into clusters, using desired clustering method.
        """
        # By default, use 'cluster_method', but can override.
        # (mainly for using kmeans when training on init_traj_distr).
        if mode is None:
            mode = self._hyperparams['cluster_method']

        # Dispatch to clustering method.
        if mode == 'random':
            self._cluster_random()
        elif mode == 'kmeans':
            self._cluster_kmeans()
        elif mode == 'traj_em':
            self._cluster_traj_em()
        else:
            raise NotImplementedError

    def _cluster_random(self):
        """
        Set cluster_idx using random clustering.
        """
        self.cluster_idx = np.random.choice(K, size=self.M)
        LOGGER.debug("Clustering randomly: %s", self.cluster_idx)

    def _cluster_kmeans(self):
        """
        Set cluster_idx using kmeans clustering on initial state.
        """
        K = self._hyperparams['num_clusters']
        kmeans = KMeans(K)
        x0 = self._get_samples().get_X()[:, 0, :] # (N, dX)
        self.cluster_idx = kmeans.fit_predict(x0)
        LOGGER.debug("Clustering by kmeans: %s", self.cluster_idx)

    def _cluster_traj_em(self):
        """
        Set cluster_idx using trajectory aware clustering.
        """
        # Initialize starting with random data points as centers.
        K = self._hyperparams['num_clusters']
#        self.cluster_idx = np.random.choice(K, size=self.M)
#        LOGGER.debug("Traj EM initialization: %s", self.cluster_idx)
        LOGGER.debug("Traj EM initialization to kmeans")
        self._cluster_kmeans()
        self._update_dynamics()
        self._update_policy_fit()

        # Begin EM steps.
        logp = np.empty([self.M, K])
        max_iters = self._hyperparams['traj_em_iters']
        T = self._hyperparams['traj_em_horizon']
        for i in range(max_iters):
            # E step.
            LOGGER.debug("E Step (%d/%d)", i+1, max_iters)
            for k in range(K):
                for m in range(self.M):
                    logp[m, k] = self._traj_log_prob(m, k, T)

            cluster_idx = logp.argmax(axis=1)
            maxlogp = np.max(logp, axis=1, keepdims=True)
            logp -= maxlogp + np.log(np.sum(np.exp(logp-maxlogp), axis=1, keepdims=True))
            cluster_weights = np.exp(logp)
            cluster_masses = cluster_weights.sum(axis=0)
            LOGGER.debug("Traj EM cluster idx: %s", cluster_idx)
            LOGGER.debug("Total cluster weights: %s", cluster_masses)

            # Convergence check.
            if np.all(cluster_idx == self.cluster_idx):
#            if (i > 0) and np.all(cluster_idx == self.cluster_idx):
                LOGGER.debug("Converged on itr (%d/%d)", i+1, max_iters)
                break

            # Reboot small clusters.
            if i < max_iters-1:
                ks = np.where(cluster_masses < 0.1)[0]
                for k in ks:
                    ms = np.random.choice(self.M, size=2)
                    LOGGER.debug("Rebooting cluster %d with %s", k, ms)
                    cluster_idx[ms] = k

            # M step.
            self.cluster_idx = cluster_idx
            LOGGER.debug("M Step (%d/%d)", i+1, max_iters)
            self._update_dynamics(update_prior=False)
            self._update_policy_fit(update_prior=False)

    def _traj_log_prob(self, m, k, T=None):
        """
        Returns the log likelihood of condition 'm' occurring under
        the trajectory defined by cluster 'k'.
        """
        # Pull out states/actions and linearizations.
        sample = self.cur[m].sample_list[0] # M = 1 for clustering.
        X = sample.get_X()
        U = sample.get_U()

        traj_info = self.cluster_traj_info[k]
        x0mu, x0sigma = traj_info.x0mu, traj_info.x0sigma
        dyn = traj_info.dynamics
        pol = self.cluster_pol_info[k]

        # Set horizon, we ignore final (X, U) pair.
        if T is None:
            T = X.shape[0] - 1
        else:
            assert T < X.shape[0] - 1

        # Begin with log prob x0mu/x0sigma.
        # TODO: use sp.linalg.cholesky for all of this?
        logp = -0.5*np.log(np.linalg.det(x0sigma))
        diff = X[0] - x0mu
        logp -= 0.5*diff.dot(np.linalg.solve(x0sigma, diff))

        # Rest of trajectory.
        for t in range(T):
            # Action term.
            sigma = pol.pol_S[t]
            logp -= 0.5*np.log(np.linalg.det(sigma))
            diff = U[t] - pol.pol_K[t].dot(X[t]) - pol.pol_k[t]
            logp -= 0.5*diff.dot(np.linalg.solve(sigma, diff))

            # Next state term.
            xu = np.concatenate([X[t], U[t]])
            sigma = dyn.dyn_covar[t]
            logp -= 0.5*np.log(np.linalg.det(sigma))
            diff = X[t+1] - dyn.Fm[t].dot(xu) - dyn.fv[t]
            logp -= 0.5*diff.dot(np.linalg.solve(sigma, diff))

        return logp
