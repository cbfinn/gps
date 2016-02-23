""" This file defines a GMM prior for policy linearization. """
import copy
import logging

import numpy as np

from gps.algorithm.policy.config import POLICY_PRIOR_GMM
from gps.utility.gmm import GMM


LOGGER = logging.getLogger(__name__)


class PolicyPriorGMM(object):
    """
    A policy prior encoded as a GMM over [x_t, u_t] points, where u_t is
    the output of the policy for the given state x_t. This prior is used
    when computing the linearization of the policy.

    See the method AlgorithmBADMM._update_policy_fit, in
    python/gps/algorithm.algorithm_badmm.py.

    Also see the GMM dynamics prior, in
    python/gps/algorithm/dynamics/dynamics_prior_gmm.py. This is a
    similar GMM prior that is used for the dynamics estimate.
    """
    def __init__(self, hyperparams):
        """
        Hyperparameters:
            min_samples_per_cluster: Minimum number of samples.
            max_clusters: Maximum number of clusters to fit.
            max_samples: Maximum number of trajectories to use for
                fitting the GMM at any given time.
            strength: Adjusts the strength of the prior.
        """
        config = copy.deepcopy(POLICY_PRIOR_GMM)
        config.update(hyperparams)
        self._hyperparams = config
        self.X = None
        self.obs = None
        self.gmm = GMM()
        self._min_samp = self._hyperparams['min_samples_per_cluster']
        self._max_samples = self._hyperparams['max_samples']
        self._max_clusters = self._hyperparams['max_clusters']
        self._strength = self._hyperparams['strength']

    def update(self, samples, policy_opt, all_samples, retrain=True):
        """ Update prior with additional data. """
        X, obs = samples.get_X(), samples.get_obs()
        all_X, all_obs = all_samples.get_X(), all_samples.get_obs()
        U = all_samples.get_U()
        dO, T = all_X.shape[2] + U.shape[2], all_X.shape[1]
        if self._hyperparams['keep_samples']:
            # Append data to dataset.
            if self.X is None:
                self.X = X
            elif X.size > 0:
                self.X = np.concatenate([self.X, X], axis=0)
            if self.obs is None:
                self.obs = obs
            elif obs.size > 0:
                self.obs = np.concatenate([self.obs, obs], axis=0)
            # Remove excess samples from dataset.
            start = max(0, self.X.shape[0] - self._max_samples + 1)
            self.X = self.X[start:, :, :]
            self.obs = self.obs[start:, :, :]
            # Evaluate policy at samples to get mean policy action.
            Upol = policy_opt.prob(self.obs.copy())[0]
            # Create dataset.
            N = self.X.shape[0]
            XU = np.reshape(np.concatenate([self.X, Upol], axis=2), [T * N, dO])
        else:
            # Simply use the dataset that is already there.
            all_U = policy_opt.prob(all_obs.copy())[0]
            N = all_X.shape[0]
            XU = np.reshape(np.concatenate([all_X, all_U], axis=2), [T * N, dO])
        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))
        LOGGER.debug('Generating %d clusters for policy prior GMM.', K)
        # Update GMM.
        if retrain:
            self.gmm.update(XU, K)

    def eval(self, Ts, Ps):
        """ Evaluate prior. """
        # Construct query data point.
        pts = np.concatenate((Ts, Ps), axis=1)
        # Perform query.
        mu0, Phi, m, n0 = self.gmm.inference(pts)
        # Factor in multiplier.
        n0 *= self._strength
        m *= self._strength
        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0
