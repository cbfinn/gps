""" This file defines the constant prior for policy linearization. """
import copy

import numpy as np

from gps.algorithm.algorithm_utils import gauss_fit_joint_prior
from gps.algorithm.policy.config import POLICY_PRIOR


class PolicyPrior(object):
    """ Constant policy prior. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(POLICY_PRIOR)
        config.update(hyperparams)
        self._hyperparams = config

    def update(self, samples, policy_opt, all_samples, retrain=True):
        """ Update dynamics prior. """
        # Nothing to update for constant policy prior.
        pass

    def eval(self, Ts, Ps):
        """ Evaluate the policy prior. """
        dX, dU = Ts.shape[-1], Ps.shape[-1]
        prior_fd = np.zeros((dU, dX))
        prior_cond = 1e-5 * np.eye(dU)
        sig = np.eye(dX)
        Phi = self._hyperparams['strength'] * np.vstack([
            np.hstack([sig, sig.dot(prior_fd.T)]),
            np.hstack([prior_fd.dot(sig),
                       prior_fd.dot(sig).dot(prior_fd.T) + prior_cond])
        ])
        return np.zeros(dX+dU), Phi, 0, self._hyperparams['strength']

    def fit(self, X, pol_mu, pol_sig):
        """
        Fit policy linearization.

        Args:
            X: Samples (N, T, dX)
            pol_mu: Policy means (N, T, dU)
            pol_sig: Policy covariance (N, T, dU)
        """
        N, T, dX = X.shape
        dU = pol_mu.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        # Collapse policy covariances. (This is only correct because
        # the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # Allocate.
        pol_K = np.zeros([T, dU, dX])
        pol_k = np.zeros([T, dU])
        pol_S = np.zeros([T, dU, dU])

        # Fit policy linearization with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = \
                    gauss_fit_joint_prior(Ys,
                            mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_S += pol_sig
        return pol_K, pol_k, pol_S
