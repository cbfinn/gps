""" This file defines the constant prior for policy linearization. """
import copy

import numpy as np

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
