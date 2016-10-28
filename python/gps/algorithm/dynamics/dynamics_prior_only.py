""" This file defines linear regression with an arbitrary prior. """
import numpy as np

from gps.algorithm.dynamics.dynamics import Dynamics
from gps.algorithm.algorithm_utils import gauss_fit_joint_prior, condition_dynamics


class DynamicsPriorOnly(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = \
            self._hyperparams['prior']['type'](self._hyperparams['prior'])

    def update_prior(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior


    def linearize(self, dX, dU, prevmu, mu):
        dS = dX+dU+dX
        pts = np.r_[prevmu, mu[:dX]]
        mu0, Phi, mm, n0 = self.prior.eval(dX, dU, np.expand_dims(pts, axis=0))
        sig_reg = np.eye(dS)*self._hyperparams['regularization']
        empmu = np.zeros(dS)
        empsig = np.eye(dS)
        N = 1e-5
        Fm, fv, dyn_covar = condition_dynamics(empmu, empsig, N, mu0, Phi, mm, n0, dX+dU, dX, sig_reg)
        return Fm, fv, dyn_covar


    def fit(self, X, U):
        """ Fit dynamics. """
        pass  # do nothing
