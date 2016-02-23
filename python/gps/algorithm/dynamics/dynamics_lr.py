""" This file defines linear regression for dynamics estimation. """
import numpy as np

from gps.algorithm.dynamics.dynamics import Dynamics


class DynamicsLR(Dynamics):
    """ Dynamics with linear regression, with constant prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None

    def update_prior(self, sample):
        """ Update dynamics prior. """
        # Nothing to do - constant prior.
        pass

    def get_prior(self):
        """ Return the dynamics prior, or None if constant prior. """
        return None

    def fit(self, sample_list):
        """ Fit dynamics. """
        X = sample_list.get_X()  # Use all samples to fit dynamics.
        U = sample_list.get_U()
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics wih least squares regression.
        for t in range(T - 1):
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            xux_mean = np.mean(xux, axis=0)
            empsig = (xux - xux_mean).T.dot(xux - xux_mean) / (N - 1)
            sigma = 0.5 * (empsig + empsig.T)
            sigma[it, it] += self._hyperparams['regularization'] * np.eye(dX+dU)

            Fm = np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip]).T
            fv = xux_mean[ip] - Fm.dot(xux_mean[it])

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5 * (dyn_covar + dyn_covar.T)
