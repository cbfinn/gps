""" This file defines a Gaussian mixture model class. """
import logging

import numpy as np
import scipy.linalg


LOGGER = logging.getLogger(__name__)


def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv


class GMM(object):
    """ Gaussian Mixture Model. """
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None

    def inference(self, pts):
        """
        Evaluate dynamics prior.
        Args:
            pts: A N x D array of points.
        """
        # Compute posterior cluster weights.
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0, Phi = self.moments(logwts)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    def estep(self, data):
        """
        Compute log observation probabilities under GMM.
        Args:
            data: A N x D array of points.
        Returns:
            logobs: A N x K array of log probabilities (for each point
                on each cluster).
        """
        # Constants.
        K = self.sigma.shape[0]
        Di = data.shape[1]
        N = data.shape[0]

        # Compute probabilities.
        data = data.T
        mu = self.mu[:, 0:Di].T
        mu_expand = np.expand_dims(np.expand_dims(mu, axis=1), axis=1)
        assert mu_expand.shape == (Di, 1, 1, K)
        # Calculate for each point distance to each cluster.
        data_expand = np.tile(data, [K, 1, 1, 1]).transpose([2, 3, 1, 0])
        diff = data_expand - np.tile(mu_expand, [1, N, 1, 1])
        assert diff.shape == (Di, N, 1, K)
        Pdiff = np.zeros_like(diff)
        cconst = np.zeros((1, 1, 1, K))

        for i in range(K):
            U = scipy.linalg.cholesky(self.sigma[i, :Di, :Di],
                                      check_finite=False)
            Pdiff[:, :, 0, i] = scipy.linalg.solve_triangular(
                U, scipy.linalg.solve_triangular(
                    U.T, diff[:, :, 0, i], lower=True, check_finite=False
                ), check_finite=False
            )
            cconst[0, 0, 0, i] = -np.sum(np.log(np.diag(U))) - 0.5 * Di * \
                    np.log(2 * np.pi)

        logobs = -0.5 * np.sum(diff * Pdiff, axis=0, keepdims=True) + cconst
        assert logobs.shape == (1, N, 1, K)
        logobs = logobs[0, :, 0, :] + self.logmass.T
        return logobs

    def moments(self, logwts):
        """
        Compute the moments of the cluster mixture with logwts.
        Args:
            logwts: A K x 1 array of log cluster probabilities.
        Returns:
            mu: A (D,) mean vector.
            sigma: A D x D covariance matrix.
        """
        # Exponentiate.
        wts = np.exp(logwts)

        # Compute overall mean.
        mu = np.sum(self.mu * wts, axis=0)

        # Compute overall covariance.
        # For some reason this version works way better than the "right"
        # one... could we be computing xxt wrong?
        diff = self.mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(diff, axis=1) * \
                np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: An N x D array of points
        Returns:
            A K x 1 array of average cluster log probabilities.
        """
        # Compute probability of each point under each cluster.
        logobs = self.estep(data)

        # Renormalize to get cluster weights.
        logwts = logobs - logsum(logobs, axis=1)

        # Average the cluster probabilities.
        logwts = logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T

    def update(self, data, K, max_iterations=100):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N = data.shape[0]
        Do = data.shape[1]

        LOGGER.debug('Fitting GMM with %d clusters on %d points', K, N)

        if (not self.warmstart or self.sigma is None or
                K != self.sigma.shape[0]):
            # Initialization.
            LOGGER.debug('Initializing GMM.')
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            N = self.N

            # Set initial cluster indices.
            if not self.init_sequential:
                cidx = np.random.randint(0, K, size=(1, N))
            else:
                raise NotImplementedError()

            # Initialize.
            for i in range(K):
                cluster_idx = (cidx == i)[0]
                mu = np.mean(data[cluster_idx, :], axis=0)
                diff = (data[cluster_idx, :] - mu).T
                sigma = (1.0 / K) * (diff.dot(diff.T))
                self.mu[i, :] = mu
                self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6

        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs, axis=1))
            LOGGER.debug('GMM itr %d/%d. Log likelihood: %f',
                         itr, max_iterations, ll)
            if np.abs(ll-prevll) < 1e-2:
                LOGGER.debug('GMM convergenced on itr=%d/%d',
                             itr, max_iterations)
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - logsum(logobs, axis=1)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - logsum(logw, axis=0)
            assert logwn.shape == (N, K)
            w = np.exp(logwn)

            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = logsum(logw, axis=0).T
            self.logmass = self.logmass - logsum(self.logmass, axis=0)
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass)
            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + \
                            1e-6 * np.eye(Do)
