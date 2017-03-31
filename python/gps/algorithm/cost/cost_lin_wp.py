""" This file defines an arbitrary linear waypoint cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_LIN_WP
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier


class CostLinWP(Cost):
    """ Computes an arbitrary linear waypoint cost. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_LIN_WP)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T, dU, dX = sample.T, sample.dU, sample.dX
        orig_A, orig_b = self._hyperparams['A'], self._hyperparams['b']

        # Discretize waypoint time steps.
        waypoint_step = np.ceil(T * self._hyperparams['waypoint_time'])
        A = np.zeros((T, dX+dU, dX+dU))
        b = np.zeros((T, dX+dU))

        if not isinstance(self._hyperparams['ramp_option'], list):
            self._hyperparams['ramp_option'] = [
                self._hyperparams['ramp_option'] for _ in waypoint_step
            ]

        # Set up time-varying matrix and vector.
        start = 0
        for i in range(len(waypoint_step)):
            wpm = get_ramp_multiplier(self._hyperparams['ramp_option'][i],
                                      waypoint_step[i] - start)
            for t in range(start, int(waypoint_step[i])):
                A[t, :, :] = wpm[t-start] * orig_A[i, :, :]
                b[t, :] = wpm[t-start] * orig_b[i, :]
            start = int(waypoint_step[i])

        # Evaluate distance function.
        XU = np.concatenate([sample.get_X(), sample.get_U()], axis=1)
        dist = np.zeros((T, dX+dU))
        for t in range(T):
            dist[t, :] = A[t, :, :].dot(XU[t, :]) + b[t, :]

        ix, iu = slice(dX), slice(dX, dX+dU)
        # Compute the loss.
        Al, Aly, Alyy = self._evalloss(dist)
        Alx = Aly[:, ix]
        Alu = Aly[:, iu]
        Alxx = Alyy[:, ix, ix]
        Aluu = Alyy[:, iu, iu]
        Alux = Alyy[:, ix, iu]

        # Compute state derivatives and value. Loss is the same.
        l = Al
        # First derivative given by A'*Aly.
        # Second derivative given by A'*Alyy*A.
        lx, lu = np.zeros((T, dX)), np.zeros((T, dU))
        lxx, luu = np.zeros((T, dX, dX)), np.zeros((T, dU, dU))
        lux = np.zeros((T, dU, dX))
        for t in range(T):
            lx[t, :] = A[t, ix, ix].T.dot(Alx[t, :])
            lu[t, :] = A[t, iu, iu].T.dot(Alu[t, :])
            lxx[t, :, :] = A[t, ix, ix].T.dot(Alxx[t, :, :]).dot(A[t, ix, ix])
            luu[t, :, :] = A[t, iu, iu].T.dot(Aluu[t, :, :]).dot(A[t, iu, iu])
            lux[t, :, :] = A[t, ix, iu].T.dot(Alux[t, :, :]).dot(A[t, iu, ix])

        return l, lx, lu, lxx, luu, lux

    def _evalloss(self, dist):
        T = dist.shape[0]
        l1, l2 = self._hyperparams['l1'], self._hyperparams['l2']
        alpha = self._hyperparams['alpha']
        logalpha, log = self._hyperparams['logalpha'], self._hyperparams['log']

        # Compute total cost.
        l = 0.5 * np.sum(dist ** 2, axis=1) * l2 + \
                np.sqrt(alpha + np.sum(dist ** 2, axis=1)) * l1 + \
                0.5 * np.log(logalpha + np.sum(dist ** 2, axis=1)) * log

        # First order derivative terms.
        lx = dist * l2 + (
            dist / np.sqrt(alpha + np.sum(dist**2, axis=1, keepdims=True)) * l1
        ) + (dist / (logalpha + np.sum(dist ** 2, axis=1, keepdims=True)) * log)

        # Second order derivative terms.
        psq = np.expand_dims(
            np.sqrt(alpha + np.sum(dist ** 2, axis=1, keepdims=True)), axis=1
        )
        psqlog = np.expand_dims(
            logalpha + np.sum(dist ** 2, axis=1, keepdims=True), axis=1
        )
        lxx = l1 * (
            (np.expand_dims(np.eye(dist.shape[1]), axis=0) / psq) -
            ((np.expand_dims(dist, axis=1) *
              np.expand_dims(dist, axis=2)) / psq ** 3)
        )
        lxx += l2 * np.tile(np.eye(dist.shape[1]), [T, 1, 1])
        lxx += log * (
            (np.expand_dims(np.eye(dist.shape[1]), axis=0) / psqlog) -
            ((np.expand_dims(dist, axis=1) *
              np.expand_dims(dist, axis=2)) / psq ** 2)
        )

        return l, lx, lxx
