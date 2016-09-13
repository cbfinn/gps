""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier


class CostState(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        # Allocate space for derivatives (full state)
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        # Compute state penalty for data_type (or X by default)
        data_type = self._hyperparams['data_type']
        x = sample.get(data_type) if data_type else sample.get_X()
        _, Ds = x.shape

        A = self._hyperparams['A']
        D = A.shape[0]
        tgt = self._hyperparams['target']

        ramp = get_ramp_multiplier(
            self._hyperparams['ramp_option'], T,
            wp_final_multiplier=self._hyperparams['wp_final_multiplier']
        )
        wp = np.ones(D) * np.expand_dims(ramp, axis=-1)

        # Compute state penalty.
        # x.shape == (T, Ds)
        # A.shape == (D, Ds)
        # x.dot(A.T).shape == (T, D)
        # tgt.shape == (D,) or scalar
        dist = x.dot(A.T) - tgt

        # Evaluate penalty term.
        l, ls, lss = evall1l2term(
            wp, dist, np.tile(A, [T, 1, 1]), np.zeros((T, D, Ds, Ds)),
            self._hyperparams['l1'], self._hyperparams['l2'],
            self._hyperparams['alpha']
        )

        # Pack into final cost
        final_l += l
        sample.agent.pack_data_x(final_lx, ls, data_types=[data_type])
        sample.agent.pack_data_x(final_lxx, lss,
                                 data_types=[data_type, data_type])

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
