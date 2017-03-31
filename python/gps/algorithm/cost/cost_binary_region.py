""" This file defines the binary region cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_BINARY_REGION
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier


class CostBinaryRegion(Cost):
    """ Computes binary cost that determines if the object 
    is inside the given region around the target state.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_BINARY_REGION)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        tt_total = 0
        for data_type in self._hyperparams['data_types']:
            config = self._hyperparams['data_types'][data_type]
            wp = config['wp']
            tgt = config['target_state']
            max_distance = config['max_distance']
            outside_cost = config['outside_cost']
            inside_cost = config['inside_cost']
            x = sample.get(data_type)
            _, dim_sensor = x.shape

            wpm = get_ramp_multiplier(
                self._hyperparams['ramp_option'], T,
                wp_final_multiplier=self._hyperparams['wp_final_multiplier']
            )
            wp = wp * np.expand_dims(wpm, axis=-1)

            # Compute binary region penalty.
            dist = np.abs(x - tgt)
            for t in xrange(T):
                # If at least one of the coordinates is outside of 
                # the region assign outside_cost, otherwise inside_cost.
                if np.sum(dist[t]) > max_distance:
                    final_l[t] += outside_cost
                else:
                    final_l[t] += inside_cost 

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
