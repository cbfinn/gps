# TODO: This cost function is hard-coded and not easy to generalize.
import copy

import numpy as np

from gps.algorithm.cost.config import COST_FK
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import get_ramp_multiplier
from gps.proto.gps_pb2 import JOINT_ANGLES, END_EFFECTOR_POINTS, \
        END_EFFECTOR_POINT_JACOBIANS


class CostFKBlock(Cost):
    """Computes cost for gripper pusher based on distance from gripper to block.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_FK)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        T = sample.T
        dX = sample.dX
        dU = sample.dU

        wpm = get_ramp_multiplier(
            self._hyperparams['ramp_option'], T,
            wp_final_multiplier=self._hyperparams['wp_final_multiplier']
        )
        wp = self._hyperparams['wp'] * np.expand_dims(wpm, axis=-1)

        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        pt = sample.get(END_EFFECTOR_POINTS)

        # The following is hard-coded: determines position of gripper and block
        # and computes the distance between them
        pt_ee_l = pt[:, 0:3]
        pt_ee_r = pt[:, 3:6]
        pt_ee_avg = 0.5 * (pt_ee_r + pt_ee_l)
        pt_block = pt[:, 6:9]
        dist = pt_ee_avg - pt_block

        wp = np.ones((T,3))
        jx = sample.get(END_EFFECTOR_POINT_JACOBIANS)

        # Also hard-coded is this Jacobian calculation
        jx_1 = 0.5 * (jx[:, 0:3, :] + jx[:, 3:6, :]) - jx[:, 6:9, :]

        jxx_zeros = np.zeros((T, dist.shape[1], jx.shape[2], jx.shape[2]))
        l, ls, lss = self._hyperparams['evalnorm'](
            wp, dist, jx_1, jxx_zeros, self._hyperparams['l1'],
            self._hyperparams['l2'], self._hyperparams['alpha']
        )

        sample.agent.pack_data_x(lx, ls, data_types=[JOINT_ANGLES])
        sample.agent.pack_data_x(lxx, lss,
                                 data_types=[JOINT_ANGLES, JOINT_ANGLES])

        return l, lx, lu, lxx, luu, lux
