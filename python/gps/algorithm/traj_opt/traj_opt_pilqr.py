""" This file defines code for PILQR. """
import logging
import copy

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from gps.algorithm.traj_opt.config import TRAJ_OPT_LQR, TRAJ_OPT_PI2, \
        TRAJ_OPT_PILQR
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.traj_opt.traj_opt_utils import  approximated_cost

from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.algorithm_badmm import AlgorithmBADMM


LOGGER = logging.getLogger(__name__)


class TrajOptPILQR(TrajOptLQRPython):
    """ PILQR trajectory optimization. """
    def __init__(self, hyperparams):
        self.config_lqr = copy.deepcopy(TRAJ_OPT_LQR)
        self.config_lqr.update(TRAJ_OPT_PILQR)
        self.config_lqr.update(hyperparams)

        self.config_pi2 = copy.deepcopy(TRAJ_OPT_PI2)
        self.config_pi2.update(hyperparams)

        TrajOptLQRPython.__init__(self, self.config_lqr)

        self.traj_opt_pi2 = TrajOptPI2(self.config_pi2)


    def update(self, m, algorithm):
        traj_distr, eta = TrajOptLQRPython.update(self, m, algorithm)
        prev_traj_distr = algorithm.cur[m].traj_distr
        traj_info = algorithm.cur[m].traj_info

        algorithm.cur[m].traj_distr = traj_distr

        self._hyperparams = self.config_pi2
        mu, c_approx = approximated_cost(
                algorithm.cur[m].sample_list, prev_traj_distr, traj_info
        )
        residual_cost = algorithm.cur[m].cs - c_approx
        _, eta_pi2 = self.traj_opt_pi2.update(
                m, algorithm, use_lqr_actions=True, use_fixed_eta=False
        )

        pi2_traj_distr, _ = self.traj_opt_pi2.update(
                m, algorithm, use_lqr_actions=True,
                fixed_eta=eta_pi2, use_fixed_eta=True, costs=residual_cost
        )

        self._hyperparams = self.config_lqr
        algorithm.cur[m].traj_distr = prev_traj_distr

        return pi2_traj_distr, eta
