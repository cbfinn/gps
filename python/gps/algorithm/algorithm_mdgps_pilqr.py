""" This file defines the MDGPS with PI2 + LQR-based trajectory optimization method. """
import copy
import logging
import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_MDGPS_PILQR, ALG_MDGPS, ALG_PILQR

LOGGER = logging.getLogger(__name__)


class AlgorithmMDGPSPILQR(AlgorithmMDGPS, AlgorithmTrajOptPILQR):
    def __init__(self, hyperparams):
        self.config_mdgps = copy.deepcopy(ALG_MDGPS_PILQR)
        self.config_mdgps.update(hyperparams)

        self.config_pilqr = copy.deepcopy(ALG_PILQR)
        self.config_pilqr.update(hyperparams)

        AlgorithmTrajOptPILQR.__init__(self, self.config_pilqr)
        hyperparams = copy.deepcopy(self._hyperparams)
        AlgorithmMDGPS.__init__(self, self.config_mdgps)
        self._hyperparams.update(self._hyperparams)
        del self.config_pilqr['agent']  # Don't want to pickle this.
        del self.config_mdgps['agent']  # Don't want to pickle this.

    def iteration(self, sample_lists):
        """
        Run iteration of PILQR.
        Args:
        sample_lists: List of SampleList objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)
        
        # Update dynamics model using all samples.
        self._update_dynamics()
        
        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            AlgorithmMDGPS._update_policy(self)
        
        # Update policy linearizations.
        for m in range(self.M):
            AlgorithmMDGPS._update_policy_fit(self, m)

        # C-step
        if self.iteration_count > 0:
            for m in range(self.M):
                AlgorithmTrajOptPILQR._stepadjust(self, m)
		
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def compute_costs(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU)) 
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB, kB = pol_info.pol_K[t, :, :], pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv
