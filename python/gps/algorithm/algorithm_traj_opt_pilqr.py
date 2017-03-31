""" This file defines the PILQR-based algorithm. """
import copy
import logging
import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.config import ALG_PILQR

from gps.algorithm.traj_opt.traj_opt_utils import approximated_cost


LOGGER = logging.getLogger(__name__)


class AlgorithmTrajOptPILQR(Algorithm):
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_PILQR)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

    def iteration(self, sample_lists):
        """
        Run iteration of PILQR.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all samples.
        self._update_dynamics()

        self._update_step_size()  # KL Divergence step size.

        # Run inner loop to compute new policies.
        for _ in range(self._hyperparams['inner_iterations']):
            self._update_trajectories()
            self._advance_iteration_variables()

    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

    def _compute_step_pi2_lqr(self, m, pdpc, pdcc, cdcc, pmc, cmc, cur_res):
        """
        Calculate new step size for pi2 and lqr.
        Args:
        m: condition
        pdpc: previous dynamics-previous controller cost
        pdcc: previous dynamics-current controller cost
        cdcc: current dynamics-current controller cost
        pmc: previous actual cost
        cmc: current actual cost
        cur_res:
        """

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Cost estimate under prev dynamics, prev controller
        pdpc = self.traj_opt.estimate_cost(
                self.prev[m].traj_distr, self.prev[m].traj_info
        )
        # Cost estimate under prev dynamics, curr controller
        pdcc = self.traj_opt.estimate_cost(
                self.cur[m].traj_distr, self.prev[m].traj_info
        )
        # Cost estimate under curr dynamics, curr controller
        cdcc = self.traj_opt.estimate_cost(
                self.cur[m].traj_distr, self.cur[m].traj_info
        )

        ent = self._measure_ent(m)

        pmc = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        cmc = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, pmc, cmc)

        _, cur_c_approx = approximated_cost(
                self.cur[m].sample_list, self.cur[m].traj_distr,
                self.cur[m].traj_info
        )
        cur_res = cur_c_approx - self.cur[m].cs

        LOGGER.debug('Previous cost: Laplace: %f MC: %f', np.sum(pdpc), pmc)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f', np.sum(pdcc), cmc)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f', np.sum(cdcc), cmc)

        max_mult = self._hyperparams['max_step_mult']
        min_mult = self._hyperparams['min_step_mult']

        if self._hyperparams['step_rule'] == 'const':
            self.cur[m].step_mult = self.prev[m].step_mult
        elif self._hyperparams['step_rule'] == 'res_percent':
            cur_res_sum = np.mean(np.sum(np.abs(cur_res), axis=1))
            act_sum = np.mean(np.sum(np.abs(self.cur[m].cs), axis=1))
            # Print cost details.
            LOGGER.debug(
                    'Cur residuals: %f. Residuals percentage: %f',
                    cur_res_sum, cur_res_sum/act_sum*100
            )
            if self.traj_opt.cons_per_step:
                T = cur_res.shape[1]
                new_mult = np.ones(T)
                for t in range(T):
                    res = np.mean(np.sum(np.abs(cur_res[:, t:]), axis=1))
                    act = np.mean(np.sum(np.abs(self.cur[m].cs[:, t:]), axis=1))
                    if res / act > self._hyperparams['step_rule_res_ratio_dec']:
                        new_mult[t] = 0.5
                    elif res / act < self._hyperparams['step_rule_res_ratio_inc']:
                        new_mult[t] = 2.0
            else:
                if cur_res_sum / act_sum > self._hyperparams['step_rule_res_ratio_dec']:
                    new_mult = 0.5
                elif cur_res_sum / act_sum < self._hyperparams['step_rule_res_ratio_inc']:
                    new_mult = 2.0
                else:
                    new_mult = 1
            new_step = np.maximum(
                    np.minimum(new_mult * self.cur[m].step_mult, max_mult),
                    min_mult
            )
            self.cur[m].step_mult = new_step
        else:
            raise NotImplementedError('Unknown step adjustment rule.')

    def compute_costs(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        multiplier = self._hyperparams['max_ent_traj']
        fCm, fcv = traj_info.Cm / (eta + multiplier), traj_info.cv / (eta + multiplier)
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k

        # Add in the trajectory divergence term.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += eta / (eta + multiplier) * np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += eta / (eta + multiplier) * np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        return fCm, fcv
