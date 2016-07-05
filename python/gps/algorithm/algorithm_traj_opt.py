""" This file defines the iLQG-based trajectory optimization method. """
import logging

import numpy as np


from gps.algorithm.algorithm import Algorithm
from gps.utility.general_utils import logsum
from gps.algorithm.algorithm_utils import fit_emp_controllers
from gps.algorithm.cost.cost_ioc_quad import CostIOCQuadratic
from gps.sample.sample_list import SampleList

LOGGER = logging.getLogger(__name__)


class AlgorithmTrajOpt(Algorithm):
    """ Sample-based trajectory optimization. """
    def __init__(self, hyperparams):
        Algorithm.__init__(self, hyperparams)
        # self.policy_opt = self._hyperparams['policy_opt']['type'](
        #         self._hyperparams['policy_opt'], self.dO, self.dU
        #         )

    def iteration(self, sample_lists):
        """
        Run iteration of LQR.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        self.N = sum(len(self.sample_list[i]) for i in self.sample_list.keys())
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            prev_samples = self.sample_list[m].get_samples()
            prev_samples.extend(sample_lists[m].get_samples())
            self.sample_list[m] = SampleList(prev_samples)
            self.N += len(sample_lists[m])
        # Update dynamics model using all samples.
        self._update_dynamics()

        # Update the cost during learning if we use IOC.
        if self._hyperparams['ioc']:
            self._update_cost()

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

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt.estimate_cost(
            self.prev[m].traj_distr, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = self.traj_opt.estimate_cost(
            self.cur[m].traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        if self._hyperparams['ioc']:
          # use prevous cost to estimate cost of current traj distr.
          self._eval_cost(m, prev_cost=True)
          new_actual_laplace_obj = self.traj_opt.estimate_cost(
              self.cur[m].traj_distr, self.cur[m].prevcost_traj_info
          )
        else:
          new_actual_laplace_obj = self.traj_opt.estimate_cost(
              self.cur[m].traj_distr, self.cur[m].traj_info
          )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def _update_cost(self):
        """ Update the cost objective in each iteration. """
        # Estimate the importance weights for fusion distributions.
        itr = self.iteration_count
        M = len(self.prev)
        ix = range(self.dX)
        iu = range(self.dX, self.dX + self.dU)
        init_samples = self.init_samples
        # itration_count + 1 distributions to evaluate
        # T: summed over time
        samples_logprob, demos_logprob = {}, {}
        # number of demo distributions
        Md = self._hyperparams['demo_M']
        demos_logiw, samples_logiw = {}, {}
        demoU = {i: self.demoU for i in xrange(Md)}
        demoX = {i: self.demoX for i in xrange(Md)}
        demoO = {i: self.demoO for i in xrange(Md)}
        # For testing purpose.
        # demoX = self.demoX
        # demoU = self.demoU
        # demoO = self.demoO
        demo_traj = {}
        # estimate demo distributions empirically
        for i in xrange(Md):
            if self._hyperparams['demo_distr_empest']:
                demo_traj[i] = fit_emp_controllers(demoX[i], demoU[i])
        for i in xrange(M):
            # This code assumes a fixed number of samples per iteration/controller
            samples_logprob[i] = np.zeros((itr + Md + 1, self.T, (self.N / M) * itr + init_samples))
            demos_logprob[i] = np.zeros((itr + Md + 1, self.T, demoX[i].shape[0]))
            sample_i_X = self.sample_list[i].get_X()
            sample_i_U = self.sample_list[i].get_U()
            # Evaluate sample prob under sample distributions
            for itr_i in xrange(itr + 1):
                traj = self.traj_distr[itr_i][i]
                for j in xrange(sample_i_X.shape[0]):
                    for t in xrange(self.T - 1):
                        # Need to add traj.ref here?
                        diff = traj.k[t, :] + \
                                traj.K[t, :, :].dot(sample_i_X[j, t, :]) - sample_i_U[j, t, :]
                        samples_logprob[i][itr_i, t, j] = -0.5 * np.sum(diff * (traj.inv_pol_covar[t, :, :].dot(diff)), 0) - \
                                                        np.sum(np.log(np.diag(traj.chol_pol_covar[t, :, :])))

            # Evaluate sample prob under demo distribution.
            for itr_i in xrange(Md):
                for j in range(sample_i_X.shape[0]):
                    for t in xrange(self.T - 1):
                        diff = demo_traj[itr_i].k[t, :] + \
                                demo_traj[itr_i].K[t, :, :].dot(sample_i_X[j, t, :]) - sample_i_U[j, t, :]
                        samples_logprob[i][itr + itr_i + 1, t, j] = -0.5 * np.sum(diff * (demo_traj[itr_i].inv_pol_covar[t, :, :].dot(diff)), 0) - \
                                                        np.sum(np.log(np.diag(demo_traj[itr_i].chol_pol_covar[t, :, :])))
            # Sum over the distributions and time.

            samples_logiw[i] = logsum(np.sum(samples_logprob[i], 1), 0)

        # Assume only one condition for the samples.
        assert Md == 1
        for idx in xrange(Md):
            if M == 1:
                i = 0
            else:
                i = idx
            # Evaluate demo prob. under sample distributions.
            for itr_i in xrange(itr + 1):
                traj = self.traj_distr[itr_i][i]
                for j in xrange(demoX[idx].shape[0]):
                    for t in xrange(self.T - 1):
                        diff = traj.k[t, :] + \
                                traj.K[t, :, :].dot(demoX[idx][j, t, :]) - demoU[idx][j, t, :]
                        demos_logprob[idx][itr_i, t, j] = -0.5 * np.sum(diff * (traj.inv_pol_covar[t, :, :].dot(diff)), 0) - \
                                                        np.sum(np.log(np.diag(traj.chol_pol_covar[t, :, :])))
            # Evaluate demo prob. under demo distributions.
            for itr_i in xrange(Md):
                for j in range(demoX[idx].shape[0]):
                    for t in xrange(self.T - 1):
                        diff = demo_traj[itr_i].k[t, :] + \
                                demo_traj[itr_i].K[t, :, :].dot(demoX[idx][j, t, :]) - demoU[idx][j, t, :]
                        demos_logprob[idx][itr + itr_i + 1, t, j] = -0.5 * np.sum(diff * (demo_traj[itr_i].inv_pol_covar[t, :, :].dot(diff)), 0) - \
                                                        np.sum(np.log(np.diag(demo_traj[itr_i].chol_pol_covar[t, :, :])))
            # Sum over the distributions and time.
            demos_logiw[idx] = logsum(np.sum(demos_logprob[idx], 1), 0)


        # Update the learned cost
        # Transform all the dictionaries to arrays
        # demoU_arr =  np.vstack((self.demo_list.get_U() for i in xrange(Md)))
        # demoX_arr =  np.vstack((self.demo_list.get_X() for i in xrange(Md)))
        # demoO_arr =  np.vstack((self.demo_list.get_obs() for i in xrange(Md)))
        # For testing purpose.
        sampleU_arr = np.vstack((self.sample_list[i].get_U() for i in xrange(M)))
        sampleX_arr = np.vstack((self.sample_list[i].get_X() for i in xrange(M)))
        sampleO_arr = np.vstack((self.sample_list[i].get_obs() for i in xrange(M)))
        demos_logiw = np.vstack((demos_logiw[i] for i in xrange(Md))).T
        samples_logiw = np.vstack((samples_logiw[i] for i in xrange(M))).T
        for i in xrange(M):
            cost_ioc_quad = self.cost[i] # set the type of cost to be cost_ioc_quad here.
            cost_ioc_quad.update(self.demoU, self.demoX, self.demoO, demos_logiw, sampleU_arr, sampleX_arr, \
                                                    sampleO_arr, samples_logiw)



    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        multiplier = self._hyperparams["max_ent_traj"]
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
