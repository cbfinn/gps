""" This file defines code for PI2-based trajectory optimization.

Optimization of trajectories with PI2 and a REPS-like KL-divergence constraint. 
References:
[1] E. Theodorou, J. Buchli, and S. Schaal. A generalized path integral control 
    approach to reinforcement learning. JMLR, 11, 2010.
[2] F. Stulp and O. Sigaud. Path integral policy improvement with covariance 
    matrix adaptation. In ICML, 2012.
[3] J. Peters, K. Mulling, and Y. Altun. Relative entropy policy search. 
    In AAAI, 2010.
 """
import copy
import logging
import numpy as np
import scipy as sp

from numpy.linalg import LinAlgError
from scipy.optimize import minimize

from gps.algorithm.traj_opt.config import TRAJ_OPT_PI2
from gps.algorithm.traj_opt.traj_opt import TrajOpt

LOGGER = logging.getLogger(__name__)

class TrajOptPI2(TrajOpt):
    """ PI2 trajectory optimization.
    Hyperparameters:
        kl_threshold: KL-divergence threshold between old and new policies.
        covariance_damping: If greater than zero, covariance is computed as a
            multiple of the old covariance. Multiplier is taken to the power
            (1 / covariance_damping). If greater than one, slows down 
            convergence and keeps exploration noise high for more iterations.
        min_temperature: Minimum bound of the temperature optimiztion for the 
            soft-max probabilities of the policy samples.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(TRAJ_OPT_PI2)
        config.update(hyperparams)
        TrajOpt.__init__(self, config)
        self._kl_threshold = self._hyperparams['kl_threshold']
        self._covariance_damping = self._hyperparams['covariance_damping']
        self._min_temperature = self._hyperparams['min_temperature']
    
    def update(self, m, algorithm, use_lqr_actions=False,
               fixed_eta=None, use_fixed_eta=False, costs=None):
        """
        Perform optimization of the feedforward controls of time-varying
        linear-Gaussian controllers with PI2. 
        Args:
            m: Current condition number.
            algorithm: Currently used algorithm.
            use_lqr_actions: Whether or not to compute actions from LQR-updated
                controller.
            fixed_eta: Fixed value of eta to use if use_fixed_eta is True.
            use_fixed_eta: Whether to use fixed_eta or compute using KL dual.
            costs: Costs to update with, defaults to sampled costs.
        Returns:
            traj_distr: Updated linear-Gaussian controller.
        """
        # We only use the scalar costs.      
        if costs is None:
            costs = algorithm.cur[m].cs

        # Get sampled controls, states and old trajectory distribution.
        cur_data = algorithm.cur[m].sample_list                
        prev_traj_distr = algorithm.cur[m].traj_distr
        X = cur_data.get_X()
        U = cur_data.get_U()
        T = prev_traj_distr.T

        # We only optimize feedforward controls with PI2. Subtract the feedback
        # part from the sampled controls using feedback gain matrix and states.       
        ffw_controls = np.zeros(U.shape)
        if use_lqr_actions:
            noise = cur_data.get_noise()
            for i in xrange(len(cur_data)):
                U_lqr = [prev_traj_distr.K[t].dot(X[i, t]) + prev_traj_distr.k[t] +
                         prev_traj_distr.chol_pol_covar[t].T.dot(noise[i, t])
                         for t in xrange(T)]
                ffw_controls[i] = [U_lqr[t] - prev_traj_distr.K[t].dot(X[i, t])
                                   for t in xrange(T)]
        else:
            for i in xrange(len(cur_data)):
                ffw_controls[i] = [U[i, t] - prev_traj_distr.K[t].dot(X[i, t])
                                   for t in xrange(T)]

        # Copy feedback gain matrix from the old trajectory distribution.                       
        traj_distr = prev_traj_distr.nans_like()
        traj_distr.K = prev_traj_distr.K

        # Optimize feedforward controls and covariances with PI2.
        k, pS, ipS, cpS, eta = self.update_pi2(
            ffw_controls, costs, prev_traj_distr.k, prev_traj_distr.pol_covar,
            fixed_eta, use_fixed_eta
        )
        traj_distr.k, traj_distr.pol_covar = k, pS
        traj_distr.inv_pol_covar, traj_distr.chol_pol_covar = ipS, cpS

        return traj_distr, eta

    def update_pi2(self, samples, costs, mean_old, cov_old,
                   fixed_eta=None, use_fixed_eta=False):
        """
        Perform optimization with PI2. Computes new mean and covariance matrices
        of the policy parameters given policy samples and their costs.
        Args:
            samples: Matrix of policy samples with dimensions: 
                     [num_samples x num_timesteps x num_controls].
            costs: Matrix of roll-out costs with dimensions:
                   [num_samples x num_timesteps]
            mean_old: Old policy mean.
            cov_old: Old policy covariance.            
            fixed_eta: Fixed value of eta to use if use_fixed_eta is True.
            use_fixed_eta: Whether to use fixed_eta or compute using KL dual.
        Returns:
            mean_new: New policy mean.
            cov_new: New policy covariance.
            inv_cov_new: Inverse of the new policy covariance.
            chol_cov_new: Cholesky decomposition of the new policy covariance.
        """
        mean_new = np.zeros(mean_old.shape)
        cov_new = np.zeros(cov_old.shape)
        inv_cov_new = np.zeros(cov_old.shape)
        chol_cov_new = np.zeros(cov_old.shape)

        # Iterate over time steps.
        T = samples.shape[1]
        etas = np.zeros(T)

        del_ = self._hyperparams['del0']
        if self._hyperparams['pi2_cons_per_step']:
            del_ = np.ones(T) * del_

        fail = True
        while fail:
            fail = False
            for t in xrange(T):

                # Compute cost-to-go for each time step for each sample.
                cost_to_go = np.sum(costs[:, t:T], axis=1)

                if use_fixed_eta:
                    eta = (fixed_eta[t] if isinstance(fixed_eta, np.ndarray)
                           else fixed_eta)
                else:
                    # Perform REPS-like optimization of the temperature eta.
                    res = minimize(self.kl_dual, 10.0,
                                   bounds=((self._min_temperature, None),),
                                   args=(self._kl_threshold, cost_to_go))
                    etas[t] = res.x
                    eta = res.x

                exponent = -cost_to_go

                exp_cost = np.exp((exponent - np.max(exponent)) / eta)
                prob = exp_cost / np.sum(exp_cost)

                # Update policy mean with weighted max-likelihood.
                mean_new[t] = np.sum(prob[:, np.newaxis] * samples[:, t], axis=0)

                # Update policy covariance with weighted max-likelihood.
                for i in xrange(samples.shape[0]):
                    mean_diff = samples[i, t] - mean_new[t]
                    mean_diff = np.reshape(mean_diff, (len(mean_diff), 1))
                    cov_new[t] += prob[i] * np.dot(mean_diff, mean_diff.T)

                # If covariance damping is enabled, compute covariance as multiple
                # of the old covariance. The multiplier is first fitted using
                # max-likelihood and then taken to the power (1/covariance_damping).
                if(self._covariance_damping is not None
                   and self._covariance_damping > 0.0):

                    mult = np.trace(np.dot(sp.linalg.inv(cov_old[t]),
                        cov_new[t])) / len(cov_old[t])
                    mult = np.power(mult, 1 / self._covariance_damping)
                    cov_new[t] = mult * cov_old[t]

                # Compute covariance inverse and cholesky decomposition.
                try:
                    inv_cov_new[t] = sp.linalg.inv(cov_new[t])
                    chol_cov_new[t] = sp.linalg.cholesky(cov_new[t])
                except LinAlgError:
                    fail = True
                    old_eta = eta
                    eta = fixed_eta if use_fixed_eta else etas
                    eta[t] += del_[t]
                    LOGGER.debug('Increasing eta %d: %f -> %f', t, old_eta, eta[t])
                    del_[t] *= 2
                    if eta[t] >= 1e16:
                        raise ValueError

        return mean_new, cov_new, inv_cov_new, chol_cov_new, etas

    def kl_dual(self, eta, kl_threshold, costs):
        """
        Dual function for optimizing the temperature eta according to the given
        KL-divergence constraint.
        
        Args:
            eta: Temperature that has to be optimized.
            kl_threshold: Max. KL-divergence constraint.
            costs: Roll-out costs.            
        Returns:
            Value of the dual function.
        """
        max_costs = np.max(-costs)
        exponent = -costs - max_costs
        return (eta * kl_threshold + max_costs
                + eta * np.log((1.0 / len(costs)) *
                np.sum(np.exp(exponent / eta))))
