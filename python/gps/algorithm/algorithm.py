""" This file defines the base algorithm class. """

import abc
import copy
import logging

import numpy as np
from numpy.linalg import LinAlgError

from gps.algorithm.config import ALG
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition, disable_caffe_logs
from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList


LOGGER = logging.getLogger(__name__)


class Algorithm(object):
    """ Algorithm superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config

        if 'train_conditions' in hyperparams:
            self._cond_idx = hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = hyperparams['conditions']
            self._cond_idx = range(self.M)
        self.iteration_count = 0

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO

        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU
        if self._hyperparams['ioc']:
            init_traj_distr['x0'] = np.zeros(self.dX)
        del self._hyperparams['agent']  # Don't want to pickle this.

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]
        self.traj_distr = {self.iteration_count: []}
        self.traj_info = {self.iteration_count: []}
        self.kl_div = {self.iteration_count:[]}
        self.dists_to_target = {self.iteration_count:[]}
        self.sample_list = {i: SampleList([]) for i in range(self.M)}

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            dynamics = self._hyperparams['dynamics']
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)
            self.traj_distr[self.iteration_count].append(self.cur[m].traj_distr)
            self.traj_info[self.iteration_count].append(self.cur[m].traj_info)

        self.traj_opt = hyperparams['traj_opt']['type'](
            hyperparams['traj_opt']
        )
        self.cost = [
            hyperparams['cost']['type'](hyperparams['cost'])
            for _ in range(self.M)
        ]
        if self._hyperparams['ioc']:
            self.gt_cost = [
                hyperparams['gt_cost']['type'](hyperparams['gt_cost'])
                for _ in range(self.M)
            ]
        self.base_kl_step = self._hyperparams['kl_step']

    @abc.abstractmethod
    def iteration(self, sample_list):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        for cond in range(self.M):
            if self.iteration_count >= 1:
                self.prev[cond].traj_info.dynamics = \
                        self.cur[cond].traj_info.dynamics.copy()
            cur_data = self.cur[cond].sample_list
            self.cur[cond].traj_info.dynamics.update_prior(cur_data)

            self.cur[cond].traj_info.dynamics.fit(cur_data)

            init_X = cur_data.get_X()[:, 0, :]
            x0mu = np.mean(init_X, axis=0)
            self.cur[cond].traj_info.x0mu = x0mu
            self.cur[cond].traj_info.x0sigma = np.diag(
                np.maximum(np.var(init_X, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[cond].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[cond].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = \
                    self.traj_opt.update(cond, self)

    def _eval_cost(self, cond, prev_cost=False):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
            prev: Whether or not to use previous_cost (for ioc stepadjust)
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU

        synN = self._hyperparams['synthetic_cost_samples']
        if synN > 0:
            agent = self.cur[cond].sample_list.get_samples()[0].agent
            X, U, _ = self._traj_samples(cond, synN)
            syn_samples = []
            for i in range(synN):
                sample = Sample(agent)
                sample.set_XU(X[i, :, :], U[i, :, :])
                syn_samples.append(sample)
            all_samples = SampleList(syn_samples +
                self.cur[cond].sample_list.get_samples())
        else:
          all_samples = self.cur[cond].sample_list
        N = len(all_samples)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        if self._hyperparams['ioc']:
            cgt = np.zeros((N, T))
        for n in range(N):
            sample = all_samples[n]
            # Get costs.
            if prev_cost:
              l, lx, lu, lxx, luu, lux = self.previous_cost[cond].eval(sample)
            else:
              l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            # Compute the ground truth cost
            if self._hyperparams['ioc'] and n >= synN:
                l_gt, _, _, _, _, _ = self.gt_cost[cond].eval(sample)
                cgt[n, :] = l_gt
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        if prev_cost:
          traj_info = self.cur[cond].prevcost_traj_info
          traj_info.dynamics = self.cur[cond].traj_info.dynamics
          traj_info.x0sigma = self.cur[cond].traj_info.x0sigma
          traj_info.x0mu = self.cur[cond].traj_info.x0mu
        else:
          traj_info = self.cur[cond].traj_info
          self.cur[cond].cs = cs[synN:]  # True value of cost.
        traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        if self._hyperparams['ioc']:
            self.cur[cond].cgt = cgt[synN:]


    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        self.cur = [IterationData() for _ in range(self.M)]
        self.traj_distr[self.iteration_count] = []
        self.traj_info[self.iteration_count] = []
        self.kl_div[self.iteration_count] = []
        self.dists_to_target[self.iteration_count] = []
        self.previous_cost = []
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
            self.traj_distr[self.iteration_count].append(self.new_traj_distr[m])
            self.traj_info[self.iteration_count].append(self.cur[m].traj_info)
            if self._hyperparams['ioc']:
              self.cur[m].prevcost_traj_info = TrajectoryInfo()
              self.previous_cost.append(self.cost[m].copy())
        delattr(self, 'new_traj_distr')

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent

    def _traj_samples(self, condition, N):
        """
        Sample from a particular trajectory distribution,
        under the estimated dynamics.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU

        X = self.cur[condition].sample_list.get_X()
        U = self.cur[condition].sample_list.get_U()
        traj_info = self.cur[condition].traj_info
        traj_distr = self.cur[condition].traj_distr

        # Allocate space.
        pX = np.zeros((N, T, dX))
        pU = np.zeros((N, T, dU))
        pProb = np.zeros((N, T))
        mu, sigma = self.traj_opt.forward(traj_distr, traj_info)
        for t in xrange(T):
            samps = np.random.randn(dX, N)
            sigma[t, :dX, :dX] = 0.5 * (sigma[t, :dX, :dX] + sigma[t, :dX, :dX].T)
            # Full version. Assuming policy synthetic samples distro fix bound is 2.
            var_limit = 2 #self._hyperparams['policy_synthetic_samples_distro_fix_bound'] # Assuming to be 2
            pemp = np.maximum(np.mean(X[:, t, :] - mu[t, :dX], 0), 1e-3)
            sigt = np.diag(1 / np.sqrt(pemp)).dot(sigma[t, :dX, :dX]).dot(np.diag(1 / np.sqrt(pemp)))
            val, vec = np.linalg.eig(sigt)
            val = np.diag(val)
            val = np.minimum(val, var_limit)
            sigt = vec.dot(val).dot(vec.T)
            sigma[t, :dX, :dX] = np.diag(np.sqrt(pemp)).dot(sigt).dot(np.diag(np.sqrt(pemp)))
            # Fix small eigenvalues only.
            val, vec = np.linalg.eig(sigma[t, :dX, :dX])
            val = np.maximum(np.real(val), 1e-6)
            sigma[t, :dX, :dX] = vec.dot(np.diag(val)).dot(vec.T)

            # Store sample probabilities.
            pProb[:, t] = -0.5 * np.sum(samps**2, 0) - 0.5 * np.sum(np.log(val))
            # Draw samples.
            try:
                samps = mu[t, :dX].reshape(dX, 1) + np.linalg.cholesky(sigma[t, :dX, :dX]).T.dot(samps)
            except LinAlgError as e:
                LOGGER.debug('Policy sample matrix is not positive definite.')
                _, L = np.linalg.qr(np.sqrt(np.diag(val)).dot(vec.T))
                samps = mu[t, :dX].reshape(dX, 1) + L.T.dot(samps)
            pX[:, t, :] = samps.T
            pU[:, t, :] = (traj_distr.K[t, :, :].dot(samps) + traj_distr.k[t, :].reshape(dU, 1) + \
                            traj_distr.chol_pol_covar[t, :, :].T.dot(np.random.randn(dU, N))).T
        return pX, pU, pProb
