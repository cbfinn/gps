""" This file defines the base algorithm class. """

import abc
import copy
import logging

import numpy as np

from gps.algorithm.config import ALG
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition
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
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
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
        del self._hyperparams['agent']  # Don't want to pickle this.

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        dynamics = self._hyperparams['dynamics']

        # Store traj_info per condition.
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        # If clustering, store traj_info per cluster and single dynamics GMM.
        if self._hyperparams['num_clusters']:
            self.dynamics = dynamics['type'](dynamics)
            self.cluster_traj_info = []
            for k in range(self._hyperparams['num_clusters']):
                self.cluster_traj_info.append(TrajectoryInfo())
                self.cluster_traj_info[k].dynamics = \
                        dynamics['type'](dynamics)

        self.traj_opt = hyperparams['traj_opt']['type'](
            hyperparams['traj_opt']
        )
        if type(hyperparams['cost']) == list:
            self.cost = [
                hyperparams['cost'][i]['type'](hyperparams['cost'][i])
                for i in range(hyperparams['conditions'])
            ]
        else:
            self.cost = [
                hyperparams['cost']['type'](hyperparams['cost'])
                for _ in range(hyperparams['conditions'])
            ]
        self.base_kl_step = self._hyperparams['kl_step']

    @abc.abstractmethod
    def iteration(self, sample_list):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _get_samples(self, idxs=None):
        """
        Helper function for extracting from 'self.cur'.
        """
        if idxs is None:
            idxs = range(self.M)
        samples = [self.cur[i].sample_list.get_samples() for i in idxs]
        return SampleList(sum(samples, []))

    def _update_dynamics(self, update_prior=True):
        """
        Update dynamics linearizations (dispatch for conditions/clusters).
        """
        if self._hyperparams['num_clusters']:
            K = self._hyperparams['num_clusters']
            if update_prior:
                all_samples = self._get_samples()
                self.dynamics.update_prior(all_samples)
            for k in range(K):
                self._update_cluster_dynamics(k)
        else:
            for m in range(self.M):
                self._update_condition_dynamics(m, update_prior)

    def _update_condition_dynamics(self, m, update_prior):
        """
        Update dynamics linearizations for condition m.
        """
        samples = self.cur[m].sample_list
        X = samples.get_X()
        U = samples.get_U()

        traj_info = self.cur[m].traj_info
        if update_prior:
            traj_info.dynamics.update_prior(samples)
        traj_info.dynamics.fit(X, U) # Stores Fm/fv/dyn_covar.

        # Fit x0mu/x0sigma and store.
        x0 = X[:, 0, :]
        x0mu = np.mean(x0, axis=0)
        x0sigma = np.diag(
            np.maximum(np.var(x0, axis=0),
                       self._hyperparams['initial_state_var'])
        )
        prior = self.cur[m].traj_info.dynamics.get_prior()
        if prior:
            mu0, Phi, priorm, n0 = prior.initial_state()
            N = len(samples)
            x0sigma += \
                    Phi + (N*priorm) / (N+priorm) * \
                    np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)
        traj_info.x0mu = x0mu
        traj_info.x0sigma = x0sigma

    def _update_cluster_dynamics(self, k):
        """
        Update dynamics linearizations for cluster k.
        """
        idxs = np.where(self.cluster_idx == k)[0]
        LOGGER.debug("Updating dynamics for cluster %d, idxs %s", k, idxs)
        if idxs.size == 0:
            return

        samples = self._get_samples(idxs)
        X = samples.get_X()
        U = samples.get_U()

        # Fit x0mu/x0sigma/dynamics and store cluster-level info.
        traj_info = self.cluster_traj_info[k]
        x0 = X[:, 0, :]
        traj_info.x0mu = np.mean(x0, axis=0)
        traj_info.x0sigma = np.diag(
            np.maximum(np.var(x0, axis=0),
                       self._hyperparams['initial_state_var'])
        )

        Fm, fv, dyn_covar = self.dynamics.fit(X, U)
        traj_info.dynamics.Fm = Fm
        traj_info.dynamics.fv = fv
        traj_info.dynamics.dyn_covar = dyn_covar

        # Copy to trajectory-level info.
        for i in idxs:
            self.cur[i].traj_info.x0mu = traj_info.x0mu
            self.cur[i].traj_info.x0sigma = traj_info.x0sigma
            self.cur[i].traj_info.dynamics.Fm = Fm
            self.cur[i].traj_info.dynamics.fv = fv
            self.cur[i].traj_info.dynamics.dyn_covar = dyn_covar

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

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[cond].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
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
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
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
