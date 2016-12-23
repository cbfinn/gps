""" This file defines the base algorithm class. """

import abc
import copy

import numpy as np
from numpy.linalg import LinAlgError

from gps.algorithm.config import ALG
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility import ColorLogger
from gps.utility.general_utils import extract_condition, disable_caffe_logs, Timer
from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList
from gps.utility.general_utils import logsum
from gps.algorithm.algorithm_utils import fit_emp_controller

LOGGER = ColorLogger(__name__)

class Algorithm(object):
    """ Algorithm superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config

        if 'train_conditions' in self._hyperparams:
            self._cond_idx = self._hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = self._hyperparams['conditions']
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
        self.policy_opts = {self.iteration_count: None}
        self.traj_info = {self.iteration_count: []}
        self.kl_div = {self.iteration_count:[]}
        self.dists_to_target = {self.iteration_count:[]}
        self.sample_list = {i: SampleList([]) for i in range(self.M)}

        dynamics = self._hyperparams['dynamics']
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            init_traj_distr['condition'] = m
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)
            self.traj_info[self.iteration_count].append(self.cur[m].traj_info)

        self.traj_opt = self._hyperparams['traj_opt']['type'](
            self._hyperparams['traj_opt']
        )

        if type(hyperparams['cost']) == list:
            self.cost = [
                hyperparams['cost'][i]['type'](hyperparams['cost'][i])
                for i in range(hyperparams['conditions'])]
        else:
            self.cost = self._hyperparams['cost']['type'](self._hyperparams['cost'])

        if type(self._hyperparams['cost']) is dict and self._hyperparams['cost'].get('agent', False):
            del self._hyperparams['cost']['agent']

        if self._hyperparams['ioc']:
            if type(hyperparams['gt_cost']) == list:
                self.gt_cost = [
                    self._hyperparams['gt_cost'][i]['type'](self._hyperparams['gt_cost'][i])
                    for i in range(self.M)
                ]
            else:
                self.gt_cost = [
                    self._hyperparams['gt_cost']['type'](self._hyperparams['gt_cost'])
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
        LOGGER.info('Fitting dynamics.')
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data.get_X()
            U = cur_data.get_U()

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(cur_data)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        LOGGER.info('Updating trajectories.')
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

        gt_conditions = self._hyperparams.get('gt_conditions', [])

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
            cost = self.cost
            if prev_cost:
                cost = self.previous_cost

            if cond in gt_conditions:
                if isinstance(self.gt_cost, list):
                    cost = self.gt_cost[cond]
                else:
                    cost = self.gt_cost

            if type(self.cost) != list:
                l, lx, lu, lxx, luu, lux = cost.eval(sample)
            else:
                l, lx, lu, lxx, luu, lux = cost[cond].eval(sample)

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
          self.cur[cond].prevcost_cs = cs[synN:]  # True value of cost.
        else:
          traj_info = self.cur[cond].traj_info
          self.cur[cond].cs = cs[synN:]  # True value of cost.
        traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        if self._hyperparams['ioc']:
            self.cur[cond].cgt = cgt[synN:]

    def _advance_iteration_variables(self, store_prev=False):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
            if store_prev:
                self.prev[m].sample_list = self.cur[m].sample_list
            else:
                self.prev[m].sample_list = True # don't pickle this.
        self.cur = [IterationData() for _ in range(self.M)]
        with Timer('Algorithm._advance_iteration_variables policy_opt_copy'):
            new_policy_opt = self.policy_opt.copy()
        self.policy_opts[self.iteration_count] = new_policy_opt
        self.traj_info[self.iteration_count] = []
        self.kl_div[self.iteration_count] = []
        self.dists_to_target[self.iteration_count] = []
        if self._hyperparams['ioc']:
            with Timer('Algorithm._advance_iteration_variables cost_copy'):
                self.previous_cost = self.cost.copy()
        else:
            self.previous_cost = []
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
            self.traj_info[self.iteration_count].append(self.cur[m].traj_info)
            if self._hyperparams['ioc']:
              self.cur[m].prevcost_traj_info = TrajectoryInfo()
        delattr(self, 'new_traj_distr')
        del self.traj_info[self.iteration_count-1]

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
            # TODO - maybe symmetrize sigma?
            val, vec = np.linalg.eig(sigma[t, :dX, :dX])
            if np.any(val < 1e-6):
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

    def _update_cost(self):
        """ Update the cost objective in each iteration. """

        # Estimate the importance weights for fusion distributions.
        demos_logiw, samples_logiw = self.importance_weights()

        # Update the learned cost
        # Transform dictionaries to arrays
        M = len(self.prev)
        Md = self._hyperparams['demo_M']
        assert Md == 1
        samples_logiw = {i: samples_logiw[i].reshape((-1, 1)) for i in xrange(M)}
        demos_logiw = {i: demos_logiw[i].reshape((-1, 1)) for i in xrange(M)}
        # TODO - make these changes in other algorithm objects too.
        sampleU_arr = np.vstack((self.sample_list[i].get_U() for i in xrange(M)))
        sampleO_arr = np.vstack((self.sample_list[i].get_obs() for i in xrange(M)))
        samples_logiw_arr = np.hstack([samples_logiw[i] for i in xrange(M)]).reshape((-1, 1))
        # TODO - this is a weird hack that is wrong, and has been in the code for awhile.
        demos_logiw_arr = demos_logiw[0].reshape((-1, 1))  # np.hstack([demos_logiw[i] for i in xrange(Md)]).reshape((-1, 1))
        self.cost.update(self.demoU, self.demoO, demos_logiw_arr, sampleU_arr,
                                                    sampleO_arr, samples_logiw_arr, itr=self.iteration_count)

    def importance_weights(self):
        """
            Estimate the importance weights for fusion distributions.
        """
        # TODO - fusion distribution is incorrect if state space is changing (via e2e feature learning)
        itr = self.iteration_count
        M = len(self.prev)
        ix = range(self.dX)
        iu = range(self.dX, self.dX + self.dU)
        init_samples = self._hyperparams['init_samples']
        # itration_count + 1 distributions to evaluate
        # T: summed over time
        samples_logprob, demos_logprob = {}, {}
        # number of demo distributions
        Md = self._hyperparams['demo_M']  # always 1
        demos_logiw, samples_logiw = {}, {}
        demoU = {i: self.demoU for i in xrange(M)}
        demoO = {i: self.demoO for i in xrange(M)}
        demoX = {i: self.demoX for i in xrange(M)}

        # For IOC+vision, recompute demoX here using self.cost. Assumes
        # that the features are the last part of the state and that the dynamics
        # are fit to the feature encoder in the cost.
        #if 'get_features' in dir(self.cost):
        #  for m in range(M):
        #    for samp in range(demoO[m].shape[0]):
        #        demoFeat = self.cost.get_features(demoO[m][samp])
        #        dF = demoFeat.shape[1]
        #        demoX[m][samp, :,-dF:] = demoFeat
        self.demo_traj = {}

        # estimate demo distributions empirically when not initializing from demo policy
        for i in xrange(Md):  # always 1
            if self._hyperparams['demo_distr_empest']:
                # Why does this not use the dynamics?
                # Answer: demoU is estimated using the dynamics if demoU are unknown. (But this is unimplemented)
                self.demo_traj[i] = fit_emp_controller(demoX[i], demoU[i])

        for i in xrange(M):
            # This code assumes a fixed number of samples per iteration/controller
            if self._hyperparams['ioc'] == 'ICML':
                samples_logprob[i] = np.zeros((itr + Md + 1, self.T, (self.N / M) * itr + init_samples))
                demos_logprob[i] = np.zeros((itr + Md + 1, self.T, demoX[i].shape[0]))
            else:
                raise NotImplementedError("Other type of ioc losses not implemented.")

            sample_i_X = self.sample_list[i].get_X()
            sample_i_U = self.sample_list[i].get_U()
            samples_per_iter = sample_i_X.shape[0] / (itr+1)
            # Evaluate sample prob under sample distributions
            for itr_i in xrange(itr + 1):
                traj = self.traj_distr[itr_i][i]
                for j in xrange(sample_i_X.shape[0]):
                    for t in xrange(self.T - 1):
                        diff = traj.k[t, :] + \
                                traj.K[t, :, :].dot(sample_i_X[j, t, :]) - sample_i_U[j, t, :]
                        samples_logprob[i][itr_i, t, j] = -0.5 * np.sum(diff * (traj.inv_pol_covar[t, :, :].dot(diff))) - \
                                                        np.sum(np.log(np.diag(traj.chol_pol_covar[t, :, :])))
            # Evaluate sample prob under demo distribution.
            if self._hyperparams['ioc'] == 'ICML':
              for m in xrange(Md):
                for j in range(sample_i_X.shape[0]):
                    for t in xrange(self.T - 1):
                        diff = self.demo_traj[m].k[t, :] + \
                                self.demo_traj[m].K[t, :, :].dot(sample_i_X[j, t, :]) - sample_i_U[j, t, :]
                        samples_logprob[i][itr + 1 + m, t, j] = -0.5 * np.sum(diff * (self.demo_traj[m].inv_pol_covar[t, :, :].dot(diff))) - \
                                                    np.sum(np.log(np.diag(self.demo_traj[m].chol_pol_covar[t, :, :])))
            else:
                raise NotImplementedError("Other type of ioc losses not implemented.")
            # Sum over the distributions and time.
            samples_logiw[i] = logsum(np.sum(samples_logprob[i], 1), 0)

        # Assume only one condition for the demos.
        assert Md == 1; idx = 0
        for i in xrange(M):  # NOTE before this was xrange(1)
            # Evaluate demo prob. under sample distributions.
            for itr_i in xrange(itr + 1):
                traj = self.traj_distr[itr_i][i]
                for j in xrange(demoX[idx].shape[0]):
                    for t in xrange(self.T - 1):
                        diff = traj.k[t, :] + \
                                traj.K[t, :, :].dot(demoX[idx][j, t, :]) - demoU[idx][j, t, :]
                        demos_logprob[i][itr_i, t, j] = -0.5 * np.sum(diff * (traj.inv_pol_covar[t, :, :].dot(diff))) - \
                                                        np.sum(np.log(np.diag(traj.chol_pol_covar[t, :, :])))
            # Evaluate demo prob. under demo distributions.
            if self._hyperparams['ioc'] == 'ICML':
              for m in xrange(Md):
                for j in range(demoX[idx].shape[0]):
                    for t in xrange(self.T - 1):
                        diff = self.demo_traj[m].k[t, :] + \
                                self.demo_traj[m].K[t, :, :].dot(demoX[idx][j, t, :]) - demoU[idx][j, t, :]
                        demos_logprob[i][itr + 1 + m, t, j] = -0.5 * np.sum(diff * (self.demo_traj[m].inv_pol_covar[t, :, :].dot(diff)), 0) - \
                                                        np.sum(np.log(np.diag(self.demo_traj[m].chol_pol_covar[t, :, :])))
            else:
                raise NotImplementedError("Other type of ioc losses not implemented.")
            demos_logiw[i] = logsum(np.sum(demos_logprob[i], 1), 0)

        return demos_logiw, samples_logiw

    """
    def __getstate__(self):
        print 'Getting state algorithm'
        for key in ['init_traj_distr', 'cost', 'gt_cost']:
            if key in self._hyperparams:
                del self._hyperparams[key]
        return {'traj_distr': self.traj_distr,
                'cur': self.cur,
                'prev': self.prev,
                '_hyperparams': self._hyperparams}

    def __setstate__(self, state):
        #import pdb; pdb.set_trace()
        #type(self).__init__(self, state['_hyperparams'])
        self._hyperparams = state['_hyperparams']
        self.traj_distr = state['traj_distr']
        self.cur = state['cur']
        self.prev = state['prev']
    """




