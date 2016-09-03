""" This file generates for a point mass for 4 starting positions and a single goal position. """

import matplotlib as mpl

mpl.use('Qt4Agg')

import sys
import os
import os.path
import logging
import copy
import argparse
import time
import threading
import numpy as np
import scipy as sp
import scipy.io
import numpy.matlib
import random
from random import shuffle

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_utils import gauss_fit_joint_prior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
                END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
                END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
                CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy  # Maybe useful if we unpickle the file as controllers

class GenDemo(object):
        """ Generator of demos. """
        def __init__(self, config):
                self._hyperparams = config
                self._conditions = config['common']['conditions']

                # if 'train_conditions' in config['common']:
                #       self._train_idx = config['common']['train_conditions']
                #       self._test_idx = config['common']['test_conditions']
                # else:
                #       self._train_idx = range(self._conditions)
                #       config['common']['train_conditions'] = config['common']['conditions']
                #       self._hyperparams=config
                #       self._test_idx = self._train_idx
                self._exp_dir = config['common']['demo_exp_dir']
                self._data_files_dir = config['common']['data_files_dir']
                self._algorithm_files_dir = config['common']['demo_controller_file']
                self.data_logger = DataLogger()

        def generate(self):
                """
                 Generate demos and save them in a file for experiment.
                 Returns: None.
                """
                # Load the algorithm
                import pickle

                algorithm_file = self._algorithm_files_dir # This should give us the optimal controller. Maybe set to 'controller_itr_%02d.pkl' % itr_load will be better?
                # algorithm_file = self._exp_dir
                # algorithm_files = self._algorithm_files_dir
                # self.algorithms = [] # A list of neural nets.
                # for i in range(4):
                #       algorithm = pickle.load(open(algorithm_files[i]))
                #       self.algorithms.append(algorithm)
                self.algorithm = pickle.load(open(algorithm_file)) # Just one demo algorithm for now.
                if self.algorithm is None:
                        print("Error: cannot find '%s.'" % algorithm_file)
                        os._exit(1) # called instead of sys.exit(), since t
                        # if algorithm is None:
                        #       print("Error: cannot find '%s.'" % algorithm_file)
                        #       os._exit(1) # called instead of sys.exit(), since t

                # Keep the initial states of the agent the sames as the demonstrations.
                if 'learning_from_prior' in self._hyperparams['algorithm']:
                        self._learning = self._hyperparams['algorithm']['learning_from_prior'] # if the experiment is learning from prior experience
                else:
                        self._learning = False
                agent_config = self._hyperparams['demo_agent']
                if agent_config['type']==AgentMuJoCo and agent_config['filename'] == './mjc_models/pr2_arm3d.xml' and not self._learning:
                        agent_config['x0'] = self.algorithm._hyperparams['agent_x0']
                        agent_config['pos_body_idx'] = self.algorithm._hyperparams['agent_pos_body_idx']
                        agent_config['pos_body_offset'] = self.algorithm._hyperparams['agent_pos_body_offset']
                self.agent = agent_config['type'](agent_config)

                # Roll out the demonstrations from controllers
                var_mult = self._hyperparams['algorithm']['demo_var_mult']
                T = self.algorithm.T
                # T = self.algorithms[0].T
                demos = []

                M = agent_config['conditions']
                N = self._hyperparams['algorithm']['num_demos']
                sampled_demo_conds = [random.randint(0, M-1) for i in xrange(M)]
                sampled_demos = []
                if not self._learning:
                        controllers = {}

                        # Store each controller under M conditions into controllers.
                        for i in xrange(M):
                                controllers[i] = self.algorithm.cur[i].traj_distr
                        controllers_var = copy.copy(controllers)
                        for i in xrange(M):

                                # Increase controller variance.
                                controllers_var[i].chol_pol_covar *= var_mult
                                # Gather demos.
                                for j in xrange(N):
                                        demo = self.agent.sample(
                                                controllers_var[i], i,
                                                verbose=(i < self.algorithm._hyperparams['demo_verbose']),
                                                save = True
                                        )
                                        demos.append(demo)
                else:
                        # demos = {i : [] for i in xrange(4)} # Take demos for 4 nn policies
                        # Extract the neural network policy.
                        for j in xrange(self.algorithm.num_policies):
                                pol = self.algorithm.policy_opts[j].policy
                                pol.chol_pol_covar *= var_mult

                                for i in range(M / self.algorithm.num_policies * j, M / self.algorithm.num_policies * (j + 1)):
                                        # Gather demos.
                                        samples = []
                                        dists = []
                                        # for _ in xrange(5):
                                        #       sample = self.agent.sample(
                                        #               pol, i,
                                        #               verbose=(i < self._hyperparams['verbose_trials'])
                                        #               )
                                        #       # if i in sampled_demo_conds:
                                        #       #       sampled_demos.append(demo)
                                        #       samples.append(sample)
                                        #       sample_end_effector = sample.get(END_EFFECTOR_POINTS)
                                        #       target_position = agent_config['target_end_effector'][:3]
                                        #       dists.append(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1])
                                        # if sum(dists) / len(dists) <= 0.3:
                                        #       controller = self.linearize_policy(SampleList(samples), j)
                                        for k in xrange(N):
                                                demo = self.agent.sample(
                                                                pol, i, # Should be changed back to controller if using linearization
                                                                verbose=(i < self._hyperparams['verbose_trials']), noisy=False
                                                                ) # Add noise seems not working. TODO: figure out why
                                                # demos.append(demo)
                                                demos.append(demo)

                # Filter out worst (M - good_conds) demos.
                if agent_config['type']==AgentMuJoCo and agent_config['filename'] == './mjc_models/pr2_arm3d.xml':
                        target_position = agent_config['target_end_effector'][:3]
                        dists_to_target = np.zeros(M*N)
                        # dists_to_target = [np.zeros(M*N) for i in xrange(4)]
                        good_indices = []
                        failed_indices = []
                        # M = len(demos)/N
                        for i in xrange(M):
                                for j in xrange(N):
                                        demo_end_effector = demos[i*N + j].get(END_EFFECTOR_POINTS)
                                        # demo_end_effector = demos[k][i*N + j].get(END_EFFECTOR_POINTS)
                                        # dists_to_target[i] = np.amin(np.sqrt(np.sum((demo_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
                                        # Just choose the last time step since it may become unstable after achieving the minimum point.
                                        dists_to_target[i*N + j] = np.sqrt(np.sum((demo_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1]
                                        # dists_to_target[k][i*N + j] = np.sqrt(np.sum((demo_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1]
                                        if dists_to_target[i*N + j] > agent_config['success_upper_bound']:
                                                failed_indices.append(i)
                        good_indices = [i for i in xrange(M) if i not in failed_indices]
                        bad_indices = np.argmax(dists_to_target)
                        import pdb; pdb.set_trace()
                        self._hyperparams['algorithm']['demo_cond'] = len(good_indices)
                        filtered_demos = []
                        demo_conditions = []
                        failed_conditions = []
                        exp_dir = self._data_files_dir.replace("data_files", "")
                        with open(exp_dir + 'log.txt', 'a') as f:
                                f.write('\nThe demo conditions are: \n')
                        for i in good_indices:
                                for j in xrange(N):
                                        filtered_demos.append(demos[i*N + j])
                                demo_conditions.append(agent_config['pos_body_offset'][i])
                                with open(exp_dir + 'log.txt', 'a') as f:
                                        f.write('\n' + str(agent_config['pos_body_offset'][i]) + '\n')
                        with open(exp_dir + 'log.txt', 'a') as f:
                                f.write('\nThe failed badmm conditions are: \n')
                        for i in xrange(M):
                                if i not in good_indices:
                                        failed_conditions.append(agent_config['pos_body_offset'][i])
                                        with open(exp_dir + 'log.txt', 'a') as f:
                                                f.write('\n' + str(agent_config['pos_body_offset'][i]) + '\n')
                        shuffle(filtered_demos)
                        demo_list =  SampleList(filtered_demos)
                        demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs(), \
                                                        'demo_conditions': demo_conditions, 'failed_conditions': failed_conditions}
                        if self._learning:
                                demo_store['pos_body_offset'] = [agent_config['pos_body_offset'][bad_indices]]
                        import matplotlib.pyplot as plt
                        from matplotlib.patches import Rectangle

                        plt.close()
                        demo_conditions_x = [demo_conditions[i][0] for i in xrange(len(demo_conditions))]
                        demo_conditions_y = [demo_conditions[i][1] for i in xrange(len(demo_conditions))]
                        failed_conditions_x = [failed_conditions[i][0] for i in xrange(len(failed_conditions))]
                        failed_conditions_y = [failed_conditions[i][1] for i in xrange(len(failed_conditions))]
                        subplt = plt.subplot()
                        subplt.plot(demo_conditions_x, demo_conditions_y, 'go')
                        subplt.plot(failed_conditions_x, failed_conditions_y, 'rx')
                        ax = plt.gca()
                        ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue'))
                        box = subplt.get_position()
                        subplt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
                        subplt.legend(['demo_cond', 'failed_badmm'], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
                                                        shadow=True, ncol=2)
                        plt.title("Distribution of demo conditions")
                        # plt.xlabel('width')
                        # plt.ylabel('length')
                        plt.savefig(self._data_files_dir + 'distribution_of_demo_conditions_multiple.png')
                        plt.close()
                else:
                        shuffle(demos)
                        demo_list = SampleList(demos)
                        demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs()}
                # Save the demos.
                self.data_logger.pickle(
                        self._data_files_dir + 'demos.pkl',
                        copy.copy(demo_store)
                )

        def linearize_policy(self, samples, cond):
                policy_prior = self.algorithms[cond]._hyperparams['policy_prior']
                init_policy_prior = policy_prior['type'](policy_prior)
                init_policy_prior._hyperparams['keep_samples'] = False
                dX, dU, T = self.algorithms[cond].dX, self.algorithms[cond].dU, self.algorithms[cond].T
                N = len(samples)
                X = samples.get_X()
                pol_mu, pol_sig = self.algorithms[cond].policy_opt.prob(samples.get_obs().copy())[:2]
                # Update the policy prior with collected samples
                init_policy_prior.update(SampleList([]), self.algorithms[cond].policy_opt, samples)
                # Collapse policy covariances. This is not really correct, but
                # it works fine so long as the policy covariance doesn't depend
                # on state.
                pol_sig = np.mean(pol_sig, axis=0)
                pol_info_pol_K = np.zeros((T, dU, dX))
                pol_info_pol_k = np.zeros((T, dU))
                pol_info_pol_S = np.zeros((T, dU, dU))
                pol_info_chol_pol_S = np.zeros((T, dU, dU))
                pol_info_inv_pol_S = np.empty_like(pol_info_chol_pol_S)
                # Estimate the policy linearization at each time step.
                for t in range(T):
                        # Assemble diagonal weights matrix and data.
                        dwts = (1.0 / N) * np.ones(N)
                        Ts = X[:, t, :]
                        Ps = pol_mu[:, t, :]
                        Ys = np.concatenate((Ts, Ps), axis=1)
                        # Obtain Normal-inverse-Wishart prior.
                        mu0, Phi, mm, n0 = init_policy_prior.eval(Ts, Ps)
                        sig_reg = np.zeros((dX+dU, dX+dU))
                        # On the first time step, always slightly regularize covariance.
                        if t == 0:
                                sig_reg[:dX, :dX] = 1e-8 * np.eye(dX)
                        # Perform computation.
                        pol_K, pol_k, pol_S = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0,
                                                                                                                dwts, dX, dU, sig_reg)
                        pol_S += pol_sig[t, :, :]
                        pol_info_pol_K[t, :, :], pol_info_pol_k[t, :] = pol_K, pol_k
                        pol_info_pol_S[t, :, :], pol_info_chol_pol_S[t, :, :] = \
                                        pol_S, sp.linalg.cholesky(pol_S)
                        pol_info_inv_pol_S[t, :, :] = sp.linalg.solve(
                                                                                        pol_info_chol_pol_S[t, :, :],
                                                                                        np.linalg.solve(pol_info_chol_pol_S[t, :, :].T, np.eye(dU))
                                                                                        )
                return LinearGaussianPolicy(pol_info_pol_K, pol_info_pol_k, pol_info_pol_S, pol_info_chol_pol_S, \
                                                                                pol_info_inv_pol_S)

