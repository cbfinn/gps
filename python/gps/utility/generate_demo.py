""" This file generates for a point mass for 4 starting positions and a single goal position. """

import logging
import copy
import scipy as sp
import scipy.io
import numpy.matlib
import random
import pickle
from random import shuffle

from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.utility.data_logger import DataLogger
from gps.utility.general_utils import compute_distance
from gps.sample.sample_list import SampleList

LOGGER = logging.getLogger(__name__)


class GenDemo(object):
        """ Generator of demos. """
        def __init__(self, config):
            self._hyperparams = config
            self._conditions = config['common']['conditions']

            self.nn_demo = config['common']['nn_demo']
            self._exp_dir = config['common']['demo_exp_dir']
            self._data_files_dir = config['common']['data_files_dir']
            self._algorithm_files_dir = config['common']['demo_controller_file']
            self.data_logger = DataLogger()

        def load_algorithms(self):
            algorithm_files = self._algorithm_files_dir
            if isinstance(algorithm_files, basestring):
                with open(algorithm_files, 'r') as f:
                    algorithms = [pickle.load(f)]
            else:
                algorithms = []
                for filename in algorithm_files:
                    with open(filename, 'r') as f:
                        algorithms.append(pickle.load(f))
            return algorithms

        def generate(self, demo_file, ioc_agent):
            """
             Generate demos and save them in a file for experiment.
             Args:
                 demo_file - place to store the demos
                 ioc_agent - ioc agent, for grabbing the observation using the ioc agent's observation data types
             Returns: None.
            """
            # Load the algorithm

            self.algorithms = self.load_algorithms()
            self.algorithm = self.algorithms[0]

            # Keep the initial states of the agent the sames as the demonstrations.
            agent_config = self._hyperparams['demo_agent']
            self.agent = agent_config['type'](agent_config)

            # Roll out the demonstrations from controllers
            var_mult = self._hyperparams['algorithm']['demo_var_mult']
            T = self.algorithms[0].T
            demos = []
            demo_idx_conditions = []  # Stores conditions for each demo

            M = agent_config['conditions']
            N = self._hyperparams['algorithm']['num_demos']
            if not self.nn_demo:
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
                            verbose=(j < self._hyperparams['verbose_trials']), noisy=True,
                            save = True
                        )
                        demos.append(demo)
                        demo_idx_conditions.append(i)
            else:
                all_pos_body_offsets = []
                # Gather demos.
                for a in xrange(len(self.algorithms)):
                    pol = self.algorithms[a].policy_opt.policy
                    for i in xrange(M / len(self.algorithms) * a, M / len(self.algorithms) * (a + 1)):
                        for j in xrange(N):
                            demo = self.agent.sample(
                                pol, i,
                                verbose=(j < self._hyperparams['verbose_trials']), noisy=True
                                )
                            demos.append(demo)
                            demo_idx_conditions.append(i)
            self.filter(demos, demo_idx_conditions, agent_config, ioc_agent, demo_file)

        def filter(self, demos, demo_idx_conditions, agent_config, ioc_agent, demo_file):
            """
            Filter out failed demos.
            Args:
                demos: generated demos
                demo_idx_conditions: the conditions of generated demos
                agent_config: config of the demo agent
                ioc_agent: the agent for ioc
                demo_file: the path to save demos
            """
            M = agent_config['conditions']
            N = self._hyperparams['algorithm']['num_demos']
            
            # Filter failed demos
            if 'filter_demos' in agent_config:
                filter_options = agent_config['filter_demos']
                filter_type = filter_options.get('type', 'min')
                targets = filter_options['target']
                pos_idx = filter_options.get('state_idx', range(4, 7))
                end_effector_idx = filter_options.get('end_effector_idx', range(0, 3))
                max_per_condition = filter_options.get('max_demos_per_condition', 999)
                dist_threshold = filter_options.get('success_upper_bound', 0.01)
                cur_samples = SampleList(demos)
                dists = compute_distance(targets, cur_samples, pos_idx, end_effector_idx, filter_type=filter_type)
                failed_idx = []
                for i, distance in enumerate(dists):
                    if (distance > dist_threshold):
                        failed_idx.append(i)

                LOGGER.debug("Removing %d failed demos: %s", len(failed_idx), str(failed_idx))
                demos_filtered = [demo for (i, demo) in enumerate(demos) if i not in failed_idx]
                demo_idx_conditions = [cond for (i, cond) in enumerate(demo_idx_conditions) if i not in failed_idx]
                demos = demos_filtered

                # Filter max demos per condition
                condition_to_demo = {
                    cond: [demo for (i, demo) in enumerate(demos) if demo_idx_conditions[i]==cond][:max_per_condition]
                    for cond in range(M)
                }
                LOGGER.debug('Successes per condition: %s', str([len(demo_list) for demo_list in condition_to_demo.values()]))
                demos = [demo for cond in condition_to_demo for demo in condition_to_demo[cond]]
                shuffle(demos)

                for demo in demos: demo.reset_agent(ioc_agent)
                demo_list = SampleList(demos)
                demo_store = {'demoX': demo_list.get_X(),
                              'demoU': demo_list.get_U(),
                              'demoO': demo_list.get_obs(),
                              'demoConditions': demo_idx_conditions}
            else:
                shuffle(demos)
                for demo in demos: demo.reset_agent(ioc_agent)
                demo_list = SampleList(demos)
                demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs()}
            # Save the demos.
            self.data_logger.pickle(
                demo_file,
                copy.copy(demo_store)
            )
            