""" This file defines the main object that runs experiments. """

import matplotlib as mpl

mpl.use('Qt4Agg')
#mpl.use('Pdf')  # for EC2

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import scipy.io
import numpy as np

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI, NUM_DEMO_PLOTS
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.utility.generate_demo import GenDemo
from gps.utility.general_utils import disable_caffe_logs
from gps.utility.demo_utils import eval_demos_xu, compute_distance_cost_plot, compute_distance_cost_plot_xu


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common'], gui_on=config['gui_on'])

        config['algorithm']['agent'] = self.agent

        if self.using_ioc():
            # demo_file = self._data_files_dir + 'demos.pkl'
            if not config['common'].get('nn_demo', False):
                demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_LG.pkl' # for mdgps experiment
            else:
                # demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn.pkl'
                # demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn_multiple_no_noise.pkl'
                # demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn_multiple_3.pkl'
            	# demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn_3pols_9conds.pkl'
                # demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn_maxent_4_cond.pkl'
                #demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn_MaxEnt_4_cond_z_0.05_noise.pkl'
                demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn_MaxEnt_4_cond_z_0.05_noise_seed1.pkl'
            demos = self.data_logger.unpickle(demo_file)
            if demos is None:
              self.demo_gen = GenDemo(config)
              self.demo_gen.generate(demo_file)
              demos = self.data_logger.unpickle(demo_file)
            config['algorithm']['init_traj_distr']['init_demo_x'] = np.mean(demos['demoX'], 0)
            config['algorithm']['init_traj_distr']['init_demo_u'] = np.mean(demos['demoU'], 0)
            self.algorithm = config['algorithm']['type'](config['algorithm'])
            # if self.algorithm._hyperparams['learning_from_prior']:
            #   config['agent']['pos_body_offset'] = demos['pos_body_offset']
            # Initialize policy using the demo neural net policy

            if 'init_demo_policy' in self.algorithm._hyperparams and \
                        self.algorithm._hyperparams['init_demo_policy']:
                demo_algorithm_file = config['common']['demo_controller_file']
                demo_algorithm = self.data_logger.unpickle(demo_algorithm_file)
                if demo_algorithm is None:
                    print("Error: cannot find '%s.'" % algorithm_file)
                    os._exit(1) # called instead of sys.exit(), since t
                var_mult = self.algorithm._hyperparams['init_var_mult']
                self.algorithm.policy_opt.var = demo_algorithm.policy_opt.var.copy() * var_mult
                self.algorithm.policy_opt.policy = demo_algorithm.policy_opt.copy().policy
                self.algorithm.policy_opt.policy.chol_pol_covar = np.diag(np.sqrt(self.algorithm.policy_opt.var))
                self.algorithm.policy_opt.solver.net.share_with(self.algorithm.policy_opt.policy.net)

                var_mult = self.algorithm._hyperparams['demo_var_mult']
                self.algorithm.demo_policy_opt = demo_algorithm.policy_opt.copy()
                self.algorithm.demo_policy_opt.var = demo_algorithm.policy_opt.var.copy() * var_mult
                self.algorithm.demo_policy_opt.policy.chol_pol_covar = np.diag(np.sqrt(self.algorithm.demo_policy_opt.var))

            self.agent = config['agent']['type'](config['agent'])
            self.algorithm.demoX = demos['demoX']
            self.algorithm.demoU = demos['demoU']
            self.algorithm.demoO = demos['demoO']
            if 'demo_conditions' in demos.keys() and 'failed_conditions' in demos.keys():
                self.algorithm.demo_conditions = demos['demo_conditions']
                self.algorithm.failed_conditions = demos['failed_conditions']

            # get samples and reward values
            if self._hyperparams['algorithm']['ioc'] == 'SUPERVISED':
                import glob
                from gps.proto.gps_pb2 import GYM_REWARD
                sample_files = glob.glob(config['common']['gt_cost_samples'])
                T = self._hyperparams['algorithm']['cost']['T']
                _, _, dX = self.algorithm.demoX.shape
                _, _, dO = self.algorithm.demoO.shape
                _, T, dU = self.algorithm.demoU.shape
                gt_cost_X = np.zeros((0, T, dX))
                gt_cost_U = np.zeros((0, T, dU))
                gt_cost_O = np.zeros((0, T, dO))
                gt_cost = np.zeros((0, T))
                import pdb; pdb.set_trace()
                for sample_file in sample_files:
                    traj_sample_lists = self.data_logger.unpickle(sample_file)
                    for sample_list in traj_sample_lists:
                        for sample in sample_list:
                            sample.agent = self.agent # need obs_datatypes to be set.
                        gt_cost_O = np.r_[gt_cost_O, sample_list.get_obs()]
                        gt_cost_X = np.r_[gt_cost_X, sample_list.get_X()]
                        gt_cost_U = np.r_[gt_cost_U, sample_list.get_U()]
                        gt_cost = np.r_[gt_cost, -sample_list.get(GYM_REWARD)]
                gt_cost = np.expand_dims(gt_cost, -1)
                import pdb; pdb.set_trace()
                gt_cost -= np.min(gt_cost)
                self.algorithm.cost._hyperparams['iterations'] = 10000
                self.algorithm.cost.update_supervised(gt_cost_U, gt_cost_X, gt_cost_O, gt_cost)
                import pdb; pdb.set_trace()

        else:
            self.algorithm = config['algorithm']['type'](config['algorithm'])


    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        import numpy as np
        import numpy.matlib

        itr_start = self._initialize(itr_load)
        for itr in range(itr_start, self._hyperparams['iterations']):
            if self.agent._hyperparams.get('randomly_sample_x0', False):
                    for cond in self._train_idx:
                        self.agent.reset_initial_x0(cond)

            if self.agent._hyperparams.get('randomly_sample_bodypos', False):
                for cond in self._train_idx:
                    self.agent.reset_initial_body_offset(cond)
            for cond in self._train_idx:
                if itr == 0:
                    for i in range(self.algorithm._hyperparams['init_samples']):
                        self._take_sample(itr, cond, i)
                else:
                    for i in range(self._hyperparams['num_samples']):
                        self._take_sample(itr, cond, i)

            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                for cond in self._train_idx
            ]

            self._take_iteration(itr, traj_sample_lists)

            if self.algorithm._hyperparams['sample_on_policy']:
            # TODO - need to add these to lines back in when we move to mdgps
                pol_sample_lists = self._take_policy_samples()
                self._log_data(itr, traj_sample_lists, pol_sample_lists)
            else:
                self._log_data(itr, traj_sample_lists)

        self._end()
        return None

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def measure_distance_and_success(self):
        """
        Take the algorithm states for all iterations and extract the
        mean distance to the target position and measure the success
        rate of inserting the peg. (For the peg experiment only)
        Args:
            None
        Returns: the mean distance and the success rate
        """
        from gps.proto.gps_pb2 import END_EFFECTOR_POINTS

        pol_iter = self._hyperparams['algorithm']['iterations']
        peg_height = self._hyperparams['demo_agent']['peg_height']
        mean_dists = []
        success_rates = []
        for i in xrange(pol_iter):
            # if 'sample_on_policy' in self._hyperparams['algorithm'] and \
            #     self._hyperparams['algorithm']['sample_on_policy']:
            #     pol_samples_file = self._data_files_dir + 'pol_sample_itr_%02d.pkl' % i
            # else:
            pol_samples_file = self._data_files_dir + 'traj_sample_itr_%02d.pkl' % i
            pol_sample_lists = self.data_logger.unpickle(pol_samples_file)
            if pol_sample_lists is None:
                print("Error: cannot find '%s.'" % pol_samples_file)
                os._exit(1) # called instead of sys.exit(), since t
            samples = []
            for m in xrange(len(pol_sample_lists)):
                curSamples = pol_sample_lists[m].get_samples()
                for sample in curSamples:
                    samples.append(sample)
            if type(self.algorithm._hyperparams['target_end_effector']) is list:
                    target_position = self.algorithm._hyperparams['target_end_effector'][m][:3]
            else:
                target_position = self.algorithm._hyperparams['target_end_effector'][:3]
            dists_to_target = [np.nanmin(np.sqrt(np.sum((sample.get(END_EFFECTOR_POINTS)[:, :3] - \
                                target_position.reshape(1, -1))**2, axis = 1)), axis = 0) for sample in samples]
            mean_dists.append(sum(dists_to_target)/len(dists_to_target))
            success_rates.append(float(sum(1 for dist in dists_to_target if dist <= peg_height))/ \
                                    len(dists_to_target))
        return mean_dists, success_rates

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                if self.algorithm.cur[0].pol_info:
                    pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                        ('pol_sample_itr_%02d.pkl' % itr_load))
                else:
                    pol_sample_lists = None
                #self.gui.update(itr_load, self.algorithm, self.agent,
                #    traj_sample_lists, pol_sample_lists)
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm._hyperparams['sample_on_policy'] and (self.algorithm.iteration_count > 0 or \
            self.algorithm._hyperparams['init_demo_policy']): # just for experiment. DELETE NOT AFTER EXPERIMENT!
        # if self.algorithm._hyperparams['sample_on_policy'] and self.algorithm.iteration_count > 0:
            if not self.algorithm._hyperparams['multiple_policy']:
                pol = self.algorithm.policy_opt.policy
            else:
                pol = self.algorithm.policy_opts[cond / self.algorithm.num_policies].policy
        else:
            pol = self.algorithm.cur[cond].traj_distr
        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            )

    def _take_iteration(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        self.algorithm.iteration(sample_lists)
        if self.gui:
            self.gui.stop_display_calculating()

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        for cond in range(len(self._test_idx)):
            if not self.algorithm._hyperparams['multiple_policy']:
                pol_samples[cond][0] = self.agent.sample(
                    self.algorithm.policy_opt.policy, self._test_idx[cond],
                    verbose=True, save=False, noisy=True)
            else:
                pol = self.algorithm.policy_opts[cond / self.algorithm.num_policies].policy
                pol_samples[cond][0] = self.agent.sample(
                    pol, self._test_idx[cond],
                    verbose=True, save=False, noisy=True)
        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """

        if self.using_ioc():
            # Produce time vs cost plots
            sample_losses = self.algorithm.cur[0].cs
            if sample_losses is None:
                sample_losses = self.algorithm.prev[0].cs
            if sample_losses.shape[0] < NUM_DEMO_PLOTS:
                sample_losses = np.tile(sample_losses, [NUM_DEMO_PLOTS, 1])[:NUM_DEMO_PLOTS]
            demo_losses = eval_demos_xu(self.agent, self.algorithm.demoX, self.algorithm.demoU, self.algorithm.cost, n=NUM_DEMO_PLOTS)

            # Produce distance vs cost plots
            dists_vs_costs = compute_distance_cost_plot(self.algorithm, self.agent, traj_sample_lists[0])
            demo_dists_vs_costs = compute_distance_cost_plot_xu(self.algorithm, self.agent, self.algorithm.demoX, self.algorithm.demoU)

        else:
            demo_losses = None
            sample_losses = None
            dists_vs_costs = None
            demo_dists_vs_costs = None

        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists, ioc_demo_losses=demo_losses, ioc_sample_losses=sample_losses,
                            ioc_dist_cost=dists_vs_costs, ioc_demo_dist_cost=demo_dists_vs_costs)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.pdf' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        if itr >= self.algorithm._hyperparams['iterations'] - 5: # Just save the last iteration of the algorithm file
            self.algorithm.demo_policy = None
            self.data_logger.pickle(
                self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
                copy.copy(self.algorithm)
            )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end:
                # Quit automatically (for running sequential expts)
                os._exit(1)

    def using_ioc(self):
        return 'ioc' in self._hyperparams['algorithm'] and self._hyperparams['algorithm']['ioc']

    def test_samples(self, N, agent_config, itr):
        from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
        import matplotlib.pyplot as plt

        pol_iter = self._hyperparams['algorithm']['iterations'] - 1
        algorithm_ioc = self.data_logger.unpickle(self._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
        M = agent_config['conditions']

        # pol_ioc = algorithm_ioc.policy_opt.policy
        controllers = {}
        for i in xrange(M):
            controllers[i] = algorithm_ioc.cur[i].traj_distr
        # pol_ioc.chol_pol_covar *= 0.0
        samples = []
        agent = agent_config['type'](agent_config)
        ioc_conditions = agent_config['pos_body_offset']
        for i in xrange(M):
            # Gather demos.
            for j in xrange(N):
                sample = agent.sample(
                    controllers[i], i,
                    verbose=(i < self._hyperparams['verbose_trials']), noisy=False
                    )
                samples.append(sample)
        target_position = agent_config['target_end_effector'][:3]
        dists_to_target = [np.zeros((M*N)) for i in xrange(len(samples))]
        dists_diff = []
        all_success_conditions = []
        only_ioc_conditions = []
        only_demo_conditions = []
        all_failed_conditions = []
        percentages = []
        for i in xrange(len(samples[0])):
            for j in xrange(len(samples)):
                sample_end_effector = samples[j][i].get(END_EFFECTOR_POINTS)
                dists_to_target[j][i] = np.nanmin(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
                # Just choose the last time step since it may become unstable after achieving the minimum point.
                # import pdb; pdb.set_trace()
                # dists_to_target[j][i] = np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1]
            if dists_to_target[0][i] <= 0.1 and dists_to_target[1][i] <= 0.1:
                all_success_conditions.append(ioc_conditions[i])
            elif dists_to_target[0][i] <= 0.1:
                only_ioc_conditions.append(ioc_conditions[i])
            elif dists_to_target[1][i] <= 0.1:
                only_demo_conditions.append(ioc_conditions[i])
            else:
                all_failed_conditions.append(ioc_conditions[i])
            dists_diff.append(np.around(dists_to_target[0][i] - dists_to_target[1][i], decimals=2))
        percentages.append(np.around(float(len(all_success_conditions))/len(ioc_conditions), decimals=2))
        percentages.append(np.around(float(len(all_failed_conditions))/len(ioc_conditions), decimals=2))
        percentages.append(np.around(float(len(only_ioc_conditions))/len(ioc_conditions), decimals=2))
        percentages.append(np.around(float(len(only_demo_conditions))/len(ioc_conditions), decimals=2))
        exp_dir = self._data_files_dir.replace("data_files", "")

        from matplotlib.patches import Rectangle

        plt.close('all')
        ioc_conditions_x = [ioc_conditions[i][0] for i in xrange(len(ioc_conditions))]
        ioc_conditions_y = [ioc_conditions[i][1] for i in xrange(len(ioc_conditions))]
        all_success_x = [all_success_conditions[i][0] for i in xrange(len(all_success_conditions))]
        all_success_y = [all_success_conditions[i][1] for i in xrange(len(all_success_conditions))]
        all_failed_x = [all_failed_conditions[i][0] for i in xrange(len(all_failed_conditions))]
        all_failed_y = [all_failed_conditions[i][1] for i in xrange(len(all_failed_conditions))]
        only_ioc_x = [only_ioc_conditions[i][0] for i in xrange(len(only_ioc_conditions))]
        only_ioc_y = [only_ioc_conditions[i][1] for i in xrange(len(only_ioc_conditions))]
        only_demo_x = [only_demo_conditions[i][0] for i in xrange(len(only_demo_conditions))]
        only_demo_y = [only_demo_conditions[i][1] for i in xrange(len(only_demo_conditions))]
        subplt = plt.subplot()
        subplt.plot(all_success_x, all_success_y, 'yo')
        subplt.plot(all_failed_x, all_failed_y, 'rx')
        subplt.plot(only_ioc_x, only_ioc_y, 'g^')
        subplt.plot(only_demo_x, only_demo_y, 'rv')
        # plt.legend(['demo_cond', 'failed_badmm', 'success_ioc', 'failed_ioc'], loc= (1, 1))
        for i, txt in enumerate(dists_diff):
            subplt.annotate(txt, (ioc_conditions_x[i], ioc_conditions_y[i]))
        ax = plt.gca()
        # ax.add_patch(Rectangle((-0.08, -0.08), 0.16, 0.16, fill = False, edgecolor = 'blue')) # peg
        ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        box = subplt.get_position()
        subplt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
        subplt.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
                        'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
                        shadow=True, ncol=2)
        plt.title("Distribution of samples drawn from demo policy and IOC policy")
        # plt.xlabel('width')
        # plt.ylabel('length')
        #plt.savefig(self._data_files_dir + 'distribution_of_sample_conditions_added_per.png')
        plt.savefig(self._data_files_dir + 'distribution_of_sample_conditions_added_per.pdf')
        plt.close('all')

    def compare_samples(self, N, agent_config, itr):
        from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        #pol_iter = self._hyperparams['algorithm']['iterations'] - 2
        #file_dir = self._hyperparams['common']['demo_exp_dir']

        pol_iter = 14
        file_dir = self._hyperparams['common']['demo_exp_dir']
        file_dir = file_dir[:-2] + '0/'
        import pdb; pdb.set_trace()
        algorithm_ioc = self.data_logger.unpickle(file_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
        algorithm_demo = self.data_logger.unpickle('/home/cfinn/code/gps/experiments/reacher_mdgps_nomaxent/data_files_maxent_9cond_z_0.05_0/algorithm_itr_14.pkl')
        #algorithm_demo = algorithm_ioc
        #algorithm_demo = self.data_logger.unpickle(self._hyperparams['common']['demo_controller_file']) # + 'data_files_maxent_9cond_z_0.05_0/algorithm_itr_09.pkl') # Assuming not using 4 policies


        #algorithm_ioc = self.data_logger.unpickle(self._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
        #algorithm_demo = self.data_logger.unpickle(self._hyperparams['common']['demo_exp_dir'] + 'data_files_maxent_4cond_0.05_z_0/algorithm_itr_11.pkl') # Assuming not using 4 policies
        #algorithm_demo = self.data_logger.unpickle(self._hyperparams['common']['demo_exp_dir'] + 'data_files_maxent_9cond_z_0.05_0/algorithm_itr_09.pkl') # Assuming not using 4 policies


        #algorithm_ioc = self.data_logger.unpickle(self._hyperparams['common']['demo_controller_file2']) # + 'data_files_maxent_9cond_z_0.05_0/algorithm_itr_09.pkl') # Assuming not using 4 policies
        #algorithm_ioc = algorithm_demo
        #import pdb; pdb.set_trace()
        pos_body_offset = self._hyperparams['agent']['pos_body_offset']
        M = agent_config['conditions']
        np.random.seed(101)
        for m in range(M):
            self.agent.reset_initial_body_offset(m)

        pol_ioc = algorithm_ioc.policy_opt.policy
        # pol_ioc.chol_pol_covar *= 0.0
        pol_demo = algorithm_demo.policy_opt.policy
        policies = [pol_ioc, pol_demo]
        samples = {i: [] for i in xrange(len(policies))}
        agent = agent_config['type'](agent_config)
        ioc_conditions = agent_config['pos_body_offset']
        for i in xrange(M):
            for j in xrange(N):
                for k in xrange(len(samples)):
                    sample = agent.sample(
                        policies[k], i,
                        verbose=1000, noisy=True
                        )
                    samples[k].append(sample)
        import pdb; pdb.set_trace()
        target_position = agent_config['target_end_effector'][:3]
        dists_to_target = [np.zeros((M*N)) for i in xrange(len(samples))]
        dists_diff = []
        all_success_conditions = []
        only_ioc_conditions = []
        only_demo_conditions = []
        all_failed_conditions = []
        percentages = []
        for i in xrange(M):
            # TODO - REACHER ONLY
            pos_body_offset = self.agent._hyperparams['pos_body_offset'][i]
            target_position =  np.array([.1,-.1,.01])+pos_body_offset
            for j in xrange(len(samples)):
                sample_end_effector = samples[j][i].get(END_EFFECTOR_POINTS)
                dists_to_target[j][i] = np.nanmin(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)

            if dists_to_target[0][i] < 0.1 and dists_to_target[1][i] < 0.1:
                all_success_conditions.append(ioc_conditions[i])
            elif dists_to_target[0][i] < 0.1:
                only_ioc_conditions.append(ioc_conditions[i])
            elif dists_to_target[1][i] < 0.1:
                only_demo_conditions.append(ioc_conditions[i])
            else:
                all_failed_conditions.append(ioc_conditions[i])
            dists_diff.append(np.around(dists_to_target[0][i] - dists_to_target[1][i], decimals=2))
        percentages.append(round(float(len(all_success_conditions))/len(ioc_conditions), 2))
        percentages.append(round(float(len(all_failed_conditions))/len(ioc_conditions), 2))
        percentages.append(round(float(len(only_ioc_conditions))/len(ioc_conditions), 2))
        percentages.append(round(float(len(only_demo_conditions))/len(ioc_conditions), 2))
        mean_ioc_dist = np.mean(dists_to_target[0])
        mean_orig_dist = np.mean(dists_to_target[1])

        import pdb; pdb.set_trace()

        from matplotlib.patches import Rectangle

        plt.close('all')
        fig = plt.figure()
        ax = Axes3D(fig)
        ioc_conditions_x = [ioc_conditions[i][0] for i in xrange(len(ioc_conditions))]
        ioc_conditions_y = [ioc_conditions[i][1] for i in xrange(len(ioc_conditions))]
        ioc_conditions_z = [ioc_conditions[i][2] for i in xrange(len(ioc_conditions))]
        all_success_x = [all_success_conditions[i][0] for i in xrange(len(all_success_conditions))]
        all_success_y = [all_success_conditions[i][1] for i in xrange(len(all_success_conditions))]
        all_success_z = [all_success_conditions[i][2] for i in xrange(len(all_success_conditions))]
        all_failed_x = [all_failed_conditions[i][0] for i in xrange(len(all_failed_conditions))]
        all_failed_y = [all_failed_conditions[i][1] for i in xrange(len(all_failed_conditions))]
        all_failed_z = [all_failed_conditions[i][2] for i in xrange(len(all_failed_conditions))]
        only_ioc_x = [only_ioc_conditions[i][0] for i in xrange(len(only_ioc_conditions))]
        only_ioc_y = [only_ioc_conditions[i][1] for i in xrange(len(only_ioc_conditions))]
        only_ioc_z = [only_ioc_conditions[i][2] for i in xrange(len(only_ioc_conditions))]
        only_demo_x = [only_demo_conditions[i][0] for i in xrange(len(only_demo_conditions))]
        only_demo_y = [only_demo_conditions[i][1] for i in xrange(len(only_demo_conditions))]
        only_demo_z = [only_demo_conditions[i][2] for i in xrange(len(only_demo_conditions))]
        # subplt = plt.subplot()
        ax.scatter(all_success_x, all_success_y, all_success_z, c='y', marker='o')
        ax.scatter(all_failed_x, all_failed_y, all_failed_z, c='r', marker='x')
        ax.scatter(only_ioc_x, only_ioc_y, only_ioc_z, c='g', marker='^')
        ax.scatter(only_demo_x, only_demo_y, only_demo_z, c='r', marker='v')
        training_positions = zip(*pos_body_offset)
        ax.scatter(training_positions[0], training_positions[1], training_positions[2], s=40, c='b', marker='*')
        # plt.legend(['demo_cond', 'failed_badmm', 'success_ioc', 'failed_ioc'], loc= (1, 1))
        # for i, txt in enumerate(dists_diff):
        #     # subplt.annotate(txt, (ioc_conditions_x[i], ioc_conditions_y[i]))
        #     ax.annotate(txt, (ioc_conditions_x[i], ioc_conditions_y[i], ioc_conditions_z[i]))
        # ax = plt.gca()
        # ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        # ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        # box = subplt.get_position()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
        ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
                        'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, 0.05), \
                        shadow=True, ncol=2)
        plt.title("Distribution of samples drawn from demo policy and IOC policy")
        # plt.xlabel('width')
        # plt.ylabel('length')
        #plt.savefig(self._data_files_dir + 'distribution_of_sample_conditions_added_per.png')
        plt.savefig(self._data_files_dir + 'distribution_of_sample_conditions_added_per.pdf')
        plt.close('all')

    def eval_samples(self):
        pol_iter = self._hyperparams['algorithm']['iterations']
        M = self._hyperparams['common']['conditions']
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d' % (pol_iter-1) + '.pkl'
        algorithm = self.data_logger.unpickle(algorithm_file)
        mean_costs = {i: None for i in xrange(pol_iter-5, pol_iter)}
        for itr in xrange(pol_iter - 5, pol_iter):
            pol_samples_file = self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr)
            pol_samples = self.data_logger.unpickle(pol_samples_file)
            pol_costs = [np.mean([np.sum(algorithm.cost.eval(s)[0]) \
                    for s in pol_samples[m].get_samples()]) \
                    for m in range(M)]
            mean_costs[itr] = np.mean(pol_costs)
        return mean_costs

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM only)')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    parser.add_argument('-m', '--measure', metavar='N', type=int,
                        help='measure success rate among all iterations') # For peg only
    parser.add_argument('-c', '--compare', metavar='N', type=int,
                        help='compare global cost to multiple costs')
    parser.add_argument('-l', '--learn', metavar='N', type=int,
                        help='learning from prior experience')
    parser.add_argument('-a', '--again', metavar='N', type=int,
                        help='run multiple experiments')
    parser.add_argument('-e', '--eval', metavar='N', type=int,
                        help='eval the policy samples over iterations')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy
    measure_samples = args.measure
    compare_costs = args.compare
    learning_from_prior = args.learn
    multiple_run = args.again
    eval_flag = args.eval

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    unset = disable_caffe_logs()
    import caffe  # Hack to avoid segfault when importing caffe later
    disable_caffe_logs(unset)
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    import matplotlib.pyplot as plt
    import random
    import numpy as np

    random.seed(0)
    np.random.seed(0)

    if args.targetsetup:
        try:
            from gps.agent.ros.agent_ros import AgentROS
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    elif test_policy_N:
        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    elif measure_samples:
        #for itr in xrange(1):
            itr=1
            random.seed(itr)
            np.random.seed(itr)
            hyperparams = imp.load_source('hyperparams', hyperparams_file)
            #hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/'
            hyperparams.config['common']['demo_exp_dir'] = exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/'
            hyperparams.config['algorithm']['policy_opt']['weights_file_prefix'] = hyperparams.config['common']['data_files_dir'] + 'policy'
            if not os.path.exists(exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/'):
                os.makedirs(exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/')
            gps_samples = GPSMain(hyperparams.config)
            #agent_config = gps_samples._hyperparams['demo_agent']
            agent_config = gps_samples._hyperparams['agent']
            plt.close()
            gps_samples.compare_samples(measure_samples, agent_config, itr)
    elif eval_flag:
        for itr in xrange(3):
            random.seed(itr)
            np.random.seed(itr)
            hyperparams = imp.load_source('hyperparams', hyperparams_file)
            hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_maxent_9cond_0.05_%d' % itr + '/'
            hyperparams.config['algorithm']['policy_opt']['weights_file_prefix'] = hyperparams.config['common']['data_files_dir'] + 'policy'
            if not os.path.exists(exp_dir + 'data_files_maxent_9cond_0.05_%d' % itr + '/'):
                os.makedirs(exp_dir + 'data_files_maxent_9cond_0.05_%d' % itr + '/')
            gps_samples = GPSMain(hyperparams.config)
            plt.close()
            sample_costs = gps_samples.eval_samples()
            print sample_costs
    elif compare_costs:
        from gps.algorithm.policy.lin_gauss_init import init_lqr

        mean_dists_global_dict, mean_dists_no_global_dict, success_rates_global_dict, \
                success_rates_no_global_dict, mean_dists_classic_dict, success_rates_classic_dict \
                 = {}, {}, {}, {}, {}, {}
        seeds = [0, 1, 2] # Seed 1, 2 not working for on classic nn
        # var_mults = [8.0, 10.0, 16.0] # 12 doesn't work
        for itr in seeds:
        # for itr in xrange(3):
            random.seed(itr)
            np.random.seed(itr)
            hyperparams = imp.load_source('hyperparams', hyperparams_file)
            # hyperparams.config['algorithm']['init_traj_distr']['type'] = init_lqr
            # hyperparams.config['algorithm']['global_cost'] = False
            hyperparams.config['common']['nn_demo'] = True
            hyperparams.config['algorithm']['init_demo_policy'] = False
            hyperparams.config['algorithm']['policy_eval'] = False
            hyperparams.config['algorithm']['ioc'] = 'ICML'
            # hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_nn_%d' % itr + '/'
            # if not os.path.exists(exp_dir + 'data_files_nn_%d' % itr + '/'):
            #     os.makedirs(exp_dir + 'data_files_nn_%d' % itr + '/')
            # hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_nn_multiple_MPF_%d' % itr + '/'
            # if not os.path.exists(exp_dir + 'data_files_nn_multiple_MPF_%d' % itr + '/'):
            #   os.makedirs(exp_dir + 'data_files_nn_multiple_MPF_%d' % itr + '/')
            # exp_dir_classic = exp_dir.replace('on_global', 'on_classic')
            hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/'
            if not os.path.exists(exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/'):
              os.makedirs(exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/')

            hyperparams.config['algorithm']['policy_opt']['weights_file_prefix'] = hyperparams.config['common']['data_files_dir'] + 'policy'
            # hyperparams.config['algorithm']['init_var_mult'] = var_mults[itr]
            # hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_no_demo_ini_%d' % itr + '/'
            # if not os.path.exists(exp_dir + 'data_files_no_demo_ini_%d' % itr + '/'):
            #   os.makedirs(exp_dir + 'data_files_no_demo_ini_%d' % itr + '/')
            gps_global = GPSMain(hyperparams.config)
            pol_iter = gps_global.algorithm._hyperparams['iterations']
            # for i in xrange(pol_iter):
            if itr != 2:
                if hyperparams.config['gui_on']:
                    gps_global.run()
                    # gps_global.test_policy(itr=i, N=compare_costs)
                    plt.close()
                else:
                    gps_global.run()
                    # gps_global.test_policy(itr=i, N=compare_costs)
                    plt.close()
            mean_dists_global_dict[itr], success_rates_global_dict[itr] = gps_global.measure_distance_and_success()

            plt.close()
            hyperparams = imp.load_source('hyperparams', hyperparams_file)
            # hyperparams.config['algorithm']['init_traj_distr']['type'] = init_lqr
            # hyperparams.config['algorithm']['global_cost'] = True
            hyperparams.config['common']['nn_demo'] = True
            hyperparams.config['algorithm']['init_demo_policy'] = False
            hyperparams.config['algorithm']['policy_eval'] = False
            hyperparams.config['algorithm']['ioc'] = 'ICML'
            # hyperparams.config['agent']['randomly_sample_bodypos'] = True
            # hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_nn_multiple_MPF_%d' % itr + '/'
            # if not os.path.exists(exp_dir + 'data_files_nn_multiple_MPF_%d' % itr + '/'):
            #     os.makedirs(exp_dir + 'data_files_nn_multiple_MPF_%d' % itr + '/')
            exp_dir_classic = exp_dir.replace('on_global', 'on_classic')
            hyperparams.config['common']['data_files_dir'] = exp_dir_classic + 'data_files_nn_ICML_3pol_9cond_%d' % itr + '/'
            if not os.path.exists(exp_dir_classic + 'data_files_nn_ICML_3pol_9cond_%d' % itr + '/'):
                os.makedirs(exp_dir_classic + 'data_files_nn_ICML_3pol_9cond_%d' % itr + '/')

            hyperparams.config['algorithm']['policy_opt']['weights_file_prefix'] = hyperparams.config['common']['data_files_dir'] + 'policy'
            gps_classic = GPSMain(hyperparams.config)
            pol_iter = gps_classic.algorithm._hyperparams['iterations']
            mean_dists_classic_dict[itr], success_rates_classic_dict[itr] = gps_classic.measure_distance_and_success()
            plt.close()

        plt.close('all')
        avg_dists_global = [float(sum(mean_dists_global_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
        avg_succ_rate_global = [float(sum(success_rates_global_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
        avg_dists_classic = [float(sum(mean_dists_classic_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
        avg_succ_rate_classic = [float(sum(success_rates_classic_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
        # avg_dists_no_global = [float(sum(mean_dists_no_global_dict[i][j] for i in xrange(3)))/3 for j in xrange(pol_iter)]
        # avg_succ_rate_no_global = [float(sum(success_rates_no_global_dict[i][j] for i in xrange(3)))/3 for j in xrange(pol_iter)]
        plt.plot(range(pol_iter), avg_dists_global, '-x', color='red')
        plt.plot(range(pol_iter), avg_dists_classic, '-x', color='green')
        # plt.plot(range(pol_iter), avg_dists_no_global, '-x', color='green')
        for i in seeds:
            plt.plot(range(pol_iter), mean_dists_global_dict[i], 'ko')
            plt.plot(range(pol_iter), mean_dists_classic_dict[i], 'co')
        #   plt.plot(range(pol_iter), mean_dists_no_global_dict[i], 'co')
        #   plt.annotate(np.around(txt, decimals=2), (i, txt))
        plt.legend(['avg MaxEnt', 'avg non-MaxEnt', 'MaxEnt', 'non-MaxEnt'], loc='upper right', ncol=2)
        plt.title("mean distances to target over time with MaxEnt demo and not-MaxEnt demo")
        plt.xlabel("iterations")
        plt.ylabel("mean distances")
        #plt.savefig(exp_dir + 'mean_dists_during_iteration_comparison_maxent.png')
        plt.savefig(exp_dir + 'mean_dists_during_iteration_comparison_maxent.pdf')
        # plt.savefig(exp_dir + 'mean_dists_during_iteration_comparison.png')
        # plt.savefig(exp_dir + 'mean_dists_during_iteration_var.png')
        plt.close()
        plt.plot(range(pol_iter), avg_succ_rate_global, '-x', color='red')
        plt.plot(range(pol_iter), avg_succ_rate_classic, '-x', color='green')
        for i in seeds:
            plt.plot(range(pol_iter), success_rates_global_dict[i], 'ko')
            plt.plot(range(pol_iter), success_rates_classic_dict[i], 'co')
            # plt.plot(range(pol_iter), success_rates_no_global_dict[i], 'co')
        plt.legend(['avg MaxEnt', 'avg non-MaxEnt', 'MaxEnt', 'non-MaxEnt'], loc='upper right', ncol=2)
        plt.xlabel("iterations")
        plt.ylabel("success rate")
        plt.title("success rates during iterations with MaxEnt demo and not-MaxEnt dem")
        # plt.title("success rates during iterations with with nn and LG demo")
        #plt.savefig(exp_dir + 'success_rate_during_iteration_comparison_maxent.png')
        plt.savefig(exp_dir + 'success_rate_during_iteration_comparison_maxent.pdf')
        # plt.savefig(exp_dir + 'success_rate_during_iteration_comparison.png')
        # plt.savefig(exp_dir + 'success_rate_during_iteration_var.png')

        plt.close()

    elif multiple_run:
        for itr in range(1, 4):
            random.seed(itr)
            np.random.seed(itr)
            hyperparams = imp.load_source('hyperparams', hyperparams_file)
            hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_%d' % itr + '/'
            if not os.path.exists(exp_dir + 'data_files_%d' % itr + '/'):
              os.makedirs(exp_dir + 'data_files_%d' % itr + '/')
            gps = GPSMain(hyperparams.config)
            if hyperparams.config['gui_on']:
                gps.run()
                # gps_global.test_policy(itr=i, N=compare_costs)
                plt.close()
            else:
                gps.run()
                # gps_global.test_policy(itr=i, N=compare_costs)
                plt.close()
    else:  # actually running GPS
        hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_maxent_9cond_z_0.05_0/'
        if not os.path.exists(exp_dir + 'data_files_maxent_9cond_z_0.05_0/'):
            os.makedirs(exp_dir + 'data_files_maxent_9cond_z_0.05_0/')
        if 'policy_opt' in hyperparams.config['algorithm']:
            hyperparams.config['algorithm']['policy_opt']['weights_file_prefix'] = hyperparams.config['common']['data_files_dir'] + 'policy'
        gps = GPSMain(hyperparams.config)


        if hyperparams.config['gui_on']:
            #gps.run(itr_load=resume_training_itr)
            #plt.close()
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)
        # print gps.measure_distance_and_success()

if __name__ == "__main__":
    main()
