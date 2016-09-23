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
# from gps.utility.generate_demo import GenDemo
from gps.utility.general_utils import disable_caffe_logs
from gps.utility.demo_utils import eval_demos_xu, compute_distance_cost_plot, compute_distance_cost_plot_xu, \
                                    measure_distance_and_success_peg, get_demos
from gps.utility.visualization import get_comparison_hyperparams, compare_experiments 

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
            demos = get_demos(self)
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

    def _run_policy(self, condition, itr_load=0, local_pol=False, repl=True):
        itr = self._initialize(itr_load)
        if itr == 0:
            raise NotImplementedError("Could not find iteration file!")
        if local_pol:
            policy = self.algorithm.cur[condition].traj_distr
        else:
            policy = self.algorithm.policy_opt.policy
        if repl:
            while True:
                cond = int(input("Condition >> "))
                self.agent.sample(policy, cond)

        else:
            self.agent.sample(policy, condition)

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
            self.algorithm._hyperparams['init_demo_policy']):
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
            sample_losses = self.algorithm.cur[6].cs
            if sample_losses is None:
                sample_losses = self.algorithm.prev[6].cs
            if sample_losses.shape[0] < NUM_DEMO_PLOTS:
                sample_losses = np.tile(sample_losses, [NUM_DEMO_PLOTS, 1])[:NUM_DEMO_PLOTS]
            demo_losses = eval_demos_xu(self.agent, self.algorithm.demoX, self.algorithm.demoU, self.algorithm.cost, n=NUM_DEMO_PLOTS)

            # Produce distance vs cost plots
            dists_vs_costs = compute_distance_cost_plot(self.algorithm, self.agent, traj_sample_lists[6])
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

        # if itr == self.algorithm._hyperparams['iterations'] - 1 or itr == self.algorithm._hyperparams['ioc_maxent_iter'] - 1: # Just save the last iteration of the algorithm file
        if ((itr+1) % 5 == 0) or itr == self.algorithm._hyperparams['iterations'] - 1: # Just save the last iteration of the algorithm file
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
    parser.add_argument('--dry_run', nargs=2, type=int, default=None,
                        help='Condition to dry-run the policy')
    parser.add_argument('-c', '--compare', metavar='N', type=int,
                    help='compare two experiments')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy
    compare = args.compare

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'
    hyperparams_file_compare = exp_dir + 'hyperparams_compare.py'

    if args.dry_run:
        import caffe
        hyperparams = imp.load_source('hyperparams', hyperparams_file)
        gps = GPSMain(hyperparams.config)
        gps._run_policy(args.dry_run[0], args.dry_run[1])
        import sys; sys.exit(0)

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
    elif compare:
        mean_dists_1_dict, mean_dists_2_dict, success_rates_1_dict, \
            success_rates_2_dict = {}, {}, {}, {}
        seeds = [0, 1, 2] 
        for itr in seeds:
            random.seed(itr)
            np.random.seed(itr)
            hyperparams_compare = get_comparison_hyperparams(hyperparams_file_compare, itr)
            gps = GPSMain(hyperparams.config)
            gps.run()
            plt.close()
            mean_dists_1_dict[itr], success_rates_1_dict[itr] = measure_distance_and_success_peg(gps) # For peg only
            gps_classic = GPSMain(hyperparams_compare.config)
            gps.run()
            plt.close()
            mean_dists_2_dict[itr], success_rates_2_dict[itr] = measure_distance_and_success_peg(gps) # For peg only
        compare_experiments(mean_dists_1_dict, mean_dists_2_dict, success_rates_1_dict, \
                                success_rates_2_dict, gps._hyperparams['algorithm']['iterations'], \
                                exp_dir, hyperparams_compare.config, hyperparams.config)
    else:
        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)

if __name__ == "__main__":
    main()
