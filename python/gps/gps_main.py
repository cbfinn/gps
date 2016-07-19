""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')

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

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.utility.generate_demo import GenDemo


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
		self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

		config['algorithm']['agent'] = self.agent

		if config['algorithm']['ioc']:
			demo_file = self._data_files_dir + 'demos.pkl'
			demos = self.data_logger.unpickle(demo_file)
			if demos is None:
			  self.demo_gen = GenDemo(config)
			  self.demo_gen.generate()
			  demo_file = self._data_files_dir + 'demos.pkl'
			  demos = self.data_logger.unpickle(demo_file)
			config['algorithm']['init_traj_distr']['init_demo_x'] = np.mean(demos['demoX'], 0)
			config['algorithm']['init_traj_distr']['init_demo_u'] = np.mean(demos['demoU'], 0)
			self.algorithm = config['algorithm']['type'](config['algorithm'])
			self.algorithm.init_samples = self._hyperparams['num_samples']
			if self.algorithm._hyperparams['learning_from_prior']:
				config['agent']['pos_body_offset'] = demos['pos_body_offset']
			self.agent = config['agent']['type'](config['agent'])
			self.algorithm.demoX = demos['demoX']
			self.algorithm.demoU = demos['demoU']
			self.algorithm.demoO = demos['demoO']
			self.algorithm.demo_conditions = demos['demo_conditions']
			self.algorithm.failed_conditions = demos['failed_conditions']
		else:
			self.algorithm = config['algorithm']['type'](config['algorithm'])
			self.algorithm.init_samples = self._hyperparams['num_samples']


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
			for cond in self._train_idx:
				for i in range(self._hyperparams['num_samples']):
					self._take_sample(itr, cond, i)

			traj_sample_lists = [
				self.agent.get_samples(cond, -self._hyperparams['num_samples'])
				for cond in self._train_idx
			]

			self._take_iteration(itr, traj_sample_lists)
			# if not self.algorithm._hyperparams['ioc']:
			# TODO - need to add these to lines back in when we move to mdgps
			#     pol_sample_lists = self._take_policy_samples()
			#     self._log_data(itr, traj_sample_lists, pol_sample_lists)
			# else:
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
			algorithm_file = self._data_files_dir + 'algorithm_i_%02d.pkl' % itr_load
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
		if self.algorithm._hyperparams['sample_on_policy'] \
				and self.algorithm.iteration_count > 0:
			pol = self.algorithm.policy_opt.policy
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
			return None
		if not N:
			N = self._hyperparams['verbose_policy_trials']
		if self.gui:
			self.gui.set_status_text('Taking policy samples.')
		pol_samples = [[None for _ in range(N)] for _ in range(self._conditions)]
		for cond in range(len(self._test_idx)):
			for i in range(N):
				pol_samples[cond][i] = self.agent.sample(
					self.algorithm.policy_opt.policy, self._test_idx[cond],
					verbose=True, save=False, noisy=False)
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
		if self.gui:
			self.gui.set_status_text('Logging data and updating GUI.')
			self.gui.update(itr, self.algorithm, self.agent,
				traj_sample_lists, pol_sample_lists)
			self.gui.save_figure(
				self._data_files_dir + ('figure_itr_%02d.png' % itr)
			)
		if 'no_sample_logging' in self._hyperparams['common']:
			return
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
	args = parser.parse_args()

	exp_name = args.experiment
	resume_training_itr = args.resume
	test_policy_N = args.policy

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
	elif exp_name == "mjc_peg_ioc_learning_example":
		ioc_conditions = [np.array([random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)]), \
						random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)])]) for i in xrange(10)]
		top_bottom = [np.array([np.random.uniform(-0.08, 0.08), \
						random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)])]) for i in xrange(10)]
		left_right = [np.array([random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)]), \
						np.random.uniform(-0.08, 0.08)]) for i in xrange(10)]
		ioc_conditions.extend(top_bottom)
		ioc_conditions.extend(left_right)
		exp_iter = hyperparams.config['algorithm']['iterations']
		data_files_dir = exp_dir + 'data_files/'
		mean_dists = []
		pos_body_offset_dists = [np.linalg.norm(ioc_conditions[i]) for i in xrange(len(ioc_conditions))]
		for i in xrange(len(ioc_conditions)):
			hyperparams = imp.load_source('hyperparams', hyperparams_file)
			# hyperparams.config['gui_on'] = False
			hyperparams.config['algorithm']['ioc_cond'] = ioc_conditions[i]
			gps = GPSMain(hyperparams.config)
			gps.agent._hyperparams['pos_body_offset'] = [ioc_conditions[i]]
			# import pdb; pdb.set_trace()
			if hyperparams.config['gui_on']:
				# run_gps = threading.Thread(
				#     target=lambda: gps.run(itr_load=resume_training_itr)
				# )
				# run_gps.daemon = True
				# run_gps.start()
				gps.run(itr_load=resume_training_itr)
				plt.close()
				# plt.ioff()
				# plt.show()
			else:
				gps.run(itr_load=resume_training_itr)
				# continue
			if i == 0:
				demo_conditions = gps.algorithm.demo_conditions
				failed_conditions = gps.algorithm.failed_conditions
			mean_dists.append(gps.algorithm.dists_to_target[exp_iter - 1][0])
			print "iteration " + repr(i) + ": mean dist is " + repr(mean_dists[i]) 
		with open(exp_dir + 'log.txt', 'a') as f:
			f.write('\nThe 50 IOC conditions are: \n' + str(ioc_conditions) + '\n')
		plt.plot(pos_body_offset_dists, mean_dists, 'ro')
		plt.title("Learning from prior experience using peg insertion")
		plt.xlabel('pos body offset distances to the origin')
		plt.ylabel('mean distances to the target')
		plt.savefig(data_files_dir + 'learning_from_prior.png')
		plt.close()

		from matplotlib.patches import Rectangle

		ioc_conditions_x = [ioc_conditions[i][0] for i in xrange(len(ioc_conditions))]
		ioc_conditions_y = [ioc_conditions[i][1] for i in xrange(len(ioc_conditions))]
		mean_dists = np.around(mean_dists, decimals=2)
		failed_ioc_x = [ioc_conditions_x[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] > 0.08]
		failed_ioc_y = [ioc_conditions_y[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] > 0.08]
		success_ioc_x = [ioc_conditions_x[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] <= 0.08]
		success_ioc_y = [ioc_conditions_y[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] <= 0.08]
		demo_conditions_x = [demo_conditions[i][0] for i in xrange(len(demo_conditions))]
		demo_conditions_y = [demo_conditions[i][1] for i in xrange(len(demo_conditions))]
		failed_conditions_x = [failed_conditions[i][0] for i in xrange(len(failed_conditions))]
		failed_conditions_y = [failed_conditions[i][1] for i in xrange(len(failed_conditions))]
		subplt = plt.subplot()
		subplt.plot(demo_conditions_x, demo_conditions_y, 'go')
		subplt.plot(failed_conditions_x, failed_conditions_y, 'rx')
		subplt.plot(success_ioc_x, success_ioc_y, 'g^')
		subplt.plot(failed_ioc_x, failed_ioc_y, 'rv')
		# plt.legend(['demo_cond', 'failed_badmm', 'success_ioc', 'failed_ioc'], loc= (1, 1))
		for i, txt in enumerate(mean_dists):
			subplt.annotate(txt, (ioc_conditions_x[i], ioc_conditions_y[i]))
		ax = plt.gca()
		ax.add_patch(Rectangle((-0.08, -0.08), 0.16, 0.16, fill = False, edgecolor = 'blue'))
		box = subplt.get_position()
		subplt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
		subplt.legend(['demo_cond', 'failed_badmm', 'success_ioc', 'failed_ioc'], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
						shadow=True, ncol=2)
		plt.title("Distribution of neural network and IOC's initial conditions")
		# plt.xlabel('width')
		# plt.ylabel('length')
		plt.savefig(data_files_dir + 'distribution_of_conditions.png')
		plt.show()
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
