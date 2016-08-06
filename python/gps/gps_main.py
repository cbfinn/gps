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

		if 'ioc' in config['algorithm'] and config['algorithm']['ioc']:
			# demo_file = self._data_files_dir + 'demos.pkl'
			if not config['common']['nn_demo']:
				demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_LG.pkl' # for mdgps experiment
			else:
				demo_file = self._hyperparams['common']['experiment_dir'] + 'data_files/' + 'demos_nn.pkl'
			demos = self.data_logger.unpickle(demo_file)
			if demos is None:
			  self.demo_gen = GenDemo(config)
			  self.demo_gen.generate()
			  demo_file = self._data_files_dir + 'demos.pkl'
			  demos = self.data_logger.unpickle(demo_file)
			config['algorithm']['init_traj_distr']['init_demo_x'] = np.mean(demos['demoX'], 0)
			config['algorithm']['init_traj_distr']['init_demo_u'] = np.mean(demos['demoU'], 0)
			self.algorithm = config['algorithm']['type'](config['algorithm'])
			# if self.algorithm._hyperparams['learning_from_prior']:
			# 	config['agent']['pos_body_offset'] = demos['pos_body_offset']
			# Initialize policy using the demo neural net policy
			if 'init_demo_policy' in self.algorithm._hyperparams and \
						self.algorithm._hyperparams['init_demo_policy']:
				demo_algorithm_file = config['common']['demo_controller_file']
				demo_algorithm = self.data_logger.unpickle(demo_algorithm_file)
				if demo_algorithm is None:
					print("Error: cannot find '%s.'" % algorithm_file)
					os._exit(1) # called instead of sys.exit(), since t
				demo_algorithm.policy_opt.solver.net.share_with(demo_algorithm.policy_opt.policy.net)
				var_mult = self.algorithm._hyperparams['init_var_mult']
				self.algorithm.policy_opt.var = demo_algorithm.policy_opt.var * var_mult
				self.algorithm.policy_opt.policy = demo_algorithm.policy_opt.policy
				self.algorithm.policy_opt.policy.chol_pol_covar = np.diag(np.sqrt(self.algorithm.policy_opt.var))
				new_policy_opt = self.algorithm.policy_opt.copy()
				self.algorithm.demo_policy_opt = new_policy_opt
			self.agent = config['agent']['type'](config['agent'])
			self.algorithm.demoX = demos['demoX']
			self.algorithm.demoU = demos['demoU']
			self.algorithm.demoO = demos['demoO']
			if 'demo_conditions' in demos.keys() and 'failed_conditions' in demos.keys():
				self.algorithm.demo_conditions = demos['demo_conditions']
				self.algorithm.failed_conditions = demos['failed_conditions']
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

		pol_iter = self.algorithm._hyperparams['iterations']
		peg_height = self._hyperparams['demo_agent']['peg_height']
		mean_dists = []
		success_rates = []
		for i in xrange(pol_iter):
			pol_samples_file = self._data_files_dir + 'pol_sample_itr_%02d.pkl' % i
			pol_sample_lists = self.data_logger.unpickle(pol_samples_file)
			if pol_sample_lists is None:
				print("Error: cannot find '%s.'" % pol_samples_file)
				os._exit(1) # called instead of sys.exit(), since t
			samples = []
			for m in xrange(len(pol_sample_lists)):
				curSamples = pol_sample_lists[m].get_samples()
				for sample in curSamples:
					samples.append(sample)
			target = self.algorithm._hyperparams["target_end_effector"][:3]
			dists_to_target = [np.amin(np.sqrt(np.sum((sample.get(END_EFFECTOR_POINTS)[:, :3] - \
								target.reshape(1, -1))**2, axis = 1)), axis = 0) for sample in samples]
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
			self.algorithm._hyperparams['init_demo_policy']):
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
		if itr == self.algorithm._hyperparams['iterations'] - 1: # Just save the last iteration of the algorithm file
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

	def good_samples(self, N, agent_config):
		from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
		import matplotlib.pyplot as plt

		algorithm_samples = self.data_logger.unpickle(self._data_files_dir + 'algorithm_itr_19.pkl')
		M = agent_config['conditions']
		pol_conditions = [[np.array([-0.05, -0.05, 0]), np.array([-0.05, -0.05, 0]),
                        np.array([0.05, 0.05, 0]), np.array([0.05, -0.05, 0])],
                        [np.array([-0.06, -0.06, 0]), np.array([-0.06, -0.06, 0]),
                        np.array([0.06, 0.06, 0]), np.array([0.06, -0.06, 0])], 
                        [np.array([-0.07, -0.07, 0]), np.array([-0.07, -0.07, 0]),
                        np.array([0.07, 0.07, 0]), np.array([0.07, -0.07, 0])], 
                        [np.array([-0.08, -0.08, 0]), np.array([-0.08, -0.08, 0]),
                        np.array([0.08, 0.08, 0]), np.array([0.08, -0.08, 0])], 
                        [np.array([-0.09, -0.09, 0]), np.array([-0.09, -0.09, 0]),
                        np.array([0.09, 0.09, 0]), np.array([0.09, -0.09, 0])], 
                        [np.array([-0.10, -0.10, 0]), np.array([-0.10, -0.10, 0]),
                        np.array([0.10, 0.10, 0]), np.array([0.10, -0.10, 0])], 
                        [np.array([-0.11, -0.11, 0]), np.array([-0.11, -0.11, 0]),
                        np.array([0.11, 0.11, 0]), np.array([0.11, -0.11, 0])], 
                        [np.array([-0.12, -0.12, 0]), np.array([-0.12, -0.12, 0]),
                        np.array([0.12, 0.12, 0]), np.array([0.12, -0.12, 0])]]
		samples = []
		pol = algorithm_samples.policy_opt.policy
		for k in xrange(len(pol_conditions)):
			agent_config['pos_body_offset'] = pol_conditions[k]
			agent = agent_config['type'](agent_config)
			for i in xrange(M):
				# Gather demos.
				for j in xrange(N):
					sample = agent.sample(
						pol, i,
						verbose=(i < self._hyperparams['verbose_trials'])
						)
					samples.append(sample)
		target_position = agent_config['target_end_effector'][:3]
		dists_to_target = np.zeros(len(pol_conditions)*M*N)
		ioc_conditions = []
		for i in xrange(len(samples)):
			sample_end_effector = samples[i].get(END_EFFECTOR_POINTS)
			# dists_to_target[i] = np.amin(np.sqrt(np.sum((demo_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
			# Just choose the last time step since it may become unstable after achieving the minimum point.
			# import pdb; pdb.set_trace()
			dists_to_target[i] = np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1]
			ioc_conditions.append(pol_conditions[i/(M*N)][(i % (M*N))/N][:2])
		good_indicators = (dists_to_target <= 0.1).tolist()
		good_indices = [i for i in xrange(len(good_indicators)) if good_indicators[i]]
		good_dists = []
		bad_dists = []
		sample_conditions = []
		failed_conditions = []
		exp_dir = self._data_files_dir.replace("data_files", "")
		for i in good_indices:
			good_dists.append(dists_to_target[i]) # Assume N = 1
			sample_conditions.append(ioc_conditions[i])
		for i in xrange(len(samples)):
			if i not in good_indices:
				bad_dists.append(dists_to_target[i])
				failed_conditions.append(ioc_conditions[i])
		
		from matplotlib.patches import Rectangle

		plt.close('all')
		demo_conditions = self.algorithm.demo_conditions
		failed_demo_conditions = self.algorithm.failed_conditions
		dists_to_target = np.around(dists_to_target, decimals=2)
		ioc_conditions_x = [ioc_conditions[i][0] for i in xrange(len(ioc_conditions))]
		ioc_conditions_y = [ioc_conditions[i][1] for i in xrange(len(ioc_conditions))]
		failed_ioc_x = [failed_conditions[i][0] for i in xrange(len(failed_conditions))]
		failed_ioc_y = [failed_conditions[i][1] for i in xrange(len(failed_conditions))]
		success_ioc_x = [sample_conditions[i][0] for i in xrange(len(sample_conditions))]
		success_ioc_y = [sample_conditions[i][1] for i in xrange(len(sample_conditions))]
		demo_conditions_x = [demo_conditions[i][0] for i in xrange(len(demo_conditions))]
		demo_conditions_y = [demo_conditions[i][1] for i in xrange(len(demo_conditions))]
		failed_conditions_x = [failed_demo_conditions[i][0] for i in xrange(len(failed_demo_conditions))]
		failed_conditions_y = [failed_demo_conditions[i][1] for i in xrange(len(failed_demo_conditions))]
		subplt = plt.subplot()
		subplt.plot(demo_conditions_x, demo_conditions_y, 'go')
		subplt.plot(failed_conditions_x, failed_conditions_y, 'rx')
		subplt.plot(success_ioc_x, success_ioc_y, 'g^')
		subplt.plot(failed_ioc_x, failed_ioc_y, 'rv')
		# plt.legend(['demo_cond', 'failed_badmm', 'success_ioc', 'failed_ioc'], loc= (1, 1))
		for i, txt in enumerate(dists_to_target):
			subplt.annotate(txt, (ioc_conditions_x[i], ioc_conditions_y[i]))
		ax = plt.gca()
		ax.add_patch(Rectangle((-0.08, -0.08), 0.16, 0.16, fill = False, edgecolor = 'blue'))
		box = subplt.get_position()
		subplt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
		subplt.legend(['demo_cond', 'failed_demo', 'success_ioc', 'failed_ioc'], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
						shadow=True, ncol=2)
		plt.title("Distribution of demos and IOC samples' initial conditions")
		# plt.xlabel('width')
		# plt.ylabel('length')
		plt.savefig(self._data_files_dir + 'distribution_of_sample_conditions.png')
		plt.close('all')



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
	parser.add_argument('-b', '--bootstrap', metavar='N', type=int,
						help='using bootstrap to collect samples as demos')
	args = parser.parse_args()

	exp_name = args.experiment
	resume_training_itr = args.resume
	test_policy_N = args.policy
	measure_samples = args.measure
	compare_costs = args.compare
	learning_from_prior = args.learn
	bootstrap = args.bootstrap

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
	elif measure_samples:

		# gps = GPSMain(hyperparams.config)
		# pol_iter = gps.algorithm._hyperparams['iterations']
		# mean_dists, success_rates = gps.measure_distance_and_success()
		# plt.close()
		# plt.plot(range(pol_iter), mean_dists, 'ro', range(pol_iter), mean_dists, '')
		# for i, txt in enumerate(mean_dists):
		# 	plt.annotate(np.around(txt,decimals=2), (i, txt))
		# plt.title("mean distances to the target during iterations")
		# plt.xlabel("iterations")
		# plt.ylabel("mean distances")
		# plt.savefig(exp_dir + 'data_files/' + 'mean_dists_during_iteration.png')
		# plt.close()
		# plt.plot(range(pol_iter), success_rates, 'ro', range(pol_iter), success_rates, '')
		# for i, txt in enumerate(success_rates):
		# 	plt.annotate(repr(txt*100) + "%", (i, txt))
		# plt.xlabel("iterations")
		# plt.ylabel("success rate")
		# plt.title("success rates during iterations")
		# plt.savefig(exp_dir + 'data_files/' + 'success_rate_during_iteration.png')
		for itr in xrange(3):
			random.seed(itr)
			np.random.seed(itr)
			hyperparams = imp.load_source('hyperparams', hyperparams_file)
			hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_50samples_%d' % itr + '/'
			if not os.path.exists(exp_dir + 'data_files_50samples_%d' % itr + '/'):
				os.makedirs(exp_dir + 'data_files_50samples_%d' % itr + '/')
			gps_samples = GPSMain(hyperparams.config)
			agent_config = gps_samples._hyperparams['agent'] 
			plt.close()
			gps_samples.good_samples(measure_samples, agent_config)

			hyperparams = imp.load_source('hyperparams', hyperparams_file)
			hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_global_%d' % itr + '/'
			if not os.path.exists(exp_dir + 'data_files_global_%d' % itr + '/'):
				os.makedirs(exp_dir + 'data_files_global_%d' % itr + '/')
			gps_synthetic = GPSMain(hyperparams.config)
			plt.close()
			gps_synthetic.good_samples(measure_samples, agent_config)

			
	elif compare_costs:
		from gps.algorithm.policy.lin_gauss_init import init_lqr

		mean_dists_global_dict, mean_dists_no_global_dict, success_rates_global_dict, \
				success_rates_no_global_dict, mean_dists_classic_dict, success_rates_classic_dict \
				 = {}, {}, {}, {}, {}, {}
		seeds = [0, 1, 3] # Seed 1, 2 not working for on classic nn
		# var_mults = [8.0, 10.0, 16.0] # 12 doesn't work
		for itr in seeds:
		# for itr in xrange(3):
			random.seed(itr)
			np.random.seed(itr)
			exp_dir_classic = exp_dir.replace('on_global', 'on_classic')
			hyperparams_file_classic = exp_dir_classic + 'hyperparams.py'
			hyperparams = imp.load_source('hyperparams', hyperparams_file)
			# hyperparams.config['algorithm']['init_traj_distr']['type'] = init_lqr
			# hyperparams.config['algorithm']['global_cost'] = True
			hyperparams.config['common']['nn_demo'] = True
			hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_nn_%d' % itr + '/'
			if not os.path.exists(exp_dir + 'data_files_nn_%d' % itr + '/'):
				os.makedirs(exp_dir + 'data_files_nn_%d' % itr + '/')
			# hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_nn_var_%d' % var_mults[itr] + '/'
			# if not os.path.exists(exp_dir + 'data_files_nn_var_%d' % var_mults[itr] + '/'):
			# 	os.makedirs(exp_dir + 'data_files_nn_var_%d' % var_mults[itr] + '/')
			# hyperparams.config['algorithm']['init_var_mult'] = var_mults[itr]
			# hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_no_demo_ini_%d' % itr + '/'
			# if not os.path.exists(exp_dir + 'data_files_no_demo_ini_%d' % itr + '/'):
			# 	os.makedirs(exp_dir + 'data_files_no_demo_ini_%d' % itr + '/')
			gps_global = GPSMain(hyperparams.config)
			pol_iter = gps_global.algorithm._hyperparams['iterations']
			# for i in xrange(pol_iter):
			if hyperparams.config['gui_on']:
				gps_global.run()
				# gps_global.test_policy(itr=i, N=compare_costs)
				plt.close()
			else:
				gps_global.run()
				# gps_global.test_policy(itr=i, N=compare_costs)
				plt.close()
			mean_dists_global_dict[itr], success_rates_global_dict[itr] = gps_global.measure_distance_and_success()
			# Plot the distribution of demos.
			# from matplotlib.patches import Rectangle

			# demo_conditions = gps_global.algorithm.demo_conditions
			# failed_conditions = gps_global.algorithm.failed_conditions
			# demo_conditions_x = [demo_conditions[i][0] for i in xrange(len(demo_conditions))]
			# demo_conditions_y = [demo_conditions[i][1] for i in xrange(len(demo_conditions))]
			# failed_conditions_x = [failed_conditions[i][0] for i in xrange(len(failed_conditions))]
			# failed_conditions_y = [failed_conditions[i][1] for i in xrange(len(failed_conditions))]
			# subplt = plt.subplot()
			# subplt.plot(demo_conditions_x, demo_conditions_y, 'go')
			# subplt.plot(failed_conditions_x, failed_conditions_y, 'rx')
			# ax = plt.gca()
			# ax.add_patch(Rectangle((-0.08, -0.08), 0.16, 0.16, fill = False, edgecolor = 'blue'))
			# box = subplt.get_position()
			# subplt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
			# subplt.legend(['demo_cond', 'failed_badmm'], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
			# 				shadow=True, ncol=2)
			# plt.title("Distribution of demo conditions")
			# # plt.xlabel('width')
			# # plt.ylabel('length')
			# plt.savefig(exp_dir + 'distribution_of_demo_conditions_seed.png')
			plt.close()
			hyperparams = imp.load_source('hyperparams', hyperparams_file)
			# hyperparams.config['algorithm']['init_traj_distr']['type'] = init_lqr
			# hyperparams.config['algorithm']['global_cost'] = True
			hyperparams.config['common']['nn_demo'] = False
			hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_LG_%d' % itr + '/'
			if not os.path.exists(exp_dir + 'data_files_LG_%d' % itr + '/'):
				os.makedirs(exp_dir + 'data_files_LG_%d' % itr + '/')
			# hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_no_demo_ini_%d' % itr + '/'
			# if not os.path.exists(exp_dir + 'data_files_no_demo_ini_%d' % itr + '/'):
			# 	os.makedirs(exp_dir + 'data_files_no_demo_ini_%d' % itr + '/')
			gps_classic = GPSMain(hyperparams.config)
			pol_iter = gps_classic.algorithm._hyperparams['iterations']
			# for i in xrange(pol_iter):
			# if itr != 0 and itr != 1:
			# 	if hyperparams.config['gui_on']:
			# 		gps_classic.run()
			# 		# gps_global.test_policy(itr=i, N=compare_costs)
			# 		plt.close()
			# 	else:
			# 		gps_classic.run()
			# 		# gps_global.test_policy(itr=i, N=compare_costs)
			# 		plt.close()
			mean_dists_classic_dict[itr], success_rates_classic_dict[itr] = gps_classic.measure_distance_and_success()
			plt.close()

			# hyperparams = imp.load_source('hyperparams', hyperparams_file)
			# # hyperparams.config['algorithm']['global_cost'] = False
			# hyperparams.config['common']['nn_demo'] = False
			# hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_global_%d' % itr + '/' #use global as sparse demos
			# if not os.path.exists(exp_dir + 'data_files_global_%d' % itr + '/'):
			# 	os.makedirs(exp_dir + 'data_files_global_%d' % itr + '/')
			# gps = GPSMain(hyperparams.config)
			# pol_iter = gps.algorithm._hyperparams['iterations']
			# # for i in xrange(pol_iter):
			# if hyperparams.config['gui_on']:
			# 	gps.run()
			# 	# gps.test_policy(itr=i, N=compare_costs)
			# 	plt.close()
			# else:
			# 	gps.run()
			# 	# gps.test_policy(itr=i, N=compare_costs)
			# mean_dists_no_global_dict[itr], success_rates_no_global_dict[itr] = gps.measure_distance_and_success()

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
		# plt.plot(range(pol_iter), mean_dists_global_dict[0], '-x', color='red')
		# plt.plot(range(pol_iter), mean_dists_global_dict[1], '-x', color='green')
		# plt.plot(range(pol_iter), mean_dists_global_dict[2], '-x', color='blue')
		# for i, txt in enumerate(avg_dists_global):
		# 	plt.annotate(np.around(txt, decimals=2), (i, txt))
		# for i, txt in enumerate(avg_dists_no_global):
		# 	plt.annotate(np.around(txt, decimals=2), (i, txt))
		plt.legend(['avg nn demo', 'avg LG demo', 'nn demo', 'LG demo'], loc='upper right', ncol=2)
		# plt.legend(['var 8', 'var 10', 'var 16'], loc='upper right', ncol=3)
		# plt.legend(['avg lqr', 'avg demo', 'init lqr', 'init demo'], loc='upper right', ncol=2)		
		plt.title("mean distances to the target over time with nn and LG demo")
		# plt.title("mean distances to the target over time with different initial policy variance")
		# plt.title("mean distances to the target during iterations with and without demo init")
		plt.xlabel("iterations")
		plt.ylabel("mean distances")
		plt.savefig(exp_dir + 'mean_dists_during_iteration_comparison.png')
		# plt.savefig(exp_dir + 'mean_dists_during_iteration_var.png')
		plt.close()
		plt.plot(range(pol_iter), avg_succ_rate_global, '-x', color='red')
		plt.plot(range(pol_iter), avg_succ_rate_classic, '-x', color='green')
		# plt.plot(range(pol_iter), avg_succ_rate_no_global, '-x', color='green')
		# plt.plot(range(pol_iter), success_rates_global_dict[0], '-x', color='red')
		# plt.plot(range(pol_iter), success_rates_global_dict[1], '-x', color='green')
		# plt.plot(range(pol_iter), success_rates_global_dict[2], '-x', color='blue')
		for i in seeds:
			plt.plot(range(pol_iter), success_rates_global_dict[i], 'ko')
			plt.plot(range(pol_iter), success_rates_classic_dict[i], 'co')
			# plt.plot(range(pol_iter), success_rates_no_global_dict[i], 'co')
		# for i, txt in enumerate(avg_succ_rate_global):
		# 	plt.annotate(repr(txt*100) + "%", (i, txt))
		# for i, txt in enumerate(avg_succ_rate_no_global):
		# 	plt.annotate(repr(txt*100) + "%", (i, txt))
		# plt.legend(['var 8', 'var 10', 'var 16'], loc='upper right', ncol=3)
		plt.legend(['avg nn demo', 'avg LG demo', 'nn demo', 'LG demo'], loc='upper right', ncol=2)
		# plt.legend(['avg lqr', 'avg demo', 'init lqr', 'init demo'], loc='upper right', ncol=2)
		plt.xlabel("iterations")
		plt.ylabel("success rate")
		plt.title("success rates during iterations with with nn and LG demo")
		# plt.title("success rates during iterations with different initial policy variance")
		# plt.title("success rates during iterations with and without demo initialization")
		plt.savefig(exp_dir + 'success_rate_during_iteration_comparison.png')
		# plt.savefig(exp_dir + 'success_rate_during_iteration_var.png')

		plt.close()

	elif learning_from_prior:
		ioc_conditions = [np.array([random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)]), \
						random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)]), 0.02]) for i in xrange(10)]
		top_bottom = [np.array([np.random.uniform(-0.08, 0.08), \
						random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)]), 0.02]) for i in xrange(10)]
		left_right = [np.array([random.choice([np.random.uniform(-0.15, -0.09), np.random.uniform(0.09, 0.15)]), \
						np.random.uniform(-0.08, 0.08), 0.02]) for i in xrange(10)]
		ioc_conditions.extend(top_bottom)
		ioc_conditions.extend(left_right)
		exp_iter = hyperparams.config['algorithm']['iterations']
		data_files_dir = exp_dir + 'data_files/'
		mean_dists = []
		success_ioc_samples = []
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
			if mean_dists[i] <= 0.1:
				success_ioc_samples.append(gps.algorithm.min_sample)
			print "iteration " + repr(i) + ": mean dist is " + repr(mean_dists[i])
		with open(exp_dir + 'log.txt', 'a') as f:
			f.write('\nThe 50 IOC conditions are: \n' + str(ioc_conditions) + '\n')
		plt.plot(pos_body_offset_dists, mean_dists, 'ro')
		plt.title("Learning from prior experience using peg insertion with nonzero z position")
		plt.xlabel('pos body offset distances to the origin')
		plt.ylabel('mean distances to the target')
		plt.savefig(data_files_dir + 'learning_from_prior_nonzero_z.png')
		plt.close()

		from matplotlib.patches import Rectangle

		ioc_conditions_x = [ioc_conditions[i][0] for i in xrange(len(ioc_conditions))]
		ioc_conditions_y = [ioc_conditions[i][1] for i in xrange(len(ioc_conditions))]
		mean_dists = np.around(mean_dists, decimals=2)
		failed_ioc_x = [ioc_conditions_x[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] > 0.1]
		failed_ioc_y = [ioc_conditions_y[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] > 0.1]
		success_ioc_x = [ioc_conditions_x[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] <= 0.1]
		success_ioc_y = [ioc_conditions_y[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] <= 0.1]
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
		plt.title("Distribution of neural network and IOC's initial conditions with z=0.02")
		# plt.xlabel('width')
		# plt.ylabel('length')
		plt.savefig(data_files_dir + 'distribution_of_conditions_nonzero_z.png')
		plt.close()
		new_demos = SampleList(success_ioc_samples)
		new_demo_store = {'demoU': new_demos.get_U(), 'demoX': new_demos.get_X(), 'demoO': new_demos.get_obs()}
		gps.data_logger.pickle(gps._data_files_dir + 'new_demos.pkl', new_demo_store)
		if bootstrap:
			# TODO: use successful samples as demonstrations.
			success_ioc_conditions = [ioc_conditions[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] <= 0.1]
			failed_ioc_conditions = [ioc_conditions[i] for i in xrange(len(ioc_conditions)) if mean_dists[i] > 0.1]
			demo_file = exp_dir + 'data_files/' + 'demos.pkl' # for mdgps experiment
			demos = gps.data_logger.unpickle(demo_file)
			new_sample_list = SampleList(success_ioc_samples)
			demos['demoU'] = np.vstack((demos['demoU'], new_sample_list.get_U()))
			demos['demoX'] = np.vstack((demos['demoX'], new_sample_list.get_X()))
			demos['demoO'] = np.vstack((demos['demoO'], new_sample_list.get_obs()))
			gps.data_logger.pickle(demo_file, demos)
			# harder_ioc_conditions = [np.array([random.choice([np.random.uniform(-0.2, -0.09), np.random.uniform(0.09, 0.2)]), \
			# 			random.choice([np.random.uniform(-0.2, -0.09), np.random.uniform(0.09, 0.2)])]) for i in xrange(1)]
			# top_bottom = [np.array([np.random.uniform(-0.08, 0.08), \
			# 				random.choice([np.random.uniform(-0.2, -0.09), np.random.uniform(0.09, 0.2)])]) for i in xrange(1)]
			# left_right = [np.array([random.choice([np.random.uniform(-0.2, -0.09), np.random.uniform(0.09, 0.2)]), \
			# 				np.random.uniform(-0.08, 0.08)]) for i in xrange(1)]
			# harder_ioc_conditions.extend(top_bottom + left_right)
			mean_dists = []
			# success_ioc_samples = []
			pos_body_offset_dists = [np.linalg.norm(failed_ioc_conditions[i]) for i in xrange(len(failed_ioc_conditions))]
			for i in xrange(len(failed_ioc_conditions)):
				hyperparams = imp.load_source('hyperparams', hyperparams_file)
				# hyperparams.config['gui_on'] = False
				hyperparams.config['algorithm']['ioc_cond'] = failed_ioc_conditions[i]
				gps = GPSMain(hyperparams.config)
				gps.agent._hyperparams['pos_body_offset'] = [failed_ioc_conditions[i]]
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
				# if i == 0:
				# 	demo_conditions = gps.algorithm.demo_conditions
				# 	failed_conditions = gps.algorithm.failed_conditions
				mean_dists.append(gps.algorithm.dists_to_target[exp_iter - 1][0]) # Assuming 1 condition
				# if mean_dists[i] <= 0.1:
				# 	success_ioc_samples.extend(gps._take_policy_samples(1))
				print "iteration " + repr(i) + ": mean dist is " + repr(mean_dists[i])
			with open(exp_dir + 'log.txt', 'a') as f:
				f.write('\nThe 50 IOC conditions are: \n' + str(ioc_conditions) + '\n')
			plt.plot(pos_body_offset_dists, mean_dists, 'ro')
			plt.title("Learning from prior experience using peg insertion and bootstrap")
			plt.xlabel('pos body offset distances to the origin')
			plt.ylabel('mean distances to the target')
			plt.savefig(data_files_dir + 'learning_from_prior_bootstrap.png')
			plt.close()

			from matplotlib.patches import Rectangle

			failed_ioc_conditions_x = [failed_ioc_conditions[i][0] for i in xrange(len(failed_ioc_conditions))]
			failed_ioc_conditions_y = [failed_ioc_conditions[i][1] for i in xrange(len(failed_ioc_conditions))]
			mean_dists = np.around(mean_dists, decimals=2)
			failed_ioc_x = [failed_ioc_conditions_x[i] for i in xrange(len(failed_ioc_conditions_x)) if mean_dists[i] > 0.1]
			failed_ioc_y = [failed_ioc_conditions_y[i] for i in xrange(len(failed_ioc_conditions_x)) if mean_dists[i] > 0.1]
			success_ioc_x = [failed_ioc_conditions_x[i] for i in xrange(len(failed_ioc_conditions_y)) if mean_dists[i] <= 0.1]
			success_ioc_y = [failed_ioc_conditions_y[i] for i in xrange(len(failed_ioc_conditions_y)) if mean_dists[i] <= 0.1]
			demo_conditions_x = [demo_conditions[i][0] for i in xrange(len(demo_conditions))]
			demo_conditions_y = [demo_conditions[i][1] for i in xrange(len(demo_conditions))]
			failed_conditions_x = [failed_conditions[i][0] for i in xrange(len(failed_conditions))]
			failed_conditions_y = [failed_conditions[i][1] for i in xrange(len(failed_conditions))]
			success_sample_x = [success_ioc_conditions[i][0] for i in xrange(len(success_ioc_conditions))]
			success_sample_y = [success_ioc_conditions[i][1] for i in xrange(len(success_ioc_conditions))]
			# failed_sample_x = [failed_ioc_conditions[i][0] for i in xrange(len(failed_ioc_conditions))]
			# failed_sample_y = [failed_ioc_conditions[i][1] for i in xrange(len(failed_ioc_conditions))]
			subplt = plt.subplot()
			subplt.plot(demo_conditions_x, demo_conditions_y, 'go')
			subplt.plot(failed_conditions_x, failed_conditions_y, 'rx')
			subplt.plot(success_sample_x, success_sample_y, 'g^')
			# subplt.plot(failed_sample_x, failed_sample_y, 'rv')
			subplt.plot(success_ioc_x, success_ioc_y, 'bp')
			subplt.plot(failed_ioc_x, failed_ioc_y, 'k*')
			# plt.legend(['demo_cond', 'failed_badmm', 'success_ioc', 'failed_ioc'], loc= (1, 1))
			for i, txt in enumerate(mean_dists):
				subplt.annotate(txt, (failed_ioc_conditions_x[i], failed_ioc_conditions_y[i]))
			ax = plt.gca()
			ax.add_patch(Rectangle((-0.08, -0.08), 0.16, 0.16, fill = False, edgecolor = 'blue'))
			box = subplt.get_position()
			subplt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
			subplt.legend(['demo_cond', 'failed_badmm', 'success_sample', 'success_boot', 'failed_boot'], \
							loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
			plt.title("Distribution of neural network and IOC's initial conditions using bootstrap")
			# plt.xlabel('width')
			# plt.ylabel('length')
			plt.savefig(data_files_dir + 'distribution_of_conditions_bootstrap.png')
	else:
		# hyperparams = imp.load_source('hyperparams', hyperparams_file)
		# hyperparams.config['common']['data_files_dir'] = exp_dir + 'data_files_more_demos/' #use global as sparse demos
		# if not os.path.exists(exp_dir + 'data_files_more_demos/'):
		# 	os.makedirs(exp_dir + 'data_files_more_demos/')
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
