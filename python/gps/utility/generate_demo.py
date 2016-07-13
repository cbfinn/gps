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
import scipy.io
import numpy.matlib
from random import shuffle

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE
# from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy  # Maybe useful if we unpickle the file as controllers

class GenDemo(object):
	""" Generator of demos. """
	def __init__(self, config):
		self._hyperparams = config
		self._conditions = config['common']['conditions']

		# if 'train_conditions' in config['common']:
		# 	self._train_idx = config['common']['train_conditions']
		# 	self._test_idx = config['common']['test_conditions']
		# else:
		# 	self._train_idx = range(self._conditions)
		# 	config['common']['train_conditions'] = config['common']['conditions']
		# 	self._hyperparams=config
		# 	self._test_idx = self._train_idx
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
		self.algorithm = pickle.load(open(algorithm_file))
		if self.algorithm is None:
			print("Error: cannot find '%s.'" % algorithm_file)
			os._exit(1) # called instead of sys.exit(), since t
		# Keep the initial states of the agent the sames as the demonstrations.
		agent_config = self._hyperparams['demo_agent']
		if agent_config['filename'] == './mjc_models/pr2_arm3d.xml':
			agent_config['x0'] = self.algorithm._hyperparams['agent_x0']
			agent_config['pos_body_idx'] = self.algorithm._hyperparams['agent_pos_body_idx']
			agent_config['pos_body_offset'] = self.algorithm._hyperparams['agent_pos_body_offset']
		self.agent = agent_config['type'](agent_config)
		# Roll out the demonstrations from controllers
		var_mult = self.algorithm._hyperparams['var_mult']
		T = self.algorithm.T
		demos = []
		controllers = {}

		M = agent_config['conditions']
		good_conds = self.ioc_algo._hyperparams['demo_cond']
		N = self.ioc_algo._hyperparams['num_demos']

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

		# Filter out worst (M - good_conds) demos.
		target_position = agent_config['target_end_effector'][:3]
		dists_to_target = np.zeros(M)
		for i in xrange(M):
			demo_end_effector = demos[i].get(END_EFFECTOR_POINTS)
			dists_to_target[i] = np.amin(np.sqrt(np.sum((demo_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
		good_indices = dists_to_target.argsort()[:good_conds - M].tolist()
		filtered_demos = []
		for i in xrange(good_conds):
			filtered_demos.append(demos[i])
		shuffle(filtered_demos)
		demo_list =  SampleList(filtered_demos)
		demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs()}
		# Save the demos.
		self.data_logger.pickle(
			self._data_files_dir + 'demos.pkl',
			copy.copy(demo_store)
		)

