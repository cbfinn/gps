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
# from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy  # Maybe useful if we unpickle the file as controllers

class GenDemo(object):
	""" Generator of demos. """
	def __init__(self, config):
		self._hyperparameter = config
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
		self.agent = config['agent']['type'](config['agent'])
		self.data_logger = DataLogger()

		config['algorithm']['agent'] = self.agent
		self.algorithm = config['algorithm']['type'](config['algorithm'])

	def generate(self):
		""" 
		 Generate demos and save them in a file for experiment.
		 Returns: None.
		"""
		# Load the algorithm
		algorithm_file = self._data_files_dir + 'algorithm_controllers.pkl' # This should give us the optimal controller. Maybe set to 'controller_itr_%02d.pkl' % itr_load will be better?
		self.algorithm = self.data_logger.unpickle(algorithm_file)
		if self.algorithm is None:
			print("Error: cannot find '%s.'" % algorithm_file)
			os._exit(1) # called instead of sys.exit(), since t
		
		# Roll out the demonstrations from controllers
		var_mult = self.algorithm._hyperparams['var_mult']
		T = self.algorithm.T
		demos = []
		controllers = {}
		M = self.algorithm._hyperparams['demo_cond']
		N = self.algorithm._hyperparams['num_demos']

		# Store each controller under M conditions into controllers.
		for i in xrange(M):
			controllers[i] = self.algorithm.cur[i].traj_distr
		good_indices = self.algorithm._hyperparams['good_indices'] # Do we still need this?
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

		self.algorithm.demo_list = self.algorithm._hyperparams['demo_list'] =  sample_list(shuffle(demos))
		# Save the demos.
		self.data_logger.pickle(
			self._data_files_dir + 'demos.pkl',
			copy.copy(self.algorithm)
		)

		# Plot the demonstrations.
		# Maybe use GUI here?
