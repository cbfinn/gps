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
import numpy.matlib

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.utility.data_logger import DataLogger

class GenDemo(object):
	""" Generator of demos. """
	def __init__(self, config):
		self._hyperparameter = config
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
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def generate(self, itr_load):
    	 """ 
    	 Generate demos.
    	 Args:
    	 	itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
         Returns: None.
    	 """
    	# Load the algorithm
    	algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        
        # Roll out the demonstrations from controllers
        controllers = {}
    	M = 40
    	N = 5
    	var_mult = 1.0
    	T = self.algorithm.T
        demos = {}

    	# Store each controller under M conditions into controllers.
    	for i in xrange(M):
    		controllers[i] = self.algorithm.cur[i].traj_distr
    		demos[i] = []
    	controllers_var = copy.copy(controllers)
    	good_indices = range(35)
    	good_indices.extend(range(36, 40))
    	for i in good_indices:

    		# Increase controller variance.
    		controllers_var[i].chol_pol_covar *= var_mult

    		# Gather demos.
    		for j in xrange(N):
    			demo = self.agent.sample(
                	controllers_var[i], i,
                	verbose=(i < self._hyperparams['verbose_trials']),
                	save = True
            	)
            	demos[i].append(demo)

        for i in xrange(M):
            self.algorithm.cur[i].demo_list = sample_list(demos[i])
        # Save the demos.
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )

        # Plot the demonstrations.
        # Maybe use GUI here?
