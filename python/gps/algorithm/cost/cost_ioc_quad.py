""" This file defines quadratic cost function. """
import copy
import numpy as np

from gps.algorithm.cost.config import COST_IOC_QUADRATIC
from gps.algorithm.cost.cost import Cost

class CostIOCQuadratic(Cost):
	""" Set up weighted quadratic norm loss on neural network. """
	def __init__(self, hyperparams):
		config = copy.deepcopy(COST_IOC_QUADRATIC) # Set this up in the config?
		config.update(hyperparams)
		Cost.__init__(self, config)

	def eval(self, sample):
		"""
		Evaluate cost function and derivatives on a sample.
		Args:
			sample:  A single sample
		"""
		pass

	def update(self, demoU, demoX, demoO, dlogis, sampleU, sampleX, sampleO, slogis, \
				eta, itr, algorithm):
		"""
		Learn cost function with generic function representation.
		Args:
			demoU: the actions of demonstrations.
			demoX: the states of demonstrations.
			demoO: the observations of demonstrations.
			dlogis: importance weights for demos.
			sampleU: the actions of samples.
			sampleX: the states of samples.
			sampleO: the observations of samples.
			slogis: importance weights for samples.
			eta: dual variable used in LQR backward pass. Is this still needed?
			itr: current iteration.
			algorithm: current algorithm object.
		"""
		pass
