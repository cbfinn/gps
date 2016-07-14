""" This file generates more demonstration positions for MJC peg insertion experiment. """
import numpy as np
import copy
import numpy.matlib

def generate_pos_body_offset(conditions):
	""" Generate position body offset for all conditions. """
	pos_body_offset = []
	for i in xrange(conditions):
		# Make the uniform distribution to be [-0.1, 0.1] for learning from prior. For peg ioc, this should be [-0.5, -0.3].
		pos_body_offset.append(np.hstack((0.2 * (np.random.rand(1, 2) - 0.1), np.zeros((1, 1)))).reshape((3, )))
	return pos_body_offset

def generate_x0(x0, conditions):
	""" Generate initial states for all conditions. """
	x0_lst = [x0.copy() for i in xrange(conditions)]
	for i in xrange(conditions):
		min_pos = np.array(([-2.0], [-0.5], [0.0]))
		max_pos = np.array(([0.2], [0.5], [0.5]))
		direct = np.random.rand(3, 1) * (max_pos - min_pos) + min_pos
		J = np.array(([-0.4233, -0.0164, 0.0], [-0.1610, -0.1183, -0.1373], [0.0191, 0.1850, 0.3279], \
    					[0.1240, 0.2397, -0.2643], [0.0819, 0.2126, -0.0393], [-0.1510, 0.0177, -0.1714], \
						[0.0734, 0.1308, 0.0003]))
		x0_lst[i][range(7)] += J.dot(direct).reshape((7, ))
	return x0_lst

def generate_pos_idx(conditions):
	""" Generate the indices of states. """
	return [np.array([1]) for i in xrange(conditions)]


	