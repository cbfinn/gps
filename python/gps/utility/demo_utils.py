""" This file generates more demonstration positions for MJC peg insertion experiment. """
import numpy as np
import copy
import numpy.matlib
import random
from gps.sample.sample import Sample

from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList

from gps.utility.data_logger import DataLogger
from gps.utility.general_utils import flatten_lists, compute_distance


def generate_pos_body_offset(conditions):
	""" Generate position body offset for all conditions. """
	pos_body_offset = []
	for i in xrange(conditions):
		# Make the uniform distribution to be [-0.15, 0.15] for learning from prior.
		pos_body_offset.append((0.3 * (np.random.rand(1, 3) - 0.5)).reshape((3, )))
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

def extract_samples(itr, sample_file):
    """ Extract samples from iteration 0 up to iteration itr. """
    sample_list = {}
    for i in xrange(itr):
        sample_file_i = sample_file + '_%02d' % itr + '.pkl'
        samples = DataLogger().unpickle(sample_file_i)
        sample_list[i] = samples[0]
    return sample_list

def extract_demos(demo_file):
    demos = DataLogger().unpickle(demo_file)
    return demos['demoX'], demos['demoU'], demos['demoO'], demos.get('demoConditions', None)

def get_target_end_effector(algorithm, condition=0):
    target_dict = algorithm._hyperparams['target_end_effector']
    if type(target_dict) is list:
        target_position = target_dict[condition][:3]
    else:
        target_position = target_dict[:3]
    return target_position

def get_demos(gps):
    """
    Gather the demos for IOC algorithm. If there's no demo file available, generate it.
    Args:
        gps: the gps object.
    Returns: the demo dictionary of demo tracjectories.
    """
    from gps.utility.generate_demo import GenDemo

    if gps._hyperparams['common'].get('nn_demo', False):
        demo_file = gps._hyperparams['common']['NN_demo_file'] # using neural network demos
    else:
        demo_file = gps._hyperparams['common']['LG_demo_file'] # using linear-Gaussian demos
    demos = gps.data_logger.unpickle(demo_file)
    if demos is None:
      gps.demo_gen = GenDemo(gps._hyperparams)
      gps.demo_gen.generate(demo_file, gps.agent)
      demos = gps.data_logger.unpickle(demo_file)
    print 'Num demos:', demos['demoX'].shape[0]
    gps._hyperparams['algorithm']['init_traj_distr']['init_demo_x'] = np.mean(demos['demoX'], 0)
    gps._hyperparams['algorithm']['init_traj_distr']['init_demo_u'] = np.mean(demos['demoU'], 0)
    gps.algorithm = gps._hyperparams['algorithm']['type'](gps._hyperparams['algorithm'])

    return demos
