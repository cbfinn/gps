""" This file generates more demonstration positions for MJC peg insertion experiment. """
import numpy as np
import copy
import numpy.matlib
import random
from gps.sample.sample import Sample

from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
from gps.utility.general_utils import flatten_lists


def generate_pos_body_offset(conditions):
	""" Generate position body offset for all conditions. """
	pos_body_offset = []
	for i in xrange(conditions):
		# Make the uniform distribution to be [-0.12, 0.12] for learning from prior. For peg ioc, this should be [-0.1, -0.1].
		pos_body_offset.append((0.2 * (np.random.rand(1, 3) - 0.5)).reshape((3, )))
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


def xu_to_sample_list(agent, X, U):
    num = X.shape[0]
    samples = []
    for demo_idx in range(num):
        sample = Sample(agent)
        sample.set_XU(X[demo_idx], U[demo_idx])
        samples.append(sample)
    return SampleList(samples)

def eval_demos(agent, demo_file, costfn, n=10):
    demos = DataLogger.unpickle(demo_file)
    demoX = demos['demoX']
    demoU = demos['demoU']
    return eval_demos_xu(agent, demoX, demoU, costfn, n=n)


def eval_demos_xu(agent, demoX, demoU, costfn, n=-1):
    num_demos = demoX.shape[0]
    losses = []
    for demo_idx in range(num_demos):
        sample = Sample(agent)
        sample.set_XU(demoX[demo_idx], demoU[demo_idx])
        l, _, _, _, _, _ = costfn.eval(sample)
        losses.append(l)
    if n>0:
        return random.sample(losses, n)
    else:
        return losses


def get_target_end_effector(algorithm, condition=0):
    target_dict = algorithm._hyperparams['target_end_effector']
    if type(target_dict) is list:
        target_position = target_dict[condition][:3]
    else:
        target_position = target_dict[:3]
    return target_position

def compute_distance(target_end_effector, sample_list):
    target_position = target_end_effector
    cur_samples = sample_list.get_samples()
    sample_end_effectors = [cur_samples[i].get(END_EFFECTOR_POINTS) for i in xrange(len(cur_samples))]
    dists = [(np.sqrt(np.sum((sample_end_effectors[i][:, :3] - target_position.reshape(1, -1))**2,
                axis=1))) for i in xrange(len(cur_samples))]
    return dists


def compute_distance_cost_plot(algorithm, agent, sample_list):
    if 'target_end_effector' not in algorithm._hyperparams:
        return None
    dists = compute_distance(get_target_end_effector(algorithm), sample_list)
    costs = eval_demos_xu(agent, sample_list.get_X(), sample_list.get_U(), algorithm.cost)
    return flatten_lists(dists), flatten_lists(costs)


def compute_distance_cost_plot_xu(algorithm, agent, X, U):
    if 'target_end_effector' not in algorithm._hyperparams:
        return None
    sample_list = xu_to_sample_list(agent, X, U)
    dists = compute_distance(get_target_end_effector(algorithm), sample_list)
    costs = eval_demos_xu(agent, sample_list.get_X(), sample_list.get_U(), algorithm.cost)
    return flatten_lists(dists), flatten_lists(costs)

