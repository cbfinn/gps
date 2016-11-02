""" This file generates more demonstration positions for MJC peg insertion experiment. """
import numpy as np
import copy
import numpy.matlib
import random
from gps.sample.sample import Sample

from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
from gps.utility.data_logger import DataLogger
from gps.utility.general_utils import flatten_lists


def generate_pos_body_offset(conditions):
	""" Generate position body offset for all conditions. """
	pos_body_offset = []
	for i in xrange(conditions):
		# Make the uniform distribution to be [-0.12, 0.12] for learning from prior. For peg ioc, this should be [-0.1, -0.1].
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
        sample_file_i = sample_file + '_%2d' % itr + '.pkl'
        samples = DataLogger().unpickle(sample_file_i)
        sample_list[i] = samples[0]
    return sample_list

def extract_demos(demo_file):
    demos = DataLogger().unpickle(demo_file)
    return demos['demoX'], demos['demoU'], demos['demoO'], demos.get('demoConditions', None)

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
        if type(costfn) is list:
            costfn = costfn[6]
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

def measure_distance_and_success_peg(gps):
    """
    Take the algorithm states for all iterations and extract the
    mean distance to the target position and measure the success
    rate of inserting the peg. (For the peg experiment only)
    Args:
        None
    Returns: the mean distance and the success rate
    """

    pol_iter = gps._hyperparams['algorithm']['iterations']
    peg_height = gps._hyperparams['demo_agent']['peg_height']
    mean_dists = []
    success_rates = []
    for i in xrange(pol_iter):
        # if 'sample_on_policy' in gps._hyperparams['algorithm'] and \
        #     gps._hyperparams['algorithm']['sample_on_policy']:
        #     pol_samples_file = gps._data_files_dir + 'pol_sample_itr_%02d.pkl' % i
        # else:
        pol_samples_file = gps._data_files_dir + 'traj_sample_itr_%02d.pkl.gz' % i
        pol_sample_lists = gps.data_logger.unpickle(pol_samples_file)
        if pol_sample_lists is None:
            print("Error: cannot find '%s.'" % pol_samples_file)
            os._exit(1) # called instead of sys.exit(), since t
        samples = []
        for m in xrange(len(pol_sample_lists)):
            curSamples = pol_sample_lists[m].get_samples()
            for sample in curSamples:
                samples.append(sample)
        if type(gps.algorithm._hyperparams['target_end_effector']) is list:
                target_position = gps.algorithm._hyperparams['target_end_effector'][m][:3]
        else:
            target_position = gps.algorithm._hyperparams['target_end_effector'][:3]
        dists_to_target = [np.nanmin(np.sqrt(np.sum((sample.get(END_EFFECTOR_POINTS)[:, :3] - \
                            target_position.reshape(1, -1))**2, axis = 1)), axis = 0) for sample in samples]
        mean_dists.append(sum(dists_to_target)/len(dists_to_target))
        success_rates.append(float(sum(1 for dist in dists_to_target if dist <= peg_height))/ \
                                len(dists_to_target))
    return mean_dists, success_rates

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

    if gps.algorithm._hyperparams.get('init_demo_policy', False):
        demo_algorithm_file = gps._hyperparams['common']['demo_controller_file']
        demo_algorithm = gps.data_logger.unpickle(demo_algorithm_file)
        if demo_algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        var_mult = gps.algorithm._hyperparams['init_var_mult']
        gps.algorithm.policy_opt.var = demo_algorithm.policy_opt.var.copy() * var_mult
        gps.algorithm.policy_opt.policy = demo_algorithm.policy_opt.copy().policy
        gps.algorithm.policy_opt.policy.chol_pol_covar = np.diag(np.sqrt(gps.algorithm.policy_opt.var))
        gps.algorithm.policy_opt.solver.net.share_with(gps.algorithm.policy_opt.policy.net)

        var_mult = gps.algorithm._hyperparams['demo_var_mult']
        gps.algorithm.demo_policy_opt = demo_algorithm.policy_opt.copy()
        gps.algorithm.demo_policy_opt.var = demo_algorithm.policy_opt.var.copy() * var_mult
        gps.algorithm.demo_policy_opt.policy.chol_pol_covar = np.diag(np.sqrt(gps.algorithm.demo_policy_opt.var))
    return demos
