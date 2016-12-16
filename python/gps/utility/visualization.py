from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, RGB_IMAGE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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

TRIALS = 20
THRESH = {'reacher': 0.05, 'pointmass': 0.1}

def compare_samples_curve(gps, N, agent_config, three_dim=True, weight_varying=False, experiment='peg'):
    """
    Compare samples between IOC and demo policies and visualize them in a plot.
    Args:
        gps: GPS object.
        N: number of samples taken from the policy for comparison
        config: Configuration of the agent to sample.
        three_dim: whether the plot is 3D or 2D.
        weight_varying: whether the experiment is weight-varying or not.
        experiment: whether the experiment is peg, reacher or pointmass.
    """
    pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
    algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + 'data_files/algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files/algorithm_itr_09.pkl') # Assuming not using 4 policies
    algorithm_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_oracle/algorithm_itr_09.pkl')
    if not weight_varying:
        pos_body_offset = gps._hyperparams['agent']['pos_body_offset']
    M = agent_config['conditions']
    if experiment == 'reacher' and not weight_varying: #reset body offsets
        np.random.seed(101)
        for m in range(M):
            self.agent.reset_initial_body_offset(m)

    pol_ioc = algorithm_ioc.policy_opt.policy
    pol_sup = algorithm_sup.policy_opt.policy
    pol_demo = algorithm_demo.policy_opt.policy
    pol_oracle = algorithm_oracle.policy_opt.policy
    policies = [pol_ioc, pol_sup, pol_demo, pol_oracle]
    successes = {i: np.zeros((TRIALS, M)) for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    if not weight_varying:
        ioc_conditions = agent_config['pos_body_offset']
    else:
        ioc_conditions = [np.log10(agent_config['density_range'][i])-4.0 \
                            for i in xrange(M)]
    pos_body_offset = gps.agent._hyperparams['pos_body_offset'][i]
    target_position = np.array([.1,-.1,.01])+pos_body_offset

    import random

    for seed in xrange(TRIALS):
        random.seed(seed)
        np.random.seed(seed)
        for i in xrange(M):
            # Gather demos.
            for j in xrange(N):
                for k in xrange(len(policies)):
                    sample = agent.sample(
                        policies[k], i,
                        verbose=(i < gps._hyperparams['verbose_trials']), noisy=True
                        )
                    sample_end_effector = sample.get(END_EFFECTOR_POINTS)
                    dists_to_target = np.nanmin(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
                    if dists_to_target <= THRESH[experiment]:
                        successes[k][seed, i] = 1.0
    
    success_rate_ioc = np.mean(successes[0], axis=0)
    success_rate_sup = np.mean(successes[1], axis=0)
    success_rate_demo = np.mean(successes[2], axis=0)
    success_rate_oracle = np.mean(successes[3], axis=0)
    print "ioc mean: " + repr(success_rate_ioc.mean())
    print "sup mean: " + repr(success_rate_sup.mean())
    print "demo mean: " + repr(success_rate_demo.mean())
    print "oracle mean: " + repr(success_rate_oracle.mean())
    print "ioc: " + repr(success_rate_ioc)
    print "sup: " + repr(success_rate_sup)
    print "demo: " + repr(success_rate_demo)
    print "oracle: " + repr(success_rate_oracle)


    from matplotlib.patches import Rectangle

    plt.close('all')
    fig = plt.figure(figsize=(8, 5))


    if three_dim:
        ax = Axes3D(fig)
        ax.scatter(all_success_zip[0], all_success_zip[1], all_success_zip[2], c='y', marker='o')
        ax.scatter(all_failed_zip[0], all_failed_zip[1], all_failed_zip[2], c='r', marker='x')
        ax.scatter(only_ioc_zip[0], only_ioc_zip[1], only_ioc_zip[2], c='g', marker='^')
        ax.scatter(only_demo_zip[0], only_demo_zip[1], only_demo_zip[2], c='r', marker='v')
        training_positions = zip(*pos_body_offset)
        ax.scatter(training_positions[0], training_positions[1], training_positions[2], s=40, c='b', marker='*')
        box = ax.get_position()
    else:
        subplt = plt.subplot()
        subplt.plot(ioc_conditions, success_rate_ioc, '-rx')
        subplt.plot(ioc_conditions, success_rate_sup, '-gx')
        subplt.plot(ioc_conditions, success_rate_demo, '-bx')
        subplt.plot(ioc_conditions, success_rate_oracle, '-yx')
        ax = plt.gca()
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax.legend(['S3G', 'cost regr', 'RL policy', 'oracle'], loc='lower left')
    plt.ylabel('Success Rate', fontsize=22)
    plt.xlabel('Log Mass', fontsize=22, labelpad=-4)
    plt.title("Generalization for 2-link reacher", fontsize=25)
    plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_average_curve.png')
    plt.close('all')

def visualize_samples(gps, N, agent_config, experiment='reacher'):
    """
    Compare samples between IOC and demo policies and visualize them in a plot.
    Args:
        gps: GPS object.
        N: number of samples taken from the policy for comparison
        config: Configuration of the agent to sample.
        experiment: whether the experiment is peg, reacher or pointmass.
    """
    pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
    algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + 'data_files/algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files/algorithm_itr_09.pkl') # Assuming not using 4 policies
    algorithm_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_oracle/algorithm_itr_09.pkl')
    M = agent_config['conditions']
    pol_ioc = algorithm_ioc.policy_opt.policy
    pol_sup = algorithm_sup.policy_opt.policy
    pol_demo = algorithm_demo.policy_opt.policy
    pol_oracle = algorithm_oracle.policy_opt.policy
    policies = [pol_ioc, pol_demo, pol_oracle, pol_sup]
    samples = {i: [] for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    ioc_conditions = [np.array([np.log10(agent_config['density_range'][i]), 0.]) \
                        for i in xrange(M)]
    print "Number of testing conditions: %d" % M

    from gps.utility.general_utils import mkdir_p

    if 'record_gif' in gps._hyperparams:
        gif_config = gps._hyperparams['record_gif']
        gif_fps = gif_config.get('fps', None)
        gif_dir = gif_config.get('gif_dir', gps._hyperparams['common']['data_files_dir'])
        mkdir_p(gif_dir)
    for i in xrange(M):
        # Gather demos.
        for j in xrange(N):
            for k in xrange(len(samples)):
                gif_name = os.path.join(gif_dir, 'pol%d_cond%d.gif' % (k, i))
                sample = agent.sample(
                    policies[k], i,
                    verbose=(i < gps._hyperparams['verbose_trials']), noisy=True, record_image=True, \
                    record_gif=gif_name, record_gif_fps=gif_fps
                    )
                samples[k].append(sample)

def get_comparison_hyperparams(hyperparam_file, itr):
    """ 
    Make the iteration number the same as the experiment data directory index.
    Args:
        hyperparam_file: the hyperparam file to be changed for two different experiments for comparison.
    """
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    hyperparams.config['plot']['itr'] = itr
    return hyperparams

def compare_experiments(mean_dists_1_dict, mean_dists_2_dict, success_rates_1_dict, \
                                success_rates_2_dict, pol_iter, exp_dir, hyperparams_compare, \
                                hyperparams):
    """ 
    Compare the performance of two experiments and plot their mean distance to target effector and success rate.
    Args:
        mean_dists_1_dict: mean distance dictionary for one of two experiments to be compared.
        mean_dists_2_dict: mean distance dictionary for one of two experiments to be compared.
        success_rates_1_dict: success rates dictionary for one of the two experiments to be compared.
        success_rates_2_dict: success rates dictionary for one of the two experiments to be compared.
        pol_iter: number of iterations of the algorithm.
        exp_dir: directory of the experiment.
        hyperparams_compare: the hyperparams of the control group.
        hyperparams: the hyperparams of the experimental group.
    """

    plt.close('all')
    avg_dists_1 = [float(sum(mean_dists_1_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_succ_rate_1 = [float(sum(success_rates_1_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_dists_2 = [float(sum(mean_dists_2_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_succ_rate_2 = [float(sum(success_rates_2_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    plt.plot(range(pol_iter), avg_dists_1, '-x', color='red')
    plt.plot(range(pol_iter), avg_dists_2, '-x', color='green')
    for i in seeds:
        plt.plot(range(pol_iter), mean_dists_1_dict[i], 'ko')
        plt.plot(range(pol_iter), mean_dists_2_dict[i], 'co')
    avg_legend, legend, avg_compare_legend, compare_legend = hyperparams_compare['plot']['avg_legend'], \
            hyperparams_compare['plot']['legend'], hyperparams_compare['plot']['avg_legend_compare'], \
            hyperparams_compare['plot']['legend_compare']
    plt.legend([avg_legend, legend, avg_compare_legend, compare_legend], loc='upper right', ncol=2)
    plt.title(hyperparams_compare['plot']['mean_dist_title'])
    plt.xlabel(hyperparams_compare['plot']['xlabel'])
    plt.ylabel(hyperparams_compare['plot']['ylabel'])
    plt.savefig(exp_dir + hyperparams_compare['plot']['mean_dist_plot_name'])
    plt.close()
    plt.plot(range(pol_iter), avg_succ_rate_1, '-x', color='red')
    plt.plot(range(pol_iter), avg_succ_rate_2, '-x', color='green')
    for i in seeds:
        plt.plot(range(pol_iter), success_rates_1_dict[i], 'ko')
        plt.plot(range(pol_iter), success_rates_2_dict[i], 'co')
    plt.legend([avg_legend, legend, avg_compare_legend, compare_legend], loc='upper right', ncol=2)
    plt.xlabel(hyperparams_compare['plot']['xlabel'])
    plt.ylabel(hyperparams_compare['plot']['ylabel'])
    plt.title(hyperparams_compare['success_title'])
    plt.savefig(exp_dir + hyperparams_compare['plot']['success_plot_name'])

    plt.close()
