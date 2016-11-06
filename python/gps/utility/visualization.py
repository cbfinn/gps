from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
import numpy as np
import matplotlib.pyplot as plt
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

def compare_samples(gps, N, agent_config, three_dim=True, weight_varying=False, experiment='peg'):
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
    # pol_iter = 9
    algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
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
    # pol_ioc.chol_pol_covar *= 0.0
    pol_demo = algorithm_demo.policy_opt.policy
    pol_oracle = algorithm_oracle.policy_opt.policy
    policies = [pol_ioc, pol_demo, pol_oracle]
    samples = {i: [] for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    if not weight_varying:
        ioc_conditions = agent_config['pos_body_offset']
    else:
        ioc_conditions = [np.array([np.log10(agent_config['density_range'][i]), 0.]) \
                            for i in xrange(M)]
        print M
        print len(agent_config['density_range'])
    for i in xrange(M):
        # Gather demos.
        for j in xrange(N):
            for k in xrange(len(samples)):
                sample = agent.sample(
                    policies[k], i,
                    verbose=(i < gps._hyperparams['verbose_trials']), noisy=True
                    )
                samples[k].append(sample)
    target_position = agent_config['target_end_effector'][:3]
    dists_to_target = [np.zeros((M*N)) for i in xrange(len(samples))]
    dists_diff = []
    # all_success_conditions, only_ioc_conditions, only_demo_conditions, all_failed_conditions, \
    #     percentages = [], [], [], [], []
    ioc_success_conditions, demo_success_conditions, ioc_failed_conditions, demo_failed_conditions, \
        oracle_success_conditions, oracle_failed_conditions, percentages = [], [], [], [], [], [], []
    for i in xrange(len(samples[0])):
        if experiment == 'reacher':
            pos_body_offset = gps.agent._hyperparams['pos_body_offset'][i]
            target_position = np.array([.1,-.1,.01])+pos_body_offset
        for j in xrange(len(samples)):
            sample_end_effector = samples[j][i].get(END_EFFECTOR_POINTS)
            dists_to_target[j][i] = np.nanmin(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
            # Just choose the last time step since it may become unstable after achieving the minimum point.
            # import pdb; pdb.set_trace()
            # dists_to_target[j][i] = np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1]
        # if dists_to_target[0][i] <= 0.15 and dists_to_target[1][i] <= 0.15:
        #     # all_success_conditions.append(ioc_conditions[i])
        #     ioc_success_conditions.append(ioc_conditions[i])
        #     demo_success_conditions.append(ioc_conditions[i])
        # elif dists_to_target[0][i] <= 0.15:
        #     # only_ioc_conditions.append(ioc_conditions[i])
        #     ioc_success_conditions.append(ioc_conditions[i])
        #     demo_failed_conditions.append(ioc_conditions[i])
        # elif dists_to_target[1][i] <= 0.15:
        #     # only_demo_conditions.append(ioc_conditions[i])
        #     ioc_failed_conditions.append(ioc_conditions[i])
        #     demo_success_conditions.append(ioc_conditions[i])
        # else:
        #     # all_failed_conditions.append(ioc_conditions[i])
        #     ioc_failed_conditions.append(ioc_conditions[i])
        #     demo_failed_conditions.append(ioc_conditions[i])
        if dists_to_target[0][i] <= 0.05:
            ioc_success_conditions.append(ioc_conditions[i])
        else:
            ioc_failed_conditions.append(ioc_conditions[i])
        if dists_to_target[1][i] <= 0.05:
            demo_success_conditions.append(ioc_conditions[i])
        else:
            demo_failed_conditions.append(ioc_conditions[i])
        if dists_to_target[2][i] <= 0.05:
            oracle_success_conditions.append(ioc_conditions[i])
        else:
            oracle_failed_conditions.append(ioc_conditions[i])
        # dists_diff.append(np.around(dists_to_target[0][i] - dists_to_target[1][i], decimals=2))
    # percentages.append(round(float(len(all_success_conditions))/len(ioc_conditions), 2))
    # percentages.append(round(float(len(all_failed_conditions))/len(ioc_conditions), 2))
    # percentages.append(round(float(len(only_ioc_conditions))/len(ioc_conditions), 2))
    # percentages.append(round(float(len(only_demo_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(ioc_success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(ioc_failed_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(demo_success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(demo_failed_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(oracle_success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(oracle_failed_conditions))/len(ioc_conditions), 2))
    from matplotlib.patches import Rectangle

    plt.close('all')
    fig = plt.figure()
    ax = Axes3D(fig)
    # ioc_conditions_zip = zip(*ioc_conditions)
    # all_success_zip = zip(*all_success_conditions)
    # all_failed_zip = zip(*all_failed_conditions)
    # only_ioc_zip = zip(*only_ioc_conditions)
    # only_demo_zip = zip(*only_demo_conditions)
    ioc_conditions_zip = zip(*ioc_conditions)
    ioc_success_zip = zip(*ioc_success_conditions)
    ioc_failed_zip = zip(*ioc_failed_conditions)
    demo_success_zip = zip(*demo_success_conditions)
    demo_failed_zip = zip(*demo_failed_conditions)
    oracle_success_zip = zip(*oracle_success_conditions)
    oracle_failed_zip = zip(*oracle_failed_conditions)

    if three_dim:
        ax.scatter(all_success_zip[0], all_success_zip[1], all_success_zip[2], c='y', marker='o')
        ax.scatter(all_failed_zip[0], all_failed_zip[1], all_failed_zip[2], c='r', marker='x')
        ax.scatter(only_ioc_zip[0], only_ioc_zip[1], only_ioc_zip[2], c='g', marker='^')
        ax.scatter(only_demo_zip[0], only_demo_zip[1], only_demo_zip[2], c='r', marker='v')
        training_positions = zip(*pos_body_offset)
        ax.scatter(training_positions[0], training_positions[1], training_positions[2], s=40, c='b', marker='*')
        box = ax.get_position()
    else:
        subplt = plt.subplot()
        # subplt.plot(all_success_zip[0], [x + 0.5 for x in all_success_zip[1]], c='y', marker='o')
        # if len(all_failed_zip) > 0:
        #     subplt.plot(all_failed_zip[0], [x + 0.5 for x in all_failed_zip[1]], c='r', marker='x')
        # else:
        #     subplt.plot([], [], c='r', marker='x')
        # subplt.plot(only_ioc_zip[0], [x + 0.5 for x in only_ioc_zip[1]], c='g', marker='^')
        # if len(only_demo_zip) > 0:
        #     subplt.plot(only_demo_zip[0], [x-0.5 for x in only_demo_zip[1]], c='r', marker='v')
        # else:
        #     subplt.plot([], [], c='r', marker='v')
        subplt.scatter(ioc_success_zip[0], [0.5 for i in xrange(len(ioc_success_zip[1]))], c='g', marker='o')
        subplt.scatter(ioc_failed_zip[0], [0.5 for i in xrange(len(ioc_failed_zip[1]))], c='r', marker='x')
        subplt.scatter(demo_success_zip[0], [0.0 for i in xrange(len(demo_success_zip[1]))], c='g', marker='^')
        subplt.scatter(demo_failed_zip[0], [0.0 for i in xrange(len(demo_failed_zip[1]))], c='r', marker='v')
        subplt.scatter(oracle_success_zip[0], [-0.5 for i in xrange(len(oracle_success_zip[1]))], c='g', marker='*')
        subplt.scatter(oracle_failed_zip[0], [-0.5 for i in xrange(len(oracle_failed_zip[1]))], c='r', marker='s')
        # for i, txt in enumerate(dists_diff):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], ioc_conditions_zip[1][i]))
        for i, txt in enumerate(dists_to_target[0]):
            subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.5))
        for i, txt in enumerate(dists_to_target[1]):
            subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.0))
        for i, txt in enumerate(dists_to_target[2]):
            subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], -0.5))
        ax = plt.gca()
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        # elif experiment == 'reacher':
        #     ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        box = subplt.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
    # ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
    #                 'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
    #                 shadow=True, ncol=2)
    ax.legend(['ioc_success: ' + repr(percentages[0]), 'ioc_failed: ' + repr(percentages[1]), 'demo_success: ' + repr(percentages[2]), \
                    'demo_failed: ' + repr(percentages[3]), 'oracle_success: ' + repr(percentages[4]), 'oracle_failed: ' + repr(percentages[5])], \
                    loc='upper center', bbox_to_anchor=(0.4, -0.05), shadow=True, ncol=3)
    # subplt.plot(all_success_zip[0], [x - 0.5 for x in all_success_zip[1]], c='y', marker='o')
    # if len(all_failed_zip) > 0:
    #     subplt.plot(all_failed_zip[0], [x - 0.5 for x in all_failed_zip[1]], c='r', marker='x')
    # else:
    #     subplt.plot([], [], c='r', marker='x')
    plt.xlabel('log of density')
    plt.title("Distribution of samples drawn from demo policy, IOC policy and oracle policy")
    plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_2.png')
    plt.close('all')


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


def load_alg(alg_file, cond=0):
    from data_logger import open_zip
    import cPickle
    with open_zip(alg_file, 'rb') as pklf:
        ss = pklf.read()
    algorithm = cPickle.loads(ss)
    controller = algorithm.cur[cond].traj_distr
    return controller

def run_alg(test_agent, alg_file, verbose=True):
    from gps.sample.sample_list import SampleList
    controller = load_alg(alg_file)

    agent_config = test_agent
    agent = agent_config['type'](agent_config)

    # Roll out the demonstrations from controllers
    # var_mult = self._hyperparams['algorithm']['demo_var_mult']
    T = controller.T
    demos = []
    demo_idx_conditions = []  # Stores conditions for each demo

    M = len(agent_config['models'])
    N = agent_config['num_demos']
    controllers = {}

    # Store each controller under M conditions into controllers.
    for i in xrange(M):
        controllers[i] = controller
    controllers_var = copy.copy(controllers)
    for i in xrange(M):
        for j in xrange(N):
            demo = agent.sample(
                controllers_var[i], i,
                verbose=j < agent_config['verbose_trials'], noisy=True,
                save = True
            )
            demos.append(demo)
            demo_idx_conditions.append(i)


    if 'filter_demos' in  agent_config:
        filter_options = agent_config['filter_demos']
        filter_type = filter_options.get('type', 'min')
        targets = filter_options['target']
        pos_idx = filter_options['state_idx']
        dist_threshold = filter_options.get('success_upper_bound', 0.01)
        cur_samples = SampleList(demos)
        sample_end_effectors = [cur_samples[i].get_X()[:,pos_idx] for i in xrange(len(cur_samples))]
        dists = [(np.sqrt(np.sum((sample_end_effectors[i] - targets.reshape(1, -1))**2,
                                 axis=1))) for i in xrange(len(cur_samples))]
        failed_idx = []
        for i, distance in enumerate(dists):
            if filter_type == 'last':
                dist_single = distance[-1]
            elif filter_type == 'min':
                dist_single = min(distance)
            else:
                raise ValueError('unrecognized filter type:', filter_type)
            print dist_single
            if(dist_single > dist_threshold):
                failed_idx.append(i)


        print "Removing %d failed demos: %s" %(len(failed_idx), str(failed_idx))
        demos_filtered = [demo for (i, demo) in enumerate(demos) if i not in failed_idx]
        demo_idx_conditions = [cond for (i, cond) in enumerate(demo_idx_conditions) if i not in failed_idx]
        demos = demos_filtered

        # Filter max demos per condition
        condition_to_demo = {
            cond: [demo for (i, demo) in enumerate(demos) if demo_idx_conditions[i]==cond]
            for cond in range(M)
            }
        print 'Successes per condition: %s' % str([len(demo_list) for demo_list in condition_to_demo.values()])
