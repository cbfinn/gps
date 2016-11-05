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
    # pol_iter = 13
    algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + 'data_files_arm/algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_arm/algorithm_itr_09.pkl') # Assuming not using 4 policies
    algorithm_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_arm_oracle/algorithm_itr_09.pkl')
    if not weight_varying:
        pos_body_offset = gps._hyperparams['agent']['pos_body_offset']
    M = agent_config['conditions']
    if experiment == 'reacher' and not weight_varying: #reset body offsets
        np.random.seed(101)
        for m in range(M):
            self.agent.reset_initial_body_offset(m)

    pol_ioc = algorithm_ioc.policy_opt.policy
    pol_sup = algorithm_sup.policy_opt.policy
    # pol_ioc.chol_pol_covar *= 0.0
    pol_demo = algorithm_demo.policy_opt.policy
    pol_oracle = algorithm_oracle.policy_opt.policy
    policies = [pol_ioc, pol_demo, pol_oracle, pol_sup]
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
        oracle_success_conditions, oracle_failed_conditions, sup_success_conditions, \
        sup_failed_conditions, percentages = [], [], [], [], [], [], [], [], []
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
        if dists_to_target[0][i] <= 0.20:
            ioc_success_conditions.append(ioc_conditions[i])
        else:
            ioc_failed_conditions.append(ioc_conditions[i])
        if dists_to_target[1][i] <= 0.20:
            demo_success_conditions.append(ioc_conditions[i])
        else:
            demo_failed_conditions.append(ioc_conditions[i])
        if dists_to_target[2][i] <= 0.20:
            oracle_success_conditions.append(ioc_conditions[i])
        else:
            oracle_failed_conditions.append(ioc_conditions[i])
        if dists_to_target[3][i] <= 0.20:
            sup_success_conditions.append(ioc_conditions[i])
        else:
            sup_failed_conditions.append(ioc_conditions[i])
        # dists_diff.append(np.around(dists_to_target[0][i] - dists_to_target[1][i], decimals=2))
    percentages.append(round(float(len(oracle_success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(oracle_failed_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(ioc_success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(ioc_failed_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(sup_success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(sup_failed_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(demo_success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(demo_failed_conditions))/len(ioc_conditions), 2))
    from matplotlib.patches import Rectangle

    plt.close('all')
    fig = plt.figure(figsize=(8, 4))
    ioc_conditions_zip = zip(*ioc_conditions)
    ioc_success_zip = zip(*ioc_success_conditions)
    ioc_failed_zip = zip(*ioc_failed_conditions)
    demo_success_zip = zip(*demo_success_conditions)
    demo_failed_zip = zip(*demo_failed_conditions)
    oracle_success_zip = zip(*oracle_success_conditions)
    oracle_failed_zip = zip(*oracle_failed_conditions)
    sup_success_zip = zip(*sup_success_conditions)
    sup_failed_zip = zip(*sup_failed_conditions)

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
        subplt.scatter(ioc_success_zip[0], [0.4 for i in xrange(len(ioc_success_zip[1]))], c='g', marker='o', s=50, lw=0)
        if len(ioc_failed_conditions) > 0:
            subplt.scatter(ioc_failed_zip[0], [0.4 for i in xrange(len(ioc_failed_zip[1]))], c='r', marker='x', s=50)
        subplt.scatter(demo_success_zip[0], [0.8 for i in xrange(len(demo_success_zip[1]))], c='g', marker='o', s=50, lw=0)
        if len(demo_failed_conditions) > 0:
            subplt.scatter(demo_failed_zip[0], [0.8 for i in xrange(len(demo_failed_zip[1]))], c='r', marker='x', s=50)
        subplt.scatter(oracle_success_zip[0], [0.2 for i in xrange(len(oracle_success_zip[1]))], c='g', marker='o', s=50, lw=0)
        if len(oracle_failed_conditions) > 0:
            subplt.scatter(oracle_failed_zip[0], [0.2 for i in xrange(len(oracle_failed_zip[1]))], c='r', marker='x', s=50)
        subplt.scatter(sup_success_zip[0], [0.6 for i in xrange(len(sup_success_zip[1]))], c='g', marker='o', s=50, lw=0)
        if len(sup_failed_conditions) > 0:
            subplt.scatter(sup_failed_zip[0], [0.6 for i in xrange(len(sup_failed_zip[1]))], c='r', marker='x', s=50)
        # for i, txt in enumerate(dists_diff):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], ioc_conditions_zip[1][i]))
        # for i, txt in enumerate(dists_to_target[0]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.5))
        # for i, txt in enumerate(dists_to_target[1]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.0))
        # for i, txt in enumerate(dists_to_target[2]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], -0.5))
        ax = plt.gca()
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        pol_names = ['Oracle', 'S3G', 'Sup cost', 'RL policy']
        yrange = [0.2, 0.4, 0.6, 0.8]
        plt.yticks(yrange, pol_names)
        # for i in xrange(len(policies)):
        #     subplt.annotate(pol_names[i], (ax.get_xticks()[0], yrange[i]), horizontalalignment='left')
        for i in xrange(len(policies)):
            subplt.annotate(repr(percentages[2*i]*100) + "%", (ax.get_xticks()[-1], yrange[i]), color='green')
        # elif experiment == 'reacher':
        #     ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        # ax.axes.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both',length=0)
    # ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
    #                 'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
    #                 shadow=True, ncol=2)
    # ax.legend(['ioc_success: ' + repr(percentages[0]), 'ioc_failed: ' + repr(percentages[1]), 'demo_success: ' + repr(percentages[2]), \
    #                 'demo_failed: ' + repr(percentages[3]), 'oracle_success: ' + repr(percentages[4]), 'oracle_failed: ' + repr(percentages[5])], \
    #                 loc='upper center', bbox_to_anchor=(0.4, -0.05), shadow=True, ncol=3)
    # subplt.plot(all_success_zip[0], [x - 0.5 for x in all_success_zip[1]], c='y', marker='o')
    # if len(all_failed_zip) > 0:
    #     subplt.plot(all_failed_zip[0], [x - 0.5 for x in all_failed_zip[1]], c='r', marker='x')
    # else:
    #     subplt.plot([], [], c='r', marker='x')
    plt.xlabel('log density', labelpad=-1)
    # plt.title("Distribution of samples drawn from various policies of 2-link arm task")
    plt.title("Distribution of samples drawn from various policies of pointmass task")
    plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_2.png')
    plt.close('all')


def manual_compare_samples(gps, N, agent_config, three_dim=True, weight_varying=False, experiment='peg'):
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
    M = agent_config['conditions']

    ioc_conditions = [np.array([np.log10(agent_config['density_range'][i])-4.0, 0.]) \
                            for i in xrange(M)]
    print M
    print len(agent_config['density_range'])
    ioc_success_conditions, demo_success_conditions, ioc_failed_conditions, demo_failed_conditions, \
        oracle_success_conditions, oracle_failed_conditions, sup_success_conditions, \
        sup_failed_conditions, percentages = [], [], [], [], [], [], [], [], []
    ioc_success_conditions = ioc_conditions[:M-5]
    ioc_success_conditions.append(ioc_conditions[M-4])
    ioc_medium_conditions = [ioc_conditions[M-5], ioc_conditions[M-3], ioc_conditions[M-2], ioc_conditions[M-1]]
    demo_success_conditions = ioc_conditions[:M-8]
    demo_medium_conditions = ioc_conditions[M-8:M-6]
    demo_failed_conditions = ioc_conditions[M-6:]
    sup_success_conditions = ioc_conditions[:M-5]
    sup_medium_conditions = [ioc_conditions[M-5], ioc_conditions[M-4], ioc_conditions[M-3]]
    sup_failed_conditions = ioc_conditions[M-2:]
    oracle_success_conditions = ioc_conditions[:M-6]
    oracle_medium_conditions = [ioc_conditions[M-5], ioc_conditions[M-4]]
    oracle_failed_conditions = [ioc_conditions[M-6], ioc_conditions[M-3], ioc_conditions[M-2], ioc_conditions[M-1]]
    percentages.append(round(float(15+14+16)/(3*len(ioc_conditions)), 2))
    percentages.append(round(float(19+18+18)/(3*len(ioc_conditions)), 2))
    percentages.append(round(float(15+18+17)/(3*len(ioc_conditions)), 2))
    percentages.append(round(float(12+13+14)/(3*len(ioc_conditions)), 2))
    from matplotlib.patches import Rectangle

    plt.close('all')
    fig = plt.figure(figsize=(11.5, 5))
    # ioc_conditions_zip = zip(*ioc_conditions)
    # all_success_zip = zip(*all_success_conditions)
    # all_failed_zip = zip(*all_failed_conditions)
    # only_ioc_zip = zip(*only_ioc_conditions)
    # only_demo_zip = zip(*only_demo_conditions)
    ioc_conditions_zip = zip(*ioc_conditions)
    ioc_success_zip = zip(*ioc_success_conditions)
    ioc_medium_zip = zip(*ioc_medium_conditions)
    ioc_failed_zip = zip(*ioc_failed_conditions)
    demo_success_zip = zip(*demo_success_conditions)
    demo_medium_zip = zip(*demo_medium_conditions)
    demo_failed_zip = zip(*demo_failed_conditions)
    oracle_success_zip = zip(*oracle_success_conditions)
    oracle_medium_zip = zip(*oracle_medium_conditions)
    oracle_failed_zip = zip(*oracle_failed_conditions)
    sup_success_zip = zip(*sup_success_conditions)
    sup_medium_zip = zip(*sup_medium_conditions)
    sup_failed_zip = zip(*sup_failed_conditions)

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
        orange = (1.0, round(float(180)/255, 2), 0.0)
        subplt.scatter(ioc_success_zip[0], [0.4 for i in xrange(len(ioc_success_zip[1]))], c='g', marker='o', s=70, lw=0)
        if len(ioc_failed_conditions) > 0:
            subplt.scatter(ioc_failed_zip[0], [0.4 for i in xrange(len(ioc_failed_zip[1]))], c='r', marker='x', s=70)
        subplt.scatter(ioc_medium_zip[0], [0.4 for i in xrange(len(ioc_medium_zip[1]))], c='chocolate', marker='^', s=70, lw=0)
        subplt.scatter(demo_success_zip[0], [0.8 for i in xrange(len(demo_success_zip[1]))], c='g', marker='o', s=70, lw=0)
        subplt.scatter(demo_failed_zip[0], [0.8 for i in xrange(len(demo_failed_zip[1]))], c='r', marker='x', s=70)
        subplt.scatter(demo_medium_zip[0], [0.8 for i in xrange(len(demo_medium_zip[1]))], c='chocolate', marker='^', s=70, lw=0)
        subplt.scatter(oracle_success_zip[0], [0.2 for i in xrange(len(oracle_success_zip[1]))], c='g', marker='o', s=70, lw=0)
        subplt.scatter(oracle_failed_zip[0], [0.2 for i in xrange(len(oracle_failed_zip[1]))], c='r', marker='x', s=70)
        subplt.scatter(oracle_medium_zip[0], [0.2 for i in xrange(len(oracle_medium_zip[1]))], c='chocolate', marker='^', s=70, lw=0)
        subplt.scatter(sup_success_zip[0], [0.6 for i in xrange(len(sup_success_zip[1]))], c='g', marker='o', s=70, lw=0)
        subplt.scatter(sup_medium_zip[0], [0.6 for i in xrange(len(sup_medium_zip[1]))], c='chocolate', marker='^', s=70, lw=0)
        subplt.scatter(sup_failed_zip[0], [0.6 for i in xrange(len(sup_failed_zip[1]))], c='r', marker='x', s=70)
        # for i, txt in enumerate(dists_diff):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], ioc_conditions_zip[1][i]))
        # for i, txt in enumerate(dists_to_target[0]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.5))
        # for i, txt in enumerate(dists_to_target[1]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.0))
        # for i, txt in enumerate(dists_to_target[2]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], -0.5))
        ax = plt.gca()
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        pol_names = ['oracle', 'S3G', 'cost regr.', 'RL policy']
        yrange = [0.2, 0.4, 0.6, 0.8]
        plt.yticks(yrange, pol_names)
        ax.set_xlim([2.2, 4.3])
        # for i in xrange(len(policies)):
        #     subplt.annotate(pol_names[i], (ax.get_xticks()[0], yrange[i]), horizontalalignment='left')
        # print ax.get_xticks()[0]
        # for i in xrange(4):
        #     subplt.annotate(pol_names[i], (2.1, yrange[i]), fontsize=22)
        for i in xrange(4):
            subplt.annotate(repr(percentages[i]*100) + "%", (4.3, yrange[i]), color='gray', fontsize=22)
        # elif experiment == 'reacher':
        #     ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        # ax.axes.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both',length=0, labelsize=21)
    # ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
    #                 'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
    #                 shadow=True, ncol=2)
    # ax.legend(['ioc_success: ' + repr(percentages[0]), 'ioc_failed: ' + repr(percentages[1]), 'demo_success: ' + repr(percentages[2]), \
    #                 'demo_failed: ' + repr(percentages[3]), 'oracle_success: ' + repr(percentages[4]), 'oracle_failed: ' + repr(percentages[5])], \
    #                 loc='upper center', bbox_to_anchor=(0.4, -0.05), shadow=True, ncol=3)
    # subplt.plot(all_success_zip[0], [x - 0.5 for x in all_success_zip[1]], c='y', marker='o')
    # if len(all_failed_zip) > 0:
    #     subplt.plot(all_failed_zip[0], [x - 0.5 for x in all_failed_zip[1]], c='r', marker='x')
    # else:
    #     subplt.plot([], [], c='r', marker='x')
    plt.xlabel('log mass', fontsize=22, labelpad=-4)
    plt.title("Generalization for 2-link reacher", fontsize=25)
    plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_average.png')
    plt.close('all')


def manual_compare_samples_curve(gps, N, agent_config, three_dim=True, weight_varying=False, experiment='peg'):
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
    # pol_iter = 13
    algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + 'data_files_arm/algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_arm/algorithm_itr_09.pkl') # Assuming not using 4 policies
    algorithm_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_arm_oracle/algorithm_itr_09.pkl')
    if not weight_varying:
        pos_body_offset = gps._hyperparams['agent']['pos_body_offset']
    M = agent_config['conditions']
    if experiment == 'reacher' and not weight_varying: #reset body offsets
        np.random.seed(101)
        for m in range(M):
            self.agent.reset_initial_body_offset(m)

    pol_ioc = algorithm_ioc.policy_opt.policy
    pol_sup = algorithm_sup.policy_opt.policy
    # pol_ioc.chol_pol_covar *= 0.0
    pol_demo = algorithm_demo.policy_opt.policy
    pol_oracle = algorithm_oracle.policy_opt.policy
    policies = [pol_ioc, pol_sup, pol_demo, pol_oracle]
    successes = {i: np.zeros((20, M)) for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    if not weight_varying:
        ioc_conditions = agent_config['pos_body_offset']
    else:
        ioc_conditions = [np.log10(agent_config['density_range'][i])-4.0 \
                            for i in xrange(M)]
        print M
        print len(agent_config['density_range'])
    pos_body_offset = gps.agent._hyperparams['pos_body_offset'][i]
    target_position = np.array([.1,-.1,.01])+pos_body_offset

    import random

    for seed in xrange(20):
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
                    if dists_to_target <= 0.05:
                        successes[k][seed, i] = 1.0
    
    success_rate_ioc = np.mean(successes[0], axis=0)
    success_rate_sup = np.mean(successes[1], axis=0)
    success_rate_demo = np.mean(successes[2], axis=0)
    success_rate_oracle = np.mean(successes[3], axis=0)
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
        subplt.plot(ioc_conditions, success_rate_ioc, '-rx')
        subplt.plot(ioc_conditions, success_rate_sup, '-gx')
        subplt.plot(ioc_conditions, success_rate_demo, '-bx')
        subplt.plot(ioc_conditions, success_rate_oracle, '-yx')
        ax = plt.gca()
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        # plt.yticks(yrange, pol_names)
        # for i in xrange(len(policies)):
        #     subplt.annotate(pol_names[i], (ax.get_xticks()[0], yrange[i]), horizontalalignment='left')
        # for i in xrange(4):
        #     subplt.annotate(repr(percentages[i]*100) + "%", (ax.get_xticks()[-1], yrange[i]), color='green', fontsize=16)
        # elif experiment == 'reacher':
        #     ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        # ax.axes.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.tick_params(axis='y', which='both',length=0, labelsize=20)
    # ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
    #                 'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
    #                 shadow=True, ncol=2)
    # ax.legend(['ioc_success: ' + repr(percentages[0]), 'ioc_failed: ' + repr(percentages[1]), 'demo_success: ' + repr(percentages[2]), \
    #                 'demo_failed: ' + repr(percentages[3]), 'oracle_success: ' + repr(percentages[4]), 'oracle_failed: ' + repr(percentages[5])], \
    #                 loc='upper center', bbox_to_anchor=(0.4, -0.05), shadow=True, ncol=3)
    ax.legend(['S3G', 'cost regr', 'RL policy', 'oracle'], loc='lower left')
    # subplt.plot(all_success_zip[0], [x - 0.5 for x in all_success_zip[1]], c='y', marker='o')
    # if len(all_failed_zip) > 0:
    #     subplt.plot(all_failed_zip[0], [x - 0.5 for x in all_failed_zip[1]], c='r', marker='x')
    # else:
    #     subplt.plot([], [], c='r', marker='x')
    plt.ylabel('Success Rate', fontsize=22)
    plt.xlabel('Log Mass', fontsize=22, labelpad=-4)
    plt.title("Generalization for 2-link reacher", fontsize=25)
    plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_average_curve.png')
    plt.close('all')

def manual_compare_samples_curve_hard(gps, N, agent_config, three_dim=True, weight_varying=False, experiment='peg'):
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
    ioc_conditions = [np.log10(agent_config['density_range'][i])-4.0 \
                            for i in xrange(10)]

    success_rate_ioc = np.array([1., 1., 1., 1., 1., 1., 0.75, 0.8, 0.65, 0.4])
    success_rate_sup = np.array([1., 1., 1., 1., 1., 0.8, 0.35, 0.15, 0., 0.])
    success_rate_demo = np.array([1., 0.9, 0.8, 0.2, 0., 0., 0., 0., 0., 0.])
    success_rate_oracle = np.array([1., 1., 1., 0.95, 0.95, 0.4, 0.25, 0.05, 0., 0.])


    from matplotlib.patches import Rectangle

    plt.close('all')
    fig = plt.figure(figsize=(8, 6))


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
        subplt.plot(ioc_conditions, 100*success_rate_ioc, '-r', linewidth=6)
        subplt.plot(ioc_conditions, 100*success_rate_sup, '--g', linewidth=6)
        subplt.plot(ioc_conditions, 100*success_rate_demo, ':b', linewidth=6)
        subplt.plot(ioc_conditions, 100*success_rate_oracle, '-.k', linewidth=3)
        ax = plt.gca()
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        # plt.yticks(yrange, pol_names)
        # for i in xrange(len(policies)):
        #     subplt.annotate(pol_names[i], (ax.get_xticks()[0], yrange[i]), horizontalalignment='left')
        # for i in xrange(4):
        #     subplt.annotate(repr(percentages[i]*100) + "%", (ax.get_xticks()[-1], yrange[i]), color='green', fontsize=16)
        # elif experiment == 'reacher':
        #     ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        # ax.axes.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.tick_params(axis='y', which='both',length=0, labelsize=20)
    # ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
    #                 'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
    #                 shadow=True, ncol=2)
    # ax.legend(['ioc_success: ' + repr(percentages[0]), 'ioc_failed: ' + repr(percentages[1]), 'demo_success: ' + repr(percentages[2]), \
    #                 'demo_failed: ' + repr(percentages[3]), 'oracle_success: ' + repr(percentages[4]), 'oracle_failed: ' + repr(percentages[5])], \
    #                 loc='upper center', bbox_to_anchor=(0.4, -0.05), shadow=True, ncol=3)
    ax.legend(['S3G', 'reward regr', 'RL policy', 'oracle'], loc='lower left')
    # subplt.plot(all_success_zip[0], [x - 0.5 for x in all_success_zip[1]], c='y', marker='o')
    # if len(all_failed_zip) > 0:
    #     subplt.plot(all_failed_zip[0], [x - 0.5 for x in all_failed_zip[1]], c='r', marker='x')
    # else:
    #     subplt.plot([], [], c='r', marker='x')
    plt.ylabel('Success Rate', fontsize=25)
    plt.xlabel('Log Mass', fontsize=25, labelpad=-4)
    plt.title("2-link reacher", fontsize=30)
    plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_average_curve.png')
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
