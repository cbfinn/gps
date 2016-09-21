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

def compare_samples(gps, N, agent_config):
        """
        Compare samples between IOC and demo policies and visualize them in a 3D plot.
        Args:
            gps: GPS object.
            N: number of samples taken from the policy for comparison
            config: Configuration of the agent to sample.
        """
        pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
        algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
        algorithm_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_maxent_9cond_z_train_demo_1/algorithm_itr_11.pkl') # Assuming not using 4 policies
        pos_body_offset = gps._hyperparams['agent']['pos_body_offset']
        M = agent_config['conditions']

        pol_ioc = algorithm_ioc.policy_opt.policy
        # pol_ioc.chol_pol_covar *= 0.0
        pol_demo = algorithm_demo.policy_opt.policy
        policies = [pol_ioc, pol_demo]
        samples = {i: [] for i in xrange(len(policies))}
        agent = agent_config['type'](agent_config)
        ioc_conditions = agent_config['pos_body_offset']
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
        all_success_conditions = []
        only_ioc_conditions = []
        only_demo_conditions = []
        all_failed_conditions = []
        percentages = []
        for i in xrange(len(samples[0])):
            for j in xrange(len(samples)):
                sample_end_effector = samples[j][i].get(END_EFFECTOR_POINTS)
                dists_to_target[j][i] = np.nanmin(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
                # Just choose the last time step since it may become unstable after achieving the minimum point.
                # import pdb; pdb.set_trace()
                # dists_to_target[j][i] = np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1]
            if dists_to_target[0][i] <= 0.1 and dists_to_target[1][i] <= 0.1:
                all_success_conditions.append(ioc_conditions[i])
            elif dists_to_target[0][i] <= 0.1:
                only_ioc_conditions.append(ioc_conditions[i])
            elif dists_to_target[1][i] <= 0.1:
                only_demo_conditions.append(ioc_conditions[i])
            else:
                all_failed_conditions.append(ioc_conditions[i])
            dists_diff.append(np.around(dists_to_target[0][i] - dists_to_target[1][i], decimals=2))
        percentages.append(round(float(len(all_success_conditions))/len(ioc_conditions), 2))
        percentages.append(round(float(len(all_failed_conditions))/len(ioc_conditions), 2))
        percentages.append(round(float(len(only_ioc_conditions))/len(ioc_conditions), 2))
        percentages.append(round(float(len(only_demo_conditions))/len(ioc_conditions), 2))

        from matplotlib.patches import Rectangle

        plt.close('all')
        fig = plt.figure()
        ax = Axes3D(fig)
        ioc_conditions_x = [ioc_conditions[i][0] for i in xrange(len(ioc_conditions))]
        ioc_conditions_y = [ioc_conditions[i][1] for i in xrange(len(ioc_conditions))]
        ioc_conditions_z = [ioc_conditions[i][2] for i in xrange(len(ioc_conditions))]
        all_success_x = [all_success_conditions[i][0] for i in xrange(len(all_success_conditions))]
        all_success_y = [all_success_conditions[i][1] for i in xrange(len(all_success_conditions))]
        all_success_z = [all_success_conditions[i][2] for i in xrange(len(all_success_conditions))]
        all_failed_x = [all_failed_conditions[i][0] for i in xrange(len(all_failed_conditions))]
        all_failed_y = [all_failed_conditions[i][1] for i in xrange(len(all_failed_conditions))]
        all_failed_z = [all_failed_conditions[i][2] for i in xrange(len(all_failed_conditions))]
        only_ioc_x = [only_ioc_conditions[i][0] for i in xrange(len(only_ioc_conditions))]
        only_ioc_y = [only_ioc_conditions[i][1] for i in xrange(len(only_ioc_conditions))]
        only_ioc_z = [only_ioc_conditions[i][2] for i in xrange(len(only_ioc_conditions))]
        only_demo_x = [only_demo_conditions[i][0] for i in xrange(len(only_demo_conditions))]
        only_demo_y = [only_demo_conditions[i][1] for i in xrange(len(only_demo_conditions))]
        only_demo_z = [only_demo_conditions[i][2] for i in xrange(len(only_demo_conditions))]
        # subplt = plt.subplot()
        ax.scatter(all_success_x, all_success_y, all_success_z, c='y', marker='o')
        ax.scatter(all_failed_x, all_failed_y, all_failed_z, c='r', marker='x')
        ax.scatter(only_ioc_x, only_ioc_y, only_ioc_z, c='g', marker='^')
        ax.scatter(only_demo_x, only_demo_y, only_demo_z, c='r', marker='v')
        training_positions = zip(*pos_body_offset)
        ax.scatter(training_positions[0], training_positions[1], training_positions[2], s=40, c='b', marker='*')
        # plt.legend(['demo_cond', 'failed_badmm', 'success_ioc', 'failed_ioc'], loc= (1, 1))
        # for i, txt in enumerate(dists_diff):
        #     # subplt.annotate(txt, (ioc_conditions_x[i], ioc_conditions_y[i]))
        #     ax.annotate(txt, (ioc_conditions_x[i], ioc_conditions_y[i], ioc_conditions_z[i]))
        # ax = plt.gca()
        # ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        # ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, fill = False, edgecolor = 'blue')) # reacher
        # box = subplt.get_position()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
        ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
                        'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, 0.05), \
                        shadow=True, ncol=2)
        plt.title("Distribution of samples drawn from demo policy and IOC policy")
        # plt.xlabel('width')
        # plt.ylabel('length')
        plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_added_per.png')
        plt.close('all')

def get_comparison_hyperparams(hyperparam_file):
    """ 
    Compare the performance of two experiments and plot their mean distance to target effector and success rate.
    Args:
        hyperparam_file: the hyperparam file to be changed for two different experiments for comparison.
    """
    hyperparams_1 = imp.load_source('hyperparams', hyperparams_file)
    hyperparams_1.config['common']['nn_demo'] = True
    hyperparams_1.config['algorithm']['init_demo_policy'] = False
    hyperparams_1.config['algorithm']['policy_eval'] = False
    hyperparams_1.config['algorithm']['ioc'] = 'ICML'
    
    hyperparams_1.config['common']['data_files_dir'] = exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/'
    if not os.path.exists(exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/'):
      os.makedirs(exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % itr + '/')
    hyperparams_1.config['algorithm']['policy_opt']['weights_file_prefix'] = hyperparams_1.config['common']['data_files_dir'] + 'policy'
    hyperparams_2 = imp.load_source('hyperparams', hyperparams_file)
    hyperparams_2.config['common']['nn_demo'] = True
    hyperparams_2.config['algorithm']['init_demo_policy'] = False
    hyperparams_2.config['algorithm']['policy_eval'] = False
    hyperparams_2.config['algorithm']['ioc'] = 'ICML'
    exp_dir_classic = exp_dir.replace('on_global', 'on_classic')
    hyperparams_2.config['common']['data_files_dir'] = exp_dir_classic + 'data_files_nn_ICML_3pol_9cond_%d' % itr + '/'
    if not os.path.exists(exp_dir_classic + 'data_files_nn_ICML_3pol_9cond_%d' % itr + '/'):
        os.makedirs(exp_dir_classic + 'data_files_nn_ICML_3pol_9cond_%d' % itr + '/')

    hyperparams_2.config['algorithm']['policy_opt']['weights_file_prefix'] = hyperparams_2.config['common']['data_files_dir'] + 'policy'
    return hyperparams_1, hyperparams_2

def compare_experiments(mean_dists_1_dict, mean_dists_2_dict, success_rates_1_dict, \
                                success_rates_2_dict):
    """ 
    Compare the performance of two experiments and plot their mean distance to target effector and success rate.
    Args:
        mean_dists_1_dict: mean distance dictionary for one of two experiments to be compared.
        mean_dists_2_dict: mean distance dictionary for one of two experiments to be compared.
        success_rates_1_dict: success rates dictionary for one of the two experiments to be compared.
        success_rates_2_dict: success rates dictionary for one of the two experiments to be compared.
    """

    plt.close('all')
    avg_dists_global = [float(sum(mean_dists_1_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_succ_rate_global = [float(sum(success_rates_1_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_dists_classic = [float(sum(mean_dists_2_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_succ_rate_classic = [float(sum(success_rates_2_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    # avg_dists_no_global = [float(sum(mean_dists_no_global_dict[i][j] for i in xrange(3)))/3 for j in xrange(pol_iter)]
    # avg_succ_rate_no_global = [float(sum(success_rates_no_global_dict[i][j] for i in xrange(3)))/3 for j in xrange(pol_iter)]
    plt.plot(range(pol_iter), avg_dists_global, '-x', color='red')
    plt.plot(range(pol_iter), avg_dists_classic, '-x', color='green')
    # plt.plot(range(pol_iter), avg_dists_no_global, '-x', color='green')
    for i in seeds:
        plt.plot(range(pol_iter), mean_dists_1_dict[i], 'ko')
        plt.plot(range(pol_iter), mean_dists_2_dict[i], 'co')
    #   plt.plot(range(pol_iter), mean_dists_no_global_dict[i], 'co')
    #   plt.annotate(np.around(txt, decimals=2), (i, txt))
    plt.legend(['avg MaxEnt', 'avg non-MaxEnt', 'MaxEnt', 'non-MaxEnt'], loc='upper right', ncol=2)
    plt.title("mean distances to target over time with MaxEnt demo and not-MaxEnt demo")
    plt.xlabel("iterations")
    plt.ylabel("mean distances")
    #plt.savefig(exp_dir + 'mean_dists_during_iteration_comparison_maxent.png')
    plt.savefig(exp_dir + 'mean_dists_during_iteration_comparison_maxent.pdf')
    # plt.savefig(exp_dir + 'mean_dists_during_iteration_comparison.png')
    # plt.savefig(exp_dir + 'mean_dists_during_iteration_var.png')
    plt.close()
    plt.plot(range(pol_iter), avg_succ_rate_global, '-x', color='red')
    plt.plot(range(pol_iter), avg_succ_rate_classic, '-x', color='green')
    for i in seeds:
        plt.plot(range(pol_iter), success_rates_1_dict[i], 'ko')
        plt.plot(range(pol_iter), success_rates_2_dict[i], 'co')
        # plt.plot(range(pol_iter), success_rates_no_global_dict[i], 'co')
    plt.legend(['avg MaxEnt', 'avg non-MaxEnt', 'MaxEnt', 'non-MaxEnt'], loc='upper right', ncol=2)
    plt.xlabel("iterations")
    plt.ylabel("success rate")
    plt.title("success rates during iterations with MaxEnt demo and not-MaxEnt dem")
    # plt.title("success rates during iterations with with nn and LG demo")
    #plt.savefig(exp_dir + 'success_rate_during_iteration_comparison_maxent.png')
    plt.savefig(exp_dir + 'success_rate_during_iteration_comparison_maxent.pdf')
    # plt.savefig(exp_dir + 'success_rate_during_iteration_comparison.png')
    # plt.savefig(exp_dir + 'success_rate_during_iteration_var.png')

    plt.close()