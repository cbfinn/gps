""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_ioc_nn import CostIOCNN
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY, evall1l2term
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_demo, init_lqr
from gps.utility.demo_utils import generate_pos_body_offset, generate_x0, generate_pos_idx
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_mdgps_ioc_example/'
# DEMO_DIR = BASE_DIR + '/../experiments/mjc_mdgps_multiple_example/on_classic/'
DEMO_DIR = BASE_DIR + '/../experiments/mjc_mdgps_example/on_classic/'
# DEMO_DIR = BASE_DIR + '/../experiments/mjc_badmm_example_'
LG_DIR = BASE_DIR + '/../experiments/mjc_peg_example/'
DEMO_CONDITIONS = 40

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    # 'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_06.pkl',
    'demo_exp_dir': DEMO_DIR,
    # 'demo_controller_file': [DEMO_DIR + '%d/' % i + 'data_files/algorithm_itr_11.pkl' for i in xrange(4)],
    # 'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_11.pkl',
    'demo_controller_file': DEMO_DIR + 'data_files_maxent_9cond_0/algorithm_itr_11.pkl',
    'LG_controller_file': LG_DIR + 'data_files/algorithm_itr_09.pkl',
    'conditions': 9,
    # 'dense': True # For dense/sparse demos experiment only
    'nn_demo': True, # Use neural network demonstrations. For experiment only
}

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'x0': np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'randomly_sample_bodypos': False,
    'randomly_sample_x0': False,
    'sampling_range_bodypos': [np.array([-0.1,-0.1, 0.0]), np.array([0.1, 0.1, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[[None, None, None, None]],
    'pos_body_idx': np.array([1]),
    # 'pos_body_offset': [np.array([-0.1, -0.1, 0]), np.array([-0.1, 0.1, 0]),
    #                     np.array([0.1, 0.1, 0]), np.array([0.1, -0.1, 0])],
    'pos_body_offset': [np.array([-0.1, -0.1, 0]), np.array([-0.1, 0, 0]), np.array([-0.1, 0.1, 0]),
                        np.array([0, -0.1, 0]), np.array([0, 0, 0]), np.array([0, 0.1, 0]),
                        np.array([0.1, 0.1, 0]), np.array([0.1, 0, 0]), np.array([0.1, -0.1, 0])],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
}

demo_agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'x0': generate_x0(np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                      np.zeros(7)]), DEMO_CONDITIONS),
    'dt': 0.05,
    'substeps': 5,
    'conditions': DEMO_CONDITIONS,
    'pos_body_idx': generate_pos_idx(DEMO_CONDITIONS),
    # 'pos_body_offset': [np.array([0, 0.2, 0]), np.array([0, 0.1, 0]),
    #                     np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],
    'pos_body_offset': generate_pos_body_offset(DEMO_CONDITIONS),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'peg_height': 0.1,
    'success_upper_bound': 0.01,
    'failure_lower_bound': 0.15,
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'learning_from_prior': True,
    'ioc' : 'ICML',
    'iterations': 20,
    'kl_step': 0.5,
    'min_step_mult': 0.05,
    'max_step_mult': 2.0,
    # 'min_step_mult': 1.0,
    # 'max_step_mult': 1.0,
    'policy_sample_mode': 'replace',
    'max_ent_traj': 1.0,
    'demo_distr_empest': True,
    'demo_var_mult': 1.0,
    'init_var_mult': 1.0,
    # 'demo_cond': 15,
    # 'num_demos': 3,
    'num_demos': 1,
    'init_samples': 5,
    'synthetic_cost_samples': 100,
    # 'synthetic_cost_samples': 0, # Remember to change back to 100 when done with the 50 samples exp
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'global_cost': True,
    'sample_on_policy': True,
    'init_demo_policy': False,
    'policy_eval': False,
    'bootstrap': False,
    'success_upper_bound': 0.10,
}

# Use for nn demos
algorithm['init_traj_distr'] = {
    # 'type': init_lqr,
    'type': init_demo,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 5.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

# Use for LG demos.
# algorithm['init_traj_distr'] = {
#     'type': init_demo,
#     'init_gains':  0.2 / PR2_GAINS,
#     'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#     # 'init_var': 1.0,
#     'init_var': 1.5,
#     'stiffness': 0.5,
#     'stiffness_vel': 0.5,
#     'final_weight': 10.0,
#     'dt': agent['dt'],
#     'T': agent['T'],
# }

torque_cost = {
    'type': CostAction,
    'wu': 1e-3 / PR2_GAINS,
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.array([2, 2, 1, 2, 2, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'evalnorm': evall1l2term,
}

# Create second cost function for last step only.
final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,
    'target_end_effector': fk_cost['target_end_effector'],
    'wp': fk_cost['wp'],
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 1e-5,
    'wp_final_multiplier': 10.0,
    'evalnorm': evall1l2term,
}

algorithm['gt_cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost, final_cost],
    'weights': [100.0, 100.0, 100.0],
}

algorithm['cost'] = {
    'type': CostIOCNN,
    'wu': 100*1e-3 / PR2_GAINS,
    'T': 100,
    'dO': 26,
    'learn_wu': False,
    'iterations': 5000,
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'iterations': 4000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 10,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
}
