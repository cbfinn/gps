from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc import AgentMuJoCo, weighted_reacher
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_action import CostAction
# from gps.algorithm.cost.cost_ioc_nn import CostIOCNN
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
#from gps.algorithm.cost.cost_gym import CostGym
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd, init_demo
# from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 2,
}

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'
DEMO_DIR = BASE_DIR + '/../experiments/reacher_mdgps_weight/'
SUPERVISED_DIR = BASE_DIR + '/../experiments/reacher_mdgps_weight_supervised/'

#CONDITIONS = 1
TRAIN_CONDITIONS = 6

np.random.seed(47)
DEMO_CONDITIONS = 4 #20
TEST_CONDITIONS = 0 # 4
TOTAL_CONDITIONS = TRAIN_CONDITIONS+TEST_CONDITIONS
# TOTAL_CONDITIONS = 13

demo_pos_body_offset = []
for _ in range(DEMO_CONDITIONS):
    # demo_pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))
    demo_pos_body_offset.append(np.array([-0.2, 0.2, 0.0])) # fix the offset for weight-varying

pos_body_offset = []
for _ in range(TOTAL_CONDITIONS):
    pos_body_offset.append(np.array([-0.2, 0.2, 0.0])) # fix the offset for weight-varying
    # pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))

#pos_body_offset.append(np.array([-0.1, 0.2, 0.0]))
#pos_body_offset.append(np.array([0.05, 0.2, 0.0]))
#demo_pos_body_offset.append(np.array([-0.1, 0.2, 0.0]))
density_range = 10**(np.array([4.5, 5, 5.5, 6, 6.5, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75]))

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    # 'data_files_dir': EXP_DIR + 'data_files/',
    'data_files_dir': EXP_DIR + 'data_files_arm/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'demo_exp_dir': DEMO_DIR,
    'supervised_exp_dir': SUPERVISED_DIR,
    # 'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_09.pkl',
    'demo_controller_file': DEMO_DIR + 'data_files_arm/algorithm_itr_09.pkl',
    'nn_demo': True, # Use neural network demonstrations. For experiment only
    # 'LG_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_LG.pkl'),
    # 'NN_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_NN.pkl'),
    'LG_demo_file': os.path.join(EXP_DIR, 'data_files_arm', 'demos_LG.pkl'),
    'NN_demo_file': os.path.join(EXP_DIR, 'data_files_arm', 'demos_NN.pkl'),
    'conditions': TOTAL_CONDITIONS,
    # 'train_conditions': range(TRAIN_CONDITIONS),
    # 'test_conditions': range(TRAIN_CONDITIONS, TOTAL_CONDITIONS),
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    # 'models': [weighted_reacher(finger_density=1e-10),
    #     weighted_reacher(finger_density=1e-9),
    #     weighted_reacher(finger_density=1e8),
    #     weighted_reacher(finger_density=1e9),
    #     ],
    'models': [weighted_reacher(arm_density=1e-5, finger_density=1e-5),
        weighted_reacher(arm_density=1e-4, finger_density=1e-4),
        weighted_reacher(arm_density=1e5, finger_density=1e5),
        weighted_reacher(arm_density=1e6, finger_density=1e6),
        weighted_reacher(arm_density=1e7, finger_density=1e7),
        weighted_reacher(arm_density=1e8, finger_density=1e8),
        ],
    # 'models': [weighted_reacher(arm_density=density_range[i], finger_density=density_range[i]) \
    #             for i in xrange(common['conditions'])],
    'density_range': density_range,
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'pos_body_offset': pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': common['conditions'],
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., 0., 3., 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(TOTAL_CONDITIONS)],
    'render': True,
}

demo_agent = {
    'type': AgentMuJoCo,
    # 'models': [weighted_reacher(finger_density=1e-8),
    #     weighted_reacher(finger_density=1e-7),
    #     weighted_reacher(finger_density=1e7),
    #     weighted_reacher(finger_density=1e8),
    #     ],
    'models': [weighted_reacher(arm_density=1e-5, finger_density=1e-5),
        weighted_reacher(arm_density=1e-4, finger_density=1e-4),
        weighted_reacher(arm_density=1e5, finger_density=1e5),
        weighted_reacher(arm_density=1e6, finger_density=1e6),
        ],
    'exp_name': 'reacher',
    'x0': np.zeros(4)*4,
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': demo_pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': DEMO_CONDITIONS,
    'T': agent['T'],
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., 0., 3., 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(DEMO_CONDITIONS)],
    'success_upper_bound': 0.01,
    'render': True,
}


algorithm = {
    'type': AlgorithmMDGPS,
    'sample_on_policy': True,
    'ioc' : 'ICML',  # IOC STUFF HERE
    'max_ent_traj': 1.0,
    'num_demos': 10,
    'synthetic_cost_samples': 100,
    'global_cost': True,
    'policy_eval': False,
    'bootstrap': False,
    'init_demo_policy': False,
    'demo_var_mult': 1.0,
    'conditions': common['conditions'],  # NON IOC STUFF HERE
    # 'train_conditions': common['train_conditions'],
    # 'test_conditions': common['test_conditions'],
    'iterations': 15,
    'ioc_maxent_iter': 10,
    'kl_step': 1.0,
    'min_step_mult': 0.2,
    'max_step_mult': 2.0,
    'policy_sample_mode': 'replace',
    'agent_pos_body_idx': agent['pos_body_idx'],
    'agent_pos_body_offset': agent['pos_body_offset'],
    'plot_dir': EXP_DIR,
    'agent_x0': agent['x0'],
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])])
        for i in range(TOTAL_CONDITIONS)],
}


#algorithm = {
#    'type': AlgorithmTrajOpt,
#    'ioc' : 'ICML',
#    'max_ent_traj': 1.0,
#    'conditions': common['conditions'],
#    'kl_step': 0.5,
#    'min_step_mult': 0.05,
#    'max_step_mult': 2.0,
#    'demo_cond': demo_agent['conditions'],
#    'num_demos': 2,
#    'demo_var_mult': 1.0,
#    'synthetic_cost_samples': 100,
#    'iterations': 15,
#    'plot_dir': EXP_DIR,
#    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])])
#                            for i in xrange(CONDITIONS)],
#}

PR2_GAINS = np.array([1.0, 1.0])
torque_cost_1 = [{
    'type': CostAction,
    'wu': 1.0 / PR2_GAINS,
} for i in range(common['conditions'])]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])]),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'evalnorm': evall1l2term,
    # 'use_jacobian': False,
} for i in range(common['conditions'])]

algorithm['gt_cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [2.0, 1.0],
}  for i in range(common['conditions'])][0]

# algorithm['cost'] = {  # TODO - make vision cost and emp. est derivatives
#     'type': CostIOCNN,
#     'wu': 200 / PR2_GAINS,
#     'T': agent['T'],
#     'dO': 16,
#     'iterations': 1000,
#     'demo_batch_size': 5,
#     'sample_batch_size': 5,
#     'ioc_loss': algorithm['ioc'],
#     'smooth_reg_weight': 0.1,
#     'mono_reg_weight': 100.0,
#     'learn_wu': False,
# }

algorithm['cost'] = {
    'type': CostIOCTF,
    'wu': 2000.0 / PR2_GAINS,
    # 'wu' : 0.0,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_image_data': [],
        'sensor_dims': SENSOR_DIMS,
    },
    'T': agent['T'],
    'dO': 16,
    'iterations': 1000, # TODO - do we need 5k?
    'demo_batch_size': 5,
    'sample_batch_size': 5,
    'ioc_loss': algorithm['ioc'],
    'approximate_lxx': False,
}

#algorithm['init_traj_distr'] = {
#    'type': init_demo,
#    'init_gains':  1.0 / PR2_GAINS,
#    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#    'init_var': 5.0,
#    'stiffness': 1.0,
##    'stiffness_vel': 0.5,
#    'final_weight': 50.0,
#    'dt': agent['dt'],
#    'T': agent['T'],
#}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  100.0 * np.ones(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 5.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.5,
    'final_weight': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 30,
        'min_samples_per_cluster': 40,
        'max_samples': 10, #len(common['conditions']),
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
    'min_eta': 1e-4,
    'max_eta': 1.0,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_image_data': [],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': example_tf_network,
    #'fc_only_iterations': 5000,
    #'init_iterations': 1000,
    'iterations': 1000,  # was 100
    'weights_file_prefix': common['data_files_dir'] + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 40,
    'min_samples_per_cluster': 40,
}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'random_seed': 1,
}

common['info'] = generate_experiment_info(config)
