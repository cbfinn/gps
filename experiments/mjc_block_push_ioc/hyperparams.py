from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc import block_push
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, RAMP_CONSTANT

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 6,
    JOINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS: 12,
    END_EFFECTOR_POINT_VELOCITIES: 12,
    ACTION: 4,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}

PR2_GAINS = np.array([ 1.0, 1.0, 1.0, 1.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = os.path.dirname(__file__)+'/'
DEMO_DIR = BASE_DIR + '/../experiments/mjc_block_push_mdgps/'

OBS_INCLUDE =  [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES]

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'demo_exp_dir': DEMO_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_20.pkl',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 8,
    'demo_conditions': 8,
    'LG_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_LG.pkl'),
    'NN_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_NN.pkl'),
    'nn_demo': True,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])


GOAL_POS =  [np.array([0.4, 0.0, -1.15])]*8
OBJECT_OFFSET = np.array([0.4,0,0])
OBJECT_POS = [
    np.array([0.1, 0.0, -0.9]),
    np.array([-0.1, 0.0, -0.9]),
    np.array([0.3, 0.0, -0.9]),
    np.array([-0.3, 0.0, -1.0]),
    np.array([0.2, 0.0, -1.0]),
    np.array([-0.2, 0.0, -1.0]),
    np.array([0.0, 0.0, -1.0]),
    np.array([-0.0, 0.0, -0.9]),
]
OBJECT_POS = [OBJECT_POS[i]+OBJECT_OFFSET for i in range(len(OBJECT_POS))]

agent = {
    'type': AgentMuJoCo,
    #'filename': './mjc_models/3link_gripper_push_2step.xml',
    'models': [block_push(object_pos=OBJECT_POS[i], goal_pos=GOAL_POS[i]) for i in range(common['conditions'])],
    'x0': np.concatenate([np.array([-np.pi/3, (3*np.pi)/4, 0., 0., 0., 0.0]), np.zeros((6,))]),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': np.array([0,0,0]),
    'pos_body_idx': np.array([6,8]),
    'conditions': common['conditions'],
    #'train_conditions': [0, 1,2,3],
    #'test_conditions': [4,5,6,7],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 150,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 8., 0., 0.3, 0., 0.3]),
 }

demo_agent = {
    'type': AgentMuJoCo,
    #'filename': './mjc_models/3link_gripper_push_2step.xml',
    'models': [block_push(object_pos=OBJECT_POS[i], goal_pos=GOAL_POS[i]) for i in range(common['demo_conditions'])],
    'x0': np.concatenate([np.array([-np.pi/3, (3*np.pi)/4, 0., 0., 0., 0.0]), np.zeros((6,))]),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': np.array([0,0,0]),
    'pos_body_idx': np.array([6,8]),
    'conditions': common['demo_conditions'],
    #'train_conditions': [0, 1,2,3],
    #'test_conditions': [4,5,6,7],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 150,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 8., 0., 0.3, 0., 0.3]),

    'filter_demos': {
        'type': 'min',
        'target': GOAL_POS[0]+np.array([0.05,0.05,0.05]),# for i in xrange(common['conditions'])],
        'state_idx': range(6+12,9+12),
        'success_upper_bound': 0.07,
        'max_demos_per_condition': 15,
    }
}


# algorithm = [{
#     'type': AlgorithmBADMM,
#     'conditions': agent[0]['conditions'],
#     'train_conditions': agent[0]['train_conditions'],
#     'test_conditions': agent[0]['test_conditions'],
#     'num_robots': common['num_robots'],
#     'iterations': 25,
#     'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
#     'policy_dual_rate': 0.2,
#     'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
#     'fixed_lg_step': 3,
#     'kl_step': 5.0,
#     'min_step_mult': 0.01,
#     'max_step_mult': 1.0,
#     'sample_decrease_var': 0.05,
#     'sample_increase_var': 0.1,
#     'init_pol_wt': 0.005,
# }]

#"""
algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': agent['conditions'],
    #'train_conditions': agent['train_conditions'],
    #'test_conditions': agent['test_conditions'],
    'iterations': 25,
    'ioc' : 'ICML',
    'num_demos': 200,
    'kl_step': 1.0,
    'min_step_mult': 0.1,
    'max_step_mult': 4.0,
    'max_ent_traj': 1.0,
    'synthetic_cost_samples': 0,

    'compute_distances': {
        'type': 'min',
        'targets': [GOAL_POS[0]+np.array([0.05,0.05,0.05]) for i in xrange(common['conditions'])],
        'state_idx': range(6+12,9+12),
    }
}
#"""

"""
algorithm = {
    'type': AlgorithmBADMM,
    'conditions': agent['conditions'],
    'iterations': 25,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'init_pol_wt': 0.005,
    'ioc' : 'ICML',
    'num_demos': 10,
    'synthetic_cost_samples': 0,
}
"""


algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 50.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}


torque_cost_1 = [{
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
} for i in range(agent['conditions'])]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0,0,0]), np.array([0,0,0]),
                                           np.array([0.05, 0.05, 0.05]) + GOAL_POS[i],
                                           np.array([0,0,0])]),
    'wp': np.array([0, 0, 0, 0, 0, 0, 1, 1, 1,0,0,0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'ramp_option': RAMP_CONSTANT,
} for i in range(agent['conditions'])]


cost_tgt = np.zeros((6,))
cost_wt = np.array([0, 0, 0, 1, 0, 0])
state_cost = [{
    'type': CostState,
    'l1': 0.0,
    'l2': 10.0,
    'alpha': 1e-5,
    'data_types': {
        JOINT_ANGLES: {
            'target_state': cost_tgt,
            'wp': cost_wt,
        },
    },
} for i in range(agent['conditions'])]
# fk_cost_1_gripper = [{
#     'type': CostFK,
#     'target_end_effector': np.concatenate([np.array([0.05, 0.05, 0.05]) + agent[0]['pos_body_offset'][i][1],
#                                            np.array([0,0,0]),
#                                            np.array([0,0,0])]),
#     'wp': np.array([1, 1, 1, 0, 0, 0, 0,0,0]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
#     'ramp_option': RAMP_QUADRATIC
# } for i in agent[0]['train_conditions']]


fk_cost_blocktouch = [{
    'type': CostFKBlock,
    'wp': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in range(agent['conditions'])]

# data_logger = DataLogger()
# data_traj = data_logger.unpickle('/home/abhigupta/gps/experiments/blockpush_free/data_files_good/traj_sample_itr_24_rn_00.pkl')
# fk_cost_2 = [{
#     'type': CostFKDev,
#     'traj': data_traj[i][0]._data[END_EFFECTOR_POINTS],
#     'wp': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
# } for i in common['train_conditions']]

algorithm['gt_cost'] = [{
    'type': CostSum,
    'costs': [fk_cost_1[i], fk_cost_blocktouch[i], state_cost[i]],
    'weights': [2.0, 1.0, 1.0],
} for i in range(agent['conditions'])]

algorithm['cost'] = {
    #'type': CostIOCQuadratic,
    'type': CostIOCTF,
    'wu': np.array([1,1,1,1])*1e-3,
    'dO': np.sum([SENSOR_DIMS[dtype] for dtype in agent['obs_include']]),
    'T': agent['T'],
    'iterations': 1000,
    'demo_batch_size': 10,
    'sample_batch_size': 10,
    'ioc_loss': algorithm['ioc'],
}


algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 10,
        'min_samples_per_cluster': 20,
        'max_samples': 20,
    },
}



algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
    'robot_number':0
}


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}


algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        #'obs_vector_data': agent['obs_include'],
        'n_layers': 3,
        'dim_hidden': 40,
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': example_tf_network,
    'fc_only_iterations': 2000,
    'init_iterations': 1000,
    'iterations': 1000,  # was 100
    'weights_file_prefix': EXP_DIR + 'policy',
}


config = {
    'iterations': 25,
    'num_samples': 7,
    'verbose_trials': 1,
    'save_wts': True,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
}

common['info'] = generate_experiment_info(config)
