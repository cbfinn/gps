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
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC

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

OBS_INCLUDE =  [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES]

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 4,
    #'train_conditions': [0,1,2,3],
    #'test_conditions': [4,5,6,7],
    'num_robots':1,
    # 'policy_opt': {
    #     'type': PolicyOptTf,
    #     'network_model': example_tf_network_multi,
    #     'network_model_feat': invariant_subspace_test,
    #     'run_feats': True,
    #     'load_weights': '/home/abhigupta/gps/subspace_newweights.pkl',
    #     'network_params': [{
    #         'dim_hidden': [10],
    #         'num_filters': [10, 20],
    #         'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    #         'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    #         'obs_image_data':[],
    #         'image_width': IMAGE_WIDTH,
    #         'image_height': IMAGE_HEIGHT,
    #         'image_channels': IMAGE_CHANNELS,
    #         'sensor_dims': SENSOR_DIMS[0],
    #         'batch_size': 25,
    #         # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
    #     }],
    #     'iterations': 4000,
    #     'fc_only_iterations': 5000,
    #     'checkpoint_prefix': EXP_DIR + 'data_files/policy',
    #     # 'restore_all_wts':'/home/abhigupta/gps/allweights_push_4link.npy'
    # }
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])


OBJECT_POS = [np.array([1.1, 0.0, -0.45]),np.array([0.9, 0.0, -0.65]),np.array([1., 0.0, 0.45]),
              np.array([0.9, 0.0, 0.65]),np.array([-0.3, 0.0, 0.6]),np.array([-0.4, 0.0, -0.5]),
              np.array([-0.3, 0.0, 0.6]),np.array([-0.4, 0.0, -0.6])][0:4:1]
GOAL_POS =  [np.array([1.25, 0.0, 0.0]), np.array([1.25, 0.0, 0.0]), np.array([1.25, 0.0, 0.0]),
 np.array([1.25, 0.0, 0.0]),
 np.array([0.5, 0.0, 0.9]),
 np.array([0.5, 0.0, -0.75]),
 np.array([0.6, 0.0, 0.85]),
 np.array([0.45, 0.0, -0.95])][0:4:1]

agent = {
    'type': AgentMuJoCo,
    #'filename': './mjc_models/3link_gripper_push_2step.xml',
    'models': [block_push(object_pos=OBJECT_POS[i], goal_pos=GOAL_POS[i]) for i in range(common['conditions'])],
    'x0': np.concatenate([np.array([np.pi/2, 0.0, 0.0, 0.0, 0., 0.]), np.zeros((6,))]),
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
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 8., 0., 0.3, 0., 0.3]),
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
algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': agent['conditions'],
    #'train_conditions': agent['train_conditions'],
    #'test_conditions': agent['test_conditions'],
    'iterations': 25,
}


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
    'ramp_option': RAMP_QUADRATIC
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

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [fk_cost_1[i], fk_cost_blocktouch[i], state_cost[i]],
    'weights': [2.0, 1.0, 1.0],
} for i in range(agent['conditions'])]


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
}


config = {
    'iterations': 25,
    'num_samples': 7,
    'verbose_trials': 1,
    'save_wts': True,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    #'train_conditions': common['train_conditions'],
    #'test_conditions': common['test_conditions'],
    'inner_iterations': 4,
    'to_log': [],
}

common['info'] = generate_experiment_info(config)