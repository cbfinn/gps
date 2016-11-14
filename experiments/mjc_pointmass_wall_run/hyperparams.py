""" Hyperparameters for MJC 2D navigation policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc import AgentMuJoCo, obstacle_pointmass
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_pointmass_wall_run/'
target_pos = np.array([1.3, 0.0, 0.])
# target_pos = np.array([0.43, 0.75, 0])
wall_1_center = np.array([0.5, -0.8, 0.])
wall_2_center = np.array([0.5, 0.8, 0.])
wall_height = 2.8


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 3,
    # 'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    # TODO: pass in wall and target position here.
    # 'models': [obstacle_pointmass(target_pos, wall_center=0.0, hole_height=0.3),
    #            obstacle_pointmass(target_pos, wall_center=0.3, hole_height=0.3),
    #            obstacle_pointmass(target_pos, wall_center=-0.3, hole_height=0.3),
    #            obstacle_pointmass(target_pos, wall_center=0.5, hole_height=0.3),
    #            ],
    'models': [
        obstacle_pointmass(target_pos, wall_center=0.1, hole_height=0.2, delete_top=True, control_limit=20),
        obstacle_pointmass(target_pos, wall_center=0.2, hole_height=0.2, delete_top=True, control_limit=20),
        obstacle_pointmass(target_pos, wall_center=0.3, hole_height=0.2, delete_top=True, control_limit=20),
    ],
    #'x0': [np.array([-0.75, 0., 0., 0.]), np.array([-0.75, -0.25, 0., 0.]),
    #      np.array([-0.75, -0.5, 0., 0.]), np.array([-0.75, -0.75, 0., 0.])],
    'x0': np.array([-1., 0., 0., 0.]),
    # 'x0': [np.array([-1., 0., 0., 0.])]*4,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 200,
    'point_linear': True,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise': False,
    #'camera_pos': np.array([1., 0., 8., 0., 0., 0.]),
    'camera_pos': np.array([0., 0., 6., 0., 1., 0.]),
}

MODELS =  [
    obstacle_pointmass(target_pos, wall_center=0.0, hole_height=0.2, delete_top=True, control_limit=20),
    obstacle_pointmass(target_pos, wall_center=0.1, hole_height=0.2, delete_top=True, control_limit=20),
    obstacle_pointmass(target_pos, wall_center=0.2, hole_height=0.2, delete_top=True, control_limit=20),
    obstacle_pointmass(target_pos, wall_center=0.3, hole_height=0.2, delete_top=True, control_limit=20),
    obstacle_pointmass(target_pos, wall_center=0.4, hole_height=0.2, delete_top=True, control_limit=20),
    obstacle_pointmass(target_pos, wall_center=0.5, hole_height=0.2, delete_top=True, control_limit=20),
    obstacle_pointmass(target_pos, wall_center=0.6, hole_height=0.2, delete_top=True, control_limit=20),
]


demo_agent = {
    'algorithm_file': EXP_DIR + 'data_files/sup_1.pkl',
    'type': AgentMuJoCo,
    'models': MODELS,
    'x0': np.array([-1., 0., 0., 0.]),
    'dt': 0.05,
    'substeps': 1,
    'conditions': len(MODELS),
    'record_reward': False,
    #'screenshot_pause': [5],
    'dont_save': True,
    'eval_only': True,
    'num_demos': 20,
    'verbose_trials': 0,
    'T': agent['T'],
    'point_linear': True,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise': False,
    'camera_pos': np.array([0., 0., 6., 0., 1., 0.]),
    'target_end_effector': target_pos,
    'filter_demos': {
        'type': 'min',
        'state_idx': range(4, 7),
        'target': target_pos,
        'success_upper_bound': 0.5,
        'max_demos_per_condition': 10,
    },
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 25,
    'kl_step': 2.0,
    'min_step_mult': 0.01,
    'max_step_mult': 4.0,
    'max_ent_traj': 10.0,
    'target_end_effector': target_pos,
    'ioc': 'ICML',
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 1.0,
    'pos_gains': 1.0,
    'vel_gains_mult': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

state_cost = {
    'type': CostState,
    'l2': 10,
    'l1': 0,
    'alpha': 1e-4,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            'target_state': target_pos[0:2],
        },
        # JOINT_VELOCITIES: {
        #     'wp': 0*np.ones(SENSOR_DIMS[ACTION]),
        # },
    },
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1, 1])*1e-3,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost, action_cost],
    'weights': [1., 1.], # used 10,1 for T=3
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 5,
        'min_samples_per_cluster': 20,
        'max_samples': 20,
    }
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

# algorithm['policy_opt'] = {
#     'type': PolicyOptCaffe,
#     'weights_file_prefix': EXP_DIR + 'policy',
#     'iterations': 10000,
#     'network_arch_params': {
#         'n_layers': 2,
#         'dim_hidden': [20],
#     },
# }

algorithm['policy_prior'] = {
    'type': PolicyPrior,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 0,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'arecord_gif': {
        'gif_dir': os.path.join(common['data_files_dir'], 'gifs'),
        'gifs_per_condition': 1,
        'save_traj_samples': False,
        'fps': 40,
    }
}
seed = 4

common['info'] = generate_experiment_info(config)
