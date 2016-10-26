""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import evall1l2term
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.gui.config import generate_experiment_info
from gps.utility.general_utils import sample_params
from gps.proto.gps_pb2 import *

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 2,
}

GAINS = np.ones(SENSOR_DIMS[ACTION])

TRAIN_CONDITIONS = 40
TEST_CONDITIONS = 10
TOTAL_CONDITIONS = TEST_CONDITIONS + TRAIN_CONDITIONS

# Set up ranges.
pos_body_lower = np.array([-0.3, -0.1, 0])
pos_body_upper = np.array([0.1, 0.3, 0])

# Create dummy train positions (will be overwritten randomly).
pos_body_offset = [np.zeros(3) for _ in range(TRAIN_CONDITIONS)]

# Add random test conditions.
np.random.seed(13)
pos_body_offset += [sample_params([pos_body_lower, pos_body_upper], []) \
        for _ in range(TEST_CONDITIONS)]

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_mdgps_example/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': TOTAL_CONDITIONS,
    'train_conditions': range(TRAIN_CONDITIONS), 
    'test_conditions': range(TRAIN_CONDITIONS, TOTAL_CONDITIONS), 
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher.xml',
    'conditions': common['conditions'],
    'x0': np.zeros(4),
    'T': 50,
    'dt': 0.05,
    'substeps': 2,
    'pos_body_offset': pos_body_offset,
    'pos_body_idx': np.array([4]),
    'randomly_sample_bodypos': True,
    'sampling_range_bodypos': [pos_body_lower, pos_body_upper],
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'sample_on_policy': True,
    'iterations': 15,
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
    'cluster_method': 'traj_em',
    'num_clusters': 8,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': 100.0*GAINS,
    'init_acc': np.zeros_like(GAINS),
    'init_var': 10.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 1e-3 / GAINS,
}

# Diff between average of end_effectors and block.
touch_cost = {
    'type': CostState,
    'data_type': END_EFFECTOR_POINTS,
    'A' : np.c_[np.eye(3), -np.eye(3)],
    'l1': 1.0,
    'evalnorm': evall1l2term,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, touch_cost],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 50,
        'min_samples_per_cluster': 40,
        'max_samples': TRAIN_CONDITIONS,
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
    'max_clusters': 50,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'gui_on': True,
    'iterations': algorithm['iterations'],
    'num_samples': 1,
    'verbose_trials': 0,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
