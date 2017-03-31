""" Hyperparameters for MJC 2D navigation with discontinous target region. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_binary_region import CostBinaryRegion
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_disc_cost_pilqr_example/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pointmass_disc_cost.xml',
    'x0': [np.array([1., 0., 0., 0.])],
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                      END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([5., 6., 6.5, 0., 0., 0.]),
    'smooth_noise_var': 3.,
}

algorithm = {
    'type': AlgorithmTrajOptPILQR,
    'conditions': common['conditions'],
    'iterations': 20,
    'step_rule': 'res_percent',
    'step_rule_res_ratio_inc': 0.05,
    'step_rule_res_ratio_dec': 0.2,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_var': 20.,
    'dt': agent['dt'],
    'T': agent['T'],
}

state_cost = {
    'type': CostState,
    'data_types' : {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
            'target_state': np.array([3., 0, 0]),
        },
    },
}

binary_region_cost = {
    'type': CostBinaryRegion,
    'data_types' : {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
            'target_state': np.array([2.5, 0.5, 0]),
            'max_distance': 0.3,
            'outside_cost': 0.,
            'inside_cost': -3.,
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost, binary_region_cost],
    'weights': [1., 10.],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 2,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    }
}

algorithm['traj_opt'] = {
    'type': TrajOptPILQR,
    'kl_threshold': 1.0,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'verbose_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
