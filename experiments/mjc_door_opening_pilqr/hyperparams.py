""" Hyperparameters for MJC door opening trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_mdgps_pilqr import AlgorithmMDGPSPILQR
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_lin_wp import CostLinWP
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_QUADRATIC
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import tf_network


SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 12,
    END_EFFECTOR_POINT_VELOCITIES: 12,
    ACTION: 6,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.152, 0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_door_opening_pilqr/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 4,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d_door.xml',
    'x0': np.concatenate([np.array([0.0, 0.5, 0.0, -0.6, -0.2, 0.0, 0.0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([0.0, 0.0, 0])], [np.array([0.0, 0.1, 0])],
                        [np.array([0.1, 0.0, 0])], [np.array([0.1, 0.1, 0])],],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                      END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([4.0, 0.0, 1.5, 0., 0., 0.25]),
    'smooth_noise': False,
}

algorithm = {
    'type': AlgorithmMDGPSPILQR,
    'step_rule': 'const',
    'conditions': common['conditions'],
    'iterations': 20,
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 10.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
}

state_cost = {
    'type': CostState,
    'data_types': {
        JOINT_ANGLES: {
            'wp': np.array([0, 0, 0, 0, 0, 0, 1]),
            'target_state': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        },
    },
    'ramp_option': RAMP_QUADRATIC,
}

_A = np.zeros((2, 44, 44))
for i in xrange(3):
    _A[0, i, 17+i] = 1.0
    _A[0, i, 20+i] = -1.0
for i in xrange(6):
    _A[1, i, 14+i] = 1.0
    _A[1, i, 20+i] = -1.0

linwp_cost = {
    'type': CostLinWP,
    'waypoint_time': np.array([0.25, 1.0]),
    'A': _A,
    'b': np.array([[0.0] * 44] * 2),
    'l1': 0.1,
    'l2': 1.0,
    'alpha': 1e-5,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost, linwp_cost],
    'weights': [0.1, 5.0, 1.0],
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
    'type': TrajOptPILQR,
    'covariance_damping': 10.0,
    'kl_threshold': 0.5,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
        'n_layers': 2,
        'dim_hidden': [100, 100],
    },
    'network_model': tf_network,
    'iterations': 4000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'random_seed': 1,
}

common['info'] = generate_experiment_info(config)
