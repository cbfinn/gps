""" Hyperparameters for MJC gripper pusher task with trajectory optimization."""
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.cost.cost_utils import RAMP_QUADRATIC
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 6,
    JOINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS: 12,
    END_EFFECTOR_POINT_VELOCITIES: 12,
    ACTION: 4,
}

GP_GAINS = np.array([1.0, 1.0, 1.0, 1.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_gripper_pusher_pilqr/'


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
    'filename': './mjc_models/3link_gripper_pusher.xml',
    'x0': np.concatenate([np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx':np.array([6, 8]),
    'pos_body_offset': [
        [np.array([0.65, 0.0, -0.9]), np.array([1.6, 0.0, -0.25])],
        [np.array([1., 0.0, 0.45]), np.array([0.2, 0.0, 1.0])],
        [np.array([0.9, 0.0, 0.9]), np.array([1.25, 0.0, 0.4])],
        [np.array([0.5, 0.0, 0.9]), np.array([-0.4, 0.0, 0.75])],
    ],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [],
    'camera_pos': np.array([0.0, 5.0, 0.0, 0.3, 0.0, 0.3]),
    'smooth_noise': False,
}

algorithm = {
    'type': AlgorithmTrajOptPILQR,
    'conditions': common['conditions'],
    'iterations': 20,
    'step_rule': 'res_percent',
    'step_rule_res_ratio_dec': 0.2,
    'step_rule_res_ratio_inc': 0.05,
    'kl_step': np.linspace(0.6, 0.2, 100),
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / GP_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 10.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / GP_GAINS,
}

fk_cost = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.zeros(6),
                                           np.array([0.05, 0.05, 0.05]) + agent['pos_body_offset'][i][1],
                                           np.zeros(3)]),
    'wp': np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'ramp_option': RAMP_QUADRATIC,
} for i in xrange(common['conditions'])]

cost_tgt = np.zeros(6)
cost_wt = np.array([0, 0, 0, 1, 0, 0])
state_cost = {
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
}

fk_cost_blocktouch = {
    'type': CostFKBlock,
    'wp': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [fk_cost[i], fk_cost_blocktouch, state_cost],
    'weights': [4.0, 1.0, 1.0],
} for i in xrange(common['conditions'])]


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
    'kl_threshold': 1.0,
    'covariance_damping': 10.0,
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
    'random_seed': 0,
}

common['info'] = generate_experiment_info(config)
