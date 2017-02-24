""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_utils import evall1l2term
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info

from gps.utility.general_utils import sample_params

SENSOR_DIMS = {
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 3,
}

GAINS = np.ones(SENSOR_DIMS[ACTION])

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}

# Set up ranges.
x0_upper = np.array([0., 0., 0., 0., 0., 0.])
#x0_upper = np.array([2., 0., 0., 0., 0., 0.])
x0_lower = -x0_upper

pos_body_idx = np.array([1]),
pos_body_upper = np.array([1., 0., 1.])
pos_body_lower = -pos_body_upper

TEST_CONDITIONS = 40
np.random.seed(47)
test_x0 = [sample_params([x0_lower, x0_upper], []) \
        for _ in range(TEST_CONDITIONS)]
test_pos_body = [sample_params([pos_body_lower, pos_body_upper], []) \
        for _ in range(TEST_CONDITIONS)]

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/arm_3link_reach.xml',
    'x0': np.zeros(6),
    'dt': 0.05,
    'substeps': 5,
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 6., 0., 0., 0., 0.]),
}

algorithm = {
    'iterations': 15,
    'policy_sample_mode': 'replace',
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
    'weights': [1.0, 1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 40,
        'min_samples_per_cluster': 25,
#        'max_samples': 40,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'iterations': 5000,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 40,
    'min_samples_per_cluster': 25,
}

config = {
    'iterations': algorithm['iterations'],
    'verbose_trials': 8,
    'verbose_policy_trials': 8,
    'agent': agent,
    'gui_on': True,
    'random_seed': list(range(5)),
}
