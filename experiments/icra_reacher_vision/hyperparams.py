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
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network, multi_modal_network_fp

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        RGB_IMAGE, RGB_IMAGE_SIZE, ACTION, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET
from gps.gui.config import generate_experiment_info

from gps.utility.general_utils import sample_params

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

SENSOR_DIMS = {
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
#    END_EFFECTOR_POINTS_NO_TARGET: 3,
#    END_EFFECTOR_POINT_VELOCITIES_NO_TARGET: 3,
    ACTION: 3,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}

GAINS = np.ones(SENSOR_DIMS[ACTION])

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}

# Set up ranges.
pos_body_idx = np.array([1]),
pos_body_upper = np.array([1., 0., 1.])
pos_body_lower = -pos_body_upper

TEST_CONDITIONS = 40
np.random.seed(47)
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
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
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
    'type': PolicyOptTf,
    'network_params': {
        'num_filters': [15, 15, 15],
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': multi_modal_network_fp,
    'fc_only_iterations': 5000,
    'init_iterations': 1000,
    'iterations': 1000,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 40,
    'min_samples_per_cluster': 25,
}

config = {
    'iterations': algorithm['iterations'],
    'verbose_trials': np.inf,
    'verbose_policy_trials': np.inf,
    'agent': agent,
    'gui_on': True,
    'random_seed': list(range(5)),
}
