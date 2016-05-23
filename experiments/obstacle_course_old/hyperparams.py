""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    ACTION: 2,
}

common = {
    'conditions': 5,
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}

ys = np.linspace(-1.5, 1.5, 5)
x0 = [np.array([0, y, 0, 0]) for y in ys]

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/obstacle_course.xml',
    'x0': x0,
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
}

algorithm = {
    'conditions': common['conditions'],
    'iterations': 15,
    'kl_step': 2.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'policy_sample_mode': 'replace',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_var': 1.0,
    'stiffness': 10.0,
    'stiffness_vel': 10.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['cost'] = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            'target_state': np.array([3, 0]),
            'wp_final_multiplier': 10,
            'l1' : 0.1,
            'l2' : 10,
            'alpha' : 1e-5,
        },
    },
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 5,
        'min_samples_per_cluster': 40,
        'max_samples': 40,
    },
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 5,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': True,
}
