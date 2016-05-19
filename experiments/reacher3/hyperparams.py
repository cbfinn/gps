""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
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
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 3,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

common = {
    'conditions': 8,
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}

# set up grid of positions
xs = np.linspace(-0.1, 0.1, 3)
ys = np.linspace(-0.1, 0.1, 3)
pos_body_offset = [np.array([x,y,0]) for x in xs for y in ys]
pos_body_offset.pop(4) # get rid of (0,0,0) point

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher3.xml',
    'x0': np.zeros(6),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': pos_body_offset,
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
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
    'init_gains': np.ones(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 1e-1*np.ones(SENSOR_DIMS[ACTION]),
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.0, 0.0]),
    'wp': np.array([2, 2, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

# Create second cost function for last step only.
final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,
    'target_end_effector': fk_cost['target_end_effector'],
    'wp': fk_cost['wp'],
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 1e-5,
    'wp_final_multiplier': 10.0,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost, final_cost],
    'weights': [1.0, 1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 40,
    },
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 1,
    'agent': agent,
    'gui_on': True,
}
