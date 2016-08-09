""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_ioc_nn import CostIOCNN
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_demo, init_lqr
from gps.utility.demo_utils import generate_pos_body_offset, generate_x0, generate_pos_idx
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_peg_ioc_example/'
DEMO_DIR = BASE_DIR + '/../experiments/mjc_peg_example/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_09.pkl',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    'demo_conditions': 15, # Does this do anything? (demo cond defined below)
    # 'demo_conditions': 25,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'x0': np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    #'pos_body_offset': [np.array([0, 0.2, 0]), np.array([0, 0.1, 0]),
    #                    np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],
    'pos_body_offset': [np.array([0.1, -0.1, 0])],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
}

demo_agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'x0': generate_x0(np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                      np.zeros(7)]), 25),
    'dt': 0.05,
    'substeps': 5,
    'conditions': 25,
    'pos_body_idx': generate_pos_idx(25),
    # 'pos_body_offset': [np.array([0, 0.2, 0]), np.array([0, 0.1, 0]),
    #                     np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],
    'pos_body_offset': generate_pos_body_offset(25),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'ioc' : 'MPF',
    'conditions': common['conditions'],
    'learning_from_prior': False,
    'kl_step': 0.5,
    #'max_step_mult': 1.0,
    'max_step_mult': 2.0,
    'min_step_mult': 0.01,
    #'min_step_mult': 1.0,
    'max_ent_traj': 1.0,
    'demo_distr_empest': True,
    'demo_cond': 15,
    'num_demos': 3,  # per condition
    'iterations': 15,
    'synthetic_cost_samples': 100,
}

# random init
#algorithm['init_traj_distr'] = {
#    'type': init_lqr,
#    'init_gains':  1.0 / PR2_GAINS,
#   'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#    'init_var': 5.0,
#    'stiffness': 1.0,
#    'stiffness_vel': 0.5,
#    'final_weight': 50.0,
#    'dt': agent['dt'],
#    'T': agent['T'],
#}

# demo init
algorithm['init_traj_distr'] = {
    'type': init_demo,
    'init_gains':  0.2 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.5,
    'stiffness': 0.5,
    'stiffness_vel': 0.5,
    'final_weight': 10.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.array([1, 1, 1, 1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

algorithm['gt_cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost],
    'weights': [1.0, 1.0],
}

algorithm['cost'] = {
    'type': CostIOCNN,
    'wu': 5e-5 / PR2_GAINS,
    'T': 100,
    'dO': 26,
    'iterations': 5000,
    'demo_batch_size': 5,
    'sample_batch_size': 5,
    'ioc_loss': algorithm['ioc'],
}

# algorithm['gt_cost'] = {
#     'type': CostFK,
#     'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
#     'wp': np.array([1, 1, 1, 1, 1, 1]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
# }

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
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 5,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
