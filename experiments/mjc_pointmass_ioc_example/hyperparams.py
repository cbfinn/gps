""" Hyperparameters for MJC 2D navigation policy optimization using ioc. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc import obstacle_pointmass
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_ioc_quad import CostIOCQuadratic
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_state import CostState
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
EXP_DIR = os.path.dirname(__file__)+'/'
DEMO_DIR = BASE_DIR + '/../experiments/mjc_pointmass_example/'
target_pos = np.array([1.3, 0.0, 0.])


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'demo_exp_dir': DEMO_DIR,
    'data_files_dir': os.path.join(EXP_DIR, 'data_files'),
    'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_09.pkl',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'demo_conditions': 4,
    'conditions': 1,
    'LG_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_LG.pkl'),
    'nn_demo': False,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': obstacle_pointmass(target_pos, wall_center=0.0, hole_height=0.3),
    'filename': '',
    #'x0': [np.array([-1., 1., 0., 0.]), np.array([1., 1., 0., 0.]),
    #       np.array([1., -1., 0., 0.]), np.array([-1., -1., 0., 0.])],
    'x0': [np.array([-1., 1., 0., 0.])],
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 100,
    'point_linear': True,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise': False,
    'camera_pos': np.array([1., 0., 8., 0., 0., 0.]),
}

demo_agent = {
    'type': AgentMuJoCo,
    'models': obstacle_pointmass(target_pos, wall_center=0.0, hole_height=0.3),
    'filename': '',
    'x0': np.array([-1., 0., 0., 0.]),
    # 'x0': [np.array([-1., 1., 0., 0.])],
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['demo_conditions'],
    'T': agent['T'],
    'point_linear': True,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise': False,
    'camera_pos': np.array([1., 0., 8., 0., 0., 0.]),
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'ioc' : 'ICML',
    'demo_distr_empest': True,
    'max_ent_traj': 1.0,
    'conditions': common['conditions'],
    'iterations': 15,
    'kl_step': 1.0,
    'min_step_mult': 0.01,
    #'min_step_mult': 1.0,
    'max_step_mult': 10.0,
    'demo_cond': 1,
    'num_demos': 5,
    'demo_var_mult': 1.0,
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

algorithm['cost'] = {
    #'type': CostIOCQuadratic,
    'type': CostIOCTF,
    'wu': np.array([1e-5, 1e-5]),
    'dO': 10,
    'T': agent['T'],
    'iterations': 5000,
    'demo_batch_size': 15,
    'sample_batch_size': 15,
    'ioc_loss': algorithm['ioc'],
}

algorithm['gt_cost'] = {
    'type': CostState,
    'l2': 10,
    'l1': 0,
    'alpha': 1e-4,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            'target_state': target_pos[0:2],
        },
    },
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 40,
        'min_samples_per_cluster': 20,
        'max_samples': 20,
    }
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_prior'] = {
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 15,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
