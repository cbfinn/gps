from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.agent.mjc.mjc_models import half_cheetah_hop
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 9,
    JOINT_VELOCITIES: 9,
    ACTION: 6,
}

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'

CONDITIONS = 1
TARGET_X = 5.0
TARGET_Z = 5.0

np.random.seed(47)
x0 = []
for _ in range(CONDITIONS):
    x0.append(np.concatenate((0.2*np.random.rand(9)-0.1, 0.1*np.random.randn(9))))

# x0 = [np.zeros(18)]

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
    #'train_conditions': range(CONDITIONS),
    #'test_conditions': range(CONDITIONS),
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': [
        half_cheetah_hop(wall_height=0.2, wall_pos=1.8, gravity=1.0)
    ],
    'x0': x0[:CONDITIONS],
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    #'train_conditions': common['train_conditions'],
    #'test_conditions': common['test_conditions'],
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., -12., 7., 0., 0., 0.]),
    'record_reward': False,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    #'train_conditions': common['train_conditions'],
    #'test_conditions': common['test_conditions'],
    'iterations': 40,
    'kl_step': 1.0,
    'min_step_mult': 0.1,
    'max_step_mult': 10.0,
    'max_ent_traj': 1.0,
    #'policy_sample_mode': 'replace',
    #'num_clusters': 0,
    #'cluster_method': 'kmeans',
    #'sample_on_policy': True,
    #'max_ent_mult': 1E-2,
    #'max_ent_decay': 0.3,

    'compute_distances': {
        'type': 'min',
        'targets': [np.array([TARGET_X]) for i in xrange(common['conditions'])],
        'state_idx': range(0, 1),
    }
}

algorithm['policy_opt'] = {
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1e-2,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost_1 = {
    'type': CostAction,
    'wu': np.array([1.0 ,1.0, 1.0, 1.0, 1.0, 1.0])*1e-2,
}

state_cost = {
    'type': CostState,
    'l1': 0.0,
    'l2': 10.0,
    'alpha': 1e-5,
    'evalnorm': evall1l2term,
    'data_types': {
        JOINT_ANGLES: {
            'target_state': np.array([TARGET_X, TARGET_Z]+[0.0]*7),
            'wp': np.array([1.0, 0.1] + [0.0]*7)
        },
        #JOINT_VELOCITIES: {
        #    'target_state': np.array([2.0]+[0.0]*8),
        #    'wp': np.array([1.0] + [0.0]*8)
        #},
    },

}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost_1, state_cost],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
        'max_iters': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}


algorithm['policy_prior'] = {
}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 1,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    #'train_conditions': common['train_conditions'],
    #'test_conditions': common['test_conditions'],
}

common['info'] = generate_experiment_info(config)
