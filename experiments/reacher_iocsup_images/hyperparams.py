from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
# Use CostIOCSupervisedTF instead?
from gps.algorithm.cost.cost_ioc_vision_supervised_tf import CostIOCVisionSupervisedTF
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd, init_demo
from gps.utility.demo_utils import generate_pos_body_offset, generate_x0, generate_pos_idx
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network, multi_modal_network_fp

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, IMAGE_FEAT
from gps.gui.config import generate_experiment_info

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS_NO_TARGET: 3,
    END_EFFECTOR_POINT_VELOCITIES_NO_TARGET: 3,
    ACTION: 2,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
    IMAGE_FEAT: 30,  # affected by num_filters
}

PR2_GAINS = np.array([1.0, 1.0])

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'
DEMO_DIR = BASE_DIR + '/../experiments/reacher_images/'

DEMO_CONDITIONS = 15
demo_pos_body_offset = [np.array([0.1,-0.1,0.0]),
                   np.array([0.1,0.1,0.0]),
                   np.array([0.1,0.3,0.0]),
                   np.array([0.0,0.2,0.0]),
                   np.array([0.0,0.0,0.0]),  # training cond up to here
                   np.array([0.1,0.0,0.0]),
                   np.array([0.1,0.2,0.0]),
                   np.array([0.05,-0.05,0.0]),
                   np.array([0.05,0.05,0.0]),
                   np.array([0.05,0.15,0.0]),
                   np.array([0.05,0.25,0.0]),
                   np.array([-0.1,0.1,0.0]),
                   np.array([-0.05,0.15,0.0]),
                   np.array([-0.05,0.05,0.0]),
                   np.array([-0.05,0.1,0.0]), # these are the 15 demo cond in which it works
                   ]


CONDITIONS = 13
pos_body_offset = [np.array([-0.1, 0.2, 0.0]),
                   np.array([-0.1,0.0, 0.0]),
                   np.array([0.0,0.3, 0.0]),
                   np.array([0.0,-0.1, 0.0]),
                   np.array([0.1,0.1,0.0]), # start training cond here.
                   np.array([0.0,0.0,0.0]),
                   np.array([0.0,0.2,0.0]), # 7 training conditions that get 6/7 success, compared to 5/7 demo policy success
                   np.array([0.0,0.2, 0.0]),
                   np.array([0.0,0.1, 0.0]),
                   np.array([0.0,0.0, 0.0]),
                   np.array([-0.1,0.1, 0.0]),
                   np.array([0.1,0.0,0.0]),
                   np.array([0.1,0.2,0.0]), # 13 conditions that gets 12/13 success, compared to 9/13 demo policy success
                   ]

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'demo_exp_dir': DEMO_DIR,
    'demo_controller_file': [DEMO_DIR + 'data_files/algorithm_itr_09.pkl'],  # NOTE - this used to be 09.
    'demo_conditions': DEMO_CONDITIONS,
    'conditions': CONDITIONS,
    'nn_demo': True,
    'NN_demo_file': EXP_DIR + 'data_files/pol_demos.pkl',
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher_img.xml',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': common['conditions'],
    'T': 50,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
                      END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, IMAGE_FEAT],  # TODO - may want to include fp velocities.
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01]) +pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(CONDITIONS)],
    'feature_encoder': common['demo_controller_file'][0], # initialize conv layers of policy
}

demo_agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher_img.xml',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': demo_pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': common['demo_conditions'],
    'T': agent['T'],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
                      END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, IMAGE_FEAT],  # TODO - may want to include fp velocities.
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+demo_pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(DEMO_CONDITIONS)],
    'feature_encoder': common['demo_controller_file'][0], # initialize conv layers of policy
}


algorithm = {
    'type': AlgorithmMDGPS,
    'ioc' : 'ICML',
    'max_ent_traj': 0.001, #1.0,
    'conditions': common['conditions'],
    'iterations': 25,
    'kl_step': 0.5,
    'min_step_mult': 0.1,
    'max_step_mult': 2.0,
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
    'demo_cond': demo_agent['conditions'],
    'num_demos': 2,
    'demo_var_mult': 1.0,
    'synthetic_cost_samples': 0, #100,  # this is broken when the obs is different from the state.
    'init_cost_params': common['demo_controller_file'][0], # initialize conv layers of policy.
    'plot_dir': EXP_DIR,
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])])
                            for i in xrange(CONDITIONS)],
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'num_filters': [15, 15, 15],
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': multi_modal_network_fp,
    'fc_only_iterations': 5000,
    'init_iterations': 0, #1000,
    'iterations': 0, #1000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  100.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost_1 = [{
    'type': CostAction,
    'wu': 2.0 / PR2_GAINS,
} for i in range(common['conditions'])]


fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])]),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 0,
    'evalnorm': evall1l2term,
} for i in range(common['conditions'])]

algorithm['gt_cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [1.0, 1.0],
}  for i in range(common['conditions'])]

demo_torque_cost_1 = [{
    'type': CostAction,
    'wu': 2.0 / PR2_GAINS,
} for i in range(common['demo_conditions'])]

demo_fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([.1, -.1, .01])+ demo_agent['pos_body_offset'][i], np.array([0., 0., 0.])]),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 0,
    'evalnorm': evall1l2term,
} for i in range(common['demo_conditions'])]

demo_gt_cost = [{
    'type': CostSum,
    'costs': [demo_torque_cost_1[i], demo_fk_cost_1[i]],
    'weights': [1.0, 1.0],
}  for i in range(common['demo_conditions'])]

import glob
algorithm['cost'] = {
    'type': CostIOCVisionSupervisedTF,
    'wu': 2.0 / PR2_GAINS,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
    },
    'T': agent['T'],
    'demo_batch_size': 5,  # are we going to run out of memory? # also should we init from policy feat?
    'sample_batch_size': 5,
    'ioc_loss': algorithm['ioc'],
    'init_fc_iterations': 6000,
    'init_iterations': 1000, #1000,
    'traj_samples': glob.glob('/home/cfinn/code/gps/experiments/reacher_images/data_files/traj_sample*.pkl'),
    'demo_file': common['NN_demo_file'],
    'gt_cost' : demo_gt_cost,
    'agent': demo_agent,
}

NUM_SAMPLES = 5
algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': NUM_SAMPLES,  # was 10 before. we want it to be the same as the number of samples.
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': NUM_SAMPLES,
    'verbose_trials': 0,  # only show first image.
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'random_seed': 1,
}

common['info'] = generate_experiment_info(config)
