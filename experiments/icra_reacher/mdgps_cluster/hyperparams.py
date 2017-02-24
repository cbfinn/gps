""" Hyperparameters for MJC peg insertion policy optimization. """
import imp
import os.path
import numpy as np
from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
default = imp.load_source('default_hyperparams', BASE_DIR+'/hyperparams.py')

EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'

TEST_CONDITIONS = default.TEST_CONDITIONS
TRAIN_CONDITIONS = 40
TOTAL_CONDITIONS = TEST_CONDITIONS + TRAIN_CONDITIONS

# Create dummy train positions (will be overwritten randomly).
x0 = [np.zeros(6) for _ in range(TRAIN_CONDITIONS)]
pos_body_offset = [np.zeros(3) for _ in range(TRAIN_CONDITIONS)]

# Add test positions.
x0 += default.test_x0
pos_body_offset += default.test_pos_body

# Update the default config.
common = default.common.copy()
common.update({
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': TOTAL_CONDITIONS,
    'train_conditions': range(TRAIN_CONDITIONS), 
    'test_conditions': range(TRAIN_CONDITIONS, TOTAL_CONDITIONS), 
})

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = default.agent.copy()
agent.update({
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'x0': x0,
    'randomly_sample_x0': True,
    'sampling_range_x0': [default.x0_lower, default.x0_upper],
    'pos_body_idx': default.pos_body_idx,
    'pos_body_offset': pos_body_offset,
    'randomly_sample_bodypos': True,
    'sampling_range_bodypos': [default.pos_body_lower, default.pos_body_upper],
})

algorithm = default.algorithm.copy()
algorithm.update({
    'type': AlgorithmMDGPS,
    'sample_on_policy': True,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
#    'cluster_method': 'traj_em',
    'cluster_method': 'kmeans',
    'num_clusters': 8,
    'kl_step': 1.0,
    'max_step_mult': 2.0,
    'min_step_mult': 0.1,
})
algorithm['policy_opt']['weights_file_prefix'] = EXP_DIR + 'policy'
algorithm['dynamics']['prior']['max_samples'] = TRAIN_CONDITIONS

config = default.config.copy()
config.update({
    'common': common,
    'algorithm': algorithm,
    'agent': agent,
    'num_samples': 1,
})

common['info'] = generate_experiment_info(config)
