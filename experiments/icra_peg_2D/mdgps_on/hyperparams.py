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
TRAIN_CONDITIONS = 4
TOTAL_CONDITIONS = TEST_CONDITIONS + TRAIN_CONDITIONS

# Create train positions.
xs = np.linspace(default.lower, default.upper, 2)
ys = np.linspace(default.lower, default.upper, 2)
pos_body_offset = [np.array([x,y,0]) for x in xs for y in ys]

# Add test positions.
pos_body_offset += default.test_pos_body_offset

# Update the defaults
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
    'pos_body_idx': np.array([1]),
    'pos_body_offset': pos_body_offset,
})

algorithm = default.algorithm.copy()
algorithm.update({
    'type': AlgorithmMDGPS,
    'sample_on_policy': True,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'kl_step': 2.0,
    'max_step_mult': 2.0,
    'min_step_mult': 0.5,
})
algorithm['policy_opt']['weights_file_prefix'] = EXP_DIR + 'policy'

config = default.config.copy()
config.update({
    'common': common,
    'algorithm': algorithm,
    'agent': agent,
    'num_samples': 5,
})

common['info'] = generate_experiment_info(config)
