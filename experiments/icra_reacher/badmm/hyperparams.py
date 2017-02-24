""" Hyperparameters for MJC peg insertion policy optimization. """
import imp
import os.path
import numpy as np
from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_badmm import AlgorithmBADMM

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
default = imp.load_source('default_hyperparams', BASE_DIR+'/hyperparams.py')

EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'

TEST_CONDITIONS = default.TEST_CONDITIONS
TRAIN_CONDITIONS = 8
TOTAL_CONDITIONS = TEST_CONDITIONS + TRAIN_CONDITIONS

# Create dummy train positions (will be overwritten randomly).
x0 = [np.zeros(6) for _ in range(TRAIN_CONDITIONS)]

pos_body_offset = [
        np.array([.4, .0, .7]),
        np.array([.4, .0, -.7]),
        np.array([-.4, .0, .7]),
        np.array([-.4, .0, -.7]),
        np.array([.7, .0, .4]),
        np.array([.7, .0, -.4]),
        np.array([-.7, .0, .4]),
        np.array([-.7, .0, -.4]) ]

# Add test positions.
x0 += default.test_x0
pos_body_offset += default.test_pos_body

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
    'x0': x0,
    'pos_body_idx': default.pos_body_idx,
    'pos_body_offset': pos_body_offset,
})

algorithm = default.algorithm.copy()
algorithm.update({
    'type': AlgorithmBADMM,
    'inner_iterations': 2,
    'sample_on_policy': False,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'lg_step_schedule': np.array([1e-4, 1e-4, 1e-3, 1e-3]),
    'policy_dual_rate': 0.1,
    'init_pol_wt': 0.002,
    'fixed_lg_step': 3,
    'kl_step': 0.5,
    'max_step_mult': 2.0,
    'min_step_mult': 0.1,
})
algorithm['policy_opt']['weights_file_prefix'] = EXP_DIR + 'policy'
algorithm['dynamics']['prior']['max_samples'] = 5*TRAIN_CONDITIONS

config = default.config.copy()
config.update({
    'common': common,
    'algorithm': algorithm,
    'agent': agent,
    'num_samples': 5,
})

common['info'] = generate_experiment_info(config)
