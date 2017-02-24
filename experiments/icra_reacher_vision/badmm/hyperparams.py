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
TRAIN_CONDITIONS = 12
TOTAL_CONDITIONS = TEST_CONDITIONS + TRAIN_CONDITIONS

# Create body positions
pos_body_offset = []

# Outer square
bound = 0.75
xs = np.linspace(-bound, bound, 2)
zs = np.linspace(-bound, bound, 2)
pos_body_offset += [np.array([x,0,z]) for x in xs for z in zs]

# Inner square
bound = 0.25
xs = np.linspace(-bound, bound, 2)
zs = np.linspace(-bound, bound, 2)
pos_body_offset += [np.array([x,0,z]) for x in xs for z in zs]

# Middle diamond
bound = 0.5
xs = np.linspace(-bound, bound, 2)
pos_body_offset += [np.array([x,0,0]) for x in xs]
zs = np.linspace(-bound, bound, 2)
pos_body_offset += [np.array([0,0,z]) for z in zs]

# Add test positions.
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
    'kl_step': 1.0,
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
