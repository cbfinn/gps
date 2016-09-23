""" Hyperparameters for MJC peg insertion policy optimization. """
import imp
import os.path
from gps.gui.config import generate_experiment_info

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
default = imp.load_source('default_hyperparams', BASE_DIR+'/hyperparams.py')

EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'

# Update the defaults
common = default.common.copy()
common.update({
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'NN_demo_file': EXP_DIR + 'data_files/demo_NN.pkl',
    'LG_demo_file': EXP_DIR + 'data_files/demo_LG.pkl',
})

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

# Algorithm
algorithm = default.algorithm.copy()
algorithm.update({
    'sample_on_policy': True,
})

algorithm['policy_opt'].update({
    'weights_file_prefix': common['data_files_dir'] + 'policy',
})

algorithm['cost'].update({
	'demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_nn_MaxEnt_9_cond_z_0.05_3pols_no_noise.pkl'),
	'weight_dir': common['data_files_dir'],
})
    
config = default.config.copy()
config.update({
    'common': common,
    'algorithm': algorithm,
})

common['info'] = generate_experiment_info(config)
