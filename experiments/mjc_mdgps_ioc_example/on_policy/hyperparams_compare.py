""" Hyperparameters for MJC peg insertion policy optimization. """
import imp
import os.path
from gps.gui.config import generate_experiment_info

EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'
default = imp.load_source('default_hyperparams', EXP_DIR+'/hyperparams.py')


# Update the defaults
common = default.common.copy()
common.update({
    data_files_dir: exp_dir + 'data_files_maxent_9cond_z_0.05_%d' % common['itr'] + '/',
    'avg_legend': 'avg_non_maxent', # Legends for visualization
    'legend': 'non_maxent', # Legends for visualization
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
    
config = default.config.copy()
config.update({
    'common': common,
    'algorithm': algorithm,
})

common['info'] = generate_experiment_info(config)
