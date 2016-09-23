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
})

plot = {
    'itr' = 0, # For comparison experiments
    'avg_legend': 'avg_maxent', # Legends for visualization
    'legend': 'maxent', # Legends for visualization
    'avg_legend_compare': 'avg_non_maxent', # Legends of control group for visualization
    'legend_compare': 'non_maxent', # Legends of control group for visualization
    'mean_dist_title': "mean distances to target over time with MaxEnt demo and not-MaxEnt demo",
    'success_title': "success rates during iterations with MaxEnt demo and not-MaxEnt demo",
    'xlabel': "iterations",
    'mean_dist_ylabel': "mean distances",
    'success_ylabel': "success rate",
    'mean_dist_plot_name': 'mean_dists_during_iteration_comparison_maxent.pdf',
    'success_plot_name': 'success_rate_during_iteration_comparison_maxent.pdf',
}


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
    'plot': plot,
})

common['info'] = generate_experiment_info(config)
