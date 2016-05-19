""" Hyperparameters for MJC peg insertion policy optimization. """
import imp
import os.path
import numpy as np
from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe

BASE_DIR = '/'.join(str.split(__file__, '/')[:-3])
default = imp.load_source('default_hyperparams', BASE_DIR+'/hyperparams.py')

EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'

# Update the defaults
common = default.common.copy()
common.update({
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
})

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

# Algorithm
algorithm = default.algorithm.copy()
algorithm.update({
    'type': AlgorithmBADMM,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.1,
    'init_pol_wt': 0.01,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
})

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': 4000,
}

config = default.config.copy()
config.update({
    'common': common,
    'algorithm': algorithm,
    'verbose_policy_trials': 1,
    'seed': 47,
})

common['info'] = generate_experiment_info(config)
