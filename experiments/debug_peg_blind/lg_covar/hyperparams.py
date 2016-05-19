""" Hyperparameters for MJC peg insertion policy optimization. """
import imp
import os.path
from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.traj_opt.traj_opt_lqr_python_mdgps import TrajOptLQRPythonMDGPS
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe

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
})

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

# Algorithm
algorithm = default.algorithm.copy()
algorithm.update({
    'type': AlgorithmMDGPS,
    'agent_use_nn_policy': False,
    'use_lg_covar': True,
})

algorithm['traj_opt'] = {
    'type': TrajOptLQRPythonMDGPS,
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
})

common['info'] = generate_experiment_info(config)
