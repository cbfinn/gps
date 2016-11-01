""" Default configuration for policy optimization. """
try:
    from gps.algorithm.policy_opt.policy_opt_utils import construct_fc_network
except ImportError:
    construct_fc_network = None

import os

# config options shared by both caffe and tf.
GENERIC_CONFIG = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 5000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # set gpu usage.
    'use_gpu': 1,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'random_seed': 1,
}


POLICY_OPT_CAFFE = {
    # Other hyperparameters.
    'network_model': construct_fc_network,  # Either a filename string
                                            # or a function to call to
                                            # create NetParameter.
    'network_arch_params': {},  # Arguments to pass to method above.
    'weights_file_prefix': '',
}

POLICY_OPT_CAFFE.update(GENERIC_CONFIG)


POLICY_OPT_TF = {
    # Other hyperparameters.
    'copy_param_scope': 'conv_params',
    'fc_only_iterations': 0,
}

POLICY_OPT_TF.update(GENERIC_CONFIG)
