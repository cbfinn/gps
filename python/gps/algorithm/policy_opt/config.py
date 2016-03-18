""" Default configuration for policy optimization. """
from gps.algorithm.policy_opt.policy_opt_utils import construct_fc_network


POLICY_OPT_CAFFE = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 20000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'use_gpu': 1,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # Other hyperparameters.
    'network_model': construct_fc_network,  # Either a filename string
                                            # or a function to call to
                                            # create NetParameter.
    'network_arch_params': {},  # Arguments to pass to method above.
    'weights_file_prefix': '',
}
