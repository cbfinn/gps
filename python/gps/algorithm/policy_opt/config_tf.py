from gps.algorithm.policy_opt.tf_model_example import example_tf_network
import os
checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..', 'policy_opt/tf_checkpoint/policy_checkpoint.ckpt'))

POLICY_OPT_TF = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 2000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'use_gpu': False,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'solver_type': 'adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # Other hyperparameters.
    'network_model': example_tf_network,  # should return TfMap object from tf_utils. See example.
    'checkpoint_prefix': checkpoint_path
}
