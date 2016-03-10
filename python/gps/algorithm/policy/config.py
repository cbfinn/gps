""" Default configuration and hyperparameter values for policies. """


# Initial Linear Gaussian Trajectory Distributions, PD-based initializer.
# Note, PD is the default initializer type.
INIT_LG_PD = {
    'init_var': 10.0,
    'pos_gains': 10.0, # position gains
    'vel_gains_mult': 0.01,  # velocity gains multiplier on pos_gains
    'init_action_offset': None,
}

# Initial Linear Gaussian Trajectory distribution, LQR-based initializer.
INIT_LG_LQR = {
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 1.0,
    # Parameters for guessing dynamics
    'init_acc': [],  # dU vector of accelerations, default zeros.
    'init_gains': [],  # dU vector of gains, default ones.
}

# PolicyPrior
POLICY_PRIOR = {
    'strength': 1e-4,
}

# PolicyPriorGMM
POLICY_PRIOR_GMM = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
    'keep_samples': True,
}
