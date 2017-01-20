""" Default configuration and hyperparameter values for algorithms. """

# Algorithm
ALG = {
    'inner_iterations': 1,  # Number of iterations.
    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
                      # trajectory optimization.
    'kl_step':0.2,
    'min_step_mult':0.01,
    'max_step_mult':10.0,
    # Trajectory settings.
    'initial_state_var':1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
                              # objects for each condition.
    # Trajectory optimization.
    'traj_opt': None,
    # Use maximum entropy term in trajectory optimization.
    'max_ent_traj': 0.0,
    # Flag if we estimate the demo distribution empirically.
    'demo_distr_empest': True,
    # Flag if the algorithm is using IOC
    'ioc': None,  # ICML
    # number of iterations to run maxent and IOC for (-1 if all iters)
    'ioc_maxent_iter': -1,
    # Dynamics hyperaparams.
    'dynamics': None,
    # Costs.
    'cost': None,  # A list of Cost objects for each condition.
    # List of demonstrations of all conditions for the current iteration used in cost learning.
    'demo_list': None,
    # Number of demos per condition.
    'num_demos': 10,
    # Demo conditions.
    'demo_cond': 4,
    # variance multiplier for demos.
    'demo_var_mult': 1.0,
    # initial policy variance multiplier.
    'init_var_mult': 1.0,
    # Demo condition during training.
    'demo_M': 1,
    # Number of synthetic samples used to estimate the cost.
    'synthetic_cost_samples': 0,
    # Whether or not to sample with neural net policy (only for badmm/mdgps).
    'sample_on_policy': False,
    # Inidicates if the algorithm requires fitting of the dynamics.
    'fit_dynamics': True,    
    # Number of samples taken in the first iteration.
    'init_samples': 5,
}

# AlgorithmBADMM
ALG_BADMM = {
    'inner_iterations': 4,
    'policy_dual_rate': 0.1,
    'policy_dual_rate_covar': 0.0,
    'fixed_lg_step': 0,
    'lg_step_schedule': 10.0,
    'ent_reg_schedule': 0.0,
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
}

# AlgorithmMD
ALG_MDGPS = {
    # TODO: remove need for init_pol_wt in MDGPS
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    # Whether to use 'laplace' or 'mc' cost in step adjusment
    'step_rule': 'laplace',
    # algorithm file with policy to copy params from to cost.
    'init_cost_params': None,
}

# AlgorithmTrajOptPi2
ALG_PI2 = {
    # Dynamics fitting is not required for PI2.
    'fit_dynamics': False,
}

# AlgorithmPIGPS
ALG_PIGPS = {    
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',    
    # Dynamics fitting is not required for PIGPS.
    'fit_dynamics': False,
}
