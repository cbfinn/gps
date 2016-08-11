""" Default configuration and hyperparameter values for algorithms. """

# Algorithm
ALG = {
    'inner_iterations': 1,  # Number of iterations.
    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
                      # trajectory optimization.
    'kl_step':0.2,
    'min_step_mult':0.01,
    'max_step_mult':10.0,
    'sample_decrease_var':0.5,
    'sample_increase_var':1.0,
    # Trajectory settings.
    'initial_state_var':1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
                              # objects for each condition.
    # Trajectory optimization.
    'traj_opt': None,
    # Use maximum entropy term in trajectory optimization.
    'max_ent_traj': 0.0,
    # Flag if we estimate the demo distribution empirically.
    'demo_distr_empest': False,
    # Flag if the algorithm is using IOC
    'ioc': False,
    # Flag if the algorithm is learning from prior experience
    'learning_from_prior': False,
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
    # verbose when generating demos.
    "demo_verbose": False,
    # Demo condition during training.
    'demo_M': 1,
    # Number of synthetic samples used to estimate the cost.
    'synthetic_cost_samples': 0,
    # Whether or not to sample with neural net policy (only for badmm/mdgps).
    'sample_on_policy': False,
    # Flag if the algorithm is using global cost.
    'global_cost': True,
    # Use demo neural net policy to initialize policy.
    'init_demo_policy': False,
    # Number of samples taken in the first iteration.
    'init_samples': 5,
    # Evaluate importance weights using nn policy.
    'policy_eval': False,
}

# Good indices.
good_indices = range(35)
good_indices.extend(range(36, 40))
ALG['good_indices'] = good_indices

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
    'max_policy_samples': 20,
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
    'max_policy_samples': 20,
    'step_rule': 'classic',
}
