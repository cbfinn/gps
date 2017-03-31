""" Default configuration and hyperparameter values for algorithms. """

import numpy as np


# Algorithm
ALG = {
    'inner_iterations': 1,  # Number of iterations.
    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
                      # trajectory optimization.
    'kl_step':0.2,
    'min_step_mult':0.01,
    'max_step_mult':10.0,
    'min_mult': 0.1,
    'max_mult': 5.0,
    # Trajectory settings.
    'initial_state_var':1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
                              # objects for each condition.
    # Trajectory optimization.
    'traj_opt': None,
    # Weight of maximum entropy term in trajectory optimization.
    'max_ent_traj': 0.0,
    # Dynamics hyperaparams.
    'dynamics': None,
    # Costs.
    'cost': None,  # A list of Cost objects for each condition.
    # Whether or not to sample with neural net policy (only for badmm/mdgps).
    'sample_on_policy': False,
    # Inidicates if the algorithm requires fitting of the dynamics.
    'fit_dynamics': True,    
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


# AlgorithmMDGPS
ALG_MDGPS = {
    # TODO: remove need for init_pol_wt in MDGPS
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    # Whether to use 'laplace' or 'mc' cost in step adjusment
    'step_rule': 'laplace',
}


# AlgorithmTrajOptPI2
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


# AlgorithmPILQR
ALG_PILQR = {
    'init_pol_wt': 0.01,
    # Dynamics fitting is not required for PI2 but it is for LQR.
    'fit_dynamics': True,
    # Whether to use 'const' or 'res_percent' in step adjusment
    'step_rule': 'res_percent',
    'step_rule_res_ratio_dec': 0.2,
    'step_rule_res_ratio_inc': 0.05,
    # The following won't work for a different horizon, but that's up to the user.
    'kl_step': np.linspace(0.4, 0.2, 100),
    'max_step_mult': np.linspace(10.0, 5.0, 100),
    'min_step_mult': np.linspace(0.01, 0.5, 100),
    'max_mult': np.linspace(5.0, 2.0, 100),
    'min_mult': np.linspace(0.1, 0.5, 100),
}


ALG_MDGPS_PILQR = {
    # TODO: remove need for init_pol_wt in MDGPS
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    # Dynamics fitting is not required for PI2 but it is for LQR.
    'fit_dynamics': True,
    # Whether to use 'const' or 'res_percent' in step adjusment
    'step_rule': 'res_percent',
    'step_rule_res_ratio_dec': 0.2,
    'step_rule_res_ratio_inc': 0.05,
    # The following won't work for a different horizon, but that's up to the user.
    'kl_step': np.linspace(0.4, 0.2, 100),
    'max_step_mult': np.linspace(10.0, 5.0, 100),
    'min_step_mult': np.linspace(0.01, 0.5, 100),
    'max_mult': np.linspace(5.0, 2.0, 100),
    'min_mult': np.linspace(0.1, 0.5, 100),
}
