""" Default configuration for trajectory optimization. """


# TrajOptLQRPython
TRAJ_OPT_LQR = {
    # Dual variable updates for non-PD Q-function.
    'del0': 1e-4,
    'eta_error_threshold': 1e16,
    'min_eta': 1e-8,
    'max_eta': 1e16,
    'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
    'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
    'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
}

# TrajOptPI2
TRAJ_OPT_PI2 = {  
    'kl_threshold': 1.0,    
    'covariance_damping': 2.0,
    'min_temperature': 0.001,
    'use_sumexp': False,
    'pi2_use_dgd_eta': False,
    'pi2_cons_per_step': True,
    'min_eta': 1e-8,
    'max_eta': 1e16,
    'del0': 1e-4,
}

# TrajOptPILQR
TRAJ_OPT_PILQR = {
    'use_lqr_actions': True,
    'cons_per_step': True,
}
