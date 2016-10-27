""" Default configuration for trajectory optimization. """


# TrajOptLQRPython
TRAJ_OPT_LQR = {
    # Dual variable updates for non-PD Q-function.
    'del0': 1e-4,
    'eta_error_threshold': 1e16,
    'min_eta': 1e-4,
    'max_eta': 1e16,
}

# TrajOptPi2
TRAJ_OPT_PI2 = {  
    'kl_threshold': 1.0,    
    'covariance_damping': 2.0,
    'min_temperature': 0.001,
}
