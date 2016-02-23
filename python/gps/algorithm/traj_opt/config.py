""" Default configuration for trajectory optimization. """


# TrajOptLQRPython
TRAJ_OPT_LQR = {
    # Dual variable updates for non-PD Q-function.
    'del0': 1e-4,
    'eta_error_threshold': 1e16,
    'min_eta': 1e-4,
}
