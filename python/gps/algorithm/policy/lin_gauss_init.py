""" Initializations for linear Gaussian controllers. """
import copy
import numpy as np
import scipy as sp

from gps.algorithm.dynamics.dynamics_utils import guess_dynamics
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.algorithm.policy.config import INIT_LG_PD, INIT_LG_LQR


def init_lqr(hyperparams):
    """
    Return initial gains for a time-varying linear Gaussian controller
    that tries to hold the initial position.
    """
    config = copy.deepcopy(INIT_LG_LQR)
    config.update(hyperparams)

    x0, dX, dU = config['x0'], config['dX'], config['dU']
    dt, T = config['dt'], config['T']

    #TODO: Use packing instead of assuming which indices are the joint
    #      angles.

    # Notation notes:
    # L = loss, Q = q-function (dX+dU dimensional),
    # V = value function (dX dimensional), F = dynamics
    # Vectors are lower-case, matrices are upper case.
    # Derivatives: x = state, u = action, t = state+action (trajectory).
    # The time index is denoted by _t after the above.
    # Ex. Ltt_t = Loss, 2nd derivative (w.r.t. trajectory),
    # indexed by time t.

    # Constants.
    idx_x = slice(dX)  # Slices out state.
    idx_u = slice(dX, dX+dU)  # Slices out actions.

    if len(config['init_acc']) == 0:
        config['init_acc'] = np.zeros(dU)

    if len(config['init_gains']) == 0:
        config['init_gains'] = np.ones(dU)

    # Set up simple linear dynamics model.
    Fd, fc = guess_dynamics(config['init_gains'], config['init_acc'],
                            dX, dU, dt)

    # Setup a cost function based on stiffness.
    # Ltt = (dX+dU) by (dX+dU) - Hessian of loss with respect to
    # trajectory at a single timestep.
    Ltt = np.diag(np.hstack([
        config['stiffness'] * np.ones(dU),
        config['stiffness'] * config['stiffness_vel'] * np.ones(dU),
        np.zeros(dX - dU*2), np.ones(dU)
    ]))
    Ltt = Ltt / config['init_var']  # Cost function - quadratic term.
    lt = -Ltt.dot(np.r_[x0, np.zeros(dU)])  # Cost function - linear term.

    # Perform dynamic programming.
    K = np.zeros((T, dU, dX))  # Controller gains matrix.
    k = np.zeros((T, dU))  # Controller bias term.
    PSig = np.zeros((T, dU, dU))  # Covariance of noise.
    cholPSig = np.zeros((T, dU, dU))  # Cholesky decomposition.
    invPSig = np.zeros((T, dU, dU))  # Inverse of covariance.
    vx_t = np.zeros(dX)  # Vx = dV/dX. Derivative of value function.
    Vxx_t = np.zeros((dX, dX))  # Vxx = ddV/dXdX.

    #TODO: A lot of this code is repeated with traj_opt_lqr_python.py
    #      backward pass.
    for t in range(T - 1, -1, -1):
        # Compute Q function at this step.
        if t == (T - 1):
            Ltt_t = config['final_weight'] * Ltt
            lt_t = config['final_weight'] * lt
        else:
            Ltt_t = Ltt
            lt_t = lt
        # Qtt = (dX+dU) by (dX+dU) 2nd Derivative of Q-function with
        # respect to trajectory (dX+dU).
        Qtt_t = Ltt_t + Fd.T.dot(Vxx_t).dot(Fd)
        # Qt = (dX+dU) 1st Derivative of Q-function with respect to
        # trajectory (dX+dU).
        qt_t = lt_t + Fd.T.dot(vx_t + Vxx_t.dot(fc))

        # Compute preceding value function.
        U = sp.linalg.cholesky(Qtt_t[idx_u, idx_u])
        L = U.T

        invPSig[t, :, :] = Qtt_t[idx_u, idx_u]
        PSig[t, :, :] = sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
        )
        cholPSig[t, :, :] = sp.linalg.cholesky(PSig[t, :, :])
        K[t, :, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, Qtt_t[idx_u, idx_x], lower=True)
        )
        k[t, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, qt_t[idx_u], lower=True)
        )
        Vxx_t = Qtt_t[idx_x, idx_x] + Qtt_t[idx_x, idx_u].dot(K[t, :, :])
        vx_t = qt_t[idx_x] + Qtt_t[idx_x, idx_u].dot(k[t, :])
        Vxx_t = 0.5 * (Vxx_t + Vxx_t.T)

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)


#TODO: Fix docstring
def init_pd(hyperparams):
    """
    This function initializes the linear-Gaussian controller as a
    proportional-derivative (PD) controller with Gaussian noise. The
    position gains are controlled by the variable pos_gains, velocity
    gains are controlled by pos_gains*vel_gans_mult.
    """
    config = copy.deepcopy(INIT_LG_PD)
    config.update(hyperparams)

    dU, dQ, dX = config['dU'], config['dQ'], config['dX']
    x0, T = config['x0'], config['T']

    # Choose initialization mode.
    Kp = 1.0
    Kv = config['vel_gains_mult']
    if dU < dQ:
        K = -config['pos_gains'] * np.tile(
            [np.eye(dU) * Kp, np.zeros((dU, dQ-dU)),
             np.eye(dU) * Kv, np.zeros((dU, dQ-dU))],
            [T, 1, 1]
        )
    else:
        K = -config['pos_gains'] * np.tile(
            np.hstack([
                np.eye(dU) * Kp, np.eye(dU) * Kv,
                np.zeros((dU, dX - dU*2))
            ]), [T, 1, 1]
        )
    k = np.tile(-K[0, :, :].dot(x0), [T, 1])
    PSig = config['init_var'] * np.tile(np.eye(dU), [T, 1, 1])
    cholPSig = np.sqrt(config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])
    invPSig = (1.0 / config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)
