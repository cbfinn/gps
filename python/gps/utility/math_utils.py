import numpy as np
import scipy as sp
import scipy.linalg

CHECK_FINITE=False
def lu_solve(L, U, A):
    """Solves LUX=A for X"""
    return sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, A, lower=True,
            check_finite=CHECK_FINITE), check_finite=CHECK_FINITE)


def solve_psd(A, B, reg=0):
    """Solve AX=B via cholesky decomposition (A must be positive semidefinite)"""
    chol = sp.linalg.cholesky(A + reg * np.eye(A.shape[0]))
    return lu_solve(chol.T, chol, B)


def invert_psd(A, reg=0):
    """Invert a PSD matrix via Cholesky + Triangular solves"""
    return solve_psd(A, np.eye(A.shape[0]), reg=reg)