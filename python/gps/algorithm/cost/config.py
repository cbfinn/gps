""" Default configuration and hyperparameter values for costs. """
import numpy as np

from gps.algorithm.cost.cost_utils import RAMP_CONSTANT, evallogl2term


# CostFK
COST_FK = {
    'wp': None,  # State weights - must be set.
    'target_end_effector': None,  # Target end-effector position.
    'evalnorm': evallogl2term,
    'alpha': 1e-5,
    'l1': 0.0,
    'l2': 1.0,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
}

# CostState (dist = Ax - tgt), x.shape depends on data_type
COST_STATE = {
    'wp': None,  # State weights - Defaults to ones.
    'A': None, # A matrix - Defaults to identity.
    'data_type': None, # Must be set.
    'target': 0.0,
    'evalnorm': evallogl2term,
    'alpha': 1e-5,
    'l1': 0.0,
    'l2': 1.0,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
}

# CostSum
COST_SUM = {
    'costs': [],  # A list of hyperparam dictionaries for each cost.
    'weights': [],  # Weight multipliers for each cost.
}


# CostAction
COST_ACTION = {
    'wu': np.array([]),  # Torque penalties, must be 1 x dU numpy array.
}
