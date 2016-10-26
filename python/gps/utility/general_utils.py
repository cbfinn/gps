""" This file defines general utility functions and classes. """
import numpy as np

class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)


def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape.
    Args:
        value: Matrix to shape check.
        expected_shape: A tuple or list of integers.
        name: An optional name to add to the exception message.
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
                         (name, str(expected_shape), str(value.shape)))


def finite_differences(func, inputs, func_output_shape=(), epsilon=1e-5):
    """
    Computes gradients via finite differences.
    derivative = (func(x+epsilon) - func(x-epsilon)) / (2*epsilon)
    Args:
        func: Function to compute gradient of. Inputs and outputs can be
            arbitrary dimension.
        inputs: Vector value to compute gradient at.
        func_output_shape: Shape of the output of func. Default is
            empty-tuple, which works for scalar-valued functions.
        epsilon: Difference to use for computing gradient.
    Returns:
        Gradient vector of each dimension of func with respect to each
        dimension of input.
    """
    gradient = np.zeros(inputs.shape+func_output_shape)
    for idx, _ in np.ndenumerate(inputs):
        test_input = np.copy(inputs)
        test_input[idx] += epsilon
        obj_d1 = func(test_input)
        assert obj_d1.shape == func_output_shape
        test_input = np.copy(inputs)
        test_input[idx] -= epsilon
        obj_d2 = func(test_input)
        assert obj_d2.shape == func_output_shape
        diff = (obj_d1 - obj_d2) / (2 * epsilon)
        gradient[idx] += diff
    return gradient


def approx_equal(a, b, threshold=1e-5):
    """
    Return whether two numbers are equal within an absolute threshold.
    Returns:
        True if a and b are equal within threshold.
    """
    return np.all(np.abs(a - b) < threshold)


def extract_condition(hyperparams, m):
    """
    Pull the relevant hyperparameters corresponding to the specified
    condition, and return a new hyperparameter dictionary.
    """
    return {var: val[m] if isinstance(val, list) else val
            for var, val in hyperparams.items()}


def get_ee_points(offsets, ee_pos, ee_rot):
    """
    Helper method for computing the end effector points given a
    position, rotation matrix, and offsets for each of the ee points.

    Args:
        offsets: N x 3 array where N is the number of points.
        ee_pos: 1 x 3 array of the end effector position.
        ee_rot: 3 x 3 rotation matrix of the end effector.
    Returns:
        3 x N array of end effector points.
    """
    return ee_rot.dot(offsets.T) + ee_pos.T

def sample_params(sampling_range, prohibited_ranges):
    """
    Samples parameters from sampling_range ensuring that the sampled_point
    doesn't lie in any of the prohibited_ranges.

    Args:
        sampling_range: A list of form [lower_lim, upper_lim] where lower_lim
            and upper_lim are two numpy arrays with size N x 1 where N is the
            dimension of the sample.
        prohibited_ranges: A list of ranges which describes the region from which
            the point can't be sampled. A range is a list of form [[l_0, u_0], ..,
            [l_N, u_N]] where l_i and u_i are the upper and lower limits of the
            i-th dimension. Can contain replace a [l_i, u_i] with None.
    Returns:
        sampled_point: A sampled vector within sampling range and lying outside
        prohibited ranges.
    """
    lower_lim, upper_lim = sampling_range
    N = len(lower_lim)
    assert N == len(upper_lim)

    valid = False
    while not valid:
        # Sample point from within lower/upper limits.
        sampled_point = np.random.rand(N)*(upper_lim - lower_lim) + lower_lim

        # Make sure it is valid.
        valid = True
        for prohibited_range in prohibited_ranges:
            for i in range(N):
                if prohibited_range[i] is None:
                    continue

                lower, upper = prohibited_range[i]
                if (lower <= sampled_point[i] <= upper):
                    valid = False
                    break

            if not valid:
                break

    return sampled_point
