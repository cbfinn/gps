""" This file defines general utility functions and classes. """
import errno
import numpy as np
import os
import time
import traceback as tb
from collections import Mapping, Container
from sys import getsizeof

from gps.utility import color_string
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS


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


class Timer(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.time_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        new_time = time.time() - self.time_start
        fname, lineno, method, _ = tb.extract_stack()[-2]  # Get caller
        _, fname = os.path.split(fname)
        id_str = '%s:%s' % (fname, method)
        print 'TIMER:'+color_string('%s: %s (Elapsed: %fs)' % (id_str, self.message, new_time), color='gray')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def flatten_lists(lists):
    """
    Flattens a list of lists into a single list
    """
    if type(lists[0]) is not list:
        return lists
    return [obj for sublist in lists for obj in sublist]


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

def logsum(vec, dim):
    """ Safe sum of log values. """
    maxv = vec.max(dim)
    maxv[maxv == -np.inf] = 0
    return np.log(np.sum(np.exp(vec - maxv), dim)) + maxv

def disable_caffe_logs(unset_glog_level=None):
    """
    Function that disables caffe printouts
    """
    if unset_glog_level is None:
        if 'GLOG_minloglevel' not in os.environ:
            os.environ['GLOG_minloglevel'] = '2'
            unset_glog_level = True
        else:
            unset_glog_level = False
        return unset_glog_level
    elif unset_glog_level:
        del os.environ['GLOG_minloglevel']

def sample_params(sampling_range, prohibited_ranges):
    """
    Samples parameters from sampling_range ensuring that sampled_point
    doesn't lie in any range in prohibited_ranges
    Args:
        sampling_range: A list of form [lower_lim, upper_lim] where lower_lim
        and upper_lim are two numpy arrays with size N x 1 where N is dimension
        of sample
        prohibited_ranges: A list of range which describes the region from which
        the point can't be sampled. A range is a list of form [[l_0, u_0], ..,
        [l_N, u_N]] where l_i and u_i is upper and lower limit of ith dimension
    Returns:
        sampled_point: A sampled vector within sampling range and lying outside
        prohibited ranges.
    """
    lower_lim, upper_lim = sampling_range
    N = len(lower_lim)
    valid_point = False
    while not valid_point:
        sampled_point = np.random.rand(N)*(upper_lim - lower_lim) + lower_lim
        for prohibited_range in prohibited_ranges:
            if all(lst is None for lst in prohibited_range):
                valid_point = True
                continue
            valid_point = False
            for i in range(len(prohibited_range)):
                if prohibited_range[i] and not (prohibited_range[i][0] <= sampled_point[i] <= prohibited_range[i][1]):
                    valid_point = True
                    break

            if not valid_point:
                break
    return sampled_point

def compute_distance(target_end_effector, sample_list, state_idxs=range(4, 7), end_effector_idxs=range(0,3), filter_type='min'):
    target_position = target_end_effector
    if type(sample_list) is not list:
        cur_samples = sample_list.get_samples()
    else:
        cur_samples = []
        for m in xrange(len(pol_sample_lists)):
            samples = pol_sample_lists[m].get_samples()
            for sample in samples:
                cur_samples.append(sample)
    sample_end_effectors = [cur_samples[i].get_X() for i in xrange(len(cur_samples))]
    if filter_type == 'min':
        if type(target_position) is not list:
            dists = [np.nanmin((np.sqrt(np.sum((sample_end_effectors[i][:, state_idxs] - target_position[end_effector_idxs].reshape(1, -1))**2,
                    axis=1))), axis=0) for i in xrange(len(cur_samples))]
        else:
            N = len(cur_samples)/len(target_position)
            dists = [np.nanmin((np.sqrt(np.sum((sample_end_effectors[i][:, state_idxs] - target_position[i/N][end_effector_idxs].reshape(1, -1))**2,
                    axis=1))), axis=0) for i in xrange(len(cur_samples))]
    elif filter_type == 'last':
        if type(target_position) is not list:
            dists = [np.sqrt(np.sum((sample_end_effectors[i][:, state_idxs] - target_position[end_effector_idxs].reshape(1, -1))**2,
                    axis=1))[-1] for i in xrange(len(cur_samples))]
        else:
            N = len(cur_samples)/len(target_position)
            dists = [np.sqrt(np.sum((sample_end_effectors[i][:, state_idxs] - target_position[i/N][end_effector_idxs].reshape(1, -1))**2,
                    axis=1))[-1] for i in xrange(len(cur_samples))]
    else:
        raise NotImplementedError()
    return dists

class BatchSampler(object):
    """ Samples data """
    def __init__(self, data, batch_dim=0):
        self.data = data
        self.batch_dim = batch_dim

        # Check that all data has same size on batch_dim
        self.num_data = data[0].shape[batch_dim]
        for d in data:
            assert d.shape[batch_dim] == self.num_data, "Bad shape on axis %d: %s, (expected %d)" % \
                                                        (batch_dim, str(d.shape), self.num_data)

    def with_replacement(self, batch_size=10):
        while True:
            batch_idx = np.random.randint(0, self.num_data, size=batch_size)
            batch = [data[batch_idx] for data in self.data]
            yield batch

    def iterate(self, batch_size=10, epochs=float('inf'), shuffle=True):
        raise NotImplementedError()


def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    if hasattr(o, '__dict__'):
        tot = 0
        for k in o.__dict__:
            tot += d(getattr(o, k), ids)
        return r+tot

    return r
