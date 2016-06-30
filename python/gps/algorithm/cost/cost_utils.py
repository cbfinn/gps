""" This file defines utility classes and functions for costs. """
import json
import numpy as np
import sys

try:
  from caffe import layers as L
  from caffe.proto.caffe_pb2 import TRAIN, TEST, EltwiseParameter
except ImportError:
  L, TRAIN, TEST, EltwiseParameter = None, None, None, None

from gps.algorithm.cost import __file__ as current_path

RAMP_CONSTANT = 1
RAMP_LINEAR = 2
RAMP_QUADRATIC = 3
RAMP_FINAL_ONLY = 4


def get_ramp_multiplier(ramp_option, T, wp_final_multiplier=1.0):
    """
    Return a time-varying multiplier.
    Returns:
        A (T,) float vector containing weights for each time step.
    """
    if ramp_option == RAMP_CONSTANT:
        wpm = np.ones(T)
    elif ramp_option == RAMP_LINEAR:
        wpm = (np.arange(T, dtype=np.float32) + 1) / T
    elif ramp_option == RAMP_QUADRATIC:
        wpm = ((np.arange(T, dtype=np.float32) + 1) / T) ** 2
    elif ramp_option == RAMP_FINAL_ONLY:
        wpm = np.zeros(T)
        wpm[T-1] = 1.0
    else:
        raise ValueError('Unknown cost ramp requested!')
    wpm[-1] *= wp_final_multiplier
    return wpm


def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (l1 * sqrt(alpha + d^2))
    Args:
        wp: T x D matrix with weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect
            to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + \
            np.sqrt(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (
        dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1
    )
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(
        np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1
    )
    d2 = l1 * (
        (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
         (np.expand_dims(wp ** 2, axis=1) / psq)) -
        ((np.expand_dims(dscls, axis=1) *
          np.expand_dims(dscls, axis=2)) / psq ** 3)
    )
    d2 += l2 * (
        np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1])
    )

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0, 2, 1])

    return l, lx, lxx


def evallogl2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (0.5 * l1 * log(alpha + d^2))
    Args:
        wp: T x D matrix with weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect
            to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + \
            0.5 * np.log(alpha + np.sum(dscl ** 2, axis=1)) * l1
    # First order derivative terms.
    d1 = dscl * l2 + (
        dscls / (alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1
    )
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(
        alpha + np.sum(dscl ** 2, axis=1, keepdims=True), axis=1
    )
    #TODO: Need * 2.0 somewhere in following line, or * 0.0 which is
    #      wrong but better.
    d2 = l1 * (
        (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
         (np.expand_dims(wp ** 2, axis=1) / psq)) -
        ((np.expand_dims(dscls, axis=1) *
          np.expand_dims(dscls, axis=2)) / psq ** 2)
    )
    d2 += l2 * (
        np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1])
    )

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0, 2, 1])

    return l, lx, lxx


def construct_quad_cost_net(dim_hidden=None, dim_input=27, T=100,
                            demo_batch_size=5, sample_batch_size=5, phase=TRAIN):
    """
    Construct an anonymous network (no layer names) for a quadratic cost
    function with the specified dimensionality, and return NetParameter proto.

    Note: this function is an example for how one might want to specify
    their network, versus providing a protoxt model file. It is not
    meant to be a general solution for specifying any network.

    Args:
        dim_hidden: Dimensionality of hidden layer.
        dim_input: Dimensionality of input.
        T: time horizon
        demo_batch_size: demo batch size.
        sample_batch_size: sample batch size.
        phase: TRAIN, TEST, or 'deploy'
    Returns:
        A NetParameter specification of the network.
    """
    if dim_hidden is None:
        dim_hidden = 42

    # Needed for Caffe to find defined python layers.
    sys.path.append('/'.join(str.split(current_path, '/')[:-1]))

    if phase == TRAIN:
        data_layer_info = json.dumps({
            'shape': [{'dim': (demo_batch_size, T, dim_input)},
                      {'dim': (demo_batch_size, 1)},
                      {'dim': (sample_batch_size, T, dim_input)},
                      {'dim': (sample_batch_size, 1)}]
        })

        [demos, d_log_iw, samples, s_log_iw] = L.Python(
            ntop=4, python_param=dict(
                module='ioc_layers', param_str=data_layer_info,
                layer='IOCDataLayer'
            )
        )
        net_input = L.Concat(demos, samples, axis=0)
    elif phase == TEST:
        data_layer_info = json.dumps({
            'shape': [{'dim': (1, T, dim_input)}]
        })
        net_input = L.Python(ntop=1,
                             python_param=dict(module='policy_layers',
                                               param_str=data_layer_info,
                                               layer='PolicyDataLayer'))
    else:
        raise Exception('Unknown network phase')

    ip_out = L.InnerProduct(net_input, num_output=dim_hidden,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0),
                            axis=2)

    # Dot product operation with two layers
    dot_prod1 = L.Eltwise(ip_out, ip_out, operation=EltwiseParameter.PROD)
    all_costs = L.InnerProduct(dot_prod1, num_output=1,
                               weight_filler=dict(type='constant', value=1),
                               bias_filler=dict(type='constant', value=0),
                               param=[dict(lr_mult=0), dict(lr_mult=0)])

    if phase == TRAIN:
        demo_costs, sample_costs = L.Slice(all_costs, axis=0, slice_point=demo_batch_size, ntop=2)

        out = L.Python(demo_costs, sample_costs, d_log_iw, s_log_iw, loss_weight=1.0,
                       python_param=dict(module='ioc_layers',
                                         layer='IOCLoss'))
        # TODO - add regularizers, maybe with python loss layers
    else:
        out = all_costs

    return out.to_proto()
