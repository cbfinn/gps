""" This file defines utility classes and functions for costs. """
import json
import numpy as np
import sys
from gps.utility.general_utils import disable_caffe_logs

unset = disable_caffe_logs()
try:
  import caffe
  from caffe import layers as L
  from caffe.proto.caffe_pb2 import TRAIN, TEST, EltwiseParameter
except ImportError:
  L, TRAIN, TEST, EltwiseParameter = None, None, None, None
disable_caffe_logs(unset)

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
                            demo_batch_size=5, sample_batch_size=5, phase=TRAIN, ioc_loss='ICML',
                            smooth_reg_weight=0.0, mono_reg_weight=0.0):
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
        ioc_loss: type of loss to use -- ICML, MPF, IOCGAN, XENTGAN
    Returns:
        A NetParameter specification of the network.
    """
    from gps.algorithm.cost.config import COST_IOC_QUADRATIC

    if dim_hidden is None:
        dim_hidden = 42

    n = caffe.NetSpec()

    # Needed for Caffe to find defined python layers.
    sys.path.append('/'.join(str.split(current_path, '/')[:-1]))
    if phase == TRAIN:
        data_layer_info = json.dumps({
            'shape': [{'dim': (demo_batch_size, T, dim_input)},
                      {'dim': (demo_batch_size, 1)},
                      {'dim': (sample_batch_size, T, dim_input)},
                      {'dim': (sample_batch_size, 1)}]
        })

        [n.demos, n.d_log_iw, n.samples, n.s_log_iw] = L.Python(
            ntop=4, python_param=dict(
                module='ioc_layers', param_str=data_layer_info,
                layer='IOCDataLayer'
            )
        )
        n.net_input = L.Concat(n.demos, n.samples, axis=0)
    elif phase == TEST:
        data_layer_info = json.dumps({
            'shape': [{'dim': (1, T, dim_input)}]
        })
        n.net_input = L.Python(ntop=1,
                               python_param=dict(module='ioc_layers',
                                                 param_str=data_layer_info,
                                                 layer='IOCDataLayer'))
    else:
        raise Exception('Unknown network phase')

    n.Ax = L.InnerProduct(n.net_input, num_output=dim_hidden,
                              weight_filler=dict(type='gaussian', std=0.01),
                              bias_filler=dict(type='constant', value=0),
                              axis=2)

    # Dot product operation with two layers
    n.AxAx = L.Eltwise(n.Ax, n.Ax, operation=EltwiseParameter.PROD)
    n.all_costs = L.InnerProduct(n.AxAx, num_output=1, axis=2,
                                 weight_filler=dict(type='constant', value=1),
                                 bias_filler=dict(type='constant', value=0),
                                 param=[dict(lr_mult=0), dict(lr_mult=0)])

    if phase == TRAIN:
        n.demo_costs, n.sample_costs = L.Slice(n.all_costs, axis=0, slice_point=demo_batch_size, ntop=2)

        # smoothness regularization
        n.costs_prev, _ = L.Slice(n.all_costs, axis=1, slice_point=T-2, ntop=2)
        _, n.costs_next = L.Slice(n.all_costs, axis=1, slice_point=2, ntop=2)
        _, n.costs_cur, _ = L.Slice(n.all_costs, axis=1, slice_point=[1,T-1], ntop=3)
        # cur-prev
        n.slope_prev = L.Eltwise(n.costs_cur, n.costs_prev, operation=EltwiseParameter.SUM, coeff=[1,-1])
        # next-cur
        n.slope_next = L.Eltwise(n.costs_next, n.costs_cur, operation=EltwiseParameter.SUM, coeff=[1,-1])
        n.reg = L.EuclideanLoss(n.slope_next, n.slope_prev, loss_weight=smooth_reg_weight)  # TODO - make loss weight a hyperparam

        n.demo_slope, _ = L.Slice(n.slope_next, axis=0, slice_point=demo_batch_size, ntop=2)
        n.demo_slope_reshape = L.Reshape(n.demo_slope, shape=dict(dim=[-1,1]))
        # TODO - add hyperparam for loss weight, maybe try l2 monotonic loss
        n.mono_reg = L.Python(n.demo_slope_reshape, loss_weight=mono_reg_weight,
                              python_param=dict(module='ioc_layers', layer='L2MonotonicLoss'))

        n.dummy = L.DummyData(ntop=1, shape=dict(dim=[1]), data_filler=dict(type='constant',value=0))
        # init logZ or Z to 1, only learn the bias
        # (also might be good to reduce lr on bias)
        n.logZ = L.InnerProduct(n.dummy, axis=0, num_output=1,
                             weight_filler=dict(type='constant', value=0),
                             bias_filler=dict(type='constant', value=1),
                             param=[dict(lr_mult=1), dict(lr_mult=1)])
        n.Z = L.Exp(n.logZ, base=2.6)

        if ioc_loss == 'XENTGAN':
            pass
        elif ioc_loss== 'IOCGAN':
            layer_name = 'IOCLossMod'
        else:
            layer_name = 'IOCLoss'
        n.out = L.Python(n.demo_costs, n.sample_costs, n.d_log_iw, n.s_log_iw, n.Z, loss_weight=1.0,
                         python_param=dict(module='ioc_layers',
                                           layer=layer_name))


    return n.to_proto()


def construct_nn_cost_net(num_hidden=1, dim_hidden=None, dim_input=27, T=100,
                          demo_batch_size=5, sample_batch_size=5, phase=TRAIN, ioc_loss='ICML',
                          Nq=1, smooth_reg_weight=0.0, mono_reg_weight=0.0, gp_reg_weight=0.0, learn_wu=False):
    """
    Construct an anonymous network (no layer names) for a quadratic cost
    function with the specified dimensionality, and return NetParameter proto.

    Note: this function is an example for how one might want to specify
    their network, versus providing a protoxt model file. It is not
    meant to be a general solution for specifying any network.

    Args:
        num_hidden: Number of hidden layers.
        dim_hidden: Dimensionality of hidden layer.
        dim_input: Dimensionality of input.
        T: time horizon
        demo_batch_size: demo batch size.
        sample_batch_size: sample batch size.
        phase: TRAIN, TEST, or 'forward_feat'
        ioc_loss: type of loss to use -- ICML, MPF, IOCGAN, XENTGAN
        Nq: number of distributions q from which the samples were drawn (only used for MPF)
        reg_weight: The weight of each of the regularizers
        learn_wu: Whether or not to learn an additional multiplier on the torque.
    Returns:
        A NetParameter specification of the network.
    """
    from gps.algorithm.cost.config import COST_IOC_NN
    num_hidden = 1
    dim_hidden = 16
    if dim_hidden is None:
        dim_hidden = 21

    n = caffe.NetSpec()

    # Needed for Caffe to find defined python layers.
    sys.path.append('/'.join(str.split(current_path, '/')[:-1]))
    if phase == 'supervised':
        data_layer_info = json.dumps({
            'shape': [{'dim': (sample_batch_size+demo_batch_size, T, dim_input)}, # sample obs
                      {'dim': (sample_batch_size+demo_batch_size, T, 1)},  # sample torque norm
                      {'dim': (sample_batch_size+demo_batch_size, T, 1)}]  # gt cost labels
        })
        n.net_input, n.all_u, n.cost_labels = L.Python(ntop=3,
                               python_param=dict(module='ioc_layers',
                                                 param_str=data_layer_info,
                                                 layer='IOCDataLayer'))
    elif phase == TRAIN:
        data_layer_info = json.dumps({
            'shape': [{'dim': (demo_batch_size, T, dim_input)},  # demo obs
                      {'dim': (demo_batch_size, T, 1)},  # demo torque norm
                      {'dim': (demo_batch_size, 1)},  # demo i.w.
                      {'dim': (sample_batch_size, T, dim_input)},  # sample obs
                      {'dim': (sample_batch_size, T, 1)},  # sample torque norm
                      {'dim': (sample_batch_size, 1)}, # sample i.w.
                      {'dim': (dim_input, )}] # length scale
        })

        [n.demos, n.demou, n.d_log_iw, n.samples, n.sampleu, n.s_log_iw, n.l] = L.Python(
            ntop=7, python_param=dict(module='ioc_layers', param_str=data_layer_info,
                                      layer='IOCDataLayer')
            )
        n.net_input = L.Concat(n.demos, n.samples, axis=0)
        n.all_u = L.Concat(n.demou, n.sampleu, axis=0)
    elif phase == TEST: # or phase == 'forward_feat':
        data_layer_info = json.dumps({
            'shape': [{'dim': (1, T, dim_input)},
                      {'dim': (1, T, 1)}]
        })
        n.net_input, n.all_u = L.Python(ntop=2,
                               python_param=dict(module='ioc_layers',
                                                 param_str=data_layer_info,
                                                 layer='IOCDataLayer'))
    elif phase == 'forward_feat':
        data_layer_info = json.dumps({
            'shape': [{'dim': (1, T, dim_input)}]
        })
        n.net_input = L.Python(ntop=1,
                               python_param=dict(module='ioc_layers',
                                                 param_str=data_layer_info,
                                                 layer='IOCDataLayer'))
    else:
        raise Exception('Unknown network phase')


    n.layer = n.net_input
    for i in range(num_hidden-1):
        n.layer = L.InnerProduct(n.layer, num_output=dim_hidden,
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0),
                                 axis=2)
        n.layer = L.ReLU(n.layer, in_place=True)

    # Necessary for computing gradients
    loss_weight = 1.0 if phase == 'forward_feat' else 0.0
    n.feat = L.InnerProduct(n.layer, num_output=dim_hidden,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0),
                            axis=2, loss_weight=loss_weight)

    if phase != 'forward_feat':
        if learn_wu:
            # learned multiplier
            n.all_u = L.InnerProduct(n.all_u, num_output=1, weight_filler=dict(type='gaussian',std=0.01),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=1), dict(lr_mult=0)],
                                     axis=2)
        n.Ax = L.InnerProduct(n.feat, num_output=dim_hidden,
                              weight_filler=dict(type='gaussian', std=0.01),
                              bias_filler=dict(type='constant', value=1),
                              axis=2)

        # Dot product operation with two layers
        n.AxAx = L.Eltwise(n.Ax, n.Ax, operation=EltwiseParameter.PROD)
        n.all_costs_preu = L.InnerProduct(n.AxAx, num_output=1, axis=2,  # all costs pre-torque penalty
                                     weight_filler=dict(type='constant', value=1),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)])
        n.all_costs = L.Eltwise(n.all_costs_preu, n.all_u, operation=EltwiseParameter.SUM, coeff=[1.0,1.0])

    if phase == 'supervised':
        n.out = L.EuclideanLoss(n.all_costs, n.cost_labels)
    elif phase == TRAIN:
        n.demo_costs, n.sample_costs = L.Slice(n.all_costs, axis=0, slice_point=demo_batch_size, ntop=2)

        # regularization
        n.costs_prev, _ = L.Slice(n.all_costs_preu, axis=1, slice_point=T-2, ntop=2)
        _, n.costs_next = L.Slice(n.all_costs_preu, axis=1, slice_point=2, ntop=2)
        _, n.costs_cur, _ = L.Slice(n.all_costs_preu, axis=1, slice_point=[1,T-1], ntop=3)
        # cur-prev
        n.slope_prev = L.Eltwise(n.costs_cur, n.costs_prev, operation=EltwiseParameter.SUM, coeff=[1,-1])
        # next-cur
        n.slope_next = L.Eltwise(n.costs_next, n.costs_cur, operation=EltwiseParameter.SUM, coeff=[1,-1])

        ### START compute normalization factor of slowness cost (std of c) ###
        # all costs is NxTx1
        n.allc_reshape = L.Reshape(n.all_costs_preu, shape=dict(dim=[-1]))
        num_costs = T*(demo_batch_size+sample_batch_size)
        n.cost_mean = L.InnerProduct(n.allc_reshape, num_output=1,
                                     weight_filler=dict(type='constant', value=-1.0/num_costs),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)], axis=0)
        n.cost_mean_tiled = L.Tile(n.cost_mean, tile_param=dict(axis=0, tiles=num_costs))
        n.cost_submean = L.Bias(n.allc_reshape, n.cost_mean_tiled, bias_param=dict(axis=0))
        n.cost_submean2 = L.Power(n.cost_submean, power=2.0)
        n.cost_var = L.InnerProduct(n.cost_submean2, num_output=1,
                                    weight_filler=dict(type='constant', value=1.0/num_costs),
                                    bias_filler=dict(type='constant', value=0),
                                    param=[dict(lr_mult=0), dict(lr_mult=0)], axis=0)
        n.cost_stdinv = L.Power(n.cost_var, power=-0.5) # 1/std(c)
        # Apply normalization
        n.next_reshaped = L.Reshape(n.slope_next, shape=dict(dim=[-1]))
        n.prev_reshaped = L.Reshape(n.slope_prev, shape=dict(dim=[-1]))
        num_cost_slopes = (T-2)*(demo_batch_size+sample_batch_size)
        n.cost_stdinv_tiled = L.Tile(n.cost_stdinv, tile_param=dict(axis=0, tiles=num_cost_slopes))
        n.slope_next_normed = L.Scale(n.next_reshaped, n.cost_stdinv_tiled, scale_param=dict(axis=0))
        n.slope_prev_normed = L.Scale(n.prev_reshaped, n.cost_stdinv_tiled, scale_param=dict(axis=0))
        ### END compute normalization factor of slowness cost (std of c) ###

        n.smooth_reg = L.EuclideanLoss(n.slope_next_normed, n.slope_prev_normed, loss_weight=smooth_reg_weight)

        n.demo_slope, _ = L.Slice(n.slope_next, axis=0, slice_point=demo_batch_size, ntop=2)
        n.demo_slope_reshape = L.Reshape(n.demo_slope, shape=dict(dim=[-1,1]))
        n.mono_reg = L.Python(n.demo_slope_reshape, loss_weight=mono_reg_weight,
                              python_param=dict(module='ioc_layers', layer='L2MonotonicLoss'))

        # n.gp_prior_reg = L.Python(n.all_costs_preu, n.net_input, n.l, loss_weight=gp_reg_weight,
        #                       python_param=dict(module='ioc_layers', layer='GaussianProcessPriors'))

        n.dummy = L.DummyData(ntop=1, shape=dict(dim=[1]), data_filler=dict(type='constant',value=0))
        # init logZ or Z to 1, only learn the bias
        # (also might be good to reduce lr on bias)
        n.logZ = L.InnerProduct(n.dummy, axis=0, num_output=1,
                             weight_filler=dict(type='constant', value=0),
                             bias_filler=dict(type='constant', value=1),
                             param=[dict(lr_mult=1), dict(lr_mult=1)])
        n.Z = L.Exp(n.logZ, base=2.6)

        # TODO - removed loss weights, changed T, batching, num samples
        # demo cond, num demos, etc.
        if ioc_loss == 'XENTGAN':
            # make multiple logZs
            n.logZs = L.InnerProduct(n.logZ, num_output=demo_batch_size+sample_batch_size, axis=0,
                                     weight_filler=dict(type='constant', value=1),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)])
            n.all_costs_sumT = L.InnerProduct(n.all_costs, num_output=1, axis=1,
                                     weight_filler=dict(type='constant', value=1),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)])


            n.demo_targets = L.DummyData(ntop=1, shape=dict(dim=[demo_batch_size, 1]), data_filler=dict(type='constant', value=1))
            n.sample_targets = L.DummyData(ntop=1, shape=dict(dim=[sample_batch_size, 1]), data_filler=dict(type='constant', value=0))
            n.all_log_iw = L.Concat(n.d_log_iw, n.s_log_iw, axis=0)
            n.all_targets = L.Concat(n.demo_targets, n.sample_targets, axis=0)

            #n.all_log_iw = L.Reshape(n.all_log_iw, shape=dict(dim=[10,1]))
            n.all_costs_sumT = L.Reshape(n.all_costs_sumT, shape=dict(dim=[10,1]))
            n.logZs = L.Reshape(n.logZs, shape=dict(dim=[10,1]))  # TODO - why is this necessary??

            # cost = 0.5*all_costs (as used to be done in the ioc loss layer
            n.all_scores = L.Eltwise(n.all_costs_sumT, n.all_log_iw, n.logZs, operation=EltwiseParameter.SUM, coeff=[-0.5,-1, -1])
            # TODO - we don't need to add demos to samples, right?
            n.out = L.SigmoidCrossEntropyLoss(n.all_scores, n.all_targets)
        elif ioc_loss == 'IOCGAN':
            n.out = L.Python(n.demo_costs, n.sample_costs, n.d_log_iw, n.s_log_iw, n.Z, loss_weight=1.0,
                             python_param=dict(module='ioc_layers',
                                               layer='IOCLossMod'))
        elif ioc_loss== 'MPF':  # MPF
            n.out = L.Python(n.demo_costs, n.sample_costs, n.d_log_iw, n.s_log_iw, loss_weight=1.0,
                             python_param=dict(module='ioc_layers',
                                               layer='LogMPFLoss'))
        else:
            n.out = L.Python(n.demo_costs, n.sample_costs, n.d_log_iw, n.s_log_iw, n.Z, loss_weight=1.0,
                             python_param=dict(module='ioc_layers',
                                               layer='IOCLoss'))

    net = n.to_proto()
    if phase == 'forward_feat':
      net.force_backward = True
    return net

def construct_fp_cost_net(num_hidden=1, dim_hidden=None, dim_input=27, T=100,
                          demo_batch_size=5, sample_batch_size=5, phase=TRAIN, ioc_loss='ICML',
                          Nq=1, smooth_reg_weight=0.0, mono_reg_weight=0.0, gp_reg_weight=0.0, image_size=[200,200]):
    """
    Construct an anonymous network (no layer names) for a quadratic cost
    function with the specified dimensionality, and return NetParameter proto.

    Note: this function is an example for how one might want to specify
    their network, versus providing a protoxt model file. It is not
    meant to be a general solution for specifying any network.

    Args:
        num_hidden: Number of hidden layers.
        dim_hidden: Dimensionality of hidden layer.
        dim_input: Dimensionality of input.
        T: time horizon
        demo_batch_size: demo batch size.
        sample_batch_size: sample batch size.
        phase: TRAIN, TEST, or 'forward_feat'
        ioc_loss: type of loss to use -- ICML, MPF, IOCGAN, XENTGAN
        Nq: number of distributions q from which the samples were drawn (only used for MPF)
    Returns:
        A NetParameter specification of the network.
    """
    from gps.algorithm.cost.config import COST_IOC_VISION

    if dim_hidden is None:
        dim_hidden = 42

    n = caffe.NetSpec()

    # Needed for Caffe to find defined python layers.
    sys.path.append('/'.join(str.split(current_path, '/')[:-1]))
    if phase == TRAIN:
        data_layer_info = json.dumps({
            'shape': [{'dim': (demo_batch_size, T, dim_input)},
                      {'dim': (demo_batch_size, T, 3, image_size[0], image_size[1])},
                      {'dim': (demo_batch_size, 1)},
                      {'dim': (sample_batch_size, T, dim_input)},
                      {'dim': (sample_batch_size, T, 3, image_size[0], image_size[1])},
                      {'dim': (sample_batch_size, 1)},
                      {'dim': (dim_input, 1)},
                      {'dim': (demo_batch_size + sample_batch_size, T, dim_input)}]
        })

        [n.demos, n.d_image, n.d_log_iw, n.samples, n.s_image, n.s_log_iw, n.l, n.total] = L.Python(
            ntop=8, python_param=dict(
                module='ioc_layers', param_str=data_layer_info,
                layer='IOCDataLayer'
            )
        )
        n.net_input = L.Concat(n.demos, n.samples, axis=0)
        n.image = L.Concat(n.d_image, n.s_image, axis=0)
        total_batch = sample_batch_size + demo_batch_size
    elif phase == TEST or phase == 'forward_feat':
        data_layer_info = json.dumps({
            'shape': [{'dim': (1, T, dim_input)},
                      {'dim': (1, T, 3, image_size[0], image_size[1])}]
        })
        n.net_input, n.image = L.Python(ntop=2,
                               python_param=dict(module='ioc_layers',
                                                 param_str=data_layer_info,
                                                 layer='IOCDataLayer'))
        total_batch = 1
    else:
        raise Exception('Unknown network phase')

    # reshape vision layers to be 4D
    n.image_reshape = L.Reshape(n.image, shape=dict(dim=[total_batch*T, 3, image_size[0], image_size[1]]))

    n.convlayer = n.image_reshape

    channel_dims = [32, 32, 32]

    for i in range(len(channel_dims)):
        stride = 2 if i==0 else 1
        n.convlayer = L.Convolution(n.convlayer, kernel_size=5, num_output=channel_dims[i],
                                    weight_filler=dict(type='gaussian', std=0.01),
                                    bias_filler=dict(type='constant', value=0),
                                    stride=stride)
        n.convlayer = L.ReLU(n.convlayer, in_place=True)

    num_fp = channel_dims[-1]

    n.convreshape = L.Reshape(n.convlayer, shape=dict(dim=[total_batch*T*num_fp, -1]))
    n.sfx = L.Softmax(n.convreshape)

    shape1 = image_size[0]/2 - 4*len(channel_dims) # stride and a few convs.
    shape2 = image_size[1]/2 - 4*len(channel_dims)
    n.sfxreshape = L.Reshape(n.sfx, shape=dict(dim=[total_batch*T, num_fp, shape1, shape2]))
    n.fp = L.InnerProduct(n.sfxreshape, num_output=2, axis=-2,
                          weight_filler=dict(type='expectation', expectation_option='xy',
                                             width=shape1, height=shape2),  # h/w might be switched
                          bias_filler=dict(type='constant', value=0),
                          param=[dict(lr_mult=0), dict(lr_mult=0)])
    n.fp = L.Reshape(n.fp, shape=dict(dim=[total_batch, T, -1]))

    n.concat = L.Concat(n.fp, n.net_input, axis=2)

    n.layer = n.concat
    for i in range(num_hidden-1):
        n.layer = L.InnerProduct(n.layer, num_output=dim_hidden,
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0),
                                 axis=2)
        n.layer = L.ReLU(n.layer, in_place=True)

    # Necessary for computing gradients
    loss_weight = 1.0 if phase == 'forward_feat' else 0.0
    n.feat = L.InnerProduct(n.layer, num_output=dim_hidden,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0),
                            axis=2, loss_weight=loss_weight)

    if phase != 'forward_feat':
        n.Ax = L.InnerProduct(n.feat, num_output=dim_hidden,
                              weight_filler=dict(type='gaussian', std=0.01),
                              bias_filler=dict(type='constant', value=0),
                              axis=2)

        # Dot product operation with two layers
        n.AxAx = L.Eltwise(n.Ax, n.Ax, operation=EltwiseParameter.PROD)
        n.all_costs = L.InnerProduct(n.AxAx, num_output=1, axis=2,
                                     weight_filler=dict(type='constant', value=1),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)])

    if phase == TRAIN:
        n.demo_costs, n.sample_costs = L.Slice(n.all_costs, axis=0, slice_point=demo_batch_size, ntop=2)

        # regularization
        n.costs_prev, _ = L.Slice(n.all_costs, axis=1, slice_point=T-2, ntop=2)
        _, n.costs_next = L.Slice(n.all_costs, axis=1, slice_point=2, ntop=2)
        _, n.costs_cur, _ = L.Slice(n.all_costs, axis=1, slice_point=[1,T-1], ntop=3)
        # cur-prev
        n.slope_prev = L.Eltwise(n.costs_cur, n.costs_prev, operation=EltwiseParameter.SUM, coeff=[1,-1])
        # next-cur
        n.slope_next = L.Eltwise(n.costs_next, n.costs_cur, operation=EltwiseParameter.SUM, coeff=[1,-1])

        ### START compute normalization factor of slowness cost (std of c) ###
        # all costs is NxTx1
        n.allc_reshape = L.Reshape(n.all_costs, shape=dict(dim=[-1]))
        num_costs = T*(demo_batch_size+sample_batch_size)
        n.cost_mean = L.InnerProduct(n.allc_reshape, num_output=1,
                                     weight_filler=dict(type='constant', value=-1.0/num_costs),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)], axis=0)
        n.cost_mean_tiled = L.Tile(n.cost_mean, tile_param=dict(axis=0, tiles=num_costs))
        n.cost_submean = L.Bias(n.allc_reshape, n.cost_mean_tiled, bias_param=dict(axis=0))
        n.cost_submean2 = L.Power(n.cost_submean, power=2.0)
        n.cost_var = L.InnerProduct(n.cost_submean2, num_output=1,
                                    weight_filler=dict(type='constant', value=1.0/num_costs),
                                    bias_filler=dict(type='constant', value=0),
                                    param=[dict(lr_mult=0), dict(lr_mult=0)], axis=0)
        n.cost_stdinv = L.Power(n.cost_var, power=-0.5) # 1/std(c)
        # Apply normalization
        n.next_reshaped = L.Reshape(n.slope_next, shape=dict(dim=[-1]))
        n.prev_reshaped = L.Reshape(n.slope_prev, shape=dict(dim=[-1]))
        num_cost_slopes = (T-2)*(demo_batch_size+sample_batch_size)
        n.cost_stdinv_tiled = L.Tile(n.cost_stdinv, tile_param=dict(axis=0, tiles=num_cost_slopes))
        n.slope_next_normed = L.Scale(n.next_reshaped, n.cost_stdinv_tiled, scale_param=dict(axis=0))
        n.slope_prev_normed = L.Scale(n.prev_reshaped, n.cost_stdinv_tiled, scale_param=dict(axis=0))
        ### END compute normalization factor of slowness cost (std of c) ###


        n.smooth_reg = L.EuclideanLoss(n.slope_next_normed, n.slope_prev_normed, loss_weight=smooth_reg_weight)

        n.demo_slope, _ = L.Slice(n.slope_next, axis=0, slice_point=demo_batch_size, ntop=2)
        n.demo_slope_reshape = L.Reshape(n.demo_slope, shape=dict(dim=[-1,1]))
        n.mono_reg = L.Python(n.demo_slope_reshape, loss_weight=mono_reg_weight,
                              python_param=dict(module='ioc_layers', layer='L2MonotonicLoss'))

        #n.gp_prior_reg = L.Python(n.total, n.all_costs, n.l, loss_weight=gp_reg_weight,
        #              python_param=dict(module='ioc_layers', layer='GaussianProcessPriors'))

        n.dummy = L.DummyData(ntop=1, shape=dict(dim=[1]), data_filler=dict(type='constant',value=0))
        # init logZ or Z to 1, only learn the bias
        # (also might be good to reduce lr on bias)
        n.logZ = L.InnerProduct(n.dummy, axis=0, num_output=1,
                             weight_filler=dict(type='constant', value=0),
                             bias_filler=dict(type='constant', value=1),
                             param=[dict(lr_mult=1), dict(lr_mult=1)])
        n.Z = L.Exp(n.logZ, base=2.6)

        # TODO - removed loss weights, changed T, batching, num samples
        # demo cond, num demos, etc.
        if ioc_loss == 'XENTGAN':
            # make multiple logZs
            n.logZs = L.InnerProduct(n.logZ, num_output=demo_batch_size+sample_batch_size, axis=0,
                                     weight_filler=dict(type='constant', value=1),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)])
            n.all_costs_sumT = L.InnerProduct(n.all_costs, num_output=1, axis=1,
                                     weight_filler=dict(type='constant', value=1),
                                     bias_filler=dict(type='constant', value=0),
                                     param=[dict(lr_mult=0), dict(lr_mult=0)])


            n.demo_targets = L.DummyData(ntop=1, shape=dict(dim=[demo_batch_size, 1]), data_filler=dict(type='constant', value=1))
            n.sample_targets = L.DummyData(ntop=1, shape=dict(dim=[sample_batch_size, 1]), data_filler=dict(type='constant', value=0))
            n.all_log_iw = L.Concat(n.d_log_iw, n.s_log_iw, axis=0)
            n.all_targets = L.Concat(n.demo_targets, n.sample_targets, axis=0)

            #n.all_log_iw = L.Reshape(n.all_log_iw, shape=dict(dim=[10,1]))
            n.all_costs_sumT = L.Reshape(n.all_costs_sumT, shape=dict(dim=[10,1]))
            n.logZs = L.Reshape(n.logZs, shape=dict(dim=[10,1]))  # TODO - why is this necessary??

            # cost = 0.5*all_costs (as used to be done in the ioc loss layer
            n.all_scores = L.Eltwise(n.all_costs_sumT, n.all_log_iw, n.logZs, operation=EltwiseParameter.SUM, coeff=[-0.5,-1, -1])
            # TODO - we don't need to add demos to samples, right?
            n.out = L.SigmoidCrossEntropyLoss(n.all_scores, n.all_targets)
        elif ioc_loss == 'IOCGAN':
            n.out = L.Python(n.demo_costs, n.sample_costs, n.d_log_iw, n.s_log_iw, n.Z, loss_weight=1.0,
                             python_param=dict(module='ioc_layers',
                                               layer='IOCLossMod'))
        elif ioc_loss== 'MPF':  # MPF
            n.out = L.Python(n.demo_costs, n.sample_costs, n.d_log_iw, n.s_log_iw, loss_weight=1.0,
                             python_param=dict(module='ioc_layers',
                                               layer='LogMPFLoss'))
        else:
            n.out = L.Python(n.demo_costs, n.sample_costs, n.d_log_iw, n.s_log_iw, n.Z, loss_weight=1.0,
                             python_param=dict(module='ioc_layers',
                                               layer='IOCLoss'))

    net = n.to_proto()
    if phase == 'forward_feat':
      net.force_backward = True
    return net

