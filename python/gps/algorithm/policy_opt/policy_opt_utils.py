""" This file defines utility functions for policy optimization. """
import json
import sys

from caffe import layers as L
from caffe.proto.caffe_pb2 import TRAIN, TEST

from gps.algorithm.policy_opt import __file__ as policy_opt_path


def construct_fc_network(n_layers=3, dim_hidden=None, dim_input=27,
                         dim_output=7, batch_size=25, phase=TRAIN):
    """
    Construct an anonymous network (no layer names) with the specified
    number of inner product layers, and return NetParameter protobuffer.

    Note: this function is an example for how one might want to specify
    their network, versus providing a protoxt model file. It is not
    meant to be a general solution for specifying any network, as there
    are many, many possible networks one can specify.

    Args:
        n_layers: Number of fully connected layers (including output).
        dim_hidden: Dimensionality of each hidden layer.
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        phase: TRAIN, TEST, or 'deploy'
    Returns:
        A NetParameter specification of the network.
    """
    if dim_hidden is None:
        dim_hidden = (n_layers - 1) * [42]

    # Needed for Caffe to find defined python layers.
    sys.path.append('/'.join(str.split(policy_opt_path, '/')[:-1]))

    if phase == TRAIN:
        data_layer_info = json.dumps({
            'shape': [{'dim': (batch_size, dim_input)},
                      {'dim': (batch_size, dim_output)},
                      {'dim': (batch_size, dim_output, dim_output)}]
        })

        [net_input, action, precision] = L.Python(
            ntop=3, python_param=dict(
                module='policy_layers', param_str=data_layer_info,
                layer='PolicyDataLayer'
            )
        )
    elif phase == TEST:
        data_layer_info = json.dumps({
            'shape': [{'dim': (batch_size, dim_input)}]
        })
        net_input = L.Python(ntop=1,
                             python_param=dict(module='policy_layers',
                                               param_str=data_layer_info,
                                               layer='PolicyDataLayer'))
    elif phase == 'deploy':
        # This is the network that runs on the robot. This data layer
        # will be bypassed.
        net_input = L.DummyData(ntop=1,
                                shape=[dict(dim=[batch_size, dim_input])])
    else:
        raise Exception('Unknown network phase')

    cur_top = net_input
    dim_hidden.append(dim_output)
    for i in range(n_layers):
        cur_top = L.InnerProduct(cur_top, num_output=dim_hidden[i],
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0))
        # Add nonlinearity to all hidden layers.
        if i < n_layers - 1:
            cur_top = L.ReLU(cur_top, in_place=True)

    if phase == TRAIN:
        out = L.Python(cur_top, action, precision, loss_weight=1.0,
                       python_param=dict(module='policy_layers',
                                         layer='WeightedEuclideanLoss'))
    else:
        out = cur_top

    return out.to_proto()
