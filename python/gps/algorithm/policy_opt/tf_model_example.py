""" This file provides an example tensorflow network used to define a policy. """

import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np


def init_weights(shape, name=None):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))


def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))


def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.batch_matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def euclidean_loss_layer(a, b, precision, batch_size):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    scale_factor = tf.constant(2*batch_size, dtype='float')
    uP = batched_matrix_vector_multiply(a-b, precision)
    uPu = tf.reduce_sum(uP*(a-b))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor


def get_input_layer(dim_input, dim_output):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    return net_input, action, precision


def get_mlp_layers(mlp_input, number_layers, dimension_hidden):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b_' + str(layer_step))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top, weights, biases


def get_loss_layer(mlp_out, action, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    return euclidean_loss_layer(a=action, b=mlp_out, precision=precision, batch_size=batch_size)


def tf_network(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    Specifying a fully-connected network in TensorFlow.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a TfMap object used to serialize, inputs, outputs, and loss.
    """
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else network_config['dim_hidden']
    dim_hidden.append(dim_output)

    nn_input, action, precision = get_input_layer(dim_input, dim_output)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    fc_vars = weights_FC + biases_FC
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out]), fc_vars, []


def multi_modal_network(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    An example a network in tf that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        A tfMap object that stores inputs, outputs, and scalar loss.
    """
    n_layers = 2
    layer_size = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    dim_hidden.append(dim_output)
    pool_size = 2
    filter_size = 3

    # List of indices for state (vector) data and image (tensor) data in observation.
    x_idx, img_idx, i = [], [], 0
    for sensor in network_config['obs_include']:
        dim = network_config['sensor_dims'][sensor]
        if sensor in network_config['obs_image_data']:
            img_idx = img_idx + list(range(i, i+dim))
        else:
            x_idx = x_idx + list(range(i, i+dim))
        i += dim

    nn_input, action, precision = get_input_layer(dim_input, dim_output)

    state_input = nn_input[:, 0:x_idx[-1]+1]
    image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]

    # image goes through 2 convnet layers
    num_filters = network_config['num_filters']

    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    image_input = tf.reshape(image_input, [-1, im_width, im_height, num_channels])

    # we pool twice, each time reducing the image size by a factor of 2.
    conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
    first_dense_size = conv_out_size + len(x_idx)

    # Store layers weight & bias
    weights = {
        'wc1': get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size)), # 5x5 conv, 1 input, 32 outputs
        'wc2': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)), # 5x5 conv, 32 inputs, 64 outputs
    }

    biases = {
        'bc1': init_bias([num_filters[0]]),
        'bc2': init_bias([num_filters[1]]),
    }

    conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'])

    conv_layer_0 = max_pool(conv_layer_0, k=pool_size)

    conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

    conv_layer_1 = max_pool(conv_layer_1, k=pool_size)

    conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])

    fc_input = tf.concat(concat_dim=1, values=[conv_out_flat, state_input])

    fc_output, _, _ = get_mlp_layers(fc_input, n_layers, dim_hidden)

    loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
    return TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss])

def multi_modal_network_fp(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    An example a network in tf that has both state and image inputs, with the feature
    point architecture (spatial softmax + expectation).
    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        A tfMap object that stores inputs, outputs, and scalar loss.
    """
    n_layers = 3
    layer_size = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    dim_hidden.append(dim_output)
    pool_size = 2
    filter_size = 5

    # List of indices for state (vector) data and image (tensor) data in observation.
    x_idx, img_idx, i = [], [], 0
    for sensor in network_config['obs_include']:
        dim = network_config['sensor_dims'][sensor]
        if sensor in network_config['obs_image_data']:
            img_idx = img_idx + list(range(i, i+dim))
        else:
            x_idx = x_idx + list(range(i, i+dim))
        i += dim

    nn_input, action, precision = get_input_layer(dim_input, dim_output)

    state_input = nn_input[:, 0:x_idx[-1]+1]
    image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]

    # image goes through 3 convnet layers
    num_filters = network_config['num_filters']

    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    image_input = tf.reshape(image_input, [-1, num_channels, im_width, im_height])
    image_input = tf.transpose(image_input, perm=[0,3,2,1])

    # we pool twice, each time reducing the image size by a factor of 2.
    conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
    first_dense_size = conv_out_size + len(x_idx)

    # Store layers weight & bias
    with tf.variable_scope('conv_params'):
        weights = {
            'wc1': init_weights([filter_size, filter_size, num_channels, num_filters[0]], name='wc1'), # 5x5 conv, 1 input, 32 outputs
            'wc2': init_weights([filter_size, filter_size, num_filters[0], num_filters[1]], name='wc2'), # 5x5 conv, 32 inputs, 64 outputs
            'wc3': init_weights([filter_size, filter_size, num_filters[1], num_filters[2]], name='wc3'), # 5x5 conv, 32 inputs, 64 outputs
        }

        biases = {
            'bc1': init_bias([num_filters[0]], name='bc1'),
            'bc2': init_bias([num_filters[1]], name='bc2'),
            'bc3': init_bias([num_filters[2]], name='bc3'),
        }

    conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'], strides=[1,2,2,1])
    conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])
    conv_layer_2 = conv2d(img=conv_layer_1, w=weights['wc3'], b=biases['bc3'])

    _, num_rows, num_cols, num_fp = conv_layer_2.get_shape()
    num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    x_map = np.empty([num_rows, num_cols], np.float32)
    y_map = np.empty([num_rows, num_cols], np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            x_map[i, j] = (i - num_rows / 2.0) / num_rows
            y_map[i, j] = (j - num_cols / 2.0) / num_cols

    x_map = tf.convert_to_tensor(x_map)
    y_map = tf.convert_to_tensor(y_map)

    x_map = tf.reshape(x_map, [num_rows * num_cols])
    y_map = tf.reshape(y_map, [num_rows * num_cols])

    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(conv_layer_2, [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)

    fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)

    fp = tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])

    fc_input = tf.concat(concat_dim=1, values=[fp, state_input])

    fc_output, weights_FC, biases_FC = get_mlp_layers(fc_input, n_layers, dim_hidden)
    fc_vars = weights_FC + biases_FC

    loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
    nnet = TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss], fp=fp)
    last_conv_vars = fc_input

    return nnet, fc_vars, last_conv_vars


def conv2d(img, w, b, strides=[1, 1, 1, 1]):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=strides, padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def get_xavier_weights(filter_shape, poolsize=(2, 2)):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32))


