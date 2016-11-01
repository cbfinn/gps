import numpy as np
import tensorflow as tf


def assert_shape(tensor, shape):
    assert tensor.get_shape().is_compatible_with(shape), "Shape mismatch: %s vs %s" % (str(tensor.get_shape()), shape)


def logsumexp(x, reduction_indices=None):
    """ Compute numerically stable logsumexp """
    max_val = tf.reduce_max(x)
    exp = tf.exp(x-max_val)
    _partition = tf.reduce_sum(exp, reduction_indices=reduction_indices)
    _log = tf.log(_partition)+max_val
    return _log


def safe_get(name, *args, **kwargs):
    """ Same as tf.get_variable, except flips on reuse_variables automatically """
    try:
        return tf.get_variable(name, *args, **kwargs)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(name, *args, **kwargs)


def find_variable(name):
    """ Find a trainable variable in a graph by its name (not including scope)

    Example:
    >>> find_variable('Wconv1')
    <tensorflow tensor>
    """
    varnames = tf.trainable_variables()
    matches = [varname for varname in varnames if varname.name.endswith(name+':0')]

    if len(matches)>1:
        raise ValueError('More than one variable with name %s. %s' % (name, [match.name for match in matches]))
    if len(matches) == 0:
        raise ValueError('No variables found with name %s. List: %s' % (name, [var.name for var in varnames]))
    return matches[0]


def jacobian(y, x):
    """Compute derivative of y (vector) w.r.t. x (another vector)"""
    dY = y.get_shape()[0].value
    dX = x.get_shape()[0].value

    deriv_list = []
    for idx_y in range(dY):
        grad = tf.gradients(y[idx_y], x)[0]
        deriv_list.append(grad)
    jac = tf.pack(deriv_list)
    assert_shape(jac, [dY, dX])
    return jac

def multimodal_nn_cost_net_tf(num_hidden=3, dim_hidden=42, dim_input=27, T=100,
                             demo_batch_size=5, sample_batch_size=5, phase=None, ioc_loss='ICML',
                             Nq=1, smooth_reg_weight=0.0, mono_reg_weight=0.0, gp_reg_weight=0.0,
                             multi_obj_supervised_wt=1.0, learn_wu=False, x_idx=None, img_idx=None,
                              num_filters=[15,15,15]):
    """ Construct cost net with images and robot config.
    Args:
        ...
        x_idx is required, and should indicate the indices corresponding to the robot config
        img_idx is required, and should indicate the indices corresponding to the imagej
    """

    inputs = {}
    inputs['demo_obs'] = demo_obs = tf.placeholder(tf.float32, shape=(demo_batch_size, T, dim_input))
    inputs['demo_torque_norm'] = demo_torque_norm = tf.placeholder(tf.float32, shape=(demo_batch_size, T, 1))
    inputs['demo_iw'] = demo_imp_weight = tf.placeholder(tf.float32, shape=(demo_batch_size, 1))
    inputs['sample_obs'] = sample_obs = tf.placeholder(tf.float32, shape=(sample_batch_size, T, dim_input))
    inputs['sample_torque_norm'] = sample_torque_norm = tf.placeholder(tf.float32, shape=(sample_batch_size, T, 1))
    inputs['sample_iw'] = sample_imp_weight = tf.placeholder(tf.float32, shape=(sample_batch_size, 1))
    sup_batch_size = sample_batch_size+demo_batch_size
    inputs['sup_obs'] = sup_obs = tf.placeholder(tf.float32, shape=(sup_batch_size, T, dim_input))
    inputs['sup_torque_norm'] = sup_torque_norm = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))
    inputs['sup_cost_labels'] = sup_cost_labels = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))

    # Inputs for single eval test runs
    inputs['test_obs'] = test_obs = tf.placeholder(tf.float32, shape=(T, dim_input), name='test_obs')
    inputs['test_torque_norm'] = test_torque_norm = tf.placeholder(tf.float32, shape=(T, 1), name='test_torque_u')

    inputs['test_obs_single'] = test_obs_single = tf.placeholder(tf.float32, shape=(dim_input), name='test_obs_single')
    inputs['test_torque_single'] = test_torque_single = tf.placeholder(tf.float32, shape=(1), name='test_torque_u_single')

    x_idx = tf.constant(x_idx)
    img_idx = tf.constant(img_idx)


    _, test_imgfeat, _, test_cost  = nn_vis_forward(test_obs, test_torque_norm, num_hidden=num_hidden, learn_wu=learn_wu, dim_hidden=dim_hidden, x_idx=x_idx, img_idx=img_idx, num_filters=num_filters)
    demo_cost_preu, _, _, demo_costs = nn_vis_forward(demo_obs, demo_torque_norm, num_hidden=num_hidden, learn_wu=learn_wu, dim_hidden=dim_hidden, x_idx=x_idx, img_idx=img_idx, num_filters=num_filters)
    sample_cost_preu, _, _, sample_costs = nn_vis_forward(sample_obs, sample_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden, x_idx=x_idx, img_idx=img_idx, num_filters=num_filters)
    sup_cost_preu, _, _, sup_costs = nn_vis_forward(sup_obs, sup_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden, x_idx=x_idx, img_idx=img_idx, num_filters=num_filters)

    # Build a differentiable test cost by feeding each timestep individually
    test_obs_single = tf.expand_dims(test_obs_single, 0)
    test_torque_single = tf.expand_dims(test_torque_single, 0)

    test_cost_single_preu, test_imgfeat_single, test_feat_single, _ = nn_vis_forward(test_obs_single, test_torque_single, num_hidden=num_hidden, dim_hidden=dim_hidden, learn_wu=learn_wu, x_idx=x_idx, img_idx=img_idx)
    test_cost_single = tf.squeeze(test_cost_single_preu)

    sup_loss = tf.nn.l2_loss(sup_costs - sup_cost_labels)*multi_obj_supervised_wt

    demo_sample_preu = tf.concat(0, [demo_cost_preu, sample_cost_preu])
    sample_demo_size = sample_batch_size+demo_batch_size
    assert_shape(demo_sample_preu, [sample_demo_size, T, 1])
    costs_prev = tf.slice(demo_sample_preu, begin=[0, 0,0], size=[sample_demo_size, T-2, -1])
    costs_next = tf.slice(demo_sample_preu, begin=[0, 2,0], size=[sample_demo_size, T-2, -1])
    costs_cur = tf.slice(demo_sample_preu, begin=[0, 1,0], size=[sample_demo_size, T-2, -1])
    # cur-prev
    slope_prev = costs_cur-costs_prev
    # next-cur
    slope_next = costs_next-costs_cur

    if smooth_reg_weight > 0:
        # regularization
        """
        """
        raise NotImplementedError("Smoothness reg not implemented")

    if mono_reg_weight > 0:
        demo_slope = tf.slice(slope_next, begin=[0,0,0], size=[demo_batch_size, -1, -1])
        slope_reshape = tf.reshape(demo_slope, shape=[-1,1])
        mono_reg = l2_mono_loss(slope_reshape)*mono_reg_weight
    else:
        mono_reg = 0

    # init logZ or Z to 1, only learn the bias
    # (also might be good to reduce lr on bias)
    logZ = safe_get('Wdummy', initializer=tf.ones(1))
    Z = tf.exp(logZ) # TODO: What does this do?

    # TODO - removed loss weights, changed T, batching, num samples
    # demo cond, num demos, etc.
    ioc_loss = icml_loss(demo_costs, sample_costs, demo_imp_weight, sample_imp_weight, Z)
    ioc_loss += mono_reg

    outputs = {
        'multiobj_loss': sup_loss+ioc_loss,
        'sup_loss': sup_loss,
        'ioc_loss': ioc_loss,
        'test_loss': test_cost,
        'test_imgfeat': test_imgfeat,
        'test_loss_single': test_cost_single,
        'test_imgfeat_single': test_imgfeat_single,
        'test_feat_single': test_feat_single,
    }
    return inputs, outputs



def construct_nn_cost_net_tf(num_hidden=3, dim_hidden=42, dim_input=27, T=100,
                             demo_batch_size=5, sample_batch_size=5, phase=None, ioc_loss='ICML',
                             Nq=1, smooth_reg_weight=0.0, mono_reg_weight=0.0, gp_reg_weight=0.0,
                             multi_obj_supervised_wt=1.0, learn_wu=False):

    inputs = {}
    inputs['demo_obs'] = demo_obs = tf.placeholder(tf.float32, shape=(demo_batch_size, T, dim_input))
    inputs['demo_torque_norm'] = demo_torque_norm = tf.placeholder(tf.float32, shape=(demo_batch_size, T, 1))
    inputs['demo_iw'] = demo_imp_weight = tf.placeholder(tf.float32, shape=(demo_batch_size, 1))
    inputs['sample_obs'] = sample_obs = tf.placeholder(tf.float32, shape=(sample_batch_size, T, dim_input))
    inputs['sample_torque_norm'] = sample_torque_norm = tf.placeholder(tf.float32, shape=(sample_batch_size, T, 1))
    inputs['sample_iw'] = sample_imp_weight = tf.placeholder(tf.float32, shape=(sample_batch_size, 1))
    sup_batch_size = sample_batch_size+demo_batch_size
    inputs['sup_obs'] = sup_obs = tf.placeholder(tf.float32, shape=(sup_batch_size, T, dim_input))
    inputs['sup_torque_norm'] = sup_torque_norm = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))
    inputs['sup_cost_labels'] = sup_cost_labels = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))

    # Inputs for single eval test runs
    inputs['test_obs'] = test_obs = tf.placeholder(tf.float32, shape=(T, dim_input), name='test_obs')
    inputs['test_torque_norm'] = test_torque_norm = tf.placeholder(tf.float32, shape=(T, 1), name='test_torque_u')

    inputs['test_obs_single'] = test_obs_single = tf.placeholder(tf.float32, shape=(dim_input), name='test_obs_single')
    inputs['test_torque_single'] = test_torque_single = tf.placeholder(tf.float32, shape=(1), name='test_torque_u_single')

    _, test_cost  = nn_forward(test_obs, test_torque_norm, num_hidden=num_hidden, learn_wu=learn_wu, dim_hidden=dim_hidden)
    demo_cost_preu, demo_costs = nn_forward(demo_obs, demo_torque_norm, num_hidden=num_hidden, learn_wu=learn_wu, dim_hidden=dim_hidden)
    sample_cost_preu, sample_costs = nn_forward(sample_obs, sample_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden)
    sup_cost_preu, sup_costs = nn_forward(sup_obs, sup_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden)

    # Build a differentiable test cost by feeding each timestep individually
    test_obs_single = tf.expand_dims(test_obs_single, 0)
    test_torque_single = tf.expand_dims(test_torque_single, 0)
    test_feat_single = compute_feats(test_obs_single, num_hidden=num_hidden, dim_hidden=dim_hidden)
    test_cost_single_preu, _ = nn_forward(test_obs_single, test_torque_single, num_hidden=num_hidden, dim_hidden=dim_hidden, learn_wu=learn_wu)
    test_cost_single = tf.squeeze(test_cost_single_preu)

    sup_loss = tf.nn.l2_loss(sup_costs - sup_cost_labels)*multi_obj_supervised_wt


    demo_sample_preu = tf.concat(0, [demo_cost_preu, sample_cost_preu])
    sample_demo_size = sample_batch_size+demo_batch_size
    assert_shape(demo_sample_preu, [sample_demo_size, T, 1])
    costs_prev = tf.slice(demo_sample_preu, begin=[0, 0,0], size=[sample_demo_size, T-2, -1])
    costs_next = tf.slice(demo_sample_preu, begin=[0, 2,0], size=[sample_demo_size, T-2, -1])
    costs_cur = tf.slice(demo_sample_preu, begin=[0, 1,0], size=[sample_demo_size, T-2, -1])
    # cur-prev
    slope_prev = costs_cur-costs_prev
    # next-cur
    slope_next = costs_next-costs_cur

    if smooth_reg_weight > 0:
        # regularization
        """
        """
        raise NotImplementedError("Smoothness reg not implemented")

    if mono_reg_weight > 0:
        demo_slope = tf.slice(slope_next, begin=[0,0,0], size=[demo_batch_size, -1, -1])
        slope_reshape = tf.reshape(demo_slope, shape=[-1,1])
        mono_reg = l2_mono_loss(slope_reshape)*mono_reg_weight
    else:
        mono_reg = 0

    # init logZ or Z to 1, only learn the bias
    # (also might be good to reduce lr on bias)
    logZ = safe_get('Wdummy', initializer=tf.ones(1))
    Z = tf.exp(logZ) # TODO: What does this do?

    # TODO - removed loss weights, changed T, batching, num samples
    # demo cond, num demos, etc.
    ioc_loss = icml_loss(demo_costs, sample_costs, demo_imp_weight, sample_imp_weight, Z)
    ioc_loss += mono_reg

    outputs = {
        'multiobj_loss': sup_loss+ioc_loss,
        'sup_loss': sup_loss,
        'ioc_loss': ioc_loss,
        'test_loss': test_cost,
        'test_loss_single': test_cost_single,
        'test_feat_single': test_feat_single,
    }
    return inputs, outputs


def compute_feats(net_input, num_hidden=1, dim_hidden=42):
    len_shape = len(net_input.get_shape())
    if  len_shape == 3:
        batch_size, T, dinput = net_input.get_shape()
    elif len_shape == 2:
        T, dinput = net_input.get_shape()

    # Reshape into 2D matrix for matmuls
    net_input = tf.reshape(net_input, [-1, dinput.value])
    with tf.variable_scope('cost_forward'):
        layer = net_input
        for i in range(num_hidden-1):
            with tf.variable_scope('layer_%d' % i):
                W = safe_get('W', (dim_hidden, layer.get_shape()[1].value))
                b = safe_get('b', (dim_hidden))
                layer = tf.nn.relu(tf.matmul(layer, W, transpose_b=True, name='mul_layer'+str(i)) + b)

        Wfeat = safe_get('Wfeat', (dim_hidden, layer.get_shape()[1].value))
        bfeat = safe_get('bfeat', (dim_hidden))
        feat = tf.matmul(layer, Wfeat, transpose_b=True, name='mul_feat')+bfeat

    if len_shape == 3:
        feat = tf.reshape(feat, [batch_size.value, T.value, dim_hidden])
    else:
        feat = tf.reshape(feat, [-1, dim_hidden])

    return feat


def nn_forward(net_input, u_input, num_hidden=1, dim_hidden=42, wu=1e-3, learn_wu=False):
    # Reshape into 2D matrix for matmuls
    u_input = tf.reshape(u_input, [-1, 1])

    feat = compute_feats(net_input, num_hidden=num_hidden, dim_hidden=dim_hidden)
    feat = tf.reshape(feat, [-1, dim_hidden])

    with tf.variable_scope('cost_forward'):
        A = safe_get('Acost', shape=(dim_hidden, dim_hidden))
        b = safe_get('bcost', shape=(dim_hidden))
        Ax = tf.matmul(feat, A, transpose_b=True)+b
        AxAx = Ax*Ax

        # Calculate torque penalty
        u_penalty = safe_get('wu', initializer=tf.constant(1.0), trainable=learn_wu)
        assert_shape(u_penalty, [])
        u_cost = u_input*u_penalty*wu

    # Reshape result back into batches
    input_shape = net_input.get_shape()
    if len(input_shape) == 3:
        batch_size, T, dinput = input_shape
        batch_size, T = batch_size.value, T.value
        AxAx = tf.reshape(AxAx, [batch_size, T, dim_hidden])
        u_cost = tf.reshape(u_cost, [batch_size, T, 1])
    elif len(input_shape) == 2:
        AxAx = tf.reshape(AxAx, [-1, dim_hidden])
        u_cost = tf.reshape(u_cost, [-1, 1])
    all_costs_preu = tf.reduce_sum(AxAx, reduction_indices=[-1], keep_dims=True)
    all_costs = all_costs_preu + u_cost
    return all_costs_preu, all_costs

def init_weights(shape, name=None):
    return safe_get(name, initializer=tf.random_normal(shape, stddev=0.01))

def init_bias(shape, name=None):
    return safe_get(name, initializer=tf.zeros(shape, dtype='float'))

def conv2d(img, w, b, strides=[1, 1, 1, 1]):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=strides, padding='SAME'), b))

def compute_image_feats(img_input, num_filters=[15,15,15]):
    filter_size = 5
    num_channels=3
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

    conv_layer_0 = conv2d(img=img_input, w=weights['wc1'], b=biases['bc1'], strides=[1,2,2,1])
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
    return fp


def nn_vis_forward(net_input, u_input, num_hidden=1, dim_hidden=42, wu=1e-3, learn_wu=False, x_idx=None, img_idx=None,
                   num_filters=[15,15,15]):

    net_input = tf.transpose(net_input)
    x_input = tf.transpose(tf.gather(net_input, x_idx))
    img_input = tf.transpose(tf.gather(net_input, img_idx))
    net_input = tf.transpose(net_input)

    # TODO don't hard code this.
    num_channels=3; im_width = 80; im_height = 64;
    img_input = tf.reshape(img_input, [-1, num_channels, im_width, im_height])
    img_input = tf.transpose(img_input, perm=[0,3,2,1])

    img_feats = compute_image_feats(img_input, num_filters=num_filters)
    if len(x_input.get_shape()) == 3:
        img_feats = tf.reshape(img_feats, [int(x_input.get_shape()[0]), int(x_input.get_shape()[1]), -1])

    dim = len(x_input.get_shape()) - 1
    # NOTE - this assumes that the image features are the last thing in the state.
    all_feat = tf.concat(dim, [x_input, img_feats])

    if all_feat.get_shape()[0] == 1:
        # hack to make this the size we want and be able to differentiate w.r.t. feat.
        return_imgfeat = tf.reshape(all_feat, [-1])
        all_feat = tf.reshape(return_imgfeat, [1, -1])  # entire state, used for jacobians
    else:
        return_imgfeat = img_feats # used for updating forward pass of image features

    # Reshape into 2D matrix for matmuls
    u_input = tf.reshape(u_input, [-1, 1])

    return_feat = compute_feats(all_feat, num_hidden=num_hidden, dim_hidden=dim_hidden)
    feat = tf.reshape(return_feat, [-1, dim_hidden])

    with tf.variable_scope('cost_forward'):
        A = safe_get('Acost', shape=(dim_hidden, dim_hidden))
        b = safe_get('bcost', shape=(dim_hidden))
        Ax = tf.matmul(feat, A, transpose_b=True)+b
        AxAx = Ax*Ax

        # Calculate torque penalty
        u_penalty = safe_get('wu', initializer=tf.constant(1.0), trainable=learn_wu)
        assert_shape(u_penalty, [])
        u_cost = u_input*u_penalty*wu

    # Reshape result back into batches
    input_shape = net_input.get_shape()
    if len(input_shape) == 3:
        batch_size, T, dinput = input_shape
        batch_size, T = batch_size.value, T.value
        AxAx = tf.reshape(AxAx, [batch_size, T, dim_hidden])
        u_cost = tf.reshape(u_cost, [batch_size, T, 1])
    elif len(input_shape) == 2:
        AxAx = tf.reshape(AxAx, [-1, dim_hidden])
        u_cost = tf.reshape(u_cost, [-1, 1])
    all_costs_preu = tf.reduce_sum(AxAx, reduction_indices=[-1], keep_dims=True)
    all_costs = all_costs_preu + u_cost
    return all_costs_preu, return_imgfeat, return_feat, all_costs


def icml_loss(demo_costs, sample_costs, d_log_iw, s_log_iw, Z):
    num_demos, T, _ = demo_costs.get_shape()
    num_samples, T, _ = sample_costs.get_shape()

    # Sum over time and compute max value for safe logsum.
    #for i in xrange(num_demos):
    #    dc[i] = 0.5 * tf.reduce_sum(demo_costs[i])
    #    loss += dc[i]
    #    # Add importance weight to demo feature count. Will be negated.
    #    dc[i] += d_log_iw[i]
    demo_reduced = 0.5*tf.reduce_sum(demo_costs, reduction_indices=[1,2])
    dc = demo_reduced + tf.reduce_sum(d_log_iw, reduction_indices=[1])
    assert_shape(dc, [num_demos])

    #for i in xrange(num_samples):
    #    sc[i] = 0.5 * tf.reduce_sum(sample_costs[i])
    #    # Add importance weight to sample feature count. Will be negated.
    #    sc[i] += s_log_iw[i]
    sc = 0.5*tf.reduce_sum(sample_costs, reduction_indices=[1,2])+tf.reduce_sum(s_log_iw, reduction_indices=[1])
    assert_shape(sc, [num_samples])

    dc_sc = tf.concat(0, [-dc, -sc])

    loss = tf.reduce_mean(demo_reduced)

    # Concatenate demos and samples to approximate partition function
    partition_samples = tf.concat(0, [demo_costs, sample_costs])
    partition_iw = tf.concat(0, [d_log_iw, s_log_iw])
    partition = 0.5*tf.reduce_sum(partition_samples, reduction_indices=[1,2])\
                +tf.reduce_sum(partition_iw, reduction_indices=[1])
    assert_shape(partition, [num_samples+num_demos])
    loss += logsumexp(-partition, reduction_indices=[0])

    assert_shape(loss, [])
    return loss

def l2_mono_loss(slope):
    #_temp = np.zeros(slope.shape[0])
    offset = 1.0
    bottom_data = slope

    #for i in range(batch_size):
    #    _temp[i] = np.maximum(0.0, bottom_data[i] + offset)
    _temp = tf.nn.relu(bottom_data+offset)
    loss = tf.nn.l2_loss(_temp)# _temp*_temp).sum() / batch_size
    return loss


def main():
    inputs, outputs= construct_nn_cost_net_tf(mono_reg_weight=1.0)
    Y, X = outputs['test_loss_single'], inputs['test_obs_single']
    dldx =  tf.gradients(Y, X)[0]
    print dldx
    print jacobian(dldx, X)
    print dfdx


if __name__ == "__main__":
    main()
