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


def jacobian(y, x):
    dY = y.get_shape()[0].value
    dX = x.get_shape()[0].value

    deriv_list = []
    for idx_y in range(dY):
        grad = tf.gradients(y[idx_y], x)[0]
        deriv_list.append(grad)
    jac = tf.pack(deriv_list)
    assert_shape(jac, [dY, dX])
    return jac


def construct_nn_cost_net_tf(num_hidden=3, dim_hidden=42, dim_input=27, T=100,
                             demo_batch_size=5, sample_batch_size=5, phase=None, ioc_loss='ICML',
                             Nq=1, smooth_reg_weight=0.0, mono_reg_weight=0.0, multi_obj_supervised_wt=1.0, learn_wu=False):

    inputs = {}
    inputs['demo_obs'] = demo_obs = tf.placeholder(tf.float32, shape=(demo_batch_size, T, dim_input))
    inputs['demo_torque_norm'] = demo_torque_norm = tf.placeholder(tf.float32, shape=(demo_batch_size, T, 1))
    inputs['demo_iw'] = demo_imp_weight = tf.placeholder(tf.float32, shape=(demo_batch_size, 1))
    inputs['sample_obs'] = sample_obs = tf.placeholder(tf.float32, shape=(sample_batch_size, T, dim_input))
    inputs['sample_torque_norm'] = sample_torque_norm = tf.placeholder(tf.float32, shape=(sample_batch_size, T, 1))
    inputs['sample_iw'] = sample_imp_weight = tf.placeholder(tf.float32, shape=(sample_batch_size, 1))
    sup_batch_size = sample_batch_size+demo_batch_size
    inputs['sub_obs'] = sup_obs = tf.placeholder(tf.float32, shape=(sup_batch_size, T, dim_input))
    inputs['sup_torque_norm'] = sup_torque_norm = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))
    inputs['sup_cost_labels'] = sup_cost_labels = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))

    # Inputs for single eval test runs
    inputs['test_obs'] = test_obs = tf.placeholder(tf.float32, shape=(1, T, dim_input))
    inputs['test_torque_norm'] = test_torque_norm = tf.placeholder(tf.float32, shape=(1, T, 1))

    inputs['test_obs_single'] = test_obs_single = tf.placeholder(tf.float32, shape=(dim_input))
    inputs['test_torque_single'] = test_torque_single = tf.placeholder(tf.float32, shape=(1))

    with tf.variable_scope('cost_ioc_nn'):
        test_feats = compute_feats(test_obs, num_hidden=num_hidden, dim_hidden=dim_hidden)
        _, test_cost  = nn_forward(test_obs, test_torque_norm, num_hidden=num_hidden, learn_wu=learn_wu, dim_hidden=dim_hidden)
        demo_cost_preu, demo_costs = nn_forward(demo_obs, demo_torque_norm, num_hidden=num_hidden, learn_wu=learn_wu, dim_hidden=dim_hidden)
        sample_cost_preu, sample_costs = nn_forward(sample_obs, sample_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden)
        sup_cost_preu, sup_costs = nn_forward(sup_obs, sup_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden)

        # Build a differentiable test cost by feeding each timestep individually
        test_obs_single = tf.expand_dims(test_obs_single, 0)
        test_torque_single = tf.expand_dims(test_torque_single, 0)
        _, test_cost_single = nn_forward(test_obs_single, test_torque_single, num_hidden=num_hidden, dim_hidden=dim_hidden, learn_wu=learn_wu)
        test_cost_single = tf.squeeze(test_cost_single)

        sup_loss = tf.nn.l2_loss(sup_costs - sup_cost_labels)*multi_obj_supervised_wt

        if smooth_reg_weight > 0:
            # regularization
            demo_sample_preu = tf.concat(0, [demo_cost_preu, sample_cost_preu])
            assert_shape(demo_sample_preu, [sample_batch_size+demo_batch_size, T, 1])
            """
            costs_prev = tf.slice(demo_sample_preu, begin=[0,0], size=[-1, T-2])
            costs_next = tf.slice(demo_sample_preu, begin=[2,0], size=[-1, T-2])
            costs_cur = tf.slice(demo_sample_preu, begin=[1,0], size=[-1, T-2])
            # cur-prev
            slope_prev = costs_cur-costs_prev
            # next-cur
            slope_next = costs_next-costs_cur
            """
            raise NotImplementedError("Smoothness reg not implemented")

        if mono_reg_weight > 0:
            #demo_slope, _ = tf.slice(slope_next, begin=[0], size=[demo_batch_size])
            #demo_slope_reshape = tf.reshape(demo_slope, shape=dict(dim=[-1,1]))
            #mono_reg = L.Python(demo_slope_reshape, loss_weight=mono_reg_weight,
            #                      python_param=dict(module='ioc_layers', layer='L2MonotonicLoss'))
            raise NotImplementedError("Monotonic reg not implemented")

        # init logZ or Z to 1, only learn the bias
        # (also might be good to reduce lr on bias)
        logZ = safe_get('Wdummy', initializer=tf.ones(1)) 
        Z = tf.exp(logZ) # TODO: What does this do?

        # TODO - removed loss weights, changed T, batching, num samples
        # demo cond, num demos, etc.
        ioc_loss = icml_loss(demo_costs, sample_costs, demo_imp_weight, sample_imp_weight, Z)

    outputs = {
        'multiobj_loss': sup_loss+ioc_loss,
        'sup_loss': sup_loss,
        'ioc_loss': ioc_loss,
        'feats': test_feats,
        'test_loss': test_cost,
        'test_loss_single': test_cost_single,
    }
    return inputs, outputs


def compute_feats(net_input, num_hidden=1, dim_hidden=42):
    if len(net_input.get_shape()) == 3:
        batch_size, T, dinput = net_input.get_shape()
    elif len(net_input.get_shape()) == 2:
        T, dinput = net_input.get_shape()

    # Reshape into 2D matrix for matmuls
    net_input = tf.reshape(net_input, [-1, dinput.value])
    with tf.variable_scope('cost_forward'):
        layer = net_input
        for i in range(num_hidden-1):
            with tf.variable_scope('layer_%d' % i):
                W = safe_get('W', (dim_hidden, layer.get_shape()[1].value))
                b = safe_get('b', (dim_hidden))
                layer = tf.nn.relu(tf.matmul(layer, W, transpose_b=True) + b)

        Wfeat = safe_get('Wfeat', (dim_hidden, layer.get_shape()[1].value))
        bfeat = safe_get('bfeat', (dim_hidden))
        feat = tf.matmul(layer, Wfeat, transpose_b=True)+bfeat

    if len(net_input.get_shape()) == 3:
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
        A = safe_get('A', shape=(dim_hidden, dim_hidden))
        Ax = tf.matmul(feat, A, transpose_b=True)
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
    loss += logsumexp(dc_sc, reduction_indices=[0])
    assert_shape(loss, [])
    return loss


def main():
    inputs, outputs= construct_nn_cost_net_tf()
    Y, X = outputs['test_loss_single'], inputs['test_obs_single']
    dldx =  tf.gradients(Y, X)[0]
    print dldx
    print jacobian(dldx, X)
    #print dfdx


if __name__ == "__main__":
    main()
