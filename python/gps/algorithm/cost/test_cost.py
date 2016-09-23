import tensorflow as tf


def assert_shape(tensor, shape):
    assert tensor.get_shape().is_compatible_with(shape), "Shape mismatch: %s vs %s" % (str(tensor.get_shape()), shape)


def logsumexp(x, reduction_indices=None):
    max_val = tf.reduce_max(x)
    exp = tf.exp(x-max_val)
    _partition = tf.reduce_sum(x, reduction_indices=reduction_indices)
    _log = tf.log(_partition)+max_val
    return _log


def construct_nn_cost_net_tf(num_hidden=3, dim_hidden=42, dim_input=27, T=100,
                             demo_batch_size=5, sample_batch_size=5, phase='train', ioc_loss='ICML',
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

    with tf.variable_scope('cost_ioc_nn'):
        demo_cost_preu, demo_costs = nn_forward(demo_obs, demo_torque_norm, num_hidden=num_hidden, learn_wu=learn_wu, dim_hidden=dim_hidden)
        sample_cost_preu, sample_costs = nn_forward(sample_obs, sample_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden, reuse=True)
        sup_cost_preu, sup_costs = nn_forward(sup_obs, sup_torque_norm, num_hidden=num_hidden,learn_wu=learn_wu, dim_hidden=dim_hidden, reuse=True)

        sup_loss = tf.nn.l2_loss(sup_costs - sup_cost_labels)

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
        logZ = tf.get_variable('Wdummy', shape=(1,))  # TODO: init to 1
        Z = tf.exp(logZ) # TODO: What does this do?

        # TODO - removed loss weights, changed T, batching, num samples
        # demo cond, num demos, etc.
        ioc_loss = icml_loss(demo_costs, sample_costs, demo_imp_weight, sample_imp_weight, Z)
    return inputs, sup_loss, ioc_loss


def nn_forward(net_input, u_input, num_hidden=1, dim_hidden=42, wu=1e-3, learn_wu=False, reuse=False):
    batch_size, T, dinput = net_input.get_shape()

    # Reshape into 2D matrix for matmuls
    net_input = tf.reshape(net_input, [-1, dinput.value])
    u_input = tf.reshape(u_input, [-1, 1])
    with tf.variable_scope('cost_forward', reuse=reuse):
        layer = net_input
        for i in range(num_hidden-1):
            with tf.variable_scope('layer_%d' % i):
                W = tf.get_variable('W', (dim_hidden, layer.get_shape()[1].value))
                b = tf.get_variable('b', (dim_hidden))
                layer = tf.nn.relu(tf.matmul(layer, W, transpose_b=True) + b)

        Wfeat = tf.get_variable('Wfeat', (dim_hidden, layer.get_shape()[1].value))
        bfeat = tf.get_variable('bfeat', (dim_hidden))
        feat = tf.matmul(layer, Wfeat, transpose_b=True)+bfeat

        A = tf.get_variable('A', shape=(dim_hidden, dim_hidden))
        Ax = tf.matmul(feat, A, transpose_b=True)
        AxAx = Ax*Ax

        # Calculate torque penalty
        wu = tf.get_variable('wu', initializer=tf.constant(wu), trainable=learn_wu)
        assert_shape(wu, [])
        u_cost = u_input*wu

        # Reshape result back into batches
        AxAx = tf.reshape(AxAx, [batch_size.value, T.value, dim_hidden])
        u_cost = tf.reshape(u_cost, [batch_size.value, T.value, 1])

        all_costs_preu = tf.reduce_sum(AxAx, reduction_indices=[2], keep_dims=True)
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
    inputs, sl, il = construct_nn_cost_net_tf()


if __name__ == "__main__":
    main()
