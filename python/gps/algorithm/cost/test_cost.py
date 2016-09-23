import tensorflow as tf

def construct_nn_cost_net_tf(num_hidden=3, dim_hidden=None, dim_input=27, T=100,
                             demo_batch_size=5, sample_batch_size=5, phase=TRAIN, ioc_loss='ICML',
                             Nq=1, smooth_reg_weight=0.0, mono_reg_weight=0.0, gp_reg_weight=0.0, multi_obj_supervised_wt=1.0, learn_wu=False):

    demo_obs = tf.placeholder(tf.float32, shape=(demo_batch_size, T, dim_input))
    demo_torque_norm = tf.placeholder(tf.float32, shape=(demo_batch_size, T, 1))
    demo_imp_weight = tf.placeholder(tf.float32, shape=(demo_batch_size, 1))
    sample_obs = tf.placeholder(tf.float32, shape=(sample_batch_size, T, dim_input))
    sample_torque_norm = tf.placeholder(tf.float32, shape=(sample_batch_size, T, 1))
    sample_imp_weight = tf.placeholder(tf.float32, shape=(sample_batch_size, 1))
    length_scale = tf.placeholder(tf.float32, shape=(dim_input))
    sup_batch_size = sample_batch_size+demo_batch_size
    sup_obs = tf.placeholder(tf.float32, shape=(sup_batch_size, T, dim_input))
    sup_torque_norm = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))
    sup_cost_labels = tf.placeholder(tf.float32, shape=(sup_batch_size, T, 1))

    with tf.variable_scope('cost_ioc_nn'):
        net_input = tf.concat(0, [demo_obs, sample_obs, sup_obs])
        all_u = tf.concat(0, [demo_torque_norm, sample_torque_norm, sup_torque_norm])

        #TODO(Justin) Make forward pass into a function which can be applied to demos, samples, and sup individually
        prev_layer = net_input
        for i in range(num_hidden-1):
            with tf.variable_scope('layer_%d' % i):
                W = tf.get_variable('W', (dim_hidden, prev_layer.get_shape()[2]))
                b = tf.get_variable('b', (dim_hidden))
                layer = tf.matmul(W, prev_layer) + b
                layer = tf.nn.relu(layer)
                prev_layer = layer

        # Necessary for computing gradients
        loss_weight = 1.0 if phase == 'forward_feat' else 0.0
        Wfeat = tf.get_variable('Wfeat', (dim_hidden, prev_layer.get_shape()[2]))
        bfeat = tf.get_variable('bfeat', (dim_hidden))
        feat = tf.matmul(Wfeat, prev_layer)+bfeat

        # TODO: What is this?
        feat_loss = loss_weight * feat

        if phase != 'forward_feat':
            if learn_wu:
                # learned multiplier
                wu = tf.get_variable('Wu', (dim_hidden, 1))
                all_u = tf.matmul(wu, all_u)

            A = tf.get_variable('A', shape=(dim_hidden, dim_hidden))
            Ax = tf.matmul(A, feat)

            AxAx = Ax*Ax
            all_costs_preu = tf.reduce_sum(AxAx, reduction_indices=[2])
            all_costs = all_costs_preu + all_u
        else:
            all_costs = None


        if phase == 'supervised':
            loss = tf.nn.l2_loss(all_costs - sup_cost_labels)
        elif phase == 'train' or phase == 'multi_objective':
            demo_costs = tf.slice(all_costs, begin=[0,0], size=[demo_batch_size, T])
            sample_costs = tf.slice(all_costs, begin=[demo_batch_size,0], size=[sample_batch_size, T])
            supervised_costs = tf.slice(all_costs, begin=[demo_batch_size+sample_batch_size,0], size=[sup_batch_size, T])

            demo_sample_preu = tf.slice(all_costs_preu, begin=[0,0], size=[demo_batch_size+sample_batch_size, T])
            supervised_preu = tf.slice(all_costs_preu, begin=[demo_batch_size+sample_batch_size,0], size=[sup_batch_size, T])

            # regularization
            """
            costs_prev = tf.slice(demo_sample_preu, begin=[0,0], size=[-1, T-2])
            costs_next = tf.slice(demo_sample_preu, begin=[2,0], size=[-1, T-2])
            costs_cur = tf.slice(demo_sample_preu, begin=[1,0], size=[-1, T-2])
            # cur-prev
            slope_prev = costs_cur-costs_prev
            # next-cur
            slope_next = costs_next-costs_cur
            """

            if smooth_reg_weight > 0:
                raise NotImplementedError("Smoothness reg not implemented")

            if mono_reg_weight > 0:
                raise NotImplementedError("Monotonic reg not implemented")

            #demo_slope, _ = tf.slice(slope_next, begin=[0], size=[demo_batch_size])
            #demo_slope_reshape = tf.reshape(demo_slope, shape=dict(dim=[-1,1]))
            #mono_reg = L.Python(demo_slope_reshape, loss_weight=mono_reg_weight,
            #                      python_param=dict(module='ioc_layers', layer='L2MonotonicLoss'))

            dummy = tf.zeros(1) #L.DummyData(ntop=1, shape=dict(dim=[1]), data_filler=dict(type='constant',value=0))
            # init logZ or Z to 1, only learn the bias
            # (also might be good to reduce lr on bias)
            Wdummy = tf.get_variable('Wdummy', shape=(1,))
            bdummy = tf.get_variable('bdummy', shape=(1,))
            logZ = Wdummy*dummy + bdummy
            Z = tf.exp(logZ)

            # TODO - removed loss weights, changed T, batching, num samples
            # demo cond, num demos, etc.
            out = icml_loss(demo_costs, sample_costs, demo_imp_weight, sample_imp_weight, Z)
            loss = out
    return loss

def icml_loss(demo_costs, sample_costs, demo_iw, sample_iw, Z):
    return 1

def main():
    pass

if __name__ == "__main__":
    main()
