""" This file defines tests for tensorflow policy optimization. """
import os
import os.path
import sys
import numpy as np
import tensorflow as tf

# Add gps/python to path so that imports work.
gps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'python'))
sys.path.append(gps_path)

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.config_tf import POLICY_OPT_TF
from gps.algorithm.policy_opt.tf_model_example import euclidean_loss_layer, \
    batched_matrix_vector_multiply


def test_policy_opt_tf_init():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    PolicyOptTf(hyper_params, deg_obs, deg_action)


def test_policy_opt_tf_forward():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    policy_opt.policy.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    policy_opt.policy.bias = -np.mean(obs_reshaped.dot(policy_opt.policy.scale), axis=0)
    policy_opt.prob(obs=obs)


def test_policy_opt_tf_backwards():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    policy_opt.policy.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    policy_opt.policy.bias = -np.mean(obs_reshaped.dot(policy_opt.policy.scale), axis=0)
    policy_opt.prob(obs=obs)


def test_policy_forward():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    policy_opt.policy.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    policy_opt.policy.bias = -np.mean(obs_reshaped.dot(policy_opt.policy.scale), axis=0)
    noise = np.random.randn(deg_action)
    policy_opt.policy.act(None, obs[0, 0], None, noise)


def test_policy_opt_backwards():
    hyper_params = POLICY_OPT_TF
    deg_obs = 20
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    # pylint: disable=W0212
    policy_opt._hyperparams['iterations'] = 100  # 100 for testing.
    N = 10
    T = 10
    obs = np.random.randn(N, T, deg_obs)
    tgt_mu = np.random.randn(N, T, deg_action)
    tgt_prc = np.random.randn(N, T, deg_action, deg_action)
    tgt_wt = np.random.randn(N, T)
    new_policy = policy_opt.update(obs, tgt_mu, tgt_prc, tgt_wt, itr=0, inner_itr=1)


def test_pickle():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    state = policy_opt.__getstate__()


def test_auto_save_state():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    policy_opt.auto_save_state()


def test_load_from_auto_save():
    import pickle
    path_to_dict = gps_path + '/gps/algorithm/policy_opt/tf_checkpoint/policy_checkpoint.ckpt_hyperparams'
    state = pickle.load(open(path_to_dict, "rb"))
    hyper_params = state['hyperparams']
    deg_obs = state['dO']
    deg_action = state['dU']
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    policy_opt.__setstate__(state)


def test_unpickle():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    bias = -np.mean(obs_reshaped.dot(scale), axis=0)
    hyper_params['scale'] = scale
    hyper_params['bias'] = bias
    hyper_params['tf_iter'] = 100
    policy_opt.__setstate__({'hyperparams': hyper_params, 'dO': deg_obs, 'dU': deg_action,
                             'scale': policy_opt.policy.scale, 'bias': policy_opt.policy.bias, 'tf_iter': 100})


def test_policy_save():
    hyper_params = POLICY_OPT_TF
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyper_params, deg_obs, deg_action)
    check_path = gps_path + '/gps/algorithm/policy_opt/tf_checkpoint/policy_checkpoint'
    policy_opt.policy.pickle_policy(deg_obs, deg_action, check_path)


def test_policy_load():
    tf_map = POLICY_OPT_TF['network_model']
    check_path = gps_path + '/gps/algorithm/policy_opt/tf_checkpoint/policy_checkpoint'
    pol = TfPolicy.load_policy(check_path, tf_map)

    deg_obs = 100
    deg_action = 7
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    pol.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    pol.bias = -np.mean(obs_reshaped.dot(pol.scale), axis=0)
    noise = np.random.randn(deg_action)
    pol.act(None, obs[0, 0], None, None)


def test_euclidean_loss_layer():
    dim_output = 4
    batch_size = 3
    predicted_action = tf.placeholder("float", [None, dim_output], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    loss = euclidean_loss_layer(predicted_action, action, precision, batch_size)
    mat_vec_prod = batched_matrix_vector_multiply(predicted_action-action, precision)
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    PRECISION = np.arange(batch_size*dim_output*dim_output).reshape((batch_size, dim_output, dim_output))
    PREDICTED_ACTION = np.arange(batch_size*dim_output).reshape((batch_size, dim_output))
    ACTION = np.ones((batch_size, dim_output))
    U = PREDICTED_ACTION - ACTION
    aleph_null = sess.run(mat_vec_prod, feed_dict={predicted_action: PREDICTED_ACTION,
                                                   precision: PRECISION, action: ACTION})

    aleph_one = sess.run(loss, feed_dict={predicted_action: PREDICTED_ACTION,
                                          precision: PRECISION, action: ACTION})

    euclidean_numpy = 0
    scale_factor = 2*batch_size
    for iter_step in range(0, batch_size):
        mat_vec_numpy = U[iter_step].dot(PRECISION[iter_step])
        assert np.allclose(mat_vec_numpy, aleph_null[iter_step])
        euclidean_numpy += mat_vec_numpy.dot(U[iter_step])

    assert np.allclose(aleph_one, euclidean_numpy/scale_factor)


def test_policy_opt_live():
    test_dir = os.path.dirname(__file__) + '/test_data/'
    obs = np.load(test_dir + 'obs.npy')
    tgt_mu = np.load(test_dir + 'tgt_mu.npy')
    tgt_prc = np.load(test_dir + 'tgt_prc.npy')
    scale = np.load(test_dir + 'scale_npy.npy')
    bias = np.load(test_dir + 'bias_npy.npy')
    hyper_params = POLICY_OPT_TF
    deg_obs = 4
    deg_action = 2

    policy = PolicyOptTf(hyper_params, deg_obs, deg_action)
    policy.policy.scale = scale
    policy.policy.bias = bias

    iterations = 200
    batch_size = 32
    batches_per_epoch = np.floor(800 / batch_size)
    idx = range(800)
    np.random.shuffle(idx)

    for i in range(iterations):
        # Load in data for this batch.
        start_idx = int(i * batch_size %
                        (batches_per_epoch * batch_size))
        idx_i = idx[start_idx:start_idx+batch_size]
        feed_dict = {policy.obs_tensor: obs[idx_i],
                     policy.action_tensor: tgt_mu[idx_i],
                     policy.precision_tensor: tgt_prc[idx_i]}
        t = policy.sess.run(policy.act_op, feed_dict={policy.obs_tensor: np.expand_dims(obs[idx_i][0], 0)})
        policy.solver(feed_dict, policy.sess)


def main():
    print 'running tf policy opt tests'
    test_policy_opt_tf_init()
    test_policy_opt_tf_forward()
    test_policy_forward()
    test_policy_opt_backwards()
    test_auto_save_state()
    test_load_from_auto_save()
    test_policy_save()
    test_policy_load()
    test_euclidean_loss_layer()
    test_policy_opt_live()
    print 'tf policy opt tests passed'


if __name__ == '__main__':
    main()




