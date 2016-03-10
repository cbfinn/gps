""" This file defines policy optimization for a tensorflow policy. """
import os
import os.path
import sys
import numpy as np

# Add gps/python to path so that imports work.
gps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'python'))
sys.path.append(gps_path)

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.config_tf import POLICY_OPT_TF


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
    policy_opt.__auto_save_state__()


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


def main():
    print 'running tf policy opt tests'
    #test_policy_opt_tf_init()
    test_policy_opt_tf_forward()
    #test_policy_forward()
    #test_policy_opt_backwards()
    #test_pickle()
    #test_unpickle()
    #test_auto_save_state()
    #test_load_from_auto_save()
    #test_policy_save()
    #test_policy_load()
    print 'tf policy opt tests passed'


if __name__ == '__main__':
    main()




