

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


def test_policy_on_robot_tf():
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
