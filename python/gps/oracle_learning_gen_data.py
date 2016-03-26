import sys
import os
import numpy as np
import pickle
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.tf_model_example import example_tf_network

policies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/mjc_pointmass_example/data_files/policies/'))


def gen_data(num_samples=100):
    policy_folders = os.listdir(policies_path)
    mjc_agent, dO, dU = init_mujoco_agent()
    conditions = [0, 1, 2, 3]
    goal_state_dim = 2
    t_steps = 100
    the_data = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps,  dO))
    the_actions = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, dU))
    the_goals = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, goal_state_dim))
    iter_count = 0
    for folder in policy_folders:
        pol_dict_path = policies_path + '/' + folder + '/_pol'
        pol = get_policy_for_folder(pol_dict_path)
        pol_dict = pickle.load(open(pol_dict_path, "rb"))
        goal_state = pol_dict['goal_state']
        for samples in range(0, num_samples):
            for cond in conditions:
                one_sample = mjc_agent.sample(pol, cond, save=False)
                obs = one_sample.get_obs()
                U = one_sample.get_U()
                the_data[iter_count] = obs
                the_actions[iter_count] = U
                the_goals[iter_count] = goal_state
                iter_count += 1
                import time
                time.sleep(1)
    np.save(policies_path + '/the_data', the_data)
    np.save(policies_path + '/the_actions', the_actions)
    np.save(policies_path + '/the_goals', the_goals)
    print 'done bitch'


def get_policy_for_folder(check_path):
    tf_map_generator = example_tf_network
    pol = TfPolicy.load_policy(check_path, tf_map_generator)
    return pol


def init_mujoco_agent():
    from gps.agent.mjc.agent_mjc import AgentMuJoCo
    from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION
    SENSOR_DIMS = {
        JOINT_ANGLES: 2,
        JOINT_VELOCITIES: 2,
        ACTION: 2,
    }
    agent = {
        'type': AgentMuJoCo,
        'filename': './mjc_models/particle2d.xml',
        'x0': [np.array([0., 0., 0., 0.]), np.array([0., 1., 0., 0.]),
               np.array([1., 0., 0., 0.]), np.array([1., 1., 0., 0.])],
        'dt': 0.05,
        'substeps': 5,
        'conditions': 4,
        'T': 100,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    }
    dO = 4
    dU = 2
    mjc_agent = agent['type'](agent)
    return mjc_agent, dO, dU

if __name__ == '__main__':
    gen_data(num_samples=1)
