import sys
import os
import numpy as np
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION


sys.path.append('/'.join(str.split(__file__, '/')[:-2]))


def main():
    gen_data()
    train_net()
    run_net()


def gen_data():
    mjc_agent = init_mujoco_agent()
    for folder in folders:
        pol = get_policy_for_folder(folder)
        conditions = [0, 1, 2, 3]
        for cond in conditions:
            one_sample = mjc_agent.sample(pol, cond, save=False)


def get_policy_for_folder():
    pass


def train_net():
    pass


def run_net():
    pass

def init_mujoco_agent():
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
    mjc_agent = agent['type'](agent)
    return mjc_agent
