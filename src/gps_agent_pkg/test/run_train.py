import time
import os
import numpy as np
import scipy
import scipy.io
import logging
import argparse
import cPickle
from gps.hyperparam_pr2 import defaults
from gps.sample.sample import Sample
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.agent.ros.agent_ros import AgentROS
from gps.proto.gps_pb2 import *
import rospy

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.debug("Test debug message")
np.set_printoptions(suppress=True)
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_agent(T=100):
    defaults['agent']['T'] = T
    #defaults['agent']['state_include'] = [JOINT_ANGLES, JOINT_VELOCITIES]
    #sample_data = SampleData(defaults['sample_data'], defaults['common'], False)
    agent = AgentROS(defaults['agent'])
    r = rospy.Rate(1)
    r.sleep()
    return agent

def run_offline():
    """
    Run offline controller, and save results to controllerfile
    """
    agent = setup_agent()
    algorithm = defaults['algorithm']['type'](defaults['algorithm'])
    conditions = 1
    idxs = [[] for _ in range(conditions)]
    for m in range(conditions):
        pol = algorithm.cur[m].traj_distr
        sample = agent.sample(pol, m)
        agent.reset(m)

    n = 0
    for itr in range(15): # Iterations
        print 'iter: ', itr
        for m in range(conditions):
            for i in range(2): # Trials per iteration
                pol = algorithm.cur[m].traj_distr
                sample = agent.sample(pol, m)
                idxs[m].append(n)
                agent.reset(m)
                n += 1
        algorithm.iteration([agent.get_samples(m, -n) for m in range(conditions)])
        print 'Finished itr ', itr
    import pdb; pdb.set_trace()

run_offline()
