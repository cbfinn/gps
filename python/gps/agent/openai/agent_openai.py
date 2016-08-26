""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import gym

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_OPENAI
from gps.proto.gps_pb2 import GYM_DATA, GYM_REWARD, ACTION

from gps.sample.sample import Sample
from gps.utility.general_utils import sample_params

class AgentOpenAI(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_OPENAI)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self.env = gym.make(self._hyperparams['env'])
        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            self.x0.append(np.zeros(11))


    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        self.x0[condition] = self.env.reset()

        new_sample = self._init_sample(condition)
        mj_X = self.x0[condition]
        U = np.zeros([self.T, self.dU])
        R = np.zeros(self.T)
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            if (t + 1) < self.T:
                avg_reward = 0
                for _ in range(self._hyperparams['substeps']):
                    self.env.render()
                    mj_X, reward, _, _ = self.env.step(mj_U)
                    avg_reward += reward
                self._set_sample(new_sample, mj_X, t, condition)
                avg_reward = avg_reward/float(self._hyperparams['substeps'])
            R[t] = avg_reward
        new_sample.set(ACTION, U)
        new_sample.set(GYM_REWARD, R)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, condition):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
        """
        sample = Sample(self)
        sample.set(GYM_DATA, self.x0[condition], t=0)
        return sample

    def _set_sample(self, sample, mj_X, t, condition):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
        """
        sample.set(GYM_DATA, np.array(mj_X), t=t+1)

