""" This file defines the sample class. """
import numpy as np

from gps.proto.gps_pb2 import ACTION


class Sample(object):
    """
    Class that handles the representation of a trajectory and stores a
    single trajectory.
    Note: must be serializable for easy saving, no C++ references!
    """
    def __init__(self, agent):
        self.agent = agent

        self.T = agent.T
        self.dX = agent.dX
        self.dU = agent.dU
        self.dO = agent.dO

        # Dictionary containing the sample data from various sensors.
        self._data = {}

        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._obs = np.empty((self.T, self.dO))
        self._obs.fill(np.nan)

    def set(self, sensor_name, sensor_data, t=None):
        """ Set trajectory data for a particular sensor. """
        if t is None:
            self._data[sensor_name] = sensor_data
            self._X.fill(np.nan)  # Invalidate existing X.
            self._obs.fill(np.nan)  # Invalidate existing obs.
        else:
            if sensor_name not in self._data:
                self._data[sensor_name] = \
                        np.empty((self.T,) + sensor_data.shape)
                self._data[sensor_name].fill(np.nan)
            self._data[sensor_name][t, :] = sensor_data
            self._X[t, :].fill(np.nan)
            self._obs[t, :].fill(np.nan)

    def get(self, sensor_name, t=None):
        """ Get trajectory data for a particular sensor. """
        return (self._data[sensor_name] if t is None
                else self._data[sensor_name][t, :])

    def get_X(self, t=None):
        """ Get the state. Put it together if not precomputed. """
        X = self._X if t is None else self._X[t, :]
        if np.any(np.isnan(X)):
            for data_type in self._data:
                if data_type not in self.agent.x_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                self.agent.pack_data_x(X, data, data_types=[data_type])
        return X

    def get_U(self, t=None):
        """ Get the action. """
        return self._data[ACTION] if t is None else self._data[ACTION][t, :]

    def get_obs(self, t=None):
        """ Get the observation. Put it together if not precomputed. """
        obs = self._obs if t is None else self._obs[t, :]
        if np.any(np.isnan(obs)):
            for data_type in self._data:
                if data_type not in self.agent.obs_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                self.agent.pack_data_obs(obs, data, data_types=[data_type])
        return obs

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('agent')
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__ = state
        self.__dict__['agent'] = None
