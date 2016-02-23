""" This file defines the base agent class. """
import abc
import copy

from gps.agent.config import AGENT
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample_list import SampleList


class Agent(object):
    """
    Agent superclass. The agent interacts with the environment to
    collect samples.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        # Store samples, along with size/index information for samples.
        self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        self.T = self._hyperparams['T']
        self.dU = self._hyperparams['sensor_dims'][ACTION]

        self.x_data_types = self._hyperparams['state_include']
        self.obs_data_types = self._hyperparams['obs_include']

        # List of indices for each data type in state X.
        self._state_idx, i = [], 0
        for sensor in self.x_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            i += dim
        self.dX = i

        # List of indices for each data type in observation.
        self._obs_idx, i = [], 0
        for sensor in self.obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dO = i
        self._x_data_idx = {d: i for d, i in zip(self.x_data_types,
                                                 self._state_idx)}
        self._obs_data_idx = {d: i for d, i in zip(self.obs_data_types,
                                                   self._obs_idx)}

    @abc.abstractmethod
    def sample(self, policy, condition, verbose=True, save=True):
        """
        Draw a sample from the environment, using the specified policy
        and under the specified condition.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self, condition):
        """ Reset environment to the specified condition. """
        pass  # May be overridden in subclass.

    def get_samples(self, condition, start=0, end=None):
        """
        Return the requested samples based on the start and end indices.
        Args:
            start: Starting index of samples to return.
            end: End index of samples to return.
        """
        return (SampleList(self._samples[condition][start:]) if end is None
                else SampleList(self._samples[condition][start:end]))

    def delete_last_sample(self, condition):
        """ Delete the last sample from the specified condition. """
        self._samples[condition].pop()

    def get_idx_x(self, sensor_name):
        """
        Return the indices corresponding to a certain state sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._x_data_idx[sensor_name]

    def get_idx_obs(self, sensor_name):
        """
        Return the indices corresponding to a certain observation sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._obs_data_idx[sensor_name]

    def pack_data_obs(self, existing_mat, data_to_insert, data_types,
                      axes=None):
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dO:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dO)
            insert_shape[axes[i]] = len(self._obs_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._obs_data_idx[data_types[i]][0],
                                   self._obs_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def pack_data_x(self, existing_mat, data_to_insert, data_types, axes=None):
        """
        Update the state matrix with new data.
        Args:
            existing_mat: Current state matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def unpack_data_x(self, existing_mat, data_types, axes=None):
        """
        Returns the requested data from the state matrix.
        Args:
            existing_mat: State matrix to unpack from.
            data_types: Names of the sensor to unpack.
            axes: Which axes to unpack along. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        return existing_mat[index]
