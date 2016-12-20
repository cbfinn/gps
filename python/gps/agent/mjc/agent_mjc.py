""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np
try:
    import imageio
except ImportError:
    print 'imageio not found'
    imageio = None

import mjcpy
import pickle

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_MUJOCO
from gps.gui.fp_plot import FPPlot
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, GYM_REWARD, END_EFFECTOR_POINTS_NO_TARGET, \
        END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, IMAGE_FEAT, CONDITION_DATA

from gps.sample.sample import Sample
from gps.utility.general_utils import sample_params
from gps.utility import ColorLogger

LOGGER = ColorLogger(__name__)


class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        if 'models' in hyperparams:
            # If MJCModel objects are provided, load those instead
            if isinstance(hyperparams['models'], list):
                files = [model.open() for model in hyperparams['models']]
                self._setup_world([f.name for f in files])
                [f.close() for f in files]
            else:
                file = hyperparams['models'].open()
                self._setup_world(file.name)
                file.close()
        else:
            self._setup_world(hyperparams['filename'])

        self.feature_encoder = None
        if 'feature_encoder' in hyperparams:
            with open(hyperparams['feature_encoder'],'r') as f:
                alg = pickle.load(f)
            self.feature_encoder = alg.policy_opt.policy

        if self._hyperparams['hardcoded_linear_dynamics']:
            dt = self._hyperparams['dt']
            F = np.array([[ 1, 0, dt, 0, dt**2., 0], [0, 1, 0, dt, 0, dt**2.],
                          [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt]])
            self.F = F


    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def randomize_world(self, condition):
        if self._hyperparams['randomize_world']:
            assert 'models' in self._hyperparams

            #Re-invoke model building.
            self._hyperparams['models'][condition].regenerate()
            with self._hyperparams['models'][condition].asfile() as model_file:
                world = mjcpy.MJCWorld(model_file.name)
                self._world[condition] = world

            if self._hyperparams['render']:
                cam_pos = self._hyperparams['camera_pos']
                self._world[condition].init_viewer(AGENT_MUJOCO['image_width'],
                                           AGENT_MUJOCO['image_height'],
                                           cam_pos[0], cam_pos[1], cam_pos[2],
                                           cam_pos[3], cam_pos[4], cam_pos[5])


    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._world = []
        self._model = []

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        if not isinstance(filename, list):
            world = mjcpy.MJCWorld(filename)
            self._world = [world for _ in range(self._hyperparams['conditions'])]
            self._model = [self._world[i].get_model().copy()
                           for i in range(self._hyperparams['conditions'])]
        else:
            for i in range(self._hyperparams['conditions']):
                self._world.append(mjcpy.MJCWorld(filename[i]))
                self._model.append(self._world[i].get_model())

        for i in range(self._hyperparams['conditions']):
            for j in range(len(self._hyperparams['pos_body_idx'][i])):
                idx = self._hyperparams['pos_body_idx'][i][j]
                # TODO: this should actually add [i][j], but that would break things
                self._model[i]['body_pos'][idx, :] += \
                        self._hyperparams['pos_body_offset'][i]

        # TODO: Seems like using multiple files wouldn't work with this.
        self._joint_idx = list(range(self._model[0]['nq']))
        self._vel_idx = [i + self._model[0]['nq'] for i in self._joint_idx]

        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            # TODO is this correct?
            if END_EFFECTOR_POINTS in self.x_data_types:
                # TODO: this assumes END_EFFECTOR_VELOCITIES is also in datapoints right?
                self._init(i)
                eepts = self._world[i].get_data()['site_xpos'].flatten()
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
                )
            elif END_EFFECTOR_POINTS_NO_TARGET in self.x_data_types:
                self._init(i)
                eepts = self._world[i].get_data()['site_xpos'].flatten()
                eepts_notgt = np.delete(eepts, self._hyperparams['target_idx'])
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts_notgt, np.zeros_like(eepts_notgt)])
                )
            else:
                self.x0.append(self._hyperparams['x0'][i])
            if IMAGE_FEAT in self.x_data_types:
                self.x0[i] = np.concatenate([self.x0[i], np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],))])
            if CONDITION_DATA in self.x_data_types:
                self.x0[i] = np.concatenate([self.x0[i], self._hyperparams['condition_data'][i]])

        if self._hyperparams['render']:
            cam_pos = self._hyperparams['camera_pos']
            for i in range(self._hyperparams['conditions']):
                self._world[i].init_viewer(AGENT_MUJOCO['image_width'],
                                           AGENT_MUJOCO['image_height'],
                                           cam_pos[0], cam_pos[1], cam_pos[2],
                                           cam_pos[3], cam_pos[4], cam_pos[5])

    def reset_initial_x0(self, condition):
        """
        Reset initial x0 randomly based on sampling_range_x0
        and prohibited_ranges_x0
        Args:
            condition: Which condition setup to run.
        """
        self._hyperparams['x0'][condition] = sample_params(
                self._hyperparams['sampling_range_x0'],
                self._hyperparams['prohibited_ranges_x0'])

    def reset_initial_body_offset(self, condition):
        """
        Reset initial body offset randomly based on sampling_range_bodypos
        and prohibited_ranges_bodypos
        Args:
            condition: Which condition setup to run.
        """
        tmp_body_pos_offset = self._hyperparams['pos_body_offset'][condition][:]
        self._hyperparams['pos_body_offset'][condition] = sample_params(
                self._hyperparams['sampling_range_bodypos'],
                self._hyperparams['prohibited_ranges_bodypos'])
        # TODO: handle the i/j stuff as above
        for j in range(len(self._hyperparams['pos_body_idx'][condition])):
            idx = self._hyperparams['pos_body_idx'][condition][j]
            self._model[condition]['body_pos'][idx, :] -= tmp_body_pos_offset
            self._model[condition]['body_pos'][idx, :] += self._hyperparams['pos_body_offset'][condition]

        if END_EFFECTOR_POINTS in self.x_data_types:
            x0 = self._hyperparams['x0'][condition]
            eepts = self._world[condition].get_data()['site_xpos'].flatten()
            self.x0[condition] = np.concatenate([x0, eepts, np.zeros_like(eepts)])

        if IMAGE_FEAT in self.x_data_types:
            import pdb; pdb.set_trace()  # TODO

    def visualize_sample(self, sample, condition):
        # replays a sample, with verbose
        self._init(condition)
        mj_X = self._hyperparams['x0'][condition]
        for t in range(self.T):
            mj_U = sample.get_U(t=t)
            self._world[condition].plot(mj_X)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)



    def sample(self, policy, condition, verbose=True, save=True, noisy=True, record_image=False, record_gif=None, record_gif_fps=None):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
            feature_encoder: Function to get image features from.
        """
        if record_gif:
            verbose = True
            record_image = True

        self.randomize_world(condition)
        # Create new sample, populate first time step.
        feature_fn = None
        if self.feature_encoder:    # TODO TODO - this will default to always using the feature encoder.
            feature_fn = self.feature_encoder.get_image_features
        elif 'get_image_features' in dir(policy):
            feature_fn = policy.get_image_features
        new_sample = self._init_sample(condition, feature_fn=feature_fn, record_image=record_image)

        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        if self._hyperparams['record_reward']:
            R = np.zeros(self.T)

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        # TODO: Does anyone use this? Is it even correct? Doesn't
        #       body_pos get overwritten everytime?
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)

        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            if self._hyperparams['record_reward']:
                R[t] = -(np.linalg.norm(X_t[4:7] - X_t[7:10]) + np.square(mj_U).sum())
            U[t, :] = mj_U
            if verbose:
                self._world[condition].plot(mj_X)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    if 'hardcoded_linear_dynamics' in self._hyperparams and self._hyperparams['hardcoded_linear_dynamics']:
                        mj_X = self.F.dot(np.r_[mj_X, mj_U])
                    else:
                        mj_X, _ = self._world[condition].step(mj_X, mj_U)
                #TODO: Some hidden state stuff will go here.
                #TODO: Will it? This TODO has been here for awhile
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition, feature_fn=feature_fn, record_image=record_image)


        if record_gif and imageio:
            # record a gif of sample.
            images = new_sample.get(RGB_IMAGE)
            size = list(new_sample.get(RGB_IMAGE_SIZE))
            images = images.reshape([-1]+size)
            images = np.transpose(images, [0,2,3,1])
            LOGGER.debug('Saving gif sample to :%s', record_gif)
            if record_gif_fps is None:
                record_gif_fps = 1./self._hyperparams['dt']
            imageio.mimsave(record_gif, images, fps=record_gif_fps)

        new_sample.set(ACTION, U)
        if self._hyperparams['record_reward']:
            new_sample.set(GYM_REWARD, R)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init(self, condition):
        """
        Set the world to a given model, and run kinematics.
        Args:
            condition: Which condition to initialize.
        """

        # Initialize world/run kinematics
        self._world[condition].set_model(self._model[condition])
        x0 = self._hyperparams['x0'][condition]
        idx = len(x0) // 2
        data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
        self._world[condition].set_data(data)
        self._world[condition].kinematics()

    def _init_sample(self, condition, feature_fn=None, record_image=False):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: function to compute image features from the observation (e.g. image).
        """
        sample = Sample(self)

        # Initialize world/run kinematics
        self._init(condition)

        # Initialize sample with stuff from _data
        data = self._world[condition].get_data()
        sample.set(JOINT_ANGLES, data['qpos'].flatten(), t=0)
        sample.set(JOINT_VELOCITIES, data['qvel'].flatten(), t=0)
        if self._hyperparams['hardcoded_linear_dynamics']:
            eepts = np.r_[data['qpos'].flatten(), [0.]]
        else:
            eepts = data['site_xpos'].flatten()
        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(eepts, self._hyperparams['target_idx']), t=0)

        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)
        if (END_EFFECTOR_POINT_VELOCITIES_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(np.zeros_like(eepts), self._hyperparams['target_idx']), t=0)

        jac = np.zeros([eepts.shape[0], self._model[condition]['nq']])
        for site in range(eepts.shape[0] // 3):
            idx = site * 3
            jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)

        if CONDITION_DATA in self.obs_data_types:
            sample.set(CONDITION_DATA, self._hyperparams['condition_data'][condition], t=0)

        # save initial image to meta data
        self._world[condition].plot(self._hyperparams['x0'][condition])

        if self._hyperparams['render']:
            img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                          self._hyperparams['image_height'])
            # mjcpy image shape is [height, width, channels],
            # dim-shuffle it for later conv-net processing,
            # and flatten for storage
            img_data = np.transpose(img["img"], (2, 1, 0)).flatten()
            # if initial image is an observation, replicate it for each time step
            if CONTEXT_IMAGE in self.obs_data_types:
                sample.set(CONTEXT_IMAGE, np.tile(img_data, (self.T, 1)), t=None)
            else:
                sample.set(CONTEXT_IMAGE, img_data, t=None)
            sample.set(CONTEXT_IMAGE_SIZE, np.array([self._hyperparams['image_channels'],
                                                    self._hyperparams['image_width'],
                                                    self._hyperparams['image_height']]), t=None)
            # only save subsequent images if image is part of observation
            if RGB_IMAGE in self.obs_data_types or IMAGE_FEAT in self.x_data_types or IMAGE_FEAT in self.obs_data_types or record_image:
                sample.set(RGB_IMAGE, img_data, t=0)
                sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                            self._hyperparams['image_width'],
                                            self._hyperparams['image_height']], t=None)
                if IMAGE_FEAT in self.obs_data_types:
                    if feature_fn is not None:
                        sample.set(IMAGE_FEAT, feature_fn(sample.get(RGB_IMAGE, t=0))[0], t=0)
                        if RGB_IMAGE not in self.obs_data_types:
                            sample.set(RGB_IMAGE, None, t=0)  # Remove image from sample for memory reasons.
                    else:
                        # TODO - need better solution than setting this to 0.
                        sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=0)
        return sample

    def _set_sample(self, sample, mj_X, t, condition, feature_fn=None, record_image=False):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation (e.g. image).
        """
        if self._hyperparams['hardcoded_linear_dynamics']:
            sample.set(JOINT_ANGLES, mj_X[self._joint_idx], t=t+1)
            sample.set(JOINT_VELOCITIES, mj_X[self._vel_idx], t=t+1)
            cur_eepts = np.r_[mj_X[self._joint_idx], [0.]]
        else:
            sample.set(JOINT_ANGLES, self._data['qpos'].flatten(), t=t+1)
            sample.set(JOINT_VELOCITIES, self._data['qvel'].flatten(), t=t+1)
            cur_eepts = self._data['site_xpos'].flatten()
        sample.set(END_EFFECTOR_POINTS, cur_eepts, t=t+1)
        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(cur_eepts, self._hyperparams['target_idx']), t=t+1)

        prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
        eept_vels = (cur_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)
        if (END_EFFECTOR_POINT_VELOCITIES_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(eept_vels, self._hyperparams['target_idx']), t=t+1)

        jac = np.zeros([cur_eepts.shape[0], self._model[condition]['nq']])
        for site in range(cur_eepts.shape[0] // 3):
            idx = site * 3
            jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t+1)

        if CONDITION_DATA in self.obs_data_types:
            sample.set(CONDITION_DATA, self._hyperparams['condition_data'][condition], t=t+1)

        if RGB_IMAGE in self.obs_data_types or IMAGE_FEAT in self.x_data_types or IMAGE_FEAT in self.obs_data_types or record_image:
            img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                          self._hyperparams['image_height'])
            sample.set(RGB_IMAGE, np.transpose(img["img"], (2, 1, 0)).flatten(), t=t+1)
            if IMAGE_FEAT in self.obs_data_types:
                if feature_fn is not None:
                    sample.set(IMAGE_FEAT, feature_fn(sample.get(RGB_IMAGE, t=t+1))[0], t=t+1)
                    if RGB_IMAGE not in self.obs_data_types:
                        sample.set(RGB_IMAGE, None, t=t+1)  # Remove image from sample for memory reasons.
                else:
                    # TODO - need better solution than setting this to 0.
                    sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=t+1)

    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        img = obs[imstart:imend]
        img = img.reshape((image_width, image_height, image_channels))
        return img
