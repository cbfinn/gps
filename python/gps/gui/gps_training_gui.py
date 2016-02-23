"""
~~~ GUI Specifications ~~~
Action Axis
    - stop, reset, start, emergency stop

Data Plotter
    - algorithm training costs
    - losses of feature points / end effector points
    - joint states, feature point states, etc.
    - save tracked data to file

Image Visualizer
    - real-time image and feature points visualization
    - overlay of initial and target feature points
    - visualize hidden states?
    - create movie from image visualizations
"""
import copy
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from gps.gui.config import common as common_config
from gps.gui.config import gps_training as gps_training_config
from gps.gui.action_axis import Action, ActionAxis
from gps.gui.output_axis import OutputAxis
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.plotter_3d import Plotter3D
from gps.gui.image_visualizer import ImageVisualizer

from gps.gui.target_setup_gui import load_data_from_npz
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS

class GPSTrainingGUI(object):
    """ GPS Training GUI class. """
    def __init__(self, hyperparams):
        self._hyperparams = copy.deepcopy(common_config)
        self._hyperparams.update(copy.deepcopy(gps_training_config))
        self._hyperparams.update(hyperparams)

        self._log_filename = self._hyperparams['log_filename']
        if 'target_filename' in self._hyperparams:
            self._target_filename = self._hyperparams['target_filename']
        else:
            self._target_filename = ''

        # GPS Training Status.
        self.mode = 'run'  # Modes: run, wait, end, request, process.
        self.request = None  # Requests: stop, reset, go, fail, None.
        self.err_msg = None
        self._colors = {
            'run': 'cyan',
            'wait': 'orange',
            'end': 'red',

            'stop': 'red',
            'reset': 'yellow',
            'go': 'green',
            'fail': 'magenta',
        }
        self._first_update = True

        # Actions.
        actions_arr = [
            Action('stop', 'stop', self.request_stop, axis_pos=0),
            Action('reset', 'reset', self.request_reset, axis_pos=1),
            Action('go', 'go', self.request_go, axis_pos=2),
            Action('fail', 'fail', self.request_fail, axis_pos=3),
        ]
        self._actions = {action._key: action for action in actions_arr}
        for key, action in self._actions.iteritems():
            if key in self._hyperparams['keyboard_bindings']:
                action._kb = self._hyperparams['keyboard_bindings'][key]
            if key in self._hyperparams['ps3_bindings']:
                action._pb = self._hyperparams['ps3_bindings'][key]

        # GUI Components.
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        # Remove 's' keyboard shortcut for saving.
        plt.rcParams['keymap.save'] = ''

        self._fig = plt.figure(figsize=(12, 12))
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)

        # Assign GUI component locations.
        self._gs = gridspec.GridSpec(16, 8)
        self._gs_action_axis        = self._gs[0:2,  0:8]
        self._gs_action_output      = self._gs[2:3,  0:4]
        self._gs_status_output      = self._gs[3:4,  0:4]
        self._gs_cost_plotter       = self._gs[2:4,  4:8]
        self._gs_algthm_output      = self._gs[4:8,  0:8]
        self._gs_traj_visualizer    = self._gs[8:16, 0:4]
        self._gs_image_visualizer   = self._gs[8:16, 4:8]

        # Create GUI components.
        self._action_axis = ActionAxis(self._fig, self._gs_action_axis, 1, 4, self._actions,
                ps3_process_rate=self._hyperparams['ps3_process_rate'],
                ps3_topic=self._hyperparams['ps3_topic'],
                ps3_button=self._hyperparams['ps3_button'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])
        self._action_output = OutputAxis(self._fig, self._gs_action_output, border_on=True)
        self._status_output = OutputAxis(self._fig, self._gs_status_output, border_on=False)
        self._algthm_output = OutputAxis(self._fig, self._gs_algthm_output, max_display_size=15,
                log_filename=self._log_filename, fontsize=10, font_family='monospace')
        self._cost_plotter = MeanPlotter(self._fig, self._gs_cost_plotter, color='blue', label='mean cost')
        self._traj_visualizer = Plotter3D(self._fig, self._gs_traj_visualizer, num_plots=self._hyperparams['conditions'])
        self._image_visualizer = ImageVisualizer(self._fig, self._gs_image_visualizer,
                cropsize=(240, 240), rostopic=self._hyperparams['image_topic'], show_overlay_buttons=True)

        # Setup GUI components.
        self._algthm_output.log_text('\n')
        self.set_output_text(self._hyperparams['info'])
        self.run_mode()

        # WARNING: Make sure the legend values in UPDATE match the below linestyles/markers and colors
        [self._traj_visualizer.set_title(m, 'Condition %d' % (m)) for m in range(self._hyperparams['conditions'])]
        self._traj_visualizer.add_legend(linestyle='-',    marker='None', color='green',     label='Trajectory Samples')
        self._traj_visualizer.add_legend(linestyle='-',    marker='None', color='blue',      label='Policy Samples')
        self._traj_visualizer.add_legend(linestyle='None', marker='x',    color=(0.5, 0, 0), label='LG Controller Means')
        self._traj_visualizer.add_legend(linestyle='-',    marker='None', color='red',       label='LG Controller Distributions')

        self._fig.canvas.draw()

    # GPS Training Functions.
    #TODO: Docstrings here.
    def request_stop(self, event=None):
        self.request_mode('stop')

    def request_reset(self, event=None):
        self.request_mode('reset')

    def request_go(self, event=None):
        self.request_mode('go')

    def request_fail(self, event=None):
        self.request_mode('fail')

    def request_mode(self, request):
        self.mode = 'request'
        self.request = request
        self.set_action_text(self.request + ' requested')
        self.set_action_bgcolor(self._colors[self.request], alpha=0.2)

    def process_mode(self):
        self.mode = 'process'
        self.set_action_text(self.request + ' processed')
        self.set_action_bgcolor(self._colors[self.request], alpha=1.0)
        if self.err_msg:
            self.set_action_text(self.request + ' processed' + '\nERROR: ' +
                                 self.err_msg)
            self.err_msg = None
            time.sleep(1.0)
        else:
            time.sleep(0.5)
        if self.request in ('stop', 'reset', 'fail'):
            self.wait_mode()
        elif self.request == 'go':
            self.run_mode()
        self.request = None

    def wait_mode(self):
        self.mode = 'wait'
        self.set_action_text('waiting')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def run_mode(self):
        self.mode = 'run'
        self.set_action_text('running')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def end_mode(self):
        self.mode = 'end'
        self.set_action_text('ended')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def estop(self, event=None):
        self.set_action_text('estop: NOT IMPLEMENTED')

    # GUI functions.
    def set_action_text(self, text):
        self._action_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output.set_bgcolor(color, alpha)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def set_status_text(self, text):
        self._status_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def set_output_text(self, text):
        self._algthm_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def append_output_text(self, text):
        self._algthm_output.append_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def set_image_overlays(self, condition):
        if len(self._target_filename) == 0:
            return
        initial_image = load_data_from_npz(self._target_filename, self._hyperparams['image_actuator'], str(condition),
                'initial', 'image', default=np.zeros((1,1,3)))
        target_image  = load_data_from_npz(self._target_filename, self._hyperparams['image_actuator'], str(condition),
                'target',  'image', default=np.zeros((1,1,3)))
        self._image_visualizer.set_initial_image(initial_image, alpha=0.3)
        self._image_visualizer.set_target_image(target_image, alpha=0.3)

    def update(self, itr, algorithm, agent, traj_sample_lists, pol_sample_lists):
        # Plot Costs
        if algorithm.M == 1:
            # Update plot with each sample's cost (summed over time).
            costs = np.sum(algorithm.prev[0].cs, axis=1)
        else:
            # Update plot with each condition's mean sample cost (summed over time).
            costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        self._cost_plotter.update(costs, t=itr)

        # Setup iteration data column titles and 3D visualization plot titles and legend
        if self._first_update:
            self.set_output_text(self._hyperparams['experiment_name'])
            condition_titles = '%3s | %8s' % ('', '')
            itr_data_fields  = '%3s | %8s' % ('itr', 'avg_cost')
            for m in range(algorithm.M):
                condition_titles += ' | %8s %9s %-7d' % ('', 'condition', m)
                itr_data_fields  += ' | %8s %8s %8s' % ('  cost  ', '  step  ', 'entropy ')
                if algorithm.prev[0].pol_info is not None:
                    condition_titles += ' %8s %8s' % ('', '')
                    itr_data_fields  += ' %8s %8s' % ('kl_div_i', 'kl_div_f')
            self.append_output_text(condition_titles)
            self.append_output_text(itr_data_fields)

            self._first_update = False

        # Print Iteration Data
        avg_cost = np.mean(costs)
        itr_data = '%3d | %8.2f' % (itr, avg_cost)
        for m in range(algorithm.M):
            cost = costs[m]
            step = algorithm.prev[m].step_mult
            entropy = 2*np.sum(np.log(np.diagonal(algorithm.prev[m].traj_distr.chol_pol_covar, axis1=1, axis2=2)))
            itr_data += ' | %8.2f %8.2f %8.2f' % (cost, step, entropy)
            if algorithm.prev[0].pol_info is not None:
                kl_div_i = algorithm.prev[m].pol_info.prev_kl[0]
                kl_div_f = algorithm.prev[m].pol_info.prev_kl[-1]
                itr_data += ' %8.2f %8.2f' % (kl_div_i, kl_div_f)
        self.append_output_text(itr_data)

        if END_EFFECTOR_POINTS not in agent.x_data_types:
            # Skip plotting samples.
            self._traj_visualizer.draw()    # this must be called explicitly
            self._fig.canvas.draw()
            self._fig.canvas.flush_events() # Fixes bug in Qt4Agg backend
            return



        # TODO(xinyutan) - this assumes that END_EFFECTOR_POINTS are in the
        # sample, which is not true for box2d. quick fix is above.
        # Calculate xlim, ylim, zlim for 3D visualizations from traj_sample_lists and pol_sample_lists
        # (this clips off LQG means/distributions that are not in the area of interest)
        all_eept = np.empty((0, 3))
        sample_lists = traj_sample_lists + pol_sample_lists if pol_sample_lists else traj_sample_lists
        for sample_list in sample_lists:
            for sample in sample_list.get_samples():
                ee_pt = sample.get(END_EFFECTOR_POINTS)
                for i in range(ee_pt.shape[1]/3):
                    ee_pt_i = ee_pt[:, 3*i+0:3*i+3]
                    all_eept = np.r_[all_eept, ee_pt_i]
        min_xyz = np.amin(all_eept, axis=0)
        max_xyz = np.amax(all_eept, axis=0)
        xlim, ylim, zlim = (min_xyz[0], max_xyz[0]), (min_xyz[1], max_xyz[1]), (min_xyz[2], max_xyz[2])

        # Plot 3D Visualizations
        for m in range(algorithm.M):
            # Clear previous plots
            self._traj_visualizer.clear(m)
            self._traj_visualizer.set_lim(i=m, xlim=xlim, ylim=ylim, zlim=zlim)

            # Linear Gaussian Controller Distributions (Red)
            mu, sigma = algorithm.traj_opt.forward(algorithm.prev[m].traj_distr, algorithm.prev[m].traj_info)
            eept_idx = agent.get_idx_x(END_EFFECTOR_POINTS)
            start, end = eept_idx[0], eept_idx[-1]
            mu_eept, sigma_eept = mu[:, start:end+1], sigma[:, start:end+1, start:end+1]

            for i in range(mu_eept.shape[1]/3):
                mu, sigma = mu_eept[:, 3*i+0:3*i+3], sigma_eept[:, 3*i+0:3*i+3, 3*i+0:3*i+3]
                self._traj_visualizer.plot_3d_gaussian(i=m, mu=mu, sigma=sigma, edges=100, linestyle='-', linewidth=1.0, color='red', alpha=0.15, label='LG Controller Distributions')

            # Linear Gaussian Controller Means (Dark Red)
            for i in range(mu_eept.shape[1]/3):
                mu = mu_eept[:, 3*i+0:3*i+3]
                self._traj_visualizer.plot_3d_points(i=m, points=mu, linestyle='None', marker='x', markersize=5.0, markeredgewidth=1.0, color=(0.5, 0, 0), alpha=1.0, label='LG Controller Means')

            # Trajectory Samples (Green)
            traj_samples = traj_sample_lists[m].get_samples()
            for sample in traj_samples:
                ee_pt = sample.get(END_EFFECTOR_POINTS)
                for i in range(ee_pt.shape[1]/3):
                    ee_pt_i = ee_pt[:, 3*i+0:3*i+3]
                    self._traj_visualizer.plot_3d_points(m, ee_pt_i, color='green', label='Trajectory Samples')

            # Policy Samples (Blue)
            if pol_sample_lists is not None:
                pol_samples = pol_sample_lists[m].get_samples()
                for sample in pol_samples:
                    ee_pt = sample.get(END_EFFECTOR_POINTS)
                    for i in range(ee_pt.shape[1]/3):
                        ee_pt_i = ee_pt[:, 3*i+0:3*i+3]
                        self._traj_visualizer.plot_3d_points(m, ee_pt_i, color='blue', label='Policy Samples')
        self._traj_visualizer.draw()    # this must be called explicitly
        self._fig.canvas.draw()
        self._fig.canvas.flush_events() # Fixes bug in Qt4Agg backend

    def save_figure(self, filename):
        self._fig.savefig(filename)
