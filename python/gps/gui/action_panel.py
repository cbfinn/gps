""" This file defines the action axis class. """
import itertools
import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from gps.gui.config import config

ros_enabled = False
try:
    import rospkg
    import rospy
    import roslib
    from sensor_msgs.msg import Joy

    roslib.load_manifest('gps_agent_pkg')
    ros_enabled = True
except ImportError as e:
    LOGGER.debug('Import ROS failed: %s', e)
except rospkg.common.ResourceNotFound as e:
    LOGGER.debug('No gps_agent_pkg: %s', e)


class Action:
    def __init__(self, key, name, func, axis_pos=None, keyboard_binding=None, ps3_binding=None):
        self._key = key
        self._name = name
        self._func = func
        self._axis_pos = axis_pos
        self._kb = keyboard_binding
        self._pb = ps3_binding

class ActionPanel:
    def __init__(self, fig, gs, rows, cols, actions_arr):
        """
        Constructs an ActionPanel assuming actions is a dictionary of
        fully initialized actions.
        Each action must have: key, name, func.
        Each action can have: axis_pos, keyboard_binding, ps3_binding.
        """
        assert len(actions_arr) <= rows*cols, 'Too many actions to put into gridspec.'

        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs)
        self._axarr = [plt.subplot(self._gs[i]) for i in range(len(actions_arr))]
        
        self._actions = {action._key: action for action in actions_arr}
        for key, action in self._actions.iteritems():
            if key in config['keyboard_bindings']:
                action._kb = config['keyboard_bindings'][key]
            if key in config['ps3_bindings']:
                action._pb = config['ps3_bindings'][key]

        self._initialize_buttons()
        self._cid = self._fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        if ros_enabled:
            self._ps3_count = 0
            rospy.Subscriber(config['ps3_topic'], Joy, self.ps3_callback)

    #TODO: Docstrings here.
    def _initialize_buttons(self):
        self._buttons = {}
        for key, action in self._actions.iteritems():
            if action._axis_pos is None:
                continue
            
            button_name = '%s\n(%s)' % (action._name, action._kb)
            if ros_enabled and action._pb:
                ps3_buttons = [config['inverted_ps3_button'][i] for i in action._pb]
                button_name += '\n(%s)' % ',\n'.join(ps3_buttons)

            self._buttons[key] = Button(self._axarr[action._axis_pos], button_name)
            self._buttons[key].on_clicked(action._func)

    def on_key_press(self, event):
        if event.key in config['inverted_keyboard_bindings']:
            key = config['inverted_keyboard_bindings'][event.key]
            if key in self._actions:
                self._actions[key]._func()
        else:
            LOGGER.debug('Unrecognized keyboard input: %s', str(event.key))

    def ps3_callback(self, joy_msg):
        self._ps3_count += 1
        if self._ps3_count % config['ps3_process_rate'] != 0:
            return
        
        buttons_pressed = tuple(np.nonzero(joy_msg.buttons)[0])
        if buttons_pressed in config['permuted_inverted_ps3_bindings']:
            self._actions[config['permuted_inverted_ps3_bindings'][buttons_pressed]]._func()
        else:
            if ((len(buttons_pressed) == 1 and buttons_pressed[0] not in (
                config['ps3_button']['rear_right_1'], config['ps3_button']['rear_right_1'])) 
                or num_buttons >= 2):
                LOGGER.debug('Unrecognized ps3 controller input:\n%s',
                        str([config['inverted_ps3_button'][b] for b in buttons_pressed]))
