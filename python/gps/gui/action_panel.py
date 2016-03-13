""" This file defines the action axis class. """
import itertools
import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from gps.gui.config import config


LOGGER = logging.getLogger(__name__)


class Action:
    def __init__(self, key, name, func, axis_pos=None, keyboard_binding=None, ps3_binding=None):
        self._key = key
        self._name = name
        self._func = func
        self._axis_pos = axis_pos
        self._kb = keyboard_binding
        self._pb = ps3_binding

class ActionPanel:
    def __init__(self, fig, gs, rows, cols, actions):
        """
        Constructs an ActionPanel assuming actions is a dictionary of
        fully initialized actions.
        Each action must have: key, name, func.
        Each action can have: axis_pos, keyboard_binding, ps3_binding.
        """
        assert len(actions) <= rows*cols, 'Too many actions to put into gridspec.'

        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs)
        self._axarr = [plt.subplot(self._gs[i]) for i in range(len(actions))]
        self._actions = actions

        # Try to import ROS.
        ros_enabled = False
        try:
            import rospkg
            import rospy
            import roslib
            from sensor_msgs.msg import Joy

            roslib.load_manifest('gps_agent_pkg')
            ros_enabled = True
        except ImportError as e:
            LOGGER.debug('PS3 not enabled: %s', e)
        except rospkg.common.ResourceNotFound as e:
            LOGGER.debug('No gps_agent_pkg: %s', e)

        # Mouse Input.
        self._buttons = {}
        for key, action in self._actions.iteritems():
            if action._axis_pos is not None:
                if ros_enabled:
                    if (config['inverted_ps3_button'] is not None and
                            action._pb is not None):
                        ps3_bindings_str = ',\n'.join([
                            config['inverted_ps3_button'][i] for i in action._pb
                        ])
                    else:
                        ps3_bindings_str = str(action._pb)
                    button_name = '%s\n(%s)\n(%s)' % \
                            (action._name, action._kb, ps3_bindings_str)
                else:
                    button_name = '%s\n(%s)' % (action._name, action._kb)
                self._buttons[key] = Button(self._axarr[action._axis_pos],
                                            button_name)
                self._buttons[key].on_clicked(action._func)

        # Keyboard Input.
        self._keyboard_bindings = {}
        for key, action in self._actions.iteritems():
            if action._kb is not None:
                self._keyboard_bindings[action._kb] = key
        self._cid = self._fig.canvas.mpl_connect('key_press_event',
                                                 self.on_key_press)

        # PS3 Input using ROS.
        if ros_enabled:
            self._ps3_bindings = {}
            for key, action in self._actions.iteritems():
                if action._pb is not None:
                    self._ps3_bindings[action._pb] = key
            for key, value in list(self._ps3_bindings.iteritems()):
                for permuted_key in itertools.permutations(key, len(key)):
                    self._ps3_bindings[permuted_key] = value
            self._ps3_count = 0
            self._ps3_process_rate = config['ps3_process_rate']
            rospy.Subscriber(config['ps3_topic'], Joy, self.ps3_callback)

    #TODO: Docstrings here.
    def on_key_press(self, event):
        if event.key in self._keyboard_bindings:
            self._actions[self._keyboard_bindings[event.key]]._func()
        else:
            LOGGER.debug('Unrecognized keyboard input: %s', str(event.key))

    def ps3_callback(self, joy_msg):
        self._ps3_count += 1
        if self._ps3_count % config['ps3_process_rate'] != 0:
            return
        buttons_pressed = tuple([
            i for i in range(len(joy_msg.buttons)) if joy_msg.buttons[i]
        ])
        if buttons_pressed in self._ps3_bindings:
            self._actions[self._ps3_bindings[buttons_pressed]]._func()
        else:
            if not (len(buttons_pressed) == 0 or (len(buttons_pressed) == 1 and
                    (buttons_pressed[0] == self._ps3_button['rear_right_1'] or
                    buttons_pressed[0] == self._ps3_button['rear_right_2']))):
                LOGGER.debug('Unrecognized ps3 controller input:\n%s',
                        str([config['inverted_ps3_button'][b] for b in buttons_pressed]))
