""" Default configuration and hyperparameter values for GUI objects. """
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

# PS3 Joystick Buttons and Axes
# (documentation: http://wiki.ros.org/ps3joy).
# Mappings from PS3 buttons to their corresponding array indices.
PS3_BUTTON = {
    'select': 0,
    'stick_left': 1,
    'stick_right': 2,
    'start': 3,
    'cross_up': 4,
    'cross_right': 5,
    'cross_down': 6,
    'cross_left': 7,
    'rear_left_2': 8,
    'rear_right_2': 9,
    'rear_left_1': 10,
    'rear_right_1': 11,
    'action_triangle': 12,
    'action_circle': 13,
    'action_cross': 14,
    'action_square': 15,
    'pairing': 16,
}
INVERTED_PS3_BUTTON = {value: key for key, value in PS3_BUTTON.iteritems()}

# Mappings from PS3 axes to their corresponding array indices.
PS3_AXIS = {
    'stick_left_leftwards': 0,
    'stick_left_upwards': 1,
    'stick_right_leftwards': 2,
    'stick_right_upwards': 3,
    'button_cross_up': 4,
    'button_cross_right': 5,
    'button_cross_down': 6,
    'button_cross_left': 7,
    'button_rear_left_2': 8,
    'button_rear_right_2': 9,
    'button_rear_left_1': 10,
    'button_rear_right_1': 11,
    'button_action_triangle': 12,
    'button_action_circle': 13,
    'button_action_cross': 14,
    'button_action_square': 15,
    'acceleratometer_left': 16,
    'acceleratometer_forward': 17,
    'acceleratometer_up': 18,
    'gyro_yaw': 19,
}
INVERTED_PS3_AXIS = {value: key for key, value in PS3_AXIS.iteritems()}

# Mappings from actions to their corresponding keyboard bindings.
keyboard_bindings = {
    # Target Setup.
    'ptn': 'left',
    'ntn': 'right',
    'pat': 'down',
    'nat': 'up',

    'sip': 'j',
    'stp': 'k',
    'sii': 'l',
    'sti': ';',

    'mti': 'u',
    'mtt': 'i',
    'rc': 'o',
    'mm': 'p',

    # GPS Training.
    'stop' : 's',
    'reset': 'r',
    'go'   : 'g',
    'fail' : 'f',

    # Image Visualizer
    'oii'  : 'i',
    'oti'  : 't',
}
inverted_keyboard_bindings = {value: key
                              for key, value in keyboard_bindings.iteritems()}

# Mappings from actions to their corresponding PS3 controller bindings.
ps3_bindings = {
    # Target Setup
    'ptn': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_left']),
    'ntn': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_right']),
    'pat': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_down']),
    'nat': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_up']),

    'sip': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_square']),
    'stp': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_circle']),
    'sii': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_cross']),
    'sti': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_triangle']),

    'mti': (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_left']),
    'mtt': (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_right']),
    'rc' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_down']),
    'mm' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_up']),

    # GPS Training
    'stop' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_square']),
    'reset': (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_triangle']),
    'go'   : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_circle']),
    'fail' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_cross']),

    # Image Visualizer
    'oii'  : (PS3_BUTTON['cross_up']    ,),
    'oti'  : (PS3_BUTTON['cross_down']  ,),
}
inverted_ps3_bindings = {value: key for key, value in ps3_bindings.iteritems()}

config = {
    'ps3_topic': 'joy',
    'ps3_process_rate': 20,  # Only process 1/20 of PS3 messages.
    'ps3_button': PS3_BUTTON,
    'inverted_ps3_button': INVERTED_PS3_BUTTON,
    'ps3_axis': PS3_AXIS,
    'inverted_ps3_ax': INVERTED_PS3_AXIS,

    'keyboard_bindings': keyboard_bindings,
    'inverted_keyboard_bindings': inverted_keyboard_bindings,
    'ps3_bindings': ps3_bindings,
    'inverted_ps3_bindings': inverted_ps3_bindings,

    'num_targets': 10,
    'actuator_types': [TRIAL_ARM, AUXILIARY_ARM],
    'actuator_names': ['trial_arm', 'auxiliary_arm'],

    'image_topic': '/camera/rgb/image_color',
    'image_actuator': 'trial_arm',
}