""" Default configuration and hyperparameter values for GUI objects. """
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

#TODO: These should probably be all caps?

# PS3 Joystick Buttons and Axes
# (documentation: http://wiki.ros.org/ps3joy).
# Mappings from PS3 buttons to their corresponding array indices.
ps3_button = {
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
inverted_ps3_button = {value: key for key, value in ps3_button.iteritems()}

# Mappings from PS3 axes to their corresponding array indices.
ps3_axis = {
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
inverted_ps3_axis = {value: key for key, value in ps3_axis.iteritems()}

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
    'ptn': (ps3_button['rear_right_1'], ps3_button['cross_left']),
    'ntn': (ps3_button['rear_right_1'], ps3_button['cross_right']),
    'pat': (ps3_button['rear_right_1'], ps3_button['cross_down']),
    'nat': (ps3_button['rear_right_1'], ps3_button['cross_up']),

    'sip': (ps3_button['rear_right_1'], ps3_button['action_square']),
    'stp': (ps3_button['rear_right_1'], ps3_button['action_circle']),
    'sii': (ps3_button['rear_right_1'], ps3_button['action_cross']),
    'sti': (ps3_button['rear_right_1'], ps3_button['action_triangle']),

    'mti': (ps3_button['rear_right_2'], ps3_button['cross_left']),
    'mtt': (ps3_button['rear_right_2'], ps3_button['cross_right']),
    'rc' : (ps3_button['rear_right_2'], ps3_button['cross_down']),
    'mm' : (ps3_button['rear_right_2'], ps3_button['cross_up']),

    # GPS Training
    'stop' : (ps3_button['rear_right_2'], ps3_button['action_square']),
    'reset': (ps3_button['rear_right_2'], ps3_button['action_triangle']),
    'go'   : (ps3_button['rear_right_2'], ps3_button['action_circle']),
    'fail' : (ps3_button['rear_right_2'], ps3_button['action_cross']),

    # Image Visualizer
    'oii'  : (ps3_button['cross_up']    ,),
    'oti'  : (ps3_button['cross_down']  ,),
}
inverted_ps3_bindings = {value: key for key, value in ps3_bindings.iteritems()}

common = {
    'ps3_button': ps3_button,
    'inverted_ps3_button': inverted_ps3_button,
    'ps3_axis': ps3_axis,
    'inverted_ps3_ax': inverted_ps3_axis,

    'keyboard_bindings': keyboard_bindings,
    'inverted_keyboard_bindings': inverted_keyboard_bindings,
    'ps3_bindings': ps3_bindings,
    'inverted_ps3_bindings': inverted_ps3_bindings,

    'ps3_topic': 'joy',
    'ps3_process_rate': 20,  # Only process 1/20 of PS3 messages.

    'image_topic': '/camera/rgb/image_color',
}

target_setup = {
    'num_targets': 10,
    'actuator_types': [TRIAL_ARM, AUXILIARY_ARM],
    'actuator_names': ['trial_arm', 'auxiliary_arm'],

    'target_setup_log_filename': 'target_setup_log.txt',
}

gps_training = {
    'gps_training_log_filename': 'gps_training_log.txt',
    'image_actuator': target_setup['actuator_names'][0],    # which actuator to get initial and target images from
}
