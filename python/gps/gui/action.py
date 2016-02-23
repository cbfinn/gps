""" This file defines the action class. """
#TODO: Should this be a BundleType?
class Action(object):
    """ Action class. """
    def __init__(self, key, name, func, axis_pos=None, keyboard_binding=None,
                 ps3_binding=None):
        self._key = key
        self._name = name
        self._func = func
        self._axis_pos = axis_pos
        self._kb = keyboard_binding
        self._pb = ps3_binding
