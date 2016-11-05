"""
Most of this code is taken from
the colorama library
"""
import logging
import sys
import traceback as tb

COLOR_CODES = {
    'utility': 'yellow',
    'algorithm': 'lblue',
    'traj_opt': 'lgreen',
    'dynamics': 'lblue',
    'default': 'white',
}


CSI = '\033['
OSC = '\033]'
BEL = '\007'


def code_to_chars(code):
    return CSI + str(code) + 'm'


def set_title(title):
    return OSC + '2;' + title + BEL


def clear_screen(mode=2):
    return CSI + str(mode) + 'J'


def clear_line(mode=2):
    return CSI + str(mode) + 'K'


class AnsiCodes(object):
    def __init__(self):
        # the subclasses declare class attributes which are numbers.
        # Upon instantiation we define instance attributes, which are the same
        # as the class attributes but wrapped with the ANSI escape sequence
        for name in dir(self):
            if not name.startswith('_'):
                value = getattr(self, name)
                setattr(self, name, code_to_chars(value))


class AnsiCursor(object):
    def UP(self, n=1):
        return CSI + str(n) + 'A'

    def DOWN(self, n=1):
        return CSI + str(n) + 'B'

    def FORWARD(self, n=1):
        return CSI + str(n) + 'C'

    def BACK(self, n=1):
        return CSI + str(n) + 'D'

    def POS(self, x=1, y=1):
        return CSI + str(y) + ';' + str(x) + 'H'


class AnsiFore(AnsiCodes):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    RESET = 39

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX = 90
    LIGHTRED_EX = 91
    LIGHTGREEN_EX = 92
    LIGHTYELLOW_EX = 93
    LIGHTBLUE_EX = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX = 96
    LIGHTWHITE_EX = 97


class AnsiBack(AnsiCodes):
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    MAGENTA = 45
    CYAN = 46
    WHITE = 47
    RESET = 49

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX = 100
    LIGHTRED_EX = 101
    LIGHTGREEN_EX = 102
    LIGHTYELLOW_EX = 103
    LIGHTBLUE_EX = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX = 106
    LIGHTWHITE_EX = 107


class AnsiStyle(AnsiCodes):
    BRIGHT = 1
    DIM = 2
    NORMAL = 22
    RESET_ALL = 0


Fore = AnsiFore()
Back = AnsiBack()
Style = AnsiStyle()
Cursor = AnsiCursor()

_COLOR_MAP = {
    'red': (Fore.RED, Back.RED),
    'blue': (Fore.BLUE, Back.BLUE),
    'green': (Fore.GREEN, Back.GREEN),
    'white': (Fore.WHITE, Back.WHITE),
    'black': (Fore.BLACK, Back.BLACK),
    'yellow': (Fore.YELLOW, Back.YELLOW),
    'magenta': (Fore.MAGENTA, Back.MAGENTA),
    'cyan': (Fore.CYAN, Back.CYAN),
    'gray': (Fore.LIGHTBLACK_EX, Back.LIGHTBLACK_EX),
    'reset': (Fore.RESET, Back.RESET),
    'lred': (Fore.LIGHTRED_EX, Back.LIGHTRED_EX),
    'lblue': (Fore.LIGHTBLUE_EX, Back.LIGHTBLUE_EX),
    'lgreen': (Fore.LIGHTGREEN_EX, Back.LIGHTGREEN_EX),
    'lwhite': (Fore.LIGHTWHITE_EX, Back.LIGHTWHITE_EX),
    'lblack': (Fore.LIGHTBLACK_EX, Back.LIGHTBLACK_EX),
    'lyellow': (Fore.LIGHTYELLOW_EX, Back.LIGHTYELLOW_EX),
    'lmagenta': (Fore.LIGHTMAGENTA_EX, Back.LIGHTMAGENTA_EX),
    'lcyan': (Fore.LIGHTCYAN_EX, Back.LIGHTCYAN_EX),
    None: (Fore.RESET, Back.RESET)
}


def print_color(text, fore=None, back=None, reset=True, outstream=sys.stdout):
    """
    Prints text in the specified colors

    Color codes:
        red, blue, green, white, black, yellow, magenta,
        cyan, reset

    :param text: A string to print
    :param fore: A string color code for the foreground.
    :param back: A string color code for the background.
    :param reset: (Default True) Whether to restore colors back to defaults
        after printing.
    :return: None
    """
    reset_ = Fore.RESET + Back.RESET if reset else ''
    outstream.write(_COLOR_MAP[fore][0] + _COLOR_MAP[back][1] + text + reset_)


def cursorup():
    print Cursor.UP()


def cursorl():
    print Cursor.BACK()


def cursorr():
    print Cursor.FORWARD()


def get_color_code(fname):
    for code_dir in COLOR_CODES:
        if code_dir in fname:
            return COLOR_CODES[code_dir]
    return COLOR_CODES['default']


def color_string(msg, color=None):
    if color==None:
        fname, lineno, method, _ = tb.extract_stack()[-2]  # Get caller
        color = get_color_code(fname)
    return _COLOR_MAP[color][0] + msg + Fore.RESET


class ColorLogger(object):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(name)

    def info(self, msg, *frmat):
        msg = color_string(msg % frmat, color=get_color_code(self.name))
        self.logger.info(msg)

    def debug(self, msg, *frmat):
        msg = color_string(msg % frmat, color=get_color_code(self.name))
        self.logger.debug(msg)

    def warning(self, msg, *frmat):
        msg = color_string(msg % frmat, color='red')
        self.logger.warning(msg)
