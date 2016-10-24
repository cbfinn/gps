""" This file defines the data logger. """
import logging
try:
   import cPickle as pickle
except:
   import pickle
import gzip
import contextlib


LOGGER = logging.getLogger(__name__)

@contextlib.contextmanager
def open_zip(filename, mode='r'):
    """
    Open a file; if filename ends with .gz, opens as a gzip file
    """
    if filename.endswith('.gz'):
        openfn = gzip.open
    else:
        openfn = open
    yield openfn(filename, mode)



class DataLogger(object):
    """
    This class pickles data into files and unpickles data from files.
    TODO: Handle logging text to terminal, GUI text, and/or log file at
        DEBUG, INFO, WARN, ERROR, FATAL levels.
    TODO: Handle logging data to terminal, GUI text/plots, and/or data
          files.
    """
    def __init__(self):
        pass

    def pickle(self, filename, data):
        """ Pickle data into file specified by filename. """
        with open_zip(filename, 'wb') as f:
            pickle.dump(data, f)

    def unpickle(self, filename):
        """ Unpickle data from file specified by filename. """
        try:
            with open_zip(filename, 'rb') as f:
                result = pickle.load(f)
            return result
        except IOError:
            LOGGER.debug('Unpickle error. Cannot find file: %s', filename)
            return None
