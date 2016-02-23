""" This file interfaces to C++ iLQG-based trajectory optimization. """
from gps.algorithm.traj_opt.traj_opt import TrajOpt


class TrajOptLQR(TrajOpt):
    """ LQR trajectory optimization. """
    def __init__(self, hyperparams, dynamics):
        TrajOpt.__init__(self, hyperparams, dynamics)

    def update(self):
        """ Update trajectory distributions. """
        # TODO: Implement this in C++, and use Boost. This is just a
        #       placeholder.
        raise NotImplementedError('TODO')
