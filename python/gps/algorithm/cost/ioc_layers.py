""" This file defines a few useful custom Caffe layers for IOC. """
import json

import caffe

import numpy as np


# TODO - this is copied from policy layers. should share code.
class IOCDataLayer(caffe.Layer):
    """ A layer for passing data into the network at training time. """
    def setup(self, bottom, top):
        info = json.loads(self.param_str)
        for ind, top_blob in enumerate(info['shape']):
            top[ind].reshape(*top_blob['dim'])

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # Nothing to do - data will already be set externally.
        # TODO - Maybe later include way to pass data to this layer and
        #        handle batching here.
        pass

    def backward(self, top, propagate_down, bottom):
        pass

class L2MonotonicLoss(caffe.Layer):
    """ A monotonic loss layer, similar to a hinge loss. """
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self._temp = np.zeros(bottom[0].shape)
        assert(bottom[0].shape[1] == 1)
        top[0].reshape(1)

    def forward(self, bottom, top):
        # TODO - make this a constant somewhere?
        offset = 1.0 # TODO - acc. to the paper, this should be -1?
        bottom_data = bottom[0].data
        batch_size = bottom[0].shape[0]

        for i in range(batch_size):
            self._temp[i] = np.maximum(0.0, bottom_data[i] + offset)

        top[0].data[...] = (self._temp*self._temp).sum() / batch_size

    def backward(self, top, propagate_down, bottom):
        loss_weight = top[0].diff[0]
        batch_size = bottom[0].shape[0]
        bottom[0].diff[...] = 2.0 * loss_weight * self._temp / batch_size
        # This is gradient of l1 loss
        # bottom[0].diff = loss_weight * np.sign(self._temp) / batch_size


class IOCLoss(caffe.Layer):
    """ IOC loss layer, based on MaxEnt IOC with sampling. """
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1)
        self.num_demos = bottom[0].data.shape[0]
        self.num_samples = bottom[1].data.shape[0]
        self.T = bottom[0].data.shape[1]

        # helper numpy arrays to store demo_counts and sample_counts
        self._demo_counts = np.zeros((self.num_demos))
        self._sample_counts = np.zeros((self.num_samples))


    def forward(self, bottom, top):
        # safely compute forward pass (objective from the input)

        # assume that bottom[0] is a NdxT matrix containing the costs of the demo
        # trajectories in at each time step, and bottom[1] stores the costs of samples.
        # also assume that bottom[2] is demo log importance weights and
        # bottom[3] is sample log importance weights

        loss = 0.0
        dc = self._demo_counts
        sc = self._sample_counts

        # log importance weights of demos and samples.
        d_log_iw = bottom[2].data
        s_log_iw = bottom[3].data

        # Sum over time and compute max value for safe logsum.
        for i in xrange(self.num_demos):
            dc[i] = 0.5 * np.sum(bottom[0].data[i,:])
            loss += dc[i]
            # Add importance weight to demo feature count. Will be negated.
            dc[i] += d_log_iw[i]
        # Divide by number of demos.
        loss /= self.num_demos

        max_val = -dc[0]
        for i in xrange(self.num_samples):
            sc[i] = 0.5 * np.sum(bottom[1].data[i,:])
            # Add importance weight to sample feature count. Will be negated.
            sc[i] += s_log_iw[i]
            if -sc[i] > max_val:
                max_val = -sc[i]
        # Do a safe log-sum-exp operation.
        max_val = np.max((max_val, np.max(-dc)))
        dc = np.exp(-dc - max_val)
        sc = np.exp(-sc - max_val)
        self._partition = np.sum(dc, axis = 0) + np.sum(sc, axis = 0)
        loss += np.log(self._partition) + max_val
        top[0].data[...] = loss
        self._demo_counts = dc
        self._sample_counts = sc


    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        loss_weight = 0.5 * top[0].diff[0]
        dc = self._demo_counts
        sc = self._sample_counts
        # Compute gradient w.r.t demos
        demo_bottom_diff = bottom[0].diff
        sample_bottom_diff = bottom[1].diff

        for i in xrange(self.num_demos):
            for t in xrange(self.T):
                demo_bottom_diff[i, t] = (1.0 / self.num_demos - (dc[i] / self._partition))

        for i in xrange(self.num_samples):
            for t in xrange(self.T):
                sample_bottom_diff[i, t] = (-sc[i] / self._partition)

        bottom[0].diff[...] = demo_bottom_diff * loss_weight
        bottom[1].diff[...] = sample_bottom_diff * loss_weight


class MPFLoss(caffe.Layer):
    """ IOC loss layer, based on MPF objective. """
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1)
        self.num_demos = bottom[0].data.shape[0]
        self.num_samples = bottom[1].data.shape[0]
        self.T = bottom[0].data.shape[1]

        # helper numpy arrays to store demo_counts and sample_counts
        self._pairs = np.zeros((self.num_demos, self.num_samples))

    def forward(self, bottom, top):
        # safely compute forward pass (objective from the input)

        # assume that bottom[0] is a NdxT matrix containing the costs of the demo
        # trajectories in at each time step, and bottom[1] stores the costs of samples.
        # also assume that bottom[2] is demo log importance weights and
        # bottom[3] is sample log importance weights
        loss = 0.0
        pairs = self._pairs

        # log importance weights of demos and samples.
        d_log_iw = bottom[2].data
        s_log_iw = bottom[3].data

        max_val = -np.inf
        for i in xrange(self.num_demos):
            for j in xrange(self.num_samples):
                pairs[i, j] = np.exp(0.5 * (np.log(d_log_iw[i]) - np.log(s_log_iw[j]) + \
                                0.5 * (np.sum(bottom[0].data[i, :]) - np.sum(bottom[1].data[j, :]))))
                loss += pairs[i, j]
        self._pairs = pairs
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        pairs = self._pairs
        loss_weight = 0.25 * top[0].diff[0] #0.5*0.5?

        # Compute gradient w.r.t demos and samples
        demo_bottom_diff = bottom[0].diff
        sample_bottom_diff = bottom[1].diff
        for i in xrange(self.num_demos):
            for t in xrange(self.T):
                demo_bottom_diff[i, t] = np.sum(pairs[i, :])
        for i in xrange(self.num_samples):
            for t in xrange(self.T):
                sample_bottom_diff[i, t] = -np.sum(pairs[:, i])

        bottom[0].diff[...] = demo_bottom_diff * loss_weight
        bottom[1].diff[...] = sample_bottom_diff * loss_weight



