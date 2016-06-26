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


class IOCLoss(caffe.Layer):
    """ IOC loss layer, based on MaxEnt IOC with sampling. """
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1)
        self.num_demos = bottom[0].data.shape[0]
        self.num_samples = bottom[1].data.shape[0]
        self.T = bottom[0].data.shape[1]

        # TODO(kevin) construct helper numpy arrays to store demo_counts_
        # and sample_counts_  (as in ioc_loss_layer.cpp)
        self.demo_counts_ = np.zeros((self.num_demos))
        self.sample_counts_ = np.zeros((self.num_samples))


    def forward(self, bottom, top):
        # safely compute forward pass (objective from the input)

        # assume that bottom[0] is the demo features in a NdxTxF matrix where F
        # is the number of features in the last layer, and bottom[1] stores
        # the sample features
        # also assume that bottom[2] is demo log importance weights and
        # bottom[3] is sample log importance weights

        # TODO(kevin) implement this (look at ioc_loss_layer.cpp in caffe)
        # python/gps/algorithm/policy_opt/policy_layers.py might also be helpful
        loss = 0.0
        dc = self.demo_counts_
        sc = self.sample_counts_

        # log importance weights of demos and samples.
        d_log_iw = bottom[2].data
        s_log_iw = bottom[3].data

        # Sum over time and compute max value for safe logsum.
        for i in xrange(self.num_demos):
            dc[i] = 0.0
            for t in xrange(self.T):
                dc[i] += 0.5 * bottom[0].data[i, t, 0] # Not sure why the feature should be 0
            loss += dc[i]
            # Add importance weight to demo feature count. Will be negated.
            dc[i] += d_log_iw[i]
        # Divide by number of demos.
        loss /= self.num_demos

        max_val = -dc[0]
        for i in xrange(self.num_samples):
            sc[i] = 0.0
            for t in xrange(self.T):
                sc[i] += 0.5 * bottom[1].data[i, t, 0]
            # Add importance weight to demo feature count. Will be negated.
            sc[i] += s_log_iw[i]
            if -sc[i] > max_val:
                max_val = -sc[i]

        # Do a safe log-sum-exp operation.
        append_demos = self.layer_params['append_demos'] # We need to set this somewhere?
        if append_demos:
            for i in xrange(self.num_demos):
                if -dc[i] > max_val:
                    max_val = -dc[i]
            for i in xrange(self.num_demos):
                dc[i] = -dc[i] - max_val
        for i in xrange(self.num_samples):
            sc[i] = -sc[i] - max_val
        dc = np.exp(dc)
        sc = np.exp(sc)

        self.partition = 0.0
        if append_demos:
            self.partition += np.sum(dc, axis = 0)
        self.partition += np.sum(sc, axis = 0)
        loss += np.log(self.partition) + max_val
        top[0].data[...] = loss
        # Need to increment iteration here?

    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        batch_size = bottom[0].shape[0] # Do we need to use this here?
        # TODO(kevin) implement this (look at ioc_loss_layer.cpp in caffe)
        # python/gps/algorithm/policy_opt/policy_layers.py might also be helpful
        loss_weight = 0.5 * top[0].diff[0]
        dc = self.demo_counts_
        sc = self.sample_counts_
        append_demos = self.layer_params['append_demos'] # We need to set this somewhere?
        # Compute gradient w.r.t demos
        demo_bottom_diff = bottom[0].diff
        sample_bottom_diff = bottom[1].diff

        for i in xrange(self.num_demos):
            for t in xrange(self.T):
                demo_bottom_diff[i * self.T + t] = (1.0 / self.num_demos)
                if append_demos:
                    demo_bottom_diff[i * self.T + t] -= (dc[i] / self.partition)

        for i in xrange(self.num_samples):
            for t in xrange(self.T):
                sample_bottom_diff[i * self.T + t] = (-sc[i] / self.partition)

        bottom[0].diff = demo_bottom_diff * loss_weight
        bottom[1].diff = sample_bottom_diff * loss_weight