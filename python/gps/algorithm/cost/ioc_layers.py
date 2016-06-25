""" This file defines a few useful custom Caffe layers for IOC. """
import json

import caffe


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
        self.num_samples = bottom[0].data.shape[0]
        self.T = bottom[0].data.shape[1]

        # TODO(kevin) construct helper numpy arrays to store demo_counts_
        # and sample_counts_  (as in ioc_loss_layer.cpp)

    def forward(self, bottom, top):
        # safely compute forward pass (objective from the input)

        # assume that bottom[0] is the demo features in a NdxTxF matrix where F
        # is the number of features in the last layer, and bottom[1] stores
        # the sample features
        # also assume that bottom[2] is demo log importance weights and
        # bottom[3] is sample log importance weights

        # TODO(kevin) implement this (look at ioc_loss_layer.cpp in caffe)
        # python/gps/algorithm/policy_opt/policy_layers.py might also be helpful

    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        batch_size = bottom[0].shape[0]
        # TODO(kevin) implement this (look at ioc_loss_layer.cpp in caffe)
        # python/gps/algorithm/policy_opt/policy_layers.py might also be helpful
