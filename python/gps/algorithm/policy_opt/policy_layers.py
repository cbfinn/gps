""" This file defines a few useful custom Caffe layers. """
import json

import caffe


class PolicyDataLayer(caffe.Layer):
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


class WeightedEuclideanLoss(caffe.Layer):
    """ A weighted Euclidean loss layer. """
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        batch_size = bottom[0].data.shape[0]
        self.diff_ = bottom[0].data - bottom[1].data
        loss = 0.0
        for i in range(batch_size):
            loss += self.diff_[i].dot(bottom[2].data[i].dot(self.diff_[i]))
        top[0].data[...] = loss / 2.0 / batch_size

    def backward(self, top, propagate_down, bottom):
        batch_size = bottom[0].shape[0]
        for i in range(2):
            if propagate_down[i]:
                sign = 1 if i == 0 else -1
                alpha = sign * top[0].diff[0] / batch_size
                for j in range(batch_size):
                    bottom[i].diff[j] = bottom[2].data[j].dot(self.diff_[j])
                bottom[i].diff[...] *= alpha
