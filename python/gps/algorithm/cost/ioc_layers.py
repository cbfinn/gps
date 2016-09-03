""" This file defines a few useful custom Caffe layers for IOC. """
import json

import caffe

import numpy as np

from gps.utility.general_utils import logsum

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

class GaussianProcessPriors(caffe.Layer):
    """ A Gaussian Process Prior layer, penalizing cost on nearby states. """
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        # Assume bottom[0] contains the (Nd+Ns)xTxdO features and bottom[1] contains the 
        # costs in the batch with zero means and normalized with shape (Nd+Ns)xT.
        self._K = np.zeros((bottom[0].shape[0], bottom[0].shape[0]))
        top[0].reshape(1)

    def forward(self, bottom, top):
        # TODO - make these constants somewhere?
        # l is the length scale and sigma is the noise constant
        l, sigma = 1.0, 1.0 # hand-engineer these. Probably optimize them in the future.
        batch_size = bottom[0].shape[0]
        X = bottom[0].data # feature matrix
        Y = np.zeros((batch_size, 1)) # cost vector

        # Compute the kernel matrix
        for i in xrange(batch_size):
            Y[i] = bottom[1].data[i, :, :].sum()
            for j in range(i + 1):
                self._K[i, j] = self._K[j, i] = np.exp(-l/2 * (np.linalg.norm(X[i, :, :] - X[j, :, :]))**2)
                if j == i:
                    self.K[i, j] += sigma**2

        top[0].data[...] = -1/2 * (Y.T.dot(np.linalg.pinv(self.K)).dot(Y) + np.log(np.linalg.det(self._K)*2*np.pi))


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
            #dc[i] = 10* 0.5 * np.sum(bottom[0].data[i,:])
            loss += dc[i]
            # Add importance weight to demo feature count. Will be negated.
            #dc[i] += d_log_iw[i] / np.log(3)
            dc[i] += d_log_iw[i]
        # Divide by number of demos.
        loss /= self.num_demos

        max_val = -dc[0]
        for i in xrange(self.num_samples):
            sc[i] = 0.5 * np.sum(bottom[1].data[i,:])
            # Add importance weight to sample feature count. Will be negated.
            #sc[i] += s_log_iw[i] * np.log(5)
            sc[i] += s_log_iw[i]
            if -sc[i] > max_val:
                max_val = -sc[i]

        # Do a safe log-sum-exp operation.
        max_val = np.max((max_val, np.max(-dc)))
        dc = np.exp(-dc - max_val)
        sc = np.exp(-sc - max_val)
        self._partition = np.sum(dc, axis = 0) + np.sum(sc, axis = 0)
        #self._partition = 0.9*np.sum(dc, axis = 0) + 0.1*np.sum(sc, axis = 0)
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
                demo_bottom_diff[i, t] = (1.0 / self.num_demos) - (dc[i] / self._partition)

        for i in xrange(self.num_samples):
            for t in xrange(self.T):
                sample_bottom_diff[i, t] = (-sc[i] / self._partition)

        bottom[0].diff[...] = demo_bottom_diff * loss_weight
        bottom[1].diff[...] = sample_bottom_diff * loss_weight

class IOCLossMod(caffe.Layer):
    """ IOC loss layer, based on MaxEnt IOC with sampling,
        modified importance weights according to Paul's writeup. """
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
        self._d_log_iw = np.zeros((self.num_demos))
        self._s_log_iw = np.zeros((self.num_samples))


    def forward(self, bottom, top):
        # safely compute forward pass (objective from the input)

        # assume that bottom[0] is a NdxT matrix containing the costs of the demo
        # trajectories in at each time step, and bottom[1] stores the costs of samples.
        # also assume that bottom[2] is demo log importance weights and
        # bottom[3] is sample log importance weights
        # bottom[4] is Z (and learned)

        loss = 0.0
        dc = self._demo_counts
        sc = self._sample_counts

        # log importance weights of demos and samples.
        d_log_q = bottom[2].data
        s_log_q = bottom[3].data
        Z_tilde = bottom[4].data[0]

        # Sum over time and compute max value for safe logsum.
        for i in xrange(self.num_demos):
            dc[i] = 0.5 * np.sum(bottom[0].data[i,:])
            loss += dc[i]
            # Add importance weight to demo feature count. Will be negated.
            #self._d_log_iw[i] = d_log_q[i]
            self._d_log_iw[i] = logsum(np.array([d_log_q[i][0], -np.log(Z_tilde)-dc[i]]).reshape((2,1)), 0)
            dc[i] += self._d_log_iw[i]
        # Divide by number of demos.
        loss /= self.num_demos

        max_val = -dc[0]
        for i in xrange(self.num_samples):
            sc[i] = 0.5 * np.sum(bottom[1].data[i,:])
            # Add importance weight to sample feature count. Will be negated.
            #self._s_log_iw[i] = s_log_q[i]
            self._s_log_iw[i] = logsum(np.array([s_log_q[i][0], -np.log(Z_tilde)-sc[i]]).reshape((2,1)), 0)
            sc[i] += self._s_log_iw[i]
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
        if np.isnan(loss):
          import pdb; pdb.set_trace()


    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        loss_weight = top[0].diff[0]
        dc = self._demo_counts
        sc = self._sample_counts
        # Compute gradient w.r.t demos
        demo_bottom_diff = bottom[0].diff
        sample_bottom_diff = bottom[1].diff
        d_log_q = bottom[2].data
        s_log_q = bottom[3].data

        Z_tilde = bottom[4].data[0]
        Z_diff = 0

        for i in xrange(self.num_demos):
            demo_bottom_diff[i, :] = (1.0 / self.num_demos) - (dc[i] / self._partition)
            #max_val = max(-np.sum(bottom[0].data[i,:]), 2*self._d_log_iw[i])
            #Z_diff += np.exp(-2.0*0.5*np.sum(bottom[0].data[i,:])-max_val) / (self._partition * Z_tilde*np.exp(2*self._d_log_iw[i])-max_val)
            cost_i = 0.5*np.sum(bottom[0].data[i,:])
            logq_i = d_log_q[i][0]
            max_val = max(-cost_i, logq_i)
            Z_diff += np.exp(-2*cost_i-2*max_val) / ( self._partition * (Z_tilde*np.exp(logq_i-max_val)+np.exp(-cost_i-max_val))**2 )

        for i in xrange(self.num_samples):
            sample_bottom_diff[i,:] = (-sc[i] / self._partition)
            #max_val = max(-np.sum(bottom[1].data[i,:]), 2*self._s_log_iw[i])
            #Z_diff += np.exp(-2.0*0.5*np.sum(bottom[1].data[i,:])-max_val) / (self._partition * Z_tilde*np.exp(2*self._s_log_iw[i]-max_val))
            cost_i = 0.5*np.sum(bottom[1].data[i,:])
            logq_i = s_log_q[i][0]
            max_val = max(-cost_i, logq_i)
            Z_diff += np.exp(-2*cost_i-2*max_val) / ( self._partition * (Z_tilde*np.exp(logq_i-max_val)+np.exp(-cost_i-max_val))**2 )

        if np.isnan(Z_diff) or np.isinf(Z_diff):
          import pdb; pdb.set_trace()

        bottom[0].diff[...] = demo_bottom_diff * loss_weight
        bottom[1].diff[...] = sample_bottom_diff * loss_weight
        bottom[4].diff[...] = Z_diff * loss_weight


class SigmoidMPFLoss(caffe.Layer):
    """ IOC loss layer, based on MPF objective. """
    def setup(self, bottom, top):
        pass

    def sigmoid(self, input):
        # input is a numpy array
        #max_val = (-input).max()
        #result = np.exp(-max_val) / (np.exp(-max_val) + np.exp(-input-max_val))
        result = 1.0 / (1.0 + np.exp(-input))
        if np.isnan(result).any():
          import pdb; pdb.set_trace()
        return result

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
                pairs[i, j] = (d_log_iw[i] - s_log_iw[j] +
                                0.5 * (np.sum(bottom[0].data[i, :]) - np.sum(bottom[1].data[j, :])))
                #if max_val < pairs[i, j]:
                #    max_val = pairs[i, j]
        #pairs = np.exp(pairs - max_val)
        pairs = self.sigmoid(pairs)
        self._pairs = pairs
        self._pairs_sum = self._pairs.sum()
        top[0].data[...] = self._pairs_sum

    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        pairs = self._pairs
        loss_weight = top[0].diff[0]

        # Compute gradient w.r.t demos and samples
        demo_bottom_diff = bottom[0].diff
        sample_bottom_diff = bottom[1].diff
        for i in xrange(self.num_demos):
            for t in xrange(self.T):
                # extra 0.5 is for the chain rule on the cost
                demo_bottom_diff[i, t] = 0.5 * np.sum(pairs[i, :]*(1-pairs[i,:]))
        for i in xrange(self.num_samples):
            for t in xrange(self.T):
                sample_bottom_diff[i, t] = -0.5 * np.sum(pairs[:, i]*(1-pairs[i,:]))
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
                pairs[i, j] = 0.5 * (d_log_iw[i] - s_log_iw[j] + \
                                0.5 * (np.sum(bottom[0].data[i, :]) - np.sum(bottom[1].data[j, :])))
                if max_val < pairs[i, j]:
                    max_val = pairs[i, j]
        pairs = np.exp(pairs - max_val)
        #pairs = np.exp(pairs)
        self._pairs = pairs # NOTE - this has the max subtracted
        self._pairs_sum = self._pairs.sum()*np.exp(max_val)
        self.max_val = max_val
        top[0].data[...] = self._pairs_sum

    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        pairs = self._pairs
        loss_weight = top[0].diff[0]

        # Compute gradient w.r.t demos and samples
        demo_bottom_diff = bottom[0].diff
        sample_bottom_diff = bottom[1].diff
        for i in xrange(self.num_demos):
            for t in xrange(self.T):
                # extra 0.5 is for the chain rule on the cost
                demo_bottom_diff[i, t] = 0.5 * 0.5 * np.sum(pairs[i, :])*np.exp(self.max_val)
        for i in xrange(self.num_samples):
            for t in xrange(self.T):
                sample_bottom_diff[i, t] = -0.5 * 0.5 * np.sum(pairs[:, i])*np.exp(self.max_val)
        bottom[0].diff[...] = demo_bottom_diff * loss_weight
        bottom[1].diff[...] = sample_bottom_diff * loss_weight


class LogMPFLoss(caffe.Layer):
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
                pairs[i, j] = 0.5 * (d_log_iw[i] - s_log_iw[j] + \
                                0.5 * (np.sum(bottom[0].data[i, :]) - np.sum(bottom[1].data[j, :])))
                if max_val < pairs[i, j]:
                    max_val = pairs[i, j]
        pairs = np.exp(pairs - max_val)
        self._pairs = pairs
        self._pairs_sum = self._pairs.sum()
        top[0].data[...] = np.log(self._pairs_sum) + max_val

    def backward(self, top, propagate_down, bottom):
        # compute backward pass (derivative of objective w.r.t. bottom)
        pairs = self._pairs
        loss_weight = top[0].diff[0]

        # Compute gradient w.r.t demos and samples
        demo_bottom_diff = bottom[0].diff
        sample_bottom_diff = bottom[1].diff
        for i in xrange(self.num_demos):
            for t in xrange(self.T):
                # extra 0.5 is for the chain rule on the cost
                demo_bottom_diff[i, t] = 0.5 * 0.5 * np.sum(pairs[i, :]) / self._pairs_sum
        for i in xrange(self.num_samples):
            for t in xrange(self.T):
                sample_bottom_diff[i, t] = -0.5 * 0.5 * np.sum(pairs[:, i]) / self._pairs_sum
        bottom[0].diff[...] = demo_bottom_diff * loss_weight
        bottom[1].diff[...] = sample_bottom_diff * loss_weight


