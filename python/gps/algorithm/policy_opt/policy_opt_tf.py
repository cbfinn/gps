""" This file defines policy optimization for a Caffe policy. """
import copy
import logging
import tempfile

import numpy as np

import tensorflow as tf

from google.protobuf.text_format import MessageToString

#from gps.algorithm.policy.tf_policy import TfPolicy
#from gps.algorithm.policy_opt.policy_opt import PolicyOpt
#from gps.algorithm.policy_opt.config import POLICY_OPT_Tf

LOGGER = logging.getLogger(__name__)


""" This file defines the base class for the policy. """
import abc


def init_weights(shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    #ToDo: This should be zeros.
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def euclidean_loss_layer(a, b, precision):
    return tf.matmul(tf.matmul(a-b, precision, transpose_a=True), a-b)


def get_input_layer(phase, batch_size, dim_input, dim_output):

    net_input = tf.placeholder("float", [batch_size, dim_input], name='nn_input')
    action = None
    precision = None
    if phase == 'TRAIN':
        action = tf.placeholder('float', [batch_size, dim_output])
        precision = tf.placeholder('float', [batch_size, dim_output, dim_output])

    return net_input, action, precision


def get_mlp_layers(mlp_input, number_layers, dimension_hidden):
    cur_top = mlp_input
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape.dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b' + str(layer_step))
        if layer_step != number_layers:  # final layer has no relu
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top


def get_loss_layer(phase, mlp_out, action, precision):
    return euclidean_loss_layer(a=action, b=mlp_out, precision=precision)
    #if phase == 'TRAIN':
    #    out = euclidean_loss_layer(a=action, b=mlp_out, precision=precision)
        #out = (action - mlp_out)*precision*(action-mlp_out)
        #(u-uhat)'*A*(u-uhat)
    #else:
    #    out = mlp_out
    #return out


def example_tf_network(n_layers=3, dim_hidden=None, dim_input=27,
                       dim_output=7, batch_size=25, phase='TRAIN'):
    """
    An example of how one might want to specify a network in tensorflow.

    Args:
        n_layers: Number of fully connected layers (including output).
        dim_hidden: Dimensionality of each hidden layer.
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        phase: 'TRAIN', 'TEST', or 'deploy'
    Returns:
        a list containing inputs, outputs, and the current session.
    """
    if dim_hidden is None:
        dim_hidden = (n_layers - 1) * [42]
    dim_hidden.append(dim_output)

    nn_input, action, precision = get_input_layer(phase, batch_size, dim_input, dim_output)
    mlp_applied = get_mlp_layers(nn_input, n_layers, dim_hidden)
    loss_out = get_loss_layer(phase=phase, mlp_out=mlp_applied, action=action, precision=precision)
    return_dict = {'inputs': [nn_input, action, precision], 'outputs': [mlp_applied], 'loss': [loss_out]}
    return return_dict
    #return [[nn_input, action, precision], loss_out]  #should this be checkpointed?

    #return out.to_proto()


POLICY_OPT_TF = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 20000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'use_gpu': 0,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # Other hyperparameters.
    'network_model': example_tf_network,  # Either a filename string
                                            # or a function to call to
                                            # create NetParameter.
    'network_arch_params': {},  # Arguments to pass to method above.
    'weights_file_prefix': '',
}


class Policy(object):
    """ Computes actions from states/observations. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, x, obs, t, noise):
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.
        Returns:
            A dU dimensional action vector.
        """
        raise NotImplementedError("Must be implemented in subclass.")


class TfPolicy(Policy):
    """
    A neural network policy implemented in Caffe. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        test_net: Initialized caffe network that can run forward.
        var: Du-dimensional noise variance vector.
    """
    def __init__(self, obs_tensor, act_op, var, sess):
        Policy.__init__(self)
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.sess = sess
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # Normalize obs.
        obs = obs.dot(self.scale) + self.bias
        action_mean = self.sess.run(self.act_op, feed_dict={self.obs_tensor: obs})
        u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u

    def get_weights_string(self):
        """ Return the weights of the neural network as a string. """
        raise 'NotImplemented - not implemented in tf'

    def get_net_param(self):
        """ Return the weights of the neural network as a string. """
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.deploy_net.share_with(self.net)
        self.deploy_net.save(f.name)
        f.close()
        with open(f.name, 'rb') as temp_file:
            weights_string = temp_file.read()
        return weights_string



""" This file defines the base policy optimization class. """
import abc


class PolicyOpt(object):
    """ Policy optimization superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, dO, dU):
        self._hyperparams = hyperparams
        self._dO = dO
        self._dU = dU

    @abc.abstractmethod
    def update(self):
        """ Update policy. """
        raise NotImplementedError("Must be implemented in subclass.")


class TfSolver:
    """ A container for holding solver hyperparams in tensorflow. Used to execute backwards pass. """
    def __init__(self):
        self.snapshot_prefix = None
        self.base_lr = None
        self.lr_policy = None
        self.momentum = None
        self.weight_decay = None
        self.type = None
        self.solver_tf_fun = self.get_solver_tf_fun()

    def get_solver_fun(self):
        pass

    def hash_solver(self, solver_string='adam'):
        pass

    def __call__(self, *args, **kwargs):
        pass


class PolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        self.batch_size = self._hyperparams['batch_size']

        #if self._hyperparams['use_gpu']:
        #    caffe.set_device(self._hyperparams['gpu_id'])
        #    caffe.set_mode_gpu()
        #else:
        #    caffe.set_mode_cpu()
        self.act_op = None
        self.loss_op = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None
        self.init_network()
        self.init_solver()
        self.tf_iter = 0
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        self.sess = tf.Session()
        self.policy = TfPolicy(self.obs_tensor, self.act_op, np.zeros(dU), self.sess)

    def init_network(self):
        tf_model = self._hyperparams['network_model']
        model_dict = tf_model()
        self.obs_tensor = model_dict['inputs'][0]
        self.precision_tensor = model_dict['inputs'][1]
        self.action_tensor = model_dict['inputs'][2]
        self.act_op = model_dict['outputs'][0]  # fix this
        self.loss_op = model_dict['loss'][0]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        solver_param = TfSolver()
        solver_param.snapshot_prefix = self._hyperparams['weights_file_prefix']
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']

        # Pass in net parameter either by filename or protostring.
        if isinstance(self._hyperparams['network_model'], basestring):
            self.solver = caffe.get_solver(self._hyperparams['network_model'])
        else:
            network_arch_params = self._hyperparams['network_arch_params']
            network_arch_params['dim_input'] = self._dO
            network_arch_params['dim_output'] = self._dU

            network_arch_params['batch_size'] = self.batch_size
            network_arch_params['phase'] = 'TRAIN'
            solver_param.train_net_param.CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward in python.
            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = 'TEST'
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward on the robot.
            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = 'deploy'
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            f.write(MessageToString(solver_param))
            f.close()

            self.solver = caffe.get_solver(f.name)

    # TODO - This assumes that the obs is a vector being passed into the
    #        network in the same place.
    #        (won't work with images or multimodal networks)
    def update(self, obs, tgt_mu, tgt_prc, tgt_wt, itr, inner_itr):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A CaffePolicy object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        #TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if itr == 0 and inner_itr == 1:
            self.policy.scale = np.diag(1.0 / np.std(obs, axis=0))
            self.policy.bias = -np.mean(obs.dot(self.policy.scale), axis=0)
        obs = obs.dot(self.policy.scale) + self.policy.bias

        blob_names = self.solver.net.blobs.keys()

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            self.solver.net.blobs[blob_names[0]].data[:] = obs[idx_i]
            self.solver.net.blobs[blob_names[1]].data[:] = tgt_mu[idx_i]
            self.solver.net.blobs[blob_names[2]].data[:] = tgt_prc[idx_i]

            self.solver.step(1)

            # To get the training loss:
            train_loss = self.solver.net.blobs[blob_names[-1]].data
            average_loss += train_loss
            if i % 500 == 0 and i != 0:
                LOGGER.debug('Caffe iteration %d, average loss %f',
                             i, average_loss / 500)
                average_loss = 0

            # To run a test:
            # if i % test_interval:
            #     print 'Iteration', i, 'testing...'
            #     solver.test_nets[0].forward()

        # Keep track of Caffe iterations for loading solver states.
        self.caffe_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)

        self.policy.net.share_with(self.solver.net)
        return self.policy

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        try:
            for n in range(N):
                obs[n, :, :] = obs[n, :, :].dot(self.policy.scale) + \
                        self.policy.bias
        except AttributeError:
            pass  #TODO: Should prob be called before update?

        output = np.zeros((N, T, dU))
        blob_names = self.solver.test_nets[0].blobs.keys()

        self.solver.test_nets[0].share_with(self.solver.net)

        for i in range(N):
            for t in range(T):
                # Feed in data.
                self.solver.test_nets[0].blobs[blob_names[0]].data[:] = \
                        obs[i, t]

                # Assume that the first output blob is what we want.
                output[i, t, :] = \
                        self.solver.test_nets[0].forward().values()[0][0]

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    # For pickling.
    def __getstate__(self):
        self.solver.snapshot()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'caffe_iter': self.caffe_iter,
        }

    # For unpickling.
    def __setstate__(self, state):
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.caffe_iter = state['caffe_iter']
        self.solver.restore(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.solverstate'
        )
        self.policy.net.copy_from(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.caffemodel'
        )
