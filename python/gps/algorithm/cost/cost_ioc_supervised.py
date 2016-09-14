""" This file defines neural network cost function. """
import copy
import logging
import numpy as np
import tempfile
from os.path import join

import caffe
from caffe.proto.caffe_pb2 import SolverParameter, TRAIN, TEST
from google.protobuf.text_format import MessageToString

from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_ioc_nn import CostIOCNN
from gps.utility.demo_utils import xu_to_sample_list, extract_demos

LOGGER = logging.getLogger(__name__)


class CostIOCSupervised(CostIOCNN):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams):
        super(CostIOCSupervised, self).__init__(hyperparams)
        self.gt_cost = hyperparams['gt_cost']  # Ground truth cost
        self.gt_cost = self.gt_cost['type'](self.gt_cost)

        self.eval_gt = hyperparams.get('eval_gt', False)

        self.agent = hyperparams['agent']  # Required for sample packing
        self.agent = self.agent['type'](self.agent)
        self.params_file = join(self.weights_dir, 'supervised_net.params')

        # Debugging
        X, U, O, cond = extract_demos(self._hyperparams['demo_file'])
        self.test_sample_list = xu_to_sample_list(self.agent, X, U)

        sup_solver = self.init_solver(phase='supervised')
        if hyperparams.get('init_from_demos', True):
            self.init_supervised_demos(sup_solver, hyperparams['demo_file'], hyperparams.get('traj_samples', []))
        solver = self.init_solver(phase=TRAIN)
        solver.net.share_with(sup_solver.net)
        solver.test_nets[0].share_with(sup_solver.net)
        solver.test_nets[1].share_with(sup_solver.net)
        #solver.net.copy_from(self.weight_file)  # Load weights into train net

        self.finetune = hyperparams.get('finetune', False)

    def eval(self, sample):
        if self.eval_gt:
            return self.gt_cost.eval(sample)
        else:
            return super(CostIOCSupervised, self).eval(sample)


    def test_eval(self):
        return self.eval(self.test_sample_list[0])[0]


    def update(self, demoU, demoX, demoO, d_log_iw, sampleU, sampleX, sampleO, s_log_iw, itr=-1):
        """
        Learn cost function with generic function representation.
        Args:
            demoU: the actions of demonstrations.
            demoX: the states of demonstrations.
            demoO: the observations of demonstrations.
            d_log_iw: log importance weights for demos.
            sampleU: the actions of samples.
            sampleX: the states of samples.
            sampleO: the observations of samples.
            s_log_iw: log importance weights for samples.
        """
        if self.finetune:
            if itr >= 5:
                super(CostIOCSupervised, self).update(demoU, demoX, demoO, d_log_iw, sampleU, sampleX, sampleO, s_log_iw, itr=itr)
        else:
            return


    def init_supervised_demos(self, solver, demo_file, traj_files):
        X, U, O, cond = extract_demos(demo_file)

        import pickle

        for traj_file in traj_files:
            with open(traj_file, 'r') as f:
                sample_lists = pickle.load(f)
                for sample_list in sample_lists:
                    X = np.r_[sample_list.get_X(), X]
                    U = np.r_[sample_list.get_U(), U]
                    # O = np.r_[sample_list.get_obs(), O]
        n_test = 5
        testX = X[-n_test:]
        testU = U[-n_test:]
        X = X[:-n_test]
        U = U[:-n_test]

        self.init_supervised(solver, U, X, O, testX, testU)

    def init_supervised(self, solver, sampleU, sampleX, sampleO, testX, testU, heartbeat=100):
        """
        """

        Ns = sampleX.shape[0]  # Num samples
        print 'Num samples:', Ns

        sample_batch_size = self.sample_batch_size + self.demo_batch_size

        blob_names = solver.net.blobs.keys()
        sbatches_per_epoch = np.floor(Ns / sample_batch_size)

        sample_idx = range(Ns)
        average_loss = 0

        sample_costs = []
        sample_list = xu_to_sample_list(self.agent, sampleX, sampleU)
        for n in range(Ns):
            l, _, _, _, _, _ = self.gt_cost.eval(sample_list[n])
            sample_costs.append(l)
        sample_costs = np.array(sample_costs)
        sample_costs = np.expand_dims(sample_costs, -1)
        T = sample_costs.shape[1]

        for i in range(self._hyperparams['init_iterations']):
            # Randomly sample batches
            np.random.shuffle(sample_idx)

            # Load in data for this batch.
            s_start_idx = int(i * sample_batch_size %
                              (sbatches_per_epoch * sample_batch_size))
            s_idx_i = sample_idx[s_start_idx:s_start_idx + sample_batch_size]
            solver.net.blobs[blob_names[0]].data[:] = sampleX[s_idx_i]
            solver.net.blobs[blob_names[1]].data[:] = np.sum(self._hyperparams['wu'] * sampleU[s_idx_i] ** 2, axis=2,
                                                             keepdims=True)
            solver.net.blobs[blob_names[2]].data[:] = sample_costs[s_idx_i]
            solver.step(2)
            train_loss = solver.net.blobs[blob_names[-1]].data
            # test_loss = solver.test_nets[0].blobs[blob_names[-1]].data
            average_loss += train_loss
            if i % heartbeat == 0 and i != 0:
                LOGGER.debug('Caffe iteration %d, average loss %f',
                             i, average_loss / heartbeat)
                # print 'test_loss:', test_loss
                print 'train_loss:', train_loss
                average_loss = 0
        # Keep track of Caffe iterations for loading solver states.
        # self.caffe_iter += self._hyperparams['iterations']
        solver.test_nets[0].share_with(solver.net)
        solver.test_nets[1].share_with(solver.net)

        # supervised_losses = []
        # for n in range(Ns):
        #    l, _, _, _, _, _ = self.eval(sample_list[n])
        #    supervised_losses.append(l)
        # supervised_losses = np.array(supervised_losses)
        # supervised_losses = np.expand_dims(supervised_losses, -1)

        import matplotlib.pyplot as plt
        test_costs = []
        nTest = testX.shape[0]
        sample_list = xu_to_sample_list(self.agent, testX, testU)
        for n in range(nTest):
            l, _, _, _, _, _ = self.gt_cost.eval(sample_list[n])
            test_costs.append(l)
        test_costs = np.array(test_costs)

        supervised_test = []
        for n in range(nTest):
            l, _, _, _, _, _ = self.eval(sample_list[n])
            supervised_test.append(l)
        supervised_test = np.array(supervised_test)
        supervised_test = np.expand_dims(supervised_test, -1)

        plt.figure()
        linestyles = ['-', ':', 'dashed']
        for i in range(4):
            plt.plot(np.arange(T), test_costs[i], color='red', linestyle=linestyles[i % len(linestyles)])
            plt.plot(np.arange(T), 2 * supervised_test[i], color='blue', linestyle=linestyles[i % len(linestyles)])
        plt.show()

        #solver.net.save(self.weight_file)

    def init_solver(self, phase='supervised', sample_batch_size=None):
        """ Helper method to initialize the solver. """
        solver_param = SolverParameter()
        solver_param.display = 0  # Don't display anything.
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']
        solver_param.random_seed = self._hyperparams['random_seed']

        # Pass in net parameter by protostring (could add option to input prototxt file).
        network_arch_params = self._hyperparams['network_arch_params']

        network_arch_params['dim_input'] = self._dO
        network_arch_params['demo_batch_size'] = self._hyperparams['demo_batch_size']
        if sample_batch_size is None:
            network_arch_params['sample_batch_size'] = self._hyperparams['sample_batch_size']
        else:
            network_arch_params['sample_batch_size'] = sample_batch_size
        network_arch_params['T'] = self._T
        network_arch_params['phase'] = TRAIN
        network_arch_params['ioc_loss'] = self._hyperparams['ioc_loss']
        network_arch_params['Nq'] = self._iteration_count
        network_arch_params['smooth_reg_weight'] = self._hyperparams['smooth_reg_weight']
        network_arch_params['mono_reg_weight'] = self._hyperparams['mono_reg_weight']
        network_arch_params['gp_reg_weight'] = self._hyperparams['gp_reg_weight']
        network_arch_params['learn_wu'] = self._hyperparams['learn_wu']
        network_arch_params['phase'] = phase
        solver_param.train_net_param.CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        # For running forward in python.
        network_arch_params['phase'] = TEST
        solver_param.test_net_param.add().CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        network_arch_params['phase'] = 'forward_feat'
        solver_param.test_net_param.add().CopyFrom(
            self._hyperparams['network_model'](**network_arch_params)
        )

        # These are required by Caffe to be set, but not used.
        solver_param.test_iter.append(1)
        solver_param.test_iter.append(1)
        solver_param.test_interval = 1000000

        #f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        #f.write(MessageToString(solver_param))
        #f.close()
        with open(self.params_file+'.'+str(phase), 'w+') as f:
            f.write(MessageToString(solver_param))
            fname = f.name
        self.solver = caffe.get_solver(fname)
        self.caffe_iter = 0
        return self.solver
