""" This file defines quadratic cost function. """
import copy
import numpy as np

from gps.algorithm.cost.config import COST_IOC_QUADRATIC
from gps.algorithm.cost.cost import Cost

class CostIOCQuadratic(Cost):
	""" Set up weighted quadratic norm loss with learned parameters. """
	def __init__(self, hyperparams):
        config = copy.deepcopy(COST_IOC_QUADRATIC) # Set this up in the config?
        config.update(hyperparams)
        Cost.__init__(self, config)
        # TODO - set-up cost function in caffe/tf here.
        # by default using caffe

        if self._hyperparams['use_gpu']:
            caffe.set_device(self._hyperparams['gpu_id'])
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self._init_solver()


    def eval(self, sample):
    	"""
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        pass

    # TODO - we might want to make the demos and samples input as SampleList objects, rather than arrays.
    def update(self, demoU, demoX, demoO, dlogis, sampleU, sampleX, sampleO, slogis, ...
    			itr, algorithm):
    	"""
    	Learn cost function with generic function representation.
    	Args:
    		demoU: the actions of demonstrations.
    		demoX: the states of demonstrations.
    		demoO: the observations of demonstrations.
    		dlogis: importance weights for demos.
    		sampleU: the actions of samples.
    		sampleX: the states of samples.
    		sampleO: the observations of samples.
    		slogis: importance weights for samples.
    		itr: current iteration.
    		algorithm: current algorithm object.
    	"""

    def _init_solver(self):
        """ Helper method to initialize the solver. """
        solver_param = SolverParameter()
        solver_param.snapshot_prefix = self._hyperparams['weights_file_prefix']
        solver_param.display = 0  # Don't display anything.
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']
        solver_param.random_seed = self._hyperparams['random_seed']

        # Pass in net parameter either by filename or protostring.
        if isinstance(self._hyperparams['network_model'], basestring):
            self.solver = caffe.get_solver(self._hyperparams['network_model'])
        else:
            network_arch_params = self._hyperparams['network_arch_params']
            network_arch_params['dim_input'] = self._dO
            network_arch_params['dim_output'] = self._dU

            network_arch_params['batch_size'] = self.batch_size
            network_arch_params['phase'] = TRAIN
            solver_param.train_net_param.CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward in python.
            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = TEST
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward on the robot.
            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = 'deploy'
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # These are required by Caffe to be set, but not used.
            solver_param.test_iter.append(1)
            solver_param.test_iter.append(1)
            solver_param.test_interval = 1000000

            f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            f.write(MessageToString(solver_param))
            f.close()

            self.solver = caffe.get_solver(f.name)



