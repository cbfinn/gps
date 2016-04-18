import tensorflow as tf


def check_list_and_convert(the_object):
    if isinstance(the_object, list):
        return the_object
    return [the_object]


class TfMap:
    """ a container for inputs, outputs, and loss in a tf graph. This object exists only
    to make well-defined the tf inputs, outputs, and losses used in the policy_opt_tf class."""

    def __init__(self, input_tensor, target_output_tensor, precision_tensor, output_op, loss_op):
        self.input_tensor = input_tensor
        self.target_output_tensor = target_output_tensor
        self.precision_tensor = precision_tensor
        self.output_op = output_op
        self.loss_op = loss_op

    @classmethod
    def init_from_lists(cls, inputs, outputs, loss):
        inputs = check_list_and_convert(inputs)
        outputs = check_list_and_convert(outputs)
        loss = check_list_and_convert(loss)
        if len(inputs) < 3:  # pad for the constructor if needed.
            inputs += [None]*(3 - len(inputs))
        return cls(inputs[0], inputs[1], inputs[2], outputs[0], loss[0])

    def get_input_tensor(self):
        return self.input_tensor

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def get_target_output_tensor(self):
        return self.target_output_tensor

    def set_target_output_tensor(self, target_output_tensor):
        self.target_output_tensor = target_output_tensor

    def get_precision_tensor(self):
        return self.precision_tensor

    def set_precision_tensor(self, precision_tensor):
        self.precision_tensor = precision_tensor

    def get_output_op(self):
        return self.output_op

    def set_output_op(self, output_op):
        self.output_op = output_op

    def get_loss_op(self):
        return self.loss_op

    def set_loss_op(self, loss_op):
        self.loss_op = loss_op


class TfSolver:
    """ A container for holding solver hyperparams in tensorflow. Used to execute backwards pass. """
    def __init__(self, loss_scalar, solver_name='adam', base_lr=None, lr_policy=None,
                 momentum=None, weight_decay=None):
        self.base_lr = base_lr
        self.lr_policy = lr_policy
        self.momentum = momentum
        self.solver_name = solver_name
        self.loss_scalar = loss_scalar
        if self.lr_policy != 'fixed':
            raise NotImplementedError('learning rate policies other than fixed are not implemented')

        self.weight_decay = weight_decay
        if weight_decay is not None:
            trainable_vars = tf.trainable_variables()
            loss_with_reg = self.loss_scalar
            for var in trainable_vars:
                loss_with_reg += self.weight_decay*tf.nn.l2_loss(var)
            self.loss_scalar = loss_with_reg

        self.solver_op = self.get_solver_op()

    def get_solver_op(self):
        solver_string = self.solver_name.lower()
        if solver_string == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.base_lr,
                                          beta1=self.momentum).minimize(self.loss_scalar)
        elif solver_string == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.base_lr,
                                             decay=self.momentum).minimize(self.loss_scalar)
        elif solver_string == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.base_lr,
                                              momentum=self.momentum).minimize(self.loss_scalar)
        elif solver_string == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.base_lr,
                                             initial_accumulator_value=self.momentum).minimize(self.loss_scalar)
        elif solver_string == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.base_lr).minimize(self.loss_scalar)
        else:
            raise NotImplementedError("Please select a valid optimizer.")

    def __call__(self, feed_dict, sess, device_string="/cpu:0"):
        with tf.device(device_string):
            loss = sess.run([self.loss_scalar, self.solver_op], feed_dict)
            return loss[0]

