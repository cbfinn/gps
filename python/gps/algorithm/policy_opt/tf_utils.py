import tensorflow as tf


def check_list_and_convert(the_object):
    if isinstance(the_object, list):
        return the_object
    return [the_object]


class TfMap:
    """ a container for inputs, outputs, and loss in a tf graph. This object exists only
    to make well-defined the tf inputs, outputs, and losses used in the policy_opt_tf class."""
    def __init__(self, inputs, outputs, loss):
        self.inputs = check_list_and_convert(inputs)
        self.outputs = check_list_and_convert(outputs)
        self.loss = check_list_and_convert(loss)

    def add_input(self, input_to_add):
        self.inputs.append(input_to_add)

    def add_output(self, output_to_add):
        self.outputs.append(output_to_add)

    def add_loss(self, loss_to_add):
        self.loss.append(loss_to_add)

    def get_input_tensor(self):
        return self.inputs[0]

    def get_act_tensor(self):
        return self.inputs[1]

    def get_precision_tensor(self):
        return self.inputs[2]

    def get_act_op(self):
        return self.outputs[0]

    def get_loss_op(self):
        return self.loss[0]


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
            print self.lr_policy
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
            return tf.train.AdamOptimizer(learning_rate=self.base_lr).minimize(self.loss_scalar)
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
            sess.run(self.solver_op, feed_dict)
