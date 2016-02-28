import tensorflow as tf
import numpy as np


def make_net():
    n_layers = 3
    dim_hidden = [42] * n_layers

    def init_weights(shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def multilayer_perceptron(nn_input, number_layers, dimension_hidden):
        #weights = []
        #biases = []
        #layers = []
        #weights.append(init_weights(dim_hidden[0]))
        #biases.append(init_weights(dim_hidden[0]))
        #layers.append(tf.nn.relu(tf.add(tf.matmul(input, weights[-1]), biases[-1])))
        cur_top = nn_input
        for layer_step in range(0, number_layers):
            #weights.append(init_weights(dimension_hidden[layer_step]))
            #biases.append(init_weights(dimension_hidden[layer_step]))
            t = cur_top.get_shape()
            llkkllk
            cur_weight = init_weights([cur_top._shape.dims[1].value, dimension_hidden[layer_step]], name='w' + str(layer_step))
            cur_bias = init_weights([dimension_hidden[layer_step]], name='b' + str(layer_step))
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
            #cur_top = layers[-1]
        #return layers[-1]
        return cur_top

    X = tf.placeholder("float", [None, 42]) # mnist data image of shape 28*28=784
    #t = X._shape.dims[1].value
    #adsafs
    Y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes
    #cur_weight = init_weights([42, 10], name='w' + str(1))
    #cur_bias = init_weights([10], name='b' + str(1))
    #cur_top = tf.nn.relu(tf.matmul(X, cur_weight) + cur_bias)
    #y_pred = cur_top
    net_out = multilayer_perceptron(X, n_layers, dim_hidden)
    cur_weight = init_weights([net_out._shape.dims[1].value, Y._shape.dims[1].value], name='w_out')
    cur_bias = init_weights([Y._shape.dims[1].value], name='b_out')
    y_pred = tf.matmul(net_out, cur_weight) + cur_bias
    return X, Y, y_pred


def main():
    trX = np.random.randn(128, 42)
    trY = np.random.randn(128, 10) #2 * trX + np.random.randn(42) * 0.33 # create a y value which is approximately linear but with some random noise
    X, Y, y_model = make_net()
    cost = (tf.pow(Y-y_model, 2)) # use sqr error for cost function
    cost_true = tf.reduce_mean((tf.pow(Y-y_model, 2)))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

    sess = tf.Session()
    init = tf.initialize_all_variables() # you need to initialize variables (in this case just variable W)
    sess.run(init)

    for i in range(100):
        for iter_step in range(128/8, 128, 128/8):
            x_batch = trX[0:iter_step, :]
            y_batch = trY[0:iter_step, :]
            sess.run(train_op, feed_dict={X: x_batch, Y: y_batch})
            t = sess.run(cost_true, feed_dict={X: x_batch, Y: y_batch})
            print t
    #print(sess.run(w))  # something around 2

if __name__ == '__main__':
    main()
