import cPickle as pkl

import tensorflow as tf
from catutil2 import *

import utils

dtype = utils.DTYPE

gpu_device = '/gpu:0'

class LR:
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_weight=0, random_seed=None):
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            b = self.vars['b']
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = xw + b
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {self.X: X}
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class FM:
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('v', [input_dim, factor_order], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']

            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v))
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 1),
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = xw + b + p
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {self.X: X}
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class FNN:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, init_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            for i in range(1, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class CCPM:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        embedding_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = embedding_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('f1', [embedding_order, layer_sizes[2], 1, 2], 'tnormal', dtype))
        init_vars.append(('f2', [embedding_order, layer_sizes[3], 2, 2], 'tnormal', dtype))
        init_vars.append(('w1', [2 * 3 * embedding_order, 1], 'tnormal', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                utils.activate(
                    tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i]
                               for i in range(num_inputs)], 1),
                    layer_acts[0]),
                layer_keeps[0])
            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embedding_order, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    num_inputs / 2),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                utils.activate(
                    tf.reshape(l, [-1, embedding_order * 3 * 2]),
                    layer_acts[1]),
                layer_keeps[1])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[2]),
                layer_keeps[2])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class PNN1:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            # This is where W_p \cdot p happens.
            # k1 is \theta, which is the weight for each field(feature) vector
            p = tf.reduce_sum(
                tf.reshape(
                    tf.matmul(
                        tf.reshape(
                            tf.transpose(
                                tf.reshape(l, [-1, num_inputs, factor_order]),
                                [0, 2, 1]),
                            [-1, num_inputs]),
                        k1),
                    [-1, factor_order, layer_sizes[2]]),
                1)
            print("l.shape", l.shape)
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1 + p,
                    layer_acts[1]),
                layer_keeps[1])
            print("after l.shape", l.shape)

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class Fast_CTR:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        init_vars.append(('bf', [factor_order], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            bf = self.vars['bf']
            l = utils.activate(
                    tf.reshape(
                        tf.reduce_sum(
                            tf.reshape(l, [-1, num_inputs, factor_order]), 1
                        ), [-1, factor_order]
                    ) + bf , 'none'
            )

            w1 = self.vars['w1']
            b1 = self.vars['b1']

            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[1]),
                layer_keeps[1])
            print("after l.shape", l.shape)

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                pass
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class Fast_CTR_Concat:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            l = tf.reshape(
                tf.concat(
                    tf.reshape(l, [-1, num_inputs, factor_order]),
                    1
                ), [-1, num_inputs * factor_order]
            )

            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']

            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[1]),
                layer_keeps[1])
            print("after l.shape", l.shape)

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class PNN1_Fixed:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w_l', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w_p', [num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        #init_vars.append(('w1', [num_inputs * factor_order + num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            w_l = self.vars['w_l']
            w_p = self.vars['w_p']
            b1 = self.vars['b1']
            # This is where W_p \cdot p happens.
            # k1 is \theta, which is the weight for each field(feature) vector
            p = tf.matmul(
                tf.reshape(l, [-1, num_inputs, factor_order]),
                tf.transpose(
                    tf.reshape(l, [-1, num_inputs, factor_order]), [0, 2, 1])
            )

            p = tf.nn.dropout(
                utils.activate(
                    tf.matmul(
                        tf.reshape(p, [-1, num_inputs * num_inputs]),
                        w_p),
                    'none'
                ),
                1.0
            )

            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w_l) + b1 + p,
                    layer_acts[1]),
                layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    if i == 1:
                        self.loss += layer_l2[i] * tf.nn.l2_loss(w_l)
                        self.loss += layer_l2[i] * tf.nn.l2_loss(w_p)
                    else:
                        wi = self.vars['w%d' % i]
                        # bi = self.vars['b%d' % i]
                        self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                pass
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class PNN2:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [factor_order * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])
            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            z = tf.reduce_sum(tf.reshape(l, [-1, num_inputs, factor_order]), 1)
            p = tf.reshape(
                tf.matmul(tf.reshape(z, [-1, factor_order, 1]),
                          tf.reshape(z, [-1, 1, factor_order])),
                [-1, factor_order * factor_order])
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + tf.matmul(p, k1) + b1,
                    layer_acts[1]),
                layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class XNN:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, interaction_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None, batch_size=None, gate_type='mul', norm_type='l2',
                 topk_pair=10, topk_triple=10):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w2', [topk_pair * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w3', [topk_triple* factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            with tf.device(gpu_device):
                self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
                self.y = tf.placeholder(dtype)
                self.vars = utils.init_var_map(init_vars, init_path)
                w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
                b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            with tf.device(gpu_device):
                x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)#bias removal
                embedX = tf.stack([xw[i] + b0[i] for i in range(num_inputs)], 1)
                l = tf.nn.dropout(
                    utils.activate(x, layer_acts[0]),
                    layer_keeps[0])
                w1 = self.vars['w1']
                w2 = self.vars['w2']
                w3 = self.vars['w3']
                b1 = self.vars['b1']
                #compute interactions
                first_indices, second_indices = get_pair_indices(num_inputs, num_inputs)
                interaction_2 = interaction(embedX, embedX,
                                          topk_pair, first_indices,
                                          second_indices, gate_type,
                                          norm_type)
                inter_2_vec = tf.reshape(interaction_2, [-1, topk_pair * factor_order])
                first_indices, second_indices = get_pair_indices(num_inputs, topk_pair)
                interaction_3 = interaction(embedX, interaction_2,
                                          topk_triple, first_indices,
                                          second_indices, gate_type,
                                          norm_type)
                inter_3_vec = tf.reshape(interaction_3, [-1, topk_triple * factor_order])
                
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, w1) + tf.matmul(inter_2_vec, w2) + tf.matmul(inter_3_vec, w3) + b1,
                        layer_acts[1]),
                    layer_keeps[1])

                for i in range(2, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    bi = self.vars['b%d' % i]
                    l = tf.nn.dropout(
                        utils.activate(
                            tf.matmul(l, wi) + bi,
                            layer_acts[i]),
                        layer_keeps[i])
                self.y_prob = tf.sigmoid(l)
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
                if layer_l2 is not None:
                    self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                    for i in range(1, len(layer_sizes) - 1):
                        wi = self.vars['w%d' % i]
                        # bi = self.vars['b%d' % i]
                        self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
                if interaction_l2 is not None: #TODO: add regularization
                    pass
                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)
            
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            config.log_device_placement=False
            self.sess = tf.Session(config=config)
            with tf.device(gpu_device):
                tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path
        
class XNN2:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, interaction_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None,batch_size=None,gate_type='mul',norm_type='l2'):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w2', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w3', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            #print('shape of xw[0]')
            #print(xw[0].shape)
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)#bias removal
            #print('shape of embX')
            #print(embedX.shape)
            #print('shape of x')
            #print(x.shape)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])
            w1 = self.vars['w1']
            w2 = self.vars['w2']
            w3 = self.vars['w3']
            b1 = self.vars['b1']
            #z = tf.reduce_sum(tf.reshape(l, [-1, num_inputs, factor_order]), 1)
            #compute interactions
            first_indices, second_indices = get_pair_indices(num_inputs)
            
            xw2left = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in first_indices]
            xw2right = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in second_indices]
            interaction_2 = interaction2(xw2left,xw2right, num_inputs,num_inputs,gate_type,norm_type)
            #interaction_2 better be list of tensors----> TODO
            #print('shape of inter2')
            #print(interaction_2.shape)
            inter_2_vec = tf.reshape(interaction_2,[-1,num_inputs* factor_order])
            
            #combined = tf.concat([x, interaction_2],1)
            #interaction_3 = interaction(embedX,interaction_2, num_inputs,num_inputs, first_indices,second_indices, gate_type,norm_type)
            #list of tensors concat then trans..
            #inter_3_vec = tf.reshape(interaction_3,[-1,num_inputs* factor_order])
            #encoded = tf.reshape(embed, [batch_size, -1])
            #encoded = tf.concat([encoded, tf.reshape(embed, [batch_size, -1])],1)
            
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + tf.matmul(inter_2_vec, w2), #+ tf.matmul(inter_3_vec, w3)+b1,
                    layer_acts[1]),
                layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])
            print(l.shape)
            self.y_prob = tf.sigmoid(l)
            print(self.y_prob.shape)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if interaction_l2 is not None: #TODO: add regularization
                pass
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)
        
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path
