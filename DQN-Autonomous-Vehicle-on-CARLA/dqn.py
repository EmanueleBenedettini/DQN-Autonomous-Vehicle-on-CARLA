import math
import numpy as np
import os
import tensorflow as tf

from state import State


class GradientClippingOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, optimizer, use_locking=False, name="GradientClipper"):
        super(GradientClippingOptimizer, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = self.optimizer.compute_gradients(*args, **kwargs)
        clipped_grads_and_vars = []
        for (grad, var) in grads_and_vars:
            if grad is not None:
                clipped_grads_and_vars.append((tf.clip_by_value(grad, -1, 1), var))
            else:
                clipped_grads_and_vars.append((grad, var))
        return clipped_grads_and_vars

    def apply_gradients(self, *args, **kwargs):
        return self.optimizer.apply_gradients(*args, **kwargs)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
            tf.compat.v1.summary.histogram('histogram', var)


class DeepQNetwork:
    def __init__(self, numActions, baseDir, args):

        self.numActions = numActions
        self.baseDir = baseDir
        self.saveModelFrequency = args.save_model_freq
        self.targetModelUpdateFrequency = args.target_model_update_freq
        self.tensorboardFrequency = args.tensorboard_logging_freq
        self.normalizeWeights = args.normalize_weights
        self.gamma = args.gamma
        self.step_frames = args.frame
        self.staleSess = None

        # Run in compatibility mode
        tf.compat.v1.disable_v2_behavior()

        self.sess = tf.compat.v1.Session()
        assert (len(tf.compat.v1.global_variables()) == 0), "Expected zero variables"
        self.x, self.y = self.build_network('policy', True, numActions)
        assert (len(tf.compat.v1.trainable_variables()) == 10), "Expected 10 trainable_variables"
        assert (len(tf.compat.v1.global_variables()) == 10), "Expected 10 total variables"
        self.x_target, self.y_target = self.build_network('target', False, numActions)
        assert (len(tf.compat.v1.trainable_variables()) == 10), "Expected 10 trainable_variables"
        assert (len(tf.compat.v1.global_variables()) == 20), "Expected 20 total variables"

        # build the variable copy ops
        self.update_target = []
        trainable_variables = tf.compat.v1.trainable_variables()
        all_variables = tf.compat.v1.global_variables()
        for i in range(0, len(trainable_variables)):
            self.update_target.append(all_variables[len(trainable_variables) + i].assign(trainable_variables[i]))

        self.a = tf.compat.v1.placeholder(tf.float32, shape=[None, numActions])
        print('a %s' % (self.a.get_shape()))
        self.y_ = tf.compat.v1.placeholder(tf.float32, [None])
        print('y_ %s' % (self.y_.get_shape()))

        self.y_a = tf.reduce_sum(tf.multiply(self.y, self.a), axis=1)
        print('y_a %s' % (self.y_a.get_shape()))

        difference = tf.abs(self.y_a - self.y_)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.loss = tf.reduce_sum(errors)

        tf.compat.v1.summary.scalar('loss', self.loss)

        optimizer = tf.compat.v1.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01)
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.compat.v1.train.Saver()

        self.merged = tf.compat.v1.summary.merge_all()
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.baseDir + '/tensorboard', self.sess.graph)

        if args.model is not None:
            print('Loading from model file %s' % args.model)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(args.model))
        else:
            # Initialize variables
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.sess.run(self.update_target)

    def build_network(self, name, trainable, numActions):
        print("Building network for %s trainable=%s" % (name, trainable))

        # First layer takes a screen, and shrinks by 2x
        x = tf.compat.v1.placeholder(tf.uint8, shape=[None, State.IMAGE_SIZE, State.IMAGE_SIZE, self.step_frames],
                                     name="screens")
        print(x)

        # x_normalized = tf.to_float(x) / 255.0
        x_normalized = tf.cast(x, dtype=tf.float32) / 255.0
        print(x_normalized)

        # Second layer convolves 32 8x8 filters with stride 4 with relu
        with tf.compat.v1.variable_scope("cnn1_" + name):
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, self.step_frames, 16], trainable, "conv1")
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1,
                                 name="h_conv1")
            print(h_conv1)

        # Third layer convolves 64 4x4 filters with stride 2 with relu
        with tf.compat.v1.variable_scope("cnn2_" + name):
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 16, 32], trainable, "conv2")
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2,
                                 name="h_conv2")
            print(h_conv2)

        # Fourth layer convolves 64 3x3 filters with stride 1 with relu
        with tf.compat.v1.variable_scope("cnn3_" + name):
            W_conv3, b_conv3 = self.make_layer_variables([3, 3, 32, 32], trainable, "conv3")
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3,
                                 name="h_conv3")
            print(h_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 32], name="h_conv3_flat")
        print(h_conv3_flat)

        # Fifth layer is fully connected with 512 relu units
        with tf.compat.v1.variable_scope("fc1_" + name):
            W_fc1, b_fc1 = self.make_layer_variables([7 * 7 * 32, 256], trainable, "fc1")
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)

        # Sixth (Output) layer is fully connected linear layer
        with tf.compat.v1.variable_scope("fc2_" + name):
            W_fc2, b_fc2 = self.make_layer_variables([256, numActions], trainable, "fc2")
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)

        return x, y

    def make_layer_variables(self, shape, trainable, name_suffix):
        if self.normalizeWeights:
            # This is best guess at what DeepMind does via torch's Linear.lua and SpatialConvolution.lua (see
            # reset methods). np.prod(shape[0:-1]) is attempting to get the total inputs to each node
            stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
            weights = tf.Variable(tf.compat.v1.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable,
                                  name='W_' + name_suffix)
            biases = tf.Variable(tf.compat.v1.random_uniform([shape[-1]], minval=-stdv, maxval=stdv),
                                 trainable=trainable, name='W_' + name_suffix)
        else:
            weights = tf.Variable(tf.random.truncated_normal(shape, stddev=0.01), trainable=trainable,
                                  name='W_' + name_suffix)
            biases = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + name_suffix)
        variable_summaries(weights)
        variable_summaries(biases)
        return weights, biases

    def inference(self, screens):
        y = self.sess.run([self.y], {self.x: screens})
        q_values = np.squeeze(y)
        return np.argmax(q_values)

    def train(self, batch, stepNumber):

        x2 = [b.state2.get_screens() for b in batch]
        y2 = self.y_target.eval(feed_dict={self.x_target: x2}, session=self.sess)
        x = [b.state1.get_screens() for b in batch]
        a = np.zeros((len(batch), self.numActions))
        y_ = np.zeros(len(batch))

        for i in range(0, len(batch)):
            a[i, batch[i].action] = 1
            if batch[i].terminal:
                y_[i] = batch[i].reward
            else:
                y_[i] = batch[i].reward + self.gamma * np.max(y2[i])

        if stepNumber % self.tensorboardFrequency == 0:
            summary, _ = self.sess.run([self.merged, self.train_step],
                                       feed_dict={
                                           self.x: x,
                                           self.a: a,
                                           self.y_: y_
                                       })
            self.summary_writer.add_summary(summary, stepNumber)
        else:
            self.sess.run([self.train_step],
                          feed_dict={
                              self.x: x,
                              self.a: a,
                              self.y_: y_
                          })

        if stepNumber % self.targetModelUpdateFrequency == 0:
            self.sess.run(self.update_target)

        if stepNumber % self.targetModelUpdateFrequency == 0 or stepNumber % self.saveModelFrequency == 0:
            dir = self.baseDir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            savedPath = self.saver.save(self.sess, dir + '/model.ckpt', global_step=stepNumber)
