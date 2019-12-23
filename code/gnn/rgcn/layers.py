import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class RGraphConvolutionLayer(keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 adj,
                 num_features_nonzero,
                 dropout_rate=0.0,
                 num_base=-1,
                 is_sparse_inputs=False,
                 featureless=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer="l2",
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(RGraphConvolutionLayer, self).__init__()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero
        self.support = len(adj)
        self.adj_list = [tf.SparseTensor(indices=adj[i][0], values=adj[i][1], dense_shape=adj[i][2])
                         for i in range(len(adj))]
        self.dropout_rate = dropout_rate
        self.num_bases = num_base
        self.W = list()

    def build(self, input_shape):
        if self.num_bases > 0:
            self.W = self.add_weight("W",
                                     shape=(self.input_dim * self.num_bases, self.output_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     trainable=True)
            self.W_comp = self.add_weight(shape=(self.input_dim, self.output_dim),
                                          initializer=self.kernel_initializer,
                                          regularizers=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        else:
            self.W = self.add_weight("W",
                                     shape=(self.input_dim*self.support, self.output_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     trainable=True)

        if self.bias is not None:
            self.B = self.add_weight(shape=(self.input_dim, self.output_dim),
                                     initializers=self.bias_initializer,
                                     regularizers=self.bias_regularizer,
                                     constraint=self.bias_constraint,
                                     trainable=True)

    def call(self, inputs, training=True):
        if training:
            inputs = tf.keras.layers.BatchNormalization()(inputs)
        supports = list()
        for i in range(self.support):
            if not self.featureless:
                # supports.append(keras.backend.dot(tf.cast(self.adj_list[i], tf.float32), inputs))
                supports.append(tf.sparse.sparse_dense_matmul(tf.cast(self.adj_list[i], tf.float32), inputs))
            else:
                supports = self.adj_list
        supports = keras.layers.concatenate(supports, axis=1)
        if self.num_bases > 0:
            self.W = tf.reshape(self.W, (self.num_bases, inputs.shape[0], self.output_dim))
            self.W = tf.transpose(self.W, (1, 0, 2))
            temp = tf.matmul(self.W_comp, self.W)
            temp = tf.reshape(temp, (self.support * (inputs.shape[0]), self.output_dim))
            output = keras.backend.dot(supports, temp)
        else:
            # output = keras.backend.dot(supports, self.W)
            output = tf.matmul(supports, self.W)
        if self.bias:
            output += self.B

        return self.activation(output)

