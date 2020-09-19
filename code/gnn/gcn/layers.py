import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


def dropout(inputs, drop_rate, noise_shape, is_sparse):
    if not is_sparse:
        return tf.nn.dropout(inputs, drop_rate)
    return sparse_dropout(inputs, drop_rate, noise_shape)


def sparse_dropout(x, drop_rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    keep_prob = 1 - drop_rate
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class Dense(keras.layers.Dense):
    """Dense layer."""

    def __init__(self,
                 output_dim,
                 placeholders,
                 dropout_rate=0.0,
                 is_sparse_inputs=False,
                 featureless=False,
                 **kwargs):
        super(Dense, self).__init__(units=output_dim, **kwargs)
        self.dropout_rate = placeholders['dropout'] if dropout_rate else 0.0
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

    def call(self, inputs):
        inputs = dropout(inputs, self.dropout_rate, self.num_features_nonzero, self.is_sparse_inputs)
        outputs = tf.matmul(inputs, self.kernel, a_is_sparse=self.is_sparse_inputs)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_signature(self, input_signature):
        pass


class GraphConvolution(keras.layers.Layer):
    """
    Graph convolution layer.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 adj,
                 num_features_nonzero,
                 dropout_rate=0.0,
                 is_sparse_inputs=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 bias_regularizer='l2',
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolution, self).__init__()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernels = list()
        self.bias = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_sparse_inputs = is_sparse_inputs
        self.num_features_nonzero = num_features_nonzero
        self.adjs = [tf.SparseTensor(indices=am[0], values=am[1], dense_shape=am[2]) for am in adj]
        self.dropout_rate = dropout_rate

    def update_adj(self, adj):
        print("gcn update adj...")
        self.adjs = [tf.SparseTensor(indices=am[0], values=am[1], dense_shape=am[2]) for am in adj]

    def build(self, input_shape):
        for i in range(len(self.adjs)):
            self.kernels.append(self.add_weight('kernel' + str(i),
                                                shape=[self.input_dim, self.output_dim],
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                dtype='float32',
                                                trainable=True))
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.output_dim, ],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype='float32',
                                        trainable=True)

    def call(self, inputs, training=True):
        # BN if training
        if training:
            inputs = tf.keras.layers.BatchNormalization()(inputs)
        # dropout if training
        if training and self.dropout_rate > 0.0:
            inputs = dropout(inputs, self.dropout_rate, self.num_features_nonzero, self.is_sparse_inputs)
        # convolve
        hidden_vectors = list()
        for i in range(len(self.adjs)):
            pre_sup = tf.matmul(inputs, self.kernels[i], a_is_sparse=self.is_sparse_inputs)
            hidden_vector = tf.sparse.sparse_dense_matmul(tf.cast(self.adjs[i], tf.float32), pre_sup)
            hidden_vectors.append(hidden_vector)
        outputs = tf.add_n(hidden_vectors)
        # bias
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        # activation
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_signature(self, input_signature):
        pass


class InputLayer(keras.layers.Layer):
    """embedding layer."""

    def __init__(self,
                 shape,
                 kernel_initializer='glorot_uniform'):
        super(InputLayer, self).__init__()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.init_embeds = self.add_weight('embedding',
                                           shape=shape,
                                           dtype='float32',
                                           trainable=True)
        # self.init_embeds = tf.nn.l2_normalize(self.init_embeds, 1)

    def call(self, inputs, **kwargs):
        return self.init_embeds

    def compute_output_signature(self, input_signature):
        pass
