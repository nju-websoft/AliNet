import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class HighwayLayer(tf.keras.layers.Layer):
    """Highway layer."""

    def compute_output_signature(self, input_signature):
        pass

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.0,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(HighwayLayer, self).__init__()
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.dropout_rate = dropout_rate

        self.shape = (input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[self.input_dim, self.output_dim],
                                      initializer=self.kernel_initializer,
                                      dtype='float32',
                                      trainable=True)

    def call(self, inputs, training=True):
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = tf.keras.layers.BatchNormalization()(input1)
        input2 = tf.keras.layers.BatchNormalization()(input2)
        gate = tf.matmul(input1, self.kernel)
        gate = tf.keras.activations.tanh(gate)
        if training and self.dropout_rate > 0.0:
            gate = tf.nn.dropout(gate, self.dropout_rate)
        gate = tf.keras.activations.relu(gate)
        output = tf.add(tf.multiply(input2, 1 - gate), tf.multiply(input1, gate))
        return self.activation(output)


class AliNetGraphAttentionLayer(keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self,
                 input_dim,
                 output_dim,
                 adj,
                 nodes_num,
                 num_features_nonzero,
                 alpha=0.0,
                 dropout_rate=0.0,
                 is_sparse_inputs=False,
                 featureless=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 bias_regularizer='l2',
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 coef_dropout=0,
                 **kwargs):
        super(AliNetGraphAttentionLayer, self).__init__()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero
        self.adjs = [tf.SparseTensor(indices=adj[0][0], values=adj[0][1], dense_shape=adj[0][2])]
        self.dropout_rate = dropout_rate
        self.coef_drop = coef_dropout
        self.nodes_num = nodes_num
        self.alpha = alpha

        self.kernel, self.kernel1, self.kernel2 = None, None, None
        self.mapping = None
        self.bias = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        self.kernel1 = self.add_weight(shape=(self.input_dim, self.input_dim),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint,
                                       trainable=True)
        self.kernel2 = self.add_weight(shape=(self.input_dim, self.input_dim),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint,
                                       trainable=True)

    def call(self, inputs, training=True):
        inputs = tf.keras.layers.BatchNormalization()(inputs)
        mapped_inputs = tf.matmul(inputs, self.kernel)
        attention_inputs1 = tf.matmul(inputs, self.kernel1)
        attention_inputs2 = tf.matmul(inputs, self.kernel2)
        con_sa_1 = tf.reduce_sum(tf.multiply(attention_inputs1, inputs), 1, keepdims=True)
        con_sa_2 = tf.reduce_sum(tf.multiply(attention_inputs2, inputs), 1, keepdims=True)
        con_sa_1 = tf.keras.activations.tanh(con_sa_1)
        con_sa_2 = tf.keras.activations.tanh(con_sa_2)
        if training and self.dropout_rate > 0.0:
            con_sa_1 = tf.nn.dropout(con_sa_1, self.dropout_rate)
            con_sa_2 = tf.nn.dropout(con_sa_2, self.dropout_rate)
        con_sa_1 = tf.cast(self.adjs[0], dtype=tf.float32) * con_sa_1
        con_sa_2 = tf.cast(self.adjs[0], dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])
        weights = tf.sparse.add(con_sa_1, con_sa_2)
        weights = tf.SparseTensor(indices=weights.indices,
                                  values=tf.nn.leaky_relu(weights.values),
                                  dense_shape=weights.dense_shape)
        attention_adj = tf.sparse.softmax(weights)
        attention_adj = tf.sparse.reshape(attention_adj, shape=[self.nodes_num, self.nodes_num])
        value = tf.sparse.sparse_dense_matmul(attention_adj, mapped_inputs)
        return self.activation(value)


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

    def call(self, inputs, **kwargs):
        return self.init_embeds

    def compute_output_signature(self, input_signature):
        pass
