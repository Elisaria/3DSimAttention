import numpy as np
import tensorflow as tf
import ClusterPartition as CP
import time


# Perform max pooling on the neighborhood of the sampling points
# to extract significant features and, simultaneously,
# make the neighborhood features possess order invariance.
class Neighborhood_MaxPooling(tf.keras.layers.Layer):

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        # High-Rank Mapping
        # Minimize the loss of feature information caused by Max Pooling
        self.HR_dense = tf.keras.layers.Dense(2 * channels, activation='relu')
        # Low-Rank Mapping
        # Reduce the input burden of subsequent network layers
        self.LR_dense = tf.keras.layers.Dense(channels, activation='relu')

    def build(self, input_shape):
        self.MaxPool = tf.keras.layers.MaxPool2D(pool_size=[1, input_shape[-2]], strides=1)

    # n : Number of sampling points
    # m : Number of points in neighborhood of a sampling point
    # inputs(B, n, m, 3)
    # return(B, n, channels)
    def call(self, inputs, *args, **kwargs):
        # (B, n, k, 2*channels)
        x = self.HR_dense(inputs)
        # (B, n, 1, 2*channels)
        x = self.MaxPool(x)
        # (B, n, 2*channels)
        x = tf.squeeze(x, axis=2)
        x = self.LR_dense(x)
        return x


# inputs: (B, n, C)
#         n : number of neighborhoods / number of sampling points
#         C : the channel length of neighborhood
# return: (B, n, num_heads, D)
#
# Carry out Multi-Head mapping, each neighborhood is extended to multi-head representation
# to enhance semantic expression


class Up_Projection(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dim = dim

    def build(self, input_shape):
        self.head_kernel = self.add_weight(shape=[input_shape[-1], self.num_heads, self.dim])
        self.head_bias = self.add_weight(shape=[self.num_heads, self.dim])

    # inputs(B, n, channels)
    # return(B, n, num_heads, dim)
    def call(self, inputs, *args, **kwargs):
        # 多头映射
        # (B, n, num_heads, dim)
        x = tf.keras.activations.relu(tf.einsum("abc,cde->abde", inputs, self.head_kernel) + self.head_bias)
        return x


class Down_Projection(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-2], input_shape[-1], self.dim])
        self.bias = self.add_weight(shape=[self.dim])

    # inputs(B, n, num_heads, dim)
    # return(B, n, dim)
    def call(self, inputs, *args, **kwargs):
        x = tf.keras.activations.relu(tf.einsum("abcd,cde->abe", inputs, self.kernel) + self.bias)
        return x


# relu_activations(B, n, num_heads, D)
# LSH_values(B, n, D)
# return the local sensitive hash of neighborhood

def Neighborhood_SimHash(relu_activations):
    # Channel coding - a positive value is 1 while a 0 value is -1
    # (B, n, h, D)
    codes = 2 * tf.minimum(tf.math.ceil(relu_activations), 1) - 1.0
    # Reduce-sum all channels along h-axis in codes
    # to obtain local sensitive hash of neighborhood
    # (B, n, D)
    LSH_values = tf.reduce_sum(codes, axis=-2)
    return LSH_values


# First, absolute position encoding is performed in Local Sensitive Hashing,
# and then Exponentially Weighted Averaging is conducted to obtain SimValus.
#
# a : smoothing factor of Exponentially Weighted Averaging
# LSH_values(B, n, D)

def Hash_ExpMovAverage(LSH_values, a):
    D = tf.shape(LSH_values)[-1]
    indexes = tf.range(0, D, 1, dtype=tf.float32)

    # First, record the index position of each weight before the LSH_values are sorted in descending order.
    positions = tf.argsort(LSH_values, axis=-1, direction='DESCENDING')
    # Sort the weights of the local sensitive hash in descending order.
    # Small weights mean long-term data,
    # and large weights mean recent data.
    LSH_values = tf.gather(LSH_values, positions, axis=-1, batch_dims=-1)
    positions = tf.cast(positions, dtype=tf.float32)

    # Create a position encoding vector
    powers = positions / tf.cast(D, dtype=tf.float32)
    pos_encoding = tf.sin(1 / (10 ** powers)) * (positions % 2) + tf.cos(1 / (10 ** powers)) * (1.0 - positions % 2)
    # Absolute position encoding
    LSH_values = LSH_values + pos_encoding / a
    # Create the exponential decay coefficient
    c = a * (1 - a) ** indexes

    # Exponentially Weighted Averaging
    # (B, n)
    simvalues = tf.einsum("abc, c->ab", LSH_values, c)
    # print(simvalues)
    return simvalues


# 1. Obtain the SimHash fingerprint of the neighborhood of each sampling point.
# 2. Finding Exponentially Weighted Average of Locally Sensitive Hash Fingerprints,
#    This scalar value can reflect the similarity between neighborhood of the sampling point.
#    Defined as SimValue.
# 3. All neighborhood features are arranged in ascending order of SimValues,
#    and adjacent neighborhood features have a strong correlation.
#
# args : relu_activations(B, n, num_heads, D)
def Neighborhoods_Arrangement(relu_activations):
    LSH_values = Neighborhood_SimHash(relu_activations)
    simvalues = Hash_ExpMovAverage(LSH_values, a=0.4)
    relu_activations = tf.gather(relu_activations, tf.argsort(simvalues, axis=-1), axis=1, batch_dims=1)
    return relu_activations


# Based on SimValues,
# a neighborhood feature sequence with local correlation is constructed,
# thereby obtaining attention in the form of a sliding window.
class SimAttention(tf.keras.layers.Layer):
    # The window size must be odd
    # If the window_size is equal to input_shape[1], it degenerates into full attention.
    def __init__(self, dim, num_heads, window_size=9, **kwargs):
        super().__init__(**kwargs)
        self.reshape = tf.keras.layers.Reshape((-1, window_size, dim))
        self.concat = tf.keras.layers.Concatenate(axis=1)
        if window_size % 2 != 1:
            raise ValueError("The window size must be odd !")
        self.window_size = window_size
        self.UP_PROJECTION = Up_Projection(dim, num_heads)
        self.DOWN_PROJECTION = Down_Projection(dim)
        self.VALUE_DENSE = tf.keras.layers.Dense(dim, activation="relu")
        self.LAYERNORM = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        if input_shape[1] < self.window_size:
            raise ValueError("The input_shape[1] cannot be less than the `window_size`, but input_shape[1] = "
                             + str(input_shape[1]) + " < self.window_size = " + str(self.window_size))

    # inputs(B, n, dim)
    # return(B, n, dim)
    def call(self, inputs, *args, **kwargs):
        # (B, n, num_heads, dim)
        query = self.UP_PROJECTION(inputs)
        # print(tf.reduce_sum(query, axis=[-1, -2]))
        query = Neighborhoods_Arrangement(query)
        # print(tf.reduce_sum(query, axis=[-1, -2]))
        # (B, n, dim)
        query = self.DOWN_PROJECTION(query)
        # print(tf.reduce_sum(query, axis=-1))
        temp_1 = tf.tile(query[:, tf.newaxis, :self.window_size], [1, int((self.window_size - 1) / 2), 1, 1])
        temp_2 = tf.image.extract_patches(images=tf.expand_dims(query, axis=2),  # (B, n, 1, dim)
                                          sizes=[1, self.window_size, 1, 1],
                                          strides=[1, 1, 1, 1],
                                          rates=[1, 1, 1, 1],
                                          padding='VALID')
        temp_2 = self.reshape(temp_2)
        # print(temp_2.shape)
        temp_3 = tf.tile(query[:, tf.newaxis, -self.window_size:], [1, int((self.window_size - 1) / 2), 1, 1])
        # (B, n, window_size, dim)
        key = self.concat([temp_1, temp_2, temp_3])
        # print(tf.reduce_sum(key, axis=-1))
        # print(tf.reduce_sum(key, axis=[-1, -2]))

        f = tf.sqrt(tf.cast(tf.shape(query)[-1], dtype=tf.float32))
        # (B, n, window_size)
        attention_scores = tf.math.softmax(tf.einsum("abc, abdc->abd", query, key) / f,
                                           axis=-1)
        value = self.VALUE_DENSE(key)
        value = tf.einsum("abcd, abc->abd", value, attention_scores)
        # (B, n, dim)
        output = self.LAYERNORM(value + inputs)
        return output


# Forward propagation
if __name__ == '__main__':
    # (2, 72, 64)
    inputs = tf.random.normal([2, 72, 64])
    print(inputs.shape)
    outputs = SimAttention(dim=64, num_heads=8, window_size=3)(inputs)
    print(outputs.shape)
