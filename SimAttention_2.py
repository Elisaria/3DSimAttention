# k_h : the number of heads
# d_l : the output dimension of down-projection
# d_h : the output dimension of up-projection
# k : the number of ramdon mapping functions
# ==============================================================================

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
    # return(B, n, d)
    def call(self, inputs, *args, **kwargs):
        # (B, n, k, 2*d)
        x = self.HR_dense(inputs)
        # (B, n, 1, 2*d)
        x = self.MaxPool(x)
        # (B, n, 2*d)
        x = tf.squeeze(x, axis=2)
        # (B, n, d)
        x = self.LR_dense(x)
        return x


class Down_Projection(tf.keras.layers.Layer):
    def __init__(self, d_l, **kwargs):
        super().__init__(**kwargs)
        self.d_l = d_l

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-2], input_shape[-1], self.d_l])
        self.bias = self.add_weight(shape=[self.d_l])

    # inputs(B, n, k_h, d_h)
    # return(B, n, d_l)
    def call(self, inputs, *args, **kwargs):
        x = tf.keras.activations.relu(tf.einsum("abcd,cde->abe", inputs, self.kernel) + self.bias)
        return x


# obtain the random mapping table
class Random_Projection(tf.keras.layers.Layer):
    # k : number of ramdom projection functions
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.d_l = input_shape[-1]

    # inputs(B, n, d_l)
    # return(B, n, k)
    def call(self, inputs, *args, **kwargs):
        # Ramdom Mapping Ensemble
        self.RME = tf.random.normal(shape=[self.d_l, self.k])
        x = tf.einsum('abc, cd->abd', inputs, self.RME)
        return x


#
# args : RMVs(B, n, k), activations(B, n, d_l)
# return (B, n, k_h, d_l)
#
def Memory_Reorganization(activations, RMVs, k_h):
    # (B, k, n)
    RMVs = tf.transpose(RMVs, perm=[0, 2, 1])
    # (B, k)
    h_vars = tf.reduce_mean((RMVs - tf.reduce_mean(RMVs, axis=-1, keepdims=True)) ** 2, axis=-1)
    # (B, k_h, n)
    h_max = tf.gather(RMVs, tf.argsort(h_vars, axis=-1, direction='DESCENDING')[:, :k_h], axis=1, batch_dims=1)
    h_max = tf.argsort(h_max, axis=-1)
    # (B, n, 1, d_l)
    permuted_activations = tf.expand_dims(tf.gather(activations, h_max[:, 0, :], axis=1, batch_dims=1), axis=2)
    for i in range(k_h - 1):
        # (B, n, 1, d_l)
        temp = tf.expand_dims(tf.gather(activations, h_max[:, i + 1, :], axis=1, batch_dims=1), axis=2)
        permuted_activations = tf.concat([permuted_activations, temp], axis=2)
    # (B, n, k_h, d_l)
    return permuted_activations


# Based on SimValues,
# a neighborhood feature sequence with local correlation is constructed,
# thereby obtaining attention in the form of a sliding window.
class SimAttention(tf.keras.layers.Layer):
    # The window size must be odd
    # If the window_size is equal to input_shape[1], it degenerates into full attention.
    def __init__(self, d_l, d_h, k_h, k, window_size=9, **kwargs):
        super().__init__(**kwargs)
        self.reshape = tf.keras.layers.Reshape((-1, window_size, d_l))
        self.concat = tf.keras.layers.Concatenate(axis=1)
        if window_size % 2 != 1:
            raise ValueError("The window size must be odd !")

        self.window_size = window_size
        self.k_h = k_h
        self.DOWN_PROJECTION_1 = tf.keras.layers.Dense(d_l, activation='relu')
        self.DOWN_PROJECTION_2 = Down_Projection(d_l)
        self.RANDOM_PROJECTION = Random_Projection(k)
        self.UP_PROJECTION = tf.keras.layers.Dense(d_h, activation='relu')

        self.LAYERNORM = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        if input_shape[1] < self.window_size:
            raise ValueError("The input_shape[1] cannot be less than the `window_size`, but input_shape[1] = "
                             + str(input_shape[1]) + " < self.window_size = " + str(self.window_size))

    # inputs(B, n, d_h)
    # return(B, n, d_h)
    def call(self, inputs, *args, **kwargs):
        # (B, n, d_l)
        query = self.DOWN_PROJECTION_1(inputs)
        # (B, n, k)
        RMVs = self.RANDOM_PROJECTION(query)
        # (B, n, k_h, d_l)
        query = Memory_Reorganization(query, RMVs, self.k_h)
        # (B, n, d_l)
        query = self.DOWN_PROJECTION_2(query)
        # print(tf.reduce_sum(query, axis=-1))
        temp_1 = tf.tile(query[:, tf.newaxis, :self.window_size], [1, int((self.window_size - 1) / 2), 1, 1])
        temp_2 = tf.image.extract_patches(images=tf.expand_dims(query, axis=2),  # (B, n, 1, dim)
                                          sizes=[1, self.window_size, 1, 1],
                                          strides=[1, 1, 1, 1],
                                          rates=[1, 1, 1, 1],
                                          padding='VALID')
        temp_2 = self.reshape(temp_2)
        temp_3 = tf.tile(query[:, tf.newaxis, -self.window_size:], [1, int((self.window_size - 1) / 2), 1, 1])
        # (B, n, window_size, d_l)
        key = self.concat([temp_1, temp_2, temp_3])
        # print(tf.reduce_sum(key, axis=-1))
        # print(tf.reduce_sum(key, axis=[-1, -2]))

        f = tf.sqrt(tf.cast(tf.shape(query)[-1], dtype=tf.float32))
        # (B, n, window_size)
        attention_scores = tf.math.softmax(tf.einsum("abc, abdc->abd", query, key) / f,
                                           axis=-1)
        # (B, n, d_h)
        value = self.UP_PROJECTION(key)
        value = tf.einsum("abcd, abc->abd", value, attention_scores)
        # (B, n, dim)
        output = self.LAYERNORM(value + inputs)
        return output


# Forward propagation
if __name__ == '__main__':
    # (2, 72, 64)
    inputs = tf.random.normal([2, 72, 64])
    print(inputs.shape)
    outputs = SimAttention(d_l=8, d_h=64, k_h=4, k=16)(inputs)
    print(outputs.shape)
