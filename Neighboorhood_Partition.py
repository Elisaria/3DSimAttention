import tensorflow as tf


# xyz(batch_size, n_points, 3)
# farthest(batch_size, 1)
def acquire_centroid_with_farthest(xyz, farthest):
    batch_indexes = tf.expand_dims(tf.range(tf.shape(farthest)[0]), axis=-1)
    # (batch_size, 2), 存储着xyz沿axis=0, axis=1上的索引
    batch_size_and_n_points_indexes = tf.concat([batch_indexes, farthest], axis=-1)
    # (batch_size, 3)
    output = tf.gather_nd(params=xyz, indices=batch_size_and_n_points_indexes)
    return tf.expand_dims(output, axis=1)  # (batch_size, 1, 3)


# The index positions where mask is True are the locations
# where the values in original_distance need to be replaced.
def update_distance_with_mask(original_distance, renovated_distance, mask):
    return original_distance * tf.cast(tf.equal(mask, False), dtype=tf.float32) \
           + renovated_distance * tf.cast(mask, dtype=tf.float32)


# inputs(batch_size, n_points, 3+dim_feature)
# return : centroids(batch_size, n_centroids, 3)
def FPS(inputs, n_centroids):
    B = tf.shape(inputs)[0]
    N = tf.shape(inputs)[1]
    xyz = inputs[:, :, :3]  # (batch_size, n_points, 3)
    distance = tf.ones((B, N), dtype=tf.float32) * 10 ** 30
    # The center point of the point cloud serves as the initial point for FPS.
    # (B, 1, 3)
    centroids = tf.reduce_mean(xyz, axis=-2, keepdims=True)

    for i in range(n_centroids):
        temp = tf.reduce_sum((xyz - centroids[:, -1:]) ** 2, axis=-1)  # (batch_size, n_points)
        mask = temp < distance  # (batch_size, n_points), bool类型
        distance = update_distance_with_mask(distance, temp, mask)
        centroids = tf.concat([centroids,
                               acquire_centroid_with_farthest(
                                   xyz, tf.argmax(distance, axis=-1, output_type=tf.int32)[:, tf.newaxis])],
                              axis=-2)  # (batch_size, ..., 3)

    return centroids[:, 1:]  # (batch_size, n_centroids, 3)


# Adaptive Sparse Sampling (ASS for short)
# Adaptively sample centroids based on the sparse distribution characteristics of the point cloud
# P(B, N, 3): Point cloud sequence
# n: Number of centroids; the square root of n must be an integer
# N and n must satisfy the condition: N/n is an integer
# return centroids(B, n, 3)
def Adaptive_Sparse_Sampling(P, n):
    B, N, C = tf.shape(P)[0], tf.shape(P)[1], tf.shape(P)[2]
    sqrt_n = tf.cast(tf.sqrt(tf.cast(n, dtype=tf.float32)), dtype=tf.int32)
    x = P[:, :, 0]
    x_index = tf.argsort(x)  # (B, N)
    # (B, N, 3)
    x_bucket = tf.gather(P, x_index, axis=1, batch_dims=1)
    # (B, sqrt_n, N/sqrt_n, 3)
    x_bucket = tf.reshape(x_bucket, [B, sqrt_n, -1, C])

    # (B, sqrt_n, N/sqrt_n)
    y = x_bucket[:, :, :, 1]
    y_index = tf.argsort(y)

    # clusters : (B, sqrt_n, N/sqrt_n, 3)
    buckets = tf.gather(x_bucket, y_index, axis=2, batch_dims=2)

    # (B, sqrt_n, sqrt_n, N/n, 3)
    buckets = tf.reshape(buckets, [B, sqrt_n, sqrt_n, -1, C])
    # (B, n, 3)
    centroids = tf.reshape(tf.reduce_mean(buckets, axis=-2), [B, -1, C])
    return centroids


# P(B, N, 3): Point cloud sequence
# centroids(B, n, 3)
# k: Number of points in each neighborhood
# Use the KNN algorithm to group the k nearest neighbor points around each centroid into a neighborhood
# The KNN evaluation criterion uses L1 distance
# return (B, n, k, 3)
def Neighboorhood_Partition(P, centroids, k):
    # (B, n, N, 3)
    temp = tf.tile(tf.expand_dims(P, axis=1), [1, tf.shape(centroids)[1], 1, 1])
    # (B, n, N)
    L1_distance = tf.reduce_sum(-tf.abs(tf.expand_dims(centroids, axis=2) - temp), axis=-1)
    # (B, n, k)
    indices = tf.nn.top_k(L1_distance, k).indices
    # (B, n, k, 3)
    return tf.gather(temp, indices, axis=2, batch_dims=2)


if __name__ == '__main__':
    data = tf.random.normal([2, 96, 3])
    print(data.shape)
    centroids = Adaptive_Sparse_Sampling(data, 16)
    print(centroids.shape)
    Neighboorhoods = Neighboorhood_Partition(data, centroids, 6)
