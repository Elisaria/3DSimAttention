import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import SimAttention as SA
import Neighboorhood_Partition as NP


class model(tf.keras.Model):
    def __init__(self, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.Neigh_pool = SA.Neighborhood_MaxPooling(d=32)
        self.sa_block = [SA.SimAttention(d_l=8, d_h=64, k_h=4, k=16) for _ in range(layer_num)]
        self.globalMaxPool = layers.GlobalMaxPooling1D()
        self.classify_head = layers.Dense(40, activation='softmax')

    # x(B, 2024, 3)
    def call(self, x, training=None, mask=None):
        KPs = NP.FPS(x, 72)
        # x(B, 72, 32, 3)
        x = NP.Neighboorhood_Partition(x, KPs, 32)
        # x(B, 72, 32)
        x = self.Neigh_pool(x)
        for i in range(self.layer_num):
            x = self.sa_block[i](x)
        x = self.globalMaxPool(x)  # (B, 72)
        return self.classify_head(x)  # (B, 40)


class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[100, 2048, 3], dtype=tf.float32)])
    def __call__(self, inputs):
        result = self.model(inputs, training=False)
        return result


if __name__ == '__main__':
    data = np.load("ModelNet40/augmented_train.npy")
    val_data = np.load("ModelNet40/test_data.npy")
    print(data.shape)
    print(val_data.shape)
    labels = np.load("ModelNet40/augmented_label.npy")
    print(labels.shape)
    val_labels = np.load("ModelNet40/test_labels.npy")
    print(val_labels)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels)).batch(16)
    val = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(16)

    model = model(layer_num=4)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.0001,
                                                   weight_decay=0.01,
                                                   beta_1=0.9,
                                                   beta_2=0.95
                                                   ),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(dataset,
              validation_data=val,
              epochs=1500)

    export_model = ExportModel(model)
    tf.saved_model.save(export_model, export_dir='ModelNet40_model')
