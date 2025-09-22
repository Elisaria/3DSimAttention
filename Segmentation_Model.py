import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import SimAttention as SA


class model(tf.keras.Model):
    def __init__(self, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.sa_block = [SA.SimAttention(d_l=8, d_h=64, k_h=4, k=16) for _ in range(layer_num)]
        self.globalMaxPool = layers.GlobalMaxPooling1D()
        self.seg_head = layers.Dense(2, activation='sigmoid')

    # x(B, 2024, 3)
    def call(self, x, training=None, mask=None):
        for i in range(self.layer_num):
            x = self.sa_block[i](x)
        return self.seg_head(x)  # (B, 2024, 2)


class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[100, 2048, 3], dtype=tf.float32)])
    def __call__(self, inputs):
        result = self.model(inputs, training=False)

        return result


if __name__ == '__main__':
    train_data = 'data address'
    val_data = 'data address'
    train_labels = 'data address'
    val_labels = 'data address'
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
    val = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32)

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
