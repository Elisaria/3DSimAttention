import tensorflow as tf
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


# Forward propagation
if __name__ == '__main__':
    inputs = tf.random.normal([2, 2048, 3])
    seg_model = model(layer_num=4)
    print(seg_model(inputs).shape)
