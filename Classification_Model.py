import tensorflow as tf
from keras import layers
import SimAttention as SA
import Neighborhood_Partition as NP


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
        x = NP.Neighborhood_Partition(x, KPs, 32)
        # x(B, 72, 32)
        x = self.Neigh_pool(x)
        for i in range(self.layer_num):
            x = self.sa_block[i](x)
        x = self.globalMaxPool(x)  # (B, 72)
        return self.classify_head(x)  # (B, 40)

# Forward propagation
if __name__ == '__main__':
    inputs = tf.random.normal([2, 2048, 3])
    cls_model = model(layer_num=4)
    print(cls_model(inputs).shape)
