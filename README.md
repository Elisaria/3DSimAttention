# 3DSimAttention
This project is the official implementation of the paper titled **<<Efficient Sparse Attention Mechanism for 3D Point Cloud Processing Using 3DSimAttention>>**.
## About
-**Core Purpose** : Designing to enhance the efficiency and generalization capability of Transformers in processing 3D point clouds.  
-**Use Case** : 3D point cloud classification, segmentation.  
-**Tech Stack** : Built with TensorFlow 2.14+ (GPU support requires CUDA 11.2+) and Python 3.8+.  
## File Description
-**Classification_Model.py** : This is a classification model case based on the ModelNet40 dataset.  
The input point cloud P has a shape of (batch_size=32, L=2048, 3).  

When the Neighborhood-Wise (NW) paradigm is adopted:  
1. P first undergoes sampling and grouping to generate neighborhoods with a shape of (batch_size, n=72, K=32, 3);  
2. Then, pooling is applied to obtain latent features with a shape of (batch_size, n=72, d=32).  

When the **Point-Wise (PW) paradigm** is used:  
P is directly fed into 3DSimAttention without additional processing. Regardless of the paradigm adopted, the final output shape of the classification task remains (batch\_size, class=40).

-**Segmentation_Model.py** : This is a case of 3DSimAttention applied to the segmentation task. Segmentation can be regarded as fine-grained classification performed on individual points, and the output length of the model is consistent with the input length. For example:(batch_size, L, 3) -> (batch_size, L, instances). Here, the specific value of "instances" depends on the number of instances that need to be segmented in the task.  

-**SimAttention.py** : The official implementation of 3DSimAttention.  

-**LSHAttention.py** : The implementation method of applying Locality-Sensitive Hashing (LSH) Attention to point clouds.  

-**Neighborhood_Partition.py** : Two point cloud sampling methods are implemented. One is the classic Farthest Point Sampling (FPS) algorithm. The other is a new Adaptive Sparse Sampling (ASS) method proposed by us, which can adaptively allocate sampling points according to the sparsity of point clouds. With significantly lower computational complexity than FPS, it is suitable for sampling large-scale point clouds and is our recommended sampling method.  

## Run the Example
Provide a minimal, runnable example — no complex configurations.  
```
import tensorflow as tf
from tensorflow import keras
import Classification_Model as Model

# saving model
class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[100, 2048, 3], dtype=tf.float32)])
    def __call__(self, inputs):
        result = self.model(inputs, training=False)
        return result

# training model
if __name__ == '__main__':
    train_data = ''
    val_data = ''
    train_labels = ''
    val_labels = ''
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
    val = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32)

    model = Model.model(layer_num=4)
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
```

By running this example, you will be able to train a classification model that achieves an accuracy of over 90% on the ModelNet40 dataset.

