# 3DSimAttention
This project is the official implementation of the paper titled 3DLSHAttention: Efficient 3D Point Cloud Processing via Locality-Sensitive Hashing Attention.
## About
-**Core Purpose** : Designing to enhance the efficiency and generalization capability of Transformers in processing 3D point clouds.  
-**Use Case** : 3D point cloud classification, segmentation.  
-**Tech Stack** : Built with TensorFlow 2.14+ (GPU support requires CUDA 11.2+) and Python 3.8+.  
## File Description
-**Classification_Model.py** : This is a classification model case based on the ModelNet40 dataset.  

The input point cloud P has a shape of (batch_size=32, L=2048, 3).  

### When the Neighborhood-Wise (NW) paradigm is adopted:  
1. P first undergoes sampling and grouping to generate neighborhoods with a shape of (batch_size, n=72, K=32, 3);  
2. Then, pooling is applied to obtain latent features with a shape of (batch_size, n=72, d=32).  


### When the **Point-Wise (PW) paradigm** is used:  
P is directly fed into 3DSimAttention without additional processing.  

Regardless of the paradigm adopted, the final output shape of the classification task remains (batch\_size, class=40).

-**Segmentation_Model.py** : This is a case of 3DSimAttention applied to the segmentation task. Segmentation can be regarded as fine-grained classification performed on individual points, and the output length of the model is consistent with the input length. For example:(batch_size, L, 3) -> (batch_size, L, instances).  
Here, the specific value of "instances" depends on the number of instances that need to be segmented in the task.
