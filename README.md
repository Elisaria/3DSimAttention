# 3DSimAttention
This project is the official implementation of the paper titled 3DLSHAttention: Efficient 3D Point Cloud Processing via Locality-Sensitive Hashing Attention.
## About
-**Core Purpose** : Designing to enhance the efficiency and generalization capability of Transformers in processing 3D point clouds.  
-**Use Case** : 3D point cloud classification, segmentation.  
-**Tech Stack** : Built with TensorFlow 2.14+ (GPU support requires CUDA 11.2+) and Python 3.8+.  
## File Description
-**Classification_Model.py** : This is a classification model case based on the ModelNet40 dataset. The input point cloud P has a shape of \((\text{batch_size}=32, L=2048, 3)\).  
When the Neighborhood-Wise (NW) paradigm is adopted : P first undergoes sampling and grouping to generate neighborhoods with a shape of \((\text{batch_size}, n=72, K=32, 3)\);Then, pooling is applied to obtain latent features with a shape of \((\text{batch_size}, n=72, d=32)\).  
When the Point-Wise (PW) paradigm is used:P is directly fed into 3DSimAttention without additional processing. Regardless of the paradigm adopted, the final output shape of the classification task remains \((\text{batch_size}, \text{class}=40)\).
