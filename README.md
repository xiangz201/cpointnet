## PointNet: *Deep Learning on Point Sets for 3D Classification and Segmentation*
根据pointnet和hgnn修改的框架，使用pointnet提取特征，使用hgnn方法产生邻接矩阵H，图卷积层D，图神经网络结构得到预测值
### Introduction
This work is based on [pointnet](https://arxiv.org/abs/1612.00593), [project webpage](http://stanford.edu/~rqi/pointnet) and [hgnn](http://gaoyue.org/paper/HGNN.pdf)，[project webpage](https://github.com/iMoonLab/HGNN)
### Framework
modelnet40 ---->pointnet----->features------>邻接矩阵H------>GCN层D
                                |                             |
                                |                             |      
                                |------------------------------>FC---->predict
