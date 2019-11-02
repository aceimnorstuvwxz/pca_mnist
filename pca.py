# 
# -*- coding: UTF-8 -*-
import tensorflow.examples.tutorials.mnist.input_data as input_data
import pickle
import numpy as np
import os
import cv2


# def load_mnist():
#     '''
#     加载mnist数据，数据首次下载后会缓存；
#     '''
#     if os.path.exists('images.npy'):
#         images = np.load('images.npy', allow_pickle=True)
#         labels = np.load('labels.npy', allow_pickle=True)
#     else:
#         images, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
#         np.save('images.npy', images)
#         np.save('labels.npy', labels)
    
#     return images, labels


def task_a():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    images = mnist.train.images
    labels = mnist.train.labels
    '''
 a. Compute the mean image and principal components for a set of images 
 (e.g. use the training images of ‘5’ in the mnist dataset). Display the
  mean image and the first 2 principal components (associated with the highest eigenvalues).  
    '''

    ims_of_5 = []
    for im, lb in zip(images, labels):
        if lb == '5':
            ims_of_5.append(im)

    print('count of 5=', len(ims_of_5))

    np.array()

    im = ims_of_5[0]
    im = im.reshape(28, 28, 1)
    cv2.imwrite('a.jpg', im)

def pca(data_mat, top_n_feat=99999999):
    """ 
    主成分分析：  
    输入：矩阵data_mat ，其中该矩阵中存储训练数据，每一行为一条训练数据  
         保留前n个特征top_n_feat，默认全保留
    返回：降维后的数据集和原始数据被重构后的矩阵（即降维后反变换回矩阵）
    """  

    # 获取数据条数和每条的维数 
    num_data,dim = data_mat.shape  
    print(num_data)  # 100
    print(dim)   # 784

    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)  #shape:(784,)
    mean_removed = data_mat - mean_vals # shape:(100, 784)

    # 计算协方差矩阵（Find covariance matrix）
    cov_mat = np.cov(mean_removed, rowvar=0) # shape：(784, 784)

    # 计算特征值(Find eigenvalues and eigenvectors)
    eig_vals, eig_vects = linalg.eig(mat(cov_mat)) # 计算特征值和特征向量，shape分别为（784，）和(784, 784)

    eig_val_index = argsort(eig_vals)  # 对特征值进行从小到大排序，argsort返回的是索引，即下标

    eig_val_index = eig_val_index[:-(top_n_feat + 1) : -1] # 最大的前top_n_feat个特征的索引
    # 取前top_n_feat个特征后重构的特征向量矩阵reorganize eig vects, 
    # shape为(784, top_n_feat)，top_n_feat最大为特征总数
    reg_eig_vects = eig_vects[:, eig_val_index] 
    
    # 将数据转到新空间
    low_d_data_mat = mean_removed * reg_eig_vects # shape: (100, top_n_feat), top_n_feat最大为特征总数
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)
    
    return low_d_data_mat, recon_mat
task_a()