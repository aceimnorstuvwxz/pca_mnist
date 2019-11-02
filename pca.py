# 
# -*- coding: UTF-8 -*-
from sklearn.datasets import fetch_openml
import pickle
import numpy as np
import os
import cv2
import numpy.linalg as linalg


def load_mnist():
    '''
    加载mnist数据，数据首次下载后会缓存；
    '''
    if os.path.exists('images.npy'):
        images = np.load('images.npy', allow_pickle=True)
        labels = np.load('labels.npy', allow_pickle=True)
    else:
        images, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
        np.save('images.npy', images)
        np.save('labels.npy', labels)
    
    return images, labels



def pca_fit(data_mat, n_feat = 2):
    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)
    mean_removed = data_mat - mean_vals

    # 计算协方差矩阵（Find covariance matrix）
    cov_mat = np.cov(mean_removed, rowvar=0)

    # 计算特征值和特征向量
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))

    # 对特征值进行从小到大排序，argsort返回的是索引，即下标
    eig_val_index = np.argsort(eig_vals) 

    # 最大的前top_n_feat个特征的索引
    eig_val_index = eig_val_index[:-(n_feat + 1) : -1]
    # 取前top_n_feat个特征后重构的特征向量矩阵reorganize eig vects, 
    reg_eig_vects = eig_vects[:, eig_val_index] 
    
    # # 将数据转到新空间
    # low_d_data_mat = mean_removed * reg_eig_vects # shape: (100, top_n_feat), top_n_feat最大为特征总数
    # recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)
    
    # 返回均值图像和特征向量矩阵
    return mean_removed, reg_eig_vects

def print_im(fn, im):
    im = im.reshape(28, 28, 1)
    cv2.imwrite(fn, im)



def task_a():
    '''
 a. Compute the mean image and principal components for a set of images 
 (e.g. use the training images of ‘5’ in the mnist dataset). Display the
  mean image and the first 2 principal components (associated with the highest eigenvalues).  
    '''
    images, labels = load_mnist()

    ims_of_5 = []
    for im, lb in zip(images, labels):
        if lb == '5':
            ims_of_5.append(im)

    #取前100个
    ims_of_5 = ims_of_5[:100]

    data_mat = np.array(ims_of_5)

    mean_images, reg_eig_vects = pca_fit(data_mat, n_feat=2)

    for i in range(mean_images.shape[0]):
        im = mean_images[i]
        print_im('tmp/mean_{}.jpg'.format(i), im)
    
    print('mean images was saved to tmp folder')

    # 显示该结果
    print('First 2 principal components is ', reg_eig_vects)


def task_b():
    '''
b. Compute and display the reconstructions of a test image using the mean image and with p principal components associated with the p highest eigenvalues (e.g. Fig 10.12) with p=10 and p=50.  

    用train数据，对所有数字算pca特性向量矩阵，并用它来重构test图像；
    '''

def pca(data_mat, top_n_feat=99999999):

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
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat)) # 计算特征值和特征向量，shape分别为（784，）和(784, 784)

    eig_val_index = np.argsort(eig_vals)  # 对特征值进行从小到大排序，argsort返回的是索引，即下标

    eig_val_index = eig_val_index[:-(top_n_feat + 1) : -1] # 最大的前top_n_feat个特征的索引
    # 取前top_n_feat个特征后重构的特征向量矩阵reorganize eig vects, 
    # shape为(784, top_n_feat)，top_n_feat最大为特征总数
    reg_eig_vects = eig_vects[:, eig_val_index] 
    
    # 将数据转到新空间
    low_d_data_mat = mean_removed * reg_eig_vects # shape: (100, top_n_feat), top_n_feat最大为特征总数
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)
    
    return low_d_data_mat, recon_mat
task_a()