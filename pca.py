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
The MNIST database of handwritten digits with 784 features, raw data available at: http://yann.lecun.com/exdb/mnist/. 
It can be split in a training set of the first 60,000 examples, and a test set of 10,000 examples
    '''
    if os.path.exists('images.npy'):
        images = np.load('images.npy', allow_pickle=True)
        labels = np.load('labels.npy', allow_pickle=True)
    else:
        images, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
        np.save('images.npy', images)
        np.save('labels.npy', labels)
    
    return images, labels



def pca_fit(data_mat, p = 2):
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
    eig_val_index = eig_val_index[:-(p + 1) : -1]
    # 取前top_n_feat个特征后重构的特征向量矩阵reorganize eig vects, 
    reg_eig_vects = eig_vects[:, eig_val_index] 
    
    # 返回均值图像和特征向量矩阵
    return mean_removed, reg_eig_vects

def pca_apply(data_mat, reg_eig_vects):
    
    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)
    mean_removed = data_mat - mean_vals

    # 将数据转到新空间
    low_d_data_mat = mean_removed * reg_eig_vects
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals

    return low_d_data_mat.astype(np.float64), recon_mat.astype(np.float64)
    
def print_im(fn, im):
    im = im.reshape(28, 28, 1)
    cv2.imwrite(fn, im)

def task_a():
    '''
    就一组5的图片，计算均值和主要成分；
    '''
    images, labels = load_mnist()

    print(images.dtype)

    ims_of_5 = []
    for im, lb in zip(images, labels):
        if lb == '5':
            ims_of_5.append(im)

    #取前100个
    ims_of_5 = ims_of_5[:100]
    data_mat = np.array(ims_of_5)

    mean_images, reg_eig_vects = pca_fit(data_mat, p=2)

    # 显示该结果
    first_2_component, recon_images = pca_apply(data_mat, reg_eig_vects)

    if not os.path.exists('task_a_result'):
        os.makedirs('task_a_result')

    for i in range(mean_images.shape[0]):
        print_im('task_a_result/{}_mean.jpg'.format(i), mean_images[i])
        print_im('task_a_result/{}_recon.jpg'.format(i), recon_images[i])

    print('Mean images was saved to task_a_result folder')
    print('First 2 principal components is ', first_2_component)


def task_b():
    '''
    用train数据算不同p值的pca特征向量矩阵；用该特征向量矩阵对test图像做重构；
    '''
    images, labels = load_mnist()
    # np.random.shuffle(images)

    # 取前1000个
    data_mat = np.array(images[:1000])

    # 取一个测试图片
    
    if not os.path.exists('task_b_result'):
        os.makedirs('task_b_result')

    n_test = 10
    test_data_map = np.array(images[7000:7000+n_test])
    for p in [10, 50]:
        _mean_images, reg_eig_vects = pca_fit(data_mat, p=p)
        _pcdata, recons = pca_apply(test_data_map, reg_eig_vects)

        for n in range(n_test):
            print_im('task_b_result/n_{}_p_{}.jpg'.format(7000+n, p), recons[n])

        

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

task_b()