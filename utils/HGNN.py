import math
import tensorflow as tf
import numpy as np
from utils import tf_util
import os
import sys



def HGNN_conv(X, G):
    if not isinstance(G, tf.Tensor):
        G = tf.convert_to_tensor(G)
    if not isinstance(X, tf.Tensor):
        X = tf.convert_to_tensor(X)
    conv1 = tf.nn.relu(tf.matmul(G, X))
    conv1 = tf.nn.dropout(conv1, 0.5)
    conv2 = tf.nn.relu(tf.matmul(G, conv1))
    return conv2


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    # x = np.array(x)
    # aa = tf.sum(np.multiply(x, x), 1)
    N = x.get_shape()[0].value

    x_2 = tf.multiply(x, x)
    aa = tf.reduce_sum(x_2, 1, keep_dims=True)
    aa = tf.tile(aa, [1, N])
    aa_T = tf.transpose(aa)
    # ab = x * x.T
    x_T = tf.transpose(x)
    ab = tf.matmul(x, x_T)
    w = tf.ones(aa.shape) * 2
    ab_2 = tf.multiply(w, ab)

    dist_mat = tf.add(aa, aa_T)
    dist_mat = tf.subtract(dist_mat, ab_2)
    # dist_mat = aa + aa.T - 2 * ab
    # dist_mat[dist_mat < 0] = 0
    zeros = tf.zeros(dist_mat.shape)
    c = tf.greater_equal(dist_mat, zeros)
    c = tf.cast(c, tf.float32)
    dist_mat = tf.multiply(dist_mat, c)
    # dist_mat = np.maximum(dist_mat, dist_mat.T)
    dist_mat = tf.sqrt(dist_mat)
    dist_mat_T = tf.transpose(dist_mat)
    c_1 = tf.greater_equal(dist_mat, dist_mat_T)
    c_1 = tf.cast(c_1, tf.float32)
    dist_mat_1 = tf.multiply(dist_mat, c_1)
    c_2 = tf.greater_equal(dist_mat_T, dist_mat, )
    c_2 = tf.cast(c_2, tf.float32)
    dist_mat_2 = tf.multiply(dist_mat_T, c_2)
    dist_mat = tf.add(dist_mat_1, dist_mat_2)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # if type(H) != list:
    #     return _generate_G_from_H(H, variable_weight)
    # else:
    #     # G = []
    #     for sub_H in H:
    #         G = generate_G_from_H(sub_H, variable_weight)
    #     return G
    G = _generate_G_from_H(H,variable_weight)
    return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # print('H',H.shape)
    # n_edge = H.shape[1]
    # I = tf.ones(n_edge)
    # I = tf.matrix_diag(I)
    # I = tf.expand_dims(I, 0)
    # I = tf.tile(I, [H.shape[0], 1, 1])
    # # the weight of the hyperedge
    # # W = np.ones(n_edge)
    # W = tf.ones(n_edge)
    # # the degree of the node
    # DV = tf.reduce_sum(H * W, axis=1)
    # DV = tf.expand_dims(DV, -1)
    # DV = tf.tile(DV, [1, 1, H.shape[1]])
    # DV = tf.to_float(DV)
    # I = tf.to_float(I)
    # DV = tf.multiply(DV,I)
    # DV2 = tf.pow(DV, -0.5)
    # # DV2 = tf.matrix_inverse(DV2)
    # # DV = DV*I
    # # the degree of the hyperedge
    # # DE = tf.reduce_sum(H, axis=0)
    # # DE = tf.expand_dims(DE,0)
    # # DE = tf.tile(DE, [H.shape[0], 1, 1])
    # # DE = tf.to_float(DE)
    # # # invDE = np.mat(np.diag(np.power(DE, -1)))
    # # invDE = tf.matrix_inverse(DE)
    # # invDE = tf.pow(DE,-1)
    # # DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    # # W = np.mat(np.diag(W))
    # # H = np.mat(H)
    # W = tf.matrix_diag(np.ones(n_edge))
    # W = tf.expand_dims(W, [0])
    # W = tf.tile(W, [H.shape[0], 1, 1])
    # # HT = H.T
    # HT = tf.transpose(H, perm=[0, 2, 1])
    # DV2 = tf.to_float(DV2)
    # H = tf.to_float(H)
    # W = tf.to_float(W)
    # # invDE = tf.to_float(invDE)
    # HT = tf.to_float(HT)
    # if variable_weight:
    #     DV2_H = DV2 * H
    #     # invDE_HT_DV2 = invDE * HT * DV2
    #     # return DV2_H, W, invDE_HT_DV2
    #     return DV2_H
    # else:
    #     G = tf.matmul(DV2 , H)
    #     G = tf.matmul(G, W)
    #     # G = tf.matmul(G, invDE)
    #     G = tf.matmul(G, HT)
    #     G = tf.matmul(G, DV2)
    #
    #     return G
    # node digree
    DV = tf.reduce_sum(H,axis=1)
    DV = tf.matrix_diag(DV)
    I = tf.ones(H.shape[0])
    I = tf.expand_dims(I, 0)
    I = tf.tile(I, [H.shape[0], 1])
    DV2 = tf.pow(DV,0.5)
    DV2 = tf.matrix_inverse(DV2)
    HT = tf.transpose(H)
    G = tf.matmul(DV2, H)
    # G = tf.matmul(G, W)
    # G = tf.matmul(G, invDE)
    G = tf.matmul(G, HT)
    G = tf.matmul(G, DV2)
    return G
def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    # dis_mat = np.array(dis_mat)
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    # n_edge = n_obj
    H = tf.ones(n_obj)
    for center_idx in range(n_obj):
        # dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx, :]
        # nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        # avg_dis = np.average(dis_vec)
        # if not np.any(nearest_idx[:k_neig] == center_idx):
        #     nearest_idx[k_neig - 1] = center_idx
        nearest_val = tf.nn.top_k(-dis_vec, k=k_neig+1).values
        min_val = tf.reduce_min(nearest_val)
        c = tf.greater_equal(-dis_vec, min_val)
        c = tf.cast(c, tf.float32)
        H = tf.concat([H, c], 0)
    H = H[n_obj:]
    H = tf.reshape(H, [n_obj, n_obj])
    return H


def construct_H_with_KNN(X, K_neigs=[3], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    # H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        # if not split_diff_scale:
        #     H = hyperedge_concat(H, H_tmp)
        # else:
        #     # H.append(H_tmp)
    return H_tmp

