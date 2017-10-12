import tensorflow as tf
import numpy as np
import cPickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

DTYPE = tf.float64

FIELD_SIZES = [0] * 26
with open('/home/ysun1/tensorflow-starter-kit_trunk/product-nets-master/data_cretio/featindex_thres1M20.txt') as fin:
    for line in fin:
        line = line.strip().split(':')
        if len(line) > 1:
            f = int(line[0]) - 1
            FIELD_SIZES[f] += 1
print 'field sizes:', FIELD_SIZES
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-2
MAXVAL = 1e-2


def read_data(file_name):
    X = []
    y = []
    with open(file_name) as fin:
        for line in fin:
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
            y.append(y_i)
            X.append(X_i)
    y = np.reshape(np.array(y), (-1, 1))
    X = libsvm_2_coo(X, (len(X), INPUT_DIM)).tocsr()
    return X, y

def weight_bias(W_shape, b_shape, bias_init=0.):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b


def dense_layer(x, W_shape, b_shape, activation):
    W, b = weight_bias(W_shape, b_shape)
    return activation(tf.matmul(x, W) + b)


def flat_highway_gate_layer(first, second, carry_bias=0.,
                            activation=tf.tanh):
    X_shape = tf.shape(first)
    first = tf.reshape(first, [X_shape[0] * X_shape[1], X_shape[2]])
    second = tf.reshape(second, [X_shape[0] * X_shape[1], X_shape[2]])
    first_second = tf.concat(1, [first, second])
    W_T_shape = [X_shape[2], X_shape[2]]
    W_H_shape = [X_shape[2] * 2, X_shape[2]]
    b_shape = [X_shape[2]]
    W_T, b_T = weight_bias(W_T_shape, b_shape)
    # W_T_2, b_T_2 = weight_bias(W_T_shape, b_shape)
    H = dense_layer(first_second, W_H_shape, b_shape, activation)
    x = tf.add(first, second)
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
    # T2 = tf.sigmoid(tf.matmul(second, W_T_2) +
    #                 b_T_2, name='transform_gate_2')
    C = tf.sub(1.0, T, name="carry_gate")
    y = tf.add(tf.mul(H, T), tf.mul(x, C))  # y = (H * T) + (x * C)
    return y.reshape(X_shape)


def get_pair_indices(sequence_length): # handles a single sample, this works.
    """get indices for each pair in a sample x"""
    pair_indices = []
    for i in range(sequence_length):
        for j in range(i + 1, sequence_length):
            pair_indices.append([i, j])
    return (np.array([e[0] for e in pair_indices]),np.array([e[1] for e in pair_indices]))


def get_batch_pair_indices(sequence_length):#handles batch
    """get the indices for a batch"""
    indices = get_pair_indices(sequence_length)
    comb_num = len(indices)
    batch_indices = []
    for i in range(batch_size):
        for j in range(len(indices)):
            batch_indices.append([[i, indices[j][0]], [i, indices[j][1]]])
    #print batch_indices
    batch_indices = np.array(batch_indices)
    #print batch_indices
    return (batch_indices[:, 0].reshape(batch_size, comb_num, 2),
            batch_indices[:, 1].reshape(batch_size, comb_num, 2))


def _gate(c1, c2, gate_type='sum'):
    """pair interaction method"""
    if gate_type == 'sum':
        return tf.add(c1, c2)
    if gate_type == 'mul':
        return tf.mul(c1, c2)
    if gate_type == 'avg':
        return tf.mul(tf.fill(tf.shape(c2), 0.5), tf.add(c1, c2))
    if gate_type == 'highway':
        return flat_highway_gate_layer(c1, c2)
    if gate_type == 'sum_tanh':
        return tf.tanh(tf.add(c1, c2))
    if gate_type == 'mul_tanh':
        return tf.tanh(tf.mul(c1, c2))
    if gate_type == 'p_norm':
        p = 2
        return tf.pow(tf.add([tf.pow(c1, p), tf.pow(c2, p)]), 1 / p)


def _norm(interactions, norm_type='l2'):
    """metrics for max-pooling"""
    if norm_type == 'l2':
        return tf.reduce_sum(tf.square(interactions), 2)
    if norm_type == 'l1':
        return tf.reduce_sum(tf.abs(interactions), 2)
    if norm_type == 'avg':
        return tf.reduce_sum(interactions, 2)

def  _getpartition(max_indices,n):
    l = [0]*n
    for i in max_indices.as_list():
        l[i]=1
    return l

def pair_wise_interaction_and_max_pooling(X1, X2, k,
                                          sequence_length,
                                          first_indices,
                                          second_indices,
                                          gate_type='sum',
                                          norm_type='l2'
                                          ):
    """
        input X: embedding inputs or max-pooling outputs
        output: pair wise interactions and max-pooling results
    """
    first_element = tf.transpose(tf.gather(tf.transpose(X1,[1,0,2]), first_indices),[1,0,2])
    second_element = tf.transpose(tf.gather(tf.transpose(X2,[1,0,2]), second_indices),[1,0,2])
    interactions = _gate(first_element, second_element, gate_type)
    norms = _norm(interactions, norm_type)
    normvec = tf.reduce_sum(norms,0)
    inter_trans = tf.transpose(interactions,[1,0,2])
    _, max_indices = tf.nn.top_k(normvec, k)
    #pooling_indices = _get_max_pooling_indices(max_indices, k)
    # print(pooling_indices.get_shape().as_list())
    inter_trans = tf.transpose(interactions,[1,0,2])
    #print(inter_trans.get_shape().as_list())
    #print(max_indices.get_shape().as_list())
    pooling_rst = tf.transpose(tf.gather(inter_trans, max_indices),[1,0,2])
    #pooling_rst = tf.transpose(tf.dynamic_partition(inter_trans, _getpartition(max_indices),1),[1,0,2])
    return pooling_rst


def interaction(X1, X2, k,#k is sequence_length.
                    sequence_length,#WTF is sequence length? set to be 24, seems to be feature dimension
                    first_indices,
                    second_indices,
                    gate_type='sum',
                    norm_type='l2'):
    """
        input X: embedding inputs or max-pooling outputs
        output: Full connection rst after interactions and max-pooling
    """
    #print(X1.shape)
    #with tf.name_scope("interaction_and_pooling_layer_%d" % i):
    rst = pair_wise_interaction_and_max_pooling(X1,X2, k, sequence_length,first_indices, second_indices)

    return rst

def interaction2(left, right, k,#k is sequence_length.
                    sequence_length,#WTF is sequence length? set to be 24, seems to be feature dimension
                    first_indices,
                    second_indices,
                    gate_type='sum',
                    norm_type='l2'):
    """
        input X: embedding inputs or max-pooling outputs
        output: Full connection rst after interactions and max-pooling
    """
    #print(X1.shape)
    #with tf.name_scope("interaction_and_pooling_layer_%d" % i):
    rst = pair_wise_interaction_and_max_pooling2(left,right, k, sequence_length,first_indices, second_indices)

    return rst

def pair_wise_interaction_and_max_pooling2(left, right, k,
                                          sequence_length,
                                          first_indices,
                                          second_indices,
                                          gate_type='sum',
                                          norm_type='l2'
                                          ):
    """
        input X: embedding inputs or max-pooling outputs
        output: pair wise interactions and max-pooling results
    """
    first_element = left
    second_element = right
    interactions = _gate(first_element, second_element, gate_type) 
    norms = _norm(interactions, norm_type)
    normvec = tf.reduce_sum(norms,1)
    _, max_indices = tf.nn.top_k(normvec, k)
    #pooling_indices = _get_max_pooling_indices(max_indices, k)
    # print(pooling_indices.get_shape().as_list())
    #inter_trans = tf.transpose(interactions,[1,0,2])
    #print(inter_trans.get_shape().as_list())
    #print(max_indices.get_shape().as_list())
    #1. return list of tensors
    #2. use dynamic_partition
    #Q. difference?
    pooling_rst = tf.transpose(tf.gather(interactions, max_indices),[1,0,2])
    print(pooling_rst.get_shape().as_list())
    #pooling_rst = tf.transpose(tf.dynamic_partition(inter_trans, _getpartition(max_indices,inter_trans.get_shape().as_list()[1]),2)[0],[1,0,2])
    #pooling_rst = tf.transpose(tf.dynamic_partition(inter_trans, _getpartition(max_indices),1),[1,0,2])
    return pooling_rst

def full_connection(X):
    """
        Input: max-pooling results
        Output: concat rst or fc rst
    """
    X_shape = X.get_shape().as_list()
    rst = tf.reshape(X, [X_shape[0], X_shape[1] * X_shape[2]])
    return rst


def _get_max_pooling_indices(indices, k):
    """
        refers from tianyao's code
        get the indices for the max-pooling
    """
    batch_indices = tf.range(batch_size)
    batch_indices = tf.expand_dims(batch_indices, 1)
    batch_indices = tf.expand_dims(batch_indices, 2)#added 2 more dimensions
    batch_indices = tf.tile(batch_indices, [1, k, 1])
    indices = tf.expand_dims(indices, 2)
    return tf.concat([batch_indices, indices],2)
 
