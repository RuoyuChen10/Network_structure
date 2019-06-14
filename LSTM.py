# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2019
@author: Ruoyu Chen
The LSTM networks
"""
import tensorflow as tf

BATCH_SIZE = 10

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)

def bias_variable(shape,name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def cell(xt,ht_1,Ct_1,name):
    #xt:(None, 9, 3)  ht_1:(None,9, 9)  Ct_1:(None,9, 9)
    with tf.name_scope(name):
        with tf.name_scope('concat'):
            concat = tf.concat([ht_1,xt],2)  # size:(None, 9, 12)
            concat = tf.reshape(concat,[-1,12])  
        with tf.name_scope('Variable'):
            Wf = weight_variable([12,9], name='Wf')
            bf = bias_variable([9,9],name='bf')
            Wi = weight_variable([12,9], name='Wi')
            bi = bias_variable([9,9],name='bi')
            Wc = weight_variable([12,9], name='Wc')
            bc = bias_variable([9,9],name='bc')
            Wo = weight_variable([12,9], name='Wo')
            bo = bias_variable([9,9],name='bo')
        with tf.name_scope('ft'):
            ft = tf.nn.sigmoid(tf.add(tf.reshape(tf.matmul(concat, Wf),[-1,9,9]), bf), name='ft')
        with tf.name_scope('it'):
            it = tf.nn.sigmoid(tf.add(tf.reshape(tf.matmul(concat, Wi),[-1,9,9]), bi), name='it')
        with tf.name_scope('C1t'):
            C1t = tf.nn.tanh(tf.add(tf.reshape(tf.matmul(concat, Wc),[-1,9,9]), bc), name='C1t')
        with tf.name_scope('Ct'):
            Ct = tf.add(tf.matmul(ft, Ct_1), tf.reshape(tf.matmul(it, C1t),[-1,9,9]), name='Ct')
        with tf.name_scope('ot'):
            ot = tf.nn.sigmoid(tf.add(tf.reshape(tf.matmul(concat, Wo),[-1,9,9]), bo), name='ot')
        with tf.name_scope('ht'):
            ht = tf.matmul(ot, tf.nn.tanh(Ct))
    return ht, Ct


def LSTM(input):
# tf.transpose(X, [1,0,2,3])
    x = tf.transpose(input, [1,0,2,3])
    ht_1 = tf.constant(0.,shape= [BATCH_SIZE,9,9])
    Ct_1 = tf.constant(0.,shape = [BATCH_SIZE,9,9])
    for i in range(0,48):
        name = 'cell_'+ str(i+1)
        ht, Ct = cell(x[i],ht_1,Ct_1,name=name)
        ht_1 = ht
        Ct_1 = Ct


def main():
    with tf.name_scope('Input_data'):
        X = tf.placeholder(tf.float32, [BATCH_SIZE, 48, 9, 3], name="Input")
    LSTM(X)
    sess = tf.Session()
    writer = tf.summary.FileWriter("./logs", sess.graph)
    writer.close()

if __name__ == '__main__':
    main()
    