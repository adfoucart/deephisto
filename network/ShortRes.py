# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

ShortRes - Short Residual network
A simple straightforward network with residual units.

Built for TensorFlow 1.4
'''

import tensorflow as tf
import numpy as np
import BaseNetwork
# from dhutil.network import initialize_uninitialized

class ShortRes(BaseNetwork):

    '''
    Add Residual Unit
     --- conv2d[3x3] --- conv2d[3x3] --- conv2d[3x3] + --- (maxpool)
      |----------------------------------------------|
    '''
    def add_residual(self, net_in, name, with_mp=True, width=None, n_convs=3):
        if width == None: width = net_in.get_shape()[3]

        net_ = net_in
        for i in range(n_convs):
            net_ = tf.contrib.layers.conv2d(net_, width, 3, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s/%d'%(name, i+1))
        
        net = tf.add(net_in, net_, name='%s/add'%name)

        if with_mp:
            net = tf.contrib.layers.max_pool2d(net, 2, 2, 'SAME', scope='%s/mp'%name)
        return net

    '''
    Add Upsampling Layer
     --- tConv[2x2] ---  
    '''
    def add_up(self, net_in, name, width=None, stride=2):
        if width==None: width = net_in.get_shape()[3]

        return tf.contrib.layers.conv2d_transpose(net_in, width, stride, stride, 'SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s'%name)

    '''
    Create full network architecture
     --- r1 -- r2 -- r3 -- up1 -- r4 -- up2 -- r5 -- out
    '''
    def create_network(self):
        with self.mainGraph.as_default():
            if( self.feature_tensor != None ):
                net = self.mainGraph.get_tensor_by_name("ae/features/%s:0"%self.feature_tensor)
            else:
                net = tf.contrib.layers.conv2d(self.X, self.width, 1, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='features/widen')

                net = self.add_residual(net, 'features/res1', True, self.width, self.convsPerRes)
                net = self.add_residual(net, 'features/res2', False, self.width, self.convsPerRes)
                net = self.add_residual(net, 'features/res3', True, self.width, self.convsPerRes)
            
            if( self.trainAE == True ):
                net = tf.contrib.layers.conv2d(net, 16, 3, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='reduce_width')
                net = self.add_residual(net, 'res4', True, 16, self.convsPerRes)    # -> /8
                net = self.add_up(net, 'up1', 16)    # -> /4
                net = self.add_up(net, 'up2', 16)  # -> /2
                self.decoded = self.add_up(net, 'up3', 3)    # -> /1
            else:
                net = self.add_up(net, 'classifier/up1')
                net = self.add_residual(net, 'classifier/res4', False, self.width, self.convsPerRes)
                net = self.add_up(net, 'classifier/up2')
                net = self.add_residual(net, 'classifier/res5', False, self.width, self.convsPerRes)

                self.segmentation = tf.contrib.layers.conv2d(net, self.output_classes, 1, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='classifier/segmentation')
                self.segmentation_softmax = tf.nn.softmax(self.segmentation, name='output/segmentation')
                # self.detection = tf.reduce_mean(self.segmentation, [1,2], name="classifier/detection")
                self.detection = tf.reduce_max(self.segmentation, [1,2], name="classifier/detection")
                # self.detection = tf.contrib.layers.conv2d(self.segmentation, 2, (self.segmentation.get_shape()[1], self.segmentation.get_shape()[2]), 1, padding='VALID', activation_fn=None, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='classifier/detection')
                self.detection_softmax = tf.nn.softmax(self.detection, name='output/detection')
