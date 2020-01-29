# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

PAN - Perfectly Adequate Network
A Network that uses the state-of-the-art ideas of Residual Units + U-net like shortcuts + combining outputs from different levels, and is therefore perfectly adequate 
for most common computer vision tasks, even though it's not particularly innovative on its own.

Built for TensorFlow 1.4
'''

import tensorflow as tf
import numpy as np
import BaseNetwork
# from dhutil.network import initialize_uninitialized

class PAN(BaseNetwork):

    '''
    Add Residual Unit
     --- conv2d[3x3] --- conv2d[3x3] -- + --- (maxpool)
      |------------ conv2d[1x1] --------|
    '''
    def add_residual(self, net_in, name, with_mp=True, width=None, n_convs=None):
        if width == None: width = net_in.get_shape()[3]
        if n_convs == None: n_convs = self.convsPerRes 

        net_ = net_in
        for i in range(n_convs):
            net_ = tf.contrib.layers.conv2d(net_, width, 3, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s/%d'%(name, i+1))
        
        shortcut = tf.contrib.layers.conv2d(net_in, width, 1, 1, activation_fn=None, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s/shortcut'%name)
        net = tf.add(net_, shortcut, name='%s/add'%name)

        if with_mp:
            net = tf.contrib.layers.max_pool2d(net, 2, 2, 'SAME', scope='%s/mp'%name)

        return net

    '''
    Add Upsampling Layer
     --- tConv[2x2] --- conv2d[3x3] -- + ---
      |--tConv[1x1]--------------------|  
    '''
    def add_up(self, net_in, name, width=None, stride=2):
        if width==None: width = net_in.get_shape()[3]

        net_ = net_in
        net_ = tf.contrib.layers.conv2d_transpose(net_in, width, stride, stride, 'SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s/up'%name)
        net_ = tf.contrib.layers.conv2d(net_, width, 3, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s/conv'%name)

        shortcut = tf.contrib.layers.conv2d_transpose(net_in, width, 1, stride, 'SAME', activation_fn=None, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s/shortcut'%name)

        return tf.add(net_, shortcut, name='%s/add'%name)

    '''
    Create full network architecture

                                |-------------------out3---|
                                |      |------------out2-- +---out
    -- r1 -- r2 -- r3 -- r4 -- ur1 -- ur2 -- ur3 -- out1---|
       |      |------------------------|      |
       |--------------------------------------|
    '''
    def create_network(self):
        with self.mainGraph.as_default():
            # Get pre-trained tensors from autoencoder
            if( self.autoencoder != None ):
                r1 = self.mainGraph.get_tensor_by_name("ae/res1/mp/MaxPool:0")
                r2 = self.mainGraph.get_tensor_by_name("ae/res2/mp/MaxPool:0")
                r3 = self.mainGraph.get_tensor_by_name("ae/res3/mp/MaxPool:0")
            else: # Or create residual units from scratch
                r1 = self.add_residual(self.X, 'res1', True, 64)
                r2 = self.add_residual(r1, 'res2', True, 128)
                r3 = self.add_residual(r2, 'res3', True, 256)

            r4 = self.add_residual(r3, 'res4', False, 512)

            ur1 = self.add_up(r4, 'ures1', 256)
            ur2 = self.add_up(tf.concat([ur1,r2], axis=3, name='concat_ur2'), 'ures2', 128)
            ur3 = self.add_up(tf.concat([ur2, r1], axis=3, name='concat_ur3'), 'ures3', 64)

            out_1 = ur3
            out_2 = tf.image.resize_bilinear(ur2, (self.tile_size, self.tile_size), name='resize_ur2')
            out_3 = tf.image.resize_bilinear(ur1, (self.tile_size, self.tile_size), name='resize_ur1')

            # For autoencoder -> output directly taken from out_3 with a convolution to get a RGB image
            if( self.trainAE == True ):
                self.decoded = tf.contrib.layers.conv2d(out_3, 3, 1, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='reduce_width')
            else: # For each of the three output, get output segmentation & patch labels
                out_1_seg = tf.contrib.layers.conv2d(out_1, self.output_classes, 1, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='out_1_seg')
                out_2_seg = tf.contrib.layers.conv2d(out_2, self.output_classes, 1, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='out_2_seg')
                out_3_seg = tf.contrib.layers.conv2d(out_3, self.output_classes, 1, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='out_3_seg')

                out_1_det = tf.reduce_max(out_1_seg, [1, self.output_classes], name='out_1_det')
                out_2_det = tf.reduce_max(out_2_seg, [1, self.output_classes], name='out_2_det')
                out_3_det = tf.reduce_max(out_3_seg, [1, self.output_classes], name='out_3_det')

                # Final output
                net = out_1_seg+out_2_seg+out_3_seg
                self.segmentation = tf.contrib.layers.conv2d(net, self.output_classes, 1, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='classifier/segmentation')
                self.segmentation_softmax = tf.nn.softmax(self.segmentation, name='output/segmentation')

                self.detection = out_1_det+out_2_det+out_3_det
                self.detection_softmax = tf.nn.softmax(self.detection, name='output/detection')