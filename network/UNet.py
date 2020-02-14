# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

UNet
From Ronneberger et al
U-Net: Convolutional Networks for Biomedical Image Segmentation,
MICCAI 2015
http://dx.doi.org/10.1007/978-3-319-24574-4_28

Built for TensorFlow 1.14
'''

import tensorflow as tf
import numpy as np
from .BaseNetwork import BaseNetwork

class UNet(BaseNetwork):

    def __init__(self, params):
        super().__init__(params)
        self.isTraining = params['isTraining'] if 'isTraining' in params else False

    def add_conv(self, input_, channels, ks, name):
        return tf.contrib.layers.conv2d(input_, channels, ks, 1, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s'%(name))

    def add_mp(self, input_, ks, name):
        return tf.contrib.layers.max_pool2d(input_, ks, ks, 'SAME', scope='%s'%name)

    def add_dropout(self, input_, name):
        return tf.contrib.layers.dropout(input_, keep_prob=0.5, scope='%s'%name, is_training=self.isTraining)

    def add_upconv(self, input_, channels, ks, name):
        return tf.contrib.layers.conv2d_transpose(input_, channels, ks, ks, 'SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer=tf.zeros_initializer(), scope='%s'%name)

    def create_network(self):
        self.setup()

        with self.mainGraph.as_default():
            ''' TODO - AutoEncoder version
            # Get pre-trained tensors from autoencoder
            if( self.autoencoder != None ):
                r1 = self.mainGraph.get_tensor_by_name("ae/res1/mp/MaxPool:0")
                r2 = self.mainGraph.get_tensor_by_name("ae/res2/mp/MaxPool:0")
                r3 = self.mainGraph.get_tensor_by_name("ae/res3/mp/MaxPool:0")
            else: # Or create residual units from scratch
                r1 = self.add_residual(self.X, 'res1', True, 64)
                r2 = self.add_residual(r1, 'res2', True, 128)
                r3 = self.add_residual(r2, 'res3', True, 256)'''
            conv11 = self.add_conv(self.X, 64, 3, 'l1/conv1')
            conv12 = self.add_conv(conv11, 64, 3, 'l1/conv2')
            pool1 = self.add_mp(conv12, 2, 'l1/pool')

            conv21 = self.add_conv(pool1, 128, 3, 'l2/conv1')
            conv22 = self.add_conv(conv21, 128, 3, 'l2/conv2')
            pool2 = self.add_mp(conv22, 2, 'l2/pool')

            conv31 = self.add_conv(pool2, 256, 3, 'l3/conv1')
            conv32 = self.add_conv(conv31, 256, 3, 'l3/conv2')
            pool3 = self.add_mp(conv32, 2, 'l3/pool')

            conv41 = self.add_conv(pool3, 512, 3, 'l4/conv1')
            conv42 = self.add_conv(conv41, 512, 3, 'l4/conv2')
            pool4 = self.add_mp(conv42, 2, 'l4/pool')

            conv51 = self.add_conv(pool4, 1024, 3, 'l5/conv1')
            conv52 = self.add_conv(conv51, 1024, 3, 'l5/conv2')
            drop5 = self.add_dropout(conv52, 'l5/drop')

            up4 = self.add_upconv(drop5, 512, 2, 'l4/up')
            concat4 = tf.concat([conv42, up4], axis=3, name='l4/concat')
            conv43 = self.add_conv(concat4, 512, 3, 'l4/conv3')
            conv44 = self.add_conv(conv43, 512, 3, 'l4/conv4')

            up3 = self.add_upconv(conv44, 256, 2, 'l3/up')
            concat3 = tf.concat([conv32, up3], axis=3, name='l3/concat')
            conv33 = self.add_conv(concat3, 256, 3, 'l3/conv3')
            conv34 = self.add_conv(conv33, 256, 3, 'l3/conv4')

            up2 = self.add_upconv(conv34, 128, 2, 'l2/up')
            concat2 = tf.concat([conv22, up2], axis=3, name='l2/concat')
            conv23 = self.add_conv(concat2, 128, 3, 'l2/conv3')
            conv24 = self.add_conv(conv23, 128, 3, 'l2/conv4')

            up1 = self.add_upconv(conv24, 64, 2, 'l1/up')
            concat1 = tf.concat([conv12, up1], axis=3, name='l1/concat')
            conv13 = self.add_conv(concat1, 64, 3, 'l1/conv3')
            conv14 = self.add_conv(conv13, 64, 3, 'l1/conv4')

            self.segmentation = self.add_conv(conv14, self.output_classes, 1, 'classifier/segmentation')
            self.segmentation_softmax = tf.nn.softmax(self.segmentation, name='output/segmentation')

            self.detection = tf.reduce_max(self.segmentation, [1,2], name="classifier/detection")
            self.detection_softmax = tf.nn.softmax(self.detection, name='output/detection')