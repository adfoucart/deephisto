# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

BaseNetwork
Superclass for the networks

Built for TensorFlow 1.14
'''

import tensorflow as tf
import numpy as np

class BaseNetwork:

    '''
    params:
    * clf_name (required) -> name of the trained classifier
    * checkpoints_dir (required) -> where to save/load checkpoints
    * summaries_dir (required) -> where to save summaries
    * verbose (bool) (default False)
    * tile_size (int) (required)
    * lr (float) -> learning rate (default 1e-4)
    * eps (float) -> epsilon of the adam optimizer (default 1e-8)
    * convsPerRes (int) -> convolutions per residual unit (default 3)
    * output_classes (int) -> number of output classes for the segmentation (default 2)
    * autoencoder (string) -> path to a trained autoencoder to get pre-trained weights (default False)
    * feature_tensor (string) -> name of the feature tensor from the autoencoder, on which to start building the rest of the network
    * fixedAE (bool) -> set to true to disable fine-tuning of the encoder on the new data (if using autoencoder)  (default False)
    * trainAE (bool) -> train as autoencoder (unsupervised) (default False)
    * weak (bool or string) -> True -> use weak version of the network; 'weakish' -> use SoftWeak; False -> use normal. (default False)
    * generative (bool) -> Use "Generated Annotations" (default False)
    * generator (string) -> path to the pretrained generator DCNN (default None)
    '''
    def __init__(self, params):
        # General parameters
        self.clf_name = params['clf_name']
        self.checkpoints_dir = params['checkpoints_dir']
        self.summaries_dir = params['summaries_dir']
        self.v = params['verbose'] if 'verbose' in params else False

        # Network parameters
        self.tile_size = params['tile_size']
        self.lr = params['lr'] if 'lr' in params else 1e-4
        self.eps = params['eps'] if 'eps' in params else 1e-8
        self.width = params['width'] if 'width' in params else 64
        self.convsPerRes = params['convsPerRes'] if 'convsPerRes' in params else 3
        self.output_classes = params['output_classes'] if 'output_classes' in params else 2
        
        # SNOW parameters
        self.autoencoder = params['autoencoder'] if 'autoencoder' in params else None
        self.feature_tensor = params['feature_tensor'] if 'feature_tensor' in params else None
        self.fixedAE = params['fixedAE'] if 'fixedAE' in params else False
        self.trainAE = params['trainAE'] if 'trainAE' in params else False
        self.weak = params['weak'] if 'weak' in params else False
        self.generative = params['generative'] if 'generative' in params else False
        self.generator = params['generator'] if 'generator' in params else None

        self.seed = params['random_seed'] if 'random_seed' in params else 56489143

    '''
    Setup the graphs & sessions
    '''
    def setup(self):
        # Setup main graph & session
        tf.reset_default_graph()
        self.mainGraph = tf.Graph()
        self.sess = tf.Session(graph=self.mainGraph)

        # Restore autoencoder if needed & setup input placeholder
        with self.mainGraph.as_default():
            tf.random.set_random_seed(self.seed)
            # Loading AutoEncoder
            if( self.autoencoder != None ):
                saver = tf.train.import_meta_graph('%s.meta'%self.autoencoder, import_scope="ae")
                saver.restore(self.sess, self.autoencoder)
                self.X = self.mainGraph.get_tensor_by_name("ae/features/X:0")
            # From scratch
            else:
                self.X = tf.placeholder(tf.float32, [None,self.tile_size,self.tile_size,3], name='features/X')

        # Generator graph & session if needed + restore generator
        if( self.generative ):
            self.genGraph = tf.Graph()
            self.genSess = tf.Session(graph=self.genGraph)
            with self.genGraph.as_default():
                saver = tf.train.import_meta_graph('%s.meta'%self.generator, import_scope="generator")
                saver.restore(self.genSess, self.generator)
            self.genX = self.genGraph.get_operation_by_name("generator/features/X").outputs[0]
            self.genOutput = self.genGraph.get_operation_by_name("generator/output/segmentation").outputs[0]

        # Setup target placeholders
        with self.mainGraph.as_default():
            self.target_seg = tf.placeholder(tf.float32, [None, self.tile_size, self.tile_size, self.output_classes], name='target_seg')
            self.target_det = tf.placeholder(tf.float32, [None, self.output_classes], name='target_det')

    '''
    Prepare loss functions & training step
    '''
    def train(self):
        with self.mainGraph.as_default():
            if( self.trainAE == True ): # Train as AE with MSE
                l1 = sum(tf.reduce_sum(tf.abs(v)) for v in tf.trainable_variables() if 'weights' in v.name)
                self.loss = tf.losses.mean_squared_error(self.X, self.decoded)+1e-5*l1
                optimizer = tf.train.AdamOptimizer(self.lr, epsilon=self.eps, name='aeopt')
                trainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self.accuracy = self.loss
            else: # Train for segmentation with Cross Entropy
                detLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.detection, labels=self.target_det, name='loss/det/softmax'), name='loss/det/reduce')
                flat_logits = tf.reshape(self.segmentation, shape=(-1,self.output_classes), name='loss/seg/flat_logits')
                flat_target = tf.reshape(self.target_seg, shape=(-1,self.output_classes), name='loss/seg/flat_target')
                segLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_target, name='loss/seg/softmax'), name='loss/seg/reduce')
                
                # Weak & SoftWeak losses 
                if( self.weak == 'weakish' ):
                    self.loss = detLoss + segLoss
                elif( self.weak == True ):
                    self.loss = detLoss
                else:    # elif( self.weak == False ):
                    self.loss = segLoss

                if( self.fixedAE == True ):
                    trainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
                else:
                    trainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

                optimizer = tf.train.AdamOptimizer(self.lr, epsilon=self.eps, name='clfopt')
            
                pred_argmax = tf.argmax(self.segmentation, axis=-1, name='accuracy/pred_argmax', output_type=tf.int32)
                target_argmax = tf.argmax(self.target_seg, axis=-1, name='accuracy/target_argmax', output_type=tf.int32)
                self.accuracy = tf.contrib.metrics.accuracy(pred_argmax, target_argmax, name="accuracy")

            self.trainingStep = optimizer.minimize(self.loss, var_list=trainableVars)
            return self.trainingStep,self.loss,self.accuracy

    def predict_segmentation(self, X):
        return self.segmentation_softmax.eval(session=self.sess, feed_dict={self.X: X})

    def predict_detection(self, X):
        return self.detection_softmax.eval(session=self.sess, feed_dict={self.X: X})

    def predict_generator(self, X):
        return self.genSess.run(self.genOutput, feed_dict={self.genX: X})