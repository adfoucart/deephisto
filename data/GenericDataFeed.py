# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

Generic DataFeed class, containing the common parts of ArtefactDataFeed, EpitheliumDataFeed & WarwickDataFeed 
'''
import numpy as np

class GenericDataFeed:

    def __init__(self, params, db, generator=None):
        # General parameters
        self.directory = params['dataset_dir']
        self.tile_size = params['tile_size']
        self.db = db
        self.v = params['verbose'] if 'verbose' in params else False

        # SNOW parameters
        self.noisy = params['noisy'] if 'noisy' in params else False
        self.pNoise = params['pNoise'] if 'pNoise' in params else 0.5
        self.generative = params['generative'] if 'generative' in params else False
        self.generator = generator
        self.onlyPositives = params['onlyPositives'] if 'onlyPositives' in params else False
        self.tda = params['tda'] if 'tda' in params else False
        self.weak = params['weak'] if 'weak' in params else False
        self.annotations = params['annotations'] if 'annotations' in params else 'full'

        # Random sampling
        self.seed = params['random_seed'] if 'random_seed' in params else 56489143
        self.pointer = 0

        self.files_X = []
        self.files_Y = []
        self.idxs = []

    '''
    Batch generator
    Yield samples from dataset
    '''
    def next_batch(self, batch_size, max_iterations, forValidation=False):
        if( self.v ): print("Starting %d iterations"%(max_iterations))
        # iterations = sampling each image at least once
        np.random.seed(self.seed)
        for it in range(max_iterations):
            # Shuffle ids so we don't always go through the images in the same order
            np.random.shuffle(self.idxs)
            # print("Iteration: %d"%it)

            # For each slide : draw a random sample. Go through each image once before starting again with new seed
            for idx in self.idxs:
                yield self.get_sample(idx, batch_size, forValidation)

    '''
    Draw a validation set from the training set, using no data augmentation. By default, use 10% of the images in the training set to create validation set.
    '''
    def validation_set(self, validation_set_size, fractionOfTrainingSet=10):
        images_used = max([len(self.files_X)//fractionOfTrainingSet, 1])
        n_per_image = max([1,validation_set_size//images_used])

        Xval = np.zeros((validation_set_size, self.tile_size, self.tile_size, 3))
        Yval_seg = np.zeros((validation_set_size, self.tile_size, self.tile_size, 2))
        Yval_det = np.zeros((validation_set_size, 2))
        n = 0
        i = 0
        while n < validation_set_size:
            n_in_this_image = min([n_per_image, validation_set_size-n])
            X,Y_seg,Y_det = self.get_sample(self.idxs[i], n_in_this_image, True)
            Xval[n:n+n_in_this_image,:,:,:] = X.copy()
            Yval_seg[n:n+n_in_this_image,:,:,:] = Y_seg.copy()
            Yval_det[n:n+n_in_this_image] = Y_det.copy()
            i += 1
            n += n_in_this_image
        return Xval,Yval_seg,Yval_det
