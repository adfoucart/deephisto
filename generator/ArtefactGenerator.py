import os
import math
import time

import numpy as np
from skimage.io import imread
from skimage.measure import regionprops,label
from skimage.transform import resize

from . import DataGenerator

class ArtefactGenerator(DataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if( self.timeit ):  self.log_time += [{'event': 'init artefact generator', 'time': time.time()}]

        if( self.mag not in ['1.25x', '2.50x', 'both'] ):
            raise ValueError('mag must be "1.25x", "2.5x" or "both"')

        x_dir = os.path.join(self.directory, self.folder_x)
        y_dir = os.path.join(self.directory, self.folder_y)

        all_tiles = [f for f in os.listdir(x_dir) if '_rgb' in f]
        if( self.mag == '1.25x' ):
            all_tiles = [f for f in all_tiles if '1.25x' in f]
        elif( self.mag == '2.5x' ):
            all_tiles = [f for f in all_tiles if '2.5x' in f]

        self.files_x = [os.path.join(x_dir, f) for f in all_tiles]
        self.files_y = [os.path.join(y_dir, f.replace('_rgb', '_anno')) for f in all_tiles]
        
        # Can't put all the dataset in RAM here -> have to load it on the fly
        # -> really need to re-do the threading stuff to improve performance
        # self.images_x = [self._preprocess(imread(f)) for f in self.files_x]
        # self.images_y = [imread(f)>0 for f in self.files_y]
        
        self.idxs = np.arange(len(self.files_x))
        self.keep_idxs = self.idxs.copy()

        if( self.pPositive is not False ):
            self.keep_idxs = []
            for idx,f in enumerate(self.files_y):
                if( imread(f).max() > 0 ): 
                    self.keep_idxs += [idx]
            
            self.keep_idxs = np.array(self.keep_idxs)

        self.isset = True

        self.split_train_val()

        self.batches_per_epoch = len(self.keep_idxs)//self.batch_size
        if( self.timeit ): self.log_time += [{'event': 'end init', 'time': time.time()}]
    
    def get_epoch(self, idxs, e):
        if( self.timeit ): self.log_time += [{'event': 'in get_epoch', 'time': time.time()}]
        
        if( self.pPositive is not False ):
            # we need to have x % of the idxs from positive samples (keep_idxs), and the rest from negative.
            # by doing the selection here, we ensure that each epoch has the right balance, but we should be 
            # aware that the balance may be wrong at the level of the mini-batch.
            take_positive = np.random.random((len(idxs),)) <= self.pPositive # -> will be true everywhere if pPositive = 1
            idxs = [idx for idx,pos in zip(idxs,take_positive) if (idx in self.keep_idxs and pos) or (idx not in self.keep_idxs and not pos)]
        
        for idb in range(self.batches_per_epoch):
            if( self.timeit ): self.log_time += [{'event': 'in batches loop', 'time': time.time()}]
            batch_x = np.array([self._preprocess(imread(self.files_x[idx])) for idx in idxs[idb*self.batch_size:(idb+1)*self.batch_size]])
            batch_y_1 = np.array([imread(self.files_y[idx])>0 for idx in idxs[idb*self.batch_size:(idb+1)*self.batch_size]])
            if( self.timeit ): self.log_time += [{'event': 'opened images from disk', 'time': time.time()}]
            batch_y = np.zeros(list(batch_y_1.shape) + [2,])
            batch_y[:,:,:,1] = batch_y_1
            batch_y[:,:,:,0] = 1-batch_y[:,:,:,1]
            if self.gamodel is not False:
                for i in range(batch_y.shape[0]):
                    if batch_y[i,:,:,1].mean() < self.maxPositiveAreaForGenerator \
                            and np.random.random() < self.pNoise :
                        batch_y[i] = self.gamodel.predict(np.array([batch_x[i]]))
            yield self._augment(*self._random_crop(batch_x, batch_y), idb)

    def _random_crop(self, batch_x, batch_y):
        """Randomly takes a cropped region of self.tile_size from each image in the mini batch"""
        if( self.timeit ): self.log_time += [{'event': 'pre-crop', 'time': time.time()}]
        shape_images = batch_y[0].shape
        shape_tiles = self.batch_y_shape[1:3]
        max_rt = np.array([shape_images[0]-shape_tiles[0], shape_images[1]-shape_tiles[1]])
        rts = (np.random.random((batch_x.shape[0], 2))*(max_rt)).astype('int')
        cropped_batch_x = np.zeros((len(batch_x),)+self.batch_x_shape[1:])
        cropped_batch_y = np.zeros((len(batch_x),)+self.batch_y_shape[1:])
        for i,(bx,by,rt) in enumerate(zip(batch_x,batch_y,rts)):
            cropped_batch_x[i] = bx[rt[0]:rt[0]+self.tile_size[0],rt[1]:rt[1]+self.tile_size[1]]
            cropped_batch_y[i] = by[rt[0]:rt[0]+self.tile_size[0],rt[1]:rt[1]+self.tile_size[1]]
        if( self.timeit ): self.log_time += [{'event': 'random crop', 'time': time.time()}]
        return cropped_batch_x,cropped_batch_y


    def get_validation_set(self):
        if( not self.isset ):
            raise ValueError(f"Cannot get validation set before dataset has been loaded")
        
        n_images = len(self.val_idxs)

        Xval = np.array([self._preprocess(imread(self.files_x[idx])) for idx in self.val_idxs])
        Yval = np.zeros(Xval.shape[:3]+(2,))
        Yval[:,:,:,1] = np.array([imread(self.files_y[idx])>0 for idx in self.val_idxs])
        Yval[:,:,:,0] = 1-Yval[:,:,:,1]
        return self._random_crop(Xval, Yval)