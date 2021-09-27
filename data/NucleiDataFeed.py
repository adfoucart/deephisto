# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

Load dataset, perform pre-processing and data augmentation, produce mini-batch for the Nuclei dataset.

Dataset downloaded from : http://www.andrewjanowczyk.com/deep-learning/

Citation:
A. Janowczyk and A. Madabhushi, “Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases,” 
J. Pathol. Inform., vol. 7, no. 29, 2016.

The dataset should be structured as follows:

- <DATASET_DIR>
|---- <db> (train, test, val...)
        |---- *.tif (RGB images)
|---- anno
        |---- *.png (Annotation mask or labels corresponding to the .tif images)
'''

import numpy as np
import os
from data.GenericDataFeed import GenericDataFeed
from skimage.io import imread
from dhutil.batch import batch_augmentation

'''
params:
* dataset_dir -> base directory of the dataset (images should be in "db" subdirectories)
* tile_size (int) (required)
* verbose (bool)
* noisy (bool) -> Use "Noisy" annotations
* pNoise (float) -> Probability of switching negative into positive examples (if noisy is true)
* generative (bool) -> Use "Generated Annotations"
* generator (method) -> predictor method of a generator DCNN
* balance (float) -> % of positive examples required in each minibatch
'''
class NucleiDataFeed(GenericDataFeed):

    def __init__(self, params, db, generator=None):
        super().__init__(params,db,generator)

        # Additional General parameters
        self.Xdir = os.path.join(self.directory, db)
        self.balance = params['balance'] if 'balance' in params else 0.
        
        # Find all RGB images & annotation masks
        self.Ydir = os.path.join(self.directory, "anno")
        files = [f for f in os.listdir(self.Xdir)]
        self.files_X = [os.path.join(self.Xdir,f) for f in files]
        self.files_Y = [os.path.join(self.Ydir,f.replace('_original.tif', '_mask.png')) for f in files]
        
        # Prepare random sampling
        self.idxs = np.arange(len(self.files_Y))
        
        if( self.v ): print("Loading data from %s"%self.directory)

        # Load images
        self.images_X = [imread(f) for f in self.files_X]
        self.images_Y = [imread(f).astype('uint16') for f in self.files_Y]
        
        self.minPxInPosSample = params['minPxInPosSample'] if 'minPxInPosSample' in params else 0.01*(self.tile_size**2)
        
        if( self.v ): print("Loaded.")

        # Loading annotation positions if necessary
        if( self.balance > 0 or self.onlyPositives == True ):
            to_remove = []
            from skimage.measure import label,regionprops
            self.positive_zones = {}
            for idx in range(len(self.files_Y)):
                im = self.images_Y[idx]
                if( im.max() == 0 ): 
                    to_remove += [idx]
                    continue
                self.positive_zones[self.files_Y[idx]] = [obj.bbox for obj in regionprops(label(im))]
            self.idxs = np.delete(self.idxs, to_remove) # Removing images with no objects
    
    def get_positive_samples(self, im, mask, idx, n_pos):
        batch_X = np.zeros((n_pos,self.tile_size, self.tile_size, 3))
        batch_Y_seg = np.zeros((n_pos,self.tile_size,self.tile_size,2))
        batch_Y_det = np.zeros((n_pos,2))
        
        rts = np.random.random((n_pos,2))   # Random translations within object
        selected = (np.random.random((n_pos,))*len(self.positive_zones[self.files_Y[idx]])).astype('int') # Select object in image
        for i in range(n_pos):
            positive = selected[i]
            bbox = self.positive_zones[self.files_Y[idx]][positive] # Positive bounding box
            # Compute available margins around the bounding box (we want the object to stay mostly in the center, so the selected patch must have a large overlap with the bounding box)
            MARGIN = 20
            margins = (min(bbox[0],self.tile_size-MARGIN), min(bbox[1],self.tile_size-MARGIN), min(im.shape[0]-bbox[2],self.tile_size-MARGIN), min(im.shape[1]-bbox[3],self.tile_size-MARGIN))
            # Limits of the coordinates where we can select the patch
            limits = (min(im.shape[0]-self.tile_size,bbox[0]-margins[0]), min(im.shape[1]-self.tile_size,bbox[1]-margins[1]), max(0,bbox[2]+margins[2]-self.tile_size), max(0,bbox[3]+margins[3]-self.tile_size))
            # Top-left position of the selected patch
            rt = (limits[0]+(rts[i,0]*(limits[2]-limits[0])).astype('int'), limits[1]+(rts[i,1]*(limits[3]-limits[1])).astype('int'))

            batch_X[i,:,:,:] = im[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size,:]
            batch_Y_seg[i,:,:,0] = mask[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size]
            batch_Y_seg[i,:,:,1] = 1-batch_Y_seg[i,:,:,0]
            batch_Y_det[i,0] = 1
            batch_Y_det[i,1] = 0
        
        return batch_X, batch_Y_seg, batch_Y_det
    
    def get_random_samples(self, im, mask, n):
        batch_X = np.zeros((n,self.tile_size, self.tile_size, 3))
        batch_Y_seg = np.zeros((n,self.tile_size,self.tile_size,2))
        batch_Y_det = np.zeros((n,2))
        
        # Random translations within image to create batch (should be small mini-batches)
        tlimits = np.array((im.shape[0]-self.tile_size, im.shape[1]-self.tile_size))
        rts = (np.random.random((n,2))*tlimits).astype('int')

        for i in range(n):
            rt = rts[i]
            batch_X[i,:,:,:] = im[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size,:]
            batch_Y_seg[i,:,:,0] = mask[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size]
        
        return batch_X, batch_Y_seg, batch_Y_det
    
    def add_noise(self, batch_X, batch_Y_seg, batch_Y_det, batch_size):
        # Add generative noise
        if( self.generative):
            toGenerate = (batch_Y_seg[:,:,:,0].sum(axis=1).sum(axis=1) < self.minPxInPosSample)*(np.random.random((batch_size,))<self.pNoise)
            
            for i in range(batch_size):
                if(toGenerate[i]):
                    batch_Y_seg[i,:,:,:] = self.generator([batch_X[i]])

        batch_Y_det[:,0] = np.clip(batch_Y_seg[:,:,:,0].sum(axis=1).sum(axis=1)/self.minPxInPosSample,0,1)

        # Add uncertainty on negative examples
        if( self.noisy ):
            toSwitch = (batch_Y_det[:,0]==0)*(np.random.random((batch_size,)) < self.pNoise)
            batch_Y_det[toSwitch,0] = 1
            batch_Y_seg[toSwitch,:,:,0] = 1
        
        return batch_Y_seg, batch_Y_det
    
    '''
    Get a batch sample from the dataset.
    '''
    def get_sample(self, idx, batch_size, forValidation=False, random_seed=None):
        # Load image & supervision
        im = (self.images_X[idx]/255.)-0.5
        supervision = self.images_Y[idx]
        if(forValidation == False):
            if( self.tda == False ):
                supervision = self.images_Y[idx]
            else:
                r = int(np.random.random()*3)
                supervision = self.images_Yset[r][idx]
        else:
            supervision = self.images_Y[idx]
        
        mask = supervision>0

        if( self.balance > 0 ):
            batch_X = np.zeros((batch_size,self.tile_size, self.tile_size, 3))
            batch_Y_seg = np.zeros((batch_size,self.tile_size,self.tile_size,2))
            batch_Y_det = np.zeros((batch_size,2))
            n_pos = int(self.balance*batch_size)
            Xpos,Yspos,Ydpos = self.get_positive_samples(im, mask, idx, n_pos)
            batch_X[:n_pos] = Xpos
            batch_Y_seg[:n_pos] = Yspos
            batch_Y_det[:n_pos] = Ydpos
            Xrand,Ysrand,Ydrand = self.get_random_samples(im, mask, batch_size-n_pos)
            batch_X[n_pos:] = Xrand
            batch_Y_seg[n_pos:] = Ysrand
            batch_Y_det[n_pos:] = Ydrand
        else:
            batch_X,batch_Y_seg,batch_Y_det = self.get_random_samples(im, mask, batch_size)
        
        batch_Y_seg, batch_Y_det = self.add_noise(batch_X, batch_Y_seg, batch_Y_det, batch_size)
        
        batch_Y_seg[:,:,:,1] = 1-batch_Y_seg[:,:,:,0]
        batch_Y_det[:,1] = 1-batch_Y_det[:,0]
        
        if self.db != 'train' or forValidation == True or self.augmentData == False:
            return batch_X,batch_Y_seg,batch_Y_det
        else:
            return batch_augmentation(batch_X,batch_Y_seg,batch_Y_det,random_seed)