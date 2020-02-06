# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

Load dataset, perform pre-processing and data augmentation, produce mini-batch for the Artefact dataset.

The dataset should be structured as follows:

- <DATASET_DIR>
|---- <db> (train, test, validation...)
        |---- *_rgb.png (RGB images)
        |---- *_bg.png (optional, may be generated, Background masks)
        |---- *_mask.png (Annotation masks or labels)
|---- <db>
        |---- ...
'''

import numpy as np
import os
from .GenericDataFeed import GenericDataFeed
from skimage.io import imread
from dhutil.artefact import getBackgroundMask 
from dhutil.batch import batch_augmentation

'''
params:
* dataset_dir (required) -> base directory of the dataset (images should be in "db" subdirectories)
* tile_size (int) (required)
* removeBackground (bool) -> remove non-tissue pixels from the slide
* justOne (bool) -> load only one slide for quick testing
* verbose (bool)
* noisy (bool) -> Use "Noisy" annotations
* pNoise (float) -> Probability of switching negative into positive examples (if noisy is true)
* generative (bool) -> Use "Generated Annotations"
* generator (method) -> predictor method of a generator DCNN
* onlyPositives (bool) -> Use "Only positive" strategy 
'''
class ArtefactDataFeed(GenericDataFeed):

    def __init__(self, params, db, generator=None):
        super().__init__(params,db,generator)

        # Additional General parameters
        self.removeBackground = params['removeBackground'] if 'removeBackground' in params else True
        self.justOne = params['justOne'] if 'justOne' in params else False
        
        # Specific code
        self.dir = os.path.join(self.directory, db)
        
        # Find all files from data directory and find the RGB image and the background and supervision masks.
        files = [f for f in os.listdir(self.dir)]
        
        self.files_X = [os.path.join(self.dir,f) for f in files if f.find('_rgb') >= 0]
        self.files_B = [os.path.join(self.dir,f) for f in files if f.find('_bg') >= 0]
        self.files_Y = [os.path.join(self.dir,f) for f in files if f.find('_mask') >= 0]

        if( self.justOne ): # Get just the first file for a quick test
            self.files_X = [self.files_X[0]]
            self.files_B = [self.files_B[0]]
            self.files_Y = [self.files_Y[0]]

        # NOTE : Label Augmentation not implemented

        # Prepare random sampling
        self.idxs = np.arange(len(self.files_Y))

        if( self.v ): print("Loading data from %s (%s)"%(self.directory, self.db))

        # Load images and generate background masks if necesseray
        self.images_X = [imread(f) for f in self.files_X]
        if( len(self.files_B) == 0 and self.removeBackground ):
            if( self.v ): print("Generating bg masks")
            self.images_B = []
            from skimage.io import imsave
            for idx,im in enumerate(self.images_X):
                bg_mask = getBackgroundMask(im)
                imsave(self.files_X[idx].replace('_rgb', '_bg'), bg_mask)
                self.images_B += [bg_mask]
        elif self.removeBackground:
            self.images_B = [imread(f)==0 for f in self.files_B]
        self.images_Y = [imread(f)>0 for f in self.files_Y]

        if( self.v ): print("Loaded.")

        # Loading nuclei positions if necessary
        if( self.onlyPositives ):
            to_remove = []
            from skimage.measure import label,regionprops
            self.positive_zones = {}
            for idx in range(len(self.files_Y)):
                im = self.images_Y[idx]
                if( im.max() == 0 ): 
                    to_remove += [idx]
                    continue
                self.positive_zones[self.files_Y[idx]] = [obj.bbox for obj in regionprops(label(im))]
            self.idxs = np.delete(self.idxs, to_remove)	# Removing images with no nuclei

    '''
    Get a batch sample from the dataset.
    '''
    def get_sample(self, idx, batch_size, forValidation=False):
        batch_X = np.zeros((batch_size,self.tile_size, self.tile_size, 3))
        batch_Y_seg = np.zeros((batch_size,self.tile_size,self.tile_size,2))
        batch_Y_det = np.zeros((batch_size,2))

        # Load image & supervision
        im = (self.images_X[idx]/255.)-0.5
        supervision = self.images_Y[idx]
        if self.removeBackground:
            bg = self.images_B[idx]
        else:
            bg = np.ones(im.shape[:2])

        nrows = im.shape[0]
        ncols = im.shape[1]

        # Using BG mask:
        # * Mask BG + Tile limits (+ positive regions if set)
        # * Take all points in mask.
        # * Draw randomly from these points Center of tiles.
        draw_mask = bg.copy()
        tile_mask = np.zeros_like(draw_mask)
        tile_mask[self.tile_size//2:-self.tile_size//2, self.tile_size//2:-self.tile_size//2] = 1
        draw_mask *= tile_mask
        if( self.onlyPositives ):
            draw_mask *= supervision
        draw_mask = draw_mask.flatten().astype('bool')
        n_candidates = draw_mask.sum()
        
        px_idx = np.arange(len(draw_mask))
        candidate_idx = np.zeros_like(draw_mask).astype('int32')
        candidate_idx[draw_mask] = np.arange(1,n_candidates+1)

        selected = 1+(np.random.random((batch_size,))*n_candidates).astype('int32')
        
        for i in range(batch_size):
            selected_idx = px_idx[candidate_idx==selected[i]][0]
            x = int(selected_idx%ncols-self.tile_size//2)
            y = int(selected_idx//ncols-self.tile_size//2)
            batch_X[i,:,:,:] = im[y:y+self.tile_size, x:x+self.tile_size,:]
            batch_Y_seg[i,:,:,0] = supervision[y:y+self.tile_size, x:x+self.tile_size]
            
            # Add noisy strategies if needed
            if( self.generative and batch_Y_seg[i,:,:,0].sum() < 80 ): # Add generative noise
                if( np.random.random() <= self.pNoise ):
                    batch_Y_seg[i,:,:,:] = self.generator([batch_X[i]])#(self.generator([batch_X[i]])>0.5).astype('float')
                else:
                    batch_Y_seg[i,:,:,1] = 1-batch_Y_seg[i,:,:,0]
            elif( self.noisy ): # Add patch-based noise
                if( np.random.random() <= self.pNoise ):
                    batch_Y_seg[i,:,:,0] = 1. # Switch every pixel to "artefact"
                    batch_Y_seg[i,:,:,1] = 0.
            else:
                batch_Y_seg[i,:,:,1] = 1-batch_Y_seg[i,:,:,0]

        # Compute patch-level label
        batch_Y_det[:,0] = batch_Y_seg[:,:,:,0].sum(axis=1).sum(axis=1)/80.
        batch_Y_det[batch_Y_det[:,0]>1,0] = 1
        batch_Y_det[:,1] = 1-batch_Y_det[:,0]

        if self.db != 'train' or forValidation == True:
            return batch_X,batch_Y_seg,batch_Y_det
        else:
            return batch_augmentation(batch_X,batch_Y_seg,batch_Y_det)