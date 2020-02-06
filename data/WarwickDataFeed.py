# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

Load dataset, perform pre-processing and data augmentation, produce mini-batch for the GlaS dataset.

Dataset downloaded from https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/

Citation:
K. Sirinukunwattana, J. P. W. Pluim, H. Chen, et al,  “Gland segmentation in colon histology images: The glas challenge contest,” 
Med. Image Anal., vol. 35, pp. 489–502, 2017.

The dataset should be structured as follows:

- <DATASET_DIR>
|---- <db>_<id>.bmp (RGB images, db in {'train', 'testA', 'testB'}, id = integer)
|---- <db>_<id>_<anno>.bmp (Annotation masks or labels, anno in {'anno', 'bb', ...)
|---- <db>_<id>_<anno>+5.bmp (Annotation masks or labels with dilation by disk(5) for Label Augmentation)
|---- <db>_<id>_<anno>-5.bmp (Annotation masks or labels with erosion by disk(5) for Label Augmentation)
'''

import numpy as np
import os
from .GenericDataFeed import GenericDataFeed
from skimage.io import imread
from dhutil.batch import batch_augmentation

class WarwickDataFeed(GenericDataFeed):

    def __init__(self, params, db, generator=None):
        super().__init__(params,db,generator)

        # Specific SNOW parameters
        self.onlyPositives = params['onlyGlands'] if 'onlyGlands' in params else self.onlyPositives # Compatibility with old config files.
        
        # Load all RGB images & annotation masks
        nPerDb = {'train': 85, 'testA': 60, 'testB': 20}
        self.files_X = [os.path.join(self.directory, "%s_%d.bmp"%(db, i+1)) for i in range(nPerDb[db])]    
        self.files_Y = [os.path.join(self.directory, "%s_%d_%s.bmp"%(db, i+1, self.annotations)) for i in range(nPerDb[db])]
        # Label augmentation
        if( self.tda ):
            self.files_Yplus = [os.path.join(self.directory, "%s_%d_%s+5.bmp"%(db, i+1, self.annotations)) for i in range(nPerDb[db])]
            self.files_Yminus = [os.path.join(self.directory, "%s_%d_%s-5.bmp"%(db, i+1, self.annotations)) for i in range(nPerDb[db])]
        
        # Prepare random sampling
        self.idxs = np.arange(len(self.files_Y))
        
        if(self.v): print("Loading data from %s"%self.directory)

        # Load images
        self.images_X = [imread(f) for f in self.files_X]
        self.images_Y = [imread(f) for f in self.files_Y]
        if( self.tda ):
            self.images_Yplus = [imread(f) for f in self.files_Yplus]
            self.images_Yminus = [imread(f) for f in self.files_Yminus]
            self.images_Yset = [self.images_Y, self.images_Yplus, self.images_Yminus]

        # Loading glands positions if necessary
        if( self.onlyPositives ):
            to_remove = []
            from skimage.measure import regionprops
            self.gland_zones = {}
            for idx in range(len(self.files_Y)):
                im = self.images_Y[idx]
                if( im.max() == 0 ): 
                    to_remove += [idx]
                    continue
                self.gland_zones[self.files_Y[idx]] = [obj.bbox for obj in regionprops(im)]
            self.idxs = np.delete(self.idxs, to_remove)    # Removing images with no glands

    '''
    Get a batch sample from the dataset.
    '''
    def get_sample(self, idx, batch_size, forValidation=False):
        batch_X = np.zeros((batch_size,self.tile_size, self.tile_size, 3))
        batch_Y_seg = np.zeros((batch_size,self.tile_size,self.tile_size,2))
        batch_Y_det = np.zeros((batch_size,2))

        # Load image & supervision
        # im = imread(self.files_X[idx])
        im = (self.images_X[idx]/255.)-0.5
        if(forValidation == False):
            if( self.tda == False ):
                supervision = self.images_Y[idx]
            else:
                r = int(np.random.random()*3)
                supervision = self.images_Yset[r][idx]
        else:
            supervision = self.images_Y[idx]
        
        mask = supervision>0

        if( self.onlyPositives == True ):
            rts = np.random.random((batch_size,2))    # Random translations within glands
            selected_glands = (np.random.random((batch_size,))*len(self.gland_zones[self.files_Y[idx]])).astype('int') # Select glands in image
            for i in range(batch_size):
                gland = selected_glands[i]
                bbox = self.gland_zones[self.files_Y[idx]][gland] # Gland bounding box
                # Compute available margins around the bounding box (we want the gland to stay mostly in the center, so the selected patch must have a large overlap with the bounding box)
                margins = (min(bbox[0],self.tile_size-50), min(bbox[1],self.tile_size-50), min(im.shape[0]-bbox[2],self.tile_size-50), min(im.shape[1]-bbox[3],self.tile_size-50))
                # Limits of the coordinates where we can select the patch
                limits = (min(im.shape[0]-self.tile_size,bbox[0]-margins[0]), min(im.shape[1]-self.tile_size,bbox[1]-margins[1]), max(0,bbox[2]+margins[2]-self.tile_size), max(0,bbox[3]+margins[3]-self.tile_size))
                # Top-left position of the selected patch
                rt = (limits[0]+(rts[i,0]*(limits[2]-limits[0])).astype('int'), limits[1]+(rts[i,1]*(limits[3]-limits[1])).astype('int'))

                batch_X[i,:,:,:] = im[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size,:]
                batch_Y_seg[i,:,:,0] = mask[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size]
                batch_Y_seg[i,:,:,1] = 1-batch_Y_seg[i,:,:,0]
                batch_Y_det[i,0] = 1
                batch_Y_det[i,1] = 0
        else:
            # Random translations within image to create batch (should be small mini-batches)
            tlimits = np.array((im.shape[0]-self.tile_size, im.shape[1]-self.tile_size))
            rts = (np.random.random((batch_size,2))*tlimits).astype('int')

            i = 0
            while i < batch_size:
                rt = rts[i]
                batch_X[i,:,:,:] = im[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size,:]
                batch_Y_seg[i,:,:,0] = mask[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size]
                
                # Add generative noise
                if( self.generative and batch_Y_seg[i,:,:,0].sum() < 80 ):
                    if( np.random.random() <= self.pNoise ):
                        batch_Y_seg[i,:,:,:] = self.generator([batch_X[i]])#(self.generator([batch_X[i]])>0.5).astype('float')
                    else:
                        batch_Y_seg[i,:,:,1] = 1-batch_Y_seg[i,:,:,0]
                else:
                    batch_Y_seg[i,:,:,1] = 1-batch_Y_seg[i,:,:,0]
        
                batch_Y_det[i,0] = min(batch_Y_seg[i,:,:,0].sum()/80., 1.)

                # Add uncertainty on negative examples
                if( self.noisy and batch_Y_det[i,0] == 0 ):
                    if( np.random.random() < self.pNoise ):
                        batch_Y_det[i,0] = 1
                        batch_Y_seg[i,:,:,0] = 1
                        batch_Y_seg[i,:,:,1] = 0

                batch_Y_det[i,1] = 1-batch_Y_det[i,0]
                i += 1

        if self.db != 'train' or forValidation == True:
            return batch_X,batch_Y_seg,batch_Y_det
        else:
            return batch_augmentation(batch_X,batch_Y_seg,batch_Y_det)