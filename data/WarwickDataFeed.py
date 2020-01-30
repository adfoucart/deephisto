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
from skimage.io import imread
from dhutil.batch import batch_augmentation

class WarwickDataFeed:

    def __init__(self, params, db, generator=None):
        # General parameters
        self.directory = params['dataset_dir']
        self.tile_size = params['tile_size']
        self.db = db
        self.v = params['verbose'] if 'verbose' in params else False

        # SNOW parameters
        self.annotations = params['annotations'] if 'annotations' in params else 'anno'
        self.noisy = params['noisy'] if 'noisy' in params else False
        self.pNoise = params['pNoise'] if 'pNoise' in params else 0.5
        self.generative = params['generative'] if 'generative' in params else False
        self.generator = generator
        self.onlyGlands = params['onlyGlands'] if 'onlyGlands' in params else False
        self.tda = params['tda'] if 'tda' in params else False
        
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
        self.pointer = 0
        self.seed = params['random_seed'] if 'random_seed' in params else 56489143
        np.random.seed(self.seed)

        if(self.v): print("Loading data from %s"%self.directory)

        # Load images
        self.images_X = [imread(f) for f in self.files_X]
        self.images_Y = [imread(f) for f in self.files_Y]
        if( self.tda ):
            self.images_Yplus = [imread(f) for f in self.files_Yplus]
            self.images_Yminus = [imread(f) for f in self.files_Yminus]
            self.images_Yset = [self.images_Y, self.images_Yplus, self.images_Yminus]

        # Loading glands positions if necessary
        if( self.onlyGlands ):
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

        if( self.onlyGlands == True ):
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

    '''
    Batch generator
    Yield samples from dataset
    '''
    def next_batch(self, batch_size, max_iterations, forValidation=False):
        if(self.v): print("Starting %d iterations"%(max_iterations))
        # iterations = sampling each image at least once
        for it in range(max_iterations):
            # Shuffle ids so we don't always go through the images in the same order
            np.random.shuffle(self.idxs)
            # print("Iteration: %d"%it)

            # For each slide : draw a random sample. Go through each image once before starting again with new seed
            for idx in self.idxs:
                yield self.get_sample(idx, batch_size,forValidation)

    '''
    Draw a validation set from the training set, using no data augmentation.
    '''
    def validation_set(self, batch_size):
        images_used = 10
        n_per_image = batch_size//images_used # 10 images in validation set
        Xval = np.zeros((batch_size, self.tile_size, self.tile_size, 3))
        Yval_seg = np.zeros((batch_size, self.tile_size, self.tile_size, 2))
        Yval_det = np.zeros((batch_size, 2))
        for i in range(images_used):
            X,Y_seg,Y_det = self.get_sample(self.idxs[i], n_per_image, True)
            Xval[int(i*n_per_image):int((i+1)*n_per_image),:,:,:] = X.copy()
            Yval_seg[int(i*n_per_image):int((i+1)*n_per_image),:,:,:] = Y_seg.copy()
            Yval_det[int(i*n_per_image):int((i+1)*n_per_image)] = Y_det.copy()
        return Xval,Yval_seg,Yval_det