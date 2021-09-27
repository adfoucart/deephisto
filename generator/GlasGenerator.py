import os
import math

import numpy as np
from skimage.io import imread
from skimage.measure import regionprops,label
from skimage.transform import resize

from . import DataGenerator

class GlasGenerator(DataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



        if( self.image_strategy not in ['tile', 'full'] ):
            raise ValueError('image_strategy must be "tile" or "full"')

        nPerDb = {'train' : 85, 'testA' : 60, 'testB': 20}
        self.files_x = [os.path.join(self.directory, f"{self.folder_x}/{self.folder_x}_{i}.bmp") for i in range(1, nPerDb[self.folder_x]+1)]
        self.files_y = [os.path.join(self.directory, f"{self.folder_y}/{self.folder_x}_{i}_anno.bmp") for i in range(1, nPerDb[self.folder_x]+1)]
        
        if self.image_strategy == 'tile':
            self.images_x = [self._preprocess(imread(f)) for f in self.files_x]
            self.images_y = [imread(f) for f in self.files_y]
            self.full_images_x = self.images_x
            self.full_images_y = self.images_y
        else:
            self.full_images_x = [self._preprocess(imread(f)) for f in self.files_x]
            self.full_images_y = [imread(f) for f in self.files_y]
            self.images_x = [resize(im, self.tile_size) for im in self.full_images_x]
            self.images_y = [resize(im, self.tile_size, preserve_range=True, order=0, anti_aliasing=False) for im in self.full_images_y]

        self.idxs = np.arange(len(self.files_x))
        self.keep_idxs = self.idxs.copy()

        if( self.pPositive is not False ):
            self.positive_zones = []
            self.keep_idxs = []
            for idx in range(len(self.images_y)):
                im = self.images_y[idx].astype('int')
                if( im.max() == 1 ): im = label(im)
                
                zones = [obj.bbox for obj in regionprops(im)]
                self.positive_zones += [zones]
                
                if len(zones) > 0:
                    self.keep_idxs += [idx]
            
            self.keep_idxs = np.array(self.keep_idxs)

        self.isset = True

        self.split_train_val()
        keep_train = [idx for idx in self.train_idxs if idx in self.keep_idxs]

        if( self.image_strategy == 'tile' ):
            self.batches_per_epoch = len(keep_train)
        else:
            self.batches_per_epoch = len(keep_train)//self.batch_size
        
    def get_epoch(self, idxs, e):
        if( self.pPositive is not False ):
            # Note that if we are in the "full image" strategy, right now we can't handle the case where pPositive != 1 or False.
            idxs = [idx for idx in idxs if idx in self.keep_idxs]
        if( self.image_strategy == 'tile' ):
            for idr,idx in enumerate(idxs):
                yield self.get_sample_tiles(idx, batch_id=e*len(idxs)+idr)
        else:
            batch_y = np.zeros(self.batch_y_shape)
            for idb in range(self.batches_per_epoch):
                batch_x = np.array([self.images_x[idx] for idx in idxs[idb*self.batch_size:(idb+1)*self.batch_size]])
                batch_y[:,:,:,1] = np.array([self.images_y[idx]>0 for idx in idxs[idb*self.batch_size:(idb+1)*self.batch_size]])
                batch_y[:,:,:,0] = 1-batch_y[:,:,:,1]
                if self.gamodel is not False:
                    for i in range(batch_y.shape[0]):
                        if batch_y[i,:,:,1].mean() < self.maxPositiveAreaForGenerator \
                                and np.random.random() < self.pNoise :
                            batch_y[i] = self.gamodel.predict(np.array([batch_x[i]]))
                yield self._augment(batch_x, batch_y, idb)

    def get_sample_tiles(self, idx, batch_id=None, augment=True):
        """Generate a batch from the image self.images_x[idx].
        batch_id may be used to set seeds for data augmtentation."""
        if( self.verbose ):
            print(f"Getting sample from file: {self.files_x[idx]}")

        batch_x = np.zeros(self.batch_x_shape)
        batch_y = np.zeros(self.batch_y_shape)

        im = self.images_x[idx]
        anno = self.images_y[idx]
        mask = anno>0
        n_positives = 0

        # Random translations within image (or bbox for positive regions)
        rts = np.random.random((self.batch_size,2))

        if( self.pPositive is not False ):
            n_positives = int(self.batch_size*self.pPositive)
            # Select positive regions:
            selected_regions = np.random.randint(0, len(self.positive_zones[idx]), size=n_positives)
            for i in range(n_positives):
                bbox = self.positive_zones[idx][selected_regions[i]]
                # we want at least 25% overlap between tile & bbox
                # -> top-left must be between [bbox[0]-tilesize[0]/2,bbox[1]-tilesize[1]/2] 
                #                       and [bbox[2]-tilesize[0]/2,bbox[3]-tilesize[1]/2] 
                tl = (np.clip((bbox[0]-self.batch_x_shape[1]/2)+rts[i,0]*(bbox[2]-bbox[0]),0,im.shape[0]-self.batch_x_shape[1]).astype('int'), 
                      np.clip((bbox[1]-self.batch_x_shape[2]/2)+rts[i,1]*(bbox[3]-bbox[1]),0,im.shape[1]-self.batch_x_shape[2]).astype('int'))
                batch_x[i] = im[tl[0]:tl[0]+self.batch_x_shape[1], tl[1]:tl[1]+self.batch_x_shape[2],:]
                batch_y[i,:,:,1] = mask[tl[0]:tl[0]+self.batch_x_shape[1], tl[1]:tl[1]+self.batch_x_shape[2]]


        for i in range(n_positives,self.batch_size):
            tl = (int(rts[i,0]*(im.shape[0]-self.batch_x_shape[1])),int(rts[i,1]*(im.shape[1]-self.batch_x_shape[2])))
            batch_x[i] = im[tl[0]:tl[0]+self.batch_x_shape[1],tl[1]:tl[1]+self.batch_x_shape[2],:]
            batch_y[i,:,:,1] = mask[tl[0]:tl[0]+self.batch_x_shape[1],tl[1]:tl[1]+self.batch_x_shape[2]]

            # Use GA if necesseray
            if self.gamodel is not False \
                    and augment \
                    and batch_y[i,:,:,1].mean() < self.maxPositiveAreaForGenerator \
                    and np.random.random() < self.pNoise :
                batch_y[i] = self.gamodel.predict(np.array([batch_x[i]]))

        batch_y[:,:,:,0] = 1-batch_y[:,:,:,1]

        if augment:
            return self._augment(batch_x,batch_y,batch_id) 

        return batch_x,batch_y

    def get_validation_set(self):
        if( not self.isset ):
            raise ValueError(f"Cannot get validation set before dataset has been loaded")
        
        n_images = len(self.val_idxs)

        if( self.image_strategy == 'tile' ):
            Xval = np.zeros((n_images*self.batch_size, self.batch_x_shape[1], self.batch_x_shape[2], 3))
            Yval = np.zeros((n_images*self.batch_size, self.batch_x_shape[1], self.batch_x_shape[2], 2))

            for i,idx in enumerate(self.val_idxs):
                Xval[i*self.batch_size:(i+1)*self.batch_size],Yval[i*self.batch_size:(i+1)*self.batch_size] = self.get_sample_tiles(idx, augment=False)

            return Xval, Yval
        else:
            Xval = np.array([self.images_x[idx] for idx in self.val_idxs])
            Yval = np.zeros(Xval.shape[:3]+(2,))
            Yval[:,:,:,1] = np.array([self.images_y[idx]>0 for idx in self.val_idxs])
            Yval[:,:,:,0] = 1-Yval[:,:,:,1]
            return Xval, Yval

    def _get_tiles_generator(self, imshape):
        if self.image_strategy == 'full':
            raise ValueError("Tiling cannot be used with the 'full' strategy.")
        
        ny = math.ceil(imshape[0]/self.tile_size[0])
        nx = math.ceil(imshape[1]/self.tile_size[1])
        step_y = (imshape[0]-self.tile_size[0])/(ny-1)
        step_x = (imshape[1]-self.tile_size[1])/(nx-1)
        coords_y = np.arange(0, imshape[0]-self.tile_size[0]+1, step_y).astype('int')
        coords_x = np.arange(0, imshape[1]-self.tile_size[1]+1, step_x).astype('int')
        mesh = np.meshgrid(coords_x,coords_y)
        return zip(mesh[0].flatten(), mesh[1].flatten())

    def tile(self, idx):
        if self.image_strategy == 'full':
            raise ValueError("Tiling cannot be used with the 'full' strategy.")

        im = self.images_x[idx]
        anno = self.images_y[idx]>0
        
        tiles_x = []
        tiles_y = []

        for tx,ty in self._get_tiles_generator(anno.shape):
            tiles_x += [im[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1]]]
            tiles_y += [anno[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1]]]

        return tiles_x,tiles_y

    def stitch(self, tiles_prediction, idx):
        if self.image_strategy == 'full':
            raise ValueError("Tiling cannot be used with the 'full' strategy.")
        
        im = self.images_x[idx]
        anno = self.images_y[idx]>0
        imshape = anno.shape

        pred_image = np.zeros(imshape+(2,)).astype('float')
        n_preds = np.zeros(imshape).astype('float') # to keep track of how many predictions were made on a given pixel

        for i,tile in enumerate(self._get_tiles_generator(imshape)):
            pred = tiles_prediction[i]

            tx = tile[0]
            ty = tile[1]
            
            pred_image[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1],0] += pred[:,:,0]
            n_preds[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1]] += 1
                
        pred_image[:,:,0] /= n_preds
        pred_image[:,:,1] = 1-pred_image[:,:,0]

        return pred_image