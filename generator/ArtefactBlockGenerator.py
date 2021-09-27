import os
import math
import time
import gc
import h5py
import threading

import numpy as np
from skimage.io import imread
from skimage.measure import regionprops,label
from skimage.transform import resize

from . import DataGenerator

class BlockLoader():
    def __init__(self, directory, cur_idx, next_idx):
        self.t0 = time.time()
        self.directory = directory
        self.current_block = (None, None) # block_x, block_y
        self.next_block = (None, None) # block_x, block_y 
        self.status = 0
        self.cur_block_idx = cur_idx
        self.next_block_idx = next_idx

    def load_cur(self):
        # print(f"{time.time()-self.t0:.3f} : load_cur")
        with h5py.File(os.path.join(self.directory,f"artefact_tiles_block_{int(self.cur_block_idx)}_annos.h5"), 'r') as hf:
            annos = hf["annos"][:]
        with h5py.File(os.path.join(self.directory,f"artefact_tiles_block_{int(self.cur_block_idx)}_images.h5"), 'r') as hf:
            images = hf["images"][:]
        self.current_block = (images,annos)

    def load_next(self):
        # print(f"{time.time()-self.t0:.3f} : load_next")
        if( self.next_block_idx == -1 ): return

        with h5py.File(os.path.join(self.directory,f"artefact_tiles_block_{int(self.next_block_idx)}_annos.h5"), 'r') as hf:
            annos = hf["annos"][:]
        with h5py.File(os.path.join(self.directory,f"artefact_tiles_block_{int(self.next_block_idx)}_images.h5"), 'r') as hf:
            images = hf["images"][:]
        self.next_block = (images,annos)

    def swap(self):
        # print(f"{time.time()-self.t0:.3f} : swap")
        self.current_block = self.next_block
        del self.next_block

    def get(self, cur_idx, next_idx):
        # print(f"{time.time()-self.t0:.3f} : get({cur_idx},{next_idx}) [status={self.status}]")
        if( self.status == 0 ):
            self.cur_block_idx = cur_idx
            self.next_block_idx = next_idx
            return False,(None,None)
        elif( self.status == 1 ):
            self.status = 2
            return True,self.current_block
        elif( self.status == 2 ):
            return False,(None,None)
        elif( self.status == 3 ):
            self.swap()
            self.next_block_idx = next_idx
            self.status = 4
            return True,self.current_block
        elif( self.status == 4 ):
            return False,(None,None)

    def run(self):
        while True:
            if self.status == 0:
                self.load_cur()
                self.status = 1
            elif self.status == 1:
                time.sleep(1) # wait for generator to get current
            elif self.status == 2 or self.status == 4:
                self.load_next()
                self.status = 3
            elif self.status == 3:
                time.sleep(1) # wait for generator to get current


class ArtefactBlockGenerator(DataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if( self.timeit ):  self.log_time += [{'event': 'init artefact generator', 'time': time.time()}]

        training_blocks = [f'artefact_tiles_block_{block}_images.h5' for block in range(5)]

        n_images = 0
        positive_per_block = []
        self.block_idxs = np.arange(5)
        self.idxs_in_block = []
        self.pos_idxs_in_block = []
        for block in range(5):
            with h5py.File(os.path.join(self.directory,f"artefact_tiles_block_{block}_annos.h5"), 'r') as hf:
                annos = hf["annos"][:]
                n_images += annos.shape[0]
                positives = annos.max(axis=1).max(axis=1)>0
                positive_per_block.append(positives)
                idxs = np.arange(annos.shape[0])
                self.idxs_in_block.append(idxs)
                self.pos_idxs_in_block.append(np.array([idx for idx in idxs if positives[idx]]))
        n_positive_images = sum([p.sum() for p in positive_per_block])

        self.isset = True

        self.split_train_val()

        if( self.pPositive == 1. ):
            self.batches_per_epoch = sum([p.sum()//self.batch_size for p in positive_per_block])
            self.idxs_in_block = self.pos_idxs_in_block
        else:
            self.batches_per_epoch = sum([p.shape[0]//self.batch_size for p in self.idxs_in_block])

        if( self.timeit ): self.log_time += [{'event': 'end init', 'time': time.time()}]

    def split_train_val(self):
        '''Doesn't care about the params, uses the predetermined split from the blocks'''
        with h5py.File(os.path.join(self.directory,f"artefact_tiles_validation_annos.h5"), 'r') as hf:
            self.validation_annos = hf["annos"][:20]
        with h5py.File(os.path.join(self.directory,f"artefact_tiles_validation_images.h5"), 'r') as hf:
            self.validation_images = hf["images"][:20]

    def next_batch(self):
        if( not self.isset ):
            raise ValueError(f"Cannot generate batch before dataset has been loaded.")

        if( self.timeit ): self.log_time += [{'event': 'start next batch', 'time': time.time()}]

        """Yield batch_x, batch_y"""
        if( self.verbose ):
            print(f'Starting {self.n_epochs} epochs.')

        '''Generate sequence'''
        np.random.seed(self.random_seed_sequence)
        self.block_sequence = np.zeros((self.n_epochs, 5)).astype('int')
        self.tile_sequence = [[[] for i in range(5)] for i in range(self.n_epochs)]
        for e in range(self.n_epochs):
            np.random.shuffle(self.block_idxs)
            self.block_sequence[e,:] = self.block_idxs[:]
            for bidx in self.block_idxs:
                tiles_idxs = self.idxs_in_block[bidx]
                np.random.shuffle(tiles_idxs)
                self.tile_sequence[e][bidx] = tiles_idxs[:]

        # self.block_sequence_flat = list(self.block_sequence.flatten()) + [-1,]

        # self.block_loader = BlockLoader(self.directory, self.block_sequence_flat[0], self.block_sequence_flat[1])
        # self.loader_thread = threading.Thread(target=self.block_loader.run)
        # self.loader_thread.start()
        for e in range(self.n_epochs):
            if( self.timeit ):  self.log_time += [{'event': 'start epoch', 'time': time.time()}]
            for epoch in self.get_epoch(e):
                yield epoch

    # def load_block(self, cur_idx, next_idx):
    #     ready = False
    #     while not ready:
    #         ready,(block_x,block_y) = self.block_loader.get(cur_idx, next_idx)
    #         time.sleep(1)
    #     return block_x,block_y

    def load_block(self, block):
        with h5py.File(os.path.join(self.directory,f"artefact_tiles_block_{int(block)}_annos.h5"), 'r') as hf:
            annos = hf["annos"][:]
        with h5py.File(os.path.join(self.directory,f"artefact_tiles_block_{int(block)}_images.h5"), 'r') as hf:
            images = hf["images"][:]
        return images,annos

    def get_epoch(self, e):
        if( self.timeit ): self.log_time += [{'event': 'in get_epoch', 'time': time.time()}]
        
        for block_idx in self.block_sequence[e]:
        # for i in range(e*5, (e+1)*5):
            if( self.timeit ): self.log_time += [{'event': 'in block loop', 'time': time.time()}]
            # load block
            block_x,block_y = self.load_block(block_idx)
            # block_idx = self.block_sequence_flat[i]
            # next_idx = self.block_sequence_flat[i+1]
            # block_x,block_y = self.load_block(block_idx,next_idx)
            tiles_idxs = self.tile_sequence[e][block_idx]
            if( self.timeit ): self.log_time += [{'event': 'block loaded', 'time': time.time()}]
            for idb in range(len(tiles_idxs)//self.batch_size):
                batch_x = self._preprocess(np.array([block_x[idx] for idx in tiles_idxs[idb*self.batch_size:(idb+1)*self.batch_size]]))
                batch_y_1 = np.array([block_y[idx] for idx in tiles_idxs[idb*self.batch_size:(idb+1)*self.batch_size]])
                batch_y = np.zeros(list(batch_y_1.shape) + [2,])
                batch_y[:,:,:,1] = batch_y_1
                batch_y[:,:,:,0] = 1-batch_y[:,:,:,1]
                if self.gamodel is not False:
                    for i in range(batch_y.shape[0]):
                        if batch_y[i,:,:,1].mean() < self.maxPositiveAreaForGenerator \
                                and np.random.random() < self.pNoise :
                            batch_y[i] = self.gamodel.predict(np.array([batch_x[i]]))
                yield self._augment(*self._random_crop(batch_x, batch_y), idb)
            del block_x # making sure we free RAM for next round
            del block_y
            gc.collect()

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
        
        Xval = self._preprocess(self.validation_images)
        Yval = np.zeros(Xval.shape[:3]+(2,))
        Yval[:,:,:,1] = self.validation_annos
        Yval[:,:,:,0] = 1-Yval[:,:,:,1]
        return self._random_crop(Xval, Yval)