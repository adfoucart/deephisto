import os
import json

import numpy as np
import tensorflow as tf
import time

class DataGenerator:
    """Abstract data generator class.

    Manages the random sequence & the next_batch/get_validation.
    Subclasses must be defined for each dataset to incorporate the 
    specific file structure."""

    def __init__(self, **kwargs):
        default_params = {
            'directory' : None,         # path to dataset
            'tile_size' : None,         # tuple (width, height) or int for square regions
            'batch_size' : None,        
            'folder_x': None,           # subfolder with the images to use
            'folder_y': None,           # subfolder with the annotations to use.
            'image_strategy': 'full',   # 'full' or 'tile'
            'name' : "data_generator",
            'pValidation': 0.1,         # proportion of training set to put aside for validation
            'train_val': False,         # False or path to .json (if set, ignore pValidation)
            'n_epochs' : 100,            
            'verbose' : False,            
            'random_sequence' : False,   # False or path to .npy
            'random_seed' : 0,                    
            'pNoise' : False,           # False or float [0,1]
            'gamodel': False,         # False or trained Model or path to model
            'maxPositiveAreaForGenerator': 0.05, # Use generator if positiveArea < maxPositiveAreaForGenerator (% of tile)
            'pPositive': False,         # False or float [0,1]. 1 = onlyPositive
            'augment_illumination': 0.05, # Scale for illumination change
            'augment_noise': 0.05,       # Scale for normal noise
            'mag': 'both',              # only used for artefact -> should refactor this better
            'timeit': False, 
            'random_seed_sequence': 1   # used for ArtefactBlockGenerator
        }

        for key,default in default_params.items():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            elif default is not None:
                setattr(self, key, default)
            else:
                raise KeyError(f'Parameter "{key}" must be set.')

        if( self.timeit ): self.log_time = [{'event': 'init generator', 'time': time.time()}] 

        if( isinstance(self.tile_size, int) ):
            self.batch_x_shape = (self.batch_size, self.tile_size, self.tile_size, 3)
            self.batch_y_shape = (self.batch_size, self.tile_size, self.tile_size,2)
        elif( isinstance(self.tile_size, tuple) and len(self.tile_size) == 2 ):
            self.batch_x_shape = (self.batch_size, self.tile_size[0], self.tile_size[1], 3)
            self.batch_y_shape = (self.batch_size, self.tile_size[0], self.tile_size[1],2)
        else:
            raise ValueError(f"Tile size must be int or tuple (height,width). Was: {self.tile_size}")

        if( isinstance(self.gamodel, str) ):
            self.gamodel = tf.keras.models.load_model(self.gamodel, compile=False, custom_objects={'leaky_relu': tf.nn.leaky_relu})
        # must be set to true by the subclass after the data is loaded into
        # self.images_x & self.images_y
        self.isset = False 

    def split_train_val(self):
        """Set pValidation% of training set aside for validation 
        (or load train/val split from json file)""" 
        if self.pValidation <= 0.:
            if(self.verbose):
                print("No validation set. Training on full set.")
            self.train_idxs = self.keep_idxs
            self.val_idxs = []
            return

        if( not self.isset ):
            raise ValueError(f"Cannot perform train/val split before dataset has been loaded")
        if( self.train_val is not False ):
            with open(self.train_val, 'r') as fp:
                train_val_idxs = json.load(fp)
                self.train_idxs = np.array(train_val_idxs['train_idxs'])
                self.val_idxs = np.array(train_val_idxs['val_idxs'])
                return True
            raise ValueError(f"Couldn't load train/val split from {self.train_val}")
        else:
            np.random.shuffle(self.idxs)
            if( isinstance(self.pValidation, int) ):
                n_val = self.pValidation
            else:
                n_val = int(len(self.idxs)*self.pValidation)
            self.val_idxs = self.idxs[:n_val]
            self.train_idxs = self.idxs[n_val:]
            with open(f'{self.name}_train_val.json', 'w') as fp:
                json.dump({'train_idxs': self.train_idxs.tolist(), 'val_idxs': self.val_idxs.tolist()}, fp)
            if( self.verbose ):
                print(f"Generated train/val split, saved to {self.name}_train_val.json")

    def next_batch(self):
        if( not self.isset ):
            raise ValueError(f"Cannot generate batch before dataset has been loaded.")

        if( self.timeit ): self.log_time += [{'event': 'start next batch', 'time': time.time()}]

        """Yield batch_x, batch_y"""
        if( self.verbose ):
            print(f'Starting {self.n_epochs} epochs.')

        if self.random_sequence is not False:
            sequence = np.load(self.random_sequence).astype('int')
        else:
            sequence = self.generate_random_sequence()

        for e in range(self.n_epochs):
            if( self.timeit ):  self.log_time += [{'event': 'start epoch', 'time': time.time()}]
            idxs = sequence[e]
            for epoch in self.get_epoch(idxs, e):
                yield epoch

    def get_epoch(self, idxs):
        raise NotImplementedError(f'Subclasses of DataGenerator must implement get_epoch method')

    def get_validation_set(self):
        if( not self.isset ):
            raise ValueError(f"Cannot get validation set before dataset has been loaded")
        
        return [self.images_x[idx] for idx in self.val_idxs],[self.images_y[idx] for idx in self.val_idxs]
    
    def generate_random_sequence(self):
        if( not self.isset ):
            raise ValueError(f"Cannot generate random sequence before dataset has been loaded")
        
        """Generates random sampling sequence. Returns np.array of 
        size n_epochs x n_samples"""
        np.random.seed(self.random_seed)
        sequence = np.zeros((self.n_epochs, len(self.train_idxs)))

        for i in range(self.n_epochs):
            np.random.shuffle(self.train_idxs)
            sequence[i,:] = self.train_idxs[:]

        # save sequence
        saveAs = f'{self.name}.npy'
        ids = 1
        while os.path.exists(saveAs):
            saveAs = f'{self.name}_{ids}.npy'
            ids += 1
        np.save(saveAs, sequence)

        if( self.verbose ):
            print(f'Generated new random sequence, saved as: {saveAs}')

        return sequence.astype('int')

    def get_sample(self, idx, item_id):
        raise NotImplementedError(f'Subclasses of DataGenerator must implement get_sample method')

    def _augment(self, X, Y,seed=None):
        if( seed != None ):
            np.random.seed(seed)
    
        mirrors = np.random.random((X.shape[0], 2))
        v_mirror = mirrors[:,0]<0.5
        h_mirror = mirrors[:,1]<0.5

        X2 = X.copy()
        Y2 = Y.copy()
        
        # Orientation
        X2[v_mirror] = X2[v_mirror,::-1,:,:]
        Y2[v_mirror] = Y2[v_mirror,::-1,:,:]
        X2[h_mirror] = X2[h_mirror,:,::-1,:]
        Y2[h_mirror] = Y2[h_mirror,:,::-1,:]

        if( self.augment_illumination is not False ):
            illumination = np.random.normal(scale=self.augment_illumination, size=(X.shape[0],))
            for i in range(X2.shape[0]):
                X2[i] += illumination[i]
        if( self.augment_noise is not False ):
            noise = np.random.normal(scale=self.augment_noise, size=X.shape)
            X2 += noise

        if( self.timeit ):  self.log_time += [{'event': 'augment', 'time': time.time()}]
        return X2,Y2

    def _preprocess(self, im):
        if( im.dtype == np.uint8):
            return (im/255)-0.5
        else:
            return im