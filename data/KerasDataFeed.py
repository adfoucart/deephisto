# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

Keras-style DataFeed for Tensorflow v2 pipelines
'''

from tensorflow import keras
import numpy as np
import os

class KerasDataFeed():

	def __init__(self, generator=None, **kwargs):
		self.accepted_params = {
			'directory' : None, 
			'tile_size' : None,
			'batch_size' : None,
			'verbose' : False,
			'augmentData' : True,
			'randomSequence' : False,
			'seed' : 0,
			'max_epochs' : 100
			}

		for key,default in accepted_params:
			if key in kwargs:
				setattr(self, key, kwargs[key])
			else:
				setattr(self, key, default)

		
	def __len__(self):
		return len(self.files_X)

	def __getitem__(self, idx):
		batch_X = np.zeros((self.batch_size,self.tile_size, self.tile_size, 3))
		batch_Y = np.zeros((self.batch_size,self.tile_size,self.tile_size))

		im = self.images_X[idx]
		mask = self.images_Y[idx]

		tlimits = np.array((im.shape[0]-self.tile_size, im.shape[1]-self.tile_size))
		rts = (np.random.random((self.batch_size,2))*tlimits).astype('int')

		i = 0
		while i < self.batch_size:
			rt = rts[i]
			batch_X[i,:,:,:] = im[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size,:]
			batch_Y[i,:,:] = mask[rt[0]:rt[0]+self.tile_size, rt[1]:rt[1]+self.tile_size]

		return batch_X,batch_Y