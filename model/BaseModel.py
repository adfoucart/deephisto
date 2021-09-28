import os
import json

import numpy as np
import tensorflow as tf

from . import MCCMetric

class BaseModel:
    """Abtract model class.
    
    Setup the pipeline and implement save/load/plot/compile/fit methods.
    """

    def __init__(self, **kwargs):
        default_params = {
            'clf_name' : None,
            'checkpoints_dir' : None,
            'summaries_dir' : None,        
            'tile_size': None,
            'verbose': False,
            'learning_rate' : 1e-4,
            'eps': 1e-8,
            'seed': 0,
            'patience': 15,
            'loadFrom': False,   # False or path to hdf5
        }

        for key,default in default_params.items():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            elif default is not None:
                setattr(self, key, default)
            else:
                raise KeyError(f'Parameter "{key}" must be set.')

    def _setup(self):
        """Load model from file or create model from _set_model method."""
        if self.loadFrom is False:
            if( isinstance(self.tile_size, int) ):
                self.input_shape = (self.tile_size, self.tile_size, 3)
            elif( isinstance(self.tile_size, tuple) and len(self.tile_size) == 2 ):
                self.input_shape = self.tile_size + (3,)
            else:
                raise ValueError(f"Tile size must be int or tuple (height,width). Was: {self.tile_size}")

            self.inputs = tf.keras.Input(shape=self.input_shape)
            self._set_model()
            self._compile()
        else:
            if( self.verbose ):
                print(f"Loading model from file: {self.loadFrom}")
            self.model = tf.keras.models.load_model(self.loadFrom, compile=False, custom_objects={'leaky_relu': tf.nn.leaky_relu})


    def _compile(self):
        """Create & compile keras model from inputs and outputs."""
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        
        opt = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, 
                epsilon=self.eps,
                name='Adam')

        self.model.compile(
            optimizer=opt, 
            loss=tf.keras.losses.BinaryCrossentropy(), 
            metrics=[
                tf.keras.losses.BinaryCrossentropy(name='crossentropy'),
                MCCMetric(name='mcc')
                ]
            )

    def plot(self):
        """Save model plot to model.png file"""
        tf.keras.utils.plot_model(self.model, show_shapes=True)

    def summary(self):
        """Print model summary"""
        self.model.summary()

    def save(self, fname=None):
        """Save to hdf5. By default, save tu checkpoint directory with model name as file name."""
        if fname is None:
            fname = os.path.join(self.checkpoints_dir, f"{self.clf_name}.hdf5")
        if( fname.endswith(".hdf5") ):
            self.model.save(fname)
        else:
            raise ValueError(f"Filename for saving the model must be .hdf5. Provided value: {fname}")

    def fit(self, dataset):
        """Use next_batch() from dataset class as generator to fit 
        the model."""
        val_data = None
        callbacks = None
        if( dataset.pValidation > 0. ):
            val_data = dataset.get_validation_set()
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_crossentropy', 
                    patience=self.patience
                    ),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_dir, f"{self.clf_name}_best.hdf5"), save_best_only=True)
                ]
        self.history = self.model.fit(
                dataset.next_batch(),
                epochs=dataset.n_epochs, 
                steps_per_epoch=dataset.batches_per_epoch, 
                validation_data=val_data, 
                callbacks=callbacks,
                verbose=2
            )

        if( dataset.pValidation > 0. ):
            np.save(os.path.join(self.summaries_dir, f"{self.clf_name}_history_crossentropy.npy"), self.history.history['val_crossentropy'])
            np.save(os.path.join(self.summaries_dir, f"{self.clf_name}_history_mcc.npy"), self.history.history['val_mcc'])
        else:
            np.save(os.path.join(self.summaries_dir, f"{self.clf_name}_history_crossentropy.npy"), self.history.history['crossentropy'])
            np.save(os.path.join(self.summaries_dir, f"{self.clf_name}_history_mcc.npy"), self.history.history['mcc'])

        return self.history