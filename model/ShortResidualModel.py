import numpy as np
import tensorflow as tf

from . import BaseModel

class ShortResidualModel(BaseModel):
    """Implement the Short Residual model.
    
    Architecture:
        Input
        -> Cond2D [tile_sizextile_sizexwidth]
        -> Residual w/ MaxPool
        -> Residual
        -> Residual w/ Maxpool [tile_size/4xtile_size/4xwidth]
        -> Conv2DTranspose
        -> Residual
        -> Conv2DTranspose
        -> Residual
        -> Conv2D [tile_sizextile_sizex2]
        -> Conv2D [softmax output]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_params = {
            'convs_per_unit' : 3,
            'width' : 64
        }

        for key,default in default_params.items():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            elif default is not None:
                setattr(self, key, default)
            else:
                raise KeyError(f'Parameter "{key}" must be set.')

        self._setup()

    def _add_residual(self, x, withMaxPooling=True):
        x_ = x
        for i in range(self.convs_per_unit):
            x_ = tf.keras.layers.Conv2D(self.width, 3, activation=tf.nn.leaky_relu, padding='same')(x_)
        x = tf.add(x, x_)
        if( withMaxPooling ):
            x = tf.keras.layers.MaxPool2D(2)(x)
        
        return x

    def _add_up(self, x):
        return tf.keras.layers.Conv2DTranspose(self.width, 3, strides=(2,2), activation=tf.nn.leaky_relu, padding='same')(x)

    def _set_model(self):
        x = tf.keras.layers.Conv2D(self.width, 3, activation=tf.nn.leaky_relu, padding='same')(self.inputs)
        x = self._add_residual(x, True)
        x = self._add_residual(x, False)
        x = self._add_residual(x, True)

        x = self._add_up(x)
        x = self._add_residual(x, False)
        x = self._add_up(x)
        x = self._add_residual(x, False)

        x = tf.keras.layers.Conv2D(2, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        self.outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax, padding='same')(x)