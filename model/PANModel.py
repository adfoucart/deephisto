import numpy as np
import tensorflow as tf

from . import BaseModel

class PANModel(BaseModel):
    """Implement the PAN model.
    
    Architecture:
                                |-------------------out3---|
                                |      |------------out2-- +-- out
 in -- r1 -- r2 -- r3 -- r4 -- ur1 -- ur2 -- ur3 -- out1---|
       |      |------------------------|      |
       |--------------------------------------|
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_params = {
            'convs_per_unit' : 3
        }

        for key,default in default_params.items():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            elif default is not None:
                setattr(self, key, default)
            else:
                raise KeyError(f'Parameter "{key}" must be set.')

        self._setup()

    def _add_residual(self, x, width=None, withMaxPooling=True):
        if width is None: width = x.shape[3]

        x_ = x
        for i in range(self.convs_per_unit):
            x_ = tf.keras.layers.Conv2D(width, 3, activation=tf.nn.leaky_relu, padding='same')(x_)

        x = tf.keras.layers.Conv2D(width, 1, activation=None)(x)
        x = tf.add(x, x_)
        if( withMaxPooling ):
            x = tf.keras.layers.MaxPool2D(2)(x)
        
        return x

    def _add_up(self, x, width=None):
        if width is None: width = x.shape[3]

        x_ = x
        x_ = tf.keras.layers.Conv2DTranspose(width, 3, strides=(2,2), activation=tf.nn.leaky_relu, padding='same')(x_)
        x_ = tf.keras.layers.Conv2D(width, 3, activation=tf.nn.leaky_relu, padding='same')(x_)
        x = tf.keras.layers.Conv2DTranspose(width, 1, strides=(2,2), activation=tf.nn.leaky_relu, padding='same')(x)
        return tf.add(x, x_)

    def _set_model(self):
        r1 = self._add_residual(self.inputs, width=64, withMaxPooling=True)
        r2 = self._add_residual(r1, width=128, withMaxPooling=True)
        r3 = self._add_residual(r2, width=256, withMaxPooling=True)

        r4 = self._add_residual(r3, width=512, withMaxPooling=False)

        ur1 = self._add_up(r4, width=256)
        ur2 = self._add_up(tf.concat([ur1, r2], axis=3), width=128)
        ur3 = self._add_up(tf.concat([ur2, r1], axis=3), width=64)

        out_1 = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.leaky_relu, padding='same')(ur3)
        out_2 = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.leaky_relu, padding='same')(tf.image.resize(ur2, self.input_shape[:2]))
        out_3 = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.leaky_relu, padding='same')(tf.image.resize(ur1, self.input_shape[:2]))

        x = out_1+out_2+out_3
        x = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.leaky_relu, padding='same')(x)
        self.outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax, padding='same')(x)