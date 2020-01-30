# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart
'''

import tensorflow as tf
from network import PAN, ShortRes

'''
Get the number of parameters in the networks
'''
def get_number_of_parameters():

    params = {
        'clf_name': 'stats',
        'checkpoints_dir': '.',
        'summaries_dir': '.',
        'tile_size': 256
    }

    pan = PAN(params)
    sr = ShortRes(params)

    for net in [pan, sr]:
        net.create_network()

        with net.mainGraph.as_default():
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                # print(shape)
                # print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    # print(dim)
                    variable_parameters *= dim.value
                # print(variable_parameters)
                total_parameters += variable_parameters
            print(total_parameters)