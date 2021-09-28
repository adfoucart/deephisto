'''
Author: Adrien Foucart
'''
import threading
from generator import PreFetch, GeneratorFactory
from model import ModelFactory
from typing import Tuple

def run_train(params: dict) -> Tuple[threading.Thread, threading.Thread]:
    """Train a network on a data generator.

    params -> dictionary.
    Required fields:
    * model_name
    * generator_name
    * dataset_dir
    * tile_size
    * clf_name
    * checkpoints_dir
    * summaries_dir

    Returns prefetch thread & model.fit thread"""

    assert 'model_name' in params
    assert 'generator_name' in params
    
    Model = ModelFactory.get_model(params['model_name'])
    Generator = GeneratorFactory.get_generator(params['generator_name'])

    model = Model(**params)
    feed = Generator(**params)
    pf = PreFetch(feed)

    t1 = threading.Thread(target=pf.fetch)
    t2 = threading.Thread(target=model.fit, args=(pf,))

    t1.start()
    t2.start()

    return t1,t2