'''
Author: Adrien Foucart
'''
import threading
from generator import ArtefactGenerator, ArtefactBlockGenerator, GlasGenerator, PreFetch
from model import PANModel, ShortResidualModel, UNetModel

'''
Train a network on a data generator.

params -> dictionary.
Required fields:
* model
* generator
* dataset_dir
* tile_size
* clf_name
* checkpoints_dir
* summaries_dir
'''
def run_train(params):
    model = None
    feed = None

    model = get_model(params)
    feed = get_feed(params)
    pf = PreFetch(feed)

    t1 = threading.Thread(target=pf.fetch)
    t2 = threading.Thread(target=model.fit, args=(pf,))

    t1.start()
    t2.start()

    return t1,t2

def get_model(params):
    if params['model_name'].lower() == 'pan' :
        model = PANModel(**params)
    elif params['model_name'].lower() == 'shortres':
        model = ShortResidualModel(**params)
    elif params['model_name'].lower() == 'unet':
        model = UNetModel(**params)

    return model

def get_feed(params):
    if params['generator_name'].lower() in ['artefact', 'artifact']:
        feed = ArtefactGenerator(**params)
    elif params['generator_name'].lower() in ['artefact_block']:
        feed = ArtefactBlockGenerator(**params)
    elif params['generator_name'].lower() in ['glas']:
        feed = GlasGenerator(**params)
    return feed