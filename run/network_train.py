# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart
'''
from data import ArtefactDataFeed, EpitheliumDataFeed, WarwickDataFeed
from network import PAN, ShortRes
from dhutil.network import train
from dhutil.tools import PreFetcher

'''
Train a network on a data feed.

params -> dictionnary.
Required fields:
* network
* feed_name
* dataset_dir
* tile_size
* clf_name
* checkpoints_dir
* summaries_dir
'''
def run_train(params):
    net = None
    feed = None

    if params['network'].lower() == 'pan' :
        net = PAN(params)
    elif params['network'].lower() == 'shortres':
        net = ShortRes(params)

    net.create_network()

    if params['feed_name'].lower() in ['artefact', 'artifact']:
        feed = ArtefactDataFeed(params, 'train', net.predict_generator)
    elif params['feed_name'].lower() in ['epithelium', 'epi']:
        feed = EpitheliumDataFeed(params, 'train', net.predict_generator)
    elif params['feed_name'].lower() in ['warwick', 'glas']:
        feed = WarwickDataFeed(params, 'train', net.predict_generator)

    batch_size = params['batch_size'] if 'batch_size' in params else 20
    nPatchesInValSet = params['nPatchesInValSet'] if 'nPatchesInValSet' in params else 40
    epochs = params['epochs'] if 'epochs' in params else 1500

    train(net, feed, batch_size, nPatchesInValSet, epochs)

'''
Train a network on a data feed, with the feed & the network on two different threads

params -> dictionnary.
Required fields:
* network
* feed_name
* dataset_dir
* tile_size
* clf_name
* checkpoints_dir
* summaries_dir
'''
def run_train_threaded(params):
    import threading

    net = None
    feed = None

    if params['network'].lower() == 'pan' :
        net = PAN(params)
    elif params['network'].lower() == 'shortres':
        net = ShortRes(params)

    net.create_network()

    if params['feed_name'].lower() in ['artefact', 'artifact']:
        feed = ArtefactDataFeed(params, 'train', net.predict_generator)
    elif params['feed_name'].lower() in ['epithelium', 'epi']:
        feed = EpitheliumDataFeed(params, 'train', net.predict_generator)
    elif params['feed_name'].lower() in ['warwick', 'glas']:
        feed = WarwickDataFeed(params, 'train', net.predict_generator)

    pf = PreFetcher(feed)

    batch_size = params['batch_size'] if 'batch_size' in params else 20
    nPatchesInValSet = params['nPatchesInValSet'] if 'nPatchesInValSet' in params else 40
    epochs = params['epochs'] if 'epochs' in params else 1500

    threads = []

    t1 = threading.Thread(target=pf.fetch, args=(batch_size, epochs))
    t2 = threading.Thread(target=train, args=(net, pf, nPatchesInValSet, batch_size, epochs, True))

    t1.start()
    t2.start()