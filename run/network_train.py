from data import ArtefactDataFeed, EpitheliumDataFeed, WarwickDataFeed
from network import PAN, ShortRes
from dhutil.network import train

def run_train(params):
    net = None
    feed = None

    if params['network'] == 'pan' :
        net = PAN(params)
    elif params['network'] == 'shortres':
        net = ShortRes(params)

    net.create_network()

    if params['feed_name'] == 'artefact':
        feed = ArtefactDataFeed(params, db, net.predict_generator)
    elif params['feed_name'] == 'epi':
        feed = EpitheliumDataFeed(params, db, net.predict_generator)
    elif params['feed_name'] == 'warwick':
        feed = WarwickDataFeed(params, db, net.predict_generator)

    batch_size = params['batch_size'] if 'batch_size' in params else 20
    nPatchesInValSet = params['nPatchesInValSet'] if 'nPatchesInValSet' in params else 40
    epochs = params['epochs'] if 'epochs' in params else 1500

    train(net, feed, batch_size, nPatchesInValSet, epochs)