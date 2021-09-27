import time

class PreFetch(object):
    '''Handles the pre-fetch for the threaded version of the network training, using the tf2 Generators.
    '''
    def __init__(self, feed):
        self.batchIsReady = False
        self.isOver = False
        self.batch = None
        self.feed = feed
        self.pValidation = feed.pValidation
        self.n_epochs = feed.n_epochs
        self.batches_per_epoch = feed.batches_per_epoch
        pass

    def setBatch(self, batch):
        while( self.batchIsReady ):
            time.sleep(0.001)

        self.batch = batch
        self.batchIsReady = True

    def getBatch(self):
        while( not self.batchIsReady ):
            time.sleep(0.001)

        self.batchIsReady = False
        return self.batch

    def next_batch(self):
        for i in range(self.n_epochs*self.batches_per_epoch):
            yield self.getBatch()

    def get_validation_set(self):
        return self.feed.get_validation_set()

    def fetch(self):
        for batch in self.feed.next_batch():
            self.setBatch(batch)
            
        self.isOver = True