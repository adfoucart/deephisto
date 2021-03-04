import time

'''
Print iterations progress
From Benjamin Cordier on https://stackoverflow.com/a/34325723
'''
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

'''
Fuse an annotation mask with an image

Either using the "artefact" blend from Janowczyk or using the "red overlay"
'''
def saveAnnotationImage(rgb, mask, output_file, mode='red'):
    import numpy as np
    if( mode == 'red' ):
        im_out = rgb.copy()
        overlay = np.zeros((im_out.shape[0],im_out.shape[1],4)).astype('float')
        overlay[mask,0] = 1.
        overlay[mask,3] = 0.5

        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(im_out)
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(output_file)
    elif( mode == 'artefact' ):
        from dhutil.artefact import blend2Images
        from skimage.io import imsave
        out = blend2Images(rgb, mask)
        imsave(output_file, out)

'''
Compute F1 Score

T = Ground truth segmentation (binary)
P = Predicted segmentation (binary)
'''
def F1(T,P):
    TP = ((T==1)*(P==1)).sum()
    TN = ((T==0)*(P==0)).sum()
    FP = ((T==0)*(P==1)).sum()
    FN = ((T==1)*(P==0)).sum()

    try:
        return 2*TP/(2*TP+FN+FP)
    except ZeroDivisionError:
        return 0

'''
Compute Matthews correlation coefficient

T = Ground truth segmentation (binary)
P = Predicted segmentation (binary)
'''
def MCC(T,P):
    TP = float(((T==1)*(P==1)).sum())
    TN = float(((T==0)*(P==0)).sum())
    FP = float(((T==0)*(P==1)).sum())
    FN = float(((T==1)*(P==0)).sum())

    try:
        return ((TP*TN)-(FP*FN))/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(.5)
    except ZeroDivisionError:
        return 0

'''
Compute accuracy

T = Ground truth segmentation (binary)
P = Predicted segmentation (binary)
'''
def ACC(T,P):
    return (T==P).sum()/((T==P).sum()+(T!=P).sum())

'''
Compute Jaccard Index

T = Ground truth segmentation (binary)
P = Predicted segmentation (binary)
'''
def JACC(T,P):
    TP = float(((T==1)*(P==1)).sum())
    TN = float(((T==0)*(P==0)).sum())
    FP = float(((T==0)*(P==1)).sum())
    FN = float(((T==1)*(P==0)).sum())

    try:
        return TP/(TP+FN+FP)
    except ZeroDivisionError:
        return 0

'''
Handle the pre-fetch for the threaded version of the network training.
'''
class PreFetcher(object):
    def __init__(self, feed):
        self.batchIsReady = False
        self.isOver = False
        self.batch = None
        self.feed = feed
        pass

    def setBatch(self, batch):
        while( self.batchIsReady ):
            time.sleep(0.001)

        self.batch = batch
        self.batchIsReady = True

    def getBatch(self):
        while( self.batchIsReady == False ):
            time.sleep(0.001)

        self.batchIsReady = False
        return self.batch

    def validation_set(self, n):
        return self.feed.validation_set(n)

    def fetch(self, batch_size, epochs):
        for X,Y,c in self.feed.next_batch(batch_size,epochs):
            self.setBatch((X,Y,c))
            
        self.isOver = True