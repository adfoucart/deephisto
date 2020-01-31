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