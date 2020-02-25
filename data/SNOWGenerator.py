# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

Add SNOW imperfections to the annotations by removing labels and deforming the contours
'''

import os
import numpy as np
from skimage.measure import find_contours, label
from skimage.morphology import disk, erosion, dilation
from skimage.draw import polygon
from skimage.io import imread, imsave
from dhutil.tools import printProgressBar

'''
params:
* input_dir -> directory containing annotation masks
* output_dir -> where to put the results.
* pRemove -> probability of removing an object (= noise %)
* defStd -> std of random erosion/diation of the object
* simplificationFactor -> ratio of points kept in the border of the polygon
* boundingBoxes -> output bounding boxes instead of segmentation
* doLabelAugmentation -> add +5 -5 versions of the label masks 

Warning: this is a very slow and unoptimized process... Go grab a coffee while it's running !
'''
def SNOWGenerator(input_dir, output_dir, pRemove, defStd, simplificationFactor, boundingBoxes=False, doLabelAugmentation=False, verbose=False):
    Yfiles = os.listdir(input_dir)
    total = len(Yfiles)

    if( verbose ): 
        print("Generating SNOW annotations for %d files in %s"%(total, input_dir))
        print("Noise %%: %d"%(pRemove*100))
        print("Std of erosion/dilation: %d"%defStd)
        print("Simplification factor: %d"%simplificationFactor)
        if( boundingBoxes ):
            print("Using bounding boxes as output.")
        if( doLabelAugmentation ):
            print("Adding label augmentation.")

    nLabelsIn = 0
    nLabelsOut = 0

    for f in Yfiles:
        Y = imread(os.path.join(input_dir, f))
        Yout = np.zeros_like(Y)

        # Check if annotations are already labelled or if it's a mask:
        labels = np.unique(Y[Y>0])
        if(len(labels) == 1):
            Y = label(Y>0)
            labels = np.unique(Y[Y>0])

        nLabels = len(labels)
        if( verbose ): 
            print("Processing: %s"%f)
            if( nLabels > 0 ): printProgressBar(0, nLabels)

        nLabelsIn += nLabels
        
        # SNOW generation:

        # Draw which objects will be removed from the image
        toRemove = np.random.random(nLabels) < pRemove
        # Draw random deformation parameters:
        if( defStd > 0 ):
            diskRadii = np.random.normal(0, defStd, labels.max()+1).astype('int')

        # If label augmentation : prepare additional outputs:
        if( doLabelAugmentation ):
            Yout_p5 = np.copy(Yout)
            Yout_m5 = np.copy(Yout)
            disk5 = disk(5)

        # To be able to get the complete contour of all objects, we first zero-pad the label image:
        Ypadded = np.zeros((Y.shape[0]+2, Y.shape[1]+2)).astype('uint8')
        Ypadded[1:-1,1:-1] = Y

        newLabel = 0 # We will re-label from 0 the resulting annotation.
        for idl,lab in enumerate(labels):

            if( toRemove[idl] ): # Ignore remove objects so they won't be added to Yout
                continue

            # Select current object
            Yobj = (Ypadded==lab).astype('uint8')

            #  Deforming objects
            if( defStd > 0 ):
                if( diskRadii[lab] < 0 ):
                    Yobj = erosion(Yobj, disk(abs(diskRadii[lab])))
                elif( diskRadii[lab] > 0 ):
                    Yobj = dilation(Yobj, disk(abs(diskRadii[lab])))
            
            # Check if we completely removed the object in the process...
            if( Yobj.sum() == 0 ): continue

            # Find the contour.
            if( simplificationFactor > 1 ):
                cont = find_contours(Yobj, 0)[0]
                sCont = cont[::simplificationFactor]
                sContours = np.vstack([sCont, cont[0]]) # Close contour
                rr, cc = polygon(sCont[:,0], sCont[:,1], Yout.shape)
                # Replace object
                Yobj[Yobj>0] = 0
                Yobj[rr,cc] = 1
            
            # Replace with bounding boxes
            if( boundingBoxes ):
                # Compute boundaries
                rows = np.any(Yobj, axis=1)
                cols = np.any(Yobj, axis=0)
                rmin,rmax = np.nonzero(rows)[0][[0,-1]]
                cmin,cmax = np.nonzero(cols)[0][[0,-1]]
                # Replace object
                Yobj[Yobj>0] = 0
                Yobj[rmin:rmax+1, cmin:cmax+1] = 1

            # Add to output array with new label after de-padding
            newLabel += 1
            Yout += Yobj[1:-1,1:-1]*newLabel

            # Label Augmentation
            if( doLabelAugmentation ):
                Yobj_p5 = dilation(Yobj, disk5)
                Yobj_m5 = erosion(Yobj, disk5)
                Yout_p5 += Yobj_p5[1:-1,1:-1]*newLabel
                Yout_m5 += Yobj_m5[1:-1,1:-1]*newLabel

            if( verbose ): printProgressBar(idl+1, nLabels)

        nLabelsOut += Yout.max()

        # Save output file(s)
        imsave(os.path.join(output_dir, f), Yout)
        if( doLabelAugmentation ):
            ext = f.rsplit('.')[-1]
            imsave(os.path.join(output_dir, f.replace('.%s'%ext, '-p5.%s'%ext)), Yout_p5)
            imsave(os.path.join(output_dir, f.replace('.%s'%ext, '-m5.%s'%ext)), Yout_m5)

    if( verbose ): print("In: %d -> Out: %d"%(nLabelsIn, nLabelsOut))