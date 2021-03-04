# -*- coding: utf-8 -*-
'''
Author: Adrien Foucart

Script to produce a whole-slide artefact detection using the PAN-GA50 network.

Produces a 1.25x mag output w/ 3 images:
    * input_image_rgb.png -> RGB
    * input_image_prob.png -> P(artefact)
    * input_image_fused.png -> Output image with green artefacts
'''

import tensorflow as tf
import os
import numpy as np
from skimage.io import imsave
from dhutil.artefact import getBackgroundMask, blend2Images, get_image, imageWithOverlay
from dhutil.tools import printProgressBar
from dhutil.network import restore
import time

'''
Process an 1.25x mag RGB image.

* sess -> tensorflow session
* rgb -> the image
* X -> the input placeholder Tensor
* Y_seg -> the segmentation output Tensor
* tile_size
'''
def process(sess, rgb, X, Y_seg, tile_size, bgDetection=True, verbose=False):
    if(verbose): print("Processing RGB image")

    # Background detection
    if(verbose): print("Background detection.")
    bg_mask = np.ones((rgb.shape[0],rgb.shape[1])).astype('bool')
    if( bgDetection ): bg_mask = getBackgroundMask(rgb) # 1 = foreground, 0 = background
    
    overlap = 2
    
    if(verbose): print("Tiling")
    im = rgb/255. - 0.5 # Offset image
    imshape = im.shape
    nr,nc = (overlap*np.ceil((imshape[0]-1)/tile_size), overlap*np.ceil((imshape[1]-1)/tile_size))
    yr,xr = (np.arange(0, nr)*((imshape[0]-1-tile_size)//(nr-1))).astype('int'), (np.arange(0, nc)*((imshape[1]-1-tile_size)//(nc-1))).astype('int')
    mesh = np.meshgrid(yr,xr)
    tiles = zip(mesh[0].flatten(), mesh[1].flatten())

    im_pred = np.zeros(imshape[:2]).astype('float')

    if(verbose): 
        print("Prediction")
        printProgressBar(0, len(mesh[0].flatten()))
    for idt,t in enumerate(tiles):
        batch_X = [im[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size]]
        sm = Y_seg.eval(session=sess, feed_dict={X:batch_X})[:,:,:,0]
        im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size] = np.maximum(im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size], sm[0,:,:])
        if(verbose): printProgressBar(idt+1, len(mesh[0].flatten()))

    mask_pred = im_pred<=0.5 # mask_pred will be 0=artefact, 1=no artefact
    mask_out = mask_pred*bg_mask # 1=normal tissue, 0 = background or artefact
    im_out = imageWithOverlay(rgb,mask_out)#blend2Images(rgb, mask_out)

    return im_pred, im_out, bg_mask

'''
Detect artefact on WSI

* input_dir -> directory with WSI
* output_dir -> directory to store result images
* network_path -> checkpoint path of the network
* asThread -> threaded version where the method polls the input folder regularly for new files to digest
* ext (optional) -> extension filter 
* bgDetection (optional) -> if True, also remove background

Results are stored in output_dir.
'''
def artefact_detector(input_dir, output_dir, network_path, asThread=False, ext=None, bgDetection=True, verbose=False):
    if( ext == None ):
        ext = ['svs', 'ndpi']
    
    sess,saver = restore(network_path)

    # Load input placeholders
    try:
        X = tf.get_default_graph().get_tensor_by_name("features/X:0")
    except KeyError:
        X = tf.get_default_graph().get_tensor_by_name("ae/features/X:0")

    # Get tile size from placeholder
    tile_size = X.get_shape().as_list()[1]

    # Load output tensor
    Y_seg = tf.get_default_graph().get_tensor_by_name("output/segmentation:0")
    
    attemptsRemaining = {}

    while True:
        # Load input files list
        input_files = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir,f)) and f.split('.')[-1] in ext]
        if( asThread ): 
            time.sleep(5)

        # Process files
        for input_image in input_files:
            if(verbose): print(input_image)
            # Get RGB
            try:
                rgb = get_image(input_image)
            except Exception as e:
                if( input_image in attemptsRemaining ):
                    attemptsRemaining[input_image] -= 1
                else:
                    attemptsRemaining[input_image] = 10

                if( attemptsRemaining[input_image] > 0 ):
                    print('Error trying to process %s... %d attempts remaining.'%(input_image, attemptsRemaining[input_image]))
                    continue
                else:
                    print('Too many errors. Passing file %s'%input_image)
                    continue

            # Process
            im_pred, im_out, bg_mask = process(sess, rgb, X, Y_seg, tile_size, bgDetection, verbose)

            # Compute & save quick stat
            if( bgDetection ):
                total_artefact_tissue = ((im_pred>0.5)*bg_mask).sum()
                total_tissue = (bg_mask).sum()
                with open(os.path.join(output_dir, "%s_stat.txt"%os.path.basename(input_image)), 'w') as fp:
                    fp.write("Artefact in non-background tissue: %.4f %% of pixels."%(100*(total_artefact_tissue/total_tissue)))

            # Save results
            imsave(os.path.join(output_dir, "%s_rgb.png"%os.path.basename(input_image)), rgb)
            imsave(os.path.join(output_dir, "%s_prob.png"%os.path.basename(input_image)), im_pred)
            imsave(os.path.join(output_dir, "%s_fused.png"%os.path.basename(input_image)), im_out)
            if( bgDetection ):
                imsave(os.path.join(output_dir, "%s_bg.png"%os.path.basename(input_image)), bg_mask.astype('uint8')*255)

            # Rename file to make sure we don't redo it after
            os.rename(input_image, '%s.done'%input_image)

        if( not asThread ): break