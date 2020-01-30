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
import sys
import openslide
from skimage.morphology import opening, closing, disk
from skimage.color import rgb2gray, rgb2hsv
from skimage.transform import downscale_local_mean, resize
from dhutil.tools import printProgressBar
from dhutil.network import restore

'''
Load a whole-slide image (ndpi, svs) & extract image @ 1.25x magnification
'''
def get_image(fpath, verbose=False):
    if( verbose ): print("Loading RGB image @ 1.25x magnification")
    slide = openslide.OpenSlide(fpath)
    op = float(slide.properties['openslide.objective-power']) # maximum magnification
    down_factor = op/1.25
    level = slide.get_best_level_for_downsample(down_factor)
    relative_down = down_factor / slide.level_downsamples[level]

    rgb_image = slide.read_region((0,0), level, slide.level_dimensions[level])
    newsize = np.array((slide.level_dimensions[level][0]/relative_down, slide.level_dimensions[level][1]/relative_down)).astype('int')
    rgb_image = np.array(rgb_image.resize(newsize))[:,:,:3].astype('uint8')
    return rgb_image

'''
blend2Images mathod From HistoQC (Janowczyk et al, 2019)
https://github.com/choosehappy/HistoQC

Produces output image with artefact regions in green.
'''
def blend2Images(img, mask):
    if (img.ndim == 3):
        img = rgb2gray(img)
    if (mask.ndim == 3):
        mask = rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out

'''
Get a mask with the low-saturation regions, which are regions with no tissue.
'''
def getBackground(rgb):
    # Work at lower resolution
    scale_factor = 8
    lr = downscale_local_mean(rgb, (scale_factor,scale_factor,1))
    hsv = rgb2hsv(lr)

    bg = hsv[:,:,1]<0.04
    bg = resize(bg, (rgb.shape[0],rgb.shape[1]))<0.5
    bg = opening(closing(bg, disk(5)), disk(10)).astype('bool')

    return bg

'''
Process an 1.25x mag RGB image.

* sess -> tensorflow session
* rgb -> the image
* X -> the input placeholder Tensor
* Y_seg -> the segmentation output Tensor
* tile_size
'''
def process(sess, rgb, X, Y_seg, tile_size, verbose=False):
    if(verbose): print("Processing RGB image")

    # Background detection
    if(verbose): print("Background detection.")
    bg_mask = getBackground(rgb)
    
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

    mask_pred = im_pred<=0.5
    mask_out = mask_pred*bg_mask
    im_out = blend2Images(rgb, mask_out)

    return im_pred,im_out, bg_mask

'''
Detect artefact on WSI

* input_dir -> directory with WSI
* output_dir -> directory to store result images
* network_path -> checkpoint path of the network
* ext (optional) -> extension filter 

Results are stored in output_dir.
'''
def artefact_detector(input_dir, output_dir, network_path, ext=None, verbose=False):
    if( ext == None ):
        ext = ['svs', 'ndpi']
    
    sess,saver = restore(network_path)

    # Load input placeholders
    try:
        X = tf.get_default_graph().get_tensor_by_name("features/X:0")
    except KeyError:
        X = tf.get_default_graph().get_tensor_by_name("ae/features/X:0")

    # Get tile size from placeholder
    tile_size = X.get_shape()[1]

    # Load output tensor
    Y_seg = tf.get_default_graph().get_tensor_by_name("output/segmentation:0")
    
    # Load input files list
    input_files = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir,f)) and f.split('.')[-1] in ext]
    
    # Process files
    for input_image in input_files:
        if(verbose): print(input_image)
        # Get RGB
        rgb = get_image(input_image)
        # Process
        im_pred, im_out, bg_mask = process(sess, rgb, X, Y_seg, params)
        # Save results
        imsave(os.path.join(output_dir, "%s_rgb.png"%os.path.basename(input_image)), rgb)
        imsave(os.path.join(output_dir, "%s_prob.png"%os.path.basename(input_image)), im_pred)
        imsave(os.path.join(output_dir, "%s_fused.png"%os.path.basename(input_image)), im_out)
        imsave(os.path.join(output_dir, "%s_bg.png"%os.path.basename(input_image)), bg_mask)