from skimage.transform import downscale_local_mean, resize
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import opening, closing, disk
OPENSLIDE_ACTIVE = True
try:
    import openslide
except ImportError:
    OPENSLIDE_ACTIVE = False

import numpy as np

'''
Return image mask with 0 = background (no tissue), 1 = foreground (tissue)
'''
def getBackgroundMask(rgb):
    # Work at lower resolution
    scale_factor = 8
    lr = downscale_local_mean(rgb, (scale_factor,scale_factor,1))
    hsv = rgb2hsv(lr)

    bg = hsv[:,:,1]<0.04
    bg = resize(bg, (rgb.shape[0],rgb.shape[1]))<0.5
    bg = opening(closing(bg, disk(5)), disk(10)).astype('bool')

    return bg

'''
Load a whole-slide image (ndpi, svs) & extract image @ 1.25x magnification
'''
def get_image(fpath, verbose=False):
    if( OPENSLIDE_ACTIVE == False ): 
        print("Error - Openslide could not be loaded")
        return False
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
