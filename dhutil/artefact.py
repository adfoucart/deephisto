import numpy as np
from skimage.transform import downscale_local_mean, resize
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops
import xml.etree.ElementTree as etree
from shapely.geometry import Polygon, MultiPolygon, Point
from PIL import Image, ImageDraw
OPENSLIDE_ACTIVE = True
try:
    import openslide
except ImportError:
    OPENSLIDE_ACTIVE = False
    print("WARNING - Openside couldn't be loaded. Some functionalities will not work.")
import tensorflow as tf

def getBackgroundMask(rgb, scale_factor=8):
    '''Return image mask with 
    0 = background (no tissue), 
    1 = foreground (tissue).

    Background mask is computed at a lower resolution with scale_factor
    '''

    lr = downscale_local_mean(rgb, (scale_factor,scale_factor,1))
    hsv = rgb2hsv(lr)

    bg = hsv[:,:,1]<0.04
    bg = resize(bg, (rgb.shape[0],rgb.shape[1]))<0.5
    bg = opening(closing(bg, disk(5)), disk(10)).astype('bool')

    return bg

def getBackgroundMask_v2(rgb, scale_factor=8, min_area_bg=10000, min_area_fg=1280, maxPoolingLayer=None):
    '''Return image mask with 
    0 = background (no tissue), 
    1 = foreground (tissue).

    Background mask is computed at a lower resolution with scale_factor.
    Small objects are removed. min_area is set as pixels from the original rgb image
    '''

    # Using MaxPooling for downscaling so that regions with even a small amount 
    # of visible tissue may be kept (e.g. fatty tissue)
    x = tf.constant(rgb2hsv(rgb)[:,:,1])
    x = tf.reshape(x, [1,rgb.shape[0],rgb.shape[1],1])
    if maxPoolingLayer is None:
        maxPoolingLayer = tf.keras.layers.MaxPooling2D(pool_size=(scale_factor, scale_factor), strides=(scale_factor, scale_factor))
    lr = maxPoolingLayer(x).numpy()[0,:,:,0]

    min_area_bg_ = min_area_bg/(scale_factor**2)
    min_area_fg_ = min_area_fg/(scale_factor**2)
    bg = lr<0.04 # bg==1, fg==0 at this stage
    # Remove "small background regions"
    lab = label(bg, connectivity=1)
    for obj in regionprops(lab):
        if obj.area < min_area_bg_:
            bg[lab==obj.label] = 0 # -> set as foreground
    # Remove "small foreground regions"
    lab_fg = label(bg==False)
    for obj in regionprops(lab_fg):
        if obj.area < min_area_fg_:
            bg[lab_fg==obj.label] = 1 # -> set as background

    bg = resize(bg, (rgb.shape[0],rgb.shape[1]))<0.5 # re-binarize and flip so that 0=bg

    return bg

def get_image(fpath, mag=1.25, verbose=False):
    '''Load a whole-slide image (ndpi, svs) & extract image @ 1.25x or 2.5x magnification'''
    if( OPENSLIDE_ACTIVE == False ):
        print("Error - Openslide could not be loaded")
        return False
    if( verbose ): 
        print(f"Loading RGB image @ {mag}x magnification")
    slide = openslide.OpenSlide(fpath)
    op = float(slide.properties['openslide.objective-power']) # maximum magnification
    down_factor = op/mag
    level = slide.get_best_level_for_downsample(down_factor)
    relative_down = down_factor / slide.level_downsamples[level]

    rgb_image = slide.read_region((0,0), level, slide.level_dimensions[level])
    newsize = np.array((slide.level_dimensions[level][0]/relative_down, slide.level_dimensions[level][1]/relative_down)).astype('int')
    rgb_image = np.array(rgb_image.resize(newsize))[:,:,:3].astype('uint8')
    return rgb_image

def get_image_and_anno(fpath, apath, mag=1.25, verbose=False):
    '''Load a whole-slide image (ndpi, svs) & extract image & annotation mask @ given magnification'''
    if( OPENSLIDE_ACTIVE == False ):
        print("Error - Openslide could not be loaded")
        return False
    if( verbose ): 
        print(f"Loading RGB image @ {mag}x magnification")
    slide = openslide.OpenSlide(fpath)
    op = float(slide.properties['openslide.objective-power']) # maximum magnification
    down_factor = op/mag
    level = slide.get_best_level_for_downsample(down_factor)
    relative_down = down_factor / slide.level_downsamples[level]

    rgb_image = slide.read_region((0,0), level, slide.level_dimensions[level])
    newsize = np.array((slide.level_dimensions[level][0]/relative_down, slide.level_dimensions[level][1]/relative_down)).astype('int')
    rgb_image = np.array(rgb_image.resize(newsize))[:,:,:3].astype('uint8')
    
    osa = OpenSlideAnnotation(apath, slide)
    anno_mask = osa.getMask([level])[0]
    return rgb_image, anno_mask

def imageWithOverlay(img, mask):
    '''Display green overlay on image where there is an artefact'''
    imask = mask==False
    output_image = img.copy()
    output_image[imask,:] *= np.array([0,1,0]).astype('uint8')

    return output_image

def blend2Images(img, mask):
    '''blend2Images mathod From HistoQC (Janowczyk et al, 2019)
    
    Source: https://github.com/choosehappy/HistoQC
    Produces output image with artefact regions in green.
    '''
    if (img.ndim == 3):
        img = rgb2gray(img)
    if (mask.ndim == 3):
        mask = rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out

class OpenSlideAnnotation:
    '''Reads .ndpa annotations files & produce annotation mask'''
    
    def __init__(self, fname, slide):
        self.fname = fname
        self.slide = slide
        tree = etree.parse(fname)    # Annotations
        self.root = tree.getroot()
        
        mppx = float(self.slide.properties['openslide.mpp-x']) #um/px
        mppy = float(self.slide.properties['openslide.mpp-y'])
        self.nppx = mppx*1000 # nm/px
        self.nppy = mppy*1000
        self.xoff = float(self.slide.properties['hamamatsu.XOffsetFromSlideCentre']) # in nm
        self.yoff = float(self.slide.properties['hamamatsu.YOffsetFromSlideCentre'])
        self.cx,self.cy = self.slide.level_dimensions[0][0]/2, self.slide.level_dimensions[0][1]/2

        self.C = np.array([self.cx, self.cy])
        self.T = np.array([self.xoff, self.yoff])
        self.S = np.array([self.nppx, self.nppy])
    
    def getAllAnnotations(self):
        '''Generator to get all the annotations in the XML file.

        Applies the offset & the conversion to get the coordinates in pixels'''
        pointlists = [ann.find('annotation').find('pointlist') for ann in self.root]
        for plist in pointlists:
            points = [[float(p.find('x').text),float(p.find('y').text)] for p in plist.findall('point')]
            points += [points[0]]
            points = self.C+(np.array(points)-self.T)/self.S
            yield points

    def getMask(self, levels):
        '''Get annotation masks at the required levels of magnification.'''
        ratios = [(self.slide.dimensions[0]//self.slide.level_dimensions[level][0], self.slide.dimensions[1]//self.slide.level_dimensions[level][1]) for level in levels]
        masks = [Image.new('L', self.slide.level_dimensions[level], 0) for level in levels]
        for points in self.getAllAnnotations():
            points_ = [list(np.round(points/np.array(r)).astype('int').flatten()) for r in ratios]
            for i in range(len(levels)):
                ImageDraw.Draw(masks[i]).polygon(points_[i], outline=255, fill=255)
        masks = [np.array(mask) for mask in masks]

        return masks