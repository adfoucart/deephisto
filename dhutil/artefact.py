from skimage.transform import downscale_local_mean, resize
from skimage.color import rgb2hsv
from skimage.morphology import opening, closing, disk

def getBackgroundMask(rgb):
    # Work at lower resolution
    scale_factor = 8
    lr = downscale_local_mean(rgb, (scale_factor,scale_factor,1))
    hsv = rgb2hsv(lr)

    bg = hsv[:,:,1]<0.04
    bg = resize(bg, (rgb.shape[0],rgb.shape[1]))<0.5
    bg = opening(closing(bg, disk(5)), disk(10)).astype('bool')

    return bg