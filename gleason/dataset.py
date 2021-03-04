'''
Script to correct the Gleason2019 publicly released training dataset (https://gleason2019.grand-challenge.org/Register/)

Requires skimage and SimpleITK

To use:
* Download the whole dataset into a directory, with each expert's map in a directory called Maps1_T, Maps2_T, ...
* Copy the final_list_to_remove.json file to that directory
* Run this script with 

    python dataset.py /path/to/directory

The script will:
* Remap the annotations (grade > 5 are mistakes and should be = 1) (results are put into Remaps1_T, Remaps2_T...)
* Generate STAPLE consensus maps for each image (not using the maps with unclosed borders, which are listed in the JSON file)
* Generate Majority Vote and Weighted Vote consenus maps for each image
* Generate the 6 "Leave-one-Out" maps for STAPLE, MV & WV.

(Note: this takes a lot of time...)
'''

import os
from skimage.io import imread,imsave
from dhutil.tools import printProgressBar
import SimpleITK as sitk
import numpy as np
import json
from skimage.morphology import binary_dilation, disk

'''
Correction of "grades" > 5 (which should be grade 1)
'''
def remap(directory):
    train_dir = os.path.join(directory, 'train')
    maps_dir = [os.path.join(directory, 'Maps%d_T'%i) for i in range(1,7)] # Original maps
    maps_out = [os.path.join(directory, 'Remaps%d_T'%i) for i in range(1,7)] # Remaps

    # Create "Remaps" directories if they don't exist yet
    for m in maps_out:
        if( os.path.exists(m) == False ):
            os.mkdir(m)

    images = [f for f in os.listdir(train_dir)]

    printProgressBar(0,len(images))
    for idi,im in enumerate(images):
        for idm,m in enumerate(maps_dir):
            f_out = os.path.join(maps_out[idm], '%s_classimg_nonconvex.png'%im.replace('.jpg',''))
            if os.path.isfile(f_out): break

            fname = os.path.join(m,'%s_classimg_nonconvex.png'%im.replace('.jpg',''))
            if( os.path.isfile(fname) ):
                anno = imread(fname)
                anno[anno>5] = 1
                imsave(f_out, anno)
        printProgressBar(idi,len(images))

'''
Generate STAPLE consensus for a set of annotation maps
'''
def coreSTAPLE(directory, core, maps, msf, reader):
    images = []

    mmax = 0
    for m in maps:
        fname = os.path.join(directory, os.path.join(m, '%s_classimg_nonconvex.png'%core))

        if( os.path.isfile(fname) ):
            reader.SetFileName(fname)
            im = reader.Execute()
            images += [im]
            im_max = sitk.GetArrayFromImage(im).max()
            if( im_max > mmax ): mmax = im_max
    try:
        cimage = sitk.GetArrayFromImage(msf.Execute(images))
        cmax = cimage.max()
        if( cmax > mmax ):
            mask = cimage==cmax
            maskd = binary_dilation(mask, disk(1))
            maskd[mask] = False
            newv = np.median(cimage[maskd])
            cimage[mask] = newv
        return cimage
    except RuntimeError:
        return None

'''
Generate STAPLE consensus for all requested annotation maps, with mistaken maps removed
'''
def generate_STAPLE(directory, maps, keep_annotations, map_out):
    train_dir = os.path.join(directory, 'train')

    m_out = os.path.join(directory,map_out)
    if( os.path.exists(m_out) == False ):
        os.mkdir(m_out)

    # Get all the cores from the training set
    f_train = os.listdir(train_dir)
    cores = [f.split('.')[0] for f in f_train]

    reader = sitk.ImageFileReader()
    msf = sitk.MultiLabelSTAPLEImageFilter()

    printProgressBar(0,len(cores)-1)
    for i,core in enumerate(cores):
        fname_out = os.path.join(m_out, '%s.png'%core)
        if os.path.isfile(fname_out): continue # pass if it's already been done

        maps_ = []
        for idm,m in enumerate(maps):
            if(keep_annotations[i,idm]):
                maps_ += [m]

        im_ = coreSTAPLE(directory, core, maps_, msf, reader)
        if( im_ == None ):
            print("EXCLUDE: ",core)
            continue
        imsave(fname_out, im_)
        
        printProgressBar(i,len(cores)-1)

'''
Majority vote if weight = 'majority', otherwise apply weights
'''
def generate_vote(directory, maps, keep_annotations, weights, map_out):
    train_dir = os.path.join(directory, 'train')
    maps_dir = [os.path.join(directory, m) for m in maps]
    images = [f for f in os.listdir(train_dir)]
    m_out = os.path.join(directory,map_out)

    if( os.path.exists(m_out) == False ):
        os.mkdir(m_out)

    printProgressBar(0,len(images)-1)
    for idi,im in enumerate(images):
        annos = []
        for idm,m in enumerate(maps_dir):
            if( keep_annotations[idi,idm] ):
                annos += [imread(os.path.join(m,'%s_classimg_nonconvex.png'%im.replace('.jpg','')))]
        annos = np.array(annos)
        
        if( weights == 'majority'):
            ws = np.ones(keep_annotations[idi].sum())
        else:
            ws = weights[keep_annotations[idi]]
        ws /= ws.sum()
        votes = np.zeros((annos.shape[1],annos.shape[2],6))
        for v in range(6):
            for idr,w in enumerate(ws):
                mask = annos[idr,:,:]==v
                votes[mask,v] += w*mask[mask]
                
        final_v = np.argmax(votes, axis=2)
        
        imsave(os.path.join(m_out, '%s.png'%im.replace('.jpg','')), final_v)
        printProgressBar(idi,len(images)-1)

'''
Generate STAPLE, MV, WV & all leave-one-out versions
'''
def generate_all_annotations(directory):
    remap(directory)

    with open(os.path.join(directory, 'final_list_to_remove.json'), 'r') as fp:
        to_remove_by_core = json.load(fp)

    train_dir = os.path.join(directory, 'train')

    # Get all the cores from the training set
    f_train = os.listdir(train_dir)
    cores = [f.split('.')[0] for f in f_train]

    maps = ['Remaps%d_T'%i for i in range(1,7)]

    keep_annotations = np.zeros((len(cores), len(maps))).astype('bool')
    for idi,im in enumerate(cores):
        for idm,m in enumerate(maps):
            if( os.path.isfile(os.path.join(os.path.join(directory,m),'%s_classimg_nonconvex.png'%im.replace('.jpg',''))) ):
                if( '%s.jpg'%im in to_remove_by_core and m in to_remove_by_core['%s.jpg'%im] ):
                    continue
                keep_annotations[idi,idm] = 1

    weights = np.array([0.401,0.413,0.612,0.583,0.591,0.532])

    # Generate main STAPLE:
    generate_STAPLE(directory, maps, keep_annotations, 'Remaps_STAPLE_no')
    generate_vote(directory, maps, keep_annotations, 'majority', 'Majority_Voting')
    generate_vote(directory, maps, keep_annotations, weights, 'Weighted_Voting')

    # Create "LoO" directories if they don't exist yet
    for i in range(1,7):
        maps = ['Remaps%d_T'%j for j in range(1,7) if j!=i]
        keep = np.array([i!=j for j in range(1,7)])
        
        m_out = os.path.join(directory, 'LoO_STAPLE_%d'%i)
        if( os.path.exists(m_out) == False ):
            os.mkdir(m_out)
        generate_STAPLE(directory, maps, keep_annotations[:,keep], m_out)

        m_out = os.path.join(directory, 'LoO_MV_%d'%i)
        if( os.path.exists(m_out) == False ):
            os.mkdir(m_out)
        generate_vote(directory, maps, keep_annotations[:,keep], 'majority', m_out)

        m_out = os.path.join(directory, 'LoO_WV_%d'%i)
        if( os.path.exists(m_out) == False ):
            os.mkdir(m_out)
        generate_vote(directory, maps, keep_annotations[:,keep], weights, m_out)

if __name__ == '__main__':
    import sys

    if( len(sys.argv) == 2 ):
        directory = sys.argv[1]
        generate_all_annotations(directory)