'''
Script to compute the core-level grades of the Gleason2019 dataset (using all the maps generated in dataset.py), and 
compute all the head-to-head kappas between those maps.

Requires skimage and sklearn

For the core-level grades, 4 versions are computed:
* Gleason score (sum of two most prominent patterns) using simple pixel count and half-area rule (only count pattern 2 if it occupies > 50% area of pattern 1)
* Epstein score (<=3+3 -> 1, 3+4 -> 2, 4+3 -> 3, 4+4 -> 4, >4+4 -> 5) using simple pixel count and half-area rule

First, the number of pixels for each Gleason grade is computed for every annotation map. This is the part that takes up the longest, 
so once it's computed & saved to a file, that file may be reused to compute the scores.

Usage:
    python grades.py /path/to/main/gleason2019/directory [/path/to/existing/pxPerGrades_all.npy]
'''

from sklearn.metrics import cohen_kappa_score
import os
from skimage.io import imread
import numpy as np
import json
from dhutil.tools import printProgressBar

def compute_pxPerGrades(directory, maps, saveAs=None):
    train_dir = os.path.join(directory, 'train')
    images = [f for f in os.listdir(train_dir)]
    
    maps_dir = [os.path.join(directory, m) for m in maps]
    
    pxPerGrades_all = np.zeros((len(images),len(maps),6))
    for idm,m in enumerate(maps_dir):
        printProgressBar(0,len(images))
        for idi,im in enumerate(images):
            fname = os.path.join(m,'%s.png'%im.replace('.jpg',''))
            if(os.path.isfile(fname) == False):
                fname = os.path.join(m,'%s_classimg_nonconvex.png'%im.replace('.jpg',''))
            if(os.path.isfile(fname) == False):
                continue
            anno = imread(fname)
            for v in range(6):
                pxPerGrades_all[idi,idm,v] = (anno==v).sum()
            printProgressBar(idi+1,len(images))

    if saveAs != None:
        np.save(saveAs, pxPerGrades_all)

    return pxPerGrades_all

'''
Simple pixel count, Gleason groups
'''
def get_ssp(pxPerGrades, saveAs=None):
    ssp = np.zeros((pxPerGrades.shape[0], pxPerGrades.shape[1])).astype('int')
    printProgressBar(0,pxPerGrades.shape[0])
    for idi in range(pxPerGrades.shape[0]):
        for idm in range(pxPerGrades.shape[1]):
            if(pxPerGrades[idi,idm].sum() > 0):
                h = pxPerGrades[idi,idm][1:]
                if( h.sum()==0 ):
                    ssp[idi,idm] = 0
                else:            
                    s = np.argsort(h)[::-1]
                    if( h[s][1] > 0 ): # 2 types of glands present:
                        ssp[idi,idm] = (s[:2]+1).sum()
                    else: # only 1 type
                        ssp[idi,idm] = (s[0]+1)*2
        printProgressBar(idi+1,pxPerGrades.shape[0])

    if( saveAs != None ):
        np.save(saveAs, ssp)
    return ssp

'''
Simple pixel count, Epstein groups
'''
def get_ssp_epstein(pxPerGrades, saveAs=None):
    sspg = np.ones((pxPerGrades.shape[0], pxPerGrades.shape[1])).astype('int')
    printProgressBar(0,pxPerGrades.shape[0])
    for idi in range(pxPerGrades.shape[0]):
        for idm in range(pxPerGrades.shape[1]):
            if(pxPerGrades[idi,idm].sum() > 0):
                h = pxPerGrades[idi,idm][1:]
                if( h.sum()==0 ):
                    sspg[idi,idm] = 0
                else:            
                    s = np.argsort(h)[::-1]
                    if( h[s][1] > 0 ): # >=2 types of glands present:
                        p1 = s[0]+1
                        p2 = s[1]+1
                        if(p1+p2 <= 6): sspg[idi,idm] = 1
                        elif(p1==3 and p2==4): sspg[idi,idm] = 2
                        elif(p1==4 and p2==3): sspg[idi,idm] = 3
                        elif(p1==5 or p2==5): sspg[idi,idm] = 5
                        else: print("error!",idi,idm)
                    else: # only 1 type
                        p1 = s[0]+1
                        if( p1 <= 3 ): sspg[idi,idm] = 1
                        elif( p1 == 4 ): sspg[idi,idm] = 4
                        elif( p1 == 5 ): sspg[idi,idm] = 5
                        else: print("error!",idi,idm)
        printProgressBar(idi+1,pxPerGrades.shape[0])

    if( saveAs != None ):
        np.save(saveAs, sspg)
    return sspg

'''
Half-area rule, Epstein groups
'''
def get_msp_epstein(pxPerGrades, saveAs=None):
    mspg = np.ones((pxPerGrades.shape[0], pxPerGrades.shape[1])).astype('int')
    printProgressBar(0,pxPerGrades.shape[0])
    for idi in range(pxPerGrades.shape[0]):
        for idm in range(pxPerGrades.shape[1]):
            if(pxPerGrades[idi,idm].sum() > 0):
                h = pxPerGrades[idi,idm][1:]
                if( h.sum()==0 ):
                    mspg[idi,idm] = 0
                else:            
                    s = np.argsort(h)[::-1]
                    if( h[s][1] > 0 and h[s][1] > h[s][0]/2 ): # 2 types of glands present:
                        p1 = s[0]+1
                        p2 = s[1]+1
                        if(p1+p2 <= 6): mspg[idi,idm] = 1
                        elif(p1==3 and p2==4): mspg[idi,idm] = 2
                        elif(p1==4 and p2==3): mspg[idi,idm] = 3
                        elif(p1==5 or p2==5): mspg[idi,idm] = 5
                    else: # only 1 type
                        p1 = s[0]+1
                        if( p1 <= 3 ): mspg[idi,idm] = 1
                        elif( p1 == 4 ): mspg[idi,idm] = 4
                        elif( p1 == 5 ): mspg[idi,idm] = 5
                        else: print("error!",idi,idm)
        printProgressBar(idi+1,pxPerGrades.shape[0])
    
    if( saveAs != None ):
        np.save(saveAs, mspg)
    return mspg

'''
Half-area rule, Gleason groups
'''
def get_msp(pxPerGrades, saveAs=None):
    msp = np.zeros((pxPerGrades.shape[0], pxPerGrades.shape[1])).astype('int')
    printProgressBar(0,pxPerGrades.shape[0])
    for idi in range(pxPerGrades.shape[0]):
        for idm in range(pxPerGrades.shape[1]):
            if(pxPerGrades[idi,idm].sum() > 0):
                h = pxPerGrades[idi,idm][1:]
                if( h.sum()==0 ):
                    msp[idi,idm] = 0
                else:            
                    s = np.argsort(h)[::-1]
                    if( h[s][1] > 0 and h[s][1] > h[s][0]/2 ): # 2 types of glands present:
                        msp[idi,idm] = (s[:2]+1).sum()
                    else: # only 1 type
                        msp[idi,idm] = (s[0]+1)*2
        printProgressBar(idi+1,pxPerGrades.shape[0])
    if( saveAs != None ):
        np.save(os.path.join(directory,saveAs), msp)
    return msp

def get_kappas(grades, annotations, w=None):
    kappas = np.zeros((annotations.shape[1],annotations.shape[1]))
    kappas_n = np.zeros((annotations.shape[1],annotations.shape[1]))
    for i in range(annotations.shape[1]):
        for j in range(i+1,annotations.shape[1]):
            mask = (annotations[:,i]*annotations[:,j])>0
            kappas_n[i,j] = mask.sum()
            kappas[i,j] = cohen_kappa_score(grades[mask,i], grades[mask,j], weights=w)
    kappas = kappas + kappas.T
    kappas[np.eye(kappas.shape[0])==1] = 1
    kappas_n += kappas_n.T
    return kappas,kappas_n

def compute_coreGrades(directory, maps, pxPerGrades_all, saveAs=None):
    print("Computing SSP")
    ssp = get_ssp(pxPerGrades_all)
    print("Computing MSP")
    msp = get_msp(pxPerGrades_all)
    print("Computing SSPG")
    sspg = get_ssp_epstein(pxPerGrades_all)
    print("Computing MSPG")
    mspg = get_msp_epstein(pxPerGrades_all)

    if( saveAs != None ):
        print("Saving...")
        np.save(os.path.join(directory, 'ssp_%s'%saveAs), ssp)
        np.save(os.path.join(directory, 'msp_%s'%saveAs), msp)
        np.save(os.path.join(directory, 'sspg_%s'%saveAs), sspg)
        np.save(os.path.join(directory, 'mspg_%s'%saveAs), mspg)

    return ssp,msp,sspg,mspg

def compute_all_kappas(directory, pxPerGradeFile=None):
    maps_experts = ['Remaps%d_T'%i for i in range(1,7)]
    maps_STAPLE = ['Remaps_STAPLE_no'] + ['LoO_STAPLE_%d'%i for i in range(1,7)]
    maps_MV = ['Majority_Voting'] + ['LoO_MV_%d'%i for i in range(1,7)]
    maps_WV = ['Weighted_Voting'] + ['LoO_WV_%d'%i for i in range(1,7)]

    all_maps = maps_experts + maps_STAPLE + maps_MV + maps_WV
    if( pxPerGradeFile == None ):
        pxPerGrades_all = compute_pxPerGrades(directory, all_maps, 'pxPerGrades_all.npy')
    else:
        pxPerGrades_all = np.load(pxPerGradeFile)

    ssp, msp, sspg, mspg = compute_coreGrades(directory, all_maps, pxPerGrades_all, 'scores.npy')

    train_dir = os.path.join(directory, 'train')
    images = [f for f in os.listdir(train_dir)]
    with open(os.path.join(directory, 'final_list_to_remove.json'), 'r') as fp:
        to_remove_by_core = json.load(fp)

    has_annotation_ = np.zeros((len(images), len(maps_experts))).astype('bool')
    for idi,im in enumerate(images):
        for idm,m in enumerate(maps_experts):
            if( os.path.isfile(os.path.join(os.path.join(directory,m),'%s_classimg_nonconvex.png'%im.replace('.jpg',''))) ):
                if( im in to_remove_by_core and m in to_remove_by_core[im] ):
                    continue
                has_annotation_[idi,idm] = 1

    print("Kept annotations per expert:")
    print(has_annotation_.sum(axis=0))

    tables_dir = os.path.join(directory, "result_tables")
    if( os.path.isdir(tables_dir) == False ):
        os.mkdir(tables_dir)

    print(len(all_maps))
    keep_annotations = np.zeros((len(images), len(all_maps)))
    keep_annotations[:,:6] = has_annotation_
    keep_annotations[:,6] = 1 #ST
    keep_annotations[:,7:13] = has_annotation_ #LoO-ST
    keep_annotations[:,13] = 1 #MV
    keep_annotations[:,14:20] = has_annotation_ #LoO-MV
    keep_annotations[:,20] = 1 #WV
    keep_annotations[:,21:] = has_annotation_ #LoO-WV

    kappas, kappas_n = get_kappas(ssp, keep_annotations)
    np.save(os.path.join(tables_dir, 'ssp_kappa.npy'), kappas)
    np.save(os.path.join(tables_dir, 'ssp_kappa_n.npy'), kappas_n)

    kappas, kappas_n = get_kappas(sspg, keep_annotations)
    np.save(os.path.join(tables_dir, 'sspg_kappa.npy'), kappas)
    np.save(os.path.join(tables_dir, 'sspg_kappa_n.npy'), kappas_n)

    kappas, kappas_n = get_kappas(msp, keep_annotations)
    np.save(os.path.join(tables_dir, 'msp_kappa.npy'), kappas)
    np.save(os.path.join(tables_dir, 'msp_kappa_n.npy'), kappas_n)

    kappas, kappas_n = get_kappas(mspg, keep_annotations)
    np.save(os.path.join(tables_dir, 'mspg_kappa.npy'), kappas)
    np.save(os.path.join(tables_dir, 'mspg_kappa_n.npy'), kappas_n)

if __name__ == '__main__':
    import sys

    if( len(sys.argv) >= 2 ):
        directory = sys.argv[1]
        if( len(sys.argv) == 3 ):
            pxPerGradeFile = sys.argv[2]
        else:
            pxPerGradeFile = None
        compute_all_kappas(directory,pxPerGradeFile)
    
