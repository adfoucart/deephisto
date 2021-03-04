'''
Script to produce MDS visualisation of the kappas computed by grades.py for the 6 experts + 3 main consensus (ST, MV, WV)

Requires sklearn

Usage:
    python mds.py /path/to/main/gleason2019/directory [grade] [seed]

    [grade] : 
        ssp > Simple pixel count, Gleason score
        sspg > Simple pixel count, Epstein score
        msp > Half-area rule, Gleason score
        mspg > Half-area rule, Epstein score

    [seed] : integer, random seed for the MDS algorithm for reproducable results.
'''

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import numpy as np
import os

def compute_and_plot_mds(directory, grade='mspg', seed=None):
    kappas = np.load(os.path.join(directory, 'result_tables/%s_kappa.npy'%grade))
    kappas_n = np.load(os.path.join(directory, 'result_tables/%s_kappa_n.npy'%grade))
    
    mds = MDS(dissimilarity='precomputed', random_state=seed)
    
    diss = 1-kappas
    keep = np.array([True for i in range(6)]+[True]+[False for i in range(6)]+[True]+[False for i in range(6)]+[True]+[False for i in range(6)])
    labels = ['1', '2', '3', '4', '5', '6', 'ST', 'MV', 'WV']

    if( len(keep) > 0 ):
        diss = diss[keep,:][:,keep]
    if( labels == None ):
        labels = ['%d'%i for i in range(diss.shape[0])]
        
    coords = mds.fit_transform(diss)
    dcoords = np.zeros_like(diss)
    for i in range(diss.shape[0]):
        for j in range(diss.shape[1]):
            dcoords[i,j] = np.abs(np.sqrt(((coords[i]-coords[j])**2).sum()))
    err = np.abs(dcoords-diss).sum(axis=0)/5 #np.sqrt(((dcoords-diss)**2).sum(axis=0)/4)

    fig,ax = plt.subplots()
    for i in range(diss.shape[0]):
        plt.plot(coords[i,0],coords[i,1],'o')
        circle = plt.Circle(coords[i,:], err[i], fill=False)
        ax.add_artist(circle)
        plt.text(coords[i,0]+0.01,coords[i,1]+0.01, labels[i])
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    import sys
    if( len(sys.argv) >= 2 ):
        directory = sys.argv[1]
        if( len(sys.argv) >= 3 ):
            grade = sys.argv[2]
        else:
            grade = 'mspg'
        if( len(sys.argv) == 4 ):
            seed = sys.argv[3]
        else:
            seed = None
    
        compute_and_plot_mds(directory, grade)