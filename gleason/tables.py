'''
Script to produce the results for expert-v-expert, expert-v-consensus and consensus-v-consensus from the computed kappas (from grades.py)

The script assumes that the kappas have been computed and are stored in the result_tables subdirectory of the main gleason2019 directory.

Usage:
    python tables.py /path/to/main/gleason2019/directory
'''

import numpy as np
import os

def print_all_tables(directory):
    directory = os.path.join(directory, "result_tables")
    grading = ['ssp', 'msp', 'sspg', 'mspg']

    # 1 - Expert v Expert
    print("Expert v Expert:")
    for g in grading:
        print(g)
        all_kappas = np.load(os.path.join(directory, '%s_kappa.npy'%g))
        all_kappas_n = np.load(os.path.join(directory, '%s_kappa_n.npy'%g))
        
        kappas = all_kappas[:6,:6]
        kappas_n = all_kappas_n[:6,:6]
        kappas2 = kappas*(1-np.eye(kappas.shape[0]))
        print("Expert\tAvg\tMin\tMax")
        for i in range(6):
            avg = ((kappas[i,:]*kappas_n[i,:]).sum()+(kappas[:,i]*kappas_n[:,i]).sum())/(kappas_n[i,:].sum()+kappas_n[:,i].sum())
            print("%d\t%.3f\t%.3f\t%.3f"%(i+1, avg, kappas[i,:].min(), kappas2[i,:].max()))
        print("Avg: %.3f"%((kappas*kappas_n).sum()/kappas_n.sum()))

    # 2 - Expert v Consensus
    print("Expert v Consensus")
    for g in grading:
        print(g)
        all_kappas = np.load(os.path.join(directory, '%s_kappa.npy'%g))
        all_kappas_n = np.load(os.path.join(directory, '%s_kappa_n.npy'%g))

        print("Expert\tST\tLoO-ST\tMV\tLoO-MV\tWV\tLoO-WV")
        for i in range(1,7):
            # Expert, ST, LoO-ST, MV, LoO-MV, WV, LoO-WV
            keep = np.array([j==i for j in range(1,7)]+[True]+[j==i for j in range(1,7)]+[True]+[j==i for j in range(1,7)]+[True]+[j==i for j in range(1,7)])
            
            kappas = all_kappas[i-1,keep]
            kappas_n = all_kappas_n[i-1,keep]
            print("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"%(i,kappas[1],kappas[2],kappas[3],kappas[4],kappas[5],kappas[6]))

        # Consensus v Consensus
        print("\tST\tMV\tWV")
        keep = np.array([False for i in range(6)]+[True]+[False for i in range(6)]+[True]+[False for i in range(6)]+[True]+[False for i in range(6)])
        kappas = all_kappas[keep,:][:,keep]
        kappas_n = all_kappas_n[keep,:][:,keep]
        print("ST\t%.3f\t%.3f\t%.3f"%(kappas[0,0],kappas[0,1],kappas[0,2]))
        print("MV\t%.3f\t%.3f\t%.3f"%(kappas[1,0],kappas[1,1],kappas[1,2]))
        print("WV\t%.3f\t%.3f\t%.3f"%(kappas[2,0],kappas[2,1],kappas[2,2]))

if __name__ == '__main__':
    import sys

    if( len(sys.argv) == 2 ):
        directory = sys.argv[1]
        print_all_tables(directory)