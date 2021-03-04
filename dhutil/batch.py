import numpy as np

'''
On-the-fly data augmentation on a batch
* Horizontal/Vertical symmetry
* Random noise
* Random illumination change
'''
def batch_augmentation(X,Y_seg,Y_det,seed=None):
    if( seed != None ):
        np.random.seed(seed)
    
    params = np.random.random((X.shape[0], 3))
    params[:,0] = np.floor(params[:,0]*4)     # Horizontal & Vertical mirrors
    params[:,1] *= 0.1                        # Random noise max value (X'(i,j) = X(i,j) + random(i,j)*max_noise)
    params[:,2] = (params[:,2]-0.5)*0.1       # Illumination change (X'(i,j) = X(i,j) + illumination)

    X2 = X.copy()
    Y2_seg = Y_seg.copy()
    Y2_det = Y_det.copy()

    # Orientation
    do_vswap = (params[:,0]==1)+(params[:,0]==3)
    do_hswap = (params[:,0]==2)+(params[:,0]==3)
    X2[do_vswap] = X2[do_vswap,::-1,:,:]
    X2[do_hswap] = X2[do_hswap,:,::-1,:]
    Y2_seg[do_vswap] = Y2_seg[do_vswap,::-1,:,:]
    Y2_seg[do_hswap] = Y2_seg[do_hswap,:,::-1,:]

    # Noise & illumination
    for i in range(X2.shape[0]):
        X2[i] += np.random.random(X2[i].shape)*params[i,1]-params[i,1]/2+params[i,2]

    return X2,Y2_seg,Y2_det

'''
On-the-fly data augmentation on a sample
* Horizontal/Vertical symmetry
* Random noise
* Random illumination change

Produces N augmented samples from one original
'''
def sample_augmentation(X,N,seed=None):
    if( seed != None ):
        np.random.seed(seed)
    
    params = np.random.random((N, 3))
    # params[:,0] = np.floor(params[:,0]*4)     # Horizontal & Vertical mirrors
    params[:,1] *= 0.1                        # Random noise max value (X'(i,j) = X(i,j) + random(i,j)*max_noise)
    params[:,2] = (params[:,2]-0.5)*0.1       # Illumination change (X'(i,j) = X(i,j) + illumination)

    X2 = np.array([X.copy() for i in range(N)])
    
    # Orientation
    # do_vswap = (params[:,0]==1)+(params[:,0]==3)
    # do_hswap = (params[:,0]==2)+(params[:,0]==3)
    # X2[do_vswap] = X2[do_vswap,::-1,:,:]
    # X2[do_hswap] = X2[do_hswap,:,::-1,:]

    # Noise & illumination
    for i in range(X2.shape[0]):
        X2[i] += np.random.random(X2[i].shape)*params[i,1]-params[i,1]/2+params[i,2]

    return X2