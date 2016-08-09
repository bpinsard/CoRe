import numpy as np


def neural_distance_mat(n,m, remove_center=True):
    coords = np.mgrid[:n,:m].reshape(2,-1).T
    if remove_center:
        coords = coords[np.arange(len(coords))!=len(coords)/2]
    dists = np.sqrt(((coords[np.newaxis]-coords[:,np.newaxis])**2).sum(-1))
    return dists #-np.diag(np.ones(len(dists)))
    
