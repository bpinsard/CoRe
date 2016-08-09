import numpy as np
from mvpa2.measures.base import CrossValidation

from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.clfs.gda import LDA
from mvpa2.clfs.knn import kNN
from mvpa2.clfs.gnb import GNB

from mvpa2.generators.partition import FactorialPartitioner, NFoldPartitioner

from mvpa2.datasets.base import Dataset
from mvpa2.misc.errorfx import mean_match_accuracy

n_slght = 1000
n_samples = [16,32,64]
slght_size = [16,32,64]

prtnr_2fold_factpart = FactorialPartitioner(
    NFoldPartitioner(cvtype=2,attr='chunks'),
    attr='targets',
    selection_strategy='equidistant',
    count=32)

clfs = [GNB, LinearCSVMC, kNN]
cvtes = [
    CrossValidation(
        clf(),
        prtnr_2fold_factpart,
        errorfx=mean_match_accuracy,
    ) for clf in clfs]

ds_rand = Dataset.from_wizard(
    np.random.normal(size=(64,10000)),
    targets=[0,1]*32,
    chunks=np.arange(64))

accuracies = np.empty((n_slght,len(clfs),len(n_samples),len(slght_size)))
accuracies.fill(np.nan)

for ni, n_samp in enumerate(n_samples):
    ds_rand_subset = ds_rand[:n_samp]
    for sli in range(n_slght):
        print sli
        for sl_szi, sl_sz in enumerate(slght_size):        
            rand_idx = np.random.randint(0, ds_rand.nfeatures, size=sl_sz)
            ds_rand_slght = ds_rand_subset[:,rand_idx]
            for cvi,cvte in enumerate(cvtes):
                accuracies[sli,cvi,ni,sl_szi] = cvte(ds_rand_slght).samples.mean()
        
    
    

