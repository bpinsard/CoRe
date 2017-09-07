import sys, os, glob
import numpy as np
import scipy.stats, scipy.ndimage.measurements, scipy.sparse
from ..mvpa import searchlight
from ..mvpa import dataset as mvpa_ds
from . import mvpa_nodes
from mvpa2.datasets import Dataset, vstack
from mvpa2.misc.errorfx import mean_mismatch_error, mean_match_accuracy
from mvpa2.mappers.fx import mean_sample
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.fx import BinomialProportionCI, mean_sample, mean_group_sample
from mvpa2.measures.base import RepeatedMeasure
from mvpa2.clfs.gnb import GNB
from mvpa2.misc.neighborhood import CachedQueryEngine
from mvpa2.measures.rsa import CrossNobisSearchlight
from mvpa2.generators.partition import NFoldPartitioner, FactorialPartitioner, CustomPartitioner
from mvpa2.algorithms.group_clusterthr import (GroupClusterThreshold, Counter, 
                                               get_cluster_sizes, _transform_to_pvals, _clusterize_custom_neighborhood)
import statsmodels.stats.multitest as smm
from mvpa2 import debug
import joblib
import __builtin__

preproc_dir = '/home/bpinsard/data/analysis/core_sleep'
dataset_subdir = 'dataset_mvpa_moco_bc_hptf'

proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
output_subdir = 'searchlight_cnbis_dtd'
compression= 'gzip'

subject_ids = [1, 11, 23, 22, 63, 50, 79, 54, 107, 128, 162, 102, 82, 155, 100, 94, 87, 192, 195, 220, 223, 235, 268, 267,237,296]
group_Int = [1,23,63,79,82,87,100,107,128,192,195,220,223,235,268,267,237,296]
ulabels = ['CoReTSeq','CoReIntSeq','mvpa_CoReOtherSeq1','mvpa_CoReOtherSeq2','rest']

seq_groups = {
    'mvpa_new_seqs' : ulabels[2:4],
    'tseq_intseq' : ulabels[:2],
    'all_seqs': ulabels[:4]
}
block_phases = [
    'instr',
    'exec'
]
scan_groups = dict(
    mvpa1=['d3_mvpa1'],
    mvpa2=['d3_mvpa2'],
    mvpa_all=['d3_mvpa1','d3_mvpa2']
)

def mvcs_reactivation(sid, mvcs_win_len=100):

    ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))    
    ds_train = ds[dict(scan_name=['d1_training_TSeq'])]
    ds_boost = ds[dict(scan_name=['d1_retest_TSeq_1block'])]
    ds_rest_pre = ds[dict(scan_name=['d1_resting1'])]
    ds_rest_post = ds[dict(scan_name=['d1_resting2','d1_resting3'])]
    del ds
    
    for roi in rois:
        mvcs_training = np.corrcoef(ds_training[:])

def nlsa_training(sid):

    ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))    
    ds_d1 = ds[dict(scan_name=['d1_resting1','d1_training_TSeq','d1_resting2','d1_resting3','d1_retest_TSeq_1block'])]
    del ds

    
