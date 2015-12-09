import sys, os
import numpy as np
from ..mvpa import searchlight
from . import mvpa_nodes
from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.fx import mean_sample
from mvpa2.clfs.gnb import GNB
from mvpa2.misc.neighborhood import CachedQueryEngine
from mvpa2.generators.partition import NFoldPartitioner
import joblib


preproc_dir = '/home/bpinsard/data/analysis/core_sleep'
dataset_subdir = 'dataset_noisecorr'
#dataset_subdir = 'dataset_smoothed'
#dataset_subdir = 'dataset_raw'

proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
output_subdir = 'searchlight_new'
#output_subdir = 'searchlight_smooth'
#output_subdir = 'searchlight_raw'
compression= 'gzip'

subject_ids = [1,11,23,22,63,50,79,54,107,128,162,102]
group_Int = [1,23,63,79,107,128]
#subject_ids=subject_ids[5:]
ulabels = ['CoReTSeq','CoReIntSeq','mvpa_CoReOtherSeq1','mvpa_CoReOtherSeq2','rest']
#ulabels = ulabels[1:]


def subject_searchlight_new(sid):
    print('______________   CoRe %03d   ___________'%sid)
    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'glm_ds_%d.h5'%sid))
    ds_all = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))
    ds = ds_all[ds_all.sa.match(dict(scan_name=mvpa_nodes.training_scans+mvpa_nodes.testing_scans),strict=False)]
    del ds_all
    targets_num(ds, ulabels)
    targets_num(ds_glm, ulabels)

    mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'd3_mvpa' in n]
    if len(mvpa_scan_names)==0:
        mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'mvpa' in n]
    ds_mvpa = ds[dict(scan_name=mvpa_scan_names)]
    ds_mvpa.fa = ds.fa # cheat CachedQueryEngine hashing

    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
    ds_glm_mvpa.fa = ds.fa # cheat CachedQueryEngine hashing
    
    svqe = searchlight.SurfVoxQueryEngine(max_feat=128,vox_sl_radius=3.2,surf_sl_radius=20)
    svqe_cached = CachedQueryEngine(svqe)

    gnb = GNB(space='targets_num')
    spltr = Splitter(attr='balanced_partitions', attr_values=[1,2])

    slght_loso = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_loso_cv,
        svqe_cached,
        splitter=spltr,
        errorfx=None,
        pass_attr=ds.sa.keys()+ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        postproc=mvpa_nodes.scan_blocks_confmat)

    slght_loco = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_loco_cv,
        svqe_cached,
        splitter=spltr,
#        reuse_neighbors=True,
        errorfx=None,
        pass_attr=ds.sa.keys()+ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        postproc=mvpa_nodes.scan_blocks_confmat)

    slght_d123_train_test = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_d123_train_test,
        svqe_cached,
        splitter=spltr,
        errorfx=None,
        pass_attr=ds.sa.keys()+ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        postproc=mvpa_nodes.scan_blocks_confmat)

    #slmap_d123_train_test = slght_d123_train_test(ds)
    ds_exec = ds[ds.sa.subtargets=='exec']
    ds_exec.fa = ds.fa # cheat CachedQueryEngine hashing
    slmap_crossday_exec = slght_d123_train_test(ds_exec)
    slmap_crossday_exec.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_crossday_exec_confusion.h5'%sid),
                             compression=compression)
    del ds_exec, slmap_crossday_exec

    ds_mvpa = ds[dict(scan_name=mvpa_scan_names)]
    ds_mvpa.fa = ds.fa # cheat CachedQueryEngine hashing    
    del ds
    mvpa_slght_subset = {
#        'all': slice(None),
        'instr': dict(subtargets=['instr']),
        'exec': dict(subtargets=['exec'])}
    for subset_name, subset in mvpa_slght_subset.items():
        ds_subset = ds_mvpa[subset]
        ds_subset.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing
        slmap_loco = slght_loco(ds_subset)
        slmap_loco.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_loco_confusion.h5'%(sid,subset_name)),
                        compression=compression)
        del slmap_loco
        slmap_loso = slght_loso(ds_subset)
        slmap_loso.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_loso_confusion.h5'%(sid,subset_name)),
                        compression=compression)
        del slmap_loso

        ds_glm_subset = ds_glm_mvpa[subset]
        ds_glm_subset.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing
        slmap_glm_loco = slght_loco(ds_glm_subset)
        slmap_glm_loco.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_glm_loco_confusion.h5'%(sid,subset_name)),
                            compression=compression)
        del slmap_glm_loco
        slmap_glm_loso = slght_loso(ds_glm_subset)
        slmap_glm_loso.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_glm_loso_confusion.h5'%(sid,subset_name)),
                            compression=compression)
        del slmap_glm_loso

    slght_loco_delay = searchlight.GNBSearchlightOpt(
        gnb,
        NFoldPartitioner(attr='chunks'),
        svqe_cached,
        splitter=spltr,
        errorfx=None,
        pass_attr=ds_mvpa.sa.keys()+ds_mvpa.fa.keys()+ds_mvpa.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        postproc=mvpa_nodes.confmat_all)

    start = -2
    end = 22
    delays = range(start, end)
    delay_slmaps_confusion = []
    blocks_tr = np.where(np.diff(ds_mvpa.sa.blocks_idx_no_delay)>0)[0]+1
    for d in delays:
        print('######## computing searchlight for delay %d #######'%d)
        delay_trs = blocks_tr+d
        for sn in mvpa_scan_names:
            scan_trs = ds_mvpa.sa.scan_name[delay_trs-d]==sn
            scan_mask = np.where(ds_mvpa.sa.scan_name==sn)[0]
            min_tr = scan_mask[0]
            max_tr = scan_mask[-1]
            delay_trs = delay_trs[np.logical_or(np.logical_and(delay_trs >= min_tr,delay_trs <= max_tr),~scan_trs)]
        delay_ds = ds_mvpa[delay_trs]
        delay_ds.targets = ds_mvpa.sa.targets_no_delay[delay_trs-d]
        delay_ds.sa.targets_num = ds_mvpa.sa.targets_num[delay_trs-d+3]
        delay_ds.chunks = np.arange(delay_ds.nsamples)
        delay_ds.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing
        slmap_confmat = slght_loco_delay(delay_ds)
        delay_slmaps_confusion.append(slmap_confmat)
        del delay_ds
    slmap_delay = vstack(delay_slmaps_confusion)
    slmap_delay.fa = ds_mvpa.fa
    slmap_delay.sa['delays']=delays
    slmap_delay.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_delay_confusion.h5'%sid),
                     compression=compression)
    del delay_slmaps_confusion, slmap_delay

    

def targets_num(ds, utargets, targets_src='targets', targets_num_attr='targets_num'):
    targets2idx = dict([(t,i) for i,t in enumerate(utargets)])
    ds.sa['targets_num'] = np.asarray([targets2idx[t] for t in ds.sa[targets_src].value], dtype=np.uint8)

def all_searchlight():
    joblib.Parallel(n_jobs=2)([joblib.delayed(subject_searchlight_new)(sid) for sid in subject_ids])


subjects_4targ = ['S01_ED_pilot','S349_AL_pilot','S341_WC_pilot','S02_PB_pilot','S03_MC_pilot']
ntargets = 4

def searchlight_stats():
    results = dict([(cls,[]) for cls in [
                'slmap_glm_exec_d2_mvpa', 'slmap_glm_instr_d2_mvpa',
                'slmap_glm_exec_all', 'slmap_glm_instr_all',
                'slmap_tr_instr_d2_mvpa', 'slmap_tr_exec_d2_mvpa',
                'slmap_tr_instr_all', 'slmap_tr_exec_all',
                'slmap_tr_loso_instr_all', 'slmap_tr_loso_exec_all',
                'slmap_glm_loso_instr_all', 'slmap_glm_loso_exec_all']])
    for subj in subjects_4targ:
        print(subj)
        slmaps = Dataset.from_hdf5(os.path.join(proc_dir,output_subdir, '%s_accuracy_slmaps.h5'%subj))
        for cls in results.keys():
            idx = np.asarray([i for i,sl in enumerate(slmaps.sa.slmap) if cls in sl])
            print(cls, idx)
            if len(idx)<1:
                continue
            zscored = (slmaps.samples[idx]-1./ntargets)/slmaps.samples[idx].std(1)[:,np.newaxis]
            results[cls].append(zscored.mean(0))
    return results


bas_labels = {
    'BA1':10,
    'BA2b':11,
    'BA3a':12,
    'BA3b':13,
    'BA4a':4,
    'BA4p':5,
    'BA44':2,
    'BA45':3}

fs_clt = np.loadtxt('/home/bpinsard/softs/freesurfer/FreeSurferColorLUT.txt',np.str)
rois = np.hstack([np.asarray([46,29,70,69,28,4,3,7,8])+a for a in [11100,12100]]+[53,17,10,49,51,12,8,47,11,50])
aparc_labels = dict([(fs_clt[fs_clt[:,0]==str(r)][0,1],r) for r in rois])


from mvpa2.measures.base import CrossValidation
from mvpa2.generators.splitters import Splitter

def test_clf(clf, ds, partitioner, rois_fa, roi_labels):

    spltr = Splitter(attr='partitions',attr_values=[1,2])
    cvte = CrossValidation(clf, partitioner, splitter=spltr, enable_ca=['stats'])

    stats = dict()

    for roi_name,roi in roi_labels.items():
        print(np.count_nonzero(ds.fa[rois_fa].value==roi))
        res = cvte(ds[:,ds.fa[rois_fa].value==roi])
        stats[roi_name] = cvte.ca.stats
        print(stats[roi_name].stats['ACC'])
    return stats



def stats_all_subjects(subjects, clf, partitioner, rois_fa, roi_labels):
    stats = dict()
    for subj in subjects:
        print(subj)
        ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_%s'%subj, dataset_subdir, 'ds_%s.h5'%subj))
        mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'd2_mvpa' in n]
        if len(mvpa_scan_names)==0:
            mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'mvpa' in n]
        mvpa_tr_scans_mask = np.logical_and(
            reduce(
                lambda mask,msn: np.logical_or(mask,ds.sa.scan_name==msn), 
                mvpa_scan_names, np.ones(ds.nsamples,dtype=np.bool)),
            ds.sa.targets!='rest')
        cv_ds = ds[mvpa_tr_scans_mask]
        stats[subj]=test_clf(clf,cv_ds,partitioner,rois_fa,roi_labels)
#        del ds, cv_ds
    return stats



def all_rois_analysis():
    
    for subj in subjects:
        print('______________   %s   ___________'%subj)
        ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_%s'%subj, dataset_subdir, 'glm_ds_%s.h5'%subj))
        ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_%s'%subj, dataset_subdir, 'ds_%s.h5'%subj))
             


def confusion2acc(ds):
    return Dataset(
        ds.samples[:,:,np.eye(ds.samples.shape[-1],dtype=np.bool)].sum(-1).astype(np.float)/\
        ds.samples[:,0].sum(-1).sum(-1)[:,np.newaxis],
        sa=ds.sa,
        fa=ds.fa,
        a=ds.a)
