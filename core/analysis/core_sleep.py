import sys, os, glob
import numpy as np
from ..mvpa import searchlight
from . import mvpa_nodes
from mvpa2.datasets import Dataset, vstack
from mvpa2.misc.errorfx import mean_mismatch_error, mean_match_accuracy
from mvpa2.mappers.fx import mean_sample
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.fx import BinomialProportionCI
from mvpa2.measures.base import RepeatedMeasure
from mvpa2.clfs.gnb import GNB
from mvpa2.misc.neighborhood import CachedQueryEngine
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.algorithms.group_clusterthr import GroupClusterThreshold
from mvpa2 import debug
import joblib


preproc_dir = '/home/bpinsard/data/analysis/core_sleep'
#dataset_subdir = 'dataset_noisecorr'
#dataset_subdir = 'dataset_smoothed'
dataset_subdir = 'dataset_mvpa_wd_interp_hrf_gam1'

proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
output_subdir = 'searchlight_wd_interp_hrf_gam1'
compression= 'gzip'

subject_ids = [1, 11, 23, 22, 63, 50, 79, 54, 107, 128, 162, 102, 82, 155, 100, 94, 87, 192, 195, 220, 223, 235, 268, 267,237,296]
#subject_ids = subject_ids[:-1]
group_Int = [1,23,63,79,82,87,100,107,128,192,195,220,223,235,268,267,237,296]
ulabels = ['CoReTSeq','CoReIntSeq','mvpa_CoReOtherSeq1','mvpa_CoReOtherSeq2','rest']
#ulabels = ulabels[1:]

seq_groups = {
    'mvpa_new_seqs' : ulabels[2:4],
    'tseq_intseq' : ulabels[:2],
#    'all_seqs': ulabels[:4]
}
block_phases = [
    #'instr',
    'exec'
]
scan_groups = dict(
    mvpa1=['d3_mvpa1'],
    mvpa2=['d3_mvpa2'],
#    mvpa_all=['d3_mvpa1','d3_mvpa2']
)


def searchlight_permutation(gnb,svqe,generator,splitter,npermutation=100):
    permutator = AttributePermutator(gnb.space, count=npermutation, limit='scan_name')

    slght = searchlight.GNBSearchlightOpt(
        gnb,
        generator,
        svqe,
        errorfx=mean_match_accuracy,
        splitter=splitter,
        postproc=mean_sample())
    null_slght = RepeatedMeasure(slght, permutator)

    return null_slght


def do_single_slmap_perm(gnb, svqe, spltr, map_name, subset, prtnr, ds, npermutation=100, overwrite=True):
    import gc
    print map_name
    print subset

    out_filename = os.path.join(proc_dir, output_subdir, map_name+'.h5')
    if os.path.exists(out_filename) and not overwrite:
        return
    ds_subset = ds[subset]
    del ds
    gc.collect()

    slght_perm = searchlight_permutation(gnb, svqe, prtnr, spltr, npermutation=npermutation)
    slmap_perms = slght_perm(ds_subset)
    slmap_perms.sa['subject_id'] = [ds_subset.sa.subject_id[0]]*slmap_perms.nsamples
    slmap_perms.sa['group'] = [ds_subset.sa.group[0]]*slmap_perms.nsamples
    slmap_perms.save(
        out_filename,
        compression=compression)
    del slmap_perms, ds_subset, slght_perm
    gc.collect()
    
def subject_searchlight_permutation(sid):

    prtnrs = dict(
        loso = mvpa_nodes.prtnr_loso_cv(),
        loco = mvpa_nodes.prtnr_loco_cv()
    )

    block_phases = ['instr']
#    block_phases = ['exec','instr']

    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'glm_ds_%d.h5'%sid))
    mvpa_scan_names = [n for n in np.unique(ds_glm.sa.scan_name) if 'd3_mvpa' in n]
    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
    del ds_glm
    
    targets_num(ds_glm_mvpa, ulabels)
    ds_glm_mvpa.sa['subject_id'] = [sid]*ds_glm_mvpa.nsamples
    ds_glm_mvpa.sa['group'] = [sid in group_Int]*ds_glm_mvpa.nsamples

    poly_detrend(ds_glm_mvpa, chunks_attr='scan_id', polyord=0)
    zscore(ds_glm_mvpa, chunks_attr='scan_id')

    ds_all = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))
    ds_mvpa = ds_all[dict(scan_name=mvpa_scan_names)]
    ds_mvpa.sa['subject_id'] = [sid]*ds_mvpa.nsamples
    ds_mvpa.sa['group'] = [sid in group_Int]*ds_mvpa.nsamples
    del ds_all

    targets_num(ds_mvpa, ulabels)
    poly_detrend(ds_mvpa, chunks_attr='scan_id', polyord=0)
    zscore(ds_mvpa, chunks_attr='scan_id', param_est=('targets','rest'))

    sample_types = {
#        '': ds_mvpa,
        '_glm':ds_glm_mvpa
    }

    slmaps = [('CoRe_%03d_%s_%s_%s%s_%s_perms'%(sid,scan_group_name, sg_name, bp, sample_type, prtnr_name),
               dict(scan_name=scan_group,subtargets=[bp],targets=seq_groups[sg_name]),
               prtnr, ds)
              for sg_name in seq_groups.keys() 
              for bp in block_phases 
              for prtnr_name,prtnr in prtnrs.items()
              for sample_type,ds in sample_types.items()
              for scan_group_name, scan_group in scan_groups.items() 
              if (prtnr_name=='loco' or scan_group_name=='mvpa_all')]
    print len(slmaps),[s[0] for s  in slmaps]
    
    svqe = searchlight.SurfVoxQueryEngine(max_feat=128,vox_sl_radius=3.2,surf_sl_radius=20)
    svqe_cached = searchlight.CachedQueryEngineAlt(svqe)

    gnb = GNB(space='targets_num')
    spltr = Splitter(attr='balanced_partitions', attr_values=[1,2])
    # precompute neighborhood
    print 'precompute cached neighborhood'
    svqe_cached.train(ds_glm_mvpa)
    for i in xrange(ds_glm_mvpa.nfeatures): svqe_cached.query_byid(i)
    print 'compute permutations'
    #do_single_slmap_perm(gnb, svqe_cached, spltr, *slmaps[0])
    joblib.Parallel(n_jobs=8)(
        [joblib.delayed(do_single_slmap_perm)(gnb, svqe_cached, spltr, *args, overwrite=False) for args in slmaps])

    del ds_mvpa, ds_glm_mvpa
    

def group_cluster_threshold_analysis():
    #debug.active += ["GCTHR"]
    prtnrs = dict(
        loso = mvpa_nodes.prtnr_loso_cv(),
        loco = mvpa_nodes.prtnr_loco_cv()
    )
    sample_types = ['_glm']
    #sample_types = ['']

    #ftp = 1e-3
    #ftp = 5e-4
    ftp = 2e-4

    block_phases = ['exec','instr']
    #block_phases = ['instr']

    conn_all = np.load(os.path.join(proc_dir,'connectivity_96k.npy')).tolist()
    gct = GroupClusterThreshold(
        n_bootstrap=10000,
        feature_thresh_prob=ftp,
        chunk_attr='subject_id',
        n_blocks=32,
        n_proc=8, 
        neighborhood=conn_all)

    slmaps = ['%s_%s_%s%s_%s'%(scan_group_name, sg_name, bp, sample_type, prtnr_name)
              for sg_name in seq_groups.keys()
              for bp in block_phases
              for prtnr_name,prtnr in prtnrs.items()
              for sample_type in sample_types
              for scan_group_name, scan_group in scan_groups.items()
              if (prtnr_name=='loco' or scan_group_name=='mvpa_all')]
    groups = dict(
        all_subjects = [False,True],
        group_Int = [True],
        group_NoInt = [False]
    )
    for slmap_name in slmaps:
        print slmap_name
        slmaps_perm = [
            Dataset.from_hdf5(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_perms.h5'%(sid, slmap_name)))
            for sid in subject_ids]
        all_slmap_perm = vstack(slmaps_perm)
        del slmaps_perm

        slmaps_conf = [
            Dataset.from_hdf5(os.path.join(proc_dir,'searchlight_new/CoRe_%03d_%s_confusion.h5'%(sid,slmap_name))) \
            for sid in subject_ids]
        slmaps_acc = [confusion2acc(sl) for sl in slmaps_conf]
        del slmaps_conf
        slmaps = Dataset(np.asarray([sl.samples.mean(0) for sl in slmaps_acc]).astype(np.float32))
        slmaps.sa['subject_id'] = subject_ids
        del slmaps_acc
        #slmaps = Dataset.from_hdf5(os.path.join(proc_dir, 'searchlight_group', 'CoRe_group_%s_mean_acc.h5'%slmap_name))
        slmaps.sa['group'] = [sid in group_Int for sid in slmaps.sa.subject_id]
        for group_name, group in groups.items():
            print slmap_name, group_name
            try:
                gct.train(all_slmap_perm[dict(group=group)])
                gct_res = gct(slmaps[dict(group=group)])
                gct_res.save(
                    os.path.join(proc_dir, 'searchlight_gct',
                                 'CoRe_group_%s_%s_gct_ftp%.4f.h5'%(group_name, slmap_name, ftp)),
                    compression=compression)
                print gct_res.a.clusterstats[:1]
                del gct_res
            except Exception as e:
                print '%s for group %s failed' % (slmap_name,group_name)
                print e
        del all_slmap_perm, slmaps


def all_searchlight_2fold():
    new_sids = [sid for sid in subject_ids if len(glob.glob(os.path.join(proc_dir,output_subdir,'CoRe_%03d_*confusion.h5'%sid)))==0]
    print new_sids
    joblib.Parallel(n_jobs=3)([joblib.delayed(subject_searchlight_2fold)(sid) for sid in new_sids])

def subject_searchlight_2fold(sid):
    print('______________   CoRe %03d   ___________'%sid)
    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'glm_ds_%d.h5'%sid))
    ds = ds_glm 
    #ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))
    #targets_num(ds, ulabels)
    targets_num(ds_glm, ulabels)

    mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'd3_mvpa' in n]
    if len(mvpa_scan_names)==0:
        mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'mvpa' in n]
    svqe = searchlight.SurfVoxQueryEngine(max_feat=64, vox_sl_radius=2.3, surf_sl_radius=20)
    svqe_cached = searchlight.CachedQueryEngineAlt(svqe)

    gnb = GNB(space='targets_num')
#   gnb = GNB(space='sequence_type')

    spltr = Splitter(attr='partitions', attr_values=[1,2])
#    ds_glm.sa['sequence_type'] = ds_glm.sa.targets_num/2
    
    slght_2fold = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_2fold_factpart,
        svqe_cached,
        splitter=spltr,
        errorfx=mean_match_accuracy,
        pass_attr=ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        #postproc=mvpa_nodes.scan_blocks_confmat
    )


    slght_loso = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_loso_cv(1),
        svqe_cached,
        splitter=spltr,
        errorfx=mean_match_accuracy,
        pass_attr=ds.sa.keys()+ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        #postproc=mvpa_nodes.scan_blocks_confmat
    )

    mvpa_slght_subset = dict([('%s_%s_%s'%(scan_group, sg_name, bp),dict(targets=seqs,subtargets=[bp],scan_name=scans)) \
                              for sg_name,seqs in seq_groups.items() \
                              for bp in block_phases for scan_group,scans in scan_groups.items()])

    print mvpa_slght_subset
    for subset_name, subset in mvpa_slght_subset.items():
        #ds_subset = ds[subset]
        #slmap_2fold = slght_2fold(ds_subset)
        #slmap_2fold.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_2fold_confusion.h5'%(sid,subset_name)),
        #                 compression=compression)
        #del slmap_2fold

        ds_glm_subset = ds_glm[subset]
        slmap_glm_2fold = slght_2fold(ds_glm_subset)
        slmap_glm_2fold.save(os.path.join(proc_dir, output_subdir,
                                         'CoRe_%03d_%s_glm_2fold_confusion.h5'%(sid,subset_name)),
                            compression=compression)
        del slmap_glm_2fold

        if len(ds_glm_subset.sa['scan_name'].unique) > 1 and False:
            slmap_glm_loso = slght_loso(ds_glm_subset)
            slmap_glm_loso.save(
                os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_glm_loso_confusion.h5'%(sid,subset_name)),
                compression=compression)
            del slmap_glm_loso


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
        
    svqe = searchlight.SurfVoxQueryEngine(max_feat=128,vox_sl_radius=3.2,surf_sl_radius=20)
    svqe_cached = searchlight.CachedQueryEngineAlt(svqe)

    gnb = GNB(space='targets_num')
    spltr = Splitter(attr='balanced_partitions', attr_values=[1,2])

    slght_loso = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_loso_cv(nrepeat=1),
        svqe_cached,
        splitter=spltr,
        errorfx=None,
        pass_attr=ds.sa.keys()+ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        postproc=mvpa_nodes.scan_blocks_confmat)

    slght_loco = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_loco_cv(nrepeat=1),
        svqe_cached,
        splitter=spltr,
        errorfx=None,
        pass_attr=ds.sa.keys()+ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        postproc=mvpa_nodes.scan_blocks_confmat)

    slght_d123_train_test = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_d123_train_test(nrepeat=1),
        svqe_cached,
        splitter=spltr,
        errorfx=None,
        pass_attr=ds.sa.keys()+ds.fa.keys()+ds.a.keys()+[('ca.roi_sizes','fa')],
        enable_ca=['roi_sizes'],
        postproc=mvpa_nodes.scan_blocks_confmat)

    crossday=False
    if crossday:
        #zscore(ds, chunks_attr='scan_id', param_est=('targets','rest'))
        ds_exec = ds[dict(subtargets=['exec'])]
        #poly_detrend(ds_exec, chunks_attr='scan_id', polyord=0)
        ds_exec.fa = ds.fa # cheat CachedQueryEngine hashing
        slmap_crossday_exec = slght_d123_train_test(ds_exec)
        slmap_crossday_exec.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_crossday_exec_confusion.h5'%sid),
                                 compression=compression)
        del ds_exec, slmap_crossday_exec

    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
    poly_detrend(ds_glm_mvpa, chunks_attr='scan_id', polyord=0)
    zscore(ds_glm_mvpa, chunks_attr='scan_id')
    ds_glm_mvpa.fa = ds.fa # cheat CachedQueryEngine hashing

    ds_mvpa = ds[dict(scan_name=mvpa_scan_names)]
    poly_detrend(ds_mvpa, chunks_attr='scan_id', polyord=0)
    zscore(ds_mvpa, chunks_attr='scan_id', param_est=('targets','rest'))
    ds_mvpa.fa = ds.fa # cheat CachedQueryEngine hashing
    del ds

    mvpa_slght_subset = dict([('%s_%s_%s'%(scan_group, sg_name, bp),dict(targets=seqs,subtargets=[bp],scan_name=scans)) \
                              for sg_name,seqs in seq_groups.items() \
                              for bp in block_phases for scan_group,scans in scan_groups.items()])

    
    for subset_name, subset in mvpa_slght_subset.items():
        ds_subset = ds_mvpa[subset]
        ds_subset.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing

        do_slmap_loco = True
        if do_slmap_loco:
            slmap_loco = slght_loco(ds_subset)
            slmap_loco.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_loco_confusion.h5'%(sid,subset_name)),
                            compression=compression)
            del slmap_loco
        if len(ds_subset.sa['scan_name'].unique) > 1:
            slmap_loso = slght_loso(ds_subset)
            slmap_loso.save(
                os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_loso_confusion.h5'%(sid,subset_name)),
                compression=compression)
            del slmap_loso

        ds_glm_subset = ds_glm_mvpa[subset]
        #zscore(ds_glm_subset, chunks_attr='scan_id')

        if do_slmap_loco:
            ds_glm_subset.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing
            slmap_glm_loco = slght_loco(ds_glm_subset)
            slmap_glm_loco.save(os.path.join(proc_dir, output_subdir,
                                             'CoRe_%03d_%s_glm_loco_confusion.h5'%(sid,subset_name)),
                                compression=compression)
            del slmap_glm_loco
        if len(ds_glm_subset.sa['scan_name'].unique) > 1:
            slmap_glm_loso = slght_loso(ds_glm_subset)
            slmap_glm_loso.save(
                os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_glm_loso_confusion.h5'%(sid,subset_name)),
                compression=compression)
            del slmap_glm_loso

    return 
    slght_loco_delay = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_loco_delay,
        svqe_cached,
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
    new_sids = [sid for sid in subject_ids if len(glob.glob(os.path.join(proc_dir,output_subdir,'CoRe_%03d_*confusion.h5'%sid)))==0]
    print new_sids
    joblib.Parallel(n_jobs=2)([joblib.delayed(subject_searchlight_new)(sid) for sid in new_sids])

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
#rois = np.hstack([np.asarray([46,29,70,69,28,4,3,7,8])+a for a in [11100,12100]]+[53,17,10,49,51,12,8,47,11,50])
rois = np.asarray([53,17,10,49,51,12,8,47,11,50])
aparc_labels = dict([(fs_clt[fs_clt[:,0]==str(r)][0,1],r) for r in rois])

bas_labels = dict([('%s_%s'%(l,h),k+hi*1000) for l,k in bas_labels.items() for hi,h in enumerate('lr')])

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

from mvpa2.generators.base import Repeater
from mvpa2.base.node import ChainNode
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.clfs.stats import MCNullDist
def create_cv_nullhyp(clf, partitioner,splitter, nrepeat=200):
    repeater = Repeater(count=nrepeat)
    permutator = AttributePermutator('targets',
                                     limit={'partitions': 1},
                                     count=1)
    null_cv = CrossValidation(
        clf,
        ChainNode(
            [partitioner, permutator],
            space=partitioner.get_space()),
        splitter=splitter,
        postproc=mean_sample())
    distr_est = MCNullDist(repeater, tail='left',
                           measure=null_cv,
                           enable_ca=['dist_samples']) 

    cv_mc_corr = CrossValidation(clf,
                                 partitioner,
                                 splitter=splitter,
                                 postproc=mean_sample(),
                                 null_dist=distr_est,
                                 enable_ca=['stats'])
    return cv_mc_corr

def subject_rois_analysis(subj, clf):

    
    print('______________   %s   ___________'%subj)
    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%s'%subj, dataset_subdir, 'glm_ds_%s.h5'%subj))
    ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%s'%subj, dataset_subdir, 'ds_%s.h5'%subj))

    mvpa_scan_names = [n for n in np.unique(ds_glm.sa.scan_name) if 'mvpa' in n]
    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
    targets_num(ds_glm_mvpa, ulabels)
    poly_detrend(ds_glm_mvpa, chunks_attr='scan_id', polyord=0)

    #ds_glm_mvpa.samples/=ds_glm_mvpa.samples.std(0)
    #ds_glm_mvpa.samples[np.isnan(ds_glm_mvpa.samples)]=0
    ds_mvpa = ds[dict(scan_name=mvpa_scan_names)]
    targets_num(ds_mvpa, ulabels)
    if 'ba_thresh' not in ds_mvpa.fa:
        ds_mvpa.fa['ba_thresh'] = ds_mvpa.fa.ba_thres
    if 'ba_thresh' not in ds_glm_mvpa.fa:
        ds_glm_mvpa.fa['ba_thresh'] = ds_glm_mvpa.fa.ba_thres
    poly_detrend(ds_mvpa, chunks_attr='scan_id', polyord=0)
    zscore(ds_mvpa,chunks_attr='scan_id')
    del ds_glm, ds

    glm_rois_stats = dict()
    tr_rois_stats = dict()

    rois_groups = dict(
        ba=bas_labels,
        aparc=aparc_labels
    )

    seq_groups = {
        'mvpa_new_seqs' : ulabels[2:4],
        'tseq_intseq' : ulabels[:2],
#        'all_seqs': ulabels[:4]
    }
    block_phases = [
        'instr',
        'exec']
    cvte_subsets = dict([('%s_%s'%(sg_name, bp),dict(targets=seqs,subtargets=[bp])) \
                    for sg_name,seqs in seq_groups.items() \
                    for bp in block_phases])

    spltr = Splitter(attr='partitions',attr_values=[1,2])

    prtnrs = dict(
        #loco=mvpa_nodes.prtnr_loco_cv(1),
        twofold=mvpa_nodes.prtnr_2fold_sift,
        #loso=mvpa_nodes.prtnr_loso_cv(1)
    )
    
    for prtnr_name, prtnr in prtnrs.items():
        glm_rois_stats[prtnr_name] = {}
        tr_rois_stats[prtnr_name] = {}
        #cvte = CrossValidation(
        #    clf, prtnr, splitter=spltr, enable_ca=['stats'],
        #    #postproc=BinomialProportionCI(width=.95, meth='jeffreys')
        #)
        cvte = create_cv_nullhyp(clf,prtnr,spltr)
        for subset_name, subset in cvte_subsets.items():
            print '_'*10,subset_name, subset,'_'*10
            glm_rois_stats[prtnr_name][subset_name] = {}
            tr_rois_stats[prtnr_name][subset_name] = {}
            ds_glm_subset = ds_glm_mvpa[subset]
#            ds_glm_subset.samples/=ds_glm_subset.samples.std(0)
#            ds_glm_subset.samples[np.isnan(ds_glm_subset.samples)]=0
            ds_subset = ds_mvpa[subset]
            for roi_fa, rois_labels in rois_groups.items():
                for roi_name, roi_label in rois_labels.items():
                    cv_glm_res = cvte(ds_glm_subset[:,{roi_fa:[roi_label],'nans':[False]}])
                    glm_rois_stats[prtnr_name][subset_name][roi_name] = cvte.ca.stats
                    pvalue = cvte.ca.null_prob.samples
                    glm_rois_stats[prtnr_name][subset_name][roi_name].stats['pvalue'] = pvalue
                    print('glm\t%s\t%s\t%s\tacc=%f\tp=%.5f'%(prtnr_name, roi_name, subset_name, glm_rois_stats[prtnr_name][subset_name][roi_name].stats['ACC'],pvalue))
                    """
                    cv_res = cvte(ds_subset[:,{roi_fa:[roi_label]}])
                    tr_rois_stats[prtnr_name][subset_name][roi_name] = cvte.ca.stats
                    pvalue = cvte.ca.null_prob.samples
                    tr_rois_stats[prtnr_name][subset_name][roi_name].stats['pvalue'] = pvalue
                    print('tr\t%s\t%s\t%s\tacc=%f\tp=%.5f'%(prtnr_name, roi_name, subset_name, tr_rois_stats[prtnr_name][subset_name][roi_name].stats['ACC'],pvalue))
                    """
    return glm_rois_stats, tr_rois_stats
    

def confusion2acc(ds):
    return Dataset(
        ds.samples[:,:,np.eye(ds.samples.shape[-1],dtype=np.bool)].sum(-1).astype(np.float)/\
        ds.samples[:,0].sum(-1).sum(-1)[:,np.newaxis],
        sa=ds.sa,
        fa=ds.fa,
        a=ds.a)

import scipy.stats
from matplotlib import pyplot
def group_searchlight():
    import hcpview
    groupintmask = np.asarray([sid in group_Int for sid in subject_ids])

    groups = dict(
        All=np.ones(groupintmask.shape,dtype=np.bool),
        Int=groupintmask,
        NoInt=~groupintmask
    )
    chance_acc = [.5,.5,.25]
    prtnrs = ['loso','loco']
    sample_types = ['','_glm']

    slmaps = [('%s_%s_%s%s_%s'%(scan_group, sg_name, bp, sample_type, prtnr), .25 if sg_name=='all_seqs' else 0.5)
              for sg_name in seq_groups.keys() 
              for bp in block_phases
              for prtnr in prtnrs
              for sample_type in sample_types
              for scan_group in scan_groups.keys()
              if (prtnr=='loco' or scan_group=='mvpa_all')]
    print slmaps

    t_range = [0,5]
    pvalue = 0.01
#    hv = hcpview.HCPViewer()
#    hv.set_range(t_range)

    for sln, ch_acc in slmaps:
        print sln
        slmaps_conf = [Dataset.from_hdf5(os.path.join(proc_dir,'searchlight_new/CoRe_%03d_%s_confusion.h5'%(s,sln))) \
                       for s in subject_ids]
        slmaps_acc = [confusion2acc(sl) for sl in slmaps_conf]
        del slmaps_conf
        mean_acc = Dataset(np.asarray([sl.samples.mean(0) for sl in slmaps_acc]).astype(np.float32))
        mean_acc.sa['subject_id'] = subject_ids
        mean_acc.sa['groups'] = groupintmask

        mean_acc.save(
            os.path.join(proc_dir,'searchlight_group','CoRe_%s_mean_acc.h5'%sln),
            compression=compression)

        """
        for group_name, group_mask in groups.items():
            tp = Dataset(np.asarray(scipy.stats.ttest_1samp(mean_acc.samples[group_mask], ch_acc)))
            tp.save(
                os.path.join(proc_dir,'searchlight_group','CoRe_group_%s_%s_tp.h5'%(group_name,sln)),
                compression=compression)
            hv.set_data(tp.samples[0]*(tp.samples[1]<pvalue))
            hv._pts.glyph.glyph.scale_factor = 3
            montage = hv.montage_screenshot()
            fig = hcpview.plot_montage(montage, t_range)
            fig.savefig(os.path.join(proc_dir,'searchlight_group',
                                     'CoRe_group_%s_%s_t%d-%dp%0.3f.png'%(group_name,sln,t_range[0],t_range[1],pvalue)))
            pyplot.close(fig)
            del tp
        """
        del slmaps_acc, mean_acc

def group_contrast(contrast_name, sln1, sln2, group1, group2, tfunc=scipy.stats.ttest_ind,
                    t_range = [0,5], pvalue = 0.05, chance_level=.5):
    import hcpview
    slmaps1 = Dataset.from_hdf5(os.path.join(proc_dir, 'searchlight_group','CoRe_group_%s_mean_acc.h5'%sln1))
    slmaps2 = slmaps1 if sln1 == sln2 else Dataset.from_hdf5(os.path.join(proc_dir, 'searchlight_group','CoRe_group_%s_mean_acc.h5'%sln2))
    t,p = tfunc(slmaps1[dict(groups=group1)].samples,slmaps2[dict(groups=group2)].samples)
    t2,p2 = scipy.stats.ttest_1samp(slmaps1[dict(groups=group1)].samples, chance_level)
    #t3,p3 = scipy.stats.ttest_1samp(slmaps2[dict(groups=group2)].samples, chance_level)
    conj_t = np.min([t,t2],0)
    conj_p = np.max([p,p2],0)
    conj_tp = Dataset(np.asarray([conj_t,conj_p]))
    conj_tp.save(
        os.path.join(proc_dir,'searchlight_group','CoRe_ctx_%s_tp.h5'%(contrast_name)),
        compression=compression)
    hv = hcpview.HCPViewer()
    t_thresh = conj_t.copy()
    t_thresh[(conj_p>pvalue)] = np.nan
    hv.set_data(t_thresh)
    hv.set_range(t_range)
    hv._pts.glyph.glyph.scale_factor = 3
    montage = hv.montage_screenshot()
    fig = hcpview.plot_montage(montage, t_range)
    fig.savefig(os.path.join(proc_dir,'searchlight_group',
                             'CoRe_ctx_%s_t%d-%dp%0.3f.png'%(contrast_name,t_range[0],t_range[1],pvalue)))
    pyplot.close(fig)
    

def all_group_contrasts():
    group_contrast('mvpa_new_seqs_exec_mvpa2-mvpa1',
                   'mvpa2_mvpa_new_seqs_exec_loco',
                   'mvpa1_mvpa_new_seqs_exec_loco',
                   [True,False],[True,False],
                   scipy.stats.ttest_rel)
    group_contrast('mvpa_new_seqs_instr_mvpa2-mvpa1',
                   'mvpa2_mvpa_new_seqs_instr_loco',
                   'mvpa1_mvpa_new_seqs_instr_loco',
                   [True,False],[True,False],
                   scipy.stats.ttest_rel)
    group_contrast('intgroup_tseqintseq-mvpa_new_seqs_exec_loco',
                   'mvpa_all_tseq_intseq_exec_loco',
                   'mvpa_all_mvpa_new_seqs_exec_loco',
                   [True],[True],
                   scipy.stats.ttest_rel)
    group_contrast('intgroup_tseqintseq-mvpa_new_seqs_exec_loso',
                   'mvpa_all_tseq_intseq_exec_loso',
                   'mvpa_all_mvpa_new_seqs_exec_loso',
                   [True],[True],
                   scipy.stats.ttest_rel)
