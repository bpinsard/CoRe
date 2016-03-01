import sys, os, glob
import numpy as np
from ..mvpa import searchlight
from . import mvpa_nodes
from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.fx import mean_sample
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.fx import BinomialProportionCI
from mvpa2.clfs.gnb import GNB
from mvpa2.misc.neighborhood import CachedQueryEngine
from mvpa2.generators.partition import NFoldPartitioner
import joblib


preproc_dir = '/home/bpinsard/data/analysis/core_sleep'
dataset_subdir = 'dataset_noisecorr'
#dataset_subdir = 'dataset_smoothed'
dataset_subdir = 'dataset_nofilt'

proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
output_subdir = 'searchlight_new'
compression= 'gzip'

subject_ids = [1,11,23,22,63,50,79,54,107,128,162,102,82,155,100,94,87,192]
group_Int = [1,23,63,79,107,128,82,100,94,87,192]
#subject_ids=subject_ids[5:]
ulabels = ['CoReTSeq','CoReIntSeq','mvpa_CoReOtherSeq1','mvpa_CoReOtherSeq2','rest']
#ulabels = ulabels[1:]
subject_ids = [102,107,11,128,155,162,1,22,23,50,54,63,79,82,87]

seq_groups = {
    'mvpa_new_seqs' : ulabels[2:4],
    'tseq_intseq' : ulabels[:2],
    'all_seqs': ulabels[:4]
}
block_phases = ['instr','exec']
scan_groups = dict(
    mvpa1=['d3_mvpa1'],
    mvpa2=['d3_mvpa2'],
    mvpa_all=['d3_mvpa1','d3_mvpa2'])


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

    ds_exec = ds[dict(subtargets=['exec'])]
    poly_detrend(ds_exec, chunks_attr='scan_id', polyord=0)
    ds_exec.fa = ds.fa # cheat CachedQueryEngine hashing
    slmap_crossday_exec = slght_d123_train_test(ds_exec)
    slmap_crossday_exec.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_crossday_exec_confusion.h5'%sid),
                             compression=compression)
    del ds_exec, slmap_crossday_exec

    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
    ds_glm_mvpa.fa = ds.fa # cheat CachedQueryEngine hashing

    ds_mvpa = ds[dict(scan_name=mvpa_scan_names)]
    ds_mvpa.fa = ds.fa # cheat CachedQueryEngine hashing
    del ds

    mvpa_slght_subset = dict([('%s_%s_%s'%(scan_group, sg_name, bp),dict(targets=seqs,subtargets=[bp],scan_name=scans)) \
                              for sg_name,seqs in seq_groups.items() \
                              for bp in block_phases for scan_group,scans in scan_groups.items()])

    
    for subset_name, subset in mvpa_slght_subset.items():
        ds_subset = ds_mvpa[subset]
        poly_detrend(ds_subset, chunks_attr='scan_id', polyord=0)
        ds_subset.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing

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

        ds_glm_subset = ds_glm_mvpa[subset].copy()
        poly_detrend(ds_glm_subset, chunks_attr='scan_id', polyord=0)
        
        ds_glm_subset.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing
        slmap_glm_loco = slght_loco(ds_glm_subset)
        slmap_glm_loco.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_glm_loco_confusion.h5'%(sid,subset_name)),
                            compression=compression)
        del slmap_glm_loco
        if len(ds_glm_subset.sa['scan_name'].unique) > 1:
            slmap_glm_loso = slght_loso(ds_glm_subset)
            slmap_glm_loso.save(
                os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_glm_loso_confusion.h5'%(sid,subset_name)),
                compression=compression)
            del slmap_glm_loso

    slght_loco_delay = searchlight.GNBSearchlightOpt(
        gnb,
        NFoldPartitioner(attr='chunks'),
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
        poly_detrend(delay_ds, chunks_attr='scan_id', polyord=0)
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
    new_sids = [sid for sid in subject_ids if len(glob.glob(os.path.join(proc_dir,output_subdir,'CoRe_%03d_*'%sid)))==0]
    joblib.Parallel(n_jobs=2)([joblib.delayed(subject_searchlight_new)(sid) for sid in new_sids])


subjects_4targ = ['S01_ED_pilot','S349_AL_pilot','S341_WC_pilot','S02_PB_pilot','S03_MC_pilot']
ntargets = 4

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
#rois = np.asarray([53,17,10,49,51,12,8,47,11,50])
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

from mvpa2.generators.base import Repeater
from mvpa2.base.node import ChainNode
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.clfs.stats import MCNullDist
def create_cv_nullhyp(clf, partitioner,splitter):
    repeater = Repeater(count=200)
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
#    ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%s'%subj, dataset_subdir, 'ds_%s.h5'%subj))

    mvpa_scan_names = [n for n in np.unique(ds_glm.sa.scan_name) if 'mvpa' in n]
    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
#    ds_glm_mvpa.samples/=ds_glm_mvpa.samples.std(0)
#    ds_glm_mvpa.samples[np.isnan(ds_glm_mvpa.samples)]=0
#    ds_mvpa = ds[dict(scan_name=mvpa_scan_names)]
    del ds_glm#, ds

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
        loco=mvpa_nodes.prtnr_loco_cv,
        loso=mvpa_nodes.prtnr_loso_glm_cv
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
            poly_detrend(ds_glm_subset, chunks_attr='scan_id', polyord=0)
#            ds_glm_subset.samples/=ds_glm_subset.samples.std(0)
#            ds_glm_subset.samples[np.isnan(ds_glm_subset.samples)]=0
#            ds_subset = ds_mvpa[subset]
#            poly_detrend(ds_subset, chunks_attr='scan_id', polyord=0)
            for roi_fa, rois_labels in rois_groups.items():
                for roi_name, roi_label in rois_labels.items():
                    cv_glm_res = cvte(ds_glm_subset[:,{roi_fa:[roi_label]}])
                    glm_rois_stats[prtnr_name][subset_name][roi_name] = cvte.ca.stats
                    pvalue = cvte.ca.null_prob.samples
                    print('glm\t%s\t%s\t%s\tacc=%f\tp=%.5f'%(prtnr_name, roi_name, subset_name, glm_rois_stats[prtnr_name][subset_name][roi_name].stats['ACC'],pvalue))
                    """
                    cv_res = cvte(ds_subset[:,{roi_fa:[roi_label]}])
                    tr_rois_stats[prtnr_name][subset_name][roi_name] = cvte.ca.stats
                    print('tr\t%s\t%s\t%s\tacc=%f'%(prtnr_name, roi_name, subset_name, tr_rois_stats[prtnr_name][subset_name][roi_name].stats['ACC']))
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
from matplotlib.pyplot import imsave
def group_searchlight():
    
    groupintmask = np.asarray([sid in group_Int for sid in subject_ids])

    groups = dict(
        All=np.ones(groupintmask.shape,dtype=np.bool),
        Int=groupintmask,
        NoInt=~groupintmask)
    chance_acc = [.5,.5,.25]
    prtnrs = ['loso','loco']
    sample_types = ['','_glm']

    slmaps = [('%s_%s_%s%s_%s'%(scan_group, sg_name, bp, sample_type, prtnr), .25 if sg_name=='all_seqs' else 0.5) for sg_name in seq_groups.keys() for bp in block_phases for prtnr in prtnrs for sample_type in sample_types for scan_group in scan_groups.keys() if (prtnr=='loco' or scan_group=='mvpa_all')]
    print slmaps


    import hcpview
    t_range = [0,5]
    pvalue = 0.01
    hv = hcpview.HCPViewer()
    hv.set_range(t_range)

    for sln, ch_acc in slmaps:
        print sln
        slmaps_conf = [Dataset.from_hdf5(os.path.join(proc_dir,'searchlight_new/CoRe_%03d_%s_confusion.h5'%(s,sln))) \
                       for s in subject_ids]
        slmaps_acc = [confusion2acc(sl) for sl in slmaps_conf]
        del slmaps_conf
        mean_acc = Dataset(np.asarray([sl.samples.mean(0) for sl in slmaps_acc]).astype(np.float32))
        mean_acc.sa['subject_id'] = subject_ids
        
        mean_acc.save(os.path.join(proc_dir,'searchlight_group','CoRe_group_%s_mean_acc.h5'%sln))
        
        for group_name, group_mask in groups.items():
            tp = Dataset(np.asarray(scipy.stats.ttest_1samp(mean_acc.samples[group_mask], ch_acc)))
            tp.save(os.path.join(proc_dir,'searchlight_group','CoRe_group_%s_%s_tp.h5'%(group_name,sln)))
            hv.set_data(tp.samples[0]*(tp.samples[1]<pvalue))
            hv._pts.glyph.glyph.scale_factor = 3
            montage = hv.montage_screenshot()
            imsave(
                os.path.join(proc_dir,'searchlight_group',
                             'CoRe_group_%s_%s_t%d-%dp%0.3f.png'%(group_name,sln,t_range[0],t_range[1],pvalue)),
                montage)
            del tp
        del slmaps_acc, mean_acc
