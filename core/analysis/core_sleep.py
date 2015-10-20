import sys, os
import numpy as np
from ..mvpa import searchlight
from . import mvpa_nodes
from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.fx import mean_sample
from mvpa2.clfs.gnb import GNB
from mvpa2.misc.neighborhood import CachedQueryEngine
import joblib


preproc_dir = '/home/bpinsard/data/analysis/core_sleep'
dataset_subdir = 'dataset_noisecorr'
#dataset_subdir = 'dataset_smoothed'
#dataset_subdir = 'dataset_raw'

proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
output_subdir = 'searchlight_new'
#output_subdir = 'searchlight_smooth'
#output_subdir = 'searchlight_raw'

subject_ids = [1,11,23,22,63,50,67,79,54,107]
#subject_ids=subject_ids[:4]

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
    
    svqe = searchlight.SurfVoxQueryEngine(max_feat=64)
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
    slmap_crossday_exec.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_crossday_exec_confusion.h5'%sid))
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
        slmap_loco.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_loco_confusion.h5'%(sid,subset_name)))
        del slmap_loco
        slmap_loso = slght_loso(ds_subset)
        slmap_loso.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_%s_loso_confusion.h5'%(sid,subset_name)))
        del slmap_loso

    slght_loco_delay = searchlight.GNBSearchlightOpt(
        gnb,
        mvpa_nodes.prtnr_loco_cv,
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
        delay_ds.sa.targets_num = ds_mvpa.sa.targets_num[delay_trs-d+2]
        delay_ds.chunks = np.arange(delay_ds.nsamples)
        delay_ds.fa = ds_mvpa.fa # cheat CachedQueryEngine hashing
        slmap_confmat = slght_loco_delay(delay_ds)
        delay_slmaps_confusion.append(slmap_confmat)
        del delay_ds
    slmap_delay = vstack(delay_slmaps_confusion)
    slmap_delay.fa = ds_mvpa.fa
    slmap_delay.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_delay_confusion.h5'%sid))
    del delay_slmaps_confusion, slmap_delay

    

def targets_num(ds, utargets, targets_num_attr='targets_num'):
    targets2idx = dict([(t,i) for i,t in enumerate(utargets)])
    ds.sa['targets_num'] = np.asarray([targets2idx[t] for t in ds.targets], dtype=np.uint8)

def all_searchlight():
    joblib.Parallel(n_jobs=2)([joblib.delayed(subject_searchlight_new)(sid) for sid in subject_ids])

def subject_searchlight(sid):
    print('______________   CoRe %03d   ___________'%sid)
    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'glm_ds_%d.h5'%sid))
    ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))
    mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'd3_mvpa' in n]
    if len(mvpa_scan_names)==0:
        mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'mvpa' in n]

        scan_names=ds.sa.scan_name
        mvpa_tr_scans_mask = reduce(
            lambda mask,msn: np.logical_or(mask,scan_names==msn), 
            mvpa_scan_names,
            np.zeros(ds.nsamples,dtype=np.bool))
        ds_mvpa = ds[mvpa_tr_scans_mask]
        del ds
        
        svqe = core.mvpa.searchlight.SurfVoxQueryEngine(max_feat=64)
        svqe.train(ds_glm)
        
        
        slght_loco = GNBSearchlight(
            GNB(space='targets_num'),
            core.analysis.mvpa_nodes.prtnr_loco_cv,
            svqe,
            splitter=Splitter(attr='balanced_partitions',attr_values=[1,2]),
            reuse_neighbors=True, errorfx=None,
            pass_attr=ds.sa.keys())

        searchlight.GNBSurfVoxSearchlight(
            ds_mvpa,
            GNB(), 
            mvpa_nodes.prtnr_loco_cv,
            surf_sl_radius=20,
            surf_sl_max_feat=64,
            vox_sl_radius=2,
            postproc=mean_sample())
        slght_loso = searchlight.GNBSurfVoxSearchlight(
            ds_mvpa,
            GNB(),
            mvpa_nodes.prtnr_loso_cv,
            surf_sl_radius=20,
            surf_sl_max_feat=64,
            vox_sl_radius=2,
            postproc=mean_sample())

        # do loco searchlight on each mvpa scan separately
        slmaps_accuracy = []
        slmaps_confusion = []

        scans_subsets = [(msn, [msn]) for msn in mvpa_scan_names]
        if len(mvpa_scan_names)>1:
            scans_subsets.append(('all', mvpa_scan_names))

        tr_subsets = [
            ('all', np.ones(ds_mvpa.nsamples, dtype=np.bool)),
            ('norest',ds_mvpa.sa.subtargets!='rest'),
            ('instr', ds_mvpa.sa.subtargets=='instr'),
            ('exec', ds_mvpa.sa.subtargets=='exec'),]
        glm_subsets = [
            ('all', np.ones(ds_glm.nsamples, dtype=np.bool)),
            ('instr', ds_glm.sa.subtargets=='instr'),
            ('exec', ds_glm.sa.subtargets=='exec'),]

        for scan_subset, scans in scans_subsets:
            # using trs
            mvpa_tr_scans_mask = reduce(
                lambda mask,msn: np.logical_or(mask,ds_mvpa.sa.scan_name==msn), 
                scans,
                np.zeros(ds_mvpa.nsamples,dtype=np.bool))

            mvpa_glm_scans_mask = reduce(
                lambda mask,msn: np.logical_or(mask,ds_glm.sa.scan_name==msn), 
                scans,
                np.zeros(ds_glm.nsamples,dtype=np.bool))
            
            for subset_name, subset in tr_subsets:
                print('@@@@@@@@@@@@@@@@  %s %s tr @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name))
                subs = np.logical_and(mvpa_tr_scans_mask, subset)
                if not len(subs):
                    raise RuntimeError
                slmaps = slght_loco(ds_mvpa[subs])
                for slmap in slmaps:
                    slmap.sa['slmap'] = ['slmap_tr_%s_%s'%(subset_name,scan_subset)]
                slmaps_accuracy.append(slmaps[1])
                slmaps_confusion.append(slmaps[0])

            for subset_name, subset in glm_subsets:
                print('@@@@@@@@@@@@@@@@  %s %s glm @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name))
                subs = np.logical_and(mvpa_glm_scans_mask, subset)
                if not len(subs):
                    raise RuntimeError
                slmaps = slght_loco(ds_glm[subs])
                for slmap in slmaps:
                    slmap.sa['slmap'] = ['slmap_glm_%s_%s'%(subset_name,scan_subset)]
                slmaps_accuracy.append(slmaps[1])
                slmaps_confusion.append(slmaps[0])

            if len(scans)>1:

                for subset_name, subset in tr_subsets:
                    print('@@@@@@@@@@@@@@@@  %s %s loso tr @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name))
                    slmaps = slght_loso(ds_mvpa[np.logical_and(mvpa_tr_scans_mask, subset)])
                    for slmap in slmaps:
                        slmap.sa['slmap'] = ['slmap_tr_loso_%s_%s'%(subset_name,scan_subset)]
                    slmaps_accuracy.append(slmaps[1])
                    slmaps_confusion.append(slmaps[0])

                for subset_name, subset in glm_subsets:
                    print('@@@@@@@@@@@@@@@@  %s %s loso glm @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name))
                    slmaps = slght_loso(ds_glm[np.logical_and(mvpa_glm_scans_mask, subset)])
                    for slmap in slmaps:
                        slmap.sa['slmap'] = ['slmap_glm_loso_%s_%s'%(subset_name,scan_subset)]
                    slmaps_accuracy.append(slmaps[1])
                    slmaps_confusion.append(slmaps[0])
                
        all_slmaps_accuracy = vstack(slmaps_accuracy)
        all_slmaps_accuracy.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_accuracy_slmaps.h5'%sid))
        print('all accuracies ', all_slmaps_accuracy.samples.max(1))
        all_slmaps_confmat = vstack(slmaps_confusion)
        all_slmaps_confmat.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_confusion_slmaps.h5'%sid))
        del all_slmaps_accuracy, all_slmaps_confmat, slmaps_accuracy, slmaps_confusion

        start = -2
        end = 22
        delays = range(start, end)
        delay_slmaps_accuracy = []
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
            delay_ds.chunks = np.arange(delay_ds.nsamples)
            slmaps = slght_loco(delay_ds)
            print('$$$$ delay %d : max accuracy %f'%(d, slmaps[1].samples.max()))
            delay_slmaps_accuracy.append(slmaps[1])
            delay_slmaps_confusion.append(slmaps[0])
            del delay_ds

        delay_slmaps_confusion = vstack(delay_slmaps_confusion)
        delay_slmaps_accuracy = vstack(delay_slmaps_accuracy)
        delay_slmaps_confusion.sa['delays'] = delays
        delay_slmaps_accuracy.sa['delays'] = delays
        delay_slmaps_confusion.save(os.path.join(proc_dir, output_subdir, 'CoRe %03d_delay_confusion_slmaps.h5'%sid))
        delay_slmaps_accuracy.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_delay_accuracy_slmaps.h5'%sid))
        del delay_slmaps_accuracy, delay_slmaps_confusion


def searchlight_cross_day(sid):
    print('______________   CoRe %03d   ___________'%sid)
    ds_all = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))
    ds = ds_all[np.logical_and(ds_all.sa.scan_name!='d1_sleep',ds_all.sa.scan_name!='d2_sleep')]
    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'glm_ds_%d.h5'%sid))
    del ds_all
    slght_d3_retest = searchlight.GNBSurfVoxSearchlight(
        ds,
        GNB(),
        mvpa_nodes.targets_balancer,
        surf_sl_radius=20,
        surf_sl_max_feat=64,
        vox_sl_radius=2,
        postproc=mean_sample())

    subsets = {
        'all':lambda x: slice(0,None),
        'noinstr':lambda x:x.sa.subtargets!='instr',
        'exec':lambda x:x.sa.subtargets=='exec'}
    slmaps_accuracy = []
    slmaps_confusion = []
    for learn_sn, part in zip(mvpa_nodes.learning_scan_names,mvpa_nodes.prtnr_d3_retest.generate(ds)):
        for subset, sel in subsets.items():
            print 'slmap_d3_retest_%s_%s'%('_'.join(learn_sn),subset)
            slmaps = slght_d3_retest(part[sel(part)])
            for slmap in slmaps:
                slmap.sa['slmap'] = ['slmap_d3_retest_%s_%s'%('_'.join(learn_sn),subset)]
            slmaps_accuracy.append(slmaps[1])
            slmaps_confusion.append(slmaps[0])
        
    slght_d1d2_training = searchlight.GNBSurfVoxSearchlight(
        ds,
        GNB(), 
        mvpa_nodes.targets_balancer,
        surf_sl_radius=20,
        surf_sl_max_feat=64,
        vox_sl_radius=2,
        postproc=mean_sample())

    # add last subset for training on last 7 blocks
    subsets['exec_last7blocks']=lambda x:np.logical_and(
        x.sa.subtargets=='exec',
        np.logical_or(x.sa.blocks_idx>6,x.sa.partitions!=2))

    for learn_sn, part in zip(mvpa_nodes.learning_scan_names,mvpa_nodes.prtnr_d1d2_training.generate(ds)):
        for subset, sel in subsets.items():
            sub_sel=sel(part)
            print 'slmap_d1d2_training_%s_%s : %d samples'%('_'.join(learn_sn),subset,np.count_nonzero(sub_sel))
            slmaps = slght_d3_retest(part[sub_sel])
            for slmap in slmaps:
                slmap.sa['slmap'] = ['slmap_d1d2_training_%s_%s'%('_'.join(learn_sn),subset)]
            slmaps_accuracy.append(slmaps[1])
            slmaps_confusion.append(slmaps[0])

    all_slmaps_accuracy = vstack(slmaps_accuracy)
    all_slmaps_accuracy.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_crossday_accuracy_slmaps.h5'%sid))
    print('all accuracies ', all_slmaps_accuracy.samples.max(1))
    all_slmaps_confmat = vstack(slmaps_confusion)
    all_slmaps_confmat.save(os.path.join(proc_dir, output_subdir, 'CoRe_%03d_crossday_confusion_slmaps.h5'%sid))
    del all_slmaps_accuracy, all_slmaps_confmat, slmaps_accuracy, slmaps_confusion



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
        
#        cross_vals = [
#            ('intra_run1', mvpa_nodes.prtnr_loco_cv
             
