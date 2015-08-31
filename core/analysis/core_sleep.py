import sys, os
import numpy as np
from ..mvpa import searchlight
from . import mvpa_nodes
from mvpa2.datasets import Dataset, vstack
from mvpa2.clfs.gnb import GNB
import joblib


preproc_dir = '/home/bpinsard/data/analysis/core_sleep'
dataset_subdir = 'dataset_noisecorr'
#dataset_subdir = 'dataset_smoothed'
#dataset_subdir = 'dataset_raw'

proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
output_subdir = 'searchlight'
#output_subdir = 'searchlight_smooth'
#output_subdir = 'searchlight_raw'

subject_ids = [1,11,23]
#subject_ids=subject_ids[-1:]

def all_searchlight():
    joblib.Parallel(n_jobs=10)([joblib.delayed(subject_searchlight)(sid) for sid in subject_ids])

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
        
        slght_loco = searchlight.GNBSurfVoxSearchlight(
            ds_mvpa,
            GNB(), 
            mvpa_nodes.prtnr_loco_cv,
            surf_sl_radius=20,
            surf_sl_max_feat=64,
            vox_sl_radius=2)
        slght_loso = searchlight.GNBSurfVoxSearchlight(
            ds_mvpa,
            GNB(), 
            mvpa_nodes.prtnr_loso_cv,
            surf_sl_radius=20,
            surf_sl_max_feat=64,
            vox_sl_radius=2)

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
            delay_trs = delay_trs[delay_trs < ds_mvpa.nsamples]
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
             
