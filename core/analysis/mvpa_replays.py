import sys, os
import numpy as np
from ..mvpa import searchlight
from . import mvpa_nodes
from mvpa2.datasets import Dataset, vstack
from mvpa2.clfs.gnb import GNB

preproc_dir = '/home/bpinsard/data/analysis/core'
dataset_subdir = 'dataset_noisecorr'
dataset_subdir = 'dataset_smoothed'
proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
output_subdir = 'searchlight'
output_subdir = 'searchlight_smooth2'


subjects = ['S00_BP_pilot','S01_ED_pilot','S349_AL_pilot','S341_WC_pilot','S02_PB_pilot','S03_MC_pilot']
#subjects = subjects[2:3]
#subjects=subjects[-1:]

def all_searchlight():
    for subj in subjects:
        print '______________   %s   ___________'%subj
        ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_%s'%subj, dataset_subdir, 'glm_ds_%s.h5'%subj))
        ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_%s'%subj, dataset_subdir, 'ds_%s.h5'%subj))

        mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'd3_mvpa' in n]
        if len(mvpa_scan_names)==0:
            mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'mvpa' in n]
        
#        ds_glm = Dataset.from_hdf5(os.path.join(proc_dir, 'tests', 'glm_ds_%s.h5'%subj))
#        ds = Dataset.from_hdf5(os.path.join(proc_dir, 'tests', 'ds_%s.h5'%subj))
        slght_loco = searchlight.GNBSurfVoxSearchlight(
            ds,
            GNB(), 
            mvpa_nodes.prtnr_loco_cv,
            surf_sl_radius=30,
            surf_sl_max_feat=128,
            vox_sl_radius=3)
        slght_loso = searchlight.GNBSurfVoxSearchlight(
            ds,
            GNB(), 
            mvpa_nodes.prtnr_loso_cv,
            surf_sl_radius=30,
            surf_sl_max_feat=128,
            vox_sl_radius=3)

        # do loco searchlight on each mvpa scan separately
        slmaps_accuracy = []
        slmaps_confusion = []

        scans_subsets = [(msn, [msn]) for msn in mvpa_scan_names]
        if len(mvpa_scan_names)>1:
            scans_subsets.append(('all', mvpa_scan_names))

        tr_subsets = [
            ('all', np.ones(ds.nsamples, dtype=np.bool)),
            ('norest',ds.sa.subtargets!='rest'),
            ('instr', ds.sa.subtargets=='instr'),
            ('exec', ds.sa.subtargets=='exec'),]
        glm_subsets = [
            ('all', np.ones(ds_glm.nsamples, dtype=np.bool)),
            ('instr', ds_glm.sa.subtargets=='instr'),
            ('exec', ds_glm.sa.subtargets=='exec'),]

        for scan_subset, scans in scans_subsets:
            # using trs
            mvpa_tr_scans_mask = reduce(
                lambda mask,msn: np.logical_or(mask,ds.sa.scan_name==msn), 
                scans,
                np.zeros(ds.nsamples,dtype=np.bool))

            mvpa_glm_scans_mask = reduce(
                lambda mask,msn: np.logical_or(mask,ds_glm.sa.scan_name==msn), 
                scans,
                np.zeros(ds_glm.nsamples,dtype=np.bool))
            
            for subset_name, subset in tr_subsets:
                print '@@@@@@@@@@@@@@@@  %s %s tr @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name)
                slmaps = slght_loco(ds[np.logical_and(mvpa_tr_scans_mask, subset)])
                for slmap in slmaps:
                    slmap.sa['slmap'] = ['slmap_tr_%s_%s'%(subset_name,scan_subset)]
                slmaps_accuracy.append(slmaps[1])
                slmaps_confusion.append(slmaps[0])

            for subset_name, subset in glm_subsets:
                print '@@@@@@@@@@@@@@@@  %s %s glm @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name)
                slmaps = slght_loco(ds_glm[np.logical_and(mvpa_glm_scans_mask, subset)])
                for slmap in slmaps:
                    slmap.sa['slmap'] = ['slmap_glm_%s_%s'%(subset_name,scan_subset)]
                slmaps_accuracy.append(slmaps[1])
                slmaps_confusion.append(slmaps[0])

            if len(scans)>1:

                for subset_name, subset in tr_subsets:
                    print '@@@@@@@@@@@@@@@@  %s %s loso tr @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name)
                    slmaps = slght_loso(ds[np.logical_and(mvpa_tr_scans_mask, subset)])
                    for slmap in slmaps:
                        slmap.sa['slmap'] = ['slmap_tr_loso_%s_%s'%(subset_name,scan_subset)]
                    slmaps_accuracy.append(slmaps[1])
                    slmaps_confusion.append(slmaps[0])

                for subset_name, subset in glm_subsets:
                    print '@@@@@@@@@@@@@@@@  %s %s loso glm @@@@@@@@@@@@@@@@@@@@'%(scan_subset, subset_name)
                    slmaps = slght_loso(ds_glm[np.logical_and(mvpa_glm_scans_mask, subset)])
                    for slmap in slmaps:
                        slmap.sa['slmap'] = ['slmap_glm_loso_%s_%s'%(subset_name,scan_subset)]
                    slmaps_accuracy.append(slmaps[1])
                    slmaps_confusion.append(slmaps[0])
                
        all_slmaps_accuracy = vstack(slmaps_accuracy)
        all_slmaps_accuracy.save(os.path.join(proc_dir, output_subdir, '%s_accuracy_slmaps.h5'%subj))
        all_slmaps_confmat = vstack(slmaps_confusion)
        all_slmaps_confmat.save(os.path.join(proc_dir, output_subdir, '%s_confusion_slmaps.h5'%subj))
        del all_slmaps_accuracy, all_slmaps_confmat, slmaps_accuracy, slmaps_confusion

        mvpa_tr_scans_mask = reduce(
                lambda mask,msn: np.logical_or(mask,ds.sa.scan_name==msn), 
                mvpa_scan_names,
                np.zeros(ds.nsamples,dtype=np.bool))
        ds_mvpa = ds[mvpa_tr_scans_mask]
        start = -2
        end = 22
        delays = range(start, end)
        delay_slmaps_accuracy = []
        delay_slmaps_confusion = []
        blocks_tr = np.where(np.diff(ds_mvpa.sa.blocks_idx_no_delay)>0)[0]+1
        for d in delays:
            print '######## computing searchlight for delay %d #######'%d
            delay_trs = blocks_tr+d
            delay_trs = delay_trs[delay_trs < ds_mvpa.nsamples]
            delay_ds = ds_mvpa[delay_trs]
            delay_ds.targets = ds_mvpa.sa.targets_no_delay[delay_trs-d]
            delay_ds.chunks = np.arange(delay_ds.nsamples)
            slmaps = slght_loco(delay_ds)
            print '$$$$ delay %d : max accuracy %f'%(d, slmaps[1].samples.max())
            delay_slmaps_accuracy.append(slmaps[1])
            delay_slmaps_confusion.append(slmaps[0])
            del delay_ds

        delay_slmaps_confusion = vstack(delay_slmaps_confusion)
        delay_slmaps_accuracy = vstack(delay_slmaps_accuracy)
        delay_slmaps_confusion.sa['delays'] = delays
        delay_slmaps_accuracy.sa['delays'] = delays
        delay_slmaps_confusion.save(os.path.join(proc_dir, output_subdir, '%s_delay_confusion_slmaps.h5'%subj))
        delay_slmaps_accuracy.save(os.path.join(proc_dir, output_subdir, '%s_delay_accuracy_slmaps.h5'%subj))
        del delay_slmaps_accuracy, delay_slmaps_confusion



def all_rois_analysis():
    
    for subj in subjects:
        print '______________   %s   ___________'%subj
        ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_%s'%subj, dataset_subdir, 'glm_ds_%s.h5'%subj))
        ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_%s'%subj, dataset_subdir, 'ds_%s.h5'%subj))
