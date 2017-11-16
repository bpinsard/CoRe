import sys, os
import numpy as np
import glob
import time_decoding.decoding as timedec
from .core_sleep import ulabels, preproc_dir, mvpa_ds, proc_dir
from mvpa2.datasets import Dataset

data_dir='/home/bpinsard/data/raw/UNF/CoRe'
tr=2.16

def subject_mvpa_ds(sid, hptf_thresh=8, reg_sa='regressors_exec'):

    ts_files = [ os.path.join(preproc_dir, '_subject_id_%d'%sid, 'moco_bc_mvpa_aniso','mapflow',
                              '_moco_bc_mvpa_aniso%d'%scan_id,'ts.h5') for scan_id in range(2)]
    ds_mvpa = [mvpa_ds.ds_from_ts(f) for f in ts_files]
    dss_mvpa = []
    for dsi, ds in enumerate(ds_mvpa):
        mvpa_ds.ds_set_attributes(
            ds, sorted(glob.glob(data_dir+'/Behavior/CoRe_%03d_D3/CoRe_%03d_mvpa-%d-D-Three_*.mat'%(sid,sid,dsi+1)))[-1])
        
        mvpa_ds.add_aparc_ba_fa(ds,sid,os.path.join(preproc_dir, 'surface_32k', '_subject_id_%s'))
        
        regs = ds.sa[reg_sa].value.astype(np.float)
        reg_names = ds.sa[reg_sa].value.dtype.names
        exec15 = regs[:,32+15]
        instr16 = regs[:,16]
        last_part1 = np.where(np.abs(exec15) >0)[0][-1]
        first_part2 = np.where(np.abs(instr16) >0)[0][0]
        
        chunks = np.hstack([np.tile(np.asarray([-1,1]).repeat(16),2),[0]])

        print last_part1, first_part2
        ds_part1 = ds[:last_part1]
        ds_part1.a.blocks_tr = ds_part1.a.blocks_tr[:16]
        ds_part1.a.blocks_targets = ds_part1.a.blocks_targets[:16]
        ds_part1.a.blocks_durations = ds_part1.a.blocks_durations[:16]
        mvpa_ds.preproc_ds(ds_part1, detrend=True, hptf=True, hptf_thresh=hptf_thresh)

        ds_part1.sa[reg_sa] = regs[:last_part1, chunks<=0].astype(
            [(n,np.float )for n,c in zip(reg_names,chunks) if c<=0])

        ds_part1.sa.chunks = [dsi*2]*ds_part1.nsamples
        dss_mvpa.append(ds_part1)

        ds_part2 = ds[first_part2:]
        ds_part2.a.blocks_tr = ds_part2.a.blocks_tr[16:]-first_part2
        ds_part2.a.blocks_targets = ds_part2.a.blocks_targets[16:]
        ds_part2.a.blocks_durations = ds_part2.a.blocks_durations[16:]
        mvpa_ds.preproc_ds(ds_part2, detrend=True, hptf=True, hptf_thresh=hptf_thresh)
        
        ds_part2.sa[reg_sa] = regs[first_part2:,chunks>=0].astype(
            [(n,np.float )for n,c in zip(reg_names,chunks) if c>=0])
        ds_part2.sa['chunks'] = [dsi*2+1]*ds_part2.nsamples
        
        dss_mvpa.append(ds_part2)

    return dss_mvpa

def wb_time_decoding(sid, n_alpha=5, logistic_window=9):
    dss_mvpa = subject_mvpa_ds(sid)
    nchunks = len(dss_mvpa)
    custom_cv = [[tr,te] for tr in range(nchunks) for te in range(nchunks) if tr!=te]
    design_mtxs = [timedec.design_matrix(ds.nsamples,tr,
                                         ds.a.blocks_tr*float(tr),
                                         ds.a.blocks_targets,
                                         durations=ds.a.blocks_durations,
                                         drift_model=None
    ) for ds in dss_mvpa]
    stimuli_onehot = []
    classes = ulabels[:4]
    nclasses = len(classes)
    for ds in dss_mvpa:
        onehot = np.zeros((ds.nsamples, nclasses))
        for seqi,seq in enumerate(classes):
            onehot[ds.a.blocks_tr[ds.a.blocks_targets==seq],seqi] = 1
        stimuli_onehot.append(onehot)


        
    scores = dict()

    rois = Dataset.from_hdf5(os.path.join(proc_dir,'msl_rois_new.h5'))

    for ri,roi_name in enumerate(rois.a.roi_labels):
        scores[roi_name]=[]
        mask = rois.samples[0]==ri+1
        for tr,te in custom_cv:
            fmri_train, fmri_test = dss_mvpa[tr].samples[:,mask], dss_mvpa[te].samples[:,mask]
            accuracy = cv_acc(fmri_train, fmri_test,
                              stimuli_onehot[tr], stimuli_onehot[te],
                              design_mtxs[tr],design_mtxs[te],
                              n_alpha, logistic_window)
        
            scores[roi_name].append(accuracy)
    scores['wb']=[]
    for tr,te in custom_cv:
        
        fmri_train, fmri_test = timedec.feature_selection(
            dss_mvpa[tr].samples, dss_mvpa[te].samples,
            np.argmax(stimuli_onehot[tr], axis=1),k=10000)
        
        accuracy = cv_acc(fmri_train, fmri_test,
                          stimuli_onehot[tr], stimuli_onehot[te],
                          design_mtxs[tr],design_mtxs[te],
                          n_alpha, logistic_window)
        scores['wb'].append(accuracy)
    return scores


def cv_acc(fmri_train, fmri_test,
           stimuli_train, stimuli_test,
           design_mtx_train, design_mtx_test,
           n_alpha, logistic_window):
    classes = ulabels[:4]
    nclasses = len(classes)

    alphas = np.logspace(- n_alpha / 2, n_alpha - (n_alpha / 2), num=n_alpha)
    ridge = timedec.linear_model.RidgeCV(alphas=alphas)
    ridge.fit(fmri_train, design_mtx_train)
    prediction_train = ridge.predict(fmri_train)
    prediction_test = ridge.predict(fmri_test)
    
    score = timedec.metrics.r2_score(
        stimuli_test, prediction_test[:,:nclasses],
        multioutput='raw_values')
    
    log = timedec.linear_model.LogisticRegressionCV()
    train_mask = np.sum(stimuli_train, axis=1).astype(bool)
    test_mask = np.sum(stimuli_test, axis=1).astype(bool)
    time_windows_train = [
        prediction_train[scan: scan + logistic_window,:nclasses].ravel()
        for scan in xrange(len(prediction_train) - logistic_window + 1)
        if train_mask[scan]]
    time_windows_test = [
        prediction_test[scan: scan + logistic_window,:nclasses].ravel()
        for scan in xrange(len(prediction_test) - logistic_window + 1)
        if test_mask[scan]]
    
    stimuli_train = np.argmax(stimuli_train[train_mask], axis=1)
    stimuli_test = np.argmax(stimuli_test[test_mask], axis=1)
    
    log.fit(time_windows_train, stimuli_train)
    accuracy = log.score(time_windows_test, stimuli_test)
    return accuracy
    
