import sys, os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import nibabel as nb
from ..behavior import load_behavior

from nipy.modalities.fmri.experimental_paradigm import BlockParadigm, EventRelatedParadigm
from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.modalities.fmri.design_matrix import dmtx_light
from nipy.modalities.fmri.hemodynamic_models import _sample_condition
default_tr = 2.16

from mvpa2.misc.fx import single_gamma_hrf
from scipy.interpolate import interp1d

def events_to_mtx(evts, frametimes, hrf_func=single_gamma_hrf, tr=default_tr, oversampling=64, time_length=32):
    dt = tr / oversampling
    hkernel = hrf_func(np.linspace(0, time_length, time_length/dt))
    hkernel /= hkernel.sum()
    regressors = []
    hr_regs = []
    for evt in evts:
        hr_regressor, hr_frametimes = _sample_condition(([evt[2]],evt[3],[1]), frametimes, oversampling=oversampling)
        conv_reg = np.convolve(hr_regressor, hkernel)[:hr_regressor.size]

        f = interp1d(hr_frametimes, conv_reg)
        regressors.append(f(frametimes))
        #regressors[-1]/=regressors[-1].sum()
    regs = np.asarray(regressors+[np.ones_like(regressors[0])]).T.astype(
        dtype=[(evt[0],np.float) for evt in evts]+[('constant', np.float)])
    return regs

def blocks_to_attributes_new(ds, blocks, hrf_rest_thresh=.2, tr=default_tr):
    # remove blocks out of scan range
    scan_len_sec = ds.nsamples*tr
    blocks = [b for b in blocks if b[3]<scan_len_sec and b[5]<scan_len_sec]
    
    instrs = [['instr_%03d_%s'%(bi,b[0]),b[0],b[2],b[3]-b[2],b[5]-b[2]] for bi,b in enumerate(blocks) if b[2]>0]
    gos = [['go_%03d_%s'%(bi,b[0]),b[0],b[3],b[4]-b[3]] for bi,b in enumerate(blocks)]
    execs = [['exec_%03d_%s'%(bi,b[0]),b[0],b[5],b[6]-b[5]] for bi,b in enumerate(blocks)]
    whole_blocks = [['block_%03d_%s'%(bi,b[0]),b[0],b[2],b[6]-b[2]] for bi,b in enumerate(blocks)]


    frametimes = ds.sa.time - ds.sa.time[0]+tr/2

    ds.sa['regressors_exec'] = events_to_mtx(instrs+execs, frametimes)
    ds.sa['regressors_stim'] = events_to_mtx(instrs+gos, frametimes)
    ds.sa['regressors_blocks'] = events_to_mtx(whole_blocks, frametimes)
    
    instrs_evt = [['instr_%03d_%s'%(bi,b[0]),b[0],b[2],0,b[5]-b[2]] for bi,b in enumerate(blocks) if b[2]>0]
    execs_evt = [['exec_%03d_%s'%(bi,b[0]),b[0],b[5],0] for bi,b in enumerate(blocks)]
    gos_evt = [['exec_%03d_%s'%(bi,b[0]),b[0],b[3],0] for bi,b in enumerate(blocks)]
    ds.sa['regressors_exec_evt'] = events_to_mtx(instrs_evt+execs_evt, frametimes)
    ds.sa['regressors_stim_evt'] = events_to_mtx(instrs_evt+gos_evt, frametimes)
    
    
    n_correct_sequences = np.asarray([sum([np.sum(s['match'])==len(b[1]) for s in b[-1]]) for b in blocks]+[-1])
    n_failed_sequences = np.asarray([sum([np.all(~s['match']) for s in b[-1]]) for b in blocks]+[-1])
    rts = [np.diff(np.hstack([s['time'] for s in b[-1]])) for b in blocks]
    rts_stats = np.asarray([[rt.mean(), rt.std()] for rt in rts])

    
    mtxs = dict(regressors_exec='',regressors_stim='_stim')#,regressors_exec_evt='_evt')
    for mtx, suffix in mtxs.items():
        mtx_vals = ds.sa[mtx].value.astype(np.float)
        reg_names = ds.sa[mtx].value.dtype.names
        targ_idx = np.argmax(mtx_vals[:,:-1],1)
        rest_mask = mtx_vals[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh*mtx_vals[:,:-1].max()

        ds.sa['targets%s'%suffix] = np.asarray(['_'.join(reg_names[i].split('_')[2:]) for i in targ_idx])
        ds.sa['targets%s'%suffix].value[rest_mask] = 'rest'
        ds.sa['subtargets%s'%suffix] = np.asarray([reg_names[i].split('_')[0] for i in targ_idx])
        ds.sa['subtargets%s'%suffix].value[rest_mask] = 'rest'
        ds.sa['blocks_idx%s'%suffix] = np.asarray([int(reg_names[i].split('_')[1]) for i in targ_idx])
        ds.sa['blocks_idx%s'%suffix].value[rest_mask] = -1

    ds.sa['sequence'] = np.asarray([''.join(blocks[i][1].astype(np.str)) for i in ds.sa.blocks_idx])
    ds.sa['n_correct_sequences'] = n_correct_sequences[ds.sa.blocks_idx]
    ds.sa['n_failed_sequences'] = n_failed_sequences[ds.sa.blocks_idx]
        

    # add time from instruction
    ds.sa['delay_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['tr_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_go'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_first_key'] = [np.nan]*ds.nsamples

    ds.sa['targets_no_delay'] = ds.targets.copy()
    ds.sa['subtargets_no_delay'] = ds.sa.subtargets.copy()
    ds.sa['blocks_idx_no_delay'] = np.zeros(ds.nsamples)-1

    last_vol = 0
    for bi,instr,go,ex in zip(range(len(blocks)),instrs, gos, execs):
        first_vol = int(np.round(instr[2]/tr+1e-4))
        prev_vol = last_vol
        last_vol = min(int(np.ceil((ex[2]+ex[3])/tr)), ds.nsamples-1)

        ds.sa.delay_from_instruction[prev_vol:last_vol+1] = np.arange(last_vol-prev_vol+1)*tr - (first_vol-prev_vol)*tr
        ds.sa.tr_from_instruction[prev_vol:last_vol] = np.arange(last_vol-prev_vol) - (first_vol-prev_vol)

        ds.sa.subtargets_no_delay[first_vol:] = 'instr'
        ds.sa.targets_no_delay[first_vol:] = blocks[bi][0]
        ds.sa.blocks_idx_no_delay[first_vol:] = bi
        
        first_vol = int(np.floor(go[2]/tr))
        ds.sa.delay_from_go[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (go[2]-first_vol*tr)

        first_vol = int(np.floor(ex[2]/tr))
        last_vol = min(int(np.ceil((ex[2]+ex[3])/tr)),ds.nsamples-1)
        ds.sa.delay_from_first_key[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (ex[2]-first_vol*tr)

        ds.sa.subtargets_no_delay[first_vol:] = 'exec'
        ds.sa.subtargets_no_delay[last_vol:] = 'rest'
        ds.sa.blocks_idx_no_delay[last_vol:] = -1
        ds.sa.targets_no_delay[last_vol:] = 'rest'

def blocks_to_attributes(ds, blocks, hrf_rest_thresh=.5, tr=default_tr):
    
    scan_len_sec = ds.nsamples*tr
    blocks = [b for b in blocks if b[3]<scan_len_sec and b[5]<scan_len_sec]
    
    instrs = np.asarray(
        [['instr_%03d_%s'%(bi,b[0]),b[0],b[2],b[3]-b[2],b[5]-b[2]] for bi,b in enumerate(blocks) if b[2]>0],
        dtype=np.object).reshape(-1,5)
    gos = np.asarray(
        [['go_%03d_%s'%(bi,b[0]),b[0],b[3],b[4]-b[3]] for bi,b in enumerate(blocks)],
        dtype=np.object)
    execs = np.asarray(
        [['exec_%03d_%s'%(bi,b[0]),b[0],b[5],b[6]-b[5]] for bi,b in enumerate(blocks)],
        dtype=np.object)

    ds.a['blocks_tr'] = np.round(np.asarray([b[2] for b in blocks])/tr).astype(np.int)
    ds.a['blocks_targets'] = [b[0] for b in blocks]
    ds.a['blocks_durations'] = [b[6]-b[2] for b in blocks]
        
    #"""
    par_exec = BlockParadigm(
        con_id = np.hstack([instrs[:,0], execs[:,0]]),
        onset = np.hstack([instrs[:,2], execs[:,2]]),
        duration = np.hstack([instrs[:,4], execs[:,3]]))

    par_stim = BlockParadigm(
        con_id = np.hstack([instrs[:,0], gos[:,0]]),
        onset = np.hstack([instrs[:,2], gos[:,2]]),
        duration = np.hstack([instrs[:,3], gos[:,3]]))
    """
    par_exec = EventRelatedParadigm(
        con_id = np.hstack([instrs[:,0], execs[:,0]]),
        onset = np.hstack([instrs[:,2], execs[:,2]]))

    par_stim = EventRelatedParadigm(
        con_id = np.hstack([instrs[:,0], gos[:,0]]),
        onset = np.hstack([instrs[:,2], gos[:,2]]))
    """
    
    frametimes = ds.sa.time-ds.sa.time[0]

    n_correct_sequences = np.asarray([sum([np.all(s['match']) for s in b[-1]]) for b in blocks]+[-1])
    n_failed_sequences = np.asarray([sum([np.all(~s['match']) for s in b[-1]]) for b in blocks]+[-1])
    rts = [np.diff(np.hstack([s['time'] for s in b[-1]])) for b in blocks]
    rts_stats = np.asarray([[rt.mean(), rt.std()] for rt in rts])
    
    mtx_exec, names_exec = dmtx_light(frametimes, par_exec, hrf_model='canonical', drift_model='blank')
    mtx_stim, names_stim = dmtx_light(frametimes, par_stim, hrf_model='canonical', drift_model='blank')

    ds.sa['regressors_exec'] = np.array(mtx_exec, dtype=[(n,np.float) for n in names_exec])
    ds.sa['regressors_stim'] = np.array(mtx_stim, dtype=[(n,np.float) for n in names_stim])

    targ_idx = np.argmax(mtx_exec[:,:-1],1)
    rest_mask = mtx_exec[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh*mtx_exec[:,:-1].max()
    ds.sa['targets'] = np.asarray(['_'.join(names_exec[i].split('_')[2:]) for i in targ_idx])
    ds.sa.targets[rest_mask] = 'rest'
    ds.sa['subtargets'] = np.asarray([names_exec[i].split('_')[0] for i in targ_idx])
    ds.sa.subtargets[rest_mask] = 'rest'
    ds.sa['blocks_idx'] = np.asarray([int(names_exec[i].split('_')[1]) for i in targ_idx])
    ds.sa.blocks_idx[rest_mask] = -1
    ds.sa['sequence'] = np.asarray([''.join(blocks[i][1].astype(np.str)) for i in ds.sa.blocks_idx])

    ds.sa['n_correct_sequences'] = n_correct_sequences[ds.sa.blocks_idx]
    ds.sa['n_failed_sequences'] = n_failed_sequences[ds.sa.blocks_idx]

    targ_idx = np.argmax(mtx_stim[:,:-1],1)
    rest_mask = mtx_stim[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh*mtx_stim[:,:-1].max()
    ds.sa['targets_stim'] = np.asarray(['_'.join(names_stim[i].split('_')[2:]) for i in targ_idx])
    ds.sa.targets_stim[rest_mask] = 'rest'
    ds.sa['subtargets_stim'] = np.asarray([names_stim[i].split('_')[0] for i in targ_idx])
    ds.sa.subtargets_stim[rest_mask] = 'rest'
    ds.sa['blocks_idx_stim'] = np.asarray([int(names_stim[i].split('_')[1]) for i in targ_idx])
    ds.sa.blocks_idx_stim[rest_mask] = -1

    # add time from instruction
    ds.sa['delay_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['tr_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_go'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_first_key'] = [np.nan]*ds.nsamples

    ds.sa['targets_no_delay'] = ds.targets.copy()
    ds.sa['subtargets_no_delay'] = ds.sa.subtargets.copy()
    ds.sa['blocks_idx_no_delay'] = np.zeros(ds.nsamples)-1

    last_vol = 0
    for bi,instr,go,ex in zip(range(len(blocks)),instrs, gos, execs):
        first_vol = int(np.round(instr[2]/tr+1e-4))
        prev_vol = last_vol
        last_vol = min(int(np.ceil((ex[2]+ex[3])/tr)), ds.nsamples-1)

        ds.sa.delay_from_instruction[prev_vol:last_vol+1] = np.arange(last_vol-prev_vol+1)*tr - (first_vol-prev_vol)*tr
        ds.sa.tr_from_instruction[prev_vol:last_vol] = np.arange(last_vol-prev_vol) - (first_vol-prev_vol)

        ds.sa.subtargets_no_delay[first_vol:] = 'instr'
        ds.sa.targets_no_delay[first_vol:] = blocks[bi][0]
        ds.sa.blocks_idx_no_delay[first_vol:] = bi
        
        first_vol = int(np.floor(go[2]/tr))
        ds.sa.delay_from_go[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (go[2]-first_vol*tr)

        first_vol = int(np.floor(ex[2]/tr))
        last_vol = min(int(np.ceil((ex[2]+ex[3])/tr)),ds.nsamples-1)
        ds.sa.delay_from_first_key[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (ex[2]-first_vol*tr)

        ds.sa.subtargets_no_delay[first_vol:] = 'exec'
        ds.sa.subtargets_no_delay[last_vol:] = 'rest'
        ds.sa.blocks_idx_no_delay[last_vol:] = -1
        ds.sa.targets_no_delay[last_vol:] = 'rest'

    
    
import h5py
import datetime 

seq_info = [('CoReTSeq',np.asarray([1,4,2,3,1])),('CoReIntSeq',np.asarray([1,3,2,4,1]))]
seq_idx = [0]*7

from mvpa2.datasets import Dataset
from mvpa2.mappers.detrend import poly_detrend
import hrf_estimation as he
from ..pipelines.wavelet_despike import wavelet_despike_loop, wavelet_despike
import scipy.ndimage

def preproc_ds(ds,
               detrend=False,
               mean_divide=False,
               median_divide=False,
               add_shift=None,
               wav_despike=False,
               wav_threshold=2.5,
               threshold_wav_low=5,
               threshold_wav_high=None,
               sg_filt=False,
               sg_filt_win=210,
               hptf=False,
               hptf_thresh=32,
               tr=default_tr):
    add_trend_chunk(ds, min_time_per_chunk=16)

    ds_mean = np.nanmean(ds.samples,0)
    ds_median = np.nanmedian(ds.samples,0)

    if mean_divide: # convert to pct change per trend chunk
        #for tc in np.unique(ds.sa.trend_chunks):
        #    ds.samples[ds.sa.trend_chunks==tc] /= np.nanmean(ds.samples[ds.sa.trend_chunks==tc],0)
        ds.samples /= ds_mean # convert to pct change
    elif median_divide:
        ds.samples /= ds_median

    if np.count_nonzero(np.isnan(ds.samples)) > 0:
        print 'Warning : dataset contains NaN, replaced with 0 and created nans_mask'
        nans_mask = np.any(np.isnan(ds.samples), 0)
        ds.fa['nans'] = nans_mask
        ds.samples[:,nans_mask] = 0

    if add_shift!=None:
        ds.samples += add_shift

    if detrend:
        polyord = (np.bincount(ds.sa.trend_chunks)>(64./tr)).astype(np.int)
        print polyord
        poly_detrend(ds, chunks_attr='trend_chunks', polyord=polyord)

    if wav_despike and ds.nsamples>10:
        # detrendind seems necessary to avoid border effects on wavelets
        poly_detrend(ds, chunks_attr=None, polyord=1)
        if ds.nsamples>500:
            ds.samples[:] = wavelet_despike_loop(
                ds.samples, threshold=wav_threshold,
                threshold_wavelet_low=threshold_wav_low,
                threshold_wavelet_high=threshold_wav_high)
        else:
            ds.samples[:] = wavelet_despike(
                ds.samples, threshold=wav_threshold,
                threshold_wavelet_low=threshold_wav_low,
                threshold_wavelet_high=threshold_wav_high)
    if sg_filt:
        sg_win = int(sg_filt_win/float(tr))
        if not sg_win%2:
            sg_win += 1
        print sg_win
        if ds.nsamples > sg_win:
            ds.samples -= he.savitzky_golay.savgol_filter(ds.samples, sg_win, 3, axis=0)
    if hptf:
        ds.samples -= scipy.ndimage.gaussian_filter1d(ds.samples,sigma=hptf_thresh,axis=0,truncate=2.5)
    

def ds_from_ts(
        ts_file,
        data_path='FMRI/DATA',
        tr=default_tr):
    ts = h5py.File(ts_file,'r')
                    
    ds = Dataset(np.transpose(ts[data_path][:]))
    print ds.shape

    ds.fa['coordinates'] = ts['COORDINATES'][:]
    ds.a['triangles'] = np.vstack([
            ts['STRUCTURES/CORTEX_LEFT/TRIANGLES'],
            ts['STRUCTURES/CORTEX_RIGHT/TRIANGLES'][:]+\
                (np.max(ts['STRUCTURES/CORTEX_LEFT/TRIANGLES'])+1)])
            
    ds.fa['node_indices'] = np.arange(ds.nfeatures,dtype=np.uint)
    if 'STRUCTURES/SUBCORTICAL_CEREBELLUM/INDICES' in ts:
        ds.fa['voxel_indices'] = np.empty((ds.nfeatures,3),dtype=np.int)
        ds.fa.voxel_indices.fill(np.nan)
        rois_offset = ts['STRUCTURES/SUBCORTICAL_CEREBELLUM/ROIS'][0,'IndexOffset']
        ds.fa.voxel_indices[rois_offset:] = ts['STRUCTURES/SUBCORTICAL_CEREBELLUM/INDICES']

    if 'scan_time' in ts[data_path].attrs and 'scan_date' in ts[data_path].attrs:
        date = ts[data_path].attrs['scan_date']
        time = ts[data_path].attrs['scan_time']
        dt = datetime.datetime.strptime(date+':'+time,'%Y%m%d:%H%M%S.%f')
        tstp = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
        ds.sa['time'] = tstp+np.arange(ds.nsamples)*tr
    else:
        ds.sa['time'] = np.arange(ds.nsamples)*tr

    return ds

def ds_set_attributes(
        ds,
        design_file=None,
        remapping=None, seq_info=None, seq_idx=None,
        default_target='rest', tr=default_tr):
    target_chunk_len = 8
    if not design_file is None:
        blocks = load_behavior(
            design_file,
            remapping=remapping,
            seq_info=seq_info,
            seq_idx=seq_idx)

        is_mvpa = 'mvpa' in design_file
        if is_mvpa:
            blocks_to_attributes_new(ds, blocks, tr=tr)
        else:
            blocks_to_attributes(ds, blocks, tr=tr)
        ds.sa['chunks'] = np.cumsum(np.ediff1d(ds.sa.blocks_idx, to_begin=[0])>0)
        chunks_count = np.bincount(ds.chunks)
        
        ds.sa['chunks'] = np.cumsum(np.ediff1d(ds.chunks, to_begin=[0])!=0)
        # rounding is to remove numerical small errors
        ds.a['blocks_tr'] = np.asarray([int(np.round(b[2]/tr)) for b in blocks])
        ds.a['blocks_targets'] = np.asarray([b[0] for b in blocks])
        ds.a['blocks_durations'] = np.asarray([b[6]-b[2] for b in blocks])
    else:
        ds.sa['chunks'] = np.arange(int(ds.nsamples/float(target_chunk_len))+1).repeat(target_chunk_len)[:ds.nsamples]
        ds.sa['targets'] = [default_target]*ds.nsamples
        ds.sa['subtargets'] = ds.sa.targets
        ds.sa['targets_stim'] = ds.sa.targets
        ds.sa['subtargets_stim'] = ds.sa.targets
        ds.sa['sequence'] = ['']*ds.nsamples

        for attr in ['n_correct_sequences',
                     'n_failed_sequences',
                     'delay_from_instruction',
                     'delay_from_first_key',
                     'delay_from_go',
                     'tr_from_instruction',
                     'blocks_idx',
                     'blocks_idx_stim',
                     'targets_no_delay',
                     'subtargets_no_delay',
                     'blocks_idx_no_delay']:
            ds.sa[attr] = [np.nan]*ds.nsamples

def add_aparc_ba_fa(ds, subject, pproc_tpl):
    pproc_path = pproc_tpl%subject
    import generic_pipelines
    roi_aparc = np.loadtxt(
        os.path.join(generic_pipelines.__path__[0],'../data','Atlas_ROIs.csv'),
        skiprows=1,
        delimiter=',')[:,-1].astype(np.int)
    
    aparcs_surf = np.hstack([nb.load(os.path.join(pproc_path,'label_resample/mapflow/_label_resample%d/%sh.aparc.a2009s.annot_converted.32k.gii'%(i,h))).darrays[0].data.astype(np.int)+11100+i*1000 for i,h in enumerate('lr')])
    ds.fa['aparc'] = np.hstack([aparcs_surf, roi_aparc]).astype(np.int32)
        
    ba_32k = np.hstack([nb.load(os.path.join(pproc_path,'BA_resample/mapflow/_BA_resample%d/%sh.BA_exvivo.annot_converted.32k.gii'%(i,h))).darrays[0].data.astype(np.int) for i,h in enumerate('lr')] + [np.zeros(len(roi_aparc))]).astype(np.int32)
    
    ba_thresh_32k = np.hstack([nb.load(os.path.join(pproc_path,'BA_thresh_resample/mapflow/_BA_thresh_resample%d/%sh.BA_exvivo.thresh.annot_converted.32k.gii'%(i,h))).darrays[0].data.astype(np.int) for i,h in enumerate('lr')] + [np.zeros(len(roi_aparc))]).astype(np.int32)
    for ba in [ba_32k, ba_thresh_32k]:
        ba[32492:2*32492] = ba[32492:2*32492]+1000*(ba[32492:2*32492]>0)
    ds.fa['ba'] = ba_32k
    ds.fa['ba_thresh'] = ba_thresh_32k

def add_trend_chunk(ds, tr=default_tr, min_time_per_chunk=32):
    ds.sa['trend_chunks'] = np.zeros(ds.nsamples)
    min_trend_chunk_len = min_time_per_chunk/float(tr)
    newchunk = np.zeros(ds.nsamples,dtype=np.bool)
    diffmean = np.nanmean(np.abs(np.diff(ds.samples,1,0)),1)
    diffmean = np.hstack([0,diffmean])
    cutoff = diffmean.mean()+2*diffmean.std()
    ds.sa['diffmean'] = diffmean.copy()
    while True:
        c = np.argmax(diffmean)
        if diffmean[c] < cutoff:
            break
        tc = ds.sa.trend_chunks[c]
        cm = ds.sa.trend_chunks==tc
        if c > 0 and np.count_nonzero(cm[:c])>min_trend_chunk_len and np.count_nonzero(cm[c:])>min_trend_chunk_len:
            around = diffmean < cutoff
            newchunk[:] = np.logical_and(cm,np.arange(len(cm))>=c)
            ds.sa.trend_chunks[newchunk] = ds.sa.trend_chunks.max()+1
        diffmean[c] = 0
                
    ds.sa.trend_chunks = np.cumsum(np.ediff1d(ds.sa.trend_chunks,to_begin=[0])!=0)

def ds_tr2glm_st(ds, regressors_attr, group_regressors, group_ignore=[],
              model='ols', sample_type='t_values', 
              hptf=None, return_resid=False):
    
    betas = []
    max_ind = []
    targets = []
    resid = None
    regs = ds.sa[regressors_attr].value.astype(np.float)
    reg_names = ds.sa[regressors_attr].value.dtype.names
    reg_groups = [n.split('_')[0] for n in reg_names]
    grouped = np.asarray([g in group_regressors for g in reg_groups])
    if not hptf is None:
        regs[:,grouped] -= gaussian_filter1d(regs[:,grouped], hptf, axis=0, truncate=2.5)

    for reg_i, reg_name in enumerate(reg_names):
        if reg_groups[reg_i] in group_ignore:
            continue
        print 'fitting %s'%reg_name
        max_ind.append(np.argmax(ds.sa[regressors_attr].value[:,reg_i].astype(np.float)))

        summed_regs = np.asarray([regs[:,np.asarray([(g==rt and n!=reg_name) for n,g in zip(reg_names, reg_groups)])].sum(1)
                                  for rt in group_regressors]).T

        mtx = np.hstack([regs[:,reg_i,np.newaxis], summed_regs, regs[:,~grouped]])
        glm = GeneralLinearModel(mtx)
        glm.fit(ds.samples, model=model)
        ctx_mtx = np.zeros(mtx.shape[-1])
        ctx_mtx[0] = 1
        if sample_type == 't_values':
            betas.append(glm.contrast(ctx_mtx).stat())
        elif sample_type == 'betas':
            betas.append(np.squeeze(glm.get_beta(0)))
        del glm, mtx, summed_regs
        
    ds_glm = Dataset(np.asarray(betas), fa=ds.fa, a=ds.a)
    for attr in ds.sa.keys():
        if 'regressor' in attr:
            continue
        ds_glm.sa[attr] = ds.sa[attr].value[max_ind]
    ds_glm.sa['chunks'] = np.arange(ds_glm.nsamples)

    if return_resid and sample_type=='betas':
        resid = ds.samples-np.dot(regs[:, grouped], ds_glm.samples)
        resid -= resid.mean(0)
        ds_resid = Dataset(resid, sa=ds.sa, fa=ds.fa, a=ds.a)
        return ds_glm, ds_resid
    return ds_glm


def ds_tr2glm(ds, regressors_attr, group_regressors, group_ignore=[],
                  model='ols', sample_type='t_values', 
                  hptf=None, return_resid=False,nproc=10):
    
    betas = []
    max_ind = []
    targets = []
    resid = None
    regs = ds.sa[regressors_attr].value.astype(ds.samples.dtype)
    reg_names = ds.sa[regressors_attr].value.dtype.names
    reg_groups = [n.split('_')[0] for n in reg_names]
    grouped = np.asarray([g in group_regressors for g in reg_groups])
    if not hptf is None:
        regs[:,grouped] -= gaussian_filter1d(regs[:,grouped], hptf, axis=0, truncate=2.5)

    def reg_glm(reg_i, reg_name):
        print 'fitting %s'%reg_name

        summed_regs = np.asarray([regs[:,np.asarray([(g==rt and n!=reg_name) for n,g in zip(reg_names, reg_groups)])].sum(1)
                                  for rt in group_regressors]).T

        mtx = np.hstack([regs[:,reg_i,np.newaxis], summed_regs, regs[:,~grouped]])
        glm = GeneralLinearModel(mtx)
        glm.fit(ds.samples, model=model)
        ctx_mtx = np.zeros(mtx.shape[-1])
        ctx_mtx[0] = 1
        if sample_type == 't_values':
            return glm.contrast(ctx_mtx).stat()
        elif sample_type == 'betas':
            return np.squeeze(glm.get_beta(0))

    import pprocess
    betas = pprocess.Map(limit=nproc)
    compute = betas.manage(pprocess.MakeParallel(reg_glm))
    for reg_i, reg_name in enumerate(reg_names):
        if reg_groups[reg_i] in group_ignore:
            continue
        max_ind.append(np.argmax(ds.sa[regressors_attr].value[:,reg_i].astype(np.float)))
        compute(reg_i, reg_name)

    betas = [b for b in betas]
    ds_glm = Dataset(np.asarray(betas, dtype=ds.samples.dtype), fa=ds.fa, a=ds.a)
    for attr in ds.sa.keys():
        if 'regressor' in attr:
            continue
        ds_glm.sa[attr] = ds.sa[attr].value[max_ind]
    ds_glm.sa['chunks'] = np.arange(ds_glm.nsamples)

    if return_resid and sample_type=='betas':
        resid = ds.samples-np.dot(regs[:, grouped], ds_glm.samples)
        resid -= resid.mean(0)
        ds_resid = Dataset(resid.astype(ds.samples.dtype), sa=ds.sa, fa=ds.fa, a=ds.a)
        return ds_glm, ds_resid
    return ds_glm


def ds_to_conn(ds):
    import scipy.sparse
    max_vertex = ds.a.triangles.max()+1
    vox_idx = ds.fa.voxel_indices[max_vertex:]
    row,col = [],[]
    for i,vox in enumerate(vox_idx):
        #sbset = (vox_idx == vox).sum(1) > 1
        #sbset[sbset] = np.all(np.abs(vox_idx[sbset]-vox)<=1, 1)
        sbset = np.abs(vox_idx-vox).sum(1) == 1
        col.append(np.where(sbset)[0]+max_vertex)
        row.append(np.ones(col[-1].size)*(i+max_vertex))
    row = np.hstack(row)
    col = np.hstack(col)
    

    conn = scipy.sparse.coo_matrix((
        np.ones(3*ds.a.triangles.shape[0]+row.size),
        (np.hstack([ds.a.triangles[:,:2].T.ravel(),ds.a.triangles[:,1],row]),
         np.hstack([ds.a.triangles[:,1:].T.ravel(),ds.a.triangles[:,2],col]))))
    conn = conn+conn.T
    conn.data.fill(1)
    return conn


def interp_bad_ts(ds, smooth_size=4, ratio=1.5):
    import surfer.utils as surfutils
    conn = ds_to_conn(ds)
    smooth_mat = surfutils.smoothing_matrix(np.arange(ds.nfeatures), conn, smooth_size)
    conn_ext = (smooth_mat>0).astype(np.float)
    
    tss_std = ds.samples.std(0)
    tss_std_locmean = np.asarray((conn_ext.dot(tss_std)+tss_std)/(np.squeeze(conn_ext.sum(1))+1))

    good_vox = np.squeeze(tss_std < ratio*tss_std_locmean)
    ds.fa['good_voxels'] = good_vox

    smooth_mat = surfutils.smoothing_matrix(np.where(good_vox)[0], conn, smooth_size)
    tss_sm = smooth_mat.dot(ds.samples[:,good_vox].T).T
    tss_sm[:,good_vox]= ds.samples[:,good_vox]
    return Dataset(tss_sm, a=ds.a, sa=ds.sa, fa=ds.fa)
    

def split_mvpa(ds, regressors_attr):
    regs_orig = ds.sa[regressors_attr].value
    regs = regs_orig.astype(np.float)
    regs_mask1 = np.ones(regs.shape[1],dtype=np.bool)
    regs_mask1[:-1] = np.tile(np.arange(32),2) < 16
    regs_mask2 = np.ones(regs.shape[1],dtype=np.bool)
    regs_mask2[:-1] = np.logical_not(regs_mask1[:-1])
    
    regs_sum_split1 = np.abs(regs[:,regs_mask1]).sum(1)
    split1_last_idx = np.argwhere(regs_sum_split1>1).flatten()[-1]+1
    regs_sum_split2 = np.abs(regs[:,regs_mask2]).sum(1)
    split2_first_idx = np.argwhere(regs_sum_split2>1).flatten()[0]-1

    ds_split1 = ds[:split1_last_idx]
    ds_split1.sa[regressors_attr] = regs[:split1_last_idx, regs_mask1].astype(
        dtype=[(n,np.float) for n in np.asarray(regs_orig.dtype.names)[regs_mask1]])
    ds_split2 = ds[split2_first_idx:]
    ds_split2.sa[regressors_attr] = regs[split2_first_idx:, regs_mask2].astype(
        dtype=[(n,np.float) for n in np.asarray(regs_orig.dtype.names)[regs_mask2]])
    return ds_split1, ds_split2
    
