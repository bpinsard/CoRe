import sys, os
import numpy as np
import nibabel as nb
from ..behavior import load_behavior

from nipy.modalities.fmri.experimental_paradigm import BlockParadigm, EventRelatedParadigm
from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.modalities.fmri.design_matrix import dmtx_light

default_tr = 2.16

def blocks_to_attributes(ds, blocks, hrf_rest_thresh=.2, tr=default_tr):
    
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
        
    par_exec = BlockParadigm(
        con_id = np.hstack([instrs[:,0], execs[:,0]]),
        onset = np.hstack([instrs[:,2], execs[:,2]]),
        duration = np.hstack([instrs[:,4], execs[:,3]]))

    par_stim = BlockParadigm(
        con_id = np.hstack([instrs[:,0], gos[:,0]]),
        onset = np.hstack([instrs[:,2], gos[:,2]]),
        duration = np.hstack([instrs[:,3], gos[:,3]]))
    
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
    ds.sa['targets'] = np.asarray(['_'.join(names_exec[i].split('_')[2:]) for i in targ_idx])
    ds.sa.targets[mtx_exec[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'
    ds.sa['subtargets'] = np.asarray([names_exec[i].split('_')[0] for i in targ_idx])
    ds.sa.subtargets[mtx_exec[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'
    ds.sa['blocks_idx'] = np.asarray([int(names_exec[i].split('_')[1]) for i in targ_idx])
    ds.sa.blocks_idx[mtx_exec[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = -1
    ds.sa['sequence'] = np.asarray([''.join(blocks[i][1].astype(np.str)) for i in ds.sa.blocks_idx])

    ds.sa['n_correct_sequences'] = n_correct_sequences[ds.sa.blocks_idx]
    ds.sa['n_failed_sequences'] = n_failed_sequences[ds.sa.blocks_idx]

    targ_idx = np.argmax(mtx_stim[:,:-1],1)
    ds.sa['targets_stim'] = np.asarray(['_'.join(names_stim[i].split('_')[2:]) for i in targ_idx])
    ds.sa.targets_stim[mtx_stim[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'
    ds.sa['subtargets_stim'] = np.asarray([names_stim[i].split('_')[0] for i in targ_idx])
    ds.sa.subtargets_stim[mtx_stim[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'

    # add time from instruction
    ds.sa['delay_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['tr_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_go'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_first_key'] = [np.nan]*ds.nsamples

    ds.sa['targets_no_delay'] = ds.targets.copy()
    ds.sa['blocks_idx_no_delay'] = np.zeros(ds.nsamples)-1
    for bi,b in enumerate(blocks):
        if b[2]>0:
            stim_tr = int(np.round(b[2]/tr))
        else:
            stim_tr = int(np.round(b[3]/tr))
        ds.sa.targets_no_delay[stim_tr:] = b[0]
        ds.sa.blocks_idx_no_delay[stim_tr:] = bi

    last_vol = 0
    for instr,go,ex in zip(instrs, gos, execs):
        first_vol = int(np.round(instr[2]/tr+1e-4))
        prev_vol = last_vol
        last_vol = min(int(np.ceil((go[2]+go[3])/tr)),ds.nsamples-1)
        ds.sa.delay_from_instruction[prev_vol:last_vol+1] = np.arange(last_vol-prev_vol+1)*tr - (first_vol-prev_vol)*tr
        ds.sa.tr_from_instruction[prev_vol:last_vol] = np.arange(last_vol-prev_vol) - (first_vol-prev_vol)
        
        first_vol = int(np.floor(go[2]/tr))
        ds.sa.delay_from_go[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (go[2]-first_vol*tr)

        first_vol = int(np.floor(ex[2]/tr))
        last_vol = min(int(np.ceil((ex[2]+ex[3])/tr)),ds.nsamples-1)
        ds.sa.delay_from_first_key[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (ex[2]-first_vol*tr)

    
    
import h5py
import datetime 

seq_info = [('CoReTSeq',np.asarray([1,4,2,3,1])),('CoReIntSeq',np.asarray([1,3,2,4,1]))]
seq_idx = [0]*7

from mvpa2.datasets import Dataset
from mvpa2.mappers.detrend import poly_detrend

def ds_from_ts(ts_file, design_file=None,
               remapping=None, seq_info=None, seq_idx=None,
               default_target='rest', tr=default_tr, data_path='FMRI/DATA'):

    ts = h5py.File(ts_file,'r')    
    ds = Dataset(np.transpose(ts[data_path]))
    if np.count_nonzero(np.isnan(ds.samples)) > 0:
        print 'Warning : dataset contains NaN, replaced with 0 and created nans_mask'
        nans_mask = np.any(np.isnan(ds.samples), 0)
        ds.fa['nans'] = nans_mask
        ds.samples[:,nans_mask] = 0

    add_trend_chunk(ds)
    polyord = (np.bincount(ds.sa.trend_chunks)>(64./tr)).astype(np.int)
    poly_detrend(ds, chunks_attr='trend_chunks', polyord=polyord)
    
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

    target_chunk_len = 6
    if not design_file is None:
        blocks = load_behavior(
            design_file,
            remapping=remapping,
            seq_info=seq_info,
            seq_idx=seq_idx)
        blocks_to_attributes(ds, blocks, tr=tr)
        ds.sa['chunks'] = np.hstack([[0],np.cumsum(ds.sa.targets[:-1]!=ds.sa.targets[1:])])
        chunks_count = np.bincount(ds.chunks)
        for chk in np.where(chunks_count>2*target_chunk_len)[0]:
            ds.chunks[ds.chunks==chk] = chk+(np.arange(target_chunk_len)*1000).repeat(
                int(np.ceil(chunks_count[chk]/float(target_chunk_len))))[:chunks_count[chk]]
        
        ds.sa['chunks'] = np.cumsum(np.ediff1d(ds.chunks, to_begin=[0])!=0)
        # rounding is to remove numerical small errors
        ds.a['blocks_tr'] = [int(np.round(b[2]/tr)) for b in blocks]
        ds.a['blocks_targets'] = [b[0] for b in blocks]
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
                     'targets_no_delay',
                     'blocks_idx_no_delay']:
            ds.sa[attr] = [np.nan]*ds.nsamples
    return ds

def add_aparc_ba_fa(ds, subject, pproc_tpl):
    pproc_path = pproc_tpl%subject
    import generic_pipelines
    roi_aparc = np.loadtxt(
        os.path.join(generic_pipelines.__path__[0],'../data','Atlas_ROIs.csv'),
        skiprows=1,
        delimiter=',')[:,-1].astype(np.int)
    
    aparcs_surf = np.hstack([nb.gifti.read(os.path.join(pproc_path,'label_resample/mapflow/_label_resample%d/%sh.aparc.a2009s.annot_converted.32k.gii'%(i,h))).darrays[0].data.astype(np.int)+11100+i*1000 for i,h in enumerate('lr')])
    ds.fa['aparc'] = np.hstack([aparcs_surf, roi_aparc]).astype(np.int32)
        
    ba_32k = np.hstack([nb.gifti.read(os.path.join(pproc_path,'BA_resample/mapflow/_BA_resample%d/%sh.BA_exvivo.annot_converted.32k.gii'%(i,h))).darrays[0].data.astype(np.int) for i,h in enumerate('lr')] + [np.zeros(len(roi_aparc))]).astype(np.int32)
    ba_thresh_32k = np.hstack([nb.gifti.read(os.path.join(pproc_path,'BA_thresh_resample/mapflow/_BA_thresh_resample%d/%sh.BA_exvivo.thresh.annot_converted.32k.gii'%(i,h))).darrays[0].data.astype(np.int) for i,h in enumerate('lr')] + [np.zeros(len(roi_aparc))]).astype(np.int32)
    ds.fa['ba'] = ba_32k
    ds.fa['ba_thres'] = ba_thresh_32k

def add_trend_chunk(ds,tr=default_tr):
    ds.sa['trend_chunks'] = np.zeros(ds.nsamples)
    min_trend_chunk_len = 32./tr
    newchunk = np.zeros(ds.nsamples,dtype=np.bool)
    diffmean = np.mean(np.abs(np.diff(ds.samples,1,0)),1)
    diffmean = np.hstack([0,diffmean])
    cutoff = diffmean.mean()+2*diffmean.std()
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
                
    ds.sa.trend_chunks = np.cumsum(np.ediff1d(ds.sa.trend_chunks,to_begin=[0])>0)

def ds_tr2glm(ds, regressors_attr, group_regressors):
    
    betas = []
    max_ind = []
    targets = []
    for reg_i, reg_name in enumerate(ds.sa[regressors_attr].value.dtype.names[:-1]):
        print 'fitting %s'%reg_name
        max_ind.append(np.argmax(ds.sa[regressors_attr].value[:,reg_i].astype(np.float)))

        summed_regs = np.asarray([ds.sa[regressors_attr].value.astype(np.float)[:,np.asarray([(n.split('_')[0]==rt and n!=reg_name) for n in  ds.sa[regressors_attr].value.dtype.names])].sum(1) for rt in group_regressors]).T
        mtx = np.hstack([ds.sa[regressors_attr].value[:,reg_i,np.newaxis].astype(np.float), summed_regs])
        glm = GeneralLinearModel(mtx)
        glm.fit(ds.samples)
        betas.append(np.squeeze(glm.get_beta(0)))
        del glm, mtx, summed_regs
        
    ds_glm = Dataset(np.asarray(betas), fa=ds.fa, a=ds.a)
    for attr in ['targets','subtargets','time','scan_id','scan_name']:
        ds_glm.sa[attr] = ds.sa[attr].value[max_ind]

    ds_glm.sa['chunks'] = np.arange(ds_glm.nsamples)
    return ds_glm



