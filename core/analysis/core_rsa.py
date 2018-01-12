import sys, os, glob
import numpy as np
import scipy.stats, scipy.ndimage.measurements, scipy.sparse
from ..mvpa import searchlight
from ..mvpa import dataset as mvpa_ds
from . import mvpa_nodes
from .core_sleep import targets_num
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
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.splitters import Splitter
from mvpa2.algorithms.group_clusterthr import (GroupClusterThreshold, Counter, 
                                               get_cluster_sizes, _transform_to_pvals, _clusterize_custom_neighborhood)
import statsmodels.stats.multitest as smm
from mvpa2 import debug
import joblib
import __builtin__

preproc_dir = '/home/bpinsard/data/analysis/core_sleep'
#dataset_subdir = 'dataset_noisecorr'
#dataset_subdir = 'dataset_smoothed'
dataset_subdir = 'dataset_wb_hptf'
dataset_subdir = 'dataset_mvpa_moco_bc_hptf'
            
proc_dir = '/home/bpinsard/data/analysis/core_mvpa'
#output_subdir = 'searchlight_wb_hptf'
#output_subdir = 'searchlight_cnbis_mnorm'
output_subdir = 'searchlight_cnbis_newmoco'
compression= 'gzip'

subject_ids = [1, 11, 23, 22, 63, 50, 79, 54, 107, 128, 162, 102, 82, 155, 100, 94, 87, 192, 195, 220, 223, 235, 268, 267,237,296]
#subject_ids = subject_ids[:-1]
#subject_ids = [296]
group_Int = [1,23,63,79,82,87,100,107,128,192,195,220,223,235,268,267,237,296]
ulabels = ['CoReTSeq','CoReIntSeq','mvpa_CoReOtherSeq1','mvpa_CoReOtherSeq2','rest']
#ulabels = ulabels[1:]


mean_gpi_tseq_intseq_retest = dict(zip([1,23,63,79,107,128,82,100,87,192,195,220,223,235,268,267,237,296],
                                       [0.691678762,0.6476318601,0.7146664634,0.6942514435,0.6439363702,0.7129282095,
                                        0.7786872682,0.6831603428,0.655332343,0.6189194603,0.7243437075,0.6760921572,
                                        0.7275203957,0.7322032598,0.7174293011,0.7663752402,0.6448094237,0.6884314187]))

seq_groups = {
    'mvpa_new_seqs' : ulabels[2:4],
    'tseq_intseq' : ulabels[:2],
    'all_seqs': ulabels[:4]
}
block_phases = [
    'instr',
    'exec'
]

def subject_searchlight_rsa_euc(sid):
    print('______________   CoRe %03d   ___________'%sid)
    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'glm_ds_%d.h5'%sid))
    ds = ds_glm
    if 'node_indices' not in ds_glm.fa.keys():
        ds_glm.fa['node_indices'] =  np.arange(ds_glm.nfeatures, dtype=np.uint)
        #ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))
        #targets_num(ds, ulabels)
    targets_num(ds_glm, ulabels)

    mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'd3_mvpa' in n]
    if len(mvpa_scan_names)==0:
        mvpa_scan_names = [n for n in np.unique(ds.sa.scan_name) if 'mvpa' in n]
    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
    ds_glm_mvpa.sa['chunks'] = np.tile(np.arange(4)[:,np.newaxis],(2,16)).ravel()

    svqe = searchlight.SurfVoxQueryEngine(max_feat=64, vox_sl_radius=2.4, surf_sl_radius=25)
    svqe_cached = searchlight.CachedQueryEngineAlt(svqe)

    part = CustomPartitioner([([a],[b]) for a in range(4) for b in range(a+1,4)],attr='chunks')
    cnbis_sl = CrossNobisSearchlight(part, svqe_cached, space='targets', nproc=10)

    mgs = mean_group_sample(attrs=['targets','chunks'])
    block_phases = ['instr','exec']

    for st in block_phases:
        ds_tmp = mgs(ds_glm_mvpa[dict(subtargets=[st])])
        slmap_cnbis = cnbis_sl(ds_tmp)
        slmap_cnbis.save(os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,st)),
                         compression=compression)


def subject_mvpa_ds_fir_residuals(sid, fir_delays, hptf_thresh=8):

    ts_files = [ os.path.join(preproc_dir, '_subject_id_%d'%sid, 'moco_bc_mvpa_aniso','mapflow',
                              '_moco_bc_mvpa_aniso%d'%scan_id,'ts.h5') for scan_id in range(2)]
    ds_mvpa = [mvpa_ds.ds_from_ts(f) for f in ts_files]
    glm_ds_mvpa = []
    residuals_mvpa = []
    n_fir_delays = len(fir_delays)
    for dsi, ds in enumerate(ds_mvpa):
        beh_file = sorted(glob.glob('/home/bpinsard/data/raw/UNF/CoRe/Behavior/CoRe_%03d_D3/CoRe_%03d_mvpa-%d-D-Three_*.mat'%(sid,sid,dsi+1)))[-1]
        beh = mvpa_ds.load_behavior(beh_file)
        evt_times = np.asarray([b[2] for b in beh])
        evt_frames = np.round(evt_times/mvpa_ds.default_tr).astype(np.int)
        evt_type = [b[0] for b in beh]
        seq_no = np.asarray([ulabels.index(l) for l in evt_type])
        mtx = np.zeros((ds.nsamples,4*n_fir_delays+1))
        mtx[:,-1] = 1 #constant
        for sno in range(4):
            for d in fir_delays:
                mtx[evt_frames[seq_no==sno]+d,sno*n_fir_delays+d] = 1
        cut = evt_frames[15]+fir_delays[-1]
        
        mvpa_ds.preproc_ds(ds, detrend=True, hptf=True, hptf_thresh=hptf_thresh)

        mtx_part1 = mtx[:cut]
        mtx_part2 = mtx[cut:]
        mtx_pinv_part1 = np.linalg.pinv(mtx_part1)
        mtx_pinv_part2 = np.linalg.pinv(mtx_part2)
        ds_part1 = ds[:cut]
        ds_part2 = ds[cut:]
        betas_part1 = mtx_pinv_part1.dot(ds_part1.samples)
        betas_part2 = mtx_pinv_part2.dot(ds_part2.samples)
        glm_ds_part1 = Dataset(betas_part1[:-1],
                               sa=dict(targets=np.repeat(ulabels[:4],n_fir_delays),
                                       targets_num=np.repeat(np.arange(4),n_fir_delays),
                                       fir_delays=np.tile(fir_delays,4)),
                               fa=ds.fa,
                               a=ds.a)
        glm_ds_part2 = Dataset(betas_part2[:-1],
                               sa=glm_ds_part1.sa,
                               fa=ds.fa,
                               a=ds.a)
        resid_part1 = Dataset(ds_part1.samples - mtx_part1.dot(betas_part1))
        resid_part2 = Dataset(ds_part2.samples - mtx_part2.dot(betas_part2))

        glm_ds_part1.sa['chunks'] = [dsi*2]*glm_ds_part1.nsamples
        glm_ds_part2.sa['chunks'] = [dsi*2+1]*glm_ds_part2.nsamples
        resid_part1.sa['chunks'] = [dsi*2]*resid_part1.nsamples
        resid_part2.sa['chunks'] = [dsi*2+1]*resid_part2.nsamples

        glm_ds_mvpa.append(glm_ds_part1)
        residuals_mvpa.append(resid_part1)
        glm_ds_mvpa.append(glm_ds_part2)
        residuals_mvpa.append(resid_part2)

        del ds
        
    glm_ds_mvpa = vstack(glm_ds_mvpa, a='drop_nonunique')
    mvpa_ds.add_aparc_ba_fa(glm_ds_mvpa,sid,os.path.join(preproc_dir, 'surface_32k', '_subject_id_%s'))
    residuals_mvpa = vstack(residuals_mvpa, a='drop_nonunique')
    residuals_mvpa.fa = glm_ds_mvpa.fa
    return glm_ds_mvpa, residuals_mvpa

def subject_mvpa_ds_residuals(sid, hptf_thresh=8, reg_sa='regressors_exec'):

    ts_files = [ os.path.join(preproc_dir, '_subject_id_%d'%sid, 'moco_bc_mvpa_aniso_new','mapflow',
                              '_moco_bc_mvpa_aniso_new%d'%scan_id,'ts.h5') for scan_id in range(2)]
    ds_mvpa = [mvpa_ds.ds_from_ts(f) for f in ts_files]
    dss_mvpa = []
    glm_ds_mvpa = []
    residuals_mvpa = []
    for dsi, ds in enumerate(ds_mvpa):
        mvpa_ds.ds_set_attributes(
            ds, sorted(glob.glob('/home/bpinsard/data/raw/UNF/CoRe/Behavior/CoRe_%03d_D3/CoRe_%03d_mvpa-%d-D-Three_*.mat'%(sid,sid,dsi+1)))[-1])
        mvpa_ds.preproc_ds(ds, detrend=True, hptf=True, hptf_thresh=hptf_thresh)
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
        
        ds_part1.sa[reg_sa] = regs[:last_part1, chunks<=0].astype(
            [(n,np.float )for n,c in zip(reg_names,chunks) if c<=0])
        glm_ds_part1, residuals_part1 = mvpa_ds.ds_tr2glm(
            ds_part1,reg_sa,['instr','exec'],['constant'],
            sample_type='betas', return_resid=True)#, hptf=hptf_thresh)
        glm_ds_part1.sa.chunks = [dsi*2]*glm_ds_part1.nsamples
        residuals_part1.sa['chunks'] = [dsi*2]*residuals_part1.nsamples
        
        dss_mvpa.append(ds_part1)
        glm_ds_mvpa.append(glm_ds_part1)
        residuals_mvpa.append(residuals_part1)

        ds_part2 = ds[first_part2:]
        ds_part2.sa[reg_sa] = regs[first_part2:,chunks>=0].astype(
            [(n,np.float )for n,c in zip(reg_names,chunks) if c>=0])
        glm_ds_part2, residuals_part2 = mvpa_ds.ds_tr2glm(
            ds_part2,reg_sa,['instr','exec'],['constant'],
            sample_type='betas',return_resid=True)#, hptf=hptf_thresh)
        glm_ds_part2.sa['chunks'] = [dsi*2+1]*glm_ds_part1.nsamples
        residuals_part2.sa['chunks'] = [dsi*2+1]*residuals_part2.nsamples
        
        dss_mvpa.append(ds_part2)
        glm_ds_mvpa.append(glm_ds_part2)
        residuals_mvpa.append(residuals_part2)

        del ds_part1, ds_part2
    del ds_mvpa

    for res in residuals_mvpa:
        for k in res.sa.keys():
            if len(res.sa[k].value.dtype)>0:
                del res.sa[k]

    glm_ds_mvpa = vstack(glm_ds_mvpa, a='drop_nonunique')
    residuals_mvpa = vstack(residuals_mvpa, a='drop_nonunique')
    return glm_ds_mvpa, residuals_mvpa

def subject_rois_rsa_crossnobis(sid, hptf_thresh=8, reg_sa='regressors_exec', fir_delays=None):
    import sklearn.covariance
    print('______________   CoRe %03d   ___________'%sid)

    if fir_delays is None:
        glm_ds_mvpa, residuals_mvpa = subject_mvpa_ds_residuals(sid, hptf_thresh, reg_sa=reg_sa)
        subsets = [dict(subtargets=[block_phase]) for block_phase in block_phases]
    else:
        glm_ds_mvpa, residuals_mvpa = subject_mvpa_ds_fir_residuals(sid, fir_delays,hptf_thresh)
        subsets = [dict(fir_delays=[fir_delay]) for fir_delay in fir_delays]
    
#def subject_rois_rsa_crossnobis_tmp(glm_ds_mvpa, residuals_mvpa):
    import sklearn.covariance

    #mgs = mean_group_sample(attrs=['subtargets','targets','chunks'])
    #glm_ds_mvpa = mgs(glm_ds_mvpa)

    ### FOR TEST ###
    """
    svqe = searchlight.SurfVoxQueryEngine(max_feat=64, vox_sl_radius=3.2, surf_sl_radius=15)
    svqe.train(glm_ds_mvpa)
    neigh = svqe.query_byid(0)
    rois = Dataset(np.zeros((1,glm_ds_mvpa.nfeatures),dtype=np.uint))
    rois.samples[:,neigh] = 1
    rois.a['roi_labels'] = ['test']
    """
    
    targets_num(glm_ds_mvpa, ulabels)
    splitter = Splitter(attr='chunks')
    partnr = CustomPartitioner([([a],[b]) for a in range(4) for b in range(a+1,4)], attr='chunks')
    part_splitter = Splitter(attr=partnr.space)

    rois_results = dict()
    rois = Dataset.from_hdf5(os.path.join(proc_dir,'msl_rois_new.h5'))

    for ri,roi_name in enumerate(rois.a.roi_labels):
        print roi_name
        mask = rois.samples[0]==ri+1
        res = residuals_mvpa[:,mask]
        betas = glm_ds_mvpa[:,mask]
        dists_mnorm = []
        dists_targets = []
        nfeats = betas.nfeatures

        print 'multivariate normalization'
        for res_split, betas_split in zip(splitter.generate(res),splitter.generate(betas)):
            emp_cov = sklearn.covariance.empirical_covariance(res_split.samples)
            shrinkage = sklearn.covariance.ledoit_wolf_shrinkage(res_split.samples, assume_centered=True)
            #print shrinkage

            cov_shrink = sklearn.covariance.shrunk_covariance(emp_cov, shrinkage=shrinkage)
            cov_eigval, cov_eigvec = np.linalg.eigh(cov_shrink)
            cov_powminushalf = cov_eigvec.dot((cov_eigvec/np.sqrt(cov_eigval)).T)

            for subset in subsets:
                betas_phase = betas_split[subset]
                for bi,beta in enumerate(betas_phase):
                    for beta2 in betas_phase[:bi]:
                        diff = beta.samples - beta2.samples
                        targ = (beta.sa.targets_num[0], beta2.sa.targets_num[0])
                        if targ[1] < targ[0]:
                            diff *= -1
                            targ = targ[::-1]
                        diff_mnorm = Dataset(np.dot(diff,cov_powminushalf),sa=beta.sa.copy(), fa=beta.fa, a=beta.a)
                        dists_targets.append(targ)
                        dists_mnorm.append(diff_mnorm)
        dists_mnorm = vstack(dists_mnorm)
        dists_mnorm.sa['targets'] = dists_targets
        
        upair_targets = sorted(list(set(dists_targets)))
        n_pair_targets = len(upair_targets)
        pair_targets2num = dict([(upt,upti) for upti,upt in enumerate(upair_targets)])
        dists_mnorm.sa['targets_num'] = np.asarray([pair_targets2num[pt] for pt in dists_targets])

        parts = list(partnr.generate(dists_mnorm))
        nparts = len(parts)
        results = dict(
            (p.values()[0][0],Dataset(np.empty((n_pair_targets*nparts), dtype=glm_ds_mvpa.samples.dtype),
                                      sa=dict(targets=upair_targets*nparts))) for p in subsets)

        print 'compute dists'
        for part_idx, part in enumerate(parts):
            for subset in subsets:
                train, test = list(part_splitter.generate(part[subset]))[1:3]
                for pt_num,pair_target in enumerate(upair_targets):
                    train_pairs = train[train.sa.targets_num==pt_num]
                    test_pairs = test[test.sa.targets_num==pt_num]
                    corr = np.dot(train_pairs.samples,test_pairs.samples.T)/nfeats
                    results[subset.values()[0][0]].samples[part_idx*n_pair_targets+pt_num] = corr.mean()
        rois_results[roi_name] = results
    return rois_results
    

def subject_searchlight_rsa_crossnobis(sid, hptf_thresh=8, reg_sa='regressors_exec'):
    print('______________   CoRe %03d   ___________'%sid)

    glm_ds_mvpa, residuals_mvpa = subject_mvpa_ds_residuals(sid, hptf_thresh, reg_sa=reg_sa)

    svqe = searchlight.SurfVoxQueryEngine(max_feat=64, vox_sl_radius=2.5, surf_sl_radius=15)
    svqe_cached = searchlight.CachedQueryEngineAlt(svqe)
    svqe_cached.train(glm_ds_mvpa)

    part = CustomPartitioner([([a],[b]) for a in range(4) for b in range(a+1,4)],attr='chunks')
    #part = CustomPartitioner([([a],[b]) for a in range(4) for b in range(4) if a!=b],attr='chunks')
    cnbis_sl = CrossNobisSearchlight(part, svqe_cached, space='targets', nproc=2, enable_ca=['roi_sizes'])

    mgs = mean_group_sample(attrs=['targets','chunks'])
    block_phases = ['exec','instr']
    #block_phases = ['exec']
    cnbis_sl.nproc=1
    cnbis_sl.train(residuals_mvpa)
    cnbis_sl.nproc=2
    print('trained')

    for st in block_phases:
        print(st)
        ds_tmp = glm_ds_mvpa[dict(subtargets=[st])]
        slmap_cnbis = cnbis_sl(mgs(ds_tmp))
        slmap_cnbis.save(os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,st)),
                         compression=compression)

def subject_searchlight_rsa_crossnobis_fir(sid, hptf_thresh=8,
                                           fir_delays=range(0,9)):
    print('______________   CoRe %03d   ___________'%sid)

    glm_ds_mvpa, residuals_mvpa = subject_mvpa_ds_fir_residuals(sid, fir_delays=fir_delays,hptf_thresh=hptf_thresh)

    svqe = searchlight.SurfVoxQueryEngine(max_feat=64, vox_sl_radius=2.5, surf_sl_radius=15)
    svqe_cached = searchlight.CachedQueryEngineAlt(svqe)
    svqe_cached.train(glm_ds_mvpa)

    part = CustomPartitioner([([a],[b]) for a in range(4) for b in range(a+1,4)],attr='chunks')
    cnbis_sl = CrossNobisSearchlight(part, svqe_cached, space='targets', nproc=1, enable_ca=['roi_sizes'])

    mgs = mean_group_sample(attrs=['targets','chunks'])
    cnbis_sl.train(residuals_mvpa)
    cnbis_sl.nproc = 2
    print('trained')

    delays_slmap = [cnbis_sl(glm_ds_mvpa[dict(fir_delays=[d])]) for d in fir_delays]
    for d,sl in zip(fir_delays,delays_slmap):
        sl.sa['fir_delay'] = [d]*sl.nsamples
    delays_slmap = vstack(delays_slmap)
    delays_slmap.save(os.path.join(proc_dir, 'searchlight_cnbis_delays_slmap','CoRe_%03d_cnbis_delays.h5'%(sid)),
                      compression=compression)


def block_argsort(data, idxes, axis=0):
    idx = np.argsort(data, axis=axis, kind='quicksort')
    return idx[idxes]

def group_rsa_cnbis_cluster(block_phase='exec',groupInt=None,
                            main_fxs=[0,5], contrasts=[(0,5)],
                            nperm=1000, voxp=.001, fwe_rate=.01, multicomp_correction='fdr_bh',blocksize=10000):
    
    
    if groupInt is not None:
        files = [os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,block_phase)) \
                 for sid in subject_ids if sid in group_Int]
    else:
        files = [os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,block_phase)) \
                 for sid in subject_ids]    

    sl_ress = np.asarray([Dataset.from_hdf5(f).samples.reshape(6,6,-1).mean(0).astype(np.float32) for f in files])
    nsubj = len(sl_ress)

    neighborhood = np.load(os.path.join(proc_dir,'connectivity_96k.npy')).tolist()
    results = dict()
    results['main_fx'] = dict()
    nfeat = sl_ress.shape[-1]
    permttest = np.empty((nperm, nfeat),dtype=np.float32)

    thr_permidx = int(voxp*nperm)
    if thr_permidx == 0:
        raise ValueError('not enough permutation to compute this pvalue')

    for main_fx in main_fxs:
        print('main_fx',main_fx)
        data = sl_ress[:,main_fx]
        #t,p = scipy.stats.ttest_1samp(data, 0, 0)
        t,p = data.mean(0), None
        # include real contrast for lower-bound on p-values
        for i in range(nperm-1):
            sys.stdout.write('\r permutations: %d/%d' % (i,nperm))
            sys.stdout.flush()
            signs = np.random.randint(0,2,nsubj)*2-1
            #permttest[i], _ = scipy.stats.ttest_1samp(data*signs[:,np.newaxis], 0, 0)
            permttest[i] = reduce(lambda x,y: x+y[0]*y[1], zip(signs,data),0)/nsubj
        sys.stdout.write(' done\n')
        permttest[-1] = t
        
        sum_higher = (permttest > t).sum(0)
        vox_pvalue = sum_higher/float(nperm)
        
        thridx = np.hstack([block_argsort(permttest[:,i*blocksize:(i+1)*blocksize], -thr_permidx) \
                            for i in range(int(nfeat/blocksize+1))])
        thr = permttest[thridx,np.arange(permttest.shape[1])]
        labels, cluster_prob_raw, labels_fwe, cluster_prob_fwe = cluster_corr(t, thr, permttest, neighborhood,
                                                                  fwe_rate, multicomp_correction)
        results['main_fx'][main_fx] = (t, p, vox_pvalue, thr,
                                       labels, cluster_prob_raw, labels_fwe, cluster_prob_fwe)
        del data

    results['contrasts'] = dict()
    for contrast in contrasts:
        print('contrast',contrast)
        
        #data = sl_ress[:,contrast]
        #t,p = scipy.stats.ttest_rel(data[:,0],data[:,1])
        data = sl_ress[:,contrast[0]]-sl_ress[:,contrast[1]]
        t,p = data.mean(0),None

        #permutations = [ np.random.randint(0,2,nsubj, dtype=np.uint8) for i in xrange(nperm-1)]
        #perm_bool = [np.asarray([perm==0,perm==1]).T for perm in permutations]
        # include real contrast for lower-bound on p-values
        for i,perm in enumerate(perm_bool):
            sys.stdout.write('\r permutations: %d/%d' % (i,nperm))
            sys.stdout.flush()
            #permttest[i],_ = scipy.stats.ttest_rel(data[perm],data[~perm])
            signs = np.random.randint(0,2,nsubj)*2-1
            permttest[i] = reduce(lambda x,y: x+y[0]*y[1], zip(signs,data),0)/nsubj
        sys.stdout.write(' done\n')
        permttest[-1] = t
        

        sum_higher = (permttest > t).sum(0)
        sum_lower = (permttest < t).sum(0)
        two_tailed_voxp = np.minimum(sum_higher, sum_lower)/float(nperm)

        thridx_low, thridx = np.hstack([block_argsort(permttest[:,i*blocksize:(i+1)*blocksize],
                                                      [thr_permidx, -thr_permidx]) \
                                        for i in range(int(nfeat/blocksize+1))])
        
        thr_low = permttest[thridx_low,np.arange(permttest.shape[1])]
        thr = permttest[thridx,np.arange(permttest.shape[1])]

        labels, cluster_prob_raw, labels_fwe, cluster_prob_fwe = cluster_corr(
            t, thr,
            permttest, neighborhood, fwe_rate, multicomp_correction)

        sc_idx = 2*32492
        sc_labels, sc_cluster_prob_raw, sc_labels_fwe, sc_cluster_prob_fwe = cluster_corr(
            t[sc_idx:], thr[sc_idx:],
            permttest[:,sc_idx:], neighborhood.todok()[sc_idx:,sc_idx:].tocoo(), fwe_rate, multicomp_correction)

        labels_low, cluster_prob_raw_low, labels_fwe_low, cluster_prob_fwe_low = cluster_corr(
            t, thr_low,
            permttest, neighborhood, fwe_rate, multicomp_correction,
            neg=True)

        results['contrasts'][contrast] = (t, p, two_tailed_voxp, thr,thr_low,
                                          labels, cluster_prob_raw, labels_fwe, cluster_prob_fwe,
                                          labels_low, cluster_prob_raw_low, labels_fwe_low, cluster_prob_fwe_low,
                                          sc_labels, sc_cluster_prob_raw, sc_labels_fwe, sc_cluster_prob_fwe)
        del data
    del permttest
    np.save('results_group_cluster_%s.npy'%block_phase,results)
    return results

def cluster_size_thresh(thrd, neighborhood, npts=10):
    labels, num = _clusterize_custom_neighborhood(thrd, neighborhood)
    area = scipy.ndimage.measurements.sum(thrd, labels, index=np.arange(1, num + 1)).astype(np.int)
    new_labels = labels.copy()
    for l,a in enumerate(area):
        if a < npts:
            new_labels[labels==l+1] = 0
    return new_labels

def perm_tfce(data, nperms, neighborhood, h=2, e=.5, d=.1):
    nsubj, nfeat = data.shape
    perms = np.empty((nperms, nfeat), dtype=np.float32)
    perms.fill(0)
    perm_data = data.astype(np.float32)
    for i in range(nperms):
        sys.stdout.write('\r permutations: %d/%d    ' % (i,nperms))
        sys.stdout.flush()
        signs = np.random.randint(0,2,nsubj)*2-1
        #perm_mean = reduce(lambda x,y: x+y[0]*y[1], zip(signs,data),0)/nsubj
        perm_data *= signs[:,np.newaxis]
        perm_mean,_ = scipy.stats.ttest_1samp(perm_data,0)
        perm_mean[np.isnan(perm_mean)] = 0
        perms[i] = tfce_map(perm_mean, neighborhood, h, e, d)
    sys.stdout.write('\n done')
    return perms

def perm_tfce_reg(data, regs, nperms, neighborhood, h=2, e=.5, d=.1):
    nsubj, nfeat = data.shape
    perms = np.zeros((nperms, nfeat), dtype=np.float32)
    regs_perm = regs.copy()
    for i in range(nperms):
        sys.stdout.write('\r permutations: %d/%d    ' % (i,nperms))
        sys.stdout.flush()
        regs_perm[:,0] = np.random.permutation(regs[:,0])
        perm_mean = np.linalg.lstsq(regs_perm, data)[0][0]
        perm_mean[np.isnan(perm_mean)] = 0
        perms[i] = tfce_map(perm_mean, neighborhood, h, e, d)
    sys.stdout.write('\n done')
    return perms


def group_rsa_cnbis_reg_tfce(reg, block_phase='exec',groupInt=None,
                             main_fxs=[0], contrasts=[],
                             nperm=1000, h=2, e=.5,
                             nproc=1):
    
    if groupInt is not None:
        files = [os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,block_phase)) \
                 for sid in subject_ids if sid in group_Int]
    else:
        files = [os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,block_phase)) \
                 for sid in subject_ids]    

    sl_ress = np.asarray([Dataset.from_hdf5(f).samples.reshape(6,6,-1).mean(0).astype(np.float32) for f in files])
    nsubj = len(sl_ress)

    neighborhood = np.load(os.path.join(proc_dir,'connectivity_96k.npy')).tolist()
    results = dict()

    # add intercept
    regs = np.asarray([reg,np.ones(reg.shape)]).T
    
    if nproc>1:
        import pprocess

    results['main_fx'] = dict()        
    for main_fx in main_fxs:
        print('main_fx', main_fx)
        data = sl_ress[:,main_fx]
        betas,_,_,_= np.linalg.lstsq(regs, data)
        betas[np.isnan(betas)] = 0
        max_val = betas[0].max()
        min_val = -betas[0].min()
        d = max_val/100.
        d_neg = min_val/100.
        
        tfce = tfce_map(betas[0], neighborhood, h, e, d)
        
        if nproc>1:
            blocksize = nperm/nproc
            blocksizes = [blocksize]*(nproc-1)+[nperm-1-(blocksize*(nproc-1))]
            df = joblib.delayed(perm_tfce_reg)
            permttest = np.vstack(
                joblib.Parallel(n_jobs=nproc)([df(data, regs, b, neighborhood, h, e, d) for b in blocksizes])+
                [np.asarray([tfce])])

        else:
            permttest = perm_tfce_reg(data, regs, nperm, neighborhood, h, e, d)
            # include real contrast for lower-bound on p-values
            permttest[-1] = tfce
            sys.stdout.write(' done\n')
        
        sum_higher = (permttest >= tfce).sum(0)
        vox_pvalue = sum_higher/float(nperm)

        del permttest
        tfce_neg = tfce_map(-betas[0], neighborhood, h, e, d)

        if nproc>1:
            blocksize = nperm/nproc
            blocksizes = [blocksize]*(nproc-1)+[nperm-1-(blocksize*(nproc-1))]
            df = joblib.delayed(perm_tfce_reg)
            permttest_neg = np.vstack(
                joblib.Parallel(n_jobs=nproc)([df(data, -regs, b, neighborhood, h, e, d_neg) for b in blocksizes])+
                [np.asarray([tfce_neg])])
        else:
            permttest_neg = perm_tfce_reg(data, -regs, nperm, neighborhood, h, e, d_neg)
            # include real contrast for lower-bound on p-values
            permttest_neg[-1] = tfce_neg
            sys.stdout.write(' done\n')

        sum_higher_neg = (permttest_neg >= tfce_neg).sum(0)
        vox_pvalue_neg = sum_higher_neg/float(nperm)
        del permttest_neg
        
        results['main_fx'][main_fx] = (betas[0], tfce, vox_pvalue, tfce_neg, vox_pvalue_neg)
        del data
    return results
    

def group_rsa_cnbis_tfce(block_phase='exec',groupInt=None,
                         main_fxs=[0,5], contrasts=[(0,5)],
                         nperm=1000, h=2, e=.5,
                         nproc=1):
    
    if groupInt is not None:
        files = [os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,block_phase)) \
                 for sid in subject_ids if sid in group_Int]
    else:
        files = [os.path.join(proc_dir, output_subdir,'CoRe_%03d_%s_cnbis.h5'%(sid,block_phase)) \
                 for sid in subject_ids]    

#    sl_ress = np.asarray([Dataset.from_hdf5(f).samples.reshape(6,6,-1).mean(0).astype(np.float32) for f in files])
    sl_ress = np.asarray([Dataset.from_hdf5(f).samples.reshape(6,6,-1).astype(np.float32) for f in files])

    nsubj = len(sl_ress)

    neighborhood = np.load(os.path.join(proc_dir,'connectivity_96k.npy')).tolist()
    results = dict()
    results['main_fx'] = dict()
    nfeat = sl_ress.shape[-1]

    if nproc>1:
        import pprocess

    for main_fx in main_fxs:
        print('main_fx',main_fx)
        if isinstance(main_fx, tuple):
            data = sl_ress[:,main_fx[0],main_fx[1]]
        else:
            data = sl_ress[:,:,main_fx].mean(1)
        #data_mean = data.mean(0)
        data_mean,_ = scipy.stats.ttest_1samp(data,0)
        data_mean[np.isnan(data_mean)] = 0
        max_val = data_mean.max()
        d = max_val/100.

        tfce = tfce_map(data_mean, neighborhood, h, e, d)

        if nproc>1:
            blocksize = nperm/nproc
            blocksizes = [blocksize]*(nproc-1)+[nperm-1-(blocksize*(nproc-1))]
            df = joblib.delayed(perm_tfce)
            permttest = np.vstack(
                joblib.Parallel(n_jobs=nproc)([df(data, b, neighborhood, h, e, d) for b in blocksizes])+
                [np.asarray([tfce])])
                
        else:
            permttest = perm_tfce(data, nperm, neighborhood, h, e, d)
            # include real contrast for lower-bound on p-values
            permttest[-1] = tfce
            sys.stdout.write(' done\n')
        
        sum_higher = (permttest >= tfce).sum(0)
        vox_pvalue = sum_higher/float(nperm)
        
        results['main_fx'][main_fx] = (data_mean, tfce, vox_pvalue)
        del data, permttest

    results['contrasts'] = dict()

    #if len(contrasts)>0:
    #    permttest_low = np.empty((nperm, nfeat),dtype=np.float32)
    for contrast in contrasts:
        print('contrast',contrast)

        if isinstance(contrast[0], tuple):
            data = sl_ress[:,contrast[0][0],contrast[0][1]]-\
                   sl_ress[:,contrast[1][0],contrast[1][1]]
        else:
            data = (sl_ress[:,:,contrast[0]]-sl_ress[:,:,contrast[1]]).mean(1)
#        data_mean = data.mean(0)
        data_mean,_ = scipy.stats.ttest_1samp(data,0)
        data_mean[np.isnan(data_mean)] = 0
        max_val = data_mean.max()
        d = max_val/100.
        tfce = tfce_map(data_mean, neighborhood, h, e, d)
        tfce_low = tfce_map(-data_mean, neighborhood, h, e, d)

        if nproc>1:

            blocksize = nperm/nproc
            blocksizes = [blocksize]*(nproc-1)+[nperm-2-(blocksize*(nproc-1))]
            df = joblib.delayed(perm_tfce)
            permttest = np.vstack(
                joblib.Parallel(n_jobs=nproc)([df(data, b, neighborhood, h, e, d) for b in blocksizes])+
                [np.asarray([tfce, tfce_low])])
            
            p_results = pprocess.Queue(limit=nproc)
            compute = p_results.manage(pprocess.MakeParallel(perm_tfce))
        else:
            permttest = perm_tfce(data, nperm, neighborhood, h, e, d)
            # include real contrasts for lower-bound on p-values
            permttest[-1] = tfce
            permttest[-2] = tfce_low
        sys.stdout.write(' done\n')

        sum_higher = (permttest >= tfce).sum(0)
        # as the permutations plays on the sign of difference, the null distribution should be the same for both tails
        sum_lower = (permttest >= tfce_low).sum(0)

        p_high = sum_higher/float(nperm)
        p_low = sum_lower/float(nperm)
        two_tailed_voxp = np.minimum(sum_higher, sum_lower)/float(nperm)
        results['contrasts'][contrast] = (data_mean, tfce, tfce_low, two_tailed_voxp, p_high, p_low)

        del data, permttest
    np.save('results_group_cluster_%s.npy'%block_phase,results)
    return results
 
def tfce_map(map_, neighborhood, h, e, d=None):
    max_value = map_.max()
    if d is None:
        d = max_value/100.
    tfce = np.zeros_like(map_)
    mask = np.empty_like(map_, dtype=np.bool)
    keep_edges = np.empty(len(neighborhood.col),dtype=np.bool)
    labels_map = np.empty_like(map_, dtype=np.uint)
    
    for t in np.arange(d, max_value+d/2., d):
        mask[:] = map_ > t
        np.logical_and(mask[neighborhood.col], mask[neighborhood.row], keep_edges)
        neigh_thr = scipy.sparse.coo_matrix(
            (neighborhood.data[keep_edges],
             (neighborhood.row[keep_edges],
              neighborhood.col[keep_edges])),
            neighborhood.shape)
        labels_map[:] = (scipy.sparse.csgraph.connected_components(neigh_thr, directed=False)[1]+1)*mask
        labels, labels_map[:], area = np.unique(labels_map, return_inverse=True, return_counts=True)

        tfce_vals = d*(area**e)*(t**h)
        
        tfce[mask] += tfce_vals[labels_map[mask]]
        del neigh_thr, labels, area, tfce_vals
    return tfce
    

def cluster_corr(map_, thr, perms, neighborhood, fwe_rate,multicomp_correction='fdr_bh', neg=False):
    counter_perms = Counter()
    if neg:
        thrd = map_ < thr
        for tt in perms:
            get_cluster_sizes(Dataset([tt<thr]), counter_perms, neighborhood)
    else:
        thrd = map_ > thr
        for tt in perms:
            get_cluster_sizes(Dataset([tt>thr]), counter_perms, neighborhood)

    null_cluster_size = scipy.sparse.dok_matrix((1, len(map_) + 1), dtype=int)
    for s in counter_perms:
        null_cluster_size[0, s] = counter_perms[s]
    del counter_perms

    labels, num = _clusterize_custom_neighborhood(thrd, neighborhood)
    area = scipy.ndimage.measurements.sum(thrd, labels, index=np.arange(1, num + 1)).astype(np.int)
   
    cluster_probs_raw = np.asarray(_transform_to_pvals(area, null_cluster_size.astype(np.float)))
    
    labels_raw = labels.copy()
    for i, cp in enumerate(cluster_probs_raw):
        if cp > fwe_rate:
            labels_raw[labels == i + 1] = 0

    rej, cluster_probs_corr = smm.multipletests(
        cluster_probs_raw,
        alpha=fwe_rate,
        method=multicomp_correction)[:2]

    labels_fwe = labels.copy()
    for i, r in enumerate(rej):
        if not r:
            labels_fwe[labels == i + 1] = 0

    return labels_raw, cluster_probs_raw, labels_fwe, cluster_probs_corr        


def group_rsa_cnbis_delay_tfce(groupInt=None,
                               main_fxs=[0,5],
                               nperm=1000, h=2, e=.5,
                               nproc=1):
    
    if groupInt is not None:
        files = [os.path.join(proc_dir, 'searchlight_cnbis_delays_slmap','CoRe_%03d_cnbis_delays.h5'%sid) \
                 for sid in subject_ids if sid in group_Int]
    else:
        files = [os.path.join(proc_dir, 'searchlight_cnbis_delays_slmap','CoRe_%03d_cnbis_delays.h5'%sid) \
                 for sid in subject_ids]

    sl_ress = np.asarray([Dataset.from_hdf5(f).samples.reshape(9,6,6,-1).mean(1).astype(np.float32) for f in files])

    nsubj = len(sl_ress)
    nfeat = sl_ress.shape[-1]

    neighborhood = np.load(os.path.join(proc_dir,'connectivity_96k.npy')).tolist()
    neighborhood_st = scipy.sparse.coo_matrix((
        np.ones(neighborhood.col.shape[0]*9+nfeat*8,dtype=np.bool),
        (np.hstack([(np.repeat(neighborhood.col[np.newaxis],9,0)+np.arange(9)[:,np.newaxis]*nfeat).flatten(),
                    np.arange(nfeat,nfeat*9)]),
         np.hstack([(np.repeat(neighborhood.row[np.newaxis],9,0)+np.arange(9)[:,np.newaxis]*nfeat).flatten(),
                    np.arange(nfeat*8)]))))

    results = dict()
    results['main_fx'] = dict()

    if nproc>1:
        import pprocess

    for main_fx in main_fxs:
        print('main_fx',main_fx)
        data = sl_ress[:,:,main_fx].reshape(nsubj,-1)
        #data_mean = data.mean(0)
        data_mean,_ = scipy.stats.ttest_1samp(data,0)
        data_mean[np.isnan(data_mean)] = 0
        max_val = data_mean.max()
        d = max_val/100.

        tfce = tfce_map(data_mean, neighborhood_st, h, e, d)

        if nproc>1:
            blocksize = nperm/nproc
            blocksizes = [blocksize]*(nproc-1)+[nperm-1-(blocksize*(nproc-1))]
            df = joblib.delayed(perm_tfce)
            permttest = np.vstack(
                joblib.Parallel(n_jobs=nproc)([df(data, b, neighborhood_st, h, e, d) for b in blocksizes])+
                [np.asarray([tfce])])
                
        else:
            permttest = perm_tfce(data, nperm, neighborhood_st, h, e, d)
            # include real contrast for lower-bound on p-values
            permttest[-1] = tfce
            sys.stdout.write(' done\n')
        
        sum_higher = (permttest >= tfce).sum(0)
        vox_pvalue = sum_higher/float(nperm)
        
        results['main_fx'][main_fx] = (data_mean, tfce, vox_pvalue)
        del data, permttest
    return results

