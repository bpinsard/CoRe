import sys, os, glob
import numpy as np
import scipy.stats, scipy.ndimage.measurements, scipy.sparse
from ..mvpa import searchlight
from ..mvpa import dataset as mvpa_ds
from . import mvpa_nodes
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
output_subdir = 'searchlight_cnbis_mnorm'
compression= 'gzip'

subject_ids = [1, 11, 23, 22, 63, 50, 79, 54, 107, 128, 162, 102, 82, 155, 100, 94, 87, 192, 195, 220, 223, 235, 268, 267,237,296]
#subject_ids = subject_ids[:-1]
#subject_ids = [296]
group_Int = [1,23,63,79,82,87,100,107,128,192,195,220,223,235,268,267,237,296]
ulabels = ['CoReTSeq','CoReIntSeq','mvpa_CoReOtherSeq1','mvpa_CoReOtherSeq2','rest']
#ulabels = ulabels[1:]


seq_groups = {
    'mvpa_new_seqs' : ulabels[2:4],
    'tseq_intseq' : ulabels[:2],
    'all_seqs': ulabels[:4]
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

    block_phases = ['exec']
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

    """
    ds_all = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%sid, dataset_subdir, 'ds_%d.h5'%sid))
    ds_mvpa = ds_all[dict(scan_name=mvpa_scan_names)]
    ds_mvpa.sa['subject_id'] = [sid]*ds_mvpa.nsamples
    ds_mvpa.sa['group'] = [sid in group_Int]*ds_mvpa.nsamples
    del ds_all

    targets_num(ds_mvpa, ulabels)
    poly_detrend(ds_mvpa, chunks_attr='scan_id', polyord=0)
    zscore(ds_mvpa, chunks_attr='scan_id', param_est=('targets','rest'))
    """

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

    # del ds_mvpa, ds_glm_mvpa
    

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

    block_phases = ['exec']
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

def subject_mvpa_ds_residuals(sid, hptf_thresh=8, reg_sa='regressors_exec'):

    ts_files = [ os.path.join(preproc_dir, '_subject_id_%d'%sid, 'moco_bc_mvpa_aniso','mapflow',
                              '_moco_bc_mvpa_aniso%d'%scan_id,'ts.h5') for scan_id in range(2)]
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
        last_part1 = np.where(np.abs(exec15) > 0)[0][-1]
        first_part2 = np.where(np.abs(instr16) > 0)[0][0]

        chunks = np.hstack([np.tile(np.asarray([-1,1]).repeat(16),2),[0]])

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

def subject_rois_rsa_crossnobis(sid, hptf_thresh=8, reg_sa='regressors_exec'):
    print('______________   CoRe %03d   ___________'%sid)

    glm_ds_mvpa, residuals_mvpa = subject_mvpa_ds_residuals(sid, hptf_thresh, reg_sa=reg_sa)

    rois = Dataset.from_hdf5(os.path.join(proc_dir,'msl_rois.h5'))
    for ri,roi_name in enumerate(rois.a.roi_labels):
        mask = rois.samples[0]==ri+1
        res = residuals_mvpa.samples[:,mask]
        res2
    
    

def subject_searchlight_rsa_crossnobis(sid, hptf_thresh=8, reg_sa='regressors_exec'):
    print('______________   CoRe %03d   ___________'%sid)

    glm_ds_mvpa, residuals_mvpa = subject_mvpa_ds_residuals(sid, hptf_thresh, reg_sa=reg_sa)

    svqe = searchlight.SurfVoxQueryEngine(max_feat=128, vox_sl_radius=3.2, surf_sl_radius=15)
    svqe_cached = searchlight.CachedQueryEngineAlt(svqe)
    svqe_cached.train(glm_ds_mvpa)

    part = CustomPartitioner([([a],[b]) for a in range(4) for b in range(a+1,4)],attr='chunks')
    #part = CustomPartitioner([([a],[b]) for a in range(4) for b in range(4) if a!=b],attr='chunks')
    cnbis_sl = CrossNobisSearchlight(part, svqe_cached, space='targets', nproc=2)

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

        permutations = [ np.random.randint(0,2,nsubj, dtype=np.uint8) for i in xrange(nperm-1)]
        perm_bool = [np.asarray([perm==0,perm==1]).T for perm in permutations]
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
        
def tfce_map(map_, neighborhood, h, e, d=None):
    max_value = map_.max()
    if d is None:
        d = max_value/100.
    tfce = np.zeros_like(map_)
    mask = np.zeros_like(map_, dtype=np.bool)
    
    for t in np.arange(d, max_value+d/2., d):
        mask = map_ > t
        keep_edges = np.logical_and(mask[neighborhood.col], mask[neighborhood.row])
        neigh_thr = scipy.sparse.coo_matrix(
            (neighborhood.data[keep_edges],
             (neighborhood.row[keep_edges],
              neighborhood.col[keep_edges])),
            neighborhood.shape)
        labels_map = (scipy.sparse.csgraph.connected_components(neigh_thr, directed=False)[1]+1)*mask
        labels, labels_map, area = np.unique(labels_map,return_inverse=True,return_counts=True)

        tfce_vals = d*(area**e)*(t**h)
        
        tfce[mask] += tfce_vals[labels_map[mask]]
        del neigh_thr
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
    

def all_searchlight_2fold():
    new_sids = [sid for sid in subject_ids if len(glob.glob(os.path.join(proc_dir,output_subdir,'CoRe_%03d_*confusion.h5'%sid)))==0]
    print new_sids
    new_sids = subject_ids
    joblib.Parallel(n_jobs=4)([joblib.delayed(subject_searchlight_2fold)(sid) for sid in new_sids])

def subject_searchlight_2fold(sid):
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

    svqe = searchlight.SurfVoxQueryEngine(max_feat=128, vox_sl_radius=3.2, surf_sl_radius=25)
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
rois = np.hstack([np.asarray([46,29,70,69,28,4,3,7,8,26,27,16])+a for a in [11100,12100]]+[53,17,10,49,51,12,8,47,11,50,13,52])
#rois = np.asarray([53,17,10,49,51,12,8,47,11,50])
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
#    ds = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%s'%subj, dataset_subdir, 'ds_%s.h5'%subj))

    mvpa_scan_names = [n for n in np.unique(ds_glm.sa.scan_name) if 'mvpa' in n]
    ds_glm_mvpa = ds_glm[dict(scan_name=mvpa_scan_names)]
    targets_num(ds_glm_mvpa, ulabels)
    poly_detrend(ds_glm_mvpa, chunks_attr='scan_id', polyord=0)

    #ds_glm_mvpa.samples/=ds_glm_mvpa.samples.std(0)
    #ds_glm_mvpa.samples[np.isnan(ds_glm_mvpa.samples)]=0
    #ds_mvpa = ds[dict(scan_name=mvpa_scan_names)]
    """
    targets_num(ds_mvpa, ulabels)
    if 'ba_thresh' not in ds_mvpa.fa:
        #ds_mvpa.fa['ba_thresh'] = ds_mvpa.fa.ba_thres
        if 'ba_thresh' not in ds_glm_mvpa.fa:
            ds_glm_mvpa.fa['ba_thresh'] = ds_glm_mvpa.fa.ba_thres
            #poly_detrend(ds_mvpa, chunks_attr='scan_id', polyord=0)
            #zscore(ds_mvpa,chunks_attr='scan_id')
            del ds_glm, ds
    """
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
        #'instr',
        'exec']
    cvte_subsets = dict([('%s_%s_%s'%(sg_name, bp,scan_group_name),dict(targets=seqs,subtargets=[bp],scan_name=scans)) \
                         for sg_name,seqs in seq_groups.items() \
                         for bp in block_phases
                         for scan_group_name,scans in scan_groups.items()])

    spltr = Splitter(attr='partitions',attr_values=[1,2])

    prtnrs = dict(
        #loco=mvpa_nodes.prtnr_loco_cv(1),
        twofold=FactorialPartitioner(
            NFoldPartitioner(cvtype=4,attr='chunks'),
            attr='targets',
            selection_strategy='equidistant',
            count=16)
        #loso=mvpa_nodes.prtnr_loso_cv(1)
    )
    rois = Dataset.from_hdf5(os.path.join(proc_dir,'msl_rois.h5'))

    for prtnr_name, prtnr in prtnrs.items():
        glm_rois_stats[prtnr_name] = {}
        tr_rois_stats[prtnr_name] = {}
        #cvte = CrossValidation(
        #    clf, prtnr, splitter=spltr, enable_ca=['stats'],
        #    #postproc=BinomialProportionCI(width=.95, meth='jeffreys')
        #)
        cvte = create_cv_nullhyp(clf,prtnr,spltr,nrepeat=100)
        for subset_name, subset in cvte_subsets.items():
            print '_'*10,subset_name, subset,'_'*10
            glm_rois_stats[prtnr_name][subset_name] = {}
            tr_rois_stats[prtnr_name][subset_name] = {}
            ds_glm_subset = ds_glm_mvpa[subset]
            #            ds_glm_subset.samples/=ds_glm_subset.samples.std(0)
            #            ds_glm_subset.samples[np.isnan(ds_glm_subset.samples)]=0
            #ds_subset = ds_mvpa[subset]
            for ri,roi_name in enumerate(rois.a.roi_labels):
                cv_glm_res = cvte(ds_glm_subset[:,rois.samples[0]==ri+1])
                glm_rois_stats[prtnr_name][subset_name][roi_name] = cvte.ca.stats
                pvalue = cvte.ca.null_prob.samples
                glm_rois_stats[prtnr_name][subset_name][roi_name].stats['pvalue'] = pvalue
                print('glm\t%s\t%s\t%s\tacc=%f\tp=%.5f'%(
                    prtnr_name, roi_name, subset_name, 
                    glm_rois_stats[prtnr_name][subset_name][roi_name].stats['ACC'],pvalue))
                
            """
            for roi_fa, rois_labels in rois_groups.items():
                for roi_name, roi_label in rois_labels.items():
                    cv_glm_res = cvte(ds_glm_subset[:,{roi_fa:[roi_label],'nans':[False]}])
                    glm_rois_stats[prtnr_name][subset_name][roi_name] = cvte.ca.stats
                    pvalue = cvte.ca.null_prob.samples
                    glm_rois_stats[prtnr_name][subset_name][roi_name].stats['pvalue'] = pvalue
                    print('glm\t%s\t%s\t%s\tacc=%f\tp=%.5f'%(prtnr_name, roi_name, subset_name, glm_rois_stats[prtnr_name][subset_name][roi_name].stats['ACC'],pvalue))
            

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
        slmaps_conf = [Dataset.from_hdf5(os.path.join(proc_dir,'searchlight_wb_hptf/CoRe_%03d_%s_confusion.h5'%(s,sln))) \
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


def subject_rsa_analysis(subject_id):
#    ds_glm = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%s'%subject_id, 
#                                            dataset_subdir, 'glm_ds_%s.h5'%subject_id))
    
    ds_all = Dataset.from_hdf5(os.path.join(preproc_dir, '_subject_id_%d'%subject_id, dataset_subdir, 'ds_%d.h5'%subject_id))
    zscore(ds_all, chunks_attr='scan_id', param_est=('subtargets','rest'))


    pat = dict()
    pat['training_end'] = ds_all[dict(scan_name=['d1_training_TSeq'],subtargets=['exec'],blocks_idx=range(7,14))].samples.mean(0)
    """
    for sn in ['d3_mvpa1', 'd3_mvpa2']:
        for seq in ulabels[:4]:
            pat['%s_%s'%(sn,seq)] = ds_all[dict(scan_name=[sn],targets=[seq],subtargets=['exec'])].samples.mean(0)
    """
#    del ds_glm

    all_tvalues = dict()
    all_pvalues = dict()
    
    rois = Dataset.from_hdf5('/mnt/data/analysis/msl_rois.h5')

    for roi_id, roi_name in enumerate(rois.a.roi_labels):
        roi_mask = rois.samples[0]==roi_id+1
        ds_tmp = ds_all[:,roi_mask]
#        ds_tmp.samples -= ds_tmp.samples.mean(1)[:,np.newaxis]
        for patname, pat_data in pat.items():
            print roi_name, patname
            roi_pat = pat_data[roi_mask]
            mtx = np.column_stack([roi_pat,np.ones(len(roi_pat))])

            xx1 = np.linalg.inv(mtx.T.dot(mtx))
            betas = xx1.dot(mtx.T).dot(ds_tmp.samples.T)
            res = ds_tmp.samples-mtx.dot(betas).T
            varb = res.var(1)*xx1[0,0]
            
            """
            all_corrs['%s_%s'%(roi_name, patname)] = np.dot(
                (ds_tmp.samples-ds_tmp.samples.mean(1)[:,np.newaxis])/ds_tmp.samples.std(1)[:,np.newaxis],
                ((roi_pat-roi_pat.mean())/roi_pat.std())[:,np.newaxis]).ravel()/ds_tmp.nfeatures
            """
            all_tvalues['%s_%s'%(roi_name, patname)] = betas[0]/np.sqrt(varb)
            all_pvalues['%s_%s'%(roi_name, patname)] = 2 * scipy.stats.distributions.t.sf(np.abs(all_tvalues['%s_%s'%(roi_name, patname)]),len(roi_pat)-2)
            
            del res
        del ds_tmp
    sas = ds_all.sa
    del ds_all
    return all_tvalues, all_pvalues, sas


def all_subjects_rsa_analysis(shift=30):
    pyplot.style.use('ggplot')
    all_subject_tvalues = dict()
    all_subject_corrs = dict()
    sas = dict()
    for s in subject_ids:
        print '_'*20,s,'_'*20
        all_subject_tvalues[s],all_subject_corrs[s],sas[s] = subject_rsa_analysis(s)
        f,ax = pyplot.subplots(
            1,1,
            sharex=True,
            figsize=(50,15),
            gridspec_kw=dict(left=.04, right=.999,top=.99,bottom=.02))
        roi_names = sorted(all_subject_tvalues[s].keys())
        
        scans_diff = sas[s].scan_name[1:]!=sas[s].scan_name[:-1]
        scans_start = np.hstack([[0],np.where(scans_diff)[0]+1])
        scans_length = np.bincount(np.cumsum(scans_diff))
        
        ax.bar(left=scans_start,
               bottom=-shift*2,
               width=scans_length,
               height=[shift*len(roi_names)+shift*4]*len(scans_length),
               color=[pyplot.rcParams['axes.color_cycle'][('sleep' in n or 'rest' in n)] \
                      for n in sas[s].scan_name[scans_start]],
               alpha=.1)

        ax.plot(
            np.asarray([all_subject_tvalues[s][r]+ri*shift for ri,r in enumerate(roi_names)]).T,
            lw=.1)
        ax.set_ylim(-shift*1,shift*(len(roi_names)+1))
        ax.set_xlim(0,len(all_subject_tvalues[s].values()[0]))
        ax.set_yticks((np.arange(len(roi_names))*shift+np.array([[-shift/2,0,shift/2]]).T).T.ravel())
        ax.set_yticklabels(__builtin__.sum([[-shift/2,r,shift/2] for r in roi_names],[]))
        f.savefig('/home/bpinsard/data/projects/CoRe/results/rsa/rsa_tvalues_%03d.pdf'%s)
        pyplot.close(f)
        #for ri,r in enumerate(sorted(all_subject_tvalues[s].keys())):
        #    ax[ri].plot(all_subject_tvalues[s][r])
    np.savez(os.path.join(proc_dir,'all_subjects_rsa.npy'),tvalues=all_subject_tvalues,corrs=all_subject_corrs, sas=sas)
    return all_subject_tvalues, all_subject_corrs, sas
            

def create_rois():

    hemis = 'lr'
    subctx_rois = dict(
        hip_l=17,
        hip_r=53,
        thal_l=10,
        thal_r=49,
        put_l=12,
        put_r=51,
        cer_l=8,
        cer_r=47,
        caud_l=11,
        caud_r=50,
        pal_l=13,
        pal_r=52    
    )
    t_pat = Dataset.from_hdf5('/home/bpinsard/data/projects/CoRe/results/mean_task_pattern.h5')

    bas = t_pat.fa.ba
    parc = t_pat.fa.aparc
    parcmod100 = np.mod(parc,100)

    ctx_rois_masks = [
#        ('dlpfc', np.logical_and(parcmod100==15,bas!=7)),
        ('sma', np.logical_and(bas==7, parcmod100==16)),
        ('pmd', np.logical_and(bas==7, np.logical_or(np.logical_or(parcmod100==70, parcmod100==55),parcmod100==29))),
        ('m1', np.logical_and(np.logical_or(bas==5,bas==6),~np.logical_or(parcmod100==3,parcmod100==16))),
        ('s1', np.logical_and(bas>0,bas<5)),
        ('opj', parcmod100==27),
        ('ips', parcmod100==57),
    ]



    svqe = searchlight.SurfVoxQueryEngine(vox_sl_radius=8, surf_sl_radius=25)
    svqe.train(t_pat)
    
    idx=0
    rois_mask = np.zeros(t_pat.nfeatures,dtype=np.int)
    rois_labels = []
    for hi,h in enumerate(hemis):
        for roi_name, roi_mask in ctx_rois_masks:
            print roi_name
            idx+=1
            tmp_mask = roi_mask.copy()
            tmp_mask[(1-hi)*32492:(2-hi)*32492] = 0
            tmp_mask[2*32492:] = 0
            if roi_name in ['s1','m1','dlpfc']:
                if roi_name=='s1':
                    neighs = np.asarray(svqe.query_byid(7816+32492*hi))
                elif roi_name=='m1':
                    neighs = np.asarray(svqe.query_byid(8080+32492*hi))
                elif roi_name=='dlpfc':
                    neighs = np.asarray(svqe.query_byid(29972+32492*hi))
                neighs = neighs[tmp_mask[neighs]]
                rois_mask[neighs] = idx
            else:
                rois_mask[tmp_mask] = idx
            rois_labels.append('%s_%s'%(roi_name,h))
    for roi_name, roi_label in subctx_rois.items():
        print roi_name
        idx += 1
        mask = (parc == roi_label)
        neighs = np.asarray(svqe.query_byid(t_pat.fa.node_indices[mask][np.argmax(t_pat.samples[0,mask])]))
        neighs = neighs[mask[neighs]]
        rois_mask[neighs] = idx
        rois_labels.append(roi_name)
    return rois_mask, rois_labels
            

