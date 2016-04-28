import numpy as np

from mvpa2.support.nibabel.surf import Surface
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.misc.neighborhood import Sphere, IndexQueryEngine, QueryEngine, CachedQueryEngine, idhash_
from mvpa2.generators.partition import HalfPartitioner, NFoldPartitioner, CustomPartitioner
from mvpa2.generators.resampling import NonContiguous
from mvpa2.measures.searchlight import Searchlight, sphere_searchlight
from mvpa2.measures.base import CrossValidation
from mvpa2.measures.gnbsearchlight import GNBSearchlight, sphere_gnbsearchlight
from mvpa2.measures.rsa import PDist
from mvpa2.base.node import ChainNode
from mvpa2.datasets import Dataset, hstack, vstack
from mvpa2.base.hdf5 import h5load
from mvpa2.generators.resampling import Balancer
from mvpa2.mappers.fx import mean_sample, mean_group_sample
from mvpa2.clfs.gnb import GNB
from mvpa2.measures.base import Measure
from mvpa2.clfs.transerror import Confusion
from mvpa2.base.node import Node
from mvpa2.generators.splitters import Splitter
from mvpa2.clfs.transerror import ConfusionMatrixError, ConfusionMatrix
from mvpa2 import debug
if __debug__:
    debug.active += ["SLC"]


from mvpa2.base.dochelpers import borrowkwargs, _repr_attrs
from mvpa2.measures.adhocsearchlightbase import \
     SimpleStatBaseSearchlight, _STATS

class GNBSearchlightOpt(GNBSearchlight):

    @borrowkwargs(GNBSearchlight, '__init__')
    def __init__(self, *args, **kwargs):

        # init base class first
        GNBSearchlight.__init__(self, *args, **kwargs)

        self.__pl_train = None
        self._max_num_of_samples = 200
    
    def _reserve_pl_stats_space(self, shape):
        # per each label: to be (re)computed within each loop split
        # Let's try to reuse the memory though
        pl = self.__pl_train = _STATS()
        pl.sums = np.zeros(shape)
        pl.means = np.zeros(shape)
        # means of squares for stddev computation
        pl.sums2 = np.zeros(shape)
        pl.variances = np.zeros(shape)
        # degenerate dimension are added for easy broadcasting later on
        pl.nsamples = np.zeros(shape[:1] + (1,)*(len(shape)-1))


    """ 
    def _pass_attr(self, dataset, result):
        if self.errorfx is None:
            # when output is the raw predictions, sa can be guessed from generator (if not random)
            ds_sa = vstack([list(self._splitter.generate(part))[1] for part in self._generator.generate(dataset[:,:1])])
            # set fa to wrong shape, but only to cheat the _pass_attr and not create huge data replicating splits
            ds_sa.fa = dataset.fa
            ds_sa.a = dataset.a
            return super(GNBSearchlightOpt, self)._pass_attr(ds_sa, result)
        else:
            return super(GNBSearchlightOpt, self)._pass_attr(dataset, result)
    """
            
    def _sl_call_on_a_split(self,
                            split, X,
                            training_sis, testing_sis,
                            nroi_fids, roi_fids,
                            indexsum_fx,
                            labels_numeric,
                            ):
        """Call to GNBSearchlight
        """
        # Local bindings
        gnb = self.gnb
        params = gnb.params

        pl = self.__pl_train # we want to reuse the same storage across
                             # splits

        training_nsamples, non0labels = \
            self._compute_pl_stats(training_sis, pl)

        nlabels = len(pl.nsamples)

        if params.common_variance:
            pl.variances[:] = \
                np.sum(pl.sums2 - pl.sums * pl.means, axis=0) \
                / training_nsamples
        else:
            pl.variances[non0labels] = \
                (pl.sums2 - pl.sums * pl.means)[non0labels] \
                / pl.nsamples[non0labels]

        # assign priors
        priors = gnb._get_priors(
            nlabels, training_nsamples, pl.nsamples)

        # proceed in a way we have in GNB code with logprob=True,
        # i.e. operating within the exponents -- should lead to some
        # performance advantage
        norm_weight = -0.5 * np.log(2*np.pi*pl.variances)
        # last added dimension would be for ROIs
        logpriors = np.log(priors[:, np.newaxis, np.newaxis])

        if __debug__:
            debug('SLC', "  'Training' is done")

        # Now it is time to "classify" our samples.
        # and for that we first need to compute corresponding
        # probabilities (or may be un
        test_data = X[split[1].samples[:, 0]]

        predictions = np.zeros((len(test_data), nroi_fids),dtype=np.int)

        for s in range(0,len(test_data),self._max_num_of_samples):

            data = test_data[s:s+self._max_num_of_samples]
            # argument of exponentiation
            scaled_distances = \
                -0.5 * (((data - pl.means[:, np.newaxis, ...])**2) \
                / pl.variances[:, np.newaxis, ...])

            # incorporate the normalization from normals
            lprob_csfs = norm_weight[:, np.newaxis, ...] + scaled_distances
            del scaled_distances

            ## First we need to reshape to get class x samples x features
            lprob_csf = lprob_csfs.reshape(lprob_csfs.shape[:2] + (-1,))

            ## Now we come to naive part which requires looping
            ## through all spheres
            if __debug__:
                debug('SLC', "  Doing 'Searchlight'")
            # resultant logprobs for each class x sample x roi
            lprob_cs_sl = np.zeros(lprob_csfs.shape[:2] + (nroi_fids,))
            indexsum_fx(lprob_csf, roi_fids, out=lprob_cs_sl)

            lprob_cs_sl += logpriors
            lprob_cs_cp_sl = lprob_cs_sl
            # for each of the ROIs take the class with maximal (log)probability
            preds = lprob_cs_cp_sl.argmax(axis=0)
            # no need to map back [self.ulabels[c] for c in winners]
            #predictions = winners

            predictions[s:s+len(data)] = preds
            del lprob_csf, lprob_cs_sl, lprob_cs_cp_sl, preds

        targets = labels_numeric[testing_sis]

        return targets, predictions



class SurfVoxQueryEngine(QueryEngine):
    """ 
    """

    def __init__(self,
                 surf_sl_radius=20,
                 vox_sl_radius=2.4,
                 max_feat=None):
        QueryEngine.__init__(self, voxel_indices=Sphere(vox_sl_radius), coordinates=None)
        self._surf_sl_radius = surf_sl_radius
        self._vox_sl_radius = vox_sl_radius
        self._max_feat = max_feat

    def _train(self, ds):

        self._include = np.logical_and(~ds.fa.nans, (ds.samples==0).sum(0)==0)

        self._max_vertex = ds.a.triangles.max()+1
        
        self._surface = Surface(
            ds.fa.coordinates[:self._max_vertex],
            ds.a.triangles)

        self._sqe = SurfaceQueryEngine(
            self._surface,
            radius = self._surf_sl_radius,
            max_feat = self._max_feat)

        self._idx_qe = IndexQueryEngine(
            sorted=True,
            voxel_indices=Sphere(self._vox_sl_radius))

        self._sqe.train(ds[:,:self._max_vertex])
        self._idx_qe.train(ds[:,self._max_vertex:])

    def query_byid(self, fid):
        if fid < self._max_vertex:
            ids = self._sqe.query_byid(fid)
        else:
            ids = self._max_vertex+self._idx_qe.query_byid(fid-self._max_vertex)#[:self._max_feat]
        ids = np.asarray(ids)[self._include[ids]]
        return ids.tolist()

class CachedQueryEngineAlt(CachedQueryEngine):
    
    def train(self, dataset):
        """'Train' `CachedQueryEngineAlt`.

        Raises
        ------
        ValueError
          If `dataset`'s .fa were changed -- it would raise an
          exception telling to `untrain` explicitly, since the idea is
          to reuse CachedQueryEngine with the same engine and same
          dataset (up to variation of .sa, such as labels permutation)
        """
        ds_fa_hash = 1
#        ds_fa_hash = ':'.join([idhash_(dataset.fa[qo]) for qo in self._queryengine._queryobjs.keys()]) +\
#                     ':%d' % dataset.fa._uniform_length
        # ds_fa_hash = idhash_(dataset.fa) + ':%d' % dataset.fa._uniform_length
        if self._trained_ds_fa_hash is None:
            # First time is called
            self._trained_ds_fa_hash = ds_fa_hash
            self._queryengine.train(dataset)     # train the queryengine
            self._lookup_ids = [None] * dataset.nfeatures # lookup for query_byid
            self._lookup = {}           # generic lookup
            self.ids = self.queryengine.ids # used in GNBSearchlight??
        elif self._trained_ds_fa_hash != ds_fa_hash:
            raise ValueError, \
                  "Feature attributes of %s (idhash=%r) were changed from " \
                  "what this %s was trained on (idhash=%r). Untrain it " \
                  "explicitly if you like to reuse it on some other data." \
                  % (dataset, ds_fa_hash, self, self._trained_ds_fa_hash)
        else:
            pass

class GroupConfusionMatrixError(ConfusionMatrixError):

    def __init__(self, labels, group_attr, *args, **kwargs):
        ConfusionMatrixError.__init__(self, *args, **kwargs)
        self.labels = labels
        self._group_attr = group_attr

    def __call__(self, predictions, targets):

        ugroup_attr = np.unique (self._group_attr)
        conf_mxs = []
        for ga in ugroup_attr:
            cm = ConfusionMatrix(labels=list(self.labels),
                                 targets=targets[self._group_attr==ga],
                                 predictions=predictions[self._group_attr==ga])
            conf_mxs.append(cm.matrix[None, :])
        return np.vstack(conf_mxs)


"""
class SurfVoxSearchlight(Measure):
    
    def __init__(self, ds, clf, prtnr, postproc=None,
                 surf_sl_radius=20, vox_sl_radius=2.5,
                 surf_sl_max_feat=64, *args, **kwargs):
        Measure.__init__(self, *args, **kwargs)
        self._prtnr = prtnr
        self._postproc = postproc
        self._surf_sl_radius = surf_sl_radius
        self._vox_sl_radius = vox_sl_radius
        self._surf_sl_max_feat = surf_sl_max_feat
#        ulabels =         
        nlabels = len(ulabels)
        label2index = dict((l, il) for il, l in enumerate(ulabels))
        labels_numeric = np.array([label2index[l] for l in labels])
        self._ulabels_numeric = [label2index[l] for l in ulabels]

        self._setup_qe(ds)
        self._setup_surf_vox_searchlight(ds, clf)

    def _setup_qe(self, ds):
        self.max_vertex = ds.a.triangles.max()+1
        self._lrh_surf = Surface(
            ds.fa.coordinates[:self.max_vertex],
            ds.a.triangles)
               
        self._sqe = SurfaceQueryEngine(
            self._lrh_surf,
            radius = self._surf_sl_radius,
            max_feat = self._surf_sl_max_feat)

        self._idx_qe = IndexQueryEngine(
            voxel_indices=Sphere(self._vox_sl_radius))

    def _setup_surf_vox_searchlight(self, ds, clf):
        
        class AppendDim(Node):
            def _call(self,ds):
                return Dataset(ds.samples[None,None])
        errorfx = Confusion(
            labels=ds.uniquetargets,
            postproc=AppendDim())

        spltr = Splitter(attr='partitions', attr_values=[1,2])
        cvte = CrossValidation(
            clf, self._prtnr, splitter=spltr,
            errorfx=errorfx,
        )

        self.slght_surf_confmat = Searchlight(cvte, self._sqe)
        self.slght_vox_confmat = Searchlight(cvte, self._idx_qe)

    def __call__(self, ds):

        print('nsamples: %d, targets: %s'%(ds.nsamples, ', '.join(ds.uniquetargets)))
        
        slmap_surf_confmat = self.slght_surf_confmat(ds[:,:self.max_vertex])
        slmap_vox_confmat = self.slght_vox_confmat(ds[:,self.max_vertex:])
        
        
        slmap_confmat = Dataset(
            np.concatenate([
                slmap_surf_confmat.samples,
                slmap_vox_confmat.samples],1),#.astype(np.float32),
            sa=slmap_surf_confmat.sa,
            fa=ds.fa,
            a=ds.a)
        if self._postproc:
            slmap_confmat = Dataset(self._postproc(slmap_confmat.samples)[np.newaxis])

        slmap_confmat.samples /= slmap_confmat.samples[0,0].sum()

        slmap_accuracy = Dataset(
            slmap_confmat.samples[...,np.eye(slmap_confmat.shape[2],dtype=np.bool)].sum(2),
            fa = ds.fa,
            a = ds.a)
        print 'max accuracy: ', slmap_accuracy.samples.max()
        return slmap_confmat, slmap_accuracy

class GNBSurfVoxSearchlight(SurfVoxSearchlight):

    def _setup_surf_vox_searchlight(self, ds, clf):

        ds = ds.copy(deep=False)
#        ds.sa['targets_numeric'] = 
        spltr = Splitter(attr='partitions',attr_values=[1,2])

        self.slght_surf_confmat = GNBSearchlight(
            clf,
            self._prtnr,
            self._sqe,
            splitter=spltr,
            errorfx=errorfx,
            reuse_neighbors=True,
            pass_attr=ds.sa.keys())

        self.slght_vox_confmat = GNBSearchlight(
            clf,
            self._prtnr,
            self._idx_qe,
            splitter=spltr,
            errorfx=errorfx,
            reuse_neighbors=True,
            pass_attr=ds.sa.keys())

class RSASurfVoxSearchlight(SurfVoxSearchlight):

    def __init__(self, ds, **kwargs):
        SurfVoxSearchlight.__init__(self, ds, None, None, **kwargs)

    def _setup_surf_vox_searchlight(self, ds, clf):

        pdist = PDist(square=False)

        self.slght_surf_pdist = Searchlight(pdist, self._sqe)
        self.slght_vox_pdist = Searchlight(pdist, self._idx_qe)

    def __call__(self, ds):
        
        slmap_surf_pdist = self.slght_surf_pdist(ds[:,:self.max_vertex])
        slmap_vox_pdist = self.slght_vox_pdist(ds[:,self.max_vertex:])

        slmap_pdist = hstack([slmap_surf_pdist, slmap_vox_pdist])
        return slmap_pdist

"""

from scipy.sparse.csgraph import connected_components
from scipy.sparse.coo import coo_matrix

def cluster_labels(thr_map, neigh):
    keep_edges = np.logical_and(thr_map[neigh.col], thr_map[neigh.row])
    neigh_thr = coo_matrix(
        (neigh.data[keep_edges],
         (neigh.row[keep_edges],
          neigh.col[keep_edges])),
        neigh.shape)
    return connected_components(neigh_thr, directed=False)[1]

def cluster_counts(thr_map, neigh):
    cl_lbls = cluster_labels(thr_map, neigh)
    labels, counts = np.unique(cl_lbls*thr_map, return_counts=True)
    if labels[0] == 0:
        counts = counts[1:]
    return counts

def clusterize(thr_map, neigh):
    cl_lbls = cluster_labels(thr_map, neigh)
    labels, counts = np.unique(cl_lbls*thr_map, return_counts=True)
    if labels[0] == 0:
        labels, counts = labels[1:], counts[1:]
    # reassign labels
    new_labels = np.zeros(cl_lbls.shape, dtype=np.uint)
    for li,l in enumerate(labels):
        new_labels[cl_lbls==l] = li+1
    return new_labels
    

"""
conn = scipy.sparse.coo_matrix((
    np.ones(3*tris.shape[0]),
    (np.hstack([tris[:,:2].T.ravel(),tris[:,1]]),
     np.hstack([tris[:,1:].T.ravel(),tris[:,2]]))))
"""
