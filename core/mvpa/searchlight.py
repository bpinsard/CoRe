import numpy as np

from mvpa2.support.nibabel.surf import Surface
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.misc.neighborhood import Sphere, IndexQueryEngine
from mvpa2.generators.partition import HalfPartitioner, NFoldPartitioner, CustomPartitioner
from mvpa2.generators.resampling import NonContiguous
from mvpa2.measures.searchlight import Searchlight, sphere_searchlight
from mvpa2.measures.base import CrossValidation
from mvpa2.measures.gnbsearchlight import GNBSearchlight, sphere_gnbsearchlight
from mvpa2.base.node import ChainNode
from mvpa2.datasets import Dataset, hstack
from mvpa2.base.hdf5 import h5load
from mvpa2.generators.resampling import Balancer
from mvpa2.mappers.fx import mean_sample
from mvpa2.clfs.gnb import GNB
from mvpa2.clfs.transerror import Confusion
from mvpa2.base.node import Node
from mvpa2.generators.splitters import Splitter
from mvpa2.clfs.transerror import ConfusionMatrixError, ConfusionMatrix
from mvpa2 import debug
if __debug__:
    debug.active += ["SLC"]

class SurfVoxSearchlight():
    
    def __init__(self, ds, clf, prtnr,
                 surf_sl_radius=20, vox_sl_radius=2.5,
                 surf_sl_max_feat=64):
        self._prtnr = prtnr
        self._surf_sl_radius = surf_sl_radius
        self._vox_sl_radius = vox_sl_radius
        self._surf_sl_max_feat = surf_sl_max_feat
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

        spltr = Splitter(attr='partitions',attr_values=[1,2])
        cvte = CrossValidation(
            clf, self._prtnr, splitter=spltr,
            errorfx=errorfx)

        self.slght_surf_confmat = Searchlight(cvte, self._sqe)
        self.slght_vox_confmat = Searchlight(cvte, self._idx_qe)

    def __call__(self, ds):
        
        slmap_surf_confmat = self.slght_surf_confmat(ds[:,:self.max_vertex])
        slmap_vox_confmat = self.slght_vox_confmat(ds[:,self.max_vertex:])
        
        slmap_confmat = Dataset(
            np.concatenate([
                    slmap_surf_confmat.samples.sum(0)[np.newaxis],
                    slmap_vox_confmat.samples.sum(0)[np.newaxis]],1),
            fa=ds.fa,
            a=ds.a)
        slmap_confmat.samples /= slmap_confmat.samples[0,0].sum()

        slmap_accuracy = Dataset(
            slmap_confmat.samples[:,:,np.eye(slmap_confmat.shape[2],dtype=np.bool)].sum(2),
            fa = ds.fa,
            a = ds.a)
        return slmap_confmat, slmap_accuracy

class GNBSurfVoxSearchlight(SurfVoxSearchlight):

    def _setup_surf_vox_searchlight(self, ds, clf):

        errorfx = ConfusionMatrix(labels=ds.uniquetargets)
        spltr = Splitter(attr='partitions',attr_values=[1,2])

        self.slght_surf_confmat = GNBSearchlight(
            clf,
            self._prtnr,
            self._sqe,
            splitter=spltr,
            errorfx=errorfx,
            reuse_neighbors=True)

        self.slght_vox_confmat = GNBSearchlight(
            clf,
            self._prtnr,
            self._idx_qe,
            splitter=spltr,
            errorfx=errorfx,
            reuse_neighbors=True)

def searchlight_delays(ds, prtnr):
    surf_vox_slght = GNBSurfVoxSearchlight(ds, GNB(), prtnr)
    start = -2
    end = 22
    delays = range(start, end)
    slmaps = []
    for d in delays:
        print '######## computing searchlight for delay %d #######'%d
        delay_ds = ds[ds.a.blocks_tr+d]
        delay_ds.targets = ds.a.blocks_targets
        delay_ds.chunks = np.arange(delay_ds.nsamples)
        slmap = surf_vox_slght(delay_ds)
        print '$$$$ delay %d : max accuracy %f'%(d, 1-slmap.samples.min())
        slmaps.append(slmap)
        del delay_ds
    slmap = mvpa2.datasets.vstack(slmaps)
    slmap.sa['delays'] = delays
    del slmaps
    return slmap

def all_searchlights(ds, prtnr, surf_vox_slght=None):
    slmaps = dict()

    if surf_vox_slght is None:
        surf_vox_slght = SurfVoxSearchlight(ds, prtnr)
    slmaps['all'] = surf_vox_slght(ds)
    slmaps['norest'] = surf_vox_slght(ds[ds.sa.targets!='rest'])
    slmaps['instr'] = surf_vox_slght(ds[ds.sa.subtargets=='instr'])
    slmaps['exec'] = surf_vox_slght(ds[ds.sa.subtargets=='exec'])
    
    return slmaps

