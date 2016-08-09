import numpy as np
from mvpa2.generators.partition import HalfPartitioner, NFoldPartitioner, CustomPartitioner, FactorialPartitioner
from mvpa2.generators.base import Sifter
from mvpa2.generators.resampling import NonContiguous
from mvpa2.base.node import ChainNode, Node
from mvpa2.generators.resampling import Balancer
from mvpa2.generators.splitters import Splitter
from mvpa2.measures.base import Measure, FeaturewiseMeasure, RepeatedMeasure
from mvpa2.datasets import Dataset
from mvpa2.clfs.gda import GDA

class BalancedPartitions(Node):
    def __init__(self, partition_attr='partitions', balanced_attr='balanced_set',
                 space='balanced_partitions',*args, **attr):
        Node.__init__(self, space=space, *args, **attr)
        self.partition_attr = partition_attr
        self.balanced_attr = balanced_attr
    
    def _call(self,ds):
        ds.sa[self.get_space()] = ds.sa[self.partition_attr].value.copy()
        ds.sa[self.get_space()].value[~ds.sa[self.balanced_attr].value] = 3
        return ds

def prtnr_loco_cv(nrepeat=1):
    return ChainNode([
        NFoldPartitioner(attr='chunks'), 
        NonContiguous(dist_attr='time',dist=60),
        Balancer(
            amount='equal',
            attr='targets',
            count=nrepeat,
            apply_selection=False,
            limit=dict(partitions=[1]),
            include_offlimit=True),
        BalancedPartitions()],
        space='balanced_partitions')
    
prtnr_loco_delay = NFoldPartitioner(attr='chunks')

def prtnr_loso_cv(nrepeat=1):
    return ChainNode([
        NFoldPartitioner(attr='scan_id'),
        Balancer(
            amount='equal',
            attr='targets',
            count=nrepeat,
            apply_selection=False,
            limit=dict(partitions=[1]),
            include_offlimit=True),
        BalancedPartitions()],
        space='balanced_partitions')

prtnr_2fold_sift = ChainNode(
    [ NFoldPartitioner(
        attr='chunks',
        cvtype=.5,
        count=32,
        selection_strategy='random'),
      Balancer(                                                                                                             
          amount='equal',                                                                                                    
          attr='targets',                                                                                                    
          count=1,                                                      
          apply_selection=False,                                                                                            
          limit=['partitions'],
          include_offlimit=True),
      BalancedPartitions()],
    space='balanced_partitions')

prtnr_2fold_factpart = FactorialPartitioner(
    NFoldPartitioner(cvtype=2,attr='chunks'),
    attr='targets_num',
    selection_strategy='equidistant',
    count=32)

training_scans = ['d3_mvpa1','d3_mvpa2']

testing_scans = [
    'd1_resting1','d1_training_TSeq','d1_resting2', 'd2_resting3','d1_retest_TSeq_1block',
    'd2_resting1','d2_retest_TSeq_1block','d2_training_IntSeq','d2_resting2','d2_resting3',
    'd3_retest_TSeq','d3_retest_IntSeq','d3_resting1','d3_resting2']

def prtnr_d123_train_test(nrepeat=1):
    return ChainNode([
        CustomPartitioner(
            [(training_scans, testing_scans)],
            attr='scan_name'),
        Balancer(
            amount='equal',
            attr='targets',
            count=nrepeat,
            apply_selection=False,
            limit=dict(partitions=[1]),
            include_offlimit=True),
        BalancedPartitions()],
        space='balanced_partitions')

class FeaturewiseConfusionMatrix(FeaturewiseMeasure):

    def __init__(self, attr='targets', labels=None, *args, **kwargs):
        """
        Parameters
        ----------
        labels : list
          Class labels for confusion matrix columns/rows
        """
        super(FeaturewiseConfusionMatrix, self).__init__(*args, **kwargs)
        self._attr = attr
        self._labels = np.asarray(labels)

    def _call(self, ds):
        nlabels = len(self._labels)
        confmat = np.zeros((1,ds.nfeatures, nlabels, nlabels), dtype=np.uint16) # not many samples -> uint16
        
        tl_test = np.zeros((nlabels,ds.nsamples), dtype=np.bool)
        for ti,tl in enumerate(self._labels):
            tl_test[ti] = ds.get_attr(self._attr)[0].value==tl
        for pl in self._labels:
            pl_test = (ds.samples==pl)
            for ti in range(nlabels):
                confmat[0,:,pl,ti] = np.logical_and(pl_test,tl_test[ti,:,None]).sum(0)
            del pl_test
        res = Dataset(confmat)
        return res
    
    def _pass_attr(self, ds, results):
        # drop all nonunique ds.sa attribute and take sa from a single sample
        ds = ds.copy(deep=False)
#        for attr in ds.sa.keys():
#            if len(ds.sa[attr].unique)>1:
#                del ds.sa[attr]
        return super(FeaturewiseConfusionMatrix, self)._pass_attr(ds[:1], results)

confmat_pass_attr_sa = [
    'scan_name','scan_id', 'blocks_idx','sequence','n_correct_sequences','n_failed_sequences','targets',]
pass_attr_fa  = [
    'ba_thresh', 'ba', 'aparc', 'coordinates','nans','node_indices','voxel_indices', 'roi_sizes']
pass_attr_a = [
    'triangles']
# split block in each scan

confmat_all = FeaturewiseConfusionMatrix(
    attr='targets_num',labels=range(5),
    pass_attr=confmat_pass_attr_sa+pass_attr_fa+pass_attr_a,
    auto_train=True)

# splits scans
split_scans = Splitter(attr='scan_name')
scan_confmat = RepeatedMeasure(
    FeaturewiseConfusionMatrix(
        attr='targets_num',labels=range(5),
        pass_attr=['scan_name','scan_id'],
        auto_train=True),
    ChainNode([
        split_scans,
        Balancer(
            amount='equal',
            attr='targets',
            apply_selection=True),]))
split_scans_blocks = ChainNode([split_scans, Splitter(attr='blocks_idx', ignore_values=[np.nan, -1])])
scan_blocks_confmat = RepeatedMeasure(
    FeaturewiseConfusionMatrix(
        attr='targets_num',labels=range(5), auto_train=True,
        pass_attr=confmat_pass_attr_sa+pass_attr_fa+pass_attr_a),
    split_scans_blocks)

split_trs = Splitter(attr='tr_from_instruction',attr_values=range(-3,7))
delay_confmat = RepeatedMeasure(
    FeaturewiseConfusionMatrix(attr='targets_num',labels=range(5), auto_train=True),
    ChainNode([
        split_trs,
        Balancer(
            amount='equal',
            attr='targets',
            apply_selection=True),]))

class LDAReg(GDA):
    """Linear Discriminant Analysis.
    """

    __tags__ = GDA.__tags__ + ['linear', 'lda']


    def _untrain(self):
        self._w = None
        self._b = None
        super(LDAReg, self)._untrain()


    def _train(self, dataset):
        super(LDAReg, self)._train(dataset)
        nlabels = len(self.ulabels)
        # Sum and scale the covariance
        self.cov = cov = \
            np.sum(self.cov, axis=0) \
            / (np.sum(self.nsamples_per_class) - nlabels)

        # For now as simple as that -- see notes on top
        shrinkage = .01
        mu = np.trace(cov) / dataset.nfeatures
        shrunk_cov = cov* (1-shrinkage)
        shrunk_cov.flat[::dataset.nfeatures+1] += mu*shrinkage
        covi = self._inv(shrunk_cov)

        # Precompute and store the actual separating hyperplane and offset
        self._w = np.dot(covi, self.means.T)
        self._b = b = np.zeros((nlabels,))
        for il in xrange(nlabels):
            m = self.means[il]
            b[il] = np.log(self.priors[il]) - 0.5 * np.dot(np.dot(m.T, covi), m)

    def _g_k(self, data):
        """Return decision function values"""
        return np.dot(data, self._w) + self._b
