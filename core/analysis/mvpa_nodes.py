from mvpa2.generators.partition import HalfPartitioner, NFoldPartitioner, CustomPartitioner
from mvpa2.generators.resampling import NonContiguous
from mvpa2.base.node import ChainNode
from mvpa2.generators.resampling import Balancer
from mvpa2.generators.splitters import Splitter

prtnr_loco_cv = ChainNode([
        NFoldPartitioner(attr='chunks'), 
        NonContiguous(dist_attr='time',dist=60)])
prtnr_loso_cv = NFoldPartitioner(attr='scan_id')

prtnr_d3_retest = CustomPartitioner(
    [(learning_set,['d3_retest_TSeq','d3_retest_IntSeq','d3_resting1'])\
     for learning_set in [['d3_mvpa1'],['d3_mvpa2'],['d3_mvpa1','d3_mvpa2']]], attr='scan_name')

prtnr_d3_retest = ChainNode([
    prtnr_d3_retest,
    Balancer(amount='equal', attr='targets', count=5, apply_selection=True, limit='partitions')])

prtnr_d1d2_training = CustomPartitioner(
    [(learning_set,['d1_resting1','d1_training_TSeq','d1_retest_TSeq_1block',
                    'd2_resting1','d2_retest_TSeq_1block','d2_training_IntSeq']) \
     for learning_set in [['d3_mvpa1'],['d3_mvpa2'],['d3_mvpa1','d3_mvpa2']]], attr='scan_name')

prtnr_d1d2_training = ChainNode([
    prtnr_d1d2_training,
    Balancer(amount='equal', attr='targets', count=5, apply_selection=True, limit='partitions')])
