from mvpa2.generators.partition import HalfPartitioner, NFoldPartitioner, CustomPartitioner
from mvpa2.generators.resampling import NonContiguous
from mvpa2.base.node import ChainNode
from mvpa2.generators.resampling import Balancer
from mvpa2.generators.splitters import Splitter

prtnr_loco_cv = ChainNode([
        NFoldPartitioner(attr='chunks'), 
        NonContiguous(dist_attr='time',dist=60)])
prtnr_loso_cv = NFoldPartitioner(attr='scan_id')

prtnr_mvpa2_retest_pilot = CustomPartitioner(
    [(['d3_mvpa1'],['d3_retest_TSeq','d3_retest_IntSeq','d3_resting1','d3_resting2']),
     (['d3_mvpa2'],['d3_retest_TSeq','d3_retest_IntSeq','d3_resting1','d3_resting2']),
     (['d3_mvpa1','d3_mvpa2'],['d3_retest_TSeq','d3_retest_IntSeq','d3_resting1','d3_resting2']),
     ], attr='scan_name')

prtnr_mvpa_d1 = CustomPartitioner(
    [(learning_set,['d1_resting1','d1_training_TSeq','d1_resting2','d1_resting3','d1_retest_TSeq_1block']) \
         for learning_set in [['d3_mvpa1'],['d3_mvpa2'],['d3_mvpa1','d3_mvpa2']]])
prtnr_mvpa_d2 = CustomPartitioner(
    [])
