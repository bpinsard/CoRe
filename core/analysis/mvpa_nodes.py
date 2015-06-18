from mvpa2.generators.partition import HalfPartitioner, NFoldPartitioner, CustomPartitioner
from mvpa2.generators.resampling import NonContiguous
from mvpa2.base.node import ChainNode
from mvpa2.generators.resampling import Balancer
from mvpa2.generators.splitters import Splitter

prtnr_loco_cv = ChainNode([
        NFoldPartitioner(attr='chunks'), 
        NonContiguous(dist_attr='time',dist=60)])
prtnr_loso_cv = NFoldPartitioner(attr='scan_id')

prtnr_mvpa2_retest = CustomPartitioner(
    [(['d3_mvpa1'],['d3_retest_TSeq','d3_retest_IntSeq','d3_resting1','d3_resting2']),
     (['d3_mvpa2'],['d3_retest_TSeq','d3_retest_IntSeq','d3_resting1','d3_resting2']),
     (['d3_mvpa1','d3_mvpa2'],['d3_retest_TSeq','d3_retest_IntSeq','d3_resting1','d3_resting2']),
     ], attr='scan_name')

