import os,sys,glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
color_cycle = plt.style.library['ggplot']['axes.color_cycle']

uniq_seq = ['CoReTSeq', 'CoReIntSeq', 'mvpa_CoReOtherSeq1', 'mvpa_CoReOtherSeq2']
seq_article_names = ['Trained sequence 1', 'Trained sequence 2', 'New sequence 1', 'New sequence 2']
seq_article_shortnames = ['TSeq1', 'TSeq2', 'NewSeq1', 'NewSeq2']


def plot(blocks, uniq_seq = None):
    
    if uniq_seq is None:
        uniq_seq = np.unique([b[0] for b in blocks])    
    good_sequence_rts = [(b[0],np.hstack([sq['rt_pre'][1:] for sq in b[-1] if np.all(sq['match'])])) for b in blocks]
    number_of_errors = [(b[0], sum([np.any(~sq['match']) for sq in b[-1] ])) for b in blocks]

    good_sequence_exec_time = [(b[0],np.hstack([sq['time'][-1]-sq['time'][0] for sq in b[-1] if np.all(sq['match']) and len(sq)==5])) for b in blocks]
    # time from first key press to the key press completing the blocks (not the last executed)
    blocks_exec_time = [(b[0], np.asarray(b[4]-b[5])) for b in blocks]
    
    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(20,10))

    axes[0,0].bar(left=[0]*len(uniq_seq),
                  bottom = -np.arange(len(uniq_seq)),
                  width = [np.hstack([gsrts[1] for gsrts in good_sequence_rts if gsrts[0]==seq]).mean() for seq in uniq_seq],
                  height = [.5]*len(uniq_seq),
                  xerr= [np.hstack([gsrts[1] for gsrts in good_sequence_rts if gsrts[0]==seq]).std() for seq in uniq_seq],
                  orientation='horizontal',
                  color=color_cycle,
                  ecolor='k')
    axes[0,0].set_yticks(-np.arange(len(uniq_seq))+.25)
    axes[0,0].set_yticklabels(uniq_seq)
    axes[0,0].set_xlabel('RT')

    axes[0,1].bar(left=[0]*len(uniq_seq),
                  bottom = -np.arange(len(uniq_seq)),
                  width = [sum([noe[1] for noe in number_of_errors if noe[0]==seq]) for seq in uniq_seq],
                  height = [.5]*len(uniq_seq),                  
                  orientation='horizontal',
                  color=color_cycle)
    axes[0,1].set_xlabel('Error count')
    axes[0,1].set_yticks([])
     
    axes[0,2].bar(left=[0]*len(uniq_seq),
                  bottom = -np.arange(len(uniq_seq)),
                  width = [np.hstack([gset[1] for gset in good_sequence_exec_time if gset[0]==seq]).mean() for seq in uniq_seq],
                  height = [.5]*len(uniq_seq),                  
                  xerr= [np.hstack([gset[1] for gset in good_sequence_exec_time if gset[0]==seq]).std() for seq in uniq_seq],
                  orientation='horizontal',
                  color=color_cycle,
                  ecolor='k')
    axes[0,2].set_xlabel('sequence execution time')
    axes[0,2].set_yticks([])

    axes[0,3].bar(left=[0]*len(uniq_seq),
                  bottom = -np.arange(len(uniq_seq)),
                  width = [np.mean([bet[1] for bet in blocks_exec_time if bet[0]==seq]) for seq in uniq_seq],
                  height = [.5]*len(uniq_seq),                  
                  xerr= [np.std([bet[1] for bet in blocks_exec_time if bet[0]==seq]) for seq in uniq_seq],
                  orientation='horizontal',
                  color=color_cycle,
                  ecolor='k')
    axes[0,3].set_xlabel('blocks execution time')
    axes[0,3].set_yticks([])


    for li,l in enumerate(['RT','errors','sequence execution time','block execution time']):
        axes[1,li].set_ylabel(l)


    max_nblocks = 0
    for seq in uniq_seq:
        nblocks=len([gsrts[0] for gsrts in good_sequence_rts if gsrts[0]==seq])
        if nblocks > max_nblocks:
            max_nblocks = nblocks
        axes[1,0].errorbar(
            x=np.arange(nblocks)+1,
            y=[gsrts[1].mean() for gsrts in good_sequence_rts if gsrts[0]==seq],
            yerr=[gsrts[1].std() for gsrts in good_sequence_rts if gsrts[0]==seq])
        axes[1,1].plot(np.arange(nblocks)+1, [noe[1] for noe in number_of_errors if noe[0]==seq],'-+')
        
        axes[1,2].errorbar(
            x=np.arange(nblocks)+1,
            y=[gset[1].mean() for gset in good_sequence_exec_time if gset[0]==seq],
            yerr=[gset[1].std() for gset in good_sequence_exec_time if gset[0]==seq])
        
        axes[1,3].plot(np.arange(nblocks)+1, [bet[1] for bet in blocks_exec_time if bet[0]==seq],'-+')

    plt.legend(uniq_seq, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    for axi in range(4):
        axes[1,axi].set_xlabel('block')
        axes[1,axi].set_xticks(np.arange(max_nblocks)+1)
        axes[1,axi].set_xlim(0,max_nblocks+1)


"""
import sys
sys.path.insert(0,'/home/bpinsard/data/projects/CoRe/code/')
import glob
import behavior, stats_mvpa_task
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
mvpas=[(f,behavior.load_behavior(f)) for f in glob.glob('2015-06-05_output_new/CoRe_S*_??_mvpa-1-D-Three_1.mat')]
with PdfPages('/home/bpinsard/data/tests/behavior.pdf') as pdf:
    for f,blocks in mvpas[1:]:
        stats_mvpa_task.plot(blocks,stats_mvpa_task.uniq_seq)
        plt.suptitle(f)
        pdf.savefig()
        plt.close()
"""

data_dir = '/home/bpinsard/data/raw/UNF/CoRe/Behavior/'

def get_correct_seq_dur(blocks,seq):
    return np.asarray([np.mean([sq['rt_pre'][1:].sum() for sq in b[-1] if np.all(sq['match']) and len(sq)==5]) \
                       for b in blocks if b[0] in seq])

def get_num_correct_seq(blocks,seq):
    return np.asarray([sum([np.all(sq['match']) and len(sq)==5 for sq in b[-1] ]) \
                       for b in blocks if b[0] in seq])

def get_block_dur(blocks,seq):
    return np.asarray([np.hstack(b[-1])['rt_pre'][1:60].sum() for b in blocks if b[0] in seq])

from . import core_rsa as core_rsa
from .. import behavior as core_behavior
import pandas
def mvpa_stats():

    mvpa1_files = [sorted(glob.glob(os.path.join(data_dir,'CoRe_%03d_D3/CoRe_%03d_mvpa-1-D-Three_*.mat'%(s,s))))[-1] \
                   for s in core_rsa.group_Int]
    mvpa2_files = [sorted(glob.glob(os.path.join(data_dir,'CoRe_%03d_D3/CoRe_%03d_mvpa-2-D-Three_*.mat'%(s,s))))[-1] \
                   for s in core_rsa.group_Int]

    mvpa_blocks = [core_behavior.load_behavior(m1)+core_behavior.load_behavior(m2) \
                   for m1,m2 in zip(mvpa1_files,mvpa2_files)]
    
    mean_seq_duration_per_block = dict([(seq,np.asarray([get_correct_seq_dur(m,seq) for m in mvpa_blocks]))
                                        for seq in uniq_seq])
    num_correct_seq_per_block = dict([(seq,np.asarray([get_num_correct_seq(m,seq) for m in mvpa_blocks]))
                                      for seq in uniq_seq])
    


    color_cycle_elife = ['#90CAF9','#FFB74D','#9E86C9','#E57373']
    
    f,ax=plt.subplots()
    for seqi,seq in enumerate(uniq_seq):
        dur_mean = np.nanmean(mean_seq_duration_per_block[seq],0)
        dur_std = np.nanstd(mean_seq_duration_per_block[seq],0)
        ax.bar(width=1,left=np.arange(16)*5+seqi,height=dur_mean,
               yerr=dur_std,
               color=color_cycle_elife[seqi],
               error_kw=dict(ecolor=color_cycle_elife[seqi]))
        ax.legend(seq_article_names,fontsize='medium')
        ax.set_ylabel('average sequence execution duration (s)')
        ax.set_xlabel('mvpa task block')
        ax.set_xticks(np.arange(16)*5+2)
        ax.set_xticklabels(['# %d'%bi for bi in range(1,17)])
        ax.grid(axis='x')
        ax.set_ylim(0,2.6)
        ax.tick_params(axis='x', which='both',length=0)
        


    nsubj = len(core_rsa.group_Int)

    seq_consolidated = np.asarray([True]*2+[False]*2)[np.newaxis,:,np.newaxis].repeat(nsubj,0).repeat(16,2).ravel()
    # flat order: subject, sequence, block
    data = pandas.DataFrame(dict(
        subjects = np.asarray(core_rsa.group_Int)[:,np.newaxis].repeat(16*4).ravel(),
        sequences = np.asarray(seq_article_shortnames)[np.newaxis,:,np.newaxis].repeat(nsubj,0).repeat(16,2).ravel(),
        seq_consolidated = seq_consolidated,
        seq_new = np.logical_not(seq_consolidated),
        blocks = np.arange(1,17)[np.newaxis].repeat(nsubj*4,0).ravel(),
        mean_seq_duration = np.hstack([mean_seq_duration_per_block[seq][:,np.newaxis] for seq in uniq_seq]).ravel(),
        num_correct_seq = np.hstack([num_correct_seq_per_block[seq][:,np.newaxis] for seq in uniq_seq]).ravel()
    ))
    

    return data
    
    md_seq_duration = smf.mixedlm(
        "mean_seq_duration ~ seq_new + seq_new*blocks",
        data_rem_missing,groups=data_rem_missing['subjects'],re_formula='~blocks')

    md_num_correct_seq = smf.mixedlm(
        "num_correct_seq ~ seq_new + seq_new*blocks",
        data, groups=data['subjects'], re_formula='~blocks')
    
