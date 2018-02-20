import os,sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
color_cycle = plt.style.library['ggplot']['axes.color_cycle']

uniq_seq = ['CoReTSeq', 'CoReIntSeq', 'mvpa_CoReOtherSeq1', 'mvpa_CoReOtherSeq2']
seq_article_names = ['TSeq', 'IntSeq', 'NewSeq1', 'NewSeq2']

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


def mvpa_stats():

    mvpa1_files = [sorted(glob.glob(os.path.join(data_dir,'CoRe_%03d_D3/CoRe_%03d_mvpa-1-D-Three_*.mat'%(s,s))))[-1] \
                   for s in core.analysis.core_rsa.group_Int]
    mvpa2_files = [sorted(glob.glob(os.path.join(data_dir,'CoRe_%03d_D3/CoRe_%03d_mvpa-2-D-Three_*.mat'%(s,s))))[-1] \
                   for s in core.analysis.core_rsa.group_Int]

    mvpa_blocks = [core.behavior.load_behavior(m1)+core.behavior.load_behavior(m2) \
                   for m1,m2 in zip(mvpa1_files,mvpa2_files)]
    
    mean_seq_duration_per_block = dict([(seq,np.asarray([get_correct_seq_dur(m,seq) for m in mvpa_blocks])) for seq in uniq_seq])
    mean_num_correct_seq_per_block = dict([(seq,np.asarray([get_num_correct_seq(m,seq) for m in mvpa_blocks])) for seq in uniq_seq])
    

    scipy.stats.ttest_rel(np.nanmean(mean_seq_duration_per_block['CoReTSeq'],1),
                          np.nanmean(mean_seq_duration_per_block['CoReIntSeq'],1))
    #Ttest_relResult(statistic=-1.8905212446386963, pvalue=0.075859215284321824)
    scipy.stats.ttest_rel(np.nanmean(mean_seq_duration_per_block['mvpa_CoReOtherSeq1'],1),
                          np.nanmean(mean_seq_duration_per_block['mvpa_CoReOtherSeq2'],1))
    #Ttest_relResult(statistic=0.82409522866903795, pvalue=0.42129767376196381)

    scipy.stats.ttest_rel(np.nanmean(mean_seq_duration_per_block['CoReTSeq'],1)+
                          np.nanmean(mean_seq_duration_per_block['CoReIntSeq'],1),
                          np.nanmean(mean_seq_duration_per_block['mvpa_CoReOtherSeq1'],1)+
                          np.nanmean(mean_seq_duration_per_block['mvpa_CoReOtherSeq2'],1))
    #Ttest_relResult(statistic=-5.6066375364018368, pvalue=3.1445329252915e-05)
    (np.nanmean(mean_seq_duration_per_block['CoReTSeq'],1)+
     np.nanmean(mean_seq_duration_per_block['CoReIntSeq'],1))/2-(
         np.nanmean(mean_seq_duration_per_block['mvpa_CoReOtherSeq1'],1)+
         np.nanmean(mean_seq_duration_per_block['mvpa_CoReOtherSeq2'],1))/2
    

    scipy.stats.ttest_rel(
        mean_num_correct_seq_per_block['CoReTSeq'].mean(1),
        mean_num_correct_seq_per_block['CoReIntSeq'].mean(1))
    #Out[252]: Ttest_relResult(statistic=-1.3867504905630728, pvalue=0.18343486443854889)

    scipy.stats.ttest_rel(
        mean_num_correct_seq_per_block['mvpa_CoReOtherSeq1'].mean(1),
        mean_num_correct_seq_per_block['mvpa_CoReOtherSeq2'].mean(1))
    #Out[253]: Ttest_relResult(statistic=0.55683017708274662, pvalue=0.58490030737698162)

    scipy.stats.ttest_rel(
        mean_num_correct_seq_per_block['CoReTSeq'].mean(1)+\
        mean_num_correct_seq_per_block['CoReIntSeq'].mean(1),
        mean_num_correct_seq_per_block['mvpa_CoReOtherSeq1'].mean(1)+\
        mean_num_correct_seq_per_block['mvpa_CoReOtherSeq2'].mean(1))
    #Out[251]: Ttest_relResult(statistic=2.8639420742641848, pvalue=0.010752736412558923)

    scipy.stats.ttest_rel(mean_seq_duration_per_block['mvpa_CoReOtherSeq1'][:,0]+
                          mean_seq_duration_per_block['mvpa_CoReOtherSeq2'][:,0],
                          mean_seq_duration_per_block['mvpa_CoReOtherSeq1'][:,-1]+
                          mean_seq_duration_per_block['mvpa_CoReOtherSeq2'][:,-1])
    #Out[380]: Ttest_relResult(statistic=3.7829720168998384, pvalue=0.001484733433173324)

    scipy.stats.ttest_rel(mean_num_correct_seq_per_block['mvpa_CoReOtherSeq1'][:,0]+
                          mean_num_correct_seq_per_block['mvpa_CoReOtherSeq2'][:,0],
                          mean_num_correct_seq_per_block['mvpa_CoReOtherSeq1'][:,-1]+
                          mean_num_correct_seq_per_block['mvpa_CoReOtherSeq2'][:,-1])
    #Out[381]: Ttest_relResult(statistic=-0.97182531580754983, pvalue=0.34476303104414352)1


    scipy.stats.ttest_rel(mean_num_correct_seq_per_block['CoReTSeq'][:,0]+
                          mean_num_correct_seq_per_block['CoReIntSeq'][:,0],
                          mean_num_correct_seq_per_block['CoReTSeq'][:,-1]+
                          mean_num_correct_seq_per_block['CoReIntSeq'][:,-1])
    #Out[390]: Ttest_relResult(statistic=0.19458138494594823, pvalue=0.84802706229304969)

    scipy.stats.ttest_rel(mean_seq_duration_per_block['CoReTSeq'][:,0]+
                          mean_seq_duration_per_block['CoReIntSeq'][:,0],
                          mean_seq_duration_per_block['CoReTSeq'][:,-1]+
                          mean_seq_duration_per_block['CoReIntSeq'][:,-1])
    #Out[391]: Ttest_relResult(statistic=2.4917714996317795, pvalue=0.023337365371333949)


    color_cycle_elife = ['#90CAF9','#FFB74D','#9E86C9','#E57373']
    
    f,ax=subplots()
    for seqi,seq in enumerate(uniq_seq):
        dur_mean = np.nanmean(mean_seq_duration_per_block[seq],0)
        dur_std = np.nanstd(mean_seq_duration_per_block[seq],0)
        ax.bar(width=1,left=np.arange(16)*5+seqi,height=dur_mean,
               yerr=dur_std,
               color=color_cycle_elife[seqi],
               error_kw=dict(ecolor=color_cycle_bars[seqi]))
    ax.legend(seq_article_names,fontsize='medium')
    ax.set_ylabel('average sequence execution duration (s)')
    ax.set_xlabel('mvpa task block')
    ax.set_xticks(np.arange(16)*5+2)
    ax.set_xticklabels(['# %d'%bi for bi in range(1,17)])
    ax.grid(axis='x')
    ax.set_ylim(0,2.6)
    ax.tick_params(axis='x', which='both',length=0)
    
