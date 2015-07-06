import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
color_cycle = plt.style.library['ggplot']['axes.color_cycle']

uniq_seq = ['CoReTSeq', 'CoReIntSeq', 'mvpa_CoReOtherSeq1', 'mvpa_CoReOtherSeq2']

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
