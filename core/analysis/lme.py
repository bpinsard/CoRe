import os, sys, glob
import numpy as np
from .. import behavior
from ..pipelines.mvpa_pilot import SEQ_INFO
from numpy.lib.recfunctions import append_fields

data_dir = '/home/bpinsard/Dropbox/CoRe_AB_EG/01_CoRe_Behav/data'

subjects = [p.split('/')[-1] for p in sorted(glob.glob(os.path.join(data_dir,'*_CoRe_S*_??')))]
#subjects = subjects[:1]
#subjects = ['React8hInt_CoRe_S371_EF']
subjects_ids = ['_'.join(s.split('_')[-2:]) for s in subjects]
groups = [s.split('_')[0] for s in subjects]

seqs_idx = [
    ('Training-TSeq-D_One',0),
    ('Training-SeqA-D_One',0),
    ('Reactivation-TSeq-D-Two',0),
    ('Training-IntSeq-D-Two',1),
    ('Testing-TSeq-D-Three',0),
    ('Testing-IntSeq-D-Three',1),
#    ('mvpa-1-D-Three',None)
]

compulsory = [seqs_idx[i][0] for i in [0,2,4,5]]


def data_for_lme():
    all_data = []
    for sname,sid in zip(subjects,subjects_ids):
        filenames = [(tn, sorted(glob.glob(os.path.join(data_dir,sname,'CoRe_%s_%s_?.mat'%(sid,tn))))) for tn,_ in seqs_idx]
        filenames = [(tn,f[-1]) for tn,f in filenames if len(f)]

        tns = [f[0] for f in filenames]
        for tn in compulsory:
            if tn not in tns:
                continue

        task_data = [
            (tn,behavior.load_behavior(
                    f, seq_info=SEQ_INFO, 
                seq_idx=([dict(seqs_idx)[tn]]*14) if not tn is None else None)) for tn,f in filenames]

        group = sname.split('_')[0]
        subj_data = []
        
        for tn, blocks in task_data:
            rt_all = behavior.blocks_to_rt_all(blocks)
            nlines = len(rt_all)
            if tn=='Training-SeqA-D_One':
                tn='Training-TSeq-D_One'
            rt_all = append_fields(
                rt_all,
                ['group','subject_id','subject','task'], 
                [[group]*nlines,[sid]*nlines,[sname]*nlines,[tn]*nlines],
                [np.dtype('S64')]*4,
                usemask=False)
            subj_data.append(rt_all)
        subj_data = np.hstack(subj_data)
        subj_data['correct_seq_idx'] = np.asarray([
            np.cumsum((np.ediff1d(subj_data['correct_seq_idx']*(subj_data['sequence']==seq),to_begin=[0])>0))*(subj_data['sequence']==seq) for seq in np.unique(subj_data['sequence'])]).sum(0)+1
        all_data.append(subj_data)
    return np.hstack(all_data)
            
