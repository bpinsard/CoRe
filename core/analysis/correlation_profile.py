from matplotlib import pyplot
import glob
import numpy as np
import os,sys
from .. import behavior
from ..pipelines.mvpa_pilot import SEQ_INFO

data_dir = '/home/bpinsard/Dropbox/CoRe_AB_EG/01_CoRe_Behav/Data'

subjects = [p.split('/')[-1] for p in sorted(glob.glob(os.path.join(data_dir,'*_CoRe_S*_??')))]
subjects = subjects[:1]
subjects = ['React8hInt_CoRe_S371_EF']
subjects_ids = ['_'.join(s.split('_')[-2:]) for s in subjects]
groups = [s.split('_')[0] for s in subjects]

seqs_idx = [
    ('Training-TSeq-D_One',0),
    ('Training-SeqA-D_One',0),
    ('Reactivation-TSeq-D-Two',0),
    ('Training-IntSeq-D-Two',1),
    ('Testing-TSeq-D-Three',0),
    ('Testing-IntSeq-D-Three',1),
    ('mvpa-1-D-Three',None)]

def sorted_glob_time(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(glob.glob(path), key=mtime))

def perf_profile_corr():
    for sname,sid in zip(subjects,subjects_ids):
        filenames = [(tn, sorted(glob.glob(os.path.join(data_dir,sname,'CoRe_%s_%s_?.mat'%(sid,tn))))) for tn,_ in seqs_idx]
        filenames = [(tn,f[-1]) for tn,f in filenames if len(f)]

        task_data = [
            (tn,behavior.load_behavior(
                    f, seq_info=SEQ_INFO, 
                    seq_idx=([dict(seqs_idx)[tn]]*14) if not tn is None else None)) for tn,f in filenames]
        rtss = [(tn,behavior.blocks_to_rts(td)) for tn,td in task_data]
        all_rtss = np.vstack(reduce(lambda l,x: l+x[1],rtss,[]))

        f,ax = pyplot.subplots(1,3,sharey=True,figsize=(24,8))
        ax[0].matshow(np.corrcoef(all_rtss),aspect='auto')
        ax[1].matshow(np.sqrt((np.abs(all_rtss[:,np.newaxis]-all_rtss)**2)).sum(-1),aspect='auto')
        ax[2].matshow(all_rtss,aspect='auto',vmax=1,vmin=0)
        labels = [tn for tn,rts in rtss]
        ticks = [0]+np.cumsum([sum([len(b) for b in rts]) for tn,rts in rtss]).tolist()[:-1]
        ax[0].set_xticks(ticks)
        ax[1].set_xticks(ticks)
        ax[0].set_xticklabels(labels, rotation=90)
        ax[1].set_xticklabels(labels, rotation=90)
        ax[0].set_yticks(ticks)
        ax[1].set_yticks(ticks)
        ax[0].set_yticklabels(labels)
        ax[2].set_xticks([])
        pyplot.title(sname)
        f.savefig('../projects/CoRe/results/behavior/corr_profiles/corr_prof_%s.pdf'%sname)
        pyplot.close()

        median_rtss = [(tn,np.asarray([np.median(b,0) for b in rts])) for tn,rts in rtss]
        all_median_rtss = np.vstack(reduce(lambda l,x: l+[x[1]],median_rtss,[]))
        f,ax = pyplot.subplots(1,3,sharey=True,figsize=(24,8))
        ax[0].matshow(np.corrcoef(all_median_rtss),aspect='auto')
        ax[1].matshow(np.sqrt((np.abs(all_median_rtss[:,np.newaxis]-all_median_rtss)**2)).sum(-1),aspect='auto')
        ax[2].matshow(all_median_rtss,aspect='auto',vmin=0,vmax=1)
        labels = [tn for tn,rts in median_rtss]
        ticks = [0]+np.cumsum([len(rts) for tn,rts in median_rtss]).tolist()[:-1]
        ax[0].set_xticks(ticks)
        ax[0].set_xticklabels(labels, rotation=90)
        ax[0].set_yticks(ticks)
        ax[0].set_yticklabels(labels)

        ax[1].set_xticks(ticks)
        ax[1].set_xticklabels(labels, rotation=90)
        ax[1].set_yticks(ticks)

        ax[2].set_xticks([])
        pyplot.title(sname)
        f.savefig('../projects/CoRe/results/behavior/corr_profiles/corr_median_prof_%s.pdf'%sname)
        pyplot.close()



def discontinuity_fit(n, a, b, c, y, ):
    print n.shape
    k=n<14
#    r = k* (a+b*n**-c) + (1-k) * (a+b*n**-c-y)
    r = a+b*n**-c + (1-k) * -y

#    print a, b, c, y
    return r
    
import scipy.optimize

tseqs_idx = [
    ('Training-TSeq-D_One',0),
    ('Training-SeqA-D_One',0),
    ('Reactivation-TSeq-D-Two',0),
    ('Testing-TSeq-D-Three',0),]


def fit_learning_curv():
    for sname,sid in zip(subjects,subjects_ids):
        filenames = [(tn, sorted(glob.glob(os.path.join(data_dir,sname,'CoRe_%s_%s_?.mat'%(sid,tn))))) for tn,_ in tseqs_idx]
        filenames = [(tn,f[-1]) for tn,f in filenames if len(f)]

        task_data = [
            (tn,behavior.load_behavior(
                    f, seq_info=SEQ_INFO, 
                    seq_idx=([dict(tseqs_idx)[tn]]*14) if not tn is None else None)) for tn,f in filenames]
        rtss = [(tn,behavior.blocks_to_rts(td)) for tn,td in task_data]
        blocks_mean_rts = np.hstack([[np.nanmean(rt) for rt in rts] for tn,rts in rtss])
        seq_dur = np.hstack([np.nansum(rt,1) for tn,rts in rtss for rt in rts])
#        print blocks_mean_rts
        pyplot.figure()
        pyplot.plot(seq_dur)

        all_rtss = np.vstack(reduce(lambda l,x: l+x[1],rtss,[]))
        pyplot.figure()
        pyplot.plot(all_rtss)
        
        f = pyplot.figure()

        x=np.arange(len(blocks_mean_rts))+1

        pyplot.plot(x[:14],blocks_mean_rts[:14],'+-',color=pyplot.rcParams['axes.color_cycle'][0])
        pyplot.plot(x[14:],blocks_mean_rts[14:],'+-',color=pyplot.rcParams['axes.color_cycle'][0])

        try:
            popt, pcov = scipy.optimize.curve_fit(
                discontinuity_fit,
                x,
                blocks_mean_rts,
                [1,1,1,0])

            print popt
            a,b,c,y=popt
            print y
            pyplot.plot(x[:14],a+b*x[:14]**-c,'+-',color=pyplot.rcParams['axes.color_cycle'][1])
            pyplot.plot(x[14:],a+b*x[14:]**-c-y,'+-',color=pyplot.rcParams['axes.color_cycle'][1])
        except RuntimeError as e:
            print e
            pass
        pyplot.title(sname)
        f.savefig('../projects/CoRe/results/behavior/rt_fits/fit_median_noerror_%s.pdf'%sname)
#        pyplot.close()


