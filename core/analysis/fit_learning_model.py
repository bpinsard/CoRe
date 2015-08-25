from matplotlib import pyplot
import glob
import numpy as np
import os,sys
from .. import behavior
from ..pipelines.mvpa_pilot import SEQ_INFO
import scipy.optimize


data_dir = '/home/bpinsard/Dropbox/CoRe_AB_EG/01_CoRe_Behav/Data'

sname = 'ReactNoInt_CoRe_S381_AC'
sid = '_'.join(sname.split('_')[-2:]) 

tseqs_idx = [
    ('Training-TSeq-D_One',0),
    ('Training-SeqA-D_One',0),
    ('Reactivation-TSeq-D-Two',0),
    ('Testing-TSeq-D-Three',0),]


def fit_data(sname, sid):
    filenames = [(tn, sorted(glob.glob(os.path.join(data_dir,sname,'CoRe_%s_%s_?.mat'%(sid,tn))))) for tn,_ in tseqs_idx]
    filenames = [(tn,f[-1]) for tn,f in filenames if len(f)]
    filenames = filenames[:2]
    task_data = [
        (tn,behavior.load_behavior(
                f, seq_info=SEQ_INFO, 
                seq_idx=([dict(tseqs_idx)[tn]]*14) if not tn is None else None)) for tn,f in filenames]
    rtss = [(tn,behavior.blocks_to_rts(td)) for tn,td in task_data]
    all_rtss = np.vstack(reduce(lambda l,x: l+x[1],rtss,[]))

    print np.count_nonzero(np.isnan(all_rtss))

    train_end = sum([len(rt) for rt in rtss[0][1]])    
    print train_end
    
    ss = np.hstack([np.arange(len(r)) for tn,rts in rtss for r in rts])
    kk = np.arange(len(ss)) < train_end
    xx = np.arange(len(ss))+1

    f = pyplot.figure()
    pyplot.plot(xx, all_rtss)
    pyplot.legend(range(5))

    for i in range(all_rtss.shape[1]):
        try:
            print i
            nan_mask = ~np.isnan(all_rtss[:,i])
            k=kk[nan_mask]
            x=xx[nan_mask]
            def discontinuity_fit(n, a, b, c, y):
                r = a+b*n**-c + (1-k) * -y #+ f*s*(n-t)
                return r
            popt, pcov = scipy.optimize.curve_fit(
                discontinuity_fit,
                x,
                all_rtss[nan_mask,i],
                [.5,.5,.1,0,])
        
            print popt
            a,b,c,y=popt
            pyplot.plot(xx[:train_end],a+b*xx[:train_end]**-c,'+-',color=pyplot.rcParams['axes.color_cycle'][i])
            pyplot.plot(xx[train_end:],a+b*xx[train_end:]**-c-y,'+-',color=pyplot.rcParams['axes.color_cycle'][i])
#        a,b,c,y,f,t=popt
        except RuntimeError as e:
            print e
