import numpy as np
from fluiddata.spectral.wmtsa import modwt
from fluiddata.spectral.wmtsa import dwtArray
import scipy.ndimage.filters

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def compute_extrema(arr, window):
    halfwin = int((window-1)/2)
    rw = np.empty(arr.shape[1])
    arr = np.concatenate([arr[...,-halfwin:], arr, arr[...,:halfwin]],-1)
    rw = rolling_window(arr, window)
    mx = rw.max(-1)
    mn = rw.min(-1)
    del rw
    return mn,mx


def modwt_scales(N,method,wavelet):
    if method == 'liberal':
        NJ = np.ceil(np.log2(N))-1
    elif method == 'extreme':
        NJ = np.ceil(np.log2(N*1.5))-1
    elif method == 'conservative':
        wtf = modwt.wtfilter(wavelet)
        NJ= np.floor(np.log2(N/(wtf.L-1))+1)
    return int(NJ)

def wavelet_despike(tss,
                    wavelet='d4',
                    threshold=2,
                    ratio=.5,
                    boundary='periodic',
                    chsearch='moderate',
                    nscale='liberal',
                    threshold_wavelet_low=None):
    out = np.empty(tss.shape)
    
    NJ = modwt_scales(len(tss), nscale, wavelet)
    cshift = np.asarray([modwt.advance_wavelet_filter(wavelet,j+1) for j in range(NJ)])
    cshift[cshift<-tss.shape[0]] += len(tss)
    cshift = np.abs(cshift)

    WJt = np.empty((NJ,tss.shape[1],len(tss)))
    Wstd = np.empty((NJ,tss.shape[1]))
    for tsi in range(tss.shape[1]):
        WJtmp, _ = modwt.modwt(tss[:,tsi].astype(np.float64), wavelet, NJ, boundary=boundary, RetainVJ=False)
        WJt[:,tsi] = WJtmp
        wvar, CI_wvar, edof, MJ = modwt.wvar(WJtmp)
        Wstd[:,tsi] = np.sqrt(wvar)

    for i in range(NJ):
        WJt[i] = np.roll(WJt[i], -cshift[i], -1)

    mn,mx = compute_extrema(WJt,5)
#    thr_cond = np.abs(WJt)>threshold
    thr_cond = np.abs(WJt)>threshold*Wstd[...,np.newaxis]
    mx2 = np.logical_and(WJt>=ratio*mx, thr_cond)
    mn2 = np.logical_and(WJt<=ratio*mn, thr_cond)
    mx3 = np.logical_and(
        mx2,
        scipy.ndimage.filters.convolve(mx2.astype(np.uint8),np.ones((3,1,5)),mode='constant')>=2)
    mn3 = np.logical_and(
        mn2,
        scipy.ndimage.filters.convolve(mn2.astype(np.uint8),np.ones((3,1,5)),mode='constant')>=2)
    WJt[np.logical_or(mx3,mn3)] = 0
    #WJt[np.logical_or(mx2,mn2)] = 0
    # remove low frequencies
    if threshold_wavelet_low is not None:
        WJt[threshold_wavelet_low:] = 0
    for i in range(NJ):
        WJt[i] = np.roll(WJt[i], cshift[i], -1)
    for tsi in range(tss.shape[1]):
        out[:,tsi] = modwt.imodwt_details(dwtArray(WJt[:,tsi], info=WJtmp.info)).sum(0)
    return out#, -mn3.astype(np.int8)+mx3.astype(np.int8), WJt

def wavelet_despike_loop(tss,
                         wavelet='d4',
                         threshold=2,
                         ratio=.5,
                         boundary='periodic',
                         chsearch='moderate',
                         nscale='liberal',
                         threshold_wavelet_low=None):
    out = np.empty(tss.shape)
    
    NJ = modwt_scales(len(tss), nscale, wavelet)
    cshift = np.asarray([modwt.advance_wavelet_filter(wavelet,j+1) for j in range(NJ)])
    cshift[cshift<-tss.shape[0]] += len(tss)
    cshift = np.abs(cshift)
    kern_conv=np.ones((3,5))

    for tsi in range(tss.shape[1]):
        WJt, _ = modwt.modwt(tss[:,tsi].astype(np.float64), wavelet, NJ, boundary=boundary, RetainVJ=False)
        wvar, CI_wvar, edof, MJ = modwt.wvar(WJt)
        Wstd = np.sqrt(wvar)

        for i in range(NJ):
            WJt[i] = np.roll(WJt[i], -cshift[i], -1)

        mn,mx = compute_extrema(WJt,5)
        #thr_cond = np.abs(WJt)>threshold
        thr_cond = np.abs(WJt)> threshold*Wstd[..., np.newaxis]
        mx2 = np.logical_and(WJt>=ratio*mx, thr_cond)
        mn2 = np.logical_and(WJt<=ratio*mn, thr_cond)
        
        mx3 = np.logical_and(
            mx2,
            scipy.ndimage.filters.convolve(mx2.astype(np.int8),kern_conv,mode='constant')>=2)
        mn3 = np.logical_and(
            mn2,
            scipy.ndimage.filters.convolve(mn2.astype(np.int8),kern_conv,mode='constant')>=2)
        WJt[np.logical_or(mx3,mn3)] = 0
        #WJt[np.logical_or(mx2,mn2)] = 0
        # remove low frequencies
        if threshold_wavelet_low is not None:
            WJt[threshold_wavelet_low:] = 0
        for i in range(NJ):
            WJt[i] = np.roll(WJt[i], cshift[i], -1)
        out[:,tsi] = modwt.imodwt_details(WJt).sum(0)
    return out

