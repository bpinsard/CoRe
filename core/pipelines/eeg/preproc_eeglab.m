function EEG=preproc_eeglab(path, filename, tr, outpath)

[EEG,com] = pop_loadbv(path,filename);

% compute a number of component depending on the length of recording
%ncomp = round(log(EEG.pnts/EEG.srate/60)*2+3);
ncomp = round((EEG.pnts/EEG.srate/60)/6+3)

etype='R  1';
Trigs=[];
for E=1:length(EEG.event)
    if strcmp(EEG.event(E).type,etype)
        Trigs(end+1)=round(EEG.event(E).latency);
    end
end

% add dummy triggers
%Trigs = [linspace(-3,-1,3)*(EEG.srate*tr)+Trigs(1) Trigs Trigs(end)+EEG.srate*tr];

ecg_chans = [];
for i=1:EEG.nbchan
    chanlab = EEG.chanlocs(i).labels;
    if length(strfind(chanlab,'ECG')) > 0
        ecg_chans(end+1) = i;
    end
end
ecg_chans

EEG_fastr=fmrib_fastr(EEG,70,4,30,Trigs,0,0,0,0,0,0.03,ecg_chans,'auto');
EEG_qrs = EEG_fastr;

qrs_events = {};
EEG_qrs = EEG_fastr;
for i=1:EEG_fastr.nbchan
    chanlab = EEG_fastr.chanlocs(i).labels;
    if length(strfind(chanlab,'ECG')) > 0
        qrs_event = ['qrs_' chanlab];
        qrs_events{end+1} = qrs_event;
        EEG_qrs=pop_fmrib_qrsdetect(EEG_qrs,i,qrs_event,'no');
    end
end

EEG_qrs = correct_qrs(EEG_qrs, qrs_events);

EEG_downsample = pop_resample(EEG_qrs, 250);

pop_writebva(EEG_downsample,[outpath '/' filename(1:end-5) '_gca.vhdr']);

EEG_pas = pop_fmrib_pas(EEG_downsample,'qrs_ECG','obs',ncomp);

EEG_filt = pop_eegfiltnew(EEG_qrs, .5, 35);

pop_writebva(EEG_pas,[outpath '/' filename(1:end-5) '_gca_pas_filt.vhdr']);
