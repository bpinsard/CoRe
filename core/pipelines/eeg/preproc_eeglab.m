function EEG=preproc_eeglab(path, filename, tr, outpath)

gca_file = [outpath '/' filename(1:end-5) '_gca.vhdr'];

if exist(gca_file,'file') ~= 2
[EEG,com] = pop_loadbv(path,filename);

% compute a number of component depending on the length of recording
ncomp = round(log(EEG.pnts/EEG.srate/60)+3)
%ncomp = round((EEG.pnts/EEG.srate/60)/6+3)

Trigs=find_trig(EEG,'Volume');
fprintf('%d Volume Triggers found\n',length(Trigs));
%Trigs = find_trig(EEG,'R  1');
if length(Trigs)==0
  Trigs=find_trig(EEG,'Stimulus');
  fprintf('%d Stimulus Triggers found\n',length(Trigs));
end

% add dummy triggers
%Trigs = [linspace(-3,-1,3)*(EEG.srate*tr)+Trigs(1) Trigs Trigs(end)+EEG.srate*tr];

ecg_chans = [];
for i=1:EEG.nbchan
    chanlab = EEG.chanlocs(i).labels;
    if length(strfind(chanlab,'ECG')) > 0
        ecg_chans(end+1) = i;
    end;
end;
ecg_chans

EEG_fastr=fmrib_fastr(EEG,70,4,30,Trigs,0,0,0,0,0,0.03,ecg_chans,'auto');
EEG_qrs = pop_select(EEG_fastr,'point',[Trigs(1)-1 Trigs(end)+1]);

qrs_events = {};
for i=1:EEG_qrs.nbchan
    chanlab = EEG_qrs.chanlocs(i).labels;
    if length(strfind(chanlab,'ECG')) > 0
        qrs_event = ['qrs_' chanlab];
        qrs_events{end+1} = qrs_event;
        EEG_qrs=pop_fmrib_qrsdetect(EEG_qrs,i,qrs_event,'no');
    end;
end;

EEG_qrs = correct_qrs(EEG_qrs, qrs_events);
%pop_writebva(EEG_qrs,[outpath '/' filename(1:end-5) '_gca_hr.vhdr']);

% remove drifts
EEG_hp = pop_eegfiltnew(EEG_qrs, .2, []);

EEG_downsample = pop_resample(EEG_hp, 250);
clear EEG EEG_fastr EEG_qrs EEG_hp
pop_writebva(EEG_downsample,gca_file);
else
disp('load existing gca file');
[pathstr,name,ext] = fileparts(gca_file);
EEG_downsample = pop_loadbv(pathstr,[name,ext]);
ncomp = round(log(EEG_downsample.pnts/EEG_downsample.srate/60)+3);
qrs_events = {};
for i=1:EEG_downsample.nbchan
    chanlab = EEG_downsample.chanlocs(i).labels;
    if length(strfind(chanlab,'ECG')) > 0
        qrs_event = ['qrs_' chanlab];
        qrs_events{end+1} = qrs_event;
    end
end
end

EEG_pas = pop_fmrib_pas(EEG_downsample,qrs_events{2},'obs',ncomp);
pop_writebva(EEG_pas,[outpath '/' filename(1:end-5) '_gca_pas.vhdr']);

%save([outpath filename(1:end-5) '_gca_pas_pca.mat'], '-struct','EEG_pas','pca','-v7.3')

clear EEG_downsample
%EEG_filt = pop_eegfiltnew(EEG_pas, .5, 35);
%clear EEG_pas

%pop_writebva(EEG_filt,[outpath '/' filename(1:end-5) '_gca_pas_filt.vhdr']);

function Trigs=find_trig(EEG,etype)
Trigs=[];
for E=1:length(EEG.event)
    if strcmp(EEG.event(E).code(1:length(etype)),etype)
        Trigs(end+1)=round(EEG.event(E).latency);
    end;
end;
