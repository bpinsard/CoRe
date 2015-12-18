function EEG=preproc_eeglab(path, filename, tr, outpath)

gca_file = [outpath '/' filename(1:end-5) '_gca_mx.vhdr'];
gca_file_vec = [filename(1:end-5) '_gca.vhdr'];

if (exist(gca_file,'file') ~= 2) & (exist([outpath '/' gca_file_vec],'file') == 2)
    disp('vec to mx');
    [EEG_vec,com] = pop_loadbv(outpath, gca_file_vec);
    pop_writebva(EEG_vec, gca_file, 'MULTIPLEXED');
end

if (exist(gca_file,'file') ~= 2)
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

% remove drifts
EEG_hp = pop_eegfiltnew(EEG_qrs, .2, []);

EEG_downsample = pop_resample(EEG_hp, 250);
clear EEG EEG_fastr EEG_qrs EEG_hp
pop_writebva(EEG_downsample,gca_file,'MULTIPLEXED');
del EEG_downsample
end

addpath('~/softs/FASST');

[EEG_fasst,Fdata] = crc_eeg_rdata_brpr(gca_file);
D_cica=crc_par([EEG_fasst.path '/' EEG_fasst.fname],struct('bcgmethod','acica','ecgchan',32,'badchan',[32,65]));

function Trigs=find_trig(EEG,etype)
Trigs=[];
for E=1:length(EEG.event)
    if strcmp(EEG.event(E).code(1:length(etype)),etype)
        Trigs(end+1)=round(EEG.event(E).latency);
    end;
end;
