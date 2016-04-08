function EEG=preproc_eeglab_fasst(path, filename, tr, outpath)

gca_file = fullfile(outpath, [filename(1:end-5) '_gca_mx.vhdr']);
gca_file_vec = [filename(1:end-5) '_gca.vhdr'];

if (exist(gca_file,'file') ~= 2) & (exist(fullfile(outpath,gca_file_vec),'file') == 2)
    disp('vec to mx');
    [EEG_vec,com] = pop_loadbv(outpath, gca_file_vec);
    pop_writebva(EEG_vec, gca_file, 'MULTIPLEXED');
    clear EEG_vec
end

if (exist(gca_file,'file') ~= 2)
disp('starting from raw data')
fullfile(path,filename)
[EEG,com] = pop_loadbv(path,filename);


Trigs=find_trig(EEG,'Volume');
fprintf('%d Volume Triggers found\n',length(Trigs));
%Trigs = find_trig(EEG,'R  1');
if length(Trigs)==0
  Trigs=find_trig(EEG,'Stimulus');
  fprintf('%d Stimulus Triggers found\n',length(Trigs));
end

% add dummy triggers
%Trigs = [linspace(-3,-1,3)*(EEG.srate*tr)+Trigs(1) Trigs Trigs(end)+EEG.srate*tr];

ecg_chans = get_ecg_chans(EEG)

EEG_fastr = fmrib_fastr(EEG,70,4,30,Trigs,0,0,0,0,0,0.03,ecg_chans,'auto');
EEG_qrs = pop_select(EEG_fastr,'point',[Trigs(1)-1 Trigs(end)+1]);
EEG_qrs.event(1).code = 'New Segment';
%EEG_qrs.event(1).bvtime = EEG_fastr.event(1).bvtime + Trigs(1); % matlab doesnt read it correct + raw date

qrs_events = {};
for i=1:EEG_qrs.nbchan
    chanlab = EEG_qrs.chanlocs(i).labels;
    if length(strfind(chanlab,'ECG')) > 0
        qrs_event = ['qrs_' chanlab];
        qrs_events{end+1} = qrs_event;
        EEG_qrs=pop_fmrib_qrsdetect(EEG_qrs,i,qrs_event,'no');
    end;
end;

[EEG_qrs,~,~] = correct_qrs(EEG_qrs, qrs_events);

% remove drifts
EEG_hp = pop_eegfiltnew(EEG_qrs, .2, []);

EEG_downsample = pop_resample(EEG_hp, 250);
clear EEG EEG_fastr EEG_qrs EEG_hp
pop_writebva(EEG_downsample,gca_file,'MULTIPLEXED');

else
  disp('loading existing gca data')
  [EEG_downsample,com] = pop_loadbv(outpath,[filename(1:end-5) '_gca_mx.vhdr']);
  ecg_chans = get_ecg_chans(EEG_downsample);
  qrs_events = strcat('qrs_',{EEG_downsample.chanlocs(ecg_chans).labels});
end

[EEG_downsample,outliers,best_ecg_peaks] = correct_qrs(EEG_downsample, qrs_events);
clear EEG_downsample
[min_outliers,min_outliers_ind] = min([outliers.num_Outliers]);
fprintf(' %s has less outliers\n', qrs_events{min_outliers_ind});

gca_file

addpath('~/softs/FASST');

[EEG_fasst,Fdata] = crc_eeg_rdata_brpr(gca_file);
EEG_fasst.CRC.EKGPeaks = round(best_ecg_peaks);
save(EEG_fasst);
fullfile(EEG_fasst.path, EEG_fasst.fname)

global crc_def;
par_args = struct('bcgmethod','acica','ecgchan',32,'badchan',ecg_chans,'fqrsdet',0);
try;
    D_cica=crc_par(fullfile(EEG_fasst.path ,EEG_fasst.fname),par_args);
catch e;
% if it fails, try it without removing to much of "movements"
  old_scSNR = crc_def.par.bcgrem.scSNR;
  crc_def.par.bcgrem.scSNR = 10
  D_cica=crc_par(fullfile(EEG_fasst.path, EEG_fasst.fname),par_args);
  crc_def.par.bcgrem.scSNR = old_scSNR;
end;
rmpath('~/softs/FASST');

function Trigs = find_trig(EEG,etype)
Trigs=[EEG.event(strcmpi({EEG.event.code}, etype)).latency]

function ecg_chans = get_ecg_chans(EEG)
ecg_chans = [];
for i=1:EEG.nbchan
    chanlab = EEG.chanlocs(i).labels;
    if length(strfind(chanlab,'ECG')) > 0
        ecg_chans(end+1) = i;
    end;
end;


