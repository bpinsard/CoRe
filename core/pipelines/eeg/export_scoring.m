function EEG=export_scoring(path, filename)

[EEG,com] = pop_loadbv(path,filename);
[ECG,com] = pop_loadbv(path,[filename(1:end-9),'.vhdr'],[],[32]);

EEG.data(32,:)=ECG.data(1,:)
EEG_scoring = pop_select(EEG,'channel',{'Fp1','Fp2','T7','T8','Fz','Cz','Cpz','Pz','Poz','Oz','Fpz','M1','M2','ECG','AF7','AF8','FT9','FT10'})
EEG_scoring = pop_eegfiltnew(EEG_scoring,0,35);

pop_writebva(EEG_scoring,[path '/' filename(1:end-5) '_scoring.vhdr'],'MULTIPLEXED');

