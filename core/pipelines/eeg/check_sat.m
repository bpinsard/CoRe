function EEG=preproc_eeglab(path, filename)

[EEG,com] = pop_loadbv(path,filename);
