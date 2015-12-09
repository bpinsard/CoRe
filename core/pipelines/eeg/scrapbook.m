
[EEG_gca,com]=pop_loadbv('CoRe_011_D1','CoRe_011_Day1_Night_01_gca.vhdr') 
pop_eegplot(EEG_gca,[],[],[],[],'eloc_file',EEG_gca.chanlocs,'winlength',30,'spacing',200,'dispchans',5,'data2',EEG_gca_pas.data)
