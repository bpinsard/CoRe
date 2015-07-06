[EEG, com] = pop_loadbv('/scratch/bsl/raw/UNF/SleepEEG_MSL/EEG/raw/SleepEEG_MSL_15MG_01', 'SleepEEG_MSL_15MG_01.vhdr');

etype='R  1';
Trigs=[];
for E=1:length(EEG.event)
	if strcmp(EEG.event(E).type,etype)
		Trigs(end+1)=round(EEG.event(E).latency);
	end
end

% add dummy triggers
Trigs = [linspace(-3,-1,3)*10800+Trigs(0) Trigs]

EEG_fastr=fmrib_fastr(EEG,70,4,30,Trigs,0,0,0,0,0,0.03,[32,65,66,67,68],'auto');

EEG_qrs=pop_fmrib_qrsdetect(EEG_fastr,68,'qrs','no')

pop_saveset(EEG_fastr,'filename','SleepEEG_MSL_15MG_01_gac.set','filepath','./')
