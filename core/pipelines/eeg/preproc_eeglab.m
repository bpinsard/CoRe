[EEG,com] = pop_loadbv('/scratch/bsl/EEG/', 'SleepCoRe_MVPA1_01.vhdr');

etype='R  1';
Trigs=[];
for E=1:length(EEG.event)
	if strcmp(EEG.event(E).type,etype)
		Trigs(end+1)=round(EEG.event(E).latency);
	end
end

% add dummy triggers
Trigs = [linspace(-3,-1,3)*10800+Trigs(1) Trigs];

EEG_fastr=fmrib_fastr(EEG,70,4,30,Trigs,0,0,0,0,0,0.03,[32,65,66,67,68],'auto');
EEG_qrs = EEG_fastr;
pop_saveset(EEG_qrs,'filename','SleepCoRe_MVPA1_01_gac.set','filepath','./')

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


pop_saveset(EEG_qrs,'filename','SleepCoRe_MVPA1_01_gac.set','filepath','./')



for E=1:length(EEG_qrs.event)
	if strcmp(EEG_qrs.event(E).type,'qrs_ECG4')
		qrs_ecg4(end+1)=round(EEG_qrs.event(E).latency);
	end
end
