function EEG=correct_qrs(EEG, qrs_events)

winsize = 20;
for qrs_id=1:length(qrs_events)
    qrs_event = qrs_events{qrs_id};
    qrs_idx = [];
    for E=1:length(EEG.event)
        if strcmp(EEG.event(E).type,qrs_event)
            qrs_idx(end+1) = E;
           end
    end
    qrs_times = [EEG.event(qrs_idx).latency];
    heartrate = diff(qrs_times);
    %remove duplicate qrs
    qrs_times = qrs_times(heartrate>0);
    to_remove = qrs_idx(heartrate<=0);
    heartrate = diff(qrs_times);
    length(qrs_times);
    slide_median_heartrate = zeros(1,length(heartrate));
    for qrs_i=1:length(heartrate)
        idx_start = max(1,qrs_i-winsize/2);
        idx_stop = idx_start+winsize-1;
        if idx_stop>length(heartrate)
            idx_stop=length(heartrate);
            idx_start=max(idx_stop-winsize,1);
        end
        slide_median_heartrate(1,qrs_i) = median(heartrate(1,idx_start:idx_stop));
    end
    figure();

    
    for i=1:length(EEG.chanlocs)
	    if strcmp(EEG.chanlocs(i).labels,qrs_event(5:end))
        ecg_chan = i
      end
    end
      
    plot(EEG.data(ecg_chan,:)-mean(EEG.data(ecg_chan,:)));
    hold on;

    plot(qrs_times(1,1:end-1),slide_median_heartrate,'+-g');
    demean_heartrate = heartrate-slide_median_heartrate;

    plot(qrs_times(1,1:end-1),heartrate,'rx-');
    
    missing_qrs = heartrate./slide_median_heartrate>1.5;
    false_pos = heartrate./slide_median_heartrate<.75;
    plot(qrs_times(missing_qrs),heartrate(missing_qrs),'*y');
    plot(qrs_times(false_pos),heartrate(false_pos),'*y');
    
    EEG.event(to_remove)=[];
    EEG.urevent(to_remove)=[];
end
