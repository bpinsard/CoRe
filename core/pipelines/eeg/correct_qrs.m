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
    %figure();
    %plot(EEG.data(qrs_id,1:20:end)+4000);
    %hold on;
    %plot(qrs_times(1,1:end-1)/20,slide_median_heartrate,'+-');
    demean_heartrate = heartrate-slide_median_heartrate;

    %plot(qrs_times(1,1:end-1)/20,heartrate,'rx-');
    
    missing_qrs = heartrate./slide_median_heartrate>1.5;
    false_pos = heartrate./slide_median_heartrate<.75;
    %plot(qrs_times(missing_qrs)/20,heartrate(missing_qrs),'*g');
    %plot(qrs_times(false_pos)/20,heartrate(false_pos),'*g');
    
    EEG.event(to_remove)=[];
    EEG.urevent(to_remove)=[];
end
