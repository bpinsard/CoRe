function EEG=correct_qrs(EEG, qrs_events)

winsize=60
for qrs_id=1:length(qrs_events)
    qrs_event = qrs_events{qrs_id};
    qrs_times = [];
    for E=1:length(EEG.event)
        if strcmp(EEG.event(E).type,qrs_event)
            qrs_times(end+1) = EEG.event(E).latency;
           end
    end
    heartrate = diff(qrs_times);
    %remove duplicate qrs
    qrs_times = qrs_times(heartrate>0);
    heartrate = diff(qrs_times);
    slide_avg_heartrate = zeros(length(heartrate));
    for qrs_idx=1:length(heartrate)
        idx_start = max(1,qrs_idx-winsize/2);
        idx_stop = idx_start+winsize
        if idx_stop>length(heartrate)
            idx_stop=length(qrs_times);
            idx_start=max(idx_stop-winsize,1);
        end
        slide_avg_heartrate(qrs_idx) = mean(heartrate(idx_start: ...
                                                      idx_stop));
    end
    demean_heartrate = heartrate-slide_avg_heartrate;
    heartrate_std = std(demean_heartrate)
    
    %missing markers
    sum(heartrate > demean_heartrate+heartrate_std)
    sum(heartrate < demean_heartrate-heartrate_std)