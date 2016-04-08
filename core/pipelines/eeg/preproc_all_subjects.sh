
tr=2.16
for d in $1/SleepEEG_MSL_*_0[12] ; do
#for d in $1/CoRe_???_D[12] ; do
    echo $d
    sd=${d##*/}
    mkdir -p $sd
    pushd $sd
    pids=''
    for f in `ls $d/*.vhdr`; do
	fname=${f##*/}
	fbase=${fname%.vhdr}
	of=${fbase}_gca_cica_mx
	echo $f
	if [ ! -f ./${of}.log ] ; then
	    echo "${of}.log does not exists : running preprocessing"
	    (
	    matlab -nodisplay -nosplash -nodesktop -r "try;addpath ~;path(pathdef);addpath('~/scratch/projects/CoRe/core/pipelines/eeg/');preproc_eeglab_fasst('$d','$fname',$tr,'.');catch e ;disp(e); disp(getReport(e)); end;exit;" > ${of}.log 2>&1 
	    if [ -f cICA_${fbase}_gca_mx.dat ] ; then
		sed "s/${fbase}_gca_mx.dat/cICA_${fbase}_gca_mx.dat/g" ${fbase}_gca_mx.vhdr | \
		    sed "s/${fbase}_gca_mx.vmrk/${fbase}_gca_cica_mx.vmrk/g" > ${fbase}_gca_cica_mx.vhdr
		sed "s/${fbase}_gca_mx.dat/cICA_${fbase}_gca_mx.dat/g" ${fbase}_gca_mx.vmrk > ${fbase}_gca_cica_mx.vmrk
	    fi
	    )&
	    pids=$pids" "$!
	else
	    echo "${of}.log exists, to rerun delete this file"
	fi
    done
    wait $pids
    popd
done
