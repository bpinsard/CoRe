#!/bin/bash
DATA_DIR='/home/bpinsard/data/analysis/core_sleep/'
#EXPORT_DIR='/home/bpinsard/data/analysis/core_mri_for_ella/'
EXPORT_DIR='stark.criugm.qc.ca:/home/bore/Public/consciousness/'

for d in ${DATA_DIR}/_subject_id_* ;
do
    sid=${d##*_}
    echo $sid
    sdir=${EXPORT_DIR}`printf 'CoRe_%03d' $sid`
    mkdir -p "$sdir"
    echo "$sdir"
    pushd "$sdir"
    for day in `seq 3`; do
#	i=0
	mkdir -p D$day
	
	if [ `ls $d/apply_registration/mapflow/*/*_D${day}_*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/apply_registration/mapflow/*/*_D${day}_*.nii.gz ;do
		scp -R $f $EXPORT_DIR
	    done
	elif [ `ls $d/apply_registration/mapflow/*/fmri_201*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/apply_registration/mapflow/*/fmri_201*.nii.gz ;do
		scp -R $f $EXPORT_DIR
	    done    
	fi

	if [ `ls $d/mcflirt/mapflow/*/*_D${day}_*.par | wc -l` -gt 0 ] ; then
	    for f in $d/mcflirt/mapflow/*/*_D${day}_*.par ;do
		scp -R $f $EXPORT_DIR
	    done
	elif [ `ls $d/mcflirt/mapflow/*/fmri_201*.par | wc -l` -gt 0 ] ; then
	    for f in $d/mcflirt/mapflow/*/fmri_201*.par ;do
		f2=${f##*/}
		scp -R $f $EXPORT_DIR
	    done
	fi

    done
    popd
done
