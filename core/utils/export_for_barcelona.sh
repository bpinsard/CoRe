#!/bin/bash
DATA_DIR='/home/bpinsard/data/analysis/core_sleep/'
#EXPORT_DIR='/home/bpinsard/data/analysis/core_mri_for_ella/'
EXPORT_DIR='stark.criugm.qc.ca:/home/bore/Public/projects/consciousness/'

for d in ${DATA_DIR}/_subject_id* ;
do
    sid=${d##*_}
    echo $sid
    sdir=${EXPORT_DIR}`printf 'CoRe_%03d' $sid`/rfMRI/
    for day in `seq 3`; do
#	i=0
	mkdir -p D$day
	
	if [ `ls $d/apply_registration/mapflow/*/*_D${day}_*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/apply_registration/mapflow/*/*_D${day}_*Sleep*.nii.gz ;do
		scp $f $sdir
	    done
	elif [ `ls $d/apply_registration/mapflow/*/fmri_201*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/apply_registration/mapflow/*/fmri_201*Sleep*.nii.gz ;do
		scp $f $sdir
	    done    
	fi

	if [ `ls $d/mcflirt/mapflow/*/*_D${day}_*.par | wc -l` -gt 0 ] ; then
	    for f in $d/mcflirt/mapflow/*/*_D${day}_*Sleep*.par ;do
		scp $f $sdir
	    done
	elif [ `ls $d/mcflirt/mapflow/*/fmri_201*.par | wc -l` -gt 0 ] ; then
	    for f in $d/mcflirt/mapflow/*/fmri_201*Sleep*.par ;do
		f2=${f##*/}
		scp $f $sdir
	    done
	fi

    done
done
