#!/bin/bash
DATA_DIR='/home/bpinsard/data/analysis/core_sleep/'
EXPORT_DIR='/home/bpinsard/data/analysis/core_mri_for_ella/'
for d in ${DATA_DIR}/_subject_id_* ;
do
    sid=${d##*_}
    echo $sid
    sdir=${EXPORT_DIR}`printf 'CoRe_%03d' $sid`
    mkdir -p $sdir
    echo $sdir
    pushd $sdir
    ln -s $d/convert_t1_dicom/*.nii.gz ./
    if [ -f $d/convert_t1_12ch_2mm_iso_dicom/*.nii.gz ] ; then
	ln -s $d/convert_t1_12ch_2mm_iso_dicom/*.nii.gz ./
    fi
    for day in `seq 3`; do
#	i=0
	mkdir -p D$day
	if [ `ls $d/fmri_convert/mapflow/*/*_D${day}_*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/fmri_convert/mapflow/*/*_D${day}_*.nii.gz ;do
# 		i=$(($i+1))
		ln -s $f `printf "D${day}/${f##*/}" $i`
	    done
	fi
	if [ `ls $d/convert_fieldmap_dicom/mapflow/*/*_D${day}_*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/convert_fieldmap_dicom/mapflow/*/*_D${day}_*.nii.gz ;do
		ln -s $f `printf "D${day}/${f##*/}" $i`
	    done
	fi
    done
    popd
done
