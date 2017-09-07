#!/bin/bash
DATA_DIR='/home/bpinsard/data/analysis/core_sleep/'
#EXPORT_DIR='/home/bpinsard/data/analysis/core_mri_for_ella/'
EXPORT_DIR='/media/bpinsard/Seagate Expansion Drive/CoRe/02_CoRe_EEG_MRI/MRI_moco_topup_nii/'
for d in ${DATA_DIR}/_subject_id_* ;
do
    sid=${d##*_}
    echo $sid
    sdir=${EXPORT_DIR}`printf 'CoRe_%03d' $sid`
    mkdir -p "$sdir"
    echo "$sdir"
    pushd "$sdir"
    f=`ls $d/convert_t1_dicom/*.nii.gz`
    f2=${f%.gz}
    gunzip -d -c $f > ./${f2##*/}
#    ln -s $d/convert_t1_dicom/*.nii.gz ./
    if [ -f $d/convert_t1_12ch_2mm_iso_dicom/*.nii.gz ] ; then
	f=`ls $d/convert_t1_12ch_2mm_iso_dicom/*.nii.gz`
	f2=${f%.gz}
	gunzip -d -c $f > ./${f2##*/}
    fi
    for day in `seq 3`; do
#	i=0
	mkdir -p D$day
	
	if [ `ls $d/apply_registration/mapflow/*/*_D${day}_*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/apply_registration/mapflow/*/*_D${day}_*.nii.gz ;do
		f2=${f%.gz}
		gunzip -d -c $f > `printf "D${day}/${f2##*/}" $i`
	    done
	elif [ `ls $d/apply_registration/mapflow/*/fmri_201*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/apply_registration/mapflow/*/fmri_201*.nii.gz ;do
		f2=${f%.gz}
		gunzip -d -c $f > `printf "D${day}/${f2##*/}" $i`
	    done    
	fi

	if [ `ls $d/mcflirt/mapflow/*/*_D${day}_*.par | wc -l` -gt 0 ] ; then
	    for f in $d/mcflirt/mapflow/*/*_D${day}_*.par ;do
		f2=${f##*/}
		cp $f `printf "D${day}/${f2%%.*}_motion.txt" $i`
	    done
	elif [ `ls $d/mcflirt/mapflow/*/fmri_201*.par | wc -l` -gt 0 ] ; then
	    for f in $d/mcflirt/mapflow/*/fmri_201*.par ;do
		f2=${f##*/}
		cp $f `printf "D${day}/${f2%%.*}_motion.txt" $i`
	    done
	fi

	if [ `ls $d/convert_fieldmap_dicom/mapflow/*/*_D${day}_*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/convert_fieldmap_dicom/mapflow/*/*_D${day}_*.nii.gz ;do
		f2=${f%.gz}
		gunzip -d -c $f > `printf "D${day}/${f2##*/}" $i`
		#ln -s $f `printf "D${day}/${f##*/}" $i`
	    done
	elif [ `ls $d/convert_fieldmap_dicom/mapflow/*/fieldmap_bold_201*.nii.gz | wc -l` -gt 0 ] ; then
	    for f in $d/convert_fieldmap_dicom/mapflow/*/fieldmap_bold_201*.nii.gz ;do
		f2=${f%.gz}
		gunzip -d -c $f > `printf "D${day}/${f2##*/}" $i`
		#ln -s $f `printf "D${day}/${f##*/}" $i`
	    done

	fi
    done
    popd
done
