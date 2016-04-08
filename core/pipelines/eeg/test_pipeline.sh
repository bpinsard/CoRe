#!/bin/bash

rm -Rf /scratch/bsl/test/test_eeg_pipeline/*
matlab -nodisplay -nosplash -nodesktop -r "dbstop if error; addpath ~;path(pathdef);addpath('~/scratch/projects/CoRe/core/pipelines/eeg/');preproc_eeglab_fasst('/home/bpinsard/data/raw/UNF/CoRe/EEG/CoRe_087_D1/','CoRe_087_Day1_TestBoostTSeq_01.vhdr',2.16,'/scratch/bsl/test/test_eeg_pipeline');"
