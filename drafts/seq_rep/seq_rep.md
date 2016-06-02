---
title: Mapping of Motor Sequence Representations in the Human Brain across learning and consolidation
tags: [example, citation]
bibliography: [abstract_rbiq.bib]
author:
- family: Pinsard
  given: Basile
  affiliation: 1,2
  email: basile.pinsard@gmail.com
- family: Gabitov
  given: Ella
  affiliation: 1
- family: Boutin
  given: Arnaud
  affiliation: 1
- family: Benali
  given: Habib
  affiliation: 1,2
- family: Doyon
  given: Julien
  affiliation: 1
institute: here
organization:
- id: 1
  name: Functional Neuroimaging Unit, Centre de Recherche de l'Institut Universitaire de Gériatrie de Montréal
  address: Montreal, Quebec, Canada
  url: http://unf-montreal.ca
- id: 2
  name: Sorbonne Universités, UPMC Univ Paris 06, CNRS, INSERM, Laboratoire d’Imagerie Biomédicale (LIB) 
  address: 75013, Paris, France
  url: http://lib.upmc.fr
date: 
---

## Introduction

Sequences of motor movements frame the complex evolving action repertoire that animals and humans possesses.
Motor sequence learning (MSL), as acquisition of temporally ordered succession of coordinated movements, is thus an adequate paradigm to study neural plasticity [@doyon_reorganization_2005] occurring from initial fast learning to automation of a skill.

While training sessions induce critical plasticity changes [@ungerleider_imaging_2002;@doyon_reorganization_2005], evidences of further offline reprocessing of acquired procedural memory have accumulated [@walker_sleep_2003;@press_time_2005;@hotermans_early_2006;@korman_daytime_2007;@morin_motor_2008;@debas_brain_2010;@barakat_sleep_2013;@cousins_cued_2014;@debas_off-line_2014;@gregory_resting_2014;@cellini_temporal_2015;@ramanathan_sleep-dependent_2015;@laventure_nrem2_2016] despite controversies regarding wake and sleep stages respective roles and interplay [@nettersheim_role_2015;@landry_effects_2016].
Explicitly acquired sequential skills specifically benefit from off-line reprocessing [@doyon_contributions_2009], observed by either gains or maintenance of performance compared to a decay if no offline period is allowed to the subject [@nettersheim_role_2015].

Models of consolidation hypothesize an underlying "transfer" or rather a "re-balancing" between cerebral structures [@rasch_reactivation_2008;@born_system_2012;@albouy_hippocampus_2013;@dudai_consolidation_2015] encoding memory.
Herein, sequential motor skills recruits an extended network of cerebral [@hardwick_quantitative_2013], cerebellar and spinal regions [@vahdat_simultaneous_2015], which involvement evolves differently across learning stages [@dayan_neuroplasticity_2011].

Primary sensorimotor and supplementary motor cortices as well as posterior parietal and dorso-lateral prefrontal cortices [@hardwick_quantitative_2013] are the major areas activated during motor sequence tasks.
These cortical areas are parts of both cortico-cerebellar and cortico-striatal loop circuits which differently contributes to motor sequence learning [@doyon_reorganization_2005].

Cerebellar cortex anatomical functions [@diedrichsen_cerebellar_2014] includes integration of sensory and motor ipsilateral finger representation in Lobule V [@wiestler_integration_2011], complex movements, including coordination, sequencing and spatial processing, recruiting bilateral Lobule VI [@schlerf_evidence_2010] and Crus I in Lobule VII [@doyon_contributions_2009], being respectively connected to primary and premotor and supplementary cerebral cortices [@buckner_organization_2011].
Lobule VIII also shows a somatotopicaly organized representation of the whole body, but weaker relatively to Lobule V/VI.
It's function includes model-based prediction and sensory feedback processing [@tseng_sensory_2007], thus contributing to error-based learning which supports early MSL observable through an activation shift from cerebellar cortex to dentate nuclei [@doyon_experience-dependent_2002] with lowered activation in extended practice, paralleling striatal-cortical network increased recruitment.

Striatum have been associated in selection, preparation and execution of free or instructed movement [@hikosaka_central_2002;@gerardin_distinct_2004;@doyon_contributions_2009].
The striatum participates in automation of sequences processing through reinforcement learning [@jin_basal_2014], but is not limited solely to procedural motor skills [@graybiel_striatum_2015].
Automation is progressively attained through local optimization of transition [@rosenbaum_hierarchical_1983;@wymbs_differential_2012;@diedrichsen_motor_2015], which variations explored during learning process then stabilizes [@albouy_neural_2012] and is influenced by learning strategy [@lungu_striatal_2014].
Progressive shift from associative and premotor to sensorimotor basal ganglia distributed network concurring to automation of the skill has been observed [@lehericy_distinct_2005].

Hippocampus is a major structure for memory [@battaglia_hippocampus:_2011] that not only encodes episodic but also supports procedural memory acquisition in it's early stages [@albouy_both_2008] it's activity and interaction with striatum forecasting sequences long-term retention [@albouy_interaction_2013].
This hippocampal representation could concurrently represent episodic memory of task execution, an allocentric representation[@albouy_maintaining_2015], tagging and buffering of memories to be further consolidated, then fading when skill is consolidated.

The abovementioned structures undergo non-linear changes of activation level along the course of learning [@dayan_neuroplasticity_2011], notably after consolidation [@lehericy_distinct_2005;@debas_brain_2010;@debas_off-line_2014], reflecting both increased implication of specialized circuits and decreased non-specific support networks activation.

While these large-scale activation changes excerpt the dynamic of network recruitment, if does not provide evidence that these acquires  motor sequence specific representation.
This can however be identified with Multivariate Pattern Analysis (MVPA), a set of techniques recently adapted to neuroimaging [@pereira_machine_2009] which evaluates how local patterns of activity are able to discriminate between different stimuli or memory items.


In a recent study focusing on upper cerebral cortex [@wiestler_skill_2013], classifier performance was evaluated on sets of sequences, either trained or not, showing an potential increase of the representation strength with training in a network bilaterally spanning primary, pre and supplementary motor to parietal cortices.

As preparation activates a network overlapping execution related one [@lee_subregions_1999;@zang_functional_2003;@orban_richness_2015], another study [@nambu_decoding_2015] specifically analyzed patterns of activity after instruction and prior to execution of extensively trained sequences.
Patterns related to motor planning without contamination of motor execution or sensory feedback related activity were detected in a restricted network spanning only contralateral dorsal premotor and supplementary motor area, maybe related to use of dominant hand instead, and not in the basal ganglia.

Using high-resolution imaging and averaging multiple trials, another study [@bednark_basal_2015] found activation differences in striatum between rhythmic and ordering motor sequences executions, whereas MVPA failed to find discriminating patterns. Both the fact that sequences were not trained and the use of cross-conditions classification of averaged trials could explains these negative findings.


Despite numerous neuroimaging [@hikosaka_central_2002;@albouy_both_2008;@orban_multifaceted_2010;@penhune_parallel_2012;@albouy_neural_2012] and neuronal level studies [@alexander_parallel_1986;@graybiel_building_1995;@miyachi_differential_2002] corroborating the implication of basal ganglia, notably the striatum, in acquisition of motor sequence, no evidence of representation have yet been found in human using fMRI MVPA, which could be partly accounted by the limited measurable signal level in these regions.
The studies listed above did not reported hippocampus representation neither, despites it's involvment in early sequential skills acquisition [@albouy_both_2008;@albouy_hippocampus_2013].


The goal of our study is to provide further insight in brain representation of motor sequence, combining MVPA and robust statistics to assess decoding significance [@stelzer_statistical_2013;@allefeld_valid_2015].

Using these techniques we intend to map subnetwork showing sequence specific patterns of activity replicating and extending previous research [@wiestler_skill_2013;@kornysheva_human_2014] by scanning larger extent of human brain.
This larger coverage will enable probing activity patterns in sub-cortical structures contributing to MSL: cerebellum, basal ganglia including striatum and lower temporal cortex including hippocampus.

The dynamic of sequential movement planning to execution will also be decomposed to analyze instruction and execution phases separately as was previously observed [@nambu_decoding_2015].

The current study also intend to measure changes induced by learning and sleep-dependent consolidation by comparing MVPA classifying performance between sequences at different stages of these processes.

##Method

### Participants
 
The study includes ## right-handed young (18-35 years old) healthy volunteers recruited by advertising on scholar and public website.

The subject were not included in case of history of neurological psychological or psychiatric disorders or scoring 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961].

Subjects with BMI greater than 27, smokers, extreme chronotype, night-workers, having travelled across meridian during the 3 previous months, or training as musician or professional typist (for over-training on coordinated finger movements) were also excluded.

Sleep quality was assessed by Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989], and daytime sleepiness  (Epworth Sleepiness Scale [@johns_new_1991]) had to be lower or equal to 9.

Subject were also instructed to abstain from caffeine, alcohol and nicotine, and have regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the study.
Instruction compliance was controlled by non-dominant hand wrist actigraphy (Actiwatch 2, Philips Respironics, Andover, MA, USA) for the week preceding and the duration of the experiment.

### Behavioral experiment

The experiment was conducted over 3 consecutive days, at the end of the day, with all motor task performed in the scanner using an ergonomic MRI-compatible 4-keys response pad.

#### First evening (D1)

Subjects were trained to perform with left-hand a 5 elements sequence (TSeq) for 14 blocks of 12 sequences or a maximum of 12x5=60 keypresses.
Subject were instructed to execute repeatedly as fast and as accurate as possible the sequence of keypresses and to start from the beginning of the sequences in case they noticed that they did an error.

They were retested approximately 20 minutes later for an additional single block of 12 sequences.

#### Second evening (D2)

Subjects were first tested for 1 block on TSeq, then half of the subjects of "Interference" (Int) group were trained on an interfering sequences (IntSeq) of 5 elements with left-hand for 14 blocks of 12 sequences as for TSeq.
"No-Interference" (NoInt) group had scanned resting-state which duration was yoked to a IntGroup subject.

#### Third evening (D3)

Subjects first performed TSeq for 7 blocks of 12 repetitions for retest, then practised IntSeq for 7 blocks of 12 sequences. 

This was followed by a task specifically designed for MVPA analysis, that will be called "MVPA task" thereafter.
This task was similar to [@wiestler_skill_2013], rapidly alternating short blocks of practice of 4 different sequences.

Significant differences are that 4 sequences were performed with the left hand four fingers excluding the thumb and no feedback was given regarding the correctness of performance. Also sequences were repeated uninterruptedly as in training, in order to probe the processes underlying automation of the skill.

Each block, composed of an instruction period of 4 sec when was displayed 5 numbers (eg. 1-4-2-3-1) representing in reading order the sequence of fingers to be pressed, followed by an execution period indicated by a green cross.
Subject had to perform 5 times the sequence, or a maximum of 5x5=25 key-presses before being instructed to stop and rest by displaying a red cross.

Rest duration was variable and set to synchronize the beginning of each block with the same slice in the volume being acquired to allow study of dynamics as further described.

Ordering of the sequences in blocks was chosen to include all possible successive pairs of the sequences using De Bruijn cycles [@aguirre_bruijn_2011] allowing unbiased analysis of dynamics described below. Given 4 sequences, a 2-length De Bruijn cycle would contains 16 blocks, repeated twice to give 8 repetitions of each of the 4 sequences which amounts to 32 blocks.

Each subject performed the task twice in scans separated by few minutes to allow rest and enable study of cross-scans pattern stability using classification generalization.

### Scan acquisition

MRI data were acquired on a Siemens Trio 3T scanner on 2 two separate sessions.
The first session used 32-channel coil to acquire high-resolution anatomical T1 weighted image using Multi-Echo (4) MPRAGE (MEMPRAGE) at 1mm isotropic resolution. 

Functional data were acquired during the second session with 12-channel coil for comparison with other dataset. EPI sequence consists of 40 slices with 3.44mm in-plane resolution and 3.3mm slice thickness to provide full cortical and cerebellum coverage with a TR of 2.16 sec.
Consecutively fieldmap was obtained to measure B0 field inhomogeneity to allow retrospective compensation of induced distortions.

### Preprocessing

Custom pipeline was used to preprocess fMRI data prior to analysis.
First, high-resolution anatomical T1 weighted image was preprocessed with Freesurfer [@dale_cortical_1999;@fischl_high-resolution_1999;@fischl_cortical_2008] to segment subcortical regions, reconstruct cortical surfaces and provide inter-subjects alignment of cortical folding patterns. 
Pial and grey/white matter interface surfaces were downsampled to match the 32k sampling of Human Connectome Project (HCP) [@glasser_minimal_2013] and we averaged pial and white surface to get coordinates at the half of the thickness of cortical sheet.
HCP subcortical rois coordinates were warped onto individual T1 data using non-linear registration based on Ants software [@avants_symmetric_2008;@klein_evaluation_2009]. Combination of cortical and subcortical coordinates then corresponds to grayordinates of HCP datasets [@glasser_minimal_2013].

fMRI data underwent estimation of subject motion [@roche_four-dimensional_2011] and coregistration to T1.
Registration and motion parameters were used to interpolate Blood-Oxygen-Level-Dependent (BOLD) signal at anatomical grayordinates above-mentioned taking into account B0 inhomogeneity induced distortions using fieldmap acquisition.

BOLD signal was further processed to remove drifts and motion-related abrupt signal change.

Of note is that our preprocessing does not includes smoothing, even though interpolation inherent to any motion correction causes averaging of values of neighboring voxels. We intended to minimize blurring of data to preserve fine-grained patterns of activity, resolution of relevant patterns being hypothetically at columnar scale.

### Multivariate Pattern Analysis

Similarly to [@wiestler_skill_2013;@nambu_decoding_2015] we aim to uncover activity patterns predicting the sequence prepared or executed.
We also aim to analyze the classification of trained sequences versus untrained one, however TSeq and IntSeq are probably still undergoing consolidation by the third day of the study, potentially lowering their pattern stability.

For this reason, we first aimed at classifying untrained sequences which are comparable as being completely new to the subjects, and are learned along the course of the "MVPA task". This 2-class scheme allows mapping representation of non-consolidated sequence.

Then TSeq and IntSeq are also analyzed separately in IntGroup mapping sequences consolidated on two separate nights of sleep.

Moreover, the instruction stimuli presented before each execution, enable disambiguation of the memory traces from explicit recall from that during execution of the motor plan.

The MVPA analysis was based on PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom cross-validation scheme, Searchlight and measures to adapt to the study design and analyses.

#### Samples

Each blocks was modeled by having 2 boxcars, respectively instruction and execution phase, convolved with Hemodynamic Response Functions (HRF). Volumes (TRs) corresponding to HRF level above 50% of maximum response level were taken as samples for the performed sequence. Maximum value of instruction and execution regressors determine the TR to pertain to instruction or execution phase, for which HRF is overlapping.
A TR based approach was chosen to explore the fine dynamic of patterns related to the task, that model driven such as GLM cannot fully analyze at the cost of lower signal-to-noise ratio.

Regular GLM-based approach was also performed using least-square separate (LS-S) regression of each event [@mumford_deconvolving_2012] shown to provide improved activation patterns estimates for MVPA. For each blocks, regressors for instruction and execution phases provided t-value maps that was further used as MVPA samples.

#### Cross-validation

The De Bruijn cycles ordering of the sequence in the task aims at providing unbiased cross-validation by balancing the temporal succession of any pair of the 4 sequences. 

Chosen cross-validation schema includes:

- Leave-One-Chunk-Out (LOCO): each block is successively taken out of the dataset to be used in prediction. Classifier is trained on remaining data by randomly selecting balanced number of samples of the 4 sequences which are further than 60 seconds to the block of test data. Random selection of balanced data is performed 5 times for each of the 64 blocks amounting to 64*5 = 320 folds of cross-validation. When applied for each scans separately, each contained 32 blocks generating 160 fold cross-validation.
- Leave-One-Scan-Out (LOSO): random balanced subset of samples from a scan is fed for training to the classifier which then predicts one the other scan the sequences. A random balanced subsets was selected 5 times for the 2 scans giving 10 cross-validation folds.

#### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that apply MVPA techniques repeatedly on small spatial neighborhood with the purpose to localize representation of information of interest across brain while avoiding high-dimensional limitation of multivariate algorithms.

Applying cross-validation using the Searchlight schema allowed to extract brain-wise map of classifier performance giving information of regions having stable sequences related patterns.
Gaussian Naive Bayes (GNB) linear classifier, optimized for Searchlight, was performed with the 2 proposed cross-validation schema analysis on the execution labelled TRs.
Also GNB-based Searchlight have been argued to allow smoothness of generated maps [@raizada_smoothness_2013] despite unsmoothed data, allowing more-reliable cross-subject study and thus higher cross-scans generalization.

Searchlight was configured to select for each grayordinate the 64 closest neighboring coordinates, using surface distance for cortical grayordinates, as the subset of features.
Searchlight size has been shown to inflate the extent of significant clusters in searchlight analysis [@viswanathan_geometric_2012;@etzel_searchlight_2013] which motivated the small neighborhood for our analysis.

For both cross-validation schema, confusion matrix was computed for each block of practice providing a more complete representation of classification performance and biases from which can be derived specific or global accuracy percentage.

#### Searchlight Group Analysis

In order to obtain the cluster with significant prediction at the group level, we used non-parametric permutation based statistical approach as described in [@stelzer_statistical_2013]. Subject-level searchlight accuracy map were computed 100 times after permuting the labels, giving a null distribution of classification accuracy per spatial coordinate.
A non-parametric null distribution of mean group accuracy map was obtained by repeating 10000 averages of randomly picked subject-level permutated maps to create a featurewise threshold map at p<0.001.
Above threshold group permutation maps clusters were used to estimate a null distribution of cluster size enabling computation of clusters significance in the non-permutated thresholded map.

This permutation procedure was conducted for each type of searchlight.


#### Dynamics analysis

Also we complemented the analysis by taking TRs subsamples with similar delay from instruction time, ranging from -2 to 20 TRs, and then to discriminate the sequences in cross-validation schema. Having 64 blocks in total across the 2 scans give 64 samples for each TR delay. This is similar to [@wiestler_skill_2013 fig.4,D-E] ROI based temporal analysis. 

Such analysis aims at uncovering the dynamics of motor sequence execution, as instruction phase might causes motor planning and simulation of sequence performance, while execution phase generally includes warm-up and then automation of motor chunks execution.
Furthermore, this method is independent of HRF model allowing potential non-hemodynamic related neurally-driven BOLD signal changes to be taken into account.

While the rapid block design might hampers temporal disambiguation, De Bruijn cycle ordering imposes balanced successive blocks pairs in the dataset. Thus temporal leaking of BOLD activity of the previous sequence production is then balanced across the 4 sequences and should yield chance level prediction if no signal related to the present sequence is observable in the data, allowing unbiased analysis relative to the chance level.

#### Region of Interest (ROI) analyzes

A network of local neuronal populations has been shown to contribute to sequence production learning [@dayan_neuroplasticity_2011], with their activity [@albouy_hippocampus_2013;@barakat_sleep_2013] evolving in accordance theoretical models of consolidation [@born_system_2012]. To extract whether their activity independently encodes the spatio-temporal pattern of the sequence, we conducted ROI cross-validation using a priori atlas including:

- hippocampus, cerebellum and striatum including caudate nucleus and putamen
- cortical network including posterior parietal, primary motor, premotor, supplementary motor area and dorso-lateral prefrontal.

To assess the significance of each of these cross-validation accuracy measures, non-parametric permutation test [@stelzer_statistical_2013] of xxx repetitions was conducted for each subject and ROI to estimate chance level distribution of accuracy.

## Results

### Searchlight

![All subjects t-values threshold (p<.01) of untrained sequences classification accuracy during execution](../../../../analysis/core_mvpa/searchlight_group/CoRe_group_All_mvpa_all_mvpa_new_seqs_exec_loco_t0-5p0.010.png){#fig:all_subj_untrained_exec .slmap}

Non-consolidated additional sequences were analyzed separately to provide a localizer of early sequence representation (fig. @fig:all_subj_untrained_exec). A limited network emerges from both instruction and execution phase patterns, left hippocampus and primary motor and posterior parietal cortex, bilateral cerebellum, anterior putamen and insular cortex. This weaker representation of new sequence is in agreement with [@wiestler_skill_2013] but could be sourced in higher variability in motor production for a part.

![Contrast conjunction (scan2-scan1,scan2>chance) for untrained sequences classifier accuracy during execution t-values (p<0.05)](../../../../analysis/core_mvpa/searchlight_group/CoRe_ctx_mvpa_new_seqs_exec_mvpa2-mvpa1_t0-5p0.050.png){#fig:intgroup_trained_min_untrained_exec .slmap}

![Int Group t-values threshold (p<.01) of trained sequences classification accuracy during execution](../../../../analysis/core_mvpa/searchlight_group/CoRe_group_Int_mvpa_all_tseq_intseq_exec_loco_t0-5p0.010.png){#fig:intgroup_trained_exec .slmap} 

Consolidated sequences in Int group has, despite lower number of subjects, an extended representation spanning over posterior parietal and premotor cortex, cerebellum, caudate and putamen bilaterally (fig @fig:intgroup_trained_exec).
When contrasted with the untrained sequences (fig @fig:intgroup_trained_min_untrained_exec), only bilateral anterior putamen clusters contains significantly higher discrimination patterns relative to untrained sequences.

![Contrast conjuction (trained-untrained,trained>chance) for Int accuracy during execution (p<0.05)](../../../../analysis/core_mvpa/searchlight_group/CoRe_ctx_intgroup_tseqintseq-mvpa_new_seqs_exec_loco_t0-5p0.050.png){#fig:intgroup_trained_min_untrained_exec .slmap}

### ROIS

## Discussion

Using Searchlight even in a very controlled design gave individual maps with variability despite important common network found across subjects as highlighted by group analysis (fig). This raise an interest regarding potential causes of variable extended representation of sequences in each subject. As the task requires switching between different sequences, different strategies could be used by the subject to maintain in working memory the sequence first shown on the screen then practised physically.

Our group analysis revealed implication of a cortical network specific to sequences features, similar to [@wiestler_skill_2013], as well as subcortical contributions to this representation.

TODO:

- representation of sequences characteristics and ROIs: [@kornysheva_human_2014] 
- address all limitation and confounds [@todd_confounds_2013; @etzel_searchlight_2013; @etzel_looking_2012] of MVPA and Searchlight

comparison with [@wiestler_skill_2013]:

- worst acquisition (12-channel, resolution...) but shorter TR.
- sequences not trained using same rapid-block design
- sequences not as intensively trained
- sequences at different stages of training/consolidation
- continuous execution of sequences
- using TRs (and GLM)

## References
