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

Sequences of motor movements constructs the complex action repertoire of animal and human beings, acquired and evolving with their practice.
Motor sequence learning (MSL) is thus an adequate paradigm to study neural plasticity [@doyon_reorganization_2005] involved from initial fast learning to automation of a skill such as a temporally ordered succession of coordinated movements.

Sequential motor skills recruits an extended network of cerebral [@hardwick_quantitative_2013] and spinal regions [@vahdat_simultaneous_2015], which involvement evolves differently across learning stages [@dayan_neuroplasticity_2011].
Many areas undergo non-linear changes in activity reflecting both increase implication of specialized circuits and decrease of non-specific support networks, thus lowering cognitive demand.

At the cerebral level, cortical networks are critical to sequence execution for which large-scale network is activated including primary sensorimotor and supplementary motor cortices as well as posterior parietal and dorso-lateral prefrontal cortices [@hardwick_quantitative_2013].

Subcortical regions, hippocampus and striatum [@albouy_hippocampus_2013;@doyon_contributions_2009;@hikosaka_central_2002] were also identified as important players in the learning and consolidation of motor sequences.
Hippocampus is a major structure for memory [@battaglia_hippocampus:_2011] that not only encodes episodic but also supports procedural memory acquisition in it's early stages [@albouy_both_2008] it's activity forecasting sequences long-term retention [@albouy_interaction_2013].

Among basal ganglia, striatal regions have been associated in selection, preparation and execution of free or instructed movement [@gerardin_distinct_2004].
The striatum also participates in automation of sequences processing through reinforcement learning [@jin_basal_2014], but is not limited solely to procedural motor skills [@graybiel_striatum_2015].
Automation is progressively attained through local optimization of transition achieved through chunking of sub-sequences [@rosenbaum_hierarchical_1983;@wymbs_differential_2012;@diedrichsen_motor_2015]. Variations are explored during learning process  then stabilizes [@albouy_neural_2012] and learning strategy influences the related implication of striatal structures [@lungu_striatal_2014].
Progressive shift from associative and premotor to sensorimotor basal ganglia distributed network concurs automation of the skill [@lehericy_distinct_2005].

In addition, subthalamic nuclei (STN), substantia nigra (SN) and pallidum were also reported to be activated differently in extended training over multiple days [@lehericy_distinct_2005;@boecker_role_2008].

Cerebellar cortex anatomical functions [@diedrichsen_cerebellar_2014] includes integration of sensory and motor ipsilateral finger representation in Lobule V [@wiestler_integration_2011], complex movements, including coordination, sequencing and spatial processing, recruiting bilateral lobule VI [@schlerf_evidence_2010] and Crus I in Lobule VII [@doyon_contributions_2009], being respectively connected to primary and premotor and supplementary cerebral cortices [@buckner_organization_2011].
Lobule VIII also shows a somatotopicaly organized representation of the whole body, but weaker relatively to Lobule V/VI.
Increased activation of cerebellum dentate nuclei during learning further decline parralleling increased implication of striatal-cortical network [@doyon_experience-dependent_2002].
Model of cerebellar function hypothesize a major role in model-based prediction and sensory feedback processing [@tseng_sensory_2007], thus contributing to error-based learning refining motor production for optimal performance.
This function, mainly studied in motor adaptation experiments, similarly optimize motor primitives and chunking patterns.

It is therefore of higher interest to study how subcortical structures including striatum, hippocampus and cerebellum progressively acquires sequence specific codes in the course of learning.

Physical practice engages a number of structures related to execution, however both preparation [@zang_functional_2003;@lee_subregions_1999], motor imagery[@hetu_neural_2013] and observation [@gazzola_observation_2009;@buccino_neural_2004] activates an overlapping network and induce behavioral performances and plastic changes.
During preparation, representation of the whole sequence can be observed in contralateral premotor and supplementary motor cortex [@nambu_decoding_2015].
These patterns likely reflect the loading of a motor plan [@cisek_neural_2005] independent of forthcoming execution's implementation.
Multiple representation of the sequence could however co-exists encoding sequence in a hierarchical structures [@diedrichsen_motor_2015] from abstract to motor allocentric and egocentric spaces [@wiestler_effector-independent_2014] with both segregated and intertwined spatial and temporal codes [@kornysheva_human_2014].

While training sessions induce critical changes in the networks, evidences of offline reprocessing of acquired procedural memory have accumulated [@walker_sleep_2003;@hotermans_early_2006;@barakat_sleep_2013;@cellini_temporal_2015;@cousins_cued_2014;@gregory_resting_2014;@korman_daytime_2007;@ramanathan_sleep-dependent_2015] despite controversies regarding wake and sleep stages respective roles [@nettersheim_role_2015;@landry_effects_2016].

Sleep-supported consolidation restructures the trace of memory, alleviating the implication of hippocampus, while orthogonal increasing striatum's activation enables the skill to become automated [@albouy_hippocampus_2013], requiring lower cognitive load, and long-term retention of behavior.
Congruent with models of consolidation [@born_system_2012], hippocampus acts as a buffer for recent memories, delaying selective transfer to cortical or specialized structures during offline period, enabling balanced plasticity versus stability, necessary for system homeostasis.
Active System Consolidation model hypothesize that large-scale motor network activity undergo "transfer", or rather rebalancing, across days after learning, supported by offline processing notably during sleep [@born_system_2012].

While MSL aggregates cognitive abilities to improve performance of execution, the local or distributed network specifically encoding different features of the sequence is only partially known.
Motor execution becomes less variable with automation of the skill, and subparts of the networks concurrently stabilize [@costa_differential_2004;@albouy_neural_2012;@peters_emergence_2014] as well as evoked BOLD fMRI activity patterns [@wiestler_skill_2013].

The goal of our study is to provide further insight in brain representation of motor sequence with Multivariate Pattern Analysis (MVPA).
Indeed, changes of activity level during learning [@dayan_neuroplasticity_2011] can reflect multiple processes but fine-grained changes in activity patterns [@wiestler_skill_2013] has the potential to better explain the evolution of localized networks encoding the skill, also excerpting subtle contribution of areas not visible using GLM analysis.
For example, while motor execution related network activity is strongly lateralized to the cortex contralateral to the effector limb, ipsilateral finger specific neural patterns are also observed [@diedrichsen_two_2013] that might enable bi-manual coordination.

Using these techniques we intend to map subnetwork showing sequence specific patterns of activity replicating and extending previous research [@wiestler_skill_2013;@kornysheva_human_2014] by scanning larger extent of human brain. This larger coverage will enable probing activity patterns in sub-cortical structures contributing to motor sequence learning including cerebellum, basal ganglia and lower temporal cortex.

The dynamic of sequential movement planning to execution will also be decomposed to analyze instruction and execution phases separately, similarly to [@nambu_decoding_2015].

The current study also intend to measure changes induced by learning and sleep-dependent consolidation by comparing MVPA classifying performance between sequences at different stages of these processes.

##Method

### Participants
 
The study includes ## right-handed young (18-35 years old) healthy volunteers recruited by advertising on scholar and public website.

The subject were not included in case of history of neurological psychological or psychiatric disorders or scoring 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961].

Subjects with BMI greater than 27, smokers, extreme chronotype, night-workers, having travelled across meridian during the 3 previous months, or training as musician or professional typist (for over-training on coordinated finger movements) were also excluded.

Sleep quality was assessed by Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989], and daytime sleepiness  (Epworth Sleepiness Scale [@johns_new_1991]) had to be lower or equal to 9.

Subject were also instructed to abstain from caffeine, alcohol and nicotine, and have regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the study.
Instruction compliance was controlled by non-dominant hand wrist actigraphy (Actiwatch 2, Philips Respironics, Andover, MA, USA) for the week preceding the experiment.

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
Searchlight size has been shown to inflate the extent of significant clusters in searchlight analysis [@etzel_searchlight_2013; @viswanathan_geometric_2012] which motivated the small neighborhood for our analysis.

For both cross-validation schema, confusion matrix was computed for each block of practice providing a more complete representation of classification performance and biases from which can be derived specific or global accuracy percentage.

#### Searchlight Group Analysis

TODO: change to permutation testing [@stelzer_statistical_2013]

Group searchlight maps were computed using mass-univariate one-sample one-tailed T-test to find regions which consistently departed from chance level across subjects.

A contrast of all subject's untrained sequences accuracy maps between the second and first "MVPA task" scans using conjunction with significant second scan accuracy using minimum t-value was computed to only extract representation enhancement during early learning reaching above chance level.

To further investigate the change in representation strength and localization between trained and untrained sequences, we contrasted their accuracy maps in Int group only using a subject pair t-test in conjunction with significant trained sequence accuracy using minimum t-value to chance level.

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
