---
title: Evolution of procedural memory trace during learning and early consolidation
tags: [example, citation]
bibliography: [abstract_rbiq.bib]
author:
- family: Pinsard
  given: Basile
  affiliation: 1,2
  email: basile.pinsard@gmail.com
- family: Boutin
  given: Arnaud
  affiliation: 1,3
- family: Gabitov
  given: Ella
  affiliation: 1,3
- family: Benali
  given: Habib
  affiliation: 2,5
- family: Doyon
  given: Julien
  affiliation: 1,3,4
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
- id: 3
  name: McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University, 
  address: 3801, University Street, Montreal, H3A2B4, Canada
  url: https://www.mcgill.ca/bic/
- id: 4
  name: Department of Psychology, University of Montreal
  address: Montreal, Quebec, Canada
- id: 5
  name: PERFORM Centre, Concordia University
  address: Montreal, Quebec, Canada


date: 
---
## Introduction

The capacity to acquire novel sequences of movements have shown to contribute critically to the complex evolving repertoire of motor actions in animals and humans.
Motor sequence learning (MSL), defined as the acquisition of temporally ordered succession of coordinated movements, has thus often been employed to study the behavioral determinants of this type of procedural memory.

While practice of MSL over several training sessions lead to substantial improvement in performance, evidence of further offline reprocessing of the acquired procedural memory have accumulated [@walker_sleep_2003;@press_time_2005;@hotermans_early_2006;@korman_daytime_2007;@morin_motor_2008;@debas_brain_2010;@barakat_sleep_2013;@cousins_cued_2014;@debas_off-line_2014;@gregory_resting_2014;@cellini_temporal_2015;@ramanathan_sleep-dependent_2015;@laventure_nrem2_2016].
This reprocessing is observed as either gains or maintenance of performance compared to a decay if no offline period is allowed to the subject [@nettersheim_role_2015], particularly benefiting explicit MSL [@doyon_contributions_2009].
However, controversies remains regarding the respective roles that both wake and sleeps' different stages play in consolidating motor memory traces [@nettersheim_role_2015;@landry_effects_2016].

Sequential motor skills recruit an extended network of cerebral [@hardwick_quantitative_2013], cerebellar and spinal regions [@vahdat_simultaneous_2015], which involvement evolves differently across learning stages [@dayan_neuroplasticity_2011].
Herein, critical plasticity changes [@ungerleider_imaging_2002;@doyon_reorganization_2005] occurs at both training and consolidation, the latter being modelled [@rasch_reactivation_2008;@born_system_2012;@albouy_hippocampus_2013;@dudai_consolidation_2015;@bassett_learning-induced_2015] as a "transfer" or "reorganization" between nervous system structures supporting such function.

Both cortico-cerebellar and cortico-striatal loop circuits contributes to MSL [@doyon_reorganization_2005], recruiting primary sensorimotor and supplementary motor cortices as well as posterior parietal and dorso-lateral prefrontal cortices [@hardwick_quantitative_2013] concurrently to sub-parts of cerebellum and striatum on which we will further expand.

Cerebellar cortex anatomical functions [@diedrichsen_cerebellar_2014] include integration of sensory and motor ipsilateral finger representation in Lobule V [@wiestler_integration_2011], complex movements, including coordination, sequencing and spatial processing, which recruit the bilateral Lobule VI [@schlerf_evidence_2010] and Crus I in Lobule VII [@doyon_contributions_2009], being respectively connected to primary and premotor and supplementary cerebral cortices [@buckner_organization_2011].
Lobule VIII also shows a somatotopicaly organized representation of the whole body, but weaker relatively to Lobule V/VI.
Cerebellar contributes to model-based prediction and sensory feedback processing [@tseng_sensory_2007], thus support error-based learning in early MSL.
This is observed by an activation shift from cerebellar cortex to dentate nuclei [@doyon_experience-dependent_2002] and lowered activation with extended practice, orthogonal to striatal-cortical loop increased recruitment.

The striatum has been associated in selection, preparation and execution of free or instructed movements [@hikosaka_central_2002;@gerardin_distinct_2004;@doyon_contributions_2009], as well as automation of sequences processing through reinforcement learning [@jin_basal_2014], the caudate nuclei activity increasing with decreased execution variability [@albouy_neural_2012].
The striatum, and more specifically the putamen, was revealed to be further activated after consolidation of a motor memory trace [@debas_brain_2010;@albouy_hippocampus_2013;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
With extended training over multiple weeks, a progressive shift from associative and premotor to sensorimotor basal ganglia distributed network concurs to the automation of the skill [@lehericy_distinct_2005].

Hippocampus is a major structure for memory [@battaglia_hippocampus:_2011] that not only encodes episodic memories, including spatial and sequenced events [@howard_time_2015;@moser_place_2015], but also supports procedural memory acquisition in its early stages [@albouy_both_2008], its activity and interaction with striatum predicting sequences long-term retention [@albouy_interaction_2013].
This hippocampal procedural representation could concurrently bind episodic memory of task execution, an allocentric representation [@albouy_maintaining_2015] and temporary tagging and buffering of memories to be further consolidated.

The abovementioned structures thus undergo non-linear changes of activation level during learning [@dayan_neuroplasticity_2011], and notably after consolidation [@lehericy_distinct_2005;@debas_brain_2010;@debas_off-line_2014], reflecting both increased implication and optimization of specialized circuits and decreased activation of non-specific support networks.

While these large-scale activation changes excerpt the dynamics of the recruited networks, it does not provide evidence that these are directly involved in acquiring motor sequence specific representation.
This can, however, be identified with Multivariate Pattern Analysis (MVPA), a set of techniques recently adapted to neuroimaging [@pereira_machine_2009], which evaluates how local patterns of activity are able to discriminate between stimuli or memories of the same class.

To our knowledge, few studies have applied MVPA to MSL, with various design aimed at specific identification of whole sequence or features representation.

In a recent study focusing on motor-related cerebral cortices [@wiestler_skill_2013], classifier performance was evaluated on sets of sequences (either trained or not) and showed a potential increase of the representation strength with training in a network spanning bilaterally the primary motor, pre-motor and supplementary motor areas to the parietal cortex.
Analyzing the dynamics of classifier decoding performance they also uncovered the timecourse of motor sequence representation in different cortical areas from instruction to execution, slightly preceding the BOLD response.

Prior to motor sequence execution, preparation activates a network that overlaps with the one activated during movements per se [@lee_subregions_1999;@zang_functional_2003;@orban_richness_2015], thus, another study [@nambu_decoding_2015] specifically analyzed patterns of activity after instruction and prior to execution of extensively trained sequences with dominant hand.
Patterns related to motor planning, without contamination of motor execution or sensory feedback related activity, were detected in a restricted network involving the contralateral dorsal premotor and supplementary motor area, but not the basal ganglia.

Numerous neuroimaging [@hikosaka_central_2002;@albouy_both_2008;@orban_multifaceted_2010;@penhune_parallel_2012;@albouy_neural_2012] and neuronal level studies [@alexander_parallel_1986;@graybiel_building_1995;@miyachi_differential_2002] have corroborated the implication of basal ganglia, notably the striatum, as well as hippocampus, in acquisition and consolidation of sequential motor skills.
Nevertheless, the acquisition of motor sequence representation during MSL in these regions critical to MSL is still to be assessed.

The goal of the present study is thus to identify newly acquired finger-presses sequences representation and their reorganization after consolidation.
We hypothesized that strengthened cortical and striatal with lowered hippocampal representation, as assessed by multivariate difference in local activity, reflects this reorganization.

##Method

### Participants

The study includes 18 right-handed young (18-35 years old) healthy volunteers recruited by advertising on scholar and public website.

The subject were not included in case of history of neurological psychological or psychiatric disorders or scoring 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961].

Subjects with BMI greater than 27, smokers, extreme chronotype, night-workers, having traveled across meridian during the 3 previous months, or training as musician or professional typist (for over-training on coordinated finger movements) were also excluded.

Sleep quality was assessed by Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989], and daytime sleepiness  (Epworth Sleepiness Scale [@johns_new_1991]) had to be lower or equal to 9.

Subject were also instructed to abstain from caffeine, alcohol and nicotine, and have regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the study.
Instruction compliance was controlled by non-dominant hand wrist actigraphy (Actiwatch 2, Philips Respironics, Andover, MA, USA) for the week preceding and the duration of the experiment.

### Behavioral experiment

The experiment was conducted over 3 consecutive days, at the end of the day, with all motor task performed in the scanner using an ergonomic MRI-compatible 4-keys response pad.

On the first evening (D1), subjects were trained to perform with their left-hand a 5 elements sequence (TSeq) for 14 blocks of 12 sequences or a maximum of 12x5=60 keypresses.
Subject were instructed to execute repeatedly as fast and as accurate as possible the sequence of keypresses and to start from the beginning of the sequences in case they noticed that they did an error.
They were retested approximately 20 minutes later for an additional single block of 12 sequences.

On the second evening (D2), subjects were first tested for 1 block on TSeq, then were trained on an interfering sequences (IntSeq) of 5 elements with left-hand for 14 blocks of 12 sequences as for TSeq.

On the third evening (D3), subjects first performed TSeq for 7 blocks followed by 7 blocks of IntSeq, each block including 12 repetitions of the sequence or 60 keypresses. 

This was followed by a task specifically designed for MVPA analysis, similar to @wiestler_skill_2013, alternating short blocks of practice of 4 different sequences.
However it differed in that the 4 sequences used the left hand four fingers excluding the thumb, and were, as for the initial training, performed repeatedly without interruption not and given error feedback, this in order to probe the processes underlying automation of the skill.

Each block, composed of an instruction period of 4 sec when was displayed 5 numbers (eg. 1-4-2-3-1) representing in reading order the sequence of fingers to be pressed, followed by an execution period indicated by a green cross.
Subject had to perform 5 times the sequence, or a maximum of 5x5=25 key-presses, before being instructed to stop and rest by displaying a red cross.

Ordering of the sequences in blocks was chosen to include all possible successive pairs of the sequences using De Bruijn cycles [@aguirre_bruijn_2011] allowing unbiased analysis of dynamics described below.
Given 4 sequences, a 2-length De Bruijn cycle would repeat each 4 times giving 16 blocks.
This cycle was repeated twice in each of 2 scans separated by few minutes, giving 4 groups of 16 practice blocks or a total of 64 blocks.

### Scan acquisition

MRI data were acquired on a Siemens Trio 3T scanner on 2 two separate sessions.
The first session used 32-channel coil to acquire high-resolution anatomical T1 weighted image using Multi-Echo (4) MPRAGE (MEMPRAGE) at 1mm isotropic resolution. 

Functional data were acquired during the second session with 12-channel coil for comparison with other dataset. EPI sequence consists of 40 slices with 3.44mm in-plane resolution and 3.3mm slice thickness to provide full cortical and cerebellum coverage with a TR of 2.16 sec.
Consecutively fieldmap was obtained to measure B0 field inhomogeneity to allow retrospective compensation of induced distortions.

### Preprocessing

Custom pipeline was used to preprocess fMRI data prior to analysis.
First, high-resolution anatomical T1 weighted image was preprocessed with Freesurfer [@dale_cortical_1999;@fischl_high-resolution_1999;@fischl_cortical_2008] to segment subcortical regions, reconstruct cortical surfaces and provide inter-subjects alignment of cortical folding patterns. 
Pial and grey/white matter interface surfaces were downsampled to match the 32k sampling of Human Connectome Project (HCP) [@glasser_minimal_2013].
HCP subcortical atlas coordinates were warped onto individual T1 data using non-linear registration based on Ants software [@avants_symmetric_2008;@klein_evaluation_2009].

fMRI data was processed using an integrated method (under review) which combines slice-wise motion estimation and intensity correction followed by resampling of cortical and subcortical gray matter timecourse extraction removing B0 inhomogeneity induced EPI distortion.
BOLD signal was further processed to remove drifts and motion-related abrupt signal change.

Importantly, this preprocessing did not include smoothing, even though interpolation inherent to any motion correction causes averaging of values of neighboring voxels.
This intend to minimize blurring of data to preserve fine-grained patterns of activity, the resolution of relevant patterns being hypothetically at columnar scale.

### Multivariate Pattern Analysis

#### Samples

Each block was modeled by having 2 boxcars, respectively instruction and execution phase, convolved with Hemodynamic Response Functions (HRF). Least-square separate (LS-S) regression of each event [@mumford_deconvolving_2012] shown to provide improved activation patterns estimates for MVPA.
For each blocks, regressors for instruction and execution phases provided betas maps that was further used as MVPA samples.

#### Cross-validated multivariate distance

Similarly to @wiestler_skill_2013 and @nambu_decoding_2015 we aim to uncover activity patterns representing the different sequences that is performed by the subject.
Instead of applying cross-validated classification, we opted for a representational approach by computing multivariate distance between evoked activity patterns, in order to avoid the formers' ceiling effect and noise sensitivity [@walther_reliability_2016].
Cross-validated Mahalanobis distance [@walther_reliability_2016], is an unbiased metric that uses multivariate normalization by estimating the covariance from the GLM fitting residuals, regularized through Ledoit-Wolf optimal shrinkage [@ledoit_honey_2012].
Distance were estimated between pairs of sequences that were in a comparable acquisition stage, that is for consolidated sequences separately from newly acquired sequences.

#### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that applies MVPA repeatedly on small spatial neighborhoods covering the whole brain while avoiding high-dimensional limitation of multivariate algorithms.
Searchlight was configured to select for each grayordinate the 128 closest neighboring coordinates, using geodesic distance for cortical grayordinates, as the subset of features for representational distance estimation.

#### Statistical testing

To assess statistical significance of multivariate distance and contrasts, group-level Monte-Carlo non-parametric statistical testing using 10000 permutations was conducted on searchlight distance maps with Threshold-Free-Cluster-Enhancement correction and thresholded at $p<.05$ (with confidence interval $\pm.0044$ for 10000 permutations) with a minimum cluster size of 25 features.

The MVPA analysis was based on PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom samples extraction, cross-validation scheme, Searchlight and measures to adapt to the study design and data.

## Results

For both new and consolidated sequences (@fig:new_crossnobis_tfce_map,@fig:cons_crossnobis_tfce_map), a large network shows differentiated patterns of activity, including primary visual cortex that processes the visual instructions, as well as posterior parietal, primary and supplementary motor, premotor and dorsolateral prefrontal cortices.
Subcortical regions also show differing activity patterns, including ipsilateral cerebellum, bilateral thalamus, hippocampus and striatum.

![Group searchlight map of cross-validated Mahalanobis distance between the 2 new unconsolidated sequences (z-score thresholded at p<.05 TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_crossnobis_tfce_map.pdf){#fig:new_crossnobis_tfce_map}

![Group searchlight map of cross-validated Mahalanobis distance between the 2 consolidated sequences (z-score thresholded at p<.05 TFCE-cluster-corrected) ](../../results/crossnobis_tfce/cons_crossnobis_tfce_map.pdf){#fig:cons_crossnobis_tfce_map}

When contrasting the multivariate distance between consolidated and unconsolidated sequences ( @fig:contrast_cons-new_crossnobis_tfce_map) higher discriminability or representation is found in bilateral putamen, contralateral caudate nuclei, thalamus, ventral and dorsal premotor, supplementary motor and dorsolateral prefrontal cortices for consolidated sequences.
Conversely, the representation strength decreases for consolidated sequences in bilateral hippocampus and ipsilateral body of the caudate nuclei.

![Group searchlight contrasts of cross-validated Mahalanobis distance between consolidated and unconsolidated sequences (z-score thresholded at p<.05 TFCE-cluster-corrected) ](../../results/crossnobis_tfce/contrast_cons_new_crossnobis_tfce_map.pdf){#fig:contrast_cons-new_crossnobis_tfce_map}

While patterns differentiating newly acquired sequences exists in contralateral putamen and bilateral caudate, this distances increases for consolidated sequences in bilateral putamen.

## Discussion

Behaviorally assessed consolidation is subtended by long-term changes in neuronal circuits that support the efficient retrieval and expression of memories.
In the present study we aimed to measure the changes in activity patterns that reflect the specialization of distributed network for the execution of different motor sequential skills.
 
(recapitulate the results)
The present results first shows which regions are differentially recruited to performed two sequences which underwent similar consolidation and two sequences that are newly acquired.
An extended network of regions showing reliable sequence specific activity patterns for both sequence types emcompasses bilateral supplementary and pre-motor to posterior parietal cortices, while contralateral primary sensorimotor regions are only different between novel sequences.

We further investigated how early consolidation restructures the distributed representation of sequential motor skills, notably between cortical and subcortical regions.
Interestingly bilateral putamen, more precisely ventral posterior regions, show higher discriminability.
This also aligns with recent results [@kawai_motor_2015] showing that striatum is "tutored" by the cortex during training and is able afterward to express the skills in the absence of neo-cortical inputs.


frontal results:
These results could be explained by the task itself, which require more cognitive processing, notably switching between and inhibiting interfering sequences.

M1 for new sequences:
The stronger pattern difference in primary motor cortex for newly learned sequences could reflect their structure difference.
This first finger press was recently shown to elicit stronger activation thus driving separability of patterns for sequences with different starts [@yokoi_does_human_2017] unbeknownst to us during experimental design.
While our design differs in that the sequences are uninterruptedly repeated five times, the first finger effect could remain for the first execution of the newly learned sequence only, the consolidated one having the same initiating finger.

(limitations)





(what's next)




# Funding

This work was supported by the Canadian Institutes of Health Research (MOP 97830) to JD, as well as by French Education and Research Ministry and Sorbonne Universités to BP. __+Ella? +Arnaud(QBIN)__
