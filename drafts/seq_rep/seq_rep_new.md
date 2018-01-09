---
title: Evolution of procedural distributed memory trace during acquisition and after consolidation
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
Motor sequence learning (MSL) has thus often been behaviorally studied as the acquisition of temporally ordered succession of coordinated movements, either implicitly or explicitly instructed, in order to understand the mechanisms underlying procedural memory [@abrahamse_control_2013;@diedrichsen_motor_2015;@verwey_cognitive_2015].

While practice of an explicit MSL task leads to substantial within-session improvements in performance, evidence showing additional offline performance gains and associated reprocessing of the memory trace has accumulated [@walker_sleep_2003;@press_time_2005;@hotermans_early_2006;@korman_daytime_2007;@morin_motor_2008;@debas_brain_2010;@barakat_sleep_2013;@cousins_cued_2014;@debas_off-line_2014;@gregory_resting_2014;@cellini_temporal_2015;@ramanathan_sleep-dependent_2015;@laventure_nrem2_2016;@king_sleeping_2017].
The reprocessing of a motor memory trace during subsequent sleep periods yields gains, or at least maintenance of performance as compared to the decay observed following a waking period of equal duration [@nettersheim_role_2015].
However, controversies remains regarding the respective roles that both wake and sleep play in the consolidation of motor memories [@brawn_consolidating_2010;@nettersheim_role_2015;@landry_effects_2016].

Sequential motor skills recruit an extended network of cerebral [@hardwick_quantitative_2013], cerebellar and spinal regions [@vahdat_simultaneous_2015], which activities differentiate across learning stages [@dayan_neuroplasticity_2011].
Herein, critical plastic changes [@ungerleider_imaging_2002;@doyon_reorganization_2005] occur at both training and consolidation stages, the latter being modeled as a "transfer" or more accurately as a "reorganization" between nervous system structures supporting such function [@rasch_reactivation_2008;@born_system_2012;@albouy_hippocampus_2013;@dudai_consolidation_2015;@bassett_learning-induced_2015;@fogel_reactivation_2017].
Primary sensorimotor and supplementary motor cortices as well as posterior parietal and dorso-lateral prefrontal cortices [@hardwick_quantitative_2013] are parts of cortico-cerebellar (CC) and cortico-striatal (CS) loops which undergo such reorganization in the course of MSL [@doyon_reorganization_2005].

Cerebellar cortices involved early-on in MSL [@diedrichsen_cerebellar_2014] includes lobule V which contribute to the integration of sensory and motor ipsilateral finger representation [@wiestler_integration_2011], bilateral Lobule VI [@schlerf_evidence_2010] and Crus I in Lobule VII [@doyon_contributions_2009] support complex movements, coordination, sequencing and spatial processing.
They compose the MSL-recruited CC loop, being respectively connected functionally to primary, premotor and supplementary motor cerebral cortices [@buckner_organization_2011].
Similarly to their neocortical counterpart, Lobule V/VI exhibit a somatotopicaly organized representation of the whole body.
The cerebellum, known to be critically involved in motor adaptation [@doyon_distinct_2003;@diedrichsen_cerebellar_2014], also contributes to MSL in building an internal model for optimizing performance, controlling movements and correcting errors [@penhune_parallel_2012].
However, these are less required with extended practice and cerebellar cortex activation lowers and shifts to dentate nuclei [@doyon_experience-dependent_2002], orthogonally to the increased recruitment of cortical-striatal loop.

Indeed, the striatum has been associated in the selection, preparation and execution of free or instructed movements [@hikosaka_central_2002;@gerardin_distinct_2004;@doyon_contributions_2009], as well as automation of sequences processing through reinforcement learning [@jin_basal_2014].
This notably reflects in the increase of caudate nuclei activity with lower execution variability [@albouy_neural_2012].
Importantly, the putamen was revealed to be further activated after consolidation of a motor memory trace [@debas_brain_2010;@albouy_hippocampus_2013;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
With extended training over multiple weeks, a progressive shift from associative and premotor to sensorimotor basal ganglia distributed network, notably the putamen, concurs to the automation of the skill [@lehericy_distinct_2005].

As for the hippocampus, mainly known to encode episodic memories [@battaglia_hippocampus:_2011] including spatial and sequences of events [@howard_time_2015;@moser_place_2015], it has been shown to also support procedural memory acquisition in its early stages [@albouy_both_2008], its activity and interaction with striatum predicting long-term retention of motor sequences [@albouy_interaction_2013].
Hence, the hippocampus contribution to procedural memory is thought to favor the building and storage of an effector-independent and allocentric (visual-spatial) representation [@albouy_maintaining_2015] and to tag memories temporarily for further reactivation during succeeding offline consolidation [@dudai_consolidation_2015].

All the aforementioned structures thus undergo non-linear changes of activation level during learning [@dayan_neuroplasticity_2011;@diedrichsen_motor_2015], and notably after consolidation [@lehericy_distinct_2005;@debas_brain_2010;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
However, while increasing activity can suggest an increased implication of specialized circuits, decreasing activity could either reflect optimization of these same circuits or decreased recruitment of non-specific support networks.
Therefore, the observed large-scale activation changes does not provide evidence that these regions are directly involved in the acquisition of motor sequence specific representation [@berlot_search_2018].

However, sequences specific representation can be identified with Multivariate Pattern Analysis (MVPA), a set of techniques recently adapted to neuroimaging [@pereira_machine_2009], which evaluates how local patterns of activity are able to discriminate between stimuli or memories of the same type.
In the MSL literature, few studies used an MVPA approach to localize the representation of motor sequences with various design aimed at specific identification of whole sequence [@wiestler_skill_2013] or specific features [@wiestler_integration_2011;@kornysheva_human_2014;@yokoi_does_2017].
In a recent study focusing on motor-related cerebral cortices [@wiestler_skill_2013], classifier performance was evaluated on trained and untrained sets of sequences and showed a potential increase of the representation strength with training in a network spanning bilaterally the primary motor, pre-motor, supplementary motor areas and parietal cortices.
By analyzing the dynamics of classifier decoding performance they also uncovered that significant motor sequence representation precedes the expected BOLD response in different cortical areas during instruction and execution.
Indeed, prior to motor sequence execution, preparation activates a network that overlaps with the one activated during movements per se [@lee_subregions_1999;@zang_functional_2003;@orban_richness_2015].
This motivated another group [@nambu_decoding_2015] to specifically analyze patterns of preparatory activity after instruction and prior to execution of extensively trained sequences to remove confounds from execution or sensory feedback.
During this preparatory period, sequence representations were localized in contralateral dorsal premotor and supplementary motor cortices only, while during execution these were detected in a larger bilateral network comprising premotor, somatosensory and posterior parietal cortices as well as cerebellum, but surprisingly not in basal ganglia.

Nevertheless, numerous neuronal level [@alexander_parallel_1986;@graybiel_building_1995;@miyachi_differential_2002] and neuroimaging studies [@hikosaka_central_2002;@albouy_both_2008;@orban_multifaceted_2010;@penhune_parallel_2012;@albouy_neural_2012] have corroborated the implication of the basal ganglia, in particular the putamen, as well as the hippocampus, in the acquisition and consolidation of sequential motor skills.
Nevertheless, the genuine acquisition of motor sequence representation in such MSL-involved regions still needs to be assessed.

The goal of the present study was thus to identify newly acquired finger-presses sequences representation across the whole brain and their reorganizations after consolidation.
Hence, and as assessed by multivariate difference in local subcortico-cortical activity, we hypothesized that strengthened motor-related cortical and striatal representations, in conjunction with weaker hippocampal-based representations, would reflect the reorganization of the memory trace during consolidation.

##Method

### Participants

The study includes 18 right-handed young (18-35 years old) healthy volunteers recruited by advertising on scholar and public website.

The subject were not included in case of history of neurological psychological or psychiatric disorders or scoring 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961].

Subjects with BMI greater than 27, smokers, extreme chronotype, night-workers, having traveled across meridian during the 3 previous months, or training as musician or professional typist (for over-training on coordinated finger movements) were also excluded.

Sleep quality was assessed by Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989], and daytime sleepiness  (Epworth Sleepiness Scale [@johns_new_1991]) had to be lower or equal to 9.

Subject were also instructed to abstain from caffeine, alcohol and nicotine, and have regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the study.
Instruction compliance was controlled by non-dominant hand wrist actigraphy (Actiwatch 2, Philips Respironics, Andover, MA, USA) for the week preceding and the duration of the experiment.

### Behavioral experiment

The experiment was conducted over 3 consecutive days, at the end of the day, with the motor tasks performed in the scanner using an ergonomic MRI-compatible 4-keys response pad.

On the first evening (D1), subjects were trained to perform with their non-dominant left-hand a 5 elements sequence (TSeq) for 14 blocks (indicated by a green cross displayed in the center of the screen) each composed of 12 repetitions of the motor sequences (ie. 60 keypresses per block).
Subjects were instructed to execute repeatedly as fast and accurate as possible the sequence of keypresses until completion of the practice block.
Practice blocks were interspersed with 25-s rest periods (indicated by the onset of a red cross on the screen) to prevent fatigue.
In case of mistake during sequence production, subjects were asked to stop their performance and to immediately start practicing again from the beginning of the sequence until the end of the block.
Approximately 20 minutes after the completion of the training phase, subjects were administered a retention test, which consisted of a single block similar to training ones.

On the second evening (D2), subjects were evaluated on the TSeq during a retest session (1 block), and were then trained on an interfering sequences (IntSeq) of 5 elements with their left-hand for 14 blocks of 12 sequences as for TSeq.

On the third evening (D3), subjects first performed TSeq for 7 blocks followed by 7 blocks of IntSeq, each block including 12 repetitions of the sequence or 60 keypresses. 

This was followed by a task specifically designed for MVPA analysis, which alternates short practice blocks of 4 different sequences similarly to @wiestler_skill_2013.
It however differed in that, in our study, all 4 sequences used the left-hand 4 fingers excluding the thumb.
Also, as for the initial training, sequences were performed repeatedly and without interruption nor any feedback, this in order to probe the processes underlying automatization of the skill.

Each block was composed of an instruction period of 4 sec. when was displayed 5 numbers (eg. 1-4-2-3-1) representing in reading order the sequence of fingers to be pressed, which was followed by an execution period indicated by a green cross.
Subjects had to perform 5 times the sequence, or a maximum of 25 key-presses, before being instructed to stop and rest by displaying a red cross.

Ordered assignement of sequences to blocks was chosen to include all possible successive pairs of the sequences using De Bruijn cycles [@aguirre_bruijn_2011].
This prevented systematic leakage of BOLD activity between blocks, allowing unbiased analysis of the dynamic of activity pattern described below.
A 2-length De Bruijn cycle of the 4 sequences repeats each one 4 times, yielding a total of 16 blocks.
This cycle was repeated twice in each of the 2 scanning sessions separated by approximately 5 minutes, thus resulting in a total of 64 blocks (4 groups of 16 practice blocks).

### Scan acquisition

MRI data were acquired on a Siemens Trio 3T scanner on 2 two separate sessions.
The first session used a 32-channel coil to acquire high-resolution anatomical T1 weighted image using Multi-Echo (4) MPRAGE (MEMPRAGE) at 1mm isotropic resolution. 

Functional data were acquired during the second session with a 12-channel coil for comparison with other dataset. EPI sequence consists of 40 slices with 3.44mm in-plane resolution and 3.3mm slice thickness to provide full cortical and cerebellum coverage with a TR of 2.16 sec.
Following fMRI data acquisition, a short EPI set of data was acquired with reversed phase encoding to correct for B0 field inhomogeneity induced distortions.

### Preprocessing

Custom pipeline was used to preprocess fMRI data prior to analysis.
First, high-resolution anatomical T1 weighted image was preprocessed with Freesurfer [@dale_cortical_1999;@fischl_high-resolution_1999;@fischl_cortical_2008] to segment subcortical regions, reconstruct cortical surfaces and provide inter-subjects alignment of cortical folding patterns. 
Pial and grey/white matter interface surfaces were downsampled to match the 32k sampling of Human Connectome Project (HCP) [@glasser_minimal_2013].
HCP subcortical atlas coordinates were warped onto individual T1 data using non-linear registration using the Ants software [@avants_symmetric_2008;@klein_evaluation_2009].

fMRI data was processed using an integrated method (under review) which combines slice-wise motion estimation and intensity correction followed by the resampling of cortical and subcortical gray matter timecourse extraction.
This interpolation concurrently removed B0 inhomogeneity induced EPI distortion estimated by FSL Topup using reversed phase encoding data [@andersson_how_2003].
BOLD signal was further processed to remove drifts and motion-related abrupt signal changes.

Importantly, this preprocessing did not include smoothing, even though interpolation inherent to any motion correction causes averaging of values of neighboring voxels.
This intended to minimize the blurring of data in order to preserve fine-grained patterns of activity, the resolution of relevant patterns being hypothetically at columnar scale.

### Multivariate Pattern Analysis

#### Samples

Each block was modeled by having 2 boxcars, respectively instruction and execution phases, convolved with Hemodynamic Response Functions (HRF).
Least-square separate (LS-S) regression of each event [@mumford_deconvolving_2012], shown to provide improved activation patterns estimates for MVPA, yielded instruction and execution phases beta maps for each block that were further used as MVPA samples.

#### Cross-validated multivariate distance

Similarly to @wiestler_skill_2013 and @nambu_decoding_2015, we aimed to uncover activity patterns representing the different sequences that were performed by the subject.
However, instead of applying cross-validated classification, we opted for a representational approach by computing multivariate distance between evoked activity patterns, in order to avoid the formers' ceiling effect and noise sensitivity [@walther_reliability_2016].
Cross-validated Mahalanobis distance [@walther_reliability_2016] is an unbiased metric that uses multivariate normalization by estimating the covariance from the GLM fitting residuals, regularized through Ledoit-Wolf optimal shrinkage [@ledoit_honey_2012].
Distance were estimated between pairs of sequences that were in a comparable acquisition stage, that is separately for the newly acquired and consolidated sequences.

#### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that applies MVPA repeatedly on small spatial neighborhoods covering the whole brain while avoiding high-dimensional limitation of multivariate algorithms.
Searchlight was configured to select for each grayordinate the 128 closest neighboring coordinates, using geodesic distance for cortical grayordinates, as the subset of features for representational distance estimation.

#### Statistical testing

To assess statistical significance of multivariate distance and contrasts, group-level Monte-Carlo non-parametric statistical testing using 10000 permutations was conducted on searchlight distance maps with Threshold-Free-Cluster-Enhancement correction and thresholded at $p<.05$ (with confidence interval $\pm.0044$ for 10000 permutations) with a minimum cluster size of 25 features.

The MVPA analysis was done using the PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom samples extraction, cross-validation scheme, Searchlight and measures to adapt to the study design and data.

## Results

For both new and consolidated sequences (@fig:new_cons_crossnobis_tfce_map), a conjunction statistic reveals a large network with differentiated patterns of activity, including primary visual cortex that processes the visual instructions, as well as posterior parietal, primary and supplementary motor, premotor and dorsolateral prefrontal cortices.
When looking at separate results, subcortical regions also show differing activity patterns, including ipsilateral cerebellum, bilateral thalamus, hippocampus and striatum (@fig:new_crossnobis_tfce_map,@fig:cons_crossnobis_tfce_map).

![Group searchlight conjunction map cross-validated Mahalanobis distance within new and consolidated sequences (z-score thresholded at p<.05 TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_cons_conj_crossnobis_tfce_map.pdf){#fig:new_cons_conj_crossnobis_tfce_map}


When contrasting the multivariate distance between consolidated and unconsolidated sequences ( @fig:contrast_cons-new_crossnobis_tfce_map) higher discriminability or representation is found in bilateral putamen, contralateral caudate nuclei, thalamus, ventral and dorsal premotor, supplementary motor and dorsolateral prefrontal cortices for consolidated sequences.
Conversely, the representation strength decreases for consolidated sequences in bilateral hippocampus and ipsilateral body of the caudate nuclei.

![Group searchlight contrasts of cross-validated Mahalanobis distance between consolidated and unconsolidated sequences (z-score thresholded at p<.05 TFCE-cluster-corrected) ](../../results/crossnobis_tfce/contrast_cons_new_crossnobis_tfce_map.pdf){#fig:contrast_cons-new_crossnobis_tfce_map}

While patterns differentiating newly acquired sequences exists in contralateral putamen and bilateral caudate, this distances increases for consolidated sequences in bilateral putamen.

## Discussion

Behaviorally assessed consolidation is subtended by long-term changes in neuronal circuits that support the efficient retrieval and expression of memories.
In the present study we aimed to measure the changes in activity patterns that reflect the specialization of distributed network for the execution of different motor sequential skills.
 
(recapitulate the results)
The present results first shows which regions are differentially recruited to performed two sequences which underwent consolidation and two sequences that are newly acquired.
An extended network of regions showing reliable sequence specific activity patterns for both sequence types emcompasses bilateral supplementary and pre-motor to posterior parietal cortices, while contralateral primary sensorimotor regions are only different between novel sequences.
The herein found ipsilateral representation have already been described [@wiestler_skill_2013], notably when non-dominant hand is used for such skills.

We further investigated how early consolidation restructures the distributed representation of sequential motor skills, notably between cortical and subcortical regions.
Interestingly bilateral putamen, more precisely ventral posterior regions, show increased sequence representation, corroborating previous accounts of consolidation related increased BOLD activation [@].
This aligns with recent results [@kawai_motor_2015] showing that striatum is "tutored" by the cortex during training and is afterward able to express acquired skills even in the absence of neo-cortical inputs.
However when cortex is intact from damages, the distributed representation of the skill precise execution yet include large cortical networks [@].

Concurrently to striatal representation emergence, the only few regions which show decreased sequence discrimination include ipsilateral caudate nuclei and bilateral hippocampus, which purportedly supports motor sequential skills early acquisition [@albouy_hippocampus_2013].
However, the present study was not designed to more precisely investigate the space of hippocampal and striatal sequence representation that were previously assessed at cortical level for finger sequences [@wiestler_effector-independent_2014] as well as for larger forearm movements [@haar_effector-invariant_2017].
As our results reflect, motor sequences consolidation reorganizes both hippocampal and striatal representations, which respecive contributions to extrinsic and intrinsic skill encoding will need to be assessed.

Interestingly, our results investigated sequences representation early after consolidation while previous studies compared sequences intensely trained for multiple days to newly acquired ones.
It is therefore possible that these representations would further evolve with either additional training or offline memory reprocessing supported in part by sleep.

(further discuss the results)
(frontal results:)
These results could be explained by the task itself, which require more cognitive processing, notably the implied switch between sequences and inhibition of interfering ones.

(M1 for new sequences:)
Pattern difference in primary motor cortex is only found in newly learned sequences, but this could reflect their structural difference.
Indeed, the first finger press was recently shown to elicit stronger activation thus driving separability of patterns for sequences with different starts [@yokoi_does_2017] unbeknownst to us during experimental design.
This initiating finger effect could remain for the first execution of the newly learned sequence only, the consolidated one having the same initiating finger.
The relatively weak primary motor representation, as compared to @wiestler_skill_2013 study, could be partly accounted by our subjects executing the five sequences uninterruptedly, singling the effect of initiating finger.

(limitations)
To bound the difficulty and the duration of the task, only four sequences were practiced by the subjects, two consolidated and two newly acquired.
This could factor limitating the power of our analysis, as only a single multivariate distance is assessed for each of these condition.
Moreover, the sequences were trained independently during longer blocks, and thus the present task further induces demands for instruction processing, switching and inhibition that could trigger novel learning.
However we supposed that the previously encoded motor sequences engram of consolidated sequences is retrieved during this task.


## Conclusion

To conclude our study shows 

# Acknowledgment



# Funding

This work was supported by the Canadian Institutes of Health Research (MOP 97830) to JD, as well as by French Education and Research Ministry and Sorbonne Universités to BP. __+Ella? +Arnaud(QBIN)__

# Supplementary

![Group searchlight map of cross-validated Mahalanobis distance between the 2 new unconsolidated sequences (z-score thresholded at p<.05 TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_crossnobis_tfce_map.pdf){#fig:new_crossnobis_tfce_map}

![Group searchlight map of cross-validated Mahalanobis distance between the 2 consolidated sequences (z-score thresholded at p<.05 TFCE-cluster-corrected) ](../../results/crossnobis_tfce/cons_crossnobis_tfce_map.pdf){#fig:cons_crossnobis_tfce_map}

# References
