---
documentclass: elife
elife: true
title: Representational changes in a distributed striato-cerebello-hippocampo-cortical network underlies the consolidation of sequential motor memories.
title: Evolution of distributed sequence-specific motor skills encoding.
title: Consolidation alters the distributed motor skill sequence-specific representations.
tags: [example, citation]
bibliography: [seq_rep_paper.bib]
author:
- family: Pinsard
  given: Basile
  affiliation: 1,2
  email: basile.pinsard@gmail.com
  initials: BP
- family: Boutin
  given: Arnaud
  affiliation: 2,3
- family: Gabitov
  given: Ella
  affiliation: 2,3
- family: Lungu
  given: Ovidiu
  affiliation: 2
- family: Benali
  given: Habib
  affiliation: 1,5
- family: Doyon
  given: Julien
  affiliation: 2,3,4
institute: here
organization:
- id: 1
  name: Sorbonne Université, CNRS, INSERM, Laboratoire d’Imagerie Biomédicale, LIB
  address: 75006 Paris, France
  url: http://lib.upmc.fr
- id: 2
  name: Functional Neuroimaging Unit, Centre de Recherche de l'Institut Universitaire de Gériatrie de Montréal
  address: Montreal, Canada
  url: http://unf-montreal.ca
- id: 3
  name: McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University
  address: Montreal, Canada
  url: https://www.mcgill.ca/bic/
- id: 4
  name: Department of Neurology and Neurosurgery, Montreal Neurological Institute, McGill University
  address: Montreal, Canada
- id: 5
  name: PERFORM Centre, Concordia University
  address: Montreal, Canada
date: 
chapters: True
chaptersDepth: 1
chapDelim: ""
abstract: |
  The acquisition of new motor sequential skills combines processes leading asymptotically to optimal performance which are supported by dynamic implication of different brain networks.
  Numerous investigations using functional magnetic resonance imaging in humans have revealed that these processes induce a functional reorganization within the cortico-striatal and cortico-cerebellar motor systems with contribution of the hippocampus, manifested by learning-related increases of activity in some regions and decreases in others.
  The functional significance of these changes is not fully understood as they convey both the evolution of sequence-specific knowledge and unspecific expertise in the task.
  Moreover, a local increase of activity can indicate some form of specialization or more likely result from faster execution, a decrease could either reflect gained efficiency or lower involvement of non-specific circuits contributing to initial phases of learning, while plasticity could occur even without significant modification of average regional activity level.
  For these reasons, we investigated whole-brain local representational changes using novel multivariate distance between fine-grained patterns evoked through the production of motor sequences, either trained in a single session and consolidated or newly acquired.
  While both consolidated and new sequences were represented in large cortical networks, specific patterns in cortical prefrontal and motor secondary areas as well as dorsolateral striatal and associative cerebellar cortex were heightened by consolidation, while ones in hippocampus and dorsomedial striatum were fading.
  These results show that complementary sequence-specific motor knowledge representations distributed in different striatal, cerebellar, cortical and hippocampal regions evolve differentially during the critical phases of skill acquisition and consolidation and thus specify their roles in its support.
---

# Introduction {#sec:introduction}

Animals and humans are able to acquire and automatize new sequences of movements, allowing them to expand and update their repertoire of complex goal-oriented motor actions for long-term use.
To study the mechanisms underlying this type of procedural memory in humans, a large body of behavioral experiments have used motor sequence learning (MSL) tasks designed to test the ability to perform temporally ordered coordinated movements acquired either implicitly or explicitly and assess their performances at different timescales [@abrahamse_control_2013;@diedrichsen_motor_2015;@verwey_cognitive_2015].
While the practice of an explicit MSL task leads to substantial within-session improvements in performance, there is now ample evidence indicating that between-session increases, conservation or decreases in performance can also be observed [@landry_effects_2016;@nettersheim_role_2015].
It is when this offline period includes sleep, that between-session gains are particularly stabilized, whereas performance tends to decay during an equal waking period [@brawn_consolidating_2010;@nettersheim_role_2015;@landry_effects_2016].
Therefore, it is thought that sleep favors reprocessing of the motor memory trace, thus promoting its consolidation for long-term skill proficiency [see @king_sleeping_2017 for a recent in-depth review].

Functional magnetic resonance imaging (fMRI) studies using General-Linear-Model (GLM) contrasts have indicated that the MSL is associated with the recruitment of an extended network of cerebral [@hardwick_quantitative_2013], cerebellar and spinal regions [@vahdat_simultaneous_2015], whose contributions differentiate as learning progresses [@dayan_neuroplasticity_2011;@doyon_current_2018].
In fact, critical plastic changes [@ungerleider_imaging_2002;@doyon_reorganization_2005] are known to occur within the initial training session, as well as the post-learning consolidation phase, the latter being characterized by a "reorganization" of the nervous system structures supporting this type of procedural memory function [@rasch_reactivation_2008;@born_system_2012;@albouy_hippocampus_2013;@dudai_consolidation_2015;@bassett_learning-induced_2015;@fogel_reactivation_2017;@vahdat_network-wide_2017].
More specifically, MSL practice activates a cortical and cerebellar motor network throughout learning but is also assisted by the hippocampus, and striato-frontal associative regions in the fast-learning phase of the initial practice [@albouy_hippocampus_2013].
These support activities decrease when approaching asymptotic behavioral performance, in parallel with an increase in sensorimotor striatum and cerebellar nuclei activations [@doyon_experience-dependent_2002], both conveying the transition to the slow-learning phase.
This network further emerges after being stabilized by consolidation and reactivated by further MSL practice extending over multiple days [@lehericy_distinct_2005].

A critical issue typically overlooked by previous MSL neuroimaging research using GLM-based contrasts is that changes in brain activity may suggest differentially recruited processes, only some of which may be specifically related to plasticity induced by MSL.
For instance, increases in activity could as well reflect a greater implication of specialized circuits as simply result from the faster motor execution.
Likewise, activity decreases could either indicate optimization and efficiency gains of specific circuits or reduced recruitment of non-specific networks supporting acquisition.
Therefore, even with the use of control conditions to dissociate sequence-specific and unspecific processes [@orban_multifaceted_2010], the observed large-scale activation differences associated with different learning phases do not provide direct evidence of plasticity related to the processing of a motor sequence-specific representation [@berlot_search_2018].
Moreover, it is conceivable that these plastic changes may even occur locally, without significant changes in the GLM-based regional activity level.
Besides, in most studies investigating the consolidation of explicit MSL, these changes are assessed by contrasting brain activity level of novice subjects between their initial training and a delayed practice session, and thus measure not only plasticity for sequence-specific (e.g. optimized chunks) but also task-related expertise (eg. habituation to experimental apparatus, optimized execution strategies, attentional processes).
The latter is notably observed if subject then practice other sequences, which initial performance are then better than that of the one learned first.

To address these specificity limitations, Multivariate Pattern Analysis (MVPA) has been employed to evaluate how local patterns of activity are able to stably discriminate between stimuli or evoked memories of the same type over repeated occurrences, allowing to test information-based hypotheses that GLM contrasts cannot inquire [@hebart_deconstructing_2017].
In the MSL literature, only a few studies have used such MVPA approach to identify the regions that specialize in processing the representation of motor sequences [@wiestler_skill_2013;@nambu_decoding_2015;@wiestler_integration_2011;@kornysheva_human_2014;@yokoi_does_2017], though focusing on later phases of MSL after repeated practice over multiple days.
For instance, in a recent study covering dorsal cerebral cortices only [@wiestler_skill_2013], cross-validated classification accuracy was measured separately on activity patterns evoked by the practice of trained and untrained sets of sequences, showing that training increased the sequence discriminability in a network spanning bilaterally the primary and secondary motor to parietal cortices.
In another study [@nambu_decoding_2015] that separately analyzed the preparation and execution, the representations of extensively trained sequences were identified in the contralateral dorsal premotor and supplementary motor cortices during preparation, and ispilateral parietal, bilateral premotor and motor cortices as well as cerebellum during execution.
In both studies, the regions carrying sequence-specific representations overlapped only partly with GLM-based measures, illustrating the fact that directly measuring coarser differences between novel and trained sequences evoked activity levels cannot assess if these truly purport sequential information.
However, the classification-based measures they used can bias parametric statistical results [@jamalabadi_classification_2016;@allefeld_valid_2015;@combrisson_exceeding_2015;@varoquaux_cross-validation_2017] and are suboptimal to detect representational changes [@walther_reliability_2016].


As a part of a larger study, the present experimental manipulation aims to address both the critical issues overlooked by studies investigating the MSL consolidation with GLM-based approach and the limitations of classifier-based MVPA studies.
Specifically, we adopt a recently developed MVPA approach [@nili_toolbox_2014] that is unbiased and more sensitive to continuous representational changes [@walther_reliability_2016], such that occur in the early stage of MSL consolidation [@albouy_interaction_2013] onto which our focus here is.
Our experimental manipulation hence allows to isolate sequence-specific plasticity, by extracting patterns evoked by the practice of consolidated and new sequences at the same task expertise level and computing this novel multivariate distance using a searchlight approach over the whole brain, in order to cover cortical and subcortical regions critical to MSL.
Based on theoretical model [@albouy_hippocampus_2013;@doyon_current_2018] that account for imaging and animal studies, we hypothesized that offline consolidation following training should induce greater sequence-specific cortical and striatal representations, in conjunction with weaker hippocampal-based ones.

# Results {#sec:results}

To investigate the representations of sequences along the course of learning, subjects (n=18) that learned two finger presses sequences separately in the first two days of the experiment, on the third day, performed a task in which they practiced these two sequences, as well as two new sequences in pseudo-randomly ordered short blocks.
All sequences were executed using their non-dominant hand while functional MRI data was acquired.

## Behavioral performance

New sequences were performed slower ($\beta=.365 , SE=0.047, p<.001$) and less accurately ($\beta=-0.304, SE=0.101,p<.001$) than consolidated one.
Significant improvement across blocks in term of speed ($\beta=-0.018, SE=0.002, p<.001$) but not of accuracy ($\beta=0.014, SE=0.010, p=0.152$) was observed for new sequences, thus showing an expected learning curve visible in @fig:mvpa_task_groupInt_seq_duration.
The consolidated sequences, on the contrary, did not show significant changes in speed ($\beta=-0.006, SE=0.005, p=0.192$) nor accuracy ($\beta=-0.006, SE=0.057, p=0.919$), the asymptotic performance being already reached through practice and consolidation.

![Average and standard deviation of correct sequence durations across the MVPA task blocks.](../../results/behavior/mvpa_task_groupInt_seq_duration.pdf){#fig:mvpa_task_groupInt_seq_duration}

We also verified that there was no significant difference between consolidated sequences in term of speed ($\beta=0.031, SE=0.026, p=0.234$) and accuracy ($\beta=-0.030, SE=0.111, p=0.789$), and ran a similar verification between new sequences which speed ($\beta=0.025, SE=0.045, p=0.577$) and accuracy ($\beta=-0.245, SE=0.138, p=0.076$) did not differ.

## A common distributed sequence representation for consolidated and new sequences

From preprocessed functional MRI data we extracted patterns of activity for each block of practice and computed a cross-validated Mahalanobis distance [@nili_toolbox_2014;@walther_reliability_2016] using Searchlight approach [@kriegeskorte_information-based_2006] over brain cortical surfaces and subcortical regions of interest.
This multivariate distance, when positive, assesses that there is a stable pattern difference between the conditions compared, and thus reflect the discriminability of these conditions.
To assess true pattern and not mere global activity differences, we computed this discriminability measure for sequences which are in the same stage of learning, thus separately for consolidated and new sequences.
From the individual discriminability maps, we then measured the prevalence of discriminability at the group level, using non-parametric testing with Threshold-Free-Cluster-Enhancement (TFCE) [@smith_threshold-free_2009] to enable locally adaptive cluster-correction.

To extract the common regions that show sequence representation at both stages of learning, we then submitted these separate consolidated and new sequences discriminability group results to a minimum-statistic conjunction.
A large distributed network (@fig:new_cons_conj_crossnobis_map) displays significant discriminability, including primary visual, as well as posterior parietal, primary and supplementary motor, premotor and dorsolateral prefrontal cortices.
When looking at separate results for each learning stages, subcortical regions also show differing activity patterns, including ipsilateral cerebellum, bilateral thalamus, hippocampus and striatum (@fig:new_crossnobis_map,@fig:cons_crossnobis_map), which does not overlap across learning stages.

![Group searchlight conjunction of new and consolidated sequences discriminability maps (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_cons_conj_crossnobis_tfce_map_vert_labels.pdf){#fig:new_cons_conj_crossnobis_map width=14cm}

## Reorganization of the distributed sequence representation with consolidation

In order to evaluate the reorganization of sequence representation undergone by consolidation at the group level, the consolidated and new sequences' discriminability maps from all participants were submitted to pairwise t-test with non-parametric testing with TFCE (@fig:contrast_conj_cons_new_crossnobis_map).
To ascertain that differences were supported by significant discriminability, we then calculated the conjunction of the contrast maps with the separate consolidated and new sequences group results respectively for positive and negative differences.

Discriminability was found to be significantly higher for consolidated sequences as compared to new sequences in bilateral sensorimotor putamen, thalamus as well as frontal, anterior insular, posterior cingulate and parietal cortices, ispilateral cerebellar lobule IX, contralateral lateral and dorsal premotor, supplementary motor, insular, and dorsolateral prefrontal cortices and cerebellar Crus I.
Conversely, the representation strength was higher for new sequences in bilateral hippocampus as well as ipsilateral body of the caudate nuclei and subthalamic nuclei.
Hence, while striatal activity patterns differentiating newly acquired sequences exists in contralateral putamen (@fig:new_crossnobis_map), this distance was significantly larger for consolidated sequences in motor regions of bilateral putamen.

![Conjunction of group searchlight contrast (paired t-test) between consolidated and new sequences discriminability maps and separate group discriminability maps for new and consolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/contrast_conj_cons_new_crossnobis_tfce_map_vert.pdf){#fig:contrast_conj_cons_new_crossnobis_map width=14cm}

# Discussion {#sec:discussion}

In the present study we aimed to measure the changes in activity patterns in a distributed network supporting the execution of multiple motor sequential skills associated with behaviorally assessed consolidation.
Locally stable patterns of activity are here used as a proxy for the specialization of neuronal circuits for the support of efficient memory retrieval and expression.
To investigate the differential pattern strength, we computed novel unbiased multivariate distance and applied robust permutation-based statistics with adaptive cluster correction.

## A distributed representation of finger motor sequence

Our results provide evidence that an extended network of regions shows reliable sequence-specific activity patterns for both consolidated and novel sequences.
Cortically, a previously described network [@wiestler_skill_2013;@nambu_decoding_2015] encompasses bilateral supplementary and premotor areas, as well as posterior parietal cortices, while contralateral primary sensorimotor regions were only shown to elicit different patterns for novel sequences.
It is noteworthy that discrimination of motor sequence representations within the ipsilateral motor, premotor and parietal cortices has been previously described [@wiestler_skill_2013;@waters-metenier_bihemispheric_2014;@waters_cooperation_2017], notably when non-dominant hand is used for fine dexterous manual skills.

Difference in activity patterns within the primary motor cortex was only found for newly learned sequences [@fig:new_crossnobis_map], which could reflect their motoric differences in terms of finger presses ordering.
Unbeknownst to us during experimental design, the first finger press was recently shown to elicit higher activation in this somatotopically organized region [@ejaz_hand_2015], thus driving separability of patterns for sequences with different initiating finger [@yokoi_does_2017].
However, the primary motor representation was found to be relatively weak in our study in comparison to [@wiestler_skill_2013], likely explained by the uninterrupted repetition of the motor sequences during the practice singling this effect to the beginning of the block, as well as our 5-element sequences not engaging the thumb which distinctive M1 pattern would have brought stronger difference if initiating the sequence [@ejaz_hand_2015].

The conjunction map reveals that a common cortical processing stream including non-motor support regions present sequential information from visually presented instruction to motor sequence production.
Herein, occipital cortex, as well as ventro-temporal regions are found to discriminate the sequences [@fig:new_cons_conj_crossnobis_map], but likely reflect the processing of the visual stimuli respectively as low-level visual mapping of shapes [@pilgramm_motor_2016;@miyawaki_visual_2008] and higher level Arabic number representation [@shum_brain_2013;@peters_neural_2015] and thus do not differ between the stages of learning studied here [@fig:contrast_conj_cons_new_crossnobis_map].
Interestingly these regions were not reported in previous study [@wiestler_skill_2013] which imaging field-of-view did not cover ventral cortex.
The dorsolateral prefrontal cortex (DLPFC) also exhibit pattern specificity, and was previously reported as encoding the sequence spatial information in working memory, preceding motor command [@robertson_role_2001].
In fact, the cognitive processing required by the MVPA task, implying notably to switch between sequences, maintain them in working memory and to inhibit interfering ones, could here magnify this frontal associative representation.

## Cortico-subcortical representational reorganization underlying memory consolidation

We then investigated how representations are restructured after early consolidation of MSL by contrasting maps of multivariate distance for consolidated and newly acquired sequences [@fig:contrast_conj_cons_new_crossnobis_map].
At the cortical level, we found that contralateral premotor and bilateral parietal regions acquire a stronger representation during consolidation, that likely reflects that the tuning of these neural populations to coordinated movements are consolidated early after learning [@makino_transformation_2017;@yokoi_does_2017;@pilgramm_motor_2016], as was previously observed with longer training [@wiestler_skill_2013].

Exploring similar changes at subcortical level, significant differences are found in bilateral putamen and more specifically ventral posterior regions, which determine the previous report of their increased activation after consolidation [@debas_brain_2010;@albouy_hippocampus_2013;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
Significant representational changes are also found in cerebellum ipsilateral lobule IX as well as contralateral Crus I and II [@doyon_experience-dependent_2002;@penhune_cerebellum_2005;@doyon_contributions_2009;@tomassini_structural_2011], while none is found in finger somatotopic cerebellar regions [@wiestler_integration_2011] concurring with cortical results.

[@alexander_parallel_1986]

Concurrently to this consolidation induced representational emergence, strikingly few regions showed decreased sequence discrimination, namely ipsilateral caudate nuclei and bilateral hippocampus.
The hippocampal early representation have been hypothesized to buffer novel explicit motor sequence learning and concur to the reactivations of the distributed network for reprocessing during offline periods, though progressively disengaging afterward [@albouy_hippocampus_2013].
Our novel findings of differential implication of dorsomedial and dorsolateral striatum in sequence representation during learning and expression of a mastered skill specifies the earlier described activity change in the course of MSL [@lehericy_distinct_2005;@jankowski_distinct_2009;@francois-brosseau_basal_2009;@kupferschmidt_parallel_2017;@corbit_corticostriatal_2017] in humans and corroborate to changes observed in animals [@yin_dynamic_2009].
Here also, the alternate production of different sequences, require shifting between overlapping set of motor commands which could further implicate the dorsal striatum in collaboration with prefrontal cortex [@monchi_functional_2006;].

While our results show that the distributed representational network during learning is reorganized during memory consolidation, the present study was not designed to investigate the nature of hippocampal, striatal or cerebellar sequence representation that were previously assessed at cortical level for finger sequences [@wiestler_effector-independent_2014;@kornysheva_human_2014] as well as for larger forearm movements [@haar_effector-invariant_2017].
Notably, the hypothesized respective extrinsic and intrinsic skill encoding in hippocampal and striatal systems remains to be assessed with dedicated experimental design.

Interestingly, our study investigated sequence representations after limited training and following sleep-dependent consolidation while previous research compared sequences intensely trained for multiple days to newly acquired ones [@wiestler_skill_2013;@lehericy_distinct_2005].
Therefore, in our study, sequence representations may further evolve, strengthen or decline with either additional training or offline memory reprocessing supported in part by sleep.

## Methodological considerations

To bound the difficulty and the duration of the task, only four sequences were practiced by the participant, two consolidated and two newly acquired, which could be a factor limiting the power of our analysis, as only a single multivariate distance is assessed for each of these conditions.
Moreover, the sequences were trained independently during longer blocks, and thus the presently used task further induces demands for instruction processing, retention in working memory, switching and inhibition that could trigger novel learning for the consolidated sequences.
We however suppose that the previously encoded motor sequence engrams are invariably retrieved during this task and as a matter of fact, is manifested by significant differences in behavioral measures and multivariate distance contrast.

The present representational analysis disregard the behavioral performance, however the chained non-linear relations between behavior, neural activity and BOLD signal were recently established to have limited influence on the representational geometry extracted from Mahalanobis cross-validated distance in primary cortex, this across a wide range of speed of repeated finger-presses and visual stimulation [@arbuckle_stability_2018].

Our results also suggest that it is possible to investigate learning-related representational changes in a shorter time-frame and with more regular training level than what was investigated before [@wiestler_skill_2013].
The use of a novel multivariate distance could have contributed to excerpt such changes, having increased sensitivity by removing classifiers' ceiling-effect and baseline shift sensitivity.

# Conclusion {#sec:conclusion}

Our study shows that the consolidation of sequential motor knowledge is supported by the reorganization of newly acquired representations within a distributed cerebral network.
We uncover that in days following learning, local activity patterns tuned to represent sequential knowledge are enhanced not only in extended cortical areas, similarly to what was shown for longer training [@wiestler_skill_2013], but also in dorsolateral striatal, thalamic and cerebellar regions.
Conversely, a small extent of the network show a decrease of sequence specific activation after consolidation, occurring specifically in dorsomedial striatum that support cognitive processing during early-learning [@doyon_current_2018] and hippocampus which carry explicit encoding of motor sequential extrinsic representation [@albouy_hippocampus_2013;@king_sleeping_2017] and play a significant role in the offline reprocessing.
These results prompt for further investigation of the dynamic of this sleep-dependent mnemonic process, which progressively reorganizes cerebral network by repeatedly reactivating the memory [@boutin_transient_2018;@fogel_reactivation_2017;@vahdat_network-wide_2017].

# Materials and methods {#sec:materials_and_methods}

## Participants

Eigtheen right-handed young ($25\pm6.2$yr.) healthy individuals (14 female) participated in the study.
They were recruited by advertising on academic and public website.
Participants were excluded if they had a history of neurological psychological or psychiatric disorders or scoring 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961], had a BMI greater than 27, smoke, had extreme chronotype, were night-workers, had traveled across meridians during the 3 previous months, or were trained as  musician or professional typist.
Their sleep quality was assessed, and individuals with score to the Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989] greater or equal to 5, or daytime sleepiness Epworth Sleepiness Scale [@johns_new_1991] score greater than 9 were excluded.

Participants included in the study were also instructed to abstain from caffeine, alcohol and nicotine, and have regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the experiment.
Their sleep schedule was assessed by analyzing the data obtained from an actigraph (Actiwatch 2, Philips Respironics, Andover, MA, USA) worn on the wrist of the non-dominant hand for the week preceding and for the duration of the experiment, and cerfified that all included subjects complied to the instructions.

All participants provided written informed consent and received financial compensation for their participation.
This study protocol was approved by the Research Ethics Board of the “Regroupement en Neuroimagerie du Québec” (RNQ).

## Procedures and tasks

The study was conducted over 3 consecutive days.
Each day, in the evening, participants performed the experimental tasks while their brain activity was recorded by an MRI scanner, using an ergonomic MRI-compatible 4-keys response pad.

TODO:do we describe CoRe?

On the first day (D1), participants were trained to perform repeatdly with their non-dominant hand (left) a 5 elements sequence (TSeq1) for 14 blocks (indicated by a green cross displayed in the center of the screen) each composed of 12 repetitions of the motor sequences (ie. 60 keypresses per block).
Participants were instructed to execute repeatedly as fast and accurate as possible the sequence of keypresses until completion of the practice block.
Practice blocks were interspersed with 25-s rest periods (indicated by the onset of a red cross on the screen) to prevent fatigue.
In case of mistake during sequence production, participants were asked to stop their performance and to immediately start practicing again from the beginning of the sequence until the end of the block.
Approximately 20 minutes after the completion of the training phase, they were administered a retention test, which consisted of a single block similar to training ones.

On the second evening (D2), participants were evaluated on the TSeq1 during a retest session (1 block), and were then trained on a new sequence (TSeq2) of 5 elements with their left-hand for 14 blocks of 12 sequences as for TSeq1.

On the third evening (D3), they first performed TSeq1 for 7 blocks followed by 7 blocks of TSeq2, each block including 12 repetitions of the sequence or 60 keypresses. 

This was followed by a task specifically designed for MVPA analysis, similar to a previous study [@wiestler_skill_2013], which alternates short practice blocks of 4 different sequences, including TSeq1 et TSeq2 as well as two new sequences NewSeq1 and NewSeq2.
It however differed in that, in our study, all 4 sequences used the left-hand 4 fingers excluding the thumb.
Also, as for the initial training, sequences were performed repeatedly and without interruption nor any feedback, this in order to probe the processes underlying automatization of the skill.

Each block was composed of an instruction period of 4 sec. when was displayed 5 numbers (eg. 1-4-2-3-1) representing in reading order the sequence of fingers to be pressed, which was followed by an execution period indicated by a green cross.
Participants had to perform 5 times the sequence, or a maximum of 25 key-presses, before being instructed to stop and rest by displaying a red cross.

Ordered assignment of sequences to blocks was chosen to include all possible successive pairs of the sequences using De Bruijn cycles [@aguirre_bruijn_2011], this preventing systematic leakage of BOLD activity between blocks of this rapid design.
A 2-length De Bruijn cycle of the 4 sequences repeats each one 4 times, yielding a total of 16 blocks.
This cycle was repeated twice in each of the 2 scanning sessions separated by approximately 5 minutes, thus resulting in a total of 64 blocks (4 groups of 16 practice blocks).

## Behavioral statistics

We entered the mean duration per block of correctly performed sequences into a linear mixed-effect model with a sequence learning stage (new/consolidated) by block (1-16) interaction to test for difference in their performance level and evolution during the task, while sequences and subjects were set as random effects.
The same model was run with the number of correct sequences as the outcome variable.
Two other models were used to separately test if there was any significant difference in performance (speed and accuracy) between the two consolidated sequences and between the two new sequences.
Full models outputs are reported in supplementary materials.

## MRI data acquisition

MRI data were acquired on a Siemens TIM Trio 3T scanner on 2 two separate sessions.
The first session used a 32-channel coil to acquire high-resolution anatomical T1 weighted saggital images using a Multi-Echo MPRAGE sequence (MEMPRAGE; voxel size=1mm isometric; TR=2530ms; TE=1.64,3.6,5.36,7.22ms; FA=7; GRAPPA=2; FoV=$256\times256\times176mm$) with the different echoes combined using a Root-Mean-Square (RMS). 

Functional data were acquired during the second session with a 12-channel coil, which allowed to fit an additional EEG cap in sleep scans, and using an EPI sequence providing complete cortical and cerebellum coverage (40 axial slices, acquire in ascending order, TR=2160ms;FoV=$220\times220\times132$mm, voxel size=$3.44\times3.44\times3.3$mm, TE=30ms, FA=90, GRAPPA=2).
Following fMRI data acquisition during task, four volume were acquired using the same EPI sequence but with reversed phase encoding to enable retrospective correction of distortions induced by B0 field inhomogeneity.

## MRI data preprocessing

High-resolution anatomical T1 weighted image was preprocessed with Freesurfer [@dale_cortical_1999;@fischl_high-resolution_1999;@fischl_cortical_2008] to segment subcortical regions, reconstruct cortical surfaces and provide inter-individual alignment of cortical folding patterns.
Pial and grey/white matter interface surfaces were downsampled to match the 32k sampling of Human Connectome Project (HCP) [@glasser_minimal_2013].
HCP subcortical atlas coordinates were warped onto individual T1 data using non-linear registration using the Ants software [@avants_symmetric_2008;@klein_evaluation_2009].

A custom pipeline was used to preprocess fMRI data prior to analysis and relied on an integrated method [@pinsard_integrated_2018] which combines slice-wise motion estimation and intensity correction followed by the extraction of BOLD timecourses in cortical and subcortical gray matter.
This interpolation concurrently removed B0 inhomogeneity induced EPI distortion estimated by FSL Topup tool using the fMRI data with reversed phase encoding [@andersson_how_2003] acquired after the task.
BOLD signal was further processed by detecting whole-brain intensity changes that corresponded to large motion, and each continuous period without such detected event was then separately detrended to remove linear signal drifts.

Importantly, the fMRI data preprocessing did not include smoothing, even though the interpolation inherent to any motion correction was based on averaging of values of neighboring voxels.
This approach was intended to minimize the blurring of data in order to preserve fine-grained patterns of activity, with the resolution of relevant patterns being hypothetically at the columnar scale.

## Multivariate Pattern Analysis

### Samples

Each block was modeled by having 2 boxcars, corresponding to the instruction and execution phases respectively, which were convolved with the Hemodynamic Response Functions.
Least-square separate (LS-S) regression of each event, which have been shown to provide improved activation patterns estimates for MVPA [@mumford_deconvolving_2012], yielded instruction and execution phases beta maps for each block that were further used as MVPA samples.

### Cross-validated multivariate distance

Similarly to @wiestler_skill_2013 and @nambu_decoding_2015, we aimed to uncover activity patterns that represent the different sequences that were performed by the participants.
However, instead of calculating cross-validated classification accuracies, we opted for a representational approach by computing multivariate distance between evoked activity patterns, in order to avoid the former's ceiling effect and baseline drift sensitivity [@walther_reliability_2016].
In the current study, we computed cross-validated Mahalanobis distance [@nili_toolbox_2014;@walther_reliability_2016;@diedrichsen_distribution_2016] which is an unbiased metric that uses multivariate normalization by estimating the covariance from the GLM fitting residuals and then regularized through Ledoit-Wolf optimal shrinkage [@ledoit_honey_2004].
This distance, that measures discriminability of conditions, was estimated separately for pairs of sequences that were in a similar acquisition stage, that is, for the newly acquired, and for the consolidated sequences.

### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that applies MVPA repeatedly on small spatial neighborhoods covering the whole brain while avoiding high-dimensional limitation of multivariate algorithms.
Searchlight was configured to select for each gray-matter coordinate their 64 closest neighbors as the subset of features for representational distance estimation.
The neighborhood was limited to coordinates in the same structure (hemisphere or region of interest), and proximity was determined using respectively euclidian and geodesic distance for subcortical and cortical coordinates.
The extent of the searchlight was thus kept to such a local range to limit the inflation of false positive or negative results [@etzel_looking_2012;@etzel_searchlight_2013].

### Statistical testing

To assess statistical significance of multivariate distance and contrasts, group-level Monte-Carlo non-parametric statistical testing using 10000 permutations was conducted on searchlight distance maps with Threshold-Free-Cluster-Enhancement (TFCE) correction [@smith_threshold-free_2009].
The statistical significance level was set at $p<.05$ (with confidence interval $\pm.0044$ for 10000 permutations) with a minimum cluster size of 25 features.
TFCE enabled a locally adaptive statistics and cluster size correction that particularly fitted our BOLD sampling of sparse gray-matter coordinates, as well as the large differences in the sizes of the structures that were investigated.

The MVPA analysis was done using the PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom samples extraction, cross-validation scheme, efficient searchlight and multivariate measure computation, optimally adapted to the study design and the anatomy-constrained data sampling.

# Acknowledgments {#sec:acknowledgments}

We thank J.Diedrichsen for methodological advice on multivariate representational analysis.

# Funding {#sec:funding}

This work was supported by the Canadian Institutes of Health Research (MOP 97830) to JD, as well as by French Education and Research Ministry and Sorbonne Universités to BP. __+Ella? +Arnaud(QBIN)__

# Supplementary materials{ label="S"}
\beginsupplement
## Behavioral linear mixed-effect model outputs

#### Test for differences in speed as mean duration to perform a correct sequence per block

```
mean_seq_duration ~ seq_new * blocks + (blocks+sequences | subjects)
==========================================================================================
Model:                     MixedLM          Dependent Variable:          mean_seq_duration
No. Observations:          1146             Method:                      REML
No. Groups:                18               Scale:                       0.0368
Min. group size:           62               Likelihood:                  165.9658
Max. group size:           64               Converged:                   Yes
Mean group size:           63.7
------------------------------------------------------------------------------------------
                                                Coef.  Std.Err.   z    P>|z| [0.025 0.975]
------------------------------------------------------------------------------------------
Intercept                                        1.269    0.076 16.790 0.000  1.121  1.417
seq_new[T.True]                                  0.365    0.047  7.776 0.000  0.273  0.457
blocks                                          -0.006    0.005 -1.304 0.192 -0.016  0.003
seq_new[T.True]:blocks                          -0.018    0.002 -7.403 0.000 -0.023 -0.013
Intercept RE                                     0.132    0.246
Intercept RE x sequences[T.NewSeq2] RE          -0.004    0.051
sequences[T.NewSeq2] RE                          0.007    0.021
Intercept RE x sequences[T.TSeq1] RE            -0.039    0.098
sequences[T.NewSeq2] RE x sequences[T.TSeq1] RE  0.001    0.024
sequences[T.TSeq1] RE                            0.025    0.056
Intercept RE x sequences[T.Tseq2] RE            -0.038    0.092
sequences[T.NewSeq2] RE x sequences[T.Tseq2] RE  0.001    0.023
sequences[T.TSeq1] RE x sequences[T.Tseq2] RE    0.023    0.049
sequences[T.Tseq2] RE                            0.022    0.048
Intercept RE x blocks RE                        -0.005    0.010
sequences[T.NewSeq2] RE x blocks RE              0.000    0.002
sequences[T.TSeq1] RE x blocks RE                0.002    0.005
sequences[T.Tseq2] RE x blocks RE                0.002    0.004
blocks RE                                        0.000    0.001
==========================================================================================
```

#### Test for differences in accuracy as the number of correct sequences over the 5 repetitions in a block

```
num_correct_seq ~ seq_new * blocks + (blocks+sequences | subjects)
==========================================================================================
Model:                       MixedLM          Dependent Variable:          num_correct_seq
No. Observations:            1152             Method:                      REML
No. Groups:                  18               Scale:                       0.6018
Min. group size:             64               Likelihood:                  -1409.7169
Max. group size:             64               Converged:                   No
Mean group size:             64.0
------------------------------------------------------------------------------------------
                                                Coef.  Std.Err.   z    P>|z| [0.025 0.975]
------------------------------------------------------------------------------------------
Intercept                                        4.691    0.079 59.215 0.000  4.536  4.846
seq_new[T.True]                                 -0.304    0.101 -3.003 0.003 -0.503 -0.106
blocks                                          -0.006    0.057 -0.101 0.919 -0.117  0.106
seq_new[T.True]:blocks                           0.014    0.010  1.434 0.152 -0.005  0.034
Intercept RE                                     0.002    0.021
Intercept RE x sequences[T.NewSeq2] RE          -0.003    0.019
sequences[T.NewSeq2] RE                          0.016    0.028
Intercept RE x sequences[T.TSeq1] RE            -0.005    0.022
sequences[T.NewSeq2] RE x sequences[T.TSeq1] RE  0.019    0.032
sequences[T.TSeq1] RE                            0.026    0.047
Intercept RE x sequences[T.Tseq2] RE            -0.004    0.025
sequences[T.NewSeq2] RE x sequences[T.Tseq2] RE  0.017    0.042
sequences[T.TSeq1] RE x sequences[T.Tseq2] RE    0.027    0.058
sequences[T.Tseq2] RE                            0.034    0.089
Intercept RE x blocks RE                        -0.001    0.021
sequences[T.NewSeq2] RE x blocks RE              0.001    0.016
sequences[T.TSeq1] RE x blocks RE                0.002    0.018
sequences[T.Tseq2] RE x blocks RE                0.002
blocks RE                                        0.038
==========================================================================================
```

#### Test for differences in speed and accuracy between the new sequences

```
mean_seq_duration ~ sequences*blocks + (1|subjects)
======================================================================
Model:               MixedLM   Dependent Variable:   mean_seq_duration
No. Observations:    571       Method:               REML
No. Groups:          18        Scale:                0.0655
Min. group size:     30        Likelihood:           -76.5056
Max. group size:     32        Converged:            Yes
Mean group size:     31.7
----------------------------------------------------------------------
                            Coef.  Std.Err.   z    P>|z| [0.025 0.975]
----------------------------------------------------------------------
Intercept                    1.630    0.071 22.931 0.000  1.490  1.769
sequences[T.NewSeq2]         0.025    0.045  0.558 0.577 -0.063  0.113
blocks                      -0.023    0.003 -7.157 0.000 -0.030 -0.017
sequences[T.NewSeq2]:blocks -0.005    0.005 -1.174 0.241 -0.015  0.004
groups RE                    0.073    0.102
======================================================================

num_correct_seq ~ sequences*blocks + (1|subjects)
======================================================================
Model:               MixedLM    Dependent Variable:    num_correct_seq
No. Observations:    571        Method:                REML
No. Groups:          18         Scale:                 0.6209
Min. group size:     30         Likelihood:            -689.3501
Max. group size:     32         Converged:             Yes
Mean group size:     31.7
----------------------------------------------------------------------
                            Coef.  Std.Err.   z    P>|z| [0.025 0.975]
----------------------------------------------------------------------
Intercept                    4.553    0.102 44.450 0.000  4.353  4.754
sequences[T.NewSeq2]        -0.245    0.138 -1.772 0.076 -0.517  0.026
blocks                      -0.007    0.010 -0.728 0.467 -0.027  0.012
sequences[T.NewSeq2]:blocks  0.028    0.014  1.936 0.053 -0.000  0.056
groups RE                    0.018    0.017
======================================================================
```

#### Test for differences in speed and accuracy between the consolidated sequences

```
mean_seq_duration ~ sequences*blocks + (1|subjects)
====================================================================
Model:               MixedLM  Dependent Variable:  mean_seq_duration
No. Observations:    575      Method:              REML
No. Groups:          18       Scale:               0.0222
Min. group size:     31       Likelihood:          226.1710
Max. group size:     32       Converged:           Yes
Mean group size:     31.9
--------------------------------------------------------------------
                          Coef.  Std.Err.   z    P>|z| [0.025 0.975]
--------------------------------------------------------------------
Intercept                  1.256    0.057 21.949 0.000  1.144  1.368
sequences[T.TSeq2]         0.031    0.026  1.191 0.234 -0.020  0.082
blocks                    -0.008    0.002 -4.023 0.000 -0.011 -0.004
sequences[T.TSeq2]:blocks -0.000    0.003 -0.165 0.869 -0.006  0.005
groups RE                  0.053    0.125
====================================================================

num_correct_seq ~ sequences*blocks + (1|subjects)
====================================================================
Model:               MixedLM   Dependent Variable:   num_correct_seq
No. Observations:    575       Method:               REML
No. Groups:          18        Scale:                0.4050
Min. group size:     31        Likelihood:           -569.8356
Max. group size:     32        Converged:            Yes
Mean group size:     31.9
--------------------------------------------------------------------
                          Coef.  Std.Err.   z    P>|z| [0.025 0.975]
--------------------------------------------------------------------
Intercept                  4.694    0.081 58.093 0.000  4.535  4.852
sequences[T.TSeq2]        -0.030    0.111 -0.267 0.789 -0.248  0.188
blocks                    -0.012    0.008 -1.414 0.157 -0.028  0.004
sequences[T.TSeq2]:blocks  0.014    0.012  1.207 0.228 -0.009  0.036
groups RE                  0.006    0.010
====================================================================
```


## Representational distance maps

![Group searchlight map of cross-validated Mahalanobis distance between the two new sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_crossnobis_tfce_map2.pdf){#fig:new_crossnobis_map}

![Group searchlight map of cross-validated Mahalanobis distance between the two consolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/cons_crossnobis_tfce_map2.pdf){#fig:cons_crossnobis_map width=20cm}

# References
