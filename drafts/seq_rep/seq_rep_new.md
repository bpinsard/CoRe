---
tags: [example, citation]
bibliography: [abstract_rbiq.bib]
title: Evolution of a distributed motor memory trace from learning to consolidation.
author:
- family: Pinsard
  given: Basile
  affiliation: 1,2
  email: basile.pinsard@gmail.com
- family: Boutin
  given: Arnaud
  affiliation: 2,3
- family: Gabitov
  given: Ella
  affiliation: 2,3
- family: Benali
  given: Habib
  affiliation: 1,5
- family: Doyon
  given: Julien
  affiliation: 2,3,4
institute: here
organization:
- id: 1
  name: Sorbonne Universités, UPMC Univ Paris 06, CNRS, INSERM, Laboratoire d’Imagerie Biomédicale (LIB) 
  address: 75013, Paris, France
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
date: " "
chapters: True
chaptersDepth: 1
chapDelim: ""
---
## Introduction

Animals and humans are able to acquire and perfect sequences of movements, expanding and updating their complex and evolving goal-oriented repertoire of motor actions.
Numerous behavioral experiments studied motor sequence learning (MSL), as the acquisition temporally ordered coordinated movements, either implicitly or explicitly instructed, thereby improving the knowledge of the mechanisms underlying this type of procedural memory [@abrahamse_control_2013;@diedrichsen_motor_2015;@verwey_cognitive_2015;@doyon_current_2018].

While practice of an explicit MSL task leads to substantial within-session improvements in performance, evidence showing additional offline performance gains and reprocessing of the memory trace associated with this type of learning has recently accumulated [see @king_sleeping_2017 for a recent in-depth review].
Indeed, the reprocessing of the motor memory trace during subsequent sleep periods yields gains, or at least maintenance of performance as compared to the decay observed following a waking period of equal duration [@nettersheim_role_2015].
However, controversy remains regarding the respective roles that both wake and sleep play in such memory consolidation [@brawn_consolidating_2010;@nettersheim_role_2015;@landry_effects_2016], and calls for the investigation of underlying dynamic changes in neural substrates supporting the production of motor sequences, including across offline periods.

Sequential motor skills recruit an extended network of cerebral [@hardwick_quantitative_2013], cerebellar and spinal regions [@vahdat_simultaneous_2015], which activities differentiate across learning stages [@dayan_neuroplasticity_2011;@doyon_current_2018].
Herein, critical plastic changes [@ungerleider_imaging_2002;@doyon_reorganization_2005] occur during both training and post-learning consolidation stages, the latter being modeled as a "reorganization" between nervous system structures supporting such function [@rasch_reactivation_2008;@born_system_2012;@albouy_hippocampus_2013;@dudai_consolidation_2015;@bassett_learning-induced_2015;@fogel_reactivation_2017;@vahdat_network-wide_2017].
Primary sensorimotor and supplementary motor cortices as well as posterior parietal and dorso-lateral prefrontal cortices [@hardwick_quantitative_2013] are parts of cortico-cerebellar (CC) and cortico-striatal (CS) loops which undergo such reorganization in the course of MSL [@doyon_reorganization_2005;@doyon_current_2018].

More specifically, cerebellar cortices involved early-on in MSL [@diedrichsen_cerebellar_2014] includes Lobule V which contribute to the integration of sensory and motor ipsilateral finger representation [@wiestler_integration_2011], Lobule VI bilaterally [@schlerf_evidence_2010] and Crus I in Lobule VII [@doyon_contributions_2009] have been shown to support complex movements, coordination, sequencing and spatial processing.
They compose the MSL-recruited CC loop, being respectively connected functionally to primary, premotor and supplementary motor cerebral cortices [@buckner_organization_2011].
Similarly to their neocortical counterpart, Lobule V/VI exhibit a somatotopicaly organized representation of the whole body.
The cerebellum, known to be critically involved in motor adaptation [@doyon_distinct_2003;@diedrichsen_cerebellar_2014], thus also contributes to MSL, building an internal model for performance optimization, controlling movements and correcting errors [@penhune_parallel_2012].
However, as these are less required with extended practice, cerebellar cortex activation lowers and shifts to dentate nuclei [@doyon_experience-dependent_2002], orthogonally to the increased recruitment of cortical-striatal loop.

Indeed, the striatum has been associated with the selection, preparation and execution of free or instructed movements [@hikosaka_central_2002;@gerardin_distinct_2004;@doyon_contributions_2009], as well as automation of sequences processing through reinforcement learning [@jin_basal_2014].
This notably reflects in the increase of caudate nuclei activity with lower execution variability [@albouy_neural_2012] and the putamen being further activated during and after consolidation of a motor memory trace [@debas_brain_2010;@albouy_hippocampus_2013;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
With extended training over multiple weeks, a progressive shift from associative and premotor to sensorimotor basal ganglia distributed network, notably the putamen, concurs to the automation of the skill [@lehericy_distinct_2005].

The hippocampus, mainly known to encode episodic memories [@battaglia_hippocampus:_2011] including spatial and sequences of events [@howard_time_2015;@moser_place_2015], has also been shown to support procedural memory acquisition in its early stages [@albouy_both_2008]; its activity and interaction with striatum predicting long-term retention of motor sequences [@albouy_interaction_2013].
Hence, the hippocampus contribution to procedural memory is thought to favor the building and storage of an effector-independent and allocentric (visual-spatial) representation [@albouy_maintaining_2015] and to tag memories temporarily for further reactivation during succeeding offline consolidation [@dudai_consolidation_2015].

All the aforementioned structures thus undergo non-linear changes in functional activity level during learning [@dayan_neuroplasticity_2011;@diedrichsen_motor_2015], and notably after the memory trace has been consolidated [@lehericy_distinct_2005;@debas_brain_2010;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
However, while increasing activity can suggest a heightened implication of specialized circuits, decreasing activity could either reflect optimization and efficiency gains of these same circuits or lowered recruitment of non-specific support networks.
Therefore, the observed large-scale activation changes described above do not provide evidence that these regions show plasticity for motor sequence specific representation [@berlot_search_2018].

To address this issue, investigators have recently been employing Multivariate Pattern Analysis (MVPA) to identify sequences specific learning representation changes using a set of techniques recently adapted to neuroimaging [@pereira_machine_2009].
The latter evaluate how local patterns of activity are able to discriminate between stimuli or memories of the same type.
In the MSL literature, few studies have used such MVPA approach to localize the representation of motor sequences with various design aimed at specific identification of whole sequence [@wiestler_skill_2013] or specific features [@wiestler_integration_2011;@kornysheva_human_2014;@yokoi_does_2017].
For instance, in a recent study focusing on motor-related cerebral cortices [@wiestler_skill_2013], classifier performance was evaluated on trained and untrained sets of sequences.
Results showed a potential increase of the representation strength with training in a network spanning bilaterally the primary motor, pre-motor, supplementary motor areas and parietal cortices.
By analyzing the dynamics of classifier decoding performance the authors also uncovered that significant motor sequence representation precedes the peak of the BOLD response in different cortical areas during instruction and execution.
Indeed, prior to motor sequence execution, preparation activates a network that overlaps with the one activated during movements per se [@lee_subregions_1999;@zang_functional_2003;@orban_richness_2015].
In a second study, Nambu et. al @nambu_decoding_2015 specifically analyzed patterns of preparatory activity, after instruction and prior to execution of extensively trained sequences, this in order to remove confounds from execution or sensory feedback.
During this preparatory period, sequence representations were localized in contralateral dorsal premotor and supplementary motor cortices only, while during execution these were detected in a larger bilateral network comprising premotor, somatosensory and posterior parietal cortices as well as cerebellum, but surprisingly not in basal ganglia.

Nevertheless, numerous neuronal level [@alexander_parallel_1986;@graybiel_building_1995;@miyachi_differential_2002] and neuroimaging studies [@hikosaka_central_2002;@albouy_both_2008;@orban_multifaceted_2010;@penhune_parallel_2012;@albouy_neural_2012] have corroborated the implication of the basal ganglia, in particular the putamen, as well as the hippocampus, in the acquisition and consolidation of sequential motor skills.
However, the genuine acquisition of motor sequence-specific representation in such MSL-involved regions still needs to be assessed in humans.
The goal of the present study is thus to identify newly acquired motor sequences representation distributed across cortical aand subcortical networks and their reorganization following sleep-dependent consolidation.
Hence, and as assessed by multivariate statistics in local subcortico-cortical activity, we hypothesized that strengthened motor-related cortical and striatal representations, in conjunction with weaker hippocampal-based representations, would reflect the reorganization of the memory trace during consolidation.

##Method

### Participants

The study includes 18 right-handed young (18-35 years old) healthy volunteers recruited by advertising on scholar and public website.

The subject were not included in case of history of neurological psychological or psychiatric disorders or scoring 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961].

Subjects with BMI greater than 27, smokers, extreme chronotype, night-workers, having traveled across meridian during the 3 previous months, or training as musician or professional typist (for over-training on coordinated finger movements) were also excluded.

Sleep quality was assessed by Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989], and daytime sleepiness  (Epworth Sleepiness Scale [@johns_new_1991]) had to be lower or equal to 9.

Subject were also instructed to abstain from caffeine, alcohol and nicotine, and have regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the study.
Instruction compliance was controlled by non-dominant hand wrist actigraphy (Actiwatch 2, Philips Respironics, Andover, MA, USA) for the week preceding and the duration of the experiment.

This study protocol was approved by the Research Ethics Board of the “Regroupement en Neuroimagerie du Québec” (RNQ).
All subjects provided written informed consent and received financial compensation for their participation.

### Behavioral experiment

The experiment was conducted over 3 consecutive days, at the end of the day, with the motor tasks performed in the scanner using an ergonomic MRI-compatible 4-keys response pad.

On the first evening (D1), subjects were trained to perform with their non-dominant left-hand a 5 elements sequence (TSeq) for 14 blocks (indicated by a green cross displayed in the center of the screen) each composed of 12 repetitions of the motor sequences (ie. 60 keypresses per block).
Subjects were instructed to execute repeatedly as fast and accurate as possible the sequence of keypresses until completion of the practice block.
Practice blocks were interspersed with 25-s rest periods (indicated by the onset of a red cross on the screen) to prevent fatigue.
In case of mistake during sequence production, subjects were asked to stop their performance and to immediately start practicing again from the beginning of the sequence until the end of the block.
Approximately 20 minutes after the completion of the training phase, subjects were administered a retention test, which consisted of a single block similar to training ones.

On the second evening (D2), subjects were evaluated on the TSeq during a retest session (1 block), and were then trained on an interfering sequences (IntSeq) of 5 elements with their left-hand for 14 blocks of 12 sequences as for TSeq.

On the third evening (D3), subjects first performed TSeq for 7 blocks followed by 7 blocks of IntSeq, each block including 12 repetitions of the sequence or 60 keypresses. 

This was followed by a task specifically designed for MVPA analysis, similar to a previous study [@wiestler_skill_2013], which alternates short practice blocks of 4 different sequences, including TSeq et IntSeq as well as two new sequences.
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
However, instead of applying cross-validated classification, we opted for a representational approach by computing multivariate distance between evoked activity patterns, in order to avoid the former's ceiling effect and noise sensitivity [@walther_reliability_2016].
Cross-validated Mahalanobis distance [@walther_reliability_2016] is an unbiased metric that uses multivariate normalization by estimating the covariance from the GLM fitting residuals, regularized through Ledoit-Wolf optimal shrinkage [@ledoit_honey_2004].
Distance were estimated for pairs of sequences that were in a comparable acquisition stage, that is separately betweem the newly acquired and between consolidated sequences.

#### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that applies MVPA repeatedly on small spatial neighborhoods covering the whole brain while avoiding high-dimensional limitation of multivariate algorithms.
Searchlight was configured to select for each grayordinate the 128 closest neighboring coordinates, using geodesic distance for cortical grayordinates, as the subset of features for representational distance estimation.
The extent of the searchlight was thus kept to a limited range to limit the inflation of false positive or negative results [@etzel_looking_2012;@etzel_searchlight_2013].

#### Statistical testing

To assess statistical significance of multivariate distance and contrasts, group-level Monte-Carlo non-parametric statistical testing using 10000 permutations was conducted on searchlight distance maps with Threshold-Free-Cluster-Enhancement (TFCE) correction and thresholded at $p<.05$ (with confidence interval $\pm.0044$ for 10000 permutations) with a minimum cluster size of 25 features.
TFCE enabled a locally adaptive statistics and cluster size correction that particularly fitted our non-regular BOLD sampling in grayordinates, as well as the different sizes of the structures that we investigated.

The MVPA analysis was done using the PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom samples extraction, cross-validation scheme, Searchlight and measures to adapt to the study design and data.

## Results

To behaviorally measure consolidation, subjects (n=18) were trained to perform two finger presses sequences, each practiced separately on two successive days, this in order to avoid their consolidation to interfere.
On the third day the subjects were retested on both sequences separately, and then performed a task (MVPA-task) designed to investigate neuronal representation from fMRI BOLD activity pattern.
By practising the two consolidated sequences as well as two new sequences during interleavedly ordered blocks, we were able to extract patterns of activity during the execution.

### MVPA-task behavioral performance

The consolidated sequences (TSeq-IntSeq) showed no difference in average sequence duration ($t(17)=-1.89, p=0.07$) and number of correct sequences ($t(17)=-1.38, p=0.18$).
Similarly, the newly acquired sequences (NewSeq1-NewSeq2) were not found different in term of average sequence duration ($t(17)=0.82, pvalue=0.42$) and number of correct sequences ($t=0.55, p=0.58$).
As expected, the consolidated sequences significantly differ from the newly learned ones in both sequence duration ($t(17)=-5.60, p=0.00003$) and number of correctly performed sequence ($t(17)=2.86, p=0.01$).
The difference in execution speed were observed until the last block ($t(17)=-2.69, p=0.02$) showing a persistent benefit of previous training and consolidation as compared to newly trained sequences of similar difficulty.

The newly trained sequences indeed showed a learning curve (@fig:mvpa_task_groupInt_seq_duration), speed increasing from first to last block ($t(17)=-3.78, p=0.001$) while accuracy did not significantly improve ($t(17)=0.97, p=0.34$) likely due to the number of correct sequence being discrete and bounded (ie. 0 to 5) suffers from ceiling effect.
The consolidated sequences speed also significantly improved during this new task ($t(17)=-2.49, p=0.02$), potentially combining improvements in sequence performance or more general knowledge in the novel task.

![Average and standard deviation of sequence duration across the MVPA task blocks.](../../results/behavior/mvpa_task_groupInt_seq_duration.pdf){#fig:mvpa_task_groupInt_seq_duration}
        
### A common distributed sequence representation for consolidated and new sequences

In order to measure sequences representation, we used cross-validated Mahalanobis distance [].
This unbiased measure of stable pattern difference avoids the ceiling effect of previously used classification-based measures [@walther_reliability_2016].
For multivariate approaches not to only reflect global activity levels differences, the conditions needs to be similar.
Thus we separately computed cross-validated Mahalanobis distance between consolidated sequences and between new sequences.
We explored local representation across the whole brain using a Searchlight approach [], each gray matter location being assigned the distance between the evoked activity pattern from its neighborhood, yielding two maps of multivariate distance per subject for the two stages of learning.


This unbiased measure, when consistently larger than zero, assesses that the pattern of activity significantly differs between conditions imaged.
Therefore, we tested for significant difference from zero at the group level using sign-flipping Monte-Carlo non-parametric testing with TFCE.
To excerpt the network that discriminate sequences at both stages of learning, we then submitted these results to a minimum-statistic conjunction.
It reveals a large network (@fig:new_cons_conj_crossnobis_tfce_map) with differentiated patterns of activity, including primary visual cortex that processes the visual instructions, as well as posterior parietal, primary and supplementary motor, premotor and dorsolateral prefrontal cortices.
When looking at separate results, subcortical regions also show differing activity patterns, including ipsilateral cerebellum, bilateral thalamus, hippocampus and striatum (@fig:new_crossnobis_tfce_map,@fig:cons_crossnobis_tfce_map).

![Group searchlight conjunction map cross-validated Mahalanobis distance within new and consolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_cons_conj_crossnobis_tfce_map.pdf){#fig:new_cons_conj_crossnobis_tfce_map}

### Reorganization of the distributed sequence representation with consolidation

In order to evaluate the changes in sequence representation during consolidation, we submitted searchlight pattern distance maps from all subjects to pairwise t-test between consolidated and new sequences, assessing signficance by permutation testing with TFCE (@fig:contrast_cons-new_crossnobis_tfce_map).
Discriminability is found to be significantly higher in bilateral putamen, contralateral caudate nuclei, thalamus, ventral and dorsal premotor, supplementary motor and dorsolateral prefrontal cortices for consolidated sequences.
Conversely, the representation strength decreases for consolidated sequences in bilateral hippocampus and ipsilateral body of the caudate nuclei.

![Group searchlight contrasts of cross-validated Mahalanobis distance between consolidated and unconsolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/contrast_cons_new_crossnobis_tfce_map.pdf){#fig:contrast_cons-new_crossnobis_tfce_map}

While patterns differentiating newly acquired sequences exists in contralateral putamen and bilateral caudate, these distances increase for consolidated sequences in bilateral putamen.

## Discussion


In the present study we aimed to measure the changes in activity patterns in a distributed network supporting the execution of multiple motor sequential skills associated with behaviorally assessed consolidation.
Locally stable patterns of activity are here used as a proxy for the specialization of neuronal circuits that support the efficient retrieval and expression of memories.

### Representation of finger motor sequence

Our findings provide evidence that an extended network of regions shows reliable sequence-specific activity patterns for both consolidated and novel sequences.
Cortically, a previously described network [@wiestler_skill_2013;@nambu_decoding_2015] encompasses bilateral supplementary and pre-motor areas, as well as posterior parietal cortices, while contralateral primary sensorimotor regions were only shown to elicit different patterns for novel sequences.
It is noteworthy that discrimination of motor sequence representations within the ipsilateral motor, premotor and parietal cortices has been previously described [@wiestler_skill_2013;@waters_metenier_bihemispheric_2014;@waters_cooperation_2017], notably when non-dominant hand is used for such skills.

Difference in activity patterns within the primary motor cortex was only found for newly learned sequences [@fig:new_crossnobis_tfce_map], which could reflect their motoric differences in terms of finger presses ordering.
Unbeknownst to us during experimental design, the first finger press was recently shown to elicit higher activation, thus driving separability of patterns for sequences with different initiating finger [@yokoi_does_2017].
However, the primary motor representation was found to be relatively weak in our study in comparison to [@wiestler_skill_2013], likely explained by the uninterrupted repetition of the motor sequences during the practice singling this effect to the beginning of the block, as well as our 5-element sequences not engaging the thumb which distinctive M1 pattern would have elicited stronger difference if initiating the sequence [@ejaz_hand_2015].

The conjunction map reveals that a common cortical processing stream including non-motor support regions present sequential information from visually presented instruction to motor sequence production.
Herein, occipital cortex, as well as ventro-temporal regions are found to discriminate the sequences [@fig:new_cons_conj_crossnobis_tfce_map], but likely reflect the processing of the visual stimuli respectively as low-level visual mapping [@pilgramm_motor_2016] and higher level Arabic number representation [@shum_brain_2013;@peters_neural_2015], and thus do not differ between the stages of learning studied here [@fig:contrast_cons-new_crossnobis_tfce_map].
The dorsolateral prefrontal cortex (DLPFC) also exhibit pattern specificity, and was previously reported to be reflect increased attentional demand during slow learning phase [].
In fact, the cognitive processing required by the MVPA task, implying notably to switch between sequences and to inhibit interfering ones, could magnify this frontal associative representation.

### Representational changes underlying memory consolidation

We then investigated how representations are restructured after early consolidation of MSL by contrasting maps of multivariate distance for consolidated and newly acquired sequences.
At the cortical level, we found that contralateral premotor and bilateral parietal regions acquire a stronger representation during consolidation, that likely reflects that the tuning of these neural populations to coordinated movements are consolidated early after learning [@makino_transformation_2017;@yokoi_does_2017;@pilgramm_motor_2016].

Further investigating such changes at subcortical level, significant differences are found in bilateral putamen and more specifically ventral posterior regions, which specifies the previous report of their increased activation after consolidation [@debas_brain_2010;@albouy_hippocampus_2013;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
Significant representational changes are also found in cerebellum ipsilateral lobule IX as well as contralateral Crus I and II in lobule VII. (TODO: explanation?)
Concurrently to this consolidation induced representational emergence, only few regions showed decreased sequence discrimination, namely ipsilateral caudate nuclei and bilateral hippocampus.
The hippocampal early representation have been hypothesized to buffer novel explicit motor sequence learning and concur to the reactivations of the distributed network for reprocessing during offline periods, though progressively disengaging afterward [@albouy_hippocampus_2013].
Our novel findings of differential implication of dorsomedial and dorsolateral striatum in sequence representation during learning and expression of a mastered skill specifies their previously reported activity change in the course of MSL [@lehericy_distinct_2005;@jankowski_distinct_2009;@kupferschmidt_parallel_2017;@corbit_corticostriatal_2017].

While our results show that the distributed representational network during learning is reorganized during memory consolidation, the present study was not designed to investigate which  of hippocampal, striatal or cerebellar sequence representation that were previously assessed at cortical level for finger sequences [@wiestler_effector-independent_2014;@kornysheva_human_2014] as well as for larger forearm movements [@haar_effector-invariant_2017].
Notably, the respective extrinsic and intrinsic skill encoding in hippocampal and striatal systems remains to be assessed.

Interestingly, our study investigated sequence representations after limited training and following sleep-dependent consolidation while previous research compared sequences intensely trained for multiple days to newly acquired ones [@wiestler_skill_2013]. Therefore, in our study, sequence representations may further evolve, strengthen or decline with either additional training or offline memory reprocessing supported in part by sleep.

To limit the difficulty and the duration of the task, only four sequences were practiced by the subjects, two consolidated and two newly acquired, which could be a factor limiting the power of our analysis, as only a single multivariate distance is assessed for each of these conditions.
Moreover, the sequences were trained independently during longer blocks, and thus the present task further induces demands for instruction processing, switching and inhibition that could trigger novel learning for the consolidated sequences.
We however suppose that the previously encoded motor sequence engrams are invariably retrieved during this task and is reflected by significant differences in behavioral measures and multivariate distance contrast.

##Conclusion

Our study shows that memory consolidation is supported by the reorganization of newly acquired sequential representations within a distributed cerebral network.
epresentation  not only in extended cortical areas, similarly to what was shown for longer training [@wiestler_skill_2013], but also in dorsolateral striatal, thalamic and cerebellar regions.
Conversely, a small extent of the network show a decrease of sequence specific activation after consolidation, occuring specifically in dorsomedial striatum that support cognitive processing during early-learning [@doyon_current_2018] and hippocampus which carry explicit encoding of motor sequential extrinsic representation [@albouy_hippocampus_2013;@king_sleeping_2017] and play a significant role in the offline reactivation.
These results prompt for further investigation of the dynamic of this sleep-dependent mnemonic process, which progressively reorganizes cerebral network by repeatedly reactivating the memory.

# Acknowledgment

We thank J.Diedrichsen for methodological advices on multivariate representational analysis.

# Funding

This work was supported by the Canadian Institutes of Health Research (MOP 97830) to JD, as well as by French Education and Research Ministry and Sorbonne Universités to BP. __+Ella? +Arnaud(QBIN)__

# Supplementary {#sec:suppl label="S"}

![Group searchlight map of cross-validated Mahalanobis distance between the 2 new unconsolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_crossnobis_tfce_map.pdf){#fig:new_crossnobis_tfce_map}

![Group searchlight map of cross-validated Mahalanobis distance between the 2 consolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/cons_crossnobis_tfce_map.pdf){#fig:cons_crossnobis_tfce_map}

# References
