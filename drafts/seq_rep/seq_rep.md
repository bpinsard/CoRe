
## Introduction

Motor skills recruits an extended network of cerebral and spinal regions, which involvement evolves differently across learning stages [@dayan_neuroplasticity_2011].
Motor sequence learning (MSL) is a adequate paradigm [@] to study neural processes involved from initial learning to automation of a skill such as a temporally ordered succession of coordinated movements.
While MSL aggregates cognitive abilities to improve performance of execution, the local or distributed network specifically encoding different features of the sequence is only partialy known.

Cortical networks are critical to sequence learning for which large-scale network is activated [@dayan_neuroplasticity_2011] including primary and supplementary motor cortices as well as posterior parietal and dorso-lateral prefrontal cortices.
The execution implicates primary motor cortex which is topologically structured to encode both single and multi-joint finger movements [@diedrichsen_two_2013;@ejaz_hand_2015].
While motor execution network is strongly lateralized to the cortex contralateral to the effector, ipsilateral activity and finger specific neural patterns are also observed [@diedrichsen_two_2013] that might enable bi-manual coordination.

At the subcortical level, hippocampus and striatum [@albouy_hippocampus_2013] were also identified as important players in the learning and consolidation of motor sequences.
Hippocampus is a major memory structure that not only encodes episodic memory but also supports procedural memory acquisition in it's early stages [@ref].
On the other hand, the striatum participates in automation of sequences processing through reinforcement learning [@jin_basal_2014;@graybiel_striatum:_2015], but is not limited to procedural motor skills.
Optimality of motor production is progressively attained through local optimization of transition often achieved through chunking of sub-sequences [@rosenbaum_hierarchical_1983;@diedrichsen_motor_2015], which variations are explored during learning process and depends on learning strategy [@lungu_striatal_2014].
The generality of such structures for learning sequences could also support the transfer between modality [@mosha_unstable_2016].

Sleep-supported consolidation restructures the trace of memory, alleviating the implication of hippocampus, while striatum enables the skill to become automated, requiring lower cognitive load, and long-term storage of behavior.
Congruent with models of consolidation [@born_system_2012], hippocampus acts as a buffer for recent memories, delaying selective transfer to cortical or specialized structures during offline period, enabling balanced plasticity versus stability, necessary for system homeostasis.

In the light of memory consolidation theory, large-scale motor network activity exhibit changes across days after learning which are supported by offline processing notably during sleep [@born_system_2012]. 
Active System Consolidation model hypothesize that relative implication of subparts of the network is modified by consolidation, sometimes called a "transfer" but is rather a rebalancing.
However while motor practice is less variable with automation of the skill, subparts of the networks stabilize [@costa_differential_2004,@peters_emergence_2014] as well as evoked BOLD fMRI activity patterns [@wiestler_skill_2013].
Premotor patterns sub-components can both encode spatial and temporal features [@kornysheva_human_2014], the former characterizing the learned sequences.

The goal of our study is to provide further insight in brain representation of motor sequence with Multivariate Pattern Analysis (MVPA).
We extend previous research [@wiestler_skill_2013;@kornysheva_human_2014] by scanning larger extent of human brain to measures sub-cortical activity including cerebellum, basal ganglia and lower temporal cortex.
We also intend to measure changes induced by learning and consolidation by comparing MVPA performances between sequences at their early stages of these processes.

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

1. In the first evening (D1), subjects were trained to perform with left-hand a 5 elements sequence (TSeq) for 14 blocks of 12 sequences or a maximum of 12x5=60 keypresses.
Subject were instructed to execute repeatedly as fast and as accurate as possible the sequence of keypresses and to start from the beginning of the sequences in case they noticed that they did an error.
They were retested approximately 20 minutes later for an additional single block of 12 sequences.

2. On the second evening (D2), subjects were tested for 1 block on TSeq.
Then half of the subjects of "Interference" (Int) group were trained on an interfering sequences (IntSeq) of 5 elements with left-hand for 14 blocks of 12 sequences as for TSeq.
"No-Interference" (NoInt) group had scanned resting-state which duration was yoked to a IntGroup subject.

3. On the last evening (D3) subjects first practised TSeq for 7 blocks x 12 repetitions for retest, then practised IntSeq for 7 blocks of 12 sequences, which was followed by a task specifically designed for MVPA analysis, that will be called "MVPA task" thereafter.
This task was similar to [@wiestler_skill_2013], which rapidly alternates short blocks of practice of 4 different sequences.
Significant differences are that 4 sequences were performed with the left hand four fingers excluding the thumb and no feedback was given regarding the correctness of performance and sequence was performed uninterruptedly.
Each block, composed of an instruction period of 4 sec when 5 numbers (eg. 1-4-2-3-1) representing in reading order the sequence of finger to be pressed, was displayed followed by an execution period indicated by a green cross.
Subject had to perform 5 times the sequence, or a maximum of 5x5=25 key-presses before being instructed to stop and rest when a red cross was displayed.
Rest duration was variable and set to synchronize the beginning of each block with the same slice in the volume being acquired.

Ordering of the sequences in blocks was chosen to include all possible successive pairs of the sequences using De Bruijn cycles [@aguirre_bruijn_2011] allowing unbiased analysis of dynamics described below. Given 4 sequences, a 2-length De Bruijn cycle would contains 16 blocks, repeated twice to give 8 repetitions of each of the 4 sequences which amounts to 32 blocks.

Each subject performed the task twice in scans separated by few minutes to allow rest and enable study of cross-scans pattern stability using classification generalization.

### Scan acquisition

MRI data were acquired on a Siemens Trio 3T scanner on 2 two separate sessions.
The first session used 32-channel coil to acquire high-resolution anatomical T1 weighted image using Multi-Echo (4) MPRAGE (MEMPRAGE) at 1mm isotropic resolution. 

Functional data were acquired during the second session with 12-channel coil for comparison with other dataset. EPI sequence consists of 40 slices with 3.44mm in-plane resolution and 3.3mm slice thickness to provide full cortical and cerebellum coverage.
Consecutively fieldmap was obtained to measure B0 field inhomogeneity to allow retrospective compensation of induced distortions.

### Preprocessing

Custom pipeline was used to preprocess fMRI data prior to analysis.
First, high-resolution anatomical T1 weighted image was preprocessed with Freesurfer [@dale_cortical_1999;@fischl_high-resolution_1999;@fischl_cortical_2008] to segment subcortical regions, reconstruct cortical surfaces and provide inter-subjects alignment of cortical folding patterns. Pial and grey/white matter interface surfaces were downsampled to match the 32k sampling of Human Connectome Project (HCP) [@glasser_minimal_2013] and we averaged pial and white surface to get coordinates at the half of the thickness of cortical sheet. HCP subcortical rois coordinates were warped onto individual T1 data using non-linear registration based on Ants software [@avants_symmetric_2008;@klein_evaluation_2009]. Combination of cortical and subcortical coordinates then corresponds to grayordinates of HCP datasets [@glasser_minimal_2013].

fMRI data underwent estimation of subject motion [@roche_four-dimensional_2011] and coregistration to T1. Registration and motion parameters were used to interpolate Blood-Oxygen-Level-Dependent (BOLD) signal at anatomical grayordinates above-mentioned taking into account B0 inhomogeneity induced distortions using fieldmap acquisition.

BOLD signal was further processed to remove drifts and abrupt motion-related signal change.

Of note is that our preprocessing does not includes smoothing, even though interpolation inherent to any motion correction causes averaging of values of neighboring voxels. We intended to minimize blurring of data to preserve fine-grained patterns of activity, resolution of relevant patterns being hypothetically at columnar scale.

### Multivariate Pattern Analysis

Similarly to [@wiestler_skill_2013] we aim to uncover activity patterns to predict the sequence being produced.
We also aim to analyze the classification of trained sequences versus untrained one, however TSeq and IntSeq are probably still undergoing consolidation by the third day of the study, potentially lowering their pattern stability.

For this reason, we first aimed at classifying untrained sequences which are comparable as being completely new to the subjects, and are learned along the course of the "MVPA task". This 2-class scheme allows mapping representation of non-consolidated sequence.

Then TSeq and IntSeq are also analyzed separately in IntGroup mapping sequences consolidated on two separate nights of sleep.

The MVPA analysis was based on PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom cross-validation scheme, Searchlight and measures to adapt to the study design and analyses.

#### Samples

Each blocks was modelled by having 2 boxcars, respectively instruction and execution phase, convolved with Hemodynamic Response Functions (HRF). Volumes (TRs) corresponding to HRF level above 50% of maximum response level were taken as samples for the performed sequence. Maximum value of instruction and execution regressors determine the TR to pertain to instruction or execution phase, for which HRF is overlapping.
A TR based approach was chosen to explore the fine dynamic of patterns related to the task, that model driven such as GLM cannot fully analyze at the cost of lower signal-to-noise ratio.

Regular GLM-based approach was also performed using least-square separate (LS-S) regression of each event [@mumford_deconvolving_2012] shown to provide improved activation patterns estimates for MVPA. For each blocks, regressors for instruction and execution phases provided t-value maps that was further used as MVPA samples.

#### Cross-validation

The De Bruijn cycles ordering of the sequence in the task aims at providing unbiased cross-validation by balancing the temporal succession of any pair of the 4 sequences. 

Chosen cross-validation schema includes:

- Leave-One-Chunk-Out (LOCO): each block is successively taken out of the dataset to be used in prediction. Classifier is trained on remaining data by randomly selecting balanced number of samples of the 4 sequences which are further than 60 secondes to the block of test data. Random selection of balanced data is performed 5 times for each of the 64 blocks amounting to 64*5 = 320 folds of cross-validation.
- Leave-One-Scan-Out (LOSO): random balanced subset of samples from a scan is fed for training to the classifier which then predicts one the other scan the sequences. Random subset was selected 5 times for the 2 scans giving 10 cross-validation folds.

#### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that apply MVPA techniques repeatedly on small spatial neighborhood with the purpose to localize representation of information of interest across brain while avoiding high-dimensional limitation of multivariate algorithms.

Applying cross-validation using the Searchlight schema allowed to extract brain-wise map of classifier performance giving information of regions having stable sequences related patterns.
Gaussian Naive Bayes (GNB) linear classifier, optimized for Searchlight, was performed with the 2 proposed cross-validation schema analysis on the execution labelled TRs.
Also GNB-based Searchlight have been argued to allow smoothness of generated maps [@raizada_smoothness_2013] despite unsmoothed data, allowing more-reliable cross-subject study and thus higher cross-scans generalization.

Searchlight was configured to select for each grayordinate the 64 closest neighboring coordinates, using surface distance for cortical grayordinates as the subset of features to be classified.
Searchlight size has been shown to inflate the extent of significant clusters in searchlight analysis [@etzel_searchlight_2013; @viswanathan_geometric_2012] which motivated the small neighborhood for our analysis.

Cross-validation confusion matrix was computed for each block of practice providing a more complete representation of classification performance and biases from which can be derived specific or global accuracy percentage.

For both LOCO and LOSO cross-validation schema, we measured following accuracy searchlights maps:

- global accuracy : the percentage of samples for which the correct sequence is detected,
- sequence specific sensitivity : for each sequence, TP/(TP+FN) with TP: true positives, FN: false negatives.

#### Searchlight Group Analysis

Group searchlight maps were computed using mass-univariate one-sample T-test to find regions which consistently departed from chance level across subjects.

To further investigate the change in representation strenght and localization between trained and untrained sequences, we contrasted their accuracy maps in Int group only using a subject pair t-test. Then a conjunction with significant trained sequence accuracy using minimum t-value was computed to only extract representation enhancement reaching above chance level.

#### Dynamics analysis

Also we complemented the analysis by taking TRs subsamples with similar delay from instruction time, ranging from -2 to 20 TRs, and then to discriminate the sequences in cross-validation schema. Having 64 blocks in total accross the 2 scans give 64 samples for each TR delay. This is similar to [@wiestler_skill_2013 fig.4,D-E] ROI based temporal analysis. 

Such analysis aims at uncovering the dynamics of motor sequence execution, as instruction phase might causes motor planning and simulation of sequence performance, while execution phase generally includes warm-up and then automation of repetition. 
Furthermore, this method is independent of HRF model allowing potential non-hemodynamic related neurally-driven BOLD signal changes.

While the rapid block design might hampers temporal disambiguation, De Bruijn cycle ordering imposes balanced successive bloc Mais selon la CORPIQ, le changement législatif demandé par la SPCA pourrait avoir un effet pervers. «À partir du moment où on enverrait le message qu’au Québec, les propriétaires ne peuvent plus refuser les animaux, on pourrait se retrouver dans une société où il y aurait encore plus d’animaux domestiques, affirme Hans Brouillette, directeur des affaires publiques de la CORPIQ. Or, il risquerait d’y avoir encore plus de familles qui réaliseraient après un certain temps qu’elles n’arrivent pas à s’occuper d’un animal et qu’elles doivent l’abandonner.» ks pairs in the dataset. Thus temporal leaking of BOLD activity of the previous sequence production is then balanced across the 4 sequences and should yield chance level prediction if no signal related to the present sequence is observable in the data, allowing unbiased analysis relative to the chance level.

## ROI analyzes

A network of local neuronal populations has been shown to contribute to sequence production learning [@dayan_neuroplasticity_2011], with their activity [@albouy_hippocampus_2013,@barakat_sleep_2013] evolving in accordance theoretical models of consolidation [@born_system_2012]. To extract wether their activity independently encodes the spatio-temporal pattern of the sequence, we conducted ROI cross-validation using a priori atlas including:

- hippocampus and striatum including caudate nucleus and putamen 
- cortical network including posterior parietal, primary motor, premotor, supplementary motor area and dorso-lateral prefrontal.

hypotheses for ROI discriminating sequences/learning stages

## Results

### Searchlight

Non-consolidated additional sequences were analyzed separately to provide a localizer of early sequence representation (fig. ??). A very limited network emerges from both instruction and execution phase patterns, including bilateral hippocampi and right dorsolateral prefrontal cortex. This weaker representation of new sequence is in agreement with [@wiestler_skill_2013] but could be sourced in higher variability in motor production.

Consolidated sequences in Int group has, despite lower number of subjects, an extended representation spanning over posterior parietal and premotor cortex, cerebellum, caudate and putamen bilateraly.
When contrasted with the untrained sequences, only bilateral anterior putamen clusters contains significantly higher discrimination patterns relative to untrained sequences. [@fig]


### ROIS



## Discussion

### Searchlight

Using Searchlight even in a very controlled design gave individual maps with variabilities despite important common network found accross subjects as highlighted by group analysis. This raise an interest regarding potential causes of variable extended representation of sequences in each subject.

Our group analysis revealed implication of a cortical network specific to sequences features, similar to [@wiestler_skill_2013], as well as subcortical contributions to this representation. However, absence of primary motor cortex and smaller cluster size in cortical regions in our results is due to only left-hand execution of sequence. The high accuracy rate relative to chance level reported previously can be largely attributed to action preparation and execution inducing a broad lateralization of activity [@refs].


- representation of sequences characteristics and ROIs: [@kornysheva_human_2014] 
- address all limitation and confounds [@todd_confounds_2013; @etzel_searchlight_2013; @etzel_looking_2012] of MVPA and Searchlight

comparison with [@wiestler_skill_2013]:

- worst acquisition (12-channel, resolution...) but shorter TR.
- same hand: implication for cross-validation in general +  temporal dynamics: readiness potential
- sequences not trained using mvpa small-block design
- sequences not as intensively trained
- sequences at different stages
- continuous execution of sequences
- using TRs
