
## Introduction

Motor skills recruits an extended network of cerebral and spinal regions, which involvement evolves differently across learning stages [@dayan_neuroplasticity_2011].
The learning processes implicates primary motor cortex which is topologicaly structured to encode multi-joint finger movements [@ejaz_hand_2015].
While motor execution network is strongly lateralized to the cortex contralateral to the effector, ipsilateral activity and finger specific neural patterns are also observed [@diedrichsen_two_2013] that might enable bimanual coordination.


Hippocampi and striatal areas [@albouy_hippocampus_2013] were also identified as subcortical actors in the learning and consolidation processes.
Motor sequence learning (MSL) is a adequate paradigm [@] to study neural processes involved from initial learning to automation of skill as structured sequences of movements.

While motor sequence learning aggregate cognitive abilities to improve performance of execution, the network, distributed or not, specifically encoding one sequence is unknown.

In the light of memory consolidation theory, large-scale motor network activity exhibit changes across days after learning which are supported by offline processing notably during sleep [@]. 
However while motor practice is stabilized across automation of the skill, subparts of the networks concurrently stabilize concurrently to activity patterns measured with fMRI [@wiestler_skill_2013]. 
Premotor patterns sub-components can both encode spatial and temporal features [@kornysheva_human_2014], the former characterizing the learned sequences.

The goal of our study is to provide further insight in brain representation of motor sequence with Multivariate Pattern Analysis (MVPA).
We extend previous research [@wiestler_skill_2013;@kornysheva_human_2014] by scanning larger extent of human brain to measures sub-cortical activity including cerebellum.
Also we intend to measures changes induced by learning and consolidation processes.

##Method

### Participants
 
The study includes ## right-handed young (18-35 years old) healthy volunteers recruited by advertising on scholar and public website.

The subject were not included in case of history of neurological psychological or psychiatric disorders or scoring 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961].

Subjects with BMI greater than 27, smokers, extreme chronotype, night-workers, having traveled accross meridian during the 3 previous months, or training as musician or professional typist (for overtraining on coordinated finger movements) were also excluded.

Sleep quality was assessed by Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989], and daytime sleepiness  (Epworth Sleepiness Scale [@johns_new_1991]) had to be lower or equal to 9.

Subject were also instructed to abstain from caffeine, alcohol and nicotine, and have regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the study.
Instruction compliance was controlled by non-dominant hand wrist actigraphy (Actiwatch 2, Philips Respironics, Andover, MA, USA) for the week preceding the experiment.

### Behavioral experiment

Do we describe the full experiment here? Consolidation.... 

The experiment was conducted over 3 consecutive days, at the end of the day, with all motor task performed in the scanner using an ergonomic MRI-compatible 4-keys response pad.

1. In the first evening (D1), subjects were trained to perform with left-hand a 5 elements sequence (TSeq) for 14 blocks of 12 sequences or a maximum of 12x5=60 keypresses.
Subject were instructed to execute repeatedly as fast and as accurate as possible the sequence of keypresses and to start from the beginning of the sequences in case they noticed that they did an error.
They were retested approximately 20 minutes later for an additional single block of 12 sequences.

2. On the second evening (D2), subjects were tested for 1 block on TSeq. 
Then half of the subjects of "Interference" (Int) group were trained on an interfering sequences (IntSeq) of 5 elements with left-hand for 14 blocks of 12 sequences as for TSeq.
"No-Interference" (NoInt) group had scanned resting-state which duration was yoked to a IntGroup subject.

3. On the last evening (D3) subjects first practiced TSeq for 7 blocks x 12 repetitions for retest, then practiced IntSeq for 7 blocks of 12 sequences, which was followed by a task specifically designed for MVPA analysis.
This task was similar to [@wiestler_skill_2013], which rapidly alternates short blocks of practice of 4 different sequences.
Significant differences are that 4 sequences were performed with the left hand four fingers excluding the thumb and no feedback was given regarding the correctness of performance.
Each block, composed of an instruction period of 4 sec when 5 numbers (eg. 1-4-2-3-1) representing in reading order the sequence of finger to be pressed, was displayed followed by an execution period indicated by a green cross.
Subject had to perform 5 times the sequence, or a maximum of 5x5=25 key-presses before being instructed to stop and rest when a red cross was displayed.
Rest duration was variable and set to synchronize the beginning of each block with the same slice in the volume being acquired.

Ordering of the sequences in blocks was chosen to include all possible successive pairs of the sequences using De Bruijn cycles [@aguirre_bruijn_2011]. Given 4 sequences, a 2-length De Bruijn cycle would contains 16 blocks, repeated twice to give 8 repetitions of each of the 4 sequences which amounts to 32 blocks.

Each subject performed the task twice in scans separated of few minutes to allow rest and enable study of cross-scans pattern stability and classifier generalization.

### Scan acquisition

MRI data were acquired on a Siemens Trio 3T scanner on 2 two separate sessions.
The first session used 32-channel coil to acquire high-resolution anatomical T1 weighted image using Multi-Echo (4) MPRAGE (MEMPRAGE) at 1mm isotropic resolution. 

Functional data were acquired during the second session with 12-channel coil for comparison with other dataset. EPI sequence consists of 40 slices with 3.44mm in-plane resolution and 3.3mm slice thickness to provide full cortical and cerebellum coverage.
Consecutively fieldmap was acquired to measure B0 field inhomogeneity to allow retrospective compensation of induced distortions.

### Preprocessing

Custom pipeline was used to preprocess fMRI data prior to analysis.
First, high-resolution anatomical T1 weighted image was preprocessed with Freesurfer [@dale_cortical_1999;@fischl_high-resolution_1999;@fischl_cortical_2008] to segment subcortical regions, reconstruct cortical surfaces and provide inter-subjects alignment of cortical folding patterns. Pial and grey/white matter interface surfaces were downsampled to match the 32k sampling of Human Connectome Project (HCP) [@glasser_minimal_2013] and we averaged pial and white surface to get coordinates at the half of the thickness of cortical sheet. HCP subcortical rois coordinates were warped onto individual T1 data using non-linear registration based on Ants software [@avants_symmetric_2008;@klein_evaluation_2009]. Combination of cortical and subcortical coordinates then corresponds to grayordinates of HCP datasets [@glasser_minimal_2013].

fMRI data underwent estimation of subject motion [@roche_four-dimensional_2011] and coregistration to T1. Registration and motion parameters were used to interpolate Blood-Oxygen-Level-Dependent (BOLD) signal at anatomical grayordinates above-mentioned taking into account B0 inhomogeneity induced distortions using fieldmap acquisition.

BOLD signal was further processed to remove drifts and abrupt motion-related signal change.

Of note is that our preprocessing does not includes smoothing, even though interpolation causes averaging of values of neighboring voxels. We intended to minimize blurring of data to preserve fine-grained patterns of activity.

### Multivariate Pattern Analysis

Similarly to [@wiestler_skill_2013] we aim to uncover activity patterns to predict the sequence being produced.

The MVPA analysis was based on PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom cross-validation, Searchlight and measures to adapt to the study design and analyses.

#### Samples

Blocks were modelled by having 2 boxcars, for instruction and execution phase respectively, convolved with Hemodynamic Response Functions (HRF). Volumes (TRs) corresponding to HRF level above 20% of response level were taken as samples for the performed sequence. Maximum value of instruction and execution regressors determine the TR to pertain to instruction or execution phase.
A TR based approach was chosen to explore the fine dynamic of patterns related to the task, that model driven such as GLM cannot fully analyze at the cost of lower signal-to-noise ratio.

Regular GLM-based approach was also performed using least-square separate (LS-S) regression of individual blocks [@mumford] as it was shown to provide improved activation patterns for MVPA.

#### Cross-validation

The De Bruijn cycles ordering of the sequence in the task aims at providing unbiased cross-validation by balancing the temporal succession of any pair of the 4 sequences.

Chosen cross-validation schema includes:

- Leave-One-Chunk-Out (LOCO): each block is successively taken out of the dataset to be used in prediction. Classifier is trained on remaining data by randomly selecting balanced number of samples of the 4 sequences which are further than 60 secondes to the block of test data. Random selection of balanced data is performed 5 times for each of the 64 blocks amounting to 64*5 = 320 folds of cross-validation.
- Leave-One-Scan-Out (LOSO): random balanced subset of samples from a scan is fed for training to the classifier which then predicts one the other scan the sequences. Random subset was selected 5 times for the 2 scans giving 10 cross-validation folds.

#### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that apply MVPA techniques repeatedly on small spatial neighborhood with the purpose to localize representation of information of interest across brain while avoiding high-dimensional limitation of multivariate algorithms.

Applying cross-validation using the Searchlight schema allowed to extract brain-wise map of classifier performance giving information of regions having stable sequences related patterns.
Gaussian Naive Bayes (GNB) linear classifier, optimized for Searchlight, was performed with the 2 proposed cross-validation schema analysis on the execution labeled TRs.
Also GNB-based Searchlight have been argued to allow smoothness of generated maps [@raizada_smoothness_2013] despite unsmoothed data, allowing more-reliable cross-subject study and thus higher cross-scans generalization.

Searchlight was configured to select for each grayordinate the 64 closest neighboring coordinates, using surface distance for cortical grayordinates as the subset of features to be classified.
Searchlight size has been shown to inflate the extent of significant clusters in searchlight analysis [@etzel_searchlight_2013; @viswanathan_geometric_2012] which motivated the small neighborhood for our analysis.

Cross-validation confusion matrix was computed for each block of practice providing a more complete representation of classification performance and biases from which can be derived specific or global accuracy percentage.

For both LOCO and LOSO cross-validation schema, we measured following accuracy searchlights maps:

- global accuracy : the percentage of samples for which the correct sequence is detected,
- sequence specific sensitivity : for each sequence, TP/(TP+FN) with TP: true positives, FN: false negatives.

#### Dynamics analysis

Also we complemented the analysis by taking TRs subsamples with similar delay from instruction time, ranging from -2 to 20 TRs, and then to discriminate the sequences in cross-validation schema. Having 64 blocks in total accross the 2 scans give 64 samples for each TR delay. This is similar to [@wiestler_skill_2013 fig.4,D-E] ROI based temporal analysis.

Such analysis aims at uncovering the dynamics of motor sequence execution, as instruction phase might causes motor planning and simulation of sequence performance, while execution phase generally includes warm-up and then automation of repetition. 
Furthermore, this method is independent of HRF model allowing potential non-hemodynamic related neurally-driven BOLD signal changes.

While the rapid block design might hamper temporal disambiguation, De Bruijn cycle ordering imposes balanced successive blocks pairs in the dataset. Performing cross-validation in TRs in which the previous or next sequence specific neural code is present will have balanced number of mismatch which will yield chance level.

#### Searchlight Group Analysis

A group searchlight map was computed using mass-univariate one-sample T-test to find regions which consistently departed from chance level (25%) across subjects.

## ROI analyzes

hypotheses for ROI discriminating sequences/learning stages

## Results



## Discussion

- representation of sequences characteristics and ROIs: [@kornysheva_human_2014] 
- address all limitation and confounds [@todd_confounds_2013; @etzel_searchlight_2013; @etzel_looking_2012] of MVPA and Searchlight

comparison with [@wiestler_skill_2013]:

- worst acquisition (12-channel, resolution...) but shorter TR.
- same hand: implication for cross-validation in general +  temporal dynamics: readiness potential
- sequences not trained using mvpa small-block design
- sequences not as trained
- sequences at different stages
- using TRs
