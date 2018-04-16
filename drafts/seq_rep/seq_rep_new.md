---
documentclass: elife
elife: true
title: Consolidation alters motor sequence-specific distributed representations.
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
  The acquisition of new motor sequential skills combines processes leading asymptotically to optimal performance that are mediated by the dynamic contribution of different brain networks.
  Indeed, numerous investigations using functional magnetic resonance imaging in humans have revealed that these processes induce a functional reorganization within the cortico-striatal and cortico-cerebellar motor systems in link with the hippocampus, which is manifested by learning-related increases of activity in some regions and decreases in others.
  Yet, the functional significance of these changes is not fully understood as they convey both the evolution of sequence-specific knowledge, as well as the development of unspecific expertise in the task.
  Moreover, the presence or absence of activity level changes does not specifically assess the occurrence of learning-related plasticity.
  To address these issues, we investigated here the whole-brain local representational changes using novel multivariate distances between fine-grained patterns evoked through the production of motor sequences, either trained in a single session and consolidated or newly acquired.
  Results revealed that while sequences were discriminable in a large cortical network common to both learning stages, sequence representations in prefrontal, motor secondary cortices as well as dorsolateral striatum and associative cerebellum were greater when executing consolidated sequences than untrained ones
  By contrast, sequence representations in the hippocampus and dorsomedial striatum were less engaged.
  These results show that complementary sequence-specific motor knowledge representations distributed in different striatal, cerebellar, cortical and hippocampal regions evolve distinctively during the critical phases of skill acquisition and consolidation, hence specifying further their roles in supporting the evolution of acquired motor memories.
---

# Introduction {#sec:introduction}

Animals and humans are able to acquire and automatize new sequences of movements, hence allowing them to expand and update their repertoire of complex goal-oriented motor actions for long-term use.
To investigate the mechanisms underlying this type of procedural memory in humans, a large body of behavioral studies has used motor sequence learning (MSL) tasks designed to test the ability to perform temporally ordered coordinated movements, learned either implicitly or explicitly and has assessed their performances in different phases of the acquisition process [@korman_multiple_2003;@abrahamse_control_2013;@diedrichsen_motor_2015;@verwey_cognitive_2015].
While practice of an explicit MSL task leads to substantial within-session execution improvements, there is now ample evidence indicating that between-session maintenance, and even increases, in performance can be observed after a night of sleep [@landry_effects_2016;@nettersheim_role_2015], while performance tends to decay during an equal period of wake [@doyon_contribution_2009;@brawn_consolidating_2010;@nettersheim_role_2015;@landry_effects_2016].
Therefore, it is thought that sleep favors reprocessing of the motor memory trace, thus promoting its consolidation for long-term skill proficiency [see @king_sleeping_2017;@doyon_current_2018 for a recent in-depth reviews].

Functional magnetic resonance imaging (fMRI) studies using General-Linear-Model (GLM) contrasts of activation have also revealed that MSL is associated with the recruitment of an extended network of cerebral [@hardwick_quantitative_2013], cerebellar and spinal regions [@vahdat_simultaneous_2015], whose contributions differentiate as learning progresses [@karni_acquisition_1998;@dayan_neuroplasticity_2011;@doyon_current_2018].
In fact, critical plastic changes [@ungerleider_imaging_2002;@doyon_reorganization_2005] are known to occur within the initial training session, as well as during the offline consolidation phase, the latter being characterized by a functional "reorganization" of the nervous system structures supporting this type of procedural memory function [@rasch_reactivation_2008;@born_system_2012;@albouy_hippocampus_2013;@dudai_consolidation_2015;@bassett_learning-induced_2015;@fogel_reactivation_2017;@vahdat_network-wide_2017].
More specifically, MSL practice is known to activate a cortical, associative striatal and cerebellar motor network which is assisted by the hippocampus at the beginning of the fast-learning phase [@albouy_hippocampus_2013].
Yet, when approaching asymptotic behavioral performance after longer practice, activity within the hippocampus decreases while activity within the sensorimotor striatum increases [@doyon_experience-dependent_2002], both effects conveying the transition to the slow-learning phase.
The same striatal regions are reactivated during sleep spindles [@fogel_reactivation_2017] contributing to the progressive emergence of a reorganized network [@debas_brain_2010;@vahdat_network-wide_2017], which is further stabilized when additional MSL practice extending over multiple days is separated by consolidation periods [@lehericy_distinct_2005].

A critical issue typically overlooked by previous MSL neuroimaging research using GLM-based activation contrasts, however, is that learning-related changes in brain activity do reflect the temporal evolution of recruited processes during blocks of practice, only some of which may be specifically related to plasticity induced by MSL.
For instance, increases in activity could not only signal a greater implication of the circuits specialized in movement sequential learning *per se*, but could also result from the inherent faster execution of the motor task.
Likewise, a decrease in activity could either indicate some form of optimization and greater efficiency of the circuits involved in executing the task [@wu_how_2004], or could show the reduced recruitment of non-specific networks supporting the acquisition process.
Therefore, even with the use of control conditions to dissociate sequence-specific from non-specific processes [@orban_multifaceted_2010], the observed large-scale activation differences associated with different learning phases do not necessarily provide direct evidence of plasticity related to the processing of a motor sequence-specific representation [@berlot_search_2018].
Furthermore, it is also conceivable that these plastic changes could even occur locally without significant changes in the GLM-based regional activity level.
Finally, in most studies investigating the neural substrate mediating the consolidation process of explicit MSL, the neural changes associated with this mnemonic mechanism are assessed by contrasting brain activity level of novice participants between their initial training and a delayed practice session,
Therefore, they measure not only plasticity for sequence-specific (e.g. optimized chunks), but also task-related expertise (e.g. habituation to experimental apparatus, optimized execution strategies, attentional processes).
The latter expertise is notably observed when participants practice two motor sequences in succession and the initial performance during sequence execution is significantly better for the subsequent than for the first sequence.

To address these specificity limitations, multivariate pattern analysis (MVPA) has been proposed to evaluate how local patterns of activity are able to reliably discriminate between stimuli or evoked memories of the same type over repeated occurrences, hence allowing to test information-based hypotheses that GLM contrasts cannot inquire [@hebart_deconstructing_2017].
In the MSL literature, only a few studies have used such MVPA approaches to identify the regions that specialize in processing the representation of learned motor sequences [@wiestler_skill_2013;@nambu_decoding_2015;@wiestler_integration_2011;@kornysheva_human_2014;@yokoi_does_2017], although they have mainly focused on later phases of MSL after repeated practice over multiple days.
For instance, in a recent study covering dorsal cerebral cortices only [@wiestler_skill_2013], cross-validated classification accuracy was measured separately on activity patterns evoked by the practice of trained and untrained sets of sequences.
These authors showed that the extended training increased sequence discriminability in a network spanning bilaterally the primary and secondary motor as well as parietal cortices.
In another study [@nambu_decoding_2015] that aimed to analyze separately the preparation and execution of sequential movements, representations of extensively trained sequences were identified in the contralateral dorsal premotor and supplementary motor cortices during preparation, while representations related to the execution were found in the parietal cortex ispilaterally, the premotor and motor cortices bilaterally as well as the cerebellum.
In both studies, the regions carrying sequence-specific representations overlapped only partly with those identified using GLM-based measures, hence illustrating the fact that coarser differences in activation between novel and trained sequences does not necessarily provide evidence of plasticity for sequential information.
However, the classification-based measures they used may have biased their parametric statistical results by violating both the normality assumption and theoretical null-distribution [@jamalabadi_classification_2016;@allefeld_valid_2015;@combrisson_exceeding_2015;@varoquaux_cross-validation_2017] and may have thus been suboptimal in detecting representational changes [@walther_reliability_2016].

The present study is part of a larger research program.
It aimed to address both the critical issues overlooked by previous research investigating the MSL consolidation with GLM-based approach described above, as well as the limitations encountered when using classifier-based MVPA methods.
Specifically, we employed a recently developed MVPA approach [@nili_toolbox_2014] that is unbiased and more sensitive to continuous representational changes [@walther_reliability_2016], such as those that occur in the early stage of MSL and consolidation [@albouy_interaction_2013].
Our experimental manipulation allowed to isolate sequence-specific plasticity, by extracting patterns evoked through practice of both consolidated and new sequences at the same level of task expertise and by computing this novel multivariate distance metric using a searchlight approach over the whole brain in order to cover cortical and subcortical regions critical to MSL.
Based on theoretical models [@albouy_hippocampus_2013;@doyon_current_2018] derived from imaging and invasive animal studies, we hypothesized that offline consolidation following training would induce greater cortical and striatal as well as weaker hippocampal sequence-specific representations.

# Results {#sec:results label=}

To investigate changes in the neural representations of motor sequences occurring during the course of learning, young healthy participants (n=18) practiced two sequences of finger movements (executed through button presses) separately on two consecutive days.
On the third day, participants were required to execute again the same two sequences, then considered to be consolidated, together with two new untrained ones, in 64 pseudo-randomly ordered short blocks splitted in two runs, with 16 blocks of each sequence.
All four sequences were executed using their non-dominant left hand while functional MRI data was acquired.

## Behavioral performance

We analyzed the behavioral performance related to the four different sequences using a repeated-measure mixed-effects model.
As expected, new sequences were performed more slowly ($\beta=.365 , SE=0.047, p<.001$) and less accurately ($\beta=-0.304, SE=0.101,p<.001$) than the consolidated ones.
Significant improvement across blocks was observed for new sequences as compared to consolidated sequences in term of change of speed ($\beta=-0.018, SE=0.002, p<.001$), thus showing an expected learning curve visible in @fig:mvpa_task_groupInt_seq_duration.
Yet accuracy did not show significant improvement ($\beta=0.014, SE=0.010, p=0.152$) likely explained by the limited precision of this measure that ranges discretely from 0 to 5.
By contrast, the consolidated sequences did not show significant changes in speed ($\beta=-0.006, SE=0.005, p=0.192$) nor accuracy ($\beta=-0.006, SE=0.057, p=0.919$), the asymptotic performances being already reached through practice and the consolidation process.

![Average and standard deviation of correct sequence durations across the MVPA task blocks.](../../results/behavior/mvpa_task_groupInt_seq_duration.pdf){#fig:mvpa_task_groupInt_seq_duration width=15cm}

Importantly, there were also no significant differences between the two consolidated sequences in term of speed ($\beta=0.031, SE=0.026, p=0.234$) and accuracy ($\beta=-0.030, SE=0.111, p=0.789$), nor between the two new sequences speeds ($\beta=0.025, SE=0.045, p=0.577$) and accuracies ($\beta=-0.245, SE=0.138, p=0.076$)].

## A common distributed sequence representation for sequences irrespective of learning stage

From the preprocessed functional MRI data we extracted patterns of activity for each block of practice, and computed a cross-validated Mahalanobis distance [@nili_toolbox_2014;@walther_reliability_2016] using a Searchlight approach [@kriegeskorte_information-based_2006] over brain cortical surfaces and subcortical regions of interest.
Such multivariate distance, when positive, demonstrate that there is a stable difference in activity patterns between the conditions compared, and thus reflect the level of discriminability between these conditions.
To assess true patterns and not mere global activity differences, we computed this discriminability measure for sequences that were at the same stage of learning, thus separately for consolidated and new sequences.
From the individual discriminability maps, we then measured the prevalence of discriminability at the group level, using non-parametric testing with a Threshold-Free-Cluster-Enhancement approach (TFCE) [@smith_threshold-free_2009] to enable locally adaptive cluster-correction.

To extract the brain regions that show discriminative activity patterns for specific sequence during both learning stages, we then submitted these separate group results for the consolidated and new sequences to a minimum-statistic conjunction.
A large distributed network (@fig:new_cons_conj_crossnobis_map) displayed significant discriminability, including the primary visual, as well as the posterior parietal, primary and supplementary motor, premotor and dorsolateral prefrontal cortices.(see the statistical maps for each learning stage separately in the Supplementary material (@fig:new_crossnobis_map,@fig:cons_crossnobis_map).

![Group searchlight conjunction of new and consolidated sequences discriminability maps (z-score thresholded at $p<.05$ TFCE-cluster-corrected) showing a large distributed cortical network showing sequence disciminative patterns at both learning stages; Regions of interest with Freesurfer colors: Acc.:Accumbens; Pt.:Putamen; Caud.:Caudate; Pal.:Pallidum; vDC:ventral Diencephalon; Am.:Amygdala; Hc.:Hippocampus; Thal.:Thalamus; Cb.:Cerebellum; BS:brain-stem](../../results/crossnobis_tfce/new_cons_conj_crossnobis_tfce_map_vert_rois_fade_labels.pdf){#fig:new_cons_conj_crossnobis_map width=14cm}

## Reorganization of the distributed sequence representation after consolidation

In order to evaluate the reorganization of sequence representation undergone by consolidation at the group level, the consolidated and new sequences' discriminability maps from all participants were submitted to a non-parametric pairwise t-test with TFCE.
To ascertain that a greater discriminability in one stage versus the other was supported by a significant level of discriminability within that stage, we then calculated the conjunction of the contrast maps with the consolidated and new sequences group results, respectively with the positive and negative contrast differences (@fig:contrast_conj_cons_new_crossnobis_map).

Discriminability between the consolidated sequences was significantly higher than that between the new sequences in bilateral sensorimotor putamen, thalamus and anterior insula, as well as in the ispilateral cerebellar lobule IX, posterior cingulate and parietal cortices, and contralaterally in the lateral and dorsal premotor, supplementary motor, frontopolar and dorsolateral prefrontal cortices in addition to cerebellar Crus I.
By contrast, the pattern dissimilarity was higher for the new sequences in bilateral hippocampi as well as the body of the caudate nuclei, subthalamic nuclei, and cerebellar Crus II ipsilaterally.
Hence, while striatal activity patterns differentiating newly acquired sequences were found in contralateral putamen (@fig:new_crossnobis_map), this discriminability was significantly larger for consolidated sequences in motor putamen bilaterally.

![Conjunction of group searchlight contrast (paired t-test) between consolidated and new sequences discriminability maps and separate group discriminability maps for new and consolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) showing a reorganization of the distributed memory trace between these two stages; Acc.: Accumbens; Pt.:Putamen; Caud.:Caudate; Pal.:Pallidum; vDC:ventral Diencephalon; Am.:Amygdala; Hc.:Hippocampus; Thal.:Thalamus; Cb.:Cerebellum; BS:brain-stem](../../results/crossnobis_tfce/contrast_conj_cons_new_crossnobis_tfce_map_vert_rois_fade_labels.pdf){#fig:contrast_conj_cons_new_crossnobis_map width=14cm}

# Discussion {#sec:discussion}

In the present study, we aimed to identify the brain networks whose activity patterns differentiate between representations of multiple motor sequences during their execution in different phases of learning (newly learned vs consolidated).
Using an MVPA approach, we considered that locally stable patterns of activity could be used as proxy for the specialization of neuronal circuits supportive of the efficient sequential motor memory retrieval and expression.
To investigate the differential pattern strength, we then computed novel unbiased multivariate distance and applied robust permutation-based statistics with adaptive cluster correction.

## A distributed representation of finger motor sequence

Our results provide evidence for an extended network of brain regions that shows reliable discrimination of sequence-specific activity patterns for both consolidated and novel sequences.
At the cortical level, a network encompassing the bilateral supplementary motor and premotor areas, as well as posterior parietal cortices and contralateral somatosensory motor cortex was found, consistent with earlier similar MVPA investigations [@wiestler_skill_2013;@nambu_decoding_2015].
It is noteworthy that such discrimination of motor sequence representations within the ipsilateral premotor and parietal cortices has been previously described [@wiestler_skill_2013;@waters-metenier_bihemispheric_2014;@waters_cooperation_2017], notably when the non-dominant hand is used for fine dexterous manual skills.
We also found a significant neural representation for both learning stages in contralateral primary motor and somatosensory (M1/S1) cortices, more specifically around the hand knob area [@yousry_localization_1997] for which finger somatotopy is measurable in fMRI [@ejaz_hand_2015].
However, previous reports of such a representation in primary regions has recently been revised, being likely explained by the stronger activity evoked by the first finger press in the sequence [@yokoi_does_2017].
In our study, the new sequences started with different fingers and this first-finger effect could explain this significant discriminability.
Nevertheless, the consolidated sequences for which first finger press was similar, were significantly discriminable in M1/S1, but such pattern separability could have been driven by another factor that was not based on sequential features, such as the second finger pressed.
Yet, the spatial extent of this M1/S1 representation was smaller compared to that found by @wiestler_skill_2013.
This is likely explained by difference in our design, notably the uninterrupted repetition of the motor sequences during practice that singularize this effect to the beginning of the block.
Moreover, none of our sequences engaged the thumb, which, having a more distinctive M1/S1 pattern, would have accentuated this effect if used to initiate the sequence [@ejaz_hand_2015].

The conjunction of new and consolidated sequences discriminability maps further revealed that a common cortical processing network, including non-motor support regions, carries sequential information across learning stages, that can originate from visually presented instruction and short-term-memory to motor sequence production.
Herein, the occipital cortex likely reflecting the processing of the visual stimuli as low-level visual mapping of shapes [@pilgramm_motor_2016;@miyawaki_visual_2008], as well as the ventro-temporal regions, known to support higher level Arabic number representation [@shum_brain_2013;@peters_neural_2015] were found to discriminate between sequences in both stages of learning [@fig:new_cons_conj_crossnobis_map].
The dorsolateral prefrontal cortex (DLPFC), which also exhibited pattern discriminability, was suggested previously to process the sequence spatial information in working memory, preceding motor command [@robertson_role_2001].
In fact, we believe that the cognitive processing requirements posed by our task, implying notably to switch between sequences, to maintain them in working memory and to inhibit interfering ones, could have magnified this frontal associative representation in our study.

In sum, this network reflecting sequence information shows some overlap with the main effect of activation-based studies [@hardwick_quantitative_2013;@dayan_neuroplasticity_2011], but also reveals discrepancies attributable to information-based inference, some of which originate from task-induced additional processing of sequential knowledge.

## Cortico-subcortical representational reorganization underlying memory consolidation following MSL

By contrasting the maps of multivariate distances for consolidated and newly acquired sequences, we identified the networks that reveal increased versus decreased discriminability of sequential representations in the early stages of the MSL consolidation (@fig:contrast_conj_cons_new_crossnobis_map).

At the cortical level, we found that the contralateral premotor and bilateral parietal regions showed a stronger representation for consolidated sequences.
This pattern likely reflects that the tuning of these neural populations to coordinated movements is consolidated early after learning [@makino_transformation_2017;@yokoi_does_2017;@pilgramm_motor_2016], as was previously observed when contrasting sequence that underwent a longer training to new ones [@wiestler_skill_2013].

Interestingly, however, significant differences at the subcortical level were found in bilateral putamen and more specifically in their sensorimotor regions, a result consistent with findings from activation studies that reported increased functional activity after consolidation in this structure [@debas_brain_2010;@albouy_hippocampus_2013;@debas_off-line_2014;@fogel_reactivation_2017;@vahdat_network-wide_2017].
Significant representational changes were also found in the bilateral thalami, and could reflect the relay of information between the cortex and cerebellum, striatum or spinal regions [@haber_cortico-basal_2009;@doyon_contributions_2009].
Finally, representation changes were detected in the cerebellum, including ipsilateral Lobule IX, shown to correlate with sequential skill performance [@orban_multifaceted_2010;@tomassini_structural_2011] as well as contralateral Crus II which connectivity with prefrontal cortex is thought to support motor functions [@ramnani_primate_2006].
However, no significant difference was observed in Lobule V of the cerebellum that is known to carry finger somatotopic representations [@wiestler_integration_2011] and to show global activation during practice [@doyon_experience-dependent_2002].

Concurrently with the representational increase in the above-mentioned network, we found only a few disparate regions that showed decreased sequence discrimination, namely the caudate nuclei, subthalamic nuclei and cerebellar Crus II ipsilaterally as well as bilateral hippocampi.
Hippocampal activation in early learning has formerly been hypothesized to support the temporary storage of novel explicitly acquired motor sequence knowledge and to contribute to the reactivations of the distributed network during offline periods and sleep in particular.
Yet such contribution of the hippocampus has been shown to be progressively disengaging afterward [@albouy_hippocampus_2013], and thus our results are consistent with the idea of the hippocampus playing a transient supportive role in early MSL, notably in encoding sequential information [@davachi_how_2015].
Our findings of a differential implication of dorsomedial and dorsolateral striatum in sequence representation during learning and expression of a mastered skill specifies the changes in activity in these regions in the course of MSL described by earlier studies [@lehericy_distinct_2005;@jankowski_distinct_2009;@francois-brosseau_basal_2009;@kupferschmidt_parallel_2017;@corbit_corticostriatal_2017;@fogel_reactivation_2017;@reithler_continuous_2010].
Indeed, our results uncover that this shift in activity purports a genuine reorganization of circuits processing sequence-specific information, similar to what was reported at neuronal level in animals [@costa_differential_2004;@yin_dynamic_2009;@miyachi_differential_2002].
Importantly, in our task, the alternate production of different sequences required shifting between overlapping sets of motor commands, and could thus have further implicated the dorsal striatum in collaboration with the prefrontal cortex [@monchi_functional_2006].

While our results show that the distributed representational network is reorganized during MSL consolidation, the present study was not designed to investigate the information-content of hippocampal, striatal or cerebellar sequence representations that were previously assessed at cortical level for finger sequences [@wiestler_effector-independent_2014;@kornysheva_human_2014] as well as for larger forearm movements [@haar_effector-invariant_2017].
Notably, the hypothesized extrinsic and intrinsic skill encoding in hippocampal and striatal systems respectively [@albouy_daytime_2013], remains to be assessed with a dedicated experimental design similar to that used by @wiestler_effector-independent_2014 to investigate such representations at the cortical level.

Importantly, our study investigated the change in neural substrates of sequence representation after limited training and following sleep-dependent consolidation.
This is in contrast to previous investigations that studied sequences trained intensively for multiple days [@nambu_decoding_2015] and compared their discriminability to that of newly acquired ones [@wiestler_skill_2013].
Therefore, in our study, the engagement of these representations for expressing the sequential skill may further evolve, strengthen or decline locally with either additional training or offline memory reprocessing supported in part by sleep.

## Methodological considerations

To limit the level of difficulty and the duration of the task, only four sequences were practiced by participants, two consolidated and two newly acquired.
This low number of sequence per condition could be a factor limiting the power of our analysis, as only a single multivariate distance is assessed for each of these conditions.
Moreover, the training sessions were each comprised of a single sequence performed in blocks longer than in the present task, thus the latter further induced demands for instruction processing, retention in working memory, switching and inhibition of other sequences that could have triggered some novel learning for the consolidated sequences.

Another potential limitation relates to the fact that the present representational analysis disregarded the behavioral performance.
Nevertheless, the chained non-linear relations between behavior, neural activity and BOLD signal were recently established to have limited influence on the representational geometry extracted from Mahalanobis cross-validated distance in primary cortex, sampled across a wide range of speed of repeated finger-presses and visual stimulation [@arbuckle_stability_2018].
Therefore, despite behavioral variability and potential ongoing evolution of the memory trace, we assumed that the previously encoded motor sequence engrams were nevertheless retrieved during this task as supported by the significant differences in activity pattern discriminability and the persistent behavioral advantage observed for the consolidated sequences.

Finally, our results also entail that it is possible to investigate learning-related representational changes in a shorter time-frame and with less extended training than what was investigated before [@wiestler_skill_2013;@nambu_decoding_2015], including in subcortical regions which neuronal organization differ to that of the cortex.
The use of a novel multivariate distance could have contributed to obtain these results by achieving increased sensitivity and statistical robustness [@walther_reliability_2016].

# Conclusion {#sec:conclusion}

Our study shows that the consolidation of sequential motor knowledge is supported by the reorganization of newly acquired representations within a distributed cerebral network.
We uncover that following learning, local activity patterns tuned to represent sequential knowledge are enhanced not only in extended cortical areas, similarly to those shown after longer training [@wiestler_skill_2013], but also in dorsolateral striatum, thalamus and cerebellar regions.
Conversely, a smaller network showed a decrease of sequence specific patterned activation after consolidation, occurring specifically in dorsomedial striatum that support cognitive processing during early-learning [@doyon_current_2018] as well as in the hippocampus which carry explicit encoding of motor sequential extrinsic representation [@albouy_hippocampus_2013;@king_sleeping_2017] and play a significant role in the offline reprocessing.
Despite discrepancies with anterior GLM-based activity changes, the results of our novel representational approach corroborate their interpretations that the differential plasticity changes in the latter regions subtend MSL consolidation [@albouy_maintaining_2015] and in addition, specifies that these convey sequential information.
Yet such consolidation-related representational changes need to be further investigated through exploration of the dynamic mechanism mediating this sleep-dependent mnemonic process, which is known to reorganize progressively the cerebral network by repeatedly reactivating the memory [@boutin_transient_2018;@fogel_reactivation_2017;@vahdat_network-wide_2017].

# Materials and methods {#sec:materials_and_methods}

## Participants

Right-handed young ($n=34, 25\pm6.2$yr.) healthy individuals (19 female), recruited by advertising on academic and public website, participated in the study.
Participants were excluded if they had a history of neurological psychological or psychiatric disorders, scored 4 and above on the short version of Beck Depression Scale [@beck_inventory_1961], had a BMI greater than 27, smoked, had an extreme chronotype, were night-workers, had traveled across meridians during the three previous months, or were trained as musician or professional typist for more than a year.
Their sleep quality was subjectively assessed, and individuals with score to the Pittsburgh Sleep Quality Index questionnaire [@buysse_pittsburgh_1989] greater or equal to 5, or daytime sleepiness Epworth Sleepiness Scale [@johns_new_1991] score greater than 9, were excluded.

Participants included in the study were also instructed to abstain from caffeine, alcohol and nicotine, to maintain a regular sleep schedule (bed-time 10PM-1AM, wake-time 7AM-10AM) and avoid taking daytime nap for the duration of the experiment.
In a separate screening session, EEG activity was also recorded while participants slept at night in a mock MRI scanner and gradients sounds were played to both screen for potential sleep disorders and test their ability to sleep in the experimental environment; 18 participants were excluded for not meeting the criterion of a minimum of 20min. in NREM2 sleep.
After this last inclusion step, their sleep schedule was assessed by analyzing the data obtained from an actigraph (Actiwatch 2, Philips Respironics, Andover, MA, USA) worn on the wrist of the non-dominant hand for the week preceding as well as during the three days of experiment, hence certifying that all participants complied to the instructions.

Among the 34 participants, one did not show within-session improvement on the task, one didn't sleep on the first night, two were withdrawn for technical problems <todo:check with Arnaud and Ella>, and four withdrew themselves before completion of the experiment.
Thus, among the 26 participants that completed the research project, a group of 18 which, by design, followed the appropriate behavioral intervention for the present study, were retained for our analysis.

All participants provided written informed consent and received financial compensation for their participation.
This study protocol was approved by the Research Ethics Board of the “Comité mixte d'éthique de la recherche - Regroupement en Neuroimagerie du Québec” (CMER-RNQ).

## Procedures and tasks

The present study was conducted over 3 consecutive evenings and is part of an experiment that aimed to investigate the neural substrates mediating the re-consolidation of motor sequence memories.
On each day, participants performed the experimental tasks while their brain activity was recorded using MRI.
Their non-dominant hand (left) was placed on an ergonomic MRI-compatible response pad equipped with 4-keys corresponding to each of the fingers excluding the thumb.

On the first day (D1), participants were trained to perform repeatedly a 5-elements sequence (TSeq1: 1-4-2-3-1  where 1 indicate the little finger and 4 the index finger).
The motor sequence was performed in blocks separated by rest periods to avoid fatigue.
Apart for a green or a red cross displayed in the center of the screen, respectively instructing the participants to execute the sequence or to rest, there were no other visual stimuli presented during the task.
Participants were instructed to execute the sequence repeatedly, and as fast and accurately as possible, as long as the cross was green.
They were then instructed to rest for the period of 25 sec. as indicated by the red cross.
During each of the 14 practice blocks, participants performed repeatedly 12 motor sequences (i.e. 60 keypresses per block).
In case participants made a mistake during sequence production, they were instructed to stop their performance and to immediately start practicing again from the beginning of the sequence until the end of the block.
After completion of the training phase, participants were then instructed to visually fixate a white cross for two ensuing resting-state scans of 7 minutes each, and were then administered a retention test, which consisted of a single block comprising 12 repetitions of the sequence.
Then the participants were scanned with concurrent EEG and fMRI for approximately two hours while instructed to sleep.

On the second day (D2), participants were first evaluated on the TSeq1 (1 block retest) to test their level of consolidation of the motor sequence, and were then trained on a new sequence (TSeq2: 1-3-2-4-1) which was again performed for 14 blocks of 12 sequences each, similarly to TSeq1 training on D1.  Again, they were then scanned during sleep while EEG recordings were acquired.

Finally, on the third day (D3), participants first performed TSeq1 for 7 blocks followed by 7 blocks of TSeq2, each block including 12 repetitions of the sequence or 60 keypresses
Following this last testing session, participants were then asked to complete an experimental task (here called MVPA task) that constituted the object of the current study, similar to a previous study that investigated sequence representation by means of multivariate classification [@wiestler_skill_2013].
Specifically, participants performed short practice blocks of 4 different sequences, including TSeq1 and TSeq2 that were then consolidated, as well as two new finger sequences (NewSeq1: 1-2-4-3-1, NewSeq2: 4-1-3-2-4).
In contrast to @wiestler_skill_2013, however, all four sequences used only four fingers of the left-hand, excluding the thumb.
Also, as for the initial training, sequences were instead repeated uninterruptedly and without feedback, in order to probe the processes underlying automatization of the skill.

Each block was composed of an instruction period of 4 seconds during which the sequences to be performed was displayed as a series of 5 numbers (e.g. 1-4-2-3-1).
The latter was then followed by an execution phase triggered by the appearance of a green cross.
Participants performed 5 times the same sequence (or a maximum of 25 key-presses), before being instructed to stop and rest when the red cross was displayed.

The four sequences were assigned to blocks such as to include all possible successive pairs of the sequences using De Bruijn cycles [@aguirre_bruijn_2011], thus preventing the systematic leakage of BOLD activity patterns between blocks in this rapid design.
As a 2-length De Bruijn cycle of the 4 sequences has to include each sequence 4 times, this yielded a total of 16 blocks.
In our study, two different De Bruijn cycles were each repeated twice in two separate scanning runs separated by approximately 5 minutes of rest, hence resulting in a total of 64 blocks (4 groups of 16 practice blocks for a total of 16 blocks per sequence).


## Behavioral statistics

Using data from the MVPA-task, we entered the mean duration per block of correctly performed sequences into a linear mixed-effect model with a sequence learning stage (new/consolidated) by block (1-16) interaction to test for difference in their performance level, as well as the evolution during the task, with sequences and blocks as random effects and participants as the grouping factor.
The same model was run with the number of correct sequences as the outcome variable.
Two other models were also used on subsets of data to test separately if there was any significant difference in performance (speed and accuracy) between the two consolidated sequences and between the two new sequences.
Full models outputs are reported in supplementary materials.

## MRI data acquisition

MRI data were acquired on a Siemens TIM Trio 3T scanner on 2 two separate sessions.
The first session used a 32-channel coil to acquire high-resolution anatomical T1 weighted sagittal images using a Multi-Echo MPRAGE sequence (MEMPRAGE; voxel size=1mm isometric; TR=2530ms; TE=1.64,3.6,5.36,7.22ms; FA=7; GRAPPA=2; FoV=$256\times256\times176mm$) with the different echoes combined using a Root-Mean-Square (RMS).

Functional data were acquired during the second session with a 12-channel coil, which allowed to fit an EEG cap to monitor sleep after training, and using an EPI sequence providing complete cortical and cerebellum coverage (40 axial slices, acquire in ascending order, TR=2160ms;FoV=$220\times220\times132$mm, voxel size=$3.44\times3.44\times3.3$mm, TE=30ms, FA=90, GRAPPA=2).
Following task fMRI data acquisition, four volume were acquired using the same EPI sequence but with reversed phase encoding to enable retrospective correction of distortions induced by B0 field inhomogeneity.

## MRI data preprocessing

High-resolution anatomical T1 weighted images were preprocessed with Freesurfer [@dale_cortical_1999;@fischl_high-resolution_1999;@fischl_cortical_2008] to segment subcortical regions, reconstruct cortical surfaces and provide inter-individual alignment of cortical folding patterns.
Pial and grey/white matter interface surfaces were downsampled to match the 32k sampling of Human Connectome Project (HCP) [@glasser_minimal_2013].
HCP subcortical atlas coordinates were warped onto individual T1 data using non-linear registration with the Ants software [@avants_symmetric_2008;@klein_evaluation_2009].

A custom pipeline was then used to preprocess fMRI data prior to analysis and relied on an integrated method [@pinsard_integrated_2018] which combines slice-wise motion estimation and intensity correction followed by the extraction of BOLD timecourses in cortical and subcortical gray matter.
This interpolation concurrently removed B0 inhomogeneity induced EPI distortion estimated by the FSL Topup tool using the fMRI data with reversed phase encoding [@andersson_how_2003] acquired after the task.
BOLD signal was further processed by detecting whole-brain intensity changes that corresponded to large motion, and each continuous period without such detected event was then separately detrended to remove linear signal drifts.

Importantly, the fMRI data preprocessing did not include smoothing, even though the interpolation inherent to any motion correction was based on averaging of values of neighboring voxels.
This approach was intended to minimize the blurring of data in order to preserve fine-grained patterns of activity, with the resolution of relevant patterns being hypothetically at the columnar scale.

## Multivariate Pattern Analysis

### Samples

Each block was modeled by two boxcars, corresponding to the instruction and execution phases respectively, convolved with the single-gamma Hemodynamic Response Function.
Least-square separate (LS-S) regression of each event, which have been shown to provide improved activation patterns estimates for MVPA [@mumford_deconvolving_2012], yielded instruction and execution phases beta maps for each block that were further used as MVPA samples.

### Cross-validated multivariate distance

Similarly to @wiestler_skill_2013 and @nambu_decoding_2015, we aimed to uncover activity patterns that represented the different sequences performed by the participants.
However, instead of calculating cross-validated classification accuracies, we opted for a representational approach by computing multivariate distance between activity patterns evoked by the execution of sequences, in order to avoid ceiling effect and baseline drift sensitivity [@walther_reliability_2016].
In the current study, we computed the cross-validated Mahalanobis distance [@nili_toolbox_2014;@walther_reliability_2016;@diedrichsen_distribution_2016], which is an unbiased metric that uses multivariate normalization by estimating the covariance from the GLM fitting residuals and regularizing it through Ledoit-Wolf optimal shrinkage [@ledoit_honey_2004].
This distance, which measures discriminability of conditions, was estimated separately for pairs of sequences that were in a similar acquisition stage, that is, for the newly acquired and consolidated sequences.

### Searchlight analysis

Searchlight [@kriegeskorte_information-based_2006] is an exploratory technique that applies MVPA repeatedly on small spatial neighborhoods covering the whole brain while avoiding high-dimensional limitation of multivariate algorithms.
Searchlight was configured to select for each gray-matter coordinate their 64 closest neighbors as the subset of features for representational distance estimation.
The neighborhood was limited to coordinates in the same structure (hemisphere or region of interest), and proximity was determined using respectively Euclidian and geodesic distance for subcortical and cortical coordinates.
The extent of the searchlight was thus kept to such a local range to limit the inflation of false positive or negative results [@etzel_looking_2012;@etzel_searchlight_2013].

### Statistical testing

To assess statistical significance of multivariate distance and contrasts, group-level Monte-Carlo non-parametric statistical testing using 10000 permutations was conducted on searchlight distance maps with Threshold-Free-Cluster-Enhancement (TFCE) correction [@smith_threshold-free_2009].
The statistical significance level was set at $p<.05$ (with confidence interval $\pm.0044$ for 10000 permutations) with a minimum cluster size of 10 features.
TFCE enabled a locally adaptive statistics and cluster size correction that particularly fitted our BOLD sampling of sparse gray-matter coordinates, as well as the large differences in the sizes of the structures that were investigated.

The MVPA analysis was done using the PyMVPA software [@hanke_pymvpa_2009] package with additional development of custom samples extraction, cross-validation scheme, efficient searchlight and multivariate measure computation, optimally adapted to the study design and the anatomy-constrained data sampling.

# Acknowledgments {#sec:acknowledgments}

We thank J.Diedrichsen for methodological advice on our multivariate representational analysis.

# Author contributions

- Conceptualization: BP, AB, EG, HB, JD
- Investigation: AB, EG, BP
- Analysis: BP
- Software development: BP
- Writing: BP
- Review and editing: BP, AB, EG, OL, HB, JD

# Funding {#sec:funding}

This work was supported by the Canadian Institutes of Health Research (MOP 97830) to JD, as well as by French Education and Research Ministry and Sorbonne Universités to BP.

\newpage

# Supplementary materials {#supplementary_materials label="S"}

## Behavioral linear mixed-effect model outputs

#### Test for differences in speed as mean duration to perform a correct sequence per block

\footnotesize

```
mean_seq_duration ~ seq_new * blocks + (blocks+sequences | participants)
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

\newpage


#### Test for differences in accuracy as the number of correct sequences over the 5 repetitions in a block

```
num_correct_seq ~ seq_new * blocks + (blocks+sequences | participants)
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

\newpage

#### Test for differences in speed and accuracy between the new sequences

```
mean_seq_duration ~ sequences*blocks + (1|participants)
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

num_correct_seq ~ sequences*blocks + (1|participants)
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

\newpage

#### Test for differences in speed and accuracy between the consolidated sequences

```
mean_seq_duration ~ sequences*blocks + (1|participants)
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

num_correct_seq ~ sequences*blocks + (1|participants)
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

\newpage

## Representational distance maps

\beginsupplement

![Group searchlight map of cross-validated Mahalanobis distance between the two new sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/new_crossnobis_tfce_map_rois_fade.pdf){#fig:new_crossnobis_map width=16cm}

![Group searchlight map of cross-validated Mahalanobis distance between the two consolidated sequences (z-score thresholded at $p<.05$ TFCE-cluster-corrected) ](../../results/crossnobis_tfce/cons_crossnobis_tfce_map_rois_fade.pdf){#fig:cons_crossnobis_map width=16cm}

# References
