Dear Basile,

Thank you for submitting your article "Consolidation alters motor sequence-specific distributed representations" for consideration by eLife. Your article has been reviewed by 3 peer reviewers, and the evaluation has been overseen by Tim Verstynen, the Reviewing Editor and Rich Ivry, the Senior Editor. The following individuals involved in review of your submission have agreed to reveal their identity: Timothy Verstynen (Reviewer #1); Atsushi Yokoi (Reviewer #2).

The reviewers have discussed the reviews with one another and the Reviewing Editor has drafted this decision to help you prepare a revised submission. We hope you will be able to submit the revised version within two months.

SUMMARY

The present work examines motor skill learning by using multi-voxel pattern analysis to compare patterns of the brain activity elicited by two well-trained sequences that were consolidated by at least one overnight sleep and two novel sequences. The key findings were that, while there was some overlap between brain regions in which the two learning stages of sequences were represented (e.g.., bilateral premotor and parietal regions), there were regions where the sequences of either one of the stages were more strongly represented, especially around the sub-cortical structures. These sub-cortical structures included, for example, some parts of bilateral hippocampi, ipsilateral cerebellar cortex, and ipsilateral caudate, where the new sequences were more strongly represented. On the other hand, in the areas including bilateral sensorimotor putamen and thalamus the consolidated sequences were more strongly represented, suggesting that neural substrate storing sequence information drastically changes over the early consolidation process.

ESSENTIAL REVISIONS:

All three reviewers highlighted major concerns that fall under seven general themes.

1. Performance or representational differences?
The behavioral results clearly show that the performance (both speed and accuracy) is different between the trained and untrained sequences. Thus the differences in multivariate patterns could be driven by differences in performance rather than differences in encoding. There is (at least) one way to look at this issue: The behavioral performance shown in Figure 1 reveals that performance for the untrained sequences steadily improves across runs and appears to reach an asymptote half-way through the experiment. If the difference in mutlivariate patterns is truly related to long-term consolidation, as opposed to being the consequent result of changes in performance, then comparing the last 8 blocks of the trained vs. untrained sequences should replicate the key results (i.e., Figure 3). In contrast, comparing the last 8 blocks of the untrained sequences to the first 8 blocks of the of the untrained sequences should not reveal a similar set of clusters. However, if this split-half comparison on the untrained sequences produces qualitatively similar maps as the trained vs. untrained comparison, then it would strongly suggest that the differences in multivariate distances is driven largely by performance effects.

*TODO: compute contrast on second run only : doesn't give the same results
compute contrast of new sequences between run1 and run2 : 
- we go from 6-folds to 1-fold, huge loss of statistical power
- among the 6-folds, 4 folds are across the beginning and end of task, thus the measure mainly reflects what patterns are stable across the runs independently of the behavioral changes.
- The behavioral difference is a limitation common to all design comparing trained and untrained task? The previous articles investigating trained vs. untrained sequence representation just omitted to present the behavioral performance evolution whithin run.
*

2. Elaboration of representational distances.
The true power of the RSA approach is that you can directly measure the distinctiveness of representations across conditions. Yet it is used mainly here as an alternative to traditional decoding methods. It would be nice for the authors to show the representational distances across sequences in key regions (e.g., cortical motor clusters, striatum, cerebellum). This would give the reader a sense of how training may be altering sequence-related representations.

*TODO: whut?? run ROI analysis?? distance matrix?*

3. Distinguishing between signal amplitude and representational discriminability.
The map of clusters that discriminate between any of the four sequences (Figure 2), reveals a pretty standard sensorimotor network. Is RSA discriminability really driven by regions with significant task-related BOLD responses as estimated from traditional univariate GLM maps (in contrast to true differences in the covariance pattern of local voxels)? How well does the discriminability of the searchlight correlate with local task-related activity maps from univariate GLM? If they are correlated, how can you distinguish between searchlight results just being the result of local signal-to-noise differences from results driven by true differences in encoding?


*TODO: run GLM analysis*

Related to this was concern that, even after accounting for the potential first-finger effect, there remains potential differences in overall activity levels across the new and old sequences. Can this be addressed, especially given the differences in behavioural performance between new and consolidated sequences. How much were the overall activities different across the blocks and sequences?). Note that the pair-wise dissimilarities between fingers changed quite a bit in different activation levels (= tapping frequencies) (see Fig. 4a *which article?*), indicating that the direct comparison between thumb/index distance in one tapping frequency and index/little one in another frequency would not be meaningful. The authors could try, for instance, using multivariate pattern angle, which is less affected by pattern scaling (Walther et al., 2016), or assessing the XOR (disjunctive union) of the prevalence of cross-validated distances for the consolidated and the new sequences, avoiding the direct comparison between them.

*TODO:*

4. Elaboration of learning mechanisms & dynamics.
The authors make an extensive effort in the Introduction and Discussion to try to link these results to hippocampus. However, there is not a direct assessment of the role of the hippocampus to the rest of the network. There is only the observation of a cluster in each hippocampus that reliably distinguishes between the trained and untrained sequences, not an analysis that shows this is driving the rest of the changes in encoding in other areas.

*The approach that we adopted is a data-driven mapping technique that explores the localized representations of information using an alternative metric and as such does not model interactions between regions. The results obtained show the stronger implication in the initial learning phase of the hippocampus, that we discuss in relation to previous studies with conventional measures.
As per the reviewers comment below, we removed references of our results as network, as we are not conducting any sort of network analysis, such as connectivity.
We hope that this clarification will improve the reader's understanding of our results.*

5. Issues with behavioral data:
One reviewer noted that you have referenced a recent paper that shows movement rate differences that are performed with a single digit do not appreciably alter classification results. However, there are likely speed accuracy tradeoff differences between the samples, which may bias classification comparisons.
Moreover, there is no evidence here for consolidation. While there is a significant difference between condition for the trained and novel sequences, the trained were reactivated prior to imaging. Might the advantage for trained have come from the warm-up session (which was not given for the novel)? Performance for the novel sequences becomes similar to the TSeqs after block 12, which corresponds to 60 trials of practice for either novel sequence. The performance distinction between conditions may be entirely driven by the warmup period. What did performance look like for the trained sequences at the very start of this reactivation/warmup period?

*TODO: compute consolidation gains between day1/day2 and day3??
compute sequence duration difference between 5 first sequences of block1 of day3-retests and first blocks of new sequences. even if not really comparable*

6. Controlling for first-finger effects.
As you note in the Discussion, the pattern discriminability between the sequences starting with different fingers might reflect a "first-finger effect", where the discriminability of two sequences is almost solely driven by which finger was moved the first, not by the sequential order per se. This also applies to the pattern dissimilarity between the two new sequences (1-2-4-3-1 and 4-1-3-2-4) which, in contrast to the two consolidated sequences that had the same first finger (1-4-2-3-1, and 1-3-2-4-1), had different first fingers. Without accounting for this potential confound, the comparison made between the "new/new" and the "old/old" dissimilarities is hard to interpret, as it is unclear whether we are comparing between genuine sequence representations, or between sequence representation and individual finger representation.

*The reviewers are right that this is a limitation for a part of the current results as we discuss in the manuscript.
However, the design of the present task is different of the one that uncovered this effect in that each sequence is executed separately in that the execution is uninterrupted and should thus theoretically reduce considerably this initiating effect.
Moreover, the main results of our article are the differences in striatal and hippocampal representations, and these regions are not expected to show strong and reliable somatotopy for individual finger movements.
Therefore, we do not expect a pattern associated with the first finger in these regions.
While this effect is known to impact the motor cortical regions, it would be expected to positively bias the new sequences discriminability, but no cortical regions show higher discriminability than for consolidated sequences.
At most, this effect could have caused some false-negative results in the contrast between conditions.
As such we still believe that these results are of interest for the understanding of motor sequence learning.*


7. Clarification of methods.
How did participants know that they made an error? Were they explicitly told a key press was incorrect? Or did they have to self-monitor performance during training? Was the same button box apparatus used during scanning as in training? Was the presentation of sequence blocks counterbalanced during scanning? How many functional runs were performed? Is the classification performance different between runs?

*The reviewers are right that this description is missing. The following sentences have been added to the method: "No feedback was provided to the participant regarding their performance.
Prior to the practice, they were given the instruction that if they realized that they made an error, they should not try to correct it but instead restart practicing from the beginning of the sequence."*

Additional detail is needed regarding the correction of motion artefact. Starting on line 505, the authors state that BOLD signal was further processed by detecting intensity changes corresponding to "large motion". What additional processing was performed? What was considered large motion? Was motion DOFs included in the GLM? Are there differences in motion between sequences and between sequence conditions? More information is needed on the detrending procedure. Is there evidence that detrending worked similarly between the different time windows between spikes? How were spikes identified? What happened to the spikes? In general, what software was used for the imaging analysis? Just PyMVPA?

*Description of the detrending procedure was extended with the following sentence:*

The description of the analysis on lines 184-187 should be reworded. Is this describing how you analyze the difference in the consolidated and novel sequence discriminability maps? But how is this a conjunction? A conjunction reflects the overlap between 2 contrasts, and in this case what we are looking at is a difference. Related to this, there are different types of conjunctions. Please provide more details, as conjunctions can inflate significance. What software was used and how were the thresholds set for the conjunction?

*We agree with the reviewers that this sentence needs to be clarified. It is in fact a union of two conjunctions ((Consolidate>New)∩(Consolidated>0))∪((New>Consolidated)∩(New>0)). As we are using minimum statistics conjunctions, no inflation is expected. We rephrased it as such: ""*



MINOR POINTS (taken from the reviews):

The flat map images are difficult to process. Please consider the following: (1) For the hemispheres, use orthogonal views. Have lateral, dorsal and medial images, instead of rotating laterally the dorsal images; (2) if possible, smooth the subcortical activation pixels so they match the cortical surfaces; (3) reduce the shading of the subcortical anatomy, or increase the brightness of the activation so it is easier to see (particularly the magenta pixels

Although the authors stated that the results presented in this study is a part of a larger research program, I think, it would be necessary to briefly describe what the main aim of the "larger research program" is and how it is different from the particular results presented in this paper.

Some tempering of 'expertise' and 'automatization' seems warranted here given that the data describe learning over less than 100 trials/sequence.

*The reviewers are right that 'expertise' is not truly applicable to our data. As the consolidated sequences have been trained over 12 blocks x 14 repetitions = 168 sequences, not accounting for the retests, ... TODO*

Similarly, the description of the results as 'networks' needs some tempering or clarification. The results do not reflect networks, but regional searchlight-derived effects performed over the cortex and subcortical ROIs.

*We agree with the reviewers that we are not conducting here a network analysis (eg. connectivity) and that our approach reveals sets of regions which carries signal discriminating sequences. We changed the manuscript to remove the descriptions of our results as networks.*

Why are the results in Figures 2, 3, S1, & S2 not shown as H-statistics (i.e., the crossnobis statistic). The cross-validated nature of this metric means it is a statistical estimator with meaningful units. I'm not exactly sure what the z-scores are showing.

*TODO: check if it maps well and change figures*

It is not clear from the Methods precisely how the different sequences were cued during the imaging sessions. This is important as it will help to clarify why dorsal and ventral stream clusters are present in both the all sequence discriminability test (Figure 2) and the trained vs. untrained sequence discriminability test (Figure 3). (As described in the Discussion, lines 237-250).

*In the "Material and methods/Procedures and task" section we slightly modified the following sentences to better explain the task design: 
"During this task, each block was composed of an instruction period of 4 seconds during which the sequences to be performed was visually displayed as a series of 5 numbers (e.g. 1-4-2-3-1), that could easily be remembered by the participant.
The latter was then followed by an execution phase triggered by the removal of the instruction stimuli and the appearance of a green cross on the screen.
In each block, participants performed 5 times the same sequence (or a maximum of 25 key-presses), before being instructed to stop and rest when the red cross was displayed."*

The use of TFCE is pretty liberal as a multiple comparisons measure. How well do the results hold up when using FDR? (This can be a supplemental analysis, but it is important).

Line 15: What does it mean "in link" with the hippocampus?

*We tried to rephrase that imprecise wording within the Elife abstracts word limit: "...with initial hippocampal contribution."*

Lines 15-18: The sentence starting with "Yet, the functional..." is a run-on.

*Hoping we understood the reviewer's comment correctly, we rephrased the sentence as such: Yet, the functional significance of these activity level changes remains ambiguous as they convey the evolution of both sequence-specific knowledge and unspecific task expertise.*

Line 42: "wake" should be "wakefulness"

*changed*

Lines 120-123: The use of RSA is highlighted as a contrast to traditional decoding approaches. However, the Yokoi paper cited in the previous paragraph (line 97) uses this same approach.

*This sentence does not specifically state that RSA has never been used to investigate sequence representations, but that previous studies contrasting learned and new sequences were based on classifier-based approaches.*

The supplementary tables are really hard to parse. Consider using standard naming conventions as opposed to operational variable names.



I think it would be necessary to discuss the result of the discriminability map for each learning stage, which is now put in the Supplementary material, as it is closely related to how we interpret the re-organization result. Particularly, I am very much interested in what caused so widespread discriminability of newly-acquired sequences in subcortical structures, especially around the cerebellum and the brainstem.

*TODO: *

It would be helpful if the authors could add a little more detail about how the cross-validated Mahalanobis distances were calculated, such as whether it has been cross-validated run-wise or block-wise. Similarly, a little more detail would help a lot at the statistics part in the Method, especially on the distance measure.

*We added the following sentence to the "Cross-validated multivariate distance" section: "Cross-validation was done accross the four imaging runs."*

Lines 232-236 and 270-275: This reasoning looks somewhat contradicting with what was stated in the previous part that sequences are represented around M1/S1 (lines 227-230). There is no prior reason to think that genuine sequence representation would reflect the property of the single finger representation (i.e., the thumb has a distinct pattern). Maybe a little more elaborated explanation would be needed here.

*TODO??*

This is slightly nitpicking, but matching the first finger alone may not guarantee that the observed pattern discriminability is actually reflecting genuine sequence representation. The weight decay on each press might be more like exponential when examined by high-frequency band activity of LFP (see Hermes et al., J Neurosci, 2012), meaning that there might be the "second-finger" effect, although this may be very small compared to the first-finger effect. Just a heads-up.

*We agree with the reviewers that it cannot be ruled-out that sequence are discriminated by single finger temporal position.*

It seems that most of the detailed description of experimental design/tasks described in the Method is not directly related to the particular result presented in the manuscript. Please consider re-structure them into what is directly relevant to the current results, and what is not. Perhaps the complete description could go into the Supplementary material?

I am curious to see the result of instruction-phase activity patterns.

*TODO*

Please re-summarise the demographic info for the survived subjects.

*We added the following information  to the manuscript: "(14 females, $25\pm6.2$yr.)"*

De-Bruijn cycle: A brief explanation of what De-Bruijn cycle is would be helpful.

*We changed the sentence to : "This prevents the systematic leakage of BOLD activity patterns between blocks in this rapid design as across blocks each sequence is preceded and followed by any sequence the same number of time within each run."*

What was the typical trial duration in the scanner?

*The trial duration varies depending on the sequence across time. To get the block duration, the sequence durations presented in the figure 1 can be multiplied by 5 (the number of sequence per block), which give approximately 5-10sec.*

Lines 511-513: Reference about the relationship between columnar scale and voxel size seems messing?

*this have been removed*

Line 542: Probably adding "volume-based" and "surface-based" would be easier to understand.

*added*

Figure1: What do the error-bars represent?

*The error-bars represent the standard deviation of the mean of the sequence duration across participants.*

It would be helpful to put labels for the subcortical structures in the Supplementary figures, as well.

*TODO*

Some of the pre-print articles cited seems to have already been published in some peer-reviewed journal. Please consider updating. Or was this for the openness/accessibility's sake?

*The preprint that have been published during the course of manuscript writing have been updated, thanks for pointing it.*

Line 206: What is adaptive cluster correction?

*The reviewers are right that this term is used in the results and discussion and only shortly explained in the methods.
We added the following to the method section: "Our BOLD sampling of sparse gray-matter coordinates induces large differences in the sizes of the structures being investigated, and thus conventional cluster corrections based on whole-brain cluster-size distribution would have biased the results.
Therefore, TFCE enabled a locally adaptive statistics and cluster size correction that particularly fitted this sampling."*

Line 211 -212: SMA is a premotor area. Maybe include something like medial and lateral premotor areas?

*changes done accordingly*

Line 222: Please consider changing out 'critical role' for something less definite. Further, an alternative explanation could be that SMC reflects different patterns of movement, and nothing to do with a representation of the sequence.

*changed to: "contribute to the acquisition of"*

Line 241-242: Please change 'likely reflecting' to something less definite. The description of visual cortex is not the focus of the paper, and discussion should reflect this point. Otherwise it is overstating a reverse inference.

*changed to: "which could reflect the"*

Line 266: Please change 'likely reflecting' to something less definite. 'This pattern suggests' or something like that is more appropriate.

*changed*

Line 297:What is meant by 'novel explicitly acquired motor sequence knowledge'? What is novel knowledge? New?

*changed*

Lines 303-306: This sentence on striatum involvement should be reworded. It is unclear what is meant by 'specifies the changes in activity in these regions in the course if MSL...'. Further, please change 'mastered skill' to something less definite, like 'consolidated'.

*"specifies" changed to "give specificity" and "mastered skill" to "consolidated skill"*

Line 308: Please change 'uncover that this shift in activity purports a genuine' to something less definite. Something like: 'Is consistent with animal studies showing a reorganization...'

*changed to "...this shift in activity is consistent with animal studies showing a reorganization at the neuronal level in these regions."*


Line 373-375: This statement is unclear. Are the authors suggesting that changes in regional representation magnitude lead to consolidation? These results do not show this.

*This statement was nuanced by changing "determined" to "accompanied".*
