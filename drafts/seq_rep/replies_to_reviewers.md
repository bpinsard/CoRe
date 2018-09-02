Dear Basile,

Thank you for submitting your article "Consolidation alters motor sequence-specific distributed representations" for consideration by eLife. Your article has been reviewed by 3 peer reviewers, and the evaluation has been overseen by Tim Verstynen, the Reviewing Editor and Rich Ivry, the Senior Editor. The following individuals involved in review of your submission have agreed to reveal their identity: Timothy Verstynen (Reviewer #1); Atsushi Yokoi (Reviewer #2).

The reviewers have discussed the reviews with one another and the Reviewing Editor has drafted this decision to help you prepare a revised submission. We hope you will be able to submit the revised version within two months.

SUMMARY

The present work examines motor skill learning by using multi-voxel pattern analysis to compare patterns of the brain activity elicited by two well-trained sequences that were consolidated by at least one overnight sleep and two novel sequences. The key findings were that, while there was some overlap between brain regions in which the two learning stages of sequences were represented (e.g.., bilateral premotor and parietal regions), there were regions where the sequences of either one of the stages were more strongly represented, especially around the sub-cortical structures. These sub-cortical structures included, for example, some parts of bilateral hippocampi, ipsilateral cerebellar cortex, and ipsilateral caudate, where the new sequences were more strongly represented. On the other hand, in the areas including bilateral sensorimotor putamen and thalamus the consolidated sequences were more strongly represented, suggesting that neural substrate storing sequence information drastically changes over the early consolidation process.

ESSENTIAL REVISIONS:

All three reviewers highlighted major concerns that fall under seven general themes.

1. Performance or representational differences?
The behavioral results clearly show that the performance (both speed and accuracy) is different between the trained and untrained sequences. Thus the differences in multivariate patterns could be driven by differences in performance rather than differences in encoding. There is (at least) one way to look at this issue: The behavioral performance shown in Figure 1 reveals that performance for the untrained sequences steadily improves across runs and appears to reach an asymptote half-way through the experiment. If the difference in mutlivariate patterns is truly related to long-term consolidation, as opposed to being the consequent result of changes in performance, then comparing the last 8 blocks of the trained vs. untrained sequences should replicate the key results (i.e., Figure 3). In contrast, comparing the last 8 blocks of the untrained sequences to the first 8 blocks of the of the untrained sequences should not reveal a similar set of clusters. However, if this split-half comparison on the untrained sequences produces qualitatively similar maps as the trained vs. untrained comparison, then it would strongly suggest that the differences in multivariate distances is driven largely by performance effects.

TODO: compute contrast on second run only

2. Elaboration of representational distances.
The true power of the RSA approach is that you can directly measure the distinctiveness of representations across conditions. Yet it is used mainly here as an alternative to traditional decoding methods. It would be nice for the authors to show the representational distances across sequences in key regions (e.g., cortical motor clusters, striatum, cerebellum). This would give the reader a sense of how training may be altering sequence-related representations.

TODO: whut?? run ROI analysis??

3. Distinguishing between signal amplitude and representational discriminability.
The map of clusters that discriminate between any of the four sequences (Figure 2), reveals a pretty standard sensorimotor network. Is RSA discriminability really driven by regions with significant task-related BOLD responses as estimated from traditional univariate GLM maps (in contrast to true differences in the covariance pattern of local voxels)? How well does the discriminability of the searchlight correlate with local task-related activity maps from univariate GLM? If they are correlated, how can you distinguish between searchlight results just being the result of local signal-to-noise differences from results driven by true differences in encoding?
Related to this was concern that, even after accounting for the potential first-finger effect, there remains potential differences in overall activity levels across the new and old sequences. Can this be addressed, especially given the differences in behavioural performance between new and consolidated sequences. How much were the overall activities different across the blocks and sequences?). Note that the pair-wise dissimilarities between fingers changed quite a bit in different activation levels (= tapping frequencies) (see Fig. 4a), indicating that the direct comparison between thumb/index distance in one tapping frequency and index/little one in another frequency would not be meaningful. The authors could try, for instance, using multivariate pattern angle, which is less affected by pattern scaling (Walther et al., 2016), or assessing the XOR (disjunctive union) of the prevalence of cross-validated distances for the consolidated and the new sequences, avoiding the direct comparison between them.

TODO: run GLM analysis

4. Elaboration of learning mechanisms & dynamics.
The authors make an extensive effort in the Introduction and Discussion to try to link these results to hippocampus. However, there is not a direct assessment of the role of the hippocampus to the rest of the network. There is only the observation of a cluster in each hippocampus that reliably distinguishes between the trained and untrained sequences, not an analysis that shows this is driving the rest of the changes in encoding in other areas.

The approach that we adopted is a data-driven mapping technique that explores the localized representations of information using an alternative metric and as such does not model interactions between regions. The results obtained show the stronger implication in the initial learning phase of the hippocampus, that we discuss in relation to previous studies with conventional measures.

5. Issues with behavioral data:
One reviewer noted that you have referenced a recent paper that shows movement rate differences that are performed with a single digit do not appreciably alter classification results. However, there are likely speed accuracy tradeoff differences between the samples, which may bias classification comparisons.
Moreover, there is no evidence here for consolidation. While there is a significant difference between condition for the trained and novel sequences, the trained were reactivated prior to imaging. Might the advantage for trained have come from the warm-up session (which was not given for the novel)? Performance for the novel sequences becomes similar to the TSeqs after block 12, which corresponds to 60 trials of practice for either novel sequence. The performance distinction between conditions may be entirely driven by the warmup period. What did performance look like for the trained sequences at the very start of this reactivation/warmup period?

TODO: compute consolidation gains between day1/day2 and day3??
compute sequence duration difference between 5 first sequences of block1 of day3-retests and first blocks of new sequences. even if not really comparable

6. Controlling for first-finger effects.
As you note in the Discussion, the pattern discriminability between the sequences starting with different fingers might reflect a "first-finger effect", where the discriminability of two sequences is almost solely driven by which finger was moved the first, not by the sequential order per se. This also applies to the pattern dissimilarity between the two new sequences (1-2-4-3-1 and 4-1-3-2-4) which, in contrast to the two consolidated sequences that had the same first finger (1-4-2-3-1, and 1-3-2-4-1), had different first fingers. Without accounting for this potential confound, the comparison made between the "new/new" and the "old/old" dissimilarities is hard to interpret, as it is unclear whether we are comparing between genuine sequence representations, or between sequence representation and individual finger representation.

already discussed, it is impossible to account for?

7. Clarification of methods.
How did participants know that they made an error? Were they explicitly told a key press was incorrect? Or did they have to self-monitor performance during training? Was the same button box apparatus used during scanning as in training? Was the presentation of sequence blocks counterbalanced during scanning? How many functional runs were performed? Is the classification performance different between runs?

- No feedback was given to the subject during practice, they were instructed that on

Additional detail is needed regarding the correction of motion artefact. Starting on line 505, the authors state that BOLD signal was further processed by detecting intensity changes corresponding to "large motion". What additional processing was performed? What was considered large motion? Was motion DOFs included in the GLM? Are there differences in motion between sequences and between sequence conditions? More information is needed on the detrending procedure. Is there evidence that detrending worked similarly between the different time windows between spikes? How were spikes identified? What happened to the spikes? In general, what software was used for the imaging analysis? Just PyMVPA?
The description of the analysis on lines 184-187 should be reworded. Is this describing how you analyze the difference in the consolidated and novel sequence discriminability maps? But how is this a conjunction? A conjunction reflects the overlap between 2 contrasts, and in this case what we are looking at is a difference. Related to this, there are different types of conjunctions. Please provide more details, as conjunctions can inflate significance. What software was used and how were the thresholds set for the conjunction?



MINOR POINTS (taken from the reviews):

The flat map images are difficult to process. Please consider the following: (1) For the hemispheres, use orthogonal views. Have lateral, dorsal and medial images, instead of rotating laterally the dorsal images; (2) if possible, smooth the subcortical activation pixels so they match the cortical surfaces; (3) reduce the shading of the subcortical anatomy, or increase the brightness of the activation so it is easier to see (particularly the magenta pixels

Although the authors stated that the results presented in this study is a part of a larger research program, I think, it would be necessary to briefly describe what the main aim of the "larger research program" is and how it is different from the particular results presented in this paper.

Some tempering of 'expertise' and 'automatization' seems warranted here given that the data describe learning over less than 100 trials/sequence. Similarly, the description of the results as 'networks' needs some tempering or clarification. The results do not reflect networks, but regional searchlight-derived effects performed over the cortex and subcortical ROIs.

Why are the results in Figures 2, 3, S1, & S2 not shown as H-statistics (i.e., the crossnobis statistic). The cross-validated nature of this metric means it is a statistical estimator with meaningful units. I'm not exactly sure what the z-scores are showing.

It is not clear from the Methods precisely how the different sequences were cued during the imaging sessions. This is important as it will help to clarify why dorsal and ventral stream clusters are present in both the all sequence discriminability test (Figure 2) and the trained vs. untrained sequence discriminability test (Figure 3). (As described in the Discussion, lines 237-250).

The use of TFCE is pretty liberal as a multiple comparisons measure. How well do the results hold up when using FDR? (This can be a supplemental analysis, but it is important).

Line 15: What does it mean "in link" with the hippocampus?

Lines 15-18: The sentence starting with "Yet, the functional..." is a run-on.

Line 42: "wake" should be "wakefulness"

Lines 120-123: The use of RSA is highlighted as a contrast to traditional decoding approaches. However, the Yokoi paper cited in the previous paragraph (line 97) uses this same approach.

The supplementary tables are really hard to parse. Consider using standard naming conventions as opposed to operational variable names.

I think it would be necessary to discuss the result of the discriminability map for each learning stage, which is now put in the Supplementary material, as it is closely related to how we interpret the re-organization result. Particularly, I am very much interested in what caused so widespread discriminability of newly-acquired sequences in subcortical structures, especially around the cerebellum and the brainstem.

It would be helpful if the authors could add a little more detail about how the cross-validated Mahalanobis distances were calculated, such as whether it has been cross-validated run-wise or block-wise. Similarly, a little more detail would help a lot at the statistics part in the Method, especially on the distance measure.

Lines 232-236 and 270-275: This reasoning looks somewhat contradicting with what was stated in the previous part that sequences are represented around M1/S1 (lines 227-230). There is no prior reason to think that genuine sequence representation would reflect the property of the single finger representation (i.e., the thumb has a distinct pattern). Maybe a little more elaborated explanation would be needed here.

This is slightly nitpicking, but matching the first finger alone may not guarantee that the observed pattern discriminability is actually reflecting genuine sequence representation. The weight decay on each press might be more like exponential when examined by high-frequency band activity of LFP (see Hermes et al., J Neurosci, 2012), meaning that there might be the "second-finger" effect, although this may be very small compared to the first-finger effect. Just a heads-up.

It seems that most of the detailed description of experimental design/tasks described in the Method is not directly related to the particular result presented in the manuscript. Please consider re-structure them into what is directly relevant to the current results, and what is not. Perhaps the complete description could go into the Supplementary material?

I am curious to see the result of instruction-phase activity patterns.

Please re-summarise the demographic info for the survived subjects.

De-Bruijn cycle: A brief explanation of what De-Bruijn cycle is would be helpful.

What was the typical trial duration in the scanner?

Lines 511-513: Reference about the relationship between columnar scale and voxel size seems messing?

Line 542: Probably adding "volume-based" and "surface-based" would be easier to understand.

Figure1: What do the error-bars represent?

It would be helpful to put labels for the subcortical structures in the Supplementary figures, as well.

Some of the pre-print articles cited seems to have already been published in some peer-reviewed journal. Please consider updating. Or was this for the openness/accessibility's sake?

Line 206: What is adaptive cluster correction?

Line 211 -212: SMA is a premotor area. Maybe include something like medial and lateral premotor areas?

Line 222: Please consider changing out 'critical role' for something less definite. Further, an alternative explanation could be that SMC reflects different patterns of movement, and nothing to do with a representation of the sequence.

Line 241-242: Please change 'likely reflecting' to something less definite. The description of visual cortex is not the focus of the paper, and discussion should reflect this point. Otherwise it is overstating a reverse inference.

Line 266: Please change 'likely reflecting' to something less definite. 'This pattern suggests' or something like that is more appropriate.

Line 297:What is meant by 'novel explicitly acquired motor sequence knowledge'? What is novel knowledge? New?

Lines 303-306: This sentence on striatum involvement should be reworded. It is unclear what is meant by 'specifies the changes in activity in these regions in the course if MSL...'. Further, please change 'mastered skill' to something less definite, like 'consolidated'.

Line 308: Please change 'uncover that this shift in activity purports a genuine' to something less definite. Something like: 'Is consistent with animal studies showing a reorganization...'

Line 373-375: This statement is unclear. Are the authors suggesting that changes in regional representation magnitude lead to consolidation? These results do not show this.
