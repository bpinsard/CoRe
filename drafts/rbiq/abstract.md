---
layout: article
title: Mapping of Motor Sequence Representations in the Human Brain
tags: [example, citation]
bibliography: abstract_rbiq.bib
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
organization:
- id: 1
  name: Functional Neuroimaging Unit, Centre de Recherche de l'Institut Universitaire de Gériatrie de Montréal
  address: Montreal, Quebec, Canada
  url: http://unf-montreal.ca
- id: 2
  name: Sorbonne Universités, UPMC Univ Paris 06, CNRS, INSERM, Laboratoire d’Imagerie Biomédicale (LIB) 
  address: 75013, Paris, France
  url: http://lib.upmc.fr
---

Motor skills recruit an extended network of cerebral and spinal regions, which involvement evolves differently across learning stages [@dayan_neuroplasticity_2011; @doyon_contributions_2009]. Motor sequence learning causes a subset of this network to specialize for a specific learned sequence leading to performance improvement and long term consolidation. Here we aimed at finding regions that encode novel or consolidated sequence-specific representations through execution evoked BOLD patterns across the whole-brain.

### Methods

Subjects (n=16) were asked to repeatedly perform 5 repetitions of 4 different 5-elements sequences (eg. 1-4-2-3-1) as instructed on the screen in a rapid block design while BOLD fMRI scan was acquired. Among the 4 sequences, 2 were completely new to the subject, 1 was trained 2 days before, and 1 was either trained one day before or the same day.
The 64 trials were split into 2 scans to test across-scan generalization.

Preprocessed whole-brain gray-matter Blood-Oxygen-Level-Dependent(BOLD) signal was analyzed with Multivariate Patterns Classification (MVPC). Gaussian Naive Bayes classifier Searchlight[@kriegeskorte_information-based_2006] with 4 sequences was performed at the subject-level, hence mapping individual brain-wise cross-validated accuracy.

Group searchlight map was computed using mass-univariate one-sample T-test to find regions in which accuracy consistently exceeded the chance level (25%) across subjects.

### Results

While subjects exhibited differences in searchlight maps, an extended common network with BOLD patterns discriminating the 4 different sequences was identified (fig.1). Activity patterns predictive of the sequence were located in:

- bilateral posterior parietal (PP), premotor (PM), dorso-lateral prefontal (DLPF) cortex, caudate nuclei (Caud), putamen (Pu) and hippocampus (Hc),
- contralateral supplementary motor area (SMA) presupplementary motor area (preSMA) and insular cortex (Ins),
- ipsilateral cerebellum (Cer) and lateral temporal (lTemp) cortex

![Group t-values threshold (p<.001) of classification accuracy during execution](../../results/slmaps/slmap_loso_highlights.png)

### Discussion and conclusion

A characteristic network known to participate in motor sequence production [@doyon_reorganization_2005; @doyon_contributions_2009; @albouy_hippocampus_2013] emerged of this pattern analysis and partly replicates [@wiestler_skill_2013] with further insights into subcortical nuclei and cerebellum implications. The main difference resides in the absence of representation in M1, which can be attributed to the fact that sequences were performed with left hand only in our study.

Distinctive patterns in subcortical regions suggest their implication in consolidation processes and potential sleep-related transfer of representations that needs to be further explored.

These results bring further insights into our understanding of the neural processes at play in procedural learning in healthy individuals.

### References
