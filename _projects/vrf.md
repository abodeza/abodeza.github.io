---
layout: page
title: Predictive Refrigerant Leak Modelling in VRF Systems
description: Project Overview
img: assets/img/F24 Predictive LEak Modeling VRF Final Review 6.jpeg
importance: 1
category: work
related_publications: false
---

# ML-Based Refrigerant Leak Detection in VRF HVAC Systems

## Context and Motivation
Variable refrigerant flow (VRF) systems offer high efficiency but their distributed piping makes refrigerant leaks hard to detect. Traditional methods, such as installing pressure sensors at every branch, are invasive, costly, and can introduce new leak points. We partnered with a clean energy consultant and a state energy authority to build a non-intrusive, data-driven solution.

## Challenge
A commercial facility’s VRF system logs hundreds of diagnostic parameters across five indoor units and a shared outdoor unit. Key questions:
- Which signals indicate the early stages of a leak?
- How do we distinguish natural operating variation from actual refrigerant loss?
- Can we achieve real-time detection without adding sensors?

## Approach

### Data exploration and feature selection  
- Applied principal component analysis to reduce 200+ parameters to a handful of principal components  
- Ran pairwise correlation analysis to confirm which raw signals drive those components  

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/pca_results.png" title="Parameters Identified via PCA" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Correlation matrix showing relationships among top parameters used for feature selection.
</div>

### Model development  
- Built a baseline model for normal operation  
- Simulated under-charge (10–20 percent refrigerant loss) scenarios  
- Trained a fully connected neural network to classify normal vs. under-charged states  

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/VRF_CM.png" title="Confusion Matrix" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Model performance on simulated leak scenarios, showing high detection rate and low false-alarm rate.
</div>

## Results
- Achieved over 95 percent recall on under-charge events  
- Maintained false-alarm rate below 5 percent  
- Demonstrated real-time inference capability on edge devices  

## Impact and Next Steps
- Enabled proactive maintenance without new hardware  
- Integrated into a demo application for regulatory reporting  
- Next steps: fine-tune thresholds per site, expand to multi-facility deployments  

## My Role
- Led end-to-end modeling pipeline  
- Engineered preprocessing and feature selection workflows  
- Designed, trained, and validated the neural network  
- Co-authored demo application supporting preventive maintenance  

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/F24 Predictive LEak Modeling VRF Final Review 6.jpeg" title="Team Award" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Team awarded “Outstanding Technical Progress in One Semester.”
</div>