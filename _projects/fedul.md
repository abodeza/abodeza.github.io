---
layout: page
title: A Bi-Objective Optimization Approach for Enhancing FedUL
description: Algorithm Analysis
img: assets/img/vrf_diagram.png
importance: 1
category: work
related_publications: false
---

## Brief  
In collaboration with a clean energy consulting firm and a state energy authority, our team developed a machine learning-based tool to detect refrigerant leaks in Variable Refrigerant Flow (VRF) HVAC systems. While VRF systems are highly efficient, they are prone to leaks due to their complex piping. Using real diagnostic data from a commercial facility, we identified key indicators of leak behavior, built a baseline model for normal operation, and tested against simulated leak scenarios.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vrf_diagram.png" title="VRF Diagram" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The diagram above shows a simple 5-indoor unit VRF system with heating and cooling capabilities.
</div>

## Problem  
In VRF systems with many indoor and outdoor units, detecting refrigerant leaks is difficult. Invasive methods—like placing pressure sensors at various pipe locations—can cause more leaks and are impractical in complex systems. As seen in the figure above, each indoor unit connects through its own set of pipes, making complete physical monitoring costly and unreliable. This calls for a smarter, non-intrusive approach.

## Solution  
We designed a non-invasive solution using existing system diagnostics. By applying Principal Component Analysis (PCA) and correlation analysis, we reduced hundreds of parameters to a core set that reliably signals abnormal conditions. A fully connected neural network (FCNN) was then trained to classify normal and undercharged states. As seen in our evaluation results, the model accurately detected leak scenarios with minimal false alarms, enabling effective real-time monitoring without additional hardware.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/VRF_CM.png" title="Model Accuracy: Confusion Matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Confusion matrix illustrating model performance across simulated VRF system states, including normal and undercharged (10–20%) conditions.
</div>

## My Contribution  
I led the modeling pipeline, from selecting key system variables using PCA to designing and training the neural network. I engineered the preprocessing steps, handled large-scale diagnostic data, and validated the model’s performance across different leak conditions. I contributed in performing a correlation study to uncover meaningful system behaviors, helping guide sensor selection and model interpretability. This work directly contributed to the creation of a working demo application capable of supporting regulatory reporting and preventive maintenance.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/pca_results.png" title="Top Parameters Identified via PCA" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">
    This correlation matrix shows the correlation factors deduced for each pair of system variables which aided in feature selection.
</div>
