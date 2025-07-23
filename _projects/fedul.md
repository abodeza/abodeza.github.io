---
layout: page
title: A Bi-Objective Optimization Approach for Enhancing FedUL [WIP]
description: Algorithm Analysis
img: assets/img/fed_infographic.png
importance: 1
category: research
related_publications: false
---

Formal report can be found here: [pdf](/assets/pdf/ECSE_4964_Project_Report.pdf).

---

<div id="toc">
  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#our-contribution">Our Contribution</a></li>
    <li><a href="#proposed-algorithm">Proposed Algorithm</a></li>
    <li><a href="#empirical-results">Empirical Results</a></li>
    <li><a href="#conclusion-and-future-work"> Conclusion and Future Work</a></li>
    <li><a href="#references"> References</a></li>
    <li><a href="#appendix"> Appendix</a></li>
  </ul>
</div>

---




<h1 id="introduction">Introduction</h1>
<p>In this project, we aim to minimize the cost function associated with
multi-class classification tasks. Training local models in unsupervised
federated learning settings using decentralized datasets can encounter
numerous obstacles. These include the absence of a definitive objective
function for model evaluation. Unlike unsupervised learning, where
generative models like GANs or diffusion models excel at modeling data
distributions, federated learning contends with heterogeneous datasets.
This heterogeneity can prevent local models from acquiring generalizable
knowledge and complicate the aggregation of these models on the server
side.</p>
<h2 id="related-work">Related Work</h2>
<h4 id="fedul">FedUL</h4>
<p>is an algorithm designed to overcome the challenge of unclear
objectives in the unsupervised FL setting. It begins with the natural
segregation of data across clients, splitting each client’s dataset into
multiple subsets. Each subset is indexed relative to the client’s
dataset, serving as the surrogate label. This approach facilitates
training on surrogate data using a supervised task. Thus, the objective
becomes minimizing the loss function associated with predicting the
surrogate labels. The algorithm assumes we have access to the class
priors and a shared class-conditional distribution among clients. The
primary goal of FedUL is to use an injective transition function <span
class="math inline"><em>Q</em></span> to recover the optimal model from
the surrogate task, ensuring that the knowledge acquired is applicable
for classifying the actual data.</p>
<p>This goal is achieved by implementing Theorem 1 from the paper, which
establishes a method to correlate the posterior probabilities of the
real and surrogate data. Specifically, it assesses the likelihood of
each surragate dataset chosen per client. Which is approximated as the
number of features for that dataset over the total number of features
for the client. Then it maps posterior probability from the real space
to the space of the surrogate data. Then it utilizes the prior
probability of the real classes to assess the likelihood of a class’s
presence in a client’s dataset, effectively normalizing the posterior
probability predicted by the global model, which is the last matrix in
the unnormalized representation of the injective function. By optimizing
the surrogate classifier, the classifier for the actual task is
concurrently enhanced, supported by other assumptions detailed in the
paper.</p>
<h2 class="unnumbered" id="theorem-1">Theorem 1</h2>
<p>For each client <span
class="math inline"><em>c</em> ∈ [<em>C</em>]</span>, let <span
class="math inline"><em>η̄</em><sub><em>c</em></sub> : 𝒳 → <em>Δ</em><sup><em>M</em> − 1</sup></span>
and <span
class="math inline"><em>η</em> : 𝒳 → <em>Δ</em><sup><em>K</em> − 1</sup></span>
be the surrogate and original class-posterior probability functions,
respectively, where <span
class="math inline">(<em>η̄</em><sub><em>c</em></sub>(<em>x</em>))<sub><em>m</em></sub> = <em>p̄</em><sub><em>c</em></sub>(<em>ȳ</em> = <em>m</em> ∣ <em>x</em>)</span>
for <span class="math inline"><em>m</em> ∈ [<em>M</em>]</span>, and
<span
class="math inline">(<em>η</em>(<em>x</em>))<sub><em>k</em></sub> = <em>p</em>(<em>y</em> = <em>k</em> ∣ <em>x</em>)</span>
for <span class="math inline"><em>k</em> ∈ [<em>K</em>]</span>. Here,
<span class="math inline"><em>Δ</em><sup><em>M</em> − 1</sup></span> and
<span class="math inline"><em>Δ</em><sup><em>K</em> − 1</sup></span> are
the <span class="math inline"><em>M</em></span>-dimensional and <span
class="math inline"><em>K</em></span>-dimensional simplexes,
respectively.</p>
<p>Let <span
class="math inline"><em>π̄</em><sub><em>c</em></sub> = [<em>π̄</em><sub>1<em>c</em></sub>, …, <em>π̄</em><sub><em>M</em><em>c</em></sub>]<sup>⊤</sup></span>
and <span
class="math inline"><em>π</em> = [<em>π</em><sub>1</sub>, …, <em>π</em><sub><em>K</em></sub>]<sup>⊤</sup></span>
be vector forms of the surrogate and original class priors,
respectively. Let <span
class="math inline"><em>Π</em><sub><em>c</em></sub> ∈ ℝ<sup><em>M</em> × <em>K</em></sup></span>
be the matrix form of <span
class="math inline"><em>π</em><sub><em>m</em>, <em>k</em></sub><sup><em>c</em></sup> = <em>p</em><sub><em>m</em></sub><sup><em>c</em></sup>(<em>y</em> = <em>k</em>)</span>
. Then we have: <span class="math display">$$\bar{\eta}_c(x) =
Q_c(\eta(x); \pi, \bar{\pi_c}, \Pi_c)$$</span> where the unnormalized
version of the vector-valued function <span
class="math inline"><em>Q</em><sub><em>c</em></sub></span> is given by:
<span
class="math display"><em>Q̃</em><sub><em>c</em></sub>(<em>η</em>(<em>x</em>); <em>π</em>, <em>π̄</em><sub><em>c</em></sub>, <em>Π</em><sub><em>c</em></sub>) = <em>D</em><sub><em>π̄</em><sub><em>c</em></sub></sub> ⋅ <em>Π</em><sub><em>c</em></sub> ⋅ <em>D</em><sub><em>π</em></sub><sup>−1</sup> ⋅ <em>η</em>(<em>x</em>).</span>
Here, <span class="math inline"><em>D</em><sub><em>a</em></sub></span>
denotes the diagonal matrix with the diagonal terms being vector <span
class="math inline"><em>a</em></span>, and ‘<span
class="math inline">⋅</span>’ denotes matrix multiplication. <span
class="math inline"><em>Q</em><sub><em>c</em></sub></span> is normalized
by the sum of all entries, i.e., <span class="math inline">$(Q_c)_i =
\frac{(\tilde{Q}_{ec})_i}{\sum_j (Q_{c})_j}$</span>.</p>
<h1 id="our-contribution">Our Contribution</h1>
<h2 id="overview">Overview</h2>
<p>We introduce an enhanced algorithm designed to enhance the
performance of FedUL in terms of accuracy and robustness. Recognizing
FedUL’s effectiveness, our approach integrates FedUL with an additional
optimization phase utilizing supervised learning techniques. This
integration involves training local models using FedUL, aggregating
these models at the server, and subsequently implementing a bi-objective
optimization task trained on publicly accessible labeled data, aiming
for Pareto optimality. This approach has shown to improve FedUL’s
performance, especially in non-IID settings.</p>
<h2 id="significance">Significance</h2>
<p>Despite the scarcity of publicly accessible labeled datasets in many
areas, our method leverages any applicable available small-sized labeled
dataset in conjunction with a larger decentralized dataset. This
approach preserves the privacy of client data while enhancing the
model’s performance through bi-objective optimization techniques.</p>
<h2 id="integration-techniques">Integration Techniques</h2>
<p>To integrate the updates from local models with the supervised
model’s updates, we explored two methods:</p>
<h3 id="method-1-feedback-optimization">Method 1: Feedback
Optimization</h3>
<p>In this approach, we fine-tune the aggregated global model using the
labeled data retained at the server. This phase optimizes the objective
of the server’s model based on the labeled data. Following this, we
establish a feedback loop by redistributing the fine-tuned model back to
the clients. This loop allows the clients to further optimize the model
with their local data, enhancing the overall accuracy of the model.</p>
<h3 id="method-2-independent-enhancement">Method 2: Independent
Enhancement</h3>
<p>Alternatively, we execute the FedUL process to aggregate the local
models into a global model. Parallel to this, we optimize an independent
model, intialized uniformly with the global model, specifically on the
labeled data. The parameters learned from this supervised task are then
combined with the global model’s parameters. This combination is
regulated through a manually tuned hyper-parameter, effectively blending
the insights of both the local models and the supervised model. A
similar loop to method 1 is implemented.</p>
<h2 id="proposed-algorithm">Proposed Algorithm</h2>

<div class="algorithm-block">
  <strong>Algorithm 1:</strong> Federation of unsupervised learning (FedUL)<br>
  <b>Server Input:</b> initial \( f \), global step-size \( \alpha_g \), and global communication round \( R \)<br>
  <b>Client \( c \)’s Input:</b> local model \( f_c \), unlabeled training sets \( U_c = \{U_{c}^m\}_{m=1}^{M_c} \), class priors \( \Pi_c \) and \( \pi \), local step-size \( \alpha_l \), and local updating iterations \( L \)

  <ol>
    <li>Start with initializing clients with Procedure A.</li>
    <li>For \( r = 1 \to R \):
      <ul>
        <li>Run Procedure B and Procedure C iteratively.</li>
      </ul>
    </li>

    <li><b>Procedure A. ClientInit(c):</b>
      <ul>
        <li>Transform \( U_c \) to a surrogate labeled dataset \( U_c \) according to (6)</li>
        <li>Modify \( f \) to \( g_c = Q_c(f) \), where \( Q_c \) is computed according to Theorem 1</li>
      </ul>
    </li>

    <li><b>Procedure B. ClientUpdate(c):</b>
      <ul>
        <li>\( f_c \leftarrow f \) &nbsp; <i>// Receive updated model from ServerExecute</i></li>
        <li>\( g_c \leftarrow Q_c(f_c) \)</li>
        <li>For \( l = 1 \to L \):
          <ul>
            <li>\( g_c \leftarrow g_c - \alpha_l \nabla J_b(g_c; U_c) \) &nbsp; <i>// SGD update based on (7)</i></li>
            <li>\( f_c \leftarrow f_c - \alpha_l \nabla J_b(Q_c(f_c); U_c) \) &nbsp; <i>// Update on \( g_c \) induces update on \( f_c \)</i></li>
          </ul>
        </li>
        <li>Send \( f_c - f \) to ServerExecute</li>
      </ul>
    </li>

    <li><b>Procedure C. ServerExecute(r):</b>
      <ul>
        <li>If using Method 1:
          <ul>
            <li>Fine-tune \( f \) on the server’s labeled data</li>
          </ul>
        </li>
        <li>Else if using Method 2:
          <ul>
            <li>Independently train a model \( f' \) on labeled data</li>
            <li>\( f \leftarrow (1 - \text{hyperparameter}) \times f + \text{hyperparameter} \times f' \)</li>
            <li>Broadcast updated \( f \) to ClientUpdate</li>
          </ul>
        </li>
      </ul>
    </li>
  </ol>
</div>

<h1 id="empirical-results">Empirical Results</h1>
<h2 id="experimental-setup">Experimental setup</h2>
<p>The experiments were done on the MNIST dataset using a CNN with the
exact dimensions and parameters as conducted in the experiments of
FedUL, with exception of a batch size of 100 instead of 128 for
splitting considerations. The goal is achieve a performance better than
that of FedUL’s and validate the approach of bi-objective optimization.
Below is a table comparing our model’s mean error compared to FedUL’s,
with FedUL+M<span class="math inline"><em>x</em></span> being a place
holder for our algorithm with method <span
class="math inline">#<em>x</em></span>. The best performing model on the
validation data is chosen.</p>
<div id="sample-table">
<table>
<caption>Setup</caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;"></th>
<th style="text-align: left;"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>1-3</span> Clients</td>
<td style="text-align: left;">Parameters</td>
<td style="text-align: left;">Data Splitting</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline">#</span>C =
5</td>
<td style="text-align: left;"><span
class="math inline"><em>α</em> = 1<em>e</em> − 4</span></td>
<td style="text-align: left;">Test: 10k</td>
</tr>
<tr>
<td style="text-align: left;">Set/C = 10</td>
<td style="text-align: left;">batch = 100</td>
<td style="text-align: left;">FedUL: 36k</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">epochs = 100</td>
<td style="text-align: left;">Supervised task: 12k</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">method 2’s hyperparameter: 0.5</td>
<td style="text-align: left;">Validation: 12k</td>
</tr>
</tbody>
</table>
</div>
<h2 id="results">Results</h2>
<div id="sample-table">
<table>
<caption>Mean Error Rates</caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;"></th>
<th style="text-align: left;"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>1-3</span> Algorithm</td>
<td style="text-align: left;">Mean Error (IID)</td>
<td style="text-align: left;">Mean Error (non-IID)</td>
</tr>
<tr>
<td style="text-align: left;">FedUL</td>
<td style="text-align: left;">0.78</td>
<td style="text-align: left;">2.98</td>
</tr>
<tr>
<td style="text-align: left;">FedUL+M1</td>
<td style="text-align: left;">0.69</td>
<td style="text-align: left;">0.77</td>
</tr>
<tr>
<td style="text-align: left;">FedUL+M2</td>
<td style="text-align: left;">0.69</td>
<td style="text-align: left;">0.79</td>
</tr>
</tbody>
</table>
</div>
<h2 class="unnumbered" id="note">Note</h2>
<p>Additionally, plots showing the mean error and losses per epoch are
included and briefly discussed in the appendix.</p>
<h1 id="conclusion-and-future-work">Conclusion and Future Work</h1>
<p>In conclusion, we found that implementing a bi-objective optimization
with FedUL and a supervised model can significantly enhance the
performance of a standalone FedUL model, particularly when applied to
non-IID datasets. Our research has demonstrated two key findings. First,
FedUL is capable of learning concurrently with a supervised federated
learning technique without the objectives interfering destructively with
each other. Second, the integration of knowledge from a supervised model
into FedUL not only boosts its accuracy but also facilitates faster
convergence. In the current setting, method 1 seems to be perform better
than method 2.<br />
</p>
<p>Looking ahead, future work could focus on several areas. Extensive
validation of the algorithm under diverse settings and with various
models could further establish its robustness and adaptability.
Additionally, optimizing the hyperparameter tuning process to enhance
performance and ease of use. Finally, examining the impacts of different
data distributions and more extreme cases of non-IID data could provide
deeper understanding for broader applications in real-world
scenarios.</p>
<h1 class="unnumbered" id="references">References</h1>
<p>[1] Lu, N., Wang, Z., Li, X., Niu, G., Dou, Q., &amp; Sugiyama, M.
(2022) Federated Learning from Only Unlabeled Data with
Class-Conditional-Sharing Clients. In <em>International Conference on
Learning Representations</em>. Available: <a
href="https://openreview.net/forum?id=WHA8009laxu"
class="uri">https://openreview.net/forum?id=WHA8009laxu</a></p>
<p><strong>Code:</strong> The code used to provide the empirical results
is a heavily modified version of the code the authors Lu, N et. al. used
and can be found here, under federated. choose fedulmnist.py:</p>
<div class="center">
<p><a href="https://rpi.box.com/s/y86pvaxxs5sn2sj6huwhrvs2xahszmi6"
class="uri">https://rpi.box.com/s/y86pvaxxs5sn2sj6huwhrvs2xahszmi6</a></p>
</div>
<p><strong>Instructions:</strong> This is the link for the GitHub
repository of FedUL algorithm, which mostly covers how to run the code.
For extra information please contact me:</p>
<div class="center">
<p><a href="https://github.com/lunanbit/FedUL?tab=readme-ov-file"
class="uri">https://github.com/lunanbit/FedUL?tab=readme-ov-file</a></p>
</div>
<h1 class="unnumbered" id="appendix">Appendix</h1>
<h2 class="unnumbered" id="discussion">Discussion</h2>
<h3 id="loss">Loss</h3>
<p>Shown in Figure 1 is the supervised learning loss and surrogate task
loss when performing the hybrid model, compared to the loss of
performing standalone FedUL. The loss of the surrogate task and FedUL
seems to be exact, with the exception that the surrogate task’s drops
faster initially. This shows the positive influence of the hybrid model
on adjusting the parameters of the clients’ end. Additionally, the loss
of all models seem to consistently decrease, indicating a shared
objective.</p>
<p>It is also worth mentioning that Figures 2 and 4 show the loss
dropping much faster than the Figures 1 and 3. This indicates that
method 2 can lead to faster convergence.</p>
<h3 id="error-rate">Error Rate</h3>
<p>In figure 5, we see a clearer picture of how the hybrid model
outperforms the Standalone FedUL Algorithm. It consistently has lower
error rates and converges faster. Additionally, in figures 7 and 8, the
mean error shows a steep decrease which indicates that the hybrid model
handles the non-IID datasets well.</p>

<h2 id="supplementing-figures">Supplementing Figures</h2>


<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Learning Loss and Surrogate loss Method 1 (IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 1: Learning Loss and Surrogate loss Method 1 (IID).
</div>

<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Learning Loss and Surrogate loss Method 2 (IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 2: Learning Loss and Surrogate loss Method 2 (IID).
</div>

<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Learning Loss and Surrogate loss Method 1 (non-IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 3: Learning Loss and Surrogate loss Method 1 (non-IID).
</div>

<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Learning Loss and Surrogate loss Method 2 (non-IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 4: Learning Loss and Surrogate loss Method 2 (non-IID).
</div>

<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Mean Error Rate Method 1 (IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 5: Mean Error Rate Method 1 (IID).
</div>

<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Mean Error Rate Method 2 (IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 6: Mean Error Rate Method 2 (IID).
</div>

<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Mean Error Rate Method 1 (non-IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 7: Mean Error Rate Method 1 (non-IID).
</div>

<div style="display: flex; justify-content: center; margin-bottom: 1.5em;">
  <img
    src="{{ '/assets/img/fedul/Mean Error Rate Method 2 (non-IID).png' | relative_url }}" 
    alt="Confusion Matrix for Classification Model Testing" 
    style="width: 600px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
  />
</div>
<div class="caption">
  Figure 8: Mean Error Rate Method 2 (non-IID).
</div>

