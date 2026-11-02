**Dynamic Survival Blending via Information Geometry and Relative
Uncertainty**

<https://share.gemini.google/IGTnWJHCKeQD>

**1. Introduction and Background**

In health technology assessment (HTA), economic evaluations of
life-extending therapies typically adopt a lifetime analytic horizon.
For immuno-oncology (IO) agents, chimeric antigen receptor T-cell
(CAR-T) therapies, and advanced therapeutic medicinal products (ATMPs),
randomized controlled trial (RCT) follow-up is frequently immature at
the time of initial appraisal. Furthermore, conventional extrapolation
techniques—which assume that relative treatment effects (e.g., constant
proportional hazard ratios) persist indefinitely—fail to capture
clinical dynamics such as administrative stopping rules and gradual
treatment effect waning.

To resolve these challenges without imposing abrupt, clinically
implausible step-changes in hazard rates, Che et al. (2023) and Zhu et
al. (2026) introduced the blended survival and hazard extrapolation
frameworks. Under this paradigm, an arm-specific model fitted to
internal trial data is smoothly transitioned into an external model
informed by mature, long-term registry or standard-of-care data across a
pre-specified time interval \[t1, t2\]:

> 
> ``` math
> S_{b}lend(t) = \left\lbrack 1 - \pi(t) \right\rbrack \cdot S_{i}nt(t) + \pi(t) \cdot S_{e}xt(t)
> ```

In existing implementations, the weight schedule π(t) is modeled using
the cumulative distribution function of a Beta(a, b) distribution
evaluated at (t - t1) / (t2 - t1). While mathematically continuous,
standard blended extrapolation introduces a major methodological
drawback: it relies on four subjective user-specified parameters (t1,
t2, a, b). As Zhu et al. (2026) explicitly observe, the shape parameters
a and b have no direct physical or biological interpretation and cannot
be informed by clinical evidence. Consequently, varying (a, b) across
sensitivity scenarios can dramatically swing restricted mean survival
times (RMST) and incremental cost-effectiveness ratios (ICERs), exposing
submissions to committee criticism.

| **CORE INNOVATION:** Rather than forcing an arbitrary, textbook Beta curve onto the extrapolation, the transition rate should be self-calibrating. The blending schedule must be governed by the empirical distance between the curves and the relative statistical uncertainty (precision loss) of the internal trial model. |
|----|

**Limitations of Heuristic Weight Parameterizations:**

While the blended survival framework (Che et al., 2023; Zhu et al.,
2026) avoids the clinically implausible hazard discontinuities of
piecewise models, its practical implementation relies on four
user-specified parameters: the blending interval endpoints
$`\lbrack a,b\rbrack`$ and the Beta shape parameters $`(\alpha,\beta)`$
governing the weight schedule
$`\pi(t;\alpha,\beta,a,b) = F_{\text{Beta}}\left( \frac{t - a}{b - a};\alpha,\beta \right)`$.
As noted by Zhu et al. (2026), the shape parameters $`(\alpha,\beta)`$
have no direct biological interpretation and cannot be directly informed
by clinical trial evidence.

Preliminary factorial simulation evidence across 144
scenarios—evaluating combinations of sample size
($`N \in \left\{ 75,250,1000 \right\}`$), administrative censoring times
(ACT $`\in \left\{ 100,360,500,750 \right\}`$ days), and blending
interval lengths ($`\left\{ 50,150,250 \right\}`$ days)—demonstrates
that extrapolation performance is highly sensitive to the choice of
$`(\alpha,\beta)`$. Specifically, when trial follow-up is short (e.g.,
ACT $`= 100`$ or $`360`$ days) and the internal survival curve plateaus
prematurely due to heavy censoring, right-skewed weight distributions
such as $`{Beta}(1,5)`$ (where $`\alpha < \beta`$ shifts weight rapidly
toward the external reference curve early in the blending window)
consistently achieve the lowest bias, lowest empirical standard error
(EmpSE), and highest confidence interval coverage. Conversely,
left-skewed parameterizations such as $`{Beta}(7,3)`$ (where
$`\alpha > \beta`$ delays the transition and lingers on the uncertain
internal extrapolation) induce severe positive bias, inflated empirical
variance, and coverage collapse.

Crucially, in a prospective health technology assessment (HTA), the
analyst does not have access to ground-truth long-term outcomes to
visually select $`{Beta}(1,5)`$ over $`{Beta}(7,3)`$ a priori. This
vulnerability motivates a self-calibrating weighting mechanism that
automatically accelerates the transition toward the external anchor when
internal trial precision degrades.

**2. Mathematical Formulations of Dynamic Blending**

**Information-Geometric Duality of Arithmetic and Geometric Blending:**

Let $`S_{\text{obs}}\left( t \mid \theta_{\text{obs}} \right)`$ and
$`S_{\text{ext}}\left( t \mid \theta_{\text{ext}} \right)`$ denote the
fitted internal and external survival functions, respectively. In the
foundational survival-scale formulation of Che et al. (2023), the
blended survival function $`S_{\text{ble}}(t)`$ is defined as a
time-varying geometric mixture:

``` math
S_{\text{ble}}(t \mid \theta) = S_{\text{obs}}\left( t \mid \theta_{\text{obs}} \right)^{1 - \pi(t)} \times S_{\text{ext}}\left( t \mid \theta_{\text{ext}} \right)^{\pi(t)}
```

Taking the negative natural logarithm yields a linear convex combination
on the cumulative hazard scale,
$`H_{\text{ble}}(t) = \left\lbrack 1 - \pi(t) \right\rbrack H_{\text{obs}}(t) + \pi(t)H_{\text{ext}}(t)`$.
Differentiating with respect to $`t`$ yields the exact analytical
blended hazard function $`h_{\text{ble}}(t)`$ and density
$`f_{\text{ble}}(t)`$:

``` math
h_{\text{ble}}(t) = \left\lbrack 1 - \pi(t) \right\rbrack h_{\text{obs}}(t) + \pi(t)h_{\text{ext}}(t) + \pi'(t)\ln\left( \frac{S_{\text{obs}}(t)}{S_{\text{ext}}(t)} \right),\quad\quad f_{\text{ble}}(t) = S_{\text{ble}}(t)h_{\text{ble}}(t)
```

From the perspective of information geometry, this geometric survival
pool and the linear survival pool
$`S_{\text{ble}}(t) = \left\lbrack 1 - \pi(t) \right\rbrack S_{\text{obs}}(t) + \pi(t)S_{\text{ext}}(t)`$
correspond to the two directional projections of the Kullback–Leibler
(KL) divergence:

1.  **Forward KL Projection (Arithmetic Mixture):** Finding an
    intermediate survival distribution $`M_{t}`$ that minimizes
    $`\left\lbrack 1 - \pi(t) \right\rbrack D_{\text{KL}}\left( P_{\text{obs},t} \parallel M_{t} \right) + \pi(t)D_{\text{KL}}\left( P_{\text{ext},t} \parallel M_{t} \right)`$
    yields the linear mixture. Under **Approach 1 (Divergence-Paced
    Clock)**, we govern the transition rate $`\pi(t)`$ by integrating
    the pointwise Gaussian KL divergence
    $`D_{\text{KL}}\left( P_{\text{obs},t} \parallel P_{\text{ext},t} \right)`$
    across $`\lbrack a,b\rbrack`$, regularized by a cosine window taper
    $`w(t) = \sin^{2}\left( \pi\frac{t - a}{b - a} \right)`$ so that
    $`\pi'(a) = \pi'(b) = 0`$, guaranteeing $`C^{1}`$ continuity in
    $`h_{\text{ble}}(t)`$.

2.  **Reverse KL Projection (Information Barycenter / Precision
    Pooling):** Finding an intermediate distribution
    $`M_{t} \sim \mathcal{N}\left( \mu_{M}(t),\sigma_{M}^{2}(t) \right)`$
    that minimizes
    $`\left\lbrack 1 - \lambda(t) \right\rbrack D_{\text{KL}}\left( M_{t} \parallel P_{\text{obs},t} \right) + \lambda(t)D_{\text{KL}}\left( M_{t} \parallel P_{\text{ext},t} \right)`$
    yields an exponential barycenter weighted by statistical precision
    $`\tau(t) = 1/\sigma^{2}(t)`$:

``` math
S_{\text{ble}}^{(2)}(t) = \mu_{M}(t) = \frac{\left\lbrack 1 - \lambda(t) \right\rbrack\tau_{\text{obs}}(t){\widehat{S}}_{\text{obs}}(t) + \lambda(t)\tau_{\text{ext}}(t){\widehat{S}}_{\text{ext}}(t)}{\left\lbrack 1 - \lambda(t) \right\rbrack\tau_{\text{obs}}(t) + \lambda(t)\tau_{\text{ext}}(t)}
```

Under **Approach 2**, as administrative censoring depletes the internal
trial risk set and $`\sigma_{\text{obs}}^{2}(t)`$ inflates
($`\tau_{\text{obs}}(t) \rightarrow 0`$), the effective weight
automatically shifts toward the high-precision external anchor,
replicating the favorable early-transition behavior of $`{Beta}(1,5)`$
without subjective tuning.

Let S_obs(t) and S_ext(t) represent the survival point estimates at time
t ∈ \[t1, t2\] for the extrapolated trial and external reference models,
respectively, with associated asymptotic standard errors σ_obs(t) and
σ_ext(t) obtained via the delta method on the complementary log-log
scale. At each time point t, we represent the state of epistemic belief
regarding survival as univariate Gaussian predictive distributions:

> 
> ``` math
> P_{o}bs,t\ N\left( S_{o}bs(t),\sigma_{o}bs²(t) \right),P_{e}xt,t\ N\left( S_{e}xt(t),\sigma_{e}xt²(t) \right)
> ```

**2.1 Approach 1: Dynamic Divergence Clock (Speedometer)**

In Approach 1, the instantaneous information discrepancy between the
observed trial model and the external anchor is quantified by the
directed Kullback–Leibler (KL) divergence from P_obs,t to P_ext,t. For
two Gaussian distributions, this yields the closed-form expression:

> 
> ``` math
> D_{K}L(t) = 0.5 \cdot \left\lbrack \ln\left( \sigma_{e}xt²(t)/\sigma_{o}bs²(t) \right) + \left( \sigma_{o}bs²(t) + \left( S_{o}bs(t) - S_{e}xt(t) \right)² \right)/\sigma_{e}xt²(t) - 1 \right\rbrack
> ```

This formulation encapsulates two critical dynamics simultaneously:

- **Geometric Separation:** The squared deviation (S_obs(t) - S_ext(t))²
  directly penalizes trajectories where the parametric trial
  extrapolation diverges away from long-term registry mortality.

- **Relative Uncertainty:** As trial follow-up ends and patient-level
  censoring dominates the tail, σ_obs²(t) inflates rapidly while
  σ_ext²(t) remains tightly constrained. The ratio σ_obs²(t) / σ_ext²(t)
  accelerates D_KL(t), signaling an objective loss of credibility in the
  trial model.

Boundary Regularization for C¹ Differentiability: If the cumulative
divergence integral were cut off abruptly at t2, a non-zero D_KL(t2)
would induce a jump discontinuity in the derivative π'(t), producing a
sharp corner (kink) in survival and a step jump in the implied hazard
rate h(t) = -S'(t)/S(t). To guarantee full C¹ differentiability, we
apply a smooth windowing taper w(t) = sin²(π · (t - t1)/(t2 - t1)),
ensuring π'(t1) = π'(t2) = 0:

> 
> ``` math
> \pi(t) = \left\lbrack \int_{t1}^{t}D_{K}L(u) \cdot w(u)du \right\rbrack/\left\lbrack \int_{t1}^{t2}D_{K}L(u) \cdot w(u)du \right\rbrack
> ```

**2.2 Approach 2: Information Barycenter (Precision Pooling)**

Rather than utilizing divergence solely as a scalar clock to pace an
arithmetic mixture, Approach 2 reformulates blending as an explicit
optimization problem in information space. We define the intermediate
blended survival state
$`M_{t}\ N\left( \mu_{M}(t),\sigma_{M}²(t) \right)`$ as the distribution
that minimizes the weighted sum of reverse KL divergences (information
projections) to both parent distributions:

> 
> ``` math
> min_{M_{t}}\left\lbrack \left( 1 - \lambda(t) \right) \cdot D_{K}L\left( M_{t}||P_{o}bs,t \right) + \lambda(t) \cdot D_{K}L\left( M_{t}||P_{e}xt,t \right) \right\rbrack
> ```

where λ(t) = 3u² - 2u³ (with u = (t - t1)/(t2 - t1)) is a smoothstep
progression parameter on \[0, 1\]. In information geometry, this
minimization yields an exponential barycenter. Expressing this in terms
of parameter precisions τ = 1/σ²:

> 
> ``` math
> \tau_{M}(t) = \left( 1 - \lambda(t) \right) \cdot \tau_{o}bs(t) + \lambda(t) \cdot \tau_{e}xt(t){S_{b}}^{(2)}(t) = \mu_{M}(t) = \left\lbrack \left( 1 - \lambda(t) \right) \cdot \tau_{o}bs(t) \cdot S_{o}bs(t) + \lambda(t) \cdot \tau_{e}xt(t) \cdot S_{e}xt(t) \right\rbrack/\tau_{M}(t)
> ```

Under Approach 2, the curve is directly pulled by relative parameter
precision. If trial follow-up ends early and trial variance explodes,
τ_obs(t) drops toward zero. Even if the nominal progress clock is only
at λ(t) = 0.50, the high precision of the external registry τ_ext(t)
mathematically overwhelms the trial model, automatically snapping the
blended curve to the reliable external evidence.

\*\*Yes, exactly.\*\* The distinction comes from the mathematical
property that the Kullback–Leibler (KL) divergence is \*\*strictly
asymmetric\*\*:

``` math
D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)
```

Because it is asymmetric, the order of the arguments matters
fundamentally. Depending on which distribution sits in the first slot
versus the second slot, minimizing the divergence produces either the
\*\*arithmetic mean\*\* (linear mixture) or the \*\*geometric mean\*\*
(exponential pooling).

**Forward KL: The Arithmetic Mean (Linear Mixture)**

When you look for an intermediate distribution \$M\$ that minimizes the
\*\*forward KL\*\* from the two boundary curves (\$P_1, P_2\$) to \$M\$:

``` math
\min_{M}\left\lbrack (1 - \pi)\, D_{KL}\left( P_{1} \parallel M \right)\mspace{6mu} + \mspace{6mu}\pi\, D_{KL}\left( P_{2} \parallel M \right) \right\rbrack
```

\* \*\*Mathematical solution:\*\*

``` math
M(x) = (1 - \pi)\, P_{1}(x) + \pi\, P_{2}(x)
```

\* \*\*What this is:\*\* This is the standard \*\*arithmetic mean\*\*
(linear combination).

\* \*\*Connection to Survival:\*\*

When you perform standard blending directly on the survival curve:

``` math
S_{blend}(t) = (1 - \pi)\, S_{obs}(t) + \pi\, S_{ext}(t)
```

you are using the exact intermediate distribution \$M\$ that minimizes
the \*forward\* KL divergence.

\* \*\*Behavior:\*\* It is \*\*mean-preserving\*\* (moment-matching). It
averages the probabilities linearly.

**Reverse KL: The Geometric Mean (Exponential / Precision Pooling)**

When you flip the order and look for an intermediate distribution \$M\$
that minimizes the \*\*reverse KL\*\* from \$M\$ to each boundary curve:

``` math
\min_{M}\left\lbrack (1 - \pi)\, D_{KL}\left( M \parallel P_{1} \right)\mspace{6mu} + \mspace{6mu}\pi\, D_{KL}\left( M \parallel P_{2} \right) \right\rbrack
```

\* \*\*Mathematical solution:\*\*

``` math
M(x) \propto \left( P_{1}(x) \right)^{1 - \pi} \cdot \left( P_{2}(x) \right)^{\pi}
```

\* \*\*What this is:\*\* This is the \*\*geometric mean\*\* (normalized
log-linear pooling / exponential barycenter).

\* \*\*Connection to Survival:\*\*

Taking the logarithm of both sides shows what happens on the cumulative
hazard scale (\$H(t) = -\ln S(t)\$):

``` math
\ln M(x) = (1 - \pi){\ln P}_{1}(x) + \pi{\ln P}_{2}(x) - \ln Z
```

``` math
H_{blend}(t) = (1 - \pi)\, H_{obs}(t) + \pi\, H_{ext}(t)
```

So blending cumulative hazards linearly corresponds to a \*\*geometric
mean\*\* of the survival functions.

\* \*\*Behavior:\*\* When applied to Gaussian beliefs with means and
variances \$(\mu_1, \sigma_1^2)\$ and \$(\mu_2, \sigma_2^2)\$, the
geometric mean leads to \*\*precision weighting\*\* (\$\tau =
1/\sigma^2\$):

``` math
\mu_{M} = \frac{(1 - \pi)\,\tau_{1}\,\mu_{1} + \pi\,\tau_{2}\,\mu_{2}}{(1 - \pi)\,\tau_{1} + \pi\,\tau_{2}}
```

The intermediate curve is actively pulled by the model with higher
statistical precision (smaller variance).

\### Summary of the Directional Duality

\| Order \| Optimization Objective \| Mathematical Result \| Survival
Framework Equivalent \|

\| --- \| --- \| --- \| --- \|

\| \*\*Forward KL\*\* \| \$\min_M \sum w_i D\_{\mathrm{KL}}(P_i
\parallel M)\$ \| \*\*Arithmetic Mean\*\* \$\sum w_i P_i\$ \| Direct
linear survival blending: \$S\_{\mathrm{blend}} =
(1-\pi)S\_{\mathrm{obs}} + \pi S\_{\mathrm{ext}}\$ \|

\| \*\*Reverse KL\*\* \| \$\min_M \sum w_i D\_{\mathrm{KL}}(M \parallel
P_i)\$ \| \*\*Geometric Mean\*\* \$\prod P_i^{w_i} / Z\$ \| Hazard /
log-survival pooling; yields precision-weighted \$(\tau = 1/\sigma^2)\$
intermediate distributions \|

**3. Comparative Discrepancy & Optimal Transport Metrics**

The informational approach can be generalized beyond Gaussian KL
divergence. Table 1 outlines the operational characteristics of
alternative statistical divergences and probability metrics for
governing survival extrapolation.

| **Metric** | **Mathematical Form** | **Operational Dynamics** | **HTA Decision Utility** |
|----|----|----|----|
| **Kullback–Leibler (KL)** | ∫ p(s) ln(p(s)/q(s)) ds | Asymmetric, unbounded; heavily penalizes tail deviations. | Rapidly registers precision collapse in small trials. |
| **Jensen–Shannon (JSD)** | 0.5 D_KL(P\|\|M) + 0.5 D_KL(Q\|\|M) | Symmetric, bounded \[0, 1\]; strictly stable everywhere. | Provides an organic, data-driven S-curve transition. |
| **Hellinger (H²)** | 0.5 ∫ (√p - √q)² ds | Bounded \[0, 1\]; square-root amplifies tail mass shifts. | Highly sensitive trigger for early treatment effect waning. |
| **1-Wasserstein (EMD)** | ∫ \|S_obs(u) - S_ext(u)\| du | Optimal transport work; linear in cumulative survival area. | Direct visual translation of Expected Opportunity Loss. |

**5. Methodological Comparison & HTA Defensibility**

| **Framework** | **Parameters Required** | **Uncertainty Response** | **Hazard Smoothness** |
|----|----|----|----|
| **Standard Beta (Che/Zhu)** | 4 Subjective (t1, t2, a, b) | Static; blind to sample size or variance explosion | C¹ continuous only if a, b \> 1 |
| **Approach 1 (KL Clock)** | 2 Boundary Limits (t1, t2) | Dynamic; accelerates clock as trial variance inflates | Strictly C¹ continuous via window taper |
| **Approach 2 (Barycenter)** | 2 Boundary Limits (t1, t2) | Intrinsic; precision directly pulls curve to anchor | Strictly C¹ continuous via smoothstep λ(t) |

If you replace the **Kullback–Leibler (KL) divergence** with an
alternative probability distance or divergence metric, you change **how
the blending mechanism perceives and penalizes discrepancy** between the
observed trial model ($`P_{\text{obs},t}`$) and the external anchor
($`P_{\text{ext},t}`$).

The mathematical and behavioral differences manifest differently
depending on whether you use the metric as a **transition clock**
(Approach 1) or as an **intermediate barycenter / precision compromise**
(Approach 2).

**1. Alternative Divergence Metrics and Their Behavioural Differences**

**A. 1-Wasserstein Metric (Earth Mover’s Distance)**

Instead of measuring relative entropy, 1-Wasserstein measures the
minimal physical “work” required to transport one probability density
into another (mass $`\times`$ distance).

- **For Gaussian predictive distributions**
  $`\mathcal{N}\left( \mu_{1},\sigma_{1}^{2} \right)`$ and
  $`\mathcal{N}\left( \mu_{2},\sigma_{2}^{2} \right)`$:

``` math
W_{1}\left( P_{\text{obs},t},P_{\text{ext},t} \right) = |{\widehat{S}}_{\text{obs}}(t) - {\widehat{S}}_{\text{ext}}(t)| + |\sigma_{\text{obs}}(t) - \sigma_{\text{ext}}(t)|
```

- **On the whole survival curve directly**, $`W_{1}(t)`$ is simply the
  integrated absolute area:

``` math
W_{1}(t) = \int_{0}^{t}|S_{\text{obs}}(u) - S_{\text{ext}}(u)|\, du
```

- **What changes compared to KL?**

  - **Linear, not exponential:** KL has a logarithmic/quadratic penalty
    that accelerates rapidly when variances diverge. Wasserstein is
    purely linear.

  - **Area-proportional & robust:** Wasserstein treats deviations
    proportionally across the board. If the trial hazard has a brief
    localized spike or the standard error widens slowly, Wasserstein
    does not overreact with a sharp jump. It produces a very smooth,
    steady transition.

  - **HTA / Economic alignment:** In decision theory, 1-Wasserstein is
    directly equivalent to the Expected Opportunity Loss (EVPI). Using
    Wasserstein ties the blending rate directly to the physical expected
    loss of the survival difference.

**B. Jensen–Shannon Divergence (JSD)**

The Jensen–Shannon divergence is a symmetrized, smoothed version of KL
divergence computed against the midpoint distribution
$`M = \frac{1}{2}\left( P_{\text{obs}} + P_{\text{ext}} \right)`$:

``` math
{JSD}\left( P_{\text{obs}} \parallel P_{\text{ext}} \right) = \frac{1}{2}D_{\text{KL}}\left( P_{\text{obs}} \parallel M \right) + \frac{1}{2}D_{\text{KL}}\left( P_{\text{ext}} \parallel M \right)
```

- **What changes compared to KL?**

  - **Strictly bounded in** $`\lbrack 0,1\rbrack`$**:** Unlike KL, which
    is asymmetric
    ($`D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)`$)
    and can blow up to infinity ($`\infty`$) if a tail density vanishes,
    JSD is symmetric and strictly bounded between 0 and 1 (when using
    base-2 logarithms).

  - **Zero numerical instability:** When trial survival drops near zero
    and standard errors become volatile, KL can produce extreme spikes
    in R. JSD remains numerically stable, naturally tracing out an
    organic S-curve without needing ad hoc clamping.

**C. Hellinger Distance (**$`\mathbf{H}^{\mathbf{2}}`$**)**

Hellinger distance measures the difference between the *square roots* of
the probability densities:

``` math
H^{2}\left( P_{\text{obs},t},P_{\text{ext},t} \right) = 1 - \sqrt{\frac{2\sigma_{\text{obs}}\sigma_{\text{ext}}}{\sigma_{\text{obs}}^{2} + \sigma_{\text{ext}}^{2}}}\exp\left( - \frac{1}{4}\frac{\left( {\widehat{S}}_{\text{obs}}(t) - {\widehat{S}}_{\text{ext}}(t) \right)^{2}}{\sigma_{\text{obs}}^{2} + \sigma_{\text{ext}}^{2}} \right)
```

- **What changes compared to KL?**

  - **Hyper-sensitive to tails:** Because it evaluates $`\sqrt{f(t)}`$,
    the Hellinger metric is disproportionately sensitive to small
    differences in the sparse tails of distributions.

  - **Earlier transition trigger:** If you want the model to
    aggressively start handing over to the external baseline the instant
    trial censoring begins to introduce tail uncertainty, Hellinger will
    fire earlier than KL or Wasserstein.

**D. Maximum Mean Discrepancy (MMD) / Kernel Distances**

MMD computes distance in a Reproducing Kernel Hilbert Space (RKHS)
without assuming a Gaussian form for the parameter uncertainties:

``` math
{MMD}^{2}\left( P_{\text{obs}},P_{\text{ext}} \right) = \mathbb{E}\left\lbrack k(X,X') \right\rbrack - 2\mathbb{E}\left\lbrack k(X,Y) \right\rbrack + \mathbb{E}\left\lbrack k(Y,Y') \right\rbrack
```

- **What changes compared to KL?**

  - **Non-parametric:** You do not have to approximate the survival
    belief as Gaussian
    $`\mathcal{N}\left( \widehat{S},\sigma^{2} \right)`$ via the delta
    method. If your Bayesian model outputs MCMC posterior draws from
    Stan/JAGS with skewness or multi-modality, MMD directly evaluates
    the sample draws without parametric assumptions.

**2. What Changes Under the Two Blending Approaches?**

| **Metric** | **Under Approach 1 (Clock / Speedometer π(t)\pi(t)π(t)) DOCX** | **Under Approach 2 (Intermediate Barycenter MtM_tMt​) DOCX** |
|----|----|----|
| **Kullback–Leibler (KL)** | **Aggressive pacing:** As $`\sigma_{\text{obs}}^{2}`$ explodes, the clock $`\pi(t)`$ accelerates rapidly. | **Precision weighting:** Yields inverse-variance pooling ($`\tau = 1/\sigma^{2}`$). Low-precision curves lose influence quickly. |
| **Jensen–Shannon (JSD)** | **Smooth S-curve:** Naturally self-normalizing without risk of infinite values. | **Geometric/Harmonic compromise:** Yields a balanced midpoint that softens the effect of one model's exploding variance. |
| **1-Wasserstein (**$`W_{1}`$**)** | **Area-proportional pacing:** Linear and gradual; less prone to sudden jumps from localized trial fluctuations. | **Quantile / Area barycenter:** The barycenter is the weighted average of the quantiles, directly averaging survival durations rather than precisions. |
| **Hellinger (**$`H^{2}`$**)** | **Early alert trigger:** Transitions earlier in the timeline due to square-root tail magnification. | **Affinity-weighted pooling:** Pulls heavily toward the model with broader support, avoiding degenerate boundary estimates. |

**Summary Recommendation for Your Project**

- If your goal is **sensitivity to loss of trial precision** (detecting
  when trial data becomes too uncertain to rely on), **KL** is
  mathematically tailored for the job because of its variance-ratio
  penalty $`\frac{\sigma_{\text{obs}}^{2}}{\sigma_{\text{ext}}^{2}}`$.

- If you want a **numerically stable, elegant bounded S-curve**, use
  **Jensen–Shannon (JSD)**.

- If you want an intuitive, **area-based economic metric** that directly
  reflects physical opportunity loss and avoids derivative volatility,
  use **1-Wasserstein (Earth Mover’s Distance)**.

**Simulation Study Design (ADEMP Framework):**

Following the ADEMP guidelines of Morris et al. (2019), we conducted a
factorial Monte Carlo simulation study ($`N_{\text{sim}} = 1,000`$
replications per scenario) to evaluate dynamic information-theoretic
blending against static Beta parameterizations.

- **Data-Generating Mechanism (DGM):** Individual patient event times
  $`T_{i}`$ were generated across combinations of trial sample size
  ($`N \in \left\{ 75,250,1000 \right\}`$), administrative censoring
  times ($`\text{ACT} \in \left\{ 100,360,500,750 \right\}`$ days), and
  blending interval lengths ($`b - a \in \left\{ 50,150,250 \right\}`$
  days), subject to random drop-out
  $`s_{i} \sim \text{Bernoulli}\left( 1 - p_{\text{cens}} \right)`$ with
  uniform censoring
  $`c_{i} \sim \text{Uniform}\left( 0,\text{ACT} \right)`$, yielding
  observed times
  $`T_{i}^{\text{obs}} = \min\left( T_{i},\text{ACT},c_{i} \right)`$.

- **Estimands:** To comprehensively assess curve accuracy across both
  early and late portions of the distribution, we evaluated five primary
  estimands: the lower quartile ($`T_{0.75}`$), median ($`T_{0.50}`$),
  and upper quartile ($`T_{0.25}`$) survival times, defined as
  $`T_{q} = \min\left\{ t:S_{\text{ble}}(t) \leq q \right\}`$, alongside
  restricted mean survival times at 360 and 750 days
  ($`\text{RMST}_{360}`$ and $`\text{RMST}_{750}`$), approximated via
  the trapezoidal rule:

``` math
\text{RMST}\left( t^{*} \right) = \int_{0}^{t^{*}}S(u)\, du \approx \frac{\Delta}{2}\sum_{k = 1}^{K}\left\lbrack S\left( t_{k - 1} \right) + S\left( t_{k} \right) \right\rbrack
```

- **Methods Compared:** We compared four manual static
  parameterizations—$`{Beta}(1,1)`$ (uniform reference), $`{Beta}(3,3)`$
  (symmetric S-curve), $`{Beta}(1,5)`$ (rapid early transition), and
  $`{Beta}(7,3)`$ (delayed transition)—against the self-calibrating
  **Approach 1 (Tapered KL Clock)**, **1-Wasserstein (Optimal Transport)
  Clock**, and **Approach 2 (Reverse-KL Precision Barycenter)**.

- **Performance Measures:** For each estimand $`\phi`$, we computed
  Monte Carlo Bias, Empirical Standard Error ($`\text{EmpSE}`$), 95%
  confidence interval Coverage, and their associated Monte Carlo
  Standard Errors (MCSE). Additionally, to evaluate global
  distributional fidelity strictly over the extrapolation window
  $`\left\lbrack a,t^{*} \right\rbrack`$ without dilution from the
  pre-blending observation period $`\lbrack 0,a)`$ where curves
  coincide, we evaluated the **Dynamic Restricted Kullback–Leibler
  (**$`\text{DRKL}`$**) Divergence** (extending Cox and Czanner, 2016):

``` math
\text{DRKL}\left( a,t^{*};f_{\text{true}},f_{\text{ble}} \right) = \int_{a}^{t^{*}}f_{\text{true}}(u)\ln\left( \frac{f_{\text{true}}(u)}{f_{\text{ble}}(u)} \right)du
```

- **Visual Presentation:** Multi-factor performance trends across all
  scenarios are synthesized using **Nested Loop Plots** (Rücker and
  Schwarzer, 2014), while replication-level interval calibration and
  coverage breakdown under model misspecification are diagnosed using
  **Zipper Plots**.

**Appendix: Derivation of the Implied Blended Hazard and Density
Functions**

Under the geometric blended survival formulation
$`S_{\text{ble}}(t) = S_{\text{obs}}(t)^{1 - \pi(t)}S_{\text{ext}}(t)^{\pi(t)}`$,
the implied probability density function
$`f_{\text{ble}}(t) = - \frac{\partial}{\partial t}S_{\text{ble}}(t)`$
is obtained via the product and chain rules of calculus:

``` math
\begin{aligned}
f_{\text{ble}}(t) & = - \frac{\partial}{\partial t}\left\lbrack \left( 1 - F_{\text{obs}}(t) \right)^{1 - \pi(t)}\left( 1 - F_{\text{ext}}(t) \right)^{\pi(t)} \right\rbrack \\
 & = - \left\lbrack \frac{\partial}{\partial t}\left( \left( 1 - F_{\text{obs}}(t) \right)^{1 - \pi(t)} \right)\left( 1 - F_{\text{ext}}(t) \right)^{\pi(t)} + \left( 1 - F_{\text{obs}}(t) \right)^{1 - \pi(t)}\frac{\partial}{\partial t}\left( \left( 1 - F_{\text{ext}}(t) \right)^{\pi(t)} \right) \right\rbrack \\
 & = - S_{\text{ble}}(t)\left\lbrack - \pi'(t){\ln S}_{\text{obs}}(t) - \frac{\left( 1 - \pi(t) \right)f_{\text{obs}}(t)}{S_{\text{obs}}(t)} + \pi'(t){\ln S}_{\text{ext}}(t) - \frac{\pi(t)f_{\text{ext}}(t)}{S_{\text{ext}}(t)} \right\rbrack \\
 & = S_{\text{ble}}(t)\left\lbrack \left( 1 - \pi(t) \right)h_{\text{obs}}(t) + \pi(t)h_{\text{ext}}(t) + \pi'(t)\ln\left( \frac{S_{\text{obs}}(t)}{S_{\text{ext}}(t)} \right) \right\rbrack \\
 & = S_{\text{ble}}(t)h_{\text{ble}}(t)
\end{aligned}
```

This identity explicitly demonstrates why boundary differentiability
($`\pi'(a) = \pi'(b) = 0`$) is vital: if $`\pi'(t)`$ experiences a step
discontinuity at $`t = a`$ or $`t = b`$ (as occurs under a uniform
$`{Beta}(1,1)`$ weight or an untapered cumulative divergence integral),
the term
$`\pi'(t)\ln\left( S_{\text{obs}}(t)/S_{\text{ext}}(t) \right)`$
introduces an artificial step-jump in the implied hazard
$`h_{\text{ble}}(t)`$. Applying our cosine window taper $`w(t)`$ or
smoothstep progression $`\lambda(t)`$ forces $`\pi'(a) = \pi'(b) = 0`$,
eliminating boundary jumps entirely.

<img src="media/image1.png" style="width:7.08333in;height:5.13542in" />
