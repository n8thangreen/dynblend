# Dynamic Survival Blending via Information Geometry, Bayesian Predictive Modeling, and Relative Uncertainty

**Nathan Green**  
*Department of Statistical Science, University College London, UK*  
*Target Journal:* Medical Decision Making / Value in Health  

---

## 1. Introduction and Background

In health technology assessment (HTA), economic evaluations of life-extending health technologies—such as immuno-oncology (IO) agents, chimeric antigen receptor T-cell (CAR-T) therapies, and advanced therapy medicinal products (ATMPs)—typically adopt a lifetime analytic horizon (often 20 to 40 years). However, at the time of initial regulatory and reimbursement appraisals by bodies such as the National Institute for Health and Care Excellence (NICE), the Canadian Agency for Drugs and Technologies in Health (CADTH), or the Pharmaceutical Benefits Advisory Committee (PBAC), pivotal randomized controlled trial (RCT) follow-up is frequently immature. 

Conventional parametric survival models (e.g., Weibull, Gompertz, log-normal, log-logistic) fitted to immature patient-level data are heavily leveraged by early events and administrative censoring. Extrapolating these models over decades routinely produces clinically implausible long-term survival tails or assumes that relative treatment effects (e.g., constant hazard ratios) persist indefinitely, directly contradicting clinical evidence of treatment effect waning or administrative stopping rules.

To address these vulnerabilities, HTA practitioners historically resorted to **piecewise cutoff models** (e.g., NICE appraisal TA366), where the parametric trial extrapolation is abruptly switched to standard-of-care registry mortality at an arbitrary milestone (e.g., 2, 3, or 5 years). While piecewise models prevent indefinite extrapolation of immature trial effects, they introduce an artificial mathematical discontinuity: a sharp step-jump in the instantaneous hazard rate $h(t) = -\frac{d}{dt}\ln S(t)$ at the cutoff time $t_1$. Clinicians and appraisal committees routinely criticize these "cliff-edges" as biological artifacts without empirical justification.

To resolve the hazard discontinuity problem, **Che et al. (2023)** and **Zhu et al. (2026)** introduced the blended survival extrapolation framework. Under this paradigm, an arm-specific model fitted to internal trial data smoothly transitions into an external reference model informed by mature, long-term registry or standard-of-care data across a pre-specified time interval $[t_1, t_2]$:

$$S_{\text{blend}}(t) = 
\begin{cases} 
S_{\text{obs}}(t), & t < t_1 \\
\mathcal{B}\left(S_{\text{obs}}(t), S_{\text{ext}}(t); \pi(t)\right), & t_1 \le t \le t_2 \\
S_{\text{ext}}(t), & t > t_2 
\end{cases}$$

where $\pi(t) \in [0, 1]$ is a monotonic weight schedule satisfying $\pi(t_1) = 0$ and $\pi(t_2) = 1$, and $\mathcal{B}(\cdot)$ denotes a blending operator.

### Limitations of Current Heuristic Blending Parameterizations

In existing literature, the weight schedule $\pi(t)$ is modeled using the regularized incomplete Beta function (the cumulative distribution function of a $\text{Beta}(a, b)$ random variable):

$$\pi(t) = I_u(a, b) = \frac{B(u; a, b)}{B(a, b)}, \quad u = \frac{t - t_1}{t_2 - t_1} \in [0, 1]$$

While mathematically continuous, this standard approach introduces a severe methodological drawback: **it requires the analyst to subjectively specify four arbitrary parameters $(t_1, t_2, a, b)$**. 

As Zhu et al. (2026) explicitly observe, the shape parameters $a$ and $b$ possess no physical, biological, or pharmacological interpretation and cannot be informed by clinical trial evidence. Preliminary factorial simulation evidence across 144 scenarios—evaluating combinations of sample size ($N \in \{75, 250, 1000\}$), administrative censoring times ($\text{ACT} \in \{100, 360, 500, 750\}$ days), and blending interval lengths ($\{50, 150, 250\}$ days)—demonstrates that extrapolation performance is highly sensitive to the choice of $(a, b)$. 

Specifically, when trial follow-up is short (e.g., $\text{ACT} = 100$ or $360$ days) and the internal survival curve plateaus prematurely due to heavy censoring, right-skewed weight distributions such as $\text{Beta}(1, 5)$ (where $a < b$ shifts weight rapidly toward the external reference curve early in the blending window) consistently achieve the lowest bias, lowest empirical standard error (EmpSE), and highest confidence interval coverage. Conversely, left-skewed parameterizations such as $\text{Beta}(7, 3)$ or $\text{Beta}(5, 1)$ (where $a > b$ delays the transition and lingers on the uncertain internal extrapolation) induce severe positive survival bias, inflated empirical variance, and coverage collapse.

Crucially, in a prospective HTA submission, **the analyst does not possess ground-truth lifetime outcomes to select $\text{Beta}(1, 5)$ over $\text{Beta}(7, 3)$ a priori**. Choosing between arbitrary Beta shapes can dramatically swing Restricted Mean Survival Times (RMST) and Incremental Cost-Effectiveness Ratios (ICERs), exposing submissions to committee criticism of strategic parameter tuning.

### The Proposed Solution: Self-Calibrating Dynamic Blending

Rather than forcing an arbitrary, heuristic textbook Beta curve onto the extrapolation, the blending schedule must be **self-calibrating**. The transition rate must be governed by two objective, data-driven forces:
1. **Geometric Separation / Clinical Divergence:** The empirical discrepancy between the internal trial extrapolation and the external anchor.
2. **Relative Epistemic Uncertainty:** The statistical precision loss (risk-set depletion) of the internal trial model as administrative censoring depletes patient numbers.

This paper establishes an objective mathematical architecture unifying **information geometry**, **optimal transport**, and **Bayesian predictive modeling** for dynamic survival blending, providing a comprehensive framework for health economic decision modeling.

---

## 2. Mathematical Formulations of Dynamic Survival Blending

### 2.1 Information-Geometric Duality of Arithmetic and Geometric Blending

A fundamental question in survival blending is the scale upon which the blending operator $\mathcal{B}(\cdot)$ operates. In the literature, two primary formulations have emerged:

#### 1. Geometric Survival Blending / Linear Hazard Pooling (Che et al., 2023)
The blended survival function is defined as a time-varying geometric mixture:
$$S_{\text{ble}}(t) = S_{\text{obs}}(t)^{1 - \pi(t)} \cdot S_{\text{ext}}(t)^{\pi(t)}$$
Taking the negative natural logarithm reveals that this is strictly equivalent to a convex linear combination on the **cumulative hazard scale**:
$$H_{\text{ble}}(t) = (1 - \pi(t)) H_{\text{obs}}(t) + \pi(t) H_{\text{ext}}(t)$$
Differentiating with respect to $t$ yields the exact analytical blended hazard function $h_{\text{ble}}(t) = -\frac{d}{dt}\ln S_{\text{ble}}(t)$:
$$h_{\text{ble}}(t) = (1 - \pi(t)) h_{\text{obs}}(t) + \pi(t) h_{\text{ext}}(t) + \pi'(t) \ln\left(\frac{S_{\text{obs}}(t)}{S_{\text{ext}}(t)}\right)$$
and the corresponding event density $f_{\text{ble}}(t) = S_{\text{ble}}(t) h_{\text{ble}}(t)$ *(see Appendix A.1 for full derivation)*.

#### 2. Arithmetic Survival Blending / Linear Probability Mixture (Zhu et al., 2026)
$$S_{\text{blend}}(t) = (1 - \pi(t)) S_{\text{obs}}(t) + \pi(t) S_{\text{ext}}(t)$$
which directly blends survival probabilities as a mixture distribution.

#### The Directional Duality under Kullback–Leibler (KL) Divergence
From the perspective of **information geometry** (Amari, 2016), arithmetic and geometric blending are not arbitrary ad-hoc choices; they represent the two canonical directional projections of the asymmetric Kullback–Leibler divergence:
$$D_{\mathrm{KL}}(P \parallel Q) = \int p(x) \ln\left(\frac{p(x)}{q(x)}\right) dx \quad \neq \quad D_{\mathrm{KL}}(Q \parallel P)$$

Because KL divergence is strictly asymmetric, minimizing divergence between anchor models and an intermediate consensus distribution $M$ produces two fundamentally different mathematical operations:

* **Forward KL Projection (Moment-Matching / Arithmetic Mean):**
  Finding an intermediate distribution $M$ that minimizes the weighted sum of forward divergences from the parent distributions:
  $$M^* = \arg\min_M \sum_{i} w_i D_{\mathrm{KL}}(P_i \parallel M)$$
  **Mathematical Solution:** $M^*(x) = \sum_i w_i P_i(x)$.  
  *Survival Framework Equivalent:* Direct linear survival blending ($S_{\text{blend}} = (1-\pi)S_{\text{obs}} + \pi S_{\text{ext}}$). It preserves probability moments linearly.

* **Reverse KL Projection (Information Barycenter / Geometric Mean):**
  Finding an intermediate distribution $M$ that minimizes the weighted sum of reverse divergences from $M$ to each parent distribution:
  $$M^* = \arg\min_M \sum_{i} w_i D_{\mathrm{KL}}(M \parallel P_i)$$
  **Mathematical Solution:** Normalized log-linear / exponential pooling:
  $$M^*(x) = \frac{1}{Z} \prod_i P_i(x)^{w_i}$$
  *Survival Framework Equivalent:* The Che et al. (2023) geometric survival pool ($S_{\text{ble}} = S_{\text{obs}}^{1-\pi} S_{\text{ext}}^\pi$). Taking logarithms linearizes cumulative hazards: $\ln S_{\text{ble}} = (1-\pi)\ln S_{\text{obs}} + \pi \ln S_{\text{ext}}$.

| Dimension | Forward KL Projection (Zhu et al., 2026) | Reverse KL Projection (Che et al., 2023) |
| :--- | :--- | :--- |
| **Optimization Objective** | $\min_M \sum w_i D_{\mathrm{KL}}(P_i \parallel M)$ | $\min_M \sum w_i D_{\mathrm{KL}}(M \parallel P_i)$ |
| **Mathematical Operator** | Arithmetic Mean: $(1-\pi)S_{\text{obs}} + \pi S_{\text{ext}}$ | Geometric Mean: $S_{\text{obs}}^{1-\pi} S_{\text{ext}}^\pi$ |
| **Linear Scale** | Direct Survival Probability $S(t)$ | Cumulative Hazard $H(t) = -\ln S(t)$ |
| **Analytical Hazard** | $h(t) = \frac{(1-\pi)f_{\text{obs}} + \pi f_{\text{ext}} - \pi'(S_{\text{obs}}-S_{\text{ext}})}{(1-\pi)S_{\text{obs}} + \pi S_{\text{ext}}}$ | $h(t) = (1-\pi)h_{\text{obs}} + \pi h_{\text{ext}} + \pi' \ln(S_{\text{obs}}/S_{\text{ext}})$ |
| **Hazard Stability** | Can exhibit non-monotonic hazards if curves cross | Strictly additive on log-scale; well-behaved transition |

---

### 2.2 Representation of Epistemic Uncertainty & Predictive Beliefs

At each time point $t \in [t_1, t_2]$, the analyst possesses point estimates of survival $\hat{S}_{\text{obs}}(t)$ and $\hat{S}_{\text{ext}}(t)$ alongside their respective estimation uncertainties. In health economics, epistemic uncertainty can be represented across three standard modalities:

#### 1. Frequentist Delta-Method on Transformed Scales
In standard parametric survival packages (e.g., `flexsurvreg` in R), parameter uncertainty is captured by the asymptotic covariance matrix $\widehat{\boldsymbol{\Sigma}} = \mathcal{I}^{-1}(\hat{\boldsymbol{\theta}})$. The pointwise variance of estimated survival $\hat{S}(t) = g(\hat{\boldsymbol{\theta}})$ is obtained via the delta method:
$$\sigma_S^2(t) = \nabla_{\boldsymbol{\theta}} g(\hat{\boldsymbol{\theta}})^\top \widehat{\boldsymbol{\Sigma}} \, \nabla_{\boldsymbol{\theta}} g(\hat{\boldsymbol{\theta}})$$

Because survival probabilities are strictly bounded on $[0, 1]$, assuming an unconstrained Gaussian distribution directly on $\hat{S}(t)$ is a local approximation. To ensure theoretical rigor, asymptotic normality is evaluated on the unbounded **complementary log-log (cloglog) scale**:
$$Y(t) = \ln(-\ln S(t)) = \ln H(t) \in (-\infty, \infty)$$
where $\hat{Y}(t) \sim \mathcal{N}\left(\mu_Y(t), \sigma_Y^2(t)\right)$ with $\sigma_Y(t) \approx \frac{\sigma_S(t)}{\hat{S}(t) |\ln \hat{S}(t)|}$. Applying Gaussian divergence operations on $Y(t)$ strictly guarantees that transformed survival values remain within $(0, 1)$.

#### 2. Bounded Beta Belief Distributions
Alternatively, the analyst can represent epistemic uncertainty directly on $[0, 1]$ using a Beta distribution $\text{Beta}(\alpha_t, \beta_t)$ moment-matched to the point estimate $\hat{S}(t)$ and its delta-method variance $\sigma_S^2(t)$:
$$\alpha_t = \hat{S}(t)\left(\frac{\hat{S}(t)(1-\hat{S}(t))}{\sigma_S^2(t)} - 1\right), \quad \beta_t = (1-\hat{S}(t))\left(\frac{\hat{S}(t)(1-\hat{S}(t))}{\sigma_S^2(t)} - 1\right)$$
The KL divergence between two Beta distributions has an exact closed-form in terms of Digamma functions $\psi(\cdot)$ and Beta functions $B(\cdot)$:
$$D_{\mathrm{KL}}(\text{Beta}_1 \parallel \text{Beta}_2) = \ln\frac{B(\alpha_2, \beta_2)}{B(\alpha_1, \beta_1)} + (\alpha_1 - \alpha_2)\psi(\alpha_1) + (\beta_1 - \beta_2)\psi(\beta_1) - (\alpha_1 + \beta_1 - \alpha_2 - \beta_2)\psi(\alpha_1 + \beta_1)$$

#### 3. Bayesian Posterior Credible Intervals and MCMC Samples
When models are estimated via Bayesian Markov Chain Monte Carlo (MCMC) (e.g., in Stan, WinBUGS, or JAGS), the analyst has direct access to the empirical posterior draws $\{S_{\text{obs}}^{(s)}(t)\}_{s=1}^B$ and $\{S_{\text{ext}}^{(s)}(t)\}_{s=1}^B$. 
- Under this framework, **no parametric assumption is required**.
- The 1-Wasserstein optimal transport distance has an exact, closed-form empirical representation over sorted draws:
  $$W_1(t) = \frac{1}{B} \sum_{s=1}^B \left| S_{\text{obs}}^{(s)\ast}(t) - S_{\text{ext}}^{(s)\ast}(t) \right|$$
- The empirical posterior variance $\widehat{\mathrm{Var}}_{\text{post}}(S(t))$ directly informs precision weighting $\tau(t) = 1/\widehat{\mathrm{Var}}_{\text{post}}(S(t))$. If only a published 95% Credible Interval $[L(t), U(t)]$ is available from registry reports, the effective posterior precision is:
  $$\tau_{\text{ext}}(t) \approx \left(\frac{2 \times 1.96}{U(t) - L(t)}\right)^2 = \left(\frac{3.92}{U(t) - L(t)}\right)^2$$

---

### 2.3 Approach 1: Unified Divergence-Paced Clock ("Speedometer")

In Approach 1, the dynamic weight schedule $\pi(t)$ is generated organically by treating pointwise statistical divergence between the trial model and external evidence as a **divergence odometer**.

Let $D(t) = D(P_{\text{obs},t}, P_{\text{ext},t})$ denote the instantaneous discrepancy at time $t$. Under Gaussian predictive beliefs $P_{\text{obs},t} = \mathcal{N}(\hat{S}_{\text{obs}}(t), \sigma_{\text{obs}}^2(t))$ and $P_{\text{ext},t} = \mathcal{N}(\hat{S}_{\text{ext}}(t), \sigma_{\text{ext}}^2(t))$, the directed KL divergence is:
$$D_{\mathrm{KL}}(t) = \frac{1}{2}\left[\ln\left(\frac{\sigma_{\text{ext}}^2(t)}{\sigma_{\text{obs}}^2(t)}\right) + \frac{\sigma_{\text{obs}}^2(t) + \left(\hat{S}_{\text{obs}}(t) - \hat{S}_{\text{ext}}(t)\right)^2}{\sigma_{\text{ext}}^2(t)} - 1\right]$$
This formulation encapsulates two critical dynamics simultaneously:
- **Geometric Separation:** The squared deviation $(\hat{S}_{\text{obs}}(t) - \hat{S}_{\text{ext}}(t))^2$ directly penalizes trajectories where the parametric trial extrapolation diverges away from long-term registry mortality.
- **Relative Uncertainty:** As trial follow-up ends and patient-level censoring dominates the tail, $\sigma_{\text{obs}}^2(t)$ inflates rapidly while $\sigma_{\text{ext}}^2(t)$ remains tightly constrained. The variance ratio $\sigma_{\text{obs}}^2(t)/\sigma_{\text{ext}}^2(t)$ accelerates $D_{\mathrm{KL}}(t)$, signaling an objective loss of credibility in the trial model.

The cumulative weight schedule is defined as:

$$\pi_D(t) = \frac{\int_{t_1}^t w(u) D(u) \, du}{\int_{t_1}^{t_2} w(u) D(u) \, du}, \quad t \in [t_1, t_2]$$

with boundary conditions $\pi_D(t) = 0$ for $t < t_1$ and $\pi_D(t) = 1$ for $t > t_2$.

#### Boundary Regularization for Guaranteed $C^1$ Differentiability

If the cumulative divergence integral were terminated abruptly at $t_1$ or $t_2$ where $D(t) > 0$, the derivative $\pi'(t)$ would exhibit a jump discontinuity:
$$\pi'(t_1^+) > 0 \quad \text{while} \quad \pi'(t_1^-) = 0$$
In Che et al.'s hazard formula, $h_{\text{ble}}(t)$ includes the term $\pi'(t) \ln(S_{\text{obs}}/S_{\text{ext}})$. A discontinuous $\pi'(t)$ induces an instantaneous step jump in hazard, recreating the very "hazard kink" blending was designed to avoid.

To guarantee **$C^1$ differentiability of survival and $C^0$ continuity of hazard**, we introduce a smooth sine-squared window taper:

$$w(t) = \sin^2\left(\pi \frac{t - t_1}{t_2 - t_1}\right), \quad t \in [t_1, t_2]$$

Because $w(t_1) = w(t_2) = 0$ and $w'(t_1) = w'(t_2) = 0$, the tapered weight derivative satisfies:
$$\pi_D'(t_1) = 0 \quad \text{and} \quad \pi_D'(t_2) = 0$$
This guarantees that both the survival curve and the hazard function transition seamlessly into the boundaries with zero artificial curvature.

#### Table 1: Full Divergence Family Catalog for Pacing Clock $\pi_D(t)$

| Metric | Pointwise Distance $D(P_{\text{obs},t}, P_{\text{ext},t})$ | Theoretical Bounds | Sensitivity Characteristics | HTA Clinical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Kullback–Leibler (KL)** | $\frac{1}{2}\left[\ln\frac{\sigma_{\text{ext}}^2}{\sigma_{\text{obs}}^2} + \frac{\sigma_{\text{obs}}^2 + (S_{\text{obs}} - S_{\text{ext}})^2}{\sigma_{\text{ext}}^2} - 1\right]$ | $[0, \infty)$ | Asymmetric, unbounded; explodes when trial variance $\sigma_{\text{obs}}^2 \gg \sigma_{\text{ext}}^2$. | **Primary Default:** Automatically accelerates transition as trial risk set depletes. |
| **Jensen–Shannon (JSD)** | $\frac{1}{2} D_{\mathrm{KL}}(P_{\text{obs}} \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(P_{\text{ext}} \parallel M)$ | $[0, \ln 2]$ | Symmetric, bounded; strictly stable everywhere. | **Robust Transition:** Prevents premature switching caused by extreme single-point variance spikes. |
| **Hellinger ($H^2$)** | $1 - \sqrt{\frac{2\sigma_{\text{obs}}\sigma_{\text{ext}}}{\sigma_{\text{obs}}^2 + \sigma_{\text{ext}}^2}} \exp\left(-\frac{1}{4}\frac{(S_{\text{obs}} - S_{\text{ext}})^2}{\sigma_{\text{obs}}^2 + \sigma_{\text{ext}}^2}\right)$ | $[0, 1]$ | Bounded; square-root density overlap emphasizes tail mass shifts. | **Early Warning Trigger:** Sensitive detector of treatment effect waning. |
| **1-Wasserstein (EMD)** | $\|S_{\text{obs}}(t) - S_{\text{ext}}(t)\| + \text{OT}(\sigma_{\text{obs}}, \sigma_{\text{ext}})$ | $[0, \infty)$ | Linear in survival probability distance (area between curves). | **Opportunity Loss:** Directly proportional to the area between curves (Expected Opportunity Loss). |

---

### 2.4 Approach 2: Generalized Information & Metric Barycenters (Fréchet Means)

Rather than utilizing divergence solely as a scalar clock to pace a pre-selected blending formula, Approach 2 reformulates blending as an explicit **Fréchet mean optimization problem** directly in information space:

$$M^*(t) = \arg\min_M \left[ (1 - \lambda(t)) D(M, P_{\text{obs},t}) + \lambda(t) D(M, P_{\text{ext},t}) \right]$$

where $\lambda(t) = 3u^2 - 2u^3$ (with $u = \frac{t - t_1}{t_2 - t_1}$) is a baseline Hermite smoothstep progression parameter ensuring $C^1$ boundary compliance ($\lambda'(t_1) = \lambda'(t_2) = 0$).

#### 1. Reverse-KL Barycenter: Precision-Weighted Pooling
When $D(M, P_i) = D_{\mathrm{KL}}(M \parallel P_i)$, the solution is the exponential barycenter. Defining precision as inverse variance $\tau(t) = 1/\sigma^2(t)$, the intermediate cumulative hazard (or survival mean) is weighted directly by statistical precision:

$$H_{\text{bary}}(t) = \frac{(1 - \lambda(t))\tau_{\text{obs}}(t) H_{\text{obs}}(t) + \lambda(t)\tau_{\text{ext}}(t) H_{\text{ext}}(t)}{(1 - \lambda(t))\tau_{\text{obs}}(t) + \lambda(t)\tau_{\text{ext}}(t)}$$

and the blended survival curve is $S_{\text{bary}}(t) = \exp(-H_{\text{bary}}(t))$ (or directly on the survival scale, $S_{\text{bary}}(t) = \frac{(1 - \lambda(t))\tau_{\text{obs}}(t) \hat{S}_{\text{obs}}(t) + \lambda(t)\tau_{\text{ext}}(t) \hat{S}_{\text{ext}}(t)}{(1 - \lambda(t))\tau_{\text{obs}}(t) + \lambda(t)\tau_{\text{ext}}(t)}$).

**Clinical Mechanism:** As trial follow-up ends and patient-level censoring depletes the risk set, $\sigma_{\text{obs}}^2(t) \to \infty$, driving $\tau_{\text{obs}}(t) \to 0$. Even if the nominal progress clock is only at $\lambda(t) = 0.50$, the high precision of the external registry ($\tau_{\text{ext}} \gg \tau_{\text{obs}}$) mathematically dominates the ratio, **automatically snapping the blended curve to the reliable external evidence without subjective human intervention**.

#### 2. Optimal Transport (Wasserstein-2) Barycenter: Displacement Interpolation
When $D(M, P_i) = W_2^2(M, P_i)$, the solution is the displacement interpolation:
$$\mu_M(t) = (1 - \lambda(t))\mu_{\text{obs}}(t) + \lambda(t)\mu_{\text{ext}}(t), \quad \sigma_M(t) = (1 - \lambda(t))\sigma_{\text{obs}}(t) + \lambda(t)\sigma_{\text{ext}}(t)$$
In contrast to KL precision pooling where precisions add, in Wasserstein space standard deviations interpolate linearly. This prevents the "magnetic snapping" of extreme variance collapse and provides a strictly linear uncertainty bridge.

#### Table 1b: Fréchet Barycenter Taxonomy for Survival Blending (Approach 2 Family)

| Objective / Divergence | Mathematical Solution $M^*$ | Parameter / Curve Operation | HTA Clinical Utility |
| :--- | :--- | :--- | :--- |
| **Reverse-KL ($\min_M \sum w_i D_{\mathrm{KL}}(M \parallel P_i)$)** | Exponential Barycenter: $M \propto P_{\text{obs}}^{1-\lambda} P_{\text{ext}}^\lambda$ | **Precision Weighting:** $\tau_M = \sum w_i \tau_i$, $\mu_M = \frac{\sum w_i \tau_i \mu_i}{\tau_M}$ | Automatically snaps to registry when trial risk set is depleted. |
| **Forward-KL ($\min_M \sum w_i D_{\mathrm{KL}}(P_i \parallel M)$)** | Mixture Barycenter: $M = (1-\lambda)P_{\text{obs}} + \lambda P_{\text{ext}}$ | **Linear Probability Mixture:** Preserves expected survival probability linearly | Reproduces Zhu et al. linear mixture; moment-preserving. |
| **Wasserstein-2 ($\min_M \sum w_i W_2^2(M, P_i)$)** | Displacement Interpolation (McCann interpolant) | **Standard-Deviation Interpolation:** $\sigma_M = (1-\lambda)\sigma_{\text{obs}} + \lambda\sigma_{\text{ext}}$ | Smooth uncertainty bridging without extreme variance distortions. |
| **$\alpha$-Divergence ($\min_M \sum w_i D_\alpha(M, P_i)$)** | Chernoff / Rényi $\alpha$-Mixture: $M \propto \left(\sum w_i P_i^{\frac{1-\alpha}{2}}\right)^{\frac{2}{1-\alpha}}$ | **Power-Mean Interpolation:** Bridges geometric ($\alpha=-1$) and linear ($\alpha=1$) | Tunable information geometry linking hazard and survival spaces. |

#### Justifications for Linear Survival Barycenters in HTA
While geometric pooling on cumulative hazards possesses elegant hazard additivity, linear probability barycenters provide distinct methodological and clinical advantages:
1. **Mathematical Demands of the Metric:** Under Forward-KL and 1-Wasserstein metrics, the Fréchet mean is mathematically proven to be linear.
2. **Hazard Independence:** Linear pooling blends probabilities directly without imposing proportional or additive hazard constraints.
3. **Finite Population Mixture Interpretation:** In HTA decision models, $\pi(t)$ carries a clean epidemiological meaning: it represents the proportion of the patient cohort whose survival is governed by general background population mortality versus trial-specific treatment mechanisms.

---

### 2.5 Generalization to Alternative Probability Metrics (1-Wasserstein, TVD, JSD, Hellinger, MMD)

Replacing the Kullback–Leibler divergence with an alternative probability distance or divergence metric alters how the blending mechanism perceives and penalizes discrepancy between the observed trial model ($P_{\text{obs},t}$) and the external anchor ($P_{\text{ext},t}$):

#### 1. 1-Wasserstein Distance ($W_1$ / Earth Mover's Distance)
Instead of relative entropy, the 1-Wasserstein distance measures the minimal physical "work" required to transport one probability distribution into another:
- **For Gaussian predictive distributions** $\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$:
  $$W_1\left(P_{\text{obs},t}, P_{\text{ext},t}\right) = |\hat{S}_{\text{obs}}(t) - \hat{S}_{\text{ext}}(t)| + |\sigma_{\text{obs}}(t) - \sigma_{\text{ext}}(t)|$$
- **For discrete Bernoulli survival states** on $\{0, 1\}$, the 1-Wasserstein distance simplifies to:
  $$W_1\left(S_{\text{obs}}(t), S_{\text{ext}}(t)\right) = \int_0^1 |F_{\text{obs}}(u) - F_{\text{ext}}(u)| du = |S_{\text{obs}}(t) - S_{\text{ext}}(t)|$$
- **Key Properties:**
  - **Symmetric & Linear:** $W_1(P_{\text{obs}}, P_{\text{ext}}) = W_1(P_{\text{ext}}, P_{\text{obs}})$, removing subjective reference choices and avoiding quadratic/logarithmic volatility when variances diverge slowly.
  - **Strictly Bounded $[0, 1]$ on Bernoulli States:** Completely avoids $\log(0)$ numerical instabilities when $S(t) \to 0$ in the extreme tail.
  - **Direct Economic Interpretation:** In health economics, integrated 1-Wasserstein distance $\int |S_{\text{obs}}(u) - S_{\text{ext}}(u)| du$ is directly proportional to the area between survival curves—representing the physical **Expected Opportunity Loss (EVPI)**.

#### 2. Total Variation Distance (TVD)
For binary survival outcomes at time $t$, the total variation distance is:
$$\text{TVD}(t) = \frac{1}{2} \left( |S_{\text{obs}} - S_{\text{ext}}| + |(1 - S_{\text{obs}}) - (1 - S_{\text{ext}})| \right) = |S_{\text{obs}}(t) - S_{\text{ext}}(t)|$$
Notice that for pointwise Bernoulli survival states, **TVD is mathematically identical to 1-Wasserstein distance** *(see Appendix A.2)*.

#### 3. Jensen–Shannon Divergence (JSD)
The Jensen–Shannon divergence is a symmetrized, smoothed version of KL divergence computed against the midpoint mixture distribution $M = \frac{1}{2}(P_{\text{obs}} + P_{\text{ext}})$:
$$\text{JSD}(P_{\text{obs}} \parallel P_{\text{ext}}) = \frac{1}{2} D_{\mathrm{KL}}(P_{\text{obs}} \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(P_{\text{ext}} \parallel M)$$
Unlike KL divergence, JSD is strictly symmetric and bounded on $[0, \ln 2]$ (or $[0, 1]$ in base-2 logarithms), naturally tracing out an organic, numerically stable S-curve transition without risk of infinite values when tail survival drops near zero.

#### 4. Hellinger Distance ($H^2$)
Hellinger distance measures the discrepancy between the square roots of the probability densities:
$$H^2(P_{\text{obs},t}, P_{\text{ext},t}) = 1 - \sqrt{\frac{2\sigma_{\text{obs}}\sigma_{\text{ext}}}{\sigma_{\text{obs}}^2 + \sigma_{\text{ext}}^2}} \exp\left(-\frac{1}{4}\frac{(\hat{S}_{\text{obs}}(t) - \hat{S}_{\text{ext}}(t))^2}{\sigma_{\text{obs}}^2 + \sigma_{\text{ext}}^2}\right)$$
(and for Bernoulli survival states, $H^2(t) = 1 - \sqrt{S_{\text{obs}}S_{\text{ext}}} - \sqrt{(1 - S_{\text{obs}})(1 - S_{\text{ext}})}$). Hellinger distance is symmetric, strictly bounded on $[0, 1]$, and maps directly to the Bhattacharyya overlap coefficient. Because it evaluates square roots, it is disproportionately sensitive to differences in sparse tails, acting as an early-warning trigger for treatment effect waning.

#### 5. Maximum Mean Discrepancy (MMD) / Kernel Distances
MMD computes distances in a Reproducing Kernel Hilbert Space (RKHS) without assuming parametric Gaussian forms for parameter uncertainties:
$$\text{MMD}^2(P_{\text{obs}}, P_{\text{ext}}) = \mathbb{E}[k(X, X')] - 2\mathbb{E}[k(X, Y)] + \mathbb{E}[k(Y, Y')]$$
This allows dynamic blending to operate directly on non-parametric Bayesian MCMC draws from Stan/JAGS with skewness or multi-modality without delta-method approximations.

#### Table 2: Methodological Comparison & HTA Defensibility

| Framework | Parameters Required | Uncertainty Response | Hazard Smoothness |
| :--- | :--- | :--- | :--- |
| **Standard Beta (Che / Zhu)** | 4 Subjective ($t_1, t_2, a, b$) | Static; blind to sample size or variance explosion | $C^1$ continuous only if $a, b > 1$ |
| **Approach 1 (KL Clock)** | 2 Boundary Limits ($t_1, t_2$) | Dynamic; accelerates clock as trial variance inflates | Strictly $C^1$ continuous via window taper |
| **Approach 2 (Barycenter)** | 2 Boundary Limits ($t_1, t_2$) | Intrinsic; precision directly pulls curve to anchor | Strictly $C^1$ continuous via smoothstep $\lambda(t)$ |

#### Table 3: Operational Behaviors across the Two Blending Frameworks

| Metric | Under Approach 1 (Pacing Clock $\pi_D(t)$) | Under Approach 2 (Intermediate Barycenter $M_t$) |
| :--- | :--- | :--- |
| **Kullback–Leibler (KL)** | **Aggressive pacing:** As $\sigma_{\text{obs}}^2$ explodes, the clock accelerates rapidly. | **Precision weighting:** Yields inverse-variance pooling ($\tau = 1/\sigma^2$). Low-precision curves lose influence quickly. |
| **Jensen–Shannon (JSD)** | **Smooth S-curve:** Naturally self-normalizing without risk of infinite values. | **Geometric/Harmonic compromise:** Yields a balanced midpoint that softens extreme variance spikes. |
| **1-Wasserstein ($W_1$)** | **Area-proportional pacing:** Linear and gradual; less prone to sudden jumps from localized trial fluctuations. | **Quantile / Area barycenter:** The barycenter is the weighted average of quantiles, averaging survival durations directly. |
| **Hellinger ($H^2$)** | **Early alert trigger:** Transitions earlier in timeline due to square-root tail magnification. | **Affinity-weighted pooling:** Pulls heavily toward the model with broader support, avoiding degenerate estimates. |

---

## 3. The Expanded Bayesian Framework: Time-Varying BMA, Dynamic Stacking, and ELPD

Transitioning survival blending into a fully Bayesian paradigm shifts the analytical output from static point estimates with asymptotic standard errors to a full joint posterior distribution of survival trajectories:
$$p\left(S_{\text{blend}}(t) \mid \mathcal{D}_{\text{obs}}, \mathcal{D}_{\text{ext}}\right)$$
This posterior feeds seamlessly into **Probabilistic Sensitivity Analysis (PSA)**, allowing health economists to propagate structural and parameter uncertainty directly into Markov and semi-Markov cost-effectiveness models.

Bayesian survival blending can be operationalized across three increasingly sophisticated modalities:

```mermaid
graph TD
    B[Bayesian Blending Engine] --> B1[3.1 Posterior Predictive Pooling]
    B --> B2[3.2 Dynamic Bayesian Shrinkage]
    B --> B3[3.3 Time-Varying Bayesian Model Averaging BMA]
    B1 --> C1[Sample-by-sample blend across MCMC draws]
    B2 --> C2[External registry acts as time-varying informative prior]
    B3 --> C3[Pointwise ELPD & Softmax Pseudo-BMA Logic Gate]
    C3 --> D[Entropy explosion in unobserved tail automatically demotes trial model]
    D --> E[Closed-form weight: π_obs = σ_ext / (σ_obs + σ_ext)]
```

### 3.1 Posterior Predictive Pooling (Sample-by-Sample Blending for PSA)
Rather than calculating a single deterministic curve from aggregate means and variances, blending is applied across every MCMC iteration $k \in \{1, \dots, K\}$ sampled from the posterior distributions:
$$S_{\text{blend}}^{(k)}(t) = (1 - \pi^{(k)}(t)) S_{\text{obs}}^{(k)}(t) + \pi^{(k)}(t) S_{\text{ext}}^{(k)}(t)$$
The weight schedule $\pi^{(k)}(t)$ can either use the empirical posterior variance and divergence clock computed across the ensemble of posterior draws, or be evaluated dynamically per draw using the pointwise distance between the sampled trajectories. Computing this across thousands of MCMC draws produces an exact posterior predictive distribution for the blended curve. Crucially, this captures parameter covariance across all extrapolation intervals and generates robust 95% Credible Intervals for reimbursement dossiers without relying on asymptotic normal approximations.

### 3.2 Dynamic Bayesian Shrinkage (The External Anchor as an Informative Prior)
Rather than fitting two separate survival models and combining them post-hoc, dynamic blending can be formulated as a single hierarchical Bayesian model fitted to the internal trial data, where the external population curve functions as an informative, time-varying prior.

Let $\boldsymbol{\theta}$ denote the survival model parameters. The posterior distribution is:
$$p(\boldsymbol{\theta} \mid \mathcal{D}_{\text{obs}}, t) \propto p(\mathcal{D}_{\text{obs}} \mid \boldsymbol{\theta}, t) \, p(\boldsymbol{\theta} \mid \mathcal{D}_{\text{ext}})$$
- **Trial Period ($t \le \text{ACT}$):** The likelihood $p(\mathcal{D}_{\text{obs}} \mid \boldsymbol{\theta})$ contains concentrated patient data that overwhelms the external prior, allowing the posterior curve to faithfully follow the clinical trial.
- **Extrapolation Period ($t > \text{ACT}$):** As patient follow-up terminates, the likelihood provides zero additional Fisher information. By Bayes' theorem, the posterior distribution organically shrinks back toward the informative prior $p(\boldsymbol{\theta} \mid \mathcal{D}_{\text{ext}})$.
Under this formulation, blending occurs naturally without specifying explicit transition boundaries $[t_1, t_2]$.

### 3.3 Time-Varying Bayesian Model Averaging (Dynamic BMA & Stacking)

#### 1. Why Standard BMA Fails for Survival Extrapolation
Standard Bayesian Model Averaging (BMA) assigns each candidate model $M_j$ a single, global posterior model probability across the entire dataset:
$$P(M_j \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid M_j) P(M_j)}{\sum_k p(\mathcal{D} \mid M_k) P(M_k)}$$
In survival extrapolation, global BMA is fundamentally invalid: a model fitted to internal trial data might achieve superior global fit across the observed 24 months, yet produce completely disastrous, explosive survival predictions at year 10. A single static weight cannot adapt as time moves into the unobserved extrapolation window.

#### 2. The Local Predictive Solution: Pointwise Expected Log Predictive Density (ELPD)
To make BMA time-varying, we replace global marginal likelihoods with **pointwise local predictive distributions**. At each time point $t$, we evaluate the model's out-of-sample predictive accuracy using the **Expected Log Predictive Density (ELPD)**:
$$\text{ELPD}_j(t) = \mathbb{E}_{\tilde{y}_t}\left[\ln p(\tilde{y}_t \mid \mathcal{D}, M_j)\right] = \int p_{\text{true}}(\tilde{y}_t) \ln p(\tilde{y}_t \mid \mathcal{D}, M_j) \, d\tilde{y}_t$$
where $\tilde{y}_t$ denotes a hypothetical new survival observation drawn from the true data-generating process at time $t$.

#### 3. The Exact Mathematical Equivalence between ELPD and KL Divergence
A foundational theorem in Bayesian model evaluation establishes that ELPD is strictly equivalent to the negative Kullback–Leibler divergence from the true data distribution to the model's predictive distribution:
$$D_{\mathrm{KL}}\left(p_{\text{true}, t} \parallel p_{j, t}\right) = \int p_{\text{true}}(\tilde{y}_t) \ln\left(\frac{p_{\text{true}}(\tilde{y}_t)}{p(\tilde{y}_t \mid \mathcal{D}, M_j)}\right) d\tilde{y}_t = \int p_{\text{true}}(\tilde{y}_t) \ln p_{\text{true}}(\tilde{y}_t) \, d\tilde{y}_t - \text{ELPD}_j(t)$$
Defining the intrinsic differential entropy of the true survival process as $\mathcal{H}(p_{\text{true}, t}) = -\int p_{\text{true}}(\tilde{y}_t) \ln p_{\text{true}}(\tilde{y}_t) \, d\tilde{y}_t$, we obtain:

$$\text{ELPD}_j(t) = - D_{\mathrm{KL}}\left(p_{\text{true}, t} \parallel p_{j, t}\right) - \mathcal{H}(p_{\text{true}, t})$$

Because $\mathcal{H}(p_{\text{true}, t})$ is a constant property of nature that does not depend on model $M_j$, **maximizing ELPD is mathematically identical to minimizing Kullback–Leibler divergence**:
$$\arg\max_{M_j} \text{ELPD}_j(t) \equiv \arg\min_{M_j} D_{\mathrm{KL}}\left(p_{\text{true}, t} \parallel p_{j, t}\right)$$

#### 4. Softmax / Dynamic Pseudo-BMA Weights
To convert unbounded pointwise ELPD values into proper posterior model probabilities $\pi_j(t) \in [0, 1]$ satisfying $\sum \pi_j(t) = 1$, we apply the Boltzmann softmax transformation:
$$\pi_{\text{obs}}(t) = P(M_{\text{obs}} \mid \mathcal{D}, t) = \frac{\exp\left(\text{ELPD}_{\text{obs}}(t)\right)}{\exp\left(\text{ELPD}_{\text{obs}}(t)\right) + \exp\left(\text{ELPD}_{\text{ext}}(t)\right)}$$
Dividing the numerator and denominator by $\exp\left(\text{ELPD}_{\text{obs}}(t)\right)$ yields the logistic form:

$$\pi_{\text{obs}}(t) = \frac{1}{1 + \exp\left(\text{ELPD}_{\text{ext}}(t) - \text{ELPD}_{\text{obs}}(t)\right)} = \frac{1}{1 + \exp\left(\Delta\text{ELPD}(t)\right)}$$

where $\Delta\text{ELPD}(t) = \text{ELPD}_{\text{ext}}(t) - \text{ELPD}_{\text{obs}}(t)$. In terms of KL divergences, this corresponds to:
$$\pi_{\text{obs}}(t) = \frac{1}{1 + \exp\left(D_{\mathrm{KL}}^{\text{obs}}(t) - D_{\mathrm{KL}}^{\text{ext}}(t)\right)}$$

#### 5. The Unobserved Extrapolation Horizon: How Entropy Drives the Transition
In a health economic model, analysts must extrapolate 10, 20, or 40 years into the future—an unobserved horizon where zero clinical trial patients will ever be observed. *How can a predictive density shift weights in the complete absence of future data?*

The transition is governed organically by **predictive entropy explosion**.

Under Gaussian predictive beliefs $\hat{S}_j(t) \sim \mathcal{N}\left(\mu_j(t), \sigma_j^2(t)\right)$, the expected log self-predictive density of a model evaluating its own predictions is:
$$\text{ELPD}_j(t) = \mathbb{E}\left[\ln \frac{1}{\sqrt{2\pi\sigma_j^2(t)}} \exp\left(-\frac{(y - \mu_j(t))^2}{2\sigma_j^2(t)}\right)\right] = -\frac{1}{2} \ln\left(2\pi \sigma_j^2(t)\right) - \frac{1}{2}$$
The differential entropy of the predictive distribution is:
$$\mathcal{H}_j(t) = \frac{1}{2} \ln\left(2\pi e \, \sigma_j^2(t)\right)$$

* **The Trial Model ($M_{\text{obs}}$):** During trial follow-up ($t \le \text{ACT}$), trial data is abundant, $\sigma_{\text{obs}}^2(t)$ is small, and $\text{ELPD}_{\text{obs}}(t)$ is high. However, as $t$ moves beyond trial follow-up, the parameter covariance matrix $\widehat{\boldsymbol{\Sigma}}$ projects exploding variance onto survival: $\sigma_{\text{obs}}^2(t) \to \infty$. The probability mass spreads infinitely thin across the real line. Consequently:
  $$\lim_{t \to \infty} \text{ELPD}_{\text{obs}}(t) = -\infty, \quad \lim_{t \to \infty} \mathcal{H}_{\text{obs}}(t) = +\infty$$
* **The External Registry Model ($M_{\text{ext}}$):** Because the registry contains mature, long-term follow-up from thousands of patients, its parameter variance is tightly constrained. Its predictive variance $\sigma_{\text{ext}}^2(t)$ remains small and stable over the extrapolation horizon, maintaining a high, finite $\text{ELPD}_{\text{ext}}(t)$.

Now examine the difference $\Delta\text{ELPD}(t)$:
$$\Delta\text{ELPD}(t) = \text{ELPD}_{\text{ext}}(t) - \text{ELPD}_{\text{obs}}(t) = -\frac{1}{2}\ln\left(2\pi\sigma_{\text{ext}}^2\right) - \left(-\frac{1}{2}\ln\left(2\pi\sigma_{\text{obs}}^2\right)\right) = \frac{1}{2}\ln\left(\frac{\sigma_{\text{obs}}^2(t)}{\sigma_{\text{ext}}^2(t)}\right) = \ln\left(\frac{\sigma_{\text{obs}}(t)}{\sigma_{\text{ext}}(t)}\right)$$

Substituting this exact expression back into our BMA logistic weight formula yields an extraordinary result:

$$\pi_{\text{obs}}(t) = \frac{1}{1 + \exp\left(\ln\frac{\sigma_{\text{obs}}(t)}{\sigma_{\text{ext}}(t)}\right)} = \frac{1}{1 + \frac{\sigma_{\text{obs}}(t)}{\sigma_{\text{ext}}(t)}} = \frac{\sigma_{\text{ext}}(t)}{\sigma_{\text{obs}}(t) + \sigma_{\text{ext}}(t)}$$

This is an analytical breakthrough for the dynblend methodology: **time-varying Bayesian Model Averaging in the unobserved extrapolation window simplifies directly to standard-deviation / precision weighting!**
The analyst does not have to engineer artificial time-decay functions. The weight schedule $\pi_{\text{obs}}(t)$ shifts automatically from 1 to 0 purely through the mathematical physics of Bayesian entropy expansion *(see Appendix A.3 for proof)*.

#### 6. Practical Computation via Leave-One-Out (LOO) Stacking in R
In applied HTA, Bayesian models fitted in Stan or JAGS can estimate pointwise ELPD empirically using the `loo` package in R via Pareto-smoothed importance sampling (PSIS-LOO):
$$\widehat{\text{elpd}}_{\text{loo}, i} = \ln \left( \frac{\sum_{s=1}^S w_i^{(s)} p(y_i \mid \boldsymbol{\theta}^{(s)})}{\sum_{s=1}^S w_i^{(s)}} \right)$$
These pointwise out-of-sample log-likelihoods can be fed directly into stacking algorithms (Yao et al., 2018) to solve for optimal combination weights via quadratic programming, providing a fully automated, empirical Bayesian survival blend.

---

## 4. Methodological Visualizations

![Figure 1: Methodological Demonstration Dashboard. (A) Comparison of static Beta weight schedules against dynamic divergence clocks and precision barycenter schedules across blending window [100d, 360d]; (B) Blended overall survival extrapolations S(t); (C) Implied instantaneous hazard rates h(t), illustrating seamless C^1 survival differentiability and continuous hazard rates without cliff-edge kinks.](fig1_methodology_dashboard.png)

---

## 5. Monte Carlo Simulation Study Design (ADEMP Protocol)

To evaluate dynamic survival blending against current HTA practices, we conducted a factorial Monte Carlo simulation study ($N_{\text{sim}} = 1,000$ replications per scenario) adhering strictly to the **ADEMP** protocol established by Morris et al. (2019).

### 5.1 Aims
The primary objective is to evaluate whether self-calibrating dynamic blending estimators (Approach 1 Divergence Clocks, Approach 2 Precision Barycenters, and Time-Varying BMA) reduce bias, minimize empirical standard error (EmpSE), improve confidence interval coverage, and maintain hazard fidelity compared to current static Beta parameterizations and unadjusted parametric extrapolations across varying trial sample sizes and administrative censoring severities.

### 5.2 Data-Generating Mechanisms (DGM)
To ensure immediate relevance to health economics, the simulation timeline and baseline hazards are calibrated to real-world Small-Cell Lung Cancer (SCLC) clinical trial dynamics over a $T_{\text{max}} = 750$-day analytic horizon ($dt = 2$ days):

1. **True Underlying Patient Survival ($S_{\text{true}}(t)$):**
   Survival times $T_i$ are generated from a true Weibull distribution with shape $\alpha = 1.15$ and scale $\sigma = 380$ days:
   $$S_{\text{true}}(t) = \exp\left(-\left(\frac{t}{380}\right)^{1.15}\right), \quad f_{\text{true}}(t) = \frac{1.15}{380}\left(\frac{t}{380}\right)^{0.15} S_{\text{true}}(t)$$
2. **External Registry Anchor ($S_{\text{ext}}(t)$):**
   A mature, high-precision external cohort (e.g., national cancer registry, $N_{\text{ext}} = 1,500$) with follow-up extending to 750 days:
   $$S_{\text{ext}}(t) = \exp\left(-\left(\frac{t}{365}\right)^{1.20}\right), \quad \sigma_{\text{ext}}(t) \approx 0.015$$
3. **Factorial Simulation Grid (144 Scenarios):**
   The experimental conditions vary systematically across:
   - **Trial Sample Size:** $N \in \{75, 250, 1000\}$ (representing rare oncology vs. standard phase III trials).
   - **Administrative Censoring Time (ACT):** $\text{ACT} \in \{100, 360, 500, 750\}$ days (representing highly immature follow-up vs. mature trials). Censoring times are uniformly distributed $C_i \sim \text{Uniform}(\text{ACT} - 50, \text{ACT})$.
   - **Blending Window Duration:** $\tau = b - a \in \{50, 150, 250\}$ days, where blending initiates at $a = \text{ACT}$.
   - **Monte Carlo Repetitions:** $B = 1,000$ independent simulation runs per factorial cell.

### 5.3 Estimands
For each simulation run, candidate models are evaluated against five clinically critical targets:
1. **Quartile Survival Times ($T_{0.75}, T_{0.50}, T_{0.25}$):**
   $$T_q = \inf \{t : S(t) \le q\}$$
2. **Restricted Mean Survival Times ($\text{RMST}_{360}$ and $\text{RMST}_{750}$):**
   $$\text{RMST}(t^*) = \int_0^{t^*} S(u) \, du \approx \frac{\Delta}{2} \sum_{k=1}^K [S(t_{k-1}) + S(t_k)]$$
   RMST directly dictates quality-adjusted life years (QALYs) in HTA cost-effectiveness models.
3. **Curve Discrepancy via Dynamic Restricted KL Divergence (DRKL):**
   To evaluate global distributional fidelity strictly over the extrapolation window $[a, t^*]$ without dilution from the pre-blending observation period $[0, a)$ where curves coincide, we evaluate the **Dynamic Restricted Kullback–Leibler (DRKL)** divergence (extending Cox and Czanner, 2016):
   $$\text{DRKL}\left(a, t^*; f_{\text{true}}, f_{\text{ble}}\right) = \int_a^{t^*} f_{\text{true}}(u) \ln\left(\frac{f_{\text{true}}(u)}{f_{\text{ble}}(u)}\right) du$$

### 5.4 Candidate Estimators Compared
1. **Unadjusted Trial Extrapolation:** Extrapolating the fitted internal Weibull model to 750 days without blending.
2. **Piecewise Cutoff Model:** Retaining trial survival up to $t_1$, then abruptly switching the hazard to external registry hazard $h_{\text{ext}}(t)$ (TA366 analog).
3. **Static Beta Blending Grid (Che et al. / Zhu et al.):**
   - $\text{Beta}(1, 1)$: Uniform linear transition.
   - $\text{Beta}(1, 5)$: Right-skewed transition (rapid early external transfer).
   - $\text{Beta}(7, 3)$: Left-skewed transition (lingers on trial extrapolation).
   - $\text{Beta}(3, 3)$: Symmetric bell-shaped S-curve.
4. **Approach 1 Divergence Clocks (Tapered $C^1$ Continuity):**
   - KL Divergence Clock ($\pi_{\text{KL}}$)
   - Jensen–Shannon Divergence Clock ($\pi_{\text{JSD}}$)
   - Hellinger Distance Clock ($\pi_{\text{Hel}}$)
   - 1-Wasserstein Optimal Transport Clock ($\pi_{\text{Wass}}$)
5. **Approach 2 Precision Barycenter:**
   - Information Barycenter with inverse-variance precision weighting on the cumulative hazard scale.
6. **Time-Varying Bayesian Model Averaging (Dynamic BMA):**
   - Softmax Pseudo-BMA weight based on pointwise ELPD / entropy differentials.

### 5.5 Performance Measures with Monte Carlo Standard Errors (MCSE)
Following Morris et al. (2019), for each estimand $\theta$ and estimator $\hat{\theta}$, we compute:

* **Bias:**
  $$\widehat{\text{Bias}} = \frac{1}{B}\sum_{b=1}^B \hat{\theta}_b - \theta, \quad \text{MCSE}(\widehat{\text{Bias}}) = \sqrt{\frac{1}{B(B-1)}\sum_{b=1}^B (\hat{\theta}_b - \bar{\hat{\theta}})^2}$$
* **Empirical Standard Error (EmpSE):**
  $$\widehat{\text{EmpSE}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B (\hat{\theta}_b - \bar{\hat{\theta}})^2}, \quad \text{MCSE}(\widehat{\text{EmpSE}}) = \frac{\widehat{\text{EmpSE}}}{\sqrt{2(B-1)}}$$
* **Root Mean Squared Error (RMSE):**
  $$\widehat{\text{RMSE}} = \sqrt{\widehat{\text{Bias}}^2 + \widehat{\text{EmpSE}}^2}, \quad \text{MCSE}(\widehat{\text{RMSE}}) \approx \sqrt{\frac{\text{Var}((\hat{\theta} - \theta)^2)}{4 B \, \widehat{\text{RMSE}}^2}}$$
* **95% Confidence Interval Coverage:**
  $$\widehat{\text{Cov}} = \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\hat{\theta}_{\text{low}, b} \le \theta \le \hat{\theta}_{\text{upp}, b}), \quad \text{MCSE}(\widehat{\text{Cov}}) = \sqrt{\frac{\widehat{\text{Cov}}(1 - \widehat{\text{Cov}})}{B}}$$

Multi-factor performance trends across all scenarios are synthesized using **Nested Loop Plots** (Rücker and Schwarzer, 2014), while replication-level interval calibration and coverage breakdown under model misspecification are diagnosed using **Zipper Plots**.

---

## 6. Practical Guidance for Health Technology Assessment

To assist health economists and evidence synthesis teams preparing submissions for regulatory and HTA bodies (e.g., NICE, CADTH, PBAC), we propose the following practical implementation hierarchy:

```mermaid
graph TD
    A[Trial Extrapolation Required] --> B{Is Trial Sample Size or Tail Risk Set Severely Depleted?}
    B -- Yes: Rare Disease / CAR-T / Small N -- --> C[Use Approach 2: Reverse-KL Precision Barycenter]
    C --> C1[Automatic snap to external registry prevents catastrophic tail variance]
    B -- No: Standard Phase III Oncology -- --> D[Use Approach 1: KL-Paced Speedometer Clock]
    D --> D1[Self-calibrating weight schedule π_KL replaces arbitrary Beta a,b parameters]
    D1 --> E[Run Sensitivity Analysis with JSD and 1-Wasserstein Clocks]
    C1 --> E
    B -- Full Bayesian Workflow / PSA -- --> F[Use Section 3: Time-Varying BMA / Stacking]
    F --> F1[Local ELPD and entropy explosion naturally transition survival without tuning]
```

### Recommendations for HTA Submissions
1. **Base-Case Model:** We recommend **Approach 1 (KL-Paced Clock)** as the primary self-calibrating default for standard oncology submissions. It eliminates the 4-parameter burden $(a, b)$ while preserving the biological plausibility of the trial during early follow-up and transitioning smoothly into registry evidence.
2. **Small-Sample and Rare Disease Appraisals:** For CAR-T therapies, pediatric oncology, or orphan indications where patient risk sets drop below 10–15 patients rapidly, we recommend **Approach 2 (Information Barycenter)**. Its explicit precision weighting prevents wide confidence interval flaring from distorting the base-case ICER.
3. **Bayesian Models and Probabilistic Sensitivity Analysis (PSA):** In PSA iterations, rather than drawing arbitrary Beta shape parameters $(a, b)$, the dynamic weight schedule $\pi(t)$ should be recomputed at each PSA draw using that iteration's parameter covariance matrix $\widehat{\boldsymbol{\Sigma}}^{(s)}$ or MCMC sample draws, ensuring fully coherent joint uncertainty propagation. Under a full Bayesian Stan/JAGS model, time-varying BMA via pointwise ELPD provides a principled, fully automated transition engine.

---

## 7. Appendix: Complete Mathematical Derivations

### A.1 Derivation of the Implied Blended Hazard and Density Functions
Under the geometric blended survival formulation $S_{\text{ble}}(t) = S_{\text{obs}}(t)^{1 - \pi(t)} S_{\text{ext}}(t)^{\pi(t)}$, the implied probability density function $f_{\text{ble}}(t) = -\frac{\partial}{\partial t}S_{\text{ble}}(t)$ is obtained via the product and chain rules of calculus:

$$\begin{aligned}
f_{\text{ble}}(t) &= -\frac{\partial}{\partial t}\left[ S_{\text{obs}}(t)^{1 - \pi(t)} S_{\text{ext}}(t)^{\pi(t)} \right] \\
&= -\left[ \frac{\partial}{\partial t}\left( S_{\text{obs}}(t)^{1 - \pi(t)} \right) S_{\text{ext}}(t)^{\pi(t)} + S_{\text{obs}}(t)^{1 - \pi(t)} \frac{\partial}{\partial t}\left( S_{\text{ext}}(t)^{\pi(t)} \right) \right]
\end{aligned}$$

Using logarithmic differentiation, for any function $g(t) = A(t)^{B(t)}$, $\frac{\partial}{\partial t}g(t) = g(t)\left[ B'(t)\ln A(t) + B(t)\frac{A'(t)}{A(t)} \right]$. Setting $A(t) = S(t)$ and noting $S'(t) = -f(t)$:
$$\frac{\partial}{\partial t}\left(S_{\text{obs}}(t)^{1 - \pi(t)}\right) = S_{\text{obs}}(t)^{1 - \pi(t)}\left[ -\pi'(t)\ln S_{\text{obs}}(t) - (1 - \pi(t))\frac{f_{\text{obs}}(t)}{S_{\text{obs}}(t)} \right]$$
$$\frac{\partial}{\partial t}\left(S_{\text{ext}}(t)^{\pi(t)}\right) = S_{\text{ext}}(t)^{\pi(t)}\left[ \pi'(t)\ln S_{\text{ext}}(t) - \pi(t)\frac{f_{\text{ext}}(t)}{S_{\text{ext}}(t)} \right]$$

Substituting these derivatives back into the expression for $f_{\text{ble}}(t)$ and factoring out $S_{\text{ble}}(t) = S_{\text{obs}}^{1 - \pi} S_{\text{ext}}^\pi$:
$$\begin{aligned}
f_{\text{ble}}(t) &= -S_{\text{ble}}(t)\left[ -\pi'(t)\ln S_{\text{obs}}(t) - (1 - \pi(t)) h_{\text{obs}}(t) + \pi'(t)\ln S_{\text{ext}}(t) - \pi(t) h_{\text{ext}}(t) \right] \\
&= S_{\text{ble}}(t)\left[ (1 - \pi(t)) h_{\text{obs}}(t) + \pi(t) h_{\text{ext}}(t) + \pi'(t)\left(\ln S_{\text{obs}}(t) - \ln S_{\text{ext}}(t)\right) \right] \\
&= S_{\text{ble}}(t)\left[ (1 - \pi(t)) h_{\text{obs}}(t) + \pi(t) h_{\text{ext}}(t) + \pi'(t)\ln\left(\frac{S_{\text{obs}}(t)}{S_{\text{ext}}(t)}\right) \right]
\end{aligned}$$
Since $h_{\text{ble}}(t) = \frac{f_{\text{ble}}(t)}{S_{\text{ble}}(t)}$, the exact analytical hazard rate is:
$$h_{\text{ble}}(t) = (1 - \pi(t)) h_{\text{obs}}(t) + \pi(t) h_{\text{ext}}(t) + \pi'(t)\ln\left(\frac{S_{\text{obs}}(t)}{S_{\text{ext}}(t)}\right)$$
which completes the derivation. $\blacksquare$

### A.2 Proof of the 1-Wasserstein & TVD Identity for Bernoulli Survival
For two Bernoulli probability distributions $P = \{p, 1-p\}$ and $Q = \{q, 1-q\}$ on $\{0, 1\}$:
1. Total Variation Distance:
   $$\text{TVD}(P, Q) = \frac{1}{2}\sum_{x \in \{0, 1\}} |P(x) - Q(x)| = \frac{1}{2}\left( |(1 - p) - (1 - q)| + |p - q| \right) = \frac{1}{2}(|q - p| + |p - q|) = |p - q|$$
2. 1-Wasserstein Distance:
   The cumulative distribution functions are step functions on $[0, 1]$:
   $$F_P(u) = \begin{cases} 0, & u < 0 \\ 1 - p, & 0 \le u < 1 \\ 1, & u \ge 1 \end{cases}, \quad F_Q(u) = \begin{cases} 0, & u < 0 \\ 1 - q, & 0 \le u < 1 \\ 1, & u \ge 1 \end{cases}$$
   By the Kantorovich–Rubinstein theorem:
   $$W_1(P, Q) = \int_{-\infty}^\infty |F_P(u) - F_Q(u)| du = \int_0^1 |(1 - p) - (1 - q)| du = \int_0^1 |q - p| du = |p - q|$$
Therefore, $\text{TVD}(P, Q) \equiv W_1(P, Q) \equiv |p - q|$ identically on discrete binary states. $\blacksquare$

### A.3 Proof of the BMA Entropy-Collapse Theorem
Let the predictive belief distributions at time $t$ beyond follow-up be univariate Gaussians:
$$\hat{S}_{\text{obs}}(t) \sim \mathcal{N}\left(\mu_{\text{obs}}(t), \sigma_{\text{obs}}^2(t)\right), \quad \hat{S}_{\text{ext}}(t) \sim \mathcal{N}\left(\mu_{\text{ext}}(t), \sigma_{\text{ext}}^2(t)\right)$$
The expected log predictive self-density is:
$$\text{ELPD}_j(t) = \int \mathcal{N}(y; \mu_j, \sigma_j^2) \ln \mathcal{N}(y; \mu_j, \sigma_j^2) \, dy = -\frac{1}{2}\ln(2\pi\sigma_j^2) - \frac{1}{2\sigma_j^2}\mathbb{E}[(y - \mu_j)^2] = -\frac{1}{2}\ln(2\pi\sigma_j^2) - \frac{1}{2}$$
The difference in ELPD is:
$$\Delta\text{ELPD}(t) = \text{ELPD}_{\text{ext}}(t) - \text{ELPD}_{\text{obs}}(t) = \left[-\frac{1}{2}\ln(2\pi\sigma_{\text{ext}}^2) - \frac{1}{2}\right] - \left[-\frac{1}{2}\ln(2\pi\sigma_{\text{obs}}^2) - \frac{1}{2}\right] = \frac{1}{2}\ln\left(\frac{\sigma_{\text{obs}}^2(t)}{\sigma_{\text{ext}}^2(t)}\right) = \ln\left(\frac{\sigma_{\text{obs}}(t)}{\sigma_{\text{ext}}(t)}\right)$$
The softmax posterior model probability for the trial model is:
$$\pi_{\text{obs}}(t) = \frac{\exp(\text{ELPD}_{\text{obs}}(t))}{\exp(\text{ELPD}_{\text{obs}}(t)) + \exp(\text{ELPD}_{\text{ext}}(t))} = \frac{1}{1 + \exp(\Delta\text{ELPD}(t))}$$
Substituting $\Delta\text{ELPD}(t) = \ln\left(\frac{\sigma_{\text{obs}}(t)}{\sigma_{\text{ext}}(t)}\right)$:
$$\pi_{\text{obs}}(t) = \frac{1}{1 + \exp\left(\ln\frac{\sigma_{\text{obs}}(t)}{\sigma_{\text{ext}}(t)}\right)} = \frac{1}{1 + \frac{\sigma_{\text{obs}}(t)}{\sigma_{\text{ext}}(t)}} = \frac{\sigma_{\text{ext}}(t)}{\sigma_{\text{obs}}(t) + \sigma_{\text{ext}}(t)}$$
Dividing the numerator and denominator by $\sigma_{\text{obs}}(t)\sigma_{\text{ext}}(t)$:
$$\pi_{\text{obs}}(t) = \frac{1/\sigma_{\text{obs}}(t)}{1/\sigma_{\text{obs}}(t) + 1/\sigma_{\text{ext}}(t)}$$
which proves that time-varying BMA under entropy explosion simplifies strictly to inverse-standard-deviation weighting. $\blacksquare$
