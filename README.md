# Preference Shift Detection

**Diagnosing and forecasting distribution shift in sequential preference models**

This project studies when uncertainty and observed preference feedback can detect, diagnose, and potentially forecast failures in sequential preference models.

The full research proposal is available [here](docs/research_proposal_preference_shift.pdf).

## Overview

Pairwise preference models are used to learn latent reward functions from comparisons between alternatives. In sequential applications, however, the data-generating process may not remain stationary. The distribution of compared alternatives may change, the underlying preference rule may evolve, or both processes may interact over time.

These mechanisms can produce similar changes in predictive performance while requiring different interpretations. A model may encounter unfamiliar comparisons, previously hidden estimation errors may become relevant, or the true preference function may no longer agree with the model learned from historical data.

The primary research question is:

> Under which distribution shifts do uncertainty and prediction surprise provide reliable and distinguishable signals of failure in a sequential preference model?

A later stage considers a prospective question:

> When only some covariate shifts are followed by delayed preference drift, can the geometry of the covariate shift forecast which shifts are genuine precursors?

The project uses controlled synthetic data and a deliberately simple probabilistic preference model. This makes it possible to isolate the relevant statistical mechanisms, evaluate detectors against known change points, and distinguish irreducible outcome randomness from model error.

## Problem Setting

At time $t$, two alternatives $A$ and $B$ are represented by feature vectors

$$
x_{A,t},x_{B,t}\in\mathbb{R}^d.
$$

Their feature difference is

$$
z_t=x_{A,t}-x_{B,t}.
$$

The latent reward function is linear:

$$
r_t(x)=w_t^\top x.
$$

The preference outcome $y_t=1$ denotes that $A$ is preferred to $B$. Preferences are generated according to a covariate Bradley–Terry model:

$$
y_t\mid z_t
\sim
\operatorname{Bernoulli}\left(p_t^*(z_t)\right),
\qquad
p_t^*(z_t)
=
\sigma\left(w_t^\top z_t\right),
$$

where

$$
\sigma(a)=\frac{1}{1+e^{-a}}.
$$

An initial stationary sample is used to estimate the preference parameter. Using Bayesian logistic regression and a Laplace approximation, the posterior is represented as

$$
w\mid\mathcal D_0
\approx
\mathcal N(\hat w,\Sigma).
$$

The estimated model predicts

$$
\hat p_t
=
\sigma\left(\hat w^\top z_t\right).
$$

## Distribution-Shift Settings

The initial experiments distinguish three fundamental settings:

| Setting          | Comparison distribution  | Preference rule       |
| ---------------- | ------------------------ | --------------------- |
| Covariate shift  | $Q_t(z)$ changes       | $w_t$ remains fixed |
| Preference shift | $Q_t(z)$ remains fixed | $w_t$ changes       |
| Combined shift   | $Q_t(z)$ changes       | $w_t$ changes       |

Covariate shifts are not assumed to be uniformly harmful. A shift may move comparisons into well-understood directions, poorly identified directions, or directions that have little influence on preference predictions.

This distinction allows the project to study not only whether a distribution changed, but whether the change is relevant to the reliability of the preference model.

## Detection Signals

### Epistemic uncertainty

The uncertainty in the latent utility difference for comparison $z_t$ is

$$
u_t^2=z_t^\top\Sigma z_t.
$$

This quantity is large when a comparison depends strongly on feature directions for which the preference weights are poorly identified.

Its expected value under comparison distribution $Q_t$ is

$$
\mathbb E_{Q_t}[u_t^2]
=
\operatorname{tr}
\left(
\Sigma\operatorname{Cov}_{Q_t}(z)
\right)
+
\mathbb E_{Q_t}[z]^\top
\Sigma
\mathbb E_{Q_t}[z].
$$

This provides a geometric description of when covariate shift should affect model uncertainty.

### Prediction surprise

Once the preference outcome is observed, disagreement between the model and the outcome is measured using predictive log loss:

$$
\ell_t
=
-y_t\log\hat p_t
-
(1-y_t)\log(1-\hat p_t).
$$

A one-sided CUSUM detector monitors whether the loss remains persistently above its pre-change level:

$$
S_t
=
\max
\left\{
0,
S_{t-1}+\ell_t-\mu_0-k
\right\}.
$$

An alarm is raised when $S_t$ exceeds a calibrated threshold.

### Accuracy baseline

Rolling preference accuracy provides a simple baseline. It discards information about predicted probabilities and is therefore expected to respond more slowly than log-loss monitoring in some settings.

All detector thresholds will be calibrated to a common pre-change false-alarm rate.

## Research Hypotheses

The first hypothesis is that epistemic uncertainty reacts primarily when the comparison distribution moves into weakly identified feature directions. Such an increase does not necessarily imply model failure, and harmful covariate shift may remain undetected when the model is confidently misspecified.

The second hypothesis is that pure preference drift with an unchanged comparison distribution cannot be detected from $z_t$ alone. Preference outcomes are required, after which log-loss monitoring should detect systematic disagreement more efficiently than rolling accuracy.

The third hypothesis is that uncertainty and prediction surprise jointly distinguish some shift mechanisms, but cannot identify every cause without additional structural assumptions. An important objective is therefore to characterize both identifiable and non-identifiable cases.

## Forecasting Delayed Preference Drift

The final stage studies whether covariate changes can provide advance warning of future preference drift.

A covariate shift occurs at time $\tau_c$. Some shifts remain benign, while others affect the preference parameter after a delay $L$:

$$
w_{\tau_c+L}=w_0+\Delta w.
$$

Let

$$
\Delta\mu
=
\mathbb E_{Q_{\mathrm{after}}}[z]
-
\mathbb E_{Q_{\mathrm{before}}}[z].
$$

A controlled exposure mechanism models the induced preference change as

$$
\Delta w=\alpha B\Delta\mu,
$$

where $B$ is a fixed low-rank preference-susceptibility map specified independently of the detector. Shifts in the null space of $B$, including selected covariance-only shifts, do not induce preference drift. Separate preference shifts without covariate precursors are also included.

The study will test whether an uncertainty-weighted change statistic

$$
R_u
=
\left|
\mathbb E_{Q_{\mathrm{after}}}
\left[z^\top\Sigma z\right]
-
\mathbb E_{Q_{\mathrm{before}}}
\left[z^\top\Sigma z\right]
\right|
$$

predicts subsequent preference drift more reliably than an unweighted description of the covariate shift.

If benign and preference-inducing shifts are observationally identical before the preference change, forecasting is impossible. This provides a negative control and clarifies which structural assumptions are necessary for genuine early warning.

## Evaluation

The principal evaluation quantities are:

* False-alarm probability
* Detection delay
* Lead time before meaningful model degradation
* Predictive log loss and excess log loss
* Rolling preference accuracy
* Uncertainty calibration
* Discrimination between benign and preference-inducing covariate shifts

Because the experiments are synthetic, the true parameters and change points are known. This allows observed log loss to be separated into irreducible Bernoulli entropy and additional loss caused by model mismatch.

## Scope

The project intentionally retains a simple pairwise-comparison setup throughout. The objective is to develop a precise statistical understanding of uncertainty, preference feedback, and delayed drift without introducing unnecessary model complexity.

The initial study uses low-dimensional synthetic data, a linear Bradley–Terry preference model, and a small number of interpretable monitoring methods. Possible later extensions include nonlinear reward models, learned representations, real preference datasets, and applications to reward modeling for foundation models.

## Status

Research design and initial implementation in progress.

**Author:** Marvin Ernst
**Started:** August 31, 2026
