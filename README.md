# Preference Shift Detection

Detecting distribution shift and model failure in sequential preference learning.

## Overview

Preference models are increasingly used to learn latent reward functions from pairwise comparisons. In many applications, however, the data-generating process is not stationary. The distribution of observed alternatives may change over time, underlying preferences may shift, or both may occur simultaneously.

This project studies whether uncertainty estimates and sequential change-point detection methods can provide early warning of model failure in such settings.

The main research question is:

> Under which types of distribution shift can predictive uncertainty provide early warning of failure in a pairwise preference model, and when is feedback-based change-point detection necessary?

The initial focus is on a controlled synthetic setting using probabilistic pairwise preference models. This allows the behavior of different detection methods to be studied under precisely specified changes in the data-generating process.

## Problem Setting

Consider items represented by feature vectors

$$
x \in \mathbb{R}^d
$$

with a latent reward function

$$
r_t(x) = w_t^\top x.
$$

Given two items $x_A$ and $x_B$, pairwise preferences are generated using a Bradley-Terry style model:

$$
P(A \succ B) = \sigma\left(w_t^\top(x_A-x_B)\right).
$$

At an unknown change point, the environment may change.

The project will initially consider three settings:

1. **Covariate shift**
   The distribution of observations changes while the underlying preference function remains fixed.

2. **Preference shift**
   The latent reward or preference function changes while the observation distribution remains fixed.

3. **Combined shift**
   Both the observation distribution and the preference function change.

## Research Questions

The initial questions are:

* Can predictive uncertainty detect an upcoming failure of a preference model before predictive performance deteriorates substantially?
* How does uncertainty-based detection behave differently under covariate shift and preference shift?
* Can sequential detectors based on prediction error or predictive log loss detect changes that uncertainty alone cannot identify?
* Are uncertainty-based and feedback-based detection methods complementary?

## Initial Approach

The project will begin with a probabilistic linear preference model and synthetic pairwise comparison data.

Candidate monitoring methods include:

* predictive uncertainty
* sequential monitoring of predictive log loss
* CUSUM or Page-Hinkley style change detection
* rolling predictive accuracy as a simple baseline

Evaluation will focus on quantities such as:

* detection delay
* false alarm rate
* predictive performance after a change
* uncertainty calibration
* lead time between detection and meaningful model failure

## Motivation

A learning system may remain confident even after the relationship between observations and outcomes has changed. This is particularly important for preference and reward models, where a model may continue assigning confident scores despite no longer representing the current preference function.

Understanding when uncertainty provides a useful warning signal, and when labeled feedback is necessary to detect a change, is relevant to sequential learning, probabilistic machine learning, and reward modeling.

## Scope

The first stage of the project intentionally uses a small synthetic setting. The goal is to obtain a clear understanding of the underlying statistical behavior before introducing more complex models or real-world preference data.

Possible later extensions include nonlinear reward models, learned representations, real preference datasets, and applications to reward modeling for foundation models.

## Status

Research and initial implementation in progress.

Author: Marvin Ernst

Date: August 31, 2026
