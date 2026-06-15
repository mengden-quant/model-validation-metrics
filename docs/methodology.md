# Validation Metrics Description

## 1. Discrimination

### 1.1. Receiver Operating Characteristic

The Receiver Operating Characteristic (ROC) curve evaluates the discrimination power of a binary classification model across all possible score thresholds.

For each threshold, observations with scores above the threshold are classified as positive, and observations with scores below the threshold are classified as negative. The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR).

AUC-ROC is the area under the ROC curve. It measures how well the model ranks positive observations above negative observations.

In credit risk validation, AUC-ROC is commonly used to assess whether higher model scores are associated with higher observed default rates. AUC-ROC measures ranking quality only. It does not assess probability calibration.

#### Required inputs

The calculation requires:
* actual binary target values;
* model scores or predicted probabilities.

The target must be binary, with values 0 and 1. The score must be numeric. Higher scores are assumed to indicate a higher probability of the positive class.

#### Calculation procedure

For each score threshold $s$, define:

$$
\text{TPR}(s) = \frac{\text{TP}(s)}{P}
$$

$$
\text{FPR}(s) = \frac{\text{FP}(s)}{N}
$$

where:
* $s$ is a score threshold;
* $\text{TP}(s)$ is the number of true positives at threshold $s$;
* $\text{FP}(s)$ is the number of false positives at threshold $s$;
* $P$ is the total number of actual positives;
* $N$ is the total number of actual negatives.

The ROC curve is obtained by varying the threshold over all possible score values.

AUC-ROC can be represented as:

$$
\text{AUC-ROC} = \underset{0}{\overset{1}{\int}} \text{TPR}(\text{FPR}) d\text{FPR}
$$

Equivalently, AUC-ROC can be interpreted as the probability that a randomly selected positive observation receives a higher score than a randomly selected negative observation, with ties counted as one half.

#### Interpretation

AUC-ROC ranges from 0 to 1:
* AUC-ROC = 0.5: the model has no discrimination power and performs like random ranking;
* 0.5 < AUC-ROC < 1.0: the model has positive discrimination power;
* AUC-ROC = 1.0: the model perfectly ranks all positive observations above all negative observations;
* AUC-ROC < 0.5: the model ranks observations in the wrong direction.

#### Implementation in `valmetrics`
Function:

```python
from valmetrics.discrimination import roc_auc

auc = roc_auc(y_true, y_score, dropna=False)
```

Key implementation assumptions:
* target values (`y_true`) must be binary: 0 and 1;
* scores (`y_score`) must be numeric;
* missing values are rejected by default;
* set `dropna=True` to drop observations where either target or score is missing;
* both classes must be present in the target;
* ties follow the standard AUC-ROC convention: tied positive-negative pairs receive half credit.

---

### 1.2. Gini

The Gini coefficient is a discrimination metric commonly used in credit risk model validation. For binary default models, it measures how well the model ranks defaulted observations above non-defaulted observations.


#### Required inputs

The calculation requires:
* actual binary target values;
* model scores or predicted probabilities.

The target must be binary, with values 0 and 1. Higher scores are assumed to indicate a higher probability of the positive class.

#### Calculation procedure

The standard Gini coefficient is derived from ROC AUC:

$$
\text{Gini} = 2 \times \text{AUC-ROC} - 1
$$

The Gini coefficient can also be defined through the Cumulative Accuracy Profile (CAP) curve, where it is also known as the Accuracy Ratio. In this package, the AUC-ROC formulation is used because it gives the same standard Gini value and is directly aligned with binary ranking performance.

In credit risk validation, higher Gini generally indicates stronger ranking power. However, Gini measures discrimination only. It does not assess probability calibration.

**Conservative treatment of ties**

Standard AUC-ROC gives half credit to tied positive-negative pairs. This can be optimistic when many observations share the same score, for example in rating-grade models or heavily bucketed scorecards.

The conservative Gini adjustment penalizes mixed tied-score groups and assigns a lower discrimination value when defaults and non-defaults cannot be ordered within the same score bucket.

For a detailed discussion of conservative tie handling, see [Gini: Conservative Handling of Ties](https://www.linkedin.com/posts/alexey-mengden_gini-conservative-handling-of-ties-ugcPost-7381397726837735424-3THK/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFbp6qYBEB_ykfytA7IUHlecuioC7Dl2D0w).

#### Interpretation

The Gini coefficient ranges from -1 to 1:
* Gini = 0: the model has no discrimination power and performs like random ranking;
* 0 < Gini < 1: the model has positive discrimination power;
* Gini = 1: the model perfectly ranks all positive observations above all negative observations;
* Gini < 0: the model ranks observations in the wrong direction.

#### Implementation in `valmetrics`

```python
from valmetrics.discrimination import gini_standard, gini_conservative

gini_std = gini_standard(y_true, y_score, dropna=False)
gini_cns = gini_conservative(y_true, y_score, dropna=False)
```

Key implementation assumptions:
* target values (`y_true`) must be binary: 0 and 1;
* scores (`y_score`) must be numeric;
* missing values are rejected by default;
* set `dropna=True` to remove observations with missing target or score values;
* both target classes must be present;
* `gini_standard` follows the standard AUC-ROC tie convention;
* `gini_conservative` applies an additional penalty for mixed positive-negative tied-score groups.

---

### 1.3. Kolmogorov-Smirnov statistic

The Kolmogorov-Smirnov (KS) statistic is a discrimination metric used to measure the maximum separation between the score distributions of positive and negative observations.

In credit risk model validation, KS is commonly used to assess how well a PD model separates defaulted observations from non-defaulted observations across score thresholds.

#### Required inputs

The calculation requires:
* actual binary target values;
* model scores or predicted probabilities.

The target must be binary, with values 0 and 1. Higher scores are assumed to indicate a higher probability of the positive class.

#### Calculation procedure

For each score threshold $s$, define:

$$
\text{TPR}(s) = \frac{\text{TP}(s)}{P}
$$

$$
\text{FPR}(s) = \frac{\text{FP}(s)}{N}
$$

where:
* $s$ is a score threshold;
* $\text{TP}(s)$ is the number of true positives at threshold $s$;
* $\text{FP}(s)$ is the number of false positives at threshold $s$;
* $P$ is the total number of actual positives;
* $N$ is the total number of actual negatives.

The KS statistic is the maximum absolute distance between the two cumulative distributions:

$$
\text{KS} = \max_s \left| \text{TPR}(s) - \text{FPR}(s) \right|
$$

Equivalently, it is the maximum vertical distance between the ROC curve and the diagonal after transforming the score thresholds into cumulative positive and negative rates.

**Treatment of ties**

When multiple observations share the same score, the threshold should move across the whole tied-score group at once. The statistic should not depend on the arbitrary order of observations inside the tied group.

#### Interpretation

The KS statistic ranges from 0 to 1:
* KS = 0: the score distributions of positive and negative observations are not separated;
* 0 < KS < 1: the model provides some separation between positive and negative observations;
* KS = 1: there is perfect separation between the two classes.

A higher KS value generally indicates stronger discrimination power. However, KS measures only the single largest separation point between the two cumulative distributions. It does not assess calibration and does not summarize the full ranking performance as completely as AUC-ROC.

#### Implementation in `valmetrics`

```python
from valmetrics.discrimination import ks_statistic
ks = ks_statistic(y_true, y_score, dropna=True)
```

Key implementation assumptions:
* target values (`y_true`) must be binary: 0 and 1;
* scores (`y_score`) must be numeric;
* missing values are rejected by default;
* set `dropna=True` to remove observations with missing target or score values;
* both target classes must be present;
* tied score groups are evaluated as whole groups, not observation by observation.

---

## 2. Calibration

### 2.1. Binomial test

The binomial test is a group-level backtest for binary default models. It compares the observed number of defaults in a group with the number of defaults expected from the model probabilities.

In credit risk model validation, the test can be used to check whether observed defaults are statistically consistent with predicted probabilities within rating grades or score buckets.

#### Required inputs

The calculation requires:
* actual binary target values;
* predicted probabilities;
* predefined group labels (optionally).

The target must be binary, with values 0 and 1. Predicted probabilities must be numeric values in the range $[0; 1]$.

#### Calculation procedure

For each group $g$, define:

$$
n_g = \text{number of observations in group } g
$$

$$
O_g = \sum_{j \in g} y_j
$$

$$
E_g = \sum_{j \in g} p_j
$$

where:
* $y_j$ is the observed binary target for observation $j$;
* $p_j$ is the predicted probability for observation $j$;
* $O_g$ is the observed number of defaults in group $g$;
* $E_g$ is the expected number of defaults in group $g$.

The group-level average predicted probability is:

$$
\bar{p}_g = \frac{E_g}{n_g}
$$

The observed number of defaults is then tested against a binomial distribution:

$$
O_g \sim \text{Binomial}(n_g, \bar{p}_g)
$$

**Important methodological limitation**

The test uses a homogeneous binomial approximation within each group:

$$
O_g \sim \text{Binomial}(n_g, \bar{p}_g)
$$

This means individual predicted probabilities $p_j$ are replaced by the group average probability $\bar{p}_g$.

If individual probabilities differ materially within the same group, the exact distribution of total defaults is Poisson-binomial, not binomial. Therefore, this implementation should be interpreted as an exact binomial test under a group-level average-PD approximation.

#### Interpretation

A low p-value indicates that the observed number of defaults in a group is unlikely under the model-implied average predicted probability for that group:
* High p-value: observed defaults are statistically consistent with predicted probabilities;
* Low p-value under `greater`: observed defaults are higher than expected, indicating possible risk underestimation;
* Low p-value under `less`: observed defaults are lower than expected, indicating possible conservatism;
* Low p-value under `two-sided`: observed defaults differ materially from expected defaults in either direction.

The result should be interpreted together with group size, expected number of defaults, rating philosophy, economic cycle, and portfolio composition.

#### Implementation in `valmetrics`

This package supports two versions:
* automatic probability groups, constructed from predicted probabilities;
* predefined groups, such as rating grades or business segments.

```python
from valmetrics.calibration import binomial_test, grouped_binomial_test

bt = binomial_test(
    y_true, y_prob, n_groups=10, confidence_level=0.95, alternative="two-sided",
    dropna=False)
grouped_bt = grouped_binomial_test(
    y_true, y_prob, groups, confidence_level=0.95, alternative="two-sided",
    dropna=False)
```

Key implementation assumptions:
* target values (`y_true`) must be binary: 0 and 1;
* predicted probabilities (`y_prob`) must be in $[0; 1]$;
* missing values are rejected by default;
* set `dropna=True` to remove observations with missing target or probability values;
* `binomial_test` constructs automatic approximately equal-frequency probability groups;
* equal predicted probabilities are not split between automatic groups;
* `grouped_binomial_test` uses user-provided group labels (`groups`);
* missing group labels are rejected;
* each group is tested using its average predicted probability;
* the result includes observed defaults, expected defaults, observed default rate, average PD, p-value, and default-count/default-rate bounds for each group.

**Automatic probability groups**

In the automatic version, observations are sorted by predicted probability and assigned to approximately equal-frequency groups.

Equal predicted probabilities are not split between groups. Therefore, the actual number of groups can be lower than the requested number of groups if many observations share the same predicted probability.

This version is useful when there is no predefined rating scale and the validation analyst wants to assess calibration across the score distribution.

**Predefined groups**

In the grouped version, the user provides group labels directly. These groups may represent rating grades, score bands, product segments, regions, or other business-defined categories.

This version is usually preferable when the model is deployed with an official rating scale or when calibration needs to be assessed at business-relevant segmentation levels.

**Alternatives**

The test supports three alternatives:

* `alternative="two-sided"`: observed defaults are tested for deviation in either direction;
* `alternative="greater"`: observed defaults are tested for being higher than expected;
* `alternative="less"`: observed defaults are tested for being lower than expected.

**Acceptance bounds**

For each group, the implementation also reports default-count bounds and default-rate bounds based on the selected confidence level.

These bounds are useful as descriptive acceptance ranges for the observed number of defaults. They should be interpreted together with the p-value.

Because the binomial distribution is discrete, the p-value decision rule and the reported quantile bounds may not always lead to exactly the same borderline conclusion.

---

### 2.2. Hosmer-Lemeshow test

The Hosmer-Lemeshow test is a grouped calibration test for binary classification models. It compares observed and expected numbers of positive outcomes across groups of observations.

In credit risk model validation, the test can be used to assess whether observed defaults are consistent with predicted probabilities across probability bands, rating grades, or other validation groups.

#### Required inputs

The calculation requires:
* actual binary target values;
* predicted probabilities;
* optionally, predefined group labels.

The target must be binary, with values 0 and 1. Predicted probabilities must be numeric values in the range $[0; 1]$.

#### Calculation procedure

For each group $g$, define:

$$
n_g = \text{number of observations in group } g
$$

$$
O_g = \sum_{j \in g} y_j
$$

$$
E_g = \sum_{j \in g} p_j
$$

where:
* $y_j$ is the observed binary target for observation $j$;
* $p_j$ is the predicted probability for observation $j$;
* $O_g$ is the observed number of positives, or defaults, in group $g$;
* $E_g$ is the expected number of positives, or defaults, in group $g$.

The number of observed non-events in group $g$ is:

$$
\bar{O}_g = n_g - O_g
$$

The number of expected non-events in group $g$ is:

$$
\bar{E}_g = n_g - E_g
$$

The Hosmer-Lemeshow statistic is calculated as:

$$
\text{HL} = \sum_g \left(
\frac{(O_g - E_g)^2}{E_g}
+
\frac{(\bar{O}_g - \bar{E}_g)^2}{\bar{E}_g}
\right)
$$

Under the usual large-sample approximation, the statistic is compared with a chi-square distribution with:

$$
\text{df} = G - 2
$$

where $G$ is the number of groups used in the test.

The p-value is calculated from the upper tail of the chi-square distribution.

**Limitations**

The Hosmer-Lemeshow test relies on an asymptotic chi-square approximation. It may be unreliable when groups are small or expected defaults/non-defaults are close to zero.

The test does not explain where the calibration problem is located unless group-level observed and expected values are reviewed separately.

The test is not a discrimination metric. A model can have strong discrimination and poor calibration, or weak discrimination and acceptable group-level calibration.

#### Interpretation

A low p-value indicates that observed outcomes differ materially from expected outcomes across the selected groups:
* High p-value: no strong evidence of group-level calibration deviation;
* Low p-value: observed and expected outcomes differ more than would be expected under the chi-square approximation.

In credit risk validation, a low p-value can indicate calibration problems, but it should not be interpreted mechanically. The result depends on group construction, sample size, expected default counts, portfolio composition, and the economic period.

The Hosmer-Lemeshow test is sensitive to the number and definition of groups. Different grouping schemes can lead to different conclusions.

#### Implementation in `valmetrics`

This package supports two versions:
* automatic probability groups, constructed from predicted probabilities;
* predefined groups, such as rating grades or business segments.

```python
from valmetrics.calibration import hosmer_lemeshow, grouped_hosmer_lemeshow
hl = hosmer_lemeshow(y_true, y_prob, n_groups=10, dropna=False)
grouped_hl = grouped_hosmer_lemeshow(y_true, y_prob, groups, dropna=False)
```

Key implementation assumptions:

* target values (`y_true`) must be binary: 0 and 1;
* predicted probabilities (`y_prob`) must be in $[0; 1]$;
* missing values are rejected by default;
* set `dropna=True` to remove observations with missing target or probability values;
* `hosmer_lemeshow` constructs automatic approximately equal-frequency probability groups;
* equal predicted probabilities are not split between automatic groups;
* `grouped_hosmer_lemeshow` uses user-provided group labels;
* missing group labels are rejected;
* at least three groups are required;
* each group must have strictly positive expected defaults and expected non-defaults;
* the result includes the HL statistic, p-value, degrees of freedom, and actual number of groups used.

**Automatic probability groups**

In the automatic version, observations are sorted by predicted probability and assigned to approximately equal-frequency groups.

Equal predicted probabilities are not split between groups. Therefore, the actual number of groups can be lower than the requested number of groups if many observations share the same predicted probability.

This version is useful when there is no predefined rating scale and the validation analyst wants to assess calibration across the score distribution.

**Predefined groups**

In the grouped version, the user provides group labels directly. These groups may represent rating grades, score bands, product segments, regions, or other business-defined categories.

This version is usually preferable when the model is deployed with an official rating scale or when calibration needs to be assessed at business-relevant segmentation levels.

---

## 3. Stability

### 3.1. Population Stability Index
The Population Stability Index (PSI) measures how much a distribution has shifted between two samples.

In credit risk model validation, PSI is commonly used to monitor stability of model scores, predicted probabilities, rating grades, input variables, or portfolio segments between a reference sample and a current sample.

PSI is a stability (or representativity) metric. It does not directly measure discrimination or calibration.

#### Required inputs

The calculation requires:
* a reference sample, often called the expected sample;
* a current sample, often called the actual sample.

#### Calculation procedure

For each bin or category $i$, define:

$$
p_i = \frac{n_i^{\text{expected}}}{N^{\text{expected}}}
$$

$$
q_i = \frac{n_i^{\text{actual}}}{N^{\text{actual}}}
$$

where:
* $p_i$ is the expected-sample proportion in bin $i$;
* $q_i$ is the actual-sample proportion in bin $i$;
* $n_i^{\text{expected}}$ is the expected-sample count in bin $i$;
* $n_i^{\text{actual}}$ is the actual-sample count in bin $i$;
* $N^{\text{expected}}$ and $N^{\text{actual}}$ are the total sample sizes.

The PSI contribution of bin $i$ is:

$$
PSI_i = (q_i - p_i) \times \ln\left(\frac{q_i}{p_i}\right)
$$

The total PSI is:

$$
\text{PSI} = \sum_i \text{PSI}_i
$$

**Continuous PSI**

For continuous variables, bins are constructed from the expected sample using quantiles. The same bin edges are then applied to the actual sample.

This means the expected sample defines the reference distribution, and the actual sample is evaluated against that reference.

**Categorical PSI**

For categorical variables, PSI is calculated over the union of categories observed in the expected and actual samples.

#### Interpretation

PSI is non-negative:
* PSI = 0: the expected and actual distributions are identical;
* Higher PSI: larger distribution shift;
* Very high PSI: material population drift or segment mix change.

Common heuristic thresholds are sometimes used in practice, for example:
* $\text{PSI} < 0.1$: low shift;
* $0.1 \leq \text{PSI} < 0.25$: moderate shift;
* $\text{PSI} \geq 0.25$: high shift.

These thresholds are only rules of thumb. They should not be used mechanically without considering sample size, portfolio context, binning method, and business materiality.

PSI is sensitive to binning choices and sample size.

A high PSI does not explain whether the shift is harmful for model performance. It only indicates that the distribution changed. The shift should be reviewed together with discrimination, calibration, default-rate analysis, and business context.

For continuous variables, PSI depends on the expected-sample binning scheme. For categorical variables, rare categories can create large contributions, especially when they appear in only one sample.

#### Implementation in `valmetrics`

```python
from valmetrics.stability import psi_continuous, psi_categorical

psi_cont = psi_continuous(
    expected, actual, bins=10, missing="raise", epsilon=1e-6)
psi_cat = psi_categorical(
    expected, actual, missing="raise", epsilon=1e-6)
```

Key implementation assumptions:
* `psi_continuous` expects numeric expected and actual samples;
* `psi_categorical` expects categorical group labels;
* missing values are rejected by default;
* set `missing="drop"` to remove missing values;
* set `missing="separate"` to treat missing values as a separate bin or category;
* continuous PSI uses expected-sample quantile bins;
* continuous PSI extends the outer bin edges to $-\infty$ and $+\infty$;
* categorical PSI uses categories observed in either sample;
* proportions are smoothed using epsilon before PSI contributions are calculated;
* the result includes total PSI and per-bin or per-category contributions.

**Smoothing**

Raw PSI is undefined when $p_i = 0$ or $q_i = 0$, because the logarithm involves division by zero.

This package applies an epsilon floor to proportions before calculating PSI:

$$
p_i^{smooth} = \max(p_i, \epsilon)
$$

$$
q_i^{smooth} = \max(q_i, \epsilon)
$$

The smoothed proportions are then renormalized before calculating PSI.

Smoothing makes PSI finite when a bin or category is present in one sample but absent in the other.

**Missing values**

Missing values can be handled in three ways:
* missing="raise": reject missing values;
* missing="drop": remove missing values before calculation;
* missing="separate": treat missing values as a separate bin or category.

For continuous PSI, the separate missing bin is represented by empty bounds. For categorical PSI, the separate missing category is represented by None.

---

## 4. Diagnostics

### 4.1. Herfindahl-Hirschman Index (HHI)

The Herfindahl-Hirschman Index (HHI) is a concentration metric used to measure how concentrated observations are across groups.

In credit risk model validation, HHI can be used to assess portfolio concentration across rating grades, score buckets, industry sectors, regions, or other categorical segments. It is a diagnostic metric, not a direct measure of model discrimination or calibration.

#### Required inputs

The calculation requires:
* group labels for each observation.

The group labels may represent rating grades, score buckets, sectors, regions, or any other categorical segmentation.

#### Calculation procedure
Let $s_i$ be the share of observations in group $i$, and let $K$ be the number of groups.

The HHI is calculated as:

$$
\text{HHI} = \sum_{i=1}^{K} s_i^2
$$

where:

$$
s_i = \frac{n_i}{N}
$$

and:
* $n_i$ is the number of observations in group $i$;
* $N$ is the total number of observations;
* $K$ is the number of groups.

The minimum HHI is reached when observations are evenly distributed across groups:

$$
\text{HHI}_{\min} = \frac{1}{K}
$$

The maximum HHI is reached when all observations belong to a single group:

$$
\text{HHI}_{\max} = 1
$$

**Adjusted HHI**

The adjusted HHI normalizes the raw HHI to the range $[0; 1]$:

$$
\text{HHI}_{\text{adj}} = \frac{\text{HHI} - \frac{1}{K}}{1 - \frac{1}{K}}
$$

where $K$ is the number of groups used as the normalization baseline.

* $\text{HHI}_{\text{adj}} = 0$: observations are evenly distributed across the groups used for normalization.
* $\text{HHI}_{\text{adj}}$ = 1: all observations are concentrated in a single group.

The normalization baseline can be based either on the number of observed groups or on a predefined total number of possible groups. The second option is useful when the full rating scale or segmentation scheme contains groups that may not appear in the current sample.

#### Interpretation

Higher HHI values indicate higher concentration:
* Low HHI: observations are relatively evenly distributed across groups;
* High HHI: observations are concentrated in a small number of groups;
* HHI close to 1: most or all observations are concentrated in one group.

In model validation, high concentration may reduce the reliability of segment-level validation metrics. For example, if most observations are concentrated in one or two rating grades, calibration or default-rate analysis for other grades may be statistically weak.

HHI should be interpreted together with sample size and the business meaning of the segmentation. A high HHI is not automatically a model defect, but it can indicate limited granularity or portfolio concentration risk.

#### Implementation in `valmetrics`

```python
from valmetrics.diagnostics import herfindahl_hirschman

hhi = herfindahl_hirschman(groups, n_groups=None, normalized=False, dropna=False)
```

Key implementation assumptions:
* group labels must be one-dimensional;
* missing group labels are rejected by default;
* set `dropna=True` to remove observations with missing group labels;
* infinite numeric group labels are rejected;
* raw HHI is returned by default;
* adjusted HHI is returned when `normalized`=True`;
* if `n_groups` is not provided, adjusted HHI uses the number of observed groups as the normalization baseline;
* if `n_groups` is provided, it is treated as the total number of possible groups and must be at least the number of observed groups;
* when the normalization baseline contains only one group, adjusted HHI is defined as 1.0 by convention in this package.
---

### 4.2. Highest Concentration Index

The Highest Concentration Index (HCI) measures the largest observed share of observations assigned to a single group.

In credit risk model validation, HCI can be used to identify whether a portfolio, rating scale, score bucket structure, or segmentation is dominated by one group. It is a concentration diagnostic, not a direct measure of model discrimination or calibration.

#### Required inputs

The calculation requires:
* group labels for each observation.

The group labels may represent rating grades, score buckets, sectors, regions, or any other categorical segmentation.

#### Calculation procedure

Let $s_i$ be the share of observations in group $i$. HCI is calculated as:

$$
\text{HCI} = \max_i s_i
$$

where:

$$
s_i = \frac{n_i}{N}
$$

and:
* $n_i$ is the number of observations in group $i$;
* $N$ is the total number of observations.

#### Interpretation

HCI ranges from 0 to 1, although in a non-empty sample it is always greater than 0:
* Low HCI: no single group dominates the sample;
* High HCI: a large share of observations is concentrated in one group;
* HCI = 1: all observations belong to one group.

In model validation, high HCI may indicate limited portfolio diversification or weak segmentation granularity. For example, if one rating grade contains most observations, validation conclusions for other grades may be less reliable due to small sample sizes.

HCI is easier to interpret than HHI because it directly reports the largest group share. However, it does not describe the full distribution across all groups. Therefore, it should usually be reviewed together with HHI.

#### Implementation in `valmetrics`

```python
from valmetrics.diagnostics import hci
hcindex = hci(groups, dropna=False)
```

If several groups have the same maximum share, all such groups are returned.

Key implementation assumptions:
* group labels must be one-dimensional;
* missing group labels are rejected by default;
* set `dropna=True` to remove observations with missing group labels;
* infinite numeric group labels are rejected;
* the function returns both the maximum concentration value and all groups sharing that maximum value.
