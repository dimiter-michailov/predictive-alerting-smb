# Predictive-Alerting-SMB

This repository is my solution for a predictive alerting / incident prediction task on multivariate metrics.

The core task is:

> given the previous **W** time steps of one or more multivariate metrics, predict whether an incident / anomaly will happen within the next **H** time steps.

I built the project around two main approaches:

- **forecasting-based anomaly detection**, formulated as single-step and multi-step future regression
- **binary classification for predictive alerting**, formulated as predicting whether an anomaly will occur within a given future horizon `H`

The main pipeline for the actual task is the **binary classification** one.  
The forecasting pipeline is included as another very justifiable approach and comparison.

---

## Quick comparison of the two approaches

### 1) Forecasting-based anomaly detection

Given a window `W` of multivariate points, predict future multivariate point(s), then detect anomalies based on prediction error.

This is a **regression** problem.

#### Benefits
- can be trained on unlabeled normal data
- is the more standard anomaly-detection formulation
- is useful especially in circumstances when labels are limited

#### Cons
- not the cleanest direct formulation of the interview task, which is really about predicting whether an anomaly will happen within the next horizon `H`
- predicts metric values first, not directly the probability of an incident within a future time window

---

### 2) Binary classification for predictive alerting

Given a window `W` of multivariate points, predict whether an anomaly will occur within a future horizon window `H`.

This is a **binary classification** problem.

The target is:
- `1` if any anomaly happens within the next `H` steps
- `0` otherwise

The model outputs a probability, which is then converted into an alert using a chosen threshold.  
Model selection is mainly driven by PR-AUC, while threshold selection is based on F1 with recall/precision tie-breaking.

#### Benefits
- directly matches the predictive alerting task
- outputs a probability score that can be thresholded into an alert
- makes the operating point explicit and tunable depending on alerting priorities

#### Cons
- cannot train directly on unlabeled normal train data
- needs supervised labels

This is the main focus of the project.

---

## Dataset used

The dataset used is **SMD (Server Machine Dataset)**.

I chose it because it is close to server / cloud monitoring.

### Useful SMD facts for this project
- there are **28 machines**
- each machine has **38 features**
- labels live only on the **test split**
- the official `train` split is normal / unlabeled

That matters a lot for how the pipelines are built.

For example:
- the forecasting pipeline can train on the unlabeled normal train data
- the supervised binary classifier cannot do that, so it is built from labeled windows made from `X_test / y_test`

### What characteristics of the dataset mattered for this task
For predictive alerting, it is not just about having labels.

What matters a lot is:
- **positive rate**
- **anomaly interval sizes**
- **gap sizes between anomaly intervals**

From exploratory analysis, the anomaly intervals are not all tiny isolated spikes. There are both short and long anomalous segments, and the gaps between intervals are often large enough to make horizon-based labeling meaningful. That makes the dataset more suitable for predictive alerting than a setup where anomalies are either too dense or too fragmented.

---

## Binary classification setup

### For the single-machine setup
The workflow is:

1. create classification windows from labeled test data
2. split them chronologically into:
   - train
   - validation
   - holdout
3. flatten each multivariate window
4. train the classifier
5. choose the threshold automatically on validation
6. evaluate once on holdout

## Multi-machine classification setup

The multi-machine extension uses **one shared classifier across multiple chosen machines**.

For each chosen machine:
- create windows separately
- split chronologically into train / validation / holdout

Then:
- pool all train windows together
- pool all validation windows together
- pool all holdout windows together

So the current multi-machine setup is:
- train one shared model on all chosen machines
- validate on the pooled validation windows of those machines
- choose the best threshold on the pooled validation probabilities
- test once on the pooled holdout windows

---

## Models developed

### Classification

#### HistogramGradientBoostingClassifier
This is the main classifier used for predictive alerting.

Used for:
- single-machine classification
- multi-machine classification

Why this model:
- handles non-linear interactions well
- fast enough for repeated validation-based tuning
- strong and practical tabular baseline for structured time-window features
- gradient-boosted tree models are widely used in industry for tabular prediction problems because they often give strong performance without heavy feature engineering

---

### Forecasting

#### HistogramGradientBoostingRegressor
Used for forecasting-based anomaly detection.

Current use:
- single future-point prediction baseline
- validation-based tuning
- train on unlabeled normal train data
- test on labeled test data

Why this model:
- can model non-linear relationships in multivariate tabular windows
- boosted tree regressors are also a common industry choice when the problem is framed as supervised regression on engineered tabular features

#### Ridge
Used as a simple forecasting baseline.

It is there mainly as a reference point against the tree-based regressor.

---

## Hyperparameter search

The tuned hyperparameters are:
- `learning_rate`
- `max_iter`
- `max_leaf_nodes`
- `min_samples_leaf`

### Search logic
The search is done in stages:

- **Stage 1**: reduced pass over a smaller set of branches with `max_iter=100`
- **Stage 2**: refine only the strongest branches with larger `max_iter`

This keeps the search more computationally reasonable than brute-forcing everything.

---

## Threshold selection

The threshold is **not manually entered**.

Instead:
- get validation probabilities
- search a sensible threshold grid on validation
- choose the best validation threshold

### Threshold rule
Current rule:
1. choose the best **F1**
2. if F1 is very close, prefer higher **recall**
3. if still very close, prefer higher **precision**

---

## Important metrics

1. **PR-AUC** as the main quality metric
2. **Brier score** as the next metric
3. **F1 / recall / precision at the chosen threshold** as operating-point metrics

### Why PR-AUC
This is an imbalanced task, so PR-AUC is more useful here than ROC-AUC for the main report.

### Why Brier score
The classifier outputs probabilities, so probability quality matters too, not just ranking quality.

---

## Forecasting setup

The idea is:

- predict future multivariate values
- compare prediction vs actual
- use prediction error as the anomaly score
- alert when the prediction error crosses a threshold

This is useful because it can train on unlabeled normal data.

---

## Repository structure

```text
predictive-alerting-smb/
├── data/
├── evaluation/
├── models/
├── notebooks/
├── pipelines/
│   ├── classification_pipeline.py
│   └── forecasting_pipeline.py
├── scoreboard_metadata/
├── utils/
├── main.py
├── prepare_machine_data.py
├── README.md
├── scoreboard.csv
└── scoreboard_multi_summary.csv