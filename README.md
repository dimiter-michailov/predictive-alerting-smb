# Predictive-Alerting-SMB

This repository is my solution for a predictive alerting task on multivariate metrics.

The core task formulation is:

> given the previous **W** time steps of multivariate metrics, predict whether an incident/anomaly will happen within the next **H** steps.

I built the project around two main approaches:

- **forecasting-based detection**, formulated as future regression.
- **binary classification**, formulated as probabilistic prediction on whether an anomaly will occur within a horizon `H`.

The pipeline that aligns closer to the given task formulation is the **binary classification**.
The forecasting pipeline is included as another justifiable approach and comparison.

---

## Quick comparison of the two approaches

### 1) Forecasting-based

Given a window `W` of multivariate points, output future multivariate point(s). Detect anomalies based on prediction error.

This is a **regression** problem.

#### Benefits
- can be trained on unlabeled normal-condition data
- is the more standard anomaly-detection approach
- is especially useful when labels are limited or(and) anomalies are rare

#### Cons
- not the cleanest direct formulation of the interview task
- predicts metric values first, not directly the probability of an incident within a future time window

---

### 2) Binary classification

Given a window `W` of multivariate points, output a probability that an anomaly will occur within a future horizon `H`.

This is a **binary classification** problem.

*Prediction quality is mainly judged by PR-AUC, while threshold selection is based on F1 with recall as the second priority.*

#### Benefits
- more closely matches the predictive alerting task description
- makes the *false alerting* trade-off easier to tune depending on priorities / budgets

#### Cons
- cannot train directly on unlabeled normal train data
- needs supervised labels

---

## Dataset used
The dataset used is **SMD (Server Machine Dataset)**.

I chose it because it is close to a server / cloud monitoring environment.

### Useful SMD summary
- there are **28 machines**
- each machine has **38 features**
- labels live only on the **test split**
- the official `train` split is unlabeled normal-condition data

This shapes the pipelines in the following way:
- the forecasting pipeline can train on the unlabeled normal train data
- the supervised binary classifier cannot do that, so it is built from labeled windows made from `X_test / y_test`

### What other characteristics of the dataset mattered
- strong class imbalance: positives (anomalies) are relatively rare
- **anomaly interval sizes** vary a lot, from short-lived intervals to much longer intervals
- **gap sizes between anomaly intervals** vary as well
- machines are heterogeneous in anomaly frequency and behavior
- many machines have constant features, but they are not the same on all machines
- splitting the labeled test set chronologically can make it hard to keep enough positives in train / validation / holdout
- within a single machine, features can be strongly correlated
- across machines, anomaly behavior is not globally synchronized

---

# Single-machine setup (Binary Classifier)
The workflow is:

1. create classification windows from labeled test data
2. split them chronologically into:
   - train
   - validation
   - holdout
3. flatten each multivariate window
4. train the classifier
5. choose the best-performing threshold on validation
6. evaluate once on holdout

## Multi-machine classification setup
The multi-machine extension uses **one shared classifier across multiple chosen machines**.

For each chosen machine:
- create windows separately
- split chronologically into train / validation / holdout
- keep the same feature width so pooled samples remain compatible

Then:
- pool all train windows together
- pool all validation windows together
- pool all holdout windows together

So the current multi-machine setup:
- trains one shared model on pooled train windows
- validates on the pooled validation windows
- chooses the best threshold on the pooled validation probabilities
- tests once on the pooled holdout windows

This works without padding / truncation / masking because samples are pooled **row-wise**:
each flattened window becomes one row, and machines can contribute different numbers of rows as long as the feature width is the same.

---

## Models developed

### Classification

#### HistogramGradientBoostingClassifier
- handles non-linear interactions well
- fast enough for repeated validation-based tuning
- tree-based models are a strong choice here because flattened windows produce tabular feature vectors, and boosted trees can model effects accross threshold and non-linear decision boundaries well

---

### Forecasting

#### HistogramGradientBoostingRegressor
- can model non-linear relationships
- tree-based regressors are useful here for similar reasons: they handle non-linearity and mixed signal strength well

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

Rather than relying on one arbitrary configuration, the search makes model selection more systematic.

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

## Important metrics (Binary Classifier)

1. **PR-AUC** as the main quality metric
2. **Brier score** as the next metric
3. **F1 / recall / precision at the chosen threshold** as threshold-specific metrics

### Why PR-AUC
This is an imbalanced task, so PR-AUC is more informative pure accuracy or the alternative ROC-AUC.

### Why Brier score
Good probability quality usually translates to strong performance on all other metrics.

---

## Forecasting setup
The idea is:

- predict future multivariate values
- compare prediction vs actual
- use prediction error as the anomaly score
- alert when the prediction error crosses a `threshold` (also chosen with a grid on validation)

---

## How to run
`main.py` provides the interactive entry flow for choosing:
- the pipeline
- the model / run mode
- the main configuration values such as window size and horizon

---

## Scoreboards
The repository writes experiment results to:

- `scoreboard.csv` for single-machine runs
- `scoreboard_multi_summary.csv` for multi-machine runs

The main fields to watch are:
- model / setup information
- `window_size`
- `horizon`
- applied `threshold`
- `pr_auc`
- `brier_score`
- `precision_at_threshold`
- `recall_at_threshold`
- `f1_at_threshold`

Even more complete metadata text files for individual runs are stored in `scoreboard_metadata/`.

For viewing the CSV scoreboards directly in VS Code, I suggest the **CSV** extension by **ReprEng**.

---

## Repository structure
predictive-alerting-smb/
├── data/
│   ├── per_machine/
│   └── ServerMachineDataset/
├── evaluation/
│   ├── classification_eval.py
│   ├── classification_metadata.py
│   ├── forecasting_eval.py
│   └── scoreboard_logger.py
├── models/
│   ├── classification/
│   │   └── hist_gb_classifier.py
│   └── forecasting/
│       ├── hist_gb_regressor.py
│       └── ridge_regressor.py
├── notebooks/
│   ├── eda.ipynb
│   └── multi_eda.ipynb
├── pipelines/
│   ├── classification_pipeline.py
│   ├── classification_search.py
│   ├── forecasting_pipeline.py
│   └── multi_classification_prepare.py
├── scoreboard_metadata/
├── utils/
│   ├── chrono_splitting.py
│   ├── load_data.py
│   └── windowing.py
├── main.py
├── prepare_machine_data.py
├── README.md
├── scoreboard.csv
└── scoreboard_multi_summary.csv