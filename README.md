# predictive-alerting-smb


The solution to the predictive alerting task uses two main approaches:
- given W (window) of multivariate points, forecasting future multivariate points and alerting based on predictive error
    This is a regression problem
        Benefits:
            can be optimized on early predicting
            unsupervised learning approach (trains on unlabelled data, scored on labelled test)
        Cons:
            Not a binary predictive-alert problem for horizon H
            (Alhough the regression on one chosen model was extended to H points in the future)
- given W (window) of multivariate points, output Binary anomaly-prediction for a future horizon window H
        This is a binary classification problem
        Benefits:
            Does not predict multivariate readings, only probability score for the horizon window H
        Cons:
            Cannot be trained on unlabelled train data
            Needs supervised learning

The dataset chose:
SMD
- summary of the dataset
- 