# Risk Model Evaluation Report

## Model Details
-**Model Type:** RandomForestClassifier
-**Number of estimators:** 100
-**Random state:** 42

## Data Overview
-**Total rows in dataset:** 51035
-**Training set rows:** 38276
-**Test set rows:** 12759
-**Target column:** `risk_flag_manual`
-**Class distribution in test set:**
  -**Class 0:** 95.99%
  -**Class 1:** 4.01%

## Evaluation Metrics on Test Set
-**ROC-AUC Score:** 1.0000
-**Precision Score:** 1.0000
-**Recall Score:** 1.0000

### Confusion Matrix
```
[[12247     0]
 [    0   512]]
```

### Classification Report
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     12247
           1       1.00      1.00      1.00       512

    accuracy                           1.00     12759
   macro avg       1.00      1.00      1.00     12759
weighted avg       1.00      1.00      1.00     12759

```

## Feature Importance
The top 10 most important features are:
```
device_change_freq         0.171456
std_tx_amount_user         0.124705
ip_risk_score              0.117645
location_change_freq       0.116981
tx_amount                  0.101763
geo_distance_delta         0.066556
avg_tx_amount              0.062121
avg_tx_hour_user           0.052589
login_time_pattern_hour    0.044529
is_vpn                     0.031554
```

## Next Steps & Considerations
-**Advanced Preprocessing:** Further explore preprocessing techniques.
-**Feature Engineering:** Create new features.
-**Class Imbalance:** Address class imbalance in the dataset.
-**Model Hyperparameter Tuning:** Optimize model parameters.
-**Cross-validation:** Implement robust model evaluation.
-**Alternative Models:** Experiment with other models.
-**Deployment:** Prepare the model for API deployment.
