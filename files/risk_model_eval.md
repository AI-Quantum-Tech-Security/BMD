# Risk Model Evaluation Report

## Model Details
-**Model Type:** RandomForestClassifier
-**Number of estimators:** 100
-**Random state:** 42

## Data Overview
-**Total rows in dataset:** 51400
-**Training set rows:** 38550
-**Test set rows:** 12850
-**Target column:** `risk_flag_manual`
-**Class distribution in test set:**
  -**Class 0:** 69.74%
  -**Class 1:** 30.26%

## Evaluation Metrics on Test Set
-**ROC-AUC Score:** 0.9074
-**Precision Score:** 0.8701
-**Recall Score:** 0.8284

### Confusion Matrix
```
[[8481  481]
 [ 667 3221]]
```

### Classification Report
```
              precision    recall  f1-score   support

           0       0.93      0.95      0.94      8962
           1       0.87      0.83      0.85      3888

    accuracy                           0.91     12850
   macro avg       0.90      0.89      0.89     12850
weighted avg       0.91      0.91      0.91     12850

```

## Feature Importance
The top 10 most important features are:
```
tx_hour                  0.273522
timestamp_hour           0.271178
device_change_freq       0.095667
location_change_freq     0.084022
tx_amount                0.083762
avg_tx_amount            0.080129
timestamp_month          0.078605
timestamp_day_of_week    0.025972
is_weekend               0.005272
is_new_device            0.001870
```

## Next Steps & Considerations
-**Advanced Preprocessing:** Further explore preprocessing techniques.
-**Feature Engineering:** Create new features.
-**Class Imbalance:** Address class imbalance in the dataset.
-**Model Hyperparameter Tuning:** Optimize model parameters.
-**Cross-validation:** Implement robust model evaluation.
-**Alternative Models:** Experiment with other models.
-**Deployment:** Prepare the model for API deployment.
