# Behavioral Risk Model Evaluation Report (Realistic Version)

## Metrics Summary

- **Macro-average ROC-AUC:** 1.0000
- **Accuracy:** 1.0000
- **Out-of-Bag (OOB) Score:** 1.0000

### Confusion Matrix
```
            Predicted fraud  Predicted legit  Predicted suspicious
fraud                   269                0                     0
legit                     0            12824                     0
suspicious                0                0                   368
```

### Classification Report
```
              precision    recall  f1-score   support

       fraud       1.00      1.00      1.00       269
       legit       1.00      1.00      1.00     12824
  suspicious       1.00      1.00      1.00       368

    accuracy                           1.00     13461
   macro avg       1.00      1.00      1.00     13461
weighted avg       1.00      1.00      1.00     13461

```

### Feature Importance (Top 15)
```
time_since_last_tx            0.271017
ip_address_reputation         0.210791
tx_hour                       0.123147
tx_amount_to_balance_ratio    0.071505
location_change_freq          0.067834
device_change_freq            0.063786
tx_amount                     0.055799
country_change_flag           0.046921
transaction_velocity_10min    0.044506
account_balance               0.021711
is_new_device                 0.021104
avg_tx_amount                 0.001457
transaction_count_24h         0.000402
is_weekend                    0.000019
```
