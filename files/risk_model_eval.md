# Raport Ewaluacji Modelu Ryzyka Behawioralnego (Wersja Realistyczna)

## Podsumowanie Metryk

- **Macro-average ROC-AUC:** 1.0000
- **Dokładność (Accuracy):** 1.0000
- **Out-of-Bag (OOB) Score:** 1.0000

### Macierz Pomyłek
```
            Predicted fraud  Predicted legit  Predicted suspicious
fraud                   263                0                     0
legit                     0            12799                     0
suspicious                0                0                   365
```

### Raport Klasyfikacyjny
```
              precision    recall  f1-score   support

       fraud       1.00      1.00      1.00       263
       legit       1.00      1.00      1.00     12799
  suspicious       1.00      1.00      1.00       365

    accuracy                           1.00     13427
   macro avg       1.00      1.00      1.00     13427
weighted avg       1.00      1.00      1.00     13427

```

### Ważność Cech (Top 15)
```
time_since_last_tx            0.265158
ip_address_reputation         0.209976
tx_hour                       0.124983
device_change_freq            0.073620
tx_amount                     0.064936
tx_amount_to_balance_ratio    0.062592
location_change_freq          0.058763
transaction_velocity_10min    0.045046
country_change_flag           0.042841
is_new_device                 0.029229
account_balance               0.019477
avg_tx_amount                 0.003068
transaction_count_24h         0.000281
is_weekend                    0.000029
```
