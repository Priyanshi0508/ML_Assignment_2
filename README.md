# Mobile Price Range Classification

This repository contains code and models for classifying mobile phones into price ranges.

## Problem Statement

Build and evaluate machine learning models that predict the `price_range` of a mobile phone (discrete classes 0, 1, 2, 3) from hardware and software features. The goal is to compare multiple classifiers and report evaluation metrics computed on a held-out 15% test split.

## Dataset Description

The dataset is provided in `Data/train.csv` (2001 examples) and contains the following columns:

- `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`, `price_range`.

`price_range` is the target with four classes (0: low, 1: medium, 2: high, 3: very high). Some preprocessing in the scripts drops a few columns (e.g., `sc_h`, `sc_w`, `wifi`) before training.

## Project Structure

- `app.py` - Streamlit app for model selection, metrics display, and predictions
- `model/train_models.py` - Trains models and saves them to `model/saved_models/`
- `model/evaluate_models.py` - Evaluates saved models
- `Data/` - Contains `train.csv` and `test.csv` used for training/evaluation
- `model/saved_models/` - Saved models, scaler, feature names, test split and metrics


## Evaluation Metrics

<!-- METRICS_START -->

| Model | Accuracy | AUC | Precision | Recall | F1 | Kappa |
|---|---|---|---|---|---|---|
| logistic | 0.9667 | 0.9989 | 0.9669 | 0.9667 | 0.9667 | 0.9556 |
| decision_tree | 0.8567 | 0.9044 | 0.8589 | 0.8567 | 0.8575 | 0.8089 |
| knn | 0.5233 | 0.7775 | 0.5551 | 0.5233 | 0.5317 | 0.3644 |
| naive_bayes | 0.8033 | 0.9518 | 0.8038 | 0.8033 | 0.8034 | 0.7378 |
| random_forest | 0.92 | 0.9891 | 0.9203 | 0.92 | 0.9198 | 0.8933 |
| xgboost | 0.94 | 0.9968 | 0.94 | 0.94 | 0.9399 | 0.92 |

<!-- METRICS_END -->

## Observations
| Model | Observation |
|---|---|
| Logistic Regression | Shows excellent and well-balanced performance across all metrics, indicating strong linear separability and reliable predictions. |
| Decision Tree | Performs reasonably well but lags behind ensemble methods, suggesting overfitting or limited generalization.|
| K-Nearest Neighbors (KNN) | Delivers poor performance, indicating sensitivity to feature scaling, noise, or an unsuitable choice of k. |
| Naive Bayes | Achieves decent accuracy and high AUC, but lower overall performance suggests its independence assumption is limiting. |
| Random Forest | Provides strong and stable results with high AUC and Kappa, demonstrating effective ensemble learning. |
| XGBoost | Achieves near-optimal performance across all metrics, making it the best-performing model on this dataset. |

