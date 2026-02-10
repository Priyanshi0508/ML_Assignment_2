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

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

Train models and save artifacts:

```bash
python model/train_models.py
```

Start the Streamlit app:

```bash
streamlit run app.py
```

## Evaluation Metrics

The table below is automatically updated with evaluation metrics computed on the 15% test split saved when running `model/train_models.py`.

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

## Notes

- To update the metrics in this README after retraining, run:

```bash
python update_readme.py
```

This script reads `model/saved_models/metrics.csv` and replaces the metrics table above.

## Observations

Below are concise observations on each model's performance based on the Evaluation Metrics table above. Interpretations reference the metrics (Accuracy, AUC, Precision, Recall, F1, Kappa) shown in the table.

| Model | Observation |
|---|---|
| Logistic Regression | Strong overall accuracy and AUC indicate good linear separability for the majority of classes. Stable precision/recall values suggest balanced performance across classes. Good baseline model. |
| Decision Tree | Lower accuracy and AUC compared to ensemble methods; may be overfitting on training data but underperforming on the held-out 15% test split. Consider pruning or limiting depth. |
| K-Nearest Neighbors (KNN) | Relatively low accuracy and F1 indicate KNN struggles with this feature space (possibly due to scale or high dimensionality). Performance sensitive to `k` and feature scaling. |
| Naive Bayes | Moderate performance; works well when feature independence assumption roughly holds. Fast to train but may be outperformed by tree-based ensembles on this dataset. |
| Random Forest | High accuracy and AUC, with strong precision/recall and F1 — indicates ensemble reduces variance and captures non-linear patterns. Reliable choice for this dataset. |
| XGBoost | Competitive or best-performing model in terms of AUC and accuracy; captures complex interactions and offers strong generalization when properly tuned. Good candidate for final model. |

These observations are qualitative summaries derived from the metrics table; run `python update_readme.py` after retraining to refresh the numeric table and confirm observations quantitatively.
