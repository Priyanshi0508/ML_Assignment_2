import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.title("Mobile Price Range Classification")

# Create three vertical columns (left: content, spacer, right: placeholder)
# Make the right column wider (approx 2.5x increase for prediction area)
left_col, mid_col, right_col = st.columns([6, 0.5, 7.5])

with left_col:
    model_name = st.selectbox(
        "Select Model",
        ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
    )

    # Load selected model
    model_path = f"model/saved_models/{model_name}.pkl"
    test_data_path = "model/saved_models/test_data.csv"

    if os.path.exists(model_path) and os.path.exists(test_data_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load test data (15% split from training)
        df_test = pd.read_csv(test_data_path)

        # Remove any NaN values
        df_test = df_test.dropna()

        # Prepare test data
        X_test = df_test.drop("price_range", axis=1)
        y_test = df_test["price_range"]

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Display Evaluation Metrics heading
        st.markdown("## **Evaluation Metrics**")

        # Calculate metrics
        metrics_dict = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Kappa"],
            "Value": [
                f"{accuracy_score(y_test, y_pred):.4f}",
                f"{precision_score(y_test, y_pred, average='weighted'):.4f}",
                f"{recall_score(y_test, y_pred, average='weighted'):.4f}",
                f"{f1_score(y_test, y_pred, average='weighted'):.4f}",
                f"{roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}",
                f"{cohen_kappa_score(y_test, y_pred):.4f}"
            ]
        }
        metrics_df = pd.DataFrame(metrics_dict)
        st.table(metrics_df)

        # Display Confusion Matrix
        st.markdown("## **Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        st.pyplot(fig)

        # Display Classification Report
        st.markdown("## **Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report.pop('macro avg', None)
        report.pop('weighted avg', None)
        report_df = pd.DataFrame(report).transpose()
        st.table(report_df)
    else:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
        if not os.path.exists(test_data_path):
            st.error(f"Test data not found. Run train_models.py first.")

with right_col:
    # File uploader to allow user to upload a test CSV and get predictions
    st.markdown("### Upload Test CSV")
    uploaded = st.file_uploader("Upload test.csv for predictions (drag & drop or browse)", type=["csv"], key="right_uploader")
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.drop(columns=["sc_h", "sc_w", "wifi"], inplace=True, errors="ignore")

            # Load feature names and scaler
            scaler_path = "model/saved_models/scaler.pkl"
            feat_path = "model/saved_models/feature_names.pkl"
            if not os.path.exists(scaler_path) or not os.path.exists(feat_path):
                st.error("Scaler or feature names not found. Run `model/train_models.py` first.")
            else:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                with open(feat_path, "rb") as f:
                    feature_names = pickle.load(f)

                # Ensure uploaded has required columns
                missing = [c for c in feature_names if c not in df_up.columns]
                if missing:
                    st.error(f"Uploaded file is missing columns: {missing}")
                else:
                    X_up = df_up[feature_names]
                    X_up_scaled = scaler.transform(X_up)

                    # Load model
                    if os.path.exists(f"model/saved_models/{model_name}.pkl"):
                        with open(f"model/saved_models/{model_name}.pkl", "rb") as mf:
                            user_model = pickle.load(mf)
                        preds = user_model.predict(X_up_scaled)

                        # Show original features with predictions (use full column width)
                        out_df = df_up[feature_names].copy()
                        out_df["predicted_price_range"] = preds
                        st.markdown("#### Predictions")
                        st.dataframe(out_df, use_container_width=True)
                    else:
                        st.error(f"Model file not found: model/saved_models/{model_name}.pkl")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
