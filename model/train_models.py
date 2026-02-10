import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, cohen_kappa_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(SAVE_DIR, exist_ok=True)
# Load dataset
df = pd.read_csv("../Data/train.csv")

# Drop columns
df.drop(columns=["sc_h", "sc_w", "wifi"], errors="ignore", inplace=True)
df = df.dropna()
X = df.drop("price_range", axis=1)
y = df["price_range"]


# Train-test split (85:15 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# Save scaler and feature names for inference
with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

feature_names = list(X.columns)
with open(os.path.join(SAVE_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    with open(os.path.join(SAVE_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # Calculate evaluation metrics on test data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
        "AUC": f"{roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}",
        "Precision": f"{precision_score(y_test, y_pred, average='weighted'):.4f}",
        "Recall": f"{recall_score(y_test, y_pred, average='weighted'):.4f}",
        "F1": f"{f1_score(y_test, y_pred, average='weighted'):.4f}",
        "Kappa": f"{cohen_kappa_score(y_test, y_pred):.4f}"
    })

results_df = pd.DataFrame(results)
print("\nEvaluation Metrics on 15% Test Data:")
print(results_df)

# Save metrics for README/update script
results_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)

# Save test data for streamlit app
test_data = pd.DataFrame(X_test)
test_data['price_range'] = y_test
test_data.to_csv(os.path.join(SAVE_DIR, "test_data.csv"), index=False)

print("\nAll models trained and saved successfully.")
