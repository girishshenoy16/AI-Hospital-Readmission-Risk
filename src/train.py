import pandas as pd
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def load_engineered_data(path: str):
    """Load engineered dataset"""
    df = pd.read_csv(path)
    X = df.drop(columns=['readmitted_flag'])
    y = df['readmitted_flag']
    return X, y


def train_model(data_path: str, model_path: str, metadata_path: str):
    """Train readmission prediction model"""

    X, y = load_engineered_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # Model configuration
    # -----------------------------
    model_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 8,
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = XGBClassifier(**model_params)

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    threshold = 0.25
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"🎯 Model ROC-AUC: {auc:.4f}")

    # -----------------------------
    # Save model
    # -----------------------------
    joblib.dump(model, model_path)
    print("✅ Model saved successfully.")

    # -----------------------------
    # Save model metadata
    # -----------------------------
    model_metadata = {
        "model_name": "XGBoost Readmission Risk Model",
        "model_type": "XGBClassifier",
        "training_date": datetime.utcnow().isoformat(),
        "roc_auc": round(auc, 4),
        "threshold": threshold,
        "intended_use": "Readmission risk stratification (not diagnosis)",
        "features_count": X.shape[1],
        "model_parameters": model_params
    }

    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=4)

    print("📄 Model metadata saved successfully.")


if __name__ == "__main__":
    engineered_data_path = "./data/processed/engineered_dataset.csv"
    model_output_path = "./models/xgb_classifier_readmission_model.pkl"
    metadata_output_path = "./models/model_metadata.json"

    train_model(
        engineered_data_path,
        model_output_path,
        metadata_output_path
    )