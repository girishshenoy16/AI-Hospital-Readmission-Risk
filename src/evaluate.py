import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def load_engineered_data(path: str):
    """Load engineered dataset"""
    df = pd.read_csv(path)
    X = df.drop(columns=['readmitted_flag'])
    y = df['readmitted_flag']
    return X, y


def evaluate_model(data_path: str, model_path: str):
    """Evaluate trained readmission prediction model"""

    # Load data
    X, y = load_engineered_data(data_path)

    # Train-test split (same strategy as training)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Load trained model
    model = joblib.load(model_path)

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.25).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n📊 MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"ROC-AUC Score: {auc:.4f}\n")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)


if __name__ == "__main__":
    engineered_data_path = "./data/processed/engineered_dataset.csv"
    model_path = "./models/xgb_classifier_readmission_model.pkl"

    evaluate_model(engineered_data_path, model_path)