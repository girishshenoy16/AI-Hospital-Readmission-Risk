import pandas as pd
import numpy as np


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw healthcare dataset"""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess healthcare data"""

    # Replace placeholder missing values
    df.replace('?', np.nan, inplace=True)

    # Create binary target variable
    df['readmitted_flag'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )

    # Drop leakage & high-missing columns
    drop_cols = [
        'readmitted',
        'encounter_id',
        'patient_nbr',
        'weight',
        'payer_code',
        'medical_specialty'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fill numeric missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical missing values
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    return df


def save_clean_data(df: pd.DataFrame, path: str):
    """Save cleaned dataset"""
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw_path = "./data/raw/diabetic_data.csv"
    processed_path = "./data/processed/cleaned_dataset.csv"

    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean, processed_path)

    print("✅ Cleaned dataset saved successfully.")