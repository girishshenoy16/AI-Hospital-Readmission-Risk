import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder


IMPORTANT_CATEGORICAL_COLS = [
    'age',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id'
]


def sanitize_column_names(columns):
    """
    Remove characters not allowed by XGBoost
    """
    clean_cols = []
    for col in columns:
        col = re.sub(r'[\[\]<>]', '', col)
        col = col.replace('(', '').replace(')', '')
        col = col.replace('-', '_')
        clean_cols.append(col)
    return clean_cols


def load_clean_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    y = df['readmitted_flag']
    X = df.drop(columns=['readmitted_flag'])

    # Keep only selected categorical columns
    cat_cols = [c for c in IMPORTANT_CATEGORICAL_COLS if c in X.columns]

    # Drop all other object columns
    object_cols = X.select_dtypes(include=['object']).columns
    drop_cols = [c for c in object_cols if c not in cat_cols]
    X = X.drop(columns=drop_cols)

    num_cols = X.select_dtypes(exclude=['object']).columns

    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    encoded_cat = encoder.fit_transform(X[cat_cols])

    encoded_cat_cols = encoder.get_feature_names_out(cat_cols)
    encoded_cat_cols = sanitize_column_names(encoded_cat_cols)

    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoded_cat_cols)

    numeric_df = X[num_cols].reset_index(drop=True)
    encoded_cat_df.reset_index(drop=True, inplace=True)

    engineered_df = pd.concat([numeric_df, encoded_cat_df], axis=1)
    engineered_df['readmitted_flag'] = y.values

    return engineered_df


def save_engineered_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    clean_path = "./data/processed/cleaned_dataset.csv"
    engineered_path = "./data/processed/engineered_dataset.csv"

    df_clean = load_clean_data(clean_path)
    df_engineered = engineer_features(df_clean)
    save_engineered_data(df_engineered, engineered_path)

    print("✅ Engineered dataset saved successfully with sanitized feature names.")