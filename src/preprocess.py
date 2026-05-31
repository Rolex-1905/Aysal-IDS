"""
Aysal IDS - Preprocessing Pipeline
Loads raw CICIDS2017 CSV files, cleans and preprocesses them,
and saves train/test splits ready for model training.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


DATA_PATH   = r"C:\Users\lenovo\Desktop\Projects\Network-IDS\data"
MODELS_PATH = r"C:\Users\lenovo\Desktop\Projects\Network-IDS\models"

DROP_COLS = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']


def load_and_sample(data_path, benign_frac=0.3):
    csv_files = [
        f for f in os.listdir(data_path)
        if f.endswith('.csv')
        and not f.startswith('X_')
        and not f.startswith('y_')
    ]

    dfs = []
    for file in csv_files:
        path = os.path.join(data_path, file)
        print(f"Loading {file}...")
        try:
            df_temp = pd.read_csv(path, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            df_temp = pd.read_csv(path, low_memory=False, encoding='latin-1')

        df_temp.columns = df_temp.columns.str.strip()
        df_temp.drop(columns=[c for c in DROP_COLS if c in df_temp.columns], inplace=True)

        non_numeric = [
            c for c in df_temp.select_dtypes(exclude=[np.number]).columns
            if c != 'Label'
        ]
        df_temp.drop(columns=non_numeric, inplace=True)

        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
        df_temp[numeric_cols] = df_temp[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_temp[numeric_cols] = df_temp[numeric_cols].fillna(df_temp[numeric_cols].median())

        benign  = df_temp[df_temp['Label'] == 'BENIGN'].sample(frac=benign_frac, random_state=42)
        attacks = df_temp[df_temp['Label'] != 'BENIGN']
        df_temp = pd.concat([benign, attacks], ignore_index=True)

        print(f"  -> {df_temp.shape} | Attacks: {(df_temp['Label'] != 'BENIGN').sum()}")
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape: {df.shape}")
    return df


def encode_labels(df):
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    le = LabelEncoder()
    df['Label_Multi'] = le.fit_transform(df['Label'])
    return df, le


def split_and_scale(df):
    X = df.drop(columns=['Label', 'Label_Binary', 'Label_Multi'])
    X = X.select_dtypes(include=[np.number])
    y = df['Label_Binary']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def save_artifacts(X_train, X_test, y_train, y_test, scaler, le):
    X_train.to_csv(os.path.join(DATA_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_PATH,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(DATA_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_PATH,  "y_test.csv"),  index=False)

    with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("Artifacts saved.")


if __name__ == "__main__":
    df = load_and_sample(DATA_PATH)
    df, le = encode_labels(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, le)
    print("Preprocessing complete.")