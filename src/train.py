"""
Aysal IDS - Model Training Script
Trains Random Forest and Logistic Regression classifiers
on preprocessed CICIDS2017 data and saves models to disk.
"""

import pandas as pd
import pickle
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


DATA_PATH   = r"C:\Users\lenovo\Desktop\Projects\Network-IDS\data"
MODELS_PATH = r"C:\Users\lenovo\Desktop\Projects\Network-IDS\models"


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).squeeze()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"Done in {time.time() - start:.1f}s")
    return model


def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    start = time.time()
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"Done in {time.time() - start:.1f}s")
    return model


def save_model(model, filename):
    path = os.path.join(MODELS_PATH, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(rf_model, "random_forest.pkl")
    save_model(lr_model, "logistic_regression.pkl")
    print("\nTraining complete.")