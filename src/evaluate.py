"""
Aysal IDS - Evaluation Script
Loads trained models, runs evaluation on test set,
prints metrics and saves charts to reports/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)


DATA_PATH    = r"C:\Users\lenovo\Desktop\Projects\Network-IDS\data"
MODELS_PATH  = r"C:\Users\lenovo\Desktop\Projects\Network-IDS\models"
REPORTS_PATH = r"C:\Users\lenovo\Desktop\Projects\Network-IDS\reports"
os.makedirs(REPORTS_PATH, exist_ok=True)


def load_artifacts():
    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).squeeze()

    with open(os.path.join(MODELS_PATH, "random_forest.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(MODELS_PATH, "logistic_regression.pkl"), "rb") as f:
        lr_model = pickle.load(f)

    return X_test, y_test, rf_model, lr_model


def print_metrics(y_test, preds, proba, name):
    print(f"\n  {name}")
    print(f"    Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"    Precision : {precision_score(y_test, preds):.4f}")
    print(f"    Recall    : {recall_score(y_test, preds):.4f}")
    print(f"    F1-Score  : {f1_score(y_test, preds):.4f}")
    print(f"    ROC-AUC   : {roc_auc_score(y_test, proba):.4f}")


def plot_confusion_matrices(y_test, rf_preds, lr_preds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, preds, title in zip(
        axes,
        [rf_preds, lr_preds],
        ["Random Forest", "Logistic Regression"]
    ):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["BENIGN", "ATTACK"],
                    yticklabels=["BENIGN", "ATTACK"])
        ax.set_title(f"Aysal IDS — {title}", fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrices.png")


def plot_roc_curve(y_test, rf_proba, lr_proba):
    fig, ax = plt.subplots(figsize=(8, 6))
    for proba, label, color in zip(
        [rf_proba, lr_proba],
        ["Random Forest", "Logistic Regression"],
        ["#2ecc71", "#e74c3c"]
    ):
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Aysal IDS — ROC Curve", fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, "roc_curve.png"), dpi=150)
    plt.close()
    print("Saved: roc_curve.png")


if __name__ == "__main__":
    X_test, y_test, rf_model, lr_model = load_artifacts()

    rf_preds = rf_model.predict(X_test)
    lr_preds = lr_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("    AYSAL IDS — EVALUATION RESULTS")
    print("=" * 50)
    print_metrics(y_test, rf_preds, rf_proba, "Random Forest")
    print_metrics(y_test, lr_preds, lr_proba, "Logistic Regression")
    print("\n" + "=" * 50)

    plot_confusion_matrices(y_test, rf_preds, lr_preds)
    plot_roc_curve(y_test, rf_proba, lr_proba)
    print("\nEvaluation complete.")