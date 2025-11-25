# scripts/evaluate_model.py
"""
Load a trained model and evaluate on the balanced dataset (or any dataset).
Saves ROC + PR + Confusion matrix PNG files in models/ and prints classification report.
"""
import os
import sys
import joblib
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "data"
BALANCED_FILE = os.path.join(DATA_PATH, "training_balanced.parquet")
MODEL_DIR = "models"

def load_dataset(path=BALANCED_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def evaluate(model_path, dataset_path=None):
    model = joblib.load(model_path)
    X, y = load_dataset(dataset_path) if dataset_path else load_dataset()
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)

    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y, y_score)
    pr_auc = auc(recall, precision)

    # ROC
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    roc_file = os.path.join(MODEL_DIR, "roc_curve.png")
    plt.savefig(roc_file)
    plt.close()

    # PR
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.grid(True)
    pr_file = os.path.join(MODEL_DIR, "pr_curve.png")
    plt.savefig(pr_file)
    plt.close()

    # confusion at 0.5
    y_pred = (y_score >= 0.5).astype(int)
    print("Classification report:\n", classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(4,3))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    cm_file = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_file)
    plt.close()

    print(f"Saved ROC -> {roc_file}, PR -> {pr_file}, CM -> {cm_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", required=True, help="path to model .pkl produced by train_model")
    parser.add_argument("--dataset", default=None, help="optional dataset to evaluate (parquet)")
    args = parser.parse_args()
    evaluate(args.model_file, args.dataset)
