# scripts/train_model.py
"""
Train AI model to predict orbital collision risk.
Supports: xgboost, lightgbm, random_forest
Saves model to models/
"""
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data"
BALANCED_FILE = os.path.join(DATA_PATH, "training_balanced.parquet")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(BALANCED_FILE):
        raise FileNotFoundError(f"{BALANCED_FILE} missing. Run merge + balance steps first.")
    df = pd.read_parquet(BALANCED_FILE)
    # drop any non-feature columns if present
    if "obj_a" in df.columns: df = df.drop(columns=["obj_a","obj_b"], errors="ignore")
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def build_model(kind="xgboost"):
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="binary:logistic",
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            return model
        except Exception as e:
            print("XGBoost not available:", e)
            print("Falling back to RandomForest.")
            kind = "rf"

    if kind == "lightgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
            )
            return model
        except Exception as e:
            print("LightGBM not available:", e)
            print("Falling back to RandomForest.")
            kind = "rf"

    # default / fallback
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    return model

def train_and_save(kind="xgboost", test_size=0.25):
    X, y = load_data()
    print(f"Loaded balanced dataset: {len(X)} rows. Labels: {y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    model = build_model(kind=kind)

    print(f"Training {kind} model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # use decision_function as fallback
        y_pred_proba = model.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-12)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    # PR plot
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f"AUC={auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    plt.legend()
    pr_path = os.path.join(MODEL_DIR, f"pr_curve_{kind}.png")
    plt.savefig(pr_path)
    plt.close()

    # classification report at threshold 0.5
    y_class = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_class, digits=3))
    cm = confusion_matrix(y_test, y_class)

    # save confusion matrix plot
    plt.figure(figsize=(4,3))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    cm_path = os.path.join(MODEL_DIR, f"confusion_matrix_{kind}.png")
    plt.savefig(cm_path)
    plt.close()

    # Feature importance (if available)
    fi_path = os.path.join(MODEL_DIR, f"feature_importance_{kind}.png")
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            idx = np.argsort(fi)[::-1][:30]
            names = X.columns[idx]
            vals = fi[idx]
            plt.figure(figsize=(8,6))
            plt.barh(np.arange(len(vals))[::-1], vals[::-1])
            plt.yticks(np.arange(len(vals))[::-1], names[::-1])
            plt.title("Feature importance")
            plt.tight_layout()
            plt.savefig(fi_path)
            plt.close()
    except Exception as e:
        print("Failed to generate feature importances:", e)

    # Save model
    model_file = os.path.join(MODEL_DIR, f"collision_predictor_{kind}.pkl")
    joblib.dump(model, model_file)
    print(f"Saved model to {model_file}")
    return model_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost","lightgbm","rf"])
    args = parser.parse_args()
    train_and_save(kind=args.model)
