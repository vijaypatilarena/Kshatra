# scripts/export_model.py
"""
Export trained model to multiple formats:
 - joblib .pkl (default)
 - ONNX (.onnx) if skl2onnx available
 - XGBoost native JSON if XGBoost model
"""
import os
import sys
import joblib
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def export_to_onnx(model, X_sample, out_path):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        print("skl2onnx not available:", e)
        return False
    initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return True

def export(model_file, dataset_file=None):
    model = joblib.load(model_file)
    base = os.path.splitext(os.path.basename(model_file))[0]
    # joblib copy (already present)
    print(f"Loaded model from {model_file}")

    # Try ONNX
    if dataset_file:
        import pandas as pd
        X = pd.read_parquet(dataset_file).drop(columns=["label"])
        X_sample = X.iloc[:4].astype(float).values
    else:
        X_sample = None

    if X_sample is not None:
        onnx_path = os.path.join(MODEL_DIR, base + ".onnx")
        ok = export_to_onnx(model, X_sample, onnx_path)
        if ok:
            print("Exported ONNX to", onnx_path)
        else:
            print("ONNX export not created (missing libs or failed).")

    # If XGBoost model, save native booster JSON if possible
    try:
        import xgboost
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            json_path = os.path.join(MODEL_DIR, base + ".json")
            booster.save_model(json_path)
            print("Saved XGBoost native model to", json_path)
    except Exception:
        pass

    print("Export step finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--dataset", default=None, help="parquet used to get sample X shape for ONNX conversion")
    args = parser.parse_args()
    export(args.model_file, args.dataset)
