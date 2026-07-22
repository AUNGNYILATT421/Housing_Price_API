import os
import numpy as np
import joblib
import pandas as pd

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")

_manual_processor = None
_stack = None
_core_features = None


def load_artifacts():
    global _manual_processor, _stack, _core_features
    _manual_processor = joblib.load(os.path.join(ARTIFACTS_DIR, "manual_processor.joblib"))
    _stack            = joblib.load(os.path.join(ARTIFACTS_DIR, "stack.joblib"))
    _core_features    = joblib.load(os.path.join(ARTIFACTS_DIR, "core_features.joblib"))
    print("Model artifacts loaded.")


def predict(input_df: pd.DataFrame) -> float:
    if _stack is None:
        raise RuntimeError("Model not loaded. Call load_artifacts() first.")

    X = input_df[_core_features].copy()
    X_processed = _manual_processor.transform(X)

    log_pred = _stack.predict(X_processed)
    return float(np.expm1(log_pred)[0])
