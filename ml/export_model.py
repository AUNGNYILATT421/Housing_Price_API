"""
Run this script once to train the ensemble and save model artifacts.

Usage:
    1. Download train.csv from Kaggle Housing Prices competition
    2. Place it in the data/ folder
    3. Run: python -m ml.export_model
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from catboost import CatBoostRegressor

from ml.preprocessing import (
    CORE_FEATURES,
    ManualFeatureProcessor,
    build_feature_pipeline,
    outlier_removal,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")


def evaluate(preds_log, y_log, name):
    rmsle = np.sqrt(mean_squared_error(preds_log, y_log))
    mae = mean_absolute_error(np.expm1(preds_log), np.expm1(y_log))
    print(f"  {name:40s} RMSLE={rmsle:.4f}  MAE=${mae:,.0f}")


def train():
    print("Loading data...")
    home_data = pd.read_csv(DATA_PATH)
    home_data = outlier_removal(home_data)

    y = np.log1p(home_data["SalePrice"])
    X = home_data[CORE_FEATURES].copy()

    # ── 80/20 split for quick validation ──────────────────────────────────────
    train_X_raw, val_X_raw, train_y, val_y = train_test_split(X, y, random_state=1)

    manual_proc_split = ManualFeatureProcessor()
    manual_proc_split.fit(train_X_raw)
    train_X = manual_proc_split.transform(train_X_raw)
    val_X = manual_proc_split.transform(val_X_raw)

    feature_pipeline = build_feature_pipeline()

    # ── Model 1: Gradient Boosting ─────────────────────────────────────────────
    print("\nTraining GBR...")
    gbr_pipeline = Pipeline([
        ("features", feature_pipeline),
        ("regressor", GradientBoostingRegressor(random_state=1, verbose=0)),
    ])
    gbr_grid = GridSearchCV(
        gbr_pipeline,
        param_grid={
            "regressor__n_estimators": [5000],
            "regressor__learning_rate": [0.03],
            "regressor__min_samples_leaf": [1],
            "regressor__min_samples_split": [2],
            "regressor__max_depth": [3],
            "regressor__subsample": [0.83],
            "regressor__max_features": ["sqrt"],
            "regressor__loss": ["huber"],
        },
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0,
    )
    gbr_grid.fit(train_X, train_y)
    gbr_best = gbr_grid.best_estimator_
    evaluate(gbr_best.predict(val_X), val_y, "GBR")

    # ── Model 2: CatBoost ──────────────────────────────────────────────────────
    print("Training CatBoost...")
    cat_pipeline = Pipeline([
        ("features", build_feature_pipeline()),
        ("regressor", CatBoostRegressor(random_state=1, verbose=0, loss_function="RMSE")),
    ])
    cat_grid = GridSearchCV(
        cat_pipeline,
        param_grid={
            "regressor__n_estimators": [1900],
            "regressor__learning_rate": [0.02],
            "regressor__depth": [6],
            "regressor__l2_leaf_reg": [3],
            "regressor__subsample": [0.7],
            "regressor__colsample_bylevel": [0.7],
        },
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0,
    )
    cat_grid.fit(train_X, train_y)
    cat_best = cat_grid.best_estimator_
    evaluate(cat_best.predict(val_X), val_y, "CatBoost")

    # ── Model 3: Kernel Ridge ──────────────────────────────────────────────────
    print("Training KernelRidge...")
    krr_pipeline = Pipeline([
        ("features", build_feature_pipeline()),
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf")),
    ])
    krr_grid = GridSearchCV(
        krr_pipeline,
        param_grid={"krr__alpha": [0.0003], "krr__gamma": [0.0001]},
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0,
    )
    krr_grid.fit(train_X, train_y)
    krr_best = krr_grid.best_estimator_
    evaluate(krr_best.predict(val_X), val_y, "KernelRidge")

    # ── Stacking Meta-Model ────────────────────────────────────────────────────
    print("Training Stacking ensemble...")
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    estimators = [("gbr", gbr_best), ("cat", cat_best), ("krr", krr_best)]

    sgd_meta = SGDRegressor(
        loss="epsilon_insensitive", epsilon=0.0, penalty="l2",
        learning_rate="invscaling", eta0=0.01, power_t=0.24,
        alpha=0.001, random_state=1,
    )
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=make_pipeline(StandardScaler(), sgd_meta),
        cv=kf, n_jobs=-1, passthrough=False, verbose=0,
    )
    stack.fit(train_X, train_y)
    evaluate(stack.predict(val_X), val_y, "Stacking (GBR + CAT + KRR)")

    # ── Retrain on full data ───────────────────────────────────────────────────
    print("\nRetraining on full dataset...")
    manual_proc_full = ManualFeatureProcessor()
    manual_proc_full.fit(X)
    X_full = manual_proc_full.transform(X)

    for name, model in estimators:
        print(f"  Fitting {name}...")
        model.fit(X_full, y)

    stack.fit(X_full, y)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(manual_proc_full, os.path.join(ARTIFACTS_DIR, "manual_processor.joblib"))
    joblib.dump(stack,            os.path.join(ARTIFACTS_DIR, "stack.joblib"))
    joblib.dump(CORE_FEATURES,    os.path.join(ARTIFACTS_DIR, "core_features.joblib"))

    print("\nArtifacts saved to ml/artifacts/")
    print("  manual_processor.joblib")
    print("  stack.joblib")
    print("  core_features.joblib")


if __name__ == "__main__":
    train()
