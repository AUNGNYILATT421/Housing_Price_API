import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# ─── Feature Definition ───────────────────────────────────────────────────────

CORE_FEATURES = [
    # Numerical
    "GrLivArea", "TotalBsmtSF", "LotArea", "GarageArea", "PoolArea", "LotFrontage",
    "2ndFlrSF", "LowQualFinSF", "BsmtUnfSF", "1stFlrSF",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
    # Counts
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath", "TotRmsAbvGrd", "Fireplaces",
    # Temporal
    "YearBuilt", "YrSold", "YearRemodAdd",
    # Quality / Condition
    "OverallQual", "OverallCond", "HeatingQC", "BsmtQual", "PoolQC",
    "ExterQual", "KitchenQual", "Functional", "FireplaceQu", "BsmtCond", "ExterCond",
    # OneHot Categorical
    "Neighborhood", "MSZoning", "MSSubClass",
    "LandSlope", "Alley", "LandContour", "BldgType",
    "Condition1", "RoofStyle", "Foundation",
    "SaleCondition", "Exterior1st", "Utilities", "Electrical",
    "GarageQual", "GarageCond",
]

OHE_CATEGORICAL_COLS = [
    "Neighborhood", "MSZoning", "LandSlope", "Alley", "LandContour", "BldgType",
    "Condition1", "RoofStyle", "Foundation", "SaleCondition", "Exterior1st",
    "Utilities", "Electrical", "GarageQual", "GarageCond",
]

QUALITY_ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]
FUNCTIONAL_ORDER = ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]
QUALITY_COLS = [
    "FireplaceQu", "BsmtCond", "KitchenQual", "ExterQual",
    "HeatingQC", "BsmtQual", "PoolQC", "ExterCond",
]
FUNCTIONAL_COLS = ["Functional"]

FILL_ZERO_COLS = [
    "PoolArea", "GrLivArea", "LotArea", "TotalBsmtSF", "BsmtUnfSF",
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
    "2ndFlrSF", "LowQualFinSF", "1stFlrSF", "3SsnPorch",
    "EnclosedPorch", "ScreenPorch", "WoodDeckSF", "OpenPorchSF", "GarageArea",
]

SKEWED_FEATURES = [
    "LotArea", "PoolArea", "LowQualFinSF", "BsmtHalfBath", "GrLivArea",
    "LotFrontage", "1stFlrSF", "2ndFlrSF", "BsmtUnfSF",
    "TotalSF", "TotalQualSF", "InteriorQualityScore",
]

# ─── Preprocessing Functions ──────────────────────────────────────────────────

def outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
    idx = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index
    return df.drop(idx, axis=0)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FILL_ZERO_COLS if c in df.columns]
    df[cols] = df[cols].fillna(0)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalQualSF"] = df["TotalSF"] * df["OverallQual"]
    df["TimeSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["InteriorQualityScore"] = df["GrLivArea"] * df["OverallQual"]
    df["TotalBaths"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    porch_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    df["HasPorchDeck"] = (df[porch_cols].sum(axis=1) > 0).astype(int)
    df["TotalPorchDeckSF"] = df[porch_cols].sum(axis=1)
    return df


def log_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in SKEWED_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


# ─── Manual Feature Processor ─────────────────────────────────────────────────

class ManualFeatureProcessor:
    """Fits on training data to learn imputation stats, then transforms any split."""

    def __init__(self):
        self.imputation_values = {}

    def fit(self, X: pd.DataFrame) -> None:
        if "LotFrontage" in X.columns:
            self.imputation_values["LotFrontage"] = X["LotFrontage"].median()
        if "YearBuilt" in X.columns:
            self.imputation_values["YearBuilt_median"] = X["YearBuilt"].median()
        if "YrSold" in X.columns:
            self.imputation_values["YrSold_mode"] = X["YrSold"].mode()[0]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "LotFrontage" in self.imputation_values:
            X["LotFrontage"] = X["LotFrontage"].fillna(self.imputation_values["LotFrontage"])
        if "YearBuilt_median" in self.imputation_values:
            X["YearBuilt"] = X["YearBuilt"].fillna(self.imputation_values["YearBuilt_median"])
            X["YearRemodAdd"] = X["YearRemodAdd"].fillna(X["YearBuilt"])
        if "YrSold_mode" in self.imputation_values:
            X["YrSold"] = X["YrSold"].fillna(self.imputation_values["YrSold_mode"])
        X["Utilities"] = X["Utilities"].fillna("AllPub")
        X = fill_missing(X)
        X = add_engineered_features(X)
        X = log_transform_features(X)
        return X


# ─── sklearn Pipeline Builders ────────────────────────────────────────────────

def build_ohe_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.03,
            sparse_output=False,
            drop="first",
        )),
    ])
    return ColumnTransformer(
        transformers=[("cat", categorical_pipeline, OHE_CATEGORICAL_COLS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_ordinal_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("quality_enc", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                ("ordinal", OrdinalEncoder(
                    categories=[["None"] + QUALITY_ORDER] * len(QUALITY_COLS),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]), QUALITY_COLS),
            ("functional_enc", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                ("ordinal", OrdinalEncoder(
                    categories=[["None"] + FUNCTIONAL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]), FUNCTIONAL_COLS),
        ],
        remainder="passthrough",
    )


def build_feature_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("ohe_proc", build_ohe_preprocessor()),
        ("ordinal_prep", build_ordinal_transformer()),
    ])
