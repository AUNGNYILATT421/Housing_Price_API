import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "ml", "artifacts")

@st.cache_resource
def load_model():
    processor = joblib.load(os.path.join(ARTIFACTS_DIR, "manual_processor.joblib"))
    stack     = joblib.load(os.path.join(ARTIFACTS_DIR, "stack.joblib"))
    features  = joblib.load(os.path.join(ARTIFACTS_DIR, "core_features.joblib"))
    return processor, stack, features


def predict(processor, stack, core_features, inputs: dict) -> float:
    df = pd.DataFrame([inputs])
    X = df[core_features].copy()
    X_processed = processor.transform(X)
    log_pred = stack.predict(X_processed)
    return float(np.expm1(log_pred)[0])


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Price Predictor")
st.caption("Stacking ensemble: GradientBoosting + CatBoost + KernelRidge · Top 16 / 4,000+ on Kaggle")
st.divider()

# ── Load model ────────────────────────────────────────────────────────────────
try:
    processor, stack, core_features = load_model()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.info("Run `python -m ml.export_model` first to generate the artifacts.")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("House Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Size**")
    GrLivArea    = st.number_input("Above-grade living area (sq ft)", 300, 6000, 1500, step=50)
    TotalBsmtSF  = st.number_input("Total basement area (sq ft)", 0, 3000, 800, step=50)
    GarageArea   = st.number_input("Garage area (sq ft)", 0, 1500, 400, step=50)
    LotArea      = st.number_input("Lot area (sq ft)", 1000, 100000, 8000, step=500)
    LotFrontage  = st.number_input("Lot frontage (ft)", 0, 200, 65, step=5)

    st.markdown("**Floors & Extras**")
    floor_1st    = st.number_input("1st floor area (sq ft)", 0, 4000, 856, step=50)
    floor_2nd    = st.number_input("2nd floor area (sq ft)", 0, 2000, 0, step=50)
    WoodDeckSF   = st.number_input("Wood deck (sq ft)", 0, 800, 0, step=10)
    OpenPorchSF  = st.number_input("Open porch (sq ft)", 0, 500, 0, step=10)

with col2:
    st.markdown("**Quality & Condition**")
    OverallQual  = st.slider("Overall quality (1–10)", 1, 10, 6)
    OverallCond  = st.slider("Overall condition (1–9)", 1, 9, 5)
    KitchenQual  = st.selectbox("Kitchen quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
    ExterQual    = st.selectbox("Exterior quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
    BsmtQual     = st.selectbox("Basement quality", ["Ex", "Gd", "TA", "Fa", "Po", None], index=2)
    HeatingQC    = st.selectbox("Heating quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
    FireplaceQu  = st.selectbox("Fireplace quality", [None, "Ex", "Gd", "TA", "Fa", "Po"], index=0)

    st.markdown("**Bathrooms & Rooms**")
    FullBath     = st.number_input("Full baths (above grade)", 0, 4, 2)
    HalfBath     = st.number_input("Half baths (above grade)", 0, 2, 0)
    BsmtFullBath = st.number_input("Basement full baths", 0, 2, 0)
    TotRmsAbvGrd = st.number_input("Total rooms above grade", 2, 14, 7)
    Fireplaces   = st.number_input("Fireplaces", 0, 4, 0)

with col3:
    st.markdown("**Year & Location**")
    YearBuilt    = st.number_input("Year built", 1870, 2024, 2000)
    YearRemodAdd = st.number_input("Year remodelled", 1950, 2024, 2000)
    YrSold       = st.number_input("Year sold", 2006, 2010, 2010)
    Neighborhood = st.selectbox("Neighborhood", [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt",
        "Gilbert", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor",
        "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr",
        "SWISU", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste",
    ])
    MSZoning     = st.selectbox("Zoning", ["RL", "RM", "FV", "RH", "C (all)"], index=0)
    BldgType     = st.selectbox("Building type", ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"], index=0)
    Foundation   = st.selectbox("Foundation", ["PConc", "CBlock", "BrkTil", "Wood", "Slab", "Stone"], index=0)

    st.markdown("**Garage & Utilities**")
    GarageQual   = st.selectbox("Garage quality", ["TA", "Gd", "Fa", "Ex", "Po"], index=0)
    GarageCond   = st.selectbox("Garage condition", ["TA", "Gd", "Fa", "Ex", "Po"], index=0)
    Electrical   = st.selectbox("Electrical system", ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"], index=0)

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    inputs = {
        "GrLivArea": float(GrLivArea),
        "TotalBsmtSF": float(TotalBsmtSF),
        "LotArea": float(LotArea),
        "GarageArea": float(GarageArea),
        "PoolArea": 0.0,
        "LotFrontage": float(LotFrontage) if LotFrontage else None,
        "2ndFlrSF": float(floor_2nd),
        "LowQualFinSF": 0.0,
        "BsmtUnfSF": 0.0,
        "1stFlrSF": float(floor_1st),
        "WoodDeckSF": float(WoodDeckSF),
        "OpenPorchSF": float(OpenPorchSF),
        "EnclosedPorch": 0.0,
        "3SsnPorch": 0.0,
        "ScreenPorch": 0.0,
        "FullBath": int(FullBath),
        "HalfBath": int(HalfBath),
        "BsmtFullBath": int(BsmtFullBath),
        "BsmtHalfBath": 0,
        "TotRmsAbvGrd": int(TotRmsAbvGrd),
        "Fireplaces": int(Fireplaces),
        "YearBuilt": int(YearBuilt),
        "YrSold": int(YrSold),
        "YearRemodAdd": int(YearRemodAdd),
        "OverallQual": int(OverallQual),
        "OverallCond": int(OverallCond),
        "HeatingQC": HeatingQC,
        "BsmtQual": BsmtQual,
        "PoolQC": None,
        "ExterQual": ExterQual,
        "KitchenQual": KitchenQual,
        "Functional": "Typ",
        "FireplaceQu": FireplaceQu,
        "BsmtCond": "TA",
        "ExterCond": "TA",
        "Neighborhood": Neighborhood,
        "MSZoning": MSZoning,
        "MSSubClass": 20,
        "LandSlope": "Gtl",
        "Alley": None,
        "LandContour": "Lvl",
        "BldgType": BldgType,
        "Condition1": "Norm",
        "RoofStyle": "Gable",
        "Foundation": Foundation,
        "SaleCondition": "Normal",
        "Exterior1st": "VinylSd",
        "Utilities": "AllPub",
        "Electrical": Electrical,
        "GarageQual": GarageQual,
        "GarageCond": GarageCond,
    }

    with st.spinner("Predicting..."):
        try:
            price = predict(processor, stack, core_features, inputs)
            st.success(f"### Estimated Sale Price: **${price:,.0f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")
