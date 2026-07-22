from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd


class HouseFeatures(BaseModel):
    """Input features for house price prediction.
    Fields starting with digits use snake_case aliases."""

    model_config = ConfigDict(populate_by_name=True)

    # Numerical area features
    GrLivArea: float = Field(default=0.0, description="Above grade living area (sq ft)")
    TotalBsmtSF: float = Field(default=0.0, description="Total basement area (sq ft)")
    LotArea: float = Field(default=0.0, description="Lot size (sq ft)")
    GarageArea: float = Field(default=0.0, description="Garage size (sq ft)")
    PoolArea: float = Field(default=0.0, description="Pool area (sq ft)")
    LotFrontage: Optional[float] = Field(default=None, description="Street frontage (ft)")
    floor_2nd_sf: float = Field(default=0.0, alias="2ndFlrSF", description="2nd floor sq ft")
    LowQualFinSF: float = Field(default=0.0, description="Low quality finished area (sq ft)")
    BsmtUnfSF: float = Field(default=0.0, description="Unfinished basement area (sq ft)")
    floor_1st_sf: float = Field(default=0.0, alias="1stFlrSF", description="1st floor sq ft")
    WoodDeckSF: float = Field(default=0.0, description="Wood deck area (sq ft)")
    OpenPorchSF: float = Field(default=0.0, description="Open porch area (sq ft)")
    EnclosedPorch: float = Field(default=0.0, description="Enclosed porch area (sq ft)")
    porch_3ssn_sf: float = Field(default=0.0, alias="3SsnPorch", description="3-season porch (sq ft)")
    ScreenPorch: float = Field(default=0.0, description="Screen porch area (sq ft)")

    # Counts
    FullBath: int = Field(default=0, description="Full bathrooms above grade")
    HalfBath: int = Field(default=0, description="Half baths above grade")
    BsmtFullBath: int = Field(default=0, description="Basement full bathrooms")
    BsmtHalfBath: int = Field(default=0, description="Basement half bathrooms")
    TotRmsAbvGrd: int = Field(default=6, description="Total rooms above grade")
    Fireplaces: int = Field(default=0, description="Number of fireplaces")

    # Temporal
    YearBuilt: int = Field(default=2000, description="Original construction year")
    YrSold: int = Field(default=2010, description="Year sold")
    YearRemodAdd: int = Field(default=2000, description="Remodel date")

    # Quality / Condition (ordinal)
    OverallQual: int = Field(default=5, ge=1, le=10, description="Overall material quality (1-10)")
    OverallCond: int = Field(default=5, ge=1, le=9, description="Overall condition (1-9)")
    HeatingQC: Optional[str] = Field(default="TA", description="Heating quality: Ex/Gd/TA/Fa/Po")
    BsmtQual: Optional[str] = Field(default="TA", description="Basement height quality")
    PoolQC: Optional[str] = Field(default=None, description="Pool quality")
    ExterQual: Optional[str] = Field(default="TA", description="Exterior material quality")
    KitchenQual: Optional[str] = Field(default="TA", description="Kitchen quality")
    Functional: Optional[str] = Field(default="Typ", description="Home functionality")
    FireplaceQu: Optional[str] = Field(default=None, description="Fireplace quality")
    BsmtCond: Optional[str] = Field(default="TA", description="Basement condition")
    ExterCond: Optional[str] = Field(default="TA", description="Exterior condition")

    # Categorical (one-hot encoded)
    Neighborhood: Optional[str] = Field(default="NAmes", description="Physical location")
    MSZoning: Optional[str] = Field(default="RL", description="General zoning classification")
    MSSubClass: Optional[int] = Field(default=20, description="Building class")
    LandSlope: Optional[str] = Field(default="Gtl", description="Slope of property")
    Alley: Optional[str] = Field(default=None, description="Alley access type")
    LandContour: Optional[str] = Field(default="Lvl", description="Flatness of property")
    BldgType: Optional[str] = Field(default="1Fam", description="Type of dwelling")
    Condition1: Optional[str] = Field(default="Norm", description="Proximity to main road/railroad")
    RoofStyle: Optional[str] = Field(default="Gable", description="Type of roof")
    Foundation: Optional[str] = Field(default="PConc", description="Type of foundation")
    SaleCondition: Optional[str] = Field(default="Normal", description="Condition of sale")
    Exterior1st: Optional[str] = Field(default="VinylSd", description="Exterior covering on house")
    Utilities: Optional[str] = Field(default="AllPub", description="Type of utilities available")
    Electrical: Optional[str] = Field(default="SBrkr", description="Electrical system")
    GarageQual: Optional[str] = Field(default="TA", description="Garage quality")
    GarageCond: Optional[str] = Field(default="TA", description="Garage condition")

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        row = {
            "GrLivArea": self.GrLivArea,
            "TotalBsmtSF": self.TotalBsmtSF,
            "LotArea": self.LotArea,
            "GarageArea": self.GarageArea,
            "PoolArea": self.PoolArea,
            "LotFrontage": self.LotFrontage,
            "2ndFlrSF": self.floor_2nd_sf,
            "LowQualFinSF": self.LowQualFinSF,
            "BsmtUnfSF": self.BsmtUnfSF,
            "1stFlrSF": self.floor_1st_sf,
            "WoodDeckSF": self.WoodDeckSF,
            "OpenPorchSF": self.OpenPorchSF,
            "EnclosedPorch": self.EnclosedPorch,
            "3SsnPorch": self.porch_3ssn_sf,
            "ScreenPorch": self.ScreenPorch,
            "FullBath": self.FullBath,
            "HalfBath": self.HalfBath,
            "BsmtFullBath": self.BsmtFullBath,
            "BsmtHalfBath": self.BsmtHalfBath,
            "TotRmsAbvGrd": self.TotRmsAbvGrd,
            "Fireplaces": self.Fireplaces,
            "YearBuilt": self.YearBuilt,
            "YrSold": self.YrSold,
            "YearRemodAdd": self.YearRemodAdd,
            "OverallQual": self.OverallQual,
            "OverallCond": self.OverallCond,
            "HeatingQC": self.HeatingQC,
            "BsmtQual": self.BsmtQual,
            "PoolQC": self.PoolQC,
            "ExterQual": self.ExterQual,
            "KitchenQual": self.KitchenQual,
            "Functional": self.Functional,
            "FireplaceQu": self.FireplaceQu,
            "BsmtCond": self.BsmtCond,
            "ExterCond": self.ExterCond,
            "Neighborhood": self.Neighborhood,
            "MSZoning": self.MSZoning,
            "MSSubClass": self.MSSubClass,
            "LandSlope": self.LandSlope,
            "Alley": self.Alley,
            "LandContour": self.LandContour,
            "BldgType": self.BldgType,
            "Condition1": self.Condition1,
            "RoofStyle": self.RoofStyle,
            "Foundation": self.Foundation,
            "SaleCondition": self.SaleCondition,
            "Exterior1st": self.Exterior1st,
            "Utilities": self.Utilities,
            "Electrical": self.Electrical,
            "GarageQual": self.GarageQual,
            "GarageCond": self.GarageCond,
        }
        return pd.DataFrame([row])


class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_price: float
    predicted_price_formatted: str


class PredictionRecord(BaseModel):
    id: int
    predicted_price: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
