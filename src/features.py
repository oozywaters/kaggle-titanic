import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Countess": "Mrs", "Lady": "Mrs", "Dona": "Mrs",
    "Capt": "Other", "Col": "Other", "Don": "Other", "Major": "Other",
    "Rev": "Other", "Sir": "Other", "Jonkheer": "Other",
}
AGE_BINS = [0, 16, 32, 48, 64, np.inf]
FARE_BINS = [-np.inf, 7.91, 14.45, 31.0, np.inf]

def add_title(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        Title=df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        .map(TITLE_MAP)
        .fillna("Other")
    )

def add_age_features(df: pd.DataFrame, title_age: pd.Series) -> pd.DataFrame:
    age = df["Age"].fillna(df["Title"].map(title_age))
    return df.assign(
        Age_missing=df["Age"].isna().astype(int),
        Age=age,
        AgeBin=pd.cut(age, bins=AGE_BINS, labels=False, include_lowest=True),
    )

def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    family_size = df["SibSp"] + df["Parch"] + 1
    return df.assign(
        FamilySize=family_size,
        IsAlone=(family_size == 1).astype(int),
    )

def add_fare_bin(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        FareBin=pd.cut(df["Fare"], bins=FARE_BINS, labels=False)
    )

class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Title, Age_missing flag, title-median Age fill, AgeBin, family features.

    Age medians are learned in fit (training folds only), so CV never sees
    validation data. Decisions documented in notebooks/eda.ipynb.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.title_age_ = add_title(X).groupby("Title")["Age"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_fare_bin(add_family_features(add_age_features(add_title(X), self.title_age_)))