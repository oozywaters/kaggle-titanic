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

class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Title, Age_missing flag, title-median Age fill, AgeBin.

    Age medians are learned in fit (training folds only), so CV never sees
    validation data. Decisions documented in notebooks/eda.ipynb.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.title_age_ = add_title(X).groupby("Title")["Age"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_age_features(add_title(X), self.title_age_)