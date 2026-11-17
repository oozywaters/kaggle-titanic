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
FARE_BINS = [-np.inf, 7.9104, 14.4542, 31.0, np.inf]

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
    """Title, Age fill, bins, family features, group survival.

    All statistics are learned in fit, so CV folds never leak. FamilySurvival
    (1 = another ticket-group member survived, 0 = all known others died,
    0.5 = unknown) uses the target, so training rows are encoded leave-one-out
    by PassengerId. Decisions documented in notebooks/eda.ipynb.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.title_age_ = add_title(X).groupby("Title")["Age"].median()

        self.group_table_ = None
        self.loo_cache_ = {}
        if y is not None:
            labels = np.asarray(y)
            groups = pd.DataFrame(
                {"key": X["Ticket"].values, "pid": X["PassengerId"].values, "y": labels}
            )
            self.group_table_ = groups.groupby("key")["y"].agg(["sum", "count"])
            for row in groups.itertuples():
                total, size = self.group_table_.loc[row.key]
                others_sum, others_count = total - row.y, size - 1
                self.loo_cache_[row.pid] = (
                    0.5 if others_count == 0 else float(others_sum > 0)
                )
        return self

    def _family_survival(self, X: pd.DataFrame) -> pd.Series:
        if self.group_table_ is None:
            return pd.Series(0.5, index=X.index)
        cached = X["PassengerId"].map(self.loo_cache_)
        any_survived = X["Ticket"].map(self.group_table_["sum"]).gt(0).astype(float)
        known_group = X["Ticket"].isin(self.group_table_.index)
        fallback = any_survived.where(known_group, 0.5)
        return cached.fillna(fallback)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = add_fare_bin(add_family_features(add_age_features(add_title(X), self.title_age_)))
        return df.assign(FamilySurvival=self._family_survival(X))