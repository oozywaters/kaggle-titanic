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

def add_title(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        Title=df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        .map(TITLE_MAP)
        .fillna("Other")
    )

class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Title, Age fill by title, ticket-group survival.

    All statistics are learned in fit, so CV folds never leak. FamilySurvival
    (1 = another ticket-group member survived, 0 = all known others died,
    0.5 = unknown) uses the target, so training rows are encoded leave-one-out
    by PassengerId. Decisions documented in notebooks/eda.ipynb.
    """

    def fit(self, X: pd.DataFrame, y):
        self.title_age_ = add_title(X).groupby("Title")["Age"].median()

        groups = pd.DataFrame(
            {"key": X["Ticket"].values, "pid": X["PassengerId"].values, "y": np.asarray(y)}
        )
        self.group_table_ = groups.groupby("key")["y"].agg(["sum", "count"])
        self.loo_cache_ = {}
        for row in groups.itertuples():
            total, size = self.group_table_.loc[row.key]
            others_sum, others_count = total - row.y, size - 1
            self.loo_cache_[row.pid] = (
                0.5 if others_count == 0 else float(others_sum > 0)
            )
        return self

    def _family_survival(self, X: pd.DataFrame) -> pd.Series:
        cached = X["PassengerId"].map(self.loo_cache_)
        any_survived = X["Ticket"].map(self.group_table_["sum"]).gt(0).astype(float)
        known_group = X["Ticket"].isin(self.group_table_.index)
        return cached.fillna(any_survived.where(known_group, 0.5))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = add_title(X)
        return df.assign(
            Age=df["Age"].fillna(df["Title"].map(self.title_age_)),
            FamilySurvival=self._family_survival(X),
        )
