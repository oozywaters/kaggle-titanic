from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import TitanicFeatures
from src.mlp import TorchMLPClassifier

MODELS = {
    "logreg": LogisticRegression,
    "ridge": RidgeClassifier,
    "lasso_logreg": LogisticRegression,
    "elasticnet_logreg": LogisticRegression,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "catboost": CatBoostClassifier,
    "lightgbm": LGBMClassifier,
    "xgboost": XGBClassifier,
    "mlp": TorchMLPClassifier,
}

ENSEMBLES = ("voting_soft", "voting_hard", "stacking_logreg", "stacking_ridge")

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySurvival"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title"]
TARGET = "Survived"

def build_preprocessor() -> ColumnTransformer:
    num_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]

    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
    ])

def make_model(name, params):
    """Instantiate a registry model by name with its params from the config."""
    if name not in MODELS:
        raise ValueError(
            f"unknown model {name!r}, choose from {sorted(MODELS) + sorted(ENSEMBLES)}"
        )
    return MODELS[name](**params.get(name, {}))

def build_ensemble(name, cfg, params):
    estimators = [(n, make_model(n, params)) for n in cfg.ensemble.estimators]
    if name == "voting_soft":
        return VotingClassifier(estimators, voting="soft")
    if name == "voting_hard":
        return VotingClassifier(estimators, voting="hard")
    final = LogisticRegression(max_iter=2000) if name == "stacking_logreg" else RidgeClassifier()
    return StackingClassifier(estimators, final_estimator=final, cv=cfg.ensemble.stacking_cv)

def build_model(cfg):
    params = OmegaConf.to_container(cfg.params, resolve=True) or {}
    if cfg.model in ENSEMBLES:
        return build_ensemble(cfg.model, cfg, params)
    return make_model(cfg.model, params)

def build_pipeline(cfg) -> Pipeline:
    return Pipeline([
        ("features", TitanicFeatures()),
        ("preprocess", build_preprocessor()),
        ("model", build_model(cfg)),
    ])
