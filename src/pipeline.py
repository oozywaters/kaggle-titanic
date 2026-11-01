from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODELS = {
    "logreg": LogisticRegression,
    "knn": KNeighborsClassifier,
    "random_forest": RandomForestClassifier,
}

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
TARGET = "Survived"

def build_preprocessor() -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]

    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
    ])


def build_model(cfg):
    """Instantiate the configured model with its per-model params."""
    if cfg.model not in MODELS:
        raise ValueError(f"unknown model {cfg.model!r}, choose from {sorted(MODELS)}")
    params = OmegaConf.to_container(cfg.params).get(cfg.model, {})
    return MODELS[cfg.model](**params)

def build_pipeline(cfg) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", build_model(cfg)),
    ])
