"""Train a baseline model and write a submission file.

Usage: uv run python src/train.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from omegaconf import OmegaConf
from config import load_config

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
TARGET = "Survived"

MODELS = {
    "logreg": LogisticRegression,
    "knn": KNeighborsClassifier,
    "random_forest": RandomForestClassifier,
}

def build_pipeline(cfg) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    if cfg.model not in MODELS:
        raise ValueError(f"Unknown model {cfg.model!r}, choose from {sorted(MODELS)}")

    params = OmegaConf.to_container(cfg.params)
    model = MODELS[cfg.model](**params)

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X, y = train[features], train[TARGET]

    pipeline = build_pipeline(cfg)

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    pipeline.fit(X, y)
    predictions = pipeline.predict(test[features])

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": predictions}
    )
    out = SUBMISSIONS_DIR / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"Submission written to {out}")


if __name__ == "__main__":
    main()
