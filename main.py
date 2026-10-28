"""Train a model and write a submission file.

Usage: uv run python main.py [model=...] [params.<model>.<param>=...] [cv.n_folds=...]
"""

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import load_config

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

    params = OmegaConf.to_container(cfg.params).get(cfg.model, {})
    model = MODELS[cfg.model](**params)

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))

    data_dir = Path(cfg.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X, y = train[features], train[TARGET]

    pipeline = build_pipeline(cfg)

    cv = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    pipeline.fit(X, y)
    predictions = pipeline.predict(test[features])

    out = Path(cfg.submission_path)
    out.parent.mkdir(exist_ok=True)
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": predictions}
    )
    submission.to_csv(out, index=False)
    print(f"Submission written to {out}")


if __name__ == "__main__":
    main()
