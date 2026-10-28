"""Train a baseline model and write a submission file.

Usage: uv run python src/train.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
TARGET = "Survived"


def build_pipeline() -> Pipeline:
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
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X, y = train[features], train[TARGET]

    pipeline = build_pipeline()

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
