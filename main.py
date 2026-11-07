"""Train a model and write a submission file.

Usage: uv run python main.py [model=...] [params.<model>.<param>=...] [cv.n_folds=...]
"""

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import load_config
from src.pipeline import ENSEMBLES, TARGET, build_pipeline

def main() -> None:
    cfg = load_config()
    params = OmegaConf.to_container(cfg.params, resolve=True).get(cfg.model, {})
    details = " ".join(f"{k}={v}" for k, v in params.items())
    if cfg.model in ENSEMBLES:
        details = f"estimators={list(cfg.ensemble.estimators)}"
    print(f"model={cfg.model} | cv={cfg.cv.n_folds} folds, seed={cfg.seed} | {details}")

    data_dir = Path(cfg.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X, y = train.drop(columns=[TARGET]), train[TARGET]

    pipeline = build_pipeline(cfg)

    cv = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    folds = " ".join(f"{s:.4f}" for s in scores)
    print(f"folds: {folds}")
    print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    pipeline.fit(X, y)
    predictions = pipeline.predict(test)

    out = Path(cfg.submission_path)
    out.parent.mkdir(exist_ok=True)
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": predictions}
    )
    submission.to_csv(out, index=False)
    print(f"Submission written to {out}")


if __name__ == "__main__":
    main()
