"""Cross-validate every model and print a ranked comparison table.

Usage: uv run python compare.py [cv.n_folds=...] [seed=...]
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import load_config
from src.pipeline import ENSEMBLES, MODELS, TARGET, build_pipeline

def main() -> None:
    cfg = load_config()
    names = list(MODELS) + list(ENSEMBLES)
    print(f"comparing {len(names)} models | cv={cfg.cv.n_folds} folds, seed={cfg.seed}")

    train = pd.read_csv(Path(cfg.data_dir) / "train.csv")
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    cv = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.seed)

    rows = []
    for name in names:
        model_cfg = cfg.copy()
        model_cfg.model = name
        scores = cross_val_score(
            build_pipeline(model_cfg), X, y, cv=cv, scoring="accuracy"
        )
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
        rows.append({"model": name, "cv_accuracy": scores.mean(), "std": scores.std()})

    results = (
        pd.DataFrame(rows)
        .sort_values("cv_accuracy", ascending=False)
        .round(4)
        .reset_index(drop=True)
    )
    results[""] = ""
    results.loc[0, ""] = "<- best"
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
