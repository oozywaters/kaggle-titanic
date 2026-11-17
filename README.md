# Kaggle Titanic

ML pipeline for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Get the data

The data is committed to the repo, so this step is optional — only needed to refresh it. Set up [Kaggle API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`), accept the competition rules on the Kaggle website, then:

```sh
uv run python src/download_data.py
```

This downloads `train.csv`, `test.csv`, and `gender_submission.csv` into `data/`.

## Train

```sh
uv run python main.py
```

Trains the model from `configs/config.yaml` (CatBoost by default), prints stratified K-fold CV accuracy (the leaderboard metric), and writes `submissions/<model>_submission.csv`.

Any config field can be overridden from the CLI:

```sh
uv run python main.py model=logreg
uv run python main.py model=mlp params.mlp.dropout=0.3 cv.n_folds=10
```

Available models: `logreg` (baseline), `ridge`, `lasso_logreg`, `elasticnet_logreg`, `knn`, `decision_tree`, `random_forest`, `catboost`, `lightgbm`, `xgboost`, `mlp` (PyTorch), plus ensembles `voting_soft`, `voting_hard`, `stacking_logreg`, `stacking_ridge` (base models set via `ensemble.estimators`).

## Compare models

```sh
uv run python compare.py
```

Cross-validates every model (ensembles included) and prints a ranked table. Best so far: **catboost, CV accuracy 0.847** (public LB 0.780). Full analysis in `notebooks/eda.ipynb`.

## Submit

```sh
uv run kaggle competitions submit -c titanic -f submissions/catboost_submission.csv -m "catboost baseline"
```

## Project layout

```
├── main.py                # entry point: load config, train with CV, write submission
├── compare.py             # cross-validate all models, print ranked table
├── configs/
│   └── config.yaml        # experiment settings (model, hyperparams, CV, ensembles)
├── src/
│   ├── config.py          # config schema + loader (OmegaConf)
│   ├── download_data.py   # fetch competition data via Kaggle API
│   ├── features.py        # feature engineering (sklearn transformer)
│   ├── mlp.py             # PyTorch MLP wrapped as a sklearn classifier
│   └── pipeline.py        # model registry, ensembles, preprocessing pipeline
├── notebooks/             # EDA (eda.ipynb)
├── data/                  # raw competition data
└── submissions/           # generated submissions (git-ignored)
```
