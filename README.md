# Kaggle Titanic

Boilerplate for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Get the data

Set up [Kaggle API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`), accept the competition rules on the Kaggle website, then:

```sh
uv run python src/download_data.py
```

This downloads `train.csv`, `test.csv`, and `gender_submission.csv` into `data/` (git-ignored).

## Train

```sh
uv run python main.py
```

Trains the model from `configs/config.yaml` (catboost by default), prints per-fold and mean stratified K-fold CV accuracy, and writes `submissions/<model>_submission.csv`. One run = one experiment; copy the printed line into your experiment notes.

Any config field can be overridden from the CLI:

```sh
uv run python main.py model=logreg
uv run python main.py model=knn params.knn.n_neighbors=25 cv.n_folds=10
```

Available models: `logreg`, `ridge`, `lasso_logreg`, `elasticnet_logreg`, `knn`, `decision_tree`, `random_forest`, `catboost`, `lightgbm`, `xgboost`, `mlp`, plus the ensembles `voting_soft`, `voting_hard`, `stacking_logreg`, `stacking_ridge`.

### Neural network

`mlp` is a PyTorch MLP wrapped as an sklearn estimator. Architecture and training are fully config-driven via `params.mlp`: `hidden` (layer sizes), `activation` (`relu`/`gelu`/`tanh`), `batchnorm`, `dropout`, `optimizer` (`adam`/`adamw`/`sgd`), `lr`, `weight_decay`, `scheduler` (`none`/`cosine`), `epochs`, `batch_size`.

```sh
uv run python main.py model=mlp params.mlp.hidden='[128,64,32]' params.mlp.dropout=0.3 params.mlp.scheduler=cosine
```

### Ensembles

Ensembles combine the base models listed in `ensemble.estimators` (every entry needs `predict_proba`, so no `ridge`): `voting_soft` averages predicted probabilities, `voting_hard` takes a majority vote, and `stacking_logreg`/`stacking_ridge` train a logistic regression / Ridge meta-model on out-of-fold predictions (`ensemble.stacking_cv` inner folds, so stacking runs take a few minutes).

## Submit

```sh
uv run kaggle competitions submit -c titanic -f submissions/catboost_submission.csv -m "catboost baseline"
```

## Project layout

```
├── main.py                # entry point: load config, train with CV, write submission
├── configs/
│   └── config.yaml        # experiment settings (model, hyperparams, CV)
├── src/
│   ├── config.py          # config schema + loader (OmegaConf)
│   ├── features.py        # feature engineering (sklearn transformer)
│   ├── pipeline.py        # model registry, ensembles, preprocessing, pipeline builder
│   ├── mlp.py             # PyTorch MLP with an sklearn interface
│   └── download_data.py   # fetch competition data via Kaggle API
├── notebooks/             # exploratory analysis
├── data/                  # raw data (git-ignored)
└── submissions/           # generated submissions (git-ignored)
```
