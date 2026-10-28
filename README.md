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

Trains the model from `configs/config.yaml` (random forest by default), prints stratified K-fold CV accuracy, and writes `submissions/<model>_submission.csv`.

Any config field can be overridden from the CLI:

```sh
uv run python main.py model=logreg
uv run python main.py model=knn params.knn.n_neighbors=25 cv.n_folds=10
```

Available models: `logreg`, `knn`, `random_forest`.

## Submit

```sh
uv run kaggle competitions submit -c titanic -f submissions/random_forest_submission.csv -m "baseline random forest"
```

## Project layout

```
├── main.py                # entry point: load config, train with CV, write submission
├── configs/
│   └── config.yaml        # experiment settings (model, hyperparams, CV)
├── src/
│   ├── config.py          # config schema + loader (OmegaConf)
│   └── download_data.py   # fetch competition data via Kaggle API
├── notebooks/             # exploratory analysis
├── data/                  # raw data (git-ignored)
└── submissions/           # generated submissions (git-ignored)
```
