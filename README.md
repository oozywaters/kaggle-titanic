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

## Train the baseline

```sh
uv run python src/train.py
```

Trains a random forest on basic features, prints 5-fold CV accuracy, and writes `submissions/submission.csv`.

## Submit

```sh
uv run kaggle competitions submit -c titanic -f submissions/submission.csv -m "baseline random forest"
```

## Project layout

```
├── src/
│   ├── download_data.py   # fetch competition data via Kaggle API
│   └── train.py           # baseline model + submission
├── notebooks/             # exploratory analysis
├── data/                  # raw data (git-ignored)
└── submissions/           # generated submissions (git-ignored)
```
