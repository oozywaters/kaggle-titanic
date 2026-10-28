"""Download competition data from Kaggle into data/.

Requires Kaggle API credentials: https://www.kaggle.com/docs/api
(place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY).
"""

from pathlib import Path
import zipfile

COMPETITION = "titanic"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    DATA_DIR.mkdir(exist_ok=True)
    api.competition_download_files(COMPETITION, path=str(DATA_DIR))

    archive = DATA_DIR / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)
    archive.unlink()

    print(f"Data extracted to {DATA_DIR}:")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
