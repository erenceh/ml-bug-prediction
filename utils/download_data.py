"""
download_data.py

A script to download Kaggle datasets for the ML Bug Triage project.
Place your Kaggle API token at ~/.kaggle/kaggle.json before running.
"""

import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "../data/raw"
DATASETS = {"github_issues": "davidshinn/github-issues", "gitbugs": "av9ash/gitbugs"}


def ensure_data_dir():
    """
    Create data/raw folder if it doesn't exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def download_dataset(api, dataset_name, dataset_slug):
    """
    Download and unzip Kaggle datasets

    Args:
        api (kaggle.api.kaggle_api_extended.KaggleApi): An authenticated Kaggle API object.
        dataset_name (str): A name for the dataset.
        dataset_slug (str): The Kaggle dataset identifier, e.g., "username/dataset-name".
    """
    print(f"Downloading {dataset_name} from Kaggle: {dataset_slug}...")
    try:
        api.dataset_download_files(dataset_slug, path=DATA_DIR, unzip=True, quiet=False)
        print(f"{dataset_name} downloaded and extracted to {DATA_DIR}")
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()

    ensure_data_dir()

    for name, slug in DATASETS.items():
        download_dataset(api, name, slug)

    print("Datasets downloaded successfully!")
