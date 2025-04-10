import kaggle
import os

def download_kaggle_dataset(dataset_name, dataset_path):
    """Download dataset from Kaggle"""
    kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
