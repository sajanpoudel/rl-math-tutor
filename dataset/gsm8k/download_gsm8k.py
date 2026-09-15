#!/usr/bin/env python3
"""Download GSM8K dataset and convert to CSV format."""

import json
import csv
import os
from pathlib import Path

def download_gsm8k():
    """Download GSM8K dataset and save as CSV files."""
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("Error: Required packages not installed.")
        print("Please install with: pip3 install --break-system-packages datasets pandas")
        print("Or use a virtual environment.")
        return False

    # Create dataset folder
    dataset_folder = Path("dataset/gsm8k")
    dataset_folder.mkdir(parents=True, exist_ok=True)

    print("Downloading GSM8K dataset...")

    # Load the dataset
    dataset = load_dataset("gsm8k", "main")

    # Convert train split to CSV
    print("Converting train split to CSV...")
    train_df = pd.DataFrame(dataset['train'])
    train_csv_path = dataset_folder / "train.csv"
    train_df.to_csv(train_csv_path, index=False)
    print(f"Saved train split to {train_csv_path} ({len(train_df)} rows)")

    # Convert test split to CSV
    print("Converting test split to CSV...")
    test_df = pd.DataFrame(dataset['test'])
    test_csv_path = dataset_folder / "test.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test split to {test_csv_path} ({len(test_df)} rows)")

    print("\nDataset download and conversion complete!")
    print(f"Files saved in: {dataset_folder.absolute()}")
    print(f"\nColumns: {list(train_df.columns)}")
    print(f"\nSample row from train set:")
    print(train_df.iloc[0].to_dict())

    return True

if __name__ == "__main__":
    download_gsm8k()
