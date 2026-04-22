"""
data_loader.py
- Loads the credit card dataset
- Provides basic exploration functions
"""

import pandas as pd
import os

def load_data(filepath="data/creditcard.csv"):
    """Load the credit card CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please download it from:\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place 'creditcard.csv' in the 'data/' folder."
        )
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """Print basic exploration of the dataset."""
    print("\n" + "=" * 60)
    print("📊 DATASET EXPLORATION")
    print("=" * 60)

    print("\n🔹 First 5 Rows:")
    print(df.head())

    print("\n🔹 Dataset Info:")
    print(df.info())

    print("\n🔹 Statistical Summary:")
    print(df.describe())

    print("\n🔹 Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found ✅")

    print("\n🔹 Class Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"\n   Fraud Rate: {class_counts[1] / len(df) * 100:.4f}%")
    print(f"   Legitimate: {class_counts[0]} ({class_counts[0]/len(df)*100:.3f}%)")
    print(f"   Fraud:      {class_counts[1]} ({class_counts[1]/len(df)*100:.3f}%)")

    return class_counts