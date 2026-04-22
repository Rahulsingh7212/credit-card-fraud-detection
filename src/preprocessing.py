"""
preprocessing.py
- Splits data into features and labels
- Scales the Amount and Time columns
- Handles class imbalance using SMOTE or undersampling
- Performs train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import os


def separate_features_labels(df):
    """Separate features (X) and target label (y)."""
    X = df.drop('Class', axis=1)
    y = df['Class']
    print(f"✅ Features shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


def scale_features(X_train, X_test):
    """Scale 'Amount' and 'Time' columns using StandardScaler."""
    scaler = StandardScaler()

    # Scale 'Amount' and 'Time' for training data
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit on training data only
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

    # Save scaler for future use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Features scaled (Amount & Time). Scaler saved to models/scaler.pkl")

    return X_train, X_test, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class ratio in both splits
    )
    print(f"✅ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print(f"   Train - Legitimate: {(y_train==0).sum()}, Fraud: {(y_train==1).sum()}")
    print(f"   Test  - Legitimate: {(y_test==0).sum()}, Fraud: {(y_test==1).sum()}")
    return X_train, X_test, y_train, y_test


def handle_imbalance(X_train, y_train, method='smote', random_state=42):
    """
    Handle class imbalance using one of:
    - 'smote': Synthetic Minority Over-sampling
    - 'undersample': Random Under-sampling
    - 'smote_tomek': Combination of SMOTE + Tomek links
    - 'none': No balancing (use class weights in models instead)
    """
    print(f"\n⚖️  Handling imbalance using method: '{method}'")

    if method == 'none':
        print("   No resampling applied. Will use class weights in models.")
        return X_train, y_train

    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'smote', 'undersample', 'smote_tomek', or 'none'.")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print(f"   Before resampling - Legitimate: {(y_train==0).sum()}, Fraud: {(y_train==1).sum()}")
    print(f"   After resampling  - Legitimate: {(y_resampled==0).sum()}, Fraud: {(y_resampled==1).sum()}")

    return X_resampled, y_resampled


def preprocess_pipeline(df, balance_method='smote'):
    """
    Full preprocessing pipeline:
    1. Separate features and labels
    2. Split into train/test
    3. Scale features
    4. Handle class imbalance (on training data only)
    """
    print("\n" + "=" * 60)
    print("🔧 PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Separate
    X, y = separate_features_labels(df)

    # Step 2: Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Scale
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Step 4: Handle imbalance (ONLY on training data!)
    X_train_balanced, y_train_balanced = handle_imbalance(
        X_train, y_train, method=balance_method
    )

    print("\n✅ Preprocessing complete!\n")

    return X_train_balanced, X_test, y_train_balanced, y_test