"""
models.py
- Defines and trains multiple classification models
- Logistic Regression, Decision Tree, Random Forest
- Saves trained models to disk
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os
import time
import numpy as np


def get_models(balance_method='smote'):
    """
    Return a dictionary of models to train.
    If balance_method is 'none', we use class_weight='balanced'.
    """
    use_balanced = (balance_method == 'none')

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced' if use_balanced else None,
            random_state=42,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced' if use_balanced else None,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced' if use_balanced else None,
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def train_model(model, X_train, y_train, model_name="Model"):
    """Train a single model and report time taken."""
    print(f"\n🚀 Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"   ✅ {model_name} trained in {elapsed:.2f} seconds")
    return model


def cross_validate_model(model, X_train, y_train, model_name="Model", cv=5):
    """Perform cross-validation and print results."""
    print(f"\n📊 Cross-Validating {model_name} ({cv}-fold)...")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    print(f"   F1 Scores: {scores}")
    print(f"   Mean F1: {scores.mean():.4f} (±{scores.std():.4f})")
    return scores


def train_all_models(X_train, y_train, balance_method='smote'):
    """Train all models and return them in a dictionary."""
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING")
    print("=" * 60)

    models = get_models(balance_method)
    trained_models = {}

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        # Train
        trained_model = train_model(model, X_train, y_train, name)

        # Cross-validate (only if dataset isn't too large from SMOTE)
        if X_train.shape[0] <= 500000:
            try:
                cross_validate_model(trained_model, X_train, y_train, name, cv=3)
            except Exception as e:
                print(f"   ⚠️ Cross-validation skipped: {e}")

        # Save model
        filename = name.lower().replace(" ", "_") + ".pkl"
        filepath = os.path.join("models", filename)
        joblib.dump(trained_model, filepath)
        print(f"   💾 Saved to {filepath}")

        trained_models[name] = trained_model

    print("\n✅ All models trained and saved!\n")
    return trained_models


def load_model(filepath):
    """Load a saved model from disk."""
    return joblib.load(filepath)


def predict(model, X):
    """Make predictions using a trained model."""
    y_pred = model.predict(X)
    return y_pred


def predict_probability(model, X):
    """Get prediction probabilities."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]  # Probability of fraud
    else:
        return None