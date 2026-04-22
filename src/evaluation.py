"""
evaluation.py
- Evaluates models using various metrics
- Confusion matrix, classification report
- ROC-AUC, Precision-Recall AUC
- Compares all models
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import pandas as pd
import numpy as np


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a single model and return metrics dictionary."""
    print(f"\n{'='*60}")
    print(f"📈 EVALUATING: {model_name}")
    print(f"{'='*60}")

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (for ROC-AUC)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
    }

    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_prob)
        metrics['PR-AUC'] = average_precision_score(y_test, y_prob)
    else:
        metrics['ROC-AUC'] = None
        metrics['PR-AUC'] = None

    # Print metrics
    print(f"\n   Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f}")
    print(f"   Recall:    {metrics['Recall']:.4f}")
    print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
    if metrics['ROC-AUC']:
        print(f"   ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print(f"   PR-AUC:    {metrics['PR-AUC']:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   ┌─────────────┬─────────────┐")
    print(f"   │ TN: {cm[0][0]:>6}  │ FP: {cm[0][1]:>6}  │")
    print(f"   ├─────────────┼─────────────┤")
    print(f"   │ FN: {cm[1][0]:>6}  │ TP: {cm[1][1]:>6}  │")
    print(f"   └─────────────┴─────────────┘")
    print(f"   (Row=Actual, Col=Predicted | 0=Legit, 1=Fraud)")

    # Classification Report
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])
    for line in report.split('\n'):
        print(f"   {line}")

    # Store curves data
    curves = {}
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        curves['fpr'] = fpr
        curves['tpr'] = tpr
        curves['precision_curve'] = prec
        curves['recall_curve'] = rec

    return metrics, cm, curves


def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models and return comparison DataFrame."""
    print("\n" + "=" * 60)
    print("📊 EVALUATING ALL MODELS")
    print("=" * 60)

    all_metrics = []
    all_curves = {}
    all_cms = {}

    for name, model in models.items():
        metrics, cm, curves = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        all_curves[name] = curves
        all_cms[name] = cm

    # Comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.set_index('Model')

    return comparison_df, all_cms, all_curves


def print_comparison(comparison_df):
    """Print a formatted comparison of all models."""
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string())

    # Best model
    best_f1 = comparison_df['F1-Score'].idxmax()
    best_recall = comparison_df['Recall'].idxmax()
    best_precision = comparison_df['Precision'].idxmax()
    best_roc = comparison_df['ROC-AUC'].idxmax()

    print(f"\n🥇 Best F1-Score:   {best_f1} ({comparison_df.loc[best_f1, 'F1-Score']:.4f})")
    print(f"🥇 Best Recall:     {best_recall} ({comparison_df.loc[best_recall, 'Recall']:.4f})")
    print(f"🥇 Best Precision:  {best_precision} ({comparison_df.loc[best_precision, 'Precision']:.4f})")
    print(f"🥇 Best ROC-AUC:    {best_roc} ({comparison_df.loc[best_roc, 'ROC-AUC']:.4f})")

    # Overall recommendation
    print(f"\n💡 RECOMMENDATION:")
    print(f"   For fraud detection, RECALL is often most important")
    print(f"   (we don't want to miss fraudulent transactions).")
    print(f"   However, PRECISION matters to avoid false alarms.")
    print(f"   F1-Score balances both — best: {best_f1}")

    return comparison_df