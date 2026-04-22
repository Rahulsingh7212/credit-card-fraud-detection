"""
visualize.py
- Creates visualizations for data exploration and model evaluation
- Class distribution, correlation, ROC curves, confusion matrices
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_class_distribution(df, save=True):
    """Plot the distribution of legitimate vs fraudulent transactions."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    ax1 = axes[0]
    counts = df['Class'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['Legitimate (0)', 'Fraud (1)'], counts.values, color=colors)
    ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Transactions')
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2000,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')

    # Percentage plot
    ax2 = axes[1]
    percentages = counts / len(df) * 100
    ax2.pie(percentages.values, labels=['Legitimate', 'Fraud'],
            colors=colors, autopct='%1.3f%%', startangle=90,
            explode=[0, 0.3], textprops={'fontsize': 12})
    ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/class_distribution.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/class_distribution.png")
    plt.show()
    plt.close()


def plot_amount_distribution(df, save=True):
    """Plot distribution of transaction amounts by class."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall amount distribution
    ax1 = axes[0]
    ax1.hist(df['Amount'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Amount')
    ax1.set_ylabel('Frequency')

    # Amount by class
    ax2 = axes[1]
    df[df['Class']==0]['Amount'].hist(bins=50, alpha=0.7, color='#2ecc71',
                                       label='Legitimate', ax=ax2, edgecolor='black')
    df[df['Class']==1]['Amount'].hist(bins=50, alpha=0.7, color='#e74c3c',
                                       label='Fraud', ax=ax2, edgecolor='black')
    ax2.set_title('Amount by Class', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Amount')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/amount_distribution.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/amount_distribution.png")
    plt.show()
    plt.close()


def plot_time_distribution(df, save=True):
    """Plot distribution of transaction time by class."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    df[df['Class']==0]['Time'].hist(bins=100, alpha=0.7, color='#2ecc71',
                                     label='Legitimate', ax=ax)
    df[df['Class']==1]['Time'].hist(bins=100, alpha=0.7, color='#e74c3c',
                                     label='Fraud', ax=ax)

    ax.set_title('Transaction Time Distribution by Class', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency')
    ax.legend()

    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/time_distribution.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/time_distribution.png")
    plt.show()
    plt.close()


def plot_correlation_heatmap(df, save=True):
    """Plot correlation heatmap of features."""
    set_style()
    fig, ax = plt.subplots(figsize=(16, 12))

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                annot=False, fmt='.2f', square=True, ax=ax,
                linewidths=0.5)

    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/correlation_heatmap.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/correlation_heatmap.png")
    plt.show()
    plt.close()


def plot_confusion_matrices(cms, save=True):
    """Plot confusion matrices for all models side by side."""
    set_style()
    n_models = len(cms)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/confusion_matrices.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/confusion_matrices.png")
    plt.show()
    plt.close()


def plot_roc_curves(curves_dict, save=True):
    """Plot ROC curves for all models on one plot."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']

    for i, (name, data) in enumerate(curves_dict.items()):
        if 'fpr' in data and data['fpr'] is not None:
            ax.plot(data['fpr'], data['tpr'],
                    color=colors[i % len(colors)],
                    label=f'{name}', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/roc_curves.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/roc_curves.png")
    plt.show()
    plt.close()


def plot_precision_recall_curves(curves_dict, save=True):
    """Plot Precision-Recall curves for all models."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']

    for i, (name, data) in enumerate(curves_dict.items()):
        if 'precision_curve' in data and data['precision_curve'] is not None:
            ax.plot(data['recall_curve'], data['precision_curve'],
                    color=colors[i % len(colors)],
                    label=f'{name}', linewidth=2)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/precision_recall_curves.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/precision_recall_curves.png")
    plt.show()
    plt.close()


def plot_model_comparison(comparison_df, save=True):
    """Plot bar chart comparing model metrics."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(comparison_df.index))
    width = 0.15

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for i, metric in enumerate(metrics_to_plot):
        if metric in comparison_df.columns:
            values = comparison_df[metric].values
            ax.bar(x + i * width, values, width, label=metric,
                   color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df.index, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)

    # Add value labels on bars
    for i, metric in enumerate(metrics_to_plot):
        if metric in comparison_df.columns:
            values = comparison_df[metric].values
            for j, v in enumerate(values):
                ax.text(x[j] + i * width, v + 0.01, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches='tight')
        print("💾 Saved: plots/model_comparison.png")
    plt.show()
    plt.close()


def plot_feature_importance(models, feature_names, save=True):
    """Plot feature importance for tree-based models."""
    set_style()

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importances[indices],
                    color='#3498db', edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top 15 Features - {name}', fontsize=14, fontweight='bold')
            ax.invert_yaxis()

            plt.tight_layout()
            if save:
                os.makedirs("plots", exist_ok=True)
                safe_name = name.lower().replace(" ", "_")
                plt.savefig(f"plots/feature_importance_{safe_name}.png",
                           dpi=150, bbox_inches='tight')
                print(f"💾 Saved: plots/feature_importance_{safe_name}.png")
            plt.show()
            plt.close()


def run_all_visualizations(df, comparison_df, cms, curves, models, feature_names):
    """Run all visualization functions."""
    print("\n" + "=" * 60)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs("plots", exist_ok=True)

    # Data exploration plots
    print("\n📊 Data Exploration Plots...")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_distribution(df)
    plot_correlation_heatmap(df)

    # Model evaluation plots
    print("\n📊 Model Evaluation Plots...")
    plot_confusion_matrices(cms)
    plot_roc_curves(curves)
    plot_precision_recall_curves(curves)
    plot_model_comparison(comparison_df)
    plot_feature_importance(models, feature_names)

    print("\n✅ All visualizations generated! Check the 'plots/' folder.")