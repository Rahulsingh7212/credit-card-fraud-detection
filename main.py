"""
main.py
=======
Credit Card Fraud Detection - Main Entry Point

This script runs the complete pipeline:
1. Load & explore data
2. Preprocess data
3. Train models (Logistic Regression, Decision Tree, Random Forest)
4. Evaluate models
5. Generate visualizations
6. Save results

Usage:
    python main.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, explore_data
from src.preprocessing import preprocess_pipeline
from src.models import train_all_models, predict, predict_probability
from src.evaluation import evaluate_all_models, print_comparison
from src.visualize import run_all_visualizations


def main():
    """Main function to run the fraud detection pipeline."""

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       💳 CREDIT CARD FRAUD DETECTION SYSTEM 💳          ║
    ║                                                          ║
    ║   Detecting fraudulent transactions using Machine Learning║
    ║   Algorithms: Logistic Regression, Decision Tree,        ║
    ║               Random Forest                              ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    total_start = time.time()

    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 1: LOADING DATA")
    print("▶" * 60)

    df = load_data("data/creditcard.csv")

    # ============================================================
    # STEP 2: Explore Data
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 2: EXPLORING DATA")
    print("▶" * 60)

    class_counts = explore_data(df)

    # ============================================================
    # STEP 3: Preprocess Data
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 3: PREPROCESSING DATA")
    print("▶" * 60)

    # Choose balance method: 'smote', 'undersample', 'smote_tomek', 'none'
    BALANCE_METHOD = 'smote'

    X_train, X_test, y_train, y_test = preprocess_pipeline(
        df,
        balance_method=BALANCE_METHOD
    )

    # Store feature names for visualization
    feature_names = X_train.columns.tolist()

    # ============================================================
    # STEP 4: Train Models
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 4: TRAINING MODELS")
    print("▶" * 60)

    models = train_all_models(X_train, y_train, balance_method=BALANCE_METHOD)

    # ============================================================
    # STEP 5: Evaluate Models
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 5: EVALUATING MODELS")
    print("▶" * 60)

    comparison_df, cms, curves = evaluate_all_models(models, X_test, y_test)
    comparison_df = print_comparison(comparison_df)

    # ============================================================
    # STEP 6: Visualizations
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("▶" * 60)

    try:
        run_all_visualizations(df, comparison_df, cms, curves, models, feature_names)
    except Exception as e:
        print(f"⚠️ Visualization error (non-critical): {e}")
        print("   This may happen in headless environments. Plots are optional.")

    # ============================================================
    # STEP 7: Save Results
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 7: SAVING RESULTS")
    print("▶" * 60)

    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv("results/model_comparison.csv")
    print("💾 Saved: results/model_comparison.csv")

    # ============================================================
    # STEP 8: Demo - Test on a Single Transaction
    # ============================================================
    print("\n" + "▶" * 60)
    print("STEP 8: DEMO - TESTING ON SAMPLE TRANSACTIONS")
    print("▶" * 60)

    # Test on a few samples from the test set
    print("\n🔍 Testing on 5 random test samples:")
    print("-" * 60)

    sample_indices = X_test.sample(5, random_state=42).index

    for idx, actual_idx in enumerate(sample_indices):
        sample = X_test.loc[[actual_idx]]
        actual_class = y_test.loc[actual_idx]

        print(f"\n   Sample {idx+1} (Index: {actual_idx}):")
        print(f"   Actual Class: {'🔴 FRAUD' if actual_class == 1 else '🟢 LEGITIMATE'}")

        for name, model in models.items():
            pred = predict(model, sample)[0]
            prob = predict_probability(model, sample)
            prob_str = f" (Prob: {prob[0]:.4f})" if prob is not None else ""
            emoji = '🔴' if pred == 1 else '🟢'
            match = '✅' if pred == actual_class else '❌'
            print(f"     {name}: {emoji} {'Fraud' if pred == 1 else 'Legitimate'}{prob_str} {match}")

    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - total_start

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║          ✅ PIPELINE COMPLETED SUCCESSFULLY ✅           ║")
    print(f"║          Total Time: {total_time:.1f} seconds" + " " * (58 - 28 - len(f"{total_time:.1f}")) + "║")
    print("║" + " " * 58 + "║")
    print("║   📁 Models saved to:  models/                          ║")
    print("║   📁 Plots saved to:   plots/                           ║")
    print("║   📁 Results saved to: results/                         ║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()