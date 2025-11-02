"""
Master script to run all analyses
"""

import subprocess
import sys

scripts = [
    'crop_analysis.py',
    'task1_yield_prediction.py',
    'task1_visualizations.py',
    'task2_crop_classification.py',
    'task2_visualizations.py'
]

print("=" * 70)
print("RUNNING COMPLETE CROP ANALYSIS PROJECT")
print("=" * 70)

for script in scripts:
    print(f"\n{'=' * 70}")
    print(f"Running: {script}")
    print("=" * 70)
    try:
        result = subprocess.run([sys.executable, script], check=True, capture_output=False)
        print(f"✓ {script} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script}: {e}")
        sys.exit(1)

print("\n" + "=" * 70)
print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files:")
print("  - agricultural_data.csv")
print("  - task1_actual_vs_predicted.png")
print("  - task1_residual_plots.png")
print("  - task1_overfitting_analysis.png")
print("  - task1_tree_structure.png")
print("  - task1_feature_importance.png")
print("  - task2_confusion_matrices.png")
print("  - task2_decision_boundaries.png")
print("  - task2_model_comparison.png")
print("  - task2_feature_distributions.png")
