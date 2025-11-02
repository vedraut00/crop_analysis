"""
Task 1: Yield Prediction (8 marks)
- Multiple Linear Regression
- Regression Tree
- Comparison and Discussion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('agricultural_data.csv')

# Prepare features and target
X = df[['soil_moisture', 'rainfall', 'temperature', 'fertilizer']]
y = df['yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 70)
print("TASK 1: YIELD PREDICTION")
print("=" * 70)

# ============================================================================
# 1. Multiple Linear Regression
# ============================================================================
print("\n" + "=" * 70)
print("1. MULTIPLE LINEAR REGRESSION")
print("=" * 70)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr_train = lr_model.predict(X_train)
y_pred_lr_test = lr_model.predict(X_test)

# Metrics
lr_train_mse = mean_squared_error(y_train, y_pred_lr_train)
lr_test_mse = mean_squared_error(y_test, y_pred_lr_test)
lr_train_r2 = r2_score(y_train, y_pred_lr_train)
lr_test_r2 = r2_score(y_test, y_pred_lr_test)
lr_test_mae = mean_absolute_error(y_test, y_pred_lr_test)

print(f"\nLinear Regression Coefficients:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"  {feature}: {coef:.6f}")
print(f"  Intercept: {lr_model.intercept_:.6f}")

print(f"\nLinear Regression Performance:")
print(f"  Training MSE: {lr_train_mse:.4f}")
print(f"  Testing MSE: {lr_test_mse:.4f}")
print(f"  Training R²: {lr_train_r2:.4f}")
print(f"  Testing R²: {lr_test_r2:.4f}")
print(f"  Testing MAE: {lr_test_mae:.4f}")

# Cross-validation
cv_scores_lr = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
print(f"\n5-Fold Cross-Validation R² Scores: {cv_scores_lr}")
print(f"Mean CV R²: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

# ============================================================================
# 2. Regression Tree (with different depths to show overfitting)
# ============================================================================
print("\n" + "=" * 70)
print("2. REGRESSION TREE ANALYSIS")
print("=" * 70)

# Test different tree depths
depths = [2, 3, 5, 10, None]
tree_results = []

for depth in depths:
    tree_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_model.fit(X_train, y_train)
    
    y_pred_tree_train = tree_model.predict(X_train)
    y_pred_tree_test = tree_model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_tree_train)
    test_mse = mean_squared_error(y_test, y_pred_tree_test)
    train_r2 = r2_score(y_train, y_pred_tree_train)
    test_r2 = r2_score(y_test, y_pred_tree_test)
    
    tree_results.append({
        'depth': depth if depth else 'None',
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_leaves': tree_model.get_n_leaves()
    })
    
    print(f"\nDepth = {depth if depth else 'Unlimited'}:")
    print(f"  Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"  Testing MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    print(f"  Number of leaves: {tree_model.get_n_leaves()}")
    print(f"  Overfitting gap (R²): {train_r2 - test_r2:.4f}")

# Best pruned tree (depth=5)
best_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
best_tree.fit(X_train, y_train)

y_pred_best_train = best_tree.predict(X_train)
y_pred_best_test = best_tree.predict(X_test)

print("\n" + "=" * 70)
print("BEST PRUNED TREE (max_depth=5, min_samples_split=10, min_samples_leaf=5)")
print("=" * 70)
print(f"Training MSE: {mean_squared_error(y_train, y_pred_best_train):.4f}")
print(f"Testing MSE: {mean_squared_error(y_test, y_pred_best_test):.4f}")
print(f"Training R²: {r2_score(y_train, y_pred_best_train):.4f}")
print(f"Testing R²: {r2_score(y_test, y_pred_best_test):.4f}")
print(f"Number of leaves: {best_tree.get_n_leaves()}")

# Feature importance
print(f"\nFeature Importance (Regression Tree):")
for feature, importance in zip(X.columns, best_tree.feature_importances_):
    print(f"  {feature}: {importance:.4f}")


# ============================================================================
# 3. Comparison and Analysis
# ============================================================================
print("\n" + "=" * 70)
print("3. MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Regression Tree (Pruned)'],
    'Training MSE': [lr_train_mse, mean_squared_error(y_train, y_pred_best_train)],
    'Testing MSE': [lr_test_mse, mean_squared_error(y_test, y_pred_best_test)],
    'Training R²': [lr_train_r2, r2_score(y_train, y_pred_best_train)],
    'Testing R²': [lr_test_r2, r2_score(y_test, y_pred_best_test)]
})

print("\n", comparison_df.to_string(index=False))

# Calculate overfitting gap
lr_gap = lr_train_r2 - lr_test_r2
tree_gap = r2_score(y_train, y_pred_best_train) - r2_score(y_test, y_pred_best_test)

print(f"\nOverfitting Analysis:")
print(f"  Linear Regression Gap: {lr_gap:.4f}")
print(f"  Regression Tree Gap: {tree_gap:.4f}")

if tree_gap > lr_gap:
    print(f"  → Regression tree shows more overfitting")
else:
    print(f"  → Linear regression shows more overfitting")

# ============================================================================
# 4. Discussion Points
# ============================================================================
print("\n" + "=" * 70)
print("4. KEY INSIGHTS")
print("=" * 70)

print("\nLinear Regression:")
print("  ✓ Simple and interpretable")
print("  ✓ Fast training and prediction")
print("  ✓ Stable across different data splits")
print("  ✗ Cannot capture non-linear relationships")
print("  ✗ May underfit complex patterns")

print("\nRegression Tree:")
print("  ✓ Captures non-linear relationships")
print("  ✓ Handles feature interactions automatically")
print("  ✓ No need for feature scaling")
print("  ✓ Interpretable decision rules")
print("  ✗ Prone to overfitting without pruning")
print("  ✗ Cannot extrapolate beyond training range")

print("\nPruning Importance:")
print("  • Prevents overfitting by limiting tree complexity")
print("  • max_depth controls vertical growth")
print("  • min_samples_split ensures sufficient data for splits")
print("  • min_samples_leaf prevents tiny, noisy leaf nodes")
print("  • Balance between bias (underfitting) and variance (overfitting)")

print("\n" + "=" * 70)
print("Yield prediction analysis completed!")
print("=" * 70)
