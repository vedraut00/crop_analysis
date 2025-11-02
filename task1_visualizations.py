"""
Task 1: Visualizations for Yield Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('agricultural_data.csv')
X = df[['soil_moisture', 'rainfall', 'temperature', 'fertilizer']]
y = df['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

tree_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

# ============================================================================
# Visualization 1: Actual vs Predicted
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression
axes[0].scatter(y_test, y_pred_lr, alpha=0.6, edgecolors='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Yield (tons/acre)', fontsize=11)
axes[0].set_ylabel('Predicted Yield (tons/acre)', fontsize=11)
axes[0].set_title(f'Linear Regression\nR² = {r2_score(y_test, y_pred_lr):.4f}', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Regression Tree
axes[1].scatter(y_test, y_pred_tree, alpha=0.6, edgecolors='k', color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Yield (tons/acre)', fontsize=11)
axes[1].set_ylabel('Predicted Yield (tons/acre)', fontsize=11)
axes[1].set_title(f'Regression Tree (Pruned)\nR² = {r2_score(y_test, y_pred_tree):.4f}', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("Saved: task1_actual_vs_predicted.png")
plt.close()

# ============================================================================
# Visualization 2: Residual Plots
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

residuals_lr = y_test - y_pred_lr
residuals_tree = y_test - y_pred_tree

axes[0].scatter(y_pred_lr, residuals_lr, alpha=0.6, edgecolors='k')
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Yield (tons/acre)', fontsize=11)
axes[0].set_ylabel('Residuals', fontsize=11)
axes[0].set_title('Linear Regression - Residual Plot', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_pred_tree, residuals_tree, alpha=0.6, edgecolors='k', color='green')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Yield (tons/acre)', fontsize=11)
axes[1].set_ylabel('Residuals', fontsize=11)
axes[1].set_title('Regression Tree - Residual Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_residual_plots.png', dpi=300, bbox_inches='tight')
print("Saved: task1_residual_plots.png")
plt.close()

# ============================================================================
# Visualization 3: Overfitting Analysis
# ============================================================================
depths = [2, 3, 5, 10, 15, 20]
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(r2_score(y_train, tree.predict(X_train)))
    test_scores.append(r2_score(y_test, tree.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training R²', linewidth=2, markersize=8)
plt.plot(depths, test_scores, 's-', label='Testing R²', linewidth=2, markersize=8)
plt.xlabel('Tree Depth', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Overfitting Analysis: Tree Depth vs Performance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task1_overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: task1_overfitting_analysis.png")
plt.close()

# ============================================================================
# Visualization 4: Decision Tree Structure
# ============================================================================
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title('Regression Tree Structure (max_depth=5)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('task1_tree_structure.png', dpi=300, bbox_inches='tight')
print("Saved: task1_tree_structure.png")
plt.close()

# ============================================================================
# Visualization 5: Feature Importance Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression Coefficients
coef_abs = np.abs(lr_model.coef_)
axes[0].barh(X.columns, coef_abs, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Absolute Coefficient Value', fontsize=11)
axes[0].set_title('Linear Regression - Feature Coefficients', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Tree Feature Importance
axes[1].barh(X.columns, tree_model.feature_importances_, color='forestgreen', edgecolor='black')
axes[1].set_xlabel('Importance', fontsize=11)
axes[1].set_title('Regression Tree - Feature Importance', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('task1_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: task1_feature_importance.png")
plt.close()

print("\nAll Task 1 visualizations completed!")
