"""
Task 2: Crop Classification (8 marks)
- SVM with RBF kernel
- Logistic Regression
- Comparison and Decision Boundary Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('agricultural_data.csv')

# Prepare features and target
X = df[['soil_moisture', 'rainfall', 'temperature', 'fertilizer']]
y = df['crop_type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 70)
print("TASK 2: CROP CLASSIFICATION")
print("=" * 70)

# ============================================================================
# 1. Logistic Regression
# ============================================================================
print("\n" + "=" * 70)
print("1. LOGISTIC REGRESSION")
print("=" * 70)

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr_train = log_reg.predict(X_train_scaled)
y_pred_lr_test = log_reg.predict(X_test_scaled)

# Metrics
lr_train_acc = accuracy_score(y_train, y_pred_lr_train)
lr_test_acc = accuracy_score(y_test, y_pred_lr_test)

print(f"\nLogistic Regression Coefficients:")
for feature, coef in zip(X.columns, log_reg.coef_[0]):
    print(f"  {feature}: {coef:.6f}")
print(f"  Intercept: {log_reg.intercept_[0]:.6f}")

print(f"\nLogistic Regression Performance:")
print(f"  Training Accuracy: {lr_train_acc:.4f}")
print(f"  Testing Accuracy: {lr_test_acc:.4f}")

print(f"\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr_test, target_names=['Wheat', 'Rice']))

print(f"\nConfusion Matrix (Logistic Regression):")
cm_lr = confusion_matrix(y_test, y_pred_lr_test)
print(cm_lr)

# Cross-validation
cv_scores_lr = cross_val_score(log_reg, X_train_scaled, y_train, cv=5)
print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores_lr}")
print(f"Mean CV Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

# ============================================================================
# 2. SVM with RBF Kernel
# ============================================================================
print("\n" + "=" * 70)
print("2. SVM WITH RBF KERNEL")
print("=" * 70)

# Grid search for best hyperparameters
print("\nPerforming Grid Search for optimal SVM parameters...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

svm_grid = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {svm_grid.best_params_}")
print(f"Best cross-validation score: {svm_grid.best_score_:.4f}")

# Use best model
svm_model = svm_grid.best_estimator_

# Predictions
y_pred_svm_train = svm_model.predict(X_train_scaled)
y_pred_svm_test = svm_model.predict(X_test_scaled)

# Metrics
svm_train_acc = accuracy_score(y_train, y_pred_svm_train)
svm_test_acc = accuracy_score(y_test, y_pred_svm_test)

print(f"\nSVM Performance:")
print(f"  Training Accuracy: {svm_train_acc:.4f}")
print(f"  Testing Accuracy: {svm_test_acc:.4f}")
print(f"  Number of support vectors: {svm_model.n_support_}")

print(f"\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm_test, target_names=['Wheat', 'Rice']))

print(f"\nConfusion Matrix (SVM):")
cm_svm = confusion_matrix(y_test, y_pred_svm_test)
print(cm_svm)

# ============================================================================
# 3. Model Comparison
# ============================================================================
print("\n" + "=" * 70)
print("3. MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM (RBF)'],
    'Training Accuracy': [lr_train_acc, svm_train_acc],
    'Testing Accuracy': [lr_test_acc, svm_test_acc],
    'Overfitting Gap': [lr_train_acc - lr_test_acc, svm_train_acc - svm_test_acc]
})

print("\n", comparison_df.to_string(index=False))

# Detailed comparison
print(f"\nDetailed Analysis:")
print(f"  Logistic Regression:")
print(f"    - Simple linear decision boundary")
print(f"    - Fast training and prediction")
print(f"    - Interpretable coefficients")
print(f"    - Testing Accuracy: {lr_test_acc:.4f}")

print(f"\n  SVM (RBF Kernel):")
print(f"    - Non-linear decision boundary")
print(f"    - More complex model")
print(f"    - Better for non-linearly separable data")
print(f"    - Testing Accuracy: {svm_test_acc:.4f}")

if svm_test_acc > lr_test_acc:
    print(f"\n  Winner: SVM outperforms Logistic Regression by {(svm_test_acc - lr_test_acc)*100:.2f}%")
else:
    print(f"\n  Winner: Logistic Regression outperforms SVM by {(lr_test_acc - svm_test_acc)*100:.2f}%")

print("\n" + "=" * 70)
print("Classification analysis completed!")
print("=" * 70)
