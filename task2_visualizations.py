"""
Task 2: Visualizations for Crop Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Load data
df = pd.read_csv('agricultural_data.csv')
X = df[['soil_moisture', 'rainfall', 'temperature', 'fertilizer']]
y = df['crop_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_svm = svm_model.predict(X_test_scaled)

# ============================================================================
# Visualization 1: Confusion Matrices
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_svm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Wheat', 'Rice'], yticklabels=['Wheat', 'Rice'])
axes[0].set_title('Logistic Regression\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Wheat', 'Rice'], yticklabels=['Wheat', 'Rice'])
axes[1].set_title('SVM (RBF Kernel)\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('task2_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: task2_confusion_matrices.png")
plt.close()

# ============================================================================
# Visualization 2: Decision Boundaries (2D projections)
# ============================================================================
def plot_decision_boundary_2d(model, X_train, y_train, feature_idx, feature_names, title, ax):
    """Plot 2D decision boundary for two features"""
    X_subset = X_train[:, feature_idx]
    
    h = 0.02
    x_min, x_max = X_subset[:, 0].min() - 0.5, X_subset[:, 0].max() + 0.5
    y_min, y_max = X_subset[:, 1].min() - 0.5, X_subset[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create full feature array with mean values for other features
    Z_input = np.zeros((xx.ravel().shape[0], X_train.shape[1]))
    for i in range(X_train.shape[1]):
        if i == feature_idx[0]:
            Z_input[:, i] = xx.ravel()
        elif i == feature_idx[1]:
            Z_input[:, i] = yy.ravel()
        else:
            Z_input[:, i] = X_train[:, i].mean()
    
    Z = model.predict(Z_input)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
    scatter = ax.scatter(X_subset[:, 0], X_subset[:, 1], c=y_train, 
                        cmap=ListedColormap(['#FF0000', '#00FF00']), 
                        edgecolors='k', s=50, alpha=0.7)
    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(*scatter.legend_elements(), labels=['Wheat', 'Rice'], loc='best')

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Soil Moisture vs Rainfall
plot_decision_boundary_2d(log_reg, X_train_scaled, y_train, [0, 1], 
                          ['Soil Moisture (scaled)', 'Rainfall (scaled)'],
                          'Logistic Regression: Soil Moisture vs Rainfall', axes[0, 0])

plot_decision_boundary_2d(svm_model, X_train_scaled, y_train, [0, 1],
                          ['Soil Moisture (scaled)', 'Rainfall (scaled)'],
                          'SVM (RBF): Soil Moisture vs Rainfall', axes[0, 1])

# Temperature vs Fertilizer
plot_decision_boundary_2d(log_reg, X_train_scaled, y_train, [2, 3],
                          ['Temperature (scaled)', 'Fertilizer (scaled)'],
                          'Logistic Regression: Temperature vs Fertilizer', axes[1, 0])

plot_decision_boundary_2d(svm_model, X_train_scaled, y_train, [2, 3],
                          ['Temperature (scaled)', 'Fertilizer (scaled)'],
                          'SVM (RBF): Temperature vs Fertilizer', axes[1, 1])

plt.tight_layout()
plt.savefig('task2_decision_boundaries.png', dpi=300, bbox_inches='tight')
print("Saved: task2_decision_boundaries.png")
plt.close()

# ============================================================================
# Visualization 3: Model Comparison
# ============================================================================
from sklearn.metrics import accuracy_score

models = ['Logistic\nRegression', 'SVM\n(RBF)']
train_acc = [accuracy_score(y_train, log_reg.predict(X_train_scaled)),
             accuracy_score(y_train, svm_model.predict(X_train_scaled))]
test_acc = [accuracy_score(y_test, y_pred_lr),
            accuracy_score(y_test, y_pred_svm)]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, test_acc, width, label='Testing Accuracy', color='coral', edgecolor='black')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Classification Model Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.7, 1.0])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('task2_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: task2_model_comparison.png")
plt.close()

# ============================================================================
# Visualization 4: Feature Distribution by Crop Type
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = ['soil_moisture', 'rainfall', 'temperature', 'fertilizer']
titles = ['Soil Moisture Distribution', 'Rainfall Distribution', 
          'Temperature Distribution', 'Fertilizer Usage Distribution']

for idx, (feature, title) in enumerate(zip(features, titles)):
    ax = axes[idx // 2, idx % 2]
    df[df['crop_type'] == 0][feature].hist(bins=20, alpha=0.6, label='Wheat', 
                                             color='orange', edgecolor='black', ax=ax)
    df[df['crop_type'] == 1][feature].hist(bins=20, alpha=0.6, label='Rice', 
                                             color='green', edgecolor='black', ax=ax)
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task2_feature_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: task2_feature_distributions.png")
plt.close()

print("\nAll Task 2 visualizations completed!")
