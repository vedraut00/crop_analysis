"""
Crop Yield Prediction and Classification
Case Study 2 - Agricultural Data Analysis

DATA SOURCE NOTE:
This project uses synthetically generated data for educational purposes.
The data generation process is based on realistic agricultural relationships
and parameters derived from real-world agricultural research.

For similar real-world datasets, see:
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Crop+recommendation
- Kaggle: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
- FAO Statistics: http://www.fao.org/faostat/en/#data
- USDA NASS: https://www.nass.usda.gov/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic agricultural dataset
def generate_agricultural_data(n_samples=200):
    """Generate synthetic agricultural data for analysis"""
    
    # Features
    soil_moisture = np.random.uniform(20, 80, n_samples)  # percentage
    rainfall = np.random.uniform(500, 2000, n_samples)  # mm per year
    temperature = np.random.uniform(15, 35, n_samples)  # celsius
    fertilizer = np.random.uniform(50, 300, n_samples)  # kg per acre
    
    # Crop type (0: wheat, 1: rice)
    # Rice prefers higher moisture and rainfall
    crop_type = ((soil_moisture > 50) & (rainfall > 1200)).astype(int)
    
    # Yield calculation with nonlinear relationships
    base_yield = (
        0.02 * soil_moisture +
        0.001 * rainfall +
        0.05 * temperature +
        0.008 * fertilizer
    )
    
    # Add interaction effects and noise
    interaction = 0.0001 * soil_moisture * rainfall
    noise = np.random.normal(0, 0.5, n_samples)
    
    # Rice typically has higher yield
    crop_bonus = crop_type * 1.5
    
    yield_tons = base_yield + interaction + crop_bonus + noise
    yield_tons = np.clip(yield_tons, 2, 12)  # Realistic yield range
    
    # Create DataFrame
    df = pd.DataFrame({
        'soil_moisture': soil_moisture,
        'rainfall': rainfall,
        'temperature': temperature,
        'fertilizer': fertilizer,
        'crop_type': crop_type,
        'yield': yield_tons
    })
    
    return df

# Generate data
print("=" * 70)
print("CROP YIELD PREDICTION AND CLASSIFICATION ANALYSIS")
print("=" * 70)
print("\nGenerating agricultural dataset...")
df = generate_agricultural_data(200)

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset statistics:")
print(df.describe())

print(f"\nCrop type distribution:")
print(df['crop_type'].value_counts())
print(f"Wheat (0): {(df['crop_type'] == 0).sum()}")
print(f"Rice (1): {(df['crop_type'] == 1).sum()}")

# Save dataset
df.to_csv('agricultural_data.csv', index=False)
print("\nDataset saved to 'agricultural_data.csv'")
