# Quick Start Guide

## Installation

1. Make sure you have Python 3.7+ installed
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Run Everything at Once
```bash
python run_all.py
```

This will:
- Generate the agricultural dataset
- Run yield prediction analysis (Task 1)
- Create yield prediction visualizations
- Run crop classification analysis (Task 2)
- Create classification visualizations

### Option 2: Run Individual Components

```bash
# Step 1: Generate data
python crop_analysis.py

# Step 2: Yield prediction
python task1_yield_prediction.py
python task1_visualizations.py

# Step 3: Crop classification
python task2_crop_classification.py
python task2_visualizations.py
```

## Output Files

After running, you'll get:

### Data
- `agricultural_data.csv` - Generated dataset with 200 samples

### Task 1 Visualizations (Yield Prediction)
- `task1_actual_vs_predicted.png` - Comparison of predictions
- `task1_residual_plots.png` - Error analysis
- `task1_overfitting_analysis.png` - Tree depth vs performance
- `task1_tree_structure.png` - Visual tree structure
- `task1_feature_importance.png` - Feature importance comparison

### Task 2 Visualizations (Crop Classification)
- `task2_confusion_matrices.png` - Classification accuracy
- `task2_decision_boundaries.png` - 2D decision boundaries
- `task2_model_comparison.png` - Model performance comparison
- `task2_feature_distributions.png` - Feature distributions by crop

## Documentation

- `DOCUMENTATION.md` - Complete analysis report with references
- `README.md` - Project overview
- This file - Quick start guide

## Submission Package

For submission, include:
1. All Python scripts (`.py` files)
2. `DOCUMENTATION.md` - Your main report
3. All generated visualizations (`.png` files)
4. `agricultural_data.csv` - The dataset
5. `requirements.txt` - Dependencies

## Troubleshooting

### Import Errors
```bash
pip install --upgrade numpy pandas matplotlib seaborn scikit-learn
```

### Visualization Issues
If plots don't show, they're saved as PNG files in the project directory.

### Memory Issues
Reduce dataset size in `crop_analysis.py`:
```python
df = generate_agricultural_data(100)  # Instead of 200
```

## Customization

### Change Dataset Size
Edit `crop_analysis.py`:
```python
df = generate_agricultural_data(500)  # More samples
```

### Adjust Model Parameters
Edit respective task files to tune hyperparameters.

### Add More Visualizations
Extend visualization scripts with your own plots.

## Questions?

Check `DOCUMENTATION.md` for detailed explanations of:
- Methodology
- Results interpretation
- Model comparisons
- References

Good luck with your submission!
