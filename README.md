# Solvation Free Energy Prediction

A machine learning project focused on predicting solvation free energy of molecules using various ML models and analyzing their feature importance.

## Project Overview

This project aims to predict solvation free energy using molecular fingerprints as features. The current implementation includes an XGBoost Regressor with hyperparameter optimization and SHAP analysis for feature importance interpretation.

## Requirements

The project uses Poetry for dependency management. Main dependencies include:

- Python ^3.11
- RDKit ^2024.3.5
- XGBoost ^2.1.2
- Pandas ^2.2.3
- SHAP ^0.46.0
- Scikit-learn ^1.5.2
- Ax-platform ^0.4.3 (for hyperparameter optimization)

For a complete list of dependencies, see `pyproject.toml`.

## Project Structure

```
solvation-free-energy/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_fp.csv        # Training data with Morgan fingerprints
│   ├── test_fp.csv         # Test data with Morgan fingerprints
│   └── optimization_results/
├── notebooks/
│   ├── data_extraction_code.ipynb    # Morgan fingerprint generation
│   ├── xgbr_optimization_code.ipynb  # XGBoost optimization
│   └── shap_plot_code.ipynb         # Feature importance analysis
└── plots/
    └── bits/                        # Molecular fragments visualization
```

## Methodology

### 1. Feature Engineering
- Molecules are represented using Morgan fingerprints (ECFP4)
- Fingerprint size: 4096 bits
- Radius: 2

### 2. Model Development
Currently implemented:
- XGBoost Regressor with hyperparameter optimization using Ax Platform
- Optimized parameters:
  - learning_rate
  - max_depth
  - colsample_bytree
  - reg_alpha

### 3. Feature Importance Analysis
- SHAP (SHapley Additive exPlanations) values analysis
- Visualization of important molecular fragments using RDKit

## Results

### 1. Model tuning
![](/plots/XGB_r2vsiter.png)


## Future Work

- Implementation of additional models:
  - CatBoost
  - Other ML algorithms
- Comparative analysis of model performance
- Feature importance comparison across different models
- Ensemble methods exploration

## Installation

1. Clone the repository
2. Install Poetry if not already installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
3. Install dependencies:
```bash
poetry install
```

## Usage

The project is currently organized in Jupyter notebooks:
1. Run `data_extraction_code.ipynb` to generate molecular fingerprints
2. Run `xgbr_optimization_code.ipynb` for model optimization
3. Run `shap_plot_code.ipynb` for feature importance analysis
