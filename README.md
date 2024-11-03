# Solvation Free Energy Prediction

A machine learning project focused on predicting solvation free energy of molecules using various ML models and analyzing their feature importance.

## Project Overview

This project aims to predict solvation free energy using both molecular fingerprints and graph-based representations. Multiple machine learning models are implemented and optimized, including Graph Neural Networks (GCN), Random Forest (RF), XGBoost (XGB), and Feed-Forward Neural Networks (FFNN).

## Requirements

The project uses Poetry for dependency management. Main dependencies include:
- Python ^3.11
- PyTorch ^2.2.1
- RDKit ^2024.3.5
- XGBoost ^2.1.2
- Pandas ^2.2.3
- Scikit-learn ^1.5.2
- Ax-platform ^0.4.3 (for hyperparameter optimization)

For a complete list of dependencies, see `pyproject.toml`.

## Project Structure

```
solvation-free-energy/
├── data/
│   ├── SAMPL.csv
│   ├── train.csv
│   ├── test.csv
│   ├── train_fp.csv        # Training data with Morgan fingerprints
│   ├── test_fp.csv         # Test data with Morgan fingerprints
│   └── optimization_results/
│       ├── GCN_optimization.csv
│       ├── RFR_optimization.csv
│       ├── XGB_optimization.csv
│       └── FFNN_optimization.csv
├── notebooks/
│   ├── EDA.ipynb                    # Exploratory Data Analysis
│   ├── gcn_optimization.ipynb       # Graph Neural Network implementation
│   ├── rfr_optimization.ipynb       # Random Forest optimization
│   ├── xgbr_optimization.ipynb      # XGBoost optimization
│   └── ffnn_optimization.ipynb      # Feed Forward Neural Network optimization
└── plots/
    └── optimization_plots/          # Model optimization visualizations
```

## Methodology

### 1. Data Preprocessing

- Initial EDA shows the distribution of molecular properties and baseline performance
- Two main approaches for molecular representation:
  - Morgan fingerprints (ECFP4) for traditional ML models
  - Graph representation for GCN

### 2. Model Development

Multiple models are implemented and optimized:

#### Graph Convolutional Network (GCN)
- Custom implementation with configurable architecture
- Features:
  - Convolution layers with adjacency matrix normalization
  - Pooling layers
  - Dropout for regularization
  - Batch normalization
- Optimized parameters:
  - hidden_nodes
  - n_conv_layers
  - n_hidden_layers
  - learning_rate

#### Random Forest Regressor
- Optimized parameters:
  - n_estimators (100-1000)
  - max_depth (10-300)
  - max_features ('sqrt', 'log2')
  - min_samples_leaf (0.001-0.5)

#### XGBoost Regressor
- Optimized parameters:
  - learning_rate (0.001-1)
  - max_depth (1-6)
  - colsample_bytree (0-1)
  - reg_alpha (1e-6-10)

#### Feed Forward Neural Network
- Architecture:
  - Multiple hidden layers with ReLU activation
  - Batch normalization
  - Dropout regularization
- Optimized parameters:
  - learning_rate
  - patience (for early stopping)
  - dropout_rate

### 3. Hyperparameter Optimization

All models use Bayesian optimization via Ax Platform for hyperparameter tuning with:
- RMSE as the optimization metric
- 20-25 trials per model
- Train/test split of 80/20

## Results

Performance metrics and optimization results for each model are stored in `data/optimization_results/`. Each model's optimization process includes:
- Optimization traces showing RMSE improvement over trials
- Feature importance analysis where applicable
- Contour plots showing parameter interactions
- Final optimized parameters and performance metrics

### Dataset EDA

The dataset is comprised of over 600 compounds with it's experimentally determined solvation free energy. Additionally, each sample has it's calculated solvation free energy using molecular dynamics. For details of the dataset, refer to the publication ![Mobley, D., Guthrie, J. P. J Comput Aided Mol Des. 2014](https://pubmed.ncbi.nlm.nih.gov/24928188/)  

The dataset was split into a training and test set. The test set was only used for final model comparison and wasn't used for the training. For comparison, the values of calculated and predicted solvation energy are shown below.  

|                  | Samples | MAE  | RMSE | R²   |
|------------------|---------|------|------|------|
| Complete dataset | 642     | 1.11 | 1.54 | 0.84 |
| Train split      | 514     | 1.10 | 1.5  | 0.85 |
| Test split       | 128     | 1.18 | 1.71 | 0.79 |  

The calculated fingerprints were used to calculate the similarity between the training and test dataset:

Cosine similarity: 0.9658  
Wasserstein distance: 38.1324  
jenJensen–Shannon divergence: 0.1843  
Bhattacharyya coefficient: 0.8284  
Hellinger distance: 0.4143  


### 2. Model comparison

TODO: Update

### 3. Feature importance

Using SHAP analysis, several features were found to impact the solvation free energy. Polar groups contributed the most towards negative values as seen here:

![](/plots/XGB_features.png) 
Feature importance with corresponding Morgan fingerprints. 
 
![](/plots/XGB_shap_dist.png) 
Beeswarm plot of the top features. Since features are binary High = 1, Low = 0. 

## Future Work

- Implementation of additional models:
  - CatBoost
  - Other ML algorithms
- Comparative analysis of model performance
- Feature importance comparison across different models
- Ensemble methods exploration

## Usage

1. Clone the repository
2. Install Poetry if not already installed:
```bash
pip install poetry
```
3. Install dependencies:
```bash
poetry install
```
4. Run notebooks in the following order:
   - EDA.ipynb for initial data analysis
   - Individual model optimization notebooks based on needs

## Future Work

- Integration of models into an ensemble
- Comparative analysis of all implemented models
- Implementation of additional architectures
- Cross-validation for more robust performance estimates
- Web interface for predictions
