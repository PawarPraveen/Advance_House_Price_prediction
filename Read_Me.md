# 🏡 House Price Prediction using Gradient Boosting & SHAP

This project builds a robust machine learning model to predict house prices using advanced regression techniques and feature importance analysis. It incorporates preprocessing, log-transformation, outlier handling, and interpretable modeling using SHAP and Gradient Boosting.

---

## 📊 Model Performance

| Metric               | Value                     |
|----------------------|---------------------------|
|    RMSE              | 26,745.21                 |
|   R² Score           | 0.9067                    |
|   Model              | GradientBoostingRegressor |

---

## 🔧 Features

- ✅ Feature selection using SHAP values
- ✅ Log transformation of skewed inputs and target (`SalePrice`)
- ✅ Handling of missing values via imputation
- ✅ Outlier detection using Z-score filtering
- ✅ One-hot encoding for categorical features
- ✅ Model evaluation with train-test split and cross-validation
- ✅ SHAP summary plots for interpretability
- ✅ XGBoost and Random Forest pipelines as alternatives

---

## 📁 Dataset

- Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Target Variable: `SalePrice` (House price in USD)
- Features: 70+ including:
  - Lot area, garage size, year built
  - Neighborhood, zoning, exterior condition
  - Basement quality, kitchen features, etc.

---

## ⚙️ Tech Stack

- Language: Python
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn` for pipelines and models
  - `xgboost`, `shap` for boosting & interpretation

---

## 📌 Project Steps

1. EDA & Preprocessing
   - Handle missing values
   - Detect and remove outliers
   - Apply `log1p` transformation on `SalePrice` and skewed features

2. Modeling
   - Use `GradientBoostingRegressor` for training
   - Compare performance with `LinearRegression` and `RandomForestRegressor`

3. Interpretation
   - Use SHAP for model explainability
   - Identify and drop low-impact features automatically

4. Evaluation
   - RMSE and R² Score on test set
   - SHAP plots to analyze prediction drivers

---

## 📉 SHAP Output

- SHAP values highlight the most influential features:
  - `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `Neighborhood`
  - Pool quality, kitchen quality, and condition

![SHAP Summary Plot](path_to_your_plot.png)

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
jupyter notebook
