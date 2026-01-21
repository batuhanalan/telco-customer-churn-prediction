# Telco Customer Churn Prediction

This project aims to predict customer churn using machine learning models based on customer demographics, service usage, and billing information.

## Problem Definition
The goal is to build a classification model that can predict whether a customer will churn and evaluate model performance using multiple metrics.

## Dataset
- Telco Customer Churn Dataset
- 7,043 customers
- Demographic information, subscribed services, billing details, and churn status

⚠️ The dataset is not included in this repository due to size limitations.  
The file path in the script should be updated by the user.

## Exploratory Data Analysis
- Identification of numerical and categorical variables
- Data type corrections
- Missing value and outlier analysis
- Distribution analysis of variables
- Churn analysis across categorical features

## Feature Engineering
- Handling missing and outlier values
- New feature creation:
  - Total number of subscribed services
  - Average monthly charges
  - Tenure-based customer segments
- Label Encoding and One-Hot Encoding
- Feature scaling with RobustScaler

## Modeling
The following classification algorithms were trained and evaluated using cross-validation:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Model performance was evaluated using:
- Accuracy
- Recall
- Precision
- F1-score
- ROC-AUC

## Final Model
- CatBoost Classifier
- Hyperparameter optimization with GridSearchCV
- Final model achieved strong performance across all evaluation metrics

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM, CatBoost
