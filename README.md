# Credit Card Fraud Detection Using Machine Learning
## 1. Project Overview
This project develops and evaluates machine learning models for detecting fraudulent credit card transactions. The main challenge is the severe class imbalance in the dataset, where fraudulent transactions represent only a very small proportion of all transactions.
<img width="515" height="388" alt="images" src="https://github.com/user-attachments/assets/3fef688b-7436-46af-a414-412d29ef564e" />

The project focuses on comparing different machine learning approaches, investigating techniques for handling class imbalance, and evaluating model performance using metrics that are appropriate for fraud detection.

## 2. Dataset
The dataset used in this project is the Credit Card Fraud Detection dataset provided through Kaggle.

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains:

284,807 transactions
31 variables
492 fraudulent transactions
284,315 legitimate transactions
Fraudulent transactions represent approximately 0.172% of all transactions.

## 3. Baseline Model
A baseline model is established before applying more advanced techniques.

The baseline provides a reference point for determining whether subsequent modelling and imbalance-handling strategies produce meaningful improvements.

The baseline model will initially use Logistic Regression.

## 4. Class Imbalance Handling
Because fraudulent transactions represent only approximately 0.172% of the dataset, different strategies will be investigated to address class imbalance.

Potential approaches include:

Class weighting
Random undersampling
Oversampling
Other appropriate imbalance-handling techniques

Resampling techniques will be applied only to the training data to prevent information from the test set from influencing the modelling process.

## 5. Machine Learning Models
The following models will be evaluated:

Logistic Regression
Random Forest
XGBoost

These models provide a progression from a relatively interpretable linear baseline to more flexible tree-based ensemble methods.

## 6. Key Findings
This section will be completed after the experiments.

The final analysis will summarize:

The best-performing model
The most effective imbalance-handling strategy
The final Precision, Recall, F1-score, ROC-AUC, and PR-AUC
The effect of hyperparameter tuning
The effect of classification threshold selection
The practical trade-off between detecting fraud and generating false alarms
