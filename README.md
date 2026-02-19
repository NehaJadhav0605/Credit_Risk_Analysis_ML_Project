# 📊 Credit Risk Analysis using Machine Learning

## 📌 Project Overview

- This project predicts whether a customer will default on a loan or not.
- It helps banks identify risky applicants before approving loans.
- Multiple Machine Learning models were trained and compared.
- Class imbalance was handled using SMOTE and dimensionality reduction using PCA.

## 🎯 Problem Statement

Loan defaults lead to major losses for banks and NBFCs.

This project builds classification models to:

- Identify defaulters (Class 1)
- Identify non-defaulters (Class 0)
- Handle class imbalance using SMOTE
- Improve model performance using PCA and ensemble methods

### Features in Dataset

- Age
- Income
- Loan Amount
- Credit Score
- Months Employed
- Number of Credit Lines
- Interest Rate
- Loan Term
- Debt-to-Income Ratio
- Education
- Employment Type
- Marital Status
- Loan Purpose
- Has Mortgage
- HasDependents
- HasCo-Signer

### Target Variable: Default

- 0 → Non-Defaulter
- 1 → Defaulter

## ⚙️ Project Workflow:

### 1️⃣ Exploratory Data Analysis (EDA)

- Checked missing values
- Studied feature distributions
- Checked relation between features and target variables
- Checked correlation between variables
- Observed class imbalance

Dataset was already clean, so no major data cleaning was required.

### 2️⃣ Data Preprocessing

- Dropped unnecessary column (Loan ID)
- Converted binary categorical columns to numeric
- Applied One-Hot Encoding to categorical features
- Scaled numeric features using StandardScaler

### 3️⃣ Train-Test Split

- 80% Training Data
- 20% Testing Data

### 4️⃣ Handling Class Imbalance

The dataset had fewer defaulters.

To solve this:
- Applied SMOTE on training data only
- Balanced both Dimensionality Reduction

### 5️⃣ Dimensionality Reduction

- Applied PCA
- Kept 95% variance
- Reduced redundant information
- Helped improve model generalization classes

## Machine Learning Models Used:

1] Logistic Regression
2] Decision Tree
3] Random Forest
4] Bagging Classifier
5] AdaBoost
6] Gradient Boosting
7] XGBoost

## Model Evaluation:

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Cross-Validation

Special focus was on predicting defaulters correctly.

## Key Observations:

- Ensemble models performed better than simple models.
- Random Forest and XGBoost gave the best performance.
- Predicting defaulters is harder than predicting non-defaulters.
- PCA reduced dimensions without major accuracy loss.
- SMOTE helped balance the dataset and improve recall for Class 1.

## Technologies Used:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost

## Final Results (After PCA):

| Model               | Train Accuracy | Test Accuracy |
| ------------------- | -------------- | ------------- |
| Logistic Regression | 0.69           | 0.68          |
| Decision Tree       | 1.00           | 0.77          |
| Random Forest       | 1.00           | 0.87          |
| Bagging Classifier  | 1.00           | 0.86          |
| AdaBoost            | 0.76           | 0.73          |
| Gradient Boosting   | 0.78           | 0.77          |
| XGBoost             | 0.90           | **0.87**      |

XGBoost performed best.

## 📌 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Try advanced resampling (SMOTEENN, Tomek Links)
- Deployment using Streamlit or Flask

## Dataset Information
### Dataset Link: https://www.kaggle.com/datasets/nikhil1e9/loan-default

## 👩‍💻 Author

### Neha Jadhav

Aspiring Data Analyst / Data Scientist
