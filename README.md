# FraudDetectAI

**FraudDetectAI** is a machine learning project focused on detecting fraudulent financial transactions. The model achieves an impressive accuracy of **99%**, making it a reliable tool for financial institutions and organizations looking to enhance their fraud detection systems.
 
## Table of Contents

- [Overview](#overview)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Model Selection](#model-selection)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Key Predictive Factors](#key-predictive-factors)
- [How to Use](#how-to-use)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

Fraud detection is a critical task in the financial industry. This project employs a RandomForestClassifier to analyze transaction data and identify potential fraud. By cleaning the data, handling missing values, and engineering meaningful features, the model achieves high accuracy and robust performance.

## Data Cleaning and Preparation

The dataset used in this project includes features such as `Time`, `Transaction_Type`, `Amount`, `Origin_ID`, `Initial_Origin_Balance`, `Final_Origin_Balance`, `Destination_ID`, `Initial_Destination_Balance`, `Final_Destination_Balance`, `Fraud`, and `Expected_Fraud`.

### Handling Missing Values

- Missing values in `Initial_Destination_Balance` and `Final_Destination_Balance` were filled using the mean of the respective columns.
- The `Fraud` column's missing values were treated as non-fraudulent transactions.

### Outlier Treatment

- The `Amount` column outliers were handled using the Interquartile Range (IQR) method, ensuring that extreme values do not skew the model's predictions.

## Model Selection

A RandomForestClassifier was chosen for its robustness and ability to handle large datasets with high dimensionality. This model was trained on the processed dataset and evaluated using various performance metrics.

## Feature Engineering

New features were engineered to enhance the model's predictive power:
- `Origin_Amount_Ratio`: Ratio of the initial origin balance to the transaction amount.
- `Large_Transaction`: Binary feature indicating whether a transaction amount exceeds 200,000.
- `Hour` and `Day`: Extracted from the `Time` feature to capture temporal patterns.

## Model Performance

The model was evaluated on a test dataset, achieving an accuracy of **99%**. Below are the key performance metrics:

- **Confusion Matrix**
![Confusion Matrix](confusion_matrix.png)

- **ROC Curve**
![ROC Curve](roc_curve.png)

- **Feature Importance**
![Feature Importance](feature_importance.png)

**F1 Score**: 99.00%  
**ROC AUC Score**: 99.00%  
**Accuracy Score**: 99.00%

## Key Predictive Factors

The key factors that contribute to detecting fraudulent transactions include:
- Transaction amount relative to the origin balance.
- Large transactions that exceed typical thresholds.
- Temporal patterns, including the time of day and day of the week.

## Credits
 Durvank Gade for starting this project.
