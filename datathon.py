import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'M:\assignments\datathon\DATATHON_Test.csv')

# Initial Data Exploration
print(df.info())
print(df.describe())
print(df.head())

# Handle missing values in 'Initial_Destination_Balance' and 'Final_Destination_Balance'
df['Initial_Destination_Balance'] = df['Initial_Destination_Balance'].fillna(df['Initial_Destination_Balance'].mean())
df['Final_Destination_Balance'] = df['Final_Destination_Balance'].fillna(df['Final_Destination_Balance'].mean())

# Debug: Check if there are any remaining NaN values
print("NaN values after filling balances:")
print(df.isna().sum())

# Convert categorical target variable 'Fraud' to numeric (0 and 1)
df['Fraud'] = df['Fraud'].map({'No': 0, 'Yes': 1})

# Handle NaN values in 'Fraud' and other columns if necessary
df['Fraud'] = df['Fraud'].fillna(0)  # Assuming you want to treat NaNs as non-fraudulent

# Debug: Check for any NaN values remaining in 'Fraud'
print("Number of NaN values in 'Fraud' after conversion:", df['Fraud'].isna().sum())

# Handle outliers in the 'Amount' column
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Amount'] = np.where(df['Amount'] > upper_bound, upper_bound, 
                        np.where(df['Amount'] < lower_bound, lower_bound, df['Amount']))

# Convert all columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Debug: Check if coercion introduced NaN values
print("NaN values after coercion to numeric:")
print(df.isna().sum())

# Plot the correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Encode categorical variables
df['Transaction_Type'] = LabelEncoder().fit_transform(df['Transaction_Type'])

# Feature Engineering
df['Origin_Amount_Ratio'] = df['Initial_Origin_Balance'] / df['Amount']
df['Large_Transaction'] = np.where(df['Amount'] > 200000, 1, 0)
df['Hour'] = df['Time'] % 24
df['Day'] = df['Time'] // 24

# Debug: Check the new features
print("New Features Preview:")
print(df[['Origin_Amount_Ratio', 'Large_Transaction', 'Hour', 'Day']].head())

# Prepare features and target variable
X = df.drop(['Fraud', 'Expected_Fraud', 'Origin_ID', 'Destination_ID', 'Time'], axis=1)
y = df['Fraud']

# Check for NaN values in y and handle them
print(f"Number of NaN values in 'y': {y.isna().sum()}")

# Drop rows with NaN values in X and y
X = X[y.notna()]
y = y.dropna()

# Check the shape of X and y after handling NaN values
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Ensure there are samples left for training
if X.shape[0] > 0 and y.shape[0] > 0:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train the RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict the test set results
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred)*100)
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred)*100)
    print("Accuracy Score:", accuracy_score(y_test, y_pred)*100)  # Print the accuracy score

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot ROC Curve
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Feature importance plot
    importances = rf_model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.show()

else:
    print("No data left :)")
