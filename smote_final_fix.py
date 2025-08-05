# Final working solution for SMOTE boolean error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("23-24Season.csv")

# Only keep relevant columns and drop rows with missing values
df = df[['HomeTeam', 'AwayTeam', 'FTR']].dropna()

# Create a binary target: 1 = Home win (H), 0 = Not home win (D or A)
df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else 0)

# Drop the original FTR column
df = df.drop(columns=['FTR'])

# One-hot encode teams (drop_first=True to avoid multicollinearity)
df = pd.get_dummies(df, columns=['HomeTeam', 'AwayTeam'], drop_first=True)

# Define features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FIX: Convert boolean features to integers to avoid SMOTE error
X_train = X_train.astype(int)
X_test = X_test.astype(int)
y_train = y_train.astype(int)

print("Original training set shape:", X_train.shape, y_train.shape)
print("Original target distribution:")
print(pd.Series(y_train).value_counts())

# Now SMOTE should work without errors
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nResampled training set shape:", X_train_sm.shape, y_train_sm.shape)
print("Resampled target distribution:")
print(pd.Series(y_train_sm).value_counts())

# Train model on resampled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sm, y_train_sm)

# Predict on test set
y_pred = model.predict(X_test)

# Report and Confusion Matrix
print("\n🔍 Classification Report (with SMOTE):")
print(classification_report(y_test, y_pred))
print("🧮 Confusion Matrix (with SMOTE):")
print(confusion_matrix(y_test, y_pred))

# Compare with original model (without SMOTE)
print("\n" + "="*50)
print("COMPARISON WITH ORIGINAL MODEL (without SMOTE)")

# Train original model
model_original = RandomForestClassifier(n_estimators=100, random_state=42)
model_original.fit(X_train, y_train)

# Predict on test set
y_pred_original = model_original.predict(X_test)

print("\n🔍 Classification Report (without SMOTE):")
print(classification_report(y_test, y_pred_original))
print("🧮 Confusion Matrix (without SMOTE):")
print(confusion_matrix(y_test, y_pred_original)) 