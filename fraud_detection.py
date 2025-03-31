import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import os

pd.set_option('display.max_columns', None)

# Load data
df = pd.read_csv('phone_data.csv', sep=';')

# Initial Exploration
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Missing Values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Categorical Variables
categorical_cols = ['CUST_LEVEL', 'GENDER_ENCR', 'INSTALMENT_IND', 'PRODUCT_NAME', 
                   'MANUFACTURER_NAME_EN', 'OPERATING_SYSTEM', 'HANDSET_FEATURE_CAT_DESC',
                   'TARIFF_LEVEL', 'CHANNEL_CLASS', 'channel_group', 'FRAUD_STATUS_6MONTH',
                   'nopay_after_12month']
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())
    print(f"Value counts for {col}:")
    print(df[col].value_counts())

# Handle '?' Values
print("\nBefore replacing '?' with NaN:")
print(df.isin(['?']).sum())
df.replace('?', np.nan, inplace=True)
print("\nAfter replacing '?' with NaN:")
print(df.isnull().sum())

# Convert Numeric Columns
numeric_cols = ['MOVING_AVERAGE_PRICE_AMT_ENCR', 'SELLING_PRICE_AMT_ENCR', 
                'UPFRONT_PYM_AMT_ENCR', 'monthly_fee_ENCR']
for col in numeric_cols:
    df[col] = df[col].str.replace(',', '.').astype(float)

print("\nStatistics for numerical columns after conversion:")
print(df[numeric_cols].describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x='CUST_LEVEL', data=df)
plt.title('Distribution of Customer Levels')
plt.xlabel('Customer Level')
plt.ylabel('Count')
plt.savefig('plots/customer_level_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df['R_AGE_Y'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('plots/age_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='CUST_LEVEL', hue='FRAUD_STATUS_6MONTH', data=df)
plt.title('Fraud Status by Customer Level')
plt.xlabel('Customer Level')
plt.ylabel('Count')
plt.savefig('plots/fraud_by_customer_level.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(x='MANUFACTURER_NAME_EN', y='SELLING_PRICE_AMT_ENCR', data=df)
plt.title('Selling Price by Manufacturer')
plt.xlabel('Manufacturer')
plt.ylabel('Selling Price')
plt.xticks(rotation=45)
plt.savefig('plots/selling_price_by_manufacturer.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Variables')
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Fraud and Non-Payment Patterns
print("\nFraud Status Distribution:")
print(df['FRAUD_STATUS_6MONTH'].value_counts(normalize=True))
print("\nNon-payment after 12 months Distribution:")
print(df['nopay_after_12month'].value_counts(normalize=True))

print("\nFraud Status by Tariff Level:")
print(pd.crosstab(df['TARIFF_LEVEL'], df['FRAUD_STATUS_6MONTH'], normalize='index'))

# Predictive Modeling
df['FRAUD_STATUS_6MONTH'] = df['FRAUD_STATUS_6MONTH'].map({'Y': 1, 'N': 0})
features = ['CUST_LEVEL', 'GENDER_ENCR', 'R_AGE_Y', 'INSTALMENT_IND', 'INSTAL_CNT',
            'PRODUCT_NAME', 'MANUFACTURER_NAME_EN', 'OPERATING_SYSTEM', 
            'HANDSET_FEATURE_CAT_DESC', 'MOVING_AVERAGE_PRICE_AMT_ENCR', 
            'SELLING_PRICE_AMT_ENCR', 'UPFRONT_PYM_AMT_ENCR', 'monthly_fee_ENCR', 
            'TARIFF_LEVEL', 'CHANNEL_CLASS', 'channel_group']
X = df[features]
y = df['FRAUD_STATUS_6MONTH']

le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown')
    X[col] = le.fit_transform(X[col])
for col in X.select_dtypes(include=['float64', 'int64']).columns:
    X[col] = X[col].fillna(X[col].median())
y = y.dropna()
X = X.loc[y.index]

print("\nClass distribution before SMOTE:")
print(y.value_counts(normalize=True))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\nEvaluation of {model_name}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:")
    print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nRandom Forest Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Random Forest Model')
plt.savefig('plots/feature_importance_rf.png')
plt.close()