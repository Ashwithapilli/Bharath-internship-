import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv(r"C:\Users\rajasree\Downloads\archive.zip")

# View dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())
# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()
# Distribution plots for continuous variables
for column in numeric_df.columns:
    sns.histplot(numeric_df[column])
    plt.title(f'Distribution of {column}')
    plt.show()
# Churn vs. Non-Churn bar charts
if 'churn' in df.columns:
    sns.countplot(x='churn', data=df)
    plt.show()
else:
    print("The 'churn' column does not exist in the DataFrame.")
# Define preprocessing steps
numeric_features = ['tenure', 'age', 'monthly_charges', 'total_charges']
categorical_features = ['gender', 'contract']

# Ensure that the specified columns exist in the DataFrame
for feature in numeric_features + categorical_features:
    if feature not in df.columns:
        print(f"The '{feature}' column does not exist in the DataFrame.")

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)# Split data into training and testing sets
if 'customerID' in df.columns and 'churn' in df.columns:
    X = df.drop(['customerID', 'churn'], axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    print("Either 'customerID' or 'churn' column does not exist in the DataFrame.")
