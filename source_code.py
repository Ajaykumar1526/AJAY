import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("exams.csv")


# Data Preprocessing & Cleaning
# Check for missing values
print(df.isnull().sum())

# Rename columns for easier access
df.columns = df.columns.str.replace(" ", "_")

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# EDA & Feature Engineering
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Feature: Average score
df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)


# Model Development & Evaluation
# Features and target
X = df.drop(['average_score'], axis=1)
y = df['average_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# Documentation & Visualization
# Coefficients
coeffs = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=coeffs, x="Coefficient", y="Feature")
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.show()