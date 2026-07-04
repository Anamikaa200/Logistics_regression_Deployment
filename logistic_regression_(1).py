import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# Load the dataset directly from a public source
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/Exercise/HR_comma_sep.csv"
df = pd.read_csv(url)

print("--- Dataset Preview ---")
print(df.head())
print("\n--- Column Info & Missing Values ---")
print(df.info())

# ---------------------------------------------------------
# 1. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
# Let's check the average metrics for employees who left vs. stayed
print("\n--- Average Metrics by Retention (left) ---")
print(df.groupby('left').mean(numeric_only=True))

"""
EDA Insights from the averages:
1. Satisfaction Level: Significantly lower for employees who left (~0.44 vs ~0.66).
2. Average Monthly Hours: Higher for employees who left (~207 vs ~199).
3. Promotion Last 5 Years: Employees who stayed were more likely to have been promoted.
"""

# ---------------------------------------------------------
# 2. Bar chart: Impact of Salary on Retention
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(x='salary', hue='left', data=df, palette='Set2')
plt.title('Employee Retention by Salary Level')
plt.xlabel('Salary')
plt.ylabel('Number of Employees')
plt.legend(['Stayed', 'Left'])
plt.show()

# ---------------------------------------------------------
# 3. Bar chart: Correlation between Department and Retention
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.countplot(x='Department', hue='left', data=df, palette='viridis')
plt.title('Employee Retention by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.legend(['Stayed', 'Left'])
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. Feature Selection & Model Building
# ---------------------------------------------------------
# Based on EDA, we select impactful features:
# Quantitative: satisfaction_level, average_montly_hours, promotion_last_5years
# Categorical (to be encoded): salary, Department
X = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary', 'Department']]
y = df['left']

# Handle Categorical Variables using One-Hot Encoding via ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['salary', 'Department'])
    ],
    remainder='passthrough'
)

# Transform features
X_processed = preprocessor.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression Model
# Increasing max_iter ensures the solver converges
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Measure Accuracy
# ---------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
