"""
Sales Prediction Project using Machine Learning
------------------------------------------------
This script loads the advertising dataset, performs exploratory data analysis,
builds and evaluates a regression model, and visualizes feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset
import os
# Load Dataset with error handling
DATA_PATH = os.path.join('data', 'advertising.csv')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure your CSV is in the 'data' folder.")
df = pd.read_csv(DATA_PATH)
print("\nFirst 5 rows of the dataset:")
print(df.head())
# Check required columns
required_cols = {'TV', 'Radio', 'Newspaper', 'Sales'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Dataset must have columns: {required_cols}. Found: {df.columns.tolist()}")

# 2. Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

try:
    sns.pairplot(df)
    plt.suptitle('Pairwise Relationships', y=1.02)
    plt.show()
except Exception as e:
    print(f"Pairplot could not be displayed: {e}")

try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
except Exception as e:
    print(f"Heatmap could not be displayed: {e}")

# 3. Feature Engineering (optional: create interaction features)
# Feature Engineering with error handling
try:
    df['TV_Radio'] = df['TV'] * df['Radio']
    df['TV_Newspaper'] = df['TV'] * df['Newspaper']
    df['Radio_Newspaper'] = df['Radio'] * df['Newspaper']
except Exception as e:
    print(f"Error in feature engineering: {e}")
    df['TV_Radio'] = df['TV_Radio'] if 'TV_Radio' in df else 0
    df['TV_Newspaper'] = df['TV_Newspaper'] if 'TV_Newspaper' in df else 0
    df['Radio_Newspaper'] = df['Radio_Newspaper'] if 'Radio_Newspaper' in df else 0

# 4. Train-Test Split
features = ['TV', 'Radio', 'Newspaper', 'TV_Radio', 'TV_Newspaper', 'Radio_Newspaper']
try:
    X = df[features]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    raise ValueError(f"Error in train-test split: {e}")

# 5. Model Training & Hyperparameter Tuning
rf = RandomForestRegressor(random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(rf, params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
try:
    grid.fit(X_train, y_train)
except Exception as e:
    raise ValueError(f"Error during Random Forest training or hyperparameter search: {e}")

print(f"\nBest Random Forest Params: {grid.best_params_}")
best_rf = grid.best_estimator_

# Also train a Linear Regression for comparison
lr = LinearRegression()
try:
    lr.fit(X_train, y_train)
except Exception as e:
    print(f"Linear Regression training failed: {e}")

# 6. Evaluation
for name, model in [('Random Forest', best_rf), ('Linear Regression', lr)]:
    try:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"\n{name} Results:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R^2 Score: {r2:.2f}")
    except Exception as e:
        print(f"Error in {name} evaluation: {e}")

# 7. Feature Importance (Random Forest)
try:
    importances = best_rf.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    feat_imp.plot(kind='bar', color='skyblue')
    plt.title('Feature Importance (Random Forest)')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Feature importance plot could not be displayed: {e}")

# 8. Residuals Analysis (for best model)
try:
    y_pred = best_rf.predict(X_test)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution (Random Forest)')
    plt.xlabel('Residuals')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Sales')
    plt.show()
except Exception as e:
    print(f"Residual analysis plots could not be displayed: {e}")

# 9. Save Model (optional)
import joblib
try:
    joblib.dump(best_rf, 'best_random_forest_model.pkl')
    print("\nBest Random Forest model saved as 'best_random_forest_model.pkl'.")
except Exception as e:
    print(f"Model could not be saved: {e}")

# 10. Predict on New Data (example)
try:
    example = pd.DataFrame({
        'TV': [150],
        'Radio': [30],
        'Newspaper': [20],
        'TV_Radio': [150*30],
        'TV_Newspaper': [150*20],
        'Radio_Newspaper': [30*20]
    })
    pred = best_rf.predict(example)
    print(f"\nPredicted Sales for example input: {pred[0]:.2f}")
except Exception as e:
    print(f"Example prediction failed: {e}")
