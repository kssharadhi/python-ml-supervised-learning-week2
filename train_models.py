"""
Week 2 - Supervised Machine Learning Models
Project: Student Performance Prediction

The script trains regression and classification models using scikit-learn.
Input: student_performance_week2.csv
"""

import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

DATA_FILE = "student_performance_week2.csv"
df = pd.read_csv(DATA_FILE)

features = [
    "Age", "CGPA", "Attendance_Percentage",
    "Absences", "Study_Hours_Per_Day", "Previous_Score"
]

X = df[features]
y_reg = df["Final_Score"]
y_cls = df["Performance_Level"]

X_train, X_test, yreg_train, yreg_test, ycls_train, ycls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=4, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
}

print("\nREGRESSION RESULTS")
for name, model in reg_models.items():
    model.fit(X_train, yreg_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(yreg_test, pred)
    rmse = math.sqrt(mean_squared_error(yreg_test, pred))
    r2 = r2_score(yreg_test, pred)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

cls_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
}

print("\nCLASSIFICATION RESULTS")
for name, model in cls_models.items():
    model.fit(X_train, ycls_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(ycls_test, pred)
    precision = precision_score(ycls_test, pred, pos_label="Pass")
    recall = recall_score(ycls_test, pred, pos_label="Pass")
    f1 = f1_score(ycls_test, pred, pos_label="Pass")
    print(f"{name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
