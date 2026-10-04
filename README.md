# Week 2 - Supervised Machine Learning Models

## Project
Student Performance Prediction using Supervised Machine Learning

## Overview
This project demonstrates regression and classification using scikit-learn. The regression task predicts Final Score. The classification task predicts Performance Level (Pass or Needs Support) using student-related input features while excluding Final Score from the classification predictors.

## Models

### Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Classification
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

## Dataset
student_performance_week2.csv contains 120 records with Age, CGPA, Attendance Percentage, Absences, Study Hours per Day, Previous Score, Final Score, and Performance Level.

The dataset is included so the experiment can be reproduced without an external download.

## Evaluation
Regression metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

Classification metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Run
```bash
pip install -r requirements.txt
python train_models.py
```

The Word report contains the methodology, model comparison, results, observations, limitations, and conclusion.
