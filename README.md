Project Overview

This project builds a machine learning model to predict house prices based on key property features:

size_sqft – Total house size in square feet

bedrooms – Number of bedrooms

bathrooms – Number of bathrooms

The goal of this project is to develop a predictive model that estimates property prices accurately using structured data and regression techniques.

Objectives

Clean and prepare real estate data

Perform exploratory data analysis (EDA)

Engineer relevant features

Train and evaluate regression models

Save the trained model for deployment

Enable future price predictions using saved model artifacts

Features Used

The trained model uses the following input features:

Feature	Description
size_sqft	Property size in square feet
bedrooms	Number of bedrooms
bathrooms	Number of bathrooms

These features were selected because they have strong predictive influence on house prices.

Technologies Used

Python

Pandas (Data manipulation)

NumPy (Numerical computation)

Scikit-learn (Machine Learning)

Joblib (Model saving)

Matplotlib / Seaborn (Visualization)

Data Preprocessing

Handled missing values

Split dataset into training and testing sets

Scaled numerical features (if required)

2️⃣ Model Training

The project uses a regression model (e.g., Linear Regression or Random Forest Regressor).

Model Evaluation Metrics

The model was evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

These metrics help measure:

Prediction accuracy

Error magnitude

Model explanatory power

Model Saving

The trained model features were saved using:

import joblib
joblib.dump(model, "model.joblib")

The uploaded .joblib file contains:

['size_sqft', 'bedrooms', 'bathrooms']

This ensures consistency when making future predictions.

import pandas as pd

new_data = pd.DataFrame({
    "size_sqft": [2000],
    "bedrooms": [3],
    "bathrooms": [2]
})

prediction = model.predict(new_data)
print(prediction)

house-price-prediction/
│
├── data/
├── notebooks/
├── model.joblib
├── README.md
└── requirements.txt