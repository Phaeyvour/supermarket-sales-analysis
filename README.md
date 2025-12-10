# Superstore Sales Predictor

A machine learning web application that predicts sales for a superstore based on product details, customer segments, and order information.

## Features
- Predict sales based on quantity, price, discount, and other factors
- Interactive web interface built with Streamlit
- 83% prediction accuracy (R² = 0.83)
- Random Forest machine learning model

## How to Run Locally

1. Clone this repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Run the app:
```bash
   streamlit run app.py
```

## Model Performance
- **R² Score:** 0.83 (explains 83% of sales variation)
- **Mean Absolute Error:** $30
- **RMSE:** $316

## Key Insights
- Unit Price drives 77.7% of sales predictions
- Quantity is the second most important factor (14.2%)
- Discounts have moderate impact (4.4%)

## Technologies Used
- Python
- Scikit-learn (Random Forest)
- Streamlit
- Pandas
- Joblib

## Dataset
Sample Superstore dataset with 9,994 transactions
