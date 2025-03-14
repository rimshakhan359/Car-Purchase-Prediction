import streamlit as st
import numpy as np
import joblib

# Load the trained model and scalers
regressor = joblib.load("car_model.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Streamlit UI
st.title("🚗 Car Purchase Prediction App")
st.write("Enter your details below to predict the estimated car purchase amount.")

# User input fields
gender = st.radio("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=80, step=1)
annual_salary = st.number_input("Annual Salary ($)", min_value=0, step=1000)
credit_card_debt = st.number_input("Credit Card Debt ($)", min_value=0, step=500)
net_worth = st.number_input("Net Worth ($)", min_value=0, step=1000)

# Convert gender to numeric
gender = 1 if gender == 'Male' else 0

# Predict button
if st.button("Predict Car Purchase Amount"):
    # Scale input data
    input_data = np.array([[gender, age, annual_salary, credit_card_debt, net_worth]])
    input_scaled = scaler_X.transform(input_data)
    
    # Predict & inverse scale
    prediction_scaled = regressor.predict(input_scaled)
    predicted_price = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    # Display result
    st.success(f"💰 Estimated Car Purchase Amount: **${predicted_price[0][0]:,.2f}**")
