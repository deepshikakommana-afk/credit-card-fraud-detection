import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("💳 Smart Credit Card Fraud Detection")

st.write("Enter transaction details:")

# Inputs
age = st.number_input("User Age", min_value=10, max_value=100, value=25)
amount = st.number_input("Transaction Amount", value=100.0)
time = st.number_input("Transaction Time (0–24 hrs)", min_value=0.0, max_value=24.0, value=12.0)

if st.button("Check Transaction"):

    # Default normal pattern
    v_features = [0]*28

    # 🚨 Rule 1: Minor restriction
    if age < 18 and amount > 5000:
        v_features = [-5]*28

    # 🚨 Rule 2: Very high amount
    elif amount > 100000:
        v_features = [-5]*28

    # 🚨 Rule 3: Night time suspicious (12 AM – 5 AM)
    elif time >= 0 and time <= 5 and amount > 10000:
        v_features = [-5]*28

    # Combine features
    features = [time] + v_features + [amount]
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")