import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
anomaly_model = pickle.load(open("anomaly.pkl", "rb"))

# Load dataset
df = pd.read_csv("creditcard.csv")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Intelligent Credit Card Fraud Detection System")

# ---------------- INPUT ----------------
st.sidebar.header("Transaction Input")

age = st.sidebar.number_input("User Age", 10, 100, 25)
amount = st.sidebar.number_input("Transaction Amount", value=1000.0)
time = st.sidebar.number_input("Transaction Time (0–24)", 0.0, 24.0, 12.0)

# ---------------- ANALYZE ----------------
if st.sidebar.button("Analyze Transaction"):

    account_type = "minor" if age < 18 else "adult"

    # ---------------- BANK POLICY ----------------
    if account_type == "minor":
        if amount > 10000:
            st.error("🚨 BLOCK - Minor exceeds limit")
            st.stop()
        elif amount > 2000:
            st.warning("⚠️ VERIFY - Parent approval required")

    if account_type == "adult":
        if amount > 200000:
            st.error("🚨 BLOCK - Extremely high transaction")
            st.stop()
        elif amount > 100000:
            st.warning("⚠️ VERIFY - OTP required")

    # ---------------- SCALE ----------------
    scaled = scaler.transform([[amount, time]])
    amount_scaled, time_scaled = scaled[0]

    v_features = np.random.normal(0, 1, 28)

    risk_score = 0
    reasons = []

    # ---------------- RULES ----------------
    if 0 <= time <= 5:
        risk_score += 0.2
        reasons.append("Odd transaction time")

    # ---------------- BEHAVIOR ----------------
    if len(st.session_state.history) > 3:
        avg = np.mean(st.session_state.history)
        if amount > avg * 3:
            risk_score += 0.3
            reasons.append("Unusual spending behavior")

    # ---------------- ML ----------------
    features = [time_scaled] + list(v_features) + [amount_scaled]
    features = np.array(features).reshape(1, -1)

    prob = model.predict_proba(features)[0][1]

    # ---------------- ANOMALY ----------------
    if anomaly_model.predict(features)[0] == -1:
        risk_score += 0.3
        reasons.append("Anomalous transaction")

    final_score = min(prob + risk_score, 1.0)

    st.session_state.history.append(amount)

    # ---------------- OUTPUT ----------------
    st.subheader(f"💳 Risk Score: {final_score:.2f}")

    if final_score > 0.75:
        st.error("🚨 BLOCK TRANSACTION")
    elif final_score > 0.4:
        st.warning("⚠️ VERIFY USER")
    else:
        st.success("✅ LEGITIMATE")

    # ---------------- BREAKDOWN (UNIQUE) ----------------
    st.write("### 📊 Risk Breakdown")
    st.write(f"ML Score: {prob:.2f}")
    st.write(f"Rule Score: {risk_score:.2f}")

    # ---------------- REASONS ----------------
    st.write("### 🔍 Reasons")
    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("- No strong indicators")

    # ---------------- USER PROFILE ----------------
    if len(st.session_state.history) > 0:
        avg = np.mean(st.session_state.history)
        st.write("### 👤 User Profile")
        st.write(f"Average Spend: ₹{avg:.2f}")
        st.write(f"Transactions Count: {len(st.session_state.history)}")

# ---------------- WHAT-IF ANALYSIS ----------------
st.header("🔄 What-if Analysis")

test_amount = st.slider("Try different amount", 0, 200000, 1000)

st.write(f"If transaction = ₹{test_amount}, risk behavior may change")

# ---------------- GRAPHS ----------------
st.header("📊 Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax1)
    ax1.set_xticklabels(['Genuine', 'Fraud'])
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.histplot(df[df['Class']==0]['Amount'], bins=50, ax=ax2)
    sns.histplot(df[df['Class']==1]['Amount'], bins=50, color='red', ax=ax2)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots()
    sns.histplot(df[df['Class']==1]['Time'], bins=50, color='red', ax=ax3)
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)