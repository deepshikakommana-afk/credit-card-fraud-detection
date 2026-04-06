# 💳 Intelligent Credit Card Fraud Detection System

## 📌 Overview
This project is a machine learning-based credit card fraud detection system enhanced with rule-based logic, behavioral analysis, and anomaly detection. It not only predicts fraud but also provides risk scoring, explanations, and decision support similar to real-world banking systems.

---

## 🚀 Features

- 🔍 Fraud Detection using Machine Learning (Random Forest)
- ⚖️ Risk Scoring System (0 to 1)
- 🏦 Bank-style Policy Rules (Minor & Transaction Limits)
- 🧠 Behavioral Analysis (User transaction history)
- 🚨 Anomaly Detection (Isolation Forest)
- 📊 Data Visualization (Graphs inside app)
- 📈 Risk Breakdown (ML vs Rules)
- 🔄 What-if Analysis (Interactive simulation)
- 👤 User Profile (Average spend & transaction count)
- 🔎 Explainable Output (Reasons for decision)

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- imbalanced-learn (SMOTE)

---

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── model.py          # Model training
├── app.py            # Streamlit application
├── graphs.py         # Data analysis (optional)
├── model.pkl         # Trained ML model
├── scaler.pkl        # Feature scaler
├── anomaly.pkl       # Anomaly detection model
├── creditcard.csv    # Dataset
└── README.md
```

---

## ⚙️ How to Run

### 1️⃣ Install dependencies
pip install pandas numpy scikit-learn streamlit seaborn matplotlib imbalanced-learn

### 2️⃣ Train the model
python model.py

### 3️⃣ Run the app
streamlit run app.py

---

## 📊 Dataset

- Source: Kaggle Credit Card Fraud Detection Dataset
- Contains anonymized features (V1–V28), Time, Amount, and Class
- Class:
  - 0 → Genuine transaction
  - 1 → Fraudulent transaction

---

## 🧠 How It Works

1. Applies bank policy rules (limits, minor restrictions)
2. Tracks user transaction behavior
3. Uses machine learning to predict fraud probability
4. Detects anomalies using Isolation Forest
5. Combines all signals into a final risk score
6. Generates decision:
   - ✅ Allow
   - ⚠️ Verify
   - 🚨 Block
7. Provides explanation and insights

---

## 🎯 Output

- 💳 Risk Score (0–1)
- 📌 Decision (Allow / Verify / Block)
- 🔍 Reasons for decision
- 👤 User profile (average spend & history)
- 🔄 What-if analysis for testing scenarios

---

## 💡 Key Innovation

Unlike basic fraud detection systems, this project:

- Combines ML + rules + behavior + anomaly detection
- Provides transparent explanations (Explainable AI)
- Includes interactive "What-if" simulation
- Shows user risk profile
- Mimics real-world banking decision systems

---

## 📌 Future Improvements

- Real-time transaction processing
- Cloud deployment (AWS / Azure)
- Mobile app integration
- User authentication system

---

## 👨‍💻 Author

Deepshika Kommana

---

## ⭐ Conclusion

This project demonstrates a complete fraud detection pipeline that not only predicts fraud but also explains decisions, making it closer to real-world intelligent financial systems.