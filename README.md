# 💳 Credit Card Fraud Detection System

A Machine Learning-based web application to detect fraudulent credit card transactions using the Random Forest algorithm and deployed with Streamlit.

---

## 🚀 Project Overview

This project aims to identify fraudulent transactions from credit card data by analyzing transaction patterns. It uses a trained machine learning model to classify transactions as fraudulent or legitimate.

---

## 🧠 Technologies Used

- Python
- Machine Learning (Random Forest)
- Scikit-learn
- Pandas, NumPy
- Streamlit (for web app)
- Matplotlib & Seaborn (for visualization)

---

## 📊 Dataset

- Source: Kaggle Credit Card Fraud Detection Dataset
- Contains over 284,000 transactions
- Highly imbalanced dataset (very few fraud cases)

---

## ⚙️ How It Works

1. Data preprocessing and feature scaling  
2. Model training using Random Forest  
3. User inputs transaction details  
4. System generates transaction features  
5. Model predicts:
   - Legitimate Transaction  
   - Fraudulent Transaction  

---

## 🌐 Application Features

- User-friendly web interface using Streamlit  
- Real-time fraud detection  
- Simulation of transaction patterns  
- Smart logic for realistic fraud scenarios  

---

## 📈 Visualizations

The project includes graphs for:
- Fraud vs Genuine transaction distribution  
- Transaction amount analysis  
- Fraud-only and genuine-only patterns  

---

## ▶️ How to Run the Project

1. Install required libraries:
pip install pandas numpy scikit-learn streamlit matplotlib seaborn

2. Train the model:
python model.py

3. Run the application:
streamlit run app.py

---

## 🎯 Output

The system predicts whether a transaction is:
- Legitimate  
- Fraudulent  

---

## ⚠️ Limitations

- This is a simulation-based system  
- Real-world systems use live transaction data  
- Features are anonymized (V1–V28)  

---

## 🔮 Future Improvements

- Real-time fraud detection system  
- Deep learning models  
- Integration with banking APIs  
- Advanced UI dashboard  

---

## 👨‍💻 Author

Developed as a Machine Learning project for academic purposes.

---

## ⭐ Conclusion

This project demonstrates how machine learning can be applied to detect financial fraud and highlights the challenges of working with imbalanced datasets.
