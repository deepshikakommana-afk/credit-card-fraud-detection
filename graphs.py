import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("creditcard.csv")

plt.figure(figsize=(14,10))

# ---------------- 1. Fraud vs Genuine ----------------
plt.subplot(2,2,1)
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Genuine Transactions")
plt.xticks([0,1], ['Genuine','Fraud'])

# ---------------- 2. Amount Distribution ----------------
plt.subplot(2,2,2)
sns.histplot(df[df['Class']==0]['Amount'], bins=50, label='Genuine', kde=True)
sns.histplot(df[df['Class']==1]['Amount'], bins=50, color='red', label='Fraud', kde=True)
plt.legend()
plt.title("Transaction Amount Distribution")

# ---------------- 3. Fraud Time Distribution ----------------
plt.subplot(2,2,3)
sns.histplot(df[df['Class']==1]['Time'], bins=50, color='red')
plt.title("Fraud Transactions Over Time")

# ---------------- 4. Correlation Heatmap ----------------
plt.subplot(2,2,4)
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()