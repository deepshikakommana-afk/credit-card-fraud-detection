import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("creditcard.csv")

plt.figure(figsize=(12,10))

# 1 Fraud vs Genuine
plt.subplot(2,2,1)
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Genuine")
plt.xticks([0,1], ['Genuine','Fraud'])

# 2 Combined Amount
plt.subplot(2,2,2)
sns.histplot(df[df['Class']==0]['Amount'], bins=50, label='Genuine')
sns.histplot(df[df['Class']==1]['Amount'], bins=50, color='red', label='Fraud')
plt.legend()
plt.title("Amount Distribution")

# 3 Fraud only
plt.subplot(2,2,3)
sns.histplot(df[df['Class']==1]['Amount'], bins=50, color='red')
plt.title("Fraud Only")

# 4 Genuine only
plt.subplot(2,2,4)
sns.histplot(df[df['Class']==0]['Amount'], bins=50)
plt.title("Genuine Only")

plt.tight_layout()
plt.show()