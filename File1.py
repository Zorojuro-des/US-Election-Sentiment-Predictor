import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')
print(data.isnull().sum())

print(data.duplicated().sum())

parties = data["party"].nunique()
candidates = data["candidate"].nunique()
print(f"Unique Parties {parties}")
print(f"Unique Candidates {candidates}")

data_updated = data[["retweets","likes","sentiment"]]

data_updated.sentiment = data_updated.sentiment.map({'negative': 0, 'neutral': 1, 'positive': 2})
print(data_updated.head())

print(data_updated.isnull().sum())
scaler = StandardScaler()
scaler.fit(data_updated)
data_updated = pd.DataFrame(scaler.transform(data_updated))
corr_table = data_updated.corr(method = 'pearson')
print(corr_table)

sns.heatmap(corr_table, annot = True,cmap='coolwarm')
# sns.boxplot(data = data_updated)
plt.show()

data_updated.dropna( inplace=True)

print(data_updated.isnull().sum())

