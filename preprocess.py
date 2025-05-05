import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import csv

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_valid = pd.read_csv('val.csv')

scaler = StandardScaler()
df_train_processed = scaler.fit_transform(df_train[['likes', 'retweets']])
df_train_processed = pd.DataFrame(df_train_processed)

df_test_processed = scaler.fit_transform(df_test[['likes', 'retweets']])
df_test_processed = pd.DataFrame(df_test_processed)

df_valid_processed = scaler.fit_transform(df_valid[['likes', 'retweets']])
df_valid_processed = pd.DataFrame(df_valid_processed)

df_train_processed.rename(columns={0: 'likes', 1: 'retweets'}, inplace=True)
df_valid_processed.rename(columns={0: 'likes', 1: 'retweets'}, inplace=True)
df_test_processed.rename(columns={0: 'likes', 1: 'retweets'}, inplace=True)


df_test_processed['sentiment'] = df_test['sentiment']
df_train_processed['sentiment'] = df_train['sentiment']
df_valid_processed['sentiment'] = df_valid['sentiment']

df_train_processed['tweet_text'] = df_train['tweet_text'].apply(lambda x: f'"{x}"' if pd.notnull(x) else x)
df_valid_processed['tweet_text'] = df_valid['tweet_text'].apply(lambda x: f'"{x}"' if pd.notnull(x) else x)
df_test_processed['tweet_text'] = df_test['tweet_text'].apply(lambda x: f'"{x}"' if pd.notnull(x) else x)

for file_name, df in {
    "train_preprocessed.csv": df_train_processed,
    "test_preprocessed.csv": df_test_processed,
    "val_preprocessed.csv": df_valid_processed
}.items():
    df.to_csv(file_name, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
