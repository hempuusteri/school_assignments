import pandas as pd
import numpy as np

df = pd.read_csv('aapl.csv')
df3 = df.copy()
df = df.drop(columns=['Unnamed: 0'] )
df = df.drop_duplicates()
print(df.isna().sum())
df2 = df.copy()

columns = ['Close', 'Open', 'Volume']
df2[columns] = df[columns].fillna(df[columns].mean())

print(df2.isna().sum())
cols = ['Close', 'High', 'Low', 'Open', 'Volume']
for col in cols:
    q1 = df2[col].quantile(0.25)
    q3 = df2[col].quantile(0.75)
    iqr = q3-q1
    lower = q1-1.5*iqr
    upper = q3+2*iqr

    df2.loc[(df2[col] < lower) | (df2[col] > upper), col] = np.nan

df2[cols] = df2[cols].fillna(df2[cols].mean())
print(df2)