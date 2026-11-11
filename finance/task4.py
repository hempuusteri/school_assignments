import pandas as pd
import numpy as np

df = pd.read_csv('tsla.csv')
#remove unnecessary column
df = df.drop(columns=['Unnamed: 0'])


#compute 5-day rolling mean of "Close" column 
# and store results in column called "Close rolling"
df['Close rolling'] = df['Close'].rolling(window=5).mean()


#compute 5-day rolling mean of "Volume" column with centered window
#and store result in column "Volume rolling

df['Volume rolling'] = df['Volume'].rolling(window=5, center=True).mean()

print(df)

#print the indexes of peaks in "Close rolling" column

y = df['Close rolling']
def find_peaks(y):
    return np.where(np.diff(1*(np.diff(y)>0))==-1)[0]+1

print(find_peaks(y))