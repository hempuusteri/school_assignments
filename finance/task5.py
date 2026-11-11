"""
stock.csv file
-compute daily returns for each stock ticker column using pct_change(),
 no logarithmic conversion
-drop missing values
-drop date column
-compute value at risk for each ticker with initial investmen of 10 000 
and confidence levels 0.9 and 0.99. 
Note that you need to modify the function from course material slightly
to reach the desired output.
-print the output from latter step
"""
import numpy as np
import pandas as pd

data = pd.read_csv('stocks.csv')

stocks = data[[ 'AAPL','AMZN','NVDA']].pct_change().dropna()

stocks_returns_mean = stocks.mean()
weights = np.random.random(len(stocks.columns))
weights /= np.sum(weights)
cov_var = stocks.cov()
port_std = np.sqrt(weights.T.dot(cov_var).dot(weights))

conf_level1 = 0.9
conf_level2 = 0.99
initial_investment = 10000


def VaR_historical(initial_investment, conf_level1):
    Hist_percentile95 = []
    for i, j in zip(stocks.columns,range(len(stocks.columns))):
        Hist_percentile95.append(np.percentile(stocks.loc[:, i],100*(1-conf_level1)))
        print("Based on historical values 95% of {}'s return is {:.4f}".format(i, Hist_percentile95[j]))

        VaR_historical = (initial_investment - initial_investment *(1 + Hist_percentile95[j]))
        print("Historical VaR result for {} is {:.2f} ".format(i, VaR_historical))

VaR_historical(initial_investment,conf_level1)
VaR_historical(initial_investment,conf_level2)