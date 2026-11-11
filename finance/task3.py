import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

#Selecting IBM stock prices
df = pd.read_csv('IBM.csv', index_col='Date', parse_dates=True)

#Handle missing values. 
#Perform exploratory data analysis (EDA): visualize the time series, check for trends and seasonality.
# Test for stationarity using the Augmented Dickey-Fuller (ADF) test and apply differencing if necessary.

print(df.isna().sum()) #no missing values
df.index = pd.to_datetime(df.index)
df = df.sort_index()

result = seasonal_decompose(df['Close'], model='multiplicative',period=52*5)

plt.figure(figsize=(10,8))
plt.subplot(3, 1, 1)
plt.plot(df['Close'], label='Close')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend()
plt.tight_layout()
plt.show()

result2 = adfuller(df['Close'])
print('ADF Statistic: %f' % result2[0])
print('p-value: %f' % result2[1])
print('Critical Values:')
for key, value in result2[4].items():
    print('\t%s: %.3f' % (key, value))

#Identify ARIMA parameters (p, d, q) using ACF and PACF plots or automated selection (e.g., auto_arima).
#Fit the ARIMA model to the training data

train_arima = df['Close']
stepwise_fit = auto_arima(train_arima, start_p = 0, start_q=0, seasonal=False, d=None,trace=True)
print(stepwise_fit.summary())


#Split the dataset into training and test sets.
#Forecast future values and compare predictions with actual test data.
#Report evaluation metrics such as RMSE, MAE, and MAPE.

# parameters from the summary
p,d,q = stepwise_fit.order

train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]

train_arima = train_data['Close']
test_arima = test_data['Close']

initial_model = ARIMA(train_arima.to_numpy(),order=(p, d, q)).fit()
fitted_values = initial_model.fittedvalues

history = train_arima.to_list()
model_fit = ARIMA(history, order=(p,d,q)).fit()

predictions = list()
for i in test_arima:
    model_fit = ARIMA(history, order=(p,d,q)).fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(i)

actual = test_arima.to_numpy()
predicted = np.asarray(predictions)

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
mape = np.mean(np.abs((actual - predicted)/actual*100))

print(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

#Plot the original series, fitted values, and forecasted values.
initial_model = ARIMA(train_arima.to_numpy(),order=(p, d, q)).fit()

fitted_values = initial_model.fittedvalues

plt.figure(figsize=(16,8))
plt.plot(df.index, df['Close'], color='green', label = 'Original')

plt.plot(train_arima.index, fitted_values, color = 'red', label = 'Fitted values')
plt.plot(test_data.index, predictions, color = 'blue', label = 'Forecasted values')
plt.title('IBM Stock Price')
plt.legend()
plt.grid(True)
plt.show()

#Interpret the model parameters and results.

"""
The ADF test showed that the close price was not stationary. 
The best model was ARIMA(0,1,0), so the data was differenced once. 
The model had RMSE 2.41, MAE 1.62, and MAPE 1.24%. 
Results showed that the model made relatively small prediction errors.
Overall, the model gave good predictions.
"""