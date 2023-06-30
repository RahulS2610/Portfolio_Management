## Imports
import warnings
warnings.filterwarnings('ignore')

# Data Manipulation and Treatment
import numpy as np
import pandas as pd
from pandas_datareader import data
import pmdarima as pm

# Plotting and Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from pandas import DataFrame
from arch import arch_model
from scipy import stats

# Plot Correlogram
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import cumsum, log, polyfit, sqrt, std, subtract



# !pip install plotly
import plotly.graph_objects as go

# Scikit-Learn for Modeling
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

# Statistics
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima


# Model
from statsmodels.tsa.statespace.sarimax import SARIMAX


## Dataset

# Fetches QQQ stocks using the start and end dates and fills any NA with 1.
start_date = '2006-10-02'
end_date = '2020-07-31'
tickers = ['MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'GOOGL', 'NVDA', 'PYPL', 'ADBE']
df = data.DataReader('AAPL', 'yahoo', start_date, end_date)
ex = data.DataReader(tickers, 'yahoo', start_date, end_date)
ex = ex.fillna(1)
print(ex['Adj Close'])



# Shows seasonal decomposition of target stock
result = seasonal_decompose(df['Adj Close'], model='multiplicative', period=30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16, 9)
plt.show()


## Changing to log returns, checking for stationarity

# Changes all data to monthly log returns and just uses Adjusted Closing price as data.
df = df.resample('M').ffill().fillna(1)
ex = ex.resample('M').ffill().fillna(1)
ex = ex['Adj Close']
df.price = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
ex = np.log(ex) - np.log(ex.shift(1))

# Cuts off first data point since no log return possible for it.
df.price = df.price[1:]
ex = ex[1:]
print(df.price)
print(ex)

# Performs stationarity test on target data for p-value to be below 0.05.
res = adfuller(df.price, autolag='AIC', regression='nc')
print('Augmented Dickey Fuller Statistic: %f' % res[0])
print('p-value: {}'.format(res[1]))
print('p-value is < 0.05 so dataset is stationary')
#print('p-value is > 0.05 so dataset is nonstationary. However, auto_arima/SARIMAX will difference the dataset to make it stationary.')



## Auto ARIMA

# Divides data into training and testing parts.
X_train, X_test = ex[:'2019-12-31'], ex['2020-01-01':]
Y_train, Y_test = df.price[:'2019-12-31'], df.price['2020-01-01':]

# Runs auto_arima to find best order and seasonal order for SARIMAX moodel and plots diagnostics.
auto_model = auto_arima(Y_train.to_numpy()[1:], exogenous=X_train.shift(1)[1:], test='adf', m=12, seasonal=True, error_action='trace', stepwise=True, trace=True, method='nm', start_P=1, start_Q=1)
auto_model.plot_diagnostics(figsize=(12,5))
plt.show()


## Model Fitting and Prediction

# Shifts exogenous stocks back one month since that data is last known data for prediction of current month.
X_shifted = X_test.shift(1).fillna(0)

# Creates and fits SARIMAX model on training data using the auto_arima orders.
model = SARIMAX(endog=Y_train[1:], exog=X_train.shift(1)[1:], order=auto_model.order, seasonal_order=auto_model.seasonal_order, enforce_stationarity=True)
model_fit = model.fit()
print(model_fit.summary())

# Predicts data for testing frame and plots results against actual values.
result = model_fit.predict(start=Y_test.index[0], end=Y_test.index[-1], exog=X_shifted)
plt.figure(figsize=(18,5))
sns.lineplot(data=pd.DataFrame({'Predicted':result,'Actual':Y_test}))


## Forecasting Scores

# Some error statistics for the result vs testing data.
r2_arima= r2_score(Y_test,result)
mse_arima= mean_squared_error(Y_test,result)
rmse_arima=np.sqrt(mean_squared_error(Y_test,result))
mae_arima=mean_absolute_error(Y_test,result)
print("R Square Score ARIMA: ",r2_arima)
print("Mean Square Error ARIMA: ",mse_arima)
print("Root Mean Square Error ARIMA: ",rmse_arima)
print("Mean Absoulute Error ARIMA: ",mae_arima)

# Creates figure with all data showing. Does not work on Github as it is dynamic (can hover over part of data line and see what the value and date is).
fig=go.Figure()
fig.add_trace(go.Scatter(x=Y_train.index.append(Y_test.index), y=Y_train.append(Y_test), mode='lines',name="Training Data for UPS"))
fig.add_trace(go.Scatter(x=Y_train.index.append(Y_test.index), y=model_fit.fittedvalues.append(result), mode='lines',name="Model Fitting"))
fig.add_vline(x=Y_test.index[0])
fig.update_layout(title="ARIMA",xaxis_title="Date",yaxis_title="Close",legend=dict(x=0,y=1,traceorder="normal"),font=dict(size=12))
fig.show()

# Static figure with all data showing.
plt.plot(Y_train.append(Y_test), color = 'blue', label="Training Data for AAPL")
plt.plot(model_fit.fittedvalues.append(result), color='red', label="Model Fitting and Prediction")
plt.title('AAPL Log Return Prediction')
plt.xlabel('Date')
plt.ylabel('AAPL Log Returns')
plt.axvline(x=18295)
plt.legend()
plt.rcParams["figure.figsize"] = (20,5)
plt.show()

# Predicted vs Actual log return score.
print("February 2020 log return prediction: {}".format(result[1]))
print("February 2020 log return actual: {}".format(Y_test[1]))
print("Root Mean Squared Error: {}".format(np.sqrt(np.square(result[1] - Y_test[1]))))



# Residuals

# Residuals to be used by GARCH model.
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


#Rahul's Garch
def simulate_GARCH(n, omega, alpha, beta=0):
    np.random.seed(4)
    # Initialize the parameters
    white_noise = np.random.normal(size=n)
    resid = np.zeros_like(white_noise)
    variance = np.zeros_like(white_noise)

    for t in range(1, n):
        # Simulate the variance (sigma squared)
        variance[t] = omega + alpha * resid[t - 1] ** 2 + beta * variance[t - 1]
        # Simulate the residuals
        resid[t] = np.sqrt(variance[t]) * white_noise[t]

    return resid, variance

# Simulate a ARCH(1) series
arch_resid, arch_variance = simulate_GARCH(n= 200, omega = 0.1, alpha = 0.7)
# Simulate a GARCH(1,1) series
garch_resid, garch_variance = simulate_GARCH(n= 200, omega = 0.1, alpha = 0.7, beta = 0.1)

# Plot the ARCH variance
plt.figure(figsize=(10,5))
plt.plot(arch_variance, color = 'red', label = 'ARCH Variance')

# Plot the GARCH variance
plt.plot(garch_variance, color = 'orange', label = 'GARCH Variance')
plt.legend()
plt.show()

# First simulated GARCH
plt.figure(figsize=(10,3))
sim_resid, sim_variance = simulate_GARCH(n = 200,  omega = 0.1, alpha = 0.3, beta = 0.2)
plt.plot(sim_variance, color = 'orange', label = 'Variance')
plt.plot(sim_resid, color = 'green', label = 'Residuals')
plt.title('First simulated GARCH, Beta = 0.2')
plt.legend(loc='best')
plt.show()

# Second simulated GARCH
plt.figure(figsize=(10,3))
sim_resid, sim_variance = simulate_GARCH(n = 200,  omega = 0.1, alpha = 0.3, beta = 0.6)
plt.plot(sim_variance, color = 'red', label = 'Variance')
plt.plot(sim_resid, color = 'deepskyblue', label = 'Residuals')
plt.title('Second simulated GARCH, Beta = 0.6')
plt.legend(loc='best')
plt.show()


# Specify GARCH model assumptions
basic_gm = arch_model(residuals, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
# Fit the model
gm_result = basic_gm.fit(update_freq = 4)

# Display model fitting summary
print(gm_result.summary())

# Plot fitted results
gm_result.plot()
plt.show()

# Make 5-period ahead forecast
gm_forecast = gm_result.forecast(horizon = 5)

# Print the forecast variance
print(gm_forecast.variance[-1:])

# Obtain model estimated residuals and volatility
gm_resid = gm_result.resid
gm_std = gm_result.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std

# Plot the histogram of the standardized residuals
plt.figure(figsize=(7,4))
sns.distplot(gm_std_resid, norm_hist=True, fit= stats.norm, bins=50, color='r')
plt.legend(('normal', 'standardized residuals'))
plt.show()

"""
# Specify GARCH model assumptions
skewt_gm = arch_model(result, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'skewt')

# Fit the model
skewt_result = skewt_gm.fit(disp = 'off')

# Get model estimated volatility
skewt_vol = skewt_result.conditional_volatility

# Plot model fitting results
plt.plot(skewt_vol, color = 'red', label = 'Skewed-t Volatility')
plt.plot(result, color = 'grey', label = 'Daily Returns', alpha = 0.4)
plt.legend(loc = 'upper right')
plt.show()
"""

# Fit GARCH model with ARMA model residuals
_garch_model = arch_model(residuals, mean='Zero', p=1, q=1)
_garch_result = _garch_model.fit(disp = 'off')
print(_garch_result.summary())

# Plot GARCH model fitted results
_garch_result.plot()
plt.show()






