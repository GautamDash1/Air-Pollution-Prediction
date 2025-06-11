import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import itertools

warnings.filterwarnings("ignore")

# === Load and preprocess data ===
file_path = 'notebooks/Data/AQI_DELHI.csv'
data = pd.read_csv(file_path)

# Drop rows where AQI is zero or NaN
data = data[data['AQI'] != 0].dropna(subset=['AQI'])

# Parse dates and set index
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y %H:%M')
data.set_index('Date', inplace=True)
data = data.asfreq('D')  # Daily frequency

# Interpolate missing AQI values if any
data['AQI'] = data['AQI'].interpolate()

# Use AQI as endogenous variable
endog = data['AQI']

# Optional: select exogenous variables, e.g., ['PM2.5', 'PM10'] or None
exog_vars = ['PM2.5', 'PM10']  # change as you want or set to None
exog = data[exog_vars] if exog_vars else None

# === ADF test to check stationarity ===
adf_result = adfuller(endog.dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

# Decide differencing order d
d = 1 if adf_result[1] > 0.05 else 0

# If differencing needed, difference series for order selection
if d == 1:
    endog_diff = endog.diff().dropna()
else:
    endog_diff = endog

# === Train-test split ===
test_size = 30
train_endog = endog[:-test_size]
test_endog = endog[-test_size:]

train_exog = exog[:-test_size] if exog is not None else None
test_exog = exog[-test_size:] if exog is not None else None

# === Grid search ARIMA and Seasonal order ===

p = q = range(0, 3)  # AR and MA terms 0-2
P = D = Q = range(0, 2)  # seasonal AR, I, MA 0-1
s = 7  # seasonal period: weekly seasonality for daily data (change as needed)

best_aic = np.inf
best_order = None
best_seasonal_order = None

# Clean exogenous variables
if exog is not None:
    exog = exog.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    exog = exog.fillna(method='ffill').fillna(method='bfill')  # Forward/backward fill remaining NaNs

    # Split again after cleaning
    train_exog = exog[:-test_size]
    test_exog = exog[-test_size:]


print("Starting grid search for SARIMAX orders...")

for param in itertools.product(p, q):
    for seasonal_param in itertools.product(P, D, Q):
        try:
            order = (param[0], d, param[1])
            seasonal_order = (seasonal_param[0], seasonal_param[1], seasonal_param[2], s)
            model = SARIMAX(train_endog,
                            order=order,
                            seasonal_order=seasonal_order,
                            exog=train_exog,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_seasonal_order = seasonal_order
            print(f"Tested SARIMAX{order}x{seasonal_order} - AIC:{model_fit.aic:.2f}")
        except Exception as e:
            continue

print(f"\nBest SARIMAX order: {best_order} Seasonal order: {best_seasonal_order} with AIC: {best_aic:.2f}")

# === Fit final model with best order ===
final_model = SARIMAX(train_endog,
                      order=best_order,
                      seasonal_order=best_seasonal_order,
                      exog=train_exog,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
final_fit = final_model.fit(disp=False)

# === Forecast ===
forecast_res = final_fit.get_forecast(steps=test_size, exog=test_exog)
forecast_mean = forecast_res.predicted_mean
forecast_ci = forecast_res.conf_int()

# === Evaluation ===
y_true = test_endog
y_pred = forecast_mean

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f"\nEvaluation Metrics:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

# === Plot results ===
plt.figure(figsize=(14, 7))
plt.plot(train_endog.index, train_endog, label='Training AQI')
plt.plot(test_endog.index, y_true, label='Actual AQI', marker='o', color='blue')
plt.plot(test_endog.index, y_pred, label='Forecasted AQI', marker='x', color='red')
plt.fill_between(test_endog.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3, label='95% CI')
plt.title('AQI Forecast with SARIMAX')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
