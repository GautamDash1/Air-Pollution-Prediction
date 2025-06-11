import pandas as pd
import numpy as np

def categorize_aqi(aqi_series):
    bins = [-np.inf, 50, 100, 200, 300, 400, np.inf]
    labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    return pd.cut(aqi_series, bins=bins, labels=labels)

df=pd.read_csv('notebooks\Data\AQI_DELHI.csv')

df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

df['AQI_Category'] = categorize_aqi(df['AQI'])

df.to_csv("notebooks\Data\AQI.csv", index=False)
