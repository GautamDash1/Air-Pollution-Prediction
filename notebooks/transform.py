import pandas as pd
import numpy as np
import os
def data_corr_coef():
    df=pd.read_csv('notebooks\Data\AQI_DELHI.csv')
    cols=['PM2.5','PM10','NO','NO2','NOx','NH3','SO2','CO','Ozone','Benzene','Toluene','RH','WS','WD','SR','BP','AT','RF','TOT-RF','AQI']

    data = df[cols]
    
    aqi_correlations = data.corr()['AQI'].drop('AQI')

    threshold = 0.80
    cols1=[]
    drop_cols = aqi_correlations[aqi_correlations.abs() > threshold].index.tolist()
    selected_features = [col for col in data.columns if col not in drop_cols and col != 'AQI']


    return cols1
drop=data_corr_coef()
