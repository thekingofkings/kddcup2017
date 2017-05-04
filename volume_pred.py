#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Task 2 volume prediction

Created on Mon May  1 13:01:10 2017

@author: hxw186
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


weather_columns = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation']


def training_features_df(filepath):
    raw = pd.read_csv(filepath)
    raw["startT"] = pd.to_datetime(raw["time_window"].apply(lambda x: x[1:-1].split(",")[0]))
    
    df = pd.DataFrame(raw.loc[raw["startT"].apply(lambda x: x.hour >= 6 and x.hour< 17)])
    
    
    df["dayofweek"] = df["startT"].dt.dayofweek
    df["daysinmonth"] = df["startT"].dt.daysinmonth
    df["slotofday"] = df["startT"].apply(lambda x: x.hour * 3 + x.minute/20)
    
    print raw.shape, df.shape
    return df


def testing_features_df(days):
    columns = ["tollgate_id", "direction", "startT"]
    tollgate_setting = [["1", "0"], ["1", "1"], ['2', '0'], ['3', '0'], ['3', '1']]
    
    tables = []
    for tollgate in tollgate_setting:
        for day in days:
            for hour in [8,9,17,18]:
                for minute in [0,20,40]:
                    row = list(tollgate)
                    dt = datetime(2016, 10, day, hour, minute)
                    row.append(dt)
                    tables.append(row)
    
    df = pd.DataFrame(tables, columns=columns)
    df["dayofweek"] = df["startT"].dt.dayofweek
    df["daysinmonth"] = df["startT"].dt.daysinmonth
    df["slotofday"] = df["startT"].apply(lambda x: x.hour*3+x.minute/20)
    
    print df.shape
    return df



def append_weather_features(df, filepath):
    weather = pd.read_csv(filepath)
    df_weather = []
    for index, row in df.iterrows():
        date = row["startT"].strftime("%Y-%m-%d")
        hour = row["startT"].hour / 3 * 3
        idx = np.where((weather['date']==date)&(weather['hour']==hour))[0]
        df_weather.append(np.squeeze(weather.ix[idx, weather_columns].values).tolist())
    df_weather = pd.DataFrame(df_weather, columns=weather_columns, index=df.index)
    dfn = pd.concat([df, df_weather], axis=1)
    return dfn





def prepare_data():
    df_train1 = training_features_df("data/training_20min_avg_volume.csv")
    df_train1 = append_weather_features(df_train1, "data/weather (table 7)_training_update.csv")
    df_train1 = df_train1.fillna(0)

    df_train2 = training_features_df("data/test1_20min_avg_volume.csv")
    df_train2 = append_weather_features(df_train2, "data/testing_phase1/weather (table 7)_test1.csv")
    df_train2 = df_train2.fillna(0)
    
    df_test = testing_features_df([18,19,20,21,22,23,24])
    df_test = append_weather_features(df_test, "data/testing_phase1/weather (table 7)_test1.csv")
    df_test = df_test.fillna(0)
    return df_train1, df_train2, df_test
    

def regression_evaluation(df_train1, df_train2, df_test, feature_columns, labelname):
    
    df_train = df_train1.append(df_train2)
    
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_absolute_error
    
    regr = DecisionTreeRegressor(max_depth=5)
    rf = RandomForestRegressor(n_estimators=10, max_depth=3)
    adb = AdaBoostRegressor(n_estimators=10)
    knn = KNeighborsRegressor(n_neighbors=5)
    bag = BaggingRegressor()
    
    X_train = df_train1.as_matrix(columns=feature_columns)
    Y_train = df_train1[labelname].values
    
    X_vald = df_train2.as_matrix(columns=feature_columns)
    Y_vald = df_train2[labelname].values
    
    X_test = df_test.as_matrix(columns=feature_columns)
    
    min_mape = 1
    best_mod = None
    for model in [regr, rf, adb, knn, bag]:
        model.fit(X_train, Y_train)
        Y_vald_est = model.predict(X_vald)
        
        mae = mean_absolute_error(Y_vald, Y_vald_est)
        mape = np.mean(np.abs(Y_vald - Y_vald_est) / Y_vald)
        print model
        print mae, mape
        if mape < min_mape:
            min_mape = mape
            best_mod = model
            
    X_train = df_train.as_matrix(columns=feature_columns)
    Y_train = df_train[labelname].values
    best_mod.fit(X_train, Y_train)
    Y_test_est = best_mod.predict(X_test)
    
    return Y_test_est
    
        
        
def generate_output(df_test, Y_test_est):
    df_test["volume"] = Y_test_est
    df_test["time_window"] = df_test["startT"].apply(lambda x: x.strftime("[%Y-%m-%d %H:%M:%S,") + 
           (x + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S)"))
    df_test.to_csv("volume.csv", columns=["tollgate_id", "time_window", "direction", "volume"], index=False)
    
    
        
if __name__ == '__main__':
    df_train1, df_train2, df_test = prepare_data()
    feature_columns = ['tollgate_id', 'direction', 'dayofweek', 'daysinmonth', 'slotofday'] + weather_columns
    Y_test_est = regression_evaluation(df_train1, df_train2, df_test, feature_columns, labelname='volume')
    generate_output(df_test, Y_test_est)