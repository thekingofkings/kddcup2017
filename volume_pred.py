#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Task 2 volume prediction

Created on Mon May  1 13:01:10 2017

@author: hxw186
"""

import pandas as pd
import numpy as np
from datetime import datetime


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
    raw = pd.read_csv(filepath)




def regression_evaluation():
    df_train1 = training_features_df("data/training_20min_avg_volume.csv")
    df_train2 = training_features_df("data/test1_20min_avg_volume.csv")
    df_train = df_train1.append(df_train2)
    
    df_test = testing_features_df([18,19,20,21,22,23,24])
    
    
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_absolute_error
    
    regr = DecisionTreeRegressor(max_depth=5)
    rf = RandomForestRegressor(n_estimators=10, max_depth=3)
    adb = AdaBoostRegressor(n_estimators=10)
    knn = KNeighborsRegressor(n_neighbors=5)
    bag = BaggingRegressor()
    
    feature_columns = ['tollgate_id', 'direction', 'dayofweek', 'daysinmonth', 'slotofday']
    
    X_train = df_train1.as_matrix(columns=feature_columns)
    Y_train = df_train1['volume'].values
    
    X_vald = df_train2.as_matrix(columns=feature_columns)
    Y_vald = df_train2["volume"].values
    
    X_test = df_test.as_matrix(columns=feature_columns)
    
    
    for model in [regr, rf, adb, knn, bag]:
        model.fit(X_train, Y_train)
        Y_vald_est = model.predict(X_vald)
        
        mae = mean_absolute_error(Y_vald, Y_vald_est)
        mape = np.mean(np.abs(Y_vald - Y_vald_est) / Y_vald)
        print model
        print mae, mape
        
        
if __name__ == '__main__':
#    regression_evaluation()
    df_train = training_features_df("data/training_20min_avg_volume.csv")
    append_weather_features(df_train, "data/weather (table 7)_training_update.csv")
