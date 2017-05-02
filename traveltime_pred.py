#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Travel time prediction

Created on Tue May  2 15:07:23 2017

@author: hxw186
"""

import pandas as pd
import numpy as np
from datetime import datetime
from volume_pred import training_features_df, append_weather_features, regression_evaluation




def testing_features_df(days):
    columns = ["intersection_id", "tollgate_id", "startT"]
    intersection_setting = [["A", '2'], ['A', '3'], ['B', '1'], ['B', '3'],
                            ['C', '1'], ['C', '3']]

    tables = []
    for intersection in intersection_setting:
        for day in days:
            for hour in [8,9,17,18]:
                for minute in [0, 20, 40]:
                    row = list(intersection)
                    dt = datetime(2016, 10, day, hour, minute)
                    row.append(dt)
                    tables.append(row)
    
    df = pd.DataFrame(data=tables, columns=columns)
    df["dayofweek"] = df["startT"].dt.dayofweek
    df["daysinmonth"] = df["startT"].dt.daysinmonth
    df["slotofday"] = df["startT"].apply(lambda x: x.hour*3+x.minute/20)
    
    print df.shape
    return df



def prepare_data():
    df_train1 = training_features_df("data/training_20min_avg_travel_time.csv")
    df_train1 = append_weather_features(df_train1, "data/weather (table 7)_training_update.csv")
    
    df_train2 = training_features_df("data/test1_20min_avg_travel_time.csv")
    df_train2 = append_weather_features(df_train2, "data/testing_phase1/weather (table 7)_test1.csv")
    
    df_test = testing_features_df([18,19,20,21,22,23,24])
    df_test = append_weather_features(df_test, "data/testing_phase1/weather (table 7)_test1.csv")
    
    # map string feature to integer
    intsec_int = {"A": 1, "B": 2, "C": 3}
    df_train1['intersection_id'] = df_train1['intersection_id'].apply(lambda x: intsec_int[x])
    df_train2['intersection_id'] = df_train2['intersection_id'].apply(lambda x: intsec_int[x])
    df_test['intersection_id'] = df_test['intersection_id'].apply(lambda x: intsec_int[x])
    return df_train1, df_train2, df_test



if __name__ == '__main__':
    df_train1, df_train2, df_test = prepare_data()
    feature_columns = ['intersection_id', 'tollgate_id', 'dayofweek', 'daysinmonth', 'slotofday']
    regression_evaluation(df_train1, df_train2, df_test, feature_columns, "avg_travel_time")
    