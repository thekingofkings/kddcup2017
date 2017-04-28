#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:15:36 2017

@author: hxw186
"""

import pandas as pd
import tensorflow as tf


COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
       "marital_status", "occupation", "relationship", "race", "gender",
       "capital_gain", "capital_loss", "hours_per_week", "native_country",
       "income_bracket"]

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                   "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]    
    
    
def retrieve_data():
    import urllib
    import os.path
    
    train_file = "census-income.train"
    test_file = "census-income.test"
    
    if not os.path.exists(train_file):
        urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file)
    if not os.path.exists(test_file):
        urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file)

    df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
    
    df_train['label'] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    df_test['label'] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    
    return df_train, df_test
    
    
def input_fn(df):
    continuous_cols = {k : tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(
            indices=[[i,0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1]
            ) for k in CATEGORICAL_COLUMNS}
    
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    label = tf.constant(df['label'].values)
    return feature_cols, label


if __name__ == '__main__':
    df_train, df_test = retrieve_data()
    
    def train_input_fn():
        return input_fn(df_train)
    
    def eval_input_fn():
        return input_fn(df_test)
    
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
    
    education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
    race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=100)
    marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
    
    age = tf.contrib.layers.real_valued_column("age")
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
    
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    
    education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
    
    age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))
    
    
    
    m = tf.contrib.learn.LinearClassifier(feature_columns=[
            gender, native_country, education, occupation, workclass, marital_status, race, relationship,
            age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
            optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0,
            l2_regularization_strength=0.001)
        )
    
    m.fit(input_fn=train_input_fn, steps=200)
    results = m.evaluate(input_fn=eval_input_fn, steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    