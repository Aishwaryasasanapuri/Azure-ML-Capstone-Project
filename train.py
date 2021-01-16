#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import __version__ as sklearnver
from packaging.version import Version
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset


run = Run.get_context()

# Retrive current run's information

ws = run.experiment.workspace
found = False
key = "Glass Dataset"
description_text = "Glass classification"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

ds= TabularDatasetFactory.from_delimited_files(path="https://raw.githubusercontent.com/Aishwaryasasanapuri/test2/main/glass.csv")

def clean_data(data):
  
    # Clean the data
    x_df = data.to_pandas_dataframe().dropna()
    #x_df.drop("name", inplace=True, axis=1)
    #jobs = pd.get_dummies(x_df.job, prefix="job")
    y_df = x_df.pop("Type")
    
    return x_df, y_df


import os

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    x, y = clean_data(ds)
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,shuffle =True)

# TODO: Split data into train and test sets.

# +
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    import joblib
    os.makedirs('outputs', exist_ok=True)  
    joblib.dump(model, 'outputs/hdmodel.joblib')
    
    
# create an output folder
# -

if __name__ == '__main__':
    main()

