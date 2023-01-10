import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_newts(filepath, do_min_max=False):
    data = pd.read_csv(filepath, delimiter=';')
    xvals_raw = data.drop(['ID', 'Green frogs', 'Brown frogs', 'Common toad', 'Tree frog', 'Common newt', 'Great crested newt', 'Fire-bellied toad'], axis=1)
    yvals = data['Fire-bellied toad']

    # optional min-max scaling
    if (do_min_max):
        #for col in ['SR', 'NR', 'TR', 'VR', 'OR', 'RR', 'BR']: 
        for col in ['SR', 'NR', 'OR']: 
            xvals_raw[col] = (xvals_raw[col] - xvals_raw[col].min())/(xvals_raw[col].max() - xvals_raw[col].min())
    xvals = pd.get_dummies(xvals_raw, columns=['Motorway', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'RR', 'BR', 'MR', 'CR'])
    return xvals, yvals

def load_heartdisease(filepath, do_min_max=False):
    data = pd.read_csv(filepath, delimiter=',')
    xvals_raw = data.drop(['NUM'], axis=1)
    yvals = data['NUM'].copy(deep=True)
    for y in range(len(yvals)):
        if (yvals[y]>0):
            yvals[y] = 1
    # optional min-max scaling <- need to figure out what this does
    if (do_min_max):
        for col in ['AGE', 'THRESTBPS', 'CHOL', 'THALACH', 'OLDPEAK', 'CA']:
            xvals_raw[col] = (xvals_raw[col] - xvals_raw[col].min())/(xvals_raw[col].max() - xvals_raw[col].min())
    xvals = pd.get_dummies(xvals_raw, columns=['SEX', 'CP','FBS', 'RESTECG', 'EXANG', 'SLOPE','THAL'])
    return xvals, yvals
