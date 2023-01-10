import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import load

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import backend as K

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

[xa, ya] = load.load_newts('amphibians.csv', True)
[xh, yh] = load.load_heartdisease('heartdisease.csv', True)

# Stage 1:

def linear_SVC(xvalues, yvalues, C):
    ls = LinearSVC(C=C, dual=False)
    ls.fit(xvalues, yvalues)
    return f1_score(ls.predict(xvalues), yvalues)

def kernel_SVC(xvalues, yvalues, C=1.0, gamma=None):
    if gamma is not None:
        s = SVC(gamma=gamma)
    else:
        s = SVC(C=C)
    s.fit(xvalues, yvalues)
    return f1_score(s.predict(xvalues), yvalues)

def kfold_linear_svc(xvals, yvals, splits, C):
    kf = KFold(n_splits=splits, shuffle=True)
    arr = []
    ls = LinearSVC(C=C, dual=False)
    for train_idxs, test_idxs in kf.split(xvals):
        xtrain_this_fold = xvals.loc[train_idxs]
        xtest_this_fold = xvals.loc[test_idxs]
        ytrain_this_fold = yvals.loc[train_idxs]
        ytest_this_fold = yvals.loc[test_idxs]
        # train a model on this fold
        ls.fit(xtrain_this_fold, ytrain_this_fold)
        # test the model on this fold
        arr.append(f1_score(ls.predict(xtest_this_fold), ytest_this_fold))
        #arr.append(f1_score(ls.predict(xtest_this_fold), ytest_this_fold, average='weighted', labels=np.unique(ytest_this_fold)))
    return np.mean(arr)

def kfold_kernel_svc(xvals, yvals, splits, C=1.0, gamma=None):
    kf = KFold(n_splits=splits, shuffle=True)
    arr = []
    if gamma is not None:
        s = SVC(gamma=gamma)
    else:
        s = SVC(C=C)
    for train_idxs, test_idxs in kf.split(xvals):
        xtrain_this_fold = xvals.loc[train_idxs]
        xtest_this_fold = xvals.loc[test_idxs]
        ytrain_this_fold = yvals.loc[train_idxs]
        ytest_this_fold = yvals.loc[test_idxs]
        # train a model on this fold
        s.fit(xtrain_this_fold, ytrain_this_fold)
        # test the model on this fold
        arr.append(f1_score(s.predict(xtest_this_fold), ytest_this_fold)) # this gives a warning
        #arr.append(f1_score(s.predict(xtest_this_fold), ytest_this_fold, average='weighted', labels=np.unique(ytest_this_fold)))
    return np.mean(arr)

# Part 1:

def plot_linear_SVC(xvalues, yvalues, splits, c_start, c_stop, num_points):
    c = []
    whole_set = []
    k_folds = []

    for i in np.logspace(c_start, c_stop, num_points):
        c.append(i)
        whole_set.append(linear_SVC(xvalues, yvalues, i))
        k_folds.append(kfold_linear_svc(xvalues,yvalues, splits, i))

    # the following code prints out the maximum f1_score and the first value
    # of C at which it occurs
    i = whole_set.index(max(whole_set))
    print("max f1 whole_set:", whole_set[i], "c:", c[i])
    i = k_folds.index(max(k_folds))
    print("max f1 k_folds:", k_folds[i], "c:", c[i])

    plt.figure()
    plt.title("LinearSVC: f1_score vs C")
    plt.ylabel("f1_score")
    plt.xlabel("C")
    plt.xscale("log")
    plt.plot(c,np.transpose([whole_set,k_folds]))
    plt.legend(labels=["whole_set", "k_folds"])

# Part 2:

def plot_kernel_SVC(xvalues, yvalues, splits, c_start, c_stop, num_points):
    c = []
    whole_set = []
    k_folds = []

    for i in np.logspace(c_start, c_stop, num_points):
        c.append(i)
        whole_set.append(kernel_SVC(xvalues, yvalues, C=i))
        k_folds.append(kfold_kernel_svc(xvalues,yvalues, splits, C=i))

    # the following code prints out the maximum f1_score and the first value
    # of C at which it occurs
    i = whole_set.index(max(whole_set))
    print("max f1 whole_set:", whole_set[i], "c:", c[i])
    i = k_folds.index(max(k_folds))
    print("max f1 k_folds:", k_folds[i], "c:", c[i])

    plt.figure()
    plt.title("kernel SVC: f1_score vs C")
    plt.ylabel("f1_score")
    plt.xlabel("C")
    plt.xscale("log")
    plt.plot(c,np.transpose([whole_set,k_folds]))
    plt.legend(labels=["whole_set", "k_folds"])

# Part 3:

def plot_kernel_SVC_gamma(xvalues, yvalues, splits, gamma_start, gamma_stop, num_points):
    gamma = []
    whole_set = []
    k_folds = []

    for i in np.logspace(gamma_start, gamma_stop, num_points):
        gamma.append(i)
        whole_set.append(kernel_SVC(xvalues, yvalues, gamma=i))
        k_folds.append(kfold_kernel_svc(xvalues,yvalues, splits, gamma=i))

    # the following code prints out the maximum f1_score and the first value
    # of gamma at which it occurs
    i = whole_set.index(max(whole_set))
    print("max f1 whole_set:", whole_set[i], "gamma:", gamma[i])
    i = k_folds.index(max(k_folds))
    print("max f1 k_folds:", k_folds[i], "gamma:", gamma[i])

    plt.figure()
    plt.title("kernel SVC: f1_score vs gamma")
    plt.ylabel("f1_score")
    plt.xlabel("gamma")
    plt.xscale("log")
    plt.plot(gamma, np.transpose([whole_set,k_folds]))
    plt.legend(labels=["whole_set", "k_folds"])

# Stage 2:

def mlp(xvals, yvals, learning_rate, epoch_num, batch_num, hidden_units):
    model = Sequential([Dense(hidden_units, activation='relu', name="hidden"), Dense(1, activation='sigmoid', name="output"),])
    model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate))
    model.fit(xvals, yvals, epochs=epoch_num, batch_size=batch_num, verbose=False)
    return f1_score(model.predict(xvals, verbose=False).round(0),yvals)

def kfold_mlp(xvals, yvals, learning_rate, epoch_num, batch_num, hidden_units, splits):
    model = Sequential([Dense(hidden_units, activation='relu', name="hidden"), Dense(1, activation='sigmoid', name="output"),])
    model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate))
    kf = KFold(n_splits=splits, shuffle=True)
    arr = []
    for train_idxs, test_idxs in kf.split(xvals):
        xtrain_this_fold = xvals.loc[train_idxs]
        xtest_this_fold = xvals.loc[test_idxs]
        ytrain_this_fold = yvals.loc[train_idxs]
        ytest_this_fold = yvals.loc[test_idxs]
        # train a model on this fold
        model.fit(xtrain_this_fold, ytrain_this_fold, epochs=epoch_num, batch_size=batch_num, verbose=False)
        # test the model on this fold
        arr.append(f1_score(model.predict(xtest_this_fold, verbose=False).round(0), ytest_this_fold))
    return np.mean(arr)

def plot_mlp(xvalues, yvalues, learning_rate, epoch_num, batch_num, splits, units_start, units_stop):
    units = []
    whole_set = []
    k_folds = []

    i = units_start
    while i <= units_stop:
        units.append(i)
        whole_set.append(mlp(xvalues, yvalues, learning_rate, epoch_num, batch_num, i))
        k_folds.append(kfold_mlp(xvalues, yvalues, learning_rate, epoch_num, batch_num, i, splits))
        i = i * 2

    # the following code prints out the maximum f1_score and the unit at which it occurs
    i = whole_set.index(max(whole_set))
    print("max f1 whole_set:", whole_set[i], "unit:", units[i])
    i = k_folds.index(max(k_folds))
    print("max f1 k_folds:", k_folds[i], "unit:", units[i])

    plt.figure()
    plt.title("MLP: f1_score vs hidden units")
    plt.ylabel("f1_score")
    plt.xlabel("hidden units")
    plt.plot(units, np.transpose([whole_set,k_folds]))
    plt.legend(labels=["whole_set", "k_folds"])

#Testing:
#Note: c_start and c_stop are exponents (c_start = 0 means 10^0 = 1)
print("Figure 1: generating linear SVC vs C for the amphibian dataset")
plot_linear_SVC(xa, ya, splits=5, c_start=0, c_stop=5, num_points=50)
print("Figure 2: generating kernel SVC vs C for the amphibian dataset")
plot_kernel_SVC(xa, ya, splits=5, c_start=0, c_stop=5, num_points=50)
print("Figure 3: generating kernel SVC vs gamma for the amphibian dataset")
plot_kernel_SVC_gamma(xa, ya, splits=5, gamma_start=0, gamma_stop=5, num_points=50)

print("Figure 4: generating MLP vs hidden units for the amphibian dataset")
plot_mlp(xa, ya, learning_rate=0.001, epoch_num=100, batch_num=5, splits=5, units_start=1, units_stop=32)

print("Figure 5: generating linear SVC vs C for the heart disease dataset")
plot_linear_SVC(xh, yh, splits=10, c_start=0, c_stop=5, num_points=50)
print("Figure 6: generating kernel SVC vs C for the heart disease dataset")
plot_kernel_SVC(xh, yh, splits=10, c_start=0, c_stop=5, num_points=50)
print("Figure 7: generating kerel SVC vs gamma for the heart disease dataset")
plot_kernel_SVC_gamma(xh, yh, splits=10, gamma_start=0, gamma_stop=5, num_points=50)

plt.show()
