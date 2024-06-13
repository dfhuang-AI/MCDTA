# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:22:25 2023

@author: dfhuang
"""
import numpy as np
from math import sqrt
from sklearn import metrics
from scipy import stats

def mae(y, f):
    mae = metrics.mean_absolute_error(y, f)
    return mae

def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def pearson(y, f):
    rp = stats.pearsonr(y, f)[0]
    return rp

def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def r_squared(y, f):
    sse = np.sum((y - f) ** 2)
    ssr = np.sum((f - np.mean(y)) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst
    return r2

def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1,1))
    lr = LinearRegression().fit(y_pred,y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))

def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci