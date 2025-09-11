#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:56 2024

@author: Sakhawat
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             f1_score,
                             balanced_accuracy_score,
                             cohen_kappa_score)

# Utility function
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):,}h {int(minutes)}m {int(seconds)}s"

# Utility function
def scores_to_metrics(scores, y_true, p_threshold = 0.5):
    '''
    
    Parameters
    ----------
    scores : DataFrame with t rows, n columns
        Each row corresponds to a model config. Each column is a sample.
    y_true : Array with n elements
        True class labels.
    p_threshold : int, optional
        The threshold applied to scores to get the predicted label. The default is 0.5.

    Returns
    -------
    metric_df : DataFrame with t rows, len(metrics) columns
        Each row corresponds to a model config. Each column is a performance metric.

    '''
    
    metrics = {'AUC': lambda y_true, y_pred, scores: \
                           roc_auc_score(y_true, scores),
               'F1': lambda y_true, y_pred, scores: \
                           f1_score(y_true, y_pred),
               'Kappa': lambda y_true, y_pred, scores: \
                           cohen_kappa_score(y_true, y_pred),
               'Balanced accuracy': lambda y_true, y_pred, scores: \
                           balanced_accuracy_score(y_true, y_pred)}
    
    metric_df = pd.DataFrame(index = scores.index,
                             columns = metrics.keys(),
                             dtype = float)
    #y_true = scores['Y_true']
    for metric_name, metric_func in metrics.items():
        for cfg in metric_df.index:
            scores_cfg = scores.loc[cfg, :]
            #print(1/0)
            y_pred = (scores_cfg >= p_threshold).astype(int)
            metric_df.loc[cfg, metric_name] = metric_func(y_true,
                                                          y_pred,
                                                          scores_cfg)
    #metric_df = metric_df.drop('Y_true')
    return metric_df

# Baseline classes

class BaselineAugmentation():
    """
    This class implements identity augmentation (baseline)
    """
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X_train, **kwargs):
        pass
    
    def resample(self, X, Y = None, **kwargs):
        if Y is None:
            return X
        return X, Y
    
    def fit_resample(self, X, Y, **kwargs):
        return self.resample(X, Y)
    
    def resample_predict_proba(self, func_predict_proba, X, **kwargs):
        return func_predict_proba(X)

class BaselineTransformation:
    """
    This class implements identity feature transformation (baseline)
    """
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X_train, **kwargs):
        pass
    
    def fit_transform(self, X, **kwargs):
        return X
    
    def transform(self, X, **kwargs):
        return X
    