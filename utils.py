#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:56 2024

@author: Sakhawat
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score)
from pathlib import Path

# Utility function
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    hours_str = f"{int(hours):,}"
    return f"{hours_str:>5s}h {int(minutes):02d}m {int(seconds):02d}s"

def cache_to_file(filepath, val, fmt = "%.6f", **kwargs):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(val, pd.DataFrame) or isinstance(val, pd.Series):
        val.to_csv(filepath, header = None, sep = '\t', float_format = fmt, **kwargs)
    else:
        np.savetxt(filepath, np.array(val), fmt = fmt, **kwargs)
    
def save_feature_importances(clf, filename, as_sparse = True):
    feature_importances = None
    if isinstance(clf, LogisticRegression):
        feature_importances = clf.coef_.ravel()
    elif isinstance(clf, (RandomForestClassifier, xgb.XGBClassifier)):
        feature_importances = clf.feature_importances_
    else:
        try:
            feature_importances = clf.feature_importances_
        except:
            pass
    if feature_importances is None:
        return
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if as_sparse:
        scipy.sparse.save_npz(
            filename,
            scipy.sparse.csr_matrix(feature_importances))
    else:
        np.savetxt(filename + '.csv', feature_importances, fmt = '%.8f')
        
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


class ThresholdSelector(ABC):
    def __init__(self):
        self.threshold_labels = []
    
    def get_threshold_labels(self):
        return self.threshold_labels
    
    @abstractmethod
    def get_thresholds(self, X):
        pass
    
class FixedThresholdSelector(ThresholdSelector):
    '''Fixed thresholds for training local models'''
    def __init__(self, H = None):
        if H is None:
            H = np.arange(0.15, 0.25+0.025, 0.025).round(3)
        self.threshold_labels = H.astype(str)
        self.H = H
        
    def get_thresholds(self, as_series = False):
        if as_series:
            return pd.Series(self.H, index = self.get_threshold_labels())
        return self.H
    
class PercentileThresholdSelector(ThresholdSelector):
    '''Percentile-based thresholds for training local models'''
    def __init__(self, low_percentile = 65, high_percentile = 90, n_thresholds = 6):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.n_thresholds = n_thresholds
        self.percentiles = np.linspace(self.low_percentile, self.high_percentile, n_thresholds)
        self.threshold_labels = self.percentiles.round(3).astype(str)
        
    def get_thresholds(self, X, corr = None, as_series = False):
        if corr is None:
            corr = squareform(np.corrcoef(X), checks = False)
        H = np.array([np.percentile(np.abs(corr), p) for p in self.percentiles])
        if as_series: 
            return pd.Series(H, index = self.get_threshold_labels())
        return H