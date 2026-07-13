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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score)
from pathlib import Path

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
        
def scores_to_metrics(scores, y_true, p_threshold = 0.5):
    '''
    Calculate performance metrics from class probability scores (y_pred)
    
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
    for metric_name, metric_func in metrics.items():
        for cfg in metric_df.index:
            scores_cfg = scores.loc[cfg, :]
            y_pred = (scores_cfg >= p_threshold).astype(int)
            metric_df.loc[cfg, metric_name] = metric_func(y_true,
                                                          y_pred,
                                                          scores_cfg)
    return metric_df

class SampleSelector():
    
    def __init__(
        self, similarity_metric = 'pearson',
        use_absolute_similarity = True,
        underflow_resolution = None):

        self.similarity_metric = similarity_metric
        self.use_absolute_similarity = use_absolute_similarity
        self.underflow_resolution = underflow_resolution
    
    def fit(self, X_train, y_train, X_test):
        assert len(X_train) == len(y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        if self.similarity_metric == 'pearson':
            self.corr = np.corrcoef(X_test, X_train)[:len(X_test), len(X_test):]
        elif self.similarity_metric == 'spearman':
            self.corr = spearmanr(X_test.T, X_train.T)[0][:len(X_test), len(X_test):]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}" )
        return self

    def get_similar_sample_idx(self, h, j = 0, as_mask = False):
        if self.use_absolute_similarity:
            mask = np.abs(self.corr[j]) >= h
        else:
            mask = self.corr[j] >= h
        if as_mask:
            return mask
        idx = np.where(mask)[0]
        return idx
        
    # return samples from X_train that are similar to the j-th test sample
    def get_similar_samples(self, h, j = 0):
        mask = self.get_similar_samples(h, j, as_mask = True)
        X_train_h = self.X_train[mask, :]
        y_train_h = self.y_train[mask]
        return X_train_h, y_train_h   


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
        self.threshold_labels = [str(h) for h in H]
        self.H = H

    def get_thresholds(self, *args, as_series = False):
        if as_series:
            return pd.Series(self.H, index = self.get_threshold_labels())
        return self.H
    
class PercentileThresholdSelector(ThresholdSelector):
    '''Percentile-based thresholds for training local models'''
    def __init__(
        self,
        low_percentile = 65,
        high_percentile = 90,
        n_thresholds = 6,
        similarity_metric = 'pearson',
        use_absolute_similarity = True):

        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.n_thresholds = n_thresholds
        self.similarity_metric = similarity_metric
        self.use_absolute_similarity = use_absolute_similarity
        self.percentiles = np.linspace(self.low_percentile, self.high_percentile, n_thresholds)
        self.threshold_labels = self.percentiles.round(3).astype(str)
        
    def get_thresholds(self, X, corr = None, as_series = False):
        if corr is None:
            if self.similarity_metric == 'pearson':
                corr = squareform(np.corrcoef(X), checks = False)
            elif self.similarity_metric == 'spearman':
                corr = squareform(spearmanr(X)[0], checks = False)
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}" )
        if self.use_absolute_similarity:
            corr = np.abs(corr)
        H = np.array([np.percentile(corr, p) for p in self.percentiles])
        if as_series: 
            return pd.Series(H, index = self.get_threshold_labels())
        return H

