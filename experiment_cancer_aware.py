#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:57:55 2025

@author: Sakhawat
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat, mmread
from scipy.spatial.distance import squareform, correlation

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


from cifrus.cifrus import CiFRUS
from utils import (format_time, scores_to_metrics,
                   BaselineAugmentation, BaselineTransformation)
from dataloader import (load_single_cancer_datasets,
                        load_pan_cancer_datasets,
                        load_tcga)

import itertools
import time
from pathlib import Path
import sys


# For parallel execution, no need to change if running a single python instance
try:
    node_id, total_nodes = int(sys.argv[1]), int(sys.argv[2])
    print('New node with nnodes={}, offset={}'.format(total_nodes, node_id))
except:
    print('Running single node')
    total_nodes, node_id = 1, 0

# -----------------------------------------------------------------------------
# Experiment set-up
# -----------------------------------------------------------------------------

SEED = 2024
dry_run = False
n_splits = 10
classifier_names = ['LR', 'RF', 'XGB'] # ('LR', 'RF', 'XGB', 'KNN')
augmentation_type = ['Baseline', 'CiFRUS'] # ('Baseline', 'CiFRUS')
experiment_type = 'pan_cancer_stratified' # 'single_cancer' | 'pan_cancer' | 'pan_cancer_stratified'
use_cache = True

# -----------------------------------------------------------------------------
# End experiment set-up
# -----------------------------------------------------------------------------

classifier_map = {
                    'RF': lambda: RandomForestClassifier(random_state = SEED, n_jobs = -1),
                    'XGB': lambda: xgb.XGBClassifier(random_state = SEED, n_jobs = None),
                    'LR': lambda: LogisticRegression(random_state = SEED, n_jobs = -1),
                 }


basedir = "./results"
scores_basedir = f"{basedir}/pred_probability/{experiment_type}"

augmentation_map = {
                    'CiFRUS': CiFRUS(random_state = SEED),
                    'Baseline': BaselineAugmentation(random_state = SEED),
                   }
transform_map = {
                'PCA': PCA(n_components = 0.95),
                'Baseline': BaselineTransformation()
                }

load_datasets = load_pan_cancer_datasets

# for dataset_name, (X, y) in load_datasets():
for dataset_name, ds_obj in load_datasets():
    X, y = ds_obj.X, ds_obj.y
    c = ds_obj.cancer_type.reshape(-1, 1)
    c_encoded = OneHotEncoder(sparse_output = False).fit_transform(c)
    
    transform_type = ['Baseline']
    if dataset_name.startswith('TCGA'):
        transform_type = ['PCA']
    augmenters = {k: v for k, v in augmentation_map.items() if k in augmentation_type}
    feature_transformers = {k: v for k, v in transform_map.items() if k in transform_type}
    

    for classifier_name in classifier_names:
        print('Dataset: ', dataset_name)
        print('Classifier:', classifier_name)
        # -----------------------------------------------------------------------------
        get_classifier = classifier_map[classifier_name]

        dir_scores = Path(f"{scores_basedir}/{dataset_name}/{classifier_name}+")
        Path(dir_scores).mkdir(parents = True, exist_ok = True)
        skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = SEED)
        y_split = y
        if experiment_type == 'pan_cancer_stratified':
            y_split = np.array([str(v0) + '_' + str(v1) for v0, v1 in zip(ds_obj.cancer_type, y)])
        for fold, (train_index, test_index) in enumerate(skf.split(X, y_split)):
            if fold % total_nodes != node_id:
                continue
            print('===================================')
            print(f'Fold {fold}, Train/Test : {len(train_index)}, {len(test_index)}')
            print('===================================')
            
            
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            c_train, c_test = c_encoded[train_index], c_encoded[test_index]
            

            if dry_run:
                continue
            
            np.savetxt(f'{dir_scores}/test_index_fold_{fold}.csv', test_index, fmt = '%d', delimiter = '\n')
            
            scores = {}
            configs = itertools.product(augmenters.items(), feature_transformers.items())
            
            for i, cfg in enumerate(configs):
                (augmenter_name, augmenter), (transform_name, transformer) = cfg
                cfg_idx = (augmenter_name, transform_name)
            
                # Apply feature transformation
                X_train_trans = transformer.fit_transform(X_train)
                X_test_trans = transformer.transform(X_test)
                
                # Append one-hot encoded cancer types as additional features
                X_train_with_ctypes = np.hstack([X_train_trans, c_train])
                X_test_with_ctypes = np.hstack([X_test_trans, c_test])
                
                # Apply augmentation
                X_train_aug, y_train_aug = augmenter.fit_resample(X_train_with_ctypes, y_train,
                                                                  r = 3, balanced = True)
                
                # Fit classifier
                clf = get_classifier()
                clf.fit(X_train_aug, y_train_aug)
                
                # get predicted probability
                y_pred = augmenter.resample_predict_proba(clf.predict_proba, X_test_with_ctypes)[:, 1]
                scores[(augmenter_name, transform_name, classifier_name + '+')] = y_pred
            scores = pd.DataFrame(scores).T
            scores = scores.astype(float).round(6)
            scores.to_csv(f'{dir_scores}/scores_fold_{fold}.csv',
                          header = None, sep = '\t', float_format = '%.6f')
            
            
          
