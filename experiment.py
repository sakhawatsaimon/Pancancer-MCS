#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:01:57 2024

@author: Sakhawat, Tanzira
"""

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import correlation

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


from cifrus.cifrus import CiFRUS
from utils import (format_time,
                   BaselineAugmentation, BaselineTransformation)
from dataloader import (load_single_cancer_datasets,
                        load_pan_cancer_datasets)

import itertools
import time
from pathlib import Path
import sys


class SampleSelector():
    
    def __init__(self, underflow_resolution = None):
        self.underflow_resolution = underflow_resolution
    
    def fit(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.corr = np.corrcoef(X_test, X_train)[:len(X_test), len(X_test):]
        return self

        
    # return samples from X_train that are similar to the j-th test sample
    def get_similar_samples(self, j, h):
        mask = np.abs(self.corr[j]) >= h
        X_train_h = self.X_train[mask, :]
        y_train_h = self.y_train[mask]
        if len(np.unique(y_train_h)) < 2:
            counts = {k: v for k, v in zip(*np.unique(y_train_h, return_counts = True))}
            print(f'\t\tDepletion: {h=:.3f}, {counts=}')
            if not self.underflow_resolution:
                pass
        if len(X_train_h) == len(self.X_train):
            print(f'\t\tSaturation: {h=:.3f}, min correlation = {self.corr[j].min()}')
        return X_train_h, y_train_h
    
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
    if as_sparse:
        scipy.sparse.save_npz(filename,
                              scipy.sparse.csr_matrix(feature_importances))
    else:
        np.savetxt(filename + '.csv', feature_importances, fmt = '%.8f')
        

# Function that will do most of the heavy lifting
def evaluate_mcs(X_train,
                 y_train,
                 X_test,
                 y_test,
                 get_classifier,
                 augmentation,
                 transform,
                 cache_dir,
                 model_attr_dir,
                 use_cache = True,
                 progress_text_prefix = ""):
    global result_cfg
    H = np.arange(0.15, 0.25+0.025, 0.025).round(3)

    configs = itertools.product(augmentation.items(), transform.items())
    n_samples_progress = min(len(X_train) // 10, 20)
    # If x_test is not 2D, make it 2D
    if len(X_test.shape) < 2:
        X_test = X_test.reshape([1, len(X_test)])
        
    scores = {}
    
    for i, cfg in enumerate(configs):
        (augmenter_name, augmenter), (transform_name, transformer) = cfg
        cfg_idx = (augmenter_name, transform_name)
        score_df = pd.DataFrame(index = range(len(X_test)),
                                columns = ['0 (baseline)'] + list(map(str, H)),
                                dtype = float)
        
        print(f'Augmentation:   {augmenter_name}')
        print(f'Transformation: {transform_name}')
        print(f'\tTrain size : {X_train.shape}')
        
        # Initialize the baseline classifier to be blank (will be lazy-loaded later)
        clf_base = None

        starttime = time.time()
        for j in range(len(X_test)):
            # progress output
            if j % n_samples_progress == 0:
                endtime = time.time()
                elapsed = format_time(endtime - starttime)
                print(f"\t\t{progress_text_prefix}: Test sample {j+1}/{len(X_test)}\t[{elapsed}]")
      
            # If available and preferred, load cached results
            cache_path = Path(f'{cache_dir}/{augmenter_name}_{transform_name}_sample_{j}.txt')
            if use_cache and cache_path.exists():
                try:
                    score_row = pd.read_csv(cache_path, header = None, index_col = 0, sep = '\t')
                    score_df.loc[j, :] = score_row.loc[score_df.columns].values[:, 0]
                    continue
                except:
                    print('\t\tInvalid cache:', str(cache_path))
                    print(f'{cfg_idx=}')

            # Lazy initialization
            if clf_base is None:
                # Transform (e.g. dim reduction)
                X_train_trans = transformer.fit_transform(X_train)
                X_test_trans = transformer.transform(X_test)
                print(f'\tTransformed: {X_train_trans.shape}')
                
                # Augment
                X_train_aug, y_train_aug = augmenter.fit_resample(X_train_trans, y_train, r = 3, balanced = True)
                print(f'\tAugmented  : {X_train_aug.shape}')
                
                # calculate sample-sample similarity
                sample_selector = SampleSelector().fit(X_train_aug, y_train_aug, X_test_trans)
                print("\tFitted sample selector")
                # Fit baseline classifier
                clf_base = get_classifier()
                clf_base.fit(X_train_aug, y_train_aug)
                print(f"\tFitted baseline classifier: {clf_base.__class__.__name__}")
            model_attr_path = Path(f'{model_attr_dir}/{augmenter_name}_{transform_name}_sample_{j}_h_0')
            model_attr_path.parent.mkdir(parents=True, exist_ok=True)
            save_feature_importances(clf_base, model_attr_path)
            xt = X_test_trans[j:j+1]
            score_df.loc[j, '0 (baseline)'] = augmenter.resample_predict_proba(clf_base.predict_proba,
                                                                               xt)[:, 1]

            # select patients based on thresholds and train additional models
            # not applicable for KNN
            if not isinstance(clf_base, KNeighborsClassifier):
                for h in H:
                    X_train_h, y_train_h = sample_selector.get_similar_samples(j, h)   
                    model_attr_path = Path(f'{model_attr_dir}/{augmenter_name}_{transform_name}_sample_{j}_h_{h}')
                    model_attr_path.parent.mkdir(parents=True, exist_ok=True)
                    if np.unique(y_train_h).shape[0] < 2:
                        # traning set contains too few samples (underflow), use baseline as placeholder
                        score_df.loc[j, str(h)] = augmenter.resample_predict_proba(clf_base.predict_proba,
                                                                                   xt)[0, 1]
                    else:     
                        # no underflow, train h-model
                        clf = get_classifier()
                        clf.fit(X_train_h, y_train_h)
                        save_feature_importances(clf, model_attr_path)
                        score_df.loc[j, str(h)] = augmenter.resample_predict_proba(clf.predict_proba,
                                                                                   xt)[0, 1]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            score_df = score_df.astype(float).round(6)
            score_df.loc[j].to_csv(cache_path, header = None, sep = '\t', float_format = '%.6f')
        if not isinstance(clf_base, KNeighborsClassifier):
            score_df['MCS'] = score_df.mean(axis = 1)
        scores[cfg_idx] = score_df
    scores = pd.concat(scores, names = ['augmentation', 'transformation', 'idx'])
    return scores

if __name__ == "__main__":
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
    classifier_names = ['LR', 'RF', 'MLP', 'XGB'] # ('LR', 'RF', 'XGB', 'KNN')
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
                        'MLP': lambda: MLPClassifier(n_jobs = -1),
                        'KNN': lambda: KNeighborsClassifier(n_neighbors = 40, metric = correlation)
                     }
    
    
    basedir = "./results"
    scores_basedir = f"{basedir}/pred_probability/{experiment_type}"
    model_attr_basedir = f"{basedir}/feature_scores/{experiment_type}"
    cache_basedir = f"{basedir}/cache/{experiment_type}"
    
    augmentation_map = {
                        'CiFRUS': CiFRUS(random_state = SEED),
                        'Baseline': BaselineAugmentation(random_state = SEED),
                       }
    transform_map = {
                    'PCA': PCA(n_components = 0.95),
                    'Baseline': BaselineTransformation()
                    }
    
    if experiment_type == 'single_cancer':
        load_datasets = load_single_cancer_datasets
    elif experiment_type.startswith('pan_cancer'):
        load_datasets = load_pan_cancer_datasets
    else:
        print('Invalid experiment type')
        load_datasets = lambda: None
    
    # for dataset_name, (X, y) in load_datasets():
    for dataset_name, ds_obj in load_datasets():
        X, y = ds_obj.X, ds_obj.y
        transform_type = ['Baseline']
        if dataset_name.startswith('TCGA'):
            transform_type = ['PCA']
    
        augmenters = {k: v for k, v in augmentation_map.items() if k in augmentation_type}
        feature_transformers = {k: v for k, v in transform_map.items() if k in transform_type}
        assert len(augmenters) > 0
        assert len(feature_transformers) > 0
    
        for classifier_name in classifier_names:
            print('Dataset: ', dataset_name)
            print('Augmentation Configs:', list(augmenters.keys()))
            print('Transform Configs:', list(feature_transformers.keys()))
            print('Classifier:', classifier_name)
            # -----------------------------------------------------------------------------
            get_classifier = classifier_map[classifier_name]
            dir_scores = Path(f"{scores_basedir}/{dataset_name}/{classifier_name}")
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
                cache_dir = Path(f'{cache_basedir}/{dataset_name}/{classifier_name}/fold_{fold}/')
                model_attr_dir = Path(f'{model_attr_basedir}/{dataset_name}/{classifier_name}/fold_{fold}/')
                np.savetxt(f'{dir_scores}/test_index_fold_{fold}.csv', test_index, fmt = '%d', delimiter = '\n')
                if dry_run:
                    continue
                scores_fold = evaluate_mcs(X_train,
                                           y_train,
                                           X_test,
                                           y_test,
                                           get_classifier,
                                           augmenters,
                                           feature_transformers,
                                           cache_dir,
                                           model_attr_dir,
                                           use_cache = use_cache,
                                           progress_text_prefix = "Node {:}".format(node_id))
                scores_fold = scores_fold.stack().unstack(level = -2)
                scores_fold.T.astype(float).to_csv(f'{dir_scores}/scores_fold_{fold}.csv',
                                                   sep = '\t', float_format = '%.6f')
