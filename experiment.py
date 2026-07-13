#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:39:07 2026

@author: Sakhawat
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from cisampling.samplers import CiFRUS, IdentitySampler
from utils import (format_time, cache_to_file, save_feature_importances,
                   SampleSelector,
                   FixedThresholdSelector, PercentileThresholdSelector)
from dataloader import (load_single_cancer_datasets,
                        load_pan_cancer_datasets)
import time
from pathlib import Path
import sys


def train_models(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold_selector,
        feature_transformer,
        augmenter,
        classifier_fn,
        sample_selector,
        cache_dir,
        runtime_dir,
        data_dir,
        model_attr_dir,
        augmentation_sampling_rate = 3,
        ctypes_train = None,
        use_cache = True,
        progress_text_prefix = ""
    ):

    n_samples_progress = min(len(X_train) // 10, 20)
    if len(X_test.shape) < 2:
        X_test = X_test.reshape([1, len(X_test)])
        
    baseline_colname = 'Baseline'
    y_pred = pd.DataFrame(
        index = range(len(X_test)),
        columns = [baseline_colname] + list(threshold_selector.get_threshold_labels()),
        dtype = float)
    
    # Initialize the baseline classifier to be blank (will be lazy-loaded later)
    clf_base = None
    X_train_trans = None
    X_train_aug = None

    starttime = time.time()
    first_noncached_sample = True
    for j in range(len(X_test)):
  
        # If available and preferred, load cached results
        path_y_pred = Path(cache_dir, f'sample_{j:04}.txt')
        
        if use_cache and path_y_pred.exists():
            try:
                yj_pred = pd.read_csv(path_y_pred, header = None, index_col = 0, sep = '\t')
                y_pred.loc[j, yj_pred.index] = yj_pred.values[:, 0]
                continue
            except:
                print(f'{progress_text_prefix}\t\t'
                      'Invalid cache:', str(path_y_pred))
                
        path_y_test = Path(data_dir, 'y_test.csv')
        cache_to_file(path_y_test, y_test, '%d', delimiter = '\n')
        
        # Feature transformation (cached, lazy)
        if X_train_trans is None:
            path_X_train_trans = Path(data_dir, 'X_train_transformed.csv')
            path_X_test_trans = Path(data_dir, 'X_test_transformed.csv')
            try:
                if not use_cache:
                    raise Exception()
                X_train_trans = np.loadtxt(path_X_train_trans)
                X_test_trans = np.loadtxt(path_X_test_trans)
            except:
                X_train_trans = feature_transformer.fit_transform(X_train)
                X_test_trans = feature_transformer.transform(X_test)
                cache_to_file(path_X_train_trans, X_train_trans, '%.8f')
                cache_to_file(path_X_test_trans, X_test_trans, '%.8f')
            print(f'{progress_text_prefix}\t\t'
                  f'Transformed: {X_train_trans.shape}')            
            H = threshold_selector.get_thresholds(X_train_trans, as_series = True)
            path_H = Path(data_dir, 'H.csv')
            cache_to_file(path_H, H)
            print(f'{progress_text_prefix}\t\t'
                  'Thresholds :', H.values.round(3))
        xj = X_test_trans[j:j+1]

        # Augmentation (cached, lazy)
        if X_train_aug is None:
            path_X_train_aug = Path(data_dir, 'X_train_augmented.csv')
            path_y_train_aug = Path(data_dir, 'y_train_augmented.csv')
            X_train_aug, y_train_aug, info = augmenter.fit_resample(
                X_train_trans,
                y_train,
                r = augmentation_sampling_rate,
                balanced = True,
                shuffle = False,
                return_info = True
            )
            if ctypes_train is not None:
                path_ctypes_train_aug = Path(data_dir, 'ctypes_train.csv')
                ctypes_train_aug = ctypes_train[info.iloc[:, 0].values]
                assert len(ctypes_train_aug) == len(X_train_aug)
                cache_to_file(path_ctypes_train_aug, ctypes_train_aug, '%s', delimiter = '\n')
                
            cache_to_file(path_X_train_aug, X_train_aug, '%.8f')
            cache_to_file(path_y_train_aug, y_train_aug, '%d', delimiter = '\n')
            print(f'{progress_text_prefix}\t\t'
                  f'Augmented  : {X_train_aug.shape}')
        path_xj_aug = Path(data_dir, 'X_test_augmented', f'X_test_augmented_{j:04}.csv')
        try:
            if not use_cache:
                raise Exception()
            xj_aug = np.loadtxt(path_xj_aug).reshape([-1, X_train_aug.shape[1]])
        except:
            xj_aug = augmenter.resample(xj, balanced = False)
            cache_to_file(path_xj_aug, xj_aug)
        if clf_base is None:
            # Fit baseline classifier
            clf_base = classifier_fn()
            t1_baseline_fit = time.time()
            clf_base.fit(X_train_aug, y_train_aug)
            t2_baseline_fit = time.time()
            print(f'{progress_text_prefix}\t\t'
                  'Fitted baseline classifier')
            runtime_path = Path(runtime_dir, 'runtime_baseline_train.txt')
            cache_to_file(runtime_path, [t2_baseline_fit - t1_baseline_fit])
            
            # Save model attributes
            model_attr_path = Path(model_attr_dir, 'model_base')
            save_feature_importances(clf_base, model_attr_path)
             
        t1_baseline_predict = time.time()
        pred_proba = clf_base.predict_proba(xj_aug)
        t2_baseline_predict = time.time()
        runtime_path = Path(runtime_dir, 'runtime_baseline_predict.txt')
        cache_to_file(runtime_path, [t2_baseline_predict - t1_baseline_predict])
        
        y_pred.loc[j, baseline_colname] = pred_proba.mean(axis = 0)[1]
        
        # progress output
        if j % n_samples_progress == 0 or first_noncached_sample:
            endtime = time.time()
            elapsed = format_time(endtime - starttime)
            first_noncached_sample = False
            print(
                f"{progress_text_prefix}\t\t"
                f"Test sample {str(j+1):>4s}/{str(len(X_test)):>4s},  "
                f"elapsed {elapsed}")
            
        # select patients based on thresholds and train additional models
        # not applicable for KNN
        if not isinstance(clf_base, KNeighborsClassifier):
            runtime_local_predict = 0
            is_fitted_sample_selector = False
            for h_label, h in H.reset_index().values:
                
                # Step 1: find similar samples (cached for faster execution)
                path_similar_sample_idx = Path(data_dir, 'local_set_index', h_label, f'local_set_idx_sample_{j:04}.csv')
                try:
                    if not use_cache:
                        raise Exception()
                    similar_sample_idx = np.loadtxt(path_similar_sample_idx).astype(int)
                except:
                    path_runtime_similarity = Path(data_dir, 'runtime_local_set', h_label, f'runtime_local_set_sample_{j:04}.csv')
                    t1_similarity = time.time()
                    if not is_fitted_sample_selector:
                        sample_selector.fit(X_train_aug, y_train_aug, xj)
                        is_fitted_sample_selector = True
                    similar_sample_idx = sample_selector.get_similar_sample_idx(h)
                    t2_similarity = time.time()
                    cache_to_file(path_runtime_similarity, [t2_similarity - t1_similarity])
                    cache_to_file(path_similar_sample_idx, similar_sample_idx, '%d')
                X_train_h, y_train_h = X_train_aug[similar_sample_idx], y_train_aug[similar_sample_idx]
                y_train_h_counts = {k: v for k, v in zip(*np.unique(y_train_h, return_counts = True))}

                # Step 2: train local models
                t1_local_predict = time.time()
                if np.unique(y_train_h).shape[0] < 2:
                    # traning set contains too few samples (depletion), use baseline as placeholder
                    print(
                        f'{progress_text_prefix}\t\t'
                        f'Depletion: sample idx={j}, '
                        f'h={h:.3f}, label counts={y_train_h_counts}')
                    pred_proba = clf_base.predict_proba(xj_aug)
                elif len(y_train_h) == len(y_train_aug):
                    # traning set contains all samples (saturation), local model is the same as baseline
                    pred_proba = clf_base.predict_proba(xj_aug)
                else:     
                    # sufficient samples to train h-model
                    clf = classifier_fn()
                    clf.fit(X_train_h, y_train_h)
                    model_attr_path = Path(model_attr_dir, f'model_sample_{j:04}_h_{h_label}')
                    save_feature_importances(clf, model_attr_path)
                    pred_proba = clf.predict_proba(xj_aug)
                t2_local_predict = time.time()
                runtime_local_predict += t2_local_predict - t1_local_predict
                y_pred.loc[j, str(h_label)] = pred_proba.mean(axis = 0)[1]
            runtime_path = Path(runtime_dir, f'runtime_mcs_predict_sample_{j:04}.txt')
            cache_to_file(runtime_path, [runtime_local_predict])
        else:
            y_pred = y_pred.iloc[:, [0]]
        y_pred = y_pred.astype(float).round(6)
        cache_to_file(path_y_pred, y_pred.loc[j])


if __name__ == "__main__":
    # For parallel execution, no need to change if running a single python instance
    try:
        total_nodes, node_id = int(sys.argv[1]), int(sys.argv[2])
        print('New node with node_id={} ({} total)'.format(node_id, total_nodes))
    except:
        print('Running single node')
        total_nodes, node_id = 1, 0
    
    # =========================================================================
    # Experiment configuration
    # =========================================================================
    SEED = 2024
    dry_run = False
    n_splits = 10
    classifier_names = ['LR', 'RF', 'XGB', 'MLP', 'KNN'] # ('LR', 'RF', 'XGB', 'MLP', 'KNN')
    augmentation_type = ['Baseline', 'CiFRUS'] # ('Baseline', 'CiFRUS')
    experiment_type = 'pan_cancer_stratified' # 'single_cancer' | 'pan_cancer' | 'pan_cancer_stratified'
    threshold_selection = 'Fixed' # 'Percentile' | 'Fixed'
    similarity_metric = 'pearson' # 'pearson' | 'spearman'
    use_absolute_similarity = True
    basedir = "./results"
    use_cache = True
    cancer_aware = False
    # =========================================================================
    # End experiment configuration
    # =========================================================================
    
    classifier_map = {
        'RF': lambda: RandomForestClassifier(random_state = SEED, n_jobs = -1),
        'XGB': lambda: xgb.XGBClassifier(random_state = SEED, n_jobs = None),
        'LR': lambda: LogisticRegression(random_state = SEED, n_jobs = -1),
        'MLP': lambda: MLPClassifier(),
        'KNN': lambda: KNeighborsClassifier(n_neighbors = 40, metric = correlation)
    }
    
    augmentation_map = {
        'Baseline': IdentitySampler(random_state = SEED),
        'CiFRUS': CiFRUS(random_state = SEED),
    }
    transform_map = {
        'PCA': PCA(n_components = 0.95),
        'Baseline': FunctionTransformer()
    }
    threshold_selector_map = {
        'Fixed': FixedThresholdSelector(H = np.arange(0.15, 0.25+0.025, 0.025).round(3)),
        'Percentile': PercentileThresholdSelector(65, 90, 6, similarity_metric, use_absolute_similarity)
    }
    sample_selector = SampleSelector(
        similarity_metric = similarity_metric,
        use_absolute_similarity = use_absolute_similarity)
    
    if experiment_type == 'single_cancer':
        load_datasets = load_single_cancer_datasets
    elif experiment_type.startswith('pan_cancer'):
        load_datasets = load_pan_cancer_datasets
    else:
        print('Invalid experiment type')
        load_datasets = lambda: None
    
    progress_text_prefix = "[Node {:}]".format(node_id)
    for dataset_name, ds_obj in load_datasets():
        X, y, ctypes = ds_obj.X, ds_obj.y, ds_obj.cancer_type
        n, m = X.shape[0], X.shape[1]
        transform_type = ['PCA']
        if dataset_name.startswith('TCGA'):
            transform_type = ['PCA']
    
        augmenters = {k: v for k, v in augmentation_map.items() if k in augmentation_type}
        feature_transformers = {k: v for k, v in transform_map.items() if k in transform_type}
        
        threshold_selector = threshold_selector_map[threshold_selection]
        skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = SEED)
        y_split = y
        if experiment_type == 'pan_cancer_stratified':
            y_split = np.array([str(v0) + '_' + str(v1) for v0, v1 in zip(ds_obj.cancer_type, y)])
        
        if cancer_aware:
            ctypes_encoded = OneHotEncoder(sparse_output = False).fit_transform(ctypes.reshape(-1, 1))
            classifier_fns = {k + '+': classifier_map[k] for k in classifier_names}
            X = np.hstack([X, ctypes_encoded])
            # only train baseline models for cancer-aware X
            threshold_selector = FixedThresholdSelector([])
        else:
            classifier_fns = {k: classifier_map[k] for k in classifier_names}
                        
        for fold, (train_index, test_index) in enumerate(skf.split(X, y_split)):
            if fold % total_nodes != node_id:
                continue
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test =  y[train_index], y[test_index]
            ctypes_train, ctypes_test = None, None
            if experiment_type.startswith('pan_cancer'):
                ctypes_train, ctypes_test = ctypes[train_index], ctypes[test_index]
            n_train, n_test = len(X_train), len(X_test)
            
            for i_trans, (transform_name, feature_transformer) in enumerate(feature_transformers.items()):
                for i_aug, (augmenter_name, augmenter) in enumerate(augmenters.items()):
                    for i_clf, (classifier_name, classifier_fn) in enumerate(classifier_fns.items()):
                        # -----------------------------------------------------
                        header_len = 60
                        print('='*header_len)
                        print('{:} {:<15s}: {:} {:}'.format(
                            progress_text_prefix,
                            'Dataset', dataset_name, X.shape))
                        print('{:} {:<15s}: {:}{:}'.format(
                            progress_text_prefix, 'Similarity',
                            similarity_metric, (' (signed)' if not use_absolute_similarity else '')))
                        print('{:} {:<15s}: {:}/{:}'.format(
                            progress_text_prefix, 'Fold',
                            fold+1, n_splits))
                        print('{:} {:<15s}: {:}/{:}'.format(
                            progress_text_prefix, 'Train/Test',
                            n_train, n_test))
                        print('{:} {:<15s}: {:} ({:}/{:})'.format(
                            progress_text_prefix, 'Transform', transform_name,
                            i_trans+1, len(feature_transformers)))
                        print('{:} {:<15s}: {:} ({:}/{:})'.format(
                            progress_text_prefix, 'Augment', augmenter_name,
                            i_aug+1, len(augmenters)))
                        print('{:} {:<15s}: {:} ({:}/{:})'.format(
                            progress_text_prefix, 'Classifier', classifier_name,
                            i_clf+1, len(classifier_fns)))
                        print('-'*header_len)
                        # -----------------------------------------------------
                        
                        experiment_dir = Path(basedir, experiment_type, dataset_name)
                        cache_dir = Path(
                            experiment_dir,
                            'pred_probability', transform_name, augmenter_name,
                            classifier_name, f'fold_{fold}')
                        runtime_dir = Path(
                            experiment_dir,
                            'training_times', transform_name, augmenter_name,
                            classifier_name, f'fold_{fold}')
                        data_dir = Path(
                            experiment_dir,
                            'data', transform_name, augmenter_name,
                            f'fold_{fold}')
                        model_attr_dir = Path(
                            experiment_dir,
                            'model_attr', transform_name, augmenter_name,
                            classifier_name, f'fold_{fold}')
                        if dry_run:
                            continue
                        if experiment_type.startswith('pan_cancer'):
                            cache_to_file(Path(data_dir, 'ctypes_test.csv'),
                                          ctypes_test, '%s', delimiter = '\n')
                        train_models(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            threshold_selector,
                            feature_transformer,
                            augmenter,
                            classifier_fn,
                            sample_selector,
                            cache_dir,
                            runtime_dir,
                            data_dir,
                            model_attr_dir,
                            ctypes_train = ctypes_train,
                            use_cache = use_cache,
                            progress_text_prefix = progress_text_prefix)
    print('{:} completed.'.format(progress_text_prefix))





