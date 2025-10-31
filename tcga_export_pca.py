#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:40:45 2025

@author: Sakhawat
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from cifrus.cifrus import CiFRUS
from pathlib import Path

from dataloader import load_tcga

# For MATLAB external use
def get_tcga_stratified_split(fold,ds_obj, n_splits = 10, pca = True, augment = False, random_state = 2024):
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    X, y, cancer_type = ds_obj.X, ds_obj.y, ds_obj.cancer_type
    y_split = np.array([str(v0) + '_' + str(v1) for v0, v1 in zip(ds_obj.cancer_type.values, y)])
    train_index, test_index = list(skf.split(X, y_split))[int(fold)]
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    ctype_train, ctype_test = cancer_type[train_index], cancer_type[test_index]
    
    X_train_trans = X_train
    X_test_trans = X_test
    if pca:
        pca_obj = PCA(n_components = 0.95)
        X_train_trans = pca_obj.fit_transform(X_train)
        X_test_trans = pca_obj.transform(X_test)
        
    X_train_aug = X_train_trans
    y_train_aug = y_train
    if augment:
        X_train_aug, y_train_aug = (CiFRUS(random_state = random_state)
                                    .fit_resample(X_train_trans,
                                                  y_train,
                                                  r = 3,
                                                  balanced = True))
    return X_train_aug, y_train_aug, X_test_trans, y_test, ctype_train, ctype_test

def write_split_files(n_splits = 10, pca = True, augment = False, random_state = 2024):
    data_dir = './datasets/TCGA/exported'
    ds_obj = load_tcga()
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    X, y, cancer_type = ds_obj.X, ds_obj.y, ds_obj.cancer_type
    y_split = np.array([str(v0) + '_' + str(v1) for v0, v1 in zip(ds_obj.cancer_type.values, y)])
    for fold, (train_index, test_index) in enumerate(skf.split(X, y_split)):
        fold_dir = Path(f'{data_dir}/fold_{fold}')
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        ctype_train, ctype_test = cancer_type[train_index], cancer_type[test_index]
        
        X_train_trans = X_train
        X_test_trans = X_test
        if pca:
            pca_obj = PCA(n_components = 0.95)
            X_train_trans = pca_obj.fit_transform(X_train)
            X_test_trans = pca_obj.transform(X_test)
            
        X_train_aug = X_train_trans
        y_train_aug = y_train
        if augment:
            X_train_aug, y_train_aug = (CiFRUS(random_state = random_state)
                                        .fit_resample(X_train_trans,
                                                      y_train,
                                                      r = 3,
                                                      balanced = True))
        #return X_train_aug, y_train_aug, X_test_trans, y_test, ctype_train, ctype_test
        #X_train, y_train, X_test, y_test, ctype_train, ctype_test = get_tcga_stratified_split(fold, ds_obj)
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'{fold_dir}/X_train.csv', X_train_aug, delimiter = '\t')
        np.savetxt(f'{fold_dir}/y_train.csv', y_train_aug, delimiter = '\t')
        np.savetxt(f'{fold_dir}/X_test.csv', X_test_trans, delimiter = '\t')
        np.savetxt(f'{fold_dir}/y_test.csv', y_test, delimiter = '\t')
        np.savetxt(f'{fold_dir}/ctype_train.csv', ctype_train, fmt = "%s", delimiter = '\t')
        np.savetxt(f'{fold_dir}/ctype_test.csv', ctype_test, fmt = "%s", delimiter = '\t')
        
if __name__ == "__main__":   
    write_split_files()