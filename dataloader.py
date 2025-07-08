#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:43:06 2024

@author: Sakhawat
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from cifrus.cifrus import CiFRUS

''' Wrapper class for data'''
class DataSet():
    def __init__(self, X = None, y = None, cancer_type = None, feature_info = None):
        self.X = X
        self.y = y
        self.cancer_type = cancer_type
        self.feature_info = feature_info

def get_tfs():
    url = 'https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
    human_tfs = pd.read_csv(url, sep = '\t', usecols=(1, 2, 4, 5, 11))
    human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
    human_tfs = human_tfs.iloc[:, [0, 1, 4]]
    human_tfs['EntrezGene ID'] = human_tfs['EntrezGene ID'].astype(float)
    return human_tfs

def load_aces():
    basepath = './datasets/ACES/'
    X  = loadmat(basepath + 'ACESExpr.mat')['data']
    y = loadmat(basepath + 'ACESLabel.mat')['label'].squeeze()
    entrez_ids = loadmat(basepath + 'ACES_EntrezIds.mat')['entrez_ids']
    X = pd.DataFrame(X)
    X.columns = entrez_ids.reshape(-1)
    return DataSet(X.values, y, cancer_type = 'ACES_BRCA', feature_info = entrez_ids)

def load_nki():
    basepath = './datasets/NKI/'
    nki_raw = loadmat(basepath + 'vijver.mat')['vijver']
    nki_p_type = loadmat(basepath + 'VijverLabel.mat')['label']
    nki_entrez_id = loadmat(basepath + 'vijver_gene_list.mat')['vijver_gene_list']
    nki_data = pd.DataFrame(nki_raw)
    nki_data.columns = nki_entrez_id.reshape(-1)
    #getting the tfs
    #human_tfs = get_tfs()
    #get common tfs with the expression data
    #common_tf = np.intersect1d(nki_data.columns, human_tfs.index)
    #get index of these tfs for each data
    #tf_locs = [nki_data.columns.get_loc(c) for c in common_tf]
    
    return DataSet(nki_data.values, nki_p_type.ravel(), cancer_type = 'NKI_BRCA')

def load_metabric():
    '''Reading METABRIC data'''
    basepath = 'datasets/METABRIC/'
    metabric_raw = pd.read_csv(basepath + 'data_mrna_illumina_microarray.txt', sep = '\t')
    metabric_raw = metabric_raw.drop(['Hugo_Symbol'], axis = 1)
    # Some entrez IDs are duplicated, use the average for those genes
    metabric_data = metabric_raw.groupby('Entrez_Gene_Id').mean().T

    metabric_p_type = pd.read_csv(basepath + 'data_clinical_patient.txt', sep = '\t', skiprows = 4, index_col = 0)
    metabric_p_type = metabric_p_type[['OS_MONTHS', 'OS_STATUS', 'RFS_MONTHS', 'RFS_STATUS']]
    metabric_p_type['label'] = 2
    metabric_p_type.loc[metabric_p_type['RFS_MONTHS'] >= 60, 'label'] = 0
    metabric_p_type.loc[(metabric_p_type['RFS_MONTHS'] < 60) & (metabric_p_type['RFS_STATUS'] == '1:Recurred'), 'label'] = 1
    metabric_p_type = metabric_p_type[metabric_p_type['label'] < 2]['label']
    metabric_p_type = metabric_p_type.loc[np.intersect1d(metabric_p_type.index, metabric_data.index)]
    metabric_data = metabric_data.loc[metabric_p_type.index, :]
    metabric_p_type = metabric_p_type.astype(int).values
    # drop genes with nan values
    metabric_data = metabric_data.loc[:, (np.isnan(metabric_data).sum(axis = 0) == 0).values]
    return DataSet(metabric_data.values, metabric_p_type.ravel(), cancer_type = 'METABRIC_BRCA')

def load_tcga(return_cancer_types = False):
    basepath = './datasets/TCGA/'
    filename_expr = 'EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz'
    filename_survival = 'Survival_SupplementalTable_S1_20171025_xena_sp'
    #reading the survival data file
    survival = pd.read_csv(basepath + filename_survival, sep = '\t', index_col = 0)
    #creating samply type id from sample id 
    survival['sample_type_id'] = survival.index.str.split('-').str[-1]

    # remove certain cancers and normal samples
    removable_cancers = ['LAML','KIRC','DLBC','PCPG','CHOL','UCS']
    mask = ~survival['cancer type abbreviation'].isin(removable_cancers) \
           & ~survival['sample_type_id'].isin(['11', '06', '02', '05', '03', '07']) 
    survival = survival.loc[mask.values, :]
    survival['PFI'] = survival['PFI'].astype(int)
    # PFI.time for censored data
    survival['new_tumor_event_type'] = survival['new_tumor_event_type'].fillna(value = 'None')

    qth = 0.75 #0.75 quartile
    c_type = 'cancer type abbreviation'
    t = survival[survival.PFI == 1]

    aggr_surv = t.groupby(by = c_type)['PFI.time'].quantile(qth)
    pfi_threshold = survival['cancer type abbreviation'].replace(aggr_surv)
    survival['s_label'] = 2
    survival.loc[(survival['PFI'] == 1) & (survival['PFI.time'] <= pfi_threshold), 's_label'] = 1
    survival.loc[(survival['PFI.time'] > pfi_threshold), 's_label'] = 0
    survival = survival[(survival.s_label == 0) | (survival.s_label == 1)]

    #reading expression file
    expr = pd.read_csv(basepath + filename_expr, index_col = 0, sep = '\t')

    expr = expr.dropna(axis = 0)
    pc_genes = pd.read_csv('./datasets/protein-coding_gene--09-24-2024.txt',
                            index_col = 0, sep = '\t')
    pc_genes = pc_genes[['symbol', 'name','entrez_id', 'ensembl_gene_id']]
    # only keep samples that are present in both expression data and survival metadata
    common_samples = np.intersect1d(survival.index, expr.columns)
    expr = expr.loc[np.intersect1d(expr.index, pc_genes.symbol),
                    common_samples]
    survival = survival[survival.index.isin(common_samples)]

    expr = expr.T
    survival.sort_index(inplace = True)
    expr.sort_index(inplace = True)
    ds_obj = DataSet(expr.values, survival['s_label'].values, survival['cancer type abbreviation'].values)
    return ds_obj

def load_single_cancer_datasets():
    yield 'NKI', load_nki()
    yield 'ACES', load_aces()
    #yield 'METABRIC', load_metabric()
    ds = load_tcga()
    X, y = ds.X, ds.y
    for cancer_type in np.unique(ds.cancer_type):
        mask = ds.cancer_type == cancer_type
        Xc, yc = X[mask], y[mask]
        yield 'TCGA_'+ cancer_type, DataSet(Xc, yc, f'TCGA_{cancer_type}')
        
def load_pan_cancer_datasets():
    yield 'TCGA', load_tcga()
