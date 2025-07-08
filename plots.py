#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:31:16 2024

@author: Sakhawat
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.cm as cm

import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import adjusted_rand_score

from utils import scores_to_metrics
from dataloader import load_single_cancer_datasets, load_pan_cancer_datasets, load_tcga

from pathlib import Path

# random seed
SEED = 2024
FIGURE_SAVEDIR = './figures'
WRITE_RESULTS = True
results_basedir = './results'
scores_basedir = f"{results_basedir}/pred_probability/"
model_attr_basedir = f"{results_basedir}/feature_scores/"
cache_basedir = f"{results_basedir}/cache/"


tcga_dataset = load_tcga()

def savefig(filename, extension = 'eps'):
    if not WRITE_RESULTS:
        return
    Path(FIGURE_SAVEDIR).mkdir(parents = True, exist_ok = True)
    plt.savefig('{}/{}.{}'.format(FIGURE_SAVEDIR, filename, extension), 
                bbox_inches = 'tight', dpi = 300)

# Utility function to add formatting to tables exported as LaTeX
def highlight_top_two(df, higher_is_better = True, ci = None,
                      precision = 2, ci_precision = 2,
                      first_start = '\\bfseries', first_end = '',
                      second_start = '\\underline{' ,second_end = '}'):
    
    precision_str = '{:.' + str(precision) + 'f}'
    ci_precision_str = '{:.' + str(ci_precision) + 'f}'
    rank = df.rank(axis = 1, method = 'min')
    if higher_is_better:
        rank = (-df).rank(axis = 1, method = 'min')
    rank[rank > 2] = -1
    if ci is None:
        table = rank.replace({1: first_start, 2: second_start, -1: ''}) \
                + df.map(precision_str.format) \
                + rank.replace({1: first_end, 2: second_end, -1: ''})
    else:
        table = rank.replace({1: first_start, 2: second_start, -1: ''}) \
                + df.map(precision_str.format) \
                + ci.map(('$\\pm$' + ci_precision_str).format) \
                + rank.replace({1: first_end, 2: second_end, -1: ''})
    return table

#%% TCGA: load pred proba scores

base_classifier_names = ['LR', 'RF', 'XGB']
knn_classifier_name = 'KNN_40'
other_classifier_names = ['LASSO', 'CMTL', 'CASO']
path_scores = Path(f'{scores_basedir}/pan_cancer_stratified/')
model_name = 'MCS'

y, cancer_types = tcga_dataset.y, tcga_dataset.cancer_type

scores_tcga = {}
test_idx = []
y_true = []

for fold in range(10):
    scores_clf = []
    test_idx_fold = np.loadtxt(f'{path_scores}/TCGA/LR/test_index_fold_{fold}.csv', dtype = int)
    test_idx.append(test_idx_fold)
    y_true.append(y[test_idx_fold].ravel())
    # load scores for MCS and associated baselines
    for classifier_name in base_classifier_names:
        p = Path(f'{path_scores}/TCGA/{classifier_name}')
        score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', 
                               sep = '\t', index_col = 0, header = [0, 1, 2]).T
        score_df = score_df.loc[(slice(None), slice(None), ['0 (baseline)', 'MCS']), :]
        score_df.index =  (score_df
                           .index
                           .set_levels(score_df
                                       .index
                                       .levels[2]
                                       .str.replace('0 (baseline)', classifier_name)
                                       .str.replace('MCS', 'MCS-' + classifier_name), level = 2))
        scores_clf.append(score_df)
    # load scores for KNN
    p = Path(f'{path_scores}/TCGA/{knn_classifier_name}')
    score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', sep = '\t', index_col = 0, header = [0, 1, 2]).T
    score_df.index =  (score_df
                       .index
                       .set_levels(score_df.index.levels[2].str.replace('0 (baseline)',
                                                                        'KNN'), level = 2))
    scores_clf.append(score_df)

    # load scores for MTL methods
    for classifier_name in other_classifier_names:
        p = Path(f'{path_scores}/TCGA/{classifier_name}')
        score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', sep = '\t', index_col = None, header = None).T
        score_df.index = pd.MultiIndex.from_tuples([('Baseline', 'PCA', classifier_name)],
                                                    names = ['augmentation', 'transformation', None])
        # Logistic function has to be applied on the scores
        scores_transformed = (1/(1+np.exp(-score_df.values)))
        score_df = pd.DataFrame(scores_transformed,
                                index = score_df.index, columns = score_df.columns)
        scores_clf.append(score_df)
        
    for classifier_name in [clf_name + '+' for clf_name in base_classifier_names]:
        p = Path(f'{path_scores}/TCGA/{classifier_name}')
        score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', sep = '\t', index_col = [0, 1, 2], header = None)
        score_df.index.names = ['augmentation', 'transformation', None]
        score_df.columns = np.arange(score_df.shape[1])
        scores_clf.append(score_df)

    scores_clf = pd.concat(scores_clf, axis = 0)
    scores_clf = scores_clf.reorder_levels([2, 0, 1]).sort_index()
    scores_clf.index.names = ['classifier'] + scores_clf.index.names[1:]
    scores_tcga[fold] = scores_clf
    
#%% TCGA: performance metrics

threshold = 0.25

metrics = {}
for (fold, score_df), y_true_fold in zip(scores_tcga.items(), y_true):
    metrics[fold] = scores_to_metrics(score_df, y_true_fold, p_threshold = threshold)
    
metrics = pd.concat(metrics)
metrics.index.names = ['fold'] + metrics.index.names[1:]

# Average and CI calculated over folds
metrics_mean = metrics.groupby(level = [1, 2, 3]).mean()
metrics_ci = metrics.groupby(level = [1, 2, 3]).sem() * 1.96


#%% TCGA: table (MCS vs base)

metric_names = ['AUC', 'F1', 'Balanced accuracy']
mask = ~(metrics_mean
         .index
         .get_level_values('classifier')
         .isin(['CASO', 'CMTL']))
tab_mean = metrics_mean.loc[mask, metric_names].droplevel('transformation')
tab_ci = metrics_ci.loc[mask, metric_names].droplevel('transformation')
sort_idx = np.array([[clf_name, clf_name + '+', 'MCS-' + clf_name] for clf_name in ['LR', 'RF', 'XGB']]).reshape(-1)
sort_idx = np.insert(sort_idx, 0, 'KNN')
tab_mean = tab_mean.loc[sort_idx, :]
tab_ci = tab_ci.loc[sort_idx, :]

tab = highlight_top_two(tab_mean.T, ci = tab_ci.T, precision = 3, ci_precision = 3).T
tab.to_latex(Path(FIGURE_SAVEDIR, 'pancancer_performance.tex'),
             index = True,
             multicolumn_format = 'c',
             multirow=False,
             column_format = ('ll' + 'r'*(tab.shape[1])))

#%% TCGA: table (MCS vs MTL)

metric_names = ['AUC', 'F1', 'Balanced accuracy']
sort_idx = (['LASSO', 'CASO', 'CMTL', 'MCS-LR', 'MCS-XGB'], 'Baseline')
tab_mean = metrics_mean.loc[sort_idx, metric_names].droplevel(['augmentation', 'transformation'])
tab_ci = metrics_ci.loc[sort_idx, metric_names].droplevel(['augmentation', 'transformation'])
tab = highlight_top_two(tab_mean.T, ci = tab_ci.T,
                        precision = 3, ci_precision = 2).T
tab.to_latex(Path(FIGURE_SAVEDIR, 'pancancer_comparison_mtl.tex'),
             index = True,
             multicolumn_format = 'c',
             multirow=False)

#%% TCGA: subtask performance

base_classifier_names = ['LR', 'RF', 'XGB']
knn_classifier_name = 'KNN_40'
other_classifier_names = ['LASSO', 'CMTL', 'CASO']
path_scores = Path(f'{results_basedir}/pred_probability/pan_cancer_stratified/')
model_name = 'MCS'
threshold = 0.25

y, cancer_types = tcga_dataset.y, tcga_dataset.cancer_type

scores_tcga = {}

test_idx = []
scores_fold = {}
for fold in range(10):
    scores_clf = {}
    test_idx.append(np.loadtxt(f'{path_scores}/TCGA/LR/test_index_fold_{fold}.csv', dtype = int))
    # load scores for MCS
    for classifier_name in base_classifier_names:
        p = Path(f'{path_scores}/TCGA/{classifier_name}')
        score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', 
                               sep = '\t', index_col = 0, header = [0, 1, 2]).T
        scores_clf[f'{classifier_name}'] = score_df.loc[('Baseline', 'PCA', '0 (baseline)'), :].rename(f'{classifier_name}')
        scores_clf[f'MCS-{classifier_name}'] = score_df.loc[('Baseline', 'PCA', 'MCS'), :].rename(f'MCS-{classifier_name}')
    # load scores for KNN
    p = Path(f'{path_scores}/TCGA/{knn_classifier_name}')
    score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', sep = '\t', index_col = 0, header = [0, 1, 2])
    scores_clf[knn_classifier_name.split('_')[0]] = score_df.loc[:, ('Baseline', 'PCA', '0 (baseline)')].rename(knn_classifier_name.split('_')[0])
    # load scores for MTL methods
    for classifier_name in other_classifier_names:
        p = Path(f'{path_scores}/TCGA/{classifier_name}')
        score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', sep = '\t', index_col = None, header = None).squeeze()
        
        # Logistic function has to be applied on the scores
        scores_transformed = 1/(1+np.exp(-score_df.values))
        score_df = pd.Series(scores_transformed,
                             index = score_df.index)
        scores_clf[classifier_name] = score_df
            
    scores_fold[fold] = pd.concat(scores_clf, axis = 1)
test_idx = np.concatenate(test_idx)
scores_fold = pd.concat(scores_fold).reset_index(drop = True).T
scores_fold = scores_fold.iloc[:, np.argsort(test_idx)]

metrics_tcga_types = {}
for cancer_type in np.unique(cancer_types):
    mask = (cancer_types == cancer_type)
    y_true_cancer_type = y[mask]
    m = scores_to_metrics(scores_fold.loc[:, mask], y_true_cancer_type, p_threshold = threshold)
    m = m.sort_index()
    metrics_tcga_types[cancer_type] = m

metrics_tcga_types = pd.concat(metrics_tcga_types, axis = 0)
metrics_tcga_types.index.names = ['Cancer type', 'Model']

#%% -- Plot

metric_names = ['AUC', 'F1', 'Balanced accuracy']
fig, axes = plt.subplots(nrows = 2, ncols = len(metric_names), 
                        figsize = (5.2*len(metric_names), 10),
                        sharex = True, sharey = False, squeeze = False,
                        gridspec_kw = {'height_ratios': [27, 1],
                                       'wspace': 0.05, 'hspace': 0.05})
classifier_order = base_classifier_names \
                   + [knn_classifier_name.split('_')[0]] \
                   + other_classifier_names \
                   + ['MCS-'+ clf_name for clf_name in base_classifier_names]
for i, metric_name in enumerate(metric_names):
    m = metrics_tcga_types[metric_name].unstack(level = -1)
    m = m[classifier_order]
    #annot = m.T.apply(lambda row: [f'*{val:.2f}' if val == row.max() else f' {val:.2f}' for val in row]).T
    annot = m.applymap(lambda val: f'{val:.2f}')
    best_non_mcs = m.loc[:, ~m.columns.str.startswith('MCS')].max(axis = 1)
    for colname in m.columns[m.columns.str.startswith('MCS')]:
        annot[colname] = ['*' if val else ' ' for val in best_non_mcs < m[colname]] + annot[colname]
    rank = (-m).rank(method = 'min', axis = 1)
    mean_rank = np.exp(np.log(rank).mean(axis = 0))
    mean_rank = pd.DataFrame(mean_rank).T
    mean_rank.index = ['mean rank']
    count_wins = (rank == 1).sum(axis = 0)
    count_wins = pd.DataFrame(count_wins).T
    count_wins.index = ['# of wins']
    count_wins = count_wins[m.columns]
    
    ax = axes[0, i]
    
    sns.heatmap(data = m, 
                annot = annot, fmt = 's', ax = ax, cbar = False,
                vmin = 0.3, vmax = 0.9)
    
    if ax != axes[0, 0]:
        ax.set_ylabel('')
        ax.set_yticks([])
    ax.set_title(metric_name)
    ax.set_xticks([])
    ax.set_xlabel('')
    
    ax = axes[1, i]
    # sns.heatmap(count_wins, ax = ax,
    #             annot = True, fmt = 'd', cmap = 'Blues', 
    #             vmin = 0, vmax = len(df), cbar = False)
    sns.heatmap(mean_rank, ax = ax,
                annot = True, fmt = '.2f', cmap = 'Blues', 
                vmin = 1, vmax = rank.shape[1], cbar = False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    if ax != axes[1, 0]:
        ax.set_yticklabels([])
    ax.set_xlabel('')
plt.subplots_adjust(wspace = 0.05)
savefig('tcga_subtask_metrics')
plt.show()

#%% Single-cancer: load pred proba scores

classifier_names = ['LR', 'RF', 'XGB']
threshold = 0.25
path_scores = Path(f'{scores_basedir}/single_cancer/')
model_name = 'MCS'

metrics_single = {}
for (dataset_name, ds_obj) in load_single_cancer_datasets():
    y = ds_obj.y
    for classifier_name in classifier_names:
        p = Path(f'{path_scores}/{dataset_name}/{classifier_name}')
        test_idx = []
        scores_fold = {}
        for fold in range(10):
            score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', 
                                   sep = '\t', index_col = 0, header = [0, 1, 2]).T
            scores_fold[fold] = score_df
            test_idx.append(np.loadtxt(f'{p}/test_index_fold_{fold}.csv', dtype = int))
        scores_fold = pd.concat(scores_fold, axis = 1)
        test_idx = np.concatenate(test_idx)
        scores_fold = scores_fold.iloc[:, np.argsort(test_idx)]
        
        metric_df = scores_to_metrics(scores_fold, y, p_threshold = threshold)
        m = metric_df.loc[(slice(None), slice(None), ['0 (baseline)', model_name]), :].droplevel(1, axis = 0)
        metrics_single[(classifier_name, dataset_name)] = m
metrics_single = pd.concat(metrics_single)
metrics_single.index.names = ['Classifier', 'Dataset', 'Augmentation', 'Model']
metrics_single = metrics_single.reset_index()
metrics_single['Model'] = metrics_single['Model'].apply(lambda val: 'Baseline' if 'baseline' in val.lower() else val)
metrics_single = metrics_single.set_index(['Classifier', 'Dataset', 'Augmentation', 'Model'])

#%% -- Plot

metric_names = ['AUC', 'F1', 'Balanced accuracy']
augmentation_name = 'Baseline'


fig, axes = plt.subplots(nrows = len(metric_names), ncols = metrics_single.index.levshape[0],
                         figsize = (6*len(metric_names), 2*metrics_single.index.levshape[0]), sharex = True)

data =  metrics_single.loc[(slice(None), slice(None), augmentation_name), metric_names]
stats = pd.DataFrame(0, index = data.index.levels[0], columns = data.columns)
data = data.reset_index()
data['Dataset'] = data['Dataset'].str.replace('_', '-')
data = data.set_index(['Classifier', 'Dataset', 'Augmentation', 'Model']).sort_index()
data = data.droplevel(2)
for i, metric_name in enumerate(metric_names):
    for j, classifier_name in enumerate(data.index.levels[0]):
        ax = axes[i, j]
        df = data.loc[classifier_name, metric_name].reset_index()
        sns.barplot(data = df, x = 'Dataset', y = metric_name,
                    hue = 'Model', ax = ax)
        # add marker for bars where MCS outperforms baseline
        df = data.loc[classifier_name, metric_name].unstack(level = -1)
        df = df.sort_index()
        df = df.reset_index()
        df['x'] = np.arange(len(df))+0.15
        df = df[df['MCS'] > df['Baseline']]
        df['MCS'] += 0.04
        stats.loc[classifier_name, metric_name] = df.shape[0]
        sns.scatterplot(data = df, x = 'x', y = 'MCS', ax = ax,
                        color = 'k', marker = '*', s = 50, label = 'MCS outperforms\nbaseline')
        
        if i == 0:
            ax.set_title(f'Classifier: {classifier_name}')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_ylim(0.0, 1.1)
        ax.set_xlim(-1, len(data.index.levels[1]))
        ax.set_yticks(np.arange(0.2, 1.1, 0.2))
        ax.set_yticklabels([])
        if j == 0:
            ax.set_ylabel(metric_name)    
            ax.set_yticklabels([f'{val:.2f}' for val in ax.get_yticks()])
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_axisbelow(True)
        ax.grid(axis = 'y', zorder = 2, color = '#CCC')
        
        ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], ncols = 1, frameon = False)
        if ax != axes[0, -1]:
            ax.legend().remove()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Dataset", labelpad = 60)        
plt.subplots_adjust(wspace = 0.02)
plt.subplots_adjust(hspace = 0.1)
savefig('single_cancer_metrics')
plt.show()


stats_total = pd.DataFrame(data.index.levshape[1], index = stats.index, columns = stats.columns)
stats_total['All'] = stats_total.sum(axis = 1)
stats_total.loc['All', :] = stats_total.sum(axis = 0)
stats['All'] = stats.sum(axis = 1)
stats.loc['All', :] = stats.sum(axis = 0)
stats = stats.astype(int)
print('==========\nStats\n==========')
print('Number/Total (%) of datasets where MCS outperforms Baseline:')
print(stats.astype(str) \
       + '/' + stats_total.astype(int).astype(str) \
       + ' (' + (stats * 100 / stats_total).round(1).astype(str) + '%)')

#%% TCGA: ablation

classifier_names = ['RF', 'XGB']
threshold = 0.25

basepath = f'{scores_basedir}/pan_cancer_stratified/TCGA'
y = tcga_dataset.y
metrics_ablation = {}

for classifier_name in classifier_names:
    p = Path(f'{basepath}/{classifier_name}')
    test_idx = []
    metrics_fold = {}
    for fold in range(10):
        score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', 
                               sep = '\t', index_col = 0, header = [0, 1, 2]).T
        # drop the "transfomation" level as the only value for TCGA dataset is "PCA"
        score_df = score_df.droplevel(1)
        score_df.index.names = ['Augmentation', 'Threshold']
        score_df = score_df.reset_index()
        score_df['Threshold'] = score_df['Threshold'].apply(lambda val: val.replace('(baseline)', '(base)'))
        score_df = score_df.set_index(['Augmentation', 'Threshold'])
        score_df = score_df.sort_index()
        score_df = score_df.loc[~score_df
                                .index
                                .get_level_values(-1).isin(['MCS']), :]
        # cumulative average of different thresholds
        score_df_cumulative = score_df.groupby(level = 0).expanding().mean().droplevel(0)
        score_df = pd.concat({'Single': score_df, 'Cumulative': score_df_cumulative})
        score_df = score_df.reorder_levels([1, 2, 0])
        score_df.index.names = np.hstack([score_df.index.names[:-1], ['Aggregation']])
        score_df = score_df.sort_index()
        test_idx = np.loadtxt(f'{p}/test_index_fold_{fold}.csv', dtype = int)
        y_fold = y[test_idx]
        metrics_fold[fold] = scores_to_metrics(score_df, y_fold, p_threshold = threshold)
        
    metrics_fold = pd.concat(metrics_fold, axis = 0)
    metrics_fold.index.names = ['Fold', 'Augmentation', 'Threshold', 'Aggregation']
    metrics_fold_mean = metrics_fold.groupby(level = [1, 2, 3]).mean()
    metrics_fold_sem = metrics_fold.groupby(level = [1, 2, 3]).sem()
    metrics_ablation[classifier_name] = metrics_fold
    
metrics_ablation = pd.concat(metrics_ablation)
metrics_ablation.index.names = np.hstack([['Base classifier'], metrics_ablation.index.names[1:]])
    
#%% -- Plot

metric_names = ['AUC', 'F1', 'Balanced accuracy']
augmentation_name = 'Baseline'

ncols = metrics_ablation.index.levshape[0]
nrows = len(metric_names)
fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (2*ncols, 2*nrows),
                         sharex = True, sharey = True)
for i, metric_name in enumerate(metric_names):
    for j, classifier_name in enumerate(metrics_ablation.index.levels[0]):
        ax = axes[i, j]
        t = metrics_ablation.loc[(classifier_name, slice(None), augmentation_name),
                                 metric_name].reset_index()
        
        data = t[t['Aggregation'] == 'Cumulative']
        sns.lineplot(data = data, x = 'Threshold', y = metric_name,
                     linestyle = '--', hue = 'Aggregation', palette = {'Cumulative': 'C1'},
                     errorbar = ('ci', 95), err_style = 'bars', err_kws = {'capsize': 1},
                     ax = ax)
        data = t[t['Aggregation'] == 'Single']
        sns.lineplot(data = data, x = 'Threshold', y = metric_name,
                     linestyle = ' ', markers=True, marker = 'o', hue = 'Aggregation',
                     errorbar = ('ci', 95), err_style = 'bars', err_kws = {'capsize': 1},
                     ax = ax)
        ax.set_xlabel('')
        if i == 0:
            ax.set_title(f'Classifier: {classifier_name}')
        if j != 0:
            ax.set_ylabel('')
        
        ax.set_ylim(0.45, 0.75)
        ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], title = 'Score\nAggregation')
        if ax != axes[0, -1]:
            ax.legend().remove()
        if i == nrows - 1:
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45,
                               ha = 'right', rotation_mode="anchor")    
            ax.tick_params(axis='x', pad=-0.5)
        ax.grid(color = '#CCC')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Correlation threshold", labelpad = 30)
plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
savefig('tcga_ablation')
plt.show()



#%% TCGA: Performance vs correlation

classifier_names = ['RF', 'XGB']
metric_names = ['AUC', 'F1', 'Balanced accuracy']
h = 0.15
n_bins = 3
path_scores = Path(f'{scores_basedir}/pan_cancer_stratified/')

dataset_name = 'TCGA'
X = tcga_dataset.X
y = tcga_dataset.y
ctypes = tcga_dataset.cancer_type
Xt = PCA(n_components = 0.95).fit_transform(X)
corr = np.corrcoef(Xt)
y_pred_all = {}
for classifier_name in classifier_names:
    p = Path(f'{path_scores}/{dataset_name}/{classifier_name}')
    test_idx = []
    scores_fold = {}
    for fold in range(10):
        score_df = pd.read_csv(f'{p}/scores_fold_{fold}.csv', 
                               sep = '\t', index_col = 0, header = [0, 1, 2]).T
        scores_fold[fold] = score_df
        test_idx.append(np.loadtxt(f'{p}/test_index_fold_{fold}.csv', dtype = int))
    scores_fold = pd.concat(scores_fold, axis = 1)
    test_idx = np.concatenate(test_idx)
    scores_fold = scores_fold.iloc[:, np.argsort(test_idx)]
    y_pred = scores_fold.loc[('Baseline', slice(None), ['0 (baseline)', 'MCS']), :].T
    y_pred = y_pred.droplevel([0, 1], axis = 1).droplevel(1, axis = 0)
    y_pred.columns = ['Baseline', 'MCS']
    y_pred = y_pred.reset_index(drop = True)
    y_pred_all[classifier_name] = y_pred
y_pred_all = pd.concat(y_pred_all, axis = 1)
  
H = list(scores_fold.index.levels[-1])
H = np.array([float(val) for val in H if 'baseline' not in val and 'MCS' not in val])

n_correlated= (np.abs(corr) >= h).sum(axis = 1) - 1
n_correlated = pd.DataFrame(n_correlated, columns = ['# samples'])
n_correlated['median abs corr'] = np.median(np.abs(corr), axis = 1).ravel()

n_correlated_binned = pd.qcut(n_correlated['# samples'], n_bins, precision = 0)

metrics_binned = {}
for bin_ in n_correlated_binned.unique().sort_values():
    mask = n_correlated_binned == bin_
    bin_str = str(int(bin_.left)) + ' - ' + str(int(bin_.right)-1)
    y_category = y[mask]
    y_pred_category = y_pred_all.loc[mask, :]
    metrics_binned[bin_str] = scores_to_metrics(y_pred_category.T, y_category)

bin_colname = f'# neighbors (absolute correlation >= {h:.3f})'
bin_colname = '# of neighbors'
metrics_binned = pd.concat(metrics_binned).reorder_levels([1, 2, 0]).sort_index()
metrics_binned.index.names = ['Classifier', 'Model type', bin_colname]
 
#%% -- Plot

ncols = len(metrics_binned.index.get_level_values('Classifier').unique())
nrows = len(metric_names)
fig, axes = plt.subplots(nrows = nrows, ncols = ncols,
                         figsize = (ncols*2, nrows*2),
                         sharex = True, sharey = True,
                         squeeze = False)
for i, metric_name in enumerate(metric_names):
    for j, classifier_name in enumerate(metrics_binned.index.get_level_values('Classifier').unique()):
        ax = axes[i, j]
        data = metrics_binned.loc[classifier_name, metric_name].reset_index()
        sns.barplot(data, x = bin_colname, y = metric_name, hue = 'Model type',
                    ax = ax)
        #ax.set_xticklabels(labels = ax.get_xticklabels(), ha = 'right')
        ax.tick_params(axis = 'x', rotation = 45)
        if i == 0 and j == ncols - 1:
            ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], title = 'Model type')
        else:
            ax.get_legend().remove()
        if j != 0:
            ax.set_ylabel('')
        if i == 0 and len(classifier_names) > 1:
            ax.set_title(classifier_name)
        ax.grid()
        ax.set_xlabel('')
        
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(bin_colname, labelpad = 50)
plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
savefig('tcga_correlation_grouped_performance')
plt.show()