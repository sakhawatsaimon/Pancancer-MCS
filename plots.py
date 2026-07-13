#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:31:16 2024

@author: Sakhawat
"""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import percentileofscore, ttest_rel, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from utils import scores_to_metrics, PercentileThresholdSelector, FixedThresholdSelector
from dataloader import load_aces, load_tcga

from pathlib import Path
from itertools import combinations_with_replacement

# random seed
SEED = 2024
FIGURE_SAVEDIR = './figures_v2'
WRITE_RESULTS = True
#results_basedir = './results_v2'
#scores_basedir = f"{results_basedir}/pred_probability"
#model_attr_basedir = f"{results_basedir}/feature_scores"
#cache_basedir = f"{results_basedir}/cache"
#data_basedir = f"{results_basedir}/data"

tcga_dataset = load_tcga(return_survival_info = True)

def savefig(filename, extension = 'eps'):
    if not WRITE_RESULTS:
        return
    Path(FIGURE_SAVEDIR).mkdir(parents = True, exist_ok = True)
    plt.savefig('{}/{}.{}'.format(FIGURE_SAVEDIR, filename, extension), 
                bbox_inches = 'tight', dpi = 300)

# Utility function to add formatting to tables exported as LaTeX
def highlight_top_two(
        df,
        higher_is_better = True,
        ci = None,
        precision = 2,
        ci_precision = 2,
        best_start = '\\bfseries',
        best_end = '',
        second_best_start = '\\underline{',
        second_best_end = '}'):
    
    precision_str = '{:.' + str(precision) + 'f}'
    ci_precision_str = '{:.' + str(ci_precision) + 'f}'
    rank = df.rank(axis = 1, method = 'min')
    if higher_is_better:
        rank = (-df).rank(axis = 1, method = 'min')
    rank[rank > 2] = -1
    if ci is None:
        table = rank.replace({1: best_start, 2: second_best_start, -1: ''}) \
                + df.map(precision_str.format) \
                + rank.replace({1: best_end, 2: second_best_end, -1: ''})
    else:
        table = rank.replace({1: best_start, 2: second_best_start, -1: ''}) \
                + df.map(precision_str.format) \
                + ci.map(('$\\pm$' + ci_precision_str).format) \
                + rank.replace({1: best_end, 2: second_best_end, -1: ''})
    return table

def load_y(path_experiment):

    pred_proba_dir = 'pred_probability'
    path_data = Path(path_experiment, 'data')
    
    y_pred = {}
    y_true = {}
    
    paths_y_true = sorted(list(path_data.rglob('*y_test.csv')))
    for path_y_true in paths_y_true:
        y_true_cfg = np.loadtxt(path_y_true, dtype = int)
        try:
            ctypes_test = pd.read_csv(Path(path_y_true.parent, 'ctypes_test.csv'), header = None)[0]
            assert len(y_true_cfg) == len(ctypes_test)
        except FileNotFoundError:
            ctypes_test = None
        
        fold_dirname = path_y_true.parent.name
        prefix_len = len(path_y_true.parts) - 5
        
        _, transform_name, augment_name, fold_dirname =\
            path_y_true.parts[prefix_len:-1]
        fold = int(fold_dirname.split('_')[1])
        y_true[(transform_name, augment_name, fold)] = pd.DataFrame({
                'y_true': y_true_cfg,
                'ctypes': ctypes_test
            })
    
        paths_y_pred = sorted(list(Path(
            *path_y_true.parts[:prefix_len],
            pred_proba_dir,
            transform_name,
            augment_name)
            .rglob(f'*/{fold_dirname}')))
    
        for path_y_pred in paths_y_pred:
            classifier_name = path_y_pred.parent.name
            sample_files = sorted(list(path_y_pred.glob('sample_*')))
            config = (transform_name, augment_name, classifier_name, fold)
            print(''.join(['{:15s}'.format(str(val)) for val in config]))
            
            y_pred_samples = {}
            for sample_file in sample_files:
                sample_id = int(sample_file.stem.split('sample_')[1])
                df = pd.read_csv(
                    sample_file, header = None,
                    sep = '\t', index_col = 0)
                df = df[1]
                y_pred_samples[sample_id] = df
            y_pred_df = pd.concat(y_pred_samples, axis = 1).T.sort_index(axis = 0)
            y_pred_df.columns.name = 'Model threshold'
            if y_pred_df.shape[1] > 1:
                y_pred_df['MCS'] = y_pred_df.mean(axis = 1)
            y_pred_df['y_true'] = y_true_cfg
            y_pred_df['ctypes'] = ctypes_test
            y_pred[config] = y_pred_df
    return y_pred

def get_y_pred_combined(y_pred):
    y_pred_combined = {}
    cfg_df = pd.DataFrame(y_pred.keys(), columns = ['Transform', 'Augment', 'Model', 'Fold'])
    folds = cfg_df['Fold'].unique()
    for (transform_name, augment_name), _ in cfg_df.groupby(['Transform', 'Augment']).groups.items():
        y_pred_models = []
        for model_name in cfg_df['Model'].unique():
            y_pred_folds = []
            for fold in folds:
                y_pred_folds.append(y_pred[(transform_name, augment_name, model_name, fold)])
            y_pred_folds = pd.concat(y_pred_folds, axis = 0)
            y_pred_folds = y_pred_folds.reset_index(drop = True)
            labels = y_pred_folds[['y_true', 'ctypes']]
            y_pred_folds = y_pred_folds.drop(labels.columns, axis = 1)
            if 'MCS' in y_pred_folds.columns:
                y_pred_folds.columns = model_name + '-' + y_pred_folds.columns
                y_pred_folds = y_pred_folds.rename(columns = {model_name + '-MCS': 'MCS-' + model_name})
            else:
                y_pred_folds.columns = [model_name]
            y_pred_models.append(y_pred_folds)
        y_pred_models = pd.concat(y_pred_models, axis = 1)
        y_pred_models.columns.name = 'Model'
        y_pred_models = pd.concat([y_pred_models, labels], axis = 1)
        y_pred_combined[(transform_name, augment_name)] = y_pred_models
    return y_pred_combined

def get_performance(y_pred):
    metrics = {}
    
    for config, y_pred_df in y_pred.items():
        y_true_cfg = y_pred_df['y_true']
        y_pred_df = y_pred_df.drop(['y_true', 'ctypes'], axis = 1)
        metrics[config] = scores_to_metrics(
            y_pred_df.T, y_true_cfg, p_threshold = threshold)
    metrics = pd.concat(metrics)
    metrics.index.names = [
        'Transform', 'Augment', 'Classifier', 'Fold', 'Model threshold']
    metrics = metrics.reorder_levels([0, 1, 2, 4, 3]).sort_index()
    metrics.columns.name = 'Metric'
    
    metrics_mean = metrics.groupby(level = [
        'Transform', 'Augment', 'Classifier', 'Model threshold']).mean()
    metrics_ci = metrics.groupby(level = [
        'Transform', 'Augment', 'Classifier', 'Model threshold']).sem() * 1.96
    return metrics, metrics_mean, metrics_ci


#%% TCGA: [plot] pre-binarization stats

bin_count = 35
plot_threshold = True
xticks_years = [0, 5, 10, 15]
n_cols = 4
threshold_color = 'C3'
pfi_map = {0: 'No', 1: 'Yes'}

survival = tcga_dataset.attributes['survival info']
n_cancers = survival['cancer type abbreviation'].nunique()
n_rows = int(np.ceil(n_cancers / n_cols))

bin_range = np.linspace(0, survival['PFI.time'].max(), bin_count)
fig, axes = plt.subplots(
    nrows = n_rows,
    ncols = n_cols,
    figsize = (n_cols*3, n_rows*0.65)
)
for i, cancer_type in enumerate(survival['cancer type abbreviation'].unique()):
    ax = axes.flat[i]
    data = survival[survival['cancer type abbreviation'] == cancer_type]
    pfi_threshold = data['PFI threshold'].unique()
    assert len(pfi_threshold) == 1
    pfi_threshold = pfi_threshold[0]
    sns.histplot(
        data = data,
        x = 'PFI.time',
        hue = 'PFI',
        multiple = 'stack',
        ax = ax, bins = bin_range,
        linewidth = 0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    if plot_threshold:
        y_top = ax.get_ylim()[1]*1.2
        ax.plot(
            [pfi_threshold, pfi_threshold],
            [0, y_top],
            linestyle = '--', 
            color = threshold_color,
            linewidth = 1)
        ax.text(
            pfi_threshold, y_top, 
            '{:.1f}y'.format(pfi_threshold / 365),
            ha = 'center', va = 'bottom',
            fontsize = 7, color = threshold_color)
    handles = ax.legend_.legend_handles
    labels = [text.get_text() for text in ax.legend_.texts]
    ax.get_legend().remove()
    plt.text(0.99, 0.12, cancer_type,
             transform = ax.transAxes, ha = 'right', va = 'bottom')
    ax.set_xlim(-500, survival['PFI.time'].max())
    ax.set_ylim(0, y_top*1.2)
    ax.set_xticks(np.array(xticks_years)*365, labels = xticks_years)
    if i + n_cols < n_cancers:
        ax.set_xticklabels([])
    sns.despine(ax = ax)
i += 1
while i < axes.size:
    axes.flat[i].axis('off')
    i += 1
    
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('PFI time (years)', labelpad = 5)
plt.ylabel('Count', labelpad = 5)
plt.legend(handles, labels, loc = 'lower center',
           bbox_to_anchor = [0.5, 1], ncols = 2,
           title = 'Progression status', frameon = False)
plt.subplots_adjust(wspace = 0.25, hspace = 0.5)
savefig('tcga_stats')
plt.show()

#%% TCGA: [plot] good/poor samples per cancer type

df = pd.DataFrame([tcga_dataset.y, tcga_dataset.cancer_type],
                  index = ['Outcome', 'Cancer type']).T
df['Outcome'] = df['Outcome'].replace({0: 'Good', 1: 'Poor'})
counts = pd.crosstab(df['Cancer type'], df['Outcome'])
fig, axes = plt.subplots(figsize = (4, 6), ncols = 3, sharey = True,
                         gridspec_kw = {'width_ratios': [5, 2, 5],
                                        'wspace': 0.05})
counts['Good'].plot.barh(ax = axes[0])
counts['Poor'].plot.barh(ax = axes[2], color = 'C1')
axes[0].set_xlim(counts.max().max(), 0)
axes[2].set_xlim(0, counts.max().max())
for i, ctype in enumerate(counts.index):
    axes[1].text(0.5, i, ctype, va = 'center', ha = 'center')
for ax in axes:
    ax.yaxis.set_inverted(True)
    
axes[0].set_yticks([])
axes[1].set_xticks([])

axes[0].tick_params(which='both', bottom=True, top=False, left=False, right=False)
axes[2].tick_params(which='both', bottom=True, top=False, left=False, right=False)
axes[1].tick_params(which='both', bottom=False, top=False, left=False, right=False)

axes[0].set_ylabel('')
axes[0].set_xlabel('Good')
axes[2].set_xlabel('Poor')
sns.despine(ax = axes[0], left = True, bottom = False)
sns.despine(ax = axes[2], left = True, bottom = False)
sns.despine(ax = axes[1], left = True, bottom = True)
axes[2].set_xticks(axes[0].get_xticks())
axes[0].set_xticks(axes[0].get_xticks())
savefig('tcga_binary_label_distribution')
plt.show()

#%% TCGA: [calc] performance


basedir = './results_v2'
path_experiment = Path(basedir, 'pan_cancer_stratified', 'TCGA')
threshold = 0.25

y_pred = load_y(path_experiment)

# Consolidate folds
y_pred_combined = get_y_pred_combined(y_pred)

metrics, metrics_mean, metrics_ci = get_performance(y_pred)

# Quick peak
model_names = metrics_mean.index.levels[-1]
# Exclude numeric (threshold) entries
model_names = model_names[model_names.str[0].str.isalpha()]
idx = (slice(None), slice(None), slice(None), model_names)
t = metrics_mean.loc[idx, :].droplevel(0)

#%% TCGA: [table] MCS vs base performance

classifier_names = ['KNN', 'LR', 'MLP', 'RF', 'XGB']
metric_names = ['AUC', 'F1', 'Balanced accuracy']
classifier_level_values = metrics_mean.index.get_level_values('Classifier')

mask = [any([val.startswith(clf_name) for clf_name in classifier_names]) \
        for val in classifier_level_values]
model_idx = (slice(None), slice(None), ['Baseline', 'MCS'])

tables = {'mean': metrics_mean, 'ci': metrics_ci}
for table_id, table in tables.items():
    table = (table
             .loc[mask, metric_names]
             .droplevel('Transform')
             .loc[model_idx, :])

    table = table.reset_index()
    table['Classifier'] = table['Model threshold'].replace(
        {'Baseline': '', 'MCS': 'MCS-'})\
        + table['Classifier']
    table['Sort dummy'] = table['Model threshold'].replace(
        {'Baseline': '', 'MCS': 'zMCS-'})\
        + table['Classifier']
    table['Augment'] = table['Augment'].replace(
        {'Baseline': '-', 'CiFRUS': 'Yes'})
    table.pop('Model threshold')
    table = table.set_index(
        ['Sort dummy', 'Classifier', 'Augment']).sort_index().droplevel(0)
    table.columns = table.columns.str.replace('Balanced accuracy', 'Bal. acc.')
    tables[table_id] = table
    
table_mean = tables['mean']
table_ci = tables['ci']

table = highlight_top_two(
    table_mean.T, ci = table_ci.T, precision = 3, ci_precision = 3).T
table.to_latex(
    Path(FIGURE_SAVEDIR, 'pancancer_performance.tex'),
    index = True,
    multicolumn_format = 'c',
    multirow=False,
    column_format = ('ll' + 'r'*(table.shape[1])))
   
#%% TCGA: [table] MCS vs MTL performance

metric_names = ['AUC', 'F1', 'Balanced accuracy']
mtl_classifier_names = ['LASSO', 'CASO', 'CMTL']
mcs_classifier_names = ['LR', 'XGB']

tables = {'mean': metrics_mean, 'ci': metrics_ci}

for table_id, table in tables.items():
    table = pd.concat([
        table.loc[(slice(None), slice(None), mtl_classifier_names)],
        table.loc[(slice(None), slice(None), mcs_classifier_names, 'MCS')]
    ], axis = 0)
    table = table.droplevel(0).reorder_levels([1, 0, 2])
    table = table[metric_names]
    table = table.reset_index()
    table['Model threshold'] = table['Model threshold'].apply(lambda val: '' if val != 'MCS' else val + '-')
    table['Classifier'] = table['Model threshold'] + table['Classifier']
    table['Augment'] = table['Augment'].replace(
        {'Baseline': '-', 'CiFRUS': 'Yes'})
    table.pop('Model threshold')
    table = table.set_index(['Classifier', 'Augment']).sort_index()
    table = table.loc[mtl_classifier_names + \
            ['MCS-' + clf_name for clf_name in mcs_classifier_names]]
    tables[table_id] = table
table_mean = tables['mean']
table_ci = tables['ci']

table = highlight_top_two(
    table_mean.T, ci = table_ci.T,
    precision = 3, ci_precision = 2).T

if WRITE_RESULTS:
    table.to_latex(
        Path(FIGURE_SAVEDIR, 'pancancer_comparison_mtl_with_augmentation.tex'),
        index = True,
        multicolumn_format = 'c',
        multirow=False)


#%% TCGA: t-test between baseline and MCS

classifier_names = ['LR', 'MLP', 'RF', 'XGB']
metric_names = ['AUC', 'F1', 'Balanced accuracy']

significance = {}
for clf_name in classifier_names:
    
    df = (metrics
          .loc[(slice(None), slice(None), clf_name, ['Baseline', 'MCS']),
               metric_names]
          .droplevel([0, 2])
          .stack()
          .unstack(level = 'Model threshold')
          .reorder_levels(['Metric', 'Augment', 'Fold'])
          .sort_index())

    pval_clf = pd.DataFrame(np.nan,
                            index = pd.MultiIndex.from_product([
                                metric_names, df.index.levels[1]]),
                            columns = ['tval', 'pval'])
    pval_clf.index.names = ['Metric', 'Augment']
    for cfg in pval_clf.index.values:
        pval_clf.loc[cfg, :] = list(ttest_rel(
            df.loc[cfg]['MCS'].sort_index(),
            df.loc[cfg]['Baseline'].sort_index()))
    significance[clf_name] = pval_clf
significance = pd.concat(significance).round(3)

count = pd.crosstab(significance['tval'] > 0, significance['pval'] <= 0.05)
count.index.names = ['MCS is better (t > 0)']
count.columns.names = ['Significant (p <= 0.05)']
print(count.stack().reset_index())

#%% TCGA: [calc] subtask performance

threshold = 0.25

metrics_tcga_types = {}
for (transform_name, augment_name), y_pred_cfg in y_pred_combined.items():
    
    metrics_cfg = {}
    for cancer_type, y_pred_ctype in y_pred_cfg.groupby('ctypes'):
        y_true_ctype = y_pred_ctype.pop('y_true')
        _ = y_pred_ctype.pop('ctypes')
        metrics_ctype = scores_to_metrics(
            y_pred_ctype.T, y_true_ctype, p_threshold = threshold)
        
        mask = [('-' in colname) \
                and ('Baseline' not in colname.split('-')) \
                and ('MCS' not in colname.split('-')) \
                    for colname in metrics_ctype.index]
        metrics_ctype = metrics_ctype.loc[~np.array(mask), :]
        metrics_ctype.index = metrics_ctype.index.map(lambda name: name.replace('-Baseline', ''))
        metrics_cfg[cancer_type] = metrics_ctype
    metrics_cfg = pd.concat(metrics_cfg)
    metrics_tcga_types[(transform_name, augment_name)] = metrics_cfg
metrics_tcga_types = pd.concat(metrics_tcga_types)
metrics_tcga_types.index.names = ['Transform', 'Augment', 'Cancer type', 'Model']
metrics_tcga_types.columns.name = 'Metric'

#%%% Plot

transform_name = 'PCA'
augment_name = 'Baseline'
metric_names = ['AUC', 'F1', 'Balanced accuracy']

base_classifier_names = ['LR', 'MLP', 'RF', 'XGB']
mtl_classifier_names = ['LASSO', 'CMTL', 'CASO']
knn_classifier_name = 'KNN'

fig, axes = plt.subplots(
    nrows = 2, ncols = len(metric_names), 
                    figsize = (6*len(metric_names), 10),
                    sharex = True, sharey = False, squeeze = False,
                    gridspec_kw = {'height_ratios': [27, 1],
                                   'wspace': 0.03, 'hspace': 0.03})
available_clf_names = metrics_tcga_types.index.unique(level='Model')
classifier_order = base_classifier_names \
                   + ([knn_classifier_name.split('_')[0]] \
                      if knn_classifier_name in available_clf_names else []) \
                   + [clf_name for clf_name in mtl_classifier_names \
                      if clf_name in available_clf_names] \
                   + ['MCS-'+ clf_name for clf_name in base_classifier_names]
for i, metric_name in enumerate(metric_names):
    m = metrics_tcga_types.loc[
        (transform_name, augment_name), metric_name].unstack(level = 'Model')
    m = m[classifier_order]
    annot = m.map(lambda val: f'{val:.2f}')
    best_non_mcs = m.loc[:, ~m.columns.str.startswith('MCS')].max(axis = 1)
    for colname in m.columns[m.columns.str.startswith('MCS')]:
        annot[colname] = ['*' if val else ' ' for val in best_non_mcs < m[colname]] + annot[colname]
    rank = (-m).rank(method = 'min', axis = 1)
    mean_rank = np.exp(np.log(rank).mean(axis = 0))
    mean_rank = pd.DataFrame(mean_rank).T
    mean_rank.index = ['mean\nrank']
    count_wins = (rank == 1).sum(axis = 0)
    count_wins = pd.DataFrame(count_wins).T
    count_wins.index = ['# of wins']
    count_wins = count_wins[m.columns]
    
    ax = axes[0, i]
    
    sns.heatmap(
        data = m, 
        annot = annot, fmt = 's', ax = ax, cbar = False,
        vmin = 0.3, vmax = 0.9)
    
    if ax != axes[0, 0]:
        ax.set_ylabel('')
        ax.set_yticks([])
    if len(metric_names) > 1:
        ax.set_title(metric_name)
    ax.set_xticks([])
    ax.set_xlabel('')
    
    ax = axes[1, i]

    sns.heatmap(
        mean_rank, ax = ax,
        annot = True, fmt = '.2f', cmap = 'Blues', 
        vmin = 1, vmax = rank.shape[1], cbar = False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')
    if ax != axes[1, 0]:
        ax.set_yticklabels([])
    ax.set_xlabel('')
plt.subplots_adjust(wspace = 0.05)
title_suffix = ''
if len(metric_names) == 1:
    title_suffix = '_' + metric_names[0].replace(' ', '_')
savefig('tcga_subtask_metrics' + title_suffix)
plt.show()


#%% Single-cancer: [calc] performance metrics

classifier_names = ['LR', 'RF', 'XGB']
threshold = 0.25
basedir = './results_v2'
path_experiment = Path(basedir, 'single_cancer')

dirnames = np.array([d for d in path_experiment.iterdir() if d.is_dir()])

y_pred_single_cancer = {}

for dirname in dirnames:
    dataset_name = dirname.name
    y_pred_dataset = load_y(dirname)
    for k, v in y_pred_dataset.items():
        y_pred_single_cancer[(dataset_name, *k)] = v.drop('ctypes', axis = 1)

metrics_single_cancer = {}

for config, y_pred_df in y_pred_single_cancer.items():
    y_true_cfg = y_pred_df['y_true']
    y_pred_df = y_pred_df.drop('y_true', axis = 1)
    metrics_single_cancer[config] = scores_to_metrics(
        y_pred_df.T, y_true_cfg, p_threshold = threshold)
metrics_single_cancer = pd.concat(metrics_single_cancer)

metrics_single_cancer.index.names = [
    'Dataset', 'Transform', 'Augment', 'Classifier', 'Fold', 'Model']
metrics_single_cancer = metrics_single_cancer.reorder_levels([
    'Classifier', 'Dataset', 'Augment', 'Transform', 'Model', 'Fold']).sort_index()

metrics_single_cancer.columns.name = 'Metric'

metrics_single_cancer_mean = metrics_single_cancer.groupby(level = [
    'Classifier', 'Dataset', 'Augment', 'Transform', 'Model']).mean().sort_index()
metrics_single_cancer_ci = metrics_single_cancer.groupby(level = [
    'Classifier', 'Dataset', 'Augment', 'Transform', 'Model']).sem().sort_index() * 1.96

# Quick peak
model_names = metrics_single_cancer_mean.index.levels[-1]
# Exclude numeric (threshold) entries
model_names = model_names[model_names.str[0].str.isalpha()]
idx = (slice(None), slice(None), slice(None), slice(None), model_names)
t = metrics_single_cancer_mean.loc[idx, :]

#%%% Plot

classifier_names = ['RF', 'XGB']
metric_names = ['AUC', 'F1', 'Balanced accuracy']
augmentation_name = 'Baseline'
transform_name = 'PCA'

fig, axes = plt.subplots(
    nrows = len(metric_names),
    ncols = metrics_single_cancer_mean.index.levshape[0],
    figsize = (6*len(metric_names), 2*metrics_single_cancer_mean.index.levshape[0]),
    sharex = True)

data =  (metrics_single_cancer_mean
         .loc[(slice(None), slice(None), augmentation_name,
              transform_name, ['Baseline', 'MCS']),
              metric_names]
         .droplevel(['Augment', 'Transform']))
stats = pd.DataFrame(0, index = data.index.levels[0], columns = data.columns)

for i, metric_name in enumerate(metric_names):
    for j, classifier_name in enumerate(data.index.levels[0]):
        ax = axes[i, j]
        df = data.loc[classifier_name, metric_name].reset_index()
        sns.barplot(
            data = df, x = 'Dataset', 
            y = metric_name,
            hue = 'Model',
            ax = ax)
        
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

stats.columns = stats.columns.str.replace('Balanced accuracy', 'Bal. acc.')
stats_total = pd.DataFrame(data.index.levshape[1], index = stats.index, columns = stats.columns)
stats_total['All'] = stats_total.sum(axis = 1)
stats_total.loc['All', :] = stats_total.sum(axis = 0)
stats['All'] = stats.sum(axis = 1)
stats.loc['All', :] = stats.sum(axis = 0)
stats = stats.astype(int)

print('='*60 + '\nStats\n' + '='*60)
print('Number/Total (%) of datasets where MCS outperforms Baseline:')
print(stats.astype(str) \
       + '/' + stats_total.astype(int).astype(str) \
       + ' (' + (stats * 100 / stats_total).round(1).astype(str) + '%)')
    
#%% TCGA: Performance comparison between MCS and component models


threshold = 0.25

metrics_ablation = {}
metrics_ttest = {}
for config, y_pred_df in y_pred.items():
    if 'MCS' not in y_pred_df.columns:
        continue
    y_true_cfg = y_pred_df['y_true']
    y_pred_df = y_pred_df.drop(['y_true', 'ctypes'], axis = 1)
    metrics_cfg = scores_to_metrics(
        y_pred_df.T, y_true_cfg, p_threshold = threshold)
    metrics_ttest[config] = metrics_cfg.T
    y_pred_df = y_pred_df.drop('MCS', axis = 1).T
    metrics_ablation[tuple(list(config) + ['Individual'])] = scores_to_metrics(
        y_pred_df, y_true_cfg, p_threshold = threshold)
    metrics_ablation[tuple(list(config) + ['Cumulative'])] = scores_to_metrics(
        y_pred_df.expanding().mean(), y_true_cfg, p_threshold = threshold)
metrics_ablation = pd.concat(metrics_ablation)

metrics_ablation.index.names = [
    'Transform', 'Augment', 'Classifier', 'Fold', 'Probability aggregation', 'Threshold']
metrics_ablation = metrics_ablation.reorder_levels([0, 1, 2, 5, 4, 3]).sort_index()
metrics_ablation.columns.name = 'Metric'

metrics_ablation_mean = metrics_ablation.groupby(level = [
    'Transform', 'Augment', 'Classifier', 'Probability aggregation', 'Threshold']).mean()
metrics_ablation_ci = metrics_ablation.groupby(level = [
    'Transform', 'Augment', 'Classifier', 'Probability aggregation', 'Threshold']).sem() * 1.96
metrics_ttest = pd.concat(metrics_ttest)
metrics_ttest.index.names = ['Transform', 'Augment', 'Classifier', 'Fold', 'Metric']

# t-test

ttest_results = {}
for cfg, metrics_cfg in metrics_ttest.groupby(['Transform', 'Augment', 'Classifier', 'Metric']):
    ttest_res_cfg = {}
    for colname in metrics_cfg.columns:
        if colname == 'MCS':
            continue
        ttest_res_cfg[colname] = ttest_rel(metrics_cfg['MCS'], metrics_cfg[colname], )[:2]
    ttest_res_cfg = pd.DataFrame(ttest_res_cfg).T
    ttest_res_cfg.columns = ['t-val', 'p-val']
    ttest_results[cfg] = ttest_res_cfg
ttest_results = pd.concat(ttest_results)
ttest_results.index.names = ['Transform', 'Augment', 'Classifier', 'Metric', 'Model']

#%%% Plot

metric_names = ['AUC', 'F1', 'Balanced accuracy']
transform_name = 'PCA'
augment_name = 'CiFRUS'
classifier_names = ['RF', 'XGB']

ncols = len(classifier_names)
nrows = len(metric_names)
fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (2*ncols, 1.5*nrows),
                         sharex = True, sharey = True)
for i, metric_name in enumerate(metric_names):
    for j, classifier_name in enumerate(classifier_names):
        ax = axes[i, j]
        t = metrics_ablation.loc[(transform_name, augment_name, classifier_name),
                                 metric_name].reset_index()
        t['Threshold'] = t['Threshold'].replace({'Baseline': '0 (Baseline)'})
        t = t.sort_values('Threshold')
        data = t[t['Probability aggregation'] == 'Cumulative']
        sns.lineplot(
            data = data, x = 'Threshold', y = metric_name,
            linestyle = '--', hue = 'Probability aggregation', palette = {'Cumulative': 'C1'},
            errorbar = ('ci', 95), err_style = 'bars', err_kws = {'capsize': 1},
            ax = ax)
        data = t[t['Probability aggregation'] == 'Individual']
        sns.lineplot(
            data = data, x = 'Threshold', y = metric_name,
            linestyle = ' ', markers=True, marker = 'o', hue = 'Probability aggregation',
            errorbar = ('ci', 95), err_style = 'bars', err_kws = {'capsize': 1},
            ax = ax)
        ax.set_xlabel('')
        ax.set_ylabel(ax.get_ylabel().replace('Balanced accuracy', 'Bal. acc.'))
        if i == 0:
            ax.set_title(f'{classifier_name}')
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
plt.xlabel("Correlation threshold percentile", labelpad = 40)
plt.subplots_adjust(wspace = 0.15, hspace = 0.1)
savefig('tcga_ablation')
plt.show()

# t-test results

df = (ttest_results
      .loc[(transform_name, augment_name), :]
      .loc[(classifier_names, metric_names), :])
is_significant = (df['t-val'] > 0) & (df['p-val'] <= 0.05)
is_significant = is_significant.unstack(level = -2)
print(is_significant)

#%% TCGA: performance for different similarity metrics

metric_names = ['AUC', 'F1', 'Balanced accuracy']

basedir = './results_v2/ablation'
dirnames = np.array([d for d in Path(basedir).iterdir() if d.is_dir()])
metrics_similarity_mean = {}
metrics_similarity_ci = {}
filter_idx = ('PCA', 'Baseline', slice(None), 'MCS')
for dirname in dirnames:
    metric_name = dirname.name.capitalize().replace('_signed', ' (signed)')
    y_pred_metric = load_y(Path(dirname, 'pan_cancer_stratified', 'TCGA'))
    _, df_mean, df_ci = get_performance(y_pred_metric)
    
    df_mean = df_mean.loc[filter_idx]
    df_ci = df_ci.loc[filter_idx]
    metrics_similarity_mean[metric_name] = df_mean
    metrics_similarity_ci[metric_name] = df_ci
    
lower_idx = df_mean.index

# Add the default (Pearson) from previously loaded results
metrics_similarity_mean['Pearson'] = (metrics_mean
                                      .loc[filter_idx]
                                      .loc[lower_idx, :])
metrics_similarity_ci['Pearson'] = (metrics_ci
                                    .loc[filter_idx]
                                    .loc[lower_idx, :])

metrics_dict = {'mean': metrics_similarity_mean,
                'ci': metrics_similarity_ci}
for k, df in metrics_dict.items():
    df = pd.concat(df)
    df.index.names = ['Distance'] + list(df.index.names)[1:]
    df = df.reorder_levels([1, 0]).sort_index()
    metrics_dict[k] = df
metrics_similarity_mean = metrics_dict['mean']
metrics_similarity_ci = metrics_dict['ci']

#%% TCGA [load] runtimes

basedir = './results_reworked'
runtime_dir = 'training_times'
transform_name = 'PCA'
augment_name = 'Baseline'
classifier_names = ['RF', 'XGB']

path_experiment = Path(basedir, 'pan_cancer_stratified', 'TCGA')

t_model = {}
t_similar_sample = {}

for classifier_name in classifier_names:
    path_runtime_clf = Path(
        path_experiment, 'training_times',
        transform_name, augment_name, classifier_name)
    for path_fold in sorted(list([d for d in path_runtime_clf.iterdir() if d.is_dir()])):
        fold_dirname = path_fold.name
        print(classifier_name, '\t', fold_dirname)
        fold_val = int(fold_dirname.split('fold_')[-1])
        t_baseline_train = float(np.loadtxt(Path(path_fold, 'runtime_baseline_train.txt')))
        t_baseline_predict = float(np.loadtxt(Path(path_fold, 'runtime_baseline_predict.txt')))
        t_model[(classifier_name, fold_val, 'train')] = t_baseline_train
        t_model[(classifier_name, fold_val, 'predict')] = t_baseline_predict
        
        for path_sample in sorted(list(path_fold.glob('*runtime_mcs_predict_sample*'))):
            sample_id = path_sample.stem.split('_')[-1]
            t_mcs_predict = float(np.loadtxt(path_sample))
            t_model[(classifier_name, fold_val, f'mcs-predict-{sample_id}')] = t_mcs_predict
            
            path_local_samples = Path(
                path_experiment, 'data', transform_name, augment_name,
                fold_dirname, 'runtime_local_set')
            
            similar_sample_idx = (fold_val, sample_id)
            if similar_sample_idx in t_similar_sample:
                continue
            t_local_samples = {}
            for p_local_samples in path_local_samples.rglob(f'*runtime_local_set_sample_{sample_id}.csv'):
                h_label = p_local_samples.parent.name
                t_local_samples[h_label] = float(np.loadtxt(p_local_samples))
            t_similar_sample[similar_sample_idx] = pd.Series(t_local_samples)
          
t_model = pd.Series(t_model)
t_baseline_train = t_model.loc[(slice(None), slice(None), 'train')].reset_index()
t_baseline_train.columns = ['Base classifier', 'Fold', 'Time']
t_baseline_predict = t_model.loc[(slice(None), slice(None), 'predict')].reset_index()
t_baseline_predict.columns = t_baseline_train.columns
t_mcs_predict = t_model.loc[t_model.index.get_level_values(-1).str.startswith('mcs')].reset_index()
t_mcs_predict.columns = ['Base classifier', 'Fold', 'Sample', 'Time']
t_mcs_predict['Sample'] = t_mcs_predict['Sample'].str.split('-').str[-1].astype(int)
t_similar_sample = pd.concat(t_similar_sample).reset_index()
t_similar_sample.columns = ['Fold', 'Sample', 'h', 'Time']

#%%% Plot

fig, axes = plt.subplots(
    nrows = 1, ncols = 4, sharey = True,
    figsize = (5, 2),
    gridspec_kw = {'width_ratios': [5, 1, 5, 5]})

# Plot baseline train
ax = axes[0]
sns.barplot(
    data = t_baseline_train, y = 'Time',
    hue = 'Base classifier', hue_order = classifier_names,
    ax = ax, legend = False)
ax.set_xlabel('Train\nglobal model')
ax.set_title('Traning')
ax.set_ylabel('Time (seconds)')

axes[1].axis('off')

# Plot sample selection
ax = axes[2]
sns.barplot(
    data = (t_similar_sample
            .groupby(['Sample'])['Time']
            .sum()
            .reset_index()),
    y = 'Time',
    ax = ax,
    width = 0.4,
    color = 'C2')
ax.set_xlim(axes[0].get_xlim())
ax.set_xlabel('Find similar\nsamples')

# Plot MCS prediction
ax = axes[3]
sns.barplot(
    data = t_mcs_predict, y = 'Time',
    hue = 'Base classifier', hue_order = classifier_names,
    ax = ax)
ax.set_xlabel('Train\nlocal models')

gs = fig.add_gridspec(1, 4, width_ratios = [5, 1, 5, 5])
fig.add_subplot(gs[:, 2:], frameon = False)
#fig.add_subplot(132, frameon=True)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.title('Inference (6 local models)')

axes[3].legend(loc = 'upper left', bbox_to_anchor = [1, 1], title = 'Classifier')

savefig('runtimes')
plt.show()

#%% TCGA [load] h-thresholds

basepath = './results_v2/pan_cancer_stratified'
h_paths = Path(basepath).rglob('*H.csv')

H = {}
for h_path in h_paths:
    fold = int(h_path.parent.name.split('_')[-1])
    # Thresholds are determined from pre-augmentation data.
    # Augmentation does not affect thresholds, kept in config for completeness.
    augment_name = h_path.parts[-3]
    transform_name = h_path.parts[-4]
    dataset_name = h_path.parts[-6]
    H_cfg = pd.read_csv(h_path, header = None, sep = '\t')
    H_cfg.columns = ['h_label', 'h_value']
    H_cfg = H_cfg.set_index('h_label', drop = True)
    H[(dataset_name, transform_name, augment_name, fold)] = H_cfg
H = pd.concat(H)
H.index.names = ['Dataset', 'Transform', 'Augment', 'Fold', 'h_label']

#%%% Plot

dataset_name = 'TCGA'
transform_name = 'PCA'
augment_name = 'Baseline'

df = H.loc[(dataset_name, transform_name, augment_name), :].reset_index()
sns.boxplot(data = df, x = 'h_label', y = 'h_value')
plt.xlabel('Similarity threshold percentile')
plt.ylabel('Similarity threshold value')
plt.show()

#%% TCGA: [calc] pairwise similarity and dataset-wise similarity thresholds

dataset_name = 'TCGA'
similarity_metric = 'pearson'
use_absolute_percentiles = True
#threshold_selector = FixedThresholdSelector(
#    H = np.arange(0.15, 0.25+0.025, 0.025).round(3)),
threshold_selector = PercentileThresholdSelector(
    65, 90, 6, similarity_metric, use_absolute_similarity = True)

X = tcga_dataset.X
y = tcga_dataset.y
ctypes = tcga_dataset.cancer_type
Xt = PCA(n_components = 0.95).fit_transform(tcga_dataset.X)
if similarity_metric == 'pearson':
    corr = np.corrcoef(Xt)
elif similarity_metric == 'spearman':
    corr = spearmanr(Xt)[0]
# Set diagonal to zero to avoid counting self-loops
np.fill_diagonal(corr, 0)
corr_flat = squareform(corr, checks = False)
H = threshold_selector.get_thresholds(Xt, as_series = True)
del Xt

# Determine similarity percentiles

if use_absolute_percentiles:
    corr_std = np.std(np.abs(corr_flat))
    percentile_std = (np.abs(corr_flat) < corr_std).sum() *100 / len(corr_flat)
    percentile_65 = np.percentile(np.abs(corr_flat), 65)
    percentile_90 = np.percentile(np.abs(corr_flat), 90)
else:
    corr_std = np.std(corr_flat)
    percentile_std = (corr_flat < corr_std).sum() *100 / len(corr_flat)
    percentile_65 = np.percentile(corr_flat, 65)
    percentile_90 = np.percentile(corr_flat, 90)
print(f'Metric              : {similarity_metric}')
print(f'Standard deviation  : {corr_std:0.3f} ({percentile_std:0.2f}th percentile)')
print(f'65th percentile     : {percentile_65:0.3f}')
print(f'90th percentile     : {percentile_90:0.3f}')

#%%% Plot correlation histogram

draw_threshold_lines = True
bins = 100
linewidth = 1
linestyle = '-'
line_pad_percent = 15
linecolor = 'C3'

if H.index.astype(float).min() < 1:
    threshold_text_formatter = r'$ ({:.3f})'
else:
    threshold_text_formatter = r'$ ({:.0f}th)'
plt.figure(figsize = (6, 3))
plt.hist(corr_flat, bins = bins, 
         alpha = 1, color = 'C0')
if draw_threshold_lines:
    ylims = plt.gca().get_ylim()
    yticks = plt.gca().get_yticks()
    y_top = ylims[1]
    y_pad = ylims[1] * line_pad_percent/100
    for i, (h_label, h) in enumerate(H.reset_index().values):
        h_label = float(h_label)
        plt.plot([-h, -h, h, h], [0, y_top, y_top, 0],
                 color = linecolor, linewidth = linewidth, linestyle = linestyle)
        line_label = r'$h_' + str(len(H)-i) + threshold_text_formatter.format(h_label)
        plt.text(0, y_top + y_pad*0.05, line_label, va = 'bottom', ha = 'center',
                 color = linecolor)
        y_top += y_pad
    plt.ylim(0, plt.gca().get_ylim()[1]+y_pad//2)
    plt.yticks(yticks)
    
plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
offset_text = plt.gca().yaxis.get_offset_text()
offset_text.set_x(-0.01)
offset_text.set_ha('right')
offset_text.set_va('top')
plt.ylabel('Number of sample pairs')
plt.xlabel('PCC')
savefig('tcga_pairwise_correlation_distribution')
plt.title('Distribution of pairwise correlation in TCGA samples')
plt.show()
std = np.std(np.abs(corr_flat))
print(f'Standard deviation of abs correlations: {std:.2f}')
percentiles = percentileofscore(np.abs(corr_flat), H)
percentiles = pd.DataFrame(percentiles, index = H,
                           columns = ['absolute correlation percentile'])
percentiles.index.name = 'threshold (h)'
print(percentiles.round(2).reset_index())

#%% TCGA: [load] number of neighboring samples for each h-threshold

basedir = Path('.', 'results_v2', 'pan_cancer_stratified', 'TCGA')
transform_name = 'PCA'
augment_name = 'Baseline'
classifier_names = ['RF', 'XGB']


y_pred_cfg = y_pred_combined[(transform_name, augment_name)]
y_pred_cfg = y_pred_cfg[
      [clf_name + '-Baseline' for clf_name in classifier_names] \
    + ['MCS-' + clf_name for clf_name in classifier_names] \
    + ['y_true']]
y_true_cfg = y_pred_cfg.pop('y_true')
n_neighbors = {}
# Load number of neighbors
paths_n_neighbors = sorted(list(Path(basedir, 'data', transform_name, augment_name).rglob('*local_set_idx_sample*')))
for path_n_neighbors in paths_n_neighbors:

    sample_id = int(path_n_neighbors.stem.split('_')[-1])
    fold = int(path_n_neighbors.parts[-4].split('_')[-1])
    h_label = path_n_neighbors.parent.name
    n_neighbors_sample = len(np.loadtxt(path_n_neighbors))
    n_neighbors[(fold, sample_id, h_label)] = n_neighbors_sample

n_neighbors = pd.Series(n_neighbors)
n_neighbors.index.names = ['Fold', 'Sample', 'h_label']
n_neighbors = n_neighbors.unstack(level = 'h_label').sort_index().reset_index(drop = True)

#%% TCGA: [plot] performance grouped by number of neighbors

metric_names = ['AUC', 'F1', 'Balanced accuracy']
n_bins = 3
h = 75.0

n_neighbors_binned = pd.qcut(n_neighbors[str(h)], n_bins, precision = 0)
metrics_binned = {}
for bin_ in n_neighbors_binned.unique().sort_values():
    mask = n_neighbors_binned == bin_
    bin_str = str('{:,}'.format(int(bin_.left))) \
              + ' - ' + str('{:,}'.format(int(bin_.right)-1))
    y_true_bin = y_true_cfg.loc[mask]
    y_pred_bin = y_pred_cfg.loc[mask, :]
    metrics_bin = scores_to_metrics(y_pred_bin.T, y_true_bin) 
    metrics_bin = metrics_bin.reset_index()
    metrics_bin['Classifier'] = metrics_bin['index'].apply(
        lambda val: val.replace('-Baseline', '').replace('MCS-', ''))
    metrics_bin['Model type'] = metrics_bin.apply(
        lambda row: row['index'].replace(row['Classifier'], '').replace('-', ''), 
        axis=1)
    metrics_bin = (metrics_bin
                   .drop('index', axis = 1)
                   .set_index(['Classifier', 'Model type'], drop = True)
                   .sort_index())
    metrics_binned[bin_str] = metrics_bin
    
bin_colname = '# of neighbors'
metrics_binned = pd.concat(metrics_binned).reorder_levels([1, 2, 0]).sort_index()
metrics_binned.index.names = ['Classifier', 'Model type', bin_colname]

ncols = len(metrics_binned.index.get_level_values('Classifier').unique())
nrows = len(metric_names)
fig, axes = plt.subplots(nrows = nrows, ncols = ncols,
                         figsize = (ncols*2, nrows),
                         sharex = True, sharey = True,
                         squeeze = False)
for i, metric_name in enumerate(metric_names):
    for j, classifier_name in enumerate(metrics_binned.index.get_level_values('Classifier').unique()):
        ax = axes[i, j]
        data = metrics_binned.loc[classifier_name, metric_name].reset_index()
        data.columns = list(data.columns[:-1]) \
                        + [data.columns[-1].replace('Balanced accuracy',
                                                    'Bal. acc.')]
        x_order = sorted(data[bin_colname].unique(),
                       key = lambda val: int(val.split(' -')[0].replace(',', '')))
        sns.barplot(data, x = bin_colname, y = data.columns[-1], hue = 'Model type',
                    order = x_order,
                    ax = ax, width = 0.5)
        if i == 0 and j == ncols - 1:
            ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], title = 'Model type')
        else:
            ax.get_legend().remove()
        if j != 0:
            ax.set_ylabel('')
        if i == 0 and len(classifier_names) > 1:
            ax.set_title(classifier_name)
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlabel('')
        if i == len(metric_names) - 1:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(bin_colname, labelpad = 50)
plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
savefig('tcga_correlation_grouped_performance')
plt.show()

#%%% Plot distribution of the % of correlated neighbors(for fixed H)

H = np.arange(0.15, 0.25+0.025, 0.025).round(3)
H = n_neighbors.columns
fig, axes = plt.subplots(nrows = len(H), figsize = (6, len(H)),
                         sharex = True)
for i, (ax, h) in enumerate(zip(axes, H)):
    ax.hist(n_neighbors[h], bins = 50,
            color = f'C{i}', label = str(h))
    ax.legend()
ax.set_xlabel('Distribution of the % of neighbors with\n' \
              + r'absolute correlation $\geq$ threshold percentile')
plt.show()

#%% TCGA: count neighbors by type

h = 0.2
bin_count = 15

ctypes = tcga_dataset.cancer_type
n_cancers = np.unique(ctypes).size
mask = pd.DataFrame(np.abs(corr) >= h)
neighbors = mask.groupby(ctypes).sum().T
index = np.array([np.where(neighbors.columns == ctype)[0][0] for ctype in ctypes])
neighbor_count = neighbors.values[range(len(neighbors)), index]
total_count = pd.Series(ctypes)
total_count = total_count.map(total_count.value_counts())
ctype_count = pd.Series(ctypes).value_counts()
neighbor_percent = neighbor_count * 100 / total_count

# Plot

split_by_types = True
n_cols = 4
n_rows = int(np.ceil(n_cancers / n_cols))

bin_range = np.linspace(0, neighbor_percent.max(), bin_count)
n_rows = int(np.ceil(n_cancers / n_cols))
fig, axes = plt.subplots(
    nrows = n_rows,
    ncols = n_cols,
    figsize = (n_cols*2.5, n_rows*1)
)

data = pd.DataFrame(
    [ctypes, tcga_dataset.y],
    index = ['cancer type', 'outcome']).T
data['percent of intra-cancer neighbors'] = neighbor_percent.values

for i, cancer_type in enumerate(neighbors.columns):
    ax = axes.flat[i]
    df = data[data['cancer type'] == cancer_type]
    #data['PFI'] = data['PFI'].replace(pfi_map)
    if split_by_types:
        sns.histplot(
            data = df,
            x = 'percent of intra-cancer neighbors',
            hue = 'outcome',
            multiple = 'stack',
            ax = ax, bins = bin_range,
            linewidth = 0)
    else:
        sns.histplot(
            data = df,
            x = 'percent of intra-cancer neighbors',
            ax = ax, bins = bin_range,
            linewidth = 0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    txt = f'{cancer_type}\nn = {ctype_count[cancer_type]}'
    plt.text(0.99, 0.12, txt,
             transform = ax.transAxes, ha = 'right', va = 'bottom')
    ax.set_xlim(-10, 180)
    ax.set_xticks([0, 50, 100])
    if i + n_cols < n_cancers:
        ax.set_xticklabels([])
    sns.despine(ax = ax)
    if i == n_cols - 1 and split_by_types:
        sns.move_legend(
            ax, "center left",
            bbox_to_anchor = [1.1, 0.5],
            title = 'Status',
            frameon = False)
    else:
        ax.legend().remove()
                        
i += 1
while i < axes.size:
    axes.flat[i].axis('off')
    i += 1
    
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'% of neighbors ($|\sigma| \geq $' + f'{h:.3f}) from within cancer', labelpad = 10)
plt.ylabel('Count', labelpad = 10)
plt.subplots_adjust(wspace = 0.3, hspace = 0.2)
plt.show()

#%% TCGA: network similarity for different distance measures

X = tcga_dataset.X
Xt = PCA(n_components = 0.95).fit_transform(X)

dist_functions = {
        'Pearson': lambda X: squareform(np.corrcoef(X), checks = False),
        'Spearman': lambda X: squareform(pd.DataFrame(X).T.corr(method = 'spearman').values, checks = False),
        'Cosine': lambda X: squareform(cosine_similarity(X), checks = False)
    }

nets = {k: func(Xt) for k, func in dist_functions.items()}

# Determine equivalent thresholds for other distance metrics
h = 0.95
q = percentileofscore(np.abs(corr_flat), h)
thresholds = {'Pearson': h}
for dist_name, net in dist_functions.items():
    if dist_name not in thresholds:
        thresholds[dist_name] = np.percentile(np.abs(nets[dist_name].flat), q)
        
# Count percent of overlapping edges for every pair

sim = pd.DataFrame(np.nan,
                   index = dist_functions.keys(),
                   columns = dist_functions.keys())
edges = {}
for f1, f2 in list(combinations_with_replacement(dist_functions.keys(), 2)):
    a1 = np.abs(nets[f1].flat) >= thresholds[f1]
    a2 = np.abs(nets[f2].flat) >= thresholds[f2]
    edges[f1] = a1.sum()
    sim_pair = np.multiply(a1, a2).sum() / a1.sum()
    sim.loc[f1, f2] = sim_pair
    sim.loc[f2, f1] = sim_pair
    
# Plot

fig, axes = plt.subplots(nrows = 2)
axes[0].hist(nets['Pearson'], bins = 100)
axes[1].hist(nets['Spearman'], bins = 100)
axes[0].set_ylabel('Pearson')
axes[1].set_ylabel('Spearman')
plt.subplots_adjust(hspace = 0.5)
plt.show()