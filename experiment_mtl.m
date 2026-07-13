% Required package:
% http://www.yelabs.net/software/MALSAR/#download

%% Utility functions

% Get list of prefix-matched files and associated paths in directory
function [entry_names, entry_paths] = listdir(path, prefix)
    entries = dir(path);
    % Exclude .. and . directories
    entries = entries(...
        ~ismember({entries.name}, {'.','..'}));
    entry_paths = fullfile({entries.folder}, {entries.name});
    entry_names = {entries.name};
    if nargin == 2
        mask = startsWith(entry_names, prefix);
        entry_names = entry_names(mask);
        entry_paths = entry_paths(mask);
    end
    [~, idx] = sort(entry_names);
    entry_names = entry_names(idx);
    entry_paths = entry_paths(idx);
end

% Sigmoid function to transform raw scores into probabilities
function [yy] = sigmoid(y)
    yy = 1 ./ (1 + exp(-y));
end

% load pre-stored data (produced by experiment.py)
function [X_train, X_test, y_train, y_test, ctype_train, ctype_test] = loadData(path)
    X_train = readmatrix(fullfile(path, 'X_train_augmented.csv'));
    [X_test_names, X_test_paths] = listdir(...
        fullfile(path, 'X_test_augmented'), 'X_test_augmented');
    X_test = cell(2, length(X_test_paths));
    for i=1:length(X_test_paths)
        xt = readmatrix(X_test_paths{i});
        X_test{1, i} = X_test_names{i};
        X_test{2, i} = xt;
    end
    y_train = readmatrix(fullfile(path, 'y_train_augmented.csv'));
    y_test = readmatrix(fullfile(path, 'y_test.csv'));
    ctype_train = table2cell(readtable(fullfile(path, 'ctypes_train.csv'), ...
        'Delimiter', '\n', 'ReadVariableNames', false));
    ctype_test = table2cell(readtable(fullfile(path, 'ctypes_test.csv'), ...
        'Delimiter', '\n', 'ReadVariableNames', false));
    % y is [0, 1], while MALSAR requires [-1, 1] labels
    y_train = y_train*2 - 1;
    y_test = y_test*2 - 1;
end

% Get data as task-specific cell arrays
function [X_train_mtl, y_train_mtl] = getAsMTLData(...
    X_train, y_train, ctypes, ctypes_unique)
    X_train_mtl = cell(size(ctypes_unique));
    y_train_mtl = cell(size(ctypes_unique));
    for t=1:size(ctypes_unique, 1)
        ctype = ctypes_unique{t};
        X_train_mtl{t} = X_train(strcmp(ctypes, ctype), :);
        y_train_mtl{t} = y_train(strcmp(ctypes, ctype));
    end
end

% Train multi-task learning model
function [W, c] = trainMTL(X_train_mtl, y_train_mtl, method_name, opts)
    if nargin < 4
        opts.maxIter = 1000;
        opts.init = 0;
    end
    if strcmp(method_name, 'CMTL')
        k = 1;     % number of clusters
        rho_L1 = 0.5; % L1-norm coefficient
        rho_L2 = 1; % L2-norm coefficient
        [W, c] = Logistic_CMTL(X_train_mtl, y_train_mtl, rho_L1, rho_L2, k, opts);
    elseif strcmp(method_name, 'CASO')
        k = 7;     % number of clusters
        rho_L1 = 1; % L1-norm coefficient
        rho_L2 = 0.9; % L2-norm coefficient
        [W, c] = Logistic_CASO(X_train_mtl, y_train_mtl, rho_L1, rho_L2, k, opts);
    elseif strcmp(method_name, 'LASSO')
        rho_L1 = 0.5;
        opts.rho_L2 = 1;
        [W, c] = Logistic_Lasso(X_train_mtl, y_train_mtl, rho_L1, opts);
    else
        error(fprintf('Unknown MTL method: %s', method_name));
    end
end

% Perform inference
function y = getAggregatedPredictions(X, ctypes, ctypes_unique, W, c, applySigmoid)
    if nargin < 6
        applySigmoid = true; % Default value
    end
    y = zeros(length(X), 1);
    for i = 1:length(y)
        t = strcmp(ctypes_unique, ctypes{i});
        if iscell(X)
            y_pred_all = X{i} * W(:, t) + c(t);
            if applySigmoid
                y_pred_all = sigmoid(y_pred_all);
            end
            y(i) = mean(y_pred_all);
        else
            y(i) = X(i, :) * W(:, t) + c(t);
            if applySigmoid
                y(i) = sigmoid(y(i));
            end
        end
    end
end

% Write inference results to file
function writeYToFile(y, filenames, dirname, model_name)
    for i=1:length(y)
        text = sprintf("%s\t%s", model_name, num2str(y(i),'%.6f '));
        path = fullfile(dirname, filenames{i});
        [~, ~] = mkdir(dirname);
        writelines(text, path);
    end
end


%% Train MTL models

%--------------------------------------------------------------------------
% Experiment configuration
%--------------------------------------------------------------------------
mtl_model_names = {'LASSO', 'CMTL', 'CASO'};
basedir = 'results';
experiment_dir = fullfile('.', basedir, ...
    'pan_cancer_stratified', 'TCGA');
apply_sigmoid = true;
%--------------------------------------------------------------------------

data_dir = fullfile(experiment_dir, 'data', '*' ,'*');
output_dir = fullfile(experiment_dir, 'pred_probability');
[~, dirnames] = listdir(data_dir);
auc = table();

for model_index = 1:length(mtl_model_names)
    model_name = mtl_model_names{model_index};
    for i = 1:length(dirnames)
        [parent, augment_name, ~] = fileparts(dirnames{i});
        [~, transform_name, ~] = fileparts(parent);
        [~, data_paths] = listdir(dirnames{i});
        for j = 1:length(data_paths)
            [~, fold, ~] = fileparts(data_paths{j});
            if ~strcmp(fold, 'fold_0') || ~(model_index == 1) || ~(i == 1)
                %continue;
            end
            fprintf("=========================================\n")
            fprintf("Transform      : %s\n", transform_name);
            fprintf("Augment        : %s\n", augment_name);
            fprintf("Fold           : %s\n", fold);
            fprintf("Classifier     : %s\n", model_name);
            
            % Load data
            [X_train, X_test, y_train, y_test, ctypes_train, ctypes_test] ...
                = loadData(data_paths{j});
            ctypes_unique = unique([ctypes_train; ctypes_test]);
            fprintf("Train/Test     : %d/%d\n", length(X_train), length(X_test));
            fprintf("-----------------------------------------\n")
    
            fprintf("\tPreparing data...\n");
            X_test_names = X_test(1, :);
            X_test_features = X_test(2, :);
            [X_train_mtl, y_train_mtl] = getAsMTLData(...
                X_train, y_train, ctypes_train, ctypes_unique);
    
            % Fit model
            fprintf("\tFitting model...\n");
            [W, c] = trainMTL(X_train_mtl, y_train_mtl, model_name);
    
            % Calculate predictions for test data
            fprintf("\tCalculating predicted output...\n");
            y_pred = getAggregatedPredictions(...
                X_test_features, ctypes_test, ctypes_unique, ...
                W, c, apply_sigmoid); 
            y_train_pred = getAggregatedPredictions( ...
                X_train, ctypes_train, ctypes_unique, ...
                W, c, apply_sigmoid);
    
            % Calculate AUC
            [~, ~, ~, auc_train] = perfcurve(y_train, y_train_pred, 1);
            [~, ~, ~, auc_test] = perfcurve(y_test, y_pred, 1);
            auc_cfg = {transform_name, augment_name, model_name, fold, ...
                auc_train, auc_test};
            auc_cfg = cell2table(auc_cfg, ...
                'VariableNames', {'Transform', 'Augment', 'Model', 'Fold', ...
                'Train AUC', 'Test AUC'});
            auc = [auc; auc_cfg];
        
            fprintf("\tTrain AUC: %.3f, test AUC: %.3f\n", auc_train, auc_test);
    
            % Write output to file
            output_path = fullfile(...
                output_dir, transform_name, augment_name, model_name, fold);
            output_files = replace(replace(X_test_names, ...
                'X_test_augmented', 'sample'), ...
                '.csv', '.txt');
            writeYToFile(y_pred, output_files, output_path, model_name);
        end
    end
end
fprintf("Completed run.\n");
mean_auc = groupsummary(auc, ["Transform", "Augment", "Model"], ...
    "mean", ["Train AUC", "Test AUC"]);
disp(mean_auc);

%% CASO hyperparams grid search

% Options
opts.maxIter = 1000;
opts.init = 0;
rho_L1 = 1;
rho_L2_vals = (0.1:0.2:1);
k_vals = (2:3:15);
basedir = 'results';
data_dir = fullfile('.', basedir, ...
    'pan_cancer_stratified', 'TCGA', 'data', 'PCA', 'Baseline');
apply_sigmoid = true;


auc_vals = zeros([length(k_vals) length(rho_L2_vals)]);
nconfigs = length(k_vals)*length(rho_L2_vals);
for i = 1:length(k_vals)
    for j = 1:length(rho_L2_vals)
        
        [~, data_paths] = listdir(data_dir);
        
        k = k_vals(i);
        rho_L2 = rho_L2_vals(j);
        aucs_folds = zeros(1, length(data_paths));
        for data_path_idx = 1:length(data_paths)
            % Load data
            [X_train, X_test, y_train, y_test, ctypes_train, ctypes_test] ...
                = loadData(data_paths{data_path_idx});
            ctypes_unique = unique([ctypes_train; ctypes_test]);
            fprintf("Train/Test     : %d/%d\n", length(X_train), length(X_test));
            fprintf("Config         : %d/%d\n", ...
                i*length(k_vals) + j, nconfigs);
            fprintf("-----------------------------------------\n")
    
            fprintf("\tPreparing data...\n");
                X_test_names = X_test(1, :);
                X_test_features = X_test(2, :);
                [X_train_mtl, y_train_mtl] = getAsMTLData(...
                    X_train, y_train, ctypes_train, ctypes_unique);
            
            % Fit model
            fprintf("\tFitting model...\n");
            [W, c] = Logistic_CASO(X_train_mtl, y_train_mtl, rho_L1, rho_L2, k);
            
            % Calculate predictions for test data
            y_pred = getAggregatedPredictions(...
                    X_test_features, ctypes_test, ctypes_unique, ...
                    W, c, apply_sigmoid); 
            
            % Calculate AUC
            [~, ~, ~, auc_test] = perfcurve(y_test, y_pred, 1);
            aucs_folds(data_path_idx) = auc_test;
        end
        auc_vals(i, j) = mean(aucs_folds);
    end
end
[max_val, linear_idx] = max(auc_vals, [], "all");
[maxrow, maxcol] = ind2sub(size(auc_vals), linear_idx);
best_k = k_vals(maxrow);
best_rho_L2 = rho_L2_vals(maxcol);
heatmap(rho_L2_vals, k_vals, auc_vals, ...
    'XLabel', '\rho_{L2}', 'YLabel', 'k', 'Title', ...
    sprintf('CASO Tuning\n(metric: AUC, best k = %d, best \\rho_{L2} = %.2f', ...
    best_k, best_rho_L2));

%% LASSO hyperparams grid search

% Options
opts.maxIter = 1000;
opts.init = 0;
rho_L1_vals = (0:0.2:1);
rho_L2_vals = (0:0.2:1);
basedir = 'results';
data_dir = fullfile('.', basedir, ...
    'pan_cancer_stratified', 'TCGA', 'data', 'PCA', 'Baseline');
apply_sigmoid = true;


auc_vals = zeros([length(rho_L1_vals) length(rho_L2_vals)]);

for i = 1:length(rho_L1_vals)
    for j = 1:length(rho_L2_vals)
        
        [~, data_paths] = listdir(data_dir);
        
        rho_L1 = rho_L1_vals(i);
        rho_L2 = rho_L2_vals(j);
        aucs_folds = zeros(1, length(data_paths));
        for data_path_idx = 1:length(data_paths)
            % Load data
            [X_train, X_test, y_train, y_test, ctypes_train, ctypes_test] ...
                = loadData(data_paths{data_path_idx});
            ctypes_unique = unique([ctypes_train; ctypes_test]);
            fprintf("Train/Test     : %d/%d\n", length(X_train), length(X_test));
            fprintf("-----------------------------------------\n")
    
            fprintf("\tPreparing data...\n");
                X_test_names = X_test(1, :);
                X_test_features = X_test(2, :);
                [X_train_mtl, y_train_mtl] = getAsMTLData(...
                    X_train, y_train, ctypes_train, ctypes_unique);
            
            % Fit model
            fprintf("\tFitting model...\n");
            opts.rho_L2 = rho_L2;
            [W, c] = Logistic_Lasso(X_train_mtl, y_train_mtl, rho_L1, opts);
            
            % Calculate predictions for test data
            y_pred = getAggregatedPredictions(...
                    X_test_features, ctypes_test, ctypes_unique, ...
                    W, c, apply_sigmoid); 
            
            % Calculate AUC
            [~, ~, ~, auc_test] = perfcurve(y_test, y_pred, 1);
            aucs_folds(data_path_idx) = auc_test;
        end
        auc_vals(i, j) = mean(aucs_folds);
    end
end
[max_val, linear_idx] = max(auc_vals, [], "all");
[maxrow, maxcol] = ind2sub(size(auc_vals), linear_idx);
best_rho_L1 = rho_L1_vals(maxrow);
best_rho_L2 = rho_L2_vals(maxcol);
heatmap(rho_L2_vals, rho_L1_vals, auc_vals, ...
    'XLabel', '\rho_{L2}', 'YLabel', '\rho_{L1}', 'Title', ...
    sprintf('CASO Tuning\n(metric: AUC, best \\rho_{L1} = %.2f, best \\rho_{L2} = %.2f', ...
    best_rho_L1, best_rho_L2));
