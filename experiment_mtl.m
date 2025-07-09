% Required package:
% http://www.yelabs.net/software/MALSAR/#download

%% Load TCGA data

basedir = ".";
data_dir = sprintf("%s/datasets/TCGA/exported/", basedir);

%%

X = table2array(readtable(strcat(data_dir, 'X.csv'), 'Delimiter', '\t', 'ReadVariableNames', false));
y = table2array(readtable(strcat(data_dir, 'y.csv'), 'Delimiter', '\n', 'ReadVariableNames', false));
ctypes = table2cell(readtable(strcat(data_dir, 'c.csv'), 'Delimiter', '\n', 'ReadVariableNames', false));
% y is in [0, 1], while MALSAR requires [-1, 1] labels
y = y*2-1;

%% Prepare data as a Multitask Learning (MTL) problem

% Other MTL algorithms evaluated using 10-fold cross validation
% CV split indices loaded from python-exported files
clc;

nfolds = 10;
load_preprocessed = true;
fold_idx_dir = sprintf("%s/results/pan_cancer_stratified/TCGA", basedir);


all_idx = 1:length(y);
ctypes_unique = unique(ctypes);

% cell array of tasks for fold
data = cell(nfolds);

for fold = 1:nfolds
    if(load_preprocessed)
        fold_dir = sprintf("%sfold_%d/", data_dir, fold-1);
        X_train = readmatrix(strcat(fold_dir, "X_train.csv"), 'Delimiter', '\t');
        y_train = readmatrix(strcat(fold_dir, "y_train.csv"), 'Delimiter', '\t');
        X_test = readmatrix(strcat(fold_dir, "X_test.csv"), 'Delimiter', '\t');
        y_test = readmatrix(strcat(fold_dir, "y_test.csv"), 'Delimiter', '\t');
        ctypes_train = table2cell(readtable(strcat(fold_dir, "ctype_train.csv"), 'Delimiter', '\t', 'ReadVariableNames', false));
        ctypes_test = table2cell(readtable(strcat(fold_dir, "ctype_test.csv"), 'Delimiter', '\t', 'ReadVariableNames', false));
        y_train = y_train*2-1;
        y_test =  y_test*2-1;
    else
        fold_idx_file = sprintf("%s/LR/test_index_fold_%d.csv", fold_idx_dir, fold-1);
        test_idx = readmatrix(fold_idx_file)+1;
        train_idx = setdiff(all_idx, test_idx);
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);
        ctypes_train = ctypes(train_idx);
        ctypes_test = ctypes(test_idx);
    end
    % convert X_train and y_train into task-specific cell arrays
    X_train_mtl = cell(size(ctypes_unique));
    y_train_mtl = cell(size(ctypes_unique));
    for t=1:length(ctypes_unique)
        ctype = ctypes_unique{t};
        X_train_mtl{t} = X_train(strcmp(ctypes_train, ctype), :);
        y_train_mtl{t} = y_train(strcmp(ctypes_train, ctype));
    end
    data_fold = {X_train_mtl, y_train_mtl, X_test, y_test, ctypes_test};
    data{fold} = data_fold;
end

%% Cross-validation Experiment

% Options
opts.maxIter = 1000;
opts.init = 0;

mtl_method_list = {'LASSO', 'CMTL', 'CASO'};

for j = 1:length(mtl_method_list)
    mtl_method = mtl_method_list{j};
    fprintf("Method: %s\n==============\n", mtl_method);
    aucs = zeros(1, nfolds);
    for fold = 1:nfolds
        fprintf("Fold %d\n--------------\n", fold);
    
        X_train_mtl = data{fold}{1};
        y_train_mtl = data{fold}{2};
        X_test = data{fold}{3};
        y_test = data{fold}{4};
        ctypes_test = data{fold}{5};

        % create directory
        savedir = sprintf("%s/%s", fold_idx_dir, mtl_method);
        mkdir(savedir);

        % call corresponding function
        if strcmp(mtl_method, 'CMTL')
            k = 1;     % number of clusters
            rho_L1 = 0.5; % L1-norm coefficient
            rho_L2 = 1; % L2-norm coefficient
            [W, c] = Logistic_CMTL(X_train_mtl, y_train_mtl, rho_L1, rho_L2, k, opts);
        elseif strcmp(mtl_method, 'CASO')
            k = 10;     % number of clusters
            rho_L1 = 1; % L1-norm coefficient
            rho_L2 = 1; % L2-norm coefficient
            [W, c] = Logistic_CASO(X_train_mtl, y_train_mtl, rho_L1, rho_L2, 15, opts);
        elseif strcmp(mtl_method, 'LASSO')
            rho_L1 = 0.5;
            opts.rho_L2 = 1;
            [W, c] = Logistic_Lasso(X_train_mtl, y_train_mtl, rho_L1, opts);
        end
        scores = zeros(length(y_test), 1);
        for t = 1:length(ctypes_unique)
            ctype = ctypes_unique{t};
            mask = strcmp(ctypes_test, ctype);
            X_test_ctype = X_test(mask, :);
            y_test_ctype = y_test(mask);
            scores(mask) = (X_test_ctype * W(:, t) + c(t));
        end

        % write scores to file
        outfile = sprintf("%s/scores_fold_%d.csv", savedir, fold-1);
        writematrix(num2str(scores,'%.6f '), outfile);
        
        % debug
        y_pred = (scores > 0)*2-1;
        fprintf("Fold %d\n========\n", fold);
        fprintf("y_pred: %d pos, %d neg, %d total\n", ...
            [sum(y_pred == 1); sum(y_pred == -1); length(y_pred)]);
        fprintf("y_true: %d pos, %d neg, %d total\n", ...
            [sum(y_test == 1); sum(y_test == -1); length(y_test)]);
        fprintf("Accuracy: %.3f percent\n", ...
            sum(y_pred == y_test)*100/length(y_test));
        [~, ~, ~, auc] = perfcurve(y_test, scores, 1);
        aucs(fold) = auc;
        fprintf("AUC: %.3f\n", auc);
        fprintf("\n");
    end
    fprintf("Mean AUC: %3f\n", mean(auc));
end

%% CMTL hyperparams grid search

% Options
opts.maxIter = 1000;
opts.init = 0;

rho_L1 = 1;
rho_L2_vals = (0.1:0.2:1);
k_vals = (1:3:25);

auc_vals = zeros([length(k_vals) length(rho_L2_vals)]);

for i = 1:length(k_vals)
    for j = 1:length(rho_L2_vals)
        aucs_folds = zeros(1, nfolds);
        k = k_vals(i);
        rho_L2 = rho_L2_vals(j);
        fprintf("k : %d, L2 penalty: %.2f (config %d/%d)\nFold: ", ...
                k, rho_L2, (i-1)*length(rho_L2_vals)+j, length(k_vals)*length(rho_L2_vals));
        for fold = 1:nfolds
            fprintf("%d ", fold);
            X_train_mtl = data{fold}{1};
            y_train_mtl = data{fold}{2};
            X_test = data{fold}{3};
            y_test = data{fold}{4};
            ctypes_test = data{fold}{5};
            opts.rho_L2 = rho_L2;
            [W, c] = Logistic_CASO(X_train_mtl, y_train_mtl, rho_L1, rho_L2, k);
            scores = zeros(length(y_test), 1);
            for t = 1:length(ctypes_unique)
                ctype = ctypes_unique{t};
                mask = strcmp(ctypes_test, ctype);
                X_test_ctype = X_test(mask, :);
                y_test_ctype = y_test(mask);
                scores(mask) = (X_test_ctype * W(:, t) + c(t));
            end
            [~, ~, ~, auc] = perfcurve(y_test, scores, 1);
            aucs_folds(fold) = auc;
        end
        fprintf("\nMean AUC: %3f\n", mean(aucs_folds));
        auc_vals(i, j) = mean(aucs_folds);
    end
end
heatmap(rho_L2_vals, k_vals, auc_vals, ...
    'XLabel', '\rho_{L2}', 'YLabel', 'k', 'Title', 'CASO AUC');

%% LASSO hyperparams grid search

% Options
opts.maxIter = 1000;
opts.init = 0;

rho_L1_vals = (0:0.2:1);
rho_L2_vals = (0:0.2:1);

auc_vals = zeros([length(rho_L1_vals) length(rho_L2_vals)]);

for i = 1:length(rho_L1_vals)
    for j = 1:length(rho_L2_vals)
        aucs_folds = zeros(1, nfolds);
        rho_L1 = rho_L1_vals(i);
        rho_L2 = rho_L2_vals(j);
        fprintf("L1 penalty: %.2f, L2 penalty: %.2f (config %d/%d)\nFold: ", ...
                rho_L1, rho_L2, (i-1)*length(rho_L2_vals)+j, length(rho_L1_vals)*length(rho_L2_vals));
        for fold = 1:nfolds
            fprintf("%d ", fold);
            X_train_mtl = data{fold}{1};
            y_train_mtl = data{fold}{2};
            X_test = data{fold}{3};
            y_test = data{fold}{4};
            ctypes_test = data{fold}{5};
            opts.rho_L2 = rho_L2;
            [W, c] = Logistic_Lasso(X_train_mtl, y_train_mtl, rho_L1, opts);
            scores = zeros(length(y_test), 1);
            for t = 1:length(ctypes_unique)
                ctype = ctypes_unique{t};
                mask = strcmp(ctypes_test, ctype);
                X_test_ctype = X_test(mask, :);
                y_test_ctype = y_test(mask);
                scores(mask) = (X_test_ctype * W(:, t) + c(t));
            end
            [~, ~, ~, auc] = perfcurve(y_test, scores, 1);
            aucs_folds(fold) = auc;
        end
        fprintf("\nMean AUC: %3f\n", mean(aucs_folds));
        auc_vals(i, j) = mean(aucs_folds);
    end
end
heatmap(rho_L2_vals, rho_L1_vals, auc_vals, ...
    'XLabel', '\rho_{L2}', 'YLabel', '\rho_{L1}', 'Title', 'LASSO AUC');