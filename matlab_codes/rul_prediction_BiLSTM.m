clc; clear; close all;

%% ================================================================
%% PURE MATHEMATICS PCA + SVR | C-MAPSS FD001-FD004
%% Step by step from scratch - no built-in PCA functions
%% ================================================================

base    = "C:\Users\shrim\Downloads\archive (4) (1)\archive (4)\CMaps\";
RUL_CAP = 125;
WINDOW  = 30;
datasets = {'FD001','FD002','FD003','FD004'};

for ds_idx = 1:4

    dsname = datasets{ds_idx};
    fprintf('\n========== %s ==========\n', dsname);

    %% ============================================================
    %% STEP 1: LOAD RAW DATA
    %% ============================================================
    train        = readmatrix(base + "train_" + dsname + ".txt");
    test         = readmatrix(base + "test_"  + dsname + ".txt");
    rul_test_raw = readmatrix(base + "RUL_"   + dsname + ".txt");

    rul_test = min(rul_test_raw, RUL_CAP);
    fprintf('STEP 1: Loaded. Train=%d rows, Test engines=%d\n', ...
        size(train,1), length(rul_test));

    %% ============================================================
    %% STEP 2: COMPUTE RUL LABELS FOR TRAINING DATA
    %% Piecewise linear: flat at RUL_CAP, then linearly decreasing
    %% RUL(t) = min( max_cycle - current_cycle , RUL_CAP )
    %% ============================================================
    units     = unique(train(:,1));
    RUL_train = zeros(size(train,1), 1);

    for i = 1:length(units)
        idx            = train(:,1) == units(i);
        max_cycle      = max(train(idx, 2));
        raw_rul        = max_cycle - train(idx, 2);   % countdown to failure
        RUL_train(idx) = min(raw_rul, RUL_CAP);       % cap at 125
    end
    fprintf('STEP 2: RUL labels computed (capped at %d)\n', RUL_CAP);

    %% ============================================================
    %% STEP 3: SENSOR SELECTION
    %% Remove sensors {1,5,6,10,16,18,19} - constant, no info
    %% Retain 14 sensors as per Dida(2025) and Peng(2022)
    %% ============================================================
    EXCLUDED     = [1, 5, 6, 10, 16, 18, 19];
    all_s_cols   = 6 : size(train,2);          % columns for s1..s21
    excl_cols    = 5 + EXCLUDED;               % map sensor# to column#
    kept_cols    = setdiff(all_s_cols, excl_cols);  % 14 columns

    X_train = train(:, kept_cols);   % [N_train x 14]
    X_test  = test(:,  kept_cols);   % [N_test  x 14]
    fprintf('STEP 3: %d sensors retained\n', length(kept_cols));

    %% ============================================================
    %% STEP 4: STANDARDISATION (Z-SCORE)
    %% For each feature j:
    %%   mu_j    = mean of column j (training only)
    %%   sigma_j = std  of column j (training only)
    %%   z_ij    = (x_ij - mu_j) / sigma_j
    %%
    %% WHY: PCA needs zero-mean data. Z-score also removes
    %%      scale differences between sensors.
    %% ============================================================
    n_train  = size(X_train, 1);
    n_feat   = size(X_train, 2);

    % Compute mean and std from TRAINING data only
    mu_train    = sum(X_train, 1) / n_train;          % [1 x 14]
    diff_mat    = X_train - mu_train;                  % [N x 14]
    var_train   = sum(diff_mat.^2, 1) / (n_train - 1);% [1 x 14]  sample variance
    sigma_train = sqrt(var_train);                     % [1 x 14]
    sigma_train(sigma_train < 1e-8) = 1;               % guard zero-std

    % Apply to both train and test
    X_train_z = (X_train - mu_train) ./ sigma_train;  % [N_train x 14]
    X_test_z  = (X_test  - mu_train) ./ sigma_train;  % [N_test  x 14]
    fprintf('STEP 4: Z-score done. mu computed from %d training rows\n', n_train);

    %% ============================================================
    %% STEP 5: SLIDING WINDOW
    %% Each sample = flatten a [30 x 14] window into [1 x 420]
    %% RUL label   = RUL at the LAST row of the window
    %%
    %%  Window i covers rows  i : i+29
    %%  Label        = RUL_train(i+29)   <-- last row
    %% ============================================================
    n_sensors = length(kept_cols);
    feat_len  = WINDOW * n_sensors;   % 30 * 14 = 420

    % Pre-count total windows to pre-allocate (avoids slow []=[]concat)
    total_wins = 0;
    for id = units'
        nc         = sum(train(:,1) == id);
        total_wins = total_wins + max(0, nc - WINDOW + 1);
    end

    X_win = zeros(total_wins, feat_len);
    y_win = zeros(total_wins, 1);
    row   = 1;

    for id = units'
        idx  = train(:,1) == id;
        data = X_train_z(idx, :);
        rv   = RUL_train(idx);
        nc   = size(data, 1);

        for i = 1 : (nc - WINDOW + 1)
            % Flatten window: stack rows left-to-right into one long vector
            win              = data(i : i+WINDOW-1, :);   % [30 x 14]
            X_win(row, :)    = win(:)';                    % [1 x 420]
            y_win(row)       = rv(i + WINDOW - 1);         % label = last row RUL
            row = row + 1;
        end
    end

    % Test: take the LAST window of each test engine
    test_units = unique(test(:,1));
    n_te       = length(test_units);
    X_test_win = zeros(n_te, feat_len);

    for k = 1:n_te
        idx  = test(:,1) == test_units(k);
        data = X_test_z(idx, :);
        nc   = size(data, 1);
        if nc >= WINDOW
            win = data(end-WINDOW+1 : end, :);
        else
            % Pad with first row if engine has < 30 cycles
            pad = repmat(data(1,:), WINDOW-nc, 1);
            win = [pad; data];
        end
        X_test_win(k, :) = win(:)';
    end
    fprintf('STEP 5: Windows built. Train=%d  Test=%d  FeatLen=%d\n', ...
        total_wins, n_te, feat_len);

    %% ============================================================
    %% STEP 6: PCA - PURE MATHEMATICS, STEP BY STEP
    %%
    %%  PCA finds directions (principal components) in which
    %%  the data varies the most.
    %%
    %%  --- THE MATH ---
    %%
    %%  (a) CENTRE the data:
    %%        X_c = X - mean(X)         [N x p]
    %%
    %%  (b) COVARIANCE MATRIX:
    %%        C = (1/(N-1)) * X_c' * X_c    [p x p]
    %%
    %%      C_ij = covariance between feature i and feature j
    %%      Diagonal = variances of each feature
    %%
    %%  (c) EIGENDECOMPOSITION of C:
    %%        C * v = lambda * v
    %%
    %%      lambda = eigenvalue  (= variance explained by that PC)
    %%      v      = eigenvector (= direction of that PC)
    %%
    %%  (d) SORT by eigenvalue DESCENDING
    %%        (largest eigenvalue = most variance = PC1)
    %%
    %%  (e) SELECT top-k eigenvectors to retain >= 95% variance:
    %%        cumulative variance = sum(lambda_1..k) / sum(all lambda)
    %%
    %%  (f) PROJECT data onto selected eigenvectors:
    %%        Z = X_c * W        W = [p x k] matrix of eigenvectors
    %%        Z is the low-dimensional representation
    %%
    %%  WHY SVD instead of eig(cov)?
    %%    For large p (420), computing [p x p] covariance = 420x420
    %%    matrix is feasible but SVD on the data matrix directly
    %%    is numerically more stable and faster.
    %%    Relation: if X_c = U*S*V', then
    %%      eigenvalues of C = diag(S)^2 / (N-1)
    %%      eigenvectors of C = columns of V
    %% ============================================================

    fprintf('STEP 6: Computing PCA...\n');

    %% (a) Centre the windowed training data
    N_win  = size(X_win, 1);   % number of training windows
    p      = size(X_win, 2);   % number of features (420)

    mu_pca  = sum(X_win, 1) / N_win;    % [1 x p]  row mean
    X_c     = X_win - mu_pca;           % [N x p]  centred

    fprintf('      Data matrix size: %d x %d\n', N_win, p);

    %% (b) Covariance matrix  C = X_c' * X_c / (N-1)
    %%     [p x p] = [p x N] * [N x p] / scalar
    C = (X_c' * X_c) / (N_win - 1);    % [p x p]
    fprintf('      Covariance matrix: %d x %d\n', p, p);

    %% (c) Eigendecomposition  C * V = V * D
    %%     V(:,i) = i-th eigenvector
    %%     D(i,i) = i-th eigenvalue = variance along that direction
    [V, D] = eig(C);                    % V=[p x p], D=[p x p] diagonal

    %% (d) Sort eigenvalues descending (eig() returns ascending)
    eigvals              = diag(D);                      % [p x 1]
    [eigvals, sort_idx]  = sort(eigvals, 'descend');     % sort descending
    V                    = V(:, sort_idx);               % reorder eigenvectors

    %% Remove tiny/negative eigenvalues (numerical noise)
    eigvals(eigvals < 0) = 0;

    %% (e) Compute explained variance ratio and find cutoff
    total_var   = sum(eigvals);
    explained   = (eigvals / total_var) * 100;      % % variance per PC
    cum_exp     = cumsum(explained);                 % cumulative %

    numPC = find(cum_exp >= 95, 1);                  % first PC reaching 95%
    if isempty(numPC), numPC = p; end

    fprintf('      Total variance: %.4f\n', total_var);
    fprintf('      PC1 explains: %.2f%%\n', explained(1));
    fprintf('      PC2 explains: %.2f%%\n', explained(2));
    fprintf('      PCs needed for 95%% variance: %d\n', numPC);

    %% (f) Select top-k eigenvectors  W = V(:, 1:numPC)
    W = V(:, 1:numPC);   % [p x numPC]  projection matrix

    %% Project training windows: Z_train = X_c * W
    Z_train = X_c * W;                        % [N_win x numPC]

    %% Project test windows:  Z_test = (X_test - mu_pca) * W
    X_test_c = X_test_win - mu_pca;           % centre with TRAINING mean
    Z_test   = X_test_c * W;                  % [n_te x numPC]

    fprintf('STEP 6: PCA done. %d -> %d dimensions (%.2f%% variance kept)\n', ...
        p, numPC, cum_exp(numPC));

    %% ============================================================
    %% STEP 7: SVR (Support Vector Regression, RBF kernel)
    %% ============================================================
    fprintf('STEP 7: Training SVR on %d samples x %d PCs...\n', ...
        size(Z_train,1), numPC);

    mdl = fitrsvm(Z_train, y_win, ...
        'KernelFunction', 'rbf',   ...
        'KernelScale',    'auto',  ...
        'Standardize',    true,    ...
        'CacheSize',      4000,    ...
        'IterationLimit', 1e5);

    fprintf('STEP 7: SVR training complete\n');

    %% ============================================================
    %% STEP 8: PREDICT AND CLAMP TO [0, RUL_CAP]
    %% ============================================================
    y_pred = predict(mdl, Z_test);
    y_pred = max(0, min(y_pred, RUL_CAP));

    %% ============================================================
    %% STEP 9: METRICS
    %% ============================================================

    % RMSE
    rmse = sqrt( mean( (rul_test - y_pred).^2 ) );

    % MAE
    mae  = mean( abs(rul_test - y_pred) );

    % R-squared: 1 - SS_residual / SS_total
    SS_res = sum( (rul_test - y_pred).^2 );
    SS_tot = sum( (rul_test - mean(rul_test)).^2 );
    r2     = 1 - SS_res / SS_tot;

    % NASA asymmetric score (Saxena 2008)
    % Penalises late prediction (over-estimate) more heavily
    d = y_pred - rul_test;
    s = zeros(size(d));
    s(d <  0) = exp(-d(d <  0) / 13) - 1;   % early: exp(d/13)-1
    s(d >= 0) = exp( d(d >= 0) / 10) - 1;   % late:  exp(d/10)-1  (harsher)
    nasa_score = sum(s);

    % Accuracy within ±30 cycles
    acc30 = mean(abs(rul_test - y_pred) <= 30) * 100;

    fprintf('\n----- %s Results -----\n', dsname);
    fprintf('  RMSE              : %.4f\n', rmse);
    fprintf('  MAE               : %.4f\n', mae);
    fprintf('  R2                : %.4f\n', r2);
    fprintf('  NASA Score        : %.4f\n', nasa_score);
    fprintf('  Accuracy (+-30)   : %.2f%%\n', acc30);
    fprintf('  PCA dims: %d -> %d (%.2f%% var)\n', p, numPC, cum_exp(numPC));
    fprintf('------------------------\n');

    %% ============================================================
    %% STEP 10: PLOTS
    %% ============================================================

    %% -- Plot A: Scree Plot --
    figure('Name', sprintf('%s | Scree Plot', dsname), 'NumberTitle','off');
    mshow = min(30, length(explained));
    yyaxis left
    bar(1:mshow, explained(1:mshow), 'FaceColor',[0.2 0.5 0.8],'EdgeColor','none');
    ylabel('Individual Variance Explained (%)');
    yyaxis right
    plot(1:mshow, cum_exp(1:mshow), '-o', 'Color',[0.9 0.3 0.1], ...
        'LineWidth',2,'MarkerFaceColor',[0.9 0.3 0.1],'MarkerSize',5);
    yline(95,'--k','95% threshold','LineWidth',1.2);
    ylabel('Cumulative Variance (%)'); ylim([0 110]);
    xlabel('Principal Component Number');
    title(sprintf('%s | Scree Plot: %d PCs retain %.1f%% variance', ...
        dsname, numPC, cum_exp(numPC)));
    legend({'Individual %','Cumulative %','95% line'},'Location','east');
    grid on;

    %% -- Plot B: Eigenvalue Spectrum (log scale) --
    figure('Name', sprintf('%s | Eigenvalue Spectrum', dsname), 'NumberTitle','off');
    semilogy(eigvals(1:min(50,length(eigvals))), '-o', ...
        'Color',[0.2 0.5 0.8],'MarkerFaceColor',[0.2 0.5 0.8],'MarkerSize',5);
    xline(numPC,'--r',sprintf('PC=%d',numPC),'LineWidth',1.5);
    xlabel('Component Index');
    ylabel('Eigenvalue (log scale)');
    title(sprintf('%s | Eigenvalue Spectrum (covariance matrix)', dsname));
    grid on;

    %% -- Plot C: True vs Predicted RUL (line) --
    figure('Name', sprintf('%s | True vs Predicted RUL', dsname), 'NumberTitle','off');
    plot(1:n_te, rul_test, '-o', 'Color',[0.1 0.5 0.8], ...
        'MarkerFaceColor',[0.1 0.5 0.8],'MarkerSize',4,'LineWidth',1.5);
    hold on;
    plot(1:n_te, y_pred,  '-x', 'Color',[0.85 0.2 0.1], ...
        'MarkerSize',6,'LineWidth',1.5);
    hold off;
    xlabel('Engine Number');
    ylabel('RUL (cycles, capped at 125)');
    title(sprintf('%s | RMSE=%.2f  MAE=%.2f  R2=%.3f  Score=%.0f', ...
        dsname, rmse, mae, r2, nasa_score));
    legend('True RUL','Predicted RUL','Location','best');
    grid on;

    %% -- Plot D: Scatter True vs Predicted --
    figure('Name', sprintf('%s | Scatter', dsname), 'NumberTitle','off');
    scatter(rul_test, y_pred, 40, [0.2 0.5 0.8],'filled','MarkerFaceAlpha',0.6);
    hold on;
    plot([0 RUL_CAP],[0 RUL_CAP],'--k','LineWidth',1.5);
    plot([0 RUL_CAP],[30 RUL_CAP+30],'--','Color',[0.6 0.6 0.6],'LineWidth',1);
    plot([0 RUL_CAP],[-30 RUL_CAP-30],'--','Color',[0.6 0.6 0.6],'LineWidth',1);
    hold off;
    xlabel('True RUL'); ylabel('Predicted RUL');
    title(sprintf('%s | Scatter  R2=%.3f', dsname, r2));
    legend('Predictions','Perfect fit','±30 band','Location','northwest');
    axis equal; axis([0 RUL_CAP 0 RUL_CAP]); grid on;

    %% -- Plot E: Error Distribution --
    figure('Name', sprintf('%s | Error Distribution', dsname), 'NumberTitle','off');
    errors = y_pred - rul_test;
    histogram(errors, 30,'FaceColor',[0.2 0.5 0.8],'EdgeColor','none');
    xline(0,  '-k','Zero','LineWidth',2);
    xline( 30,'--r','+30','LineWidth',1.3);
    xline(-30,'--r','-30','LineWidth',1.3);
    xlabel('Prediction Error (cycles)  [positive=late, negative=early]');
    ylabel('Count');
    title(sprintf('%s | Error  mean=%.1f  std=%.1f  Acc±30=%.1f%%', ...
        dsname, mean(errors), std(errors), acc30));
    grid on;

end   % END dataset loop
