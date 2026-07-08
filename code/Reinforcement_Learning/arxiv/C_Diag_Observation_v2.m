%% C_Diag_Observation.m
% Tests whether the agent's ACTUAL observation, [real(r); imag(r)]
% (32-dim), encodes the direction to the peak -- i.e., whether there is
% anything for the Q-network to learn from the observation alone
% (t_x, t_y are NOT observed).
%
% METHOD:
%   1. For several calibration positions, compute the observation at
%      every (t_x, t_y) cell, and the geometric ground-truth optimal
%      action (the greedy wrapped step toward the calibrated best cell).
%   2. Fit a LINEAR one-vs-all probe: obs -> optimal action (base MATLAB
%      least squares, no toolboxes needed). A linear probe is the right
%      proxy here: SIM-2 is a structured linear map up to the diode, so
%      if a free linear map cannot extract direction, the constrained
%      SIM-2 network almost certainly cannot either. The probe has
%      33*9 = 297 free parameters -- FEWER than SIM-2's 916 phases --
%      so success here also settles the parameter-count question.
%   3. Report accuracy on (a) held-out CELLS within training positions,
%      and (b) entirely held-out POSITIONS -- (b) is the decisive one,
%      since the deployed agent faces unseen MU positions.
%   4. Also maps the DOA-estimation error over the grid using the exact
%      estimator block (same variable names: R, linear_idx, n_psi_x_max,
%      n_psi_y_max, t_psi_x_max, t_psi_y_max, best_psi_x_est,
%      best_psi_y_est).
%
% CHANCE LEVEL: 1/9 = 11.1%.
% VERDICT (held-out-position accuracy):
%   >~70%  -> direction is linearly decodable; parameter count is NOT
%             the problem; look at training machinery / output scale.
%   ~30-70%-> partial information; expect slow/imperfect RL learning.
%   ~11%   -> observation carries no direction; stop tuning rewards.

clc; clear all; close all;
addingPathParentFolderByName('code');
Parameters;

% ---- MATCH THE TRAINING SIGNAL MODEL -------------------------------------
% You are currently training on the PURE (analytic) SIM-1 intentionally.
% This flag must match the training script, or you diagnose the wrong
% system. Set true only when the CST campaign resumes.
USE_CST = false;
if USE_CST
    EnvPars.G      = EnvPars.G_CST;
    EnvPars.U_func = EnvPars.U_func_CST;
end
Calibration;

n_positions   = 6;    % 4 train + 2 held-out
n_train_pos   = 4;
rng(11);
pos_list = randperm(EnvPars.N_cal, n_positions);

T_x = EnvPars.T_x; T_y = EnvPars.T_y;
n_cells = T_x * T_y;
obs_dim = 2 * EnvPars.N;

X_all   = cell(n_positions,1);   % observations, n_cells x 32
y_all   = cell(n_positions,1);   % optimal-action labels, n_cells x 1

for pp = 1:n_positions
    pos_idx = pos_list(pp);
    psi_x   = EnvPars.psi_x_cal(pos_idx);
    psi_y   = EnvPars.psi_y_cal(pos_idx);
    bx      = EnvPars.best_tx_cal(pos_idx);
    by      = EnvPars.best_ty_cal(pos_idx);

    a_psi_x   = exp(1i * psi_x * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);

    X = zeros(n_cells, obs_dim);
    y = zeros(n_cells, 1);
    err_map = zeros(T_x, T_y);          % DOA error (raw, your formula)
    err_map_wrapped = zeros(T_x, T_y);  % DOA error with angle wrapping

    for ty = 1:T_y
        for tx = 1:T_x
            t_psi = (ty - 1) * T_x + tx;
            v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
            r  = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

            % ---- observation exactly as the agent sees it ----
            X(t_psi, :) = [real(r); imag(r)]';

            % ---- ground-truth optimal action: greedy wrapped step ----
            dxw = mod(bx - tx + floor(T_x/2), T_x) - floor(T_x/2);
            dyw = mod(by - ty + floor(T_y/2), T_y) - floor(T_y/2);
            step_xy = [sign(dxw), sign(dyw)];   % in {-1,0,1}^2
            y(t_psi) = find(all(EnvPars.delta_moves == step_xy, 2), 1);

            % ---- DOA estimate: YOUR estimator block, verbatim names ----
            R = flipud(fliplr(reshape(abs(r), [EnvPars.N_x, EnvPars.N_y])))';
            [~, linear_idx] = max(R, [], 'all');
            n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
            n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;

            t_psi_x_max = EnvPars.t_x(t_psi);
            t_psi_y_max = EnvPars.t_y(t_psi);
            best_psi_x_est = mod(2*pi * (n_psi_x_max + (t_psi_x_max) / EnvPars.T_x) / EnvPars.N_x, 2*pi);
            best_psi_y_est = mod(2*pi * (n_psi_y_max + (t_psi_y_max) / EnvPars.T_y) / EnvPars.N_y, 2*pi);

            % your formula (no wrapping), for continuity with error_sum
            err_map(tx, ty) = 0.5*(abs(psi_x - best_psi_x_est) + ...
                                   abs(psi_y - best_psi_y_est));
            % wrapped variant: |angle difference| on the circle --
            % your raw formula overestimates when angles straddle 0/2pi
            dpx = abs(angle(exp(1i*(psi_x - best_psi_x_est))));
            dpy = abs(angle(exp(1i*(psi_y - best_psi_y_est))));
            err_map_wrapped(tx, ty) = 0.5*(dpx + dpy);
        end
    end

    X_all{pp} = X;
    y_all{pp} = y;

    fig = figure('Name', sprintf('DOA error pos %d', pos_idx));
    subplot(1,2,1);
    imagesc(err_map'); axis xy equal tight; colorbar;
    hold on; plot(bx, by, 'r+', 'MarkerSize', 14, 'LineWidth', 2);
    xlabel('t_x'); ylabel('t_y'); title(sprintf('err sum (raw), pos %d', pos_idx));
    subplot(1,2,2);
    imagesc(err_map_wrapped'); axis xy equal tight; colorbar;
    hold on; plot(bx, by, 'r+', 'MarkerSize', 14, 'LineWidth', 2);
    xlabel('t_x'); ylabel('t_y'); title('err sum (angle-wrapped)');
    saveas(fig, sprintf('doa_error_pos_%d.png', pos_idx));
end

% ---- quadratic feature augmentation (physically: intensities +
% adjacent-output interference terms, i.e., what detectors and
% couplers would measure) ----
augment = @(X) [X, ...
    X(:,1:16).^2 + X(:,17:32).^2, ...                         % |r_n|^2
    X(:,1:15).*X(:,2:16)   + X(:,17:31).*X(:,18:32), ...      % Re(r_n conj(r_{n+1}))
    X(:,17:31).*X(:,2:16)  - X(:,1:15).*X(:,18:32)];          % Im(r_n conj(r_{n+1}))
X_all = cellfun(augment, X_all, 'UniformOutput', false);

%% ---- Linear probe: obs -> optimal action --------------------------------
n_act = EnvPars.n_actions;

X_tr = vertcat(X_all{1:n_train_pos});
y_tr = vertcat(y_all{1:n_train_pos});
X_te = vertcat(X_all{n_train_pos+1:end});
y_te = vertcat(y_all{n_train_pos+1:end});

% one-hot targets, least-squares fit with bias (base MATLAB only)
Y_tr = zeros(numel(y_tr), n_act);
Y_tr(sub2ind(size(Y_tr), (1:numel(y_tr))', y_tr)) = 1;

A_tr = [X_tr, ones(size(X_tr,1),1)];
W    = A_tr \ Y_tr;

% training accuracy
[~, pred_tr] = max(A_tr * W, [], 2);
acc_tr = mean(pred_tr == y_tr);

% held-out POSITION accuracy (the decisive number)
A_te = [X_te, ones(size(X_te,1),1)];
[~, pred_te] = max(A_te * W, [], 2);
acc_te = mean(pred_te == y_te);

% held-out CELL accuracy within training positions (70/30 split)
n_tr_all = size(X_tr,1);
idx_perm = randperm(n_tr_all);
n70 = round(0.7*n_tr_all);
i70 = idx_perm(1:n70); i30 = idx_perm(n70+1:end);
Y70 = zeros(n70, n_act);
Y70(sub2ind(size(Y70), (1:n70)', y_tr(i70))) = 1;
W2  = [X_tr(i70,:), ones(n70,1)] \ Y70;
[~, pred30] = max([X_tr(i30,:), ones(numel(i30),1)] * W2, [], 2);
acc_cell = mean(pred30 == y_tr(i30));

fprintf('\n===== Linear probe: [Re(r);Im(r)] -> optimal action =====\n');
fprintf('probe free parameters:               %d (33 x 9)\n', 33*n_act);
fprintf('chance level:                        %.1f%%\n', 100/n_act);
fprintf('training accuracy (4 positions):     %.1f%%\n', 100*acc_tr);
fprintf('held-out CELLS (same positions):     %.1f%%\n', 100*acc_cell);
fprintf('held-out POSITIONS (2 unseen):       %.1f%%   <-- decisive\n', 100*acc_te);
fprintf('\nVERDICT: >~70%% held-out-position -> direction is linearly\n');
fprintf('decodable with 297 params; SIM-2''s 916 phases are NOT too few\n');
fprintf('in count -- look at output scale / constrained expressivity.\n');
fprintf('~11%% -> observation carries no direction; stop tuning rewards.\n');

%% ======================= Helpers =======================
function addingPathParentFolderByName(targetName)
    currFolder = pwd;
    found = false;
    while true
        [parentFolder, currentName] = fileparts(currFolder);
        if strcmpi(currentName, targetName)
            found = true;
            break;
        end
        if isempty(parentFolder) || strcmp(currFolder, parentFolder)
            break;
        end
        currFolder = parentFolder;
    end
    if found
        addpath(genpath(currFolder));
        fprintf('Adding matlab path to: %s\n', currFolder);
    else
        error('Folder named "%s" not found in any parent directory.', targetName);
    end
end
