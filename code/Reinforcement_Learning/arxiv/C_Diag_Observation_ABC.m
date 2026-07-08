%% C_Diag_Observation_ABC.m
% Three linear probes compared on identical data, to isolate WHAT
% operation is missing for cross-position generalization:
%
%   A. RAW linear:        [Re(r); Im(r)]                (32 feats)
%      -> baseline; known result ~39-48% held-out-position.
%   B. QUADRATIC, proper 2D pairs: raw + |r_n|^2 + x-adjacent AND
%      y-adjacent interference products                  (96 feats)
%      -> fixes the bug in the previous augmentation (which had no
%      y-pairs and 3 junk cross-row pairs).
%   C. ARGMAX-ALIGNED linear: circularly shift the 4x4 complex pattern
%      so the argmax bin is at (1,1), phase-reference to that port,
%      then raw [Re; Im]                                 (32 feats)
%      -> tests the gating hypothesis: the ONLY missing operation is
%      argmax-relative indexing, which the DOA estimator already
%      computes every step.
%
% PREDICTIONS (stated before running, so this is a real test):
%   A ~= 40-48%   (reproduces baseline)
%   B ~= 55-70%   (better features, still no gating)
%   C ~= 85%+     (gating supplied -> position-invariant)
% If C lands near the within-position ceiling (~87%), the training fix
% is: emit the ALIGNED observation from step/reset functions. No new
% hardware, no nonlinear layer.

clc; clear all; close all;
addingPathParentFolderByName('code');
Parameters;

USE_CST = false;   % must match the current (intentionally analytic) campaign
if USE_CST
    EnvPars.G      = EnvPars.G_CST;
    EnvPars.U_func = EnvPars.U_func_CST;
end
Calibration;

n_positions = 6; n_train_pos = 4;
rng(11);
pos_list = randperm(EnvPars.N_cal, n_positions);

T_x = EnvPars.T_x; T_y = EnvPars.T_y;
N_x = EnvPars.N_x; N_y = EnvPars.N_y;
n_cells = T_x * T_y;

XA = cell(n_positions,1); XB = cell(n_positions,1);
XC = cell(n_positions,1); yy = cell(n_positions,1);

for pp = 1:n_positions
    pos_idx = pos_list(pp);
    psi_x = EnvPars.psi_x_cal(pos_idx);
    psi_y = EnvPars.psi_y_cal(pos_idx);
    bx = EnvPars.best_tx_cal(pos_idx);
    by = EnvPars.best_ty_cal(pos_idx);

    a_psi_x   = exp(1i * psi_x * ((1:N_x)-1))';
    a_psi_y   = exp(1i * psi_y * ((1:N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);

    A = zeros(n_cells, 2*EnvPars.N);
    B = zeros(n_cells, 2*EnvPars.N + N_x*N_y + 2*((N_x-1)*N_y + N_x*(N_y-1)));
    C = zeros(n_cells, 2*EnvPars.N);
    y = zeros(n_cells, 1);

    for ty = 1:T_y
        for tx = 1:T_x
            t_psi = (ty - 1) * T_x + tx;
            v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
            r  = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

            % ---- label: greedy wrapped step toward calibrated best ----
            dxw = mod(bx - tx + floor(T_x/2), T_x) - floor(T_x/2);
            dyw = mod(by - ty + floor(T_y/2), T_y) - floor(T_y/2);
            y(t_psi) = find(all(EnvPars.delta_moves == [sign(dxw), sign(dyw)], 2), 1);

            % ---- A: raw ----
            A(t_psi,:) = [real(r); imag(r)]';

            % ---- B: raw + proper 2D quadratic features ----
            Rc = reshape(r, [N_x, N_y]);        % element (nx,ny), x-fastest
            mags = abs(Rc(:)).^2;               % 16
            xp = Rc(1:N_x-1, :) .* conj(Rc(2:N_x, :));   % x-adjacent, 3x4
            yp = Rc(:, 1:N_y-1) .* conj(Rc(:, 2:N_y));   % y-adjacent, 4x3
            B(t_psi,:) = [real(r); imag(r); mags; ...
                          real(xp(:)); imag(xp(:)); ...
                          real(yp(:)); imag(yp(:))]';

            % ---- C: argmax-aligned + phase-referenced raw ----
            [~, mi] = max(abs(Rc(:)));
            [mx, my] = ind2sub([N_x, N_y], mi);
            Ral = circshift(Rc, [1-mx, 1-my]);           % argmax -> (1,1)
            Ral = Ral * exp(-1i*angle(Ral(1,1)));        % phase-reference
            C(t_psi,:) = [real(Ral(:)); imag(Ral(:))]';
        end
    end
    XA{pp} = A; XB{pp} = B; XC{pp} = C; yy{pp} = y;
end

%% ---- run the three probes ------------------------------------------------
names = {'A raw linear', 'B 2D quadratic', 'C argmax-aligned'};
sets  = {XA, XB, XC};
n_act = EnvPars.n_actions;

fprintf('\n===== Probe comparison (chance = %.1f%%) =====\n', 100/n_act);
fprintf('%-20s %-12s %-18s %-22s\n', 'features', 'train acc', 'held-out cells', 'held-out POSITIONS');

for k = 1:3
    Xs = sets{k};
    X_tr = vertcat(Xs{1:n_train_pos});   y_tr = vertcat(yy{1:n_train_pos});
    X_te = vertcat(Xs{n_train_pos+1:end}); y_te = vertcat(yy{n_train_pos+1:end});

    Y_tr = zeros(numel(y_tr), n_act);
    Y_tr(sub2ind(size(Y_tr), (1:numel(y_tr))', y_tr)) = 1;
    A_tr = [X_tr, ones(size(X_tr,1),1)];
    W = A_tr \ Y_tr;

    [~, p_tr] = max(A_tr*W, [], 2);          acc_tr = mean(p_tr == y_tr);
    [~, p_te] = max([X_te, ones(size(X_te,1),1)]*W, [], 2);
    acc_te = mean(p_te == y_te);

    idx = randperm(size(X_tr,1)); n70 = round(0.7*numel(idx));
    i70 = idx(1:n70); i30 = idx(n70+1:end);
    Y70 = zeros(n70, n_act);
    Y70(sub2ind(size(Y70), (1:n70)', y_tr(i70))) = 1;
    W2 = [X_tr(i70,:), ones(n70,1)] \ Y70;
    [~, p30] = max([X_tr(i30,:), ones(numel(i30),1)]*W2, [], 2);
    acc_cell = mean(p30 == y_tr(i30));

    fprintf('%-20s %-12.1f %-18.1f %-22.1f\n', names{k}, 100*acc_tr, 100*acc_cell, 100*acc_te);
end

fprintf('\nIf C >> B > A on held-out POSITIONS, the missing operation is\n');
fprintf('argmax gating (already computed by the DOA estimator), and the\n');
fprintf('training fix is to emit the aligned observation -- no new\n');
fprintf('hardware, no nonlinear layer, no more parameters.\n');

%% ======================= Helpers =======================
function addingPathParentFolderByName(targetName)
    currFolder = pwd;
    found = false;
    while true
        [parentFolder, currentName] = fileparts(currFolder);
        if strcmpi(currentName, targetName)
            found = true; break;
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
