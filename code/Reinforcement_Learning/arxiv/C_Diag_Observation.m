%% C_Diag_Observation.m
% Tests whether the OBSERVATION [Re(r); Im(r)] carries one-shot direction
% + distance information toward the peak snapshot -- beyond the scalar
% p-landscape already verified by C_Diag_Landscape.m.
%
% Two estimators compared at EVERY grid cell:
%   (a) The argmax-bin estimator (your exact code block, same variable
%       names). Prediction: inverting best_psi_x_est always returns the
%       CURRENT t -- zero directional information. The script measures
%       this explicitly.
%   (b) A sub-bin (Jacobsen) estimator on the COMPLEX bins: the ratio of
%       the peak bin to its circular neighbors encodes the fractional
%       misalignment delta = (t* - t)/T, sign included -> one-shot
%       prediction of the peak cell from a single observation.
%
% NOTE: runs on the ANALYTIC G / idealized U_func -- deliberately NO
% CST overrides, to match the current intentional training configuration.
% (If you re-enable G_CST in training, re-enable it here too, or the
% diagnostic tests a different physics than the agent sees.)

clc; clear all; close all;
addingPathParentFolderByName('code');
Parameters;
% -- intentionally NOT overriding EnvPars.G / EnvPars.U_func (pure SIM-1) --
Calibration;

n_positions = 4;
rng(7);
pos_list = randi(EnvPars.N_cal, n_positions, 1);

for pp = 1:n_positions
    pos_idx = pos_list(pp);
    psi_x   = EnvPars.psi_x_cal(pos_idx);
    psi_y   = EnvPars.psi_y_cal(pos_idx);
    bx      = EnvPars.best_tx_cal(pos_idx);
    by      = EnvPars.best_ty_cal(pos_idx);

    a_psi_x   = exp(1i * psi_x * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);

    %% ---- Sign auto-calibration probe --------------------------------
    % The flipud/fliplr/transpose chain makes analytic sign-tracking of
    % delta error-prone (this chain has caused sign bugs before), so the
    % sign convention is calibrated empirically: at one cell to the
    % right of the best cell, the true delta_x is -1/T_x. Whichever sign
    % the raw estimator returns there fixes sgn_x. Same for y.
    tx_probe = mod(bx, EnvPars.T_x) + 1;   % bx + 1, wrapped
    [dxr, ~] = rawDeltas(tx_probe, by, a_psi_x_y, EnvPars);
    sgn_x = -sign(dxr); if sgn_x == 0, sgn_x = 1; end
    ty_probe = mod(by, EnvPars.T_y) + 1;   % by + 1, wrapped
    [~, dyr] = rawDeltas(bx, ty_probe, a_psi_x_y, EnvPars);
    sgn_y = -sign(dyr); if sgn_y == 0, sgn_y = 1; end

    %% ---- Sweep every grid cell --------------------------------------
    err_cells    = zeros(EnvPars.T_x, EnvPars.T_y);  % Chebyshev cell error of one-shot prediction
    argmax_self  = false(EnvPars.T_x, EnvPars.T_y);  % does inverted argmax point at current cell?

    for ty = 1:EnvPars.T_y
        for tx = 1:EnvPars.T_x
            t_psi = (ty - 1) * EnvPars.T_x + tx;
            v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
            r  = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

            % ================= (a) YOUR EXACT BLOCK =================
            R = flipud(fliplr(reshape(abs(r), [EnvPars.N_x, EnvPars.N_y])))';
            [~, linear_idx] = max(R, [], 'all');
            n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
            n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;
            t_psi_x_max = EnvPars.t_x(t_psi);
            t_psi_y_max = EnvPars.t_y(t_psi);
            best_psi_x_est = mod(2*pi * (n_psi_x_max + (t_psi_x_max) / EnvPars.T_x) / EnvPars.N_x, 2*pi);
            best_psi_y_est = mod(2*pi * (n_psi_y_max + (t_psi_y_max) / EnvPars.T_y) / EnvPars.N_y, 2*pi);
            % Invert the argmax estimate: which snapshot does it claim is best?
            f_x = mod(best_psi_x_est * EnvPars.N_x/(2*pi), 1);
            t_from_argmax_x = round(f_x * EnvPars.T_x);
            if t_from_argmax_x == 0, t_from_argmax_x = EnvPars.T_x; end
            argmax_self(tx, ty) = (t_from_argmax_x == tx);   % expected: TRUE everywhere

            % ============ (b) SUB-BIN (complex) ESTIMATOR ============
            % Same reordering as your block, applied to the COMPLEX r.
            % CRITICAL: .' (transpose), NOT ' (conjugate transpose) --
            % using ' here silently conjugates the field and flips signs.
            Rc = flipud(fliplr(reshape(r, [EnvPars.N_x, EnvPars.N_y])))';

            % Peak bin location in Rc: row = n_psi_y_max, col = n_psi_x_max
            % (their decomposition divides by N_x; valid because N_x == N_y --
            % if the array ever becomes non-square, this needs ind2sub.)
            cx = n_psi_x_max;  ry = n_psi_y_max;

            % Circular neighbors along x (columns) at the peak row
            cxp = mod(cx, EnvPars.N_x) + 1;  cxm = mod(cx-2, EnvPars.N_x) + 1;
            Xk  = Rc(ry, cx);  Xp = Rc(ry, cxp);  Xm = Rc(ry, cxm);
            delta_x = jacobsen(Xk, Xp, Xm);

            % Circular neighbors along y (rows) at the peak column
            ryp = mod(ry, EnvPars.N_y) + 1;  rym = mod(ry-2, EnvPars.N_y) + 1;
            Yk  = Rc(ry, cx);  Yp = Rc(ryp, cx);  Ym = Rc(rym, cx);
            delta_y = jacobsen(Yk, Yp, Ym);

            % One-shot prediction of the best snapshot cell
            t_star_x = mod(round(tx + sgn_x*delta_x*EnvPars.T_x) - 1, EnvPars.T_x) + 1;
            t_star_y = mod(round(ty + sgn_y*delta_y*EnvPars.T_y) - 1, EnvPars.T_y) + 1;

            % Torus Chebyshev error against the calibrated best cell
            ex = abs(t_star_x - bx); ex = min(ex, EnvPars.T_x - ex);
            ey = abs(t_star_y - by); ey = min(ey, EnvPars.T_y - ey);
            err_cells(tx, ty) = max(ex, ey);
        end
    end

    %% ---- Report ------------------------------------------------------
    e = err_cells(:);
    fprintf('\n=== Position %d (idx %d), best cell (%d,%d) ===\n', pp, pos_idx, bx, by);
    fprintf('(a) argmax estimator inverts to CURRENT cell: %.1f%% of cells (expect ~100%% -> directionless)\n', ...
        100*mean(argmax_self(:)));
    fprintf('(b) one-shot sub-bin prediction of the peak cell:\n');
    fprintf('    median error: %.1f cells | 95th pct: %.1f | within 1 cell: %.1f%% | within 2: %.1f%%\n', ...
        median(e), prctile(e,95), 100*mean(e<=1), 100*mean(e<=2));

    fig = figure('Name', sprintf('One-shot prediction error, pos %d', pos_idx));
    imagesc(err_cells'); axis xy equal tight; colorbar;
    hold on; plot(bx, by, 'r+', 'MarkerSize', 14, 'LineWidth', 2);
    xlabel('t_x'); ylabel('t_y');
    title(sprintf('Sub-bin one-shot error [cells], pos %d (median %.1f)', pos_idx, median(e)));
    saveas(fig, sprintf('obs_oneshot_error_pos_%d.png', pos_idx));
end

fprintf('\nINTERPRETATION:\n');
fprintf('  (a) ~100%% self-pointing confirms the argmax bin alone has no direction info.\n');
fprintf('  (b) small median error -> [Re(r);Im(r)] carries one-shot direction+distance;\n');
fprintf('      the network''s job is to approximate a wave-domain fractional-bin interpolator.\n');

%% ======================= Helpers =======================
function d = jacobsen(Xk, Xp, Xm)
    % Fractional bin offset from complex peak bin Xk and neighbors Xp/Xm.
    den = 2*Xk - Xp - Xm;
    if abs(den) < 1e-15
        d = 0;
    else
        d = -real((Xp - Xm) ./ den);
    end
    d = max(min(d, 0.75), -0.75);   % guard against blow-ups far off-peak
end

function [dx, dy] = rawDeltas(tx, ty, a_psi_x_y, EnvPars)
    % Raw (sign-uncalibrated) Jacobsen deltas at one cell -- used only by
    % the sign-calibration probe. Mirrors the main-loop computation.
    t_psi = (ty - 1) * EnvPars.T_x + tx;
    v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
    r  = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

    Rabs = flipud(fliplr(reshape(abs(r), [EnvPars.N_x, EnvPars.N_y])))';
    [~, linear_idx] = max(Rabs, [], 'all');
    n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
    n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;

    Rc = flipud(fliplr(reshape(r, [EnvPars.N_x, EnvPars.N_y]))).';
    cx = n_psi_x_max;  ry = n_psi_y_max;
    cxp = mod(cx, EnvPars.N_x) + 1;  cxm = mod(cx-2, EnvPars.N_x) + 1;
    dx = jacobsen(Rc(ry,cx), Rc(ry,cxp), Rc(ry,cxm));
    ryp = mod(ry, EnvPars.N_y) + 1;  rym = mod(ry-2, EnvPars.N_y) + 1;
    dy = jacobsen(Rc(ry,cx), Rc(ryp,cx), Rc(rym,cx));
end

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
