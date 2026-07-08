%% C_Diag_Landscape.m
% Diagnostic for observation/shaping informativeness on the T_x x T_y
% snapshot grid. For a few random calibration positions it:
%   1. Computes p(t_x,t_y) = max power / global_max over ALL grid cells
%   2. Plots the landscape heatmap with the calibrated best cell marked
%   3. Runs a GREEDY HILL-CLIMB on p from every one of the T_x*T_y starts,
%      using the exact same 9 wraparound moves as the RL agent
%   4. Reports: basin fraction (p>threshold), greedy success rate, mean
%      path length, fraction stuck at local maxima / plateaus, and a
%      consistency check between argmax(p) and best_tx/ty_cal
%
% INTERPRETATION:
%   - Greedy success >~90%  -> landscape supports navigation; any RL
%     failure is training machinery (e.g. the diode/negative-return
%     collision), NOT the observation.
%   - Greedy success low    -> landscape fragmented (local maxima /
%     plateaus); no reward tweak fixes that. Architecture-level rethink.
%
% Deterministic environment (no additive noise in r), so single pass
% per cell suffices.

clc; clear all; close all;
addingPathParentFolderByName('code');
Parameters;

% % Must MATCH training exactly -- CST front-end overrides
% EnvPars.G      = EnvPars.G_CST;
% EnvPars.U_func = EnvPars.U_func_CST;
Calibration;

n_positions = 4;            % how many random calibration positions to test
climb_cap   = 400;          % greedy step cap (way above worst-case ~20)
rng(7);                     % reproducible position picks

pos_list = randi(EnvPars.N_cal, n_positions, 1);

for pp = 1:n_positions
    pos_idx = pos_list(pp);
    psi_x   = EnvPars.psi_x_cal(pos_idx);
    psi_y   = EnvPars.psi_y_cal(pos_idx);
    gmax    = EnvPars.global_max_cal(pos_idx);
    bx      = EnvPars.best_tx_cal(pos_idx);
    by      = EnvPars.best_ty_cal(pos_idx);

    a_psi_x   = exp(1i * psi_x * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);

    %% 1. Full landscape p(t_x, t_y)
    p_map = zeros(EnvPars.T_x, EnvPars.T_y);   % p_map(tx, ty)
    for ty = 1:EnvPars.T_y
        for tx = 1:EnvPars.T_x
            t_psi = (ty - 1) * EnvPars.T_x + tx;
            v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
            r  = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;
            p_map(tx, ty) = max(abs(r).^2) / gmax;
        end
    end


    % Consistency: does argmax(p_map) match the calibrated best cell?
    [~, li] = max(p_map, [], 'all');
    [ax, ay] = ind2sub(size(p_map), li);
    consistent = (ax == bx) && (ay == by);

    %% 2. Greedy hill-climb from EVERY start, same moves as the agent
    n_start   = EnvPars.T_x * EnvPars.T_y;
    success   = false(n_start,1);
    steps_tk  = nan(n_start,1);
    stuck     = false(n_start,1);
    tie_eps   = 1e-12;

    for s = 1:n_start
        ty0 = ceil(s / EnvPars.T_x);
        tx0 = s - (ty0-1)*EnvPars.T_x;
        tx = tx0; ty = ty0;
        for k = 1:climb_cap
            if tx == bx && ty == by
                success(s) = true; steps_tk(s) = k-1; break;
            end
            % evaluate all 9 moves (same wraparound as stepFunction)
            best_v = p_map(tx, ty); best_m = 0;
            for m = 1:size(EnvPars.delta_moves,1)
                d  = EnvPars.delta_moves(m,:);
                nx = mod(tx - 1 + d(1), EnvPars.T_x) + 1;
                ny = mod(ty - 1 + d(2), EnvPars.T_y) + 1;
                if p_map(nx, ny) > best_v + tie_eps
                    best_v = p_map(nx, ny); best_m = m;
                end
            end
            if best_m == 0
                stuck(s) = true; break;   % no strictly better neighbor
            end
            d  = EnvPars.delta_moves(best_m,:);
            tx = mod(tx - 1 + d(1), EnvPars.T_x) + 1;
            ty = mod(ty - 1 + d(2), EnvPars.T_y) + 1;
        end
    end

    basin_frac  = mean(p_map(:) > EnvPars.threshold);
    succ_rate   = mean(success);
    mean_steps  = mean(steps_tk(success), 'omitnan');
    stuck_rate  = mean(stuck);

    fprintf('\n=== Calibration position %d (idx %d) ===\n', pp, pos_idx);
    fprintf('argmax(p) at (%d,%d) vs calibrated best (%d,%d): %s\n', ...
        ax, ay, bx, by, string(consistent));
    if ~consistent
        fprintf('  ** WARNING: calibration best cell is NOT the landscape max --\n');
        fprintf('  ** at_peak termination and greedy target disagree. Investigate Calibration.\n');
    end
    fprintf('basin fraction (p > %.2f):        %.1f%% of grid\n', EnvPars.threshold, 100*basin_frac);
    fprintf('greedy hill-climb success rate:   %.1f%% of %d starts\n', 100*succ_rate, n_start);
    fprintf('mean path length (successes):     %.1f steps\n', mean_steps);
    fprintf('stuck at local max/plateau:       %.1f%%\n', 100*stuck_rate);

    %% 3. Plots
    fig = figure('Name', sprintf('Landscape pos %d', pos_idx));
    imagesc(p_map'); axis xy equal tight; colorbar;
    hold on;
    plot(bx, by, 'r+', 'MarkerSize', 14, 'LineWidth', 2);
    plot(ax, ay, 'wo', 'MarkerSize', 10, 'LineWidth', 1.5);
    contour(p_map', [EnvPars.threshold EnvPars.threshold], 'w--', 'LineWidth', 1);
    xlabel('t_x'); ylabel('t_y');
    title(sprintf('p(t_x,t_y), pos %d -- greedy success %.0f%%', pos_idx, 100*succ_rate));
    saveas(fig, sprintf('landscape_pos_%d.png', pos_idx));
end

fprintf('\nVERDICT GUIDE:\n');
fprintf('  success >~90%% everywhere  -> landscape fine; failure was the diode /\n');
fprintf('                               negative-return collision (fix the reward).\n');
fprintf('  success low / stuck high   -> landscape fragmented by CST perturbations;\n');
fprintf('                               reward tweaks will NOT fix this.\n');

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
