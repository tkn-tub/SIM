%[text] # SIM Ablation Study -- Full grid (T_SIM, L, M, s_x), WITH phase training
%[text] Follows the spirit of [1, Sec. VI.A]. Sweeps all four design variables in
%[text] nested loops; for EACH combination runs full gradient-descent phase
%[text] training (same algorithm as SIM_2DFT_Numeric_Gradient_Method_CST.m,
%[text] including CST-realistic amplitude coupling via F_amp, train-aware), with:
%[text]   - beta computed in closed form every iteration, exactly as in the main
%[text]     training script (NOT fixed -- reverted per discussion)
%[text]   - a single fixed zeta (learning-rate decay) set by hand below, not swept
%[text]   - a single random seed per combo (no Monte Carlo averaging -- GD itself
%[text]     partially compensates for initialization, unlike the no-training case)
%[text] Results land in one long-format table; each of the four "rounds" (vary one
%[text] parameter, hold the other three at baseline) is a filter on this table.
%[text]
%[text] CAUTION (carried over from the no-training version): s_layer = T_SIM/L
%[text] varies across the grid; large s_layer relative to lambda moves outside the
%[text] "thin metasurface stack" regime typical SIM designs assume.
%[text]
%[text] NOTE: the main training script's gradient is a FORWARD difference
%[text] (grad = (L_p - Lval)/h), not central -- this mirrors that exactly for
%[text] consistency. Flagging again in case you still want to switch to central
%[text] difference (the noise-robustness concern from the amplitude coupling
%[text] discussion still applies); easy to swap if so.

clc;
clear all;
close all;

addingPathParentFolderByName('code');

Parameters

%% ----------------- Load CST amplitude/phase data -----------------
load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

%% ----------------- Ablation grid -----------------
% PLACEHOLDER VALUES for T_SIM and s_x -- confirm/edit these.
T_SIM_sweep = [4 5 6 7] * lambda;          % thickness
L_sweep     = [7 9 11 13 15];                    % number of intermediate layers
Mx_sweep = [12 13 14 15 16 17]
M_sweep     = Mx_sweep.^2;                  % meta-atoms per intermediate layer (perfect squares)
s_x_sweep   = [0.3 0.4 0.5 0.6] * lambda;     % meta-atom spacing (s_y = s_x throughout)

% <<< FILL IN: single fixed learning-rate decay value used for every combo.
zeta_fixed = 0.980;%as observed from the file Result.mlx;
if isempty(zeta_fixed)
    error('Set zeta_fixed to a single scalar value (e.g. 0.99) before running.');
end

% maxIter, eta0, seed come from Parameters.m
h = 1e-5;   % finite-difference step (radians), matches main training script

% baseline index used when isolating one round from the full grid
i_T_base = 2; i_L_base = 2; i_M_base = 2; i_s_base = 2;

F = dft2_matrix(N_x, N_y);
N = N_x*N_y;

n_T = numel(T_SIM_sweep);
n_L = numel(L_sweep);
n_M = numel(M_sweep);
n_s = numel(s_x_sweep);
n_combos = n_T*n_L*n_M*n_s;

%% ----------------- Preallocate long-format results table -----------------
T_SIM_col   = zeros(n_combos,1);
L_col       = zeros(n_combos,1);
M_col       = zeros(n_combos,1);
s_x_col     = zeros(n_combos,1);
s_layer_col = zeros(n_combos,1);
loss_final_col = zeros(n_combos,1);
beta_final_col = zeros(n_combos,1);   % complex
gapfrac_final_col = zeros(n_combos,1);
loss_hist_all = cell(n_combos,1);     % full convergence curve per combo, for diagnostics

% ---- flatten the 4D grid into one combo list (parfor needs a single
%      sliced loop variable, not nested loops) ----
[Tg, Lg, Mg, Sg] = ndgrid(T_SIM_sweep, L_sweep, M_sweep, s_x_sweep);
T_SIM_list = Tg(:);
L_list     = Lg(:);
M_list     = Mg(:);
s_x_list   = Sg(:);
n_combos   = numel(T_SIM_list);   % overwrites the earlier n_T*n_L*n_M*n_s value (same number)

% ---- optional: size the pool explicitly on a shared HPC node ----
% If running under SLURM, the default parpool may try to grab more workers
% than this job actually owns. Uncomment and adjust if that's your setup:
% c = parcluster('local');
% c.NumWorkers = str2double(getenv('SLURM_CPUS_PER_TASK'));
% parpool(c);

t_start = tic;
parfor idx = 1:n_combos

    fprintf('T_SIM iteration %d out of %d: %.1f %%n',idx, n_combos ,100*idx/n_combos);

    T_SIM_cur = T_SIM_list(idx);
    L_cur     = L_list(idx);
    M_cur     = M_list(idx);
    s_x_cur   = s_x_list(idx);
    s_y_cur   = s_x_cur;
    s_layer_cur = T_SIM_cur / L_cur;

    M_x_cur = sqrt(M_cur);
    if mod(M_x_cur,1) ~= 0
        error('M = %d is not a perfect square.', M_cur);
    end
    M_y_cur = M_x_cur;

    % combo-specific reproducible stream -- workers don't share a single
    % sequential rng() the way the original nested `for` did
    rng(seed + idx);

    % ---- grid coordinates ----
    [xn, yn] = grid_coords_centered(N_x, N_y, d_x, d_y);
    [xm, ym] = grid_coords_centered(M_x_cur, M_y_cur, s_x_cur, s_y_cur);

    % ---- W0 (M x N): input layer -> layer 1 ----
    W0 = zeros(M_cur, N);
    for m = 1:M_cur
        for n = 1:N
            d  = sqrt((xm(m)-xn(n))^2 + (ym(m)-yn(n))^2 + s_layer_cur^2);
            cos_epsilon = s_layer_cur/d;
            W0(m,n) = (A_atom*cos_epsilon)/(2*pi*d^2) * (1-1j*kappa*d) * exp(1j*kappa*d);
        end
    end
    WL = W0.';

    % ---- W{l}, l=1..L-1: identical geometry every layer ----
    W_single = build_W_MM(M_x_cur, M_y_cur, s_x_cur, s_y_cur, s_layer_cur, kappa, A_atom);
    W = repmat({W_single}, L_cur-1, 1);

    % ---- init phases / amplitude-coupled Upsilon ----
    xi = cell(L_cur,1);
    Upsilon = cell(L_cur,1);
    for l = 1:L_cur
        xi{l} = 2*pi*rand(M_cur,1);
        amp_l = F_amp(mod(xi{l}, 2*pi));
        Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
    end

    % ---- gradient-descent training (single combo) ----
    eta = eta0;
    beta = 1+0j;
    loss_hist_cur = zeros(maxIter,1);

    for it = 1:maxIter
        % forward G = WL*Upsilon_L*W_{L-1}*...*Upsilon_1*W0
        G_intermediate = eye(M_cur, M_cur);
        for l = 2:L_cur
            G_intermediate = Upsilon{l} * W{l-1} * G_intermediate;
        end
        G = WL * G_intermediate * Upsilon{1} * W0;

        g = G(:); f = F(:);
        beta = (g'*g) \ (g'*f);   % closed-form LS, as in main script

        E = beta*G - F;
        Lval = norm(E,'fro')^2;
        loss_hist_cur(it) = Lval;

        % ---- numeric gradient (forward difference) ----
        grads = cell(L_cur,1);
        for l = 1:L_cur
            grads{l} = zeros(M_cur,1);
        end

        for l = 1:L_cur
            for m = 1:M_cur
                xi_p = xi{l}(m) + h;
                amp_p = F_amp(mod(xi_p, 2*pi));
                Upsilon_p = Upsilon;
                Upsilon_p{l}(m,m) = amp_p * exp(1i*xi_p);

                G_intermediate = eye(M_cur, M_cur);
                for l_int = 2:L_cur
                    G_intermediate = Upsilon_p{l_int} * W{l_int-1} * G_intermediate;
                end
                G_p = WL * G_intermediate * Upsilon_p{1} * W0;
                g_p = G_p(:);
                beta_p = (g_p'*g_p) \ (g_p'*f);

                E_p = beta_p*G_p - F;
                L_p = norm(E_p,'fro')^2;

                grads{l}(m) = (L_p - Lval) / h;
            end
        end

        % ---- update phases ----
        for l = 1:L_cur
            xi{l} = mod(xi{l} - eta*grads{l}, 2*pi);
        end
        for l = 1:L_cur
            amp_l = F_amp(mod(xi{l}, 2*pi));
            Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
        end

        eta = eta * zeta_fixed;
    end

    % ---- gap diagnostic on FINAL trained phases ----
    in_gap_count = 0;
    for l = 1:L_cur
        xi_mod = mod(xi{l}, 2*pi);
        if phase_min_meas <= phase_max_meas
            out_of_range = (xi_mod < phase_min_meas) | (xi_mod > phase_max_meas);
        else
            out_of_range = (xi_mod < phase_min_meas) & (xi_mod > phase_max_meas);
        end
        in_gap_count = in_gap_count + sum(out_of_range);
    end
    gapfrac_final = in_gap_count / (M_cur*L_cur);

    % ---- write directly to this iteration's slot (no shared counter) ----
    T_SIM_col(idx) = T_SIM_cur;
    L_col(idx)     = L_cur;
    M_col(idx)     = M_cur;
    s_x_col(idx)   = s_x_cur;
    s_layer_col(idx) = s_layer_cur;
    loss_final_col(idx) = loss_hist_cur(end);
    beta_final_col(idx) = beta;
    gapfrac_final_col(idx) = gapfrac_final;
    loss_hist_all{idx} = loss_hist_cur;

    fprintf('combo %d/%d done (T_SIM=%.2f*lambda, L=%d, M=%d, s_x=%.2f*lambda)\n', ...
        idx, n_combos, T_SIM_cur/lambda, L_cur, M_cur, s_x_cur/lambda);
end
fprintf('\nFull grid done: %d combinations, %.1f s total.\n', n_combos, toc(t_start));

results = table(T_SIM_col, L_col, M_col, s_x_col, s_layer_col, ...
    loss_final_col, beta_final_col, gapfrac_final_col, ...
    'VariableNames', {'T_SIM','L','M','s_x','s_layer','loss_final','beta_final','gapfrac_final'});

%% ----------------- Save full results table -----------------
path = fullfile('..', 'Dataset', 'Ablation_FullGrid_Training.mat');
save(path, 'results', 'loss_hist_all', ...
     'T_SIM_sweep', 'L_sweep', 'M_sweep', 's_x_sweep', 'zeta_fixed', 'maxIter', 'eta0', 'seed', ...
     'i_T_base', 'i_L_base', 'i_M_base', 'i_s_base');



%% ======================= Helpers =======================
% Copied verbatim from SIM_2DFT_Numeric_Gradient_Method_CST.m so this file
% is self-contained. Worth extracting into shared files once stable.

function [x, y] = grid_coords_centered(Nx, Ny, dx, dy)
    N = Nx*Ny;
    x = zeros(N,1);
    y = zeros(N,1);
    for n = 1:N
        iy = ceil(n/Nx);
        ix = n - (iy-1)*Nx;
        x(n) = (ix - 1 - (Nx-1)/2) * dx ;
        y(n) = ((Ny-1)/2 - (iy - 1)) * dy;
    end
end

function [F_amp, phase_min, phase_max] = build_amplitude_interpolant(mag_dB, phase_deg)
    phase_rad = deg2rad(phase_deg(:));
    phase_unwrapped = unwrap(phase_rad);
    mag_lin = 10.^(mag_dB(:)/20);

    [phase_sorted, idx] = sort(phase_unwrapped);
    mag_sorted = mag_lin(idx);

    phase_wrapped = mod(phase_sorted, 2*pi);
    [phase_wrapped, idx2] = sort(phase_wrapped);
    mag_sorted = mag_sorted(idx2);

    phase_min = phase_wrapped(1);
    phase_max = phase_wrapped(end);

    F_amp = griddedInterpolant(phase_wrapped, mag_sorted, 'pchip', 'nearest');
end

function F = dft2_matrix(Nx, Ny)
    N = Nx*Ny;
    F = zeros(N,N);
    for n = 1:N
        nx = mod(n-1, Nx) + 1;
        ny = floor((n-1)/Nx) + 1;
        for nh = 1:N
            nhx = mod(nh-1, Nx) + 1;
            nhy = floor((nh-1)/Nx) + 1;
            F(n,nh) = exp(-1j*2*pi*(nx-1)*(nhx-1)/Nx) * exp(-1j*2*pi*(ny-1)*(nhy-1)/Ny);
        end
    end
end

function Wmm = build_W_MM(Mx, My, sx, sy, slayer, kappa, Ameta)
    M = Mx*My;
    Wmm = zeros(M,M);
    for m = 1:M
        [mx, my] = idx_to_xy(m, Mx);
        for mh = 1:M
            [mhx, mhy] = idx_to_xy(mh, Mx);
            dxh = (mx - mhx)*sx;
            dyh = (my - mhy)*sy;
            d = sqrt(dxh^2 + dyh^2 + slayer^2);
            Wmm(m,mh) = (Ameta * slayer) / (2*pi*d^3) * (1 - 1j*kappa*d) * exp(1j*kappa*d);
        end
    end
end

function [x, y] = idx_to_xy(m, Mx)
    y = ceil(m/Mx);
    x = m - (y-1)*Mx;
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
        warning('Folder "%s" not found in path hierarchy.', targetName);
    end
end
