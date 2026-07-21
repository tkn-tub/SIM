%[text] # SIM Ablation Study -- Full grid WITH training + per-combo checkpointing
%[text]
%[text] Same training as SIM_Ablation_FullGrid_Training_parallel.m, with one
%[text] critical addition: every completed combo immediately saves its result
%[text] to its own checkpoint file from inside the parfor body. On restart,
%[text] already-finished combos are detected at the start of their parfor
%[text] iteration and skipped entirely -- so a crash only costs the combos
%[text] that were mid-training at the time, not the ones already done.
%[text]
%[text] Checkpoint strategy chosen for parfor compatibility:
%[text]   - ONE file per combo: ablation_ckpt/combo_%04d.mat
%[text]   - Written by the worker immediately after training finishes for that
%[text]     combo, before moving on to the next one that worker picks up.
%[text]   - Final aggregation reads ALL checkpoint files and assembles the
%[text]     results table, so the table is rebuilt correctly even after a
%[text]     partial run.
%[text]   - The parfor output arrays (T_SIM_col etc.) are still populated
%[text]     for combos that DO run in this session; the aggregation step
%[text]     fills in completed-from-previous-session slots from disk.
%[text]
%[text] NOTE: this is different from the zeta sweep, where checkpoints lived
%[text] at the outer (sequential) loop level. Here the parfor IS the outer
%[text] loop, so checkpointing must happen inside each worker.

clc;
clear all;
close all;

addingPathParentFolderByName('code');

Parameters

%% ----------------- Load CST amplitude/phase data -----------------
load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

%% ----------------- Ablation grid -----------------
T_SIM_sweep = [4 5 6 7] * lambda;
L_sweep     = [7 9 11 13 15];
Mx_sweep    = [12 13 14 15 16 17];
M_sweep     = Mx_sweep.^2;
s_x_sweep   = [0.3 0.4 0.5 0.6] * lambda;

zeta_fixed  = 0.987;

% maxIter, eta0, seed come from Parameters.m
h = 1e-5;

% baseline indices for the "vary one, fix others" filter later
i_T_base = 2; i_L_base = 2; i_M_base = 2; i_s_base = 2;

F = dft2_matrix(N_x, N_y);
N_atoms = N_x*N_y;   % renamed from N to avoid clash with Parameters.m

[Tg, Lg, Mg, Sg] = ndgrid(T_SIM_sweep, L_sweep, M_sweep, s_x_sweep);
T_SIM_list = Tg(:);
L_list     = Lg(:);
M_list     = Mg(:);
s_x_list   = Sg(:);
n_combos   = numel(T_SIM_list);

%% ----------------- Checkpoint directory -----------------
ckpt_dir = sprintf('ablation_ckpt_zeta_%05.3f', zeta_fixed);
if ~exist(ckpt_dir, 'dir')
    mkdir(ckpt_dir);
end

already_done = false(n_combos,1);
for idx = 1:n_combos
    already_done(idx) = isfile(fullfile(ckpt_dir, sprintf('combo_%04d.mat', idx)));
end
fprintf('Grid: %d combos total. %d already done (checkpoint files found), %d to run.\n', ...
    n_combos, sum(already_done), sum(~already_done));

%% ----------------- Preallocate output arrays (parfor needs them) -----------------
T_SIM_col         = zeros(n_combos,1);
L_col             = zeros(n_combos,1);
M_col             = zeros(n_combos,1);
s_x_col           = zeros(n_combos,1);
s_layer_col       = zeros(n_combos,1);
loss_final_col    = zeros(n_combos,1);
beta_final_col    = complex(zeros(n_combos,1));
gapfrac_final_col = zeros(n_combos,1);
loss_hist_all     = cell(n_combos,1);

%% ----------------- Start pool -----------------
delete(gcp('nocreate'));
parpool;

%% ----------------- parfor over combos -----------------
t_start = tic;

parfor idx = 1:n_combos

    ckpt_file = fullfile(ckpt_dir, sprintf('combo_%04d.mat', idx));

    %% ---- SKIP if this combo already has a checkpoint ----
    if isfile(ckpt_file)
        C = load(ckpt_file);
        T_SIM_col(idx)         = C.T_SIM_cur;
        L_col(idx)             = C.L_cur;
        M_col(idx)             = C.M_cur;
        s_x_col(idx)           = C.s_x_cur;
        s_layer_col(idx)       = C.s_layer_cur;
        loss_final_col(idx)    = C.loss_final;
        beta_final_col(idx)    = C.beta_final;
        gapfrac_final_col(idx) = C.gapfrac_final;
        loss_hist_all{idx}     = C.loss_hist;
        fprintf('combo %d/%d: loaded from checkpoint (T_SIM=%.2f*lambda, L=%d, M=%d, s_x=%.2f*lambda)\n', ...
            idx, n_combos, C.T_SIM_cur/lambda, C.L_cur, C.M_cur, C.s_x_cur/lambda);
        continue;
    end

    %% ---- Setup for this combo ----
    T_SIM_cur   = T_SIM_list(idx);
    L_cur       = L_list(idx);
    M_cur       = M_list(idx);
    s_x_cur     = s_x_list(idx);
    s_y_cur     = s_x_cur;
    s_layer_cur = T_SIM_cur / L_cur;

    M_x_cur = round(sqrt(M_cur));
    M_y_cur = M_x_cur;

    rng(seed + idx);   % reproducible per combo, independent across workers

    [xn, yn] = grid_coords_centered(N_x, N_y, d_x, d_y);
    [xm, ym] = grid_coords_centered(M_x_cur, M_y_cur, s_x_cur, s_y_cur);

    W0 = zeros(M_cur, N_atoms);
    for m = 1:M_cur
        for n = 1:N_atoms
            dist = sqrt((xm(m)-xn(n))^2 + (ym(m)-yn(n))^2 + s_layer_cur^2);
            cos_eps = s_layer_cur/dist;
            W0(m,n) = (A_atom*cos_eps)/(2*pi*dist^2) * (1-1j*kappa*dist) * exp(1j*kappa*dist);
        end
    end
    WL = W0.';

    W_single = build_W_MM(M_x_cur, M_y_cur, s_x_cur, s_y_cur, s_layer_cur, kappa, A_atom);
    W = repmat({W_single}, L_cur-1, 1);

    xi = cell(L_cur,1);
    Upsilon = cell(L_cur,1);
    for l = 1:L_cur
        xi{l} = 2*pi*rand(M_cur,1);
        amp_l = F_amp(mod(xi{l}, 2*pi));
        Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
    end

    %% ---- Training loop ----
    eta = eta0;
    beta = 1+0j;
    loss_hist_cur = zeros(maxIter,1);

    for it = 1:maxIter
        G_intermediate = eye(M_cur, M_cur);
        for l = 2:L_cur
            G_intermediate = Upsilon{l} * W{l-1} * G_intermediate;
        end
        G = WL * G_intermediate * Upsilon{1} * W0;

        g = G(:); f = F(:);
        beta = (g'*g) \ (g'*f);
        E = beta*G - F;
        Lval = norm(E,'fro')^2;
        loss_hist_cur(it) = Lval;

        grads = cell(L_cur,1);
        for l = 1:L_cur
            grads{l} = zeros(M_cur,1);
        end

        for l = 1:L_cur
            for m = 1:M_cur
                xi_p  = xi{l}(m) + h;
                amp_p = F_amp(mod(xi_p, 2*pi));
                Upsilon_p = Upsilon;
                Upsilon_p{l}(m,m) = amp_p * exp(1i*xi_p);

                G_int_p = eye(M_cur, M_cur);
                for l_int = 2:L_cur
                    G_int_p = Upsilon_p{l_int} * W{l_int-1} * G_int_p;
                end
                G_p  = WL * G_int_p * Upsilon_p{1} * W0;
                g_p  = G_p(:);
                beta_p = (g_p'*g_p) \ (g_p'*f);
                E_p  = beta_p*G_p - F;
                L_p  = norm(E_p,'fro')^2;
                grads{l}(m) = (L_p - Lval) / h;
            end
        end

        for l = 1:L_cur
            xi{l} = mod(xi{l} - eta*grads{l}, 2*pi);
            amp_l  = F_amp(mod(xi{l}, 2*pi));
            Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
        end

        eta = eta * zeta_fixed;
    end

    %% ---- Gap diagnostic on final trained phases ----
    in_gap_count = 0;
    for l = 1:L_cur
        xi_mod = mod(xi{l}, 2*pi);
        if phase_min_meas <= phase_max_meas
            oor = (xi_mod < phase_min_meas) | (xi_mod > phase_max_meas);
        else
            oor = (xi_mod < phase_min_meas) & (xi_mod > phase_max_meas);
        end
        in_gap_count = in_gap_count + sum(oor);
    end
    gapfrac_final = in_gap_count / (M_cur*L_cur);

    %% ---- Write per-combo checkpoint (before updating output arrays) ----
    loss_final  = loss_hist_cur(end);
    beta_final  = beta;
    parsave_combo(ckpt_file, T_SIM_cur, L_cur, M_cur, s_x_cur, s_layer_cur, ...
        loss_final, beta_final, gapfrac_final, loss_hist_cur, lambda);

    %% ---- Write into parfor output slices ----
    T_SIM_col(idx)         = T_SIM_cur;
    L_col(idx)             = L_cur;
    M_col(idx)             = M_cur;
    s_x_col(idx)           = s_x_cur;
    s_layer_col(idx)       = s_layer_cur;
    loss_final_col(idx)    = loss_final;
    beta_final_col(idx)    = beta_final;
    gapfrac_final_col(idx) = gapfrac_final;
    loss_hist_all{idx}     = loss_hist_cur;

    fprintf('combo %d/%d done (T_SIM=%.2f*lam, L=%d, M=%d, s_x=%.2f*lam) loss=%.4f gapfrac=%.1f%%\n', ...
        idx, n_combos, T_SIM_cur/lambda, L_cur, M_cur, s_x_cur/lambda, loss_final, 100*gapfrac_final);
end

fprintf('\nparfor done: %.1f s total.\n', toc(t_start));
delete(gcp('nocreate'));
fprintf('Parallel pool released.\n');

%% ----------------- Aggregation pass: fill any slots from checkpoints -----------------
% Needed if the parfor was killed before all combos finished: the surviving
% checkpoint files still hold those results and we rebuild from them here.
fprintf('Aggregating results from checkpoint files...\n');
n_recovered = 0;
for idx = 1:n_combos
    ckpt_file = fullfile(ckpt_dir, sprintf('combo_%04d.mat', idx));
    if isfile(ckpt_file) && loss_final_col(idx) == 0
        C = load(ckpt_file);
        T_SIM_col(idx)         = C.T_SIM_cur;
        L_col(idx)             = C.L_cur;
        M_col(idx)             = C.M_cur;
        s_x_col(idx)           = C.s_x_cur;
        s_layer_col(idx)       = C.s_layer_cur;
        loss_final_col(idx)    = C.loss_final;
        beta_final_col(idx)    = C.beta_final;
        gapfrac_final_col(idx) = C.gapfrac_final;
        loss_hist_all{idx}     = C.loss_hist;
        n_recovered = n_recovered + 1;
    end
end
fprintf('Recovered %d additional combos from checkpoint files.\n', n_recovered);

n_complete = sum(loss_final_col ~= 0);
fprintf('%d / %d combos have valid results.\n', n_complete, n_combos);

%% ----------------- Assemble results table -----------------
results = table(T_SIM_col, L_col, M_col, s_x_col, s_layer_col, ...
    loss_final_col, beta_final_col, gapfrac_final_col, ...
    'VariableNames', {'T_SIM','L','M','s_x','s_layer','loss_final','beta_final','gapfrac_final'});

%% ----------------- Save -----------------
out_path = fullfile('..', 'Dataset', sprintf('Ablation_FullGrid_Training_zeta_%05.3f.mat', zeta_fixed));
save(out_path, 'results', 'loss_hist_all', ...
     'T_SIM_sweep', 'L_sweep', 'M_sweep', 's_x_sweep', 'zeta_fixed', 'maxIter', 'eta0', 'seed', ...
     'i_T_base', 'i_L_base', 'i_M_base', 'i_s_base');
fprintf('Saved results to %s\n', out_path);

%% ======================= Helpers =======================
function parsave_combo(fpath, T_SIM_cur, L_cur, M_cur, s_x_cur, s_layer_cur, ...
        loss_final, beta_final, gapfrac_final, loss_hist, lambda) %#ok<INUSL>
    % Wrapper so save() can be called from inside a parfor body.
    save(fpath, 'T_SIM_cur', 'L_cur', 'M_cur', 's_x_cur', 's_layer_cur', ...
         'loss_final', 'beta_final', 'gapfrac_final', 'loss_hist', 'lambda', '-v7.3');
end

function [x, y] = grid_coords_centered(Nx, Ny, dx, dy)
    N = Nx*Ny;
    x = zeros(N,1); y = zeros(N,1);
    for n = 1:N
        iy = ceil(n/Nx);
        ix = n - (iy-1)*Nx;
        x(n) = (ix - 1 - (Nx-1)/2) * dx;
        y(n) = ((Ny-1)/2 - (iy-1)) * dy;
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
    N = Nx*Ny; F = zeros(N,N);
    for n = 1:N
        nx = mod(n-1,Nx)+1; ny = floor((n-1)/Nx)+1;
        for nh = 1:N
            nhx = mod(nh-1,Nx)+1; nhy = floor((nh-1)/Nx)+1;
            F(n,nh) = exp(-1j*2*pi*(nx-1)*(nhx-1)/Nx) * exp(-1j*2*pi*(ny-1)*(nhy-1)/Ny);
        end
    end
end

function Wmm = build_W_MM(Mx, My, sx, sy, slayer, kappa, Ameta)
    M = Mx*My; Wmm = zeros(M,M);
    for m = 1:M
        [mx, my] = idx_to_xy(m, Mx);
        for mh = 1:M
            [mhx, mhy] = idx_to_xy(mh, Mx);
            dxh = (mx-mhx)*sx; dyh = (my-mhy)*sy;
            d = sqrt(dxh^2 + dyh^2 + slayer^2);
            Wmm(m,mh) = (Ameta*slayer)/(2*pi*d^3) * (1-1j*kappa*d) * exp(1j*kappa*d);
        end
    end
end

function [x, y] = idx_to_xy(m, Mx)
    y = ceil(m/Mx); x = m - (y-1)*Mx;
end

function addingPathParentFolderByName(targetName)
    currFolder = pwd; found = false;
    while true
        [parentFolder, currentName] = fileparts(currFolder);
        if strcmpi(currentName, targetName), found = true; break; end
        if isempty(parentFolder) || strcmp(currFolder, parentFolder), break; end
        currFolder = parentFolder;
    end
    if found
        addpath(genpath(currFolder));
        fprintf('Adding matlab path to: %s\n', currFolder);
    else
        warning('Folder "%s" not found in path hierarchy.', targetName);
    end
end
