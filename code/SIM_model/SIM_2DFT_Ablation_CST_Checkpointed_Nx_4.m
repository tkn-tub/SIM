%% SIM-1 ablation study with per-combination checkpoints
% Sweeps T_SIM, s_x, L, M, and zeta while preserving the file layout
% expected by the existing parallel-coordinates plotting script:
%
%   ../Dataset/ablation_ckpt_zeta_0.988/combo_*.mat
%
% Each completed combo_*.mat file contains at least:
%   T_SIM_cur, L_cur, M_cur, s_x_cur, zeta_cur,
%   loss_hist, loss_final, and lambda.
%
% Incomplete runs are stored as partial_*.mat files. The plotting script
% ignores those files and can therefore be run while this sweep is active.

clc;
clearvars;
close all;

addingPathParentFolderByName('code');
Parameters;  % supplies lambda, kappa, N_x, N_y, d_x, d_y, A_atom, eta0, seed

%% -------------------------- User settings -------------------------------

% % Values used by the parallel-coordinate figure supplied with the question.
% % T_SIM and s_x are specified in wavelengths here and converted to metres.
% T_SIM_lambda_sweep = 4:7;
% s_x_lambda_sweep   = 0.3:0.1:0.6;
% L_sweep            = 7:2:15;
% M_x_sweep          = 12:17;                 % M = M_x^2
% zeta_sweep         = 0.986:0.001:0.989;

% Example of a smaller sweep around the currently working configuration:
T_SIM_lambda_sweep = [10 12 14];
s_x_lambda_sweep   = [0.40 4/9 0.50];
L_sweep            = [23 27 31];
M_x_sweep          = [30 34 38];
zeta_sweep         = [0.986 0.988 0.990];

maxIter_ablation = 600;
h                 = 1e-6;
checkpointEvery   = 5;
autoResume        = true;
verifyGradient    = false;  % enable only for a small validation run

% Empty means all geometries. This is useful for splitting the sweep among
% independent HPC jobs, e.g. 1:50 in one job and 51:100 in another.
geometrySubset = [];

% The completed file keeps the trained phases and final G because they are
% compact and useful when selecting the best configuration later.
saveTrainedState = true;

%% -------------------------- Fixed quantities ----------------------------

dataset_path = fullfile('..', 'Dataset');
if ~exist(dataset_path, 'dir')
    mkdir(dataset_path);
end

T_SIM_sweep = T_SIM_lambda_sweep * lambda;
s_x_sweep   = s_x_lambda_sweep   * lambda;

N = N_x * N_y;
F = dft2_matrix(N_x, N_y);
[xn, yn] = grid_coords_centered(N_x, N_y, d_x, d_y);

load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = ...
    build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

% Geometry combinations exclude zeta so the propagation matrices are built
% only once and then reused for every zeta value of that geometry.
[Tg, Sg, Lg, MXg] = ndgrid(T_SIM_sweep, s_x_sweep, L_sweep, M_x_sweep);
geometry_table = table(Tg(:), Sg(:), Lg(:), MXg(:), MXg(:).^2, ...
    'VariableNames', {'T_SIM', 's_x', 'L', 'M_x', 'M'});

nGeometry = height(geometry_table);
nRuns     = nGeometry * numel(zeta_sweep);

if isempty(geometrySubset)
    geometrySubset = 1:nGeometry;
else
    geometrySubset = unique(geometrySubset(:).');
    assert(all(geometrySubset >= 1 & geometrySubset <= nGeometry), ...
        'geometrySubset contains an index outside 1:%d.', nGeometry);
end

fprintf('\nAblation design: %d geometries x %d zeta values = %d runs.\n', ...
    nGeometry, numel(zeta_sweep), nRuns);
fprintf('This job will process %d geometries (%d runs).\n\n', ...
    numel(geometrySubset), numel(geometrySubset) * numel(zeta_sweep));

if nRuns > 250
    warning(['The selected full-factorial design is large (%d runs). ' ...
        'Validate the script first with one or two values per variable, ' ...
        'then launch the full sweep or divide geometrySubset among jobs.'], nRuns);
end

sweep_config = struct();
sweep_config.T_SIM_lambda_sweep = T_SIM_lambda_sweep;
sweep_config.s_x_lambda_sweep   = s_x_lambda_sweep;
sweep_config.L_sweep            = L_sweep;
sweep_config.M_x_sweep          = M_x_sweep;
sweep_config.zeta_sweep         = zeta_sweep;
sweep_config.maxIter            = maxIter_ablation;
sweep_config.h                  = h;
sweep_config.checkpointEvery    = checkpointEvery;
sweep_config.N_x                = N_x;
sweep_config.N_y                = N_y;
sweep_config.lambda             = lambda;
sweep_config.seed               = seed;
sweep_config.created_at         = datestr(now, 30);
save(fullfile(dataset_path, 'ablation_manifest.mat'), ...
    'sweep_config', 'geometry_table', '-v7');

%% ------------------------------ Sweep -----------------------------------

for iGeom = geometrySubset
    T_SIM_cur = geometry_table.T_SIM(iGeom);
    s_x_cur   = geometry_table.s_x(iGeom);
    L_cur     = geometry_table.L(iGeom);
    M_x_cur   = geometry_table.M_x(iGeom);
    M_y_cur   = M_x_cur;
    M_cur     = geometry_table.M(iGeom);
    s_y_cur   = s_x_cur;
    s_layer_cur = T_SIM_cur / L_cur;

    combo_tag = make_combo_tag(T_SIM_cur/lambda, s_x_cur/lambda, L_cur, M_cur);

    % Do not spend time rebuilding W if this geometry is already complete
    % for every requested zeta value.
    all_complete = true;
    for iZeta = 1:numel(zeta_sweep)
        zeta_cur = zeta_sweep(iZeta);
        zeta_dir = fullfile(dataset_path, ...
            sprintf('ablation_ckpt_zeta_%.3f', zeta_cur));
        final_file = fullfile(zeta_dir, ['combo_' combo_tag '.mat']);
        if ~result_is_complete(final_file, maxIter_ablation, ...
                T_SIM_cur, s_x_cur, L_cur, M_cur, zeta_cur)
            all_complete = false;
            break;
        end
    end

    if all_complete
        fprintf('[geometry %d/%d] already complete: %s\n', ...
            iGeom, nGeometry, combo_tag);
        continue;
    end

    fprintf(['\n[geometry %d/%d] T_SIM/lambda=%.4g, s_x/lambda=%.4g, ' ...
        'L=%d, M_x=%d, M=%d\n'], ...
        iGeom, nGeometry, T_SIM_cur/lambda, s_x_cur/lambda, ...
        L_cur, M_x_cur, M_cur);

    % Build the geometry-dependent matrices once for all zeta values.
    [xm, ym] = grid_coords_centered(M_x_cur, M_y_cur, s_x_cur, s_y_cur);
    [W0, Wmm, WL] = build_propagation_matrices( ...
        xn, yn, xm, ym, s_layer_cur, A_atom, kappa);

    % Same initialization for every zeta under this geometry. This makes
    % the zeta comparison fair and reproducible.
    rng(seed, 'twister');
    xi0 = cell(L_cur, 1);
    for l = 1:L_cur
        xi0{l} = 2*pi*rand(M_cur, 1);
    end

    for iZeta = 1:numel(zeta_sweep)
        zeta_cur = zeta_sweep(iZeta);

        zeta_dir = fullfile(dataset_path, ...
            sprintf('ablation_ckpt_zeta_%.3f', zeta_cur));
        if ~exist(zeta_dir, 'dir')
            mkdir(zeta_dir);
        end

        final_file   = fullfile(zeta_dir, ['combo_'   combo_tag '.mat']);
        partial_file = fullfile(zeta_dir, ['partial_' combo_tag '.mat']);

        if result_is_complete(final_file, maxIter_ablation, ...
                T_SIM_cur, s_x_cur, L_cur, M_cur, zeta_cur)
            fprintf('  zeta %.3f: complete, skipping.\n', zeta_cur);
            continue;
        end

        meta = struct();
        meta.version       = 1;
        meta.geometryIndex = iGeom;
        meta.T_SIM_cur     = T_SIM_cur;
        meta.s_x_cur       = s_x_cur;
        meta.L_cur         = L_cur;
        meta.M_x_cur       = M_x_cur;
        meta.M_y_cur       = M_y_cur;
        meta.M_cur         = M_cur;
        meta.N_x           = N_x;
        meta.N_y           = N_y;
        meta.N             = N;
        meta.zeta_cur      = zeta_cur;
        meta.lambda        = lambda;
        meta.h             = h;
        meta.eta0          = eta0;
        meta.seed          = seed;
        meta.maxIter       = maxIter_ablation;

        fprintf('  zeta %.3f: training', zeta_cur);
        tic_combo = tic;

        result = train_one_combo(W0, Wmm, WL, F, F_amp, xi0, ...
            eta0, zeta_cur, h, maxIter_ablation, checkpointEvery, ...
            autoResume, partial_file, meta, verifyGradient);

        elapsed_seconds = toc(tic_combo);

        % Names below deliberately match the unchanged plotting code.
        loss_hist = result.loss_hist;
        loss_final = result.loss_final;      % linear scale
        loss_final_dB = 10*log10(loss_final);
        beta_hist = result.beta_hist;
        beta_final = result.beta_final;
        eta_final = result.eta_final;
        G_final = result.G_final;
        iterations_completed = result.iterations_completed;

        zeta_saved = zeta_cur; %#ok<NASGU>
        T_SIM_over_lambda = T_SIM_cur/lambda; %#ok<NASGU>
        s_x_over_lambda   = s_x_cur/lambda; %#ok<NASGU>
        saved_at = datestr(now, 30); %#ok<NASGU>

        output = struct();
        output.T_SIM_cur = T_SIM_cur;
        output.L_cur = L_cur;
        output.M_cur = M_cur;
        output.M_x_cur = M_x_cur;
        output.M_y_cur = M_y_cur;
        output.s_x_cur = s_x_cur;
        output.s_y_cur = s_y_cur;
        output.s_layer_cur = s_layer_cur;
        output.zeta_cur = zeta_cur;
        output.zeta_saved = zeta_saved;
        output.lambda = lambda;
        output.T_SIM_over_lambda = T_SIM_over_lambda;
        output.s_x_over_lambda = s_x_over_lambda;
        output.loss_hist = loss_hist;
        output.loss_final = loss_final;
        output.loss_final_dB = loss_final_dB;
        output.beta_hist = beta_hist;
        output.beta_final = beta_final;
        output.eta_final = eta_final;
        output.G_final = G_final;
        output.iterations_completed = iterations_completed;
        output.maxIter_used = maxIter_ablation;
        output.elapsed_seconds = elapsed_seconds;
        output.phase_min_meas = phase_min_meas;
        output.phase_max_meas = phase_max_meas;
        output.meta = meta;
        output.saved_at = saved_at;
        if saveTrainedState
            output.xi_trained = result.xi_trained;
        end

        atomic_save_struct(final_file, output, '-v7');

        if isfile(partial_file)
            delete(partial_file);
        end

        fprintf(' -> done, final loss %.3f dB, %.1f s.\n', ...
            loss_final_dB, elapsed_seconds);
    end

    clear W0 Wmm WL xm ym xi0
end

fprintf('\nAblation sweep finished for the selected geometrySubset.\n');

%% ============================= Helpers ==================================

function result = train_one_combo(W0, Wmm, WL, F, F_amp, xi0, ...
        eta0, zeta_cur, h, maxIter, checkpointEvery, autoResume, ...
        partial_file, meta, verifyGradient)

    L = meta.L_cur;
    M = meta.M_cur;
    f = F(:);

    loss_hist = zeros(maxIter, 1);
    beta_hist = complex(zeros(maxIter, 1));

    if autoResume && isfile(partial_file)
        C = load(partial_file);
        validate_checkpoint(C, meta, partial_file);

        xi_local = C.xi_local;
        eta_local = C.eta_local;
        beta_local = C.beta_local;
        it_done = C.it_done;

        nCopy = min(it_done, maxIter);
        loss_hist(1:nCopy) = C.loss_hist_chk(1:nCopy);
        beta_hist(1:nCopy) = C.beta_hist_chk(1:nCopy);
        it_first = it_done + 1;

        fprintf(' (resuming at iteration %d)', it_first);
    else
        xi_local = xi0;
        eta_local = eta0;
        beta_local = 1 + 0j;
        it_first = 1;
    end

    gamma = phases_to_gamma(xi_local, F_amp);

    for it = it_first:maxIter
        [A_pre, B_post, G] = prefix_suffix_products(W0, Wmm, WL, gamma);

        g = G(:);
        beta_local = (g' * f) / (g' * g);
        E = beta_local*G - F;
        Lval = norm(E, 'fro')^2;

        loss_hist(it) = Lval;
        beta_hist(it) = beta_local;

        grads = cell(L, 1);
        for l = 1:L
            xi_p_vec = xi_local{l} + h;
            gamma_p = F_amp(mod(xi_p_vec, 2*pi)) .* exp(1i*xi_p_vec);
            delta_vec = gamma_p - gamma{l};

            Bl = B_post{l};
            Al = A_pre{l};
            grad_l = zeros(M, 1);

            for m = 1:M
                G_p = G + delta_vec(m) * (Bl(:,m) * Al(m,:));
                g_p = G_p(:);
                beta_p = (g_p' * f) / (g_p' * g_p);
                E_p = beta_p*G_p - F;
                L_p = norm(E_p, 'fro')^2;
                grad_l(m) = (L_p - Lval) / h;
            end
            grads{l} = grad_l;
        end

        if verifyGradient && it == it_first
            verify_rank_one_gradient(grads, xi_local, gamma, W0, Wmm, ...
                WL, F, F_amp, h, Lval);
        end

        for l = 1:L
            xi_local{l} = mod(xi_local{l} - eta_local*grads{l}, 2*pi);
        end
        gamma = phases_to_gamma(xi_local, F_amp);
        eta_local = eta_local * zeta_cur;

        if mod(it, 10) == 0 || it == it_first || it == maxIter
            fprintf('\n    iter %d/%d, loss %.3f dB', ...
                it, maxIter, 10*log10(Lval));
        end

        if mod(it, checkpointEvery) == 0 || it == maxIter
            chk = struct();
            chk.it_done = it;
            chk.xi_local = xi_local;
            chk.eta_local = eta_local;
            chk.beta_local = beta_local;
            chk.loss_hist_chk = loss_hist(1:it);
            chk.beta_hist_chk = beta_hist(1:it);
            chk.meta = meta;
            chk.saved_at = datestr(now, 30);
            atomic_save_struct(partial_file, chk, '-v7');
        end
    end

    % Evaluate the state after the final phase update. This removes the
    % inconsistency in the original script, where xi_trained was post-update
    % but G_i and loss_hist(end) described the pre-update state.
    G_final = forward_from_gamma(W0, Wmm, WL, gamma);
    g_final = G_final(:);
    beta_final = (g_final' * f) / (g_final' * g_final);
    loss_final = norm(beta_final*G_final - F, 'fro')^2;

    result = struct();
    result.loss_hist = loss_hist;
    result.beta_hist = beta_hist;
    result.loss_final = loss_final;
    result.beta_final = beta_final;
    result.eta_final = eta_local;
    result.G_final = G_final;
    result.xi_trained = xi_local;
    result.iterations_completed = maxIter;
end

function [A_pre, B_post, G] = prefix_suffix_products(W0, Wmm, WL, gamma)
    L = numel(gamma);

    A_pre = cell(L, 1);
    A_pre{1} = W0;
    for l = 2:L
        A_pre{l} = Wmm * (gamma{l-1} .* A_pre{l-1});
    end

    B_post = cell(L, 1);
    B_post{L} = WL;
    for l = (L-1):-1:1
        B_post{l} = (B_post{l+1} .* gamma{l+1}.') * Wmm;
    end

    G = B_post{1} * (gamma{1} .* A_pre{1});
end

function G = forward_from_gamma(W0, Wmm, WL, gamma)
    X = gamma{1} .* W0;
    for l = 2:numel(gamma)
        X = gamma{l} .* (Wmm * X);
    end
    G = WL * X;
end

function gamma = phases_to_gamma(xi, F_amp)
    gamma = cell(size(xi));
    for l = 1:numel(xi)
        amp_l = F_amp(mod(xi{l}, 2*pi));
        gamma{l} = amp_l .* exp(1i*xi{l});
    end
end

function verify_rank_one_gradient(grads, xi, gamma, W0, Wmm, WL, ...
        F, F_amp, h, Lval)
    L = numel(xi);
    M = numel(xi{1});
    f = F(:);

    l_check = unique([1, ceil(L/2), L]);
    m_check = unique(round(linspace(1, M, min(5, M))));
    max_abs = 0;
    g_scale = 0;

    for l = l_check
        for m = m_check
            gamma_p = gamma;
            xi_pm = xi{l}(m) + h;
            gamma_p{l}(m) = F_amp(mod(xi_pm, 2*pi)) * exp(1i*xi_pm);

            G_chk = forward_from_gamma(W0, Wmm, WL, gamma_p);
            g_chk = G_chk(:);
            beta_chk = (g_chk' * f) / (g_chk' * g_chk);
            L_chk = norm(beta_chk*G_chk - F, 'fro')^2;
            grad_chk = (L_chk - Lval) / h;

            max_abs = max(max_abs, abs(grads{l}(m) - grad_chk));
            g_scale = max(g_scale, abs(grad_chk));
        end
    end

    fprintf('\n    gradient check: max abs %.3e, relative %.3e', ...
        max_abs, max_abs/max(g_scale, eps));
end

function [W0, Wmm, WL] = build_propagation_matrices( ...
        xn, yn, xm, ym, s_layer, A_atom, kappa)

    % Input -> first intermediate layer, M x N.
    dx0 = xm - xn.';
    dy0 = ym - yn.';
    d0 = sqrt(dx0.^2 + dy0.^2 + s_layer^2);
    W0 = rayleigh_sommerfeld_kernel(d0, s_layer, A_atom, kappa);

    % Between any two intermediate layers, M x M. All hops are identical
    % because every layer uses the same grid and the same separation.
    dxm = xm - xm.';
    dym = ym - ym.';
    dmm = sqrt(dxm.^2 + dym.^2 + s_layer^2);
    Wmm = rayleigh_sommerfeld_kernel(dmm, s_layer, A_atom, kappa);

    % Reciprocal/isomorphic arrangement used by the original script.
    WL = W0.';
end

function W = rayleigh_sommerfeld_kernel(d, s_layer, A_atom, kappa)
    W = (A_atom*s_layer) ./ (2*pi*d.^3) .* ...
        (1 - 1i*kappa.*d) .* exp(1i*kappa.*d);
end

function tf = result_is_complete(file_name, maxIter, ...
        T_SIM_cur, s_x_cur, L_cur, M_cur, zeta_cur)
    tf = false;
    if ~isfile(file_name)
        return;
    end

    try
        S = load(file_name, 'iterations_completed', 'maxIter_used', ...
            'T_SIM_cur', 's_x_cur', 'L_cur', 'M_cur', 'zeta_cur', ...
            'loss_final');

        required = {'iterations_completed', 'maxIter_used', 'T_SIM_cur', ...
            's_x_cur', 'L_cur', 'M_cur', 'zeta_cur', 'loss_final'};
        if ~all(isfield(S, required))
            return;
        end

        tf = S.iterations_completed >= maxIter && ...
             S.maxIter_used >= maxIter && ...
             nearly_equal(S.T_SIM_cur, T_SIM_cur) && ...
             nearly_equal(S.s_x_cur, s_x_cur) && ...
             S.L_cur == L_cur && S.M_cur == M_cur && ...
             nearly_equal(S.zeta_cur, zeta_cur) && ...
             isfinite(S.loss_final) && S.loss_final > 0;
    catch
        tf = false;
    end
end

function validate_checkpoint(C, meta, partial_file)
    required = {'it_done', 'xi_local', 'eta_local', 'beta_local', ...
        'loss_hist_chk', 'beta_hist_chk', 'meta'};
    assert(all(isfield(C, required)), ...
        'Checkpoint %s is incomplete or corrupt.', partial_file);

    cm = C.meta;
    assert(cm.L_cur == meta.L_cur && cm.M_cur == meta.M_cur && ...
        cm.N_x == meta.N_x && cm.N_y == meta.N_y && ...
        nearly_equal(cm.T_SIM_cur, meta.T_SIM_cur) && ...
        nearly_equal(cm.s_x_cur, meta.s_x_cur) && ...
        nearly_equal(cm.zeta_cur, meta.zeta_cur) && ...
        nearly_equal(cm.h, meta.h) && cm.seed == meta.seed, ...
        ['Checkpoint %s does not match the current combination. ' ...
         'Delete it or restore the original sweep settings.'], partial_file);
end

function atomic_save_struct(file_name, S, version_flag)
    folder = fileparts(file_name);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end

    tmp_file = [tempname(folder) '.mat'];
    try
        save(tmp_file, '-struct', 'S', version_flag);
        movefile(tmp_file, file_name, 'f');
    catch ME
        if isfile(tmp_file)
            delete(tmp_file);
        end
        rethrow(ME);
    end
end

function tag = make_combo_tag(T_SIM_lambda, s_x_lambda, L, M)
    tag = sprintf('TSIM_%s_sx_%s_L_%d_M_%d', ...
        number_tag(T_SIM_lambda), number_tag(s_x_lambda), L, M);
end

function out = number_tag(x)
    out = sprintf('%.6g', x);
    out = strrep(out, '.', 'p');
    out = strrep(out, '-', 'm');
end

function tf = nearly_equal(a, b)
    scale = max([1, abs(a), abs(b)]);
    tf = abs(a-b) <= 1e-11*scale;
end

function [x, y] = grid_coords_centered(Nx, Ny, dx, dy)
    N = Nx*Ny;
    idx = (1:N).';
    iy = ceil(idx/Nx);
    ix = idx - (iy-1)*Nx;
    x = (ix - 1 - (Nx-1)/2) * dx;
    y = ((Ny-1)/2 - (iy - 1)) * dy;
end

function [F_amp, phase_min, phase_max] = ...
        build_amplitude_interpolant(mag_dB, phase_deg)
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
    F_amp = griddedInterpolant(phase_wrapped, mag_sorted, ...
        'pchip', 'nearest');
end

function F = dft2_matrix(Nx, Ny)
    N = Nx*Ny;
    n = (0:N-1).';
    nx = mod(n, Nx);
    ny = floor(n/Nx);
    F = exp(-1i*2*pi*(nx*nx.')/Nx) .* ...
        exp(-1i*2*pi*(ny*ny.')/Ny);
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
        fprintf('Adding MATLAB path to: %s\n', currFolder);
    else
        error('Folder named "%s" not found in any parent directory.', targetName);
    end
end
