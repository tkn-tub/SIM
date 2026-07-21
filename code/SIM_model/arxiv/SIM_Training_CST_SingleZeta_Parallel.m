%[text] # SIM training (WITH CST amplitude coupling) -- single fixed zeta, parallelized
%[text] Same structure and robustness fixes as the no-CST version, with the
%[text] CST-realistic amplitude-phase coupling (F_amp, built from t_y_x.mat)
%[text] folded back in. Reads ALL geometry from Parameters.m (28 GHz, per
%[text] fc=28*10^9), exactly like the no-CST script.
%[text]
%[text] Robustness fixes carried over:
%[text]   - ONE fixed zeta (no sweep, no accidental cumulative carry-over)
%[text]   - parfor chunked by worker (NumWorkers tasks/iteration, not L*M)
%[text]   - fresh pool at start, periodic pool refresh during long runs
%[text]   - checkpoint every N iterations, with resume support
%[text]   - early-exit finalization (single forward pass, no pool) if the
%[text]     checkpoint shows training already completed

clc;
clear all;
close all;

addingPathParentFolderByName('code');

Parameters

%% ----------------- Load CST amplitude/phase data -----------------
load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

% Evaluating target 2D-DFT matrix F in Eq. (8) as
F = dft2_matrix(N_x, N_y);

%% ----------------- Single fixed zeta/eta, increased iterations -----------------
maxIter = 2000
zeta_fixed = 0.995 %0.988;   % same value established from the ablation study; adjust if needed
h = 1e-5;              % finite-difference step (radians)
checkpoint_every = 25;
checkpoint_path = 'SIM_training_CST_checkpoint.mat';
pool_refresh_every = 100;   % recycle the parallel pool periodically during long runs

%% ----------------- Geometry (unchanged from the original script) -----------------
rng(seed);
[xn, yn] = grid_coords_centered(N_x, N_y, d_x, d_y);
[xm, ym] = grid_coords_centered(M_x, M_y, s_x, s_y);

W0 = zeros(M, N);
for m = 1:M
    for n = 1:N
        d = sqrt((xm(m)-xn(n))^2 + (ym(m)-yn(n))^2 + s_layer^2);
        cos_epsilon = s_layer/d;
        W0(m,n) = (A_atom*cos_epsilon)/(2*pi*d^2) * (1-1j*kappa*d) * exp(1j*kappa*d);
    end
end
WL = W0.';

W = cell(L-1,1);
for l = 1:(L-1)
    for m = 1:M
        for n = 1:M
            d = sqrt((xm(m)-xm(n))^2 + (ym(m)-ym(n))^2 + s_layer^2);
            cos_epsilon = s_layer/d;
            W_matrix(m,n) = (A_atom*cos_epsilon)/(2*pi*d^2) * (1-1j*kappa*d) * exp(1j*kappa*d);
        end
    end
    W{l} = W_matrix;
end

%% ----------------- Init phases / CST-coupled Upsilon -----------------
%      -- or resume from a checkpoint if one exists --
if isfile(checkpoint_path)
    fprintf('Found existing checkpoint at %s -- resuming.\n', checkpoint_path);
    S = load(checkpoint_path);
    xi = S.xi;
    Upsilon = S.Upsilon;
    loss_hist = S.loss_hist;
    beta_hist = S.beta_hist;
    gap_frac_hist = S.gap_frac_hist;
    eta = S.eta;
    start_it = S.it + 1;
    if S.maxIter ~= maxIter
        warning('Checkpoint was saved with maxIter=%d but this run uses maxIter=%d. Proceeding with the current maxIter; histories will be resized.', ...
            S.maxIter, maxIter);
        loss_hist(end+1:maxIter) = 0;
        beta_hist(end+1:maxIter) = 0;
        gap_frac_hist(end+1:maxIter) = 0;
    end
    fprintf('Resuming from iteration %d (of %d).\n', start_it, maxIter);

    if start_it > maxIter
        % Training already finished in a previous run -- finalize directly
        % from the saved, fully-trained phases with ONE ordinary forward
        % pass -- no parfor, no pool.
        fprintf('Checkpoint shows training already completed (%d of %d iterations).\n', S.it, maxIter);
        fprintf('Finalizing directly from the checkpoint, without starting a parallel pool.\n');

        G_intermediate = eye(M,M);
        for l = 2:L
            G_intermediate = Upsilon{l}*W{l-1}*G_intermediate;
        end
        G = WL*G_intermediate*Upsilon{1}*W0;
        g = G(:); f = F(:);
        beta = (g'*g) \ (g'*f);

        path = fullfile('..', 'Dataset', 'SIM_training_CST_single_zeta.mat');
        save(path, 'xi', 'Upsilon', 'G', 'beta', 'loss_hist', 'beta_hist', 'gap_frac_hist', ...
             'zeta_fixed', 'maxIter', 'eta0', 'seed', 'fc', '-v7.3');
        fprintf('Finalized and saved (%s) without touching the parallel pool.\n', path);

        delete(gcp('nocreate'));
        return;
    end
else
    xi = cell(L,1);
    Upsilon = cell(L,1);
    for l = 1:L
        xi{l} = 2*pi*rand(M,1);
        amp_l = F_amp(mod(xi{l}, 2*pi));
        Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
    end
    eta = eta0;
    loss_hist = zeros(maxIter,1);
    beta_hist = complex(zeros(maxIter,1));
    gap_frac_hist = zeros(maxIter,1);
    start_it = 1;
end

%% ----------------- Start pool, cache read-only large objects -----------------
% Force a FRESH pool every run rather than reusing whatever pool may have
% accumulated state across previous script executions in this session.
delete(gcp('nocreate'));
parpool('local');
% size the pool explicitly on a shared HPC node if needed:
% c = parcluster('local');
% c.NumWorkers = str2double(getenv('SLURM_CPUS_PER_TASK'));
% parpool(c);

W_c    = parallel.pool.Constant(W);
WL_c   = parallel.pool.Constant(WL);
W0_c   = parallel.pool.Constant(W0);
F_c    = parallel.pool.Constant(F);
Famp_c = parallel.pool.Constant(F_amp);

[Lg, Mg] = ndgrid(1:L, 1:M);
l_list = Lg(:);
m_list = Mg(:);
n_pert = numel(l_list);   % = L*M

% ---- chunk perturbations across workers instead of one task per (l,m) ----
nW = gcp().NumWorkers;
chunkEdges = round(linspace(0, n_pert, nW+1));

%% ----------------- Quick one-iteration benchmark -----------------
fprintf('Benchmarking a single iteration before committing to %d...\n', maxIter);
t_bench = tic;
G_intermediate = eye(M,M);
for l = 2:L
    G_intermediate = Upsilon{l}*W{l-1}*G_intermediate;
end
G_bench = WL*G_intermediate*Upsilon{1}*W0;
g_bench = G_bench(:); f_bench = F(:);
beta_bench = (g_bench'*g_bench) \ (g_bench'*f_bench);
Lval_bench = norm(beta_bench*G_bench - F, 'fro')^2;

xi_bench = xi;
grad_bench = zeros(n_pert,1);
bench_results = cell(nW,1);
parfor c = 1:nW
    idxRange = (chunkEdges(c)+1):chunkEdges(c+1);
    local_grad = zeros(numel(idxRange),1);
    Wl = W_c.Value; WLl = WL_c.Value; W0l = W0_c.Value; Fl = F_c.Value; Fampl = Famp_c.Value;
    for ii = 1:numel(idxRange)
        k = idxRange(ii);
        l = l_list(k); m = m_list(k);
        xi_p = xi_bench{l}(m) + h;
        amp_p = Fampl(mod(xi_p, 2*pi));

        Upsilon_local = cell(L,1);
        for ll = 1:L
            amp_ll = Fampl(mod(xi_bench{ll}, 2*pi));
            Upsilon_local{ll} = diag(amp_ll .* exp(1i*xi_bench{ll}));
        end
        Upsilon_local{l}(m,m) = amp_p * exp(1i*xi_p);

        G_intermediate = eye(M,M);
        for l_int = 2:L
            G_intermediate = Upsilon_local{l_int}*Wl{l_int-1}*G_intermediate;
        end
        G_p = WLl*G_intermediate*Upsilon_local{1}*W0l;
        g_p = G_p(:);
        f_p = Fl(:);
        beta_p = (g_p'*g_p) \ (g_p'*f_p);
        local_grad(ii) = (norm(beta_p*G_p - Fl, 'fro')^2 - Lval_bench) / h;
    end
    bench_results{c} = local_grad;
end
for c = 1:nW
    idxRange = (chunkEdges(c)+1):chunkEdges(c+1);
    grad_bench(idxRange) = bench_results{c};
end
t_per_iter = toc(t_bench);
est_total_hours = t_per_iter * maxIter / 3600;
fprintf('One iteration took %.1f s (%d workers). Estimated total for %d iterations: %.2f hours.\n', ...
    t_per_iter, gcp('nocreate').NumWorkers, maxIter, est_total_hours);
fprintf('Adjust maxIter now if that estimate doesn''t fit your time budget.\n\n');

%% ----------------- Training loop -----------------
beta = 1+0j;

% fig = figure;
% h_plot = semilogy(nan, nan, 'b-', 'LineWidth', 1.5);
% grid on; xlim([1 maxIter]);
% xlabel('Iterations', 'Interpreter','latex');
% ylabel('$\mathcal{L}=\|\beta G-F\|^2$', 'Interpreter','latex');
% title(sprintf('Training (CST), $\\zeta$=%.3f, $f_c$=%.0f GHz (live)', zeta_fixed, fc/1e9), 'Interpreter','latex');
% set(gca, 'FontSize', font);
% if start_it > 1
%     set(h_plot, 'XData', 1:(start_it-1), 'YData', loss_hist(1:(start_it-1)));
% end

t_start = tic;
for it = start_it:maxIter
    % ----- forward G -----
    G_intermediate = eye(M,M);
    for l = 2:L
        G_intermediate = Upsilon{l}*W{l-1}*G_intermediate;
    end
    G = WL*G_intermediate*Upsilon{1}*W0;

    g = G(:); f = F(:);
    beta = (g'*g) \ (g'*f);

    E = beta*G - F;
    Lval = norm(E,'fro')^2;
    loss_hist(it) = Lval;
    beta_hist(it) = beta;

    % ----- parallel finite-difference gradient over all (l,m), chunked by worker -----
    xi_cur = xi;
    grad_vals = zeros(n_pert,1);
    iter_results = cell(nW,1);
    parfor c = 1:nW
        idxRange = (chunkEdges(c)+1):chunkEdges(c+1);
        local_grad = zeros(numel(idxRange),1);
        Wl = W_c.Value; WLl = WL_c.Value; W0l = W0_c.Value; Fl = F_c.Value; Fampl = Famp_c.Value;
        for ii = 1:numel(idxRange)
            k = idxRange(ii);
            l = l_list(k); m = m_list(k);
            xi_p = xi_cur{l}(m) + h;
            amp_p = Fampl(mod(xi_p, 2*pi));

            Upsilon_local = cell(L,1);
            for ll = 1:L
                amp_ll = Fampl(mod(xi_cur{ll}, 2*pi));
                Upsilon_local{ll} = diag(amp_ll .* exp(1i*xi_cur{ll}));
            end
            Upsilon_local{l}(m,m) = amp_p * exp(1i*xi_p);

            G_intermediate = eye(M,M);
            for l_int = 2:L
                G_intermediate = Upsilon_local{l_int}*Wl{l_int-1}*G_intermediate;
            end
            G_p = WLl*G_intermediate*Upsilon_local{1}*W0l;
            g_p = G_p(:);
            f_p = Fl(:);
            beta_p = (g_p'*g_p) \ (g_p'*f_p);

            E_p = beta_p*G_p - Fl;
            L_p = norm(E_p,'fro')^2;

            local_grad(ii) = (L_p - Lval) / h;
        end
        iter_results{c} = local_grad;
    end
    for c = 1:nW
        idxRange = (chunkEdges(c)+1):chunkEdges(c+1);
        grad_vals(idxRange) = iter_results{c};
    end

    grads = cell(L,1);
    for l = 1:L
        grads{l} = zeros(M,1);
    end
    for k = 1:n_pert
        grads{l_list(k)}(m_list(k)) = grad_vals(k);
    end

    % ----- update phases -----
    for l = 1:L
        xi{l} = mod(xi{l} - eta*grads{l}, 2*pi);
    end
    for l = 1:L
        amp_l = F_amp(mod(xi{l}, 2*pi));
        Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
    end

    % ----- gap diagnostic on this iteration's phases -----
    in_gap_count = 0;
    for l = 1:L
        xi_mod = mod(xi{l}, 2*pi);
        if phase_min_meas <= phase_max_meas
            out_of_range = (xi_mod < phase_min_meas) | (xi_mod > phase_max_meas);
        else
            out_of_range = (xi_mod < phase_min_meas) & (xi_mod > phase_max_meas);
        end
        in_gap_count = in_gap_count + sum(out_of_range);
    end
    gap_frac_hist(it) = in_gap_count / (M*L);

    eta = eta * zeta_fixed;

    % ----- live feedback -----
    fprintf('iter %d/%d (%.1f%%), loss=%.4f, |beta|=%.4f, eta=%.4e, gapfrac=%.1f%%, elapsed=%.1f s\n', ...
        it, maxIter, 100*it/maxIter, Lval, abs(beta), eta, 100*gap_frac_hist(it), toc(t_start));
    % set(h_plot, 'XData', 1:it, 'YData', loss_hist(1:it));
    % drawnow;

    % ----- periodic checkpoint -----
    if mod(it, checkpoint_every) == 0 || it == maxIter
        save(checkpoint_path, 'xi', 'Upsilon', 'loss_hist', 'beta_hist', 'gap_frac_hist', ...
             'it', 'eta', 'zeta_fixed', 'maxIter', 'fc', '-v7.3');
        fprintf('  [checkpoint saved at iteration %d]\n', it);
    end

    % ----- periodic pool refresh -----
    if mod(it, pool_refresh_every) == 0 && it < maxIter
        fprintf('  [recycling parallel pool at iteration %d]\n', it);
        delete(gcp('nocreate'));
        parpool('local');
        nW = gcp().NumWorkers;
        chunkEdges = round(linspace(0, n_pert, nW+1));
        W_c    = parallel.pool.Constant(W);
        WL_c   = parallel.pool.Constant(WL);
        W0_c   = parallel.pool.Constant(W0);
        F_c    = parallel.pool.Constant(F);
        Famp_c = parallel.pool.Constant(F_amp);
    end
end
fprintf('\nTraining done: %d iterations, %.1f s total.\n', maxIter, toc(t_start));

%% ----------------- Save final result -----------------
path = fullfile('..', 'Dataset', 'SIM_training_CST_zeta_0_995_Nx_8.mat');
save(path, 'xi', 'Upsilon', 'G', 'beta', 'loss_hist', 'beta_hist', 'gap_frac_hist', ...
     'zeta_fixed', 'maxIter', 'eta0', 'seed', 'fc', '-v7.3');

delete(gcp('nocreate'));
fprintf('Parallel pool released.\n');

%% ======================= Helpers =======================
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
