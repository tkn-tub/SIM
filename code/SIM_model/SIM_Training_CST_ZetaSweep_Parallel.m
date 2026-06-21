%[text] # SIM training (WITH CST) -- full zeta sweep, parallelized per value
%[text] Sweeps the FULL zeta vector already defined in Parameters.m
%[text] (zeta = zeta_ini:zeta_delta:zeta_end = 0.980:0.001:0.999, 20 values).
%[text] zeta=0.988 is already known to be a strong choice from prior work;
%[text] highlighted in the final comparison plot.
%[text]
%[text] Design choices, given this is ~20x the cost of the single-zeta run:
%[text]   - Geometry (W0, W, WL) and the initial random phases are built ONCE
%[text]     and shared across all 20 zeta values -- only the decay schedule
%[text]     differs between them. Every zeta starts from the SAME initial
%[text]     phases, for a fair comparison (isolates the effect of the decay
%[text]     schedule from initialization randomness).
%[text]   - ONE checkpoint file per zeta value -- a crash partway through the
%[text]     sweep only costs progress on whichever zeta was mid-training;
%[text]     rerunning the script skips zeta values that already finished.
%[text]   - NO live per-iteration plotting (this can run a long time
%[text]     unattended; console progress printing only). One combined
%[text]     comparison figure is built at the very end from all 20 stored
%[text]     loss histories.
%[text]   - Same chunked-parfor / fresh-pool / periodic-refresh machinery as
%[text]     the single-zeta script, reused across the whole sweep (the pool
%[text]     itself is not torn down between zeta values, only refreshed
%[text]     periodically within a given zeta's training, same as before).

clc;
clear all;
close all;

addingPathParentFolderByName('code');

Parameters
% zeta = zeta_ini:zeta_delta:zeta_end already defined here (0.980:0.001:0.999)

%% ----------------- Load CST amplitude/phase data -----------------
load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

F = dft2_matrix(N_x, N_y);

%% ----------------- Sweep settings -----------------
h = 1e-5;
checkpoint_every = 25;
pool_refresh_every = 100;
zeta_highlight = 0.988;   % known-good value from prior single-zeta work

n_zeta = numel(zeta);
fprintf('Sweeping %d zeta values: %.3f to %.3f\n', n_zeta, zeta(1), zeta(end));

%% ----------------- Geometry (built once, shared across all zeta values) -----------------
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

%% ----------------- Fixed initial phases, shared across all zeta values -----------------
xi0 = cell(L,1);
Upsilon0 = cell(L,1);
for l = 1:L
    xi0{l} = 2*pi*rand(M,1);
    amp_l = F_amp(mod(xi0{l}, 2*pi));
    Upsilon0{l} = diag(amp_l .* exp(1i*xi0{l}));
end

%% ----------------- Start pool, cache read-only large objects -----------------
delete(gcp('nocreate'));
parpool;
% c = parcluster('local'); c.NumWorkers = str2double(getenv('SLURM_CPUS_PER_TASK')); parpool(c);

W_c    = parallel.pool.Constant(W);
WL_c   = parallel.pool.Constant(WL);
W0_c   = parallel.pool.Constant(W0);
F_c    = parallel.pool.Constant(F);
Famp_c = parallel.pool.Constant(F_amp);

[Lg, Mg] = ndgrid(1:L, 1:M);
l_list = Lg(:);
m_list = Mg(:);
n_pert = numel(l_list);

nW = gcp().NumWorkers;
chunkEdges = round(linspace(0, n_pert, nW+1));

%% ----------------- Sweep over zeta values -----------------
loss_hist_all = cell(n_zeta,1);

t_sweep = tic;
for iz = 1:n_zeta
    zeta_val = zeta(iz);
    checkpoint_path = sprintf('SIM_training_CST_checkpoint_zeta_%05.3f.mat', zeta_val);

    fprintf('\n=== zeta(%d/%d) = %.3f ===\n', iz, n_zeta, zeta_val);

    % ---- resume-or-init for THIS zeta value ----
    if isfile(checkpoint_path)
        S = load(checkpoint_path);
        xi = S.xi; Upsilon = S.Upsilon;
        loss_hist = S.loss_hist; beta_hist = S.beta_hist; gap_frac_hist = S.gap_frac_hist;
        eta = S.eta;
        start_it = S.it + 1;
        if S.maxIter ~= maxIter
            warning('Checkpoint for zeta=%.3f used maxIter=%d, this run uses %d. Resizing histories.', ...
                zeta_val, S.maxIter, maxIter);
            loss_hist(end+1:maxIter) = 0;
            beta_hist(end+1:maxIter) = 0;
            gap_frac_hist(end+1:maxIter) = 0;
        end
        fprintf('Found checkpoint, resuming from iteration %d (of %d).\n', start_it, maxIter);
    else
        xi = xi0;
        Upsilon = Upsilon0;
        eta = eta0;
        loss_hist = zeros(maxIter,1);
        beta_hist = complex(zeros(maxIter,1));
        gap_frac_hist = zeros(maxIter,1);
        start_it = 1;
    end

    if start_it > maxIter
        fprintf('zeta=%.3f already complete (%d/%d iterations) -- skipping training.\n', zeta_val, S.it, maxIter);
        loss_hist_all{iz} = loss_hist;
        continue;
    end

    % ---- per-zeta training loop ----
    beta = 1+0j;
    t_start = tic;
    for it = start_it:maxIter
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

        for l = 1:L
            xi{l} = mod(xi{l} - eta*grads{l}, 2*pi);
        end
        for l = 1:L
            amp_l = F_amp(mod(xi{l}, 2*pi));
            Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
        end

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

        eta = eta * zeta_val;

        fprintf('  zeta=%.3f, iter %d/%d (%.1f%%), loss=%.4f, |beta|=%.4f, eta=%.4e, elapsed=%.1f s\n', ...
            zeta_val, it, maxIter, 100*it/maxIter, Lval, abs(beta), eta, toc(t_start));

        if mod(it, checkpoint_every) == 0 || it == maxIter
            save(checkpoint_path, 'xi', 'Upsilon', 'loss_hist', 'beta_hist', 'gap_frac_hist', ...
                 'it', 'eta', 'zeta_val', 'maxIter', 'fc', '-v7.3');
            fprintf('    [checkpoint saved at iteration %d]\n', it);
        end

        if mod(it, pool_refresh_every) == 0 && it < maxIter
            fprintf('    [recycling parallel pool at iteration %d]\n', it);
            delete(gcp('nocreate'));
            parpool;
            nW = gcp().NumWorkers;
            chunkEdges = round(linspace(0, n_pert, nW+1));
            W_c    = parallel.pool.Constant(W);
            WL_c   = parallel.pool.Constant(WL);
            W0_c   = parallel.pool.Constant(W0);
            F_c    = parallel.pool.Constant(F);
            Famp_c = parallel.pool.Constant(F_amp);
        end
    end

    loss_hist_all{iz} = loss_hist;
    fprintf('zeta=%.3f done: %d iterations, %.1f s.\n', zeta_val, maxIter, toc(t_start));
end
fprintf('\nFull sweep done: %d zeta values, %.1f s total.\n', n_zeta, toc(t_sweep));

delete(gcp('nocreate'));
fprintf('Parallel pool released.\n');

%% ----------------- Combined comparison plot -----------------
cmap = parula(n_zeta);
figure; hold on; grid on;
h_highlight = [];
for iz = 1:n_zeta
    if isempty(loss_hist_all{iz})
        continue;   % shouldn't happen, but skip defensively if a zeta value was never run
    end
    if abs(zeta(iz) - zeta_highlight) < 1e-9
        h_highlight = semilogy(loss_hist_all{iz}, 'LineWidth', 3, 'Color', 'r');
    else
        semilogy(loss_hist_all{iz}, 'LineWidth', 1, 'Color', cmap(iz,:));
    end
end
set(gca, 'YScale', 'log', 'FontSize', font);
colormap(gca, parula);
cb = colorbar; cb.Label.String = '\zeta value'; cb.Label.Interpreter = 'tex';
clim([zeta(1) zeta(end)]);
xlabel('Iterations', 'Interpreter','latex');
ylabel('$\mathcal{L}=\|\beta G-F\|^2$', 'Interpreter','latex');
title(sprintf('CST training, full $\\zeta$ sweep ($f_c$=%.0f GHz)', fc/1e9), 'Interpreter','latex');
if ~isempty(h_highlight)
    legend(h_highlight, sprintf('\\zeta=%.3f (selected)', zeta_highlight), 'Location','best');
end

%% ----------------- Save combined results -----------------
path = fullfile('..', 'Dataset', 'SIM_training_CST_zeta_sweep_28GHz.mat');
save(path, 'zeta', 'loss_hist_all', 'maxIter', 'eta0', 'seed', 'fc', 'zeta_highlight', '-v7.3');
fprintf('Saved combined sweep results to %s\n', path);

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
