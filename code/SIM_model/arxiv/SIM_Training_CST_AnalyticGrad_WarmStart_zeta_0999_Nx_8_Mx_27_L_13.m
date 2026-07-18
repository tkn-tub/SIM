%[text] # SIM training (WITH CST amplitude coupling) -- analytic gradient + warm-start
%[text] Replaces the parfor finite-difference gradient with a closed-form
%[text] forward-backward sweep. Cost per iteration: O(L) matrix multiplications
%[text] instead of O(L*M) forward passes -- roughly M=729x faster than the
%[text] parallelized FD version for Mx=27, L=13.
%[text]
%[text] Analytic gradient derivation:
%[text]   G = WL * Upsilon{L} * W{L-1} * ... * Upsilon{1} * W0
%[text]   E = beta*G - F   (residual)
%[text]   For each layer l, define:
%[text]     B{l} = W{l-1}*...*Upsilon{1}*W0    (right partial product, feeds INTO Upsilon{l})
%[text]     A{l} = WL*Upsilon{L}*...*W{l}      (left partial product, comes AFTER Upsilon{l})
%[text]   Then: dL/d(xi_{l,m}) = 2*Re( beta* * c_{l,m} * [A{l}'*E*B{l}']_{mm} )
%[text]   where c_{l,m} = (a'(xi)+i*a(xi))*exp(i*xi) encodes the CST amplitude coupling.
%[text]   a'(xi) computed by central FD on F_amp (same h as before).
%[text]   The d(beta)/d(xi) term is dropped (standard in SIM literature, same
%[text]   as An et al. 2024 -- vanishes at convergence since E->0).
%[text]
%[text] Warm-start: upsamples trained Mx=17 phase profiles to Mx=27 via bilinear
%[text] interpolation, skipping the flat random-walk phase (iters 0-600 in the
%[text] FD runs). Falls back to random init if the warm-start file is not found.
%[text]
%[text] No parfor, no parallel pool -- single-threaded, runs on any MATLAB.

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
maxIter = 8000
zeta_fixed = 0.999 %0.988;   % same value established from the ablation study; adjust if needed
M_x=27
M_y=M_x;
M=M_x*M_y;
L=13
h = 1e-5;              % finite-difference step (radians)
checkpoint_every = 25;
checkpoint_path = 'SIM_training_CST_analytic_zeta_0_999_Mx_27_L_13.mat';
% pool_refresh_every removed -- no parallel pool in analytic-gradient version

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

        path = fullfile('..', 'Dataset', 'SIM_training_CST_analytic_Mx27_L13.mat');
        save(path, 'xi', 'Upsilon', 'G', 'beta', 'loss_hist', 'beta_hist', 'gap_frac_hist', ...
             'zeta_fixed', 'maxIter', 'eta0', 'seed', 'fc', '-v7.3');
        fprintf('Finalized and saved (%s).\n', path);
        return;
    end
else
    %% ----- Warm-start: upsample trained Mx=17 phases to Mx=27 grid -----
    % Replace path below with whatever .mat holds your trained Mx=17 result.
    % The file must contain 'xi' (cell array of L phase vectors).
    warm_start_file = fullfile('..', 'Dataset', 'SIM_training_CST_zeta_0_999_Nx_8_Mx_17_L_13.mat');

    if isfile(warm_start_file)
        fprintf('Warm-starting from %s ...\n', warm_start_file);
        S_warm = load(warm_start_file, 'xi');
        M_warm   = numel(S_warm.xi{1});
        Mx_warm  = round(sqrt(M_warm));
        L_warm   = numel(S_warm.xi);

        xi = cell(L,1);
        Upsilon = cell(L,1);
        for l = 1:L
            if l <= L_warm
                xi_2d    = reshape(mod(S_warm.xi{l}, 2*pi), Mx_warm, Mx_warm);
                xi_2d_up = imresize(xi_2d, [M_x, M_x], 'bilinear');
                xi{l}    = xi_2d_up(:);
            else
                xi{l} = 2*pi*rand(M,1);
            end
            amp_l      = F_amp(mod(xi{l}, 2*pi));
            Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
        end
        fprintf('Warm-start done: %d/%d layers upsampled from %dx%d to %dx%d.\n', ...
            min(L,L_warm), L, Mx_warm, Mx_warm, M_x, M_x);
    else
        fprintf('Warm-start file not found -- using random init.\n');
        xi = cell(L,1);
        Upsilon = cell(L,1);
        for l = 1:L
            xi{l}      = 2*pi*rand(M,1);
            amp_l      = F_amp(mod(xi{l}, 2*pi));
            Upsilon{l} = diag(amp_l .* exp(1i*xi{l}));
        end
    end
    eta = eta0;
    loss_hist     = zeros(maxIter,1);
    beta_hist     = complex(zeros(maxIter,1));
    gap_frac_hist = zeros(maxIter,1);
    start_it = 1;
end

%% ----------------- Pre-compute partial products for analytic gradient -----
% B{l} = W{l-1}*...*Upsilon{1}*W0  (right partial, feeds INTO Upsilon{l})
% A{l} = WL*Upsilon{L}*...*W{l}    (left partial, comes AFTER Upsilon{l})
% Both are rebuilt every iteration inside the training loop.
% (This comment block replaces the parpool/benchmark setup.)

fprintf('Analytic gradient mode -- no parallel pool needed.\n');
fprintf('Estimated speedup vs FD: ~%dx (M=%d perturbations eliminated per iter).\n', M, M);
fprintf('Starting %d iterations from iter %d...\n', maxIter - start_it + 1, start_it);

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

    % ----- analytic gradient via forward-backward sweep -----
    % Step 1: build right partial products B{l} = input to Upsilon{l}
    %   B{1} = W0  (input to first layer)
    %   B{l} = W{l-1} * Upsilon{l-1} * B{l-1}  for l=2..L
    B = cell(L,1);
    B{1} = W0;
    for l = 2:L
        B{l} = W{l-1} * Upsilon{l-1} * B{l-1};
    end

    % Step 2: build left partial products A{l} = output from Upsilon{l}
    %   A{L} = WL
    %   A{l} = A{l+1} * Upsilon{l+1} * W{l}  for l=L-1..1
    A = cell(L,1);
    A{L} = WL;
    for l = L-1:-1:1
        A{l} = A{l+1} * Upsilon{l+1} * W{l};
    end

    % Step 3: for each layer, compute P = A{l}' * E * B{l}'
    % and extract its diagonal. That diagonal, scaled by c_{l,m}, is the
    % gradient w.r.t. all M phases in layer l simultaneously.
    grads = cell(L,1);
    for l = 1:L
        % c_{l,m} = (a'(xi) + i*a(xi)) .* exp(i*xi)
        % a'(xi) via central FD on F_amp (same h as the old FD gradient)
        xi_mod   = mod(xi{l}, 2*pi);
        amp_cur  = F_amp(xi_mod);
        amp_plus = F_amp(mod(xi{l} + h, 2*pi));
        amp_minus= F_amp(mod(xi{l} - h, 2*pi));
        damp     = (amp_plus - amp_minus) / (2*h);          % a'(xi), M x 1 real
        c_lm     = (damp + 1i*amp_cur) .* exp(1i*xi{l});   % M x 1 complex

        % diag(A{l}' * E * B{l}')_m computed without forming the full matrix:
        % sum over rows of conj(A{l}) elementwise with (E*B{l}'), then transpose
        P_diag = sum(conj(A{l}) .* (E * B{l}'), 1)';   % M x 1

        % gradient: dL/d(xi_{l,m}) = 2 * Re( conj(beta) * c_{l,m} .* P_diag )
        grads{l} = 2 * real(conj(beta) * c_lm .* P_diag);
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

    % (no pool refresh needed -- analytic gradient uses no parallel pool)
end
fprintf('\nTraining done: %d iterations, %.1f s total.\n', maxIter, toc(t_start));

%% ----------------- Save final result -----------------
path = fullfile('..', 'Dataset', 'SIM_training_CST_analytic_Mx_27_L13.mat');
save(path, 'xi', 'Upsilon', 'G', 'beta', 'loss_hist', 'beta_hist', 'gap_frac_hist', ...
     'zeta_fixed', 'maxIter', 'eta0', 'seed', 'fc','M_x','N_x','L', '-v7.3');
fprintf('Saved to %s\n', path);

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
