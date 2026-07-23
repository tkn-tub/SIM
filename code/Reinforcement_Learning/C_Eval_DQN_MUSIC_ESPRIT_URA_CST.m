%[text] ## Evaluation of Localization Precision (Episodic Navigation)
% =========================================================================
% Unified Baseline Benchmark: DQN vs. Brute Force vs. MUSIC vs. ESPRIT
% Runs on the exact same randomized episodic deployments.
% =========================================================================
clc; clearvars; close all;

fprintf('=== Initializing Random Deployment Environment ===\n'); %[output:967af77d]
% Add required codebase folders to path
addingPathParentFolderByName('code');

% Load Parameters and Calibration
Parameters;  %[output:4aa3f05d] %[output:2f8af1d6] %[output:089a36fb] %[output:3214553f] %[output:927863e1] %[output:42f49735] %[output:25b0b4a0] %[output:11014d92] %[output:32fd5a73] %[output:9149da03] %[output:8fefdb79] %[output:07c38d6d] %[output:86e31bab] %[output:6228f180]
Calibration; %[output:15c951ad]

%% Optional visual verification of the calibration sector

if ~isfield(EnvPars, 'plotCalibrationSector') || ... %[output:group:5e4e48af]
        EnvPars.plotCalibrationSector

    figure; %[output:8400aad5]
    scatter( ... %[output:8400aad5]
        X_cal, ... %[output:8400aad5]
        Y_cal, ... %[output:8400aad5]
        28, ... %[output:8400aad5]
        'filled'); %[output:8400aad5]

    hold on; %[output:8400aad5]

    plot( ... %[output:8400aad5]
        EnvPars.pos_SIM(1), ... %[output:8400aad5]
        EnvPars.pos_SIM(2), ... %[output:8400aad5]
        'kp', ... %[output:8400aad5]
        'MarkerSize', 14, ... %[output:8400aad5]
        'MarkerFaceColor', 'k'); %[output:8400aad5]

    % Local-cell boundary
    rectangle( ... %[output:8400aad5]
        'Position', ... %[output:8400aad5]
        [0, 0, EnvPars.x_max, EnvPars.y_max], ... %[output:8400aad5]
        'LineWidth', 1.5); %[output:8400aad5]

    % Sector boundary lines
    boundaryAngles = sectorBoresight + ...
        [-sectorHalfWidth, sectorHalfWidth];

    boundaryLength = ...
        sqrt(EnvPars.x_max^2 + EnvPars.y_max^2);

    for kBoundary = 1:2

        xBoundary = EnvPars.pos_SIM(1) + ...
            boundaryLength*cos(boundaryAngles(kBoundary));

        yBoundary = EnvPars.pos_SIM(2) + ...
            boundaryLength*sin(boundaryAngles(kBoundary));

        plot( ... %[output:8400aad5]
            [EnvPars.pos_SIM(1), xBoundary], ... %[output:8400aad5]
            [EnvPars.pos_SIM(2), yBoundary], ... %[output:8400aad5]
            '--', ... %[output:8400aad5]
            'LineWidth', 1.2); %[output:8400aad5]
    end

    axis equal; %[output:8400aad5]
    grid on; %[output:8400aad5]

    xlim([0, EnvPars.x_max]); %[output:8400aad5]
    ylim([0, EnvPars.y_max]); %[output:8400aad5]

    xlabel('x position [m]'); %[output:8400aad5]
    ylabel('y position [m]'); %[output:8400aad5]

    title(sprintf( ... %[output:8400aad5]
        'Calibration positions: sector %d, boresight %.0f^\\circ', ... %[output:8400aad5]
        selectedSector, sectorBoresight_deg)); %[output:8400aad5]

    legend( ... %[output:8400aad5]
        'Calibration positions', ... %[output:8400aad5]
        'SIM/gNB', ... %[output:8400aad5]
        'Cell boundary', ... %[output:8400aad5]
        'Sector boundary', ... %[output:8400aad5]
        'Location', 'best'); %[output:8400aad5]
end %[output:group:5e4e48af]

% Load the trained DQN agent
%agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Tx_50_Aligned.mat')
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_ideal.mat');

if isfile(agent_path) %[output:group:4e64244d]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:4e00c158]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:4e64244d]
%%
%[text] ### DQN Agent and Brute Force
% Benchmark Configurations
N_eval = 500;              % Evaluation episodes (random positions)
K_snap = 16;               % Snapshots per point for digital estimators
Refine = true;             % 3-point parabolic peak refinement for MUSIC/DFT


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining the DFT as ideal or with CST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CST SIM1 front-end.
EnvPars.G      = EnvPars.G_CST;
EnvPars.U_func = EnvPars.U_func_CST;

% Preallocate error arrays (in cm)
err_dqn    = zeros(N_eval, 1);
err_bf     = zeros(N_eval, 1);
steps_dqn  = zeros(N_eval, 1);
err_dqn_ref    = zeros(N_eval, 1);

err_music  = nan(N_eval, 1);
err_esprit = nan(N_eval, 1);

psi_x_music = nan(N_eval,1);
psi_y_music = nan(N_eval,1);

psi_x_esprit = nan(N_eval,1);
psi_y_esprit = nan(N_eval,1);

% Define physical constants matching Parameters / EnvPars
N_x  = EnvPars.N_x; N_y = EnvPars.N_y; N = EnvPars.N;
T_x  = EnvPars.T_x; T_y = EnvPars.T_y;
posS = EnvPars.pos_SIM(:).';
hgt  = EnvPars.pos_SIM(3) - EnvPars.h_MU;
dref = d_MU_SIM_max;

%% ================================================================
% DIGITAL MUSIC SEARCH MANIFOLD
% ================================================================

% Use approximately the same angular-grid resolution as the SIM search.
N_music_x = EnvPars.T_x;
N_music_y = EnvPars.T_y;

% Periodic electrical-angle grids in the signed domain [-pi, pi)
psi_x_axis = linspace(-pi, pi, N_music_x + 1);
psi_y_axis = linspace(-pi, pi, N_music_y + 1);

psi_x_axis(end) = [];
psi_y_axis(end) = [];

[PsiX_grid, PsiY_grid] = ...
    ndgrid(psi_x_axis, psi_y_axis);

% Convert electrical angles to direction cosines.
Ux_grid = ...
    EnvPars.lambda / ...
    (2*pi*EnvPars.d_x) * PsiX_grid;

Uy_grid = ...
    EnvPars.lambda / ...
    (2*pi*EnvPars.d_y) * PsiY_grid;

% Search only physically visible directions.
if isfield(EnvPars, 'positionFoV_deg') && ...
        ~isempty(EnvPars.positionFoV_deg)

    visibleRadiusSquared = ...
        sin(deg2rad(EnvPars.positionFoV_deg))^2;

else

    visibleRadiusSquared = 1 - 1e-8;
end

visibleGrid = ...
    Ux_grid.^2 + Uy_grid.^2 <= ...
    visibleRadiusSquared;

psi_x_visible = PsiX_grid(visibleGrid);
psi_y_visible = PsiY_grid(visibleGrid);

N_visible = numel(psi_x_visible);

% Array indices following the same convention as generateChannel.m:
%
% ax = exp(+j psi_x [0,...,N_x-1])
% ay = exp(+j psi_y [0,...,N_y-1])
% a  = kron(ay, ax)
n_x_element = (0:EnvPars.N_x-1).';
n_y_element = (0:EnvPars.N_y-1).';

A_music_ideal = complex(zeros(EnvPars.N, N_visible));

for q = 1:N_visible

    ax_q = exp( ...
        1i * psi_x_visible(q) * ...
        n_x_element);

    ay_q = exp( ...
        1i * psi_y_visible(q) * ...
        n_y_element);

    A_music_ideal(:,q) = ...
        kron(ay_q, ax_q);
end

% Normalize the steering vectors.
A_music_ideal = ...
    A_music_ideal ./ ...
    max(vecnorm(A_music_ideal, 2, 1), eps);

useCSTDigitalFrontEnd = false;
%% Optional static per-element response for the digital baseline

if useCSTDigitalFrontEnd

    c_digital = c_cst(:);

    assert(numel(c_digital) == EnvPars.N, ...
        'c_cst must contain one coefficient per array element.');

    assert(all(abs(c_digital) > 1e-8), ...
        ['The CST element response contains a coefficient too close ' ...
         'to zero for ESPRIT calibration.']);

else

    c_digital = ones(EnvPars.N,1);
end

% Calibrated MUSIC knows the static per-element response.
A_music = c_digital .* A_music_ideal;

A_music = ...
    A_music ./ ...
    max(vecnorm(A_music, 2, 1), eps);

% Generate element coordinates
[xe, ye] = grid_coords_centered(N_x, N_y, d_x, d_y);   % element coords
nx_idx = EnvPars.n_x(:); ny_idx = EnvPars.n_y(:);

% Static CST gains
load t_y_x.mat;
F_amp = @(p) ones(size(p)); % Fallback if interpolation function not loaded
try
    F_amp = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);
catch
    warning('Using default unit gains for CST modeling.');
end
th_op = zeros(N,1); % Assume uncalibrated/calibrated phase spread = 0 for perfect comparison
c_cst = F_amp(th_op) .* exp(1i*th_op);

fprintf('Running %d joint evaluation episodes...\n', N_eval); %[output:9e4a146d]

for i = 1:N_eval %[output:group:4f47927c]
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval); %[output:81fdfac7]
    end
    % -----------------------------------------------------------------
    % RESET ENV & GET TRUE UE POSITION
    % -----------------------------------------------------------------
    [obs, LoggedSignals] = resetFunction_nav_CST_Aligned(EnvPars);
    target_idx = LoggedSignals.pos_idx;
    pMU = EnvPars.pos_cal(target_idx, :); % True 3D coordinate [x, y, z]
    
    %% Exact channel shared by all estimators in this episode

    h_episode = LoggedSignals.h(:);

    assert(numel(h_episode) == EnvPars.N, ...
        ['LoggedSignals.h contains %d coefficients, ' ...
        'but EnvPars.N = %d.'], ...
        numel(h_episode), ...
        EnvPars.N);

    assert(all(isfinite(h_episode)), ...
        'The generated channel contains nonfinite values.');

    % -----------------------------------------------------------------
    % DQN EVALUATION
    % -----------------------------------------------------------------
    isDone = false;
    step = 0;
    while ~isDone
        action_out = getAction(agent, {obs});
        action = action_out{1};
        if iscategorical(action) || iscell(action)
            action = double(action); 
        end
        [obs, reward, isDone, LoggedSignals] = stepFunction_nav_CST_Aligned(action, LoggedSignals, EnvPars);
        step = step + 1;
    end
    steps_dqn(i) = step;    
    

    % Get physical error
    pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x, LoggedSignals.psi_y, EnvPars, pMU);
    pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pos_MU_true);
    err_dqn(i) = norm(pos_est_dqn(1:2) - pos_MU_true(1:2))*1e2; % cm
    %Refinement
    %Refinement
    [psi_x_dqn_ref, psi_y_dqn_ref, dbg] = refineAngleParabolic3(LoggedSignals.h, ...
        LoggedSignals.best_t_x, LoggedSignals.best_t_y, EnvPars, ...
        struct('m',10,'n_avg',12));
    pos_dqn_ref = estimatePosFromAngles(psi_x_dqn_ref, psi_y_dqn_ref, EnvPars, pos_MU_true);
    err_dqn_ref(i) = norm(pos_dqn_ref(1:2) - pos_MU_true(1:2))*1e2; % cm

    % -----------------------------------------------------------------
    % BRUTE FORCE EVALUATION (Optimal Discrete Grid search)
    % -----------------------------------------------------------------
    % Retrieve the exact DOAs for this target index from Calibration
    psi_x_true = EnvPars.psi_x_cal(target_idx);
    psi_y_true = EnvPars.psi_y_cal(target_idx);
    a_psi_x   = exp(1i * psi_x_true * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y_true * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);
     
    best_power = -inf;
    best_psi_x_est = 0; best_psi_y_est = 0;

    % Exhaustive search over all phase configurations
    for t_psi = 1:EnvPars.T
        v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
        Upsilon   = diag(v0');
        G_Upsilon = EnvPars.G * Upsilon;
        r   = sqrt(db2pow(EnvPars.SNR_dB)) * G_Upsilon * a_psi_x_y;
        power_vec = abs(r).^2;
        current_peak = max(power_vec);

        if current_peak > best_power
            best_power = current_peak;
            R = flipud(fliplr(reshape(abs(r), [EnvPars.N_x, EnvPars.N_y])))';
            [~, linear_idx] = max(R, [], 'all');
            n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
            n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;

            % Map snapshot index to temporal coords
            t_psi_x_max = EnvPars.t_x(t_psi);
            t_psi_y_max = EnvPars.t_y(t_psi);

            best_psi_x_est = mod(2*pi * (n_psi_x_max + (t_psi_x_max) / EnvPars.T_x) / EnvPars.N_x, 2*pi);
            best_psi_y_est = mod(2*pi * (n_psi_y_max + (t_psi_y_max) / EnvPars.T_y) / EnvPars.N_y, 2*pi);
        end
    end    
    % pos_MU_true = estimatePosFromAngles(psi_x_true, psi_y_true, EnvPars, pos_MU_true);
    pos_est_bf = estimatePosFromAngles(best_psi_x_est, best_psi_y_est, EnvPars, pos_MU_true);
    err_bf(i) = norm(pos_est_bf(1:2) - pos_MU_true(1:2))*1e2; % cm  
%[text] ### MUSIC and ESPRIT
    %% ================================================================
    % MUSIC AND ESPRIT USING THE EXACT DQN CHANNEL
    % ================================================================

    % Static digital front-end response, if enabled.
    h_digital = c_digital .* h_episode;

    % The channel returned by generateChannel is normalized to unit
    % average element power. Therefore this uses the same controlled SNR
    % convention as the DQN receiver.
    snrLinear = db2pow(LoggedSignals.SNR_dB);

    % Unit-amplitude transmitted symbols. The channel remains fixed for all
    % K_snap snapshots in the current coherence interval.
    symbols = exp( ...
        1i * 2*pi * rand(1, K_snap));

    % The same element-space data matrix is passed to MUSIC and ESPRIT.
    digitalNoise = ...
        (randn(EnvPars.N, K_snap) + ...
        1i*randn(EnvPars.N, K_snap)) / sqrt(2);

    X_digital = ...
        sqrt(snrLinear) * ...
        h_digital * symbols + ...
        digitalNoise;

    %% ================================================================
    % 2-D MUSIC
    % ================================================================

    R_music = ...
        (X_digital * X_digital') / K_snap;

    % Enforce Hermitian symmetry against numerical roundoff.
    R_music = ...
        0.5 * (R_music + R_music');

    [eigenvectorsMusic, eigenvaluesMusic] = ...
        eig(R_music, 'vector');

    [~, musicOrder] = ...
        sort(real(eigenvaluesMusic), 'descend');

    % One transmitted MU signal:
    % first eigenvector = signal subspace;
    % remaining eigenvectors = noise subspace.
    E_noise = ...
        eigenvectorsMusic(:, musicOrder(2:end));

    musicDenominator = ...
        sum( ...
        abs(E_noise' * A_music).^2, ...
        1);

    P_music_visible = ...
        1 ./ max(musicDenominator, 1e-18);

    % Place the spectrum back onto the complete rectangular grid so that
    % your existing parabolic peak-refinement helper can be reused.
    [psi_x_music_signed, psi_y_music_signed] = ...
        pick_peak_local( ...
        P_music_visible, ...
        PsiX_grid, ...
        PsiY_grid, ...
        visibleGrid, ...
        Refine);

    psi_x_music(i) = ...
        mod(psi_x_music_signed, 2*pi);

    psi_y_music(i) = ...
        mod(psi_y_music_signed, 2*pi);

    pos_est_music = ...
        estimatePosFromAngles( ...
        psi_x_music(i), ...
        psi_y_music(i), ...
        EnvPars, ...
        pMU);

    err_music(i) = ...
        100 * norm( ...
        pos_est_music(1:2) - ...
        pMU(1:2));

    %% ================================================================
    % 2-D ESPRIT
    % ================================================================

    % Static, element-dependent CST gains break exact shift invariance.
    % A calibrated ESPRIT receiver removes them before forming its
    % covariance matrix.
    if useCSTDigitalFrontEnd

        X_esprit = ...
            X_digital ./ c_digital;

    else

        X_esprit = X_digital;
    end

    R_esprit = ...
        (X_esprit * X_esprit') / K_snap;

    R_esprit = ...
        0.5 * (R_esprit + R_esprit');

    [eigenvectorsEsprit, eigenvaluesEsprit] = ...
        eig(R_esprit, 'vector');

    [~, espritOrder] = ...
        sort(real(eigenvaluesEsprit), 'descend');

    % Principal signal-subspace vector.
    e_signal = ...
        eigenvectorsEsprit(:, espritOrder(1));

    % x-shift selection:
    % every element except the final x element of each row.
    J1x = find(EnvPars.n_x(:) < EnvPars.N_x);
    J2x = J1x + 1;

    % y-shift selection:
    % every element except the final y row.
    J1y = find(EnvPars.n_y(:) < EnvPars.N_y);
    J2y = J1y + EnvPars.N_x;

    assert(all(J2x <= EnvPars.N), ...
        'Invalid ESPRIT x-shift indices.');

    assert(all(J2y <= EnvPars.N), ...
        'Invalid ESPRIT y-shift indices.');

    % Least-squares rotational invariance factors.
    phi_x_esprit = ...
        (e_signal(J1x)' * e_signal(J2x)) / ...
        max(e_signal(J1x)' * e_signal(J1x), eps);

    phi_y_esprit = ...
        (e_signal(J1y)' * e_signal(J2y)) / ...
        max(e_signal(J1y)' * e_signal(J1y), eps);

    % generateChannel uses exp(+j psi n), so the phase of each rotational
    % factor directly gives the corresponding electrical angle.
    psi_x_esprit(i) = ...
        mod(angle(phi_x_esprit), 2*pi);

    psi_y_esprit(i) = ...
        mod(angle(phi_y_esprit), 2*pi);

    pos_est_esprit = ...
        estimatePosFromAngles( ...
        psi_x_esprit(i), ...
        psi_y_esprit(i), ...
        EnvPars, ...
        pMU);

    err_esprit(i) = ...
        100 * norm( ...
        pos_est_esprit(1:2) - ...
        pMU(1:2));
    % % -----------------------------------------------------------------
    % % GENERATE EQUIVALENT RAW SNR FIELD FOR TRADITIONAL ALGORITHMS
    % % -----------------------------------------------------------------
    % % Compute direction cosines
    % dv = pMU - posS;
    % d  = norm(dv);
    % u_true  = dv(1)/d;  
    % v_true  = dv(2)/d;
    % 
    % % Path loss & element patterns scaling
    % cosT   = hgt / d;
    % ampRel = (dref/d) * cosT^2;
    % snrLin = db2pow(EnvPars.SNR_dB) * ampRel^2;
    % 
    % % Steer vector
    % kappa  = 2*pi / EnvPars.lambda;
    % a_true = c_cst .* exp(1i * kappa * (xe*u_true + ye*v_true));
    % 
    % % Generate multi-snapshot digital covariance
    % S = exp(1i*2*pi*rand(1, K_snap)); % Random phase symbols
    % X = sqrt(snrLin) * a_true * S + (randn(N, K_snap) + 1i*randn(N, K_snap))/sqrt(2);
    % R_cov = (X*X') / K_snap;
    % 
    % % -----------------------------------------------------------------
    % % 2-D MUSIC ESTIMATION
    % % -----------------------------------------------------------------
    % % Spatial scan grid
    % u_ax = ((1:T_x) - (T_x+1)/2) * (2/T_x);
    % v_ax = ((1:T_y) - (T_y+1)/2) * (2/T_y);
    % [Ug, Vg] = ndgrid(u_ax, v_ax);
    % vis = (Ug.^2 + Vg.^2) <= 0.98;
    % A_vis = exp(1i * kappa * (xe*Ug(vis).' + ye*Vg(vis).'));
    % A_est = A_vis.*c_cst;
    % 
    % [E_noise, lam] = eig((R_cov + R_cov')/2, 'vector');
    % [~, ix]  = sort(real(lam), 'descend');
    % En = E_noise(:, ix(2:end)); % 1 Signal subspace, remaining are noise
    % 
    % P_music = 1 ./ max(sum(abs(En'*A_est).^2, 1), 1e-18);
    % [u_mus, v_mus] = pick_peak_local(P_music, Ug, Vg, vis, Refine);
    % 
    % % Map back to position
    % w_mus = sqrt(max(1 - u_mus^2 - v_mus^2, 1e-6));
    % pos_est_music = [posS(1) + u_mus*hgt/w_mus, posS(2) + v_mus*hgt/w_mus];
    % err_music(i) = norm(pos_est_music - pMU(1:2)) * 1e2;
    % 
    % % -----------------------------------------------------------------
    % % 2-D ESPRIT ESTIMATION
    % % -----------------------------------------------------------------
    % Es = E_noise(:, ix(1)); % Signal eigenvector
    % 
    % J1x = find(nx_idx <= N_x-1); J2x = J1x + 1;
    % J1y = find(ny_idx <= N_y-1); J2y = J1y + N_x;
    % dxs = (xe(J2x(1)) - xe(J1x(1)));
    % dys = (ye(J2y(1)) - ye(J1y(1)));
    % 
    % Fx = pinv(Es(J1x)) * Es(J2x);
    % Fy = pinv(Es(J1y)) * Es(J2y);
    % 
    % u_esp = angle(Fx) / (kappa*dxs);
    % v_esp = angle(Fy) / (kappa*dys);
    % 
    % w_esp = sqrt(max(1 - u_esp^2 - v_esp^2, 1e-6));
    % pos_est_esprit = [posS(1) + u_esp*hgt/w_esp, posS(2) + v_esp*hgt/w_esp];
    % err_esprit(i) = norm(pos_est_esprit - pMU(1:2)) * 1e2;
end %[output:group:4f47927c]
%%
%[text] ### Figures
% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
%plot limits
xlim_1=0;
xlim_2=150;

figure; hold on; box on; grid on; %[output:588cd30f]
h_dqn = cdfplot(err_dqn_ref);    set(h_dqn, 'LineWidth', 1.5); %[output:588cd30f]
% h_dqn = cdfplot(err_dqn);    set(h_dqn, 'LineWidth', 1.5);
h_bf  = cdfplot(err_bf);     set(h_bf, 'LineWidth', 1.5, 'LineStyle', '--'); %[output:588cd30f]
h_mus = cdfplot(err_music);  set(h_mus, 'LineWidth', 1.5); %[output:588cd30f]
h_esp = cdfplot(err_esprit); set(h_esp, 'LineWidth', 1.5); %[output:588cd30f]

%patch
confidence=90;%confidence level
[F,X]=ecdf(err_dqn_ref);
[~,idx]=min(abs(F-confidence/100)) %[output:688ce384]
x_coords = [0  X(idx) X(idx) 0];
y_coords = [confidence/100 confidence/100 1  1];
patch(x_coords, y_coords, 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); %[output:588cd30f]
plot([xlim_1 xlim_2],[confidence/100 confidence/100],'--','Color',[0.4 0.4 0.4]); %[output:588cd30f]
plot([X(idx) X(idx)],[0 1],'--','Color',[0.4 0.4 0.4]); %[output:588cd30f]
text(5,0.9,strcat('$',num2str(confidence),'\%$ Confidence'),'Interpreter','latex','FontSize',font-2); %[output:588cd30f]
text(X(idx),0.6,strcat('$',num2str(X(idx),4),'$ cm'),'Interpreter','latex','FontSize',font-2,'Rotation',90); %[output:588cd30f]
set(gca,'FontSize',font); %[output:588cd30f]

xlim([0, 200]); %[output:588cd30f]
yticks([0,0.2,0.4,0.6,0.8,1]) %[output:588cd30f]
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', 14); %[output:588cd30f]
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', 14); %[output:588cd30f]
legend({'DQN Agent', 'Brute Force', '2D MUSIC', '2D ESPRIT'}, ... %[output:588cd30f]
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12); %[output:588cd30f]
title(''); %[output:588cd30f]

% Print Summary
fprintf('\n================ RESULTS SUMMARY ================\n'); %[output:224c4b3c]
fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n'); %[output:6c8ed15d]
fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_dqn), median(err_dqn), prctile(err_dqn, 95), mean(steps_dqn)); %[output:8be17f78]
fprintf('Brute Force   | %13.2f | %11.2f | %20.2f\n', mean(err_bf), median(err_bf), prctile(err_bf, 95)); %[output:69b96b92]
fprintf('2-D MUSIC     | %13.2f | %11.2f | %20.2f\n', mean(err_music), median(err_music), prctile(err_music, 95)); %[output:4c70112c]
fprintf('2-D ESPRIT    | %13.2f | %11.2f | %20.2f\n', mean(err_esprit), median(err_esprit), prctile(err_esprit, 95)); %[output:489befcc]

% % CDF
% figure
% hold on
% h1 = cdfplot(err_dqn);
% set(h1, 'LineWidth', 1.5);
% 
% grid on
% h2 = cdfplot(err_bf);
% set(h2, 'LineWidth', 1.5,'LineStyle','--');
% 
% grid on
% 
% %patch
% x_coords = [0    30   30 0];
% y_coords = [0.95 0.95 1  1];
% patch(x_coords, y_coords, 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% 
% 
% % Formatting the plot matching your script's style
% xlim([0, 50])
% xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', 14);
% ylabel('CDF', 'Interpreter', 'latex', 'FontSize', 14);
% title('');
% legend({'DQN Agent', 'Brute Force'}, 'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 14);
% 
% set(gca, 'Box', 'on', 'TickDir', 'out', 'LineWidth', 1, 'FontSize', 12);
% 
% % Boxplot
% figure
% g1 = repmat({'DQN agent'},N_eval,1);
% g2 = repmat({'Brute force'},N_eval,1);
% g = [g1; g2];
% h3 = boxplot([err_dqn, err_bf], g)
% set(h3, 'LineWidth', 1.5);
% grid on
%%
%[text] ### Functions
% Helper peak-picking function
function [uh, vh] = pick_peak_local(Pvis, Ug, Vg, vis, refine)
    S = -inf(size(Ug));  S(vis) = 10*log10(Pvis);
    [~, ii]  = max(S(:));
    [iu, iv] = ind2sub(size(S), ii);
    du = Ug(2,1) - Ug(1,1);  dvv = Vg(1,2) - Vg(1,1);
    uh = Ug(iu, iv);  vh = Vg(iu, iv);
    if refine
        uh = uh + du  * parab_local(S, iu, iv, 1);
        vh = vh + dvv * parab_local(S, iu, iv, 2);
    end
end

function del = parab_local(S, iu, iv, dim)
    del = 0;
    if dim == 1, if iu < 2 || iu > size(S,1)-1, return; end
        Pm = S(iu-1,iv); P0 = S(iu,iv); Pp = S(iu+1,iv);
    else,        if iv < 2 || iv > size(S,2)-1, return; end
        Pm = S(iu,iv-1); P0 = S(iu,iv); Pp = S(iu,iv+1);
    end
    den = Pm - 2*P0 + Pp;
    if isfinite(den) && den < -1e-9
        del = 0.5*(Pm - Pp)/den;
        del = max(min(del, 0.5), -0.5);
    end
end

function [x, y] = grid_coords_centered(Nx, Ny, dx, dy)
    Nloc = Nx * Ny;  x = zeros(Nloc,1);  y = zeros(Nloc,1);
    for n = 1:Nloc
        iy = ceil(n/Nx);  ix = n - (iy-1)*Nx;
        x(n) = (ix - 1 - (Nx-1)/2) * dx;
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

function pos_MU_est = estimatePosFromAngles(psi_x_est, psi_y_est, EnvPars, pos_MU_true)
    % % Shift phases back to principal domain [-pi, pi]
    % psi_x_est = mod(psi_x_est + pi, 2*pi) - pi;
    % psi_y_est = mod(psi_y_est + pi, 2*pi) - pi;
    % 
    % phi_est   = atan2(-psi_y_est, -psi_x_est);
    % theta_est = asin(EnvPars.lambda/(2*pi)*sqrt(psi_x_est^2/EnvPars.d_x^2 + psi_y_est^2/EnvPars.d_x^2));
    % 
    % pos_MU_est(1) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*cos(phi_est)+EnvPars.pos_SIM(1);
    % pos_MU_est(2) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*sin(phi_est)+EnvPars.pos_SIM(2);
    % pos_MU_est(3) = pos_MU_true(3);
    % CORRECTED inverse. psi in [0,2pi) (computePsiFromPos / estimator convention).
    % Convert to a SIGNED direction cosine, then project along the true drop.
    ux = EnvPars.lambda/(2*pi*EnvPars.d_x) * wrapToSigned(psi_x_est);
    uy = EnvPars.lambda/(2*pi*EnvPars.d_y) * wrapToSigned(psi_y_est);

    s = ux^2 + uy^2;
    s_max = sin(deg2rad(72))^2;          % FoV guard (room corner ~70.5 deg)
    was_clamped = s > s_max;
    if was_clamped
        sc = sqrt(s_max/s);  ux = ux*sc;  uy = uy*sc;  s = s_max;
    end
    uz = sqrt(1 - s);

    drop = EnvPars.pos_SIM(3) - pos_MU_true(3);
    pos_MU_est(1) = EnvPars.pos_SIM(1) + drop*ux/uz;
    pos_MU_est(2) = EnvPars.pos_SIM(2) + drop*uy/uz;
    pos_MU_est(3) = pos_MU_true(3);
end

function u = wrapToSigned(psi)
    % (pi,2pi) -> (-pi,0); [0,pi] unchanged. Maps DFT phase to signed cosine.
    u = angle(exp(1i*psi));
end

function [psi_x, psi_y] = computePsiFromPos(pos_MU, EnvPars)
    % Geometry: vector from SIM to MU. SIM on the ceiling, array in the
    % horizontal (xy) plane, z-axis pointing down toward the floor.
    
    delta = pos_MU(:) - EnvPars.pos_SIM(:);   % column vector
    rng   = norm(delta);
    
    % Direction cosines along the array's x and y axes
    u_x = delta(1) / rng;
    u_y = delta(2) / rng;
    
    % Electrical angles (assuming d_y = d_x)
    psi_x = mod(2*pi * EnvPars.d_x * u_x / EnvPars.lambda, 2*pi);
    psi_y = mod(2*pi * EnvPars.d_x * u_y / EnvPars.lambda, 2*pi);
end

function addingPathParentFolderByName(targetName)
    % Recursively places target folders into the MATLAB path
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
    else
        warning('Folder named "%s" not found in any parent directory.', targetName);
    end
end


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":41.2}
%---
%[output:967af77d]
%   data: {"dataType":"text","outputData":{"text":"=== Initializing Random Deployment Environment ===\n","truncated":false}}
%---
%[output:4aa3f05d]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:2f8af1d6]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:089a36fb]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"4"}}
%---
%[output:3214553f]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:927863e1]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"14"}}
%---
%[output:42f49735]
%   data: {"dataType":"textualVariable","outputData":{"name":"M","value":"196"}}
%---
%[output:25b0b4a0]
%   data: {"dataType":"textualVariable","outputData":{"name":"L","value":"12"}}
%---
%[output:11014d92]
%   data: {"dataType":"textualVariable","outputData":{"name":"zeta","value":"0.9800"}}
%---
%[output:32fd5a73]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:9149da03]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:8fefdb79]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"60"}}
%---
%[output:07c38d6d]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"34.6606"}}
%---
%[output:86e31bab]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     x_max: 20\n                     y_max: 20\n              calSpacing_m: 0.5000\n      sectorBoresights_deg: [0 120 240]\n            selectedSector: 1\n       sectorHalfWidth_deg: 60\n                 MU_margin: 0\n                     N_cal: 400\n              channelModel: 'rician_los_nlos'\n                    fc_GHz: 28\n                    V_hall: 72000\n                    S_hall: 18000\n                   mu_lgDS: -7.2781\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 15\n                sigma_K_dB: 8\n                        KR: 181.9810\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 1\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         M: 196\n                       M_x: 14\n                       M_y: 14\n                         T: 3600\n                       T_x: 60\n                       T_y: 60\n                    SNR_dB: 34.6606\n                 theta_min: 1.7657\n                 theta_max: 4.5175\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 24\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [10 10 4]\n                    pos_MU: [78.5072 13.8642 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 … ] (1×3600 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 … ] (1×3600 double)\n                      h_MU: 1.5000\n                    L_hall: 120\n                    W_hall: 60\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 180\n                 tolerance: 0.0262\n         StopTrainingValue: 58\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9800\n"}}
%---
%[output:6228f180]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 2.366%n","truncated":false}}
%---
%[output:15c951ad]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nFull local grid: 40 x 40 = 1600 candidate positions\nSelected sector 1: boresight = 0.0 deg, angular interval = [-60.0, 60.0] deg\nCalibration positions retained in sector: 570\nCalibration complete.\n\n","truncated":false}}
%---
%[output:8400aad5]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAQAElEQVR4Aey9D5xV1ZXnu8Q\/FFAoIfwRKLAw2MYYMcZGMRg0CT6ZTzohCaFbiK+lzRuCoxN4Aa3YLYY0vPYzanyQ0dH4XjshsSF5BJWYfJ4T6VFoMQQyo0KM2kFECxH9GCnkj6atxKnvpvbl3HPPOffeU\/f\/\/dWn1tl7r7X32mv\/zql7V+2z19793tePEBACQkAICAEhIATqDIF+ph8hIASEgBAQAkKgSARUvdoIyIGp9h1Q\/0JACAgBISAEhEDRCMiBKRoyNRACQkAIVB8BWSAEmh0BOTDN\/gRo\/EJACAiBBkbgnnvusdNPP93mzJljhw4dauCRNt\/Q5MA03z3XiIVACRCQCiFQ+wi88cYbtmnTJtuyZYuNHj3apbVvtSwsFAE5MIUipXpCQAgIASEgBIRAzSAgB6ZmbkVtG7JhwwY3DctUrCd4xVjtp3JJabdjxw4799xzM1O7\/Ld0ySWXGESeOnFUKv7ixYvduIodS6n6L4ce8OUekcbpD2MfV0\/80iDA88xzDZEvVCv3MN+9LFRXueqlsbHUz5\/\/OwarIMYjRoywqVOn2uTJk23v3r0uLRcO0lt5BPpVvkv1WG8I8OEwb968HLPhIcsR1DADR6XebC4FnLz7\/9rXvmZ8cZRCX6PoqBQut956q3V2djYKbBUdB3+v\/N3GdYoD9cADD2TE4Lxw4cLMepf58+fbrl27bPXq1dba2pqpp0z9IyAHJtU9bJ5GfHDw4TB48GBbv369+yDgw+BLX\/qSA+HRRx9N\/aV4zjnn2DPPPFOxDxY+6HC6nOG9l9tvv92Nadq0ab2c+k\/8BzYpo+E\/\/s9+9rNZ7\/8rjT121BpF4VJqG3GQWDzK31CpddeKPp4zPhNIS2lTIdhxD9esWWP+82n79u1uloU1L1Ap7ZGu2kNADkzt3ZOasuiRRx5x9nznO98xvvRcoeeydOlS90ER5PsPHKZxPfHhDb+nSc4vswHBV0jBCjhGXgd1qIucDyymiKFvfetb7vUPefjUoa5vR4rTQjtS\/gsmz5cJ9ajPf3fUw1FDBqELnfAhXxcZhC74pL495ULHCmbUh6LaoBeZJ\/qgX0\/gSTsvJw3W8e1JGcusWbPcf\/8HDx60GTNmGHUZO+NCD\/q8bmTo84QOL0MXuEDgRXtfj7Kvhz70ehkper08qIe854dTdNLWE\/1ht68X7ge7wvqoTzuvA7toR70oXLxu7PVtSONwCD+Dvj0p\/eAw80Xa1tbmvmThp6HDhw+7V63YAmFfWA88ZJ4KtRkswM638xh5\/WEMqce98XL6gUfqecjhQdjly+R9HZ\/+5je\/ca+SqYsd2BPEjnrgGLYL\/uuvv25dXV129tln2\/jx490MC6+MkO3cuZNEVGsIlNAeOTAlBLPRVPEhwntj\/rsZOXJk1vCYimVKNjhzgVPDh3WwIuX7778\/yMqbZwp4yZIlmXp88V533XXGB5tnUueHP\/yhK44ZM8aOHDli1KGuY\/Ze7r777qJmiPiwvuyyy9wXfq8KQydf\/HwIex4pDhHOEHmokLFS56677qK6I8p8OIM1DD7g0UveE334D2\/qUZ92Xk5KneAXCLxiCL30gZ5gO2zBpiAP7LEBXDx\/2bJl7v6gB1lf7QNr9Hj9pPTHPeY5iOoHu3BKkFOfe3nllVe6+0cZwi708rxQDhN6i8Eh+AwOHDgwrM6V7733Xgvec8cs8kJ7bPfNuE\/+fvfFZnAAM7DzuukHjNAbhSH1Fi1aFPt3Fb532LpixQqa5dCePXuMV5vcW4TYwesfHDbK+QgHhrZEGPGZRP0JEyaQmBwYB0NDX+TANPTt7dvg+HB79dVXC1biX8cwnQzdcMMNrm2aDxLaosNPCfPBxgehU9h78XVwpNrb223jxo3udVCwHR9ufMgxvU19mvL6i1dXwRkl+NCqVavcFx6L\/ugbXb7dfffdl3mvTt2xY8e61zLUQSe8QsbKFxpteCWHc8gXBsSXBTNP6PF14Pt+yCODaEd79HhijMiCxCLGtWvXGjp8G+5TsA55dEO+DjrRTxmbsI16nsAkWIf\/gsHZy2lHe+p48vZh08aeewWR922iUnD17Ul9G2yFwvcp+Jz4e+l1cD+p\/+yzz7p7HIULOqGg\/YyDchIOPIP+C9SPgzL8oJPvZcWm3D\/sAgOeDdrz6gRnDT6EjdhKHVLK+WxmhhXMPEa0JY8+iH4gcAM\/5FDc3w910UmKHurS7uSTT4aVQzgwzEhSz9vMZ85xxx1njJN+aUQeLMGUch9ITRsIATkwDXQza2Uo\/GfIdDD\/vaexiQ9rPvxoywdW1JQwH85TpkyhShbxXyP\/QU+cONE5F1nCPAXa+hmnG2+80U1H0wRbsIkvvpdeegmWo0mTJpn\/Ap4+fbrj5bvwgQxRDweK2R7yOD44ADhc9Oe\/9NA\/e\/ZsqhhfDB4P6jErBM68Igk7GK5BERf6p\/o111yTeVXo7aOvzZs3I3YUxJ5pe6bvnaDnUir7uH9gjtPKGKHgTJC3ly9Z6iL3zxuy4L286qqreiwzdz\/5Ekz68qUtlYvFgTblJJ4BngX6YLxg453Gvtjs2wZxJk8\/yPz9DeLM3xf4UidM8P3fUBD3q6++OlzVlRkH46HALO+QIUPIioRAQQjIgSkIpuasxAcmX9LhLzDQ4IOKDzKmiylDfMHwRcIULv9R8V86\/HIQH3R84Hnd\/CfK+3MWq65YscL4r887Cr5OqVPGWWqdheib3xtV4fHl\/uDMgH8h7ftaJ4x9WF8p7OPZY7aFL06+5OiDL9ZSOGvoKgXlwyGyjyozi7UZhxSnL\/j3xD3B6Qj+7acdFq9\/4169FaKTzwAcapwmPpNog+NFWq2\/T\/oWVQYBOTCVwblue\/EzC+G1JKxr4YPMvwtnBoDpar5s+HBjwP6DhHwxxJQ2X1a04UOJnTTJJ30gUZ92OFx8+UW9\/kpqj34+rHmXjkNwyy23ZF4Xed3MNPAfKXXTEphBtMfp2rZtG1nDNv9hTH\/+y4E6vCqgkr8X5CHvKPipd3RRH1mQ+ILgiyLIC+fpH17wPvt7yhdE1GwX9ZOoUPuSdHAvcWRwiJmZ4t4wG+TtxUnlyxW5J16RBe8lr5Log2cJpxsnG3yjcPF6S4kDffeVeAb8vWW8POveGemLzb4t2Hr8fMr983aDJ44MMu84MyPo5T6lnv8bCuLO61dfp9DU60qqz98MOPjZUe5xIZ8XSTolqx8E5MDUz72qiqW8yuDDjS8O\/svnwx\/y0\/W8AuFVg\/8g4YOVLxXq8EWc1mj0owNniC98HCPsiNPnP4jpk3bYgC1R9akT95880958YdMnfaMLW9DDNDgfquT7QiyQRK+3kRQCR\/BEd1wdvsSYaaK9J+4L98c7b7SPooMHD2aikMJy+ofQgz50k1LGJmwLt4kqF2Kfr8M4yEfp8a8hscMT9417gzOFrVDwPvl6OCfo9PeSdsi4n9TnWSJPHYgxMlZmsNAJeR7tkFEuBgf0FkL0SR+MN6k+zzJ2UZdng7r+tRJ8CBuxlTqklPPZzN8UeHiMaAv5vw+wpBwk\/\/cQdqixCfJ8rxOs3377bUSpiTHjfOKgBJXg4PLcM1bGTF\/cY\/CAgnWVbzwE5MA03j0t+Yj4j5ZFdGHF8JDB54MkGDnEh+JPf\/pTt3g0bmaAdlFEW6JavIwyCy7pw\/PCqXe0PJ8PZuyj7P9T5MMNXfD4wGPNCfkg8UXtZ5I8ny9NZjnow\/PSpnyo3nHHHZnmlLHTO0bg6f\/D9ZUYC\/\/9UgcMfv7zn7sQdi8nRc\/SpUvJ5hDt\/DoihEy3v\/POO2QzRB36oK8MsyeDLdjUky3oN419UYr57x9cwjIWfHKPsBc54w7WgefvE\/WYKeT++Trcf\/8soSOMC\/VKgQN6SknXXntt1j3nvoARfTCOtDZzv8ADXNAFgRe4gR9Y8uzDQ+aJ\/pH5cjCFz33wPJ6phQsX+mJRqXeGaMTiXmZWyQeJ55M+PI+x8BoZXDxPaWMiIAemMe9ryUfFhxLTx0GCF+yIspcz7f\/Rj37URQaR54Nyfu\/aDf\/BywckCyr58OXDhjrUhb7yla9kIoooI6MvUsoQeXie+CDz\/ZP39pCnDvVp5+sgR0aZPHWgcD1sxFZkEPbThpQyRHt46KOcRL4u9f3Yg\/XRi8xTWCdY0c7LSSnDR49vT0oZIk89iLoXXHBB5CaC9EUdT7SjPeRxAUPy8OgTfUGMPM\/rIKUOfNrQFh0QeXhRFMQJHRA8Xxd96IXvKSinHvcN27w83Cfj8zJ0oZN2xeJAmyTydoT7x+nECfMziGEd3j5e1WKftxV+uG5am7kH2OV1gxf2ev3k4Xk5abB\/8kEes0rMmDAuXndhl\/8nIqyTMXnMvR3YQp663E90Q0E+siDRB3WgpHrBNsrXPwJyYOr\/HmoEQkAI1CkCODCsGeGLuk6HkGO2nzXhVQ6znrx+4nUSFb2MvEgI9BWBZnJg+opVw7dn0SbvvvnAgfhPKjho3tPDh8gHZcoLASFQPALMHEDFt6zdFjhjvGIKW8hrJWRhvspCIC0CcmDSItdg7VhMyS6nrDFgGpb\/nli74h0VnBume3kfDpGH12AwlG04fhoe3PyUedk6k2IhUGUE\/GslPks8yXmp8k3pU\/e12VgOTG3el4pbxTtn3h37DxnKrO73odCErnKeC2HEfBmzhgJexQ1Vh0JACAgBISAEehCQA9MDgn7zI4Ajw7v64OwBvKSWbBPOTI5oi9sVWDgIBz0D6Z4BcOPzJOnzRrLmQ0AOTPPd84JGzOshwomDi+6CkRLBfJzCdevWGXs3iOYIhznCQH8HfXsGrr\/++riPGvGbFAE5ME1645OG7dfDsAmWf6WUVD+fjFdPF154oYkutPPPP99YKC0sjj4P9YHHUVsrdc8KwaR72JkWpNF\/dm7m7yvIJ99XGTqC1Fd94BjURz5J558GDsv3ESN5kyIgB6ZJb3zcsHFeZs2aZax\/CUdHBF8ZBfNxujx\/5syZxlboojX2wx\/+0NjIjlR4CI+oZ4BnI+kZ+fw3\/m87dPENWfTbj3zdrl12t5VaNvTzN2f1Q7\/l6ivO\/iMfjz4I0vTT9AjIgWn6R+AYAN55YYvysPMS9cooindMm3KlQkB6hEAQgdVbXwsWM\/nrfvSclVq2Ztu+jP5gphx9JekM9q28EPAIyIHxSNRQymK1StO\/\/uu\/2jXXXGMcWPgXf\/EXFu7\/jDPOsCeffNL++Z\/\/2RF5eOF6wbKHNMhr9vy+ffsMqmUc\/H1TWpsIvPLkYVv8Tx+06b9szTLwlbfetc0vdmXxfCGtzLcPp2n1pW0X7l9lIQACcmBAoYaILzYWq3FGSyWJxbpPPfWUseV3dr9TjfK\/\/\/f\/3n3xkkJ8CZMii6OVK1c6ZFnMG1en2fif\/vSnjWMSSGt17MzA8Ry6m6dLTSFwoPPf7Io9H7ADg\/5kj1x0KMu2cUNbDMpi9hbgQ73FrAQ+lMXMU6A+FFUN+qlVMgAAEABJREFUPlRKWZQu8YSAHJgaewb44vjVr35lt912m7HpmWi1cFhdOQwWLFhgPH819mchc3oReLln9uX95\/9oT0480ss5lnRcPt7uvOKsY4xALq1s9qRTA1qOZdPqS9vuWM\/KCYFjCNS8A3PM1ObKsVKfw9BEk00YVA4Dnrvm+kurn9Ey+\/Kzr3faKWNPsiGXDcgyfMqHhhjOxsUThhj5oJByWtlds88qqb40dgTHorwQCCIgByaIhvJCQAgIgRpF4F9ue91ZdvDKE3LWurD25YmdXcaiW\/KuYu+FclrZtWueq1hfcTb2DkNJ8Qg0fAs5MA1\/i6szwA0bNhiHPnryZyolWXPo0CG34Rtt2Ujvc5\/7nBEZRZlNwJAntU+SoQd96EUP+tCb1KYaMuzDTuyNy1fDLvVZXQSYfdn+4\/027hODbPOJByONSYriSSvDIYrqLK2+tO2ibBBPCMiBaaJngAgAPpA8lWvoOAaLFi0yDn3kILft27fbpk2brBAnxtvEeUsPP\/ywjRgxwrNKlra2trp1NaXYpK9kRvUqiht3HL+3mZIGR+Dh3ldHn7x+ZM6MiB86f9\/MtvhyME0rC+oI5gvSF2zQm0\/brre5EiGQhYAcmCw4GrfAB8fn\/8tTxpSwp48t\/6U90TPtXMpRM7tx3333uZBsvnTRjcNw44032s9\/\/nM3owIPZ8bPzrAzLbMN8D1R9jMRnodTRJtLLrnE6aEvZlII+4aP48TMBXLK0OLFi416CxcutGeffdauvPJK27p1a2amB930hQ3Upy064KMPGyBk1KEusiBRDzu+9rWvuVmnoA76RkZ7iLq+LbbBg8jDRz\/9Yet1113nbGZjQQ7ahB+0jXYQ+umH9ujBDmyNksGDqEd9Ue0j8PKTh4zQ6dN6Zl9O+0RrySONiBiCikGC+lBUG\/hQKWVRusQTAnJgmuAZeOWtdw3nhTQ4XMpM6QZ5fc2\/9NJLbg+ZKVOmZKnCmfEzKnyJswMpB7QxQ8ORBatWrcqqHy5Q97zzzjPqs0vwrbfemqny4Q9\/2PFZ7IujQhgw9ZgBwlnBphUrVrg9bu6\/\/377yEc+kmmLQ4CjwB44tEE3OrxDgCNBZA6zSOyRE2dnnH1Lly51fdH+3nvvNZwwnBQw2Lt3r8GHDh48aPBd5Z7L8OHD7c4773Q2r1271oYOHdrDPfpLPfSgj7ZwfT\/kn3\/+eeMcK8aP\/diWrz\/aNRg1xHB4dcTaFxbu\/sV3x7oxlTrSqCMheolFt67T0CWpTTlkoe5VFAIOATkwDobGvuCoQFGjhM8rpShZWt4pp5xiI0eOjG3Oq5uNGzdmXg8VsqPv2LFj7Utf+pLTedVVV9kLL7xgb775piv79sz0EHY+f\/58x8cGbHGFmMvrr79uJ598cpZuQtlxemhCvxMnTjR0s28LvCiiXti+3bt3G07K1Vdf7drjYOEEbd68OUsFur\/3ve8ZTl6WIKaAzehBH23Rv23bNjcrRROcMF69jR8\/3jlA8IJEm2L6C7ZVvrII4Lgw+8KrI98zkUZEB\/lZDtKOy9szUUillKELog\/6Jy1XXzhLUWMjioq+RUIgjIAcmDAiDVje\/OL+xFG98tY7ifJihQcOHDC+ZOPaMbvBaw9eZUDB2ZS4NmPGjLGBAwdGir0Dg5CZBnRCfMG\/8sorsGMJO99+++2MPOz0JPWbadSTyanXwzt8+LC9+uqrPbmjvzgOo0ePdgWcOBwinCNs5ZWaExRwCZ9Dhc04Yb5pEA\/P60t\/XofSyiKwZttr9rm7nrLbv\/J7++vfPe8ijIIWjP1AS6Y4bmh2WHWpZaXWh+GF6zw2TtqJhIBHQA6MR6KB0\/CHW3ioUz70gTArdZn\/+tva2iw8y8CrGpwWUl7j0AGvP3htc8MNN1BMJByBI0eObt4Vdjp8Q3QvW7bMeLWCXl6djBs3zosj0\/CXP7pxwCIrJzCj7Bs0aJDh2PhmOG7MyPgyM0XYCQ4scsb58rKkNOygYHPQCYtrm7a\/OH3ilw+Bh587ZAvW7sws2GWmlLVrrFljxpS8X7BbThn9QJXoK25sjLd8SEtzPSMgB6ae71687VkSpmbHDY3+LwY+07ZZDfpQYJaBVxp33313Zk0HX9ysK2H2gVcbQfU4HayHCfKi8p2dnW69CDKOO+A1ybBhwyjG0gMPPGD5ZmBwYPjypy6KVq1aZThgOGKUC6Uo+9rb240xs6gZDHCoWJPC+iBmXIILaQcPHpz42i1oBzajB33oRT94hLENtulLf0E9ylcGARyYqJ5Ys1bqAxuTdMY5D0ltyiGLwkI8ISAHpkmegZ\/+h\/NyRorzEsXPqVgkg9cV3\/nOd2zGjBkuKofXJHyR33777U4Ta0WYsYBPhM28efPcWhFeubgKERdeB7GYldctzGIEF6366nyBs4AXfdRDH2tFmKHg9RNOAlFIv\/3tb30Ttw6HxbI4XLRhLQkLfnHEMpUKyNAPZz+hI2ift5OxYhe4sNYFO6hHfWQsUIYf7ApHBecKjN56662MiHroQR9tEfh+yEdRIf1FtROvOgj8j1ffjeyY2RY\/GxKuUA5ZuA9fLkdfSTp9v0qFQBCB8jgwwR6UrwkEcFbeuuNTxoI8FuGRPn3TRQa\/HAbixPB6xJN3XugLR2Pjxo0ucoiUww1ZfMsXNilt+ZL2UUuU4bPwFH3kcTAg8sjRC\/nXJNQjUsfLfd1nnnnGLrjggqx9YOgLPm2wB\/vQhV7a05YyuoPjgOcJ54iZJHQE29CWMnwInbQJ89ENH1v8uLEDeyBCsz2feuhBH4R+9MHHPq8LHjLq+jz1IV+HNqLaQuBf179ji\/\/pg\/aJ7dnrWrCSv1eIfJjgQ2E+ZfgQ+TDBh8L8pDL1oag68KFSyqJ0iScE5MA02TPA66SOy8e7iIUmG7qGKwRqHgEfNj1oVD97cmLu4nr+dhVGXfO3sU8GqnHhCMiBKRyrhqn5\/e9\/v2HGUgsDYYaDmQ5mOWrBHtlQvwhs\/9F+w4mZvKTFVs6akJkhZUajo0yh0vxTwzo4ZmXpB\/RI6Q8eRDnIT2pTatmUDw2ha5EQyEFADkwOJI3NYG+Sv\/mbv2nsQWp0QqAOEcBx+ZfbX7cRHz\/eEUMoPNQ4+3VT+nYtdOtoXCA0u9T66KBwncdsop1ICHgE5MB4JJokxYFhqJqFAQWREKgdBDLnHS0eaUQhKYzajIW9cZFQtXPnZEm1EJADUy3kq9Tvt7\/9bdfzxo0bXVqOC6HRLDolwgYiD4++SDnTh+3wCQFmbxjO7aGM3BNhv7Ql9Tz2SQmGHnt+MEUe7A8ZuumTviHk6PYU1T\/tRJVHoFl7zDrvaEqrc2CisChHiHKSzjjnIalNOWRRWIgnBOTANNEzwOzL448\/7kZcrhkYHATCfglnJtoFIg8Pmes8dGEb\/uDGdzg2Tz31lNuPJViV\/V\/Cm7gF5T7Pnix+XxfPC6d+szvs4xykW265xR36GK6nshAoNwLu1dFtrxvHBvgjAxRGXW7Upb8REJAD0wh3MWIMOCtsxhakT33qU1k1gzLyyPvq2LDnyvvvv29s1uY7Y98X8uw4Sxom+sVhwXFB5s8hOuussyg6QsaBh14vMzPMoDB7wunLzLy4ij0XDodkUS0zLz3FvL\/oRPeRI0d3+s3bQBWEQAkRePnJw8Z5RxP\/6gPOiUH16JNPIMmhcUNbMgt7w8JyyMJ9+HI5+krS6ftVKgSCCMiBCaLRQHl2gX3ssceMFGfGU3CInkcK\/7\/+1\/9qc+fOJZuacITYyZZN7HjlgyK\/nwnROpTDdOqpp7qN7LzjwmwMG7uxt4qv62XoxzHBQVm\/fr09+uij9vzzz\/tqLqUtr6bYVdcx8lzo78wzz3Sb2uWpKrEQKDkC\/rRpP\/tCB0unRe8y3XH5eFMYNQiJhICZHJgaegpKbUp7e7vhxHzrW9+KVU0d5DgI5GMrFigglBjnglc07BTLLAnknZkoNWxghwPB7A0zLczGMCsSrIuTgUODfvJsRsembzhHvKIK1iXPrA+76sb1G7SNDegKOY8Jvc1C3d3dJqoMBl\/bMsGgIN7njjzBbv9Ce2a2hRmZRZ9ps1nnDbPJ7a1ZIdbllK2cdUbF+oobm8Kom+VTp\/hxyoEpHrO6a7F06VJjdiXK8KuuusqQR8n6wmO2hfUlEDMlf\/\/3f585GylK7\/Tp0401LjhSbLGPUxOsxwnM1IFHnjSJcGyWLFlinBPEkQLhujhY2AZxlMCXv\/zlRPvC7Ru9\/NprrxlricpNe\/bssX379lWkr3KPJY3+\/\/KL5+zfrdxmI274F\/vY8l8aZY\/JgQNdNjwQHT34uHczOO3fXzlZJfsCw3B\/Q0\/qbvQ\/N40vJQIBByalBjWrCwTiXqe8\/PLLJbWfGY\/gehSU89qH10rMsFCOIs70wXH5yU9+YszG4ID4eiz+RUYdeIUs5KUe5yeRxjlvyKBC7KNeM9Hw4cNt1KhRZScc1Ur1VYnxFNPHY51mSze8aX7B7t63u11556H+tuXNAbbssa4c2YuHWyyuXTlkyx\/vcjaFbSxHX3E6CSlvpr89jbVwBOTAFI5V3dbcvXu3+egjXhPNnTs3M5a+LtrNKOrN4GSEX91wajKnJ\/Nl1VstJ8Fh4cDHH\/7wh+ZnWnwlv\/iXAxnh8Xpp69atbsYE52bNmjWwc4jXTTfeeKPRP4ci5lToZTDrw3+9Sfb1Vm2apKWlxSpBAwYMsP79+1ekr0qMp5g+HtzRFfk83fTIPvvFrvciZYsf2m1x7cohq2RfSfZHglFLTNlSFQTkwFQF9sp2igNDjzgvrIlhRoIv7UsvvRS2eefGFfp4wRFZu3atLVu2zFj7AnGo4v3332+sWUlSz8wKIdU4QcF6vDKaOnWq4ZDARw+LdFkoTHj2pZdeCjuSqEuYdFgYXAODnptvvjmvfWEdKguBNAgQNn3\/F1+0zS9GOzBs3uZnPML6kSW1K7Us3L8v14od3h6lzYmAHJgmuO+8Pgov1PXODHzkpYQBJ4aN8lhfAnHSM44EfSDjVGXKOCQs+GW9DLL58+cb7ahD2Z+sDB+C54kyun\/+85\/b7373O8P5QebbkPdEXa8X3eRpGyRvg2+jVAiUCwEfNn32rv6RXbAoF4oSxoQau6rlkDnFEZdy9JWkM8IEsYSAopCa4Rk47bTTbOnSpZFDhc9C3khhjTJZZ8PMDsRsDa+ecFJq1FyZJQQyCDD78rOvd7r9Xm65\/SMZfjDzjU+3mcKog4goLwSiEdAMTDQuDcXFSUkaUNIrmKR21ZIxWxKcPWHWpVq2qF8zEwgFI8CeL1T+i++22cUThthds8\/KhEozA9FxebvNuWCUnT+mJSt82ctmTzo1tl05ZNgH0b\/1\/JBiYzn6itOpMOoe4PUbiUC\/SK6YQkAICAEhUBIEOE\/oc3c9ZeO\/ttG2\/3i\/DfnfBthpn2jN6C78VOZATHVP60q2q2RfPUOz7P5aYImEQA4CcmByIBGjzk4d7JoAABAASURBVBCQuUKgZhHAebl2zXNuwe70X7bagUF\/stsG7rUndnZZUMYAWBhL3Sd27neHOS5Yu9O1y5YltSu9DHsgvzj4mI2l7ysOE3ACA5EQCCMgByaMiMolQSC4ToW1KpxdlE8xu\/ASXURbjgv43Oc+Z4RJB9vF8YN1ypVnDNiHneXqQ3obC4HVW19zA2LB7tjXT7TOke85JybpxGbCieP2PklqVw5ZnPNQjr6SdDoQdRECIQTkwIQAKbqoBjkI4IAQOs0OvKxVYR+XTZs2GQ5ATmUxhEADI8DMxSmH+9kntg90jssjFx1yo2UmA5krhC7IFEYdAkVFIRCBgByYCFDESo8AsxNs38\/eK4RKo4lwaTaUI+TZz6jg5DAzA6WZ1bjrrrvcPjPnnnuu29COfqAovdhEH8iogw3M7jCbA5FHji2Qr0dd8vDohzOa4EFeJzKI9vDQx5410CWXXGLXXXddluOGEwehQ9T4CLDota1n5gUn5smJx047hw9FIUAINRQlow1UKVlUP\/CwASIfJvhQmE8ZPkQ+TPChMF\/lxkWgryOTA9NXBCvQ\/k9H9lihFGVOoW2pl6Z9sA0b5LGrLbvlBvk4M+z\/wj4sfMlzNhIzNMzOUC9fpBR1PL3yyivG+UbM7uAo3XLLLeadB2Z+OOeoGL3ow+FAH4dA4oCF9XGA5MGDB70Jhr2Eb9OGnX5fffVVt+MvFQ4cOGBsjMd+M1\/4wheM2Sf0QThBYWxoI2pMBP6vKR+yf9e79uXZ0\/+QGWTH5ePtzivOypSDGYVRB9FQXgjEIyAHJh6bmpG888JK69rwyYIoyuhC21IvTfs\/dP4kq9kpp5xiSdvy4wy0tbUZZxAxO3P11VcbZx3hlGQpiimg3+9dg8OBY4HjxFlLZ599tk2ePNnt2otejjV48803YzQdZaPPOxXBYwyi9B1tYUboNkSZIw7GjBlD1hH6\/PjZpwbmkSNHDBvJM25SUeMj8NlPjbDxPxthv\/1stxssMwwdl7dbXMgwMoVRZ4eWly+M2t0SXeoYATkwdXDzTho70wadd1tBFDWcQttSL037Ez84OasZMxB8+WcxQwVmLfhy5\/UL2\/ozg\/HOO++EakUXgw5CsAZHDgTLOBEnn3xykBWZL1RfsDGzSLxWwn7GwXi8PKiPGSdmapgRwnE777zznHPl6yptEgQ+fHxmoOOGDsjkyWSHDNemrLo2Koya50SUi4AcmFxMao5z4rDJ1n\/slwuiKOMLbUu9NO37DWzLNBs\/frwxu8KXdYbZk2HdCetESHuKxswJr1888bpl2LBhiPJS0EHCUaJMI3+cAHkIWdQhjvB9G+rFUVifr8dMEa+teH2F\/TgnzPp4eThlVofzofT6KIxM45eJ4okLQ46XKYzaL3BmQTM4Nf6TohGmQUAOTBrU1CYWgdbWVuPVzd13351ZXMvaj4ULFxozEcxI8Lpm69atGfnixYsN5wbHIFZxQIDz4R0kUhwmHCdmXJ599lm3FoU+WcsyadIk847RI4884rTQBh2ukHBhZoWZIWZXvL6o6sihKBk89Dz\/\/PPGqy7shCdqDgR8GHV4tEkhw4sf2u32gQm3oZzUrhyyOOehHH0l6WTsIiEQRkAOTBgRlfuMAFv9f+c73zFOefavWHBe\/JoRFvSyyNXLWaeyYsUKGzRoUEF9jxs3zl577TUXhbRmzRqjLY4TeumXV1I4DShjsS0yoqAeffRR14a26ECeRDhbS5YsMa\/v\/PPPt8GDBzs7cdJuvfVWpw\/HiBml8Cssrxs9OFIsFMYWz69uqt7LhQDnHd3158879X4mwRUCF2YWkmQKow6ApawQiEFADkwMMGL3DQGcGF6vePLOi9calPP6iC95vtxXr15tyHBGfNSSb0Pq+d\/+9rcN3b4tMoi28CF0oRM+7Z555hnXhrbohgeRp3\/q0T7YjjK6ICKcvve977k1LEE+Y4M4UDKsD528NnvhhReMmSfKosZGYPuP9htOzMtPHrJxQ1siBwsfihISQg1FyWgDVUoW1Q88bIDIhwk+FOZThg+RDxN8KMxXWQjEISAHJg4Z8cuKQLMoZ7HvZZddZp\/97GcN56ZZxt2s48Rx+ZfbX7dxnxjkzjuKC5XuUBh1ziOShElOZTGEQA8CcmB6QNCvECgXAjgtzPwwO1OuPqS3dhB4+OuddsrYk+yT1490RsWdOK0w6uxQ6Y6E0PIpHxrisNRFCIQRaFIHJgyDykJACAiBviGw+gd77JUnD9vmEw\/ajP++wx3W6DVWNwy576HZ1bU\/+jWcx1Zp8yIgB6Z5733ZRs6aD7bRZwEvRB5e2g55DcN2\/9XWkdZ+jg4gyopIprQ61K62Efj5Y28Yr444bfqXE48Yi3QJn\/bkF+x6ftzJy9TXadRHT+\/mjoNXXCQUclEVEKihLuXA1NDNaARTcDJmzZpls2fPdgtmWfxKHh6yRhijxiAEwgg88bM3bezrJxrnHeHEeHncl29SyPBihVF7+JQKgUQE5MAkwiNhsQiwSdz777+fFXFDiDF62PCNlBkVv4tteHYmKGPWYvfu3e5ARPZ38U5QsE6wPQcvEqqMbtpGzXjoEEjugKiUCLBwd\/D93e606eB5R0l9MLPgZ2XC9ZApjDqMSqasjBDIICAHJgOFMqVAgI3a2FiOPV5wKNA5YsQII9yZ0GNmYa677jpjvxZmZ9gfZeHChe4wRmR\/+7d\/a\/fff7+bvWHvGPZYufPOO40zjtjNFn2097vgBtsjY4M62gdDoeFDHNrIZnn0S3t208XJwSEiRFqHQIKSqFgE\/uW2112TRy465NJCLoQLQ1F1CaGGomS0gSoli+oHHjZA5MMEHwrzKcOHyIcJPhTmqywE4hCQAxOHTA3x2U9i+4\/3WyEUZXYh7XydNO35D9S3Y98VnAecATaAYw0M5J0ZPwvjN5rjUEZ2qOWwQ2RsFIcThD6\/twp5T8zwnHzyye4oAni05\/Rrf1DikCFDYg+S5Iwi6tOOWSH6pR06cZA4DgD72aSOzfWa+RDI7u5uE0VgEIELfzt\/nPC+dY58j0cri2adF308xqLPtNmKL5+RVdcXFlwyym66dIgvZqVJ7cohS2N\/OezIAkEFIdCLgByYXiBqOdnxo\/32s693FkRR4yi0LfXStH\/5ycM5zZhtYaYDWr9+vf393\/995uiAzs5Ow1nAsWGmhpkRnIi4nWyDyqkXPN+I4wNwTHydMWPG2MCBA30xK6Ue9bOYPYVwv9TBSeoRJf4Wqi+ohNkeXnExdpy44BEEQX3MWjEDhVPH0QeVPgSS3Yq5T+UmnM99+\/ZZufspp\/7ZWwbblfefbOePyY6WofzNi1sj+Ref2m2n9T8YKfvkqG4bc3yXfezUE4KPjqub1K4csjT2l9qOLBBUEAIBBOTABMCo1Sx7SnzlwdOtEIoaQyHtfJ007U\/7xLEjAJhp4WyjoB5mVHithPMBH+eFL2acG4h9UtgvJe7wRNp4CjsX6OS1kZcnpdSjPnVIKZMP94ss6CRRB4Lv21COo7A+X4\/XV7y24vUV4wYDsPDycBpxCGS4StnKw4cPt1GjRpWduJ+V6quc43ms0yy8boXy8se7IvkvHm6xuDY7D\/W3LW8OsKf3dWfdX\/QltSuHbHkK+0ttRxYIKgiBAAJyYAJg1GqWjbFO+0Sr29kzXxo1hnxtgvI07bHPt2NWgdcvODKexywDi3D5skLuD0hEToixX4iLnP\/Iea2DDEcIIu+JOjgXDzzwgGOtWrXKcI5wkhwj4YLzwWwGVUh9O3RiH3ayJkaHQJq1tLRUhAYMGGD9+\/evSF\/lHNODO7p4rHIojr\/4od0WJ7vpkX32i125r6NQntSuHLI4G8vRV5JOxi4SAmEE5MCEEVE5HoECJLz6YLHtsmXLjNckEAtkWVjLLAtyFuXCQ3b33XcbZfjIb775ZuO1ErK9e\/fa0qVL3ZoWnBaikDCB+rSjDs7SihUr3PlEyJJo3LhxxqsR2q1Zs8Z8O\/plUTFrdnCw0EG\/rIe58cYb7dFHH3VjoS06kCcRY1myZIl5feeff76xtmfQoEHG+ppbb73V6WOBMmtxwq+wvG70sEiZyCps8XyltYdAXERRnKVEGsW1QcZsS1RbZEntSi2LsgFerdiBLaLmRUAOTPPe+7KNnC\/ejRs3ukgiXpX4V0S+QxwGeFGy4NoZFgPzxe31oZN8sL3noZu2vg3lINGGQxu\/rUMgg7AonwIBFq3\/w8jtWS2LjZ6hPpSlpLdABBLUW8xKaANlMXsL8KHeYlYCH8pi9hbgQ73FghLqQ1GV4UOllEXpEk8I1JMDo7slBJoGARb76hDI2rzdPmz65SePhU3HHdo4e9KpkYPo0GGOObgkYZJTWQwh0IOAHJgeEPQrBGoNAWaMmKXSIZC1dWeYfSFseuJffcCtSfPWxR3aeNfsswzyMxKkHQkHFyKbc8EoF3G0ctYEo771\/JAiwyGK66scMmyH6L\/HDGdPpe2Y8qHokHLsERWKQGPWkwNTo\/eVxayiPSYMKotBjf451IxZD3+90+24+x\/\/sMs+tvyXWQc2YmTcoYdx\/KQ2tSSrrv3Z4engIhICICAHBhRqiIiMufDCC42t8Fm8KZpqwqByGPDc8fzxHNbQn0VNmMLMC6dNs2Ed5x2xkPXo4YtdzpEh7xfRehk8KMxvtsMcPR7J447GMe48qZp4KGREVRGQA1NV+HM754vjtttuMxaj1jvNnDnTDZC03sdSKvsJ0b7jjjvsBz\/4Qc3eY54\/d+N0ySDAqyPWvuC4hI8MSDqYMe7LN6kN4cQPP3dsfU3GiJ5MUrtyyNLYXw47eoauXyGQg4AcmBxIqs\/AiWGDs3onxgGapPU+llLaz068pdRXal3cL+5b41DfR\/Lyk4cNJ+bJiUdylDG74GdYcoQxjKQ2yBRGHQOc2EIggIAcmAAYygoBISAEwgjguHDMBq+Onj39D2GxW9jqF7nmCGMY1IeixIRQQ1Ey2kCVkkX1Aw8bIPJhgg+F+ZThQ+TDBB8K81UWAnEIyIGJQ0Z8IVAiBKSmvhHg1REj+OTikSQ51JEQEk1kUE6DHkZSm298us2WTos+BDKpXTlkaewvhx09kOlXCOQgIAcmBxIxhIAQEALHEGDxLmHTc\/66reiQaEKQIT+zQNqhMOqicFQY9bFnUblsBOTAZOPRgCUNSQgIgWIQYOHq5+56yoZ+4zEXKj3+ZyPsk9cfm31JE1Kcpg0210q76trRAhQiIZCDgByYHEjEEAJCoFkRwHmJCnve8YcjsaHSSaHB6IL8Il8W6FJOavPEzv1GFNKCtTutuHbRYcj5+4tvR1uomnZwT5r1edS4kxEouwOT3L2kQkAICIHaQWD11tcijUkbGhz35ZukT2HUkbdATCGQg4AcmBxIxBACQqBZEfAzDeHxM3OSRhbW48v59CmM2iPV1KkGnwcBOTB5AJJYCAiB5kGARbanHO5ni\/\/pg1mDhg9lMXtKWLu5AAAQAElEQVQL8KHeYkEJ9aGoyoRQQ1Ey2kCVkkX1Aw8bIPJhgg+F+ZThQ+TDBB8K81UWAnEIyIGJQ0Z8ISAEmg4BTpWe\/stWd97R2NdPzIy\/IyFUOkmWJgy5ZsKoe0afxv4kPNLKekzRrxDIQUAOTA4kYggBIdCsCIx94wTDcWHTOogZgY48Yc98yV88YUhkaDAh1BB6wJQ0n745F4yy88e02MpZE9wmedbzU0i7JDvSyrAdov8eM5w9+exP21dcO4VRg7woCgE5MFGoiCcEhEC1Eah4\/+y4y6Z1nHd08MoTMv2PGzogkyeTJqQ4TZu0fZWjXXXtVxg191SUi4AcmFxMxBECQqAJEeC8o1eePGycd+QX7LLYljDi5LDnUochK4w6iH9cJFcTPqIacggBOTAhQFQ0O3TokM2ZM8c2bNgQCceOHTts6tSpRhpZoRGYGkNTIcDsC+cdMfsSdd5RUthzkizuyzepzeKHdrt9YKJuQFK7csjS2F8OO6KwEE8IyIHRM5CFAM7LvHnzbMuWLVl8X0B+yy232IEDBzxLqRCoewR4dcQgHrnoEEkOMRPjZwXCwiRZuK4vJ7VBpjBqj5RSIRCPQL94kSRVRKAqXTOjMmXKFNf32LFjXRq+4Njs2bPHTjnllLBIZSFQlwgw+8J5R+M+MciOO+v4yDGMG9riFrBGCZNkUfXhJbUhhBqiXpiS2pVDFu7fl8vRV5JO369SIRBEQA5MEI0mz7e2ttpDDz1kK1asiETijTfesPvuu89uvvnmSHkcE4cH6u7uNpEwqLVn4OGvd9qgUf3sov\/zg7biy2dEPsaLPtOWSjbrvOhTpZP0LbhklN106ZCS2pHUX5Isjf1J+tLKIsEQs+kRiHZgmh6W5gRg\/PjxBsWN\/oEHHnBrX0aOPHawXVzdIH\/dunWu3fLly62zs7OpCUdu3759TY1B8BmoBTym3NHPRnz8eOs3dr+d1v+gLZ02zPwMCOm8C4bYxad2p5J98+LWovV9clS3jTm+y5Z8akjJ7Kik\/Wn7imtHSHnw80R5IeAR6OczSoVAEgK8XnrqqafsyiuvTKoWKZs5c6atXr3a5s6da6NGjWpqwvkbPnx4U2MQfAaqhcdjnWbzHthn5\/\/n3fa5VXvsnbktmXvCTOSowSeY\/\/mz0UP6JCtWn8dk0KBWK6Ud4F6sLWnblLIdr5asjn5kauUQkANTOazruqdVq1bZrFmzjA\/AYgfS1tZmkydPtvb2dmtpaWlqGjBggPXv37+pMQg+A9XA48EdXbb4od3mF8rufbvblX+9510rtWzRgy853cX0Rd1f7HrPvvmzPRWxMWncaexP0pdGxj0x\/QiBCATkwESAIlY2Aqx92bZtmxGddPrpp9uMGTOMqX\/SuFDrbA0qCYHaQaDUJ04nhQ2nCUPGuXr4uehoqKS+jspeiwQ6rSyN\/Wn7SmoXOSgxmx4BOTBN\/wjkB2DEiBG2ceNG27Vrl6P169cbsyqk06ZNy69ANYRADSGQJhz6lbfetTTt4oadTx+zMFFt87VLY2OSzigb4CW1KYeMPkVCIIyAHJgwIioLASGQGoF6aPjRkwba4n\/6YI6prLWAcgQ9DPhQTzbnFz6UI0hgUB+KqsLCYShKRhuoUrKofuBhA0Q+TPChMJ8yfIh8mOBDYb7KQiAOATkwccg0Md\/PuMTNrpxzzjm2adMmI21imDT0OkVg3gvDjB13Tzmc\/fHXcfl4u\/OKsyJHlVY2e9KpRev7xqfbXORSVMO0dqRtl8b+tH0ltYvCQjwhkP0XLDyEQF0jIOOFQDICbFjHeUcTr\/iAnTL2JFeZ\/\/o7Lm83vqwvnjDE7pp9VmbTur7K0AWhh85I8\/U154JRRujwylkTSmZH2rFhO4Td1vNDms\/+tH3FtZvyoSE9PetXCOQiIAcmFxNxhIAQaEAE2HGXIwNwXPp9sb9V8oTlSvbFrSt1f6XWV5yNOo0avES5CMiBycUkNUcNhYAQqF0Etv9ov+HE7LnkT3btmucyi3JZdEr5iZ1dRtQNeb8Ytq8ydEHF6dvvDnNcsHZnRWxMGje2Q8XZX1ocuSe1+1TJsmoiIAemmuirbyEgBCqCAI7Lv9z+unHe0W9O\/0Nkn0lhvGllcV++SfoURh15e8RsbARSjU4OTCrY1EgICIF6QsC\/Ovrk9SMzsxph+5lt8TMNpZKF9fhyvr4URu2RUioE4hGQAxOPjSRCQAg0AAIvP3nIWLx72icG2WmfaM0sjA0PjQWqUJhPGT5EPkzwoTA\/qUx9KKoOIdRQlIw2UKVkUf3AwwaIfJjgQ2E+ZfgQ+TDBh8L8ipTVSV0iIAemLm+bjBYCQqAQBNyro9tedxFHzL7QptSh0h0J4ddE1tBnmJLaKIw6jJbKQiAaATkw0bjUFZet\/i+55BJjm\/9iiDZ1NVAZKwSKRICIIxc2\/VfHwqZLHSqNkxKnkxBkyM8skHbkhmy7UXmZwqizw9gVRu0eD10iEJADEwFKvbLuvfdet9W\/3\/I\/KaVuvY5TdguBQhBgAe3n7nrKbv\/K7+0\/\/mGXizAKtqtkaHAl+2KMpe6v1PqKs1Fh1OAlykVADkwuJuIIgeZCoAFHi\/NS7fBf+vfkFwezeBdeUujyEzsVRh3Ei3vZgI+ohlQCBOTAlADEaqsY0XvYot\/6f8eOHXbuuedGvlLitRGvnKjLAY3Vtl39C4FyILB6a2lPZU4Ke06SxX35JrVZ\/NButw9MFC5J7cohS2N\/OeyIwkI8ISAHpsGegUOHDtktt9xil112WeTrJJwWHJ4aGrZMEQIlR8D\/Bx9WzAxIJWXh\/n05nx0Ko\/ZIKRUC8QjIgYnHpi4lR44csVdffdWmT59el\/bLaCHQVwQIm+a06em\/bM1RxUJZKEfQw4AP9WRzfuFDOYIeBnyoJ1vwL\/WhqAaEUENRMtpAlZJF9QMPGyDyYYIPhfmU4UPkwwQfCvNVFgJxCMiBiUOmTvkDBw60MWPG1Kn1MlsI9A2BYNj0IxcdylHWkRDyXA4ZEUo5RvQwkvpSGHUPQPoVAgUgIAemAJDqqUpra6tdffXVtmzZMmOtSz3ZLluFQF8RePnJw0bYNHu+hMNvKeNQEPJMPtgX5XLICKFGdzF9+TDqC0\/LnkFCTzlsTNKZxv4kfWlkQeyUry0Eqm2NHJhq34Ey9D9y5Ejr6uqyyZMn5yzk9Yt4y9CtVAqBqiLA7MvPvt7pNq179vQ\/5BwZwNoXH\/1DPmgs5XLIiDhCd3F9HY1C+tXL2TNI6CmHjUk609l\/9DBH7C1u3NHtgjqUFwJBBOTABNFogLxfxHvNNdekXsSLjjlz5tiGDRsyiDCbg\/PjN8pDTr1MBWWEQJUR4LwjTPiL77aZopBA4hiljQyq7SikY+NTrjkRkAPTYPfdL+KdMGFCqpHhlMybN8+2bNmSaQ9v4cKFNnv2bOcUbd++3cmWLl3qUl2EQLURYPaF847G9Z53FP7v39uXL\/qn1O18v+E0nx2KQgojprIQyEWgXy5LnHpGoC+LeNk\/ZsqUKW74Y8eOdSkX1tWsXr3a5s+fT9EoT5061fbu3Ws4N46pixCoIgIP9746Yu0LZsRFs8CHqBMm+FCYTxk+RN6TT+FDvlxISn0oqi4RSFCUjDZQpWRR\/cDDBoh8mOBDYT5l+BD5MMGHwnyVhUAcAnJg4pCpUz7OxY033mgrV64sehEvbR966CFbsWJFnY5eZjcjAi8\/ecgt3D2td\/YFDCp5YGNSRBGLVrEnTEltFIUURktlIRCNgByYaFzqlstaleuuu86effbZohfxjh8\/3qB8g6ePNWvWGLMwOD356u\/Zs8eg7u5uE5UKA+nxz9LPvr7HBo3qZ9PvGJV5via3t9rKWRPM\/0fPjMaiz7TZrPOGWSVlK2edUbQdf\/nx4XbuyBPs9i+016X9pcaY6Kt8nzGSNycC\/Zpz2I07anbZZbfduIMckVEnLQK8Mlq4cKHba+bKK68sSM26deucs7N8+XLr7OxsasKR27dvX1NjEHwG+orHf\/nFc\/bQ7P32k\/a37WPLf2mUvf79+7ts+IBjj+jg497N4F5JWbF9eUwOHKhP+0uN\/9CTuo\/dROWEQAABOTABMOo1y4wIEULBqKF8Y6EubfLVC8pxXljgC+\/ee++1QmZfqDtz5kxjDc3cuXNt1KhRTU2EuA8fPrypMQg+A33B47FOs6Ub3jQWvBI2vfftbld+8XCLBWXW81Mt2fLHu5xN2NhjhhVix85D\/W3LmwNs2WNdbmyFtivHuNPYX2o7Hn4uO5wcPERCAATkwICCKC8C3nkZPXq0c0YKdV5Q3NbW5l5ntbe3W0tLS1PTgAEDrH\/\/\/k2NQfAZ6AseD+7osqifxQ\/ttlqRpbHjpkf22S92vRc1NKv02NLYXw4bI8EQs+kRqFMHpunvWyQAzI74fVrypdSNVBLDXLp0qZP41BV0EQJVRKDUIc\/5QpvT9BcHT76+\/IxNuH2+dmlsTNIZ7t+Xk9qUQ+b7VSoEggjIgQmiUad51rSwtiVu3UscnzaFDJlXVNu2bXN7w0ycODGzuy+voJAVokN1hECpEfALdMN64UNhPmX4EPkwwYfCfMrwIfJhgg+F+Ull6kNRdVhwDEXJaANVShbVDzxsgMiHCT4U5lOGD5EPE3wozFe5xAg0kDo5MA10M0s1FO8QTZs2zan05bAjhAOEzFXSRQhUCAE2rPuHkdvt+iOjI3tMClGutGz2pFOLtlFh1JGQiSkEchCQA5MDiRhCQAjUKgLsuMuRAaeMPcm+8g+nGYcN+v\/aSTsubzecBg5srAUZNkDYBqak+Wz0hzmuDISBF9KuHOPGdoj+C7U\/jx1W7L1RGDXIi6IQkAMThYp4QkAI1AwCnMfzubuesqHfeMxuXPSc4cRw3pE3cOwHWnzWxg0NxE33cGtFVit29EBiaWxJ0yZtX7ntjt1fZCIh4BGQA+ORUCoEhEDNIYDz4k9EPuVwP2vbeJx1jnzPOkd0W1CG4Swepe4TO4+eakzeL2rNyCoswwaoODv2G6HDC9butOLalX7c2A5V0w7uM\/dXJATCCMiBCSOishAQAjWDwOqtr2Vsmf7LVjsw6E\/25Dnv2HU\/es6CskylnkwtyeK+fJNsJAwZB6ZnKDm\/Se3KIUtjfznsyAFCDCHQg4AcmB4Q9CsEKoSAuikSgc0vHt3rZezrJxrkZl96ZmCYUfGysMpakoVt8+V8NiqM2iOlVAjEIyAHJh6bupXs2LHDzj333Ey4c3BPGIU+1+1tbUrDxw1tMV4dfWLHADf78suJRxwO8CFXCF3gQyG2K8KHXCF0gQ+F2K4IH3KF0AU+FGInFqkPRVUihBqKktEGqpQsqh942ACRDxN8KMynDB8iHyb4UJivshCIQ0AOTBwydcpnx9xbbrnFLrvsMguHPe\/atcsU+lynN7ZJzb7zirOsrXf25dnT33VODFB0XD7ekJEPUy3JiMgJ20c5yUaFUYOQSAjkBtSzMwAAEABJREFUR0AOTH6M6qrGkSNH7NVXX7Xp06fXld0yVghEIXBO\/4F2xZ6hznF5cuI7xn\/oHTUWKo2TEhcaTAgyhN2MjzSf\/QqjPsvdZ4+XwqhBQhSFQCUdmKj+xSsxAgMHDrQxY8aUWKvUCYHqIMB+L0PvHmK\/\/Wx3xoBxNRoqjYFx4cZx\/KQ2tSSrrv0Ko+ZZEOUiIAcmF5O65nDI4tVXX23Lli0zbfNf17dSxvcgQBSMC+M98WBPyYzFr5RrKVQ6yRZshfyC48LsVxh1EC+eAXfzm\/qiwUchIAcmCpU65uG04Lx0dnba5MmTcxbyahFvHd\/cJjS9HkKlk8KG4758k9oojLoJH3QNORUCcmBSwVa7jUaMGOEW6kYt4IWnRby1e++a1bKVK1fGDt3\/Jx6uwExGPcjCdvtyPvtrLYza2+3TfPaX+t74fpUKgSACcmCCaCgvBIRARRHYs2eP4cCQ+o45KuD+L77oiuOGRq9\/gA+5SqELfCjEdkX4kCuELvChENsV4UOuELrAh0LsxCL1oahKhFBDUTLaQJWSRfUDDxsg8mGCD4X5lOFD5MMEHwrzVRYCcQjIgYlDps7599xzT87rI3h1PiyZ32AI3H\/\/\/W5Ejz\/+uEu5vPzkYXulhzh1+mioNNxs6lAYdTYgPaVyYEKEVY\/qnN9y9JWkM8cAMYRADwJyYHpAaLRfHJU1a9bYli1bMnvBkIeHrNHGq\/HULwK80sR6nzL78rOvdxrRRxP\/6gNFn1zcUWMh1oRQQ35mgTSfjQqjVhg1fxOi\/AjIgcmPUV3VYBEvjsqSJUuM9TDeePLwkFHH85XmR0A1yoPA7t27zc+8fP\/737db1m6xW+a\/4Dp75KJDFlwAW90w3gHOJn9JY0uaNvRXK+2qa0f0a0TwETU3AnJgmvv+a\/RCoGoIPP7441l9\/+eVP7WTf\/2+\/eb0P9jmEw8a4cee\/KJQFo\/CSwpdriUZtkLF2a8w6iBeQUc264FRoekRkANT849AcQYy0zJ79uycfWCYdSG8Ghl1krRyHMGcOXNsw4YNWdV4\/eTPVSKfJVRBCBSJwKpVq7JafPbZCW7HXX\/eEcK4L6+kMORakqWxf\/FDu+3h5w4x\/Byq9NjS2F8OG3OAEEMI9CAgB6YHhEb7nT9\/vuGoBPeBIQ8PWdJ4cV7mzZvn1s8E63FA5OrVq239+vWOyMML1lFeCAQR4LXQ+PHjLY6CMzAfb\/m8jT\/pz+3F3\/\/MXvnRFXbgH+dk0cG137B\/++1\/y6hnJsb\/l55h9mZqSdZrUk6Sz0aFUedAJoYQyEGgXw4nxFCxPhHAUWHflyDBSxoNDsmUKVNclbFjx7rUXzZv3mxtbW3uy+icc86xCy64wOB5uVIhEEZg7ty59thjj1l7e7ux3iVMvv4Hjh9tnx50je3\/415b9\/YS+9Pb+7KIeoMuv8FO+sjlZB2xGBZyhdAFPhRiuyJ8yBVCF\/hQiO2K8CFXCF3gQyF2YpH6UFQlQqihKBltoErJovqBhw0Q+TDBh8J8yvAh8mGCD4X5KguBOAT6xQnEbz4EOIbgoYceshUrVuQMfufOnTZ69GijjhfC8\/mklD0+oO7ubhM1FwY4vY8++qj93d\/9XewjMv7EPzecmP9++O6cOn\/9HxbZKV9dbf1OPjVLtugzbbbiy2dk8XyhlmSzzhvmzcpKk2xccMkou+nSIVn1fSGpXTlkaewvhx1+\/HWWytwyIyAHpswAV0I961s4IoA1Kz7v16qEU+pRJ8ouP9UfJYM3YcIEEkfBvGMkXNatW2dTp0615cuXG0ccNDPhyO3bt6\/pcPjqV79qvHaMekw+3jLDzb78z3d\/mhHj+CxYsMCWLr7Wlk4bZn5GgnTeBUPs4lO77bT+B2te9s2LW4u28ZOjum3M8V225FNDqj7uNPaX+t6cP0ZRSJk\/DGWyEJADkwVHfRZYlLtx40abNm2aC50mH3x1FMwjo34lRzpz5kz35TW355XCqFGjrJlp5MiRNnz48KbE4Itf\/KL97ne\/s0svvdSCP9fe9i278b5rMixmW1q\/fIdN\/NIChxOzfqMGn5CR\/9noIY7Pc1QPsmJt9M\/IoEGt1qdx29GfvuJVrP3cF6hU7fRa6eh91DUXATkwuZjUNYfZFWZZmI0JDwQeMuqEZYWUg6+Mgvl8bflvmkXE7e3t1tLS0tQ0YMAA69+\/f9NiEDVzd93KN90jdOeCo69bTmg7114\/bpgtfmi3LXrwJZf6Ra173+525V\/vedce3NHl8rUsS2M\/4\/nFrvfsmz\/bY+QBp1rjTmN\/qe8N9xkMREIgjIAcmDAiKkciEPXFE8WLbCxmMyIQOebdgc3rfIXNv3nXoNmfbrUpH23JijaqlTDetKHBaezHcVMYtX86lAqBeATkwMRjU1eSe3rPPmKmgzUmhEKH17\/AmzRpknvNVOzgiE7aunWrEakEkYdXrB7Vb24EcGCiEPCzMB2zjy5eDYZMR9XPF4ZcKyHWUbbDy2e\/n3mhbpDytSv1uIN9B\/O1YkfQJuWbDwE5MA1yzwmRZq0LZx4RAn3vvfdmzkGC7+n2229PNWJCp9ncbsaMGQaRh5dKWSUaqY+aRODb3\/52xi5eKXZ0dLjn9KSpHbbmvx9yMzDMwrzX+XSmXlSGdRFQrcui7IOH7RD5MLFQGQrzKdMGIh8m+FCYTxk+RD5M8KEwP6lMfSiqDnyolLIoXeIJATkwDfYMsECXhbos6E07tDgd3knCGSKfVr\/aNScCzL74zetwXtgjZunSpQ6M\/2fZQvu7Q991edbC+BmYWjkNuSPl6ddp7P\/Gp9tc5JIDI3RJa0fadmnsT9tXUrsQDCoKAYeAHBgHQ8kvFVXIolwW57JI1+fDr498mXrUqaiB6kwI9CCAA9OT2Le+9S176aWXDCeGMnTxhCF299c+ZXe+ca2NG3GCsR5m5qBn7K7ZZzny\/9GTdlzebnyx0gY5PHSQ1poM+yBsK9TGOReMMkKHV86aYMW0Kwcm2A5V044pHxpi+hECUQjIgYlCpc54wRkTn2eWJIqYnaFOnQ1R5jYAApx9hPPiZ13CQ+IL+Ob\/Y7Gd8MHJxixM955nMlWqexqyTqOuLv7aBybzh9AUmcIHKQemcKxUUwgIgT4gkOS8BNUOOHOBKy7+\/CEjiufaNc+ZX5zK4lHKT+zsqgsZtkLF2b\/fHea4YO3Oqo8b26Hi7C\/tveEZcA+ELkIghIAcmBAgjVDkFRGvisKvlOAha4Qxagz1h0DwlVGS9ScOm2xDP\/+Snf3FtbZ662uRVdOGNVe6XdyXb5IdCqOOvOUVY6qj+kFADkz93KuCLb311luNcGkW8j7wwAOuHdFJs2fPNmSOoYsQqAME\/H\/+YVOZiakHWdhuX85nv8KoPVJKhUA8AnJg4rGpSwkzLNu2bbPp06fboUOHbNOmTc6ZYd0LG88ho05dDk5GNx0CfvFoeODwoTCfMnyIfJjgQ2E+ZfgQ+TDBh8J8yvAh8oUS9aHc+ubOP1IYdRQy4gmBbATkwGTj0VClI0eO2Kuvvmo4Lg01MA2maRC484qzIsfakTKsudLtWJgcNYAkOxRGHYWYeEIgFwE5MLmY1DVn4MCBNmbMGOOsou3bt1tXV5f5HXMfeeSRzGxMXQ9SxpcNgVpTTKj0zkuvtq+3r8+YRlgtjgEy8hlBT4ZyLckIQcamHtMyv5STbPRh1Bee1pppQyZfuySdaWVp7E\/bV1w7xi4SAlEIyIGJQqWOeZwAe+ONN9rdd99tHB1w2WWXGTvmLl682Pbu3WtxIax1PGSZ3sAIsAj2V11nOgdmTMvRQx9Z++KjkMgHh0+5lmTBCB5vZ34bj0Yh\/erlQ76JS\/O3Oxr9Qz3XoPdCOS0m6ewvrR29w1AiBHIQkAOTA0n9M3BYnnnmGbdFuz86gHT16tWGg1O7I5RlQiAbAaKQvrt7hmMGZ2GSonhqSYYD5owPXZJsVBRSCCwVhUAMAnJgYoARWwgIgeojwOzBr7o+bOv2TbGZp262C4c874zKF8VDO1cxdKl0u1D3mWI+OxSFlIFKGSEQi4AcmAA0jZS95557zB8f4FN4jTRGjaXxEfCROh3Pf9UN1s\/CwIccM3SBD4XYrggfcoXQBT4UYrsifMgVQhf4UIidWKQ+FFWJCCQoSkYbqFKyqH7gYQNEPkzwoTCfMnyIfJjgQ2G+ykIgDgE5MHHI1DEfR2XNmjXG3i\/+OAHy8JDV8dBkepMhEIxC4lXShUNesC\/1zMQkRfHUkoyFqVG3LMlGRSFFISZejSNQFfPkwFQF9vJ1yh4vOCpLliwx9n7xPZGHh4w6nq9UCNQyAkQaEQnDf+Y4MHveHWa3fvgfdZjj7LMqctAj2EPgz3NC2lHhwzSJvqJvkRAIIyAHJoyIykJACNQUAsxiPH3TRfbWHZ+yMy\/qcLa988IKl3Kp7kGDhR30WA82JmFZXfsLOMwR40VNh4AcmAa75cy0cGTAsmXLLDjTQh4eMuo02LA1nCZBoP\/YL9sJH5xs77yw0u7f+GsLhvmyMJayDxkm7xfzVkuGDVBxdhwNo9Zhjmbct7hIriZ55DXMBATkwCSAU6+i+fPnG47K5MmTMwt5ycNDlnZcrJ\/xC4LnzJnjjipIq0vthEBaBPxp1T\/+zfGRKpJClMskM8K9o4yJ+\/JNskNh1FFIiicEchGQA5OLSUNwcFT8Al6fwks7uA0bNhjrZ1gMzA6\/6NGmeKAgqjQCJw6bbEM\/\/5L5WY1w\/\/zXXiuysG2+nM9GhVF7pJQKgXgE5MDEYyNJAAGOJuCIAo4qYDO8qVOnup19OTAyUE3ZaiDQpH2yoDRq6PChWpBF2QAP+yDyYSKEGgrzKdMGIh8m+FCYTxk+RD5M8KEwP6lMfSiqDnyolLIoXeIJATkwDfoMBF\/3+Nc+zKKkHS4HQnIwJAdE4rRs2rTJcGJwZtLqVDsh0BcEgiHWQT0dNXTQIwuQg7b5fJKNCqP2KCkVAskIyIFJxqcupTgv\/nWPf33Eqx8W8SJLM6hp06bZnXfeaZytNHHiRLv66qut95VUXnV79uwxqLu720TCoFTPwOT2Vls5a0ImnJhZi0WfabNZ5w2zWpGtnHVG0Tb+5ceH27kjT7Dbv9Be9bGlsb\/U+CuMOu9HbNNW6Ne0I2\/QgRNthPPCni\/BaCPy8JBRp9jh4\/hcd9119uijj7ozlh555BHjgMhC9Kxbt87N1ixfvtw6OzubmnDk9u3b19QYBJ+BvuJx8and9uBXTrXffvHH9vBVbTbnIydksN2\/v8uGB6KcBx\/3blVkxdrhMTlwoD7t9\/e32HHHtRt6UnchHzOq04QI1L8D04Q3rdJD9q+MiGLCEaL\/q666yrZu3Wo7duygmEgzZ8601atX29y5c23UqFFNTSNHjrThw4c3NQbBZ6AUeHxw79\/aSfv\/mw078eUMro91mi3d8Kb5xbB73+525RcPt1glZcsf73L9FmPHzkP9bcubA2zZY+rqYlMAABAASURBVF11aX+pMX74uUOmHyEQhYAcmChU6piHg8FMy6JFi7KcC2ZdeIUUdEIqNcy2tjYjjLu9vd1aWlqamgYMGGD9+\/dvagyCz0Ap8Gg97zbj5\/2Xv5fB9cEdXbByiBDlSsrS9HXTI\/vsF7vey7EdRj3YXw4bGbuo9AjUu0Y5MPV+B0P2e0fl4MGDNmPGjKx9YJiivfXWWzO8Sy65JNQ6ushCXRbsrlmzJrM53qpVqwzHZPz48dGNxBUCFUKg38A2Y4O77t9vsffe3OJ6VRi1gyFzyRe2HYdXRkEok1Zf2nah7lUUAg4BOTAOhsa5MAOzceNGt07FL+CNS6lX6MhZsDtp0iQ3k0JU0969e+3ee++11tbWQlWonhAoGwKDemZhcGQOP32966PUYbzog5zy0AU+FGInFqkPRVViMTIUJaMNVClZVD\/wsAEiHyb4UJhPGT5E3iz7Ch\/K5qokBOIRkAMTj40kIQRuv\/32jGPEmhY5LyGAVKwqAuzQ+6cje+ydF1ZYrYRYz550aiQmCqPOhSUJk9za4ggBMzkwegqEgBBoCAR4jeTPSfrE6EOW7xRl\/98+acfl7YazcfGEISVthw0QfQAyab6+5lwwys4f05IVfl1Iu1qxv9R2KIza9BODgByYGGDEFgJCoP4QYBYGq995YSWJjf1Ai0u5jBsaiKnuYVRSVsm+eoZW8nFX1\/5j95CxiYSAR0AOjEdCqRCoKALqrBwIcE4SszB\/6PyJ\/eD\/fzBzXhKLR69d85w9sbPLOGCRvF+4Wk4Z\/UDF9bXfCB1esHZnndpfWoy5X+V4VqSz\/hGQA1P\/9zBrBEQhEV107rnnZoVRZ1VSQQg0MAI+rPpXXR\/OGWXSKdDlkMV9+Sb1RRgyDkyO8T2MpHblkKWxvxx29Axdv0IgBwE5MDmQ1DfDRyGx5b8Po54zZ46xGV1wZMoLgUZFgGikCY\/fFzk8Zlv8bEi4Qjlk4T58OV9ffuM7X9+n+dqVemy+33BaK3aE7VK5uRCQA9Og99tHDHEGEocwcn4R4c8cCdCgQ9awhEAGgXFDo9dNwIcyFQMZ+FCAlcnChzKMQAY+FGDlzVIfiqpICDUUJaMNVClZVD\/wsAEiHyb4UJhPGT5EPkzwoTBfZSEQh0CVHJg4c8QvNQJ+Rmb79u1uDxe\/kR2vmXjdVOr+pE8I1AICCqPOvgsdl49PFVpORFG2pqOltPrStjvaq65CIBsBOTDZeDRUiddGvD5i5oUZGHbT9ZvasSndwoULG2q8GowQ8AiEw6EvGn3IOsoUKs2XfLg\/ZhLojxBqiDK2kcJPaqMw6rMyp3CDl8KoeXICpGwGATkwGSgaJ7N48WJ3XABOC6+PeI2E48Juun6U06dPN2S+rFQINBoCOAnb5h9nOy+92v6\/z2zqcWCyj72oZGhwJfviPpa6v1LrK87G6NeB6BA1NwJyYBrs\/vNaaNu2bW6bf5wWjgvgNVJ4mNOmTTNkYb7KQqCRECCsmg3uCKv25yQRWVN8aHO60GD6gfziWha\/Uk4O567pMGrLb386rOIw4X410jOpsZQOATkwpcOyJjThrOCY4KDUhEEyQghUGQHOScIEv7nd6q2vUcyhcoT\/xn35JvWlMOqcWyOGEIhEQA5MJCxiCgEhUJMIpDSKHXq7f7\/FmInxsyFhVcyOlFoW7sOX8\/WlMGqPlFIhEI+AHJh4bCQRAkKgQRAYcOZCY3+Yw09dn1kgGh4aC0ahMJ8yfIh8mOBDYX5SmfpQVB1CqKEoGW2gSsmi+oGHDRD5MMGHwnzK8CHyYYIPhfkqC4E4BOTAxCEjvhDIRUCcOkZg0Mduc9bfdt5\/c2n40pEy1DipHQuJw\/1QTmrzjU+32dJpw6iWQ0ntyiFLY3857MgBQgwh0IOAHJgeEPQrBIRA4yPAgl7OSTr\/j6vsuzMGZ2Zi+K+\/o0wh1oRQQ\/QBwqT5+lIYtcKoeVZE+RHol7+KatQMAjJECAiBPiHAWhgUfP6E71glQ4Mr2RfjK3V\/pdZXnI0KowYvUS4CcmByMRFHCAiBBkWAWRjCqlnQCzFMFtSWKzQYvZBfHFxYXwqjDuIVF8nFvRM1NwLFODDNjZRGbxs2bHAb5LGzr44i0ANRrwhc8eQcZ\/qvuj7sUn9JCm1OK4v78k3SpzBqf0eUCoFkBOTAJOMjaS8CO3bssEWLFmU2yJs9e7ZxFAHHFfRWUSIE6gIB\/rufEHFaNbMjyKIGkVYWpQtePn0KowalRiKNpRwIyIEpB6oNqHPz5s122WWXmd8gj2MJVq9eba2trQ04Wg2pkRFgIW3U+OBDpZRF6YJHPxD5MBFCDYX5lGkDkQ8TfCjMpwwfIh8m+FCYn1SmPhRVBz5USlmULvGEgBwYPQN5EWCWZdOmTTZhwoS8daMq7Nmzx6Du7m4TCYNqPwMrvnxG1GNqiz7TZqWWzTovOhw6qa8Fl4yymy4dUlIbk\/pLkqWxP0lfWlkkGGI2PQJyYJr+ESgcgMGDBxtrX4pdA7Nu3TqbOnWqLV++3Do7O5uacOT27dvX1BgEn4Fq4HFa\/4NunxU\/y0E674IhdvGp3VZq2Tcvbi26r0+O6rYxx3fZkk8NMWzjL5S0XDYmjTuN\/Un60sjOH6MoJJ4BUS4CcmByMREnBoF7773X1q5daxwSOWnSpILXwMycOdN43TR37lwbNWpUU9PIkSNt+PDhTYZB\/D2vFh7zPn2G\/fqbk2zvP1xkmz7+13bDqH\/I3BNei44afIL5nz8bPaRPsmL1eUwGDWq1UtrB316xtqRtU8p2ca+jTD9Nj4AcmKZ\/BAoHgIW7HBZJi6uuusq9FnrppZcoJlJbW5tNnjzZ2tvbraWlpalpwIAB1r9\/\/6bGIPgM1AIebm+YA7+24w89bQ\/u6LLFD+02v4h279vdrvzrPe+mki168CXXvhh91P3Frvfsmz\/bUzI7Kml\/2r7i2nFPTD9CIAIBOTARoIiVjQD\/tY0ePTqb2VM65ZRTjP8We7I1\/SvjhEASAplzkp6+3lZvfS2yalLYc5JMYdTZcCZhlSTL1qKSEDiKgByYozjomgeB6dOn25o1a+yNN95wNVetWmVnnnmm+RkZx9RFCNQpAszC\/OnIHuOYgagh5At7jgu\/jtIFL58+ZmGoF6Z87eLsSNsu3L8vp9WXtp3vV6kQCCIgByaIRlnyjaGU8OklS5a4V0Es4t27d68tXbq0MQanUTQ9AuzOyzlJX29fb2Na3szBg3UYUI6ghwEf6skW\/Et9KKoBC3ahKBltoErJovqBhw0Q+TDBh8J8yvAh8mGCD4X5KguBOATkwMQhI34OAjgxLOCFWJTLq6WcSmIIgTpFgFkYTMeJIQ1SR8qTqtOc5qzTqIPIK9\/UCOQZvByYPABJLASEQHMgwDlJzMLMPHWzXTjk+cygp3xoiOGIXDxhiJHPCHoylJNkd80+q+g2cy4YZYQOX3haqwV\/8vWVZEdaWRr70\/YV1y6IgfJCIIiAHJggGsoLASHQ1Aj8tHuRG39wFoZ1JU\/s7DIW5JJ3FXovlJNkwYMce5tYvjZP7Dx6mOOvXj7km7g0f7t0Npbe\/tLa4QZfnYt6rXEE5MDU+A2SeUJACFQOgR\/\/5nj77u4ZPTMwL\/TQsVmYpAiZJBlOT5T1SW0WP7TbHn4u23nxOpLalUOWxv5y2OHHr1QIBBGQAxNEQ3khIASaGgFmOXBg9rw7zIKnVZc6eiafPheFFHEn8rXD\/ohmlrZdlC54afWlbUefIiEQRkAOTBgRlYWAEGhaBHwUzKVbbs3CAD6UxewtwId6iwUl1IeiKhOBBEXJaANVShbVDzxsgMiHCT4U5lOGD5EPE3wozFdZCMQhIAcmDhnxhUBzI9CUo7\/zirMix60opGxY0uKRtl127yoJgaMIyIE5ioOuQkAICAEj0ojIGz8TQNpxeXsmCqlYGfUh9AAvaT59Pgpp5awJRv1C2\/konmL7S2qHLqiadhB9BQYiIRBGQA5MGBGVawMBWSEEqoQAX+hP33SRvXXHp4y04\/Lx9t6bWzLWjP1ASyY\/buiATJ5MnCyOn9SmlmTVtf8Y3mAiEgIeATkwHgmlQkAICIEIBN766Xh754WVLow6GBbNglTKPgyZvF9E62XwoDA\/qY0Po16wdqcLucYkry+53dHw5eL7i2+HLqg4++P1pbE\/LhIKXETNjYAcmOj7L64QEAJCwCHADr3dv99i\/\/zY\/+vK4UtS2HDcl29SG4VRhxFWWQhEIyAHJhoXcYWAEBACDgF\/WvWtH\/5HVw5fmB3xMxRhWVw5qQ0yhVHHISd+7SNQOQvlwFQOa\/UkBIRAnSIw6GO3Ocv\/U4QTM25oS2axratUwCWpDSHUUJSapHblkEXZAK8cfSXppE+REAgjIAcmjIjKQkAICIEQAsFzksKnVbPINy78mgXBIVWumNRGhzk6iFJf1LB5EJAD0zz3WiMVAkKgDwi0nnd0Fsa\/SmLGoCNPiDUhyBB16Zo0XxuFUZ+VmdECL4VR8+SIohCQAxOFinhCQAgIgRAC\/Qa2Wf+xX7YLh7xgr\/\/tgEyIdbDa2A8cC\/kdFwixrm4YcmGh3owjzs44flKb0smOYYpOkRDwCMiB8UgoFQJCQAjkQWBQ7yzM4aevz6pJtFFUuDE8yC\/yZYEu5eRw4v3uMEeFUZs7wwlss8BWQQj0IiAHphcIJYUjsGPHDps6daqRFt5KNSuBgPooPwI4MX86ssf+0PmTTGert76WyQczcV++CqMOonQ0n4TJ0Rq6CoFsBOTAZOOhUh4EDh06ZLfccosdOHAgT02JhUBjIsBrpKGff8m9TvIj9DMsvpwvZSYmrg0yhVHnQ1ByIWAmB0ZPQVEIbNmyxfbs2WOnnHJKRDuxhEBzIjBuaHHrNKgPRaFFCDUUJaMNVClZVD\/wsAEiHyb4UJhPGT5EPkzwoTBfZSEQh4AcmDhkxM9B4I033rD77rvPbr755hyZGEKgmRFQGPWxu99x+XiLwyOt7Jh25YTAMQQayoE5NizlyoHAAw884Na+jBw5sij1zNhA3d3dJhIGjfgMTG5vtZWzJmTCf5lBWfSZth7eGT2Uy5913jCLa\/OXHx9u5448wW7\/QnuOvqR25ZCtnFW8\/aW2Q2HURX3cNlXlfk01Wg02NQIs2H3qqafsyiuvLFrHunXrnOOzfPly6+zsbGrCkdu3b19TYxB8BhoFj9eef9j27++y4YGI5cHHvevucxwfHKJkHpMDB6L1xbWDD0XphA+lkaVpk7avqHZDT+ou+jNHDYpGoC4byIGpy9tWeaNXrVpls2bNstbW1qI7nzlzpq1evdrmzp1ro0aNampi9mr48OFNjUHwGWgEPD7w4l\/bgD232tINb5pffLv37W5XXv54l0vD\/BcPt9hjnRYp23mov215c4Ate6wrR19Su3LI0thfajsefu6Q6UcIRCEgByYKFfGyEGDty7Zt22zevHl2+umn24wZM9xCXtINGzaifvyGAAAQAElEQVRk1Y0qtLW12eTJk629vd1aWlqamgYMGGD9+\/dvagyCz0Aj4MFp1f3+bZ99vX29hX8e3NEVZrkyJ07HyW56ZJ\/9Ytd7rl74ktSuHLI4G0veV89Ak3T2iPUrBHIQkAOTA4kYYQRGjBhhGzdutF27djlav3694ZSQTps2LVxdZSHQVAgQVv2rrjOdAxM+JykOiFfeetcURp2NThIm2TVVEgJHEZADcxQHXYWAEGheBPo88ofevsLpiJqFcYLQhXBhKMR2RRYAQ64QutAGCrFdET7kCqELfCjEdkX4kCsUeKE+FFUdPlRKWZQu8YSAHBg9A0UjcM4559imTZuMtOjGaiAEGhCB\/336F4xZmJmnbrYLhzyfGeHsSadm8sFMUjjxNz7dZkunDQtWz+ST2pVDlsb+ctiRAUAZIRBAQA5MAAxlhUBVEFCndY\/AxROG2KDzbnfjYBaGGYiOy9vtrtlnOaKMkBQ+jgFtkMMLyuZcMMrOH9OSFX5NnXztknSmlWEfRP\/W80NaaTsURt0DvH4jEZADEwmLmEJACAiB4hCYes5HjQW9Fw55wbbNP846Lh+fUVDd05wDsd09FqWxJU2bnq6sNO1aUCUSAjkIyIHJgaTpGBqwEBACJUJgwJkLrd\/ANjv45GynkcMcr13zXGbBLgtVKT+xs8viZfuN0OEFa3cW2S5JZzoZtkJ+wXFh9qfrKw4TcHJg6iIEQgjIgQkBoqIQEAJCoC8IMAtD+3deWGFxp1QnnbxMODEODDrClNSuHLI456EcfSXpDOOgshAAgeo7MFghEgJCQAg0CAL9x37Zhn7+pZ7XSQszMyjhoTGT4Wc1omR+47soWVK7UsvC\/ftyPvsrZYe3R2lzIiAHpjnvu0YtBIRABRBg0WtUN\/ChKBkh1FCUjDZQpWRR\/cDDBoh8mOBDYT5l+BD5MMGHwnyV4xFodokcmGZ\/AjR+ISAEyobAnVecFambBb5xMoVRR0ImphDIQUAOTA4kYggBISAESoNAXKh0Ulhz\/YRRDyk6RDxp3HEyhVGX5llsRC1yYBrxrmpMQkAI1AwCl792kX3p1M0Ze8YN7XtYM8pKE6JcmC2V7Ct3bAqjBhNRLgJyYHIxEUcICIE6QKAeTCSKhx16rxtxlzOXxa+EJceFDB+VKYzaLwIGLzB04OkiBEIIyIEJAaKiEBACQqBUCBBG\/d3dM5y6\/\/Thf3Qpl6SQYYVRg5BICORHQA5MfoxUQwhEICCWEMiPADMJv+r6cM45ScwsIIvSgExh1FHIiCcEshGQA5ONh0pCQAgIgZIhMG7o0fUbNzz\/VaeTc5LIwIfIh4kQaijMp0wbiHyY4ENhPmX4EPkwwYfC\/KQy9aGoOvChUsqidIknBOTA1OkzILOFgBCofQR8qPSr7w6zdfumGOckXTjkeXdOkpeFR6Ew6jAiKguBaATkwETjIq4QEAJCoM8IBMOoO3pnYf7Th++zuJDhjsvbTWHUZ5mfwSFVGHWfH8OGVZDSgWlYPDQwISAEhEBJEcBZefqmi+ytOz5lg867zdpa3rQ\/dP4k00d1Q5QVRp25EcrUHQJyYOrulslgISAE6hUBzkk64YOT7fBT1+s06he73G1k0fLR8PHoU6wbLozajVqXUiAgB6YUKDaBjjfeeMMuueQSO\/300x3NmTPHDh061AQj1xCFQGkRGHDmAqfw0FOLXRq+KIw6jIjKQiAaATkw0biIG0AAR2XhwoU2e\/Zs27Vrl23fvt1Jly5d6lJdhIAQKByBE4dNdqdV+zUx4ZbMSJQxjDrcndFfXEh3TuVeRlKbcsh6u1UiBLIQkAOTBYcKUQi0trba6tWrbf78+U5MeerUqbZ3717NwjhEdBECxSPAAtWoVoRQQ1Ey2kCVkkX1Aw8bIPJhgg+F+ZThQ+TDBB8K81UWAnEIyIGJQ0b8kiGwZ88eg7q7u03U5BjoGcj8Daz48hmRf2MLLhllN106JFK26DNtFteuHLJZ5w2rCTsijRCz6RGQA9P0j0DxALAeZs2aNcYsDLMx+TSsW7fO1V2+fLl1dnY2NeHI7du3r6kxCD4DzYzHaf0P2tJpw8zPtpDOu2CIfXJUt405vss+duoJWX9a549psYtP7TbakQ8KKZdD9s2LWw3dlegrzv5g38oLgSACcmCCaCifFwG\/HmbMmDF25ZVX5q1PhZkzZ7pXUHPnzrVRo0ZVk6re98iRI2348OFVt6NW7kOz4zHv02fYr785yV7+6v9w6dIvnGVgsuXNAfb0vm4L\/rAu5sXDLfZYpxn5SsiWP95Vsb7ixhYcp\/JCIIiAHJggGsonIoDzMm\/ePFfn3nvvtUJmX6jc1tZmkydPtvb2dmtpaWlqGjBggPXv37+pMQg+A8Kjxf7tf\/yNvffiXXbSn950zwWY\/GLXexb1s\/ih3fbgjqPhx2F5OWSV7CvJ\/vBYVRYCICAHBhQqRXXcj3deRo8e7WZTCnVe6njIMl0IVAQBH1b9zgsrM\/2FZ1i8oBwRPkk6fb\/hNKlNOWTh\/lUWAiAgBwYURHkRWLp0qavjU1fQRQgIgT4jQFg1G9yxO+97b25x+lgP4zKhC1E6UIjtivAhVwhd4EMhtivCh1yhwAv1oajq8KFSyqJ0iddcCESNVg5MFCriZSHAot1t27bZli1bbOLEiW4jOza0Y2M7ZFmVVRACQqBoBDhigEZ+FobFvZTD1HH5eIs7BLIcMo5BCNtAuRx9JemkT5EQCCMgByaMiMo5CIwYMcI2btzoNrFjIztP8JDlNBBDCAiBohHgVVL377fYH175iYv8WTlrQtahhh2XtyceAomzETw8EgOYCelLu7tmn2UQekqhL42NtXGYI6MX1RoCcmBq7Y7IHiEgBJoSgQFnLrR+A9vsj88vyYy\/Vg56rK4dLRk8lBECQQTkwATRUF4ICAEhUEUEmIWh+9\/8z+\/ZgrU7zW\/xz8LYa9c8Z0\/s7LI12\/YZ+UrI6AeqRF9xY2O8YCISAmEE5MCEEVFZCAgBIVAlBFjMe8IHJ9uUE39sY1rezLHiuh89Z6u3vpbDh1EOWZzzUI6+knQyPpEQCCMgByaMiMpCQAiEEFCxkgic+KFrXXevvpu7jT8zMX42xFUKXMohC6jPypajrySdWZ2rIAR6EZAD0wuEEiEgBIRALSBw3JA\/t6n\/8weRprCYFooSwodKKYvSBY9+IPJhgg+F+ZThQ+TDBB8K81UWAnEIyIGJQ0b8mkFAhgiBZkNAYdTNdsc13jQIyIFJg5raCAEhIATKiAAHKCqM2lwYucKoy\/ig1blqOTB5b6AqCAEhIAQqj8DsSaPs6Zsusrfu+JRLOy4fn2VEJUObK9kXg8zuT2HUYCLKRUAOTC4m4ggBISAEagaBt3463g4\/dX3GHiKDKhXaTD+QXzjMQlvKcSHP5ZAx3szglakvBMpsrRyYMgMs9UJACAiBviBAWHXwnCSFUfcFTbVtJATkwDTS3dRYhIAQaDgEWs+7zY3Jn5PkZ0McM3BhdqTUsoD6rGw5+krSmdV54QXVbHAE5MA0+A3W8ISAEKhvBDhegB16OSfpvTe3uIWtUSMiBBkqpSxKFzz6gciHCT4U5lOGD5EPE3wozFdZCMQhIAcmDhnxhYAQEAJ9QaCEbf05SYefvt7uvOKsSM0s8i21bPakUyvWV5L9kUaI2fQIyIFp+kdAAAgBIVAPCDAL86cje+z8P37f7pp9VmYmhlmLjsvbDWfj4glDSiqjH4g+wIi0XH3F2a8wapAXRSEgByYKFfGEQP0joBE0GAL+nCTWwvyxx5HJDjUekDXaUstKrQ9jC9epMGrwEuUiIAcmFxNxhIAQEAI1iQCzMBj2zgsrzC\/YZfFrOcKX0empEn3FhWYrjJo7LopCQA5MFCriRSJwzz332Omnn+6IfGQlz1QqBIRAyRE4cdhke63fx2zmqZvtwiHPZ+m\/7kfPWalDrOOch3L0laQza6AqCIFeBPr1pkqEQCICO3bssNWrV9v69esdkYeX2EhCISAESo7AFU\/OcTp\/1fVhl\/oLMzF+psTzfJpW5tuH07T60rYL96+yEACBRnVgGJuohAhs3rzZ2trabPz48XbOOefYBRdcYPBK2EVTqNq3b5+tWrXKSJtiwHkGCQ7CIxukfJgcP7DNJjx+X3ajnhILbKGebM4vfChH0MOAD\/VkC\/6lPhTVAD5USlmULvGEgBwYPQMFIbBz504bPXq0tba2ZurDyxQSMnv27DHRUQz4cvrBD34gPHqfCeFx9LkI\/n3kw+TGTw6O\/Gv7ytknWqllnxz1x4r1lWR\/pBFilgmB+lErB6Z+7lXVLZ0wYULGhmA+w4zJrFu3zqZOnSrqwWDOnKPT\/6TCZKqBA48NqfA4+jcCFkmY3Pg3n7chD301h+5e9JdWatmOu+fl9EPf5egrTmfrE7cCh0gI5CAgByYHEjGEgBAQAkKgWgioXyFQKAJyYApFSvUs+MoomI+DZsGCBbZr1y6RMNAzoGegz8\/AmjVrTD9CIIiAHJggGsrHIhD1yiiKF6tAAiFQFwjISCEgBOoFATkw9XKnqmznlClTbOvWrUboNEQeXpXNUvdCQAgIASHQpAjIgWnSG1\/ssAmdZnHhjBkzDCIPr1g9qp+MgKRCQAgIASFQGAJyYArDSbV6EJg\/f37mPTb5HpZ+hYAQEAJCQAhUBQE5MFWBvVY7lV1CQAgIASEgBOoDATkw9XGfZKUQEAJCQAgIASEQQKCmHJiAXcoKASEgBISAEBACQiAWATkwsdBIIASEgBAQAkKgLhBoSiPlwDTlba\/MoBcvXmynn366ow0bNlSm0xrt5Y033rBLLrnEYeExueeee2rU2vKbxbMRHn8QI7CiXH5LaqOHKDz4m\/HPCum5557rtjGoDYvLYwX3nHvPeCGiHQ8dOpTpLCinHuWMUJmmQ0AOTNPd8soMmA\/fbdu22ZYtW+zee++1ZcuWWTN\/2Lz++ut28sknOzz87sTNGsnFl\/UDDzyQ8yDeeuutNmnSJBfpRko5p1IDMuLwYLfrL33pSw4PnplnnnnGnQRfkxCUwCgclYULF9rs2bPdmLdv3+60Ll261KVceCZ4NsCDlDJ8UXMiIAemOe972Uf9yCOPuC+jESNG2OTJk23MmDHmP5DK3nkNdoADM3jwYBs4cGANWlcZk3Bg+a\/5hRdesLPPPjurU2Q4vNOnT3f8q666yqgH3zEa8MLY4vBguDgwzbTbdWtrq61evdq8Y0+ZAz737t1rODfg1WzPCM+BKB4BOTDx2EiSEgE+bPjQCX\/48oGcUmXdN2vmsQdv3l133WWcaYMzF+Tj4L3\/\/vs2cuTIDPvAgQMGP8OIz9StJA4P\/zdUtwMrg+E8C834jJQByoZRKQemYW5l7Q3EOzD8JzV69OjaM7CCFuHA8Dpt4sSJbh1M+N1+BU2pWlfMxn30ox+N7f+UU07JODA4MpRjKzeAIAmPI0eO2Kuvvmq8ImEtCBRexPkVpQAABz5JREFUM9QAECQOgRkXnF1mYfgMoTLPBM8GeVLK5EXNiYAcmOa87xp1KRHIo8v\/N+3XM\/hXacF3+3lUSNxkCDDb0NXV5daPsd5j\/fr1dvfddxtry5oBCv5mFi5c6F49X3nllc0wZI0xBQJyYFKApiaFIcCsAzX5MOKVEvlmJP575N3+7bff7oZP+eqrrzbe5\/NfpmPqYsFXRnyBU25WWDhnjEW706ZNcxBQvuyyy4y1ZY7RwBc+L+bNm+dGSAAAfy+u0HPhmeDZ6Mm614uUyYuaEwE5MPV\/32tuBHzgRL0y8q+Uas7gKhnEwuZmXtQbhD3qdQCvB+AH6zV7vtH\/hrzzwucHTj+fJf6e8yzwTPgyKWX45EXNh4AcmOa75xUZMdEkvL9mhoG1H7zPZ\/1HRTqvsU7AgGgTP\/1PmbDy4Lv9GjO54uawHuTMM8+0VatWub5JKcN3jCa78KzwzPCsMHTKjz76qE2ZMoViw5J\/rerT4EB5FngmeDbgk1KGT1nUfAj03YFpPsw04gIQYOqbfRoIoWY6eMmSJdasHzSMe+3atW4vHBZjggnYzJ8\/vwAkm6fKDTfc4F6rgRGv1yg3z+izR8rfD38zPCvgwd\/Qd77znYbeBwZnjfvOPzz8s8O4oaAjxzNBHfiklLORU6mZEJAD00x3u8JjZc0HCxAhPpAr3H1NdYcTs3HjRrdBF3iATU0ZWEFjeC3A64GwAxfECKwoV9CsqnUVhwd\/MzwrnihXzcgKdMz95r778foUHjJMIKWMjJQy\/Hol2d03BOTA9A0\/tRYCQkAICAEhIASqgIAcmCqAri6FgBAQAtVHQBYIgfpGQA5Mfd8\/WS8E6gKBHTt2GIuWSTGYaBPWOpCHOA8IIl9qok8OQmTdRJo+2ECOthCLaUttn\/QJASGQDgE5MOlwUyshIASKQIB9TDZt2uQWoeK8sCj16aefzmhgTRCUYZQ4Q7gtm8Gl6YO1OjhbY8eOLbFVUicEhEBfEJAD0xf01FYICAEhIASEgBCoCgJyYKoCuzqtPgKNbwGvS3h1wisURsvMB2cwQeThBYn6X\/va1ww5r0sgeME6vEKB74nXK16OzmDbYN\/YwCukrVu3GrMvzGhwzg\/1aUc\/kNdFfdr7foIy+qQdYcVeHgy19TriUsJ1qU973wcpfaLb66QPbIvTI74QEALVRUAOTHXxV+9CoGwIsBnY2WefbbfccovxRXz\/\/fe7AwJXrFhhhO5GdcxmaTgahKniZLDXBl\/q1CVdtGiR8SoGOSnn88BHTn\/soIoMuuaaazJ9I4cGDBjgzvdhfxP28CCcOmwLTtKMGTMMBwM93o6gEwNv0KBBLiyds6XY1RiHiD4KpZ\/+9KfGeOmDbfrpk7aU0c\/mi2AGTyQEhEDtISAHpkr3RN0KgXIjgGNw44032rPPPms4F3zBszla0t4ZOBb+8DzqUZ8dlXfv3m2sYcEpYT0LtpNShs\/5NOHzrlg7EuWg0DaJOO+Hgy\/9vifeDpwpZk9oy3oU6pBnnDhd9I+jBq8Qmj17dmZzRXaODurkiAecokL0qI4QEALVQUAOTHVwV69CoCIIeCfjgQceML7wvVMQ1zkzKDgEXs45M++\/\/769\/PLLbvYmfBYPW9sfPHjQjjvuOOOASvrhFYx\/JeP1FJrigOCIhPthZ9aTTz7ZHeCHLpwLnAzyaSncR1o9aicEhEBFEch0JgcmA4UyQqAxEdi5c6cbGI4BDoIrlOjCzMvbb7\/ttOEc8fqFVzq8uuKVTFpHxikMXOjnwIEDAY6yQkAINDsCcmCa\/QnQ+BsaAdaTMCvC4ZGFrOkIOzk4Dscdd5yddtppxqyHd4Y8aJThB2dDmMHh1ZF3ZDZv3uyr501pyywQeoOVvR3MCAX5yguBiiOgDmsGATkwNXMrZIgQKC0CrBfBcWGx7Fe+8hVjPQuLbom2ieuJxat+4apvz1qR9vZ2txFdsD16KLP+BH1E7QQX2r700ku2Z88e4zUTck9xToqXsx4FpwvnC563gwMwWQ8DTyQEhIAQkAOjZ0AINCACvCpauHChmzXxi3J5xUO0zXXXXWc4BVHDRs6iXNaxsKAX54XFuNQlJTKIV0PISVnECx+nhOgmFtoig5DffPPNbvM62gfJOymEM4dtwU4inIh4Qg924Lyk2YQu2GeD5DUMISAEehGQA9MLhBIh0EgI4FDwGgci78eGE7Bx48ZM9I3n+3Tw4MFGG9ayQDgnXkaKcwHfU1DO7Ai6vYyU+rRjMTGOESll+MipTzvsgpBB1HvmmWdcmDT1gjL6xMbguKJ46Iki+qNfbPBy8vCQwUM3faCXskgICIHaQ0AOTO3dE1kkBGoXAVkmBISAEKgRBOTA1MiNkBlCQAiUDwEimHilFVyjU2hvbNTHa6zOzs5Cm6ieEBACFUBADkwFQFYXJUNAisqIAK9poDJ2URXVwddRacbHayReY0G8aqrKINSpEBACOQj8LwAAAP\/\/I1dpawAAAAZJREFUAwC\/Uh0H\/UM2wwAAAABJRU5ErkJggg==","height":337,"width":560}}
%---
%[output:4e00c158]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:9e4a146d]
%   data: {"dataType":"text","outputData":{"text":"Running 500 joint evaluation episodes...\n","truncated":false}}
%---
%[output:81fdfac7]
%   data: {"dataType":"text","outputData":{"text":"Evaluating episode 50 \/ 500...\nEvaluating episode 100 \/ 500...\nEvaluating episode 150 \/ 500...\nEvaluating episode 200 \/ 500...\nEvaluating episode 250 \/ 500...\nEvaluating episode 300 \/ 500...\nEvaluating episode 350 \/ 500...\nEvaluating episode 400 \/ 500...\nEvaluating episode 450 \/ 500...\nEvaluating episode 500 \/ 500...\n","truncated":false}}
%---
%[output:688ce384]
%   data: {"dataType":"textualVariable","outputData":{"name":"idx","value":"451"}}
%---
%[output:588cd30f]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAQAElEQVR4AeydB3xT1RfHz2NvKEumtICiDAURAdmIIgqKKCoiCgqI4EYFcRX\/oKKIEwcqwwEKigMVVKayBEGQJcree09Z\/3xvefU1TdKkadIkPXw4b9x9fy\/N++Xcc87Ndkb\/KQKKgCKgCCgCioAiEGUIZBP9pwgoAoqAIqAIKAIBIqDFMxsBJTCZ\/QS0f0VAEVAEFAFFQBEIGAElMAFDphUUAUVAEch8BHQEikBWR0AJTFb\/BOj8FQFFQBFQBBSBKERACUwUPjQdsiKQ+QjoCBQBRUARyFwElMBkLv7auyKgCCgCioAioAikAwElMOkATatkPgI6AkVAEVAEFIGsjYASmKz9\/HX2ioAioAgoAopAVCKgBCZdj00rKQKKgCKgCCgCikBmIqAEJjPR174VAUVAEVAEFIGshEAGzlUJTAaCqU0pAoqAIqAIKAKKQHgQUAITHpy1F0VAEVAEFIHMR0BHEEMIKIGJoYepU1EEFAFFQBFQBLIKAjFLYMaPHy+VKlWSyZMnZ5VnqfNUBBSBSEdAx6cIKAIZhkBMEphFixbJCy+8IGfOnMkwoLQhRUARUAQUAUVAEYgcBGKOwPzzzz\/Su3dv2b17d+SgrCNRBCIDAR2FIqAIKAIxg0DMEBi0LSwXdejQQdauXRszD0gnoggoAoqAIqAIKAKpEYgJArN582bp1auX3HPPPbJnzx6pUKGCFChQIPVsNSVzEdDeFQFFQBFQBBSBDEIgJgjMq6++KpMmTZJ8+fJJ3759ZdSoURIXF5dBEGkzioAioAgoAoqAIhBpCMQEgSlatKghLrNmzZLu3btLnjx5POGsaYqAIqAIKAKKgCIQIwjEBIHp16+fIS6FChWKkcei01AEFAFFQBFQBCIFgcgcR0wQmMiEVkelCCgCioAioAgoAqFCQAlMGshu2rRJVBQD\/QzoZ0A\/A5n3GVDsg8M+jddc1GYrgfHx6PijeeyxJ6Vhw1YqikFAn4FGja6Qxo0bqygG+hnQz0CmfwYIL8L7zMfrLiqzlMD4eGw88Nmzl0jPnq9JYuJolSjA4Kab+smJEwmZ\/swGD\/5CRo8erZIGBg8++KD5C3z55ZcjGCt9js7PcjQ8s49ef0GGP3WPR3HOxdO1t3p2uqc6dppdxtfZLuvt7Ksued7qkU6+uzzX+Vo5tnyG+TuLtYMSGD+eaI0aNUQlOjCoXr26eaKcM\/OZzZkzRnbu3Cn16tVT8YFB3bp1zfPirFhFx2eFZ8VD4+zpmV1c8JikJZ7qOdOCqU\/dMh\/dKudO7OtRnP14uvZWz073VMdOs8v4OttlvZ191SXPWz3SyXeXy5e8Ja+cv5VHFnOiBCbmHqlOKNYQ0PkoApmFwMmd68RdSuU+KefkOul1SFsTm0la4rXy2Yxg6lOXZnKUiJc81ZqmEvJ8iac6zrRg6tKOr\/rkUcaXUMabeKq3u3BlWXwwNkOLKIHx9knQdEVAEVAEohSBLS4S4Us29ExIc2Zr2ltCOXc5\/Xwj+bTGRq\/1Pb1E3dO8Vj6b4V7e\/f5sMY8nu2zczc9KmcRpqcRjJUeipzrONEfRVJfOct6uU1VyS\/BWz053K57i1i7jPJ\/q+rH0\/rt0inKxcqMEJlaeZMjmEV0NFytWTFq3bi2co2vkWXO0pUqVkjvuuEM4Z00E\/Jv10WXT5eD0kUb2jk2UnUO7GPFW+5irvC9BqwJB8VbfTkeL4S4SV06QbB90soulOJfxQBrc01JU8HDjXt793kOV5CS7bMGmnZPT9CI2EVACE5vPNcvOCuICgcmyAETZxCEud955Z5SNOnzDtbUoLIvYpGXvuP6GyEBo0I5AbtxHVNpFInzJuW+vFcS9nvO+4rgzpgzlnFLmjX\/EeuIXKfnUz87ieq0IhB2BiCcwYUdEO1QEFAFFIAwIoAWxxVt3thYFLQgaBSSu\/bNSotcIgaCU6T9N8lZrmqo6ab6E9pBUFTVBEYgiBJTARNHD0qEqAopASgQgAJ40EM5S5HsSNBi2OMt7umbZBrG1IO5nT3XsNLQkCEs2TiHNFrus+xmSgqABgbQgcTcnCkQGgqIkxB0x3\/cb9hyTYGTmqn3ir4yZv018yaAf14ov6TVmhfiSNkP\/EG9Sc8AcqTlgjpEek8\/IgasG+QYmSnOVwETpg9NhKwJZHQHICwSA5RVfWJDvSZwkxFd98li2QWzC436mjDdhnIh7PuTDFvc8+x6Sgtj3oT47X+72i9r9JfzKlI3y3m\/7hLP7C9bbC9WZ7ny5Bnpd9JFpEowE2p97+eve\/kP8FXds3O8H\/bjORWC8izvu7vezVu8Tb+J8jlyfzlc81B+dTGk\/JglMyZIlZcaMGbJmzRpp0aJFpgCrnSoCikDGIgAJgHTYdiE7hnYxHUACzIWXg+2V4n5Gi2GLl6rJySzbIGhAPElyQQ8XaE8QbEqcQpotHqoFncSLy11sUsIvf5tU8JK2SQHXttgvavcX7ytTNsmwefvMy9fnS9XLC9Z9TIHcBwPKuUXzSDDSoFIRCUQ61CklvqRPy3jxJUM7XCi+5NuetcSbLHqqvtjy3pWWFPqpTzDQRWzdmCQwEYu2DkwRUASSEYCQnFg5U2T1XLGXeJIz3S4oi7YFzYdtF8KZYhAKzt7E9kpxP1PPFm917XSWbRCb8Lif3V\/CNlHgPHd\/EUHcX\/b2PWTCKe6Ewb63CYfzbJMNzjYJsc+kuYtNSvj1b\/96Z+z2PJ0vePtl7f4S7n1FOel+WRHh7P6C9fZCdabbL9b0nPcMaSbplfT056wzoVctCUTcsXG\/79MywUVgvIs77u73DSsXEW\/ifI7l4\/JItiO77EccU2clMDH1OHUyikD0IIAGZeeAK+XMe7cJZ5Z5vI0eLQsaFM7YhDjFyxKLt6aSbSAgF4g7kfBGGJxkwCYJ9tmZx7VNFJxnu133M2TCKfZ43M824XCeIR+2uE\/Y+RKzr21S0sf1698mFbykbVLAtS32y9r9xdv7ivJyT90i5uUbyEvVftnaY0nP2X2Oep+1EVACk7Wfv84+RAh07\/68tGnTJkStR3ezaFMQo0EhpkjFusnRUn3NDA0KSy4QFqc46\/AyR5vh1FJwDalwJxs2ubAJhU0ibOLgJApc07Ytzj65dn8Z20TBeXZ\/2dv3kAmnuBMG+94mHM6zTTY42yTEPpPmLjYp6eP69e8kFMxBRRGINgSUwETbE9PxRgcCUThKSAXCck5aw2c5x12cHja+6tv17DJWjzEmpggExU5LzxniAhmBiEA4nALxsNu0yYZNLtyJhDfC4CQDNkmwz848rm2i4Dzb7bqfIRNOscfjfrYJh\/Nsz4WzPT89KwJZBQElMFnlSes8FYGzCNhGsJwhEzbx4BrxtZRztolU++NAfOy8tM4sAyEsCeVt2DGt4sn5kBDnkg+aEpu0OIkLFfo4lkjQWEAq3MmGTS5sQtHHpZVAbOLgJApcQxJsoQ8VRUARyFwElMBkLv6h6l3bVQS8IsDSjS1O4gGpQCAWXiufzWApx12cHjZni3k82fXQuGAYaxeCoNhiExWboLAEhLDsA1mxBW0LRAahHcgHhAUiAumwBeJBvooioAjEDgJKYGLnWaY5k337dsqgQd1lzJhX5PTpU2mWj7YCBw\/ulV9\/\/UbGjXtF3n\/\/Kfnpp09l794dAU3jzJkzsnXrWtm1a2tA9aKpMATCNoLl2iYeXCMQi7TmA9FxF091bEICwYB0YI9iC4Sk3uA\/pPab66Tk47+aoFukITZRsQkK7UBCWPaBpNjSx6VpsTUoEBeuIS2exqJpioAiEFsIhIbAxBZGQc9mw4a\/DXEYMKCzDBlyn\/Ts2Vg++WSQHD58wGPbpJP\/6KPXyJtvPiKcX3yxq3mxeqpw\/PhRmThxlNx\/fzN56qn2pvyqVYtTFF23boU891wnWbduuTRp0k6yZcueIj+tG17sf\/\/9h7z66gPSuXMtad++opFbbjlPHnjgCjOfFSvmyzPP3Crz509Oq7kMzT958oTp\/+mnb5Zy5SrL9df3Esb7\/vtPS9++bWXLljVp9sfY33vvSenW7TIzn7Vrl6VZx1eBYcP6yYQJE3wVCVkeWhWWhxBP9iwQD9sIlutgBwK5cCcnbc5GCYWMIORDYtztUqhr9w9BscUmKjZBgZywDMSyDyTFlj6uZR+bzChxsZHUsyKQNRBQAhPi5zxjxnh54om2Urfu1S5yMVIeeeQtefjhN2T69C\/kySdvlJ07N6cYAVqEQYO6yaxZE+Thh990kZIhLuIx1pThBf3PP4vMtX04cuSgDB36mAwf\/py0aNFB\/ve\/sXLffYPlrbcelRdf7CbvvdfPde4qH330vGzfvkFuuOFeKVMmwa7u15kxvfbaA67x3iRLl86WTp36yqhRi12ajjXy6afL5fbb+8i8eT+5yMstsmLFPL\/azMhCU6eONdqWu+\/uLwkJ1SRXrjyuMT4hjRq1NZqmU6dOptndhRfWkebNb06zXCQWILibU7Y820zsJSLsWQiB78+4IRMIyze2QDqcwpKOLZAShHvKOMmJfU2\/kBJIhk06ICMIhGTuo7Vkwf3xsuOlRsK9LTZRsQmKkhOQVMkKCOgc\/UdACYz\/WAVccuPGv41m4JxzzpWaNRsl1z\/vvJpy1VUdZfPm1S7i0V\/QoJDJi5blnRUuTUazZjdJ2bKVSJZ8+QpI27b3yr\/\/HjeEZO\/enSadA8skc+b8IBdf3EiuvrqTWJYlJUueK5UrXyz33DPQJc+7NDLvSLFipYSXdMOG11PNb4EgvfbagzJ79veu8VR0kaFv5IorbnGNqaBpI0eOnHLZZVfJwIFfSI0aDUza5s2rzDkcB8Y3ffp4KVgwTsq5tC92n3nzFnBpUoa4lpJ+k\/Llz7eTfZ6LFy8tuXPn81kmVJloTdwF7QlC8DbEW9\/kOYV2KHu0wzty8oGJcrBFX7N\/CyTDFkgH5ANBW4KWxBaWb2wh3yks6dhit8WZ\/iAoEBOnYDgLKSEPEoNARhCIDUJdFUVAEVAEAkVACUygiAVQHmKxb99OKed6sfKCdVblZY+mYNmyubJ27XKTtdlFaObN+9FoEMg3iWcPtAGhWb9+pSxcONWkHjq0X3777UfXtSXNmt3oevnmdV2LnDz5r5w4cVyyZ89h7hctmiF\/\/DHdpX3p6SIeBUyaPwcIFUtZf\/45UyAE3boNkBIlynqsyvx69HhBSpdO8JgfqsTdu7fJjh0bQ9V8yNpdunSpS5u1NLl9grrhAeSUrYnNBLG1K2hHIAsQCkiHLYlF+4gnaTSzitT7KrexLYGQUM8WSAhtIWhLaJvBQChYvrEFwuGUPi3jxRZIiS2QFspBTJxCmyrRhICOVRGIHgSUwIToWUEg1qxJekFBVGwyYXcHEShUqJgcPXrI9SKbY5IXLpwm+\/fvdmlLSruIQLxJsw8FChSWChUuEJEzMn\/+z4ag7Nq12SwLFS5cTM49lzxXtus\/y1J58uSX\/PkLCcs\/EMB3qAAAEABJREFUX345VBo0aCNVq17myvX\/\/z\/\/LJaZM781FS65pJlccEFtc+3tULJkOalfv5WE89+hQ\/sMhp76xFAZbdWRI4c8ZWda2qlTp+Tbb7+Vbdu2pRrDlhylxJYFuWsKMiF\/S0HQkEBAbNIB8UC+c+W7y5\/lr0+x74s7IYGEOMkHBGTPkGZmGYflG1vsMva5T8sEF4FJEgiLLZCWVJPRBEVAEVAEQoiAEpgQgXvixAk5duyIaf3MmdPGqNTcnD1AaLJlS4J\/27Z1Jp8lJ7KLFj3HpYXJy2UKKVWqgrnfuPEfOeTSvmTPntMY4+bOnVcgOGSeOXNGfv99slx0UQOTN3nyZ2aJqk2brmZ5iTL+yuzZ3yWTAwhM9rMaHV\/1b7rpftfy2O3JRRgPxr8vvXSPvPhiVxky5H7p2vUyGTDgTlm+fJ6ZN4Upt2PHJvnmm2Hy0ENXGbseltIwSsZQuEOHC8xynL3cBjHBYPiVV3q5ltaOya5dW6Rfv3auui2kf\/\/2rj7qCPW6d68rP\/30CV2kENrB8Pnxx9sYe6FnnrlFRoz4nwurpGeWorDrBlL4zjt9pVOnGoIBM+PhHg2bK9v853rChA9cfV8mX3011rWE9YBUrFgxlZx33nnywQcfyLu\/bBQISbfhv8ulc6pKm6VV5PY\/y0rH3+Pk5q1Xyj35+8k9JYdI\/6J9jJTNf1oqb54gpab0liJf3y1Fv+0ujTd\/KN\/dUdoQj0VP1RdICMK1U9wJCUTEST4ihYAYIPWgCCgCioAfCCS9Qf0oqEUCQyB37jxSqFBRU2nLlrVePY5MAdfh6NHDwnKI61Kyu4iCZVlcepRDh\/bJnj3bjZamWrW6cvz4EaNpoTDLPdu3b5RLL23hWppaJpMmfSzt2vWS4sXLkO23oLXYsGGlKc\/ykb+Gvzlz5k5epoKU8EL\/3\/\/ukMaN20rfvh\/II4+8KS+88JUhHP37dxTyKceccF+eMuVz2bx5lTEKxh6oQ4dHjQ1P\/vwFDbnBKJpBxcWVkOee+8zY+HDP\/J5\/fry89tpkefbZcTJkyI9SqdJFZKUS7GYgVHhLPf30R8boOTFxtEvDVcVowNwr\/PPPIsGAGluaESMWuojUMrnuum6C8XBi4m3GEPv48aOyevUyGTfuTVcbu1zE7HSahPGXf\/bJ5z\/NlZ9evU\/OZMsh2W9+V2r0fF9qXNNZ8myYKaV\/e1GeaZjb7Eg74qZSUnb+y9K6WhGZN+sX+eeff1zz7y8rFs2XB7t1kj0b\/jIaF\/ex670ioAgoArGKQLZYnVhGzcuyTsiUKa\/JmDH90pRcucSlOUmSvHlzSIMGLV0vsjOuF9uf8v77D6Wo\/+WXA+XAgV1mmGvWLJQvvujvKnva3BcrVlIKF86f3Jbd9+rVs00+mp1Jk95yvSyflbi4XK6yBV0v7Ztl8OB7XMtRv0rPns8J\/U+Y8J7kzp1d1qz5JUXfdnvuZ+f4T58+Yl7MdPjvv8dcWox30mzDWZ\/rFStmul72L0rBggXk77+nJtefPHmoSytxrmTPns3kv\/baXfLddy9LnTqNXMtUtehSqlSpKc8884HUqlVXVq2a6lpWgwyekR9+GOaq0ye5rRkzPjLlDx7cLTlzJmGfM+cZ1\/JZLhcGSQa5f\/wxKbn86NH9XBqam2TJklmucWU3\/YLD558\/Lc2aXefqp7RpL0eOpLZOnjwoL73U1fUsssm+fX8bzMeP\/59LA7betL9582oX7ncJabtP55MTJ0\/I6dOnJX\/ePNLhskpy0xX1jNQ8P2lJMK5gfrm2wSVSsmwFufjgTCm14HUpmvuMNC17Rv7s31TQlLzS83rXOFzLiwf2SvVCR2Txl6\/JkCd6uEjfLhc52i\/PP\/+8i1A9LQsXLnTNoaCL+O52kbD7XET2uBm789CvXz+XZso\/cdZzXvvbhrOOfZ1W3eeee07eeecd17N+xq6S6pxWG3Z+qoquBDvPn7OruMf\/\/tSljKfKpPsrnuqTFkx9f+tSjr48CXlO4Znt37\/fU1FNUwTCikC2sPYWhZ1ly3ZYKlc+V5o0qZ+m1Kgh4pROnZq6XsgXuIjJGfnrryWuX\/ilTRs1alRxvYy2uJaYjhpEatSoKg0bXiaFCiU9jri4lO3YfVevnmTnkjNnDrnkkhqmrauuaiq9ez8sAwcOcL2k33NpHvpJ3boFZNOmSa4X\/wLp2LGDtGjR2EUEqsnhw3tl0aJ5snLlEilXrqSpb7fN2Tn2Cy8UF\/kxwxP6q137olTlqeMUZ\/3q1V0v5D+\/NXOvU6e2GYOz7DXXXCVlypQx+ceOHTTzpz5zp9eEhBJmHqRRr1q1pLnnzZtbLr\/80uSx1Khhp+cRxky\/Var8K1WrimtZjZbEpYmp4Ch\/vuzdu8ulkSourVpdmZxOHxdfnMelPXIxFxGJj096BidOLHZpt\/a7yMse+fXXyTJ9+k9G5vw2U44dP+YqKbJx7xGZdbCovPLHOjkhlknLkyuHLC9QU2bnrm\/kSOnaJv245JTy9VpJz25d5KIq8XLk0AG56oqmctv1V5p8Dueff76MGjXKSIMGDVzjr+Qa817X8zssM2bMcJHJn4xMmzbNEBrqHDt2LPmae1vq168v\/opdx\/0cTP206tapU8f1N1PDRV7ruHebfJ9WG3Z+cgXHhZ3nz9lRLcWlP3Upk6LS2RvS\/ZWzVVKdgqnvb13Kper4bAJ5TuGZFS5c+GyunhSBzEMg6Y2Zef1HRc+tW7c2Owuzu7AvcZ9MwYIFXb\/Yx8nDDz8suXLlkldffVUee+wxlxbhB9cLu6GLIOR2kYOccuutt7pepq3Err9v374Uv6btPitUqGCK0FaTJk1Sjclkug5btmyRoUOHSqdOneTee+91aWniXEsrr7kIzSrzUvzoo4\/Mdc2aNVO04aqa\/D9fvnyuJarSyfd8adnj8HZOLuy62Lt3ryxblhQMrnr16in6of4tt9wi9erVc5UUOXLkiGuJqbG59nSg\/IWwE1dm8eLFDVakIZdddpkrVVwal\/zm7OlAXcoiYAi+lSpVkuuvvz7FuDzVxVsIo9v7779ffv\/9dyO93\/lOTjZ6TPJc8z\/Z1\/ZDyXvDS3K0fAOpV72iFClZXizLkmy58kqjyy6SFx\/qJDPful8e7dhS+FesSEHp3L6Ni4Be4iJF+0iSK6+80ozD3LgOlmW5CFhVadSokUvLk1cSEhJcGp9Dcu2115r+7XEsWLDALCWtWbPGpJcsWdJVO+V\/5uyvpKz5310w9dOq26pVK2nYsGGK+f\/Xc9JVWm3Y+UmlUx7tPH\/OKWv+d+dPXcr8V+O\/K9L9lf9qpbwKpr6\/dSmXstf\/7shzCs\/sv1y9UgQyDwElMCHGvkCBAsLLb\/bs2a6lnDUuTcxf8tlnn5lfy8ePHxde7hdffLF5AZcqVcqMhhc6RsDmxnE4ePCguStUqJBZYjA3bgfsScaMGWNefBAjbCV69+7t0iIcNGr6smXLGkJTrVo1Q2Lcqiff5s+fX8qXL2\/u+XW\/efNmc+3v4eTJk0ZjQHmuObsLZIK0w4cPi7cy5GekrFy50mh90mpz56F\/TeyURRsPmKLvT1wobYb+IUUfmSaL3+0rE7beZgS342971pJFT9WX7+6\/VDq0aW7KQya633aDNKxcxNw7D+A5evRosTH19Kyd5e3rHTt2JGNqp+lZEVAEFIGsioCDwGRVCMI\/b0jF999\/b7QvnTt3NnYMlmVJ7dpJywzbt2+Xo0eTlpeco1u\/fr25RXsQZ6+1mJT\/DosWLZKxY8dKr169DFGBzOzcudNoOy699NLkgpZlCZqF5AS3C8uypHnz5maMZLFcEQjJyJ07txQtWpSqhrhBrMyNhwPlKO8hK2RJ3jC2O3zw85VC7JSvF+80Sdu3bJSZf201162P\/GjOhOGHuBTLfUpsEpI9e3ajgUH7dNFFF4nthdS9e3dTZ+PGjdK0aVNDYu05U9ZkpnGg7oEDSYTKvShkGHFP13tFQBFQBGIVASUwYX6yaFEGDBggkIobb7xRrr766uQRQDCKFSsmu3btEvcYIYcOHZK1a9eashCLvHnzmmvngTKvvfaaWY5hiQlDOwgNZVimsOtAJmgfGxTyvAlLGPXqJS3zQGBYsvBW1k5nDIwTLVHlypVN8vLly82czI3jYBOymjVrSrjW1KtUqWIIxtatW+XbmUuMGzOuzMidI5fK1v3\/GcKiXWlYu4ZLY3NaCu7\/R\/qsfUx+39hcypzcZmbBxoc8z759+xqSxtwhpibTj0O5cuVMqYkTJ8qmTZvMtfPwxx9\/yJtvvmlshSCsGzZskDlz5jiLmGvIE4a9c+fONfd6UAQUgTAjoN1lCgJKYMIIOy+4Z599VmbNmiUdOnSQp59+OlnDwTDQrLRo0cIs90yZMoWkZGHpAyKAgSdlkjMcFz\/99JN5kfbs2dO0yy\/yPXv2CMtYLFXZRXlZQpLsJSI73f1MvUceeURKlChhbDDwREB75F7Ovqdd7G5+\/vlnk3THHXcY7RKEBrsNk3j2ALn666+\/zFIYRO5scshPceXOkyKlKgjLOK+\/8aZUnviQnP\/jw0JwuPmrd8m\/J06aMZQokEuIk\/LOfa0kIe9JOX7ytExcvU82Hc0uaF7i2j9rMOF5gg\/PBYKBZocGwBsCgn0KMmzYMJLNstyvv\/5qnn+NGjXMswGfRx99NHlJiYKQRbw9sL\/CBoayEM8XXnhByKMMAnkZOnSoQMhseyDSVRQBRUARiHUElMCE4Qnz4lmxYoV07NhRfvnlF2PMy8vJ1ojYQ8iRI4dxh73gggtkwoQJyRoXiM+7774rtPPggw+aX+R2HfsMeXj77belR48exuiTdLQaBE2zLEuynQ2aRxvjx4+Xli1bGnIhafzDPuezzz4Tzrxo27VrJ\/3795e\/\/\/7bLJuwDLVu3ToZOHCgkAeB6datm9Fy8BLH\/sayLBkyZEjyC5oxfP755wIpwz4IexyGwRIVxIJrXJEpxzUCBpxJ52yLne6sa+fZZe0yhMv\/cc0J2ViulZy2csih9Ytl\/pLFcuHOadLvMpEL13wk2Y7uMdXLrv9Gcq\/4xtgS3XvjlZIzezbZeDy3dF9VWQacvFKe+XW3NGvWzGiWMNK2LMvUY8z0ixE0y0km0XWwx0CeZVly1VVXCfO+7777DFbz5s0zS0sYtKL1uuuuu4RlJ8gLn5OHHnrIkL3du3cbo+\/bbrtNID2MAa0PRIpyrq70f9ZDQGesCGRJBJTAhPCx44nDy\/+mm26Sp556Su68806ZPn26XHfddeJ8uTmHgJEtdXjB8QLjJQUp4MX45ZdfGg8cZ3muyRs+fLjgoUPbpCHYWPCCxGsJ91teou+\/\/74hHs6lK8r6El6iX3zxhUA6eGGiYcEjhuUYlr0gKSwXTZ482ZAYy0p6mVuWJWhhCJuPwe4NN9xgDJq7dOkiELqvv\/7a5FuWZdyEWS6hDcaCBor4IGgWPvnkE\/n4449JFpbEXn75ZfvQ6kQAABAASURBVIE0QcQgRmRQ7sknnzTeOKtXrzYky9ZUUBdyd8\/webJ3bH95Mu8Uufmi0nJegVMyb39euXdFGZn60cvy2P3dhTmhERo0aJAZK1qom54bLh9\/OtrYKEHYGCNtMzfaxdsMmyU0K\/\/++68hJPY4MdRlicgeJx5itI2RL0t0d999t7zxxhuGdNI2Gpz4+Hjjveb09mCZ7dNPPzV2SXx2WC4CI7RxPBc+N+CgoggoAopAVkFACUwInzQvP170EA+knUt7QVpaXWI\/0qdPH4EoDB48WL766isTep5lCk91+VWPuzTh6XmZOsvw4uMXOiTnlVdeMfE2IBw5ifrmLJjGNS9NXKl52c6cOTPZfZdlEuaGWzTj9tQM42Zs81xaBkjKyJEjjRaKdLs8Nh5oMlhWYsll8eLFwlJY6dKl5fbbbxfuSScfDRAvefBEo0U6AlGBULEUB8lhuYv0CdN+k4OVWsnJtQuk+4FR0ubwj9Ijxy\/yTpUN8lPtdbJ0wgjj2g75g2xQt1atWskk07IsYXlm3LhxyfOePXt2MsFhDmg\/7rnnHqMVoZ22bdsKY4FYQETscUKuXn\/9dcGjCBzBlc8IZISxMmYIrBMb2kdIA0cwoCzYJyYmJhtLUyZTRDtVBBQBRSATEFACE0LQIQnuhCIU3fESxA7DGzni1z5LNSwz8GK2rCQNSSjGEkltbthzTAb9uPasrJPWLuLC+Ao27Swleo1IlrzVmpKcIdLRtUyIWziaF28NomnBXsleLvNWTtMVAUVAEVAEvCOgBMY7NpoTHQh4HeWY+Vtd5GWdMdDFo6j2U59JxXFnDHGBxNjitYEAMyAtnTt3lj\/\/\/NMYamOvZLtRO8\/YJeHeHmDzWlwRUAQUAUXAgYASGAcYehndCJzcuU6OzvxEzox9TPa+182QF4hLn5bx0qdlgjSsnDqoXEbO+JxzzjHRi\/Hwysh2tS1FQBFQBBSB1AhkS52kKQEhoIUjAgHIy4aeCYa4yO9fysHpI824GlQqYsgLLtEmIYQHvMjq168vXbt2FWyCsHXBVsVdCFyH\/U4Ih6JNKwKKgCIQ8wgogYn5RxzbE8TOBdk7tr+Z6JYcpWRYoTslsWgfF3GJd0mCSQ\/Xgc0XMfi1LO92Rhj8stRkbx0RrrFpP4qAIqAIRBICwY4l7AQGbxJ+oRLrBLsAAnQ99thjJgBbMJPhVy7tYKRKu9gZXHPNNYKrLS6uwbStdSMLAQgLxrlEz605YI60SvzKaFwgL\/2LPi7DCt8p1drdY8gLS0jhHH18fLzZhJHPHMa6zr4x3J03b56J50OMHMSZr9eKgCKgCCgC\/iMQNgJz5swZIQYJMVGmTp0qxMtgmHhs4IbLzsC4sJIWqFCP+rRDhFfq8\/LA3ZQ4KrjjEvKddJXoRgDywh5Fg35MMs5lNrhFc85ZIl463NJWJtxZTnpfUZ6ksAsEvXHjxvLNN98IcW6cAyDuCzFxMO4lRpAzT68VAUUg3Ahof9GOQNgIDDEviENCzBLiehAIjJgYkyZNkrp16wpEJjEx0UR4DQRUIsJSj\/poc4h5Qrt\/\/vmnoJGxg7gRPIxorYG0rWUjAwE0LUUfmSYIGhdIDJqVPi3j5duetVyalngz0PMbtpTbListZQrlMPfhPqB1efXVV80+V9jDnHvuuSmGwNIRwfZwnybOD+VTFNAbRUARUAQUAb8RCAuB4Yv6ww8\/NFoXlnWefvpps1OyZVlCcC6imRKqng0OR48ebVTs\/s6AX7TUS0hIEPaEufDCC00kVGKiEFafX7u0hfqeUPhcq0QHAhAVyAv7FNkjhrhgkJtEXJI8i44un2Gy81TLuHgupsEAD3wOiaQ7bNgwE5GYwH\/uTUBi+BsgMjLLSe75ep91ENCZKgKKQHAIhIXAsAnhwoULJU+ePCZ0PAHenMMmCit7v1iWJUQkZV8fZ763azYrZJmIfGxfypUrx2WyWJYlTZo0MWQJ19b169cn5+lF5CNAHBebvEBc9gxpJoueqi9DO1wo3NszKNlrhH2ZqWeWL9G6sJdRtrN7T3kbEJswzp8\/31u2pisCioAioAikgUBYCAz7wrAPDwaOhHn3NCaMeosVKybsFbNq1SpPRXymsTSFnY17IcgSkWrd0\/U+8hDYOzZRnFJw8iDpvn+UfCivy89xw2RLYjOPg7Z3h86byRoY9qJC24i91fPPPy9t2rRJNV6WMadNm2bS2ZjRXGTKQTtVBBQBRSC6EQgLgWHjPmAqU6aM5M+fn8tUAnnhBQAJWblyZap8TwlsVmh7cmA8SSRU93Jz5swRtC+0zYaC7vl6HzkI7B3XX5xy9br3zN5FF2\/8xngZHVs23etg425O9JoXrgy2c2DvI5ZB0bC498tn++uvv5Yff\/xRLMuSyy+\/3L2I3isCioAioAj4iUDICQzLPPv27TPDKVKkiEA6zI3bAWNb8kkOZKmnffv2glYH+5ZevXqZXY55UdAvm+KxMZ9lWcLmeuyYLPovIhCwB4Gdy8xV+wRbl4FV3hFiuCCT4u+Rgy36SFz7Z03of\/YuKp2YpLmw60baGcPde++913gg4W33ww8\/GI0iWkWuO3XqJBjv4oF36aWXSsOGDSNtCjoeRUARUASiBoGQExh+iR45csQAwnKOufBwwD6maNGiJoc65sKPg\/2L99prrxVsbThDaDDm7devnyFMAwcONC8Oy\/IeXMyPrrRIBiMAecElGsHW5asjVWRSwj3y4ocjpefL78rF97woaFbsPYsye4nIn+ljA4MGBmNdNtCEpCD33XefzJ492xioc\/\/WW28Z2yx\/2tQyioAioAgoAqkRCDmBcXbpawmHX6+QGGd5f67PnDkjv\/\/+u9lAj9gv7nUwrMQG58CBA44svYwEBCAwCAa5tkv0oqfqR8LQghoDxuRs1oin0UsvvSSEDbj11lslMTHRGKmPGjVKWG4KqhOtrAgoAopAFkcgrAQmo7GGvHzwwQfywAMPCJ5LV199tXlBEAcGu5s333xT0OqMHTtWOnbsKJ5sZPwZE207BUNMlZOSXgwgLUTSve+zFQb+9rWKm8Bz9eILpLvN9I4lVPWI94KXEUuXzzzzjDz33HNy2223Sfny5QWiHap+td30fy4VO8UuVj4DzveV+ZLNyEMEtRVWAuPLtoUPDgG+AsEGu5cRI0aYF4K9eR7xYCzLMktHLCe99957goEw7tbEooH0BNIHZXnxEF3VlgEDBsjGjRtV0oHBU+9PNKH\/947tL11W9ReCzlUpfDLDsOQPd9u2bRnWXnqfM59nPO\/SWz+r1IuU55VV8M6Ieeozi\/zvfn7Y2+8r3l+8x2JRsoV6Uti95MuXz3Tjy7YF8rJnzx5TjjrmIo0DHka8rCAod955p3iqV7NmTaN9oakZM2YYjySuAxEMgbFrsIWN+EqXLi0q\/mNwInec\/LxwnbT+pbtM2Hqb8S5iC4AfdtwhrS9NyDAszznnHLM8k9nPhiVRgilm9jgivf9IeV6RjlMkjS+Tn1mGfVdEEqYZPRbeUfb7yg7mGsg7L1rKhpzA4HVkr\/fjjYR3kCdw8MwgnzxftjLk20K8Da4x2sWYl2tPYrta404N4fFUxlcaWx0QnMyW+Ph4E5QPmx2VPGliseOIyNw\/\/pKjEwZJmZPbhLgtGObiVXTu22vTrB8IxhjP8pkLpE4oyvJ5gsSEou1YajNSnlcsYRrquegzS\/s7L9TPIK32eUfZ7yveX3wfxaKEnMAAmk0gcCdlzyLS3IWgXhAMy7KkSpUq7tk+7yFF2BX4LKSZYUcAzyJsXQZ+Mk0ajqwvaFwYRI6S8cY1Ohq8ihiviiKQjIBeKAKKQMQgEBYCQ5Rd1Onr1q0TDGw9zR4bFUgMwe78jddSqlQp05S9JmtuPBzsyL4Es7PreCimSRmEAEa6xHVBBv24Ttas\/Mu0jOaFuC6REvrfDEoPioAioAgoAlGJQFgITNWqVeWSSy4R7Fw++ugjcbeF2bt3rwwfPjw5Rgbrgf6gWbt2bWHpCOKD0ZJ7u7QBuRk3bhyX0qhRI4HEmBs9hASBDXuOCTtGo32hgw51SkmL69qagHRoXojrApEhTyVgBLSCIqAIKAKKwFkEwkJgWDMlFgbRdr\/\/\/nshYi7W8HgE\/f3339KjRw9hKwBsZbp06SLYDojj36OPPioVK1Y0GzPu2LEjOQeX1DvuuMOEZR8\/frxpl\/ZoFzKD0S4W2HgrodnB0NeyNJhdMoAZfAF5QetCsw0qFTGbLrLxYp+WCSYgnWpeQEZFEVAEFAFFICMQCAuBYaBXXHGF9O7dW9hYcfLkyYaMYHxL7BbiZbBHEoG+zj\/\/fIr7LVhbP\/zww8Luv7RLe7SLHQ1kCA0MxGjo0KGCi7XfDWvB1Ah4SYG4oHV56olEeX36JUJgOogL2hdnlVjSvECKP\/nkE0F76Jyj83rfvn3CZxLS\/uuvv8rOnTuNltFZRq8VAUVAEVAE0odA2AiMZVnStWtXIahc8+bNBW0MQ4a43HjjjWb\/mFatWpEUkOA6jUbn22+\/ldatW0vhwoVNfcuyBG+mvn37ys8\/\/ywXX3yxSddDxiIAeVk+\/l15a1k7SdwzyDT+bc9ahsSYmxg9QIrZ36h+\/foyd+7cFLOE3LBh47PPPivdu3cXthRA+4c3QLt27QQtYYoKeqMIKAKKgCIQMAJhIzCMzLIsqVWrlmCvgtHumjVrZMmSJUKcFZaIKONJBg8eLGvWrBGWhEqWLJmqiGVZwt5Hb7zxhvzxxx+mLMbC06ZNMy+QQoUKpaqjCcEhcHD6SNk5tItsTWwm5\/\/4sHGPpkW0LGhguM4Kgvs\/AevsuS5evFjYNgDNjGVZxu6KYIuQHM58FllOZXsLu46eFQFFQBFQBAJHIKwEJvDhaY1IRQDyAokpsTVJ+2CIy9trhbgukTrmUI8LIjNkyBCzVISGEQ3MyJEjzXIpxLtJkybCPTtWv\/jii2LHMQr1uLR9RUARyOoIxOb8lcDE5nMN6axYNpqQv6UkFu0jA6u8IycfmGiICyQmpB1HeOPsv4U9F8Ns06aNdOjQwRiYc2+LZVlmc0eWoNDW2Ol6VgQUAUVAEQgMASUwgeGlpV0I4GnU30VevnORmB2l68r5ja52per\/3377zYQKYGsL7L2wz\/KECukYmS9dutRTtqYpAjGHgE5IEQgFAkpgQoFqjLR5dNn0VDNB+zJr9T6TjpcR3kbmJosdICEJCQnGqw5bq61bt8qiRYsMChiME8rb3Hg4YB8zZcoU2b59u4dcTVIEFAFFQBHwBwElMP6glIXKnNy5TrYkNpM17S1joMvZSWTGzN9q0LDjvGQlg10z8bMH9lsaOHCgTJ061eyG3rJlS8FVmmwiT5PPtbsQo4igjbp85I5MKO+1bUVAEYhFBJTAxOJTDWJOkJVjZzUv2LQQ+p89i9C8sK8RWwPQfMPKRThleSGYIga5eMjhKk1YALbN6Nevn0yYMCEVPuz3hfuAfXiLAAAQAElEQVQ1Gf5umUFZFUVAEVAEFIGUCCiBSYlHlr5D+4J3ESDkqdbUGOYuu+QhweaFQHU2eUHr0qFOaYqpnEUgLi5OIC2\/\/\/67MdI9m2xOzkOePHkEA160NE2bNnVm6bUioAgoAopAAAhkC6CsFo1xBA5OG2lmiOaFsP9oXa57+w9hX6Mk0lJKCFK36Kn6wr0prIcUCLB0hAYmRaLjpmDBgvLZZ58JWhj28XJk6aUioAgoAopAAAgogQkArOgs6v+oc5SMN4ULNr1TtmQvZTQvJHSoU0ogLRjs6tIRiKgoAoqAIqAIZDYCSmAy+wlEUP8Fm3Y2o2H5aNbqfYKgaYG4mAw9BI3Ali1b5IYbbpAaNWqYfZKCblAbUAQUAUUgiyIQcgKTRXGN2mnneO+ovLGpQrL25a1bL4zauUTiwNk+Y\/ny5XL48GHBlToSx6hjUgQUAUUgGhBQAhMNTymMY8RgF2NdNC8sHemSUcaCz4aO\/fv3l549e8ojjzySsY1ra4qAIhBLCOhc0kBACUwaAGWVbAx2EXvZCM2LLh1l\/NMnAB6bPT766KPGGynje9AWFQFFQBHIGggogckazznFLHGXtoPVzexQWtoM\/UNwk0YoSJA61byAhIoikIUR0KkrAhGOgBKYCH9AwQ4PskJslw09E8SWLc82E4LVbclRSr7L19IY69KPvWzUp2UCtyqKgCKgCCgCikDEIqAEJmIfTcYMjJguB6ePFIiMU2h9WKE7ZVjhOwVblz1DmiW7SkNkyFfxjsDatWvlk08+EfY18lZq3759xtPo+++\/N9sM7Ny5U9hKwFt5TU+BgN4oAoqAIuATASUwPuGJjcyK486YqLrnvr3WnGXAChnU6Af5Ln9LE5BONS6BP2ei6RKMrn79+jJ37twUDUBuVqxYIc8++6x0795d7r\/\/fmGbAQx427VrJ3\/\/\/XeK8nqjCCgCioAiEDgCSmACxywqa6CJseWVhWdk3Lo8hrwQoE41Lh4eqZ9J\/\/77rxw6dCi59OLFiwUjXc6WZUmjRo1kxIgRhuRwLlSokNlqYNGiRcl19EIRUAQUAUUgcASUwASOWVTXwNPI3hqAbQGiejIRNniIzJAhQ4Sloly5chkNzMiRI6VJkyZSsmRJc+b+3nvvFTaAPHjwYITNQIejCCgCikD0IKAEJjKfVchGRZwXGi8fl6SB4VolYxBg2Wj+\/PmmsTZt2kiHDh3Esixzbx8syzIaGJag0NLY6XpWBBQBRUARCAwBJTCB4RWVpWeu2mc2ZIS82HFeNMZLxj\/K3377TY4dOybFihWTrl27CjFfPPVCepUqVWTp0qWesjVNEVAEFAFFwA8EPBMYPypqkehAgOWi697+w2wNwDWjJkid2r2ARPoFEpKQkCDZs2eXadOmydatW8W2a7n44otl+PDhMmHCBI8d4LnENgLbt2\/3mK+JioAioAgoAmkjoAQmbYyirsTRZdNlTXtLCFKH1oUJ4CqN1gW7Fw1SByLBSe7cuWXgwIEydepUOXXqlLRs2dK4StPqBRdcYIgN1+6CGzXkRpeP3JHRe0UgNhDQWYQPASUw4cM6LD0R62VrYjPTF3Fe0LT0aRkvkBdIjJIXA02GHcqXL28McmfMmCG4SufPn18KFCjgtf1du3YJ7tcUqFy5MicVRUARUAQUgXQgoAQmHaBFchW0L4xvQe6aJs5Ln5YJgpCmEjoE4uLipF+\/fvL7778bI11vPeXJk8fsgYSWpmnTpt6KaboikE4EtJoikHUQyJZ1ppq1ZrolxzmC9gWtS9aaeebONrdracmXBqZgwYLy2WefGS1M2bJlM3ew2rsioAgoAlGMgBKYKH547kNn+Wjv2P4meWv2UlI+Lo+51kPmILBnzx759NNPjWbmmWeekfHjx5sYMZkzmvD0qr0oAoqAIhAuBJTAhAvpEPezd2yisEkjJIZNGqu26yETetUKca9Zt3nivbz11lseASCg3YIFC6R\/\/\/7y9NNPG40L+yY9+uijUq9ePWGLAcp4rKyJioAioAgoAn4hoATGL5giv1DczYlmw0ZGOin+Hrnj2npcZjEJ33SLFCkif\/31V4ptBOzeP\/jgA\/nnn3\/MbfHixeXKK6802wtce+21cs4558jHH38szz\/\/vJw4ccKU0YMioAgoAopA4AiEncDgPkqQL4wYK1asKDVq1JDHHntM1qxZE\/joHTV4GRB345prrpHzzjtPaLtWrVrSt29f2bx5s6Nk7F5ueOQPuafkEFlc7vrYnWSEzIytAVgiWrZsWYoRkfbTTz8Jxro9evQQgtu99957hrC8+eabMmvWLLM30vTp02X27Nkp6uqNIqAIKAKKgP8IhI3AEP\/i\/fffl5tuusnEzmATPIZ5+PBh+fLLL+X666+XiRMnkhSwsPfM3XffLQ8++KD5VUxcDhrZv3+\/jB07ViA1xOsgLZSSmW0TpK7duL2C95G6Sof+SRQuXFiIpjt06FBx7mkEWUYIZnfhhRd63EqAvZEgNyxDhX6k2oMioAgoArGJQNgIzMyZM+WVV16R06dPGzdTbARWr14tkyZNkrp16wpEJjExUf7++++AkMaWANsC2sfDY\/DgwcKeNKjwR40aJeXKlTMvGFT2vFgCajxKCm\/Yc8xE2mW4fVrGq9s0QIRBWrduLUTffeGFF1IsB0FucKlmPyRvwyhTpoxs27bNW7amKwKKgCKQlRBI11zDQmCOHj0qH374oaB1QRuCYWNcXJz5dXr++efL22+\/LfxiRZMyevRoQVvj72wI4w55wXX1nXfekXbt2gmurIR4b9SokfALGWLDEhXh2\/1tN5rKQWAYb4NKRZS8AESYhM\/sddddJ59\/\/rk88cQTcuDAAcE1mr2Q2BNJvPxjufPHH3+UUqVKeSmhyYqAIqAIKAJpIRAWArN8+XJZuHChsQu44447Um1yB5m56667DKGBZGzatCmtcZt8iBHLTxAeiEv9+vVNuvOAGh8iQ4RUND728pKzTLRe43HE2MfM38pJdOnIwBC2Q44cOaRPnz7SoEED4yLdvHlz+eKLLwTbq19++cUjEUdjiMbm+++\/l4YNG4ZtrNqRIqAI+EBAs6ISgbAQGNTsfHHHx8dLpUqVPAJ1wQUXCL9ct2zZIqtWrfJYxj0RogM5ypMnj+DhYVmWexHhJYO765IlS4xbK5qZVIWiMOHg9JGyoWeC2e8I+5ekoHWlo3Am0T1ktHssjUJGMOB98cUXZeTIkUar2LlzZ\/n6668FsvLqq69Kp06d5NJLLxXy27ZtK7Vr147uyevoFQFFQBHIRATCQmCwSWGOrPujCeHaXSAvuJyiTVm5cqV7tsf79evXC3vLlC5dWiBHHgvFaOLB6aPMzL7L19KcibgLiTE3eggrAiVKlBA8jfB4sz\/ffI5\/\/fVXeeSRR+T+++8X2wOJXawpxzIq12EdqHYWqQjouBQBRSAdCIScwBw\/flz27dtnhlakSBFjn2Ju3A65cuUS8kmGmHBOS2xNDbE18ubNawyAu3XrZlyzcaOuU6eOcV\/dsWNHWk1FXf7JHevMmCfkTyIwDSrFmXs9ZA4CfP66d+8uc+bMETR+N998s7BZY7ly5cy5Q4cOMmLECLNXEuUonzkj1V4VAUVAEYgNBEJOYDBYPHLkiEHL1y9OloGKFi1qylHHXKRx2L59uykBgcEWBoNKbGjwaCJj9+7dQlAxvEGIP0NaLAi2LwhzQeuC51HDykW4VclkBDAmx1A9W7ZsRvOCLQxxYQYOHCi4T2NgnslDTN29pigCioAiEIUIhJzAODGpUKGC8zbFNbYqkJgUiWnc2PE3iPGCYSRxOfAIwYWaZSh+8fILGO8mXK1jxY364LSRBpmdpevJxMQb1PPIoKEHRUARUAQUgayEQFgJTKiAhciUL1\/eGE6ybIShLtoefvG+8cYbxjgYD6Rx48alawgYCzvl5MmTkmmyc53sHddf2O9ob9PemTeOjMMgJufABy3TPiP6bGLyM6Wfp0z83o2yvynn+4rvoliVsBIYX7Yt\/HH6ip2R1gPo1auXicHhXo5YHXgokT5jxgwhOi\/Xgchtt90mjRs3TpYBAwbIxo0bM0U2fPWaGfqC3BfLoZLVMmUMmTV3f\/rlD5cAcf6UDWUZPs943oWyj1hoO1KeVyxgGa456DPLnO\/+QJ4vphP2O4v3l3lpxOAh5AQGTUi+fPkMdL5sWyAvuKFSkDqc0xLb8whXVm\/u2ZZlmUi\/tIUxbyoCQ0Ya8vLLLwsB9mzp3Lmz4PmUGZL36C4zWpaPWl+akGnjyIy5+9Mn9lB4BflTNpRlWBLFHiaUfcRC25HyvGIBy3DNQZ9Z6Yj\/3uUdZb+v2GLHvDRi8BByAoPRIi8UsMMbCa8krt2FKL3kk+7LVoZ8WwiAZ1\/7OtuEiF\/E6SEwbHVQr149sQXihL1OZsj0ywfLpeWnmg0bM6P\/SO8T7x4+c5k9Tj6PkJjMHkek9x8pzyvScYqk8ekzy2OCskbSM3EfC+8o+33F+4vvo1BIZrcZcgLDBKtXr85JCFJnewiZBMcBjyFiuliWZTbJc2R5vYToQE7Q7KDB8VYQ4kIebtr8euA6GoUtA0bPS4q626dlQjROQcesCCgCioAioAhkCAJhITBE2UWdvm7dOsGY1tPI\/\/rrL4HEEOyO+BmeyrinsWxEecjLrFmz3LPNPQHF7LyyZcuKvZxlMqPs0GvMCpm1OimmDu7TUTZ8Ha4ioAgoAhmIgDaV1REIC4GpWrWqXHLJJQLR+OijjwSNiRP4vXv3yvDhw83eMYRkZy3Xme\/tms3wmjVrZrI\/\/fRTYcsCc+M4kEYcDsuyzHYDEClHdtRcbthzzJAXiMvQDhcK56gZvA5UEVAEFAFFQBHIYAQ8EhjckgmDzh4uxFQJtk\/WTG+\/\/XYh2i5t4jGERTXakb\/\/\/lt69OghBJrDVqZLly5m\/yJx\/COGC5F1cYvGENfOsixL7r77brO\/EtqbBx54QCArECSEvmib+eBeTaA7u260nSEwjLl8XB5h2wCuVRQBRSDzENCeFQFFIHMR8Ehg0Gawh8vp06eNtTXeQRAHW7iHfAQy9CuuuEJ69+4txGiZPHmyQEZYArr66qtl\/vz5wh4yiYmJcv755wfSrHGdfvvtt0093PsgLAS0Q9iDhiB2LGER6A5vpYAaj6DCg35ca0ajEXcNDHpQBBQBRUARyOIIeCQwYNKiRQtp06aNsbZmJ2fIB1bNnLk\/deoUxfwWy7Kka9euMnbsWGnevLnRxlAZ4nLjjTfKN998I61atSIpYDnvvPNMu2ySh2GvZVliWZZwTdpnn30mCQnRa\/S6ZuVfgrBs1KGO7jgd8AckEyo8\/\/zz5u8ndF1ry4qAIqAIZG0EvBIYXv5Agyso2hL2ciEoHGfuSSc\/ELEsS2rVqmX2J8Jod82aNQIZIs4KS0Te2ho8eLBQlkB0JUuW9FisUKFCwiZ506ZNM4bCGAtzTRp5HitFQSJ7pB9k9AAAEABJREFUHuV4vZVM2HqbjF98uZTYOjcKRq1DVAQUAUVAEVAEQouAVwLj3i1+5tiycHbP0\/vQIXB02XQxJKZEvJToNULyVmsaus4CaFmLKgKKgCKgCCgCmYmA3wTG1yDRdCxdutRXEc1LJwIHp48yNYdXflYKNu1srvWgCCgCioAioAhkdQSCJjAY886bN0\/YfyZ8YGaNntC8HHNpYLbkKCVVGl2dNSats1QEFAFFQBFQBPxAwCuBee2116RRo0bJGxi2a9dOFixYIJztTaI4E6b4vffe86MrLRIoAid2rDNVtmYvpa7TBonoOfTr108mTJgQPQPWkSoCikDWQCCGZumVwLA30ebNmwXXZIRtAIitwpl7Wwj\/H0N4RNRU0L4woJ2l63JSUQQUAUVAEVAEFIGzCHglMASFmzt3rqQlU6dOlQYNGpxtTk8ZhQCB6yZ\/+41prkvnW81ZD4qAIqAIRDkCOnxFIMMQ8Ehg2PDwqquuElyW05L4+HghGJ16J2XYMzENsedR6VPbzLUeFAFFQBFQBBQBRSAlAh4JzA033CDsX+QseuDAAbEj8R4\/ftyZJR07dhT2MEqRqDdBIcCu09eVHi3juq5R1+mgkNTKioADAb1UBBSBmEHAI4GxZ0e03W+\/\/Vbq1asnNWvWNGeuITeQHLyP8EKyy+s5YxAYM39b8saNGnk3YzDVVhQBRUARUARiCwGvBAaD3UGDBsnDDz9sNC9Mu3jx4oJAWth88Y477pCPP\/7Y7CJNvkrGIID2hZb6tEwQtg\/gWiUmENBJKAKKgCKgCGQQAl4JDC6gH374ocTFxcmQIUNk5cqVgsYF4frdd981GykS5n\/hwoUZNBxthk0bsX+BuHSoU0oBUQQUAUVAEVAEFAEPCHgkMIcOHZIvvvhC2J9o3Lhx0rZtW8mZM2dyda4x8v3oo48Eg182S0Qrk1xAL9KFwMxV+2TQj0mxXxpUKpKuNnxW0kxFQBFQBBQBRSBGEPBIYDZs2CAbN26UAQMGSEJCgtepli1bVp566imzIaPGg\/EKk98ZaF8o3KdlvAztcCGXKoqAIqAIKAKKgCLgAQGPBIZgdeXLl5dq1ap5qJIyqUaNGlK4cGF\/thJIWVHvUiHA0lHpk9ukwzfNZU17S44um56qjCZEBwLPP\/+8tGnTJjoGq6NUBBQBRSAKEfBIYNavXy9VqlSRAgUKpDml\/PnzGzuZ\/fv3p1lWC3hHgMB15CbueUnYAylPtabqPg0gKoqAIqAIZHkEFABPCHgkMHggYdviqYJ7Wu7cuQ3ROXbsmHuW3geAAAQG7Uvt44skR4l4KdlrRAC1tagioAgoAoqAIpC1EPBIYIBg3bp1xn3aDl7n7bx582ZTjjoq6Udg1uq9Uvv4YtNAjpLxhsSYGz0oAoqAIpDJCGj3ikAkIuCVwIwdOzY5cB3B67xJo0aNZObMmZE4t6gaEx5IZU4mbR2Qt2qTqBq7DlYRUAQUAUVAEQg3Al4JTPbs2aVMmTJSrlw5n8JeSZZlhXvcMdUfy0cY8MbUpHQyikCGIaANKQKKgCKQGgGvBObll182mpVffvlFfMmcOXOke\/fuqVvWFL8RsMmLvXkjS0h+V9aCEYlAv379hGCQETk4HZQioAgoAjGAgEcC07JlS2nSxL9lDMuyhC0F\/HG5jgG8MnwKaF96jVlh2q1Y5QJzxojXXOghIhDQQSgCioAioAhEHgIeCQzB69hCgIi8GO\/u2bPH435H5G3atElKly5tJPKmF\/kjsrUvRN5t88RrUnHcGclbrWnkD1xHqAgoAoqAIqAIZCICqQjMyZMn5bXXXC\/SihXl4osvlscee0z++usvj0PE3fqBBx6Q+fPne8zXRN8IOLUvfVomeCmsyYqAIqAIKAKKgCLgjkAqApMjRw6pXr265MqVSwYNGiQjR46Uyy+\/XCwrtaEuWwk8\/fTT8sorr8jOnTvd29b7NBCwl47QvjSsrHsfpQGXZisCioAioAgoAskIpCIw5CxatEhuuOEGYRNHy0pNXChjC1sJlChRQhYvTophYqfr2TcCaF9YPjq3aB7d98g3VJqrCCgCioAioAikQiAVgTlz5oxs375drr32WkEbk6qGWwJlMOD1tszkVlxvzyIAgeGyfFwegcRwraIIKAKKgCIQMwjoREKMQCoCc\/jwYTlw4IAULlw4oK7XrVsXUPmsXnjM\/K0GAl06MjDoQRFQBBQBRUARCAiBVAQmZ86ccurUKcG7yJ+WMPpdunSpUM+f8lomCYEx87eZi7qF9suWxGa6+7RBQw+KgCKQYQhoQ4pAjCOQisCwOWP58uXlq6++kqNHj6Y5\/TVr1hgvJAx\/0yysBQwC9vIRS0d1C+2TY8umm72P1H3awKMHRUARUAQUAUUgTQRSERhqtG7dWubOnStDhgzxSWLwPHrmmWfk9OnTcumll1JVxQ8EbALjLKrRd51o6HUMIKBTUAQUAUUgpAh4JDAXXXSRQGI+\/PBDufrqq+Xtt9+WlStXCoHrELyUnnvuObniiiuM9qVbt25y\/vnn+zVQvJW6du0qF1xwgVSsWFHwYiLWDJocvxrws9DBgwdNhGD6ePfdd\/2sFZ5ig35cazoi9svJnUm2QzlLxJs0PcQGAs8\/\/7y0adMmNiajs1AEFAFFIAIR8EhgsGdhL5cbb7zR2MIMHjxYWrVqlbw7dbt27Ux8GAx+b731VunUqZPHODHO+eLd9P7778tNN90kU6dOlX\/\/\/ddk08aXX34p119\/vUycONGkBXugr08++cTs5RRsWxldn12nbffpDnVKyckdSQQmR4kKGd1V1m5PZ68IKAKKgCIQ0whk8za7AgUKyEsvvSRffPGF0bTkz58\/uSjXaF\/IGzBggOTNmzc5z9vFzJkzTcA7lptuv\/12WbBggaxevVomTZokdevWFYhMYmKi\/P33396a8DsdDVGkaV3swdveRwSvQ\/uyd1x\/k5VHtw8wOOhBEVAEFAFFQBHwBwGvBIbKlmVJrVq1BM3JkiVLhGUehGvSyLMsi6I+BWNglqPQulxzzTVC9N64uDijtWHpiSUqti3Apmb06NEe913y2YEjk6UjIghzdiSH+9Jjf9i+4H2E8S7LR0eXTTflIC9qwGug0IMioAgoAoqAIuAXAj4JjF8t+FFo+fLlsnDhQsmTJ4+xS2GJylkNMnPXXXcZQjNlyhSzbOXM9\/eapaNhw4bJvHnzpEGDBlKyZEl\/q4al3KzV+0w\/5c8Grzu2bIa5z1u1iTnrQRFQBBQBRUARyNoI+D\/7sBAYlnQOHTok8fHxUqlSJY+jw6i3WLFismXLFlm1apXHMmklQlxGjRolCQkJ8sADDwgu4WnVCVc+2hen8S79lug1gpOggTEXeogZBLAhmzBhQszMRyeiCCgCikCkIRAWArNixQoz7zJlygj2M+bG7QB5KV68uFk+wuPJLTvNW5af+vfvL8ePH5eHHnpIKlSokGadcBZA+wKJYfmooWPjxnPfXiu6fBTOJ6F9KQKKgCLgHQHNiR4EQk5gIBT79u0ziBQpUsSrViRXrlxCPgXXr1\/PyW9h6eidd94R9mO67rrrjOu335XDUBDiMnpe0tYBb916YYoec6j7dAo89EYRUAQUAUVAEfAHgZATmBMnTsiRI0fMWNxtX0zi2QP2MUWLFjV31DEXfh7wcML4l6WjBx98MOK2NYDAoIFx1774OT0tpggoAlkGAZ2oIqAI+ItAyAmMcyC+lnVy5MhhjHwlwH\/YzAwcONDUevTRR6VcuXLmOiMP7AvlFPZ\/CkRs25f2tYpLIPW07MmoxYvPnz6\/6H1++uz02UXzZ8D5vuK7KFYlrAQmo0HkA4YLNrFjCK7XokWLjO7CtHfbbbdJ48aNk4XYNxs3bhR\/5Ou5\/wjalzKFckhTF7fyp46W8Q9bTzjxh7tt2za\/no2n+hmVxmcTw\/WMai9c7YS7n0h5XuGedzT3p88s\/d9P4XruH3zwQfL7iveXeZHF4CGsBMaXbQtf+MeOHQsI4unTp5tAe3g29erVK2RLRy+\/\/LKwRGVL586dpXTp0n7JhwsOmTn1aRkvtauU96uOv21rudTP4JxzzpESJUpkOs5oFAkGqc8o9TNyYhIpz8s5Jr3WZxbtnwHeUfb7CrMK8xKKwUPICQx2L\/ny5TPQ+bJtgbzs2bPHlKOOufBx2Lx5sxCwjiKPPPKIlC1blsuQCJGC69Wrl7yVAu7g2Oz4I1sPnjRjalKlhFkio87Rb1+ULZ3yCmfuVfKcxSb4M1GhcZ\/PbEx56JCYzB5HpPcfKc8r0nGKpPHpMwv+eyrUz5N3lP3O4v3F91EsSsgJDC8TfhEDHt5IeCVx7S5E6SWfdF+2MuQjuGazFQH10L6waaNTeHio6yjLlgjkNWnSxGxISVo4BONdBONdJBx9ah+KgCKgCCgCikBWQCDkBAYQq1evzskEqWPPI3Pjdti9e7fs2rXLROOtUqWKW254bjO6F8gLbRJ5l7OKIqAIKAKKgCKgCGQMAmEhMETZxR5g3bp1ZgNHT0MnhgskhmB3lStX9lQkRRoGu+zL5E3mzp0r5cuXN3Uef\/xxs4\/TjBkzwrq9wKzVe03\/zsB1JkEPMY\/A888\/L23atIn5eeoEFQFFQBEQkUwBISwEpmrVqnLJJZcIdi4fffSRuNvC7N27V4YPH26i8DZs2NAYYGYKGhnc6cxVSQH8zi2aN4Nb1uYUAUUgKyKAB1BmC15+SGaPI6v3nxU\/\/+5zDguBwejr9ttvF6Ltfv\/994LNCvYpRNDFBbpHjx6yePFi4z3SpUsXwfhRHP+I75IZNiyOIaTrcuPeJK8q9yWko8tnmPZ0DyQDgx4UAUXADwR4YT\/22GPJ7rHO0A7hvG7evLl07NhROIezX599OcJcZJVyHTp0ED4Tfnx0YrZIWAgM6F1xxRXSu3dvyZ49u0yePFkwqMX9+eqrr5b58+ebPZISExPl\/PPPp3jUy0yX9mXDnmNybtE84r6EdHLHOjO\/nCXjzVkPioAioAikhQAvq99++03cwzrY7rJ6Hp0i3EUs44FrNJ+FtD4zsZ4fNgJjWZZ07dpVxo4da5g72hjAZXPHG2+8Ub755htp1aoVSTEhtv2Lu\/YlJiank1AEFAEnAmG9xi0WL0uVesmhLbIaFnwGwvqhi9DOwkZgmL9lWVKrVi0hSiBGuxjgLlmyxPyiYImIMp5k8ODBARvhlixZUjDapQ+WqDy1G8o0NDC036dlAqcUUqb\/NGETRyRFht4oAoqAIqAIKAKKgF8IhJXA+DWiLFAI4nLu22uzwEyz7hT79esnEyZMCA8A2osioAgoAlkQASUwIXrotgEvNjAh6kKbVQQUAUVAEVAEsiwCSmBC8Ogx3kVoOgsQGKapoggoAoqAXwgQNuPTTz+Viy++WHDkwPMUT9OrrrrKxE56\/\/335cCBA6naOnXqlMyePVtuueUWueGGG+T+++8X6hBzaceOHaY8e0uS5vUAABAASURBVOpNnTpVcA6h7WHDhpnwHCbTdfjnn3+EuGA4i3Dm3pXs8z+mCN27d5ejR4\/6LKeZ4UdACUwIMVfyEkJwtWlFQBGISgTi4uLk+uuvl2rVqgkBTtnLDjvHn376SQYMGCAff\/yxXHnllSa0hj1BYoex9x2OIITkGD9+vLz55pvCmQjuBI0kFAchOHDvhhARpmPIkCEyadIkuxk577zz5KGHHjJxyThzn5zp4QJCRB8ERl2+fLmHEqFLYvwQMAhZ6HqJ7pajn8BEIP5j5m81o2pQqYg560ERUAQUAUUgbQTQyrzxxhty+vRpE3Zj8+bNphIk5MMPP5Sbb75ZWrdubbacIQMC1KNHDy7liSeekJ07d5prDgULFuQkkCNc0M2N6wDJyZYtW6p4Y+LhH\/vtofU5dOiQsWmDVHgoFpIk5gwBA4uQdBADjSqBCcFDtD2QOtQpHYLWtUlFQBFQBGIXAUjMtddeazxPMYRnA2DCbEA82ELGsqwUk8eDtU6dOrJy5coUWhvKot1hCxuM6g8ePJiinj83EydOFGKYseQ0ZcoUj4Hj1q5dKw888IDcc8890qBBA\/n666+Tm0ZzBPEieGuzZs3k1ltvNXsCsozGUhlLYGiL7rzzTqlRo4a0bdtWIG0Ic2azYjRSaJ88Lasld5TOi2ivpgQmBE9w1uqkLQScAez2jk2UNe1T\/uGFoGttUhFQBBSBqEbAsiyx45wQqZ098iAnaFuKFi2aam4Qm+rVqxtbl1mzZiXnW5YlXbp0kWuuuUZmzpwpL7zwQqptbJILe7hAm8PSEW1gUwOpmDZtWoqSEBFIEuNF08P+e08\/\/bSx0yFA68iRIwU7m9dee03effddsxcgZATt0GWXXSZsybB9+3Yhn\/AihP2AtJUtW1Zuuukm01enTp2kT58+UqhQIXOvh\/8QUALzHxYZcuXNeNfePuDosukZ0o82oggoAooA2t5eY1ZIpMqY+dt8PCTvWTlz5jSZvOBZ\/kH7AIEpXry4Sfd2cNey0E7fvn2NsTC2LESB91bXPR3yUrp0aVOXIKvFihUzAVedfSxatMiQEjQ0kJJbXRoWtCYYGGNE\/NVXX5mo84yjVKlSUq5cOaHOnj17hGUsy7KM1ga7oISEBOG8atUq96HovRcElMB4ASa9yTaBKR+XJ71NaD1FQBFQBPxCgHANkIRIlZmr9vo1D2+FeOnHx8dLkSJFjBfQ\/v37PRbF2JYMSARnp6DNYPsF8p555hlDIJz5nq7xOEITwpIOGh7IyOWXXy5Lly6VhQsXJlfBM+rw4cNij4tyEC2WjjZv3iwYGLMUxBIWmpcLL7zQLElBcpIb0Yt0I6AEJt3Qea5oG\/A6l49O7lwnx1yaFwLY5a3W1HNFTVUEFIGwIhALneEoMLTDhRKpkl47QLQUPJ+aNWtK4cKFjfcQJIHlFtKdAnkhsjtp2M9wdhfaee655wTtyWOPPZbC2Fc8\/MPjCNsWbGAgHxAfbHEoiiaHPrmuUKGCQLIgKZCWdevWCVqbKlWqCFoWlphYCsLV2xbaQhNDfZXgEFACExx+qWrbGpgGleKS8w5OG2muc5SMN2c9KAKKgCKQEQgQqqFDnVISqeL8IefvfFkygjiwLIMhbu7cuYX98qiPjYu7J9DWrVuN8e4FF1xglmMo50mwY7ntttsMifGlAaF9tC\/Ynrz44otiEw+0ONja4JWEdxJ94Ib9yiuvmLg1eAxBsthEskyZMlKiRAljt7JgwQJjn0N5BHdv7GO4VgkOASUwweGXojbkBQNevlQ8\/eHmrdokRXm9iV0E+NIjNoX3GWqOIqAIuCPAsgteObgtDxw4UCAClIHIQGK++OKLFEs4LPW8\/fbbcuzYMUM0IA2URxtCmq0pIc2yLHn44YelYcOG3HoVtCh\/\/vmnNG3aNEUZlqCIX4NR8eeff25ICeOlf\/7eMbQlRo1tbItNS+3atWXEiBFCeTQ4BMWD8MTHx6do29NN9uzZjbv4li1bjGcSRsWeymXlNCUwYXj6tgFvHl0+CgPa2oUioAhEMgIsq7DksmzZMoGooLnA0Jbouj179jQ2Ij\/\/\/LPw8rfngRHs\/\/73P0lMTJQnn3xS0KRAdHC3xgZl7NixwjIRhAWSgMfPr7\/+KkT8pT+7HUgI3kiQCzvNeV6xYoVAnNCwUN9ZF28i0ikPkcK9GYPbefPmSaNGjYwbdOPGjU10YDQ4efLkMW0RlI9xs7xFPQgMdjIYFDP\/OXPmGNdqPJxYImP5Cu8rtD3MifG+9957Jugffav8h4ASmP+wCPoKDQyNOA14I9X+hXGqKAKKgCIQbgTi4uKkY8eOZtkHQoCGgqUaPHYgNt26dTNLL+7jQiMBYRk3bpwJQoctCmloPmxCgsFtkyZNjLcQSzXEV6E\/Z1sY9eLeXLJkSWeyucbIFndm6jJGZ12Wi\/r372\/i06ChueOOO6Rq1arGTRsXauaAfU39+vXlnXfeMTFj0AgRMRgbHYRr0vLmzWu0Qcx\/1KhRRtPEFgmU+eGHH4TlM8p9+eWXAqnCfoc6ZpB6SEZACUwyFMFfbNx7zDTCEpK5cB1O7FjnOoqo\/YuBQQ+KgCKgCASFAFoUtBjYtBA35e677zbeQdiuBNVwgJVZpoJYoHVp166dQK5YNkbbUq9ePbOsFWCTWjxABLIFWD6DisdmMxv2JG325SQweB1VHHdGyiROi81J66wUAUVAEQgzAmgnsD0hVgsbLbKEhJEt+ymxjBSO4ezbt09Y7vnll1+SvZqwc0GTlCtXLrG1QuEYS1btQwlMBj55gkrRXAOHBxL3KlkPAVwvWQfPejPXGSsC4UMAIoP2A00Iu0sTmp9lpHCMgL5texlixOAFxVjwmmIpKWTjCMfkoqQPJTAZ9KCwf\/HlgZRB3WgzioAioAgoAhGCAEtF2O1g4Iv9yvfff2+WkrDNiZAhxvQwlMBk0OOFwNBUeY3ACwwqioAiEJsI6KwUgYhBQAlMxDwKHYgioAgoAoqAIqAI+IuAEhh\/kUqj3KzVSXt+eApgl0ZVzVYEFAF\/EdByioAioAicRUAJzFkg9KQIKAKKgCKgCCgC0YOAEpgMelaePJD2jk2UNe0t4ZxB3WgzmYuA9q4IKAJBIkB0W0LrE9K\/YsWKQrA5oupeccUVxgAWQ9hTp04F2Uv6q7O3EgHn8CpiU8YePXoIXoUILtukE+k3\/T1ozYxCQAlMBiGJBxJNOZeQdAsBEFFRBBQBReA\/BOLi4oSos3jwkMqmiUSynTRpkhBu\/4EHHpDPPvuMrKCEfYogHYTrD6QhdpO+9dZbhYi97DSNizZ7HSHDhg2TAQMGBNKclg0hAkpgMgBcW\/viDGAXki0EMmCs2oQioAgoApGIAPsdQWzy588v7A8UTEC6gwcPyhNPPCHr1q1L11QJ23\/OOed4rIumqGXLlh7zNDG8CCiByQC8x8zfalppUKmIOXPQLQRAQUURUAQUAf8RgHgQzZbtAggEx3ITWg8C1LHXUI0aNQTtCAQHDQ6bOh44cEBmzpxpNoFkH6QdO3YIy1ArV64U9hpi\/yI2Sjxz5owQXJIloWuuuUbYimDp0qX+D85Vct26dWZvIqLsssz17bffCstfRAVmGwHuSfc2bua3adMm6d27t7z11lvCfkrPPPOMkO5qXtauXSv33XefsD0Cm1l+8sknZtdr8lRSIxAIgUldW1OE+C9j5m8zSHSoU9qcOewd15+T5K3axJz1kLUQQN3MF1rWmrXOVhFIHwKQliVLlgjbAZQrV85s9khLBITbvn27rFq1Svbt2ydPPvmkQFLY7BD7Gcpky5ZNsKepVasWt0Zat24tlSpVMvLss89Ks2bNZMqUKfLFF1+YPthAkYLk2eSB+7SEbQPYyJFyw4cPF2xh2EyyZ8+ehsgQDZh0b+Pev3+\/ISjXXXedOUNgPv30U2ErhC1btgj1IS9sKMn3x0svvSQLFy6kOxUPCCiB8QBKIEkb9iRt4Ij2xbZ\/cS4fFWzWOZDmtKwioAgoAn4jcHTZdNnQM8Gn+Gosrbrk+6pPHmW8yc6hXSjiUz7++GNDTLB9QcNx4403yrnnnmvqFCpUSAjZX6BAAenQoYPRvtx7771CuH5TwM8DBAm7mjp16gjaHexwKleubIgRfXprBvLUq1cvQdPTvn174YcJZffs2SPseYRBb\/HixUmSmjVrCrtZk87yV8px3yqM+48\/\/pB\/\/\/1X0CRRqV69evL+++8L2iC0RBCf8847TyzLMu1hvzN\/\/nyKqnhAQAmMB1CCTXIuH+UoER9sc1pfEVAEFAGPCPBjKS3xWPFsYlp1yT9b1OuJMt7EayVHhm3EO3XqVGMg+8Ybb8j9998vvLztYpZlCZoWSec\/NB+E+4cMYNjLsg3k6NprrzVkwVuz2MEMHTpURo8eLePGjTPjo+y2bdsEcsOYLMsiSWgvISHBpJNPomVZKcaNNqVIkSLCZo\/i+gcxa968uTEYhtyw\/AVJYozz5s0zhA2y5Sqq\/z0gEHYCs3jxYunatavAXFEBwkTZ+GrNmjUehudfEmubK1asEBg8akTaRX2I2vDVV18V1kj9aynwUnYAO6cBb+CtaA1FQBFQBAJHgN3uz317rfgSWvUmvurZed7q2ul2OU\/nEr1G2MXSPFuWJY0bNzZajLlz58qiRYvSrONvgd27d8uRI0ekfv36RosCSbClevXq\/jZj7Gww4MXgGG3JsWPHBG0LDViWZcgK6eSLh38sV7EUhhbGmX306FHZuXOnlCpVyrhs22PjjGGzs6xe\/4dA2AgMJANV2U033SQwbfsBHj58WL788ku5\/vrrZeLEif+NzM+rEydOCAyZNcXvvvtOYNpUpb\/169cL\/vxXXnmlQJxID5U4CQxfKhXHnZEyidNC1Z22qwgoAoqAoOFNS3zBlFZd8n3VJ48y3oT89AikgO\/29NT1VIdlHjQkkCJnuywfTZ482VMVj2loQ9CyQDTKly8v\/PC2fyAzZggN6eR7aiA+Pl4w4kUbZOdDan799VcpXbq0aQ8iY+dBbMaPH59CG2Xn6VkkbAQGK\/FXXnlFTp8+LbfffrssWLDAWIjj+1+3bl2ByCQmJsrff\/8d0HPBqhwtC5bfLVq0MIZaWJ5jgY6BFQZhfCCwFN+8eXNAbWthRUARyAgEtA1FICUCZ86cMe+ClKkis2fPNl4+BJDzpRnJkyePFC1aVDZs2GCWbNDAswSDrQvvEsuyBC8mbFX4Ubtr1y5j\/Pvzzz8LS1QsT+GB9NRTTyXb2zjHgraG9wXvK4iJM49r7Gg6duxobGimT59OkhAAj3GwwkC+SXQ7NG3a1Njv4Bllk58XXnhBChcubIL4oZ3B\/Xvjxo2CJxNLXZAYlprcmtJbFwJhITA8gA8\/\/NAYL+G+9vTTTwtM1rIswZocFzQCGEE0WGvkw+0aW5r\/Ya5YcFOetUw0MbBjy7IEFR6ud3xYixUrZsgSa5hpNhpggZmr9plaBMdLAAAQAElEQVQaDSrFmbMeFAEQYA0bcs21iiKgCPyHAC\/mMWPGCFoHUocMGWKWTTCU5YXdtm1bGTVqlDHe5SWPcSuEg78n6orrH+SkW7durisxBrD8WGVpB+3HX3\/9ZWxMeNegYencubMxI3jkkUeE65EjRwqmBgSk+9\/\/\/mfeQaahsweICJGCsXHBM4iVA6fG5GwxueGGG4Qf5XgMvfPOO4LH0EMPPWTG423ckDLKobVhZYAVCc6XXXaZ8aTCCwsNDZ5WeCGxpAYudp96TolAWAjM8uXLjSsYrBm3MciFcxiQmbvuussYU+HqxgN05nu75kPCB8tbu9SDGDVq1IhLMwbYubnRQ5ZBQCeqCCgCkYMA3\/e8lDGo5Tv8hx9+MHYp\/HgljXgvePAwYuwZIRNo1flRQF3SEb7b0ezzDnj99deNCzI\/Ulu1amV+wBIvBjKD1gUtP1oMfjzjrk0dvJL4wUtbTmEpByNi6tIvrtZ4BjnLcG1ZlkAyGD8eRpgrcG9ZlvgaN7aZjJu2f\/rpJ+PibVmWef9Rnzxw4cy9ZVl0p+IBgbAQmEWLFpk1vHjX+h\/GtR7GIRj1oimB8eLz76mMexoaGNY1sRQn7LN7PveWlaTl4RpVINoarjNCcKGetXqfacppA2MS9KAIKAKKgCKgCCgCIUMgLASGdUFmUKZMGSFMNNfuAnnB0AqCgf2Ke76ne4IXwVJRMdK2pzK0Z9vVoHa0rIxjsxAY+oS8IFynFk1RBBQBRUARUAQUgYxGIOQEBqMqDJMYOP7v3gIQ4RdPPuXWr1\/PKUNk3bp1gj89jV100UVeCRT5gcqgH9eaKn1aJsiWxGaypr0le8cmmjQ9KAKKgCKgCCgCikAQCKRRNeQEBpc1LLoZh7vtC2m2YMeCVTn31OEcrNAOBlgsS6HhYT0x2Dad9e3lo\/bxx+TYsiRL9DzVmjqL6LUioAgoAoqAIqAIhACBkBMY55grVKjgvE1xzfIOJCZFYhA3LB2NGDFCMACzLEuwWMfjKT1NYlTsFGxpELstO\/KuxJWTnFUamsBG5KuczLJY8NnQ5591n38onj2fKZWwIhDxnXn7nDnfVxE\/iSAGGFYCE8Q4A6oKeWF\/DdzzuCaSYZcuXYyVd0ANnS2MxTzubLbgfvfbsqTlozKFcsi2rVuTSsaVFfz3VTZmGg784RLGO7OfAV8suH5m9jgivf9IeV6RjhPj43Od9EWjR0UgCQFcvvlsuAuu3fb7ivdXUunYO4aVwPiybeELnyiGwUJMQDuWjfDvJ9ovsQAIDORr+SqtPvHNx8XPFmIJHM1ewFRLKFFAisQlxYDJkzev4IKnUjrTcMAjDRfMzH4GaBRx28zscUR6\/5HyvCISp9Ip\/45sG0HzxaMHRcCFgLfvOt5R9vvqwQcfdJWMzf8hJzAQh3z58hn0sEkxFx4OkBeiJpJFHc6BCgHziCEwaNAggcgQEIl7b1ER\/W2fGAL16tUTW3AH33k0yZsJ76McB7eZpnKWiBeWwVTyZBoOeV0kEkPxzH4GfCAgMZk9jkjvP1KeV6TjZI+Pz5WKImAjYH8u3M+8o+z3Fe8vu3ysnUNOYHiZwBIBDm8kvJK4dhe0JeST7stWhnxPQkwYNoUk4i\/5hHOGyPArmPuMlg17jpomITAnd6wz1zlKeLfxMQX0oAhEDwI6UkUgZAjwYxZNeZ06dUzQN0JiEGmXJX+i7RJhnUB1xA1j+Z+tYIisjiMG9Yhk621wLKtQBq9TtiSYMWOGx6JsZ0MZ+ujfv7\/Z3gathT2m++67T9iegMqsENAOe\/YxLiIFM06WidH2N2\/eXO68804TTZeoukQGpg5hPkgnsB1zxCaTerTJj2y2TiDgHnMjUnD79u1NO8yZOVBOxTsCIScwdE34ZM54A3mLhMtuoexXYVmW8KGjvL\/CB6Jnz55CRER2Au3Tp48g6dXk+NuvXe7o8qQ\/EPVAshHRM7vI8mWrSCgCikBqBD766CMT3JQYXhAD9qx78sknTbR0ou1CFKpVqyb8AOXFPnjwYCFqLfaH2DcSft\/bBr0sB3bo0EF470CUcOZAO+8cBUQJwgQBoY927dpJ7dq1hXpEbkejAfFgywHqoU0lvH+PHj3MjtRElGec2JrMmjXLOItAar766itDyDCXoA6kpVOnTjRhNPjYY1KPcRHhlx\/abEnARsbYbI4dO1Yov3DhQrOKYCrqwSsCYSEwRNnlQ0JMFsInexoNYZshMQSkq1y5sqciHtPQvLDGx4eIWDJsjIXHEUTGY4UMStQ9kDIISG\/NaLoioAjEJAJ79uwRvu+x0+C9wP5FhOJHO89WAr4mjfaD\/e3YZLF3797ChoveymfLls1s6AgZYDsbcfzDeHzevHlC8FRHsnH0oB4\/fiExzjyuSSffsixDwGiDZVDePeSz2vDwww8LZSBJpHkSyBP7Qd18882C2O8ry7KkRYsWAh6e6mlaSgTCQmCqVq0ql1xyiWDnAvOGfTqHgQZl+PDhwgOHscKgnfnerin\/6quvCmo6PkAY2954443mQ+itTijSyyROk4rjzkhejQETCni1TUVAEYghBNh5Ge0Dmgh7WsQAgzCUKlXKTvJ6hsSweS\/7BUEEvBUk6nvr1q0N0aDcmTNnkoui+eGHNUs7yYkBXjBeCMuff\/4pvL9sLQ\/LUmzSyBKRpyb50c0SGeYVzMOyrFTFGHcgP+RTNZBFEsJCYGCot99+u9kh9Pvvv5devXoZN1s+UIT5Ry2HOpAPA2t\/qN6c+LP+yQcNFd6OHTuSs2DWX3zxhbmHtUKS2NGaMp6EPG8fKtPIf4c0rzbuPWbKYANjLvSgCCgCikCYETh9ZJOc2DU3YoXxuUOCtgHNizMd8wEIBz92nemeri3LEtswlfcH7xFP5Ui74oorhH3yeO9QljQIxOTJk4VlKjQlpKVHeE\/xvoJ8sZEj7sr0wfuOpSfyPbWLyzOrEbzvvNl7sqTGO81TfU37D4GwEBi644OEyo8PLx8eyAjGU1dffbWgNuTDm5iYKIEEm\/vuu+8Mu6Z97F9Yu7Qtrz2dYcUsU1E+WNmwRwlMsBhqfUVAEQgOgRO758rB2R0iVo6ufD3NCWLsin0L2nNPuz57aoClHNKJjePNrpJ8iADeqHzvY2dCGhp7tCf+kCXK+5KaNWsKditohfgRjkYF+zdbG+OpLj+imbOnvNhIC98swkZgLMsSDJZ42Fhss+TDNCEufHC\/+eYbadWqFUl+CcZXK1eu9KtsRheyyUtGt6vtxQ4C\/fr1E9TWsTMjnUkkIpAtbznJUaxexEq2fGXThG3OnDmCdrx79+4BL\/+z5MQ7xFsnaFhYjmErmUmTJhnNP1p7DOzRlHirF0h6QkKC0CZGuPny5RMMewcOHGiikHtqhx\/xaGew41Ei4wkh\/9PCRmAYkmVZgmqNB4wRF2uYS5YsEWxXWCKijCfBAp2yWKuXLFnSFEEFicsb6f6Ks75pJMiDLh8FCaBWVwQUgaAQyFm8nhRqMCZiJW+Vh3zODyNcXItxYw4kXteiRYtMu2hALCulDYnJcBzQ9F9++eXyzz\/\/yEsvvSSE7MDW0lEk1SV2mthspspwJdjaH0gX7x5XkkBK0PTgiQRZQqO0dm1StHbyncKS1rnnnitohZi\/M0+vA0MgrAQmsKFFbmlbA1M+Lk\/kDlJHpggoAopABCPAyxviQqR0XuoMFVKAlxLX3gQPIpaDMDfwx04EbQfmAxAPbGFwwY47Gz3dUx8Yz0JeGJ97\/qpVq4R80lm+ItYYqwHcI+SxDMYyESSINHfBiJnx4HXFioSncrSJx64v+x73drPivRKYdDz1Wav3mlrVFr4uG3omyNGzO1GbRD0oAopAFkZAp+4PAnieQl5wIYZY4HSBhgT7EV82LZs3bxacOnjBs0xD2A1v\/UEiWKYh\/9JLLzVxWCBKzZo1I8mrYOKAgS3xZujPLkhQO7yXIECksQ0GRruUoy\/SCMa6fft2E1OGpSXSPAnzJibM+PHj5a233hKnzQx9EhOHti3Lt3bJU9tZKU0JTBBPu\/bxxXJy57ogWtCqioAioAhkLQTwAiJ2F84c2L3YDhctW7YUtBOYB2ATuWzZMuOkgW1J3759hYBvBCzFIeTnn382JMETckTiffvtt014DaLZQjywd8HWEnKCYS9liMOydOlSYTws\/UCgaA\/NzmeffWa8ZrHLxDmEjRFpk61pWP6hHPYuxLDBJRqbGgLu4YnEfNDMQMwwGIbgUH7u3Lkm4B3kjTwi+BLThrkQvZd+mBvEDk9d+qaeincElMB4x8ZrzsxV+6T0yW1S+\/giyVEiXuO\/eEVKM8KNgPanCEQ6Ati6EA8M+xF3wROV5Z2OHTsKXj0soxBJF+JAlFuIDYFKCxUq5HWapUuXlscff1yws0S7gd0lhSEZkAPLsoQytEMMF8aAnSVLP5RD0J5AcMj\/9ddf5ZdffhHIEOnkIxAt5gFJwQsWokU5NEPMgaUrbG0gR\/RBObQu5FEfuxm8lqiLLSj9TJkyRYYNGxaQNy5tZVVRApPOJ1\/mVNIGjjlKxqezBa2mCCgCioAioAgoAulFQAlMgMhhwDtr9T6pfWyxqZm3ahNz1gMIqCgCioAioAgoAuFBQAlMgDhDYKhS+3gSgdENHEFDRRFQBBQBRUARCC8CMUVgwgGdvYVAOPrSPhQBRUARUAQUAUXAMwJKYDzj4jV19LytXvM0QxGwEcAdFKNB+17PioAioAhEMAJROTQlMFH52HTQioAioAgoAopA1kZACUw6n3\/pxGlScdwZyVutaTpb0GqKgCKgCCgCBgE9KALpQEAJTICg2TYwug9SgMBpcUVAEVAEziJA+HziqtSpU0fYB494KWx+eubMGSHQG8HhLr74YmEfoy5dupjou1dddZWwLEu9AwcOnG0p9YmAdMR7oS5t9O7dW9hcFSHQHPFeCKJHTSL6ElCOAHd33nmnMA6Cyr377rtmM8apU6cKweUYI4H07r\/\/fmnQoIEQz2XevHmmDHvsXXPNNWYejJFIwc2aNRPS2LrAjtLLvNj3iT5oj\/4YR4cOHeSiiy4S2u7Ro4dUqVJF6tatK4899pjY42U8jFclJQJKYFLi4fMODySEQkpgQEFFEYgJBHQSYUaAAHCQB0LzQwCIjkv4\/IULFwqB3q6\/\/nqpVq2aECyOCLcEmmODxAEDBgiRbQnnT6A7T8OGoLRr187UpQ3IDDZpCMSoa9euydU++OADmTVrlomQS8A5guVBLtavXy8EooPY2BFxiY775ptvCqSDrQbuuOMOIYpukyZN5LrrrjNt0i9jpUzTpk0NKaEOmcwL4kOkXu47depktjfInz+\/CZJHOfrIkyePIW7PPvusjB49Wp5++mlhDybqqKREQAlMSjz8ulPy4hdMWkgRUAQUgVQIsFkjUXI7d+5sSAbh+O+9915hc8P58+enKu9MQKNC+H32OEKzwr5Bznz7mn2KihQpYt8mny3LSzbebQAAEABJREFUEogHxAYChRaFbQZy5cplykBMHn74YcmWLZugDSIREsXZFsqz8zS7Wn\/55Zdm3HaefSbKLlsjUJctBOjLznOeLcuSGjVqSNWqVZ3JKa7ZYqBChQop0vQmCQElMEk4+HW0tS\/ldRdqv\/DKyoVQV6MS9wsDLaQIZCEE2O8I7UJcXFzyrIsWLSpoHkqVKpWc5u0CEkMIfsLzB\/o3xvIS2ozSpUub\/iAsbBcwfPjw5A0VWc656aabxF768TSO3LlzGw3NkSNHhOUwT2UgOCdPnjRkyFM+aWhf2BeK7RW49yRsX8Dykqe8rJ6mBCaAT8Cs1XtN6YaVUzN7k6EHRUARUAQUAZ8IoJ1AM+EstGvXLuFl7ksTYZe3LMvYiHDPjs22poT7tGTOnDnCJpGUY4kI+xrIE8s32JvQHhoW9k8in3KeZO3atcb+BXsV97lQfufOnYKmCHLD8pKnMpRTCQ4BJTDB4RcLtXUOioAiEKUIrJ99SL57YGPEyp+fJ\/3o8wUvWgrsW2688UbBfsVXWTsvZ86c5nLbtm1y+PBhc+3psHTpUsHmBXLSvn17wQ7GWa5mzZoyduxYQauDTQ2aHcocPXrUWSz5Gq0MdjsQHoyEaTs503XxySefCDYxLVq0EJZ9fvzxR7n11ltdOfo\/FAgogQkA1Zmr9pnSDUZeLmvaW3Jy5zpzrwdFQBFQBDIDgf0bTwgkIVJlw6xDacKCVgSNRffu3cWyrDTLOwuw5ITmxpnmvK5evbpgqIsx7Lhx42TAgAHObHPNEs0XX3wh7CadL18+U54dpSFWpsDZA8bDGNRCYDAMxuC3bNmyZ3OTTuyijWDzgp0O9j1JOXoMBQKZT2BCMasQtYkLdemT26SMS3KUiBckRF1ps4qAIqAIpIlAhcvzS+s3ykes1Lj1PzsXT5PBCBfXYgiBLzsQ97qLFi0ySWhQLMt\/0oNXEca1VIY0YUfDNctaGObiiVSsWDFBI8QyEXm24DWEdua5554zLtKeloUsyxLconG1xsB30qRJdnU9hwABJTB+grphzzFB2hz+0c8aWkwRUAQUgdAiULh8LrnolriIlQqXF\/AKAOQF4vLEE0+IrcmAVOCl5LWSK2PTpk0yceJEOf\/884WlGleS3\/+xd4mPj5fx48fLkiVLZNCgQYK2xG6gcuXKZhmLpSLsV+z0QM4QMYgOruGQnUWLFgVSPaCyWb2wEhg\/PwGQF4rWPr6Yk5ToNcKc9aAIKAKKgCIQGAIEdYO83HzzzYI9y44dOwQPIV78vmxaID0EioN0sMxTpkwZjx1jFEwZlnHcC6AVQetz7rnnCka7LA1BWCi3b98+2b59u9SuXVtYWiKNdjj7IjTuy02QF8Z58OBBYZ7MlzZUMhYBJTB+4ml7INU+nsSmdQsBP4HTYoqAIhCjCKRvWrzUcR0m2Bt2LwR2Q1jawcWapZlvvvnGeAtBHrBN6du3rxAJt2fPniYyLgHkIBmeRgARwt4FMrJgwQIhrgthDRDaeOCBB0zcFexnsFEhuB0RfgmYh7EvY0EzA7EiyB7ReOln5MiRMnPmTON9xD0CcSEfexjusY+xy1x99dXSrVs3wTiYcZMHcSIuDGUhTnZZ7v\/44w+jGWLOBPQbMWKEbN26lSwVLwgogfECjHsyBrzYv5Cuti+goKIIKAKKQOAIsMRCJF7sT9wlMTHRROLFEJYX\/+rVq4UX+YsvviiQBIgNpKBQoUJeO8aTiTgz1IXMsPUAmh2ENkgjH6LEOCARP\/zwgzHi\/eWXXwTNDjFqcKPGo4g8xgn5YBsA0u3OucauZsqUKeJexrIsIdge6WPGjBG2FyASL\/2Rhr2Nsz1ctxkX4ybQH7FfiFdj96Xn1AgogUmNSaoUlo9mrd4nZU5tM3k5Ssabsx4UAUUg8xDQnhUBRSBrI6AExo\/nv3HvMVOq9Mnt5qwHRUARUAQUAUVAEchcBJTA+IE\/y0cUw32ac96qTTipZGkEfE8edTXr6r5Laa4ioAgoAopAehFQAhMAcnE3PysVx52RuJsTA6ilRRUBRUARUAQUAUUgoxFQAuMHorYHUoNKcX6UDk8R7UURUASyJgLEQVHZJFkZg6z5yU89ayUwqTHRFEVAEVAEIg4BYovUrVtXcPVt3LixqGRdDPgM8FngMxFxH9QwDiidBCaMI\/SzK6I3YndQp04dqVixoommyOZd+OjbQYr8bCpVMbyQUiVqgiKgCCgCYUSAl9XLL78s7OuTmTJ8+HDjcowLcmaOI6v3zWchjB+\/iOwqJggM8QJatWplNuHavXu3ARrSQhAjYgbgW+8riqKp4Mfh3KJ5\/CilRRQBEYJmTZgwQaFQBDIUAUgMgdYyW9i9ObPHELX916snGTF2PgsZ+uGKwsainsCwdwZ7aXCuUaOGfP\/990IgIMjLPffcI9myZROCCI0dOzbdj+d0vuKmrhIYA4MeFAFFQBFQBBSBTEcg6gkMm3oRtZA9MV577TW58MILxbIsE83x8ccflx49esiZM2eEsM2QnEARPyfXSXlvxyPy+8bmcnTZ9ECra3lFQBFQBCIJAR2LIhAzCEQ1gWFPDUJL8zTatm2bvPkW94hlWcJmYWXLljUbhdl7UJDnr1xc8Jiw\/xHbB+j+R\/6ipuUUAUVAEVAEFIHQIhDVBGbjxo2ybt06s5spa4qeoCpZsqRUqVLFaGFmz57tqYjPtJbFDpr8v69+1Zz1oAgoAkEgoFUVAUVAEcggBKKawGzZskX27t0r7GB6zjnneIQkd+7cwo6jZK5fv17Y6ZNrf6RU7pOCBmZLjlKys3Q9f6pomUxGYNu2bcImaZwzeSjavR8I8Jz0efkBVAQV0WcWQQ\/Dy1D2b\/xXbCl4pqQUcImXolGdHNUEZvv2pL2J8ubNK5AYb0\/CJjeUP3r0qLdiqdKL7V9l0rZmLyXl49QDyYAR4Qe+XHHv9DJMTY4wBPR5RdgD8WM4GfnM1s8+JLb8+fle8SS\/vrxdnPLdAxvFk3xyw2oJRoZe+peEWp4\/508Jhzjn8U3bQ3LrkWF+PNnoKxLVBAYbGCDPkSOHMdzl2pOUKlXKJJ88edIsJZmbAA5bcpwj6oEUAGBaNGgE+PVkf7FH2nnt5JWCeHrZBJq2Z1Z+Oe9kc48vrkDb8lZ+0YglKV6AzpdhJF17eilHYtqu0WXl2mMDZPr9x4MiDJCNT29YI7Z4m+uvg10ExiHenvOG2YclGOFvLtQS9BeDnw0ULp9LbDlo7RDEz6pRVSyqCYyNNBoWtDD2vfu5QIEC7kkB32cIgQm4V62QlRA4fWST7Fm5Rnb98Yvw0rW\/2CPtPKbjcUG8vXACSV\/00hlpcvwB4RxIvUDK\/tD3jLi\/BCPx3tuLORLTS5+qHhRZsIlGwWI7pEyVpUYubLFSql132MhFt8SJLQ16nRBbGj16jlzzopUsrd8oL7Z0+DS3IB2\/qijpkR6\/FBCk1+8XSKik79qi0m\/7RSEX5\/iv\/7qAfJ6ve0x+lcYEgQn1k8l2ZHeW3ncjmvYcsT8LmT1mxrF58+Y0Pze8nD5xqb7frDpZXkzYI+82PiTDri4is4bmFOeXu\/0lb5+lxFKxJfu5+yQzpGjDw5IR8k+OqRnSjq+xJHQ4IdEgNR+3JNKlxG2bZWGZp6Ti\/X9I0zdzByVtxuxz1c9lpM6g+lL72fJGnBhUuuscsYVnWPzKOLGlaAPXZ\/Cs5LwgvyDp\/Vs4kj+XIGgrQiVb9kia3wkZ\/d3Fd1EsiKc5xASBScu2JRDDXSdIuwtXlrFxbeWjLUV035Eo2XuFPUJ4hpwbZ+KYP\/vsM3nppZfS\/Nw89thj5pfswd0lGbZR9fLluTX7UplxYKp8n+cpef9Ikjy36Smx5QNXmi3v7e4smSEv\/dFRMkJm5H4jQ9rxNZYnv20v0SD3Db1eIl1e+KaXLNy\/VPoN7y+3920VlDS\/4XGxJTP\/XmO5b74LY3XfpKgmMAULFjRf+mnZtmB0RsG0bGUo45T4Wg3l9mfelDcH\/y\/T9x\/J6vt+xOr8h3zez6i7r3epeZt\/c0Q4Ix3HV5RHv7hOej83Wl55K0liFQOd12j9fhkd6RhE9\/hidd+kqCYwRN+1LEvwLNq\/f7+Te6S4RkNDQlq2MpRxF\/abIMaMSr0M2b9DcUyJY7X6FaXC5QWEs2KTEhvFQ\/HQz0DGfAZ4j7m\/22LhPqoJDA+lWLFiAnmxSYr7Qzl+\/LgQ8I70ChUqSEYY9NKWiiKgCCgC4UJA+1EEFIHUCEQ1gcE9miB1J06cEG\/bBOzYsUNWrlxp3Kwvv\/zy1AhoiiKgCCgCioAioAhEHQJRTWCwgbn++usN6F9\/\/bWsXbvWXNsHNnFkF2q8QRISEuSSSy6xs\/SsCCgCfiOgBRUBRUARiDwEoprAAGeLFi3k\/PPPF7YVuPPOO+XXX3+VU6dOmS0GXnjhBXn77beN9qVLly7Cpo7UUVEEFAFFQBFQBBSB6EYg6gkMhryDBg2SEiVKGP96SMx5550ntWvXlg8++MA8nQ4dOphdqc2NHqIOAR2wIqAIKAKKgCLgjkDUExgmdPHFF8uECROka9euglEvadmzZzck5v3335f+\/ftLzpw5SVZRBBQBRUARUAQUgRhAICYIDM+hZMmS0q9fP5k\/f76sWbNG\/vnnHxk3bpw0b95cIDOUSZ9oLUVAEVAEFAFFQBGINARihsBEGrA6HkVAEVAEFAFFIEsjEOLJK4EJMcDavCKgCCgCioAioAhkPAJKYDIeU21REVAEFAFFIPMR0BHEOAJKYGL8Aev0FAFFQBFQBBSBWERACUwsPlWdkyKgCGQ+AjoCRUARCCkCSmBCCq82rggoAoqAIqAIKAKhQEAJjAdUDx06JMSPadiwoVSsWFEqVaokV111lYwfP97sfC36L6wITJw40TwHnoUvmTx5ssdxLV682MQIuuCCC0w7NWrUkMcee8y423usEBuJYZ\/F3r175YYbbpDbbrtN+BvyNQDy0\/s3FkxdX2PKann+Pi\/+fi666CLzt+Pr7+\/dd9\/1CKE+L4+w+JUIdh999JFcc801QoBW8Of7q3PnzjJv3jxhuxxvDVE31v\/GlMC4PX32Tbr55puFbQjYnoBsPiSrVq2SRx99VHr27CkHDx4kWSVMCKxevTpdPfHc+AO+6aabZOrUqfLvv\/+adg4fPixffvmlsI8W5Mgk6iEoBNhQ9eWXXxZedmk1FMzfWDB10xpXVsoP5HnxPcjLMD346PNKD2pJdYhlxndXYmKi\/PXXX2aLHHL4\/vrll1+ECPNPP\/208CzF7V8wuAdT120YIb9VAuOA+OjRoyZqLx+WcuXKyahRo0xAvD\/\/\/FP4EOXPn19mzJghr776qk\/m62gy615m0MxPnjxp\/nhpDmI5d+5c8SaNGjWiWLLMnDlTXnnlFTl9+rTcfvvtsmDBAoEMTZo0SerWrSt8EfBc\/\/777+Q6ehE4AnyBDh06VD7\/\/PM0KwfzNxZM3TQHljPxPkQAABAASURBVIUKBPK8gGXZsmWcBI00e815+\/vjb8wUPHvQ53UWiHSc0I49\/vjjwndTwYIFZfDgwbJixQqjNeYZ3HrrrabVMWPGyIgRI1K8j4LBPZi6ZkBhPiiBcQCOSg6CUqBAAeHXJC9Eovhyf8cddxhyY1mW2baAD5ajql6GCIEDBw6YP1qar1evnhBx2Zvkzp2bYkb4Q\/zwww+N1gX1K79U4uLizMaebP7JJp9sQbFz504ZPXp0ii8A04Ae\/EKAL9pHHnlE3nzzTb8wDOZvLJi6fk0mCxQK9HkdP348+QcEy0hsiOvt74\/vSSeE+rycaAR2zY8vfjiD6bBhw6Rdu3Zif7\/xDPr37y+33HKL+Zv7+OOPzT6Adg\/B4B5MXbv\/cJ5jlcAEjCG\/9LFx4ddJ48aNzT5K7o1ceeWVUrNmTdm9e7fo0oM7OqG5B+tdu3YJf8gJCQl+d7J8+XJZuHCh5MmTRyCf7nthQWbuuusuQ2imTJmS4gvA706ycMFTp07Jt99+K\/xNfP\/995IvXz6pUKGCT0SC+RsLpq7PQWWRzPQ8L6BBS8kSEtfVqlXj5Jfo8\/ILJo+FWPqeNm2aISfe3kV8n0Fg+F7k+WDiQGPB4B5MXfrODFECcxZ1fpksXbrU3NWvX19y5Mhhrp0HVHn8CiGNlyN\/3FyrhA6BTZs2GcJYokQJKVOmjN8dLVq0yBiSxsfHGyNsTxUx6mXzT+cXgKdympYaAQgiWq09e\/YImixsivhCTV3yv5Rg\/saCqfvfCLLuVXqeF2ht27ZNtm\/fLhD+tAgq5W3R52UjEfj5yJEjApkoXLiw8B3l6V1Eq6VLlzbPBcIDQSUtGNz\/qysSLe9AJTA8dZfwh8ovfZitrz\/UypUru0qLrF+\/3thQmBs9hAyBlStXml8iEBG+hHlJ8keNNf7ll18ugwYNEpaZ3AfAejFpkB5sl7h2F8hL8eLFTfv0456v994RYGm1SZMmxhgazSXLct5LJ+UE8zcWTN2k3rP2MT3PC8Qg97zYzjnnHPN31q1bN8ELhr+\/WrVqSd++fQWjT8o6RZ+XE43Arvm+euONN+SPP\/6Q++67z2vlrVu3Cs\/Gsizh+YrrXzC4B1PX1XWm\/FcCcxb2\/fv3y7Fjx8ySA8z3bHKqE3\/IJGKVD+HhWiU0CPDLwrY1wnCNJZ\/58+cbuxZ65A\/uvffeM8sYTu8X1u337dtHESlSpEjy2rFJcBxy5cpl8kmCkHJW8Q+BqlWrCl+yvMQsy\/KrUjB\/Y8HU9WtwMV4oPc8LSHBo4MwSRceOHYXlVlvzzDMZO3asXH311amW1MlL7\/cp\/an4RoDvRsJG8B4qVapUspY5GNyDqet7tKHLVQJzFlv+2LB\/QV1ns9mzWSlOkBvsKlDx2Wq7FAX0JsMQQLOydu1a0x5\/sHg5\/Pbbb8aoFwO3559\/XljWwxAXF3f7lyDPETUsFdGocfYkPMeiRYuaLOqYCz2EDIFg\/saCqRuyCcV4w3zH2QSG77qWLVsaAoMnHxpL4r7grQmheeaZZ4RlWxsSfV42EqE584Pts88+M41jh1a+fHlzHQzuwdQ1nWfCQQmMG+gYRbGs4JacfMtLz9dLMbmgXgSNAL8usmXLZgx4CTyXmJgo2MLQMM8JV8KRI0caEsOXKjFfIDrk2+JrORCyyvO0y+o5PAjw7NL7NxZMXf9mp6VsBPgRAInhRxtLt6+\/\/rokJCQYw3e+AwnuiQcfaRjbv\/baa6kCferzstHMuDM\/1Pg+BHOCrLKsZ1kptaDB4B5M3YybpX8tKYHxDyctlQkI4C6IcSjalu7du5svTvdh4BVG9FfScYFnWYlrFUVAEQgOgUKFCglaFmwx0HZCWtxbRANz5513mmTKsdRkbvQQEgSwScIuhh9s2PAR7oPvyZB0FgWNKoFxe0j86vdl22Kr2dyq6W0mIWBZlglKR\/d4xLg\/u\/Xr15PlUfh1yfP0mJlJiVmh22D+xoKpmxWwzYw54oXGr3YilOOx5ByDPi8nGsFds5zetWtXE+2apXMcGPgB56nVYHAPpq6nsYQyTQnMWXRZSuAXBi811nvPJqc62YZOLD\/4spVJVVETQoYAf8w8O\/sLlGviktChL9sWyAukh3LU4awSOgSC+RsLpm7oZqQtgwB\/azwfrm0NDPf8Ten3KagEL9i8oOnCJgnNC5GvmzdvnqrhYHAPpm6qgYQpQQnMWaAx5uRXBC81SMrZ5FQn+xcGZX2t46eqGBUJkTdIPIow0vVFKiEpfFHyB8h6PRErbVsZvJFow9PM2BuJfPJ82cqQrxI8AsH8jQVTN\/iRZ90W+Nvi74+zNxTIs\/8+8YihnD4vUMgYYR83gnESE4slO7YOYFsHT60Hg3swdT2NJRxpSmDOoswfHi7S\/DH6Wnawf2HwwsNf\/2x1PYUAAdbfL7zwQmnbtq3HWBN2l6wLY7wLqcRtmvTq1atzEvLwkjA3bgeM4FhysixLqlSp4partxmNQDB\/Y8HUzeh5ZJX2cNPl74J4P04PI\/f5o8Vk2QGNi\/2jTp+XO0qB3\/OdNmHChOQNhC+44AKzP5\/93eapxWBwD6aup7GEIy2iCEw4JuytDwzWzjvvPJM9Z84cEwnR3DgOLFFgUErSJZdcIkpgQCJ0wh8qX4oY5qJC9dQTex79\/PPPJotYF\/xC4YY\/dgjNunXrzAaOpLkL6lhIDMHu7ACF7mX0PuMQCOZvLJi6GTeDrNUSP9IgJGilZ82a5XHyvGQhOvzwwxsJwkNBfV6gEJyw6SzeRmiKL7vsMkNewNhXq8HgHkxdX2MKZZ4SmLPoYtPChlm8MH\/55Rezc\/HZrOQTL0p+iWBzccUVVySn60VoEODLEDdBviQ\/\/fRTgUC69zR9+nSzO7VlWXLjjTdK3rx5TRHIDCSTL9+PPvpI+II1GWcPRLAcPny4icKLOpaw3Gez9BQiBIL5GwumboimE\/PN8mOgZs2aZp5oAljCMDeOAz8s2AuLJALa2Uu3+rxAJP1CAM\/ExEQTtLNBgwbyzjvvJIeQ8NKqSQ4G92Dqms4z4aAExgF67dq1pV69emYPHdx2CZHOiw\/16KhRo6Rfv37mhceLEu2Ao6pehgABvgzvv\/9+IWLuvHnzhGfCHzaEBjLDH\/XDDz9s\/sivueYa4QvUHgZEhsB31OULtlevXrJx40bz\/GijR48expqfPrp06eJx7yvRfxmOQDB\/Y8HUzfCJZIEG+Rvq2bOnYDRqe8AQSBJ7F+zK+H7EsBQtJp5IXDth0eflRMP\/a+z5MNLF9ggtMhHIeQ\/t2LFDvAnvKLuHYHAPpq7dfzjPSmAcaPNhId4Byw+8IInuihaADRz79+9vXpSsB\/PStCzLUVMvQ4UApKR3795mrw++PLlHK8MXJjEQUK\/yTHhuaM+c40BLZtdFzU056tIGWxKwBMivHH\/28XG2q9fpRyCYv7Fg6qZ\/xFm7JhqY5557ziyXQ\/w7dOggLLVjm8b3I9+TfF++9dZbEhcXlwIsfV4p4PD7BrKIGQMVICZ33323+WFdz\/Xj2pt88sknFDcSDO7B1DWdh\/mgBMYNcIICEaKZTcqwjSDbsizBRmLw4MHy9ttvm8ivpKuEHgHLsoRIk2hRICR4GdErmhW+XPml8sEHH3h8JpZlCXET2K8Fl0PqUBfighbtm2++kVatWpGkEkYEgvkbC6ZuGKcYU13xN4I9xs033yzYxDA5QkhAXPgBQLBJngvp7kK6fp+6o+L7HicSnAt8l\/KdGwzuwdT1PaqMz1UC4wFTjJlYrpg5c6bZd4eohz\/99JNgI4Na1UMVTQoxAmhJ2CqAaJ9r1qwRDHBRYfPlypept+4tyxI2HITkUIe6S5YsEbQ37KjrrZ6mB44Ay3LgS3h5fsn5aiGYvzEPdUX\/Pn2h7TkvkOfFS+3FF18UlnJ5xv\/884\/88MMPgntvWt+J+rw84+8ttUWLFua9A87+Cs\/Svb1gcA+mrvs4QnmvBCaU6GrbioAioAgoAoqAIhASBJTAhARWbTRLIaCTVQQUAUVAEQg7Akpgwg65dqgIKAKKgCKgCCgCwSKgBCZYBDO\/vo5AEVAEFAFFQBHIcggogclyj1wnrAgoAoqAIqAIRD8CwROY6MdAZ6AIKAKKgCKgCCgCUYaAEpgoe2A63KyJwNatW+Wll14SYm\/g\/l2jRg1p3LixcRG\/\/PLLhUB+ROkMBzqEFyCQIH0SFdmfPokFQlBIzv6Uz4gybAZK4MLHH39ciCOEC35GtOutDSKozp07Vwi7T1RanhMBFL2V13RFQBEIDgElMMHhp7UVgbAgwF5NhHVnfyc6ZGsE9uxauHChPPHEE8J+T9dee62wVxf5KkkIEI+mU6dOAjbEA0pKDc2RvWSIlNqmTRuhz9D0oq0qAoqAjYASGBsJPSsCUYiAZVnSunVr4aXJnjRvvfWWsE9NKKfC5pds4sfeYJbl35Yat956q6xcuVI4h3Js2nYgCGhZRSC6EVACE93PT0evCIhlWcJSCVCwj8qBAwe4VFEEFAFFIKYRUAIT049XJ5fVECAEOBtcfvHFF8LeUSwt3XLLLYbgsBUDeLApHxuSEn4cbQr7frHzLXnI0aNHhd3X0ZawbIWtzWuvvSakk79371759NNPhTTuETb1Y3O\/p556SoYPHy433HCDEHqePGTz5s1mH7Fx48Zxa+TUqVPy7bffSvv27eWRRx4x46VPxkeBEydOyK+\/\/mr2s8KmhLpXXXWVsK0E5bZs2UKxgIV+sYmhjfvvv1\/A4MknnxTmRZ\/YyjCXK6+80ozvmmuuMfhdeumlMm3aNGF81D3vvPPkoosuMtsYBDwIraAIKAJBI6AEJmgItQFFIHMRgFhgPMooMK7dtm2bMSRFGwMR4aVfrlw5Yc8obGQefPBBs0EmRq68uL\/66ivhTDu8wP\/3v\/\/J0qVL5eOPPxY2y6xbt6688cYbAoGgH8jG008\/LZs2baJLIxAa7vv27St33XWXsJfY\/v37TR5Grffdd5+wGSrLXCRi\/Pvhhx+adl999VUZMmSIfP3118KGmx07dky25cGGheUqDIc3bNgg7H9FO5AMm5DRnr9i98t4qf\/mm2\/K7bffLmPGjJHXX3\/dNMNmn3\/++afZj+b33383ZI69lvLkySMPPfSQgBv1SCtRooS88847Ys\/VNKAHRUARCAsCSmDCArN2EnkIxMaI0JxAJtBUQF4gDrVr15b69eubCd57771y9913y4wZM8zGeyNGjJALL7zQCAWoU7JkSfntt99k1apVwosbz5mbb75ZcubMaZaneMFXq1ZNSpUqJRipQmAKFixI9WRZv369rF692rRBIuXwmOIa+xyMjrm2BS0GBKJRo0YCuSKdNhkrS2Bod\/DqYRPBfPnyCSTqnnvuEQgNbTMWbGoOHTpEVb8FkgUxY370R0XGd9lll8m5554rGOKWKVNGihcvbsYF8YKk4FFEGcp36dJFSIuPjxfwQ7sULg8w+ldRBBR5bVjsAAAE8klEQVSBJASUwCThoEdFIKoQeO+994wb9Y033ii8PF955RUZPXq08PJ1TqRIkSLJt5TDawmtAlqO2267zSzdoPUoXLiw0SKgoWF34QoVKiTX4yWNFgVNTnKi2wXLVSwj4X0zbNgwyePSVrDs41Ys+XbFihWCNuacc85JTuOiatWqhjiw2zHkjLSMFEjakSNHzJKQ3S4kCfduNEeW5dko2bIsyZYt5delZSWlYTR97Ngx0X+KgCIQXgRS\/kWGt+8s3ZtOXhEIBgG0EbhRI9i5tG3bViAevtqEMKDduO666wzZgfDw4p4yZYrQDrYgEAtfbXjLg8AQF4Z8bF+wgYGEcO9JWOYiHS0LZ1uYA6QGzUoolmXQ2rj3afetZ0VAEYguBJTARNfz0tEqAkEjsG7dOnF\/iWPYig0My0ZoPlgSCqSjw4cPy\/XXXy+QITQ7LCeh0di4caPHZlgCImPNmjWCXQrXTmGpCK2QMy0jrpkfmiLGlxHtaRuKgCIQdgSSO1QCkwyFXigCsY0ASyUI9jJLlixJMdkffvjBGLJWr15dWA4ZO3asYNBrF2JpCc2KO\/Gx819++WVZvny5YE+DETB2MngJsWRjl3GesdNhLBjj2toY8tEQQXrw+HFfDiM\/WCEaMCQGzRNExm4POxbi2qD5sdP0rAgoApGNgBKYyH4+OjpFwCMC3oiEx8JnE4sWLWrcm3lJd+vWTT7\/\/HNhmQf7mW+++ca4Mzdr1sy4KePtAwlhyWXixIny3HPPSbt27YyR69nmUpzQwOCVgzbFsizBwJWlIKctjbMChrvYy+BZhPcR9cjHywmbEgxlMaglLSOlZs2axhB53rx5xlNqwYIFxoD5gQcekFatWhkj4YzsT9uKQQR0ShGDgBKYiHkUOhBFwDsC7IX0wQcfGPdmSqEhwYaFdO5tgdjgcQTpIA2PG0gBS0Tcd+7cWZ599lljZMsWBHjjQD5wY8YrB60H\/WDT8vPPP5tloS+\/\/FIgOcRfwUaGGDNoL+bPn29ioKCpwWuH5SO0L7hHjxo1yrhGJyQkCCSB8dL\/jz\/+aAgD15AovI1wYe7QoYPgqTRp0iRDrCAaEC3IDRoaiBaxW9CUYPND2rJlywSPKfqnPX+EpSn6pD9cxYnngls27t94RLGMxrxpe\/v27QKx27Nnj+kHbRHzxu2ccUydOlVIY5wQPtL8GYOWUQQUgYxBQAlMxuCorSgCIUWAvZCIQYKbM3YjGN1ia0K6s2O0Fk2aNDEvXspBJHA7JgYM5Vg+ueOOO4S4KuTzAn7mmWeEAHjkI2hH8CT666+\/BIHQQETIwwW7f\/\/+JkbKjBkzBM8k2oQAsAxFW7hCs0RDv5ZlCctFtEd\/vPxxibYsy7ho48LMXChPzBnK2X1BNnALRwsEWSIgH8tOBJkjjfgwGC\/TP2PzV+Li4mTgwIHCMhrEiPg2derUMdUxIsYAmbaZOyQLzVWLFi2MfQ9zYI6Mo3nz5iYNexpIIWmmkdAetHVFQBE4i4ASmLNA6EkRUAQUAUVAEVAEogcBJTDR86x0pIpA5iMQZSNgyQctFMtPaJtCOXyW71iuoy+WvkLZl7atCCgCIkpg9FOgCCgCMYkAez2x5IONy7XXXiu1atUK6TxZvmPZjL7eeusts8zG0lNIO9XGFYEsjIASmCz88KNw6jpkRUARUAQUAUXAIPB\/AAAA\/\/8EpAaxAAAABklEQVQDAClejGI58F9TAAAAAElFTkSuQmCC","height":337,"width":560}}
%---
%[output:224c4b3c]
%   data: {"dataType":"text","outputData":{"text":"\n================ RESULTS SUMMARY ================\n","truncated":false}}
%---
%[output:6c8ed15d]
%   data: {"dataType":"text","outputData":{"text":"Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n","truncated":false}}
%---
%[output:8be17f78]
%   data: {"dataType":"text","outputData":{"text":"DQN Agent     |        139.26 |       28.26 |               590.16 (Steps: 145.4)\n","truncated":false}}
%---
%[output:69b96b92]
%   data: {"dataType":"text","outputData":{"text":"Brute Force   |         31.49 |       13.14 |                99.29\n","truncated":false}}
%---
%[output:4c70112c]
%   data: {"dataType":"text","outputData":{"text":"2-D MUSIC     |       1422.82 |     1563.84 |              1984.32\n","truncated":false}}
%---
%[output:489befcc]
%   data: {"dataType":"text","outputData":{"text":"2-D ESPRIT    |       1417.54 |     1548.03 |              1984.13\n","truncated":false}}
%---
