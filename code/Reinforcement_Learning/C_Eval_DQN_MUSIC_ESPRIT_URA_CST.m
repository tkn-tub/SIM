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

if ~isfield(EnvPars, 'plotCalibrationSector') || ... %[output:group:021cb0f7]
        EnvPars.plotCalibrationSector

    figure; %[output:48267f95]
    scatter( ... %[output:48267f95]
        X_cal, ... %[output:48267f95]
        Y_cal, ... %[output:48267f95]
        28, ... %[output:48267f95]
        'filled'); %[output:48267f95]

    hold on; %[output:48267f95]

    plot( ... %[output:48267f95]
        EnvPars.pos_SIM(1), ... %[output:48267f95]
        EnvPars.pos_SIM(2), ... %[output:48267f95]
        'kp', ... %[output:48267f95]
        'MarkerSize', 14, ... %[output:48267f95]
        'MarkerFaceColor', 'k'); %[output:48267f95]

    % Local-cell boundary
    rectangle( ... %[output:48267f95]
        'Position', ... %[output:48267f95]
        [0, 0, EnvPars.x_max, EnvPars.y_max], ... %[output:48267f95]
        'LineWidth', 1.5); %[output:48267f95]

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

        plot( ... %[output:48267f95]
            [EnvPars.pos_SIM(1), xBoundary], ... %[output:48267f95]
            [EnvPars.pos_SIM(2), yBoundary], ... %[output:48267f95]
            '--', ... %[output:48267f95]
            'LineWidth', 1.2); %[output:48267f95]
    end

    axis equal; %[output:48267f95]
    grid on; %[output:48267f95]

    xlim([0, EnvPars.x_max]); %[output:48267f95]
    ylim([0, EnvPars.y_max]); %[output:48267f95]

    xlabel('x position [m]'); %[output:48267f95]
    ylabel('y position [m]'); %[output:48267f95]

    title(sprintf( ... %[output:48267f95]
        'Calibration positions: sector %d, boresight %.0f^\\circ', ... %[output:48267f95]
        selectedSector, sectorBoresight_deg)); %[output:48267f95]

    legend( ... %[output:48267f95]
        'Calibration positions', ... %[output:48267f95]
        'SIM/gNB', ... %[output:48267f95]
        'Cell boundary', ... %[output:48267f95]
        'Sector boundary', ... %[output:48267f95]
        'Location', 'best'); %[output:48267f95]
end %[output:group:021cb0f7]

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

for i = 1:N_eval %[output:group:043c953b]
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval); %[output:82b013a1]
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
end %[output:group:043c953b]
%%
%[text] ### Figures
% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
%plot limits
xlim_1=0;
xlim_2=150;

figure; hold on; box on; grid on; %[output:65c66fc1]
h_dqn = cdfplot(err_dqn_ref);    set(h_dqn, 'LineWidth', 1.5); %[output:65c66fc1]
% h_dqn = cdfplot(err_dqn);    set(h_dqn, 'LineWidth', 1.5);
h_bf  = cdfplot(err_bf);     set(h_bf, 'LineWidth', 1.5, 'LineStyle', '--'); %[output:65c66fc1]
h_mus = cdfplot(err_music);  set(h_mus, 'LineWidth', 1.5); %[output:65c66fc1]
h_esp = cdfplot(err_esprit); set(h_esp, 'LineWidth', 1.5); %[output:65c66fc1]

%patch
confidence=90;%confidence level
[F,X]=ecdf(err_dqn_ref);
[~,idx]=min(abs(F-confidence/100)) %[output:8545d6f8]
x_coords = [0  X(idx) X(idx) 0];
y_coords = [confidence/100 confidence/100 1  1];
patch(x_coords, y_coords, 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); %[output:65c66fc1]
plot([xlim_1 xlim_2],[confidence/100 confidence/100],'--','Color',[0.4 0.4 0.4]); %[output:65c66fc1]
plot([X(idx) X(idx)],[0 1],'--','Color',[0.4 0.4 0.4]); %[output:65c66fc1]
text(5,0.9,strcat('$',num2str(confidence),'\%$ Confidence'),'Interpreter','latex','FontSize',font-2); %[output:65c66fc1]
text(X(idx),0.6,strcat('$',num2str(X(idx),4),'$ cm'),'Interpreter','latex','FontSize',font-2,'Rotation',90); %[output:65c66fc1]
set(gca,'FontSize',font); %[output:65c66fc1]

xlim([0, 200]); %[output:65c66fc1]
yticks([0,0.2,0.4,0.6,0.8,1]) %[output:65c66fc1]
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', 14); %[output:65c66fc1]
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', 14); %[output:65c66fc1]
legend({'DQN Agent', 'Brute Force', '2D MUSIC', '2D ESPRIT'}, ... %[output:65c66fc1]
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12); %[output:65c66fc1]
title(''); %[output:65c66fc1]

% Print Summary
fprintf('\n================ RESULTS SUMMARY ================\n'); %[output:0f6850ff]
fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n'); %[output:880efd45]
fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_dqn), median(err_dqn), prctile(err_dqn, 95), mean(steps_dqn)); %[output:8c27b297]
fprintf('Brute Force   | %13.2f | %11.2f | %20.2f\n', mean(err_bf), median(err_bf), prctile(err_bf, 95)); %[output:246142fb]
fprintf('2-D MUSIC     | %13.2f | %11.2f | %20.2f\n', mean(err_music), median(err_music), prctile(err_music, 95)); %[output:0a586031]
fprintf('2-D ESPRIT    | %13.2f | %11.2f | %20.2f\n', mean(err_esprit), median(err_esprit), prctile(err_esprit, 95)); %[output:7f30ae64]

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
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     x_max: 20\n                     y_max: 20\n              calSpacing_m: 0.5000\n      sectorBoresights_deg: [0 120 240]\n            selectedSector: 1\n       sectorHalfWidth_deg: 60\n                 MU_margin: 0\n                     N_cal: 400\n              channelModel: 'rician_los_nlos'\n                    fc_GHz: 28\n                    V_hall: 72000\n                    S_hall: 18000\n                   mu_lgDS: -7.2781\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 15\n                sigma_K_dB: 8\n                        KR: 1.7748\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 1\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         M: 196\n                       M_x: 14\n                       M_y: 14\n                         T: 3600\n                       T_x: 60\n                       T_y: 60\n                    SNR_dB: 34.6606\n                 theta_min: 1.7657\n                 theta_max: 4.5175\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 24\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [10 10 4]\n                    pos_MU: [63.2459 30.4489 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 … ] (1×3600 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 … ] (1×3600 double)\n                      h_MU: 1.5000\n                    L_hall: 120\n                    W_hall: 60\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 180\n                 tolerance: 0.0262\n         StopTrainingValue: 58\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9800\n"}}
%---
%[output:6228f180]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 2.366%n","truncated":false}}
%---
%[output:15c951ad]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nFull local grid: 40 x 40 = 1600 candidate positions\nSelected sector 1: boresight = 0.0 deg, angular interval = [-60.0, 60.0] deg\nCalibration positions retained in sector: 570\nCalibration complete.\n\n","truncated":false}}
%---
%[output:48267f95]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7tXQnYTVXbfkwhFDKV1xSl0ZjhMzQYo6hUKBk+hYioRLPwVYQQMpVP5iljSIm+QqGQREKpjKWICpn+617+fTrnvOecdc5+995nn73vdV3vxfuuvaZ7PWuvez\/redaT6dy5c+eEiQgQASJABIgAESACKYRAJhKYFJotdpUIEAEiQASIABFQCJDAUBCIABEgAkSACBCBlEOABCblpowdJgJEgAgQASsR+OOPP+Txxx+XX3\/9VQYNGiSXX365ldWzLpsQIIGxCVhWSwSIABEgAqmBwNtvv606es0118jkyZPl9ddfT42O+7yXJDA+FwAOnwgQASLgdwQmTJggWbJkkauvvlqmTJlCApMiAkECkyIT5fZu4svlxIkTsmrVKrnsssvUl0zw7w8++KCsXLlSBg8eLM2aNbN9OOPHj5dXXnlFmjdvLgMGDLC9vWQ08OWXX8pdd90lxYoVk\/\/9739RuxA+F8noK9u0FgGn15OZ3pvpY7LW7dGjR6Vnz57y+++\/q\/cGj5DMzLjzZUhgnMfcVS2uWLFCoD7dvHmz\/PXXX3LJJZdIxYoV5aGHHlL\/xpvCN8mRI0fK6dOnpX379nLRRReJmZdZvG3jObSHn2+++UYV++KLL+STTz6R6667TurVq5dIVSnz7IEDB2TGjBly8cUXy7\/\/\/e\/AuO+991555513AvMXPhcpM0AbOxouLzY2parGWsBX\/tChQ+XkyZMyZswYadCggelm7V5PpjsWVHD+\/Pmye\/duadiwodJsxJPiJTBm5u\/bb7+V\/\/znP7Jx40bJlCmT3HjjjdKnTx8pWLBgPF3jMy5EgATGhZPiVJfeeOMNpRFBSktLk5IlS8qePXvUSydbtmwyatSouDd\/3Ve+mRfu2bNn1YsGP7p0xx13yPbt2wMERve8V\/NffvllefPNN0MIjFfHmpFxWSUvZ86cUUcPsdIvv\/wibdu2lZ07d0rOnDnl2LFjriYw8YwpI9jHKhsvgUl0\/kAab7nlFgHpr1u3rkDjsn79eqlWrZpMnz7druGwXpsRIIGxGWC3Vv\/999+rhYz07LPPKg2JkaZOnSrPP\/+8lC9fXubNm6f+jHPhiRMnyk8\/\/ST58uWTxo0bS+\/evSV79uwqP94jpH79+snatWvlww8\/lAsvvFBpeh5++GFVx4gRI9QX6qOPPio7duyQDz74QD766COlFRo4cKC89957yksAZAuanVatWqkXUYUKFUJgfvLJJyVr1qzpjpD279+vPAw+\/vhjVa5IkSKCFyHaA2FDuuKKK9S\/aKtXr17y9ddfS\/HixeXFF1+UGjVqxDWdpUuXFtwP+d\/\/\/lf1e9euXVK0aFHp27ev1K5dO1DH3LlzFdn47rvvVPvQeMETwhgPCBwwwZfsvn37lCYLfXjuuefUV2P4ERLmxNBAoRG0Be1a+NzgZY4vWNR78OBBpcHB1+hTTz0V+Bp94IEHZM2aNfLWW2+p5zAXaL9jx44KeyRd\/\/CM0adx48bFTYYXL16sNnjIKMjBtddeK5hTQyOIdpEP7RM2pEKFCilZ6Ny5cwBbaN9ee+012bp1q+o3sHj66aeVvEaSF5SNRz4wt5kzZ1Zzi7mqU6eO9ogSX\/yQA8j+Cy+8IOvWrbOMwMRaTwADmshhw4YprQO0QOh\/hw4d5M4771RYYS3ed999UrNmTfUDuUCdOOZ9\/\/33lS0IiFeOHDmUxgj9z507tyqrm6fwjxZ4+uBdA1nCPEDGfvvtN0UgQLxbtmwpBoFBn3CMg3n++++\/pVGjRmr9QHajzV+sxblo0SLp3r27kkHIIkjaTTfdpNYV1vqVV14Z19rmQ+5CgATGXfPhWG9Gjx6tNnNsbu+++266dqGFgUYGaeHChdKjRw\/1QunSpYva0GDr8sgjj8gTTzyREIHBxnvrrbeqFxg2AbxIpk2bJtWrV1cvFtirYLPPmzevlCtXTp1Lo5\/YrPDywUt0+PDhsnfvXpk0aZJUrVpV5YFgYLPD+TU2PPQv2AYGhOW2225T5VAHXoILFixQWhu8rA1NFFTdeElef\/31cvvttyuSsGTJEilQoIA6kjIIW6yJMurAS7Fbt26ybds2gbYrT548snr1arUBgFjghQxyhq\/zQ4cOqfFccMEFCm+UxZEDVN4gmugziM7YsWMDxDKcwCxfvlypxLERY26ADTbucAKDOcRLGzgBE2xyIJSlSpVSY8UYcSQFuxrY1+AIAP2CzIA8YONCnbr+mSEwIBzAvUyZMgoXzAU2YGy+wA7k2ZATjO2ee+6RpUuXqvFgvlu0aKHwRh0YBwjXzz\/\/rGSkbNmyipDPmjUrnbxA5hKRD6yFSpUqKYzRh1gJGzDwQ8ImbSWBibWevvrqK9U3zFnr1q0VdiCkWAuvvvqqyjNkCGQexAybOtYD\/o9\/Qd6x8YNMQiML4mMQQ908hRMYyDvkHjZyIJyffvqpOrpGfwzbOIPAYP6BGY6AMWenTp1Ssg1iE2m9646oDM0k3ieQfyRjHRhYOPbyZUOWIUACYxmUqVURvkZnzpypXvh48cdKn332mTpWAqHAxoXf77\/\/fkUCoEVAilcDE2xUC00CyAtepHiJGC8vfFFDS2K89PH19Oeff6qNHC9sbGj4MoTmBloSQwuD5w0NRLgq2iAMwVolfH1hE8QLHptjsPExXtJ4WePFWblyZcHXY7xfagYWwXYOTZo0UdqcIUOGKK0PNj58fYLEYdNAwtcpvkaNOcHY5syZI4899pjaiLEhb9q0SWkUsIFGMuI1NB7BNjDBc4PjC3zNAit8fUP7Am0RyCE2KWPcxuYTLB8GqQFZbNOmjcI+Vv8wpuPHjyuSiqMT3VELnsdmBVKMucamhv5BGwcCgzEbGhTMObRzhQsXVsbiVapUkRIlSqive5BGkCxoSLp27aqwBTGGtgnaNpBGyG6wvCQqHwYGia56qwlMrPVkbNCdOnVS2lIkQxMBrGBUb8gQ8oLlG2sLGhisS8guEsgD5A9ygw8E3TyFExh8FGAdg6jXqlVLrS1oFKFVDScwIFToH+bbWO+QbWiIIq133TyAuOBdFTxv0DiCzEKODS2wrh7muwsBEhh3zYdjvTEWr0EeYjUM8oINYMOGDcpKHwkvH3wl4SWXCIGBKh1GpkggUCBS0L6AyBikw3hRGX3CyxKEBUZ42KywIeIHX5X4qouHwOAIAps6XqogCkbCixREBkc5OA4wNnsYNxsaKGiM0Db6i41Sl4w6oLHBlz0SXvbYnEFG8GVrHCXhC9RQyc+ePVttNCCK0HKhPPqLzds4YkJ\/8fWKr2kzBAbkEy\/zcM2b0T8c6T3zzDMBo2tDq4Ex4OWPzQdjAEnQ9U+HU6R8aEugCcGmhnTVVVfJv\/71L0WYcfwBexLYLURKODYEgcV84djDmNPwZyPJS6LysWzZssBxYyLjtJrAxFpPkDFoHIOP73AEbBBmEGrINTzZQOpgE2IkEFpo\/CIlkGxo62LNE8oFExgcUYK0I0FuoY0MfwbrwngHgDRB04oEMgp5wxEX7mgxQ2BAZrGmIhEYvAvxgcCUegiQwKTenFnSY+NFgQ0CxwbhCWp5bBT58+dX6nio9rG53X333UpFjxeCGQITrK6FKhibJb7CYGMTyYAPGxZefjhKgHYAxx4gO\/hiToTAGF9gxgZtjBcvRRy5QLUOI79IxsiGViNRAmMcwaAtfPnjqA64YcOIRGAMQhesJYJWBEdKOOaBHQO+YKGJwEsdm3W4G7VOAwO7FnxxAkd8jRvJeMHjRY4XeiSja9hGwA4qWLMRq3\/xHLdFEubDhw+rMeNLH+OGDOTKlUsRUBA3g8BADsLbAPnDcRv6lQiByYh8JLIgrSYwsdaTQc6DCcyPP\/4oN998s+pyMIEJd8U3CAzIPo7KghPkD5qxWPOEI9BgGYK8G\/NmaBFRp6HVC9fABGuW8JEEDYnxnjBDYIwjJIN8o21DQwWtKNYRU+ohQAKTenNmSY\/xZYYNG1\/34V8gBrHAUQ5sI3AODU8gQ6NgEA18EUNljxTvERIM97ARIhlHSFBNv\/TSSxEJDNqELQQ0Gfh\/8FebUZfxQsMXOL4okaIdIQUfewEDvFgxNpzHY7xWEhjjOAb9wZEINlWow0EIjSMkEDGDzBjHetA2wPYFz6OP2IiQQOLwYodtAzZ4aKGiEZhgshXpCAkbPwgCjqOCj5BgNIwv63gIjK5\/kBvYf6B+aJBgV6FLICs4MkJZo2\/4asaXN\/CBASrmEHOOIwH8H7hAs4T5gy2EsTEFEy1sxJANkDfYwqBcsLwYR0hm5EM3puD8aAQG2g5oN2GgHo9brzE\/sdaTgYNx1Ip+QAsBXEBCYDMV7S4hHDthbQcfrwBjyA3kCfjr5ilYhiCnIM3QoBo2b9Di4gMC9l9mCEzw\/OnmAB9pOE6ElhXEFu89fBjBCNyw\/9LVwXz3IUAC4745caxHOA6AYRw2cLw4QUiwYUL9jg0Hmxm+Zg0tRbt27dSxCl72sCeA7QhU2FD36giM8aWFL2gcIaFNvEiwCeNoBVqHSBoYHF+BaGHzA\/H54Ycf1Nk4PHugPerfv7\/ajLBx4aWElyZU5NAShRvxQjuB4yIQCJzH48gGYzU0OZGIGP5mVgMDrDBWfOniBYqx4zgM9iCGASw2K2CDFynmA8dJICcoi80aL1fgjg39yJEjgRtCQbgM9X\/w1zPIIIgJCBNsXaCWD58baFmweQFzYAGtDDAFnjBsxtzHQ2B0\/cM4E\/VCMmQAmwv6hjkFLphP2NvAHskwQMcmDFscEFvgZBA\/bMqwX0L72MQNA2nYfcDOAxtfuLxAc2NWPnQLFhs95hQJsg4ZRFs4hoGmAuvHwBvrEYRdl+JZT8ABhBcJnmPw+gO+sOeCLQn6EI3AQPOF42Uc9WDjB7GElhLkD5pEyC\/WV6x5Cpchw8YLHyOYK8gw2oddViIEBu+M8PkzSH403EBy69evr66JgHYJR5TQaGKdABOm1ESABCY1582yXuOrCi8jLGZ8AWKTveGGG9SZMDY4JGxwcJ\/E+Tk2Uqhj8RLDwsdLGMdNOgJjuOWC8MB+ABs51NA4WoFNB1K0OyBg6AovHhyf4OUDuxecicObAxs2fscLGR46cPfEFyfIVfhNvCBnULnDQBh1gbSBYOBr09AOWKmBgREvNC4gW8AJ\/Qy234CWBEcy+PpGv6GVwVEGNgkkfOViDDiKwgsXmwnIGr6g0c9Imw+exVczvnSxuYCERnKjRr9whAQiijnHyx3lMCdI8RAYXf+CyV+8btTQ1sDbBRs95guGtiDWkEcQMiRsYJAH4AebGWhe8IUPbxmQEyQQNNhQgOQBW8NN3LBJCpcXjNesfOgWo3EEEuk52OtgLAbeIOlYK7oUz3pCHbBrgSYQxzZYE5AFuIwbl+jFus0Zx5SYC8gvyA8wNFz445mncBkybruFjEK7BvIJcg98II9NmzaN+A4IP0LCuCLNnw4zaAxBEPHegFwBA\/xu2OPoyjPffQiQwLhvTtijFEdAd6lfig+P3bcJAZAZkNOM3NBrU9csqRaaFtib4UoC2NYh4bgS2rVgrzlLGmMlvkCABMYX08xBOokACYyTaHujLWziOPaCa\/ill17qjUGFjQLaDtgy4fgHx7a4gwnH0dA44ojNuEzSk4PnoGxBgATGFlhZqZ8RIIHx8+ybGzuOvGDjpLsUz1zt7igFOxR4\/MAeDHZJcN3GkSpc+I2jPXf0lL1IFQRIYFJlpthPIkAEiAARIAJEIIAACQyFISICMI6E+yo8WpBwgRs8fuA1AwM+GAbCiBLGojD2hbEpjGKZiAARIAJEgAg4gQAJjBMop2Ab8HLANe24jwReH4itA5UviAuCPcIFGufZeAaximDZjzsmmIgAESACRIAIOIEACYwTKKdgG7gkDHcrwEUVCZfbQeMC91a4HiOej+HuiXg3cO+Fa7URzTkFh8wuEwEiQASIQAohQAKTQpOVrK7iyAhX8MM4FQZ3ICu44wQxjIyEC6HgAgq3yOCEy8aYiAARIAJWIBAtPpMVdbOO1EOABCb15szRHuPyK1yuhQuvoIXBVeJwe8T\/K1asGOgLbvXEbZ\/hXhQkMI5OFxsjAp5GgATG09Ob8OBIYBKGzD8FcHMmbrzFTZWwc8HV7EjQwOCGVyOGD\/6G4yZcFW7clmqgZBAYvnj+kRvcCIqbeZnOI0A80ktCLExW7zoiTUZtTFeod8NSUqtMXlfk1SqTT1btPGxZH\/POf1DVxfcI3xrBCJDAUB4iIoCwArjiH0dDiOAanBCQDn9HLBwkRKXFNfi48jt8YyaBSWxz8qM4ksAkJiNN39gUkRyoD4koxCEZedFk2UwfSWD8+GbQj5kERo+RL59AjCLEyEH8nvCEoHqIMwMvJNwaiiMmBEWEkW94IoFJbHPyo7CRwCQmI\/kfXyktll8sxQ5mlcGtfvWFyJDA+GKaEx4kCUzCkHm\/ALQvsG\/B1d6IGm0kBDzcunWr+hUEZsqUKQIPJNwRg3tgihQpQgITh3hwww4FiXgkRmDKP7BKWnxwkcysf1R+KnQqUNiMZsMuzYyV2hfURQITx4vFh4+QwPhw0p0cMjUwiW1OTs6NW9oigUlMRl4uvFl+KnxaZtb7PaTgokcqyqqdR2Tgsu\/TVeh0XjQyZbYfJDBuWa3u6gcJjLvmw3O9IYFJbHPynADEMSASmPhl5JPBB+WTQQel5OBLZMaxXwK2MDDg7d2wpKoIRr4Dl+1Oep6V\/RjbuY4aG41441hQPnqEBMZHk52MoZLAxL85JWN+3NAmCUx8MvLjmj9lyl27pETN3LKnU+aApgXaDpCXmqXzqopAXgwtTDLzrOxH6\/qVSGDcsFhd1gcSGJdNiNe6QwIT3+bktXlPZDwkMPHJyNRm38kPq\/9Q2peuG79NV4hu1IlIHZ\/1AgIkMF6YRRePgQQmvs3JxVNoe9dIYPQyYmhfaj9ZWAZduJ9u1LZLJRtIBQRIYFJhllK4jyQw+s0phafXkq6TwOhlBNoXpFZzLxe4Ufst0YjXbzMe33hJYOLDiU+ZRIAERr85mYTWM8VIYGLLiGG4+8C80lK8Rq6oBIZu1J5ZEhxInAiQwMQJFB8zhwAJDAmMTnJIYGLLCNymYbgL7QtSsHFscEmzLsp2lKMbtU7qmW8FAiQwVqDIOqIiQAJDAqNbHiQwicuIlS7KdrlfW9lHulHrVpE\/80lg\/Dnvjo2aBCbxzcmxyXFJQyQwkWVk1reZXOEOHa71idc1m27ULllgHu4GCYwLJ\/f++++Xzz77zIU9Y5e8jkD16tVl2rRpjg6TBCY93HNWbZOOcw+ky3CTq3SsvjAataNLyLeNkcC4cOqhteCNky6cGB90KRmyRwKTXrAavPapfL7nRESJc5Oxbqy+RFsuZvpPLyQfvHxMDJEExgRodhdJxiZi95hYf2ogkAzZI4FJLxudy21QfwyPd5QaUmR9L0lgrMfUCzWSwLhwFpOxibgQBnYpCQgkQ\/ZIYEIn2nCbDo82jafMaC+SUc5K7QvqIoFJwssgBZokgXHhJNm1icCo7vzLLG8gboodwz937pyMHz9eZs6cKXv27JF8+fJJo0aNpGfPnpIrV66YTZYvX16WLl0qv\/zyizz66KPyv\/\/9TyZMmCDffvutDBgwIEPdnTNnjtxzzz2qjmuuuUaWL18ul112WYbqtLrwl19+GRj36dOn5d1335U777xTgv9udZvB9dkle7H6TALzDzrGjbuRok3jKTtcnu2ok27Udq5S1m0gQALjQlmwehOBO2OTURtDRhruSWAlDC+\/\/LK89957gn8rVKggBw8elP79+8uZM2dk8uTJcRGYQoUKydGjRyV\/\/vxxEZizZ89K5syZo9YNUlWtWjVZt26degYE6ZJLLolZxkpM4q0LpMUY95YtW2TQoEHy9ttvS\/Df463LzHNWy148fSCB+QclI95R3TculJcO\/hESMgBrdmGXCurhpm9scn2elX3cMriZGjdtA+NZUf55hgTGhXNt9SYS6+rx3167xVIEQAxq1aols2bNEmhTjPTHH3\/IggULpHnz5pI1a1ZFaD744AMBsYDnC7Qr+Hs0DczGjRvVxr5t2zYpUaKEDB8+XGlPrr\/+euncubOMGTNGPv30U9m5c6c899xz8vvvv0v27Nmlb9++UqNGDenYsaNq74orrpD\/\/ve\/Ur9+\/YAGZsmSJTJs2DBFsAoUKKCIV+nSpRVx+uabb+TkyZPy448\/KhKBdooWLRqC2bhx4wRkA8+BrKHdoUOHqv6hv+jP4cOH1d+feOIJqVevnvz999\/y1FNPyeeffy4gX5UrV1YYQNMEzdPixYulYcOGahwggU8++WRAM4PnUT\/6jVSuXDk1zosuukiNE79\/8cUXcuDAASlWrJiMHj1ajS1Sezlz5gwZi9WyF49wkcCcRyk43tHB6r\/RCylIeHiEFM9K8t8zJDAunHMrN5FI2pfgIcMV0rjIygoocCzz4osvyqpVq6JWh2cGDhyojkeQ7rrrLkVCmjRpEpXAvPbaazJv3jxFQHAUlS1bNnnllVfU5g5S9PTTT0umTJnk9ttvl\/bt20uzZs1k4cKFiuh8+OGH8ttvvykiA0KCZBwhgUDheAvkqlSpUsqFePbs2aotaD5AbN5\/\/30pWLCgPPvss+o4DO0HJxAdox0QoH79+snx48flpZdekgYNGiji0bRpU0VOcIT10UcfKbI1Y8YMmTRpkiJxIC8gVRdccEGAqIDEgAiiH8FHSIsWLVJECnkXXnih9OjRQy699FJFULp06aLGCk0XNFK33nqrvPDCC4r8RWqvSpUqJDBWCL4FdRjal2cOlhN6IYUCSgJjgYB5sAoSGBdOqpUEJtq148awrSYwc+fOVZsnCEC0hA37zz\/\/lNy5c6tHnnnmGaWt6Nq1a1QCA1sYbORIn3zyidLggFhUrFhRaUrwLxKIA0hAlixZ5Oeff1baIBCHaAQGdaGet956S5WHFuXqq69WhAFjARGDPQ8SNDfQtAwZMiQdgQEhMZ5DGZAYaGZAqL766itFroLJGohQt27dlLanZs2aSjuDFExUohGYxx9\/XPWxQ4cOqgywASGERgYEBtqcBx98UOV16tRJaXxKliwZsb3wObJS9uJdWtTAhGpfavcs7MuAjbHkhQQm3tXkr+dIYFw431ZuIjoCAwO+mqXzWobCihUrFCGJdRHfoUOHlMZh165dqt29e\/dKmzZtYhKYr7\/+OkAcsMnjqGTt2rWKuIBoQHuCBO0ECNSpU6fUkc\/WrVtVO9EIzPz581V+MCkBOQAZ+Pjjj2XTpk3quAYJBCr4dwM0ECi0M3jw4AAJQf\/eeOMN6d69e4g26t\/\/\/rc6GmrZsqU6JkJfccwELVCfPn0CR0ggJdEIDLCCturee+8NtAeiAsxBYG688UZVP1Lw75Ha4xGSZaKfoYrC4x1FO\/alF9L5qNxMRAAIkMC4UA6sJDAYXqyXoWEUaBUMOKq44YYb1NEIbFuMBM0GtBIgNzhagQ0ItAbQlODoIy0tLSaBCdZwQGsCzQW8lYIJzP79++Xmm29WR1M4atq3b5\/Url07JoFBXahn4sSJqqvQ4Fx77bWyefNmeeedd+ImMGvWrJE333xT1WH0D7Yn0MCgLsPAGMdljzzyiNKKGAmYQfuEvlatWlV7hAQNzJVXXikPP\/ywqmLlypWKgGHcsQhMpPYMLY6RZ7XsxSNXftfAhEebBmZPz9woY9ceSQefHR5DdtRJL6R4JJ\/PZBQBEpiMImhDeas3kWh2MFZrXwwosJnChRokAyQGhr0wMs2RI4ey3cAmC9sVaCmgfcC\/2Oh79+4d9QgJtiggGjCghb0LjolQZzCB2b59uyAMA8gEDIJBkHCsA+0ItDFoE2QCrtyGDQz6DO0HNDHQ4kDLAu0L+h+ucYmlgUH\/oOGA0SwIGY6EYAsEGxgQFrhCox\/oH7Qr0BrBQBcaGiSMHaQrmMDgaAskCM+i34ZbOdoZNWqUIlhoB0dRkBkYCEcjMCBmkdojgbFhAVtQJUjdvrP5VOTpVTsPn5eRIHs1KwMlMpijBRPGKpKCAAlMUmCP3ajVBMZoDS+9VTvPf9VZabgbaTTQwOB4BN47MGyF5gEbMIgHPGSgRYAhLkgFNnkYxsJQF5tw+D0wsCWB8S02YJAUaGtGjBihDGvDj5BQHkdLF198sTK6hXEtyAs2exy9wB4FtiwgEsY9MCAExnOwxQHxKl68eEIEZsOGDUp7E94\/wwvpyJEjimyAfEHTgiOtXr16BYyKgYNxrGYQFRA\/4Ib+jx07NqIXEuyJQHqef\/55RcyiERhgHKk9ww4pWAMDo2cnE+4Kwpz6NY1bdySgbbkhLYd0rJpXCp87JEv25U7398pFcyiYIpVxW56VfezV5jY1brpR+3WVRB43CYwL5cEuAuPCoXqiS7CB2bFjh\/KKSvWUDNnz8xFSNO0oiEykWEgM5kgbmFR\/x1jZfxIYK9G0qK5kbCIWdd2X1Vh1U7AbwEuG7PmZwIRf9haPDPjRkJdeSPFIhv+eIYFx4ZwnYxNxIQwp0yUSmIxNlR8JDAx36S4dv9yQwMSPlZ+eJIFx4WzbSWBgQHrTTTe5cNTskhsQsFP2oo3PjwTGcJvuVvKHhKbdj9oXAEQCk5CY+OZhEhgXTrWdm8gtt9yi3G6ZiEAkBOyUPRKY8wgEu01PP\/aLDFz2fTpootnA2OHybEeddKPm+8UJBEhgnEA5wTbs2kSgfcE9KfBcsTPBEwd3vcBzCDF44GECLyN43yDBtRpeRLjGHpex4f4W3J1i3FaLZxAGAFfkw5sJt+kioQyuzjdumY00BtT3119\/qQvtjLtXcM8K2sYFePDowR0q8IYCDvAMggcQXLKNy\/DsxMbtddsle7HG7ScNjBHvqETN3NJq7uUKlkgu0c2vPEc36iD38bGd6yis6IXk9jeIs\/0jgXEW77hCfb3xAAAgAElEQVRas2sTMQgM7ifBra92pTp16iiScd999ylSArdoEJjVq1erWELhBAYXzuEeleC4PLifBHef4E4Zg8Dgdln0GzflRksGIYI7MVylkSIRGFz3D5dpuD4jLAFuAzZCFdiFSyrUa5fs+Z3AGDdit1h+sRQ7mFVKDr5E7m99Piho8G3ZRpT4yzIfllnfZgpoZ8Kjx0cqY9yo7ZY8K\/vRun4lEphUeIE43EcSGIcBj6c5uzYRHB8hkCC0MHYdI+EK\/7Jly6pr7QsVKhQYLr6ycbcKbt4NJzCI\/AyNCG7qNQhH48aN1fMgIiAwIBq4vRbEA7f44k4T3CeDZ3APCu4SwVX+IDAImAjig\/tM8uTJE5PAoD1of0DqnL7\/JB5ZcPoZu2TPzwTGcJUu9nM2afHBRbKm3HFZc\/1f6mK6WmXySpNRG9PBQzfqUEhoA+P0myA12iOBceE8WbGJ4EgEZCU4Bf8OEhOc8Dt+rDDwxc26Bw8eFMT9QQToYCKDNsMJDLQtzz33XOAGXdyCi1trcRSFK\/ZBYKA9mjNnjjpGmjp1qjpiwr+IqwTCgmdwERwIDMgNrtXHbby44C2WBuaPP\/5Qz+BSPIQ58HuyQvYSxdDrR0iGqzS0L0gz6\/0egCiWUW40HP1oyEsCk+iq8sfzJDAunGerNhGQGGgWdMnqIyVoYaZNm6au5IfdCcaD6\/Rvu+38bZrhBAaEA4EPW7duLTh+wr+wfxk0aFCAwOB2XGhqWrRooa7Or1SpkiJISLhxF7Y2wQSmTJkyKmji9OnTFTkJt4HBDbQ43gKBge0LQg7QBkbUXDltZ+B1AoNYZDW+ulBqbM4pM+sflZ8KndItSeaHIUACQ5GIhAAJjAvlwspNxLB7iTZMq8lLeDs4+lm2bJnSbkBjgqv\/IxEY2KDgWAuamObNm6v\/w+bF0MDgSAkhBWAQjJAAiC3UrFkz1dzIkSPVEVIwgYE9DWxaEB0bWptwAmPYwID44P+w0YHhb5EiRVwoEc51yUrZi7fXfiAwPadeIj8VPk3tS5l8gdhO4fIRS7NEAhPvavLXcyQwLpxvqzeRaCTGDm8kEJFvv\/1WYG8TnNq2batiHrVq1SoigUFwRZTp3LmzCv4IrYxBYGBTY5Aa1AltTrVq1RSRiaaBAYGBxxECNSK2EIhRsBeSQWCMPoIgBWuJXCgWjnTJatmLp9NeJzBTm30nP6z+I532JZb7Mt2oQyWHBCaeleS\/Z0hgXDjnVm8i0QiMHdoXHD\/ccccd6vgHhAXHNOvXr1cRp3GcAw+iSBoYEA4cDeFZBFvEcwaBgZ3LunXrVJBFJGhiYHgLDQtsbaCNAfkJ18DgWTwH8oJgipEIzNmzZ5XB8UMPPaTsZoC9n5PVshcPll4nMMAAJGZmvaNxR5amG3VoFG66Ucezkvz3DAmMC+fc6k3E8D7CUEFaYMxrpzcSSAPconft2qUITIkSJZQ3EQgNUjQCg+jQr776qrz\/\/vvqOYPALFiwQBEUaEmQjh07prQq0PRcccUViuyAoCCYomHEG+6SDQIUfg8M6jL6h\/qM+l0oEo51yWrZi6fjfiAwiboU0436\/OV+hvs43ajjWUn+e4YExoVzbuUmEu3uF4PU2HGM5ASk0JwYF9WBLOHyOnoRZRx5K2Uv3t54ncBEizhNN+pQCYmFB4+Q4l1N\/nqOBMaF823lJgICA21LpIvr4KWEZOeldnbAC00NjpMWL16svI+gdenevbvceuutdjTnqzqtlL14gfM6gYkVcZpu1KFSEg0PEph4V5O\/niOBceF8O7mJpGJwR5AWkC54GEELg6Ofp59+OiQUgQunNSW65KTsGYB4kcAY0aYxRrhRM2UMARKYjOHn1dIkMC6c2WRsIi6EgV1KAgLJkD2vEZjweEfRCAy1L\/FpX\/AUCUwSXgYp0CQJjAsnKRmbiAthYJeSgEAyZM9rBMZwm35gXmkpXiNXSKyj4CmlG3WogMfCgwQmCS+DFGiSBMaFk5SMTcSFMLBLSUAgGbLnJQJjaF9qP1lYavcsHJjBSBGnezcsqfIZjVpUXKhYeNCNOgkvgxRokgTGhZOUjE3EShjg2YSr+RHTCDfkIgI1LpTDbbe5cuWK2VT58uVV9Gq4ReNSO9joBCfESIr0dyv7H62uCRMmKNdt3Dfj1ZQM2fMKgYGr9O4nf1PRpj\/rc1ZtyGYjRNONmm7UXn3HWDkuEhgr0bSormRsIhZ1XVUDD6H33ntP\/VuhQgV12Vz\/\/v2Vx9DkyZN9Q2AwXkTfTqWUDNnzAoGBFqVLny0h0aYx77Fcg+lGHboy6EadSm8Kd\/SVBMYd8xDSi2RsIlbBgDAAiAw9a9YsgTbFSAiaiAvpEBIgW7ZsKj7R\/PnzVTaeR0DGCy64QJXRaWBw5X+9evVUsEiUeemll1TUa9wNM3ToUPV3pHLlyglcxS+66CK56qqr1K28COyIZPyOW36feOIJqVu3rrrtF\/2HhxOicp88eVKefPJJ2bBhg1x66aWCcAf4GzQwhw8flscff1xd1od2H3zwQRVccsuWLSoMAsIf7N+\/X\/788095+OGHA4Es4QL+2muvBfpoFe5W1ZMM2fMCgYGrdPW+mdPFO1LybTL+T7Q5NVuf0+Ws7D9tYKxa4d6qhwTGhfOZjE3EKhiwQeO2X8QaipZw0y5CDcydO1cuvPBCdST0r3\/9SxGAeAgMQhVAu4P7X+bMmaMiWcOlGsEYx4wZo8gT6kVEaxCPp556KiqB+e2335QbNo6HQFpQx8SJE+Wdd96RKVOmyMKFC1VkbQSlvOuuu+SGG25QBAak6ffff1c3B+OYDAQIx10gNnfffbcMHDhQmjRpoo7SPv\/8cxk7dqyCA4QI84sxuzElQ\/a8QGBur7+W0aZtFGgSGBvBTeGqSWBcOHnhm8hvC0vF1cv8Tc+fGweneMuiTKTy+LuujuByICU4Jpo3b17UPodv4iAfiG80Y8aMuAjMPffcI1999ZU6noFGBNqUL774Qv7zn\/+osAIdOnRQbYNQgEhAIxNNAwMCAyK0efNmVWbbtm2qPAgYYjPhCAzaFSTUBYICAoNAkadOnZKcOXOqPBAYkJncuXOrWFCw1cEdNQcOHFBhEKDdAamqWrWqwqZ48eJxzanTD5HAJI64YbgbHm0aNZnVekTrhdn6nC5ndf9JYBKXSz+UIIFx4SyHbyLHtw+Pq5c5y3ZP91y8ZVEwUnn8XVdHcDmQEVzpjwCJ0RIIAY5lDINeHMFccsklSvsRjwYGgSGDNTwgLSApOPqB1gOEBAmxjzp16qT6EovAIFK20d\/t27eL8TuiXTdt2lRAmJBAshCsEgQGBApHQSA0iKcE4gPihuMqlFu7dm1g+Pfdd580a9ZMihYtqkiOcXQW16Q6\/BAJTOKAR4s2jZpiuQbTjToUa7pRJy57fi9BAuNCCUjGJmIVDEePHlXHLJMmTVJBG40ETUm\/fv0UuYFB75VXXint27dP12w8BAZBHkFOQBz+\/vtvZW8CQmTUC5sTpJUrV8qQIUNUlGmQHGhkChUqFCgDrQg0MNEITNeuXaVSpUqBfuLYCuMDgYHGBUSqRYsWqi0cPw0ePFgRmOD6kAfNEux6SpYsqUgMyrk1QfY+\/PBDR7uHI7i0tDRH27SjsS\/2npBx647I53tOqOo7VcsrHavmVf9PNK\/xZX\/IwUwFLKvPbD8yUi7RMcdqa+bTdysc8QHBRAQMBEhgXCgLqUxgACdIA1yoseGDxMAwFsa0OXLkUDYqsJMZPny4TJ8+XR25wMYExrjQdMRDYG6\/\/XYZNWqUcs2GNgM2MLCrQWwk\/B32K9mzZ1dHQMASRrp16tRRBAoGwzjmgks3tCSxCAzsYpYtW6ZsYRABG7Yt1apVUwSmYsWKiqRdf\/31qj7Y\/cAwuUiRIukIDGxl0C6Om3B8BBLj1pQM2fOCDQzmM9GI07FcrOlGTTdqt74j3NQvEhg3zcb\/9yUZm4jVMGBzx5HKjz\/+KAUKFFAGsI8++qgiKkgjR45Umzk0KBgv7Euw+esIDI5uQEig8cBxFRIIRZUqVUK8kHAXDexNnn\/+eXVUBXKD45vLLrtMkRkY18IrChqVaBoYeBChLdjXQENgkDFoWjA+kDEQMGhhcG\/N7NmzVRvwQgo\/QoNdDcgSyJWbUzJkzwsExkzEabpRh64EulG7+c3gzr6RwLhwXpKxibgQBk91CUdnsMOBfYybUzJkzwsExmzEacZDCl0NjEbt5reD+\/pGAuO+OVEaCZ71unBiTHYJt\/e2a9dOXe4HGxk3p2TIXioSmOBo05hPRpy2V6rphWQvvqlaOwmMC2cuGZuIC2HwRJdwNIZjI7h4N2jQwPVjSobspRqBCY82HYvAmHVfjiYoZutzupzV\/SeBcf2rIykdJIFJCuyxG03GJuJCGNilJCCQDNlLNQITHm0a0xRswBs8bXSjDhVis3iQwCThZZACTZLAuHCSkrGJuBAGdikJCCRD9lKJwESLNo2pSjTiNKNR75ZVOw8rKWc06iQsdg80SQLjwklMxiZiJQy4DA5X7X\/zzTcqgCM8eOC2XLt2bVPN4NZb3OVy5513miqPQoxiHR90yZC9VCIw0L6s2nlEZtb7Xd2ym5GI03SjDnWVjoVH6\/qVlADTNjC+deyXp0hgXDjTydhErIQBbsq4bRc30OKyOVziBgKzevVqyZcvX8JNIUAiYie9\/fbbcZcNjwSdSgQmmVGskyF7qUJgYLj7yaCDMrP+Ufmp0KmALJqNOE036tDlTDfquF9vfPD\/ESCBcaEoJGMTsQoGxAfCzbi4BwW33hoJmxTi\/yB+Ee5vASHBHTDFihVTd6fgWdzdguv5cT8M\/o\/7We6\/\/35p2LChCpyIuES4VA5hA4YNG6a0O7hjBhfmlS5dWt3tAu0PyApIFEiTkfA3RrHWz3IyZC9VCMzLhTdHjDYNVO0wko02W3a0ZUedVvafNjD6tevHJ0hgXDjrydhErIQBV+UfPHhQRZeuUaNGCJHBrby4hh+XvoHoIL4QLopDtGZoavA7buY9ceKEuml39OjRsm\/fPhVhGhqYvXv3qr\/jErpSpUqpZ1EXSA\/ycRsubsYND5YIAsMo1vpZTobspQKBiRXvSI8qn8goAiQwGUXQm+VJYFw4r+GbCF6e8aRWcy9P91i8ZVEwUnn8XVdHeDloYUAsoClBzCKMB9qP2267TZELkA\/jOOiPP\/5Q1\/LDXqZ3794qZpER\/Rm35OIWXdyfYhAYhChA2IC33npLjRUxllAG7aBuaHciHTWBwDCKtV6KSGDSYxQr2jS1L+eNcONNZjU9JDDxIuyv50hgXDjf4ZsIVNfxpGcOlkv3WLxlUTBSefxdV0e0cih7\/PhxFU8IN9FOnTpVPv\/8cxk6dKjkz58\/0FcQFQQQxBX8jRs3DkSTNh5AGACDwCDu0a5du1S8JSMZ0ag\/\/vhjFdQRcZbCEwgMo1jrpYgEJj1GhvYly6SLZOCy84anwcmsazCjUcePIwmMfu368QkSGBfOejI2EatgwBEPbp695ZZbQqqEPQsuckNAQ2hUcFQUnmCzUqZMGTGiSePoCM+vWbMmRAODo6aJEyeq4iBI1157rWzevFldGLdp0yZFkCIRGEax1s9yMmTPbUdIkYIynp13Umr3LCzhIQOgUVjYpYIC1sq84Y0ulu5Lfw+4GaN+u9qyo\/+o00o8tgxupjCmF5J+DfvpCRIYF852MjYRq2DACwa2JjDSBWGBF9L69euV9gPRp2F0CxsW2K3AhsU4+kG0ahw5IcjjjBkzlBFvkyZNFBn59ddflS0Mjoj2798fiEKN8jguQjkcLeH\/sQgMo1jrZzkZsucmAmN1UEazHko3pOWQz\/ecSDdhZutzuly0oyKz\/aAGRr92\/fgECYwLZz0Zm4iVMHzyySfKSwhHPSAwJUqUkC5dugSu0je8kP766y9l4wLyYkSTRqTn+fPnq8jS7du3V8QHhr+IZo37YODdhCMlHBPhd0SXhhcSjHZjERhGsY5vhpMhe24iMHYEZTRr9xFtxszW53Q5K\/tPAhPf+vXbUyQwLpzxZGwiLoSBXYJdksNRrJMhe24iMAzK6M5lRwLjznlJdq9IYJI9AxHaT8Ym4kIYfN+lZESxTobsuZHAtFh+sbpt10iprL0w7GeMa\/vDF5YdY7NS+4K6SGB8\/zqMCAAJjAvlIhmbiAth8HWXkhXFOhmy5yYCAwPeaZP3SIsPLpI15Y7Lmuv\/UnJoh6cRvZBCl3gsPEhgfP06jDp4EhgXykUyNhEXwsAuJQGBZMiemwgMIDfcpge3+lXNgC7QYKygjGbzml95TvadzaeiXMcb8NBsW3aVszK45djOddRc0AspCS8FFzdJAuPCycH1+TBWZSICTiNQvXp1dQmhk8lNBMa4tM7QvtgZsJHBHBnM0cl15sW2SGC8OKsuGhO+6PnlFDohbtqw3SAqbsIjWrwjs+6\/ZsvRjTpUMnmE5IaV6r4+kMC4b0481SMSmPTT6aYN2w3C5hY8okWbNjCyw9g1Vp3R5sbpfphtz8r+k8C4YaW6rw8kMO6bE9f0CJfM4Xbc3Llzq2CJkdKiRYuke\/fu6iI53OUSnkhgSGB0Au0GAqOLd6QbA\/PtRYAExl58U7V2EphUnTmb+43r+hHZuVKlSvL1119HJDAHDhyQ5s2bC2IZjR8\/ngQmzjlxw4YdZ1cdecwNeOiiTZvVQpgtZ6X2AnWZ7YfZclb3nwTGkaWYco2QwKTclDnT4W3btqko0nPmzFE\/kTQwiG+E6\/5x6y6u\/KcGJr65ccOGHV9PnXkq2XgY2pfaTxaWNdcfdyxgI92oQ+WLbtTOrDcvtUIC46XZtGEsiCAdicBMmjRJEDIAmpdatWqRwCSAfbI37AS66sijbsADxrtGVHUr3X\/NuijTjTrUfZxu1I4sxZRrhAQm5abM2Q5HIjDYcFq3bq20MgULFoyLwKDX0Ni0adPG2QG4sLU9e\/ZIWlqaC3uWnC4lA49x647I2LVH1IDh8dOxal6pXDSH+t0NeYXPHZIl+3K7uo86vKzEsVeb29Tc8B6Y5KxRt7ZKAuPWmXFJv8IJzJkzZ5TdS6dOnQLBGamBSWyy3KBxSKzH9j7tNB5uiTgdy8WabtShMkcbGHvXYKrWTgKTqjPnUL\/DCcz27dsVgcmTJ0+gB\/v375f8+fNLhw4dVPTo4EQvpPQT5fSG7ZComG7GaTzcFHGabtShYhMNDxIY08vL0wVJYDw9vRkfXDQbmOCaqYFJDGenN+zEeuf8007jwYjTzs9xRlskgckogt4sTwLjzXnN8KjatWunwhngyOj06dOSPXt2VSdcqrNkyRJSPwlMYnA7vWEn1jvnn3YaDxAYRJsudjCrGPGOMGqzLsN2lIs2C3a0ZUedVvefBMb5dZkKLZLApMIspXAfeYTEIySd+DpNYKZN3iu7e\/4aEm0afXQ64jTdqEMlg27UupXC\/HAESGAoE7YiQAJDAqMTMKcJDC6tQ5pZ76hrIz3TjZpu1Lp1w3wREhhKga0IkMCQwOgEzEkCY8Q72vtwZpl+7BfVtWRFnGY0akaj1q0N5sdGgASGEmIrAiQwJDA6AXOKwOjiHZmNHG1HObpRh0oNbWB0q8if+SQw\/px3x0ZNAkMCoxM2pwiMLt6RoY1ZtfNwxC7bYexKN+pQqOlGrVstzA9GgASG8mArAiQwJDA6AXOCwBjalxI1c0u3kj\/ousR8lyFADYzLJsQl3SGBcclEeLUbJDAkMDrZdoLAGNoXxDuKdg+M0xoWal\/i077gKRIY3SryZz4JjIfm\/eTJk6ZGY9zxYqqwphAJDAmMTq7sJjDB0aZr9ywsA5ftdkXEabpRh0oG3ah1K4X54QiQwHhIJgyykOiQ7AyQRgJDAqOTR7sJjOE23Wru5YGuuCHidKxI1XSjphu1bt0wn27UnpKB6667TiZPnpzQmBBVesuWLQmVSeRhEhgSGJ282EVggjUtbnSVphs13ah1a4P5sRGgBsZDElK\/fn354IMPEhqRmTKJNEACQwKjkxc7CEwqRJxmNOpQyYiFB21gdKvIn\/kkMB6e90OHDslPP\/0kkWxjqlev7sjISWBIYHSCZgeBSZWI0zTkDZUOulHrVgvzgxEggfGoPIwfP15effVVFYwxUrLT7iW4PRIYEhjdErODwDDitA711MqnBia15sup3pLAOIW0w+1UrVpV\/vOf\/wg0LZG8jOz0PCKBiT3ZdmzYDouXpc3ZgUfnchvSRZtGp93kKk3tS3zaFzxFAmPpkvNMZSQwnpnK0IHUrVtXPvzww6SPjhoYamB0Qmg1gTHcpmfWPyo\/FToV0rybIk7TjTpUMuhGrVspzA9HgATGozLx0ksvSaVKlaRRo0ZJHSEJDAmMTgCtJjAvF94suHG35KD86s4XIzQAjERjuS67KY9u1HSj1q0b5tON2rMysH37dmnZsqXkypVLChcuLJkzZw4Z6+zZsx0ZOwkMCYxO0KwkMEa06QfmlVbRpgcui99VN5Zbs9N5l2U+LLO+zZSy\/QdeVrqxt65fSYmRU7Z7OpllvjsQoAbGHfNgeS9uvfVWueiii6Ry5coRbWB69OhheZuRKiSBIYHRCZpVBCY43hG0L01GbUzXtB2Ro+2ok9GoQ6eONjC6VeTPfBIYj857lSpV5NNPP5WsWbMmdYQkMCQwOgG0isAY8Y6gfem6aUfg6Ci8fRrypp8ROzCJNu9m2iKB0a0if+aTwHh03lu0aCFwpYYWxmz68ssvpWfPnpI7d26ZN29eoJrTp08rF+0FCxbIiRMnpHTp0tKnTx8pX758uqZIYEhgdPJnBYEJj3dEN2od6qmVTwKTWvPlVG9JYJxC2uF2QDhmzJghTZs2VTYw4alevXoxe7R06VIZMWKEMgT++uuvQwjMyJEjZcmSJTJx4kQpUKCADB8+XObMmSOrV68mgYljnq3YsONoJmUesQKP8HhHqRBxmm7UoSIaCw8SmJRZzo52lATGUbidawyaj0yZMkVtcNeuXTE7s23bNkEdICb4CdbArFq1SvLkyRPQuGADqlOnjmzdulVy5swZUi81MNTA6KQ+owQm2HC3eI1cqrlUiDhNN+pQyaAbtW6lMD8cARIYj8rEuXPnYhKYeIc9derUdAQmvOyYMWNkxYoVMmvWrKgaGGS0bdtW2rRpE2\/Tnn1uz549kpaW5tnxJTowK\/BY0eW41HkjlDx\/sfeEjFt3RD7fc0J1qVO1vNKxal71f7fnNb7sDzmYqUDK9h84W4nxzKfvVvNGL6REV5e3nyeB8dD8duvWTR37JJJ0ZXQEZv78+TJgwACZPn26lCpVKiqB4YvnH2gyqnFIZH5T4dmM4hHNXddKN166Ue9O2KXbSvzpRp0KK9n5PpLAOI+5bS3CiDaSHUqsBmvWrCkw1o2WohGYs2fPyuDBg2XlypUyduxYKV68eMQqeISUHpaMbti2CVCSKs4IHtGiTkezp7DD5dmOOulGHSqMtIFJ0uJ0ebMkMC6foES6Z5CFRMro1LKRCAyOp3r16iWIdg2ND7yUoiUSGBIYnTxmhMDEijodrV0zbryoy+ly7P8\/CJDA6FaRP\/NJYDw077E0KbGGGcn92Xg+EoGZNm2a4AeGvdmyZYuJIAkMCYxuiWWEwNBdWoeuN\/JJYLwxj1aPggTGakQ9Ul+7du3ks88+kzNnzgjufTGiV8OlukmTJrJjxw7JkiVLyGhBdnDzb3AigSGB0S0JMwTGcJvuVvIHXfUh+U5rUcy2R+1LKAIkMAmJuW8eJoHxzVQnZ6AkMCQwOslLlMBEi3cU3E404sBo1OdjQwUnOzCxGn8SGN0q8mc+CYw\/592xUZPAkMDohC0eAmN4tBT7OZu0+OAiFW261dzLVdXhdjDYPBd2qRD177HKuCVveKOLpfvS30PCIRjjcksfgbFT+G8Z3Ey1RW9G3WryVz4JjL\/m2\/HRksCQwOiETkdggj2NWiy\/WIodzCoz6x+V+1unSa0yeSMGbaQXUijqdnhKxarTavypgdGtIn\/mk8D4c94dGzUJDAmMTth0BMbQsBjalzXljsua6\/9S1fI6\/lB0zdrc2FEu2rybaYsERreK\/JlPAuPReT98+LCKVbRz504VcDE8vfXWW46MnASGBEYnaDoCY3gaGdqXwa1+1VXJfI8hQALjsQm1aDgkMBYB6bZqcG0\/zourV68uF1xwQbruvfTSS450mQSGBEYnaPEQGGpfzqNoRnuRjHJWal9QFwmMbhX5M58ExqPzXqVKFXVLbqxL5pwYOgkMCYxOznQEBga8Z9oclZ8Kn5aZ9X4PVBfLe8ZqLxg7PHUYzDFUMhjMUbdSmB+OAAmMR2WiQYMG8v777yd9dCQwJDA6IdQRGMNteu\/DWWT6sZ9VdTAg7d2wpPo\/jHxBclbtPBySF+3vscq4Ja\/5ledk39l8Ecfllj46if\/YznXU3NILSbea\/JVPAuPR+R43bpy6aK59+\/aWRKU2CxMJDAmMTnZ0BAblO5fbENC+QLuCzVMXYNHKYIK6ttBHK9u7LPNhmfVtpoQDKFrdj4yM20o8GMxRt4r8mU8C49F5hw3Mhg0b1A26RYoUSUdiFi1a5MjISWBIYHSCpiMw0QI2OunG67QbMoM5hkoNbWB0q8if+SQwHp33YcOGSdasWaOOrmvXro6MnASGBEYnaDoCEytgI92oQ9F1k5FvtHk300cSGN0q8mc+CYwP5h3Ro\/GTOXNmx0dLAkMCoxM6HYFhwEYdgt7PJ4Hx\/hybGSEJjBnUUqAMAjCOGDFC5s6dK3v27FFHSCVLlpSWLVtKhw4dHLOLIYEhgdEtlxkzZii5DE4w3K3ds7D6UzQCQ+2LP7QvGCUJjG4V+TOfBMaj844jpGnTpsl9990nJUqUUKPEly42i86dOyvjXicSCQwJjE7OQK6hIQxOLxfeHIh3FGwMGvwM3ahDkXXa1dtJ\/AL\/AxYAACAASURBVElgdKvIn\/kkMB6d95tuuklGjRol1113XcgIN23aJD179pTly5c7MnISGBKYWIL2v\/\/9T26++WZ58cUXpU+fPurR4GjTxWvkUn8z4xJtpozZtqwuRzfqULd4ulE78rpOuUZIYFJuyuLr8DXXXCNffvmlZMuWLaTAmTNnFKnZtm1bfBVl8CkSGBKYWCJ0yy23yEcffaQITI7qbWXa5D0q2jQurXtg3uVaV2nUHc1d10o33oy4E8fqY7Q8ulF\/r8TGcJmnG3UGX8QeLU4C49GJbdSokTzxxBNSr169kBGuWLFC+vfvr27pdSKRwJDAxJIzHB8ZKd9jKyQ42vRPhU6pC+uiRZymG3UosmaxsqMco1E78XZlGyQwHpUBGO8+88wzUrduXTFIxI4dOxRxgar+\/vvvd2TkJDAkMNEEzTg+MvKvuWmktNpWW4KjTRtf4cYtu+F10ZA3FBEzLsoZwdgp\/GkD48jrOuUaIYFJuSmLv8PYIKZMmSK7d+8OeCG1bt1aateuHVclOIKCvQziKc2bNy9QBgaXr732msycOVNFui5fvry88sorkpaWlq5eEhgSmGjCZhwfGfkP5XtLsqZVCIl3FJeg8iHPI0AC4\/kpNjVAEhhTsHm\/0NKlS5UbdqVKleTrr78OITBTp06VN998UyZPniyFCxeWQYMGybp162T+\/PkkMHGIhu7ekziqSJlH+vbtq2xcIqXgv9fN1Vnq5HpY3s49Qr4\/9XnE52EjkzWtfCDPqa9\/uzQU7H\/82iMSmJRZ8o52lATGUbjtbWzIkCHSvHlzKVasmOD\/sRLsY2IlGPlCezJnzhz1E6yBuffee+WOO+6QBx54QFVx\/PhxqVChgrz77rtyxRVXhFRLDQw1MOGalkhy91KhLxVxefPwg+my6z3QQ74o2DTd351043XaRTlaKAGn+2G2PaujgZPA2Lt3pGrtJDCpOnMR+g1iAfsWeBnh\/7HS7Nmz4xo5tC3hBAZkZcyYMVK9evVAHbC1efzxx+W2224jgdEg6ycNjAEFNDHwNIqUDO3LW0ceku\/+Xh\/yiOFebcYl2kwZNO6GcnSjpht1XC9onz9EAuNzAdANPxKBKVu2rLoQr2LFioHijRs3Vpfj3XPPPREJDP6IAJNt2rTRNen5fNyMHMleyOsDX7t2rQwfPlw+++yzkKHWuj6HjG5TXzpP+kBWfXVC5WUrVkG6d+8uXZrdpH4ft+6IjF17RP0f2omOVfNK5aI5YuaZKWO2LavLFT53SJbsy53wmK3uhw7jWO1ZiX+vNuc\/jL777juvLxOOLwEESGASACuVHm3WrJkKIxCeDh8+LHfeeafAwDeeFE0DA\/uYYGPgWrVqybPPPitw3w5OPEJKj7IfNTDBKETSxvy6oKS8OuOIDJx+RHL+q626EwbJrIuv1W68ZvththyjUYeuGx4hxfO29t8zJDAem3N4Dm3ZskXd9fL888+nGx08kkBKtm7dGtfIIxEYxK3BkRFiKiGBFFWtWlXef\/99KVWqFAmMBlm\/E5hINjG978srvVrmVSTm9d13BAgMoDTrGhxtGszW53Q59v8fBEhg4npd++4hEhiPTfmaNWtk\/PjxyvMDHkLh6cILL1TxkQzyoRt+JAIDmxgcBcAL6dJLL5V+\/frJzp07lVt1eKIGhhqYcASiGfVCC4N0yR27BZfaMREBAwESGMpCJARIYDwqF7jvBQTDbGrXrp2yVUDoAUS2zp49u6oKLtVZsmRRBAZ3zMADqUqVKuoemCJFipDAxAG4nzUw4ZfXAS4Yg0PWgrUwb5zuo1ymzWo9qL0IRcAsjmbLWY0\/CUwcLxYfPkIC46FJB9EAucD17Ph\/rJQ1a1ZHRk4NDDUwwQiEa1\/gZQTD7mkrNkvfvi\/K47V2q6OktAE3qWMkt7jxmu2H2XJ0ow5dNyQwjryuU64REpiUm7LoHQZZmDZtmvqiNYhDtKedsuYngSGBiURggqNPGxopuC8\/99Bd8s5ju5UtTL62G6V3w\/PHSmZcm82UMduW1eXoRk03ag9tTbYNhQTGNmidr3jVqlVy\/fXXy8UXXyz4f6wEryEnEgkMCYyBgHF8FExekBd+pHZ8+3A5vn2Y5CzbQ3KW7a6Km4ksbaaM2basLsdo1IxG7cT7OdXbIIFJ9RmM0f+TJ08GbFfw\/02bNqn7R4oWLerYqElgSGCCEQCJuemm83e7GCmSTdBvC0tJtgLVJU+N6Ur70mTUxnRAMhp1KCRmXbbtKGe1GzuPkBx7ZadUQyQwKTVd8Xf2k08+kW7dusmGDRuUPczdd98dMMAdNWqUNGjQIP7KMvAkCQwJjE58IhEYaGEM7UvTNzYJo1H\/g6JZw1qny0WbdzP9IIHRrSJ\/5pPAeHTecaU\/PJFwZwuCLL766quyePFiWb9+vbz++usqbpETiQSGBEYnZzqvrPyPr9RVwXyPI0AC4\/EJNjk8EhiTwLm92FVXXSVfffWVZMuWTR577DEpWLCgPPPMM3Lq1CkpX7583BfZZXScJDAkMDoZMktgGM05FFkzmg3UYEc5K7UvqIsERreK\/JlPAuPReUfARRwj5cyZU\/71r3\/J0KFDBYa7x44dk5o1a8rmzZsdGTkJDAmMTtB0BCbYGDe4LkajDkXWrMu2HeUYjVon9cy3AgESGCtQdGEdHTt2lHPnzqk7YRA2ADfzIg0ZMkSFGsjIJXeJDJcEhgRGJy86AoPyy5f0VSEGDFsYGJ7qXKzpRn1YQR8PVnjOarysrG9s5zpqLE5d\/6CTWea7AwESGHfMg+W9+Pnnn2XgwIFy6NAheeqpp+Tqq69W\/0cgR4QawO9OJBIYEhidnOkIjOFWDQKDH3zdg7zULJ1XVR3NXZpu1OePh+LBKhaOZvOsxL91\/UokMLqF5MN8EhgfTDo0MfjJnDmznD17Vv3rVCKBIYHRyZqOwOBL\/uqvK6pqynw0IVAd3ahDkbXDHdpsnXSj1kk9861AgATGChRdWAdcp0eMGCFz586VPXv2qKOkkiVLKq8kBHLE704kEhgSGJ2c6QgM3KgrnZ4oj5ZcoDQw+DESDXlD0bXDINdsndHm3Ux9NOLVrSJ\/5pPAeHTehw0bpsIKIPJ0iRIl1CixUcyYMUM6d+4s7du3d2TkJDAkMDpB0xEYw40aBAY\/wVoYXd3M9wYCJDDemEerR0ECYzWiLqkPt53iwrrrrrsupEe4jbdnz56yfPlyR3pKAkMCoxO0RAlMsC1MtAvurPz6R11mtAYZKcf+hyJAAqNbRf7MJ4Hx6Lxfc8018uWXX6p7YILTmTNnFKnZtm2bIyMngSGB0QmajsAEG4MaWphWm3rLc61ayqqdR2TgsvNxc4KT1W68drgax6qT0ahJYHTrhvkiJDAelYJGjRrJE088IfXq1QsZ4YoVK6R\/\/\/6ycqUzt5uSwJDA6JaYjsCgfLBL7s6b28uGrO2kXuM+qurwUAMgLwu7VIj691hl3JI3vNHF0n3p7yEhFIxxuaWPwNgp\/LcMbqbaohu1bjX5K58ExqPzDeNd3Lxbt25dMUjEjh07FHHp06eP3H\/\/\/Y6MnASGBEYnaPEQmOA6gqNVb8jaNmKgR6u9YMx645gtF00DY7Y+p8tZjT+PkHSryJ\/5JDAenndE\/p0yZYrs3r074IWE+Ei1a9d2bNQkMCQwOmFLlMCgPkSrRmq3Z17UQI\/R2nXansVse+z\/PwiQwOhWkT\/zSWD8Oe8ZGjXiKb388ssCggR3bMRZeuGFFwR2N+GJBIYERidsZgiMUScDPerQ9UY+CYw35tHqUZDAWI2oS+rDxXULFiyQDz\/8UHArb\/bs2eWyyy4TRKnOqAZm5MiRirxMmjRJxVrCv2PHjpXVq1eTwMQx\/xnZsOOoPuUeyQgeiRIYs9oQp8tR+xKKAAlMyi1rRzpMAuMIzM43gntgRo8eLTVq1FDEBd5HP\/zwg6xbt06efPJJefjhh013qmvXrlK8eHHp1auXqgMbEGxt4KJ90UUXhdRLDQw1MDpBywiBiRbokV5Ioag77UVlNf4kMLpV5M98EhiPzjuIy8SJE+XKK68MGSEiVCM2UiRtSbxQzJw5U9WNgJAFChQQaGSg6Zk3bx41MHGAmJENO47qU+6RjOIRLWiglcEEdYEjAbqV7TW\/8pzsO5tPxXlKJICl1f3IyLitxIPBHFNuWTvSYRIYR2B2vpFbbrkloqs0QgxUrlxZ3RFjNuF4qlu3brJs2TLJlSuX5MiRQyZMmBDTBgZttW3bVtq0aWO2Wc+UQ2iHtLQ0z4wnowOxAo8cP78tJwq1DenKuHVHZOzaI+pv8OrpWDWvVC6aQ\/3u9rzC5w7Jkn25U7b\/wNlKjHu1uU3NG92oM7ravFWeBMZb8xkYDY55EPOofPnyISN8\/\/33ZenSpTJ06FDTI+\/Xr5\/89NNPMmTIEHVkhPrgsg0tTP78+UPq5RFSepgzqnEwPXEuLZhRPILdqnOW7a5Gia\/\/JqM2phux0+7EZtujG3Xo1PEIyaWLN8ndIoFJ8gTY1fzgwYOVcW21atWkWLFiygYG7tSwU2nWrJnSmhipd+\/eCXWjVq1a8vTTTyuDYCOVK1dOBY9ECIPgRAJDAqMTrowSGNRvuFXnb3r+Vt7wy+2C++C0Qa7Z9qLhZrY+p8tZ2X8SGN0q8mc+CYxH5\/2uu+6SrFmzxjW62bNnx\/Wc8VDHjh2VVxM0MBdccIGsWbNGBYeEBqZo0aIkMBo0rdiwE5owlz9sBR7hWphEvZNcDpHvu0cC43sRiAgACQzlImEEfvnlF8ExErQ5IEmwg+nRo0e6sAWomBoYamB0AmYFgUEbBomBFiYagXFaC2G2PSu1F6jLbD\/MlrO6\/yQwulXkz3wSGH\/Ou2OjJoEhgdEJm1UEBu3gKCln2R6CiNWRgjw67U5stj0GcwyVGhIY3SryZz4JjD\/n3bFRk8CQwOiEzUoCEx4nKVXdkOlGHeo+Tjdq3SryZz4JjD\/n3bFRk8CQwOiEzUoCY2hhshWoLm8ceyWghcFRCO40qVk6r+pO8AV4bsy7LPNhmfVtppTtP3C2EuPW9SupeaMbtW41+SufBMZf8+34aElgSGB0Qmc1gTG0MDhGwk9wMuvW7HQ5ulHzCEm3bpgvQgLjUSmoU6eOwBMJLtPhnkFODpkEhgRGJ29WExi0t3RSI6mW9xsp89GEdM2bNUx1ulw03Jzuh9n2rOw\/bWB0q8if+SQwHp133MmyZMkS2b59u9SsWVPuvvtuadiwoQq+6GQigSGB0cmbHQSGbtQ61FMrnwQmtebLqd6SwDiFdJLa2bVrlyxatEgWL14sBw8eVJfP3XPPPSqcgBOJBIYERidnThIYs9oEp8tZqb1AXanefxIY3SryZz4JjI\/mfe7cudK3b185duyYXHXVVSqeUaNGjWxFgASGBEYnYHYQmGhRqs26NTtdjm7UoVJDAqNbRf7MJ4Hx+Lzv3btXQFzmz5+v4hchyGOLFi0EAfRwk+5DDz2kiIxdiQSGBEYnW3YQGLRpZTTkjERlNtMXulHTjVq3bphPI17PygCOjRAiYNWqVVKqVCm59957lR1MwYIFA2Nev369IjAZiUytA5AEhgRGJyN2ERij3fA4Sfi7lS6+drhm0436fEwrw8WdbtS6VeTPfGpgPDrvOCJq3LixtGzZUqpWrRpxlCdPnlQEZvLkybahQAJDAqMTLrsJTHicpFSIVE03ah4h6dYN86mB8awMwM4lT548SR8fCQwJjE4I7SYwaD84TlKqRKqOhpvTBrlm27Oy\/7SB0a0if+ZTA+PPeXds1CQwJDA6YXOSwCBOUtGx5XRdYr7LECCBcdmEuKQ7JDAumQivdoMEhgRGJ9tOEJhgLUyrTb1l7ZGy6bplVtNgRzkrtReoy44+xqrT6v6TwOhWkT\/zSWD8Oe+OjZoEhgRGJ2xOERj0Awa9kUIMIM9pV+lY7dGNOlRqSGB0q8if+SQw\/px3x0ZNAkMCoxM2JwmMYQuzIWs7RWRW7TysuodYR067Ssdqj27UdKPWrRvm04iXMmAzAiQwJDA6EXOSwBhaGPxrxEliNGp7InRb6apON2rdKvJnPjUw\/px3x0ZNAkMCoxM2pwlMtGjVTkecjtUe3ah5hKRbN8ynBoYyYDMCJDAkMDoRc5rAwI260umJ8mjJBemiVTtt7OqkIazTY4s272b6QRsY3SryZz41MP6c9wyPesWKFfLyyy+rAJHFihWTF154QapXr56uXhIYEhidsDlNYBipWjcj7ssngXHfnLihRyQwbpiFFOvDd999J3feeaeMGTNGkRaELECspWnTpkmmTJlCRkMCQwKjE2+3EBgzmgGMzY5yVmov7Oqjk9ojEhjdKvJnPgmMP+c9Q6OG5uXo0aMyYMAAbT0kMCQwOiFxmsCkQqRqulGHSg0JjG4V+TOfBMaf856hUSO+Urly5WT79u2ya9cuKVq0qDzzzDNSvnz5qEdIyGjbtq20adMmQ217oTAigaelpXlhKJaMIRl4fLH3hIxbd0Q+33NCjaFTtbzSsep5bxw35DW+7A85mKmAq\/uow8tKHGc+fbeaG2h\/mYiAgQAJDGUhYQQQJPLUqVPy3\/\/+Vy699FKZMGGCjBs3TlauXCm5c+cOqY8aGGpgdALmtAYmUn9wwV3+pucjIFvp\/ms2UjWjUTMatW7dMJ9eSJQBEwjcd999Uq1aNenRo0egNLQvo0ePlho1apDAaDB1w4ZtYtptK5JsPIKjVW\/I2laajNqYbqxOu1jTjTp0CniEZNvyS+mKqYFJ6elLTuefffZZyZIli\/Tr1y+EwEALA2ITnKiBoQZGJ6XJJjDon0Fi2u2ZF7idN7zfdhjrOmkIm8r9J4HRrSJ\/5pPA+HPeMzTqr776Sh544AGZMmWKXHfddTJx4kQZO3asOkLKmTMnCQw1MAnJlxsIDDocK05SQgPiw5YjQAJjOaSeqJAExhPT6PwgZs6cKSNGjJBjx45JmTJlpG\/fvorMhCdqYKiB0UmnWwhMtBt60f9U1l54of8kMLpV5M98Ehh\/zrtjoyaBIYHRCZtbCIyhhVl75CpptalXSLedjlRNN+pQqSGB0a0if+aTwPhz3h0bNQkMCYxO2NxEYCJpYZIRqZrRqBmNWrdumE8vJMqAzQiQwJDA6ETMTQQGfV06qZFUy\/uNipOUrEjVdKOmG7Vu3TCfBIYyYDMCJDAkMDoRcxOBWb3riHyw+EUV6PH13XeoHyS6USfmWh7NZsgsjjxC0q0if+bzCMmf8+7YqElgSGB0wuYmAoNI1at2Hg4QmOC+05A3\/Uw65QZOAqNbRf7MJ4Hx57w7NmoSGBIYnbC5icAwUrVutpKTTwKTHNzd3ioJjNtnKMX7RwJDAqMT4VQgMNS+JE\/7gpZJYHSryJ\/5JDD+nHfHRk0CQwKjEzY3ERi3RKqmG3Wo1JDA6FaRP\/NJYPw5746NmgSGBEYnbG4iMOgrDHlBZGALgwTDUxj15izbXQwbGWNM0Mws7FJB\/Wpl3vBGF0v3pb+HhDWwqy07+o86rcRjy+BmCmNGo9atJn\/lk8D4a74dHy0JDAmMTujcRmDC+2vcDbPt2o2OBXpkMEdqYHTrhvl0o6YM2IwACQwJjE7EUoXABLtVh4\/JDhuZaLjZ0ZYddVrZfx4h6VaRP\/OpgfHnvDs2ahIYEhidsLmdwKD\/hham1abesvZIWd2QmG8xAiQwFgPqkepIYDwykW4dBgkMCYxONlOBwGAM0aJVu117gb7b0Uen7oBB\/0lgdKvIn\/kkMP6cd8dGTQJDAqMTtlQhMNGiVdsR6JFeSKFSQwKjW0X+zCeB8ee8OzZqEhgSGJ2wpQqBMbQw+BdxkpDsCvTIYI4M5qhbN8ynES9lwGYESGBIYHQilkoExtDC5CzbQ7lVB6fgO2QyGgSSwRwZzFG3bphPAkMZsBkBEhgSGJ2IpRKBwVgMEpO\/6flNFgl3xzQZlVjAw1iBDelGzSMk3bphPgkMZcBmBEhgSGB0IpZqBOb0r2vl6OqWEqyFCb+0LXjMZg1oo+Fmtj6ny1nZf9rA6FaRP\/NpA+PPebds1IsWLZLu3bvLzJkzpUqVKunqJYEhgdEJW6oRGEMLE3yExCCQulnOWD4JTMbw82ppEhivzqwD4zpw4IA0b95cjh49KuPHjyeBiRPzVNyw4xyaqce8gEc0AmNW62Gl9gJ1me2H2XJW958ExtTS8nwhEhjPT7F9A2zbtq00adJEhg0bJkOHDiWBiRNqL2zYcQ41rse8gIfVQSDpRh0qOiQwcS0l3z1EAuO7KbdmwJMmTZJPPvlEaV5q1apFApMArF7YsBMYrvZRr+ARKQhk74Yl1fgTzaMbNd2otQuHDwgJDIUgYQSw4bRu3VrmzZsnBQsWjIvAoBFobNq0aZNwe14rsGfPHklLS\/PasEyPxyt4jFt3RMauPaJwgAalY9W8UrloDvV7onmFzx2SJftyW1af2X5kpFyiY47VVq82tykcGY3a9DLzZEESGE9Oq32DOnPmjLJ76dSpkzRo0EA1RA1MYnh7ReOQ2KijP53qeBhu1cbldsEjjeUqTTfqUJmIhQePkKxabd6qhwTGW\/Np+2i2b9+uCEyePHkCbe3fv1\/y588vHTp0kI4dO4b0gV5I6ack1Tdsq4XMC3hEi5OkCH6ZfLJq5+GIsDkZT8hsP8yWiyYnZuojgbF61XmjPhIYb8xjUkdBDUxi8Hthw05sxLGf9gIeL77eQx4tuUBe332H+mGyFgESGGvx9EptJDBemckkjoMEJjHwvbBhJzZi7xMYuFHvvLm9GmjwUZIZbUMstMzW53Q5K7UvqIsExsoV5526SGC8M5euHAmPkHiEpBNMLxA6uFEf3z4snRbGbKRqulGHSg0JjG4V+TOfBMaf8+7YqElgSGB0wuYFAoMxwlW63MHOcurQZ0oLk5FI1XSjphu1bt0wn7GQKAM2I0ACQwKjEzGvEBiMM1q06kQjVTMaNaNR69YN80lgKAM2I0ACQwKjEzEvEZhgEmNEqzYTqZrRqHmEpFs3zCeBoQzYjAAJDAmMTsS8RmAwXrhVG9GqzUaqjoab0wa5Ztuzsv+0gdGtIn\/m0wbGn\/Pu2KhJYEhgdMLmRQKDoyQjWjUjVeskQJ9PAqPHyI9PkMD4cdYdHDMJDAmMTty8SGCCx2wmUrWV2gvUZVaLYrac1f0ngdGtIn\/mk8D4c94dGzUJDAmMTti8TmDMRKqmG3Wo1JDA6FaRP\/NJYPw5746NmgSGBEYnbF4nMBg\/o1FLhtzKx3auo8SIwRx1q8lf+SQw\/ppvx0dLAkMCoxM6PxCY5Uv6SvPlNyoocCzTu2FJqVk6r\/o9kos13ajpRq1bN8ynFxJlwGYESGBIYHQi5nUCY9wN02pTb1l7pGwADkajDpUMRqPWrRTmhyNADQxlwlYESGBIYHQC5nUCAzfqiWl3RQz0yGjUodIRDQ\/awOhWkT\/zSWD8Oe+OjZoEhgRGJ2xeJzDwQkKkakar1klC9HwSGPPYebkkCYyXZ9cFYyOBIYHRiaEfCAwwCI9WTe1LfNoXPEUCo1tF\/swngfHnvDs2ahIYEhidsHmdwBhGuuFamFiRqulGHSo1JDC6VeTPfBIYf867Y6MmgSGB0Qmb1wkMxm+4UVc6PVEdJY3PvlJ5IgXnrdp5WP0OY1ZGo2Y0at26YT69kCgDNiNAAkMCoxMxPxCYYAyC4yTh73Sj1ruVt65fSUHIe2B0q8lf+dTA+Gu+HR8tCQwJjE7o\/EZgDLdqBHvckLWtNBm1MR1EjEbNIyTdumE+NTCUAZsRIIEhgdGJmN8IDPCAFiZbgeqCu2GMoyMdTka+2fhETpeLNh4z\/aANTLzS4a\/nqIHx13xbMtrTp0\/Lq6++KgsWLJATJ05I6dKlpU+fPlK+fPl09ZPAkMDohM6PBMaIVs1I1TrpOJ9PAhMfTn57igTGbzNuwXhHjhwpS5YskYkTJ0qBAgVk+PDhMmfOHFm9ejUJTBz4+nHDjgWLn\/FIlMCY0V4Ae6fLWal9IYGJ46Xi00dIYHw68RkZ9qpVqyRPnjwBjQs2oDp16sjWrVslZ86cIVVTA0MNjE7W\/ExgokWqpht1qNRQA6NbRf7MJ4Hx57xbOuoxY8bIihUrZNasWVE1MMho27attGnTxtK2U7GyPXv2SFpaWip23ZY++x2PL\/aekHHrjsjne04ofDtVyyuNL\/tDDmYqkO7vHaueDwAZqYzb8qzs48yn71bjpheSLUswZSslgUnZqXNHx+fPny8DBgyQ6dOnS6lSpaISGL54\/oHGzxqHSFJLPCJr6WZ9m0kGLguNyhwrgrXb8iK5h5vtI92o3fG+d1svSGDcNiMp0p+zZ8\/K4MGDZeXKlTJ27FgpXrx4xJ7zCIlHSDqR9juBCXarzlm2u4Jrzqpt0nHugXTQxYrY7Ka8aDY3ZvvIIyTdKvJnPgmMP+c9Q6M+d+6c9OrVSw4dOiQjRoyQ3LlzR62PBIYERidsficwwAdu1Uj5m57XuDR47dPAkVI4fk4b5JptL9q8m6mPBEa3ivyZTwLjz3nP0KinTZsm+Jk3b55ky5YtZl0kMCQwOmEjgREJ18Ik6p2kwzjV80lgUn0G7ek\/CYw9uHq61saNG8uOHTskS5YsIeOcOnWqVK5cOeRvJDAkMLrFQAJzHiGDxEALE43AmNFeoG6ny1mpfUFdJDC6VeTPfBIYf867Y6MmgSGB0QkbCUwogUGIgX6bbpSxa4+kgy5WBGs35UUjTGb7SAKjW0X+zCeB8ee8OzZqEhgSGJ2wkcD8g5Chhfnz8qHyyMclQ8IMgBQs7FJBPdz0jU2uz7Oyj1sGN1PjpjejbjX5K58Exl\/z7fhoSWBIYHRCRwITihAMetecbiFtVjVMB51ZLx6ny9ELSSf1zLcCARIYK1BkHVERIIEhgdEtDxKYUIQMLczru+8Q\/IQnp+1ZzLYXbd7N1Mcj0DSOgQAAD1hJREFUJN0q8mc+CYw\/592xUZPAkMDohI0EJj1Chlt1mY8m6ODzRT4JjC+mOeFBksAkDBkLJIIACQwJjE5eSGDSI0QvpFBMSGB0q8if+SQw\/px3x0ZNAkMCoxM2Epj0CD09cyO9kIJgIYHRrSJ\/5pPA+HPeHRs1CQwJjE7YSGAiy8i+s\/kE8YRW7TysHoAhbu+GJdX\/V+864vo8K\/s4tnMdNW56IelWk7\/ySWD8Nd+Oj5YEhgRGJ3QkMInLiJWBEs0GWNSVs7KPDOaoW0X+zCeB8ee8OzZqEpjENyfHJsclDZHAxJYRGPTicjsj0CM0G01GbUxXyGlX6Vjt0Y3aJYvL490ggfH4BCd7eCQwJDA6GSSBiS0j4XGSwi+ICy5txkUZ5e0oF23ezbRFGxjdKvJnPgmMP+fdsVGTwJDA6ISNBEYvI8HRqv0Y6JEERreK\/JlPAuPPeXds1CQw+s3JsclwaUMkMHoZCdbCFB1bLuJMmtFspIL2BX0kgXHp4k1yt0hgkjwBXm+eBEa\/OXldBnTjI4GJT0YMEjM++0oZuOz7dIXMBkq0oxyDOeqknvlWIEACYwWKrCMqAiQw8W1OfhYhEpj4ZcQw6N2QtS3dqP28aDh2hQAJDAXBVgRIYOLfnGydCBdXTgITv4wYWpgNWdtJ8+U3qoLQduB+GJ1bM5610rVZ156VbdGN2sULOIldI4FJIvh+aJoEJv7NyQ\/yEGmMJDCJyQi0MGuPXCWtNvUKKUg3ar+uIP+OmwTGv3PvyMhJYBLbnByZFJc1QgKTmIwsX9JXKp2eqCJVh0er9qohL414XbZoXdIdEhiXTEQqdePcuXPy2muvycyZM+XEiRNSvnx5eeWVVyQtLS3dMEhgEtucUkkOrOorCUxiMgI36qkVXpVqeb8Rv0SrJoGxarV5qx4SGG\/NpyOjmTp1qrz55psyefJkKVy4sAwaNEjWrVsn8+fPJ4GJYwb69u0rffr0ieNJfzxCPNLPcyxMUiFSdTTJNashIoHxx7sg0VGSwCSKGJ+Xe++9V+644w554IEHFBrHjx+XChUqyLvvvitXXHFFCELUwKQXGGDCoHT\/4EI8EpORYOPY4JJ2uEObrZNu1NwonECABMYJlD3WBsjKmDFjpHr16oGR1a1bVx5\/\/HG57bbbIhIYj0HA4RABIpAEBEj8kwC6i5skgXHx5Li1a2XLlpUZM2ZIxYoVA11s3LixtG\/fXu655x4SGLdOHPtFBFIcARKYFJ9Ai7tPAmMxoH6oDhqYESNGSO3atQPDrVWrljz77LPSqFEjP0DAMRIBIkAEiECSESCBSfIEpGLzLVu2FBwZdejQQXX\/8OHDUrVqVXn\/\/felVKlSqTgk9pkIEAEiQARSDAESmBSbMDd0d86cOTJ8+HDlhXTppZdKv379ZOfOncqtmokIEAEiQASIgBMIkMA4gbIH2wCBmTJlivJAqlKliroHpkiRIh4cKYdEBIgAESACbkSABMaNs8I+EQEiQASIABEgAjERIIGhgBABIkAEiAARIAIphwAJTMpNGTtMBIgAESACRIAIkMBQBmxBAJ5Jzz\/\/vHz88ceSOXNmufXWWwXXo2fPnt2W9lKh0ocfflg+\/PBDyZIlS6C7bdq0kWeeeSYVum9JH3\/55Rc1XuCwbdu2EHnA73DF37Fjh+TKlUt5uT344IOWtOvmSmJhgisLYGeWKVOmwBBGjhwp9erVc\/OQTPft9OnT8uqrr8qCBQtUnLXSpUursBuIt4bkVxkxDajHC5LAeHyCkzW8bt26ycmTJ1XQRwR\/xGZUrlw5X23W4djfd999cuedd0qLFi2SNS1Jbfenn36Stm3bSrNmzZRcBBOYv\/\/+W2655RYVngKy8sMPPwjc9QcOHCh16tRJar\/tbDwWJlg3ZcqUkZUrV0rx4sXt7IZr6gY5W7JkiUycOFEKFCigvB3h9bh69Wrxq4y4ZnJc2BESGBdOSqp3CV+M+GJavHhxIDbSmjVrpHv37rJ+\/fpUH57p\/uO2YhA7v172d+DAATl79qzSKNSvXz+EwKxatUqFoli7dm1A2zBkyBBBpGpsal5NsTA5evSoijH2xRdfSL58+bwKQci4IAd58uQJaFww\/yCwW7duVTj4UUZ8MfEmB0kCYxI4FouOAL6sb7\/9dnUUgOMjpJ9\/\/lnFTvr8888lf\/78voQPtxVfffXVsmvXLqUeBx44TsGXpp8Sxh9OYN5++21ZunSpClFhpIULFyryggsSvZ4iYQLtzE033SRNmjRR6wbHr02bNpWuXbtK1qxZvQ6JGh9irq1YsUJmzZolfpcRX0x4goMkgUkQMD6uRwBfSvfff79s37498PCxY8fUVxVsYtLS0vSVePAJ2ADhpmJE8waB6dmzp5w6dUomTZrkwdFGH1KkzfqNN95Q2hdsUkYCccElifgq93qKhMm+fftUyI4GDRrIjTfeqIgvbIJwFNmlSxevQyLz58+XAQMGyPTp09W68buMeH7CTQyQBMYEaCwSGwFoYBCV+ptvvpELLrhAPbxnzx71EvaTOlwnJ8AJx0pbtmxRRqt+SdE0MDDcnDt3bgAG2D68+eab8t5773kemkiYRBr0hAkTZNGiRTJv3jzPYoJjxsGDByvbn7Fjxwbsf0Bu\/Swjnp3wDAyMBCYD4LFoZARgvAuD3XfeeUeuu+469dDy5cvVccm6det8CRswAXmrVq1awAvpyy+\/VNG7v\/766wDR8wM4kTbrzz77TDp16qQwMo5H+vfvL\/DQef311z0PSyRMMPa9e\/cqOxgjjRs3TnlweTVsBwyXe\/XqJYcOHVLap9y5cwfG7ncZ8fwiMDFAEhgToLGIHgEcjxw5ckR5m2DzhmcJolc\/8cQT+sIefAIY1KhRQ9q1a6fU\/zDQ7NGjh9K8QDXupxRps4b7LOxi7r77bkVkcPzYunVrtYnBdsjrKRImILjwWIMWChjApuzf\/\/63Wkvw5vJimjZtmuAHGqZs2bKFDNHvMuLF+c7omEhgMoogy0dEABv0c889Jx999JEy5IX7MH73i\/FhJFCwIb388svqaC1HjhzKu+Kpp56Siy++2BdSNHr0aOUWi69s2P4Yx4s4FgG5wwYNLR2O1GDo\/cgjjyhbKi8nHSawA4EhM7QxkBPYv+A+oeB7YbyED45UIQfBdyVhfFOnTpXKlSv7Uka8NL9Wj4UExmpEWR8RIAJEgAgQASJgOwIkMLZDzAaIABEgAkSACBABqxEggbEaUdZHBIgAESACRIAI2I4ACYztELMBIkAEiAARIAJEwGoESGCsRpT1EQEiQASIABEgArYjQAJjO8RsgAgQASJABIgAEbAaARIYqxFlfUSACBABIkAEiIDtCJDA2A4xGyACRADxjNq0aSPffvttxLuAEB8KwS3tuOgQ9+\/cddddKsoxwlngcrxEEi7Y279\/v\/z1118hEbQTqYPPEgEiYD0CJDDWY8oaiQARCEPg999\/V5s\/QingErY1a9aoW4gR4BMJl9fhorZixYpZjp1BYDZv3hxyNX0iDW3cuFHdEowxICo0ExEgAslHgAQm+XPAHhAB3yGAqMq4ibhVq1a2j50ExnaI2QARSAoCJDBJgZ2NEgH7EXjyySflu+++E0R1htbj8OHDUq9ePXn88ccjEgccs9SsWVN++OEHpRH5+++\/BUc73bt3D1xdj6vtEVDwxx9\/lEKFCqmjGcR2wtXvuO7++eefl7Vr18rx48elTJky0rt3bxUDK\/gICfF88Ds0Gddcc42KQB1+hBSrnaFDh8qmTZvUcRDGhsB\/1157reDv+fLlSwdsOIFBKIPSpUuriMcIigiMLrnkEhXmAAFIESzx2LFjKuYQfpCogbFfXtkCEUgUARKYRBHj80QgRRBAPKpGjRopggFNx9NPP62iG0+aNCniCEAiEERx8uTJ6mgH\/2\/atKkMHDhQxbJauXKldO7cWcXmAXmAPUv79u1VvCIEpoT9ihHAM2fOnDJ79mwZNGiQIjTr168PsYGBvUu3bt0CRCqYwOjaQftjx45VbUKTg3Eihg6OeB577DEtgcEDIDA33HCDjB8\/Xi688EIVY2jnzp3y0ksvqbree+891T\/0HXGZSGBSROjZTV8hQALjq+nmYP2GwCeffCJdu3ZVRAIRwpcuXSpFixaNSmDy5s2rNnUjQVuSO3duZfgKbQQCMI4aNSqQD60HIgd\/\/PHHKkIytCqIrm0E7Tx79qwK5hluxBuLwOjaAYGZOHGiIkVGUENom2BkG9w3o5ORjpBAYPr16xcgUCBpixYtUv1EAimqUKGC0shUrFiRBMZvC4fjTQkESGBSYprYSSJgHgFEeJ4+fboMGDBAWrRoEbUiaEGuv\/56eeGFFwLPPPvss0ozgaOWBg0aSMOGDUM8hXDUA63Hrl27VJRtkA8cPeHY6Oabb1YaoGzZsiVEYHTtgCBBQ\/Luu++G9PPAgQPy1ltvxa2BgRYHR2pIw4YNE5A9EBakkydPytVXXy3Tpk1T3lHUwJiXP5YkAnYhQAJjF7Kslwi4BIHWrVurDRhHPSAz0RIIDGxS+vbtG3jkqaeekn379qljp0jEAvYrODqCHQm0IadOnVLHLtDILFy4UIoUKaLITyJHSLp2oGVZtmyZ0pgYCUQrowRm9erV6tiLBMYlgstuEAENAiQwFBEi4GEEpk6dquxF8AMbkbfffluqVKkSccQgMLAHwTNGgm1IiRIllPamY8eO6jhozJgxgfwhQ4bI4sWLZcWKFcqYFkdQxvERjIYrV66stD8gNsH3wMQ6QtK1gyMkEhgPCy2HRgTiRIAEJk6g+BgRSDUEYLB76623yujRo6VWrVrq3xkzZsiSJUvUHSzhCQQGhrmwl6lbt666q6Vdu3bK3gRHQjhieeihh1Q9N910k\/JUwpER7GQ6deqkDHubNGmijIZhxAtS88gjj8hHH30k33\/\/fQiBgQs1jm9gFAzPoWAj3ljtGEbEJDCpJo3sLxGwHgESGOsxZY1EIOkIwFUYnkcw2AUhQTp9+rTyKoL3DQxYIxEYHCHBkwieQDDexfETSIORoNGBG\/XBgwdV3S1btlSeQNDMbN26Vfr376\/+hcbl8ssvl0cffVQdPYUb8aIOGAAXLFhQHTeFu1HHaocamKSLFztABFyBAAmMK6aBnSACyUcAJALkBne3eCnxIjsvzSbHQgT+QYAEhtJABIiAQoAEJrog0AuJi4QIuA8BEhj3zQl7RASSgoDXCQyDOSZFrNgoEbANgf8D7srtJQNcbPoAAAAASUVORK5CYII=","height":337,"width":560}}
%---
%[output:4e00c158]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:9e4a146d]
%   data: {"dataType":"text","outputData":{"text":"Running 500 joint evaluation episodes...\n","truncated":false}}
%---
%[output:82b013a1]
%   data: {"dataType":"text","outputData":{"text":"Evaluating episode 50 \/ 500...\nEvaluating episode 100 \/ 500...\nEvaluating episode 150 \/ 500...\nEvaluating episode 200 \/ 500...\nEvaluating episode 250 \/ 500...\nEvaluating episode 300 \/ 500...\nEvaluating episode 350 \/ 500...\nEvaluating episode 400 \/ 500...\nEvaluating episode 450 \/ 500...\nEvaluating episode 500 \/ 500...\n","truncated":false}}
%---
%[output:8545d6f8]
%   data: {"dataType":"textualVariable","outputData":{"name":"idx","value":"451"}}
%---
%[output:65c66fc1]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7sXQnYTVX3XzJk5lXKmCGpJLP4vDJnyFCUIWMqhJImUoZXGsyFEKEMKV4RSihjhlQylDFkyJB5zBj\/57f9z3Xf+557zz73nmkfaz2P5\/t67zp7+K19z\/7dtddeK8W1a9euEQsjwAgwAowAI8AIMAIKIZCCCYxC1uKhMgKMACPACDACjIBAgAkMLwRGgBFgBBgBRoARUA4BJjDKmYwHzAgwAowAI8AIMAJMYHgNMAKMACPACDACjIByCDCBUc5kPGBGgBFgBBgBRoARYALDa4ARYAQYAUZACgFcWv33338pQ4YMUvrBSm+88YbpZ\/BA\/\/79Iz53\/vx5wj\/IiBEjxP+Pi4ujdu3aUbZs2Uz1+f3339Nvv\/1GJ06cCDyHtkqVKkWPPPKIqbagbHV7wQPAPD\/55BPxJ8w1Xbp0psen+gNMYFS3II+fEWAEGAGHELh8+TJ9\/vnnVKNGDcqTJ4+pXkFgjMhIaINGz\/z66680Y8YM8RiIBohH9erVadGiRZQrVy46cOAAFSxYkFq1ahVxg9fIAJ6vUKECPfDAA4HnN23aRKtWrQqQIhmiYHV7obgEt6\/NHSQGhAZzliF+poznUWUmMB41DA+LEWAEGAEvIjBo0CDKkiULxcfHi41eVozIiF47Rs+MHTtWkBZ4SEBaQFbat29P+PuuXbvE3+FRweYOghPOIzN8+PCInoxgb0eXLl0Mp2x1e3qeF8wb84JMnjxZEDT896hRo+jIkSOmyaLhpDyowATGg0bhITECjAAj4CUETp06RfB27N69m1KnTk3lypWjpUuXUvHixemhhx6iW265xXC4RmQkWgKTNWtWatKkCaH9YAKD9kBm8Hds7iAh2ufBfeGYBx6Wbt26JfHShI4Xzw8cOFB4aCIdJ4VrT8+LItOeHnnB31q2bBkgZKHHSSAxr776qqFNVFdgAqO6BXn8jAAjwAjYiMDOnTtFLEfOnDkDRyvp06envXv30ldffSVITLVq1QxHYAeBwbi2bNlC8IrA66KRFvx\/EJv69etT3759BTEpWbKkICqhx1gDBgzQjXHRG68W09K9e\/ew8w3Xnt4DMu1pzxnFvBh9bmggBRWYwChoNB4yI8AIMAJOIICjiGnTplH58uXpnnvuoT\/\/\/JP+++8\/Klu2LP3444\/ivytWrEj33XdfxOFgcwWRsCoGBnEeOBLCMRGOT9AuyMBff\/0VOEIqUKCA8LggLuTJJ58UJEbTDR4siAoIEGJmQv8eOl70i+OhSPMI154eQDLt4TlZchKshzmHzsmJNeNkH0xgnESb+2IEGAFGQCEEZs2aJTb+0qVL04IFC8TIT548Sfny5aPjx48Lz0vevHkNyYt2W0aLHzG6kaQRBD0vyJw5c4QnBeMCQUGQLWI\/Lly4QCtXrhQEBkTl\/vvvFyQHfeN4aO3atQGCE46oRBpXpDHZ2Z4seQn11GgxMn4mMUxgFHqZ8FAZAUaAEXAKgcOHD9PXX39NderUEQGyOXLkEN6WkSNHErwbtWrVEsG8RgKPBTbTF1980fS1Zj0CgyMa3DQCOZkyZYrwTtStW5cefvhhQaqCA3XxGbw0CDYGqbn11ltFvIxKHhijgGA9\/DHvuXPniuO1Pn36GJlI2c+ZwChrOh44I8AIMAL2IbBx40ZC\/AtiX3DU0ahRI9HZihUrBJGRFasJDIKJQajgzQExgocFniAQmEgSLjbF6zEwwA8eJjNHcBrxmz59ejLCJms3FfSYwKhgJR4jI8AIMAIOIbB+\/XoR54L4FpCXPXv2iOOau+++m65cuUIpU6YUn2sCL8y9994bdnR6V5BjOUJCR5pXAiQGhAbHJEZHJegTRADzCBbE8ixevDiZhyjU+wPvDhLlwdtTpkyZsPMN117oA7Ltac+ZCYI2o+vQsrKlG18TGHxxkLPg008\/pRQpUtDHH39MNWvWtAVIbpQRYAQYAT8gMGTIEJGoDjEjuGH0888\/04MPPiiOX5CJN1iQ2A4xKIg7iSTYrHFl2MogXpAY7ejICHd4kuCpwXGKXiI6jRAFX00OblMjG\/ibzJGM1e2h31BSEkoCg7FlAmO0Ijz+Ob58r7\/+ushboAkTGI8bjYfHCDACriMwfvx4evbZZ4VXYvPmzaJsQPPmzQWB0ZNJkyZR69atxUfIiovbL3oSzaYa6RnZfCsYC7w03377bVjyEZo5F8HB8NSA+CCGBkHDwOHMmTMiniY0jiZ0vla3F47AhAssjgZr1xdeFAPwnQfm0qVLNHjwYBo3bpyA4+mnn6bPPvtM\/H8mMFGsEH6EEWAEbioEhg0bRk2bNiW8S7UjIySqgxc7VKCzbNkyatu2LeHG0u+\/\/069e\/d2hMCAJMDTgYBiI0KBm0uHDh0y9BSB6ICsaOn4MREcTSF5HY6N4G1CMDAy\/Br1qREnq9rT88AwgQn1CSr+VcUvhnr16lHu3Llp6NCh4ipdsWLFmMAoblcePiPACDiDwLx58wRxgWikJdI2cdttt4lgWqTsT5MmjTjqcMIDgz40QqGXxyV4DAjUBQkxCvSVQVirvyRLYozalG2PCUxyJH3ngcG1MXhfELGdMWNGOnv2LBMYo28Qf84IMAKMQJQI4Ko1juzTpk0rYmR69uzpGIFBR8EZePU61uJfkAvGbHXqcJBopEM2BscIWpn2mMDcBATm6tWrSepyMIEx+urw54wAI8AIRIcACMs\/\/\/wj6iNpAo9MOA9MNL0YBf5q2WwjtY24FVSktlI00mE0Ptk+jdrT82wFHyGF9mPVuGTH74ae7zwwoSAygXFjWXGfjAAjwAgwAoyAvQgwgbEXX26dEWAEGAFGgBFgBGxAgAmMDaByk4wAI8AIMAKMACNgLwJMYCTwrV37LXE9kIURcAKBwoUzUapUh0UOChZGgBFgBGJFANXEp06dGmsznnueCYyESXLnbkDII8CiBgINGqhtr7Vrv6FcuUikA7gZBEnDkCyMRR0E2Gbq2Aoj9au9mMBIrEMmMBIgeUiFCYyHjCExFL++XCWmrqyK1222q3HSpHsFE5OWQAgGPlRXzyiRnoe+URtu919jbQFf\/khgAiPxCmECIwGSh1SYwHjIGBJD8fpmKDGFm07F6zZjApN0STKBUfQrasU1aiYwahkfia2Mist5eUY32xESkk7KFMjzss38MjYjT4I4jki8JhKF6tnM6HmnPRFGnhO\/2M1oHl4nnEbjD\/c5e2AkkGMCIwGSh1SQ2Ar1S1SVm43A\/PXXX6KeDUtkBE4k9g0oxDXuE1bZiETENUmgcM8bPasRmHA2M3qeCYz8Kh+w4EYh4tCnDp+5RJ+u2i\/fGBEdH1rVlL4KykxgJKzEBEYCJA+pMIHxkDEkhsIEJjlIwWTlxPSEZAp2EwEjs7lts0ibu9HYQz8fsOAvs48oqc8ERgGzderUiVauXJlkpCiBDkE5eBQb06REiRI0ceJEw1kxgTGEyFMKTGA8ZQ7Dwbi9GRoOMEoFI29EugeqUM6EJbqth3sW3hNIJA9MlMM19ZiTNgsmKzcL2dCM0b1WeM9k91r5pW3GR0jSULmr2KZNG\/rxxx+lBlG6dGlKTEw01GUCYwiRpxSYwHjKHIaDsXMzDPZkgDAc6FMl2Xhi8WZEOo45mFCVzm9aGnb+kfoNHrcXCEvoJCLZLJJ3xCoCUrFQHMXfndVwbckqmCEDsm16SY8JjJes4fBYmMA4DHiM3TGBiRFAhx+PlcBgs482piMW8gKY3AgSDUcQrCIHDpvfsLu5nUvSih0nA3p+JxuGgEShwAQmCtD88ggTGLUsyQRGLXvFQmBAXhAjEo5IeN2TIWOpbK\/oHzPJPOuWjlVHH26N32\/9MoHxm0VNzIcJjAmwPKCqOoEBhKVLewBIh4ZglsDoBbi64QkxC8\/KnSep\/sh1Zh9Lpt+sbA7Kly2dbjtOeSfM2izmSXMDMSHABCYm+NR+mAmMWvZjAmOfvUI9Gno9RQow1Xv+5IkTlDUuLtBUpOf14koixaFEg4TRDZedR\/6lGb\/9E03TUT\/jtRskTGCiNqUrDzKBcQV2b3TKBMYbdpAdBRMYWaSS6snkGTG6WYMWrYwrCSUTjcddv5UxNkubIMKT\/JqxDAJuxIx4jYjI4KSnwwQmWuTceY4JjDu4e6JXJjCeMIP0IJjAGEMVSkRuSZuRrl44G3hQNqbEjAcGZOREojHZGJv5BjkJbb\/96YkU6XPjmRtryNxwceqoxni07mgwgXEH92h7ZQITLXI+eI4JjFpGZAJjbK9wnpRcfZeKq79m8ozoHbkg3mPFjhPGAzHQkCETsXRysxORaLFjAhMtcu48xwTGHdw90atfCMyxY4do4sR3qGXLN+iOO\/J4AlsrB4H5rV+\/jP788w9Knz4d5c1bmCpUqEu33qof8KjX99WrV2nXrt+pUKHiVg7NVFuqlBKIJihV70osb4amlocnlNlmnjCD9CCYwEhD5T\/FWAnM4cP7aPr0YXT16n90\/Pg\/lC5dBqpYsQHFx9fXBevSpYv01Vcf0Y4dGyhDhsy0f\/9OeuihmtSwYSdKk+ZW3WeWLZtJK1bMpbvuupeOHNlPLVq8TnfeeVdAd8uWX2j06DeoYMGi1LXrsKiMtGHDClqxYjZt3vwznThxmNKnz0jZs+cR\/woXLkm5c99N06d\/SO+\/Pyuq9mN5aMaMEfTXX5uoRYtudPz4Sfr229H066+LqESJyvTWW58aNr1u3TJauXKuIEAgMRMmrDV8xi4FLxIYI7Kid21W1rvBm6FdK8m+dtlm9mFrR8tMYOxAVZE2YyEwGzb8SEOGdKaOHfvT\/\/73qJjxpk1r6J132lCVKk9Qu3b96JZbbgkgcfnyJRo8uCMdPLibBg36RngPzp07Ta+9Vpduuy0H9ekzhVKnTkpiQEyWLZtF\/fvPovz5i9CuXX\/Qhx++RCVLVqH\/\/rtMKVOmorRpM9APP3xBgwfPo7i4O0whf\/bsKRo2rKvY3EFSmjV7lYoW\/R+lTZue\/vprM\/3xxypKTBxOly9fpJQpU9OXX24z1X6sygsXfk6TJ\/enUaOWU6ZMcYQjpOzZb6chQzrRhQv\/UkLCVMMuMMcff5xNEyYkiDb8QGC0HCmYfCzXjPt+u4uGLdqjiyE8KrFmROXN0HB5ek6BbeY5k0QcEBMYtexl6WijJTD\/\/nuWunatQVmy3C7ISLBMm\/YBwWvQqdNAqlr1ycBHICOLF08Xm+4DD5QP\/B1ECKTnscc6UMuW3QN\/X736Oxo6tDPVr\/8ctW79ZuDv48cn0LPP3giYxLMgNHXrtjWFDchTjx4N6eDBvwQB69hxgPAghQq8RAMGtBd606btSELKTHUYhXLHjhUpV66C1KvXJPF0tDEwOIJ6\/vkKniEw8eeTe4FkrigjZf6h\/vXp6vnrNcAgic+ZK1hXsVDWZDlL7LpBw5thFIve5UfYZi4bwGT3TGBMAuYn9WgJzDffTBAxJ48++jS1bds7CST79v1Jr7xSS3hDRo5cJrwqp08fpw4d\/ke33pqePv30N0qRIkXgGRw\/tW1bSvz32LE\/BeI6evduStu2\/UYjRy6n22\/PGdD\/6KPX6IUXBov\/\/umn7wRZGjhwLt1yS0pTphk\/vg\/Nnz+Zbr89N40YsZhSpUod9nnMqVu3ejR+\/FpxvOSE4LisU6eHxRHb669\/HBOBOXnyCLVrV84TBCbdigTSIzDwpKxdu5ZQxytYEEirXTEO\/vvkCp\/SsH35YjIFEqeNeur+mNqI9DBvhrZBa1vDbDPboLWlYSYwtsCqRqPREhiQCMSmtGrVgxo0aJdsss2b30c4MkLMCIJGZ88eS1Om9E+yGQc\/1L\/\/c7R27WLq0OE9qlGjGV27do3atClOWbNmp+HDFwVU9+7dRhs3rqR69Z6h8+fPCaKEuJd77zWX3nX79nX01ltPiHZffHEIVarU0NBg8MJ06PCuGJMTgricd95pTf\/7X1165ZURAQJz5513CEKIWJ0cOfJLESqvEJilAxrQHWkPUI2CmSjtA0mLD2Z9sjd16vIyTUv9WDJ4ccU4WIKvG0dK7R7JTrJxLLHYmjfDWNBz51m2mTu4R9srE5hokfPBc9ESmISE5rRp00\/UqFFneuqpV5Mh0aFDBTp+\/JDwlFSu3IgGDuxAv\/zyfbJjIu3BTz\/tR\/PmfUrVqjURMTUgMK1bP0j33Vc2SaAqjqfg9UEsB2JDzp49KfTNygcfvEirVn0r+vnii63JYm\/02kPAMrxKwXE6f\/65nr77bqLw\/uCoCZ6lEiUqUcOGHZPogXjhSAzBtH36fE5\/\/\/0nwYu1bdtaEZCMeCEEC2vy5ptPCIJy9Oh+Sp8+E+XMmV8E4P777zk6ceIQXbp0QagOGDBHBC8Hy5Url2nmzFGEsSFGCM4u2ADxSnoxMPD0AFfMD3PIk+ceql69SRJShyOoX3\/9QYy\/Vq2WVKBAUfrqqxEEkoVYpkaNOlH16k2TwXbmzAnhIfvnn720b98OOnZkL6W+hSh1+kwCe8ip81eI6BqluHKRbklB9F+aTHQ1XRxdvuNBunT\/Y0S3pKRwJMUJEmJ2bQXr82YYC3ruPMs2cwf3aHtlAhMtcj54LloCM2bMm\/TDD19SmTI1qHv3scmQeO21R2nPnq3UrNkr9MQTL9AbbzxOO3dupObNXxebe6jgJlNi4jB68MF46t17svgYMTO\/\/baExoxZJQjCihVzxGZeqlRVwpEOSNQHHyygzJmzmbbEs8+WEV6M22\/PRaNHrzD9PB6YMeMjQV4Qn5I\/\/\/VjCIx38OBOol0QFQQn7927nZYvn0Vff\/2xIDiI9cGGXrz4w7R792ZasGCKiCUaPfrHJKRn+fKvacSIV5J5YLJkyUiIjTl\/\/iwNHbqA8ua9JzB+xPUMGNCOsma9g7p0GSqOxf799wz179+Otmz5ORmBAQlFTFGXLh+IOVy8eF4ESYOsaOQU\/eCo7ZtvxgvMQGAOHNhFpUtXF0RKC3IOJVPQ6devNT3zTAJt3LiC5s+\/HsdjJG9NWk6Xfp9LgwYNonLlytHkyZMpVapURo958nPeDD1ploiDYpupZTMmMGrZy9LRVq3agp566inDNkuXrpdEB7En+EUPYtGhQ0\/KkuW2oM+v0YgRPencuTNUs2ZjKlXqYfr44350\/RijH9Ws2SKgi2u1kF9+WUqLFs2kbNmyU\/v2vcTfsJkuXvw1HTq0jwoWLEalSlWjKlUaic9AXu66Kz8VL17BcOxQCB4\/jrZwxAXJm\/duatHiJcM2Que\/deta6tWrMVWr9jg99FC1JM8vXTqHfvrpBypUqCg9+WR78RmeB+k4evQAtWnTUxyBQTD\/iRMH08GDe6lp005UoMD1cUE2bfqF5s6dTPfdV5L69ftK\/E0L4u3cubLwmLRv\/xZly3Zn4Jnvv59Bf\/zxC3XqlJAkR0zmzHnozTcbJSEw8NR07vwwxcfXogceKBto48iRAzR+\/HWvVvv2PSlbtjvE+LVjw\/Ll64hju\/Xr5wudxYtn0c8\/L6Hy5WtQlSoNxN+A8aRJQ8TY4+Nr09BhvSh1ujhKefv9dHLbd3QtVVo6W6s\/3Xt5K\/27ZSkVzpGRateuTdOnT6cXXnhBeGd69uxJZ86coWbNmtF7772XzEbffJM0eDySEevVS7p+NV3ZNvSel3n2xIkTFBcXR271r82T+5e3f6ZMmahy5cqG7wRW8AYCTGC8YQdXRvHII4\/QSy8Zb+B6L8BHH32Utm7dSnfccQc1b96ccuTIQfv37xeBmD\/99JM48nj++eepSJEi1L9\/f7H5vvPOO0I3dANZvnw5zZgxQ7zs+\/btmwyL4P7nzJlDn376KT3zzDPCo3HhwgXasGED7dy5k26\/\/XaqVq1asl\/swc+fO3eOHnzwQdFHoUKFqEuXLobYh87\/6aefJoz5lVdeofz58yd5\/vTp09SnTx\/677\/\/qEePHpQzZ06xgVWvXp3w627+\/PlUuHBh8Qw2wSlTptDPP\/8sNuoKFW4Qsl9++UV4H0qWLElffXWdwGi\/DqtWrUp79uyht956i+688zqBOXbsGPXr149KlSpFrVu3TjImeDLwD\/jCPpB58+YJspA7d2669dak19d37doldGCr8uXLi\/Gjry+++IISEhJE+9oG3mf8t3RiwwJKl+s+ylayDrU\/NZGW7vmXluy7QHdUeYZSZ8hCe5ZOpQtFGtG9d6SlSycO0s5b76PZbzagPWvmiXZhtyxZstC\/\/\/5L6dOnF31jvcC2ZcuWpWnTpjGBCUFAhkAxgbmOgBkCxwTG8HXoKQUmMJ4yh7ODicX4ly9fpuHDh4uNEBtr9uzZxQ2Sxx9\/XBCXNGnS0Lp16yhdunTUsGFDQTLeeOMNat\/+ulciWEaMGEEffPCBeD4xMTEsCGfPnqWaNWvS2LFjqWjRoqLfDh06UMaMGQltHDp0SBAEbIqRBM9iswSRgL5Zeeihh+jo0aP0448\/CgIQKhpZGTp0qMADokdg8PdQYqC19fXXXwuCVLduXTE3SCiB+eGHHwg2hMyaNYteffVV6ty5s\/jfYDly5EgyAvPhhx8K+61YsYJy5cplCEHoOLO9soTKF8xKvy5bQBnXjqXLucrQ0AJ\/UukL6+mlrTlp27+30rOjFhNRClo+dShlz3cv3R9fl4LjVkB0QZIGDx5MjRo1IowT68gvwscR6lmSbaaWzWLZw7w80xTXtChBL4\/S5bFZZXx4G1KmvH6N+e2336bPPvtMbNzYwCEgLdhsO3XqRK+99lqyWeOIYNy4cQSvzkcffRQWFbQNzw68ABcvXhT62PQWLlwoPEAQtI8NMZK0atWKVq5cKca8ceNGQbJkBcQNxAfeH3hG4CEJlbZt29KyZcvo9ddfp44dr8f82E1gBgwYQGPGjJEmMBgbxh9uDtqctHpA00e\/Tyc2fk\/\/FmtOlwpWD0w51b41gsDUqVOHembfLOoNtdpbmg4eOS6wBbnct2+f8KyNGjVKEFtNZs6cKeyl4TR69OgAXrL28LIeb4Zeto7+2NhmatnMqj3Ma7NmAiNhEauNf\/jwYcKx1Pnz5wWp0I5XcEzSu3fvJN6E4OGB2MAToh1P6A19y5YtBGLw\/fffE9y8n3zyCb3\/\/vvC29OtW7fAIyBDb755I\/GdXltTp04VMRaQ0GMtCdioYsWK4ogDnhF4SEKlXbt2tGjRIvr444+Fx8hJAgNPRiiB0\/PADBkyhEaOHJmEZGnzaDBqvShYmOLSObqW5npyv7TrJlLaPcuTEBgkgJs7d644hqwUd456FzwsdLuerkp\/\/LlbeNO0vC7fffcdAffgPC979+4VnqN8+fKJuAN4neCpCxZ43UAwNYIsYx+v6PBm6BVLyI+DbSaPlRc0rd7DvDAnjIEJjIQlrDQ+vDA4DkDsBmIxWrS4Eax7\/PhxcYSRNWtW8XmwwFGGuA0c6eBIAXEaoQKdpk2bijgRbNCQNm3aiCOc4JgS\/F2GwMCLUqNGDeEZuO222wTZyJw5c0TEQAJwGwbjwzENNuQnnnhC3JYJFZy5b968OckRk90eGMQQgcghJgm4pE59IzGfHoHR9BF\/Av2cPVYlncb5U5T2z3l0odhThLT67\/XtTVtWfJOMZAYTmPfrF6GcCUvEERaICTDGcR\/aB2YgI0aixd9AD3bHnGDT4PkYteGVz3kz9Iol5MfBNpPHyguaVu5hXpiPNgYmMBLWsMr4ly5dEoGrX375pdi8EBwaKgiWReBh6JGFFsAafOQU+iw2W\/yaDw7mBCFAHErwL3a8fBYsWCC8MkYCjw5IETZVtAVvDjZzPUEsDz5HnA5iXrZv3y4CA3GMhM0ZpEETBDI\/\/PDDhGOq4IDkcAQGcUG4fRPqfdKOV\/RiYCpVqkR\/\/\/23mOs991y\/Ro0bL\/BiYD6hsUbACHFICJD9\/fff6bZXl9K1y+cpy\/c96JZLZ+hyjuJ0tnR7SpE67fVp\/HuMKh2aIuZ8\/\/3Xr4gbxerUii9DoydPF7rAC0QT+MBr9scffwjiCs8VvCkYO4KtIQjKBpGEVw21s\/AsBF48eMlAboPJsJFdvfQ5b4ZesobcWNhmcjh5RcuqPcwr82ECY8ISVhgfm3nXrl3p5MmTIv4Fv7r1BF6YJk2aiF\/S2JyxkWGzfeyxx8TxAAgKbqKECm714Bhm0qRJgds70MEvemyKIECa4JgKMRVG3pRgsgGSgfgc9I1YneLFiwtSAFKGucHDgOBgxJjkzZs30BeOxTDf+Ph4EduB+SAuB+PC9V\/E9GBT1qRMmTIEDPBc8G2j5557jhYvXizGjaM0TSZMmCCOt9A+biNBtJcr4m5OnTolMAEp0AQEDzefILgeD0IDYgMPE+YIua9Yafo1Sw26clthSrNvNaX9dZxIIJciTTqqWK6M8HoAU\/T95JM3alk9XfpOWn4iA7WrXox6fPJ1oE94WHDLDOPAeDSBFw63xTTBbay0adOKG0+4zaTZGkQSnjvMB+ugWLFilC1bNlq\/fr0Yv1E8k4nl7rgqb4aOQx5zh2yzmCF0tAEr9jBHByzZGXtgJICK1vjYqBHPsHTpUrHh1apVS2x2oddxQ4eATQpxF6tXrxZxD3hZwFuBjTsc6QApwcYXGteCjQ+\/8uG5wOY5ceJEqlKlShJyIAGBUAGBwJEHfv3juAXzwG0YEAUECmvegtD24FlA4CliORDvA48DnsHxllaJG94GBCbDywIBQQLJwbzxdwTeXrlyRQQhw0uFYzLMBTE+iLNBm4ipgUcHx1IgI1pbsB\/iT+rXrx8YGm4VoU14WnDcBRIw4lBRyrh6GF3MXZau5C1PVzNkp3cfK0QdK+cVx3aI1cFcMGZ4XDCOuP5lRJuXrhJN\/ycrfX4gM12mlBSX8jJ17tFXXGOHZ2zgwIHCEwZiCs8XxorAXQgwhVfu119\/FUd1JUqUEDFSjRs3TgIlvFZoB1e8QVjvvvtucVUbXrngulmy9vSKHm+GXrGE\/DjYZvJYeUEz2j1WrG6JAAAgAElEQVTMC2OPNAYmMBIWitb48E5g0wEJ0TZqie6iUlm1apUgBXo3hf7880\/huQHhwGaHjc\/PYvblqgXjapgglmXFjpNJrjKHw2tX4xsFNzWd9CXrUI4355mGGMG5IEqatyhcXg7c3PJTEjGz9jINLD9gOQJsM8shtbXBaPcwWwdlQeNMYCRA9KvxJaaupIrsyxVXnwcs+CswR9wWcltwpAQvDTxV8NiFCrx6iKOC98gvImsvv8zXD\/Ngm6llRb\/uYUxgJNahX40vMXUlVWRerkgwFyx65OVE4vVsx+keqEIH+lShgonXCyvaJTiCg8cOt7+MJPgWkpGu1z+XsZfX53CzjY9tppbF\/bqHMYGRWId+Nb7E1JVUifRyhdelYqGsVH\/k9Vs8qOAcnPVW70hIA8FuAqMlKsQ1dATqhl6JRpwLAroRT8UERsml6ZtBM4FRy5R+3cOYwEisQ78aX2LqSqqEvlxDY1y0SSHWJf7urEnmGEpg4pokiM\/jGvexHQsER+P6PHIBgazoxcDAO4Mr+Chv4BfhzVA9S7LN1LKZX\/cwJjAS69CvxpeYupIqwS\/Xa9eIbnv1xnFR6YsbaMzhlynxub+kgnSdBADlH3DlHTe8IOGCeHF9G3Wm\/CK8GapnSbaZWjbz6x7GBEZiHfrV+BJTV1IFL9fSI3YnGXvFQnE0p1MJOphQVdQhgmfFCa9KNAAip06oBwZVxP16e4w3w2hWibvPsM3cxd9s737dw5jASKwEvxpfYupKqGiFFFfuPClqE4VKcICuRmCyNnyDsjV\/31Pzw7V7VNZGvh3kmAnOlIy\/ob4VsgfrJTL01ERMDoY3Q5OAeUCdbeYBI5gYgl\/3MFcJDIIRkcQL+S+QcA3JufByRj4TJCQLzp5qwlYiOywSg82bN4+2bdsmAh+R5O2uu+4SidGQQA0ZT2XFr8aXnb+X9UJvE2ljbf9wHurf8Hr5gGDRYlxy9V1KaYtU9tTUkLAPlcmRMwhZm0NLTSBLMD5Hgj4tCZ6nJhDlYHgzjBI4Fx9jm7kIfhRd+3UPc43AIBgRvzCXLLken5AmTRqRGv3YsWOBa6TIPIvU8WYE8QMgKFu3bhWPFShQQJAVpIqHGx7kBrVukOlWNhmYX41vBlcv6eqRFtwmguBGkd7LNTQ41+4bRdHgVadOHUHasV5RN0ovBga1oqBn9nsRzXiceoY3Q6eQtq4ftpl1WDrRkl\/3MNcIjHZlFNlhUSsHGWJxbfTcuXOE+jb4pYk4APwqRZp6WUGKdqSRR0p2tPPggw8GHj18+LBwzSMIEvV3kOJfr6pzaF9+Nb4spl7QC3eTCGMLzeFiRGDyjfmbUmbL7YVpJRkD6jKh\/lEkQVkIlJrAGveL8GaoniXZZmrZzK97mCsEBl4SHOXAG4LMo3pVdLWqvkjDj\/N\/mVovOIJCHRlIaNVibbnBC4M6M5AhQ4aI6sNG4lfjG83bK58jtkXL26KNSQvKxX9rcS34\/wjOPVmmtfC8qSYg36jtFE7gtUQRRxD9jRs3qja9sOPlzVA9U7LN1LKZX\/cwVwgMChz27NlTeEFwbRTHR6GCLwjc5RDEyaC4n5GgqJ8WN4MKv3pHRCgIiCrKIERIyY7jJiPxq\/GN5u2Fz5t8spF+2HJMDGVb33jKnunGWkGm3BPTr+dpgSBjbs6EJbpHSF6Yi9EY4IlEBe5wlcrx+WeffSauUCPGyy\/Cm6F6lmSbqWUzv+5hrhAY3LRA4C5Sp48bNy7sSsDL\/NChQ9SjRw9RvddIEBSMAGB4YsI9s337dqpdu7ZoCpuATD4NvxrfCE+3Pw+OdQk9JgolL8HXolV9uSJOCxWzmzZtKiqI58qVSxypIhAdtY8QxAsZPHiw+Nwvoqq9\/IJ\/NPNgm0WDmnvP+HUPc4XA4CW9adMmevbZZwlHReEEMQFr1qyhJ598kgYOHChlfcS9vPPOO8K7M3r0aPGLVhOQoZdeekl4fcz8ivWr8aUAdUkpEnnBkLSgXL18Liq\/XHHE2bFjR9qxY4c4KgquiwSC\/swzz1CvXr1csoo93apsL3sQ8X6rbDPv2yh4hH7dw1whMJpnBTkt2rdvH3YlgGwgqVelSpWE61xWEEcwatQoEeyYO3du8Q9eGWwOEBAoxMjIXkX1q\/Fl8XRaL5i8THy6KNUvlj3ZEC5sXnY9IZ1Oin\/VX64zZswQAeb\/\/PMPHTx4UASkFypUiFq2bCk8jH4T1e3lN3vIzIdtJoOSd3T8uoe5QmAQz3LmzBnDGBQQHOS8KFu2LE2bNk16Nfz+++80ceJEUVcmNPgXfTdr1owaN24s8m3IiF+NLzN3p3T0AnX1KkTLjEf1l+s333wjphmulIAMBirpqG4vlbC2aqxsM6uQdKYdv+5hrhKYcDeFNJN269aN8GvUDIFBbE337t1Fxd6nn35aHD\/lzZuXjh8\/Lo6jcD0bwb4IEP74449F1V8jgfGDBYG\/rVu3NnqMP5dEIDTt\/9hGOah07rSSTydX+\/vvvylPnjxRP+\/2gytWrBBDiDaRo9vjN9u\/6vYyO18\/6LPNvG3FSZMmiR\/xweKnCvbavFwhMPHx8cI1bnSE9OKLL9K3335LVatWpfHjxxuumP379wtiguvZiK1BjE2oIBcMrlHDA4RYmebNmxu261f2ajhxBxS0\/C64Fo3K0EhEFyw4KjrQp4r4k2z2XNV\/HbIHxoGFx13EhIDq37GYJq\/gw37dw1whMMi9smHDBmrbtm3EgETcxkDAbZMmTah\/\/\/6Gy2bs2LFCL1WqVCJIGEGQeqLdgpIN5PWr8Q0BtVEB9YsqFsoayO+iHRfhdlGw4Go0CIyZ4ouqv1yZwNi48LhpSxBQ\/TtmCQgKNeLXPcwVAoM06DNnzhTJ7ELdXMFrokyZMuLoBzljcPvCSJDXZfLkyeL4YPny5WHVBwwYIK6lImX7ypUrjZolvxrfcOI2KYSWAghOShea8l8bgpbjRWZIqr9cmcDIWJl13ERA9e+Ym9i50bdf9zBXCAwS07366quUIUMGkdY\/Xbp0yWz6xx9\/UIMGDcTf58+fT4ULFza0O\/Jj4PYREuOhOGS47L1acDCytcqkZPer8Q0BtVABHpeC2dNRhymbRavhjoxCPTDaEPRuG4UbnuovVyYwFi48bsoWBFT\/jtkCiocb9ese5gqBQfwJrlIjSVe4WBVU4kU16SJFipD2QjdaHwsXLhQFIiGRMvFWqVJFBPLiKAvlBIzEr8Y3mrdVn0fyuFjVR3A7qr9cmcDYsSq4TSsRUP07ZiUWKrTl1z3MFQIDgyPJ3KBBg4S3pHfv3uJaM2JWQG6GDx8ugnZxk+jzzz9PkowOzyKGBV8g5MTAs5qgTECtWrXEZ8hiCo9M+fLlA58fPXqU+vbtKwKD4Z2ZPXs2FS1a1HD9+dX4hhO3QCGYvMztXFIE6gYLgnTTFqlsQU83mlD95RpMYJDTCBmjZTyFloLoYGOq28tBqDzTFdvMM6aQGohf9zDXCAzICY6RcO0ZgqrUqAwNkgEiAglXqwjVqbdu3aobQ4MvFq4440YSBEQGieyQ1A6fIbMpSBOqYcumY\/er8aVWfgxKzcZtpIWbr9cx0svposW7FEy8FkMvyR9V\/eUaTGC0eC0\/XoHULKe6vSxdvIo0xjZTxFD\/P0y\/7mGuERjN\/PCGIEkdbg2dPXtWkBjcDsIV6HAFHCMRGLSLoym0iSMl1JFBuyBICO4tV66cyA9jplqxX41v11cw9Mho8ctlqETeTEm6Cw7WvRkJDPIbych3331HS5YsSVZKA\/mN\/CK8GapnSbaZWjbz6x7mOoFRYRn41fhWYq\/lcwluM\/TIKNwNo5uRwGjZqKO1gZ88MrwZRrsK3HuObeYe9tH07Nc9jAmMxGrwq\/Elpi6tYhTrcjChqqhdFCpWkxe0r8LLFUeciOXC\/0bypqxatYp+\/fVX6tKlSxLounbtKm0bryuqYC+vY+j0+NhmTiMeW39+3cOYwEisC78aX2LqUiq4Ij1gwV9CN9r6RVIdSSqp8nK9evWquPaPpI6IycqePXnRSo6BkTQ6qzmKgCrfMUdB8XBnft3DmMBILDq\/Gl9i6lIqmvfFC+RFFQ9MMLBr164VZS1QOqNatWpJMGcCI7UEWclhBJjAOAx4jN35dQ9jAiOxMPxqfImpR1QJrSAdTGCQkA7Zc62+Ii0zZhVfrqdPnxa37jJnzkxvvvmmCDqHMIGRsTjrOI2Ait8xpzHyUn9+3cOYwEisMr8aX2LqSVRCCUvwhxp5CQ3UtSPGxWjcKr9ckaV6xIgR1KxZM2rfvj0TGCNj8+euIKDyd8wVwFzu1K97GBMYiYXlV+NLTP26FyAoxiX0mWCvS3CgbsZyjSh1vmJkpgSA7HiM9FR\/uU6YMIG++OILatmyJR06dEjU7fLTraNQ+6luL6P16MfP2WZqWdWvexgTGIl16FfjS0ydQm8XrdhxkrrXyp\/00f+u0K5mNyp\/Z2v6NmV9spdM87boqP5yRSK7\/\/77j\/78809RnBTZqZnA2LJUuNEoEVD9OxbltJV9zK97GBMYiSXpV+MbTT04twuKL07MPjusRwUlAa78s4syVm1r1Kztn6v+cg3OxIu5nDx5UpTN8Kuobi+\/2iXSvNhmalndr3sYExiJdehX40eaenC8C46J7Er7LwG\/aRXVX65czNG0yfkBhxFQ\/TvmMFyud+fXPYwJjMTS8qvxw009+NhozJFXqPSF9QFVN4JyJUyUREX1l2s4ArN9+3Z69913qU6dOiLI1y+iur38Ygcz82CbmUHLfV2\/7mFMYCTWll+Nrzd1jbyUvriBxhx+OYmKCuQFA1b95RqOwCDZ3bhx40TCuzVr1kisXDVUVLeXGihbO0q2mbV42t2aX\/cwJjASK8evxg+derDn5dd9NxKqqUJctPmo\/nKN5IEBialduzZ7YCS+t6xiHwKqf8fsQ8abLft1D2MCI7He\/Gp8TD20cjSCded0KkEIykXtIjeuQUuYJKKK6i9XjoGJdQXw83YjoPp3zG58vNa+X\/cwJjASK82vxgd5gaelTN7FARS8Ug5AwixhVVR\/uTKBicX6\/KwTCKj+HXMCIy\/14dc9jAmMxCrzm\/G1G0ZanEtckwQlPS3hTKf6y1WPwGBOiYmJVL9+fbr\/\/vslVq06KqrbSx2krRsp28w6LJ1oyW97mIYZExiJ1eMX44ceFzGBkTC+zSo\/\/fQTrV69mo4ePUpYZ48++ijlzJlTt9d169bRE088QSVKlKCZM2faPDLnmufN0DmsreqJbWYVks6045c9LBQtJjAS68cvxg\/NqgsCc6BPFVF0MWfCEgkk1FBR5eXao0cPmjZtWhJQU6RIQW3atKFXX32VMmTIkAzw+Ph4OnjwoK8y86piLzVWvzOjZJs5g7NVvfhlD2MCE8WK8IPxtay63WsVCJQC0JLTMYGJYlHE+AiICwgMpFixYlSxYkVBWPbu3Uvff\/89Zc2alT777DPKkydPkp7godm6dSsTmBjx58djQ4AJTGz4Of20H\/YwPczYAyOxklQ3fnBJAL2q0apdkzYymQovVxwF4UiodevW1KdPH4LnRZPLly\/T1KlTBYFBZeqiRYsGPmMCY2R9\/twJBFT4jjmBgyp9qL6HhcOZCYzEClTV+HuOX6CS76wOzFDP++I38oLJqvByhdfl4sWLtH79ekqXLp3uKty3bx916dKFevbsSaVLlxY6TGAkvrCsYjsCKnzHbAdBoQ5U3cOMIGYCY4QQkQiuVLEacHDMix+uR0uYSqio8HLFkVH69Olp4cKFyaYVfAvp1KlT9NJLL1H79u2pQoUKTGBkFwHr2YqACt8xWwFQrHFV9zAjmJnAGCGkKIHZeeQ8lX3\/JzG7m4m8qEJgQEg2b95MK1asiEhg8OH58+cFiXnqqado0KBBHAMj8Z1lFXsRYAJjL75Wt84ExmpEFWpPReNr3hetJECuvkspbZHKCqEe\/VBVeLlu2bKFHn\/8cRo7dixVrpzULnp5YC5dukSvvPIKLV++nM6ePaukRzCcRVWwV\/Sr0Z9Pss3UsquKe5gMwuyBkUBJNeNrQbtfnulGhU7+KmZ45yvTKcP\/GkvMVn0VVV6uP\/zwAyUkJNALL7xATz75JKVKlUqAHy4T75UrV8T16rlz5zKBUX+ZKj0DVb5jSoNs4eBV28Nkp84ERgIp1Ywf6n3xY6BuJLOp9HI9ceIEffvtt3TmzBnq2LFjRAKDD69evUr9+vUTN5f8IirZyy+YxzoPtlmsCDr7vGp7mCw6TGAkkFLJ+Jr35aW8e6jVqrbktzIBEuZSIog30jy4FpKMlVnHTQSYwLiJvvm+VdrDzMyOCYwEWioZX\/O+vN2gED068i4mMBL2dUNlw4YNIoFdcP4XbRxMYNywCPdpBgEmMGbQcl9XpT3MDFpMYCTQUsX4gxbuoffn76KKheJoYvbZdGJ6AuXs9T2lK1ZDYpb+UVHh5Yo1lT17dqpTpw7Vrl2bypYtSylTpjQ8QvKPlW7MRAV7+RH3WObENosFPeefVWUPM4sMExgJxFQxvnZ8hGvTWpmAmy3+BeZU4eWKNZU5c2bx7++\/\/6Zs2bJRzZo1BaE5duyYCOitV6+exOpUX0UFe6mPsrUzYJtZi6fdramyh5nFgQmMBGIqGH\/Agt00YMFfYjY3W96XUBOq8HLFmipevDjNmjWLfv\/9d5o3b564fbR\/\/35KmzYtPfjgg9SuXTt6+OGH6dZbb5VYpeqqqGAvddG1Z+RsM3twtatVFfawaObOBEYCNRWMr1fvSGJqvlRR4eUaTGCCjQAyg1tJ+AcygwKPVatWFZ6ZKlWqhC07oLIhVbCXyvjaMXa2mR2o2temCntYNLN3lcBcu3ZN\/AL9+uuvCYm9Tp8+TVmyZKGSJUtSq1atRIXeaOW\/\/\/6jL774QrS\/Y8cOQoG8nDlzil+0zzzzDN11113STatgfC1492b3vqh0hKR5YMItRAT6fvfddwEyA0+MRmbq168vvX69rsibodctlHx8bDO1bKbCHhYNoq4RGBCK559\/npYsWSLGnSZNGhEHgPN\/fAbp1KkTvfbaa6bnhUylbdu2pbVr14pnM2bMKNzwR48eFbc+8N+ffvppoECeUQdeN752fITg3TmdShhNx\/efq\/ByDeeBiURm4JXBUdOBAwc4kZ3vV7G3J6jCd8zbCDo7Oq\/vYdGi4RqBee+992jcuHGCWLz99tsirXrq1Knp3LlzNGHCBBo6dKggGx999JEoYGdGEDuwaNEiypUrF\/Xv3z\/gyUF1XxCiX375RXwG8oQ+jcTrxg\/2viB492bM\/RJsQxVermYJTPD84JmB98YvooK9\/IK1VfNgm1mFpDPteH0PixYFVwjMkSNHxFEO6rsgq2iLFi2Sjf+tt94SR0D58uWjxYsX6+bL0Js0iuO1bt1aeHSQcv2ee+5JogYvzLPPPkt58+YVadkLFChgiJ3XjQ8CE3x1mgnMX1J2NTS8jQpYU1iDy5Yts7EXNZrmzVANO6n2I0E9VO0bsdf3sGhn7gqBmTp1KvXs2ZMyZcokvCEgG6GCl1r16tXFnxHHIvuLs3PnziJuAJV733333WhxSfKcl40fWjYAA2cC430CgxiWTZs20bBhw8hP8SzRfOGYwESDmrvPsM3cxd9s717ew8zOJVjfFQKDqroI3K1WrZo4RgonFSpUoEOHDlGPHj3ElVIjQeAurp9euHCBJk6cKLw8VohXja9HXm6mqtPhbKvKyxVHQfAylihRgpo1axaYDmfiteJby23YiYAq3zE7MVCpba\/uYbFi6AqB0X594igHR0XhBF6UNWvWiEq9AwcONJzrrl27qEaN61ln8RySgcF7g2BeFM277bbbqHz58tSoUSNT11G9aHyNvLQ\/PZHan5pI6R6oQjkTrgdE3+yi+suVCczNvoK9P3\/Vv2PeR9jaEXpxD7Nihq4QGM2z8sYbb1D79u3DzuOll14ScSyVKlWizz77zHC+iJV57rnnREr26dOni7ZxqylUEMA7fvx4uvfeew3bhIIXjR8cuHsisa+YR1xj\/1QoljJMGCXVX65MYGKxPj\/rBAKqf8ecwMhLfXhxD7MCH1cIDOJZzpw5Q3369KE2bdqEnQcIDogI6sRMmzbNcL5z5syhrl27ipgapGgvVKgQvfzyy4KoIOfM8uXLxY0nkBrkhJk\/f76IwzESrxlfuzb9VYcSVPXeOKPh33Sfq\/5yZQJz0y1Z5Sas+ndMOcBjHLDX9rAYpxN43FUCk5CQIG4MhZNu3brRjBkzpAkMdPEMBHEFiYmJgQJ5Wh\/r16+nhg0biltNr7\/+OnXs2NEQSxg\/WEC6Io3bsMEYFUqP2C1aWPti\/hhb8ufjqC2UJ08eZSeHm3SQWBI5qjR51e2lEtZWjZVtZhWS9rQzadIkEQcaLAix8Ju4QmDi4+Pp4MGDZHSE9OKLL4ospMg+iiMfI4EunoFEut0B8oFNQtaz4yX2yjWPjFaBGsUcI82CPTDGNmYNdxFgD4y7+Jvt3Ut7mNmxR9J3hcDAA4IbGMiW26tXr7Dja9q0qbhm3aRJE5GQzki0HDDQmz17triRpCc4RkJMDWJhtF+7kdr2kvG5ZIDRKmACY4yQtzR4M\/SWPWRGwzaTQck7Ol7aw6xExRUCg2y4M2fOFNecQ91cwZMrU6YMHT9+XOSMQf0iI0GCvHLlygm1KVOmEIKF9UTLAiybSMwLxl+4+Rg1G7dRTIdLBkReCaq\/XNkDY\/RN58\/dRkD175jb+Dndvxf2MDvm7AqBwdVmZMFFpd2ff\/5Z90rzH3\/8QQ0aNBBzRrBt4cKFpeaP5Hf4cuEoCQG8eoLr2ygjAIIDomMkXjC+5nnBWLWCjbh9xDePkltP9ZcrExijbyR\/7jYCqn\/H3MbP6f69sIfZMWdXCAxuIIE8oO4R8sCAUITKCy+8IArXFSlShLQXugwAI0aMoA8++EAUhly4cKH432DZuXMn1a5dm5D0zigGR3vObeOHi3tB3SNIwcRrMtDcNDqqv1yZwNw0S1XZiar+HVMW+CgH7vYeFuWwDR\/TJTCoz3Lx4kW65ZZbAonhDFsyqTB69GgaNGiQuPLcu3dvaty4sSisCHIzfPhwEbSLq8+ff\/55sqMgZPLFF6hkyZLi2WABKapZs6YIEr7\/\/vsFmdG8N5s3byY8u337dpHUDgUfcd3aSNw2foNR62nFjhMBzwvGe4O8XCWi60SG5ToCqr9cmcDwSvY6Aqp\/x7yOr9Xjc3sPs3o+Wnu6BAaTbdWqFSGIFh4QxJYECypIy2z8kQYNcoJjJJQUgKDNuLg4QrHFK1euiL+FyxOD6tRbt24NG0Ozbds2MX60BcmRI4doU\/vvLFmyiIrXIEAy4rbxQwN3DyZUpfObloqhs\/cluQX55Sqzqr2jw\/byji1kR8I2k0XKG3pu72F2oaBLYHDNGbdzkCsFgjgR1GzZsmUL1a1bV1SPRkp+KwRXn5GkDoXtzp49K0jMQw89JI6VwhVwNCIwGNepU6dozJgx9MMPPxByFly9elXkBkH9JWTrveOOO6SH76bx1+49TY98uFaMVYt94aOjyKbjl6v00vaEItvLE2YwNQi2mSm4XFd2cw+zc\/K6BAael9DMt\/v27RMemZUrVwaIjZ0D81Lbbho\/1PuikZebveJ0pPXBL1cvfXuMx8L2MsbIaxpsM69ZJPJ43NzD7ERKl8Dg+GXy5MlJ+oUH44knnhDFEW82ccv42\/45R\/8b8HMS7wv+48LmZZS2SOWbzQzS8+WXqzRUnlBke3nCDKYGwTYzBZfrym7tYXZPXJrAYCBIQKdHYJCeH6n7\/SpuGJ8z7ka\/mvjlGj12bjzJ9nID9dj6ZJvFhp\/TT7uxhzkxR10Cg8y3KKIYKuEIDGoa4Z9fxQ3j6+V98Su+Vs9L9Zcr30KyekVwe1YjoPp3zGo8vN6eG3uYE5iEvYWECeOKc7DoLVpce96\/fz\/5sVCUNnc3jM8lA6Jf\/qq\/XJnARG97ftIZBFT\/jjmDknd6cWMPc2L2YQmM2c6ZwJhFLLI+CEz3WgWoey2uOG0WWdVfrkxgzFqc9Z1GQPXvmNN4ud3fTUVgcISEIyHketGuUusZALlcTp48Sf369Ut2a8ltg1nZvxvGB4EZ1fx+alYmB2l5X3L1XcrBuxKGVf3lygRGwsis4ioCqn\/HXAXPhc7d2MOcmKauBwYp\/JFrRVaQywX5YfwqThs\/3NVpJjByK0z1lysTGDk7s5Z7CKj+HXMPOXd6dnoPc2qW0rWQUBV6z549ItFc7ty5Rdr\/m0WcNn4wgUHBxhPTrwdIc9ZduRWn+suVCYycnVnLPQRU\/465h5w7PTu9hzk1y4gE5vz58zRw4ECCh0VLw4+B4WipXr16ohQACI3fxUnja9entfgXTlxnfnWp\/nJlAmPe5vyEswio\/h1zFi33e3NyD3NytmEJzO7du0U6fyxUSKpUqYTn5cCBA3T58mXxt+zZs9OMGTMob968To7Z8b6cNH644yP2vsibXfWXKxMYeVuzpjsIqP4dcwc193p1cg9zcpZhCQzqBS1evJiaNWsmCiMWKlRIHBv9999\/tHfvXlqzZg0NHTqUbr\/9dkLMjJ\/FKeOv+esU1RnxG1UsFEdzOl1PDMh1j8yvLNVfrkxgzNucn3AWAdW\/Y86i5X5vTu1hTs9Ul8D89NNPgrQMHz6c6tSpE3ZMyP+C8gKDBw+mihUrOj12x\/pzyvh6uV8QAwOJa9zHsfmq3pHqL1cmMKqvQP+PX\/XvmP8tlHSGTu1hTuOqS2C6detGmTJlol69ehmO58svv6RVq1YJsuNXccL4w5fso4S5OwSEWtVpv+Jp97z45Wo3wta2z\/ayFk8nWmObOYGydX04sYdZN1r5lsLWQho5cqQI1jWSixcvUtu2bWnq1KlGqsp+7oTxOfOudcuDX67WYelES2wvJ1C2tg+2mbV42t2aE3uY3VXXphwAACAASURBVHPQa1+XwDRt2tRUYrp27drRJ5984sb4HenTbuNrN4+CY18cmZhPO+GXq1qGZXupZS+Mlm2mls3s3sPcQkOXwDRv3tyURwW3lcaPH+\/WHGzv127js\/fFWhPyy9VaPO1uje1lN8LWt882sx5TO1u0ew+zc+yR2tYlMAjIxRES4mCM5Ny5c9S5c2davny5kaqyn9ttfBAY9r5Ytzz45Wodlk60xPZyAmVr+2CbWYun3a3ZvYfZPf5w7XMxRwnk7TS+5n0JLdyI69NxTRL49pGEfUJV+OUaBWguPsL2chH8KLtmm0UJnEuP2bmHuTQl0W1YAlOgQAGpIN7Tp0+L81CuRh2dGcNdnUb5gHQPVKGcCUuia\/gmfkr1lytfo76JF68iU1f9O6YIzJYN86YiMKguLXOFWkPXrL5lVnGoIbuMH1o2QJuOVv+IPTDRGVj1lysTmOjszk85h4Dq3zHnkPJGT3btYW7PTtcDs2XLFrr\/\/vulx2ZWX7phjyjaZfxIwbt8hBS98VV\/uTKBid72\/KQzCKj+HXMGJe\/0Ytce5vYMdQkM6h0hvwuKOSJIF2UE9Io2\/vLLL1SqVClKmTKl2\/OwtX87jN9g1HpaseMEhca+YCKaB4brH0VnVtVfrkxgorM7P+UcAqp\/x5xDyhs92bGHeWFmugQG16KXLFlCOXPmpA4dOlDDhg11byStW7eOEhMT6b333vPCXGwbgx3GN\/K+YDK5+i6ltEUq2zYvvzas+suVCYxfV6Z\/5qX6d8w\/lpCbiR17mFzP9mrpEpjRo0fT559\/TnPnztX1vAQPacSIEVSsWDGqXNm\/G60dxg93dVor3giM2QMT3eJX\/eXKBCY6u\/NTziGg+nfMOaS80ZMde5gXZqZLYFAaoFOnTlS2bFnDMe7evZuGDBlCIDJ+FauNv3LnSao\/cp3u8ZF2hJShTH1KU6CUXyG1dV6qv1yZwNi6PLhxCxBQ\/TtmAQRKNWH1HuaVyesSmEaNGtHMmTOlx\/joo4\/SvHnzpPVVU7Ta+D\/uOEmPjVrHRRttWgiqv1yZwNi0MLhZyxBQ\/TtmGRCKNGT1HuaVaesSmMcee4xmz54tPcb4+HhauXKltL5qilYbX7s+zVWn7VkJqr9cmcDYsy64VesQUP07Zh0SarRk9R7mlVnrEpiaNWsKD0zGjBkNx3nw4EFq0aIFLV682FBXVQWrja\/dQGICY8+KUP3lygTGnnXBrVqHgOrfMeuQUKMlq\/cwr8xal8C89NJLdN9991HHjh0Nx9m7d286fPgwffzxx4a6qipYbXwu3mjvSlD95coExt71wa3HjoDq37HYEVCrBav3MK\/MXpfAfPvtt\/Tiiy\/SgAED6Mknn6QUKVIkG++VK1do0qRJhCy8w4cPp\/r163tlTpaPw2rjM4Gx3ERJGuSXq734Wt0628tqRO1vj21mP8ZW9mD1Hmbl2GJpS5fAoMF27drRokWLqHDhwlS3bl3Knz8\/Zc2alZDk7u+\/\/xZBu1jEqFwNIuNnsdr4IDB6Cey0K9R8fTq21cQv19jwc\/pptpfTiMfeH9ssdgydbMHqPczJsUfqKyyBQRbed999l6ZOnRr2+Ro1agid7NmzRzWfa9eu0axZs+jrr78mlCNAYcgsWbJQyZIlqVWrVoIcWSW47o3bUhcuXKCHH36YJk6cKN201cYHgXnnsXuoU+U8gTEcTKhK5zct5QrU0lYJr8gvVwtAdLAJtpeDYFvUFdvMIiAdasbqPcyhYRt2E5bAaE9u2LCBvvjiC9q+fbvwuCA7L+okgbzUqVPHsINwCpcvX6bnn39eZPyFpEmThrJly0bHjh0jfAZBLprXXnst6j60B69evUpNmzaltWvXij+5SWC0AN65nUtS\/N1ZA3PTvC9cgTpmc4t1imrqLGogwPZSw07Bo2SbqWWzm5bA2GUmlB8YN24c3XrrrfT222\/T448\/TqlTpxa1lyZMmEBDhw4VsTcfffSR8JzEIsgsPGjQINEXajy5SWDCxb\/w8VEsFk76LL9crcPSiZbYXk6gbG0fbDNr8bS7NSYwFiJ85MgRQSIuXbokgoBxDTtU3nrrLeH5yZcvn7iirRdILDMkHE2BHIG8oKbT5MmTPUdgNPLCtY9kLGqso\/rLlW8hGduYNdxFQPXvmLvoOd87ExgLMUdcTc+ePUWBSFS0xvFRqOALUr16dfFnxMkUL17c9AhAkEBetm7dSgMHDhTBx7gx5ZYHRktgFxrAy94X06aN+IDqL1cmMNauB27NegRU\/45Zj4i3W2QCY6F9XnnlFRG4W61aNXGMFE4qVKhAhw4doh49eohbUWYF18DHjBkjiNAnn3xCH374oasEJtzxkRbAy7ePzFpYX1\/1lysTGGvWAbdiHwKqf8fsQ8abLTOBsdAuyBmzadMmevbZZwlHReHkqaeeojVr1ohcNPCgmBEE7DZp0kRc\/Z4\/f764KeUFAlOxUBzN6VQi2VROJPaluMZ9zEyRdcMgoPrLlQkML22vI6D6d8zr+Fo9PiYwFiKqeVbeeOMNat++fdiWkRF47ty5VKlSJfrss8+kR4BA4Hr16tGePXtElWzksYF4gcDo5X+RnhgrSiGg+suVCYyUmVnJRQRU\/465CJ0rXTOBsRB2xLOcOXOG+vTpQ23atAnbMgjO9OnTqWzZsjRt2jTpEWgBwPD0DBs2LPCcmwTm2jWi217VT2AnPTFWlEJA9ZcrExgpM7OSiwio\/h1zETpXumYCYyHsGoFJSEig1q1bh225W7duNGPGDFME5scffxSkCEdGCxcuFInxNImFwAQPEu1HGrfehEqP2C3+vPbF\/BYiyU3pIYBg7Tx5biQJVA2lFStWiCFbmcjRyxiobi8vY2vX2NhmdiFrTbvIjh+arHXXrl3WNO6hVgwT2dkx1vj4eEIVa6MjJNRjQl2mqlWr0vjx4w2HcurUKapVqxb9888\/4sipcuXKSZ6JhcDEavxw5QMMJ8UKphFQ\/dche2BMm5wfcBgB1b9jDsPlenfsgbHQBMjHggy\/bdu2pV69eoVtGdlzcc0awbj9+\/c3HEGXLl0IL\/8OHTpQ9+7dk+m7RWCWbj9BjT5eT188V4xqFbktMC4O3DU0aVQKqr9cmcBEZXZ+yEEEVP+OOQiVJ7piAmOhGVAeYObMmYb5WMqUKUPHjx8XOWOeeeaZiCPYv3+\/aA9SqFAhSpUqVTL9o0ePEv5lyJCB8ubNKz5HHE65cuUith2r8Tn7roWLR6Ip1V+uTGAkjMwqriKg+nfMVfBc6DzWPcyFIUt16coREhLTvfrqq4JI\/Pzzz5QuXbpkg\/3jjz+oQYMG4u+4Bo2q2JEkOPGd1Mz\/X2ns2LGirlMkidX4kQgM1z4yYy05XX65yuHkFS22l1csIT8Otpk8Vl7QjHUP88Ic9MbgCoHBDSRcpcZ1Z9wYQj6YUHnhhRdo3rx5VKRIEXEsZIW4dYQEApPYvjhVvy9bYBo4PjoxPYGYwFhh2aRt8MvVekztbJHtZSe69rTNNrMHV7taZQJjMbJagUWUEejduzc1btxYFHMEuUG6fwTtXrt2jT7\/\/HNBdoIFmXzxBSpZsqR4VlbcIDArd56k+iPX0ZxOJalioRvVpzUCE9ckgRPYyRpQUo9frpJAeUSN7eURQ5gYBtvMBFgeUGUCY7ERQE5wjISSAhAUW4yLixMxKleuXBF\/C5cnBtWpUd\/IbE0jNwjMh4v20tvf7qTjQ6smQVCrf8QExuKFRSTIbYECBaxvmFu0BQG2lzlYmzdvTj\/99JO5h1jbdwiUL1+eUFdQRpjAyKAUhQ6uSSNJHUoLnD17VpCYhx56SBwrhSvgqBKB4fpHUSyKGB\/hDTFGAB1+nO1lDnC\/bkbmUGBtM+vAjK5KyLoSA6MSQBhrLMbnG0jOW5s3ROcxj6VHtpc59GJ5H5nribW9jICZdWBG18tzDh0bExgJa0Vr\/DE\/\/k09Zv1J09oVp0fuvxHAiy4vbF4mgnhzJiyRGAGrmEFA9Q2Rr1GbsfbNpxvt++jmQ8rfMzazDszoqoQaExgJa0Vr\/AELdtOABX8li3+R6JJVYkCACUwM4LnwqOr2chqyaN9HTo+T+7MXATPrwIyuvaO2tnUmMBJ4Rmv8BqPW04odJ5jASGBspYrqGyJ7YKxcDf5rK9r3kf+QuLlnZGYdmNFVCVUmMBLWitb44eJfJLpklRgQYAITA3guPKq6vZyGLNr3kdPjDNcfsqtPnz6dBgwYQLfccgvVrVuXUqZMGbh92qJFC8INGz357rvvRH08FOlFtvVjx45RzZo1A0lPL1++LOrgffDBB+JmK34M5M6dO9AUiv1+9NFH9Pvvv4tyM7jRhfQdkQSpPG677TaqXbu2VyAU4zCzDszoemqSBoNhAiNhrWiNDwJTsVAczelUQqIXVrEKAdU3RPbAWLUS\/NlOtO8jr6GBNBiZMmUSCUs1mTt3LiUkJFCpUqVEPrDgLO0oQbNu3TpBULRSMEeOHKEnn3ySihUrJvRTpEghmgI5GjNmjLjJiluuyDemCbK\/o5SNTH09PINbryBMX3zxheMQDhkyRKQb0RMz68CMruOTjKFDJjAS4EVrfCYwEuDaoMIExgZQbWxSdXvZCE3MG5fTYzPTX\/Xq1Sljxow0e\/bsJI8tW7aMnn76aWrdujX17dtXfKaVn0GC06pVk+bUWrVqFbVs2ZLee+89atasmdDHxr97927hrUE7IEWabNmyRfT5xhtvGA4XhAnECWtUpqSNYYMmFEC8+vXrRyirwwRGHzgmMBILKhYC071WAepeK3+gF9w+OtCnCpcQkMA9WhXVN0T2wERr+ZvjuWjfR15DJxyBwTiff\/55WrBgAa1evZpy5MhBlStXJhwPgazoCbw5\/\/33X+BzEJg6deoIr8zChQtp2LBhVL9+ffHo9u3bBSHCEZKRgOSgkPDjjz9ODRs2pHfffTfZI6dPnxbt4xhszZo1YuzoW5MpU6bQrl276LfffhOFhkFKLly4QBMnTiR4nHCkhfH+8ssv1LRpU3rzzTdFEeNu3brR8uXLqUOHDsKTFFqzz8w6MKNrhImXPmcCI2GNaI0PD0wogUF3yMLLGXglgI9ShQlMlMC59Jjq9nIattD3EW47elVQPiX+7hslVILHGYnAaKVmPv30UypdujQ9+OCDFB8fTyADetK2bVuC52bt2rUiGSoIAY5+cNQE8nHo0CFBWu655x5pAnPq1ClRqw8Eo0ePHsJrgwzImTNnTjKETp06UaVKlYT3B+Tjyy+\/pEaNGlH79u3Fkdfhw4fpxRdfFDE+FStWJMwbRAhHZ6j599JLL1Hnzp0FmUGZnBUrVoi4HfbAGK9qJjDGGJkKltKa024gjWlRhBqXvjPQC3tgJACPUUX1DZE9MDEuAJ8\/HkpgtMsCXpx2pBjASAQGwbrY1HH0U7ZsWRHoCyIydOhQ3WlqxAEkBd4KjcDcf\/\/9wvuBZ++8805Rumb\/\/v1SHpgJEybQXXfdJTwfGzZsEB6Ynj17Co+MJsgeD3I1Y8YMQbS04yyQKZCnatWqiVp+t99+u3hkzpw5wvsCPc3jgsBiEBaU0UEWepA0PMMExnhFM4ExxigqAhPpBhJ7YCRAj0GFCUwM4LnwqOr2choylTwwwCb4CF3WA4NYF3gpcDSD46MSJUpErH3Xrl07WrRoUTIPDAgMBMdIOIrBMRKIEYiM0RESYmfgAcLREAQenfPnz9PixYsDwcIgMCgqDGLTpk0bwnESxoqjH9xcKlKkCCFoOHv27MmWSSiBQSFjkC\/0iTkzgTH+ZjGBMcbIUgKjFXFM90AVzsIrgX00KqpviOyBicbqN88z0R5pew2hSB4YxH8kJiYKT0XOnDnFhg6BZ0NPcMUZcSMgC5BgD4ymj7+NHDlSXJ3G7adIBAZHRbjJlCtXrkB3OIbCkVBoIDHaXbp0KX3yySeCRMFbM3DgwACZwTzgnQkVJjCxr0gmMBIYRvPCiBT\/gi4LJl6T6JlVokFAdQITzZxVfobtZc560byPzPXgjHY4AoMgW8SQIL+LdmQEbwTiUPTIANYPjmp69eoVON4ZPHiwyNtStGjRwGSuXr1KiJXBkQ28MZEIDOJS4FUJ9pzAu4IYFlzxxlVuTXAkhbiXfPnyUYECBZKQFRCXWrVqiRtSmsCDA8\/Q33\/\/LYJ2tSMkPQ9M7969adu2bboGMbMOzOg6Y31remECI4GjWeOv3HmS6o9cRwOfKEzPxd9IonQisa+of5Sr71JKW+T6LwoW6xHgDdF6TO1ske1lDl2z7yNzrTunrZcHBgnmQB7uuOMOGjt2bCBgFjeMOnbsSP\/88w+NGzcuQCxOnDghgmXTp08vSIWWBwbBtyAw6CNYEJhbr1498S8cgcF6HDRoEI0aNSoZGJpnCAG4ICHXrl0TsTHvv\/8+5c+fP0neGjwMT8zHH39Mr7\/+uoilwS2lPXv2iOBgeHngDdLiZbQYGMwbulocED7H\/NF+sJhZB2Z0nVsBsffEBEYCQ7PGjxT\/AhIDiWvcR6JnVokGAd4Qo0HNvWfYXuawN\/s+Mte6\/dogHfCkYNNHFlzEpYB4YJM+d+4cNWnSRHhUkKU3WHCLB54YBMwiQBaxKdj0EfD63HPPibagg8y58NxgwwdJwefBgjwwP\/zwg7gZFCr4DFen0Y4WQKzp7Ny5UyS\/wzERrkMjRgfxLiBKWMMQzKNw4cKCbDVo0IAuXbok9EBGQHZAnNA+\/g7vCm42IedN165dRbwP4l9wBRseG8wfnijMC0HK8P4wgUlqMSYwEt9Xsy+McMdHEl2xigUI8IZoAYgONsH2Mge22feRuda9r41rydjYDxw4IIgLNne3BERkxIgRwmMC7xCCeuHlAUHDvwwZMtg2NDPrwIyubQO2oWEmMBKgmjG+dnx0fGjSbJES3bCKRQjwhmgRkA41w\/YyB7SZ95G5ltXRRtwJMuTiSAZkBp4WvZs+ds8IAbx58uQRsSzBAi8QShygHpNdYmYdmNG1a7x2tMsERgJVM8ZHUqkBC\/7iCtQSuNqlovqGyLeQ7FoZ\/mjXzPvIHzMOPwsE\/CLIFQGxSGBXrlw5EUjrlOD4B3ldUK8IV6b\/\/fdfEZSL47DGjRvbOgwz68CMrq2DtrhxJjASgJoxPleglgDUZhUmMDYDbHHzqtvLYjgMmzPzPjJsjBViQgCeIFyfxvVtlDq47777xE0o5IaxW8ysAzO6do\/byvaZwEigacb4TGAkALVZRfUNkT0wNi8QxZs38z5SfKo8\/AgImFkHZnRVAp0JjIS1zBhfrwK1lryO6x9JgG2BChMYC0B0sAnV7eUgVKIrM+8jp8fG\/TmHgJl1YEbXuRnE3hMTGAkMzRhf7waSRmA4eZ0E2BaoqL4hsgfGgkXg4ybMvI98DMNNPzUz68CMrkrAMoGRsJYZ4zOBkQDUZhUmMDYDbHHzqtvLYjgMmzPzPjJsjBWURcDMOjCjqxIgTGAkrGXG+CAwvR69m16ucZdoWcu+K1y\/XD5AAu3YVVTfENkDE\/sa8HMLZt5HXsQBiexQTBFZai9evCiuGyNxHPK6oPbQ888\/L5LBuSWoeTRp0iSRQRc1k1DSIFWqVCJIF4nzVq9eTXPnznV1jGI\/KVhQVNqWETO6Mu15RYcJjIQlZI2v5YD5umMJqnRPXBICw\/EvEkBbpMIExiIgHWpGdXs5BFOgG9n3kdPjMttfq1ataOPGjaL4IQQEAdlxkdsFmWtz5Mhhtskk+rhaDbIR7XXmBx98kMqUKSOy4wYLijk+9NBDhM\/dFDPrwIyum3My2zcTGAnEZI3fYNR6WrHjRJIcMAcTqtL5TUvZ+yKBs1Uqqm+I7IGxaiX4sx3Z95HXZ48surh+DBKjCapPt2zZUlSTRo2haAVeHrTz2GOPiVpJ0QgKMeI6NGovBcuRI0fEf7qROC94HGbWgRndaLBy6xkmMBLIyxpf7wo1B\/BKAGyxChMYiwG1uTnV7WUzPMmal30fOT0us\/3pEZiFCxeKIySQBtRD2rdvn6h\/hGR1IDTvvPMOde7cWRR5RNFHFElE3aFvv\/2W+vbtS2XLlqWRI0cSii0iO2\/x4sUFCenUqZMotDhlyhRx7PLbb7+Jekb9+vVLVoBRm4cegUHiPCTMA3lBsjoUXkS2XZQ32Lp1qyhrgOOvcONu0aIFoQ3Uc8Lx1MqVK6l169b06KOPim5R4BHHU\/AeoUwBikoi06+emFkHZnTN2tFNfSYwEujLGl+PwGgxMBz\/IgG0RSq8IVoEpEPNsL3MAS37PjLXqvPaGoFBtWUcH2Fj79Onj6jyPHz4cFHMEHExICIgB2+\/\/bYgCijkCHKDrLvIhAsCA+nQoYOIVQGBgaCIIz7XPDAgQngex1SIuUFxxOrVq4tii3oSSmBQaBKEB+3BBngOhRZRoRqCuBl4jlAoEvPRGzfabNeunTiWQp0kEKDBgwcTqnCDtKB9VNWGvPDCC2Le33\/\/faDKdvA4zawDM7rOr4Toe2QCI4GdjPG\/33KMmn6ykSoWiqM5nUpItMoqdiHAG6JdyNrTLtvLHK6h7yMcU0eSnAlLwn5s9CwejPQ8Po\/URtoHqlBc4z66\/YPAIO0+PCsoggivSNasWYWHpW7duoFnUJkZpGDx4sVJ2gEBgUdDIzB4DhKOwID0gNTcfvvtQm\/OnDmiDACOrcIRGJAoxLugkjTKBCA+B94etIH4GJAiLesuApKhC48KSgvojRtkBRWr8TkE84bXCfWcUKUaAcSocA0BHitWrBDzRmXtUJHZl7RnzOiaW43uajOBkcBfxvhDf9hD78zbxTWQJPC0W4U3RLsRtrZ9tpc5PEPfR9oxdbhWInl\/jZ5Fm0be40htpHugSlgCFHqEpB3J4NikZ8+eIiU\/BEQAxGH27NlRE5jz58+LWkWIuZGNXdE7QgJhwTEVYmwQHAwyA4+RJiBjIDcoL6A3bpCbWrVqCeIVKngWc65fv77UgpDZl5jASEHpbyWZhcJFHL2zBnhD9I4tZEbC9pJB6YZO6PvowuZlERtIW6Ry2M+NnsWDkZ7H50ZthHteLwYG7eF2T5YsWYT3wSoCc\/r0aeHZSExMJBATGdEjMGgHx0Y47gLhwFFQ5co38MWcEBMDL5AegUGsC46tNA9M8DjgdapTp444OpIRmX2JCYwMkj7XkVkoejeQfA6LZ6fHG6JnTaM7MLaXOXvJvI\/MteiOdjgCg8BbxIdoRzuRPDDNmjULbPiITQG5CD5CwlVt7YgJhATeD7SnCY5n4EHJmTNnMhDC3UKCIo5+8Dn6DCYjuPVUu3Zt0afeuHHMtWXLFhHXkjJlStHn3r17xf8OGDBAxLzguAzHTBDEACGAuWrV5MeEZtaBGV13VkN0vbp6hIRzxVmzZomkRjAq2C2YN84UsfBwxhmtYGF++eWXtH79ejp58qRgxQj+qlSpknDT3XHHHdJNyxifizhKw2m7Im+ItkNsaQdsL3NwyryPzLXojjbe8evWrRMBrNqGjSOaHj16iFtDr732mhgYbhfBG4NNP1hwhINjJwTP7t69m15++WVBfOBlAZEBkXjggQcEMfjnn3\/o888\/F8npcHOpRo0a4lhqz5499NZbb+kCgCMnxLmgfT354IMPxB6DvQb94uYRkvJhnLglpTdu7Zo4vDAgP7iS\/euvv1Lv3r1FPFCbNm0CV78xJxxRDR06lNKkSZNsCGbWgRldd1ZDdL26RmAQpY3rckuWXA8wg4GyZctGx44dExHckOBFLDs9PIuFj6tomuC6GhgzCBME\/41Ib9mS5zLGDy3iyNenZS1mvZ7qGyLngbF+TfipRZn3kZfni\/gRbMy4Eo2rwtjMEbwLkoENHcQEt4cQQIsfoPBw4OozbvjgiAXXoSG4cozPcPMIMSXIkosgWOR\/QTDtxIkTxeYPrwhICkgSbg4hSR72gnr16onAWfy4DRa0AbIDT0769OkFoUIbGTNmTKKHNkCINm\/eTKVKlaKDBw8KApI7d+6I48aP9o8++kiMF3MH0cGeBEG\/uEKOz8qXL08JCQmiPT0xsw7M6Hp57YSOzTUCA\/caDIXFg+txjz\/+uGDNuKo2YcIEsfCw4GBo7Y68DLBg22PGjBGqcFGC5SKoCsQG0d64pnf8+HGRshrMWY\/ZhvYjY\/zQGkgagYkUxCYzH9YxjwATGPOYufmE6vZyGjuZ95HTY+L+nEfAzDowo+v8TKLv0RUCA5b98MMPC\/aNe+96EdlgzF988QXly5dPEA3NxRhpqjiCAvNGu0899ZTu\/X6wb+26HRg6xmEkMsYHgZnQuig9XiK7aA4EhsmLEbL2fK76hsgeGHvWhV9alXkf+WWuPI\/wCJhZB2Z0VcLcFQIzdepUcU0ObrNffvlF1wuCTQjuNQhcbgjsMhItERJclLhvX7Ro0WSP4DodzkUhcM\/hWpuRGBl\/ybYT9MSY9XRwYGW6vPwzOjz6WdEkExgjZO35nAmMPbja1arq9rILl3DtGr2PnB4P9+cOAmbWgRldd2YTXa+uEJhXXnlFBO4isVBonYngaSDpEM4jcQaJ7IVWCJIN3XfffcKjg2qoCLoyEiPjB1+h1jLv5uq71PD6oVG\/\/Hl0CKi+IbIHJjq73yxPGb2PbhYcbvZ5mlkHZnRVwtUVAoNEPZs2baJnn302bAQ4QMQxECLFQTJANqwQXFHTUksj6jtcgFRwX0bGD76BxKUDrLBSbG0wgYkNP6efVt1eTuNl9D5yejzcnzsImFkHZnTdmU10vbpCYDTPCiLAI1UKRbEu3CbC1WetPkR007z+FGJkEE2Oq3NNmzal999\/X6o5I+OHXqFGam2j9NtSHbNSVAioviGyByYqs980Dxm9j24aIG7yiZpZB2Z0VYLVFQKDeJYzZ86IG0G4dhZOQHCmT58uUjcjP0AsgrgY3EpC3gFUC0U10NBrceHaNzJ+6A2kWMbJz8aOABOY2DF0sgXVG+q5ewAAIABJREFU7eUkVujL6H3k9Hi4P3cQMLMOzOi6M5voenWVwBgF0aLKJ4hGrAQGOQRwXAXPC8gLEhOZTWQXDC9Ilxb8+\/Xms9Rv0VHqVvk2alrs+l1+FncRQFXXcCXo3R2ZXO9aCvVYEjnK9eQNLdXt5TSKuNyAdxrLzY0ASMmiRYt0QcAeh1u2weLHNeMKgYmPjxdJf4yOkFD2HAmPkEZ5\/PjxUa1WJMrr2rWr8PhgQ0ByIi1pkGyDkdgrlxCQRdE5PdV\/0fMRknNrRcWe\/PprWkVbuDlmM+vAjK6bczLbtysEBkWwNmzYQG3btqVevXqFHTPiVHDNukmTJtS\/f3+zc6OxY8cGnkNmR+SW0epPmGkskvG5hIAZJJ3RZQLjDM5W9aK6vazCQbYdv2xG8BDgH\/KCISs6QgruvvtukWgUoQNISopsvChyiPf2lStXBETIG4YsteEEZWkmT54s0vznz59feCn08ojhxzF+JOMHbffu3cUPZdyORTVs6CPLL\/YeJEKF4DQA6TmQMR43aVGOAGk5kEl348aNQg8\/zPHshx9+KKpfoz0kVkU9I+QcQxmb4OKPuFSC8IgLFy6IKtmnTp2iO++8k\/CdQNxnaJbg4DmbWQdmdGXXoRf0XCEwSPU\/c+ZMYdBQN1cwKKhDgcUcXFpdFjSknYa3Bdl9EazbqFEj2UeT6RkRmIqF4mhOpxJRt88PWosAb4jW4ml3a2wvcwj7YTNCbi\/UQAI5gf1BVpC2Hxs63tkQ7A8gF\/PmzQsAhEsdCD1A6v7hw4cHygqEIoiM7qhqDcEPWZCNUEEoAI5rcSQHHU1AMDAWJD0NFewrKP6oZYdHRnn8yP7qq68E2dq5c6eI60R2eaQJgXzyySdiDwKhQqJVCMoQ4Mf7ggULRDkCrUL21atXxbzwDzWSQJbCiZl1YEbX3Gp0V9sVAoPFC3aLAlg\/\/\/yz7iL8448\/qEGDBgKd+fPni9gVWQH7xQLAIkSemUhsXaZNIwLzVp2C1HBsftFUwcTr9ZZY3EOAN0T3sI+mZ7aXOdT8sBkhLQZiHDUBSXnhhRdozpw5gQSkIBa4aDF79uwkAC1btkzUSkIcIrwfeoIij0jDAc8IyA6SpwYLvDRTpkwRXpWaNWvSiBEjAh+H6xcKCGW49957A4WGQWSKFSuW5IQAe9qOHTuoefPmok3Eo4B0geRo9ffQL+b\/6aefJvHIaIPAj3wUp4yU5sPMOjCja241uqvtCoFBPAquUoMl41gHAbahgsWMRQ03nBYTIAMV6h2hSCQKfGHhxEpeBCkpWDBs0ByOkL7uWJLu+bwhnd+0lAmMjJFs1uEN0WaALW6e7WUOUD9sRjhqyZkzZ2DiWuZ1HOvAwwGJRCTwjof3YvXq1ZQjR45kAILAdOzYURzLoCRNcLtQRnJUhDAgJ5kZAoM6ffgxrQXYY5+C12jUqFEBjwvaB0HS5qFHYHAxBQUsQytsaxNBgUgcSUW6bGJmHZjRNbca3dV2hcBgyqNHjxZnjSimiFLiqEAK1yHIDbwnYLpws6E6J8hOsOD8EQsebBbPaoLzyCpVqogz1S5duojgXSsknPGDA3i5+rQVSFvTBm+I1uDoVCtsL3NIh76PrhxbY64Bh7VT3VbOsEdUlobXAclFtXiVSARG2z\/CeTA0AoPYFhwfPfHEEyJ+BYKUGq+\/\/rrwzsObEguBWbt2rYjJQYb3Zs2aiZMFLWZGm3Qogdm\/f784HsPxWbDnxxCkEAUzpMSMrtlxuKnvGoEBOYGxEeQEQbBSXFycKCOuBWuFyxMDt93WrVuTxdBoR1NoD8dHRgG75cqVS3L2Gc4Q4YwfHMDLxRvdXMZJ++YN0Tu2kBkJ20sGpRs6oe+j43MKmGvAQe3Ut5enTBW+MOzxzTffJNxOxaauSSQCoxXlDZeKQyMwiG3BcRMIEsgRPDL4GzAEsYmVwGCs8LZg\/LiYgj0M8S4gReEIDOJucPzVsmVLESsTrZghJWZ0ox2PG8+5RmC0ycK1hyhslBY4e\/asWAAIdMKxUrgCjuEIDLw1kW41hQIMzw7OQY0kEoHRAnjZA2OEonOfq74h8jVq59aKij2Fvo\/OrHrKs9NIdVt5SnfvSxHHh40fax7hBMFiFIvy7rvv0rBhw8QxUKgEExgQFwTW4sYRsrvDY4K9AkG3egQGxCZdunQiC3yo4AgJtfRCTwXwgxxxNiAk+AEO7w5uNUFCPTDacRnmhwDfaMUMKTGjG+143HjOdQLjxqTN9hmJwLxRuwB1ybOHDvSpIprlIF6z6FqvzwTGekztbFF1e9mJjV7bftqMcNyP4yDcNAWhkCUwCIBNTEykVatWJYml0Z4PJjD4W+3atYV3H159xN9oJWz0CEyrVq1EEWG9+JSPPvpIxOYgvuXAgQOUOXPmJBndMR54VuDdR+yNHoHBTaOiRYuK69LIUxatmFkHZnSjHY8bzzGBkUA9EoH5qkMJqnpvHMEDE9ckgeIa95FokVXsRED1DZE9MHauDvXb9stmhJwn8KAgTkXLd4L4lCxZsggyE84Ds337dpEWA8c0Q4cO1TXo5cuXqVOnTgEPB7z8CNxFfAqICfqAwJvyyCOPJIlFQVwlQhtwPTo0DwvidODlx\/O4KYvjIAQUBws8LyAnuDatR2DwNxAhjB0xMMHHZlo7OI0ACdPGGSuR9cuaCcWBCYzE+0zP+Iu2HqfGYzfQ3M4lKf7urILApHugChdxlMDTbhUmMHYjbG37qtvLWjSMW\/PDZoQbqLiMgaMdLTM6CA1u9CCuBaKXBwa5Y3AMhNs5iGWBB0RP4NnBDSGthh6CbBFjU6tWLcLREwQEoVChQoLABOeBwXrEsRSuQSNbvOYZwq3Y9evXi3gXCC6c4DYRkubhfyHoF\/lfcByGoF6IlgcGR0zarVgcMyGpHW4bIZdMcMwMiBGuk+MiSqR6fWbWgRld4xXoHQ0mMBK20DP+Bz\/spX7zdtLxodfPOSEnEvuyB0YCT7tVVN8Q2QNj9wpRu33VNyMcoSCwVqv5FWwNxJDAI4HjIQTD4mYqyARuJoFwgPggOy5IQuiRk9YOSAFIA+IrcdsIGd1BdJAfDPGTuAa9b98+kYMF2W5BEuAFwk1Y3IqFIJcL9E+ePClysWAcICkYd3BWXxAPeGnQJv7+559\/ihtPWq08jAEJVXHppFKlSqKgsHYFG3EzyA2DpK4gb9p1cBw\/QS\/c\/LR5mlkHZnRV+nYwgZGwlp7xByzYTQMW\/JWEwEg0xSoOIMAExgGQLexCdXtZCIVUU37djKQmz0oBBMysAzO6KkHMBEbCWnrG5yKOEsC5pKL6hsgeGJcWjiLd+nUzUgR+zwzTzDowo+uZCUoMhAmMBEh6xucijhLAuaTCBMYl4KPsVnV7RTntqB\/z62YUNSA36YNm1oEZXZXgZAIjYS0mMBIgeUhF9Q2RPTAeWkweHIpfNyMPQu3pIZlZB2Z0PT3pkMExgZGwVjgC07\/oIarxXXPO\/SKBoZMqTGCcRDv2vlS3V+wImGvBr5uRORRY28w6MKOrErJMYCSsFWr84ABezv8iAaDDKrwhOgx4jN2xvcwB6NfNyBwKrG1mHZjRVQlZJjAS1go1vhb\/8uu+auJpzv8iAaKDKrwhOgi2BV2xvcyB6NfNyBwKrG1mHZjRVQlZJjAS1tIjMKUvbqAxh19m8iKBn9MqvCE6jXhs\/bG9zOHn183IHAqsbWYdmNFVCVkmMBLW0iMwL+XdQ61WteX4Fwn8nFbhDdFpxGPrj+1lDj+\/bEYTJ04k\/EP22pIlS4o6RXfffTcdP36cpk+fTgMGDBDJ3JDYLmXKlKJIIgTFGLWMtnrIIZMtUvUvXLiQChQoIBLQIckcsvFire3atUtUjoacP3+e+vbtSxs3bhRlBlAnCbpIYlekSBFR9HHQoEF0+vRpMQ4kuoMOShB07NhRVLdGyYAhQ4bQsWPHqHLlypQ1a1YxJ5QT6Ny5s6h8DUGZBBSIhC6y+CIpHpLn4X8xv3vuuYcuXbpEa9asEUUmkUBv7969tGPHDlq8eDHlz58\/yXTNrAMzuuZWo7vaTGAk8NcjMNrxERdvlADQYRXeEB0GPMbu2F7mAPTDZjRr1ixCWQCQAtgfZCV9+vSilACy3kL0SgmAAKDUQKlSpWj48OGiarSeoO3HHntMlALQCjdqeigxAIIDQRp\/1DxCRlyQpZ07d4ryBsgIjGy\/ENRUAoFANl2N9KDMAIo5giShXtHLL79Ms2fPJtRpSpUqFaEWEwgOyMiCBQsoV65cgWG2a9eOFi1aJPoCWUKm4ClTpoh5b9myRWCC4pYoNQDBHDAmFJAMFjPrwIyuudXorjYTGAn8g42\/cudJqj9yHW3K\/Dad37SUPTAS+Dmtwhui04jH1h\/byxx+ftiMBg4cSKgorQnqDIFYoAYQKjVDwhVzXLZsmfBaIF0\/vCd6AiKBCtR6BAalBuBdgaC0QLFixah\/\/\/6BZlBGAF4PkBTIiy++KIjKtm3bAjogHCj6qBVjRJFHlASAd0eT+fPnC\/IDwqWVFsBn8MqAqGntoV4TSAxEj8CsW7eO0qZNywRGx9BMYCTeHcEvDC4hIAGYyyqqb4icB8blBeTx7v1AYHAMkzNnzgDS+M6CsKB2kOZpCEdg8BAqQMOzsXr16kANoWCzhSMwICfFixcPVJkGaQKZGDVqVMDjohEJbRx6BGbSpEmCmKAIZI0aNUiPwOBoCYUf+\/XrJ469NAklMMHj1iMw4ZajmXVgRtfjyz\/J8JjASFgr2PicgVcCMJdVmMC4bACT3atuL5PTjVndj5vRTz\/9JEjAjz\/+GCiWGInAjB49WsSmoCAj4k5CRY\/AII4FFbCHDRtGGTJkEI+sXbtWkAvEx6B69KuvvipiYYIllMAglgUeExSXxI8NHGOFEhisaRz7oHAlSBmOmZjAxLz0kzXABEYCUyYwEiB5SEX1DZE9MB5aTB4cSiiB+XHwPx4c5fUh5auQke6qcJ0sRBJ4KuLj40X8hyaRCMx3330njmJCj2e0ZzUCg2BY4IXKz2fPnhUxKb\/99luAwEAfXg\/0j8DeuLg4UQUbVaY1AYEBCWnYsKEIvsXx0v\/+9z\/q2rWrCOKFaATm8ccfF7EyqC6NYy6QokyZMiWZOntgjFaD\/OdMYCSwCiUw3WsVoO61kkaESzTDKg4hwATGIaAt6kZ1e1kEg3QzoQTmvTs3Sj\/rtGK++IzUYub1WzjhBMQBpP2tt95KohKJwIwfP57effdd4U2pX79+sqbDHSF98MEHIqhX88BoD4LgTJ06VQTv4rbTuHHjqGrVquJjvSOk0A6DPTAI6AXhQWwLbleFChMY61YhExgJLLUXhhbA+3XHklTpnqwST7KKGwioviGyB8aNVaNOnyp5YIDqw6\/dGRZcXDfGcRBu3eAWULBEIjAIAE5MTKRVq1YliaXRng9HYI4ePUrZsmUTfeEWUebMmSljxoz\/196VgNtYdu1FlKH6QlyRpDJkSiWR+CSUecw8VUSFEGXOIb\/MQ6Ii85ikzPMQUkp\/ZCiFSEmZyVTUf90r7\/n3OWfvs993P3t6977XdZ2Lc\/Yz3ut93ufea61nPYndor3mzZtLqVKlZM6cOQERGFh6cAIKbimcmoJVx1NIYIK31khgbGBpvTDGrPtJ+i\/ZLydH\/svMKdGJAAlMdOrF16jcrq9wox0rMTBws8CC0r1798SgWsSXIF4EBMMXgQE5qVevnrp5Ro4c6RX+1E4hoQLyuyCYd\/PmzRoQ7CmwvCCHC4JwIU4tMKgDtxTcSWXKlJHJkycnxvXgs9QIDE5I1ahRI8kxal\/Pl5PnwEnZcD\/PJv2RwNhAz1J+rfHbpe7WDlL+wqeSq\/8GyVA4ZfCYjeZYJMQIuH1DpAUmxA+Iy5uPhc3o\/PnzGlCLQFcrRgSEBieCENcC8ZYHBvldOnXqJDly5NATQLCgeBMks6tVq5bmZwEB8RQkz4N1pEmTJprkbsaMGfovBBYh5H+BOwvxKxC4nDAuxL4ktxJZ7VpxMoh\/QbI7iyT17dtX42NwnNoSKw+Mt\/YQVIzkdrAwJSdWyefp5DlwUtZNy4MExoa2PAnM6PX\/+jSZwM4GcBEqQgITIeAD7Nbt+gpw2gFXc\/tmhJM5CHCF9SO5IAYFgbxwDyGYFsndEOOChG849QPi07BhQyUZvsgErB84Fo04FLhvKleurBYeuHbwGUiGlW8GVhx8VqBAAe3jhx9+kPr16yfmbYEVBuNA8G7r1q2V1CBbsKdgrEiIBwL2zDPPqAsKGYAhIDb4QgLSAmICguLZHv6GviFIqAdShiR3+fLl07w4IGG+xMlz4KRswA9mBCqSwNgA3VI+jlAzA68NwCJcxO0bIi0wEX6Aorz7WN2Mohz2qBuek+fASdmom2gqAyKBsaEtKH\/G6v\/VDLwkMDYAi3AREpgIK8Bh927Xl8PpGheP1c3IGJg4a8DJc+CkrJtgJIGxoS2LwCSMnKo3UGdpmCBZGvSzUZNFIoGA2zdEWmAi8dS4p89Y3Yzco4HoGKmT58BJ2eiYnb1RkMDYwAnKP11nkpS4vIMExgZekS7idgITafzC3T\/15QzxWN2MnKHA0k6eAydl3YQsCYwNbVkEpu3ZadL2zDRaYGxgFski3BAjib7zvqkvZ5jF6mbkDAWWdvIcOCnrJmRJYGxoiwTGBkhRVIQbYhQpw8ZQqC8bIHkUidXNyBkKLO3kOXBS1k3IksDY0JZFYHiFgA2woqAIN8QoUIKDIVBfDsASkaZNmwouP6TENwKlS5fW6w\/sCAmMHZRitEyeh6vJH2VfkR\/\/p5z8J2O6GJ1l7EyLG6K7dEl9uUtfGC115i6dkcC4S19BHW2ual3k0r21eIVAUFENXWNuf7nyFFLong22HBwE3L7GgoOCe1ohgXGPrnSkyGqIFNHbtm2TkydPSsaMGTXjIa5ERyZHX1kcvU0TCewgvAPJHQ+B21+uJDDueM7ieZRuX2PxpjsSGBdpfOzYsXrJF1JDX3fddZItWzZNI33hwgWdBXyHU6ZMSbxAzN\/USGD8IRRdn7v95UoCE13PE0eTEgG3r7F40ykJjEs0vm7dOmnTpo2OFv\/i5k\/cbor7N1auXKk3n4LM4L4K3LthRw40SKPFeIGjHbQiX8btL1cSmMg\/QxxB6gi4fY3Fm35JYFyi8WrVqullXbgQDJaY5LJ48WK9zRSWmfXr10vu3Ln9zgwEJmORxyRnwr+uJEp0I+D2lysJTHQ\/Xxwdg3jd9gyQwLhAY9i4KlasqCP96KOPpHjx4ilGDUtM2bJl5ejRo9KjRw+9Kt2fkMD4Qyi6Pu\/fv7\/06+feqx7ijcC4XV\/R9fSHZzTUWXhwDlYvJDDBQjKE7bz\/\/vvSs2dPufHGG2X79u0+A3W7dOkiCxculAoVKsikSZP8jggEJmfCBslYpLzfsiwQeQTcvljjjcC4XV+Rf+LDP4Jo0dnFvWNSTH7b4sbhByTKexwzZozMPzwoykfpfHgxlchu0KBB8t5770nhwoXF2gS8QQJl4gfuo40bN\/pFrUebp2Xwe1P9lmOB6EAgWl6udtHYNPy3JEV3ntygvxfL+pjdJlxdDmsRbl2KexAIls42DUv67LsHAfeNtNdv97lv0H5GHFME5uWXX5aPP\/5Yypcvr6eMfMncuXOlV69ecv3112u8jD\/BKSQeofaHUvR8HikCAyLy05bzcujTP4zAOHPvl1r\/P9+VNGqHlYlAvCLQ6L1jcmRv0Xidfop50wLjgkehXbt2snr1aqlSpYqMHz\/e54gXLFgg3bp108\/37dvnNycMLDDz1vm31LgAIg4xRAhUvzRQcl4N7IVZ65U++rKdsOLfwWXKf1H\/vfBDxhCNls0SgehB4KGacxMHY62B6BldbIzEybUDbppxTFlgLAJTtWpVGTdunE89IMC3a9eutgmMmxTKsRIBIkAEiAARiAcEYorA2HUhzZo1S\/r27SsZMmSQPXv2xIOeOUciQASIABEgAjGFQEwRmMGDB8uECROkUKFCsnTpUp+KGjVqlOaIyZs3ryDxHYUIEAEiQASIABFwFwIxRWDmz58vr776qmTKlEl27Nihyeq8CbLzLl++XCpVqqSEh0IEiAARIAJEgAi4C4GYIjC\/\/PKLlCtXTjWAnDAlS6Y8xfHnn39KqVKl5MyZM5KQkCAtW7Z0l8Y4WiJABIgAESACREBiisBAn40aNdKbqJGRd+LEiSlUPH36dCUuOEK9adMmyZ49Ox8DIkAEiAARIAJEwGUIxByBAXkBiYG0aNFCENiLyxyvXLmiOWL69OkjsMK89NJL0rlzZ5epi8MlAkSACBABIkAEgEDMERhMCqeMXnvtNfnnn380DgZWllOnTsnly5dV67joEYl90qZNy6eACBABIkAEiAARcCECMUlgoIedO3fqtQJbt25V8oL7kYoUKSINGjSQmjVrulBVHDIRIAJEgAgQASJgIRCzBIYqJgJEgAgQASJABGIXARKY2NUtZ0YEiAARIAJEIGYRIIGJWdVyYkSACBABIkAEYhcBEpjY1S1nRgSIABEgAkQgZhEggYlZ1XJiRIAIEAEiQARiFwESGC+6RUZfXDGwefNmOXLkiKRLl07uuOMOTY7Xtm1buemmm2L3iYjCmfXv31+mTZvmd2Q5cuSQzz\/\/PEU5HKfHDeTIA\/Ttt9\/K2bNnNTfQAw88oLmCypYt67dtFvCNAE764QqPw4cPy7333ivLli3zC5fJGjOp63dgcVLAic5WrFghL774oi1k1q5dK3fddVeKstSZLfhSFDp27Ji++3Bn308\/\/SR\/\/fWX3HLLLVKsWDF56qmnpEqVKj4bNsHcpG5gMw2sFglMMtyQnbddu3Zy6dIl\/eTWW2\/VhwZXD0Buu+02mTt3ruTJkycwxFnLMQLWLePI6ZM+fXqf9XPmzCl4gXoKdPf888\/L+vXr9c\/IwJw1a1Y5ceKE6hWCl3O3bt0cjyveKyCv0vDhw2XSpEmJUNghMCZrzKRuvOsL8w9EZ7iWpWfPnppXK2PGjKnCCLKT\/N1InQX25CEp63PPPadfuCB492XIkEHOnTuX2GDt2rVlxIgRKXKamWBuUjewmQZeiwTGAzuw3ccff1zOnz+v38oHDhyYuBh37dqlWX337dsnBQsWlCVLlvi8LDJwdbCmNwTatGmj30CeeeYZ6du3ryOQBg0apPmAbrjhBhkwYIDUqVNHXwTQ8eTJk2XkyJGSJk0aeeutt6RatWqO2o7nwtu3b5euXbvKjz\/+KHfffbfkzp1bNm7c6NcCY7LGTOrGs66suQeqM1ijBw8eLIUKFZKlS5c6gpI6cwRXYuHff\/9dKleurGQFFi1YoR955BHdc2AdGTJkiO5BkNdff12aNWuWWNcEc5O6gc3UrBYJjAd+2OCmTp2qL+OVK1em+LaBBwcEB9\/c8c2zXr16Zuizti0EkHzwq6++kk6dOumPXcFixOWeuDoi+SK32ujdu7fMmTNH7rzzTiVJIDMU\/wjAlbp69Wpp1aqVdO\/eXUaNGqVE0Z8FxmSNmdT1P6PYLxGozvANf9y4cXoJLtaKE6HOnKD1\/2UtzGExxjpDCIOnXL16VWrVqqUu8fvvv18WLFiQ+LEJ5iZ1A5upWS0SmGv4wTwKhgvmC3MpTHfeBBvo4sWLpXz58jJlyhQz9FnbFgL4JrJ\/\/37p16+fbph2Zfbs2Xr3FWKWYI7FyyC5wIKA2CYI4mSKFy9ut\/m4LgdcoResA4hl6UqNwJisMZO6ca0oj8kHojNUx7UsM2fOlCeeeELeeecd23BSZ7ahSlFw6NCh8tlnn0n+\/PkF\/\/cmsMK8++67kiVLFv2CBzHB3KRu4DM1q0kCcw2\/gwcPqnUFsnDhQg2S8iaIf+nVq5f6Infv3s1v7GbPn63aJUuW1JgVp1YvK3YGeoV1wJeUKVNGjh49mipxtTXQOCr0999\/J\/G72yEwJmvMpG4cqSXVqQaiMzTYsWNHdR01bNhQXUl2hTqzi1Rg5fA+HD9+vOTKlUsPnEBMMDepG9gMzGuRwFzDcM2aNXrCCPLNN9\/o3Une5IsvvpDGjRvrRwh2uv322821wBZSRQDf6uEGGjt2rJw+fVpdPYcOHdI6wB9WANxAnjlz5iTt4M4rkMzWrVsLXEW+pEmTJnpnFqL6fX3boYpSR8AOgTFZYyZ1qTvvCNjRGWrC6ol3HWLRHn74YVm0aJHGAiIgGIccSpQooTEYyd+F1Flon7waNWrInj17khBLE8xN6oZ2pr5bJ4G5ho0VaY9gT\/gVfYmnywF+R\/gfKaFD4OLFi3oJJwSk8o8\/\/vDaGb6FTJw4UQMNLbEsKz169Egkp94qW27B\/\/73vxoDRXGOgJ3N0GSNmdR1Ppv4qGFHZ0Cibt26smPHjlTXH9yzIP+Iy7CEOgvdczR9+nRJSEhQtzhOfuXNm1c7M8HcpG7oZpp6yyQw1\/BBPAsCPZEf5Ouvv\/aJGvLCWHlD4BfGJkkJHQJw7VgY33zzzWrORu4D5HyBWwnB1sOGDZMLFy7ot0H8Dp8wBPEsiOL3FzsDgjNv3jyBqwqLmOIcATubockaM6nrfDbxUcOOzoAEXLBwLyDAHdYYBNXj5BksMJ9++qmuP+tzxJ0h2BdCnYXmOUI+K5wARMwKSCMsx5aYYG5SNzQz9d8qCUwyAuMZEOUNPpxEwskWCAmM\/wfMtASOO+MlCUFcEnK9JBcE6MKFBGnfvr0ubk8Cg28qLVu29DkUJGGbP38+CYyBsuxshtYLMpA1ZlLXYFoxXdWOzgDAJ598omQFJ2E8LZwWOMePH5fq1asLTv099NBD+mXAk8AEou+YBj7AyYGwjB49Wl3p+D+Cq5FawlNM1olJ3QCnZFyNBOYahHbNZwcOHJBKlSppLZ5aMX7+gtYAFjJetJ65Kh599FH59ddfxZ8LyQpSrFChQpKkbEEbXBw0ZGczNFljJnXjAP798oR5AAAOGUlEQVSApmhHZ3YbtnLFYGOFuwnWUurMLnr+y8F1jmSbq1at0pxWCOAFaUwuJpib1PU\/g9CUIIG5hisytSLYE4KET1iA3mTLli3SvHlz\/Qj\/R2ZeSuQRsPImeLoALd+9vwR4sN7AiuP0lEXkZx09I7CzGZqsMZO60YNSdI3Ejs7sjhinYCwr5\/LlyzXZJ3VmF73Uy\/3888+a1mPv3r164ghH2YsWLeq1kgnmJnWDM1PnrZDAXMPM0zX04Ycf6j053sQKnkJAKU4rUaIDASsnQvbs2fVEEQTfWBBoDZdfancpwex98uRJzRnz7LPPRseEXDYKO5uhyRozqesyKMM2XDs6szsYWD8td4Z1HxJ1Zhc93+VwvxhOSSL2Eu8pkBdcheJLTDA3qWs+08BaIIHxwA3HcfHAdOnSRYNFvQmYMBZo1apVNTslJbQIIEETLGIQy0LmrUdcyohYGRzzRK4eCFx8iIfB8Wocf\/d2jwuuiLBOTiCav0CBAqGdUIy2bnczNFljJnVjFHajadnRGS4Q3LBhg+ZJwjvR111IiMtANmacioELCW4OCHUWuIpAKGAdBnnBwQXEv3hLxpm8BxPMTeoGPtPAa5LAeGCHBYiFCLcQfI3Jc8HgeDXO3sPPC5+vFQsTOPys6Q8BHI1+4403FHNkQPZmOsULE+4iiKcVBSeQcIIJgcDIA+ONAHXo0EFvTy5cuHDi3SL+xsTPUyJgZzNELZM1ZlKXOgtMZ54E35eFEuvsySefVJKDdyLejZZQZ4E9ebiuBi5tvNuQ3gEXpuIeJDtigrlJXTtjC3YZEhgPRLEQkVYeUfX4Jo+N07oaHjESsMyADXt+yw+2QtheUgSgExzjxJFpHJPGRonf06ZNK1euXNGkdiAn+By5EJAx1PNb4ttvv63HPPHNBVH7OAKKyxzR7ptvvqkvBpCjWbNm8Ui8wcNnl8CYrDGTugZTi9mqdnVmJbLDBoqA+KZNmyausZ07d+oFq3CnY13hiK\/nSSXqLLDHBxfN4jJhxPThHWelhrDTmgnmJnXtjC3YZUhgkiEKdwUWLBSJjQ2bJtiwdaV5vnz5BLkO8HdKeBCATpAFFHEqEJCRbNmy6e843gkB0cR1ARbhtEYGHcKNhBcrBKZtvAxAUkGAIP7yxIRnlu7pBd8Kkx9Lhx6QLRmCu6c8Bfe1lC5dOvFPJmvMpK57EA7+SE10hrUC6yXIiiWINYPOrfcirNVwcVjXsXjOgDpzrk9YXRC8C9KYKVMmvw0gAadn3KYJ5iZ1\/Q40yAVIYLwAinwGCJaC7xd+SGyYuK0YqelBbiz\/bpB1weZSQeDUqVMaiAud4Cg73ELYKHHNAOKRYG715Z9Hs7DM4JggrhbAkUSQGFjS8GLmBY7OHj0keqxfv77tSgh8t5I\/WpVM1phJXduDjrGCpjoD2UeuJKwjpK\/HlR7YWJEbBnETeC+mdiKTOnP2QOHdBOJoV\/BuQyJOTzHB3KSu3TEHoxwJTDBQZBtEgAgQASJABIhAWBEggQkr3OyMCBABIkAEiAARCAYCJDDBQJFtEAEiQASIABEgAmFFgAQmrHCzMyJABIgAESACRCAYCJDABANFtkEEiAARIAJEgAiEFQESmLDCzc6IABEgAkSACBCBYCBAAhMMFNkGESACRIAIEAEiEFYESGDCCjc7IwJEgAgQASJABIKBAAlMMFBkG0SACBABIkAEiEBYESCBCSvc7IwIEAEiQASIABEIBgIkMMFAkW0QASJABIgAESACYUWABCascLMzIkAEiAARIAJEIBgIkMAEA0W2QQRCjMC3334rM2bMkLlz52pPhQsXFtwA\/Pfff+sllrVr19Yf3F4balm3bp306tVLb\/8uWrSore5GjRolS5YskWXLloXtMtTevXvLnDlzpHHjxpIzZ07BDb+hvLgTl4ROmTJFLly4oHqCXrZs2WILHxYiAkTAOQIkMM4xYw0iEDEErFtqt27dKtmzZ1cC88EHH0iPHj3kgQce0Bu306dPH9LxffLJJzJgwAB56623pFChQrb6QtmVK1fqjcbhus0dBGb9+vURIRF9+vQRED0SGFuPBwsRgYAQIIEJCDZWIgKRQaB8+fJy+PBh+eqrryRLliyJg3j66adl48aNMnLkSKlTp05kBhdlvYLAgECAxIRbBg0apBYnEphwI8\/+4gkBEph40jbn6noEKlSoIIcOHUpBYBISEmT69Ony8ssvS4cOHVw\/z2BMgAQmGCiyDSIQvQiQwESvbjgyIpACAV8EBnEeX3zxhcalFCtWTBYvXqyupWnTpgncGTt27NC4jLvuukvLLVy4UH7\/\/Xc5evSotG7dOonV5urVqzJr1izZt2+fXLlyRRB\/0759e6lUqZKO58iRI2pdgPuoXLly+rdTp07JkCFDJE+ePPo7+ujXr5\/2B9mzZ498\/PHH8uyzz8ptt92WOK81a9bIhg0bJFOmTLJ3717JnTu3dO\/eXW6++Wa5fPmyLF++XFasWCH58+fX+JVx48bJ\/v37pWbNmjJw4EBJkyaNz6ckNQKzadMmWbp0qWTIkEG++eYbqVy5sjz\/\/PM6X\/SHfmHhQj\/Dhg1TLOC+e\/PNN3X+gwcPli+\/\/FLnB\/fY7bffnmQctMBw8RKB0CNAAhN6jNkDEQgaAskJzF9\/\/SUTJ06U4cOHC9xLkydPVkLwxhtv6KYL4gFSAUKCvx0\/flwDW8eOHasBvyj\/+uuvy\/jx46Vq1ao6TsS3IL7mhRde0N9h1QHhAdlAfdTdvHmztteoUSMtA5KEDb9r1676Ozb46tWrK5kCcRkzZoxajhAXkjdvXi2DgNcFCxYkxsVcvHhRmjZtKqdPn9ZgX8i2bdukVatWUqBAAalXr55UqVJFxwxr06RJkwR4+BJfBAYE5cMPP9Q5I14IcUM9e\/ZUAoZAaBA29AXcGjZsKHXr1pUDBw5I8+bNpWLFivLggw\/KU089Jb\/99pvUr19ff0BYPIUEJmiPPBsiAj4RIIHhw0EEXISARWDuv\/9++eeff9QacNNNN+nG265dO7n++ut1NrB+4NQSLCU4sWRJtWrV5LXXXpPSpUvrn44dOyalSpXSHxAbxGx06tRJPvvsM0mXLp2W+f7772XEiBFq8QCxQQBxkyZNkhCYFi1aSNq0aZVcgBgdPHhQQEisIF+QHpxEsggMiNBjjz2mp5lAWiyBZQSEBcQLZAhzvOeee5RQgKRBYDUqU6aMX3eZNwIDSxH6hTXKGhvG+corr0jLli3VygJB+\/ny5VOiZEnJkiXllltukdWrVyf+DZavs2fPJhIu6wMSGBctKg7VtQiQwLhWdRx4PCLgy4WUHAtYUaZOnarWBOvUj7Xx16hRQzJnzpykCkgQyAQ23l27dsns2bN9wrtz504lTJ4WGFh4+vbtq24etJH8dNKECRPUKmMRGPzbpk0btcLAcmQJCAvq3nfffTJv3jz9M9xHsHigPwiOK+Nz1MeYfYk3AoN+4SoCLqkdOS9btqwSJ7jgLIElBm4tyzqEv7dt21ZdUJ9\/\/nmSYZDAxOPq5JzDjQAJTLgRZ39EwAABEwKDmA24fBYtWuQzfwusH8hjgvgZX+KNwKAsXDKjR48WxNBgY+\/WrVsiSUhOYGbOnKmWIJSvVatWkq4Qa3Pp0iV1U3kjMOfPn1fXVCAE5u2339aYFk9i522eTgjM119\/rfEwnkICY\/CQsyoRsIkACYxNoFiMCEQDAiYEBkGyiHNBzEuzZs2STOfPP\/9U0gCLBo5jw02ERGyeAusDrDneCAzcMCiPPuC+QhDvSy+9JJ07d9YmkhMYy1UEkvPiiy8m6QdxJrly5VIXWLAJDFxCOLEFglaiRAmv88MfSWCi4WnnGIhA6giQwPAJIQIuQgDZZH\/++WcNbs2aNavPkXtzIYFkII4DcSw4pYRMvpYggBVxJ4iDgaumY8eO0qVLl8TP4SLBCR1s7N4IDE7iWMe3YYEBQQLh+eijj7wSmHPnzukJJmTIxYkfSzBGJORD\/Mtzzz3nlcCYuJBwGguuoEcffVTjdaykf4gFwjgQB0MC46IFwaHGNQIkMHGtfk7ebQhgcz9z5oysXbs28YiytzlYeWGwYSO+xRIrmBanehA8i5gTbNw4bYNAWbhnnnjiCfn111\/VWoOg1u+++06rWzEolisKfVgbPoJ6QZoQrwLBySUcjUbgL8TqFyeA0DcEVpBXX31VjyHjxBIEpAInhHByCeQCZAjBtCAdSNIHsQKPkbwPbihf4usUEmJgVq1apfE6CGo+ceKE7N69W+BeskgdgpoRA+MZC4Qj1SBYOI1lCY6gg0wCZ0+hC8ltK4vjdSMCJDBu1BrHHHcIJL8LCRssSEPy+BEAg81\/6NCheloHMS8gKogZgYCwIFcMjg7jBBM2aVhecDzZEtQDOYEbCSSkQYMGWgaEAhmA3333Xd3EixQpojlbYJXBKR5s7iA8sBDBSgJ3FAgBCBLIB\/K3IPgXBKJgwYLaHbLk4jj0nXfeqZYhWG3geoI7Cm0gXgWupGzZsilZgeUELirkcEHuFRAkzyBgzwfDF4FBH8AHVxugj8cff1yPUaN\/WIZArOBmwxjQJ4gdjpEjSBnH1tE\/\/gYM8DnaQH1gjfw1EBKYuFuinHAEECCBiQDo7JIIEIHQI8BMvKHHmD0QgUgiQAITSfTZNxEgAiFDAAQGOVsQUBxu6d+\/v1p4eBdSuJFnf\/GEAAlMPGmbcyUCcYQACAyCkpFs7o477tCg4aJFi4YMAbiS4O6Ciwr9Is8MCUzI4GbDREBIYPgQEAEiQASIABEgAq5DgATGdSrjgIkAESACRIAIEIH\/A4JNzM2Wzx3MAAAAAElFTkSuQmCC","height":337,"width":560}}
%---
%[output:0f6850ff]
%   data: {"dataType":"text","outputData":{"text":"\n================ RESULTS SUMMARY ================\n","truncated":false}}
%---
%[output:880efd45]
%   data: {"dataType":"text","outputData":{"text":"Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n","truncated":false}}
%---
%[output:8c27b297]
%   data: {"dataType":"text","outputData":{"text":"DQN Agent     |        128.25 |       31.42 |               500.98 (Steps: 145.0)\n","truncated":false}}
%---
%[output:246142fb]
%   data: {"dataType":"text","outputData":{"text":"Brute Force   |         41.59 |       14.24 |               105.97\n","truncated":false}}
%---
%[output:0a586031]
%   data: {"dataType":"text","outputData":{"text":"2-D MUSIC     |       1417.04 |     1554.14 |              1976.15\n","truncated":false}}
%---
%[output:7f30ae64]
%   data: {"dataType":"text","outputData":{"text":"2-D ESPRIT    |       1409.64 |     1548.01 |              1974.05\n","truncated":false}}
%---
