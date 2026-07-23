%[text] ## Evaluation of DQN agent
% =========================================================================
% Unified Baseline Benchmark: DQN vs. Brute Force vs. MUSIC vs. ESPRIT
% Runs on the exact same randomized episodic deployments.
% =========================================================================
clc; clearvars; close all;

fprintf('=== Initializing Random Deployment Environment ===\n'); %[output:99c0c360]
% Add required codebase folders to path
addingPathParentFolderByName('code');

% Load Parameters and Calibration
Parameters;  %[output:27d469b9] %[output:71214bd1] %[output:55089e94] %[output:50587bb3] %[output:830d1b28] %[output:48431b00] %[output:8f1a92c0] %[output:5d7ee21e] %[output:36c19bd5] %[output:997d3fe4] %[output:4b08851a] %[output:1151d92d] %[output:4ee4867a] %[output:41f1396b]
Calibration; %[output:87a42bb6]

% %% Optional visual verification of the calibration sector
% 
% if ~isfield(EnvPars, 'plotCalibrationSector') || ...
%         EnvPars.plotCalibrationSector
% 
%     figure;
%     scatter( ...
%         X_cal, ...
%         Y_cal, ...
%         28, ...
%         'filled');
% 
%     hold on;
% 
%     plot( ...
%         EnvPars.pos_SIM(1), ...
%         EnvPars.pos_SIM(2), ...
%         'kp', ...
%         'MarkerSize', 14, ...
%         'MarkerFaceColor', 'k');
% 
%     % Local-cell boundary
%     rectangle( ...
%         'Position', ...
%         [0, 0, EnvPars.x_max, EnvPars.y_max], ...
%         'LineWidth', 1.5);
% 
%     % Sector boundary lines
%     boundaryAngles = sectorBoresight + ...
%         [-sectorHalfWidth, sectorHalfWidth];
% 
%     boundaryLength = ...
%         sqrt(EnvPars.x_max^2 + EnvPars.y_max^2);
% 
%     for kBoundary = 1:2
% 
%         xBoundary = EnvPars.pos_SIM(1) + ...
%             boundaryLength*cos(boundaryAngles(kBoundary));
% 
%         yBoundary = EnvPars.pos_SIM(2) + ...
%             boundaryLength*sin(boundaryAngles(kBoundary));
% 
%         plot( ...
%             [EnvPars.pos_SIM(1), xBoundary], ...
%             [EnvPars.pos_SIM(2), yBoundary], ...
%             '--', ...
%             'LineWidth', 1.2);
%     end
% 
%     axis equal;
%     grid on;
% 
%     xlim([0, EnvPars.x_max]);
%     ylim([0, EnvPars.y_max]);
% 
%     xlabel('x position [m]');
%     ylabel('y position [m]');
% 
%     title(sprintf( ...
%         'Calibration positions: sector %d, boresight %.0f^\\circ', ...
%         selectedSector, sectorBoresight_deg));
% 
%     legend( ...
%         'Calibration positions', ...
%         'SIM/gNB', ...
%         'Cell boundary', ...
%         'Sector boundary', ...
%         'Location', 'best');
% end

% Load the trained DQN agent
%agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Tx_50_Aligned.mat')
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_ideal.mat');

if isfile(agent_path) %[output:group:169f459d]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:065619c1]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:169f459d]
%%
%[text] ### CDF Precision of DQN Agent for different K factors
% Benchmark Configurations
N_eval = 100;              % Evaluation episodes (random positions)
K_snap = 16;               % Snapshots per point for digital estimators
Refine = true;             % 3-point parabolic peak refinement for MUSIC/DFT


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining the DFT as ideal or with CST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CST SIM1 front-end.
% EnvPars.G      = EnvPars.G_CST;
% EnvPars.U_func = EnvPars.U_func_CST;

fprintf('Running %d joint evaluation episodes...\n', N_eval); %[output:8f349f15]

% Preallocate main arrays to satisfy parfor slicing rules
K_factor = 15:-4:7;
num_K = length(K_factor);

err_dqn_ref = zeros(N_eval, num_K);
err_dqn     = zeros(N_eval, num_K); 
steps_dqn   = zeros(N_eval, num_K); 

parfor k = 1:num_K %[output:group:47ed160a] %[output:35308db2]
    % 1. Create a local copy of the environment parameters for this worker
    EnvPars_local = EnvPars;
    EnvPars_local.mu_K_dB = K_factor(k);
    
    % 2. Temporary 1D arrays to hold inner loop data for this specific K
    temp_err_dqn_ref = zeros(N_eval, 1);
    temp_err_dqn     = zeros(N_eval, 1);
    temp_steps_dqn   = zeros(N_eval, 1);
    
    % 3. Standard sequential loop for the episodes
    for i = 1:N_eval
        % Optional: Print progress (this will print cleanly because it's sequential per worker)
        if mod(i, 50) == 0
            fprintf('Worker K=%d dB: Evaluating episode %d / %d...\n', K_factor(k), i, N_eval);
        end
        
        % -----------------------------------------------------------------
        % RESET ENV & GET TRUE UE POSITION
        % -----------------------------------------------------------------
        [obs, LoggedSignals] = resetFunction_nav_CST_Aligned(EnvPars_local);
        target_idx = LoggedSignals.pos_idx;
        pMU = EnvPars_local.pos_cal(target_idx, :); 
        
        h_episode = LoggedSignals.h(:);
    
        assert(numel(h_episode) == EnvPars_local.N, ...
            ['LoggedSignals.h contains %d coefficients, ' ...
            'but EnvPars_local.N = %d.'], ...
            numel(h_episode), EnvPars_local.N);
    
        assert(all(isfinite(h_episode)), ...
            'The generated channel contains nonfinite values.');
    
        % -----------------------------------------------------------------
        % DQN EVALUATION
        % -----------------------------------------------------------------
        isDone = false;
        step_count = 0; 
        
        while ~isDone
            action_out = getAction(agent, {obs});
            action = action_out{1};
            if iscategorical(action) || iscell(action)
                action = double(action); 
            end
            
            [obs, reward, isDone, LoggedSignals] = stepFunction_nav_CST_Aligned(action, LoggedSignals, EnvPars_local);
            step_count = step_count + 1;
        end
        temp_steps_dqn(i) = step_count;    
        
        % -----------------------------------------------------------------
        % ERROR COMPUTATION & REFINEMENT
        % -----------------------------------------------------------------
        pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x, LoggedSignals.psi_y, EnvPars_local, pMU);
        pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars_local, pos_MU_true);
        temp_err_dqn(i) = norm(pos_est_dqn(1:2) - pos_MU_true(1:2)) * 1e2; 
        
        [psi_x_dqn_ref, psi_y_dqn_ref, ~] = refineAngleParabolic3(LoggedSignals.h, ...
            LoggedSignals.best_t_x, LoggedSignals.best_t_y, EnvPars_local, ...
            struct('m', 10, 'n_avg', 12));
            
        pos_dqn_ref = estimatePosFromAngles(psi_x_dqn_ref, psi_y_dqn_ref, EnvPars_local, pos_MU_true);
        temp_err_dqn_ref(i) = norm(pos_dqn_ref(1:2) - pos_MU_true(1:2)) * 1e2; 
    end
    
    % 4. Assign the sequential results back to the sliced 2D arrays
    err_dqn_ref(:, k) = temp_err_dqn_ref;
    err_dqn(:, k)     = temp_err_dqn;
    steps_dqn(:, k)   = temp_steps_dqn;
end %[output:group:47ed160a]

% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
num_curves = length(K_factor);
leg_labels = cell(num_curves, 1);
colors = lines(num_curves); % Define distinct colors for curves and patches

figure; hold on; box on; grid on; %[output:3a34235e]

% 1. Plot the CDFs
for i = 1:num_curves
    h_dqn(i) = cdfplot(err_dqn_ref(:,i));     %[output:3a34235e]
    set(h_dqn(i), 'LineWidth', 1.5, 'Color', colors(i, :));
    leg_labels{i} = sprintf('DQN K = %d', K_factor(i));
end

% 2. Set limits first so xlim_1 and xlim_2 are accurate for the annotations
xlim([0, 250]); %[output:3a34235e]
yticks([0, 0.2, 0.4, 0.6, 0.8, 1]); %[output:3a34235e]
xl = xlim; 
xlim_1 = xl(1); 
xlim_2 = xl(2);

% 3. Patches and 90% Confidence Markers
confidence = 90; % confidence level
conf_val = confidence / 100;

% Draw the horizontal 90% line once across the whole graph
plot([xlim_1 xlim_2], [conf_val conf_val], '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off'); %[output:3a34235e]
text(5, conf_val + 0.02, strcat('$', num2str(confidence), '\%$ Confidence'), 'Interpreter', 'latex', 'FontSize', font-2); %[output:3a34235e]

% Loop specifically over the first (1) and last (num_curves) columns
cols_to_mark = [1, num_curves];

for c = cols_to_mark
    [F, X] = ecdf(err_dqn_ref(:, c));
    [~, idx] = min(abs(F - conf_val));

    x_val = X(idx);
    curve_color = colors(c, :); % Match patch to curve color

    x_coords = [0, x_val, x_val, 0];
    y_coords = [conf_val, conf_val, 1, 1];

    % Draw patch and vertical line (hidden from legend)
    patch(x_coords, y_coords, curve_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off'); %[output:3a34235e]
    plot([x_val x_val], [0 1], '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');

    % Add the X value text
    text(x_val, 0.6, strcat('$', num2str(x_val, 4), '$ cm'), ...
        'Interpreter', 'latex', 'FontSize', font-2, 'Rotation', 90, ...
        'VerticalAlignment', 'bottom');
end

% 4. Final Formatting
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', font); %[output:3a34235e]
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', font); %[output:3a34235e]
legend(leg_labels, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', font); %[output:3a34235e]
title(''); %[output:3a34235e]
set(gca, 'FontSize', font); %[output:3a34235e]

% Print Summary
% fprintf('\n================ RESULTS SUMMARY ================\n');
% fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n');
% fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_dqn), median(err_dqn), prctile(err_dqn, 95), mean(steps_dqn));
% fprintf('Brute Force   | %13.2f | %11.2f | %20.2f\n', mean(err_bf), median(err_bf), prctile(err_bf, 95));
% fprintf('2-D MUSIC     | %13.2f | %11.2f | %20.2f\n', mean(err_music), median(err_music), prctile(err_music, 95));
% fprintf('2-D ESPRIT    | %13.2f | %11.2f | %20.2f\n', mean(err_esprit), median(err_esprit), prctile(err_esprit, 95));

%%
%[text] ### Angle precision
% Preallocate error arrays (in cm)
ang_err_dqn_los    = zeros(N_eval, 1);
ang_err_bf_los     = zeros(N_eval, 1);
steps_dqn_los  = zeros(N_eval, 1);
ang_err_dqn_ref_los    = zeros(N_eval, 1);

ang_err_music  = nan(N_eval, 1);
ang_err_esprit = nan(N_eval, 1);

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
Ux_grid = EnvPars.lambda / (2*pi*EnvPars.d_x) * PsiX_grid;
Uy_grid = EnvPars.lambda / (2*pi*EnvPars.d_y) * PsiY_grid;

% Search only physically visible directions.
if isfield(EnvPars, 'positionFoV_deg') && ~isempty(EnvPars.positionFoV_deg)
    visibleRadiusSquared = sin(deg2rad(EnvPars.positionFoV_deg))^2;
else
    visibleRadiusSquared = 1 - 1e-8;
end

visibleGrid = Ux_grid.^2 + Uy_grid.^2 <= visibleRadiusSquared;

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
    ax_q = exp(1i*psi_x_visible(q) * n_x_element);
    ay_q = exp(1i*psi_y_visible(q) * n_y_element);
    A_music_ideal(:,q) = kron(ay_q, ax_q);
end

% Normalize the steering vectors.
A_music_ideal = A_music_ideal ./ max(vecnorm(A_music_ideal, 2, 1), eps);

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

A_music = A_music ./ max(vecnorm(A_music, 2, 1), eps);

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
   
EnvPars.channelModel = 'rician_los';
for i = 1:N_eval %[output:group:1b3d23e9]
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval); %[output:66a706c6]
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
    pos_est_dqn_los = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pos_MU_true);
    err_dqn_los(i) = norm(pos_est_dqn_los(1:2) - pos_MU_true(1:2))*1e2; % cm
    
    %Refinement
        [psi_x_dqn_ref, psi_y_dqn_ref, dbg] = refineAngleParabolic3(LoggedSignals.h, ...
        LoggedSignals.best_t_x, LoggedSignals.best_t_y, EnvPars, ...
        struct('m',10,'n_avg',12));
    pos_dqn_ref_los = estimatePosFromAngles(psi_x_dqn_ref, psi_y_dqn_ref, EnvPars, pos_MU_true);
    err_dqn_ref_los(i) = norm(pos_dqn_ref_los(1:2) - pos_MU_true(1:2))*1e2; % cm

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
    pos_est_bf_los = estimatePosFromAngles(best_psi_x_est, best_psi_y_est, EnvPars, pos_MU_true);
    err_bf_los(i) = norm(pos_est_bf_los(1:2) - pos_MU_true(1:2))*1e2; % cm
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
    symbols = exp(1i * 2*pi * rand(1, K_snap));

    % The same element-space data matrix is passed to MUSIC and ESPRIT.
    digitalNoise = (randn(EnvPars.N, K_snap) + 1i*randn(EnvPars.N, K_snap)) / sqrt(2);

    X_digital = sqrt(snrLinear) * h_digital * symbols + digitalNoise;

    %% ================================================================
    % 2-D MUSIC
    % ================================================================
    R_music = (X_digital * X_digital') / K_snap;

    % Enforce Hermitian symmetry against numerical roundoff.
    R_music = 0.5 * (R_music + R_music');

    [eigenvectorsMusic, eigenvaluesMusic] = eig(R_music, 'vector');
    [~, musicOrder] = sort(real(eigenvaluesMusic), 'descend');

    % One transmitted MU signal:
    % first eigenvector = signal subspace;
    % remaining eigenvectors = noise subspace.
    E_noise = eigenvectorsMusic(:, musicOrder(2:end));
    musicDenominator = sum(abs(E_noise' * A_music).^2, 1);

    P_music_visible = 1 ./ max(musicDenominator, 1e-18);

    % Place the spectrum back onto the complete rectangular grid so that
    % your existing parabolic peak-refinement helper can be reused.
    [psi_x_music_signed, psi_y_music_signed] = pick_peak_local( ...
        P_music_visible, PsiX_grid, PsiY_grid, visibleGrid, Refine);

    psi_x_music(i) = mod(psi_x_music_signed, 2*pi);
    psi_y_music(i) = mod(psi_y_music_signed, 2*pi);

    pos_est_music = estimatePosFromAngles(psi_x_music(i),psi_y_music(i),EnvPars,pMU);
    err_music(i) = 100 * norm(pos_est_music(1:2) - pMU(1:2));

    %% ================================================================
    % 2-D ESPRIT
    % ================================================================
    % Static, element-dependent CST gains break exact shift invariance.
    % A calibrated ESPRIT receiver removes them before forming its
    % covariance matrix.
    if useCSTDigitalFrontEnd
        X_esprit = X_digital ./ c_digital;
    else
        X_esprit = X_digital;
    end

    R_esprit = (X_esprit * X_esprit') / K_snap;
    R_esprit = 0.5 * (R_esprit + R_esprit');

    [eigenvectorsEsprit, eigenvaluesEsprit] = eig(R_esprit, 'vector');
    [~, espritOrder] = sort(real(eigenvaluesEsprit), 'descend');

    % Principal signal-subspace vector.
    e_signal = eigenvectorsEsprit(:, espritOrder(1));

    % x-shift selection:
    % every element except the final x element of each row.
    J1x = find(EnvPars.n_x(:) < EnvPars.N_x);
    J2x = J1x + 1;

    % y-shift selection:
    % every element except the final y row.
    J1y = find(EnvPars.n_y(:) < EnvPars.N_y);
    J2y = J1y + EnvPars.N_x;

    assert(all(J2x <= EnvPars.N), 'Invalid ESPRIT x-shift indices.');
    assert(all(J2y <= EnvPars.N), 'Invalid ESPRIT y-shift indices.');

    % Least-squares rotational invariance factors.
    phi_x_esprit = (e_signal(J1x)' * e_signal(J2x)) / max(e_signal(J1x)' * e_signal(J1x), eps);
    phi_y_esprit = (e_signal(J1y)' * e_signal(J2y)) / max(e_signal(J1y)' * e_signal(J1y), eps);

    % generateChannel uses exp(+j psi n), so the phase of each rotational
    % factor directly gives the corresponding electrical angle.
    psi_x_esprit(i) = mod(angle(phi_x_esprit), 2*pi);
    psi_y_esprit(i) = mod(angle(phi_y_esprit), 2*pi);

    pos_est_esprit = estimatePosFromAngles(psi_x_esprit(i),psi_y_esprit(i),EnvPars,pMU);
    err_esprit(i) = 100 * norm(pos_est_esprit(1:2) - pMU(1:2));
end %[output:group:1b3d23e9]

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
% end

% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
%plot limits
xlim_1=0;
xlim_2=150;

figure; hold on; box on; grid on; %[output:5b2f7b92]
h_dqn = cdfplot(err_dqn_los);    set(h_dqn, 'LineWidth', 1.5); %[output:5b2f7b92]
% h_dqn = cdfplot(err_dqn);    set(h_dqn, 'LineWidth', 1.5);
h_bf  = cdfplot(err_bf_los);     set(h_bf, 'LineWidth', 1.5, 'LineStyle', '--'); %[output:5b2f7b92]
h_mus = cdfplot(err_music);  set(h_mus, 'LineWidth', 1.5); %[output:5b2f7b92]
h_esp = cdfplot(err_esprit); set(h_esp, 'LineWidth', 1.5); %[output:5b2f7b92]

%patch
confidence=90;%confidence level
[F,X]=ecdf(err_dqn_los);
[~,idx]=min(abs(F-confidence/100)) %[output:0ee99c48]
x_coords = [0  X(idx) X(idx) 0];
y_coords = [confidence/100 confidence/100 1  1];
patch(x_coords, y_coords, 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); %[output:5b2f7b92]
plot([xlim_1 xlim_2],[confidence/100 confidence/100],'--','Color',[0.4 0.4 0.4]); %[output:5b2f7b92]
plot([X(idx) X(idx)],[0 1],'--','Color',[0.4 0.4 0.4]); %[output:5b2f7b92]
text(5,0.9,strcat('$',num2str(confidence),'\%$ Confidence'),'Interpreter','latex','FontSize',font-2); %[output:5b2f7b92]
text(X(idx),0.6,strcat('$',num2str(X(idx),4),'$ cm'),'Interpreter','latex','FontSize',font-2,'Rotation',90); %[output:5b2f7b92]
set(gca,'FontSize',font); %[output:5b2f7b92]

xlim([0, 50]); %[output:5b2f7b92]
yticks([0,0.2,0.4,0.6,0.8,1]) %[output:5b2f7b92]
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', font); %[output:5b2f7b92]
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', font); %[output:5b2f7b92]
legend({'DQN Agent', 'Brute Force', '2D MUSIC', '2D ESPRIT'}, ... %[output:5b2f7b92]
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', font); %[output:5b2f7b92]
title(''); %[output:5b2f7b92]

% Print Summary
fprintf('\n================ RESULTS SUMMARY ================\n'); %[output:4fafaeca]
fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n'); %[output:604fe58a]
fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_dqn_ref_los), median(err_dqn_ref_los), prctile(err_dqn_ref_los, 95), mean(steps_dqn)); %[output:943faa00]
fprintf('Brute Force   | %13.2f | %11.2f | %20.2f\n', mean(err_bf_los), median(err_bf_los), prctile(err_bf_los, 95)); %[output:445abb86]
fprintf('2-D MUSIC     | %13.2f | %11.2f | %20.2f\n', mean(err_music), median(err_music), prctile(err_music, 95)); %[output:0755fce3]
fprintf('2-D ESPRIT    | %13.2f | %11.2f | %20.2f\n', mean(err_esprit), median(err_esprit), prctile(err_esprit, 95)); %[output:9765fc69]
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
%   data: {"layout":"onright"}
%---
%[output:99c0c360]
%   data: {"dataType":"text","outputData":{"text":"=== Initializing Random Deployment Environment ===\n","truncated":false}}
%---
%[output:27d469b9]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:71214bd1]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:55089e94]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"4"}}
%---
%[output:50587bb3]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:830d1b28]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"14"}}
%---
%[output:48431b00]
%   data: {"dataType":"textualVariable","outputData":{"name":"M","value":"196"}}
%---
%[output:8f1a92c0]
%   data: {"dataType":"textualVariable","outputData":{"name":"L","value":"12"}}
%---
%[output:5d7ee21e]
%   data: {"dataType":"textualVariable","outputData":{"name":"zeta","value":"0.9800"}}
%---
%[output:36c19bd5]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:997d3fe4]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:4b08851a]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"60"}}
%---
%[output:1151d92d]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"34.6606"}}
%---
%[output:4ee4867a]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     x_max: 20\n                     y_max: 20\n              calSpacing_m: 0.5000\n      sectorBoresights_deg: [0 120 240]\n            selectedSector: 1\n       sectorHalfWidth_deg: 60\n                 MU_margin: 0\n                     N_cal: 400\n              channelModel: 'rician_los_nlos'\n                    fc_GHz: 28\n                    V_hall: 72000\n                    S_hall: 18000\n                   mu_lgDS: -7.2781\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 15\n                sigma_K_dB: 8\n                        KR: 1.6951\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 1\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         M: 196\n                       M_x: 14\n                       M_y: 14\n                         T: 3600\n                       T_x: 60\n                       T_y: 60\n                    SNR_dB: 34.6606\n                 theta_min: 1.7657\n                 theta_max: 4.5175\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 24\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [10 10 4]\n                    pos_MU: [45.7145 1.4865 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 … ] (1×3600 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 … ] (1×3600 double)\n                      h_MU: 1.5000\n                    L_hall: 120\n                    W_hall: 60\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 180\n                 tolerance: 0.0262\n         StopTrainingValue: 58\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9800\n"}}
%---
%[output:41f1396b]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 2.366%n","truncated":false}}
%---
%[output:87a42bb6]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nFull local grid: 40 x 40 = 1600 candidate positions\nSelected sector 1: boresight = 0.0 deg, angular interval = [-60.0, 60.0] deg\nCalibration positions retained in sector: 570\nCalibration complete.\n\n","truncated":false}}
%---
%[output:065619c1]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:8f349f15]
%   data: {"dataType":"text","outputData":{"text":"Running 100 joint evaluation episodes...\n","truncated":false}}
%---
%[output:35308db2]
%   data: {"dataType":"text","outputData":{"text":"Worker K=15 dB: Evaluating episode 50 \/ 100...\nWorker K=11 dB: Evaluating episode 50 \/ 100...\nWorker K=7 dB: Evaluating episode 50 \/ 100...\nWorker K=15 dB: Evaluating episode 100 \/ 100...\nWorker K=11 dB: Evaluating episode 100 \/ 100...\nWorker K=7 dB: Evaluating episode 100 \/ 100...\n","truncated":false}}
%---
%[output:3a34235e]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAQAElEQVR4AeydCZxP5ffHz2UwdmNfM5YoS1GJaKF9T5FoQ4v8077RavRTUlKKSqWIUqSUNmUtZEkolMpO9n0Zy+A\/72fc6c53vvsy813OvObc+9xnfz733u9z7nnOOU+BY\/qnCCgCioAioAgoAopAjCFQQPRPEVAEFAFFQBFQBAJEQLPnNwLKwOT3HdD2FQFFQBFQBBQBRSBgBJSBCRgyLaAIKAKKQP4joD1QBBIdAWVgEv0J0PErAoqAIqAIKAIxiIAyMDF407TLikD+I6A9UAQUAUUgfxFQBiZ\/8dfWFQFFQBFQBBQBRSAIBJSBCQI0LZL\/CGgPFAFFQBFQBBIbAWVgEvv+6+gVAUVAEVAEFIGYREAZmKBumxZSBBQBRUARUAQUgfxEQBmY\/ERf21YEFAFFQBFQBBIJgTCOVRmYMIKpVSkCioAioAgoAopA3iCgDEze4KytKAKKgCKgCOQ\/AtqDOEJAGZg4upk6FEVAEVAEFAFFIFEQUAbm+J3+7LPPpE6dOjJp0qTjMXpSBBQBRSDMCGh1ioAiEDYElIHJhHLhwoXSr18\/OXbsWOaV\/isCioAioAgoAopAtCOQ8AzM33\/\/LQ8\/\/LBs27Yt2u+V9k8RCBUBLa8IKAKKQNwgkLAMDNIWlos6deokK1eujJsbqgNRBBQBRUARUAQSAYGEZGDWr18vPXr0kLvuuku2b98uNWvWlBIlSiTC\/c7fMWrrioAioAgoAopAmBBISAbmlVdeke+++06KFSsmvXr1khEjRkhKSkqYINVqFAFFQBFQBBQBRSDSCCQSA5ONZdmyZQ3jMnPmTOnWrZskJydnp2lAEVAEFAFFQBFQBKIfgYRkYJ544gnDuJQqVSr675D2UBFQBBQBRUARyFcEorPxhGRgovNWaK8UAUVAEVAEFAFFwF8ElIHxFyk\/861bt06UFAN9BvQZ0GcgfM+AYhkaln5OXzGXTRmYMN4yXrInnuwll1x6oZJiEPZn4NJL2silF52rpBjoM5Agz8C5554r4SDchTA\/hXG6i4qqlIEJ423gAfnttwXyaK8e0u\/Fp5XCgMHt3TpJ9ZplFdNMLF8a2EdGvt03JOp5X0c5ocIxebnvfSHVE2o\/Yqu8Z8wVT8\/YBHuPFdP\/MB378Qj56KOPQqL7779f5syZE8aZLnqqUgYmAveibt26ohQ+DLhF7FOV6JjWqVNXGjduHBI1atQIOIVzqHVp+cYGRwBNJDy\/WV9Cnp9zLGL05b6Gsq\/FffLuygoRayOS\/Q9n3fdNzpDuk455pHe+WijWRw9K1Q86eaRrl70qoxqv5TGNO1IGJu5uqQ4o3hDQ8SgC0YLAxl0HZeLSbbJp96GI0rHkMhGtP9L9D1f9a7YfEG9UYcNsgTK2rBJvVLlwRrQ8QmHthzIwYYVTKws3AqVLl5ZzzjlHOIe77mitb92qFQJFon\/lypWTK6+8UjhHov54r5MJ\/LslW2XEz\/8a+nZ5hhxKPU++W3HEXNvxeXn+esbvsmLKONn9w9sRp2JzR8i1e78y9FbNnyUS9GzRL6TDP30jUnck+htqnY8UniiQu3q+qjtJptaf6JHuq7HavHIlW3eRE95Y6Zb+vfVjuen3GiZfvB2UgYm3Oxr28eRvhTAuMDD524u8a\/3okSOyYO4s2blze65Gv\/xomPz000+54gOJgHGBgQmkjOb9DwGkDy9OXJXNrMC4wMDkJcPi2tbXmQxMkYn9ZM+kt\/OErtv3lUDFM5mZSFCtVd\/KVSXXSSTqjrY6k+eMkFmTJsve+d\/kGu+vcxfJh79skJKT+nukA0ummYczqUJNSaqQ6oFqyqZDSSZfvB2UgYm3O6rjiVoE9uzaKZ9\/+L70feweeeTOTvLIHR1z0WN33STTv\/8qaseQqB1D8mITGJxavaR0Pquq3NSsolzXqKQ5cx0uuqm+iL9080n0SKRgShUpeWG3mKdibW4Tad5JOMfyeA417yxv7G0pnebWkfN\/qCptfqiWiy6cVE3GrCkpRZtckuu+Fa59umFIUq7vLd6oQo\/3JaVDWtZDkGDHAtE+Xu2fIhAPCBxIT5cP33ldZk6dKDu3bxU5diwehpUQY2DJ6MZhvwuEBIZBX9KwnGFgbj6zkmFgwsW4UE\/7QvPl0h+7+001fx5Il6RI5oRX6qJuEutUrM3tYrXoZCb0WB1LgZY3yzPfbZSxP6+WjbsOyDGxzD3ydCjW5NJc9437SX6YE2\/E8hH5EpGUgUnEu65jznMEtm3ZJBvWrZHa9U6Wx194TQa8+7FbeuGtUdLivAvzvH\/aoGcEUMgktVKpwmJT5VJFiIoIHVox39SLRCUQKnb6VaacHvIfgbXrNspf\/6yWZqc1lCkT3pa\/fv3CLS2ZM05uaHdJpDoc9\/UqAxP3t1gHGA0IFClaVAoVLiKtzr9EypWv6LFLSUlJcuEVbaX6Cake82hC3iGAvgkSGFq8tGF5GX3HKYaa1ChJlE\/a98sE2TL0roDo4HEGhuWTyr0miL9UpM7pPvujGfIGgRLFi0lykSJyS8crpHq1Sh4bLVQoSe6+o4M0OKm2xzya4BkBZWAysalYsaJMnz5dVqxYIRdeqF+\/mZDof5gRSEkpJ\/UbniJ7du8UX397d+8WyFc+TY8sAht3HTTKurYEBr2XQFvcP\/8rQaISCB3ZscE0k5RS1Zxj9pDAHa9SuYKc3bKpbN3m+33fvn2XQAkMV9BDVwYmaOjis+DuzMlz2LBh8s0338jRo0fjbpB79+6VefPmyaeffipjxoyRmTNnyq5duwIa57Fjx2TLli2yc+dOv8sVNJKV62TRvNmy6p9lHssdOnhA5s2cnln3do95NMF\/BDK2\/yv7MqUgwRDMxznpPws6KX1r\/yUnbpqWq66DC7+VY0snC3ndtWH3tMz1vaV8t6EBkUpUbPRi75yUVNBIVr75fob8uugPjwNIP3BQPv1ikmzcvM1jHk3wjIAyMJ6xydOUjRs3CozD0KFDZcSIEdK3b1\/56quvJD093W0\/iCf9pZdeklGjRgnnd99910ys7gocOnTImOA+99xz8tprr5n8q1evzpF1\/fr18tZbbwnnZs2aSYECgT0eTOyrVq2SDz74QJ588kl58MEHDT3yyCPSr18\/Mx6kXIMHD5bFixfnaDvSFxkZGaZ92q5SpYpcffXVQn9hZF599VXZvHmzzy7Q97Fjx0rv3r3l+eefN5t2+izkyJBUKElSypWXwS\/0zmV9ZFskPdGji1H0dRTTYHAImFIsx+wc20eCoSIT+0m33SPkmnXvCIqy7urY81lfkR8GCWd36Uhe6EhSpjQFhiQQopxS7CJQuHAhqZopienYtZfUO+0at3Rqyw7y4ZhvYneQ+dzzAvncvjafiQASgVdeeUVOOeUUueuuu6Rz585yyy23yNy5c2XQoEGZ4sWcX+NIEWB2fv31V7n11lvl5ptvlnvvvTezJpHXX39dVrswJgcOHJDRo0fL559\/LmeddZbJy+ZexMH0IIngPGHCBNm2bZtccMEFUqFCBVOfvwf6BOMCc\/T333\/LVVddZZgWxvXCCy8Y52m\/\/\/676d\/y5cv9rTZs+dgLBGlLu3btpHr16lK4cGG55ppr5PTTTzeSJn+kTbVr15YzzzwzqD5hhfTxe2\/Ir7NnBFVeCwWHgL0cg0lqsdOvlEBo94kXy4\/JZ8lfFc\/zWK5wk8tETj7fYzrtocsC4xLcCLRULCKwd99+6fnMq\/Llt9Njsfsx02dlYPL5ViF5QZKCg7H69etn96ZmzZrSsmVL2bRpk2E8kKCQeOTIEfn2228FJoDJtFKlLAWx5ORkw3gcPnzYLI2wFER+iIl74cKFQv1nn322WJYl5cuXlxNOOEGuv\/566dChg3Tp0sV4u61Vq5aZ1CnnL8EgIQWiDfSJkLy0aNFC6BN1oJjKvjkwWfSBOMbFOS+I\/v3yyy9SvHhxoX9y\/I\/+wfz16dNHKleufDzW+6lMmTKG+fGeS0RcMmCFtG71SqlSvaY88NTz8uLQD9UKyQUjb5c7xqTJ+p5nBEw4d6PejzdVkXbrrgyIeuy+Tt4p3Vm+rNHN+NlwZ8pa6rqnxLr4ASmZeXaXThymwPRBKXEQwAppyR\/Lpf6JqfL5hwPlj3mfqxVSBG5\/gQjUqVUGgACTPswGjAgTrLNovXr1zGT5zz\/\/mGUd0ljqQJKBBIF04myiDmjDhg2ydOlSE71\/\/34hv2VZwrIQ5UhgSQVmp2DBglzKn3\/+KX\/88YdRYk7OZIZMpB8HGCokN8uWLTMMS\/v27aVs2bJuS5YoUcIwSxUrerbCcVswxEh0VZAshVhNWIpfeOW1Uj21thQomIW7a6Uwe42aniGFCxV2TUroa3RMggVgS4Fy8kfhesEWlyZ+WhwF3YAWjFsEetzZQRqeXEcKFnQ\/1WKFdFGbzI+9Ivq+B\/MQuEc1mJq0TMAIwECsW7fOlCtUqFDmQ55zUktJSREmfSQIMDFkhDHZs2ePIAlAikKcTcWKFRP0O44dO2Z0TKh\/+\/btsnXrVlNP1ar\/WTXs2LFDihQpIkWLFhWWfyZNmiRNmzaVOnXq2NX5dV6zZo2wlEXmBg0aSK1MCQ5hTwRzw1KZp\/RIxMPEHTx40G3VLB3BQIKx2wxhiiydUlZKlSkrMI6+qjypUROp1\/CUXNmuvvF2SaRtFQDg4PL5gvSFMDSqzVixacDpH8otld6SSW2\/lGr9f\/FITfpNlDd63ShTHjojKMK5HG0rKQL+IlCpYnmpWKGcHDx02GeRc1ueJq1aNMmV74qLzpTnnn0yV7xG\/IeAMjD\/YZHnIaQX9sTKRIpSqbMTBQoUMMs9xGH1QjrSFa5LlSplpDOEnWQzNSxNMXHzRU89hQsXNswKealnyZIlggSHtNmzZwtLVK1bt85uj3z+0IIFC8Se\/E8++eRcTJi7Oi6++GJp1apVdhL9WbVqlbz33nuCLs6IESPkmWeeMQrFK1asENLJzBmGbMqUKYJeDTo8pKN3g6Lwo48+ahR1GQv5YUxQ2h0+fLgZH5IYdIpQkIZQNH744YeNUi7LbJRxEvWw99DLL78sH330kVDX+PHjTV3OfHaYvn388cfSq1cvo7xMf7imHyVKlpKmzVvK7\/PnyKxZMwXFYRSm165da8ZMXsYwcuRIWfjLbPlryW92teaM0vaXX34hd955p9x+++3SsWNHo0+ERM5kOH7Yt2+fUQJHNwpF5WuvvVZYIvv333+P54it046xacbCh15vKVDO7ISMN1xo0bo9RCspAlGHQNmUUnLVpefK9J9+yfxoOeK1fz\/O+lVmzl7oNY8mukeggPvoEGO1uF8IIHVBwkJmpCRMUoQ9EcyObfJb0MMShF0W5oWJE4YGqQqTMXGks9xDe40aNTJLU0ze+L9B4kO6vwTjYjNULDtV8FPxl3GTn3ZgSqZNm2aYFRRq77jjDqPE\/MADDwgMxzvvvGOUmcnL5MyEjUIuOjQsjWHuffnllws6gyXpfwAAEABJREFUPEiTYG5QiiY\/TN4999xjlq24Rmp1\/\/33y1NPPWXo8ccfN3pApLkSY4OhgtFDsfrGG2+Uu+++W5BiIQFzzY\/iNArUSMBgjvr37y9t2rQR+vrGG28YRewzWp4nezOlZz9N+cEoWiO9+uSTT8zSXo8ePQQ9qF\/nz5cfvp4ghw4fym4CKd3AgQMlKamQYaJQ4EbJ+4cffhAYH+ohM9jAPHGvYY5gtuj7okWLDFP1119\/kc0j4fdk4do9Eglal8mgIk0JlOzOflb8Svn6lN7y2CWpuUglJDZKeo4mBDpdf5ns239AJk2b47FbR44clR+mzJYDB\/973z1m1oRcCCgDkwuSvIuACUG51bIsYZKyJyK7B+6kMkz4pDM520wA1zahA0OYfJRPSkoSLG+QtiDZYFKGgUFXBenL5MmTjVUOzAzlAqGDmcsyLEVRxrIsoT4J8I++wISw9IQExy7OUhN9hNmBwWHyhdk76aSTBAVn8p1wwgkCw1O3bl2h\/02aNDHSGpbbkG6RxxuBDZIp1zxgh6I09SCVol3yMD6W2WCEuLYJZgdmoUaNGkayZNeLkjV5Yba+++ZrGfHGy7Jt80bZsupPKbhngxzavEI2\/vmLjHy9nwzu20u2LV8kBfdulE1rl8uh4z9oMKFImmBEYTJZ9pPMP8ZL3eCPtI3xYgXGeG644QZhOZH+orRdu3ZtwwzCLMHcZBbP9Y+32RuH\/S4PjV0Wdnr+41livdletr59V8BkWxH9VPQs2VqwnOAN15VyDUYjFIF8RmD9hs1ye480+X3p33LfY\/3dmlDXO+0aObnZtfLJZxNz9FYv\/EdAGRj\/sYpITvRGYC6YgPBJsmrVKjMJsxzx9ddfCxMUDdvSDcvyvikYeV2JCRhrG77Ob7vtNmM+DPPz22+\/Ce1dcsklZjkK6c53331nvvJZymF5xrWucF7DKLAEhV4IkyyTr7N+pB0VK1YUsEGqwtmZ7srElS5d2iSj04P+j7kI4gD2+KlBIkIfnFXAUMF4OuOQvrAUhGXYCy+8IEhgICQvtrRm+85dUrJUGdmze6ezqMfwkSMZJg3GjeUfGBYnPjBvaWlpZnno1FNPNZI07ieMMMtiLDVBSLJWrlxp6qKPSLHMhcvB9jbLXj94nA0nXV9ovmmNfX0wZw6UVpzUSarVTBU2UDQV6UERiHIEKlUoJ1Uqlxd\/PPFG+VCiunvKwOTz7YGRYPnjsssuMxM1OhqPPfaY0bngq5uveSZMJixnV1kOcjdJw4SQz7Is4QtcPPzBGKG4iy5KtWrVBEkIuh5YRbFcgo4F+h\/erHeQBiAFoIkjR46Iu\/6Q5omYTJlwSXfXV6QILMmQzvKIryU28oWDkJiADwySk2nwVDdjYPznn3++WZqyl6jQ4xkwYIDgC4elLBRzz7v4Srm04x1ypGQVqdX0HOk7eLgMePdjQ10eeEoySlSWQsXLZDcFA8MFzNmXHw0zzgi5tixLYPqQCHEfNm7caKQs+Pl5J3PZzSYY0XHjxsmXX34pb7\/9thwpXEqem7JVHv1suTw4Zlk2TVyylWqNhOOVDvUlnHRh1SzxeLHTr5IKdw0NmM7p+rDpD5IX08koPFhW4B8WUTgMEYmeXllW7GKalFRQWpzRWG67pa1HE+q\/fv1CFs0aI22vbOMW9K9\/mCuPP\/J\/sntmp5CowZFBMvSeY27biPXIArE+gHjof3JysqDYiodXJju86qITwYQNU8DSBMsmTFRMZIyZ5RsmTcJOOnDggLksWrSoIHkxFy4HJB8\/\/\/yzkbrgr4UJGyVV2mvbtq0xg6YsjA1pLsWzL+kPUgoiWJpAckHYX6L\/jIP8LHdxdiW7fvKR3zU9Etfo9YBRoHWjV+StTL2TG0uteid5VZS2LEuKlConJUunCPcSRoo6\/R0794D7SBl3NGnZTvlj8yFZtG5PDtq4O4vJQPLirlwocfYykDpzCwVFLRtrCLRsfqo0O62hFCxYwGPXiyYXkc6drpLKFct5zJOxbbaESqfX9Vh9TCd4Rjamh5W\/nWcJwx9y10u7HMs3LK8ULFjQKHmyzILEwl7SYGKDaXCtw55E0SFhErbrc57x94LlEToVMCqzZs2S3bt3C7okLNnYeamfNu1r17NlWcISGH2kH9TLROuaz9M1yzG0T1kYJXf9pQ+kw5ARpi6YOuI8kWsf7AkdJsk5HnflqR9miTQwhoizifLUQ7orgT1Mh53XeSYtqUiyVK9ZW\/bs2mWWCZ3lFy+YJ5vWrzFRBQoVkVIp5cw2Eja2PA+S+bdszWaZvnidIaQmNs1fvTszVWTl2n9l8q\/\/mHQ7n32evHClbN+2VZKPpUuLakky8Pr6uahWqaNmbygkeb7INJh5OLLjX9k\/f0I2bf51ijhpb4FiAu3KXEZz1plZNNe\/M91XOFfh4xG+ytnpx7PnONlp\/pxzFHRc+FOWPI4i2UHi\/aXsQi6BUMr7W5Z8Ls1mX5LmD2UXcAQoZ783KPFz7YkcxXIEPeV3jc9R6PiFax5v18eL5DrZZcqULi6NG9SS5StWy9at23K8V198NVnm\/brE\/A40PLmO8RVDRXZZzlzvSC8q0C45UQ7VH5yDSrYcLe7INd\/KEr3krsGxK80CB0+kDIwnZIKMP3z4iHz7zST5cNQYn1SmVAVxJcp9MHy0DBn8tqxft1mSChaVRQuXZte1aMESSd+XIWtWb5DDByVH+cKFisu6tZtk96502bZlt4wdMz67HPVCI4Z\/JO+8PVwOpB+R5meeLUkFispfy1aaMk1OaZajzIwf58iMH2fnqoN6IPre7IyWckKNOqb83Nm\/ysZ\/t3rMTxmIPrz37gdSpdIJUr5sFVN2+d9rRI4VylV2xk9zTDrj+WzcBJO+ZPFfJu7I4QI5xl9Aipj4Ff+slU8+\/szkpb3Jk3408Rv\/3SbjPv0yO75UyXJy6MBRk0ZZxkP+3xb9Ibt3psuqlevlw5E57yPlqQeMixQqYdpPrXmi7Nl9QP5c+k8m7ruy66cuiPvZv99AGTLoDXn8\/gfk67FjZO\/WHaZt+kC7zZq3kT8XL5PdW7bKv5n3nXbGfPK5eQZoa9IP02XT9gOy6u8V8sOnow31+mqrPPHhfOn92scybFEB2Xm4uKzOZHDGjxxu0u18nCeO+VDefbm\/\/P3Dx3LantlS4I8fpGmdE3LRS6+8Lf6SVThFoIN\/\/ybbR\/bOpte+nC9Oend9VRlZ5CoT56ybsq7kTPcVdi1rX\/sqZ6fb+Z1nO82fc1a5MnKsYCmDQ9Z1SsD42eU4+9OunYf87shO93UOpSx1uytPHGn+EHldiXKvvP6efPDRF8KZa0\/kWta+9pTfNd7O7zy75vF27SznDNtlej09QC64qrtc1v4+ufWuJ3M8F7Nmz5Ovvp8tDz35muzcXyD7+bHLcp7240wzC7045Rx54eta0nvglzmoUPkW4o5c8739ya9yoEhDU1e8HQrE24Dyezzp+w\/JNVfeID26P+yTqlauLa7U5ZbucmC\/yF9\/rJHTTm0pffsMkHvvfjS7rgfvf0IaNThD\/l27XebPW5Kj\/K7tB2Xe7N8luVAZuefuR7LLOPvSoP5pQr6HMuupWb1+5gRcWVav2CT7dh+R05q0zC7Tru1NUrpkJbnbyzjoe91ajeT\/uj0oRw8XlpXLN8oLzw+SKy9rn12Ps23C1Lv8r\/VSo1o90\/dON3SV\/XuOysJf\/5Q1KzfnKHfrTd3k4H7LpN\/W5e7stKqVasumf3dlMnAFTR30AypoFTfxxZLLyh1d78nOf0HrK0z8oQMFhDrpB1S5Qqrsy2ybuihbtXJtU+buux6S5MJlZOP6nXI0o7B0u\/0+E0+Zjtd3yWQgj5n6ihctZ9pv0ay1lCpe0Yz\/xRdez3H\/7ftZO7WBHDpUWNZuPixb9xaWjVsPmbbpA32vldpQLr\/6Ftm45bBs3rhHaIf2bu50h+zZedjco027kmRjoZMk+YzbDTUqXVAKLp4kDc++Xpo3O08OlzxJNu6wZO6STCa24nkmD3kLN+0sf+2tIJsyKknZs\/9PMk66QW7t1ksKFquei267+ynxl+zyR3buk6OZTHXh1LOleLOb5foG1XNRp\/POzlWvXd559rdt8jnLOcOk+UPOMnbYn3J2HsoUKFpdjhauLAWKVsvG0k73daa8K\/kq40x3LWtfO\/N4C9v5nWdv+V3TnOWcYdd8nq6dZewweTt1fVguu\/Z284xy7YnsMq5nT\/ld413Lce2ax9s1+d0RZW687RHZmV5IDh7KMP63rm1\/c47n\/677estTvfvJwYwC8kTaIDlklTPPD2VtaphaRFKKpss919cUFPNdydNc55oPPUdbguupTKzGF4jVjkdzv+vVq2ecxPk6O8fA8glLMGlpabJw4UJ59dVX5bXXXpNTTjklR10s2eCADXNiXPjbFiaIXfH9QT0oAWM+69o+SrFs6MieRCic0j46NSeeeKLRyyhQoIBpi2uWr9q3by9NmzY1ca51cU15CCsYHLZxpj8PPfSQYPpLWq1atQQ\/NCjDjh07VnDW1q1bN3niiSdMm1jX8MJZliX4OilevLhpjz7Mnz9fsJzp0qWLoORMmyiuUhd1s5TDeAlDYMA5OTlZKE9+iDESb1mWoNdDHEQcdXC2yxKPXhAY0s6PP\/4o6CWxhMfSEvcFp4KUefHFF819YnkL\/zLkxxIJfzEo7+IXhrEeOnTI+JChDRRpb7vtNoqbTSQJjJ63Uco+NFVuG76Iy0xG55Bc\/\/oMafHWern9p9KyrdYVchQdvCOHZd3CaTLprSflu8E9ZenoPvLc4\/fLiHsvlKGdm8g7Lz4l6Awd2LdHfnj\/Bfnrm6Gyd+EEmfJOb9m3drGMfvd1SbvlfOlxSQODsWnMccjYskqSnqzvN6243hJox9g+ppaiDc6TCj3el9O7PZ+LTr22m2kTfG0yhVwOdpo\/Z5ei2Zf+lCVPdgFHgHh\/yVEsRzCU8v6WJV+ORh0XpPlDjiLZQX\/K2XmyC7kE7HRfZ5di5pIy7JeG0QJhb2QKuDl4K+NMc1M01zPqzO8atsvvW\/CobP+yVjaV\/\/MSkV+ukV3\/zpeXbz8iU\/rukxuqvSjEOyn9hwZybuUfZPrUSTL5tQamvDO91LG\/TRMnNTnfbb9MopuDaz+5tpfR3WSP6agCMd37OOg8OhZM\/jALWK907txZpk2bJnhRtfUfXIfJJEwZFH+ZIGEKcGzGZD5u3Dgz2buWIQ0fMFg2UbedjiIuFjJMvtOnTzfbCmDBgp7JpZdeamfzeYZR+fTTT+WTTz4xDtxwsnbFFVcIP0ZnnHGG+YJAx2bSpEly3XXXGeaFSi3LErzGfvnll4KiMp5jYbC6du1q9maivrZt25r8YAVTQB2UnTx5srz55puyYcMGGTVqlODFlngYQBShV61aJZ999plhjIgnH8zf3LlzTd0oTcMkkUZZzJ5pg2vGjl8VnOuBy1VXXSU9e\/YU+sWY2iE\/+HQAABAASURBVLVrJziro68lSpQQZ\/4jR44IfaRuxka9MEDcN\/qEwjRt2P0cMek3SVr\/ixRZ+jnRhpL\/\/lasnWsyw5YcrHuxpJ\/ZXaRAQZFjRwV9odTUVIEhhLHLzGT+mzRpIh9++KHAnBYsWFDQcwIjdJ3AkfZNRg+H9CXTPKT4jk7KlGYlN2ztO6PmUATiDIHD22bnGtGedJFq5Y7KGSeKFCpo5Up3Rhw+IrJghTPmv3CBYtXNMtF\/MRpyIlDAeaHhvEegRObkx0Q\/LpPxgJjcifPVE0x8mVBhFPjaR7KCySzctruySBr+7\/\/+T8hTsmTJHFmY+PA5A5ODKTWbPiIVQck2R0YfF0yalEVyNGPGDPn7778F0SXSHMaGgzX67a4a+k3fYC5gUoYPH24kH0hS7Px4Cn7wwQeFjSepd9GiRUaygak1fm64Jp70Pn36SGrmJA+eSFGIh2BUcDCH0zzGaveRskhOaIP2LMsS8sEkkAfJCg73YP5wcgeDhHSKMSNBuefjP2XE8pJS8bpn5eJnv5TLnvtamt47TP4sf6H0nLBWBs7YIb+t2S6zMurL2b0+NukX9P5ctp7UUTYeKSkZ1c6QkZ99axhX2r\/p8nNk23udZfvANrL9lQtk0yePSfWqVQRJD\/2BgQUz8jqJOHAEA8YL9mlpacayzJnPGUbysmVIV9kzbYSJLtm6i9QeeywgOuGNlVJUGRiDnx5iB4Gj+9fJvkwJSihEHYy4zIU\/SdmrVxqqfd0sOVLmXEk6Z7a5tuOd51KX\/y3zdl1NUdlftmOufEXr32\/S9OAZAQcD4zmTpkQOAZgEV4YiEq0x0VaoUMGjaTXWR0gTkEowMVuW96+GSPQxVuvsMfoPgYnxRl8sOyT\/Hi4l344fI6PnrMuRf832A5lDPyazJn0tEydONNKmli1bZsblzf+eqcMzmZfhcuC4BCapQs28aVhbUQTyGYGDaz+VUIkhICmBCEP81iLxRNqKNJs4J\/GxOH78+Hx53539iPWwMjCxfge1\/xFHAAZjxj87xRPZHRjS6WTxSDc1kscfvleq7PhVGi5+RbqnbpC+F6UYItxy1VB5d+CzRv+FJTd0mKgX6QhLO1DW9WohHCwdXjZDZPnsHHVkbFlN1YLkBR2WlA5p5loPikC0I4D04\/DW2RIsHd2\/3gyxSI32UrzpS8FTk5dMPfYBB6RIvL\/44gtBPQDpLR61IcK33HKLWZJGP875vtvl9ewfAsrA+IeT5kpQBGBamvT9Wa5+Y4FHsqHp1KyyeKMeV54uX477RKqVLyWfDHpaBvS4ztDHrz4lSxfOE77KYFwGDx4s9lLW5sylnQ1pbQTK2LLKSEoIB0tb+l4kx4beKJztOvZMG26GgOQFJsZc6EERiAEEkJ7smdVJgiXKM8wCxaoJTEywhDkz9TgJRWQkMCj5I93m3YbQOcT3lrv33Vlew74RUAbGN0aaI4ERWLuD5R2RE8omS6s6ZTxSz0tS\/UKpevXqxkILxWCsmG6++WZh24a0tDRB4XbEiBGC+Nm1MhRkH6i2QS6sXVIIB0tJ9c8Wqd3cbR0l23RxbVavYwOBhO3l4a1zzNiTyrWQUAjGxVQU5kOw7zvdwHDg+eefJ6jkAQFlYDwAo9HRiUD\/iSsFiUheEfotIAHzMqFHU\/FEPS+pRTa\/yLIs4YcN0fKzzz4r\/EhhrYQll2VZRsqy5u5aAtl6KSnX95aqaVNDpopP\/SBW99HC2bU+LIn8GoBmUgSiDAEUXku1Gi3BklN\/JdxDsyzv73u420uk+pSBSaS7HQdj7T9xlaCTkldkQ4YExg5H+rxn2ghhuQiiLRiLQhX9k\/CQP89JG1QE8gmBo+nrTMs4lDMBPSQUAsrAJNTtjt3Bjp63QZC+2CNY+NRZkpcUiITF7qO\/ZxiVPdOGy44xaYYyNq8yRVGoxTwZgokxkXpQBKIcgaP71xnLnvRlr0owdHj5YCmyaYRw9lWetqIcDu1eBBFQBiaC4GrV4UHg390Zcv\/YfzIZmKyJHWmIg4x+SqSvwzMS97WgqIsfFjzaQjA05IRpgQgrKQKxgsCRTAYGvyrpywZlMjCB0+HlQyR58wi\/ysYKJtrPyCAQ9QwMDsbuuOMOwXU+buQbN24sjz76qHGQFgokOPmiHnyeUC8O0y6\/\/HLjuTU9PT2UqrVshBCASel5Sar0DEDfJEJdCbpamJNctDmLMcMCCF0XCOmLq2O4J554Qtg+IujGtaAikIkAUouI0vFlHfRK0E0JlArV6SEHKnYWzv6UxfyZtjKHFlf\/vOu883E1qDAPJmoZmGPHjgku7VF0nDJlivGPwdj37dsneHW95pprBI+oxAVKlKM89ezatcsUx\/073ktxy49H1j179ph4nwfNEFEEfll3QK4akbXOTUMwL5gqE441+jetjVHMRTnXSTA0jCWlQ2\/BBwsEM0OckiIQbgT2LnhUdk46J2KE9IU+FyrXQorWfyBgKlTnHjlYqbPf5YrUaE9zSgmIQNQyMLiix9U7LvAxNWVfGdy5f\/fdd9K8eXOBkUlLS5O\/\/voroNtGfspRHmkOLvSp97fffjOSHXtPIPa5Yf+agCrXzGFHwGnGHKuMC6DgeA6LIpaEvBF5lRSBSCKQcXzvHqQWkaSk8s0jOQytOw4QCHUIUcnAsIQzbNgwI3VhWefpp582jr0syxL2emFzvFNPPVW2bNkiOApCWuMvEGwaSDlMVocMGSLsiWNZlnGx3717d7PXDHV9\/\/33snLlSoJKeYgAjuOuGrJAbLIVd1vVKRPTS0d7p2XtM8SyEEq5BZ+cIT+16itvlekk75a\/RX658GXzPOch1NpUgiHAspEtHWHo7N0TSVLJCCgrRRKBqGRgli5dKr\/++qskJyebnYrZL8gJAl5Kb7vtNrNnDM6\/1q37b4nBmc81fPDgQbMRIPHovuCLg7BNlmXJeeedZ5ilrVu3yurVWS7W7XQ9Rx4BGJaZy3eKTZhL0yr6L5xjgebNmyd403X2FQkM19aZHYT9ps4991yBMWdTxlGjRglLly1atDBpe\/fuFf1TBMKNAMq1\/3merR7u6mOwPu1yrCMQlQzMwoULhR\/x1NRUqVOnjluMUeotV66csLfEP\/\/84zaPt0iWptxJbmCW2PjQW1lNCx8CMChIXWyyl4zYU+jLu5vKmNtOkrevqywPX1AjfI1GuKYyZcoYRplnmKbQcYFYOho1c5mMHDnSbBtQvnx5ueiii4wn3iuuuEIqVapk0nBs524DOOpSSkwEDoew349d1l46wmNtqZajExPIKBz14sWLBYrCrkV9l6KSgfnjjz8McFWrVpXixYubsOsB5oUJACZk2bJlrslur4sUKSKNGjUyaVg3rV+ftZGXiTh++PnnnwXpC3XXrFnzeKyeIoEATAsedZ37DMHQ0BZLRmfXLSNn102R06slExUzxM7e27dvlyVLlpg+Hz5uZbSvZA1haZJn9\/3335c5c+bI0KFDjSfe119\/XWbOnCnET5s2TdgrxRTWQ9wi4O\/AkJoEu9ePsxxmzbSJ0zd0Xwgr5T8C\/A5s3Lgx\/zsSgz2IOgaGZZ6dO3caKPmShekwFy4HlG1JJzqQpZ7rr7\/eSHXQb+nRo4fALMEE0S7i\/JdeesksTbVt21bq1q0r+hc5BGYu32EqZ3kIhsWmnpekGt8uJjEGD6VLl5b69esLOlZYsyF9YRhbClcQmOaePXsKS5WWZRGdTZaVtYSJLhbLUNkJGkhoBNBdAQCYDqQnoRKmydSnFB4EduzYYT5MMAgJlNDhHDEiSz8uPL1JrFqijoFBdL5\/\/35zF1jOMQE3B\/RjypYta1IoYwJ+HKpVq2YUfxHZo2vDmWUqlHmxuYdheu6554RJxrJyTjB+VK9ZAkAACQzZB3c8OcceQz1j2M8L44GuvPJKWTD3Z3niisay4fWuRBmymRtz4eGA5DHyX2QeGtfoPEMgfdmr4o85c\/qyQaZPRWq0C3qvH+ceQTBCpkI9hAUB5qnhw4fLvffeGzA99dRT4q8OZ1g6G2eVRB0D48TX2xJOUlKSUfKVAP+Qtvzyyy+C2TS+X1yL4xcGHZzdu3e7Jul1GBFgqQhFXaQvZ9ctE8aao6MqrOTalNou36wVeXl1edl7WKR67ROF5aMDBw547CTM+MSJE6Vy5coe82hCfCAAY4J0xRcxWpgOiLBSdCFQokQJueuuuwRGBt3Mjh07Gr02f84o7qvOZfD3M6oZmOCH5b4kzMu7774r9913n+F6L730UsGKCT8wLCWhh4BUZ8yYMXLTTTcZcb\/7mrzHwlHbhC8ZpQxxxcBG0DU+lq7Tl0wTe\/8i1\/Oez\/rKndV3ymmlD8oP20tK1zVN5PuDqQJjM23aNIFRcR0rS6dI\/xBDn3XWWW4xcy2j17mfLTCBKUB3JD1TyhFJOhzAvj2u\/bDfgRKtp4o\/VLBK21zPBGNVcv8M5CUup512mjCf9O3bV5599lm\/ieWjTp06CR\/Trv3l+XCN8\/fann84U0+8UlQzMN50W7iR3r5k3d0w9F5QkuRhueGGG2TQoEGCPxjLsoSlI5aTUKrkKxmvvPiigelxV5e3uBtvvFEwk4V4oNeuXStKOTGYuzTLx06VkkleseEFZDklGvHb8EEvYe8iT1S84FHpeWqSnH766bJj5y558cUX5YMPPpC33nrLMMg8i1gk\/e9\/\/zNfbDhoJP2CCy4QFIFdx3znnXdKkyZNvOLlWsb1OprxdO1rKNcbf3tX8HmSvmyQX3vqBJvvcAD79ri2wW\/I0cKVZf22Yz4pFCxirWwsPqMYfpx99tlG+T4QvBlrw4YNBV0513IYnPzf\/\/1fUO\/7u5kf6sw\/EPMRz1o8UoFoGxRiuGLFiplu8ZVqAm4OMC9YepBEGc6+CAsjJsNy5cpJ586djcjPtQwTBNIX4qdPn24skggHQigCo5wFdenSRapUqaKUicHhIiliU5kyZQykycnJXrHBtLhChQpe80QC3\/IFD4ovStqzyYyhZOsuwv5F7qjhYyPl7bfflocffljs5xqmmGXMfv36CcwLTAx+j3iOyccXXI0aNSIy5vzC0597VLH0EQkXlSxZwtwbFF5RWo0UsV9PIPv2uPajZMOHI3Kf\/cE7WvNE8zPqDbOrrrpK2LfPWx53addcc42gM+cuLdg45h3mH+j+++8370I8HqKOgUESwoQF2IjUsQ4i7EqHDh0S0on3pitDuk1wuYRR2kWZl7A7gvMlHq4ahodwIMSXdIsWLQRKTU01ujpM1IlMm\/eLtBiwIJs6vPdnNqTecClatKiRjnnLE+609C9fkC0Pn+yTbOsi5x5G7GPkJDzv4ngRi7fZs2cbB3cdOnQwFm7Vq1c3Z0TISGPYLoN85A\/3mOz68gNPu21v52Or35KDMy4KGyEZ4QErVL6533vq5MW+Pa5t4K3WGy6JmBatz2gs3QvmHeYfiPmIdyFsFEUVRR0DAzY2A4GTOvYsIs6Vtm2brHjRAAAQAElEQVTbZqQjlmUZk1XXdG\/XMEUsI3nLo2nhRQClXWpEaddJN55ZheioovSl001\/cDzni5C+kMcU8HFA2Y+tMV544QVjdvnjjz+aM3ovmFXDvPuoIm6TD2+dY8aGomo4CQmMqVgPioAiEHcIRCUDgyY3P\/arVq0SFGzdoY6OCkwMJqf++muxLTtYd\/QmWbE9+5YvX17sMu76kKhxuPu39yry90wZ8KqRkiwLnzormzo1q0x0vhMKuf+mtREoY\/Mq058KPd6XE95Y6ZXIYzLrISAEULLdPbOT2HQ0fZ0pX7zJSxLO\/XkKlW9h6tVD3CCgA1EEshGISgamQYMGglY3ei4oNbrqwuA46L333jPu2FGcYp0we0ReAihTsnQE44OSk2u9FIW5GTt2LEE555xzBCbGXOjBILBm+wHpP3FV9l5FmEL7S1SA9IVztBGKuAeWTBPIXhoKto8oi7O\/Ec+puzrQgYGBnjRpkmBx9NNPP5mNHIl3l9+Ow0\/RhAkT7MuYPqPQimt7m2BoYnpA2nlFIEAEWGG49tprpXHjxsJvgWtx3nXeedd4vf4PgahkYFgDvfnmmwVvu\/zAoxeAhjY\/8H\/99ZfgqZStANCV6dq1q+AT5r8hidkYr3bt2sbb6ebNm7OTUIy89dZbjafdzz77TKiX+qgXZgalXTS2mYCQ7KDoa1mJ7cwOhmX0vI1iE8wKgMKIfHl3UwmU2OOI8nlFMCN7pg0XX2T3B4lKlbSpAqG\/YscHcua5\/OabbwRTaPRenGV5tq677jpp2bKldOvWzTi+4jljnZp4nkdn\/ngKw6Rg2gyxsSBjQ6m1ZMvRYlPUS0zotJIiEAYEVqxYIThTRU0Cdx5hqDLhqohKBoa7gCkpFhk4+YE7RUcA5Vts7XGzzh5JaWlpUq9ePbL7TV26dJEHH3xQChQoYLhe6qNeXL\/DDCGBYQLCDTwm1n5XHKcZWfrpMfoPcZI9VBzQBUp22bw6bx7SVbb4QUhe6BP6LDAuENehEIrm9oaO1APTjXMrzpZlGQkfyrswOZxLlSolMO44UiR\/vNHuWZ2MaTPmzUheGB86KjAtNhGnpAgkAgJ8tPTp00fuvvtueeihhxJhyGEfY9QyMJZlGZM0nMqdf\/75RhrD6GFc2rVrJ1988YVcdtllRAVEmKoiefnyyy+N6Rqu3anAsizBmqlXr17yww8\/GIdjxCcy2dIXMEBXxZWIj3ay9VlKtu4ivggz6HAwLu4wgZEZOHCgWSpCsti7d2\/B\/TiMOT5fOHON3weUfG2LOZe6YvoSCQwDKFKjvdgE40KckiKQaAgwF\/FB88gjjwgfzYk2\/nCMN2oZGAZnWZY0bdpU0FdBaReR2++\/\/y74WWGJiDzuaMCAAUJeloSYHFzzWJYl7H302muvyYIFC0xelIWnTp1qxPp8CbuWScRrGBjGzXIRSz9O6hkl+xWxNLTieks8EUtIjAFTZ5aHvBHmz+SNBOHpGckhdeMvAtNpy8q5PGlZlpHA8GOGlIa8eUEwFuzJs\/3LWhJJssdSvOlLYpMdp2dFQBFQBAJFIKoZmEAHo\/nDi4C9WzSSF1NzFB4OLMkyefbWNZaFIG95Ip02Z84cQSkdL884u+Lry12bxLOcuXjxYnfJEYlDHwUmJiKVu1SKibRLlF4qAnGFAC46Vq1aZVwkoMMJoQaB4v6xY8fiaqz5PRhlYPL7DkRx+zP+2Wl616pOijnn54GlIPYbctVnwfyZfiFZqT32mLgjTKHJk1cEE1KrVi1Bfwup3oYNG8TWa2EvJJxMeeoLlkso9G3alOXl11O+cMbb+igo1Ja9eqVEkjCRDmfftS5FIFoQSE9PF\/Y2QrcFtQeMTewdqlHYR3Efxf5PPvlEyBst\/Y7lfsQSAxPLOMdk39fuOGD6zRKSCeTjgaUiTJ05O8leIspvCYsTGhzS4ZxuypQpZpO2Sy65RDCVJg8+jkgn7Ep8neEeIC+Xj1z7oNeKgCIQOALorKGMi1KuvcUN+pp420aNwbKylouxin388ccF8+m\/\/\/478Ia0RA4ElIHJAYde2Aig\/wLBvEB2fCTPMCNIVJx0eNkMkeWzhTTaRhEXaYsrRUr5ljaDJcz2UchFFwtTaX7QcNDoqT62rsD8mnR\/nTOS15VYDjq8dba4o2M7f5GkfQtd0v7zgutal14rAoqAdwT48GC\/sxkzZgjbhHz33XeybNkyQV8Tb9tYGQ4ePFhatWpl9j5DOou7hNtvv13ybqnY+xhiNVUZmFi9c3HWbxiUNXfXkg1pbXLQlr4XybGhN0r6jA\/NiJMq1HRrTWQSo\/SQkpIiOKRiA0fMpD11Mzk52VgjIKVp3bq1p2xe42FeUMjdM6uTuKP0ubdI8RUPCWc73V5C8lqxJioCioBbBFj2hVHBypAPFlx7sIzszIxTVBgdNqr86quv5NFHHxWkMU8\/\/bRQXvQvKAQKBFVKCykCEUKApaDkhq3FpqT6Z4vUbp59TXyEmo54tSwdeZPAlCxZUj7++GNBCoPH6FA6hLIsPlZcSUqfIRnFm4hrPNeFyqnb\/VAw17KeEYjnlPXr1wvuOM4991yPw+S9r1y5ssycOVNw1Iq7hFdeeUWwTmTZGObGY2FN8IiAMjAeoUnMBBzXsb8RjuvyCgGkLzics9urmjZVbKr41A9idR8tnImLxqUiu9+RPj\/\/\/POCCbaznfRlr2bvJ7R7ZifZu+DR7ORSrUaLKxVrPlL21R4onF3TYHqyC2tAEVAE\/EIAyamrxMW1IEq7LBGjK2On4UQVB3b4NMOBqh1vn3nXeeftaz3nRkAZmNyYJGwMOi\/OfY4AokZKMqeI0uHNq8weRDSSVDGVU0KSr71R3IFycO04YQnISeQrULQ6JyWDgB4UgcghgGSF7QDQdTvmxkwa5uWtt94yivxOC0TLsgR\/UCj6ojMTuR7Gb83KwMTvvQ14ZDAwFEJp197jaEKPpkRFjLAo2jtthKmf5SGkLOYiAQ84X7T3Rpk08Us5uPZTn2TDhGM4ez8hzkhX7DQ9KwKKQOQQYOkX3054ccfiEAtENmIcP368PPnkk9KmTRt5\/fXXhWUkPG47e0LZK664wiwtOeM17B8CysD4h1NC5LId17WqU0bsPY4iOXCWjvDrAhNDO4UqxKf0hbH5Q\/iPwAwTc8zbT52QvW\/QvsxlIU+E0i51o7+CW36biFNSBBSBvEGAvfv69u0r+HwaNmyY3H\/\/\/WZ\/o9GjRxtlXbYPYW+\/Ro0a5erQmWeeKWw1cvDgwVxpGuEdAWVgvOOTkKlIYPJy4CjuYh6Nu\/+8bDfa2mId3d4bpVyprN7ZewZ5O+OATvVXsvDSoyKQHwhYliXXXXedYI2EZdHZZ58t1atXF9whsPfe5MmT5dZbbxXLyvIH4+wj2+I89dRT2fv9OdM07B0BZWC84xMHqd6HgLfdsg9NFaj\/xFXeM4eQipRlhcueRZhN21Xi1wVGxr5OtDMSFuc+RPb4WRqy6bmxhWTKunOz9xGy44vWf8DOrmdFQBHIRwRSUlKka9eu8sEHHxhm5vvvvxckL96sCvHYjRWTZeVkbliGwv1CPg4n6ptWBibqb1FkO2gvG9mtIH1pFYGtA\/Yc13Ox27HPMC0lW3e2L+Py7M\/eKOi7uA4e02bXOL1WBBQBRUARyEIg4gxMVjN6jBYEUNTFVBozaQgJDH3reUmqbB\/YRhY+dZag\/0JcJKhK2tQc+xWxT1FKh7RINJXvdWJ9YPZGObOZeNobpcUZDeSDl9rJgUNZm7w59yFSRdx8v4XaAUUgIAR2794tW7ZskcOHD+co5\/yI0Y0dc0AT0oUyMCHBF3uFsxiYVTJ63kZDM5dnbdh4QtmiER0MmzHSQKEEMZPG3wPKuCjlbt+RhXGxIiJVyx6V8qWOiiVHgUO27DgoaW8ukM6vFJBVOyuaOD0oAopAbCGAC4QbbrhBmjRpIijjs2kjG7kyCvY8uvrqq7M\/YuyNHa+88krjyI48HkijfSCgDIwPgGIxGSYFyYo7sjdobFWnjAzpdHI2dWpWORaHmq99xgLI3X5Dh7b8LG++2ltm\/PSTtL+6tXw5\/GH5+eWjMuvd5jL58wHy04QB8kqfznLW6SdK74euldTq5WX5BpH7hx7TvVHy9Y5q44pA4AhkZGRI\/\/79Zd68ecZUGrPo0047TZ555hnBN8y9995rGBV0XS6\/\/HJhy4F+\/foJ+jLskbZw4cLAG9USBgFlYAwM8XOAeWnS92e5+o0FbollI3u0MC022XGROmMyTd3ovHCOdYJ58bTn0NofOsn0SePl2ZszpNf5U6XqzgFSqGCWgp5tTXT+tY+KVaSCVDvpMvl64nSzN8qWbbsFCwbdGyXWn4446b8Owy8EkL6wz1nVqlXl888\/Nz5f3n77bcGEmm0C\/vrrLylXrpzZJmTw4MHStm1bQVozcuRI876\/+eabwnKzX41pphwIKAOTA474uTihbLIgZfFEN55ZJX4Gm48jwXwZZVsnbTnaWEqXLiXntGyWY8+hIie0y+4pTq3w4Kl7o2RDogFFICYR2LVrl\/Hjgj8XTKLtQVSvXl3YtBHmBYbl9NNPt5PM2bIsueyyywRndjiwNJF6CAgBZWACgiu6M6Oci+TF7iVedD0Rkhc7X6TOSF3+TWsjTnPpSLUVar3sKYRExV\/aPatTdpMo2zqpXPOBUrTC6VKy+bs59iJC+mIX4osr0L1R7LIJctZhKgIxgUDp0qXN0lGBAgXEsrIkrXbHGzZsKCwp8cFixznPSUlJxlfMn3\/+6YzWsJ8IFPAzn2aLAQRQzGUJia7WyIM9jGjHG+2ZOtzscQQjQ75oXj46uHacsCwUCDEmd3sOIVnRvVFAR0kRiH8EWDo644wzZOXKlYLyvnPElmXJTTfdJDAyzng7jP7M4sWL5Z9\/\/rGj9BwAAsrABABWNGaFaUHyAtnMC\/sYIXkJV3\/3TBsuO8akBUzpS6ebLuBlF3NpyEREwQFGBamLTVzTLfYRKnPhT1LGT0LyQjknIRLWvVGciGhYEYhfBJCiPPbYY8JHC3ovx1w2dDzxxBPlnHPOcQvA\/PnzjcM7V8bHbWaNzIWAMjC5IImdCKyMUMrFgy5k9xz9Fzsc6hnpCfsV7RjbRwKlA0ummeaTKtSUaJK+wKywVJS+bJDYZDqaeShYrLqg1+IvZRZx+697o7iFRSMVgbhEAE+7KOWyISvec9nbyNdA2fvonXfeMfozbDngK7+m50ZAGZjcmERDjMc+IGWxyc4Ew4IjOgjTaK7ttFDPLANRBwxIyvW9JVBii4CSbbpQRdQRTAr7CNmEa37iwtFRy7LCvjfK888\/L1dddVU4uqd1KAKKQJgRqFChgrzwwgvGfLpo0aJ+1Y70hnLnnnturvy867zzuRI0IhsBZWCyoYj+AIwLJtI22Qq76Lv0vKSWQOFWzrWXgWBE8JgbKJVs3SWqpC\/c5SP713ES9FfYR8gmp5KtyRCGQ0oY4az6jQAAEABJREFU90YJQ3e0CkVAEYgwAjAv+Hzx1UyRIkXkrbfekjlz5kiDBg18Zdd0Nwi4Z2DcZNSo6EEACYuTImkSHY8edI+mZzEwLBdFw13F\/fjmzZuNKDka+qN9UAQUgcgisGDBAhk\/fry+8yHCrAxMiADmVXGkL+i72O2xZ5FN4Za62G1wRgeGM0tInGOZ0H1JX\/aq0XvJi3Gw\/8nvv\/8uX3\/9tXDm2tnu7Nmz5dJLL5UmTZpIixYt5NRTTxW2H2AvFWc+DSsCikBsIIAy7k8\/\/WTeeRzYue6JZI+Cdx4pTcuWLeWRRx6R9evXi\/4FjoAyMIFjli8lsDKy9y2qEQUm0vkCQoiNsnSUvmyQMZemqgLFqnGKCOGd88Ybb5RrrrlGcCXOmTVt+4fq22+\/lS5dugg\/cvyQwcDgZnzGjBnmB80fJcCIdFwrVQQUgYARgFFhOwG2EGB7AN55Pk5wXoeibnp6eo46LcuSNm3amKWjzz77zGw1kCODXviFQNQzMIsWLRJMUk866STBy2Hjxo2Nd0O0vf0aoYdMPHATJkwQJg3M3Ki7adOm0qtXr6jmhntekirhNJF2woO0JX3JNHGSMz2aw0hX3O1L5Iw7enzpCK+5mEuj+xKJMeHb4fXXX5cNGzbIkCFDBKZk0KBBRlz8+OOPC19o7JNy6NAhqVWrlnz55Zfy0UcfGRfkMDZIYDCvdO0b1g08s67xeq0I\/IeAhvIDgeHDhwvbByBlLV68uFx00UXSoUMHwRvviy++KNdee638\/fffubqG87tckccjeNd5549f6skNAlHLwGBLD+favn17mTJlivBjT\/+xtR83bpz5suXHnrhAiQni9ttvl\/vvv1\/wgMhDRx24hB4zZoxhamiTuGghlpDoS6s6KZwiQlgcbUhrI06ioVhYPjq49lPZM6uTV9q34FGGY6hQ+RbmHIkDzxeuwWFacBWOoyukL\/iIQNeFZ2\/btm3GeydWBieffHJ2NzDH7Nixo9kYLjtSA4qAIhC1CGzfvt3sgcSc1apVK+PXZejQoYJF0jfffCNz586V5s2bG4nrrFmzonYcsdixqGVg+Gp9+eWX5ejRo3LzzTcLX6TLly+X7777zjwMMDJpaWlGBB8I8IjmWXOkfhyODRgwwIjv4I5HjBhhOGbWMZlY1kfJuiTMC8tHKO6eXbdMIMMNKG\/GltUmPwxLcsPWYlPJ1p1NfDQfju7PWkPGDBoJizdy7kkUiTGxRUDFihUFyZ6zfpgTpIlIaIhH1Mz+KYSdBMOzceNGZ1TMhLWjikCiIcA8AZUoUUIeeOABSUnJ+ZHJdZ8+feSll14yJtbR9nEcy\/crKhmY9Mz1wmHDhhmpC0s8Tz\/9tHkoLMuSevXqyRtvvGEUHvnSRfQO5+vvTZg6daoR6fOwsQvoddddJ5izoYeAt0RE\/jA2LFFNnjzZ32ojmm\/m8p2m\/hoR0n3ZM2242a+IMw2ldOgtVdOmZhOm08RHGx3dv052z+wkOKVDAkP\/8OmCd1xvVKRGe7JGjJKTk6VQoUK56ud5ZXdaEmBSkLRYVs69U0jDrTjPIGElRUARiA0EUjIZlxo1anjsLAq7zDkDBw4UJDN8nHvMrAl+IRCVDAzi919\/\/VWYCG699dZckwEPym233WY2zoLJWLcuyyzW14hhjFh+OnbsmHEydtZZZ+UqgjgfRoZ1TCQ+9vJSrox5GPHR3A2mtZ6X1DLncB\/2TBshGVtWmWqRvkDmIsoPKOVmbJvtUMqtbny75He3kb7s3LlTlixZkt0VJH\/9+vUzS5aFCxc2zPngwYOFZzI7U2aA\/VQ+\/fRTQRkw81L\/FQFFIMoRQLIK+dNNpLIsL6Ee8fnnn5sVBn\/KaR73CEQlA7Nw4UKj8Jiamip16tRx2\/OTTjpJypUrJ1h78MXqNpNLJIwOzBGMETuEWlbur188IzKxYPaK2A\/JjEs1cXW5J1P6Yvt6wVkd+xUVzVw+yq9BIlVBmpJuzJ1flcPLB0uRTSPM2Y777zzIdJPlInv\/okjqtpjG\/DiwOy0WBvfcc4\/wLLE0ecUVV8j48eMF5gXFPCwPfvnlF7n00kuNRBFT62effdbodpGnRYsWfrSkWRQBRSC\/EShbtqxR0t20aZNRdfDVH5gdFH5Rh+BD3Vd+TfeMQFQyMH\/88YfpMWJ2JCHmwuUA81K+fHlBmrJs2TKXVPeXq1evFvQTqlSpIjBH7nMFHxupkvYSUiT0X9jnyCl9idQY\/K0XqQrKtunLBhl\/LYeXD5HkzSNM2I6zz0hfqBeHdOi+EI4WQm\/r4osvlldeeUVghNeuXSs8y+hW3XLLLUbXijQYZvSwMLscPny44FYcqwXO0TIW7YcioAh4RwCXCBic9O7dW6ZPny7MS95K8H6jCoFyr7d8muYdAa8MzKpVq+SGG24Q9mmwCeVX71WGlsoGV4jfqaVMmTJGP4WwK\/GVSjrxMCacfZEtqalUqZLg7hkfHHfeeadgmo0ZdbNmzYQJBksRX3XlRTrKu1Ck2rIZF+pH+pKfkhf6gPTl6HFTZxgS9FkK1ekhByp2Fs5cuxL7FxFH+Wginq++ffuaHzPchaOr9fPPP5ulS8uyTFcbNWpklNIRJWN2zZmvMuJNBj0oAopATCCAztv\/\/vc\/YXlo1KhR0q1bN8HDtrfOowoBE3Peeed5y5YoaUGN0ysDgzUOZsY7duwwHkJ\/+OEHOfvss4NqyN9C+GfZv3+\/yc5DYQJuDiwDIbojiTKcfREiPvLAwKALc\/XVVws6NFg0EY9p67vvvms2zMP\/DHH5RTAu7HXEvkeR6EP6kmlGcdeuu2TrLnYwX84sG6GMi\/SFDhQq10Lw01Kozj1ysFJnE+balVDIhdmhTLSRZVlG0oIkhiUhFMdd+8gzjgdelpg4c+2aR68VAUUg+hGwLEvwJYZ+C1SqVCmfnUZZ\/\/3335cLL7zQZ17NkBsBrwyMnR1xOJIYrHWImzdvnuDfwpbK2Gec9\/AlSZ5wUM2aNT1Wg+gdJsZjBjcJMGREY8aGQmX9+vXlk08+MQ6GWIbiQcLxENYimFqvz0czahgYCNNpqOclqXQ9bLRjbB9TFwq77DBtLvLxgPSF5mFGoMI12nGppAgoAopA5BHQFmISAb8YmFNOOcVY\/NgjZKnlq6++Mi7PWW5BstG9e3dB\/M1avp0vWs8wMpi7Ib5jLCjq8uWLKO+1114zysFYII0dOzaoIaAsbBM+P4Khmct3mLarlEySX3o1k4cvqCHB1OO2zJZVciBTAkMDKXe9IyWveyp8dWdk+Kwrfdmrgvmzkw6uHUd3pGCVa6RE66lilTnDZz1ux+ZH+7FaDoBite\/ab9\/vhWKkGDmfgVDed3v+4Uw98UoF\/BkYk7trPib9Sy65xCwpYSmENQVxrvlCufam28KNPnDgQNDV9+jRQ9AGd60AMT7ifOJRxsI7L+FAiD1wbKkUehAocAZKO3bsNE2eWilJAi3rM\/+in03dUru5bC1VJ\/z1r13rsc71y+cZhVwUcJ1kS2B2HyyRoywvIE7dfI7JS5vxUhZ9LTaBC2U8iqfnZzMYXMOEZ45nPph+xFMZxTTrGUUX7v\/+7\/+CejZQhbDnIOajrB\/8+Dv6xcB4GjZLSmXKlJEymYRSrad8gcTDLBUrVswU8abbAvOCC2cyUoazL7Itj1h3hOlyl9+yLOPplzSkS8EwMHhcRGkTQjsdq6dAadGmDLoglzapIYGW9ZS\/fMGDUmLZREn+d5GpO7loUfGUN9zxZQ\/PFKiMtdy0zTIR+xG5UrkGXXL0CX0lNPbD3Z9ErU\/xrJLj+Qr1OVA8w4sn90MxDR1T5h3mH4gtc8yPbhweQmJgIoEHTBETFnVjjYRVEmFXYm8k0on3pitDuk1ofdthb2ebIcL5WDAMDKZxKG1CME3o6gRK6PjQR86BlvWUX1bMlj3D\/k\/SJ\/SnakOe8oYzvuDehXLkz6ezyTScecBniyu5tos1D8+Ea7xeJxtHj4Hi4BbP5ODqCrTteMyveIb\/2VFMQ8eUeYf5B2I+yvy5jcv\/PGNg2Gtow4Ysj7K+kER0Rh6c1NkWQlw7CYshfLpYliUo4zrTPIVhdGBOkOwgwfGUD8aFNCRLfA0QzktCeRffLyjvhtP3i+2wjj2OsDrKK+Xd9GWDDHw4nMNqKIvamTg9KAKKgCKgCCgCwSCQZwwMoiyna3VvncXLLian+KFBmdZdXsy7YWJwdle3bl13WXLFsWxEfpiXmTNn5konAgdEdho6MvZyFml5RTAwtFUjzHsfZRzfrJHNGfPK7wu6Lei6MB78teC3BcIcmjgR0ZMioAgoAoqAIhAwAn4xMEgsAq7ZUQBz5EAc4DVo0MDsBQOj8cEHH4hr+\/ilYVM8mA380rBu6mjOY7By5cqCi3cyfPjhh8KWBYSdRNz3339vrK5Q5oWRcqbnRdi2QAp3W4e3rDJVYjptAnlwwLMuzSB9YbmIsFJwCLAFwYQJE4IrrKUUAUUgphDgXeedj6lO53Fn\/WJgHnzwQWGDQ1ur2XmeOHGisJ+DO78w5GMNjrMnSYq78bIGiu8ZFIPZIwaLIbTsYVjwntu9e3fB0Ry6Ml27dhX0RMTxhw+X2rVrC2bRKOLaSZZlye233272V0J6c99998n3mcwKDBJEW9SNmTXm1Ti6s8vmxzmcy0f50X\/aPHTcPLpQ+eZcKikCioAioAgoAmFBwC8GBoVZpCiYt7kSOire0mEgYDwC7e0FF1wgDz\/8sGCaPWnSJMOMsASEuTaO9NhXJi0tTerVqxdQ1SwL4f+FcowFhgUdGggfNjixYwkLR3dYKwVUeZgyz\/gny4S6VZ2UMNUokpEpfbF9v+TllgFOCUzYBqMVKQKKgCKgCOQ7AvndAb8YmPzopGVZcscdd8iYMWPk\/PPPF6Qx9APGpV27dvLFF18IUh\/iAqUTTzzR1NurVy9BsdeyLLNkRJi4jz\/+WGrVqhVotWHJj\/4LCrxUFk4JzOHNebd8hN7L4a2zBbL3NmLDRcakpAgoAoqAIqAIhAMBnwwMpo+9e\/cWdFhmz54twRC77QYjzbCsrL0lcMqD0u6KFSvk999\/F\/yssETkCQDaIy+O6CpWrOg2G\/tUsOHW1KlTheUtiDBxpLktlAeRMDA0gwUS53CRLX1JqhjeLQlc+wfTwp5Ge2Z1EghmxjWPXisCioAiEDoCWkOiI+CTgWFppW3btoL1DsxAMHTttdcKG9olOtiBjD\/cFkh220UbnGcHI3K2JS44qkNxF8JsmuuINKiVKgKKgCKgCCQkAl4ZmJSUFLn88suFZZtQ0LEsy9SDFVAo9cR7WaQv\/SeuDNsw0XvZMqSr\/JvWRvZMGxG2ej1VhLTF9vnCbtKlWo0WCLNpT2U0XhGIVQS034qAIpC\/CHhlYKpXry4dO3bMZeUTTJfbtGkjtoO6YMonQpkeo\/8QW\/8lHEtI6L3smTbcbNwIMwOGkVxCgnmBiaGdAsWqcVJSBKK3A3IAABAASURBVBQBRUARUAQigoBXBsZdi0eOHJEpU6ZIly5dpHHjxoIuik1NmzY18VgNYZbsrrzGeUbAZl6GdDpZIM85fafAsNh6L3jerZI2VaCSrbv4LhxEDhiXg2s\/NSWRuKijOgNFBA9atSKgCCgCiY1AQAwM2wHg3A3roB9\/\/FEwoXbCx75BxKMIi7kzCr\/OdA17RsA2nUby0qlZZc8Z\/UzZnLl0tGNsn+zcmE5D2RFhDhzeNtvUaOu8mAs9KAKKgCKgCCgCEULAbwZm4cKFwrbcOJLzpy8rV6400phvv\/3Wn+ya5zgCNcK8fQASl4o93j9ee\/hOrjUdXDPORLFdgAnoQRFQBBQBRUARiCACfjEwOLF79NFHBe+19AWfLNddd53g5h8pi02TJ0+W\/\/3vf2ZpybIswcHdM88849ZlP\/Uo\/YeAvX1AqL5f0pdMkxXXW0bvhdpLtO4skd46gOUj9jvC0ki3CwD1yNDzzz8vV111VWQq11oVAUUgqhDgXeedj6pORVlnfDIwx44dk2HDhhlfKZZlCUtIP\/\/8s+BrhX2InGbVtWrVkptuuknGjx8vI0eOFJSAYXpeeeUVsXd4Ds\/4468Wewkp1JHZei\/UA+MSyWUj2lBSBBQBRUARUATyAwGfDAxLRmwqZVmW2Udo4MCBkpLi3cW9ZVnSsmVLGTp0qMDgIKGZP39+fowv5toM1\/YBKdf3lhPeCJ9Jticgkb7Yyrue8mi8IqAIKAKKQJQgEEfd8MnAzJkzxywdsTfRgw8+KIUKFfJ7+CeffLI8+eSTJv9XX30lSHPMhR5yITBzedb+R6EuIeWqOMIRKO9iPk0zBYpW56SkCCgCioAioAhEHAGvDExGRobMnTtX2AaAjQ6LFi0acIdat25t\/L8sWbJEduzYEXD5RCiAA7twjTN96XRTFabTJhDhg1N5t0TTlyLcmlavCCgCMY6Adl8RCBsCXhmY3bt3C3sKNWvWTOrVqxdUozA\/F110kWzatEk2btwYVB3xXAjmpUnfn80QMaE2gRAOtg5MXum+oLxLdzGfRomXsFJkEHjiiSeE5dzI1K61KgKKQDQhwLvOOx9NfYq2vnhlYA4cOCB79uyRJk2aSJEiRYLuO\/sp4djO1W9M0BXGUUEYGIYD8xIO\/y\/UlR+k1kf5gbq2GTACWkARUATiBgGvDAwWREhhTjrppJAGXL58ebEsS3B0F1JFcVjYNp+Geel5Sa2gR4j5NHseBV1BEAVR4KWYSl5AQUkRUAQUAUUgLxHwysDkZUcSta3\/JDCB6xc5McPrrr18hPm0M03DUYOAdkQRUAQUAUUgTAgoAxMmIKOlmgo93o+4+TSSF0ynoWgZt\/ZDEVAEFAFFILEQ8IuBmTVrlvz777+yefPmoGjBggWCPk1iQevfaG0JjH+5PefK2LzKJHpV3jU5Qj\/sXfCo7Msk23w69Bq1BkVAEVAEFAFFIDAE\/GJghg8fLnjdbdGihQRDaWlpghJvYF1LrNw1QtgDKWNLFvOS14gVqdFe2PuoVMvRed20tqcIKAKKgCKQ4Aj4xcCECSOtxg0CtgM7rJDcJPuM2jNtuKy5u5bkBxNTuEa7TAbmAVElXp+3STMoAoqAIqAIhBkBvxiYggULStWqVc3eRuxvFAhVq1ZN2PwxzP2Oi+rs5SOYFyiYQe2ZNsIUQ3HXJhMRwcPR9HWm9oLF1POuAUIPioAioAhEFAGt3B0CPhkYTKA\/\/\/xzmTFjhvz4448B008\/\/SSTJk2SWrWCNxF213GNEyN1sS2P2PcIChWXw1tny+6ZnbwSSryhtqPlFQFFQBFQBBSBUBDwycDUrVtXUlNTQ2nDSG6aNm0aUh3xWNiWwASr\/4LvF3AJ57YBKObiXdcb0SbLRhBhJUUgrxFYt26dRAPhXRyKhr5Esg95WTd4QnnZZjS2heFLoP3K6\/cwv9vzysCwVHT33XdL8eLFQ+7n7bffLg0bNgy5nniqwHZiF+ryUcnWnUOGBckLZFdUvOlLUrLlaI9UvInue2RjlVfn559\/Xq666qq8ai5q2+FH\/dFHH5Vzzz033+n888+Xm266SThHQ3\/ioQ9gqZieKz179jQrH4Hc006dOhnGPmpf3jB3zCsDk5KSIqeeeqrxohtMu4cOHcq2PmJn6ipVqgRTTdyWsSUwwTAwKO2yfITeS8nWXULCCH8ue2Z1EgjJC5WxszTbA3gj8ikpApFHIGcLMDBz5syRl156ST766CMlxUCfgcxn4P777xfei5xvS3xfeWVgpk6dKmwmZdMjjzwiF198sfnqQSfGGzRHjhyR\/v37yx133CE7duzwljXh004oG7gX3j1ThxvcwuH35eCacaYuNmSEMI+GcTGRelAEohSB5s2bB+XWIRhXEFomOBcailve4cb7EKWvasS65ZWBadSokdmN+uOPPxZo4sSJcsUVV8gXX3xh\/MJ46xWWSw888IAcO3bMiMLS09O9Zde0ABFIXzrdlCgR4vIRCrlIXdBnKdVqtEAsH5nK9WAQ0IMioAgoAopA9CHglYGpUKGCUcCl27Vq1ZLPPvtMEFOxtEScLypZsqT06dNH\/vjjD\/n22299Zdd0PxFwLh+FKoE5sj\/LJJolIz+b12yKgCKgCCgCikC+I+CVgdm7d6\/ZQqBcuXLy8ssvy4knnhhwh2F82rVrJx9++KHs2bMn4PKLFi0yy1DsiF27dm1p3LixoMC3YsWKgOvyVoC+3XrrrUIbb731lresYUlD\/8V2Yue9wtyphzdned5NqpiaOzGAGKQvh9ZmLR8FUEyz5hMCLOVOmDAhn1rXZhUBRUARiC4EvDIwa9askWXLlknHjh2NMm+wXb\/gggtkw4YN8ueff\/pdBUtP77zzjrRv316mTJkiKARTeN++fTJu3Di55pprwibVoa1Ro0YZjW\/ayAvqMfoPgYmhrRoBbiOw97jzuqINzqN40MSeRijwBl2BFlQEFAFFQBFQBPIJAa8MDBs4ZmRkGMVdy7KC7mLlypWlaNGi8tdff\/ldB0rCSH2OHj0qN998s8yfP1+WL18u3333naCsBCOTlpYWUJ2eGl+4cKHkhdTFbh\/GBenLCWWTZUink+XsumXspIDOIUtgjnvUZT+jEk1fCqhtzawIKAKKgCKgCOQnAl4ZmH\/++UfKlCkjlSpVCqmPMC\/UwTKNPxWh8Dts2DAjdbn88svl6aeflpSUFGPOXa9ePXnjjTeMRGjLli3GfA4Jij\/1ustDn7CW4uwuPRJxo+dtMNW2qlNGOjWrbML5cWAJiXaL1tf9jMBBSRFQBBSBMCKgVUUYAa8MTLjaxqQapgSGyJ86ly5dKr\/++qskJycLeimFChXKUQxm5rbbbjMMzeTJk4N23APj8\/bbb8vcuXOlVatWUrFixRztROpixj87TdWdmgXnF+fwllWmPD5gTEAPioAioAgoAnGPAPMoqhWbN2+O+7H6M0CfDAyA7dq1y5+6POahPIBjleQxkyNhYeaSDgrEqampUqdOHUfKf0GUelEuZpnLX8bov9JZIRiXESNGmH2a7rvvPilSpEhWQoSPa3cciHALvqtPX\/aq70yaQxFQBCKKwCeffCI33nijMZDAgMAdsQ0L\/reefPJJmTdvXrZzUJ8dy8zAxyNl8OF13nnnZbdDnbZxBb+1mVlz\/B88eFCYKNE15DeYfnHGGCNHRscFfsPuvfdeOeWUU4wxBGVok6V+nA86suYI7t69W\/Ayja4kZWziN75bt26mH\/QHok9du3Y184Kdr1mzZkK7tJ+j4ii6wBda7969hfEwDl9d++CDD7IxtMfJGW\/2q1evlrJly\/qqIiHSvTIw7IO0bdu2gJRv3aGGJRF7W7CM5C7dNQ6za+KqVq3qcRsDmBc2mkSKgqIx+QMhlp8w8eZheuCBB6RmzZqBFA8pLzowVBCs7kvG5iwJTKGKqVQTMLF0lL5skCmH\/xcT0IMioAjkOQI33HCDWQa\/6667sts+88wzjc4flpb8FuK+om3btoIfLvKfdtppwocXzEl2ITeBlStXGiMIyiDVhgFavHix\/P3338YlBhMizMVZZ50lTJj8ltrV8DF35513mnwwP5ZlCen\/+9\/\/hA167XzOc5s2beT11183feNjFYYHT8m0Ub16dWfWHOFSpUoZh6njx48XGCs7kfaHDh0qnOkPRPi5554T5gZ8jd1yyy2CFJ52ad8uGy1nmzljO4CRI0cKOp2++sbchN81d\/lKlCghV199tSQlJblLTrg4rwwMD0mZMmWCNoEGTbj70aNHE5T69eubs7cDDMXOnTtNFtrmoTUXLofChQsL6UTDkXL2l3gR33zzTcOY8TBceuml\/haNyXwwLFgb2XR422wzDpiXUi2z7o2J0IMiED4EtKYAEGjSpEl2brZdYZmcCH7\/cEXBnnTTpk0TJuz9+\/cb\/1rXXXedMNmRz5WQYnfo0EH4eLzsssvMbzhSHOpj4qfOF198UQYOHGgkOmlpaQJzwm+jsy7ynn322cLESTzWoI8\/\/rhgocq1O4Ixgs4444yA9CdheuiXXSfGH5aV03iE\/n311VfChzXuNJ566imBAbLLRNMZphHjEJiX1NRUv7s2e\/ZsYeVj7NixQthJPANInPyuLM4zemVgeJjwu8JSS79+\/cyDHggePGxw4OzPcMIJJwgiQV\/lDx8+LLyg5HPVfSHOJvRjbDEaZex4f85YONEvxodjPm\/t+FNfIHls6UsgZULNi7Rl34JHxUl2nTAxdljPioAiEL0IwEQ888wzYktEfv\/9d+ncuXMuJmb9+vXGVxaTPFIQ\/AfZDJFzdJZlyZVXXmkkHMQj1eF3kbArwfiULl3aRLNs37dvXzPJmgiXA0wPhhv8RodTUsB8whLSq6++asbHBsF5+dvtMkyflw0aNJDHHnvMeK335+OdCjEmGT58uLkvSNrQy3QSc55l5WTqKJeo5JWB4SFE\/MhDwlotkz0vhz9gITpjzQ8LHx48xHuVK1f2p2h2Hm\/LOrwYvCDZmf0M8PIhgiQ7PwTeRJvkCTfNXL7TVIkFkgkEccATL8X8VeJF8kL+IjXaS05qR3R8ko5KEYhDBGAOkMIgFWF4+NbChxVhCLcXgwcPNi4nLMsSpDDVqlUjyS1Z1n95+J0eMmSIrFq1ym1eGAaW7kmcNGmS2UyTMlxHmvhIHTRokNC\/Xr16SZcuXQQsIt1uXteP8QrLfNwDwqxI5HUfYqk9rwwMA7nwwgsFZS8eVHywoGjFxI8UAyaFPBDp27dvN2u3rLXygvFiEY+kw7YaIm9+ES83Jtj4o0H8ytgi0RcU1myiTSd9NDfLhLrDaeXFGR9I2O6zP2UOb81aLjpauLIUadwvBxWqc0\/QffCnbc2TEVZ8ue+KaQYwJDQhieH3lA9LgEB3BCMJwui9oBNCGB1BlGgJeyM+4th0kTzoKn7++ecEcxHLXM8++6ywfE+iN4kN6eEimBd2Hn\/\/\/ffl4YcfNstolhV\/UgjnMOGeAAAQAElEQVSWjRgj4\/3666\/l+uuvF6Q4LP2hoMxc6g+m\/EbY8w9nf8rEah6fDAwvCWue9ovAGihKZZg380Cz1gkhqmTNE9DReUH3BVBY14Sh4SXhOhBavXq1x+zcpAMHArPmmTZtmnz66adGg71Hjx7C2Dw2EELCjTfeKKx7Qoha165dKzat3LLX1Fy72IHsODvN37OpIPPgT\/7NaxZm5hTJKH5q0O35006k8vAC8qMaqfpjqV4UGHnnQulzPODJ82Ae6gQ+sLTPby4Q8Dz89ttvBGXJkiWydetWE0by4o\/U27Is4bkyhTIPWC3Zv9+Zlzn+0Rd86KGHjAsLJlQk7DgZzZEpjBf0o2fPnsIKAEthSJ8sK\/6YFyBDAMD94oO\/ePHiRBnFaaxskX4hOIDJMQleDni9f\/fdd7PnIKzcvGSP6SSfDAyjgwlBcnHHHXcEJLbjRqDdfv7551ONXwRTUaxYMZMXTtQE3BxgXrjhJFGGszdi6YuXjTy8gLzchCNBfC2wlgwh6qxSpYrY9O\/urC\/I0+vXyI6z03ydyxc8KMk\/Dc3usq\/8pJcqksUwFS9bN+D2KJ\/fhOUam4rmdz\/ipf14wLNMmTLZ74AdQLcsmsnuZ7jOKK7y4WjXxyRHGKVdzhC6E+iuEPZFLNfbv6Mo6O7cudNtEcuyBDNmVAvIAIOBo1FPysTkCZaQ8LNchIQJNxcsh1lWaMwLKwd8WAZL9IeP+GDH5K0cc9ILL7xgrKpgSFnxgGG0l8qQjDG3wDh6q4ffS+Yd5h8I1Q9v+aMvzf8e+cXAUB36MHDAmPKxpGRziKS5EozLgAEDBG3xU0891TXZ6zUvHDeATLxEntYAeYhIJx8vH2dvhDkiWxFQDukLL7+TEKHyJUMdaOeThtTJFs0S7y+x1QH1QampqcYhH\/o6kF0H4UApY\/bHkj6hv6kC\/Rd\/yqMrRAHO\/uSPtjw8dzwT0davWO1PvODJM+2kJn1\/lmgmmCtnf0MN2++zXc+mTZtMECVQE8g8wHTz7mQGff6joMszTUZ+I7196cPoIJVHTYD86OEgZfdWhnyBEpP1N998Y4rhcBTzb3MRwoEPX6SQwRLSLV8MRAjdyy5qWZbYXudZqrPnRMyrf\/nll+x87gLcR+Yd5h+I+chdvniI85uBsQfLxM6DhZke5l2Aiw0+9O233wqcI2uw6JjwY2mXC+TcqFEjkx2FW\/Y8MhcuBzTseZgsy\/LLPNuleL5c2j9i7IEUSgdKtu4iFXq877MKzKd9ZtIMikAcIMA7Fc0UBxDnGAJSeSxT7WWsSCj18vELE0bDSHi6d+8u6PhwHSyh94h\/HW\/kLY2lGX+ZwmD76FquZcuWZq8+MIcB++GHH1yzJOx1wAyMjRRiLUSU55xzjlxxxRWGMBVDwczOE+wZc2vqQRMbqYm7euD6YWLwVYPDPXd5nHG+HlyYsRo1apgimL7xEE+fPj3PthcwDfs4ZGzJ0glKbnieFG3Y2mtumJedk84RTKi9ZtRERSAOEFj41FkSzQRzFUmYkXqHUj8O1gKVLLDkwcesbZnEx+yECRNC6UaOsu3btzf73tnSB+YDpOcwMzkyJsBFkyZNzKbKDBXdUKRkhBOdgmZgIgkcmtfYwMNtokPjqguDW+b33nvPKDghxqxSpUokuxM1dQeyB9KR\/etMv\/HzAmE+bSL0ELMIsIQbzgkiZoGI6477Nzh0T1gGITdWQTYDk5q5ZE0cxJK4pyV40p0EU0CdxKFfY+shcu2NmFgfeOCBbKVeXFSEY6nHbhMpDN54bSaJD1eWr+y+2vni\/WxZltmvj3HigTlQZpNy8UhRycCw9HTzzTcbcz3MyeC6eRm5aZhAI0pEWQ3OHIUy1oOdNwdtbZa6gtVhcdYVTWF\/thDAbHr3zE7ZkpcCRatLmQt\/EpiYaBqL9kURUASCRwCjBFs6DdNiL+XAUKCjQs3eluBJdxKMgX3NbycO0+xrX2eUa22lXhgh9iWydXJ8lfUnnTHhwM5mYqZMmSIYYoRb58afvuRnHlYlaB+LXphWwolOUcnAcFPwN4PNP0tVrK\/CjPCSopWNmR9KxGlpaUbRifyJQBl+7EJ9aO04ydg22xCYFCxWnZOSIuAXApop+hHgQw4DCZbQLcsS9A1tKTQSC1uHECYH8jUipDToNJIP5offXj4iufaHKINUBGk4+WGcQtVVoR4nNWnSRF555RWxmRjmBJavwMKZz1eYcjBowRKWuODlq51IpMMUWpZlNuS0rNCssSLRv\/yoM2oZGMuyhIdlzJgxghm2zXHCuGAF9cUXXwh7fOQHaMG2uWZ7lt+aGinJwVbhthz6Lnjbheylo6L175eSLUdL8aYvuS2jkYqAIhCbCGCFwtI6vUe3r1OnTmYJh2sUPW+66SZzjUUShhXEeyN0S5Bok4fdjvGaTjgQol2nUm8gZf3NC4PkdKQ3fPhwt\/s3easPCx0kGMFSmTJlvFUfsTSkTSjvonfE5psRayjGKo5aBgYcLcsyu5Oi+Y2IE8Va9v+A84aDJo87woSbvIEo4aKQTH7KsUTlrt5ojWNzRnufI6Qv9DOpXAspVL4FwRgi7aoioAh4QgDdB3yi8GEHc8KmfmzXAvPgLMOHHVIU4tgQcNmyZQTdEvqF\/L4izaGee+65J5fhArqI6JxwdlvJ8UgmV36bbSnJ8eiwnpDAs3xkWVkSCBSHR44cafQh\/WkIJujHH3+UYIm5JRJWSNwH9DrRc4Pp5F47xzNt2jRZsGCB+ai39Z2c6YkajmoGJt5uyszlO8yQzq4bXi4eCQwVw7SgrIv0RZkXEFFSBGIDAXsJh96yi\/GuXbsImg10WQbCE+1VV11ldD9YssGx27BhwwQrTJPRcWD5B6edWIjCmODBmf11HFlMEEdxeLkdN26cwLy8\/PLL4ip9YYmGyR7mBR9gXJvCHg4s9TilJB6yuY3GMantkI8MjJuzkyzLEvRtTjnlFBNNf\/r06SPoyBA2kVF2gNm0l9TYqBhmxbWL+DT78MMPBT8v6HziPZcyMI5vvvmmPP\/888IynS1dcy2fqNfKwDjufLQGd4xJkxXXZ31xuPZx34JHsxV2C5VvbpaMitZ\/wDWbXisCikAUIgBjwmSFpY3dPXT8mjZtKkiZcU1x7bXXyqhRowSJCw7d+BrH8qdEiRJ2kVznlJQU4zsECQ2TJnWw\/QvtYRjBho\/ssYNVG\/qFLEmxHGVZWb8z6HnAFLRt29aYMtMAjA552K+HrVyIc0euUhJ3eZxxWFOlpaWZvX\/wI2angUnHjh2Fvtnt0X+c5jkZHRgXPNj60ze77rw4Yy0LQwLDhfSENufMmSNI+NmOh3TiIAxSGMOZZ54pqEnwDCBJoyzMI9iDBTqh5FfKQkAZmCwc8uQ445+dpp1WdVLM2d\/DjrF9srMmu\/h\/Qe+FRKyMkMAQVlIEFIHYQIAJCnfvmB6zfO2O5s6dKzAaSBqYpL0xLs5RI4np3LmzMGkiXaEevuKxEmJXZ\/bXmTVrlsCQoPzrLMsyCZIbdA2xdrL7NXnyZLOVgKvlp7OsZVnSrVs3eeaZZ5zRHsPoo8DAULfdDmfahQGA8bLbAy+YL1QJyOMkynft2lXsvB4bzKMEmEiYDrwJ2\/1kTDCjKF6T7uwKDCrjtceG2gTM5oMPPphrWc9ZLkrC+dINZWDyBXb\/G3VaHtUee0yqpk3NLmwvHcG8YCqty0bZ0GhAEVAEjiPAV7u9\/GRZ\/0lY2FvneBY9KQIxiYAyMHl422Yuz5LABKIDk75kmulhUsVUc3YebIsjZ5yGFQFFQBFwRcCysjZhdFos8YWPhGb9+vWu2WPvWnuckAgoAxPlt33PtBGmh0UbnGfOzoNtcVSkRjtntIYVAUVAEciFAMq\/\/\/vf\/4QlGHvbFJgYdFZYRnKnXJqrEo1QBKIIAWVgouhmuOvKgeMSGFfdF\/Ie3jqHk6jui4Eh7g9YIrAUEPcDjb0BxkyPLcsy\/rPwZotiKHowqampMnDgQGELF5R9USb9448\/YmZM2tHERUAZmDy697YTu0A2dXPqv\/javDGPhqHNKAKKQBwggF4Mlk5Y9KAgjOIoisToxfTq1UtOPvnkOBilDiHeEVAGJgbvcPqyV4Wdpu0lpBgcgnY5nAhoXYqAIqAIJCACysBE8U0\/vHmV6Z1TgRfLo\/Rlg4QziVggcVZSBBQBRUARUAQSCQFlYPLgbrN8ZFsghdqcbXmE3gum01A+m0+HOiQtrwgoAoqAIqAIBIyAMjABQxZ4ARiYHqPDrxSn0pfA70Usl2CfFPQVYnkM2ndFQBFQBMKFQOwzMOFCIoL1rN1xwNSOAu+Xdzc14WAPtt4L2wYEW4eWUwQUAUVAEVAEYh0BZWDy8A62qlNGYGL8bdI2oXb6gDm6X51O+Yuf5lMEFAFFQBHwjECspygDE6N3UJePYvTGabcVAUVAEVAEwoKAMjBhgTH8lewYkybpS6dnV5y+7FVh5+nD22Znx2lAEVAEFIHYRUB7rgiEhoAyMKHhF5HSOLDbMbaP2EtIUmCTpC8bJAfXfvqf+XTR6hFpWytVBBQBRUARUARiAQFlYKLwLu2ZOtz0iu0DynX9nxRrfoG5xnS6eNOXpGTL0aKm0wYSPSgCQSMQDQU\/+eQTufHGG+XEE0+U2rVruyU85l588cWC19x58+ZJIHsWHTlyRCjzyCOPyHnnnZfdDnW2a9dOPvzwQ9m7d28uKA4ePCjvvPOOXHPNNVKnTh3TL87kz5X5eMTUqVPl3nvvlVNOOcXkZzy0mZaWJuvWrTueK\/dp9+7dwjYZF1xwQXY5yp500knSrVs30w\/6A9Gnrl27ZveJfM2aNTPt0n7u2qMjZseOHdK7d28zHsbhT6+4dz\/99JPZ+mHGjBn+FEm4PMrA5MEtX7M93bTirwKvvXSUcn1vsUovMUtHpoLMQ5Ea7ZV5ycRB\/xWBeEDghhtukI8++kjuuuuu7OGceeaZMn\/+fFmxYoWwJ9Fnn30mbdu2lYkTJwr5TzvtNBkxYoQwwWUXchNYuXKltG\/f3pRZunSpYYAWL14sbBlAnUz+MBdnnXWWfPDBB3Ls2LHsWooUKSJ33nmnkA\/mx7Isk85mkEyq2RkdgTZt2sjrr79u+layZEnDZDA22qhevbojZ85gqVKlBBcB48ePFxgrO5X2hw4davpBfyDinnvuOalataqwHcItt9wikydPNu3Svl02Ws42c3buuefKyJEj5ejRoz67xn2YO3euXHfdddK5c2fzHPgslKAZlIGJwhufsTnLA2+BEklim00jfSlygu46HYW3K8guaTFF4D8EmjRpkn3BPkQpKSnmmkm7Vq1acvfdd8u0adOECXv\/\/v3Sp08fM8Ft2bLF5HM9LFy4UDp06CCLFi0yX\/BITpDiUB8TP3W++OKLZhNHJDppmVISmBMmT2dd5D377LOlRIkSJvrQoUPy+OOPy5o1a8y1uwOMEXTGGWdIpUqV3GVxGwfTNm7CLgAAEABJREFUQ7\/sxMqVK4tlWfalOdO\/r776SrZt2yaPPvqoPPXUUwIDZBKj7ADT+NZbbwnMS2pqql+9A993333XSKyuuOKKXOP3q5IEyqQMTAzcbCyOSrUaLUhfYqC72kVFQBGIAAIwEc8884zYEhE2YOQL3ZWJWb9+vZncmeRZ9kG6YTNEzm5ZliVXXnmlkXAQj1QHiQlhV4LxKV26tIn+999\/pW\/fvpKeniVZNpGOA0xP0aJFJTk5WZKSkhwpoQVhXlhCevXVV8342Em7UKFCoVUawdLs7v3YY48JDGD9+vX9aqlw4cLmfiB9QTLGPferYIJmUgYmim\/8obXjTO8KREBh11SsB0VAEYgpBGAOkMIwKdLxP\/\/8U0aNGkXQUEZGhgwePFiWL19uvt6RwlSrVs2kuTtYlmUkNeSBQRgyZIisWrXKXVaBYShXrpxJmzRpkrz00ktmWclERPiAlGjQoEFC\/3r16iVdunQxS0gRblarj3IElIGJshu0Z9pwwQqJbh3emmUyXbT+\/VwqJTgCKDpeddVVCY6CDp+v8ttuu01s6QO6I5s3bzbAoPcyefJkEy5fvrxR3DUXXg7op7Ro0cLk2Lhxo3z++ecm7HpgmevZZ58VpASkeZPYkB4ugnmBWXr\/\/ffl4YcfNstolpVzaSlcbWk9sYVAPjEwsQVSqL1lLyTqOKFsUU4eCcZly5CuJj2pQuaaaYGNJqwWRwYGPSgCisBxBBo3bmyUZLlcu3at\/PbbbwRlyZIlsnXrVhNGqoIeibnwcrAsS2BO7CxYLbmzTCL90ksvlYceeshId5DY9O\/f3ygckxYJoh89e\/YUrLVYCkP6ZFnKvEQC61isUxmYKLprhzdniW5hXkpfebvpGfovJqAHRUAR8IjAjjFpEs3Ex4nHzgeRgOIqirJ20X\/++ccEUdo1gcxDxYoVBd2VzKDP\/5o1a2ZLdFDQ3blzp9sylmUJZsxYQ5EBBuPpp58WVz0c0kIlLHhYLho\/frzcd999ZqnLskJjXjBHRqk2WKI\/KNqGOraQymvhbASUgcmGIv8DtuO6kq07S8nWXfK\/Q9oDRSBGEMDxYzRTuGFEORYlWbveTZs2meCePXvMmUOVKlX8ZmBQ0LXrY4L2pKBLvSxdYYnk1MPBR423MpQLlFg2+uabb0yxt99+25h\/m4sQDgcOHDAWPvilCYaQbiF5CqELWjSMCCgDE0YwPVVlLyHVSEn2lMXE2\/5fcGB3ZH+W4ydV4DXQ6EER8IoAPpOimZCqeh1A7CSanmLy3K9fv+xlrEgo9Z566qkCE0aDSHi6d+8u6PhwHSxdeOGFxq8KPnaCIUyc\/ZVqBdtHLec\/AlHPwCASveOOOwSvjIhMWfvF\/p+Hz\/9h5swJB42DKMSSOE6iXswNcYT0yiuvCKLLnCUif4WIGQkMP3RFatUV2\/9L5FvWFmIFAXQAJkyYECvdzdN+pnRIk2imSIPh9J8STFs4WON3MZCy6NggJbEtk1DqDefziRO+N954QypUqGC6hXVUjx49IrJcZRrQQ8whELUMDC8TNv88xFOmTBHEmqC7b98+GTdunODi+ttvvyUqIEKjHVO8q6++WnCItGvXLlOe9lavXm08Ol500UXGAZRJCMNh5vKs9eQTynqWwNjbBxRt2FqQvrD3URia1ioUgfhCQEdjEED3hCUQLrAKshmY1NRUogyh3Ouv23okHNRJQfRrihUrRtAnofz7wAMPZCv14iUXT78+C\/qZASkM3nhtJgmzcZav7L76WY1mi1MEopaBQdnq5ZdfNq6Xb775ZqPpjm+D7777Tpo3by4wMmlpafLXX38FdGv4QkDKghtuxImYHFLvsmXLBK+JmBTyMuMsCodQAVXuJrO9fATzArnJYqLs5aNiZ16QLX3B+y4O7EwGPSgCioAicBwBfpv43eISpgUJMmEYCnRUCONwjt9Jwr4IxsDOg0S6bNmy9qXPM75mbKVefjvZD8nWyfFZ2I8MjImPWZuJ4YMWS6hw69z40RXNEmUIRCUDw4M5bNgwI3W5\/PLLBS33lJQUw+XXq1dPECvCmfOy4DkS6Yk\/uKLghktt8uOmGUkMXy6WZRkNfFxtv\/baa8KLwo\/D2LFj\/anWax6bganhRf\/FuXwkSYvMztNeK9XE\/EJA21UE8h0Bfr+QHuNp17Iss6WArSvC72KjRo1MH2FyIHPh5YCUhq0HyALzw6aKeNLl2h+iDFIRW6kXxilUXRXXdps0aSJ8ePLbTFqwOjeUg0ELllBnAC\/6oJT\/CEQlA8MeEr\/++qtxRX3rrbca5sIJVUomM4MjJ8uyzEZetijVmcddGL0ZxJto27urlzL8AJxzzjkEhT74+wVjCrg5zFy+w8SeXbeMObs72ObTR9PXycG1n5osRWq0F3VgZ6DQgyKgCDgQ+OWXX8zmi0QhRe7UqZP5uOMa5dqbbrrJXPPB5s8yO7ol6BpSvmHDhoIuIOFAiHadSr2BlPU3LwyS05He8OHDxd3+Td7q47cfKXuwVKZMGW\/Va1oeIxCVDAxfA6xxpqamZmu5u+KCUi\/cONy+7QPBNY\/rNS8067tsMIYCmms615ZlCVIewrjl5muHcLBkS2BO8OHEjvqTKmXpyLB0VLzpS5LLgR2ZlBQBRSAhEWDZG58oSAH4LWvWrJn07t1bYB6cgFx22WWCFIU4pMgsjxN2R+gEYlmDNId67rnnHsF\/jDMvpsf8HnN2xruG+U11KvW6pofj2ulIj\/pGjBghI0eOFH9\/p2GCfvzxRwmWBgwY4LdpOv1TiiwCUcnAYCHEsKtWrSrFixcnmItgXnCVzYPr7QV1FuThRbdm6tSpQt3ONDtMfbZeDb4WLCs0x0k2A1PDyxIS\/itov0DxrI3PCpVvzqWSIqAIJAgCfLTZQ0UCbRsXwGCwDIQnWraRQPeDJRssKFlmd\/c7xvIPHnKRJMOY3HnnnbJ48WK7+uwz1pZ4ucUoAuYFnUNX6Qu\/h0z2MC8TJ070ySiw1OOUkmQ35kdg+\/bt4vwYZdyuxSzLEvRtTjnlFJNE\/9iZGx0ZwiYyyg4wm\/aSGjuJc0\/96SKSMRhH8oO\/P2USLU8gDEyeYMP64s6dO01biOs82dyjeU86GbEe4hwO4qGZO3euqYqXxBMDZTKE4YD+y6HVM6RYy\/JS4vxLw1CjVqEIKAKxggCMyY033ihY2th9xpW\/7d6hfv36cu2115oNG5G44NBt2rRp8sADDwh7ItllXM8pmcvsGCUgoWHSpA6WzWnv66+\/Nhs+ovOHUcN5551nlqRYjrKsrA82fodhCtq2bWt0DqkfRoc87EmEdJo4d+QqJXGXxxmHCkBaWppcf\/312VsikA4mHTt2NH2z26P\/OM1zMjowLi+88IL40zfqzSvasWOHfPzxx4bhWrBggWl2zpw5gj+bzz77TEg3kS4H7j8uEyDGRjIegGFK7XqIUxKJOgYGbpMXjpvDlwZnd8Rapq0pTxl3eQKNox5eWpalkPDwxRNoHYHmR\/8l+ZQyUrhOiWzrI90+IFAUNb8iEJsIIE3AEAHdPHT03BEfVDAaSBqYpL0xLk4UkMR07txZmDSRrlAPyrZYCWHAwO7Ss2bNEhgSdP+cZflwRHLzxRdfmJ2t7X5htdm1a1dBOu3M7wxbliXdunWTZ555xhntMYw+CgwMddvtcMaQAgYAxstuD7zYbfv333\/P5ZCO8r765rETEUiAiYQBw5vwihUrTH8Z06hRo4ziNenumoVRff7558U5RvSekJbB2Lork6hxUcfAOG8E+3M4r51hHmiYGGdcKGE4XV5kOHzLsoSX19aFCbRevihsssvyBeGOju5fZ5iXo4UrS6E6PaRI435SsEpbcZdX4zISGheeJX0GMoBBKQAEChYsKHyMsfxkWZYpiYTl888\/N2E9xBcC\/EbY8w\/n+BpdztFENQOTs6uRu4J5QRFs4MCBZo0XLh9O3rKyXvZAW0YkbG8WxtcP5bdu3SI4lnKlHbv\/JFmOFqosW4u1k83S3G0+13KJcs0LuHHjRsVk7VrDvLEmHsq9jwc8eR7MS6MHvxGwLEv4TXNaLOH7BQmNO10TvyvWjFGHwIYNGwTFbHsOYj6Kuk6GqUNRzcB4022BywyHYhOa\/SwbYY6Ht1\/8ziBm9bZ85Qt7NPERC0PVq1cz2cuXr2D29cBfg5NKV84yyytauZXbdGfeRAxjMYYr8UQcu+uYUY5ET8A1PpDreMCzTJky5p3SQ2AI8JvG7xxLMDVq1DCFYWLQWUH6zBK6idRDTCPA72WXLl2E+Qe6\/\/77Y3o83jofdQwML5ntxtrbCwXzgtY6g6MM50ApPT1dUP5COQpGBoU1rtHID7QuZ348Bbdo0UKgatWqmyR7yYtlLycRTwbOzngNJxs\/QKzjsx6veGThESoO+YNnePruHDvvjFLgCFiWJZhZ480WpVz0YHBXgfS5QYMGRmGY30DbEjTwFrREfiPAe8I9Zf6BmI\/yu0+Rar9ApCoOtl4mKzhIymONxFotYVdCWkI68d50ZUh3R5i2sSkkpoik41sBRsZfBTnK+ENrdxww2U7wsg+SyaAHRUARUATyCAH0YlAIxaIHBWEURlEkRi+mV69ecvLJJ+dRT7QZRSB4BKKOgWEotitsrIE8ecLFv8HWrVuNx0lMDSnnL2G+dvfddwva4bzIaHdDwUpy\/G3XNR8m1AdXzzTRR\/eqcqIBIgIHrVIRUAQUAUUg\/hCISgYGL7tIQvDJgtmZO9hZu4WJwZFT3bp13WVxG4fkhTXBmTNnCr5kcH+NxRGMjNsCIUbajuzcSWA2D+kq+IChiaP7lIEBByVFQBFQBBQBRcAfBKKSgWEt9rTTThP0XD744ANx1YVBgvLee+8ZiyG866LE6M9gsTZiQzC88cK8oGzbrl07I8Xxp3ygedZsZ\/lIxB3z4lpXcoPWrlF6rQjkQADHVoj7c0TqhSKgCCgCCYpAVDIwKBrefPPNRkKC18gePXoYM1oYENz848mQzcfQlcE0EAVY5\/175JFHhN1G8TC5efPm7CQ2Z\/z006zNEnEIBZO0ZcsWIY87Ig3l3uwKAgzYDEwNl20EWDpacb0lB5ZME3v7gILFqgdYu2ZXBBQBRUARUATiGAEfQ4tKBoY+sxnZww8\/LCztTJo0SWBG6tSpI5j84WoZF\/94bwzE2dxXX30l+NGgfvRf2CukxXFrIXfn9u3bC8tU5A+G+k9caYqdXTen2Wd6JuNCQlKlZClQIknwvAsRp6QIKAKKgCKgCCgCvhGIWgbGsizBMmjMmDFy\/vnnG2kMw4FxYdkHF9eYAxLnD8G4+Lvpoz\/1+cqD9GXm8p1m+ahVnZTs7HumDZc900aYa7YPIFCoXAtOSoqAIqAIKALRg4D2JMoRiFoGBtwsyxJM\/fAqiNIu+2Ng7ofuCktE5HFHAwYMMPtOTJ8+XSpWrGiylChRwjj2oQ5\/yVneVBLAAQaG7Cwf2RIYlo62DOlqlrlbQQIAABAASURBVI5IQwLDuXCNdpyUFAFFQBFQBBQBRcBPBKKagfFzDFGZzZ3\/F3vpKLlha6nQ431Jqphq+q76LwYGPSgCioATAQ0rAoqAVwSUgfEKT3gT7aWjlOt7S8nWXcJbudamCCgCMYcAm8eyV82JJ55oDA+QLLsSUuiLL75YcDqH\/p+rVaa3QWOEQBkMG9AjtNuhTpbiP\/zww2y9QGc9OBBli5VrrrlG0D2kT5zJ78znDE+dOlXuvfdeOeWUU7LHQptpaWnCHlzOvM7w7t27hd2X0XukHZtwp9GtWzehH\/QHIozhBn2x87F7M+3SvrPe\/Ay\/+uqr2RjY\/XR3vu6664Tx52dfY7ltZWDy6O6xfITVUVKFVCnasLVplZ2oCagCLygoRRkC2p08QOCGG24wS9t33XVXdmtnnnmmzJ8\/3yyD49L\/s88+E7Y5mThxopAf68kRI0YIzEl2ITeBlStXCoYIlFm6dKlhgBYvXix43KVOJlSYi7POOktwV4GVp10NHtHxj0U+mB\/LsozbCvZS+umnn+xsOc5t2rSR119\/Xegb27HAZLAXD21Ur+7ZyrJUqVKCi4Dx48cblQG7UtofOnSocKY\/EOHnnntO8P+Fgcctt9wikydPNu3Svl02P89Yr3733Xd+dQFDEsbvV2bNlAsBZWByQRKZiD1Th5uKbebFXOhBEVAEFIFMBJo0aZJ5zPrHjX9KSpbiP5N2rVq1BM\/h06ZNEybs\/fv3S58+fYSvdybLrFI5jwsXLpQOHToI7iYwdkByghSH+pj4qfPFF18U9kBCopOWKSWBOXEyMdRIXnxtoUPINVu4sNntmjVruHRLMEbQGWecIWwe6jaTm0iYHvplJ1WuXDmXjy76hzUp1qFsBfPUU09JtDEAs2fPFqRFMHIwi646lzhnRfoFpuwYbY9Xz4EjoAxM4Jj5VWLN9nSTz3Zil750urku0bqzOav0xcDg+aApioAikAMBJrxnnnlGbIkIBg2dO3cWVyZm\/fr1wuTOJI8UBOmGzRA5K7QsS6688koj4SCeCReJCWFXgvEpXbq0iWaLl759+0p6etZvnIl0HGB68OXFpoKuProc2QIOwrywhMTyDONjI8q83v7FV6fB5Pvvv5enn35akK6AhWuZjRs3yi+\/\/JK9NOeartf+I6AMjP9YaU5FQBFQBPIVASZEpDBIRegI1pmjRo0iaCgjI0MGDx4sfOVblmWkMNWqVTNp7g6W9V8eGIQhQ4bIqlWr3GUVGIZy5cqZNHxzYQ1KGRMR4QNSokGDBgn9Y7PJLl26GB9hEW424Oph3NjN29ty1pIlSwQJFvewTJkyAbehBf5DQBmY\/7BwhiIePrJ\/nWmjQFHPa8Mmgx4UgeMIoOh41VVXHb\/SU6IigCTmtttuE1v6gO4InsTBA70XdEIIly9f3jgAJeyN0E\/BkSd5kA58\/vnnBHMRy1zPPvtstk8ubxKbXIVDiIB5gVl6\/\/33BeemMHCWZYVQY2SL4qvMstz3D4YPXSbyIKGxLPf5ItvD+KldGZgI3csZ\/+w0NePEzp0Cb\/qyQSa9UPnm5qwHRUARUAT8RaBx48ZmCYL8a9euld9++42g8HW\/detWE0bygh6JufBysCxLYE7sLFgt4fjTvnae8YT+0EMPGd0UJuP+\/fsbhWNnnnCG6UfPnj0Fay2WwqKdefE1dqyx0JFhaQ+LMF\/5YzM973qtDEweYJ1ubx1w3O8L+i8Z22abLQSK1n8gD3qgTSgCikA8IYDiKoqy9pj++ecfE0Rp1wQyDzjxRHclM+jzv2bNmtkSHZY3du7c6baMZVmCGTOWTWSAwUDfw1UPh7RQCfNilouQMN13331mOcyyQpNYzJgxQ1CcDZboD4rMwY4N5Wp0iDBjL1OmTLDVaLnjCCgDcxyIcJ\/W7jhgqkSJN2Nz1ppy0QbnmThdPjIw6EERCBsCOyedI9FMfLSEbbCZFaEci5JsZtD8b9q0yZz37NljzhyqVKki\/jIwKOja9TFBo4xKHe6IpSsskdDhIB09HHzUeCtDvkCJZSP2rKPc22+\/bcy\/Cfsib+kHDhwwPmmQhARDSLeQPHlrw1Ma+knoDoEfOjKWFRoz5qmdRIpXBiZCd9veSgAGxrZAwgMvP2SH1o6LUKtarSKQmAjwXkUzxdtdweS5X79+2ctYTMwwHMFO7u7wOfXUUwUmjDQkPN27dxd0fLgOli688ELjX8fVtNnfa7a18ZcpdO3jhg0b5NdffzUO7nDS55qu14EjoAxM4Jj5LGEzL3ZGWwJTKHMJ6eDaTwWy0\/SsCPiLADoAEyZM8Dd7QuUrc+FPEh3kvh+Rdlbp9J8SzI0\/evSocVQXSFl0bGBabMsklHrD+XzihO+NN96QChUqmG5hHdWjR49cZuMmMQYO9vIRCtPly5ePgR5HfxeVgYngPUL64lr94a1zTFTR+vdLqVajTVgPioAiEBoCMAjRTKGNLndpdE9YAiGlcOHCYjMwqampRBlCuReHaubCxwEJB3WSDf2aYsWKEfRJKP8+8MAD2Uq9eMnFeZvPgn5mQAqDN16bSWK5iuUru69+VpPv2XT5KDK3QBmYyOCao1askIhgGwGUdwkXqdGek5IiEFYEtLLEQGD9+vXG1wujhWnBqoUwDAU6FoRRFt23bx9BnwRjYGdCObhs2bL2pc8zHn9tpV4YIfYlsnVyfBb2IwNjwoGdzcRMmTJFsIQKt86NH10JOou9fOS8V0FXpgWzEVAGJhuKyASczIuzBb4WndcaVgQUAUXAHwTQM7Hd6VuWZbYUsHVFkFg0atTIVAOTA5kLLwekNCxvkAXmh00VccjGtT9EGaQitlIvjFOouiqu7TZp0kReeeUVsZmYYHVuKAeDFizdcccdZpsA1\/75ugZfcGF7BazDfOXXdP8QUAbGP5yCzoXFUbGW5aXY2RnGSiLoimKioHZSEVAEIo0AbujZfJF2UErt1KmTWcLhGuXam266yVxjkfTtt98S7ZXQLbHNrxs2bChYyHgt4CaRdp1KvW6yhBwFg+R0pDd8+HBxt3+Tt4awtMJxX7BUpkwZb9W7TXMuH7Vu3VqwIHObUSMDRkAZmIAhC6xAgeJJUrhOCSlQIkmwkqC0Sl9AQUkRUAQCQYDdp\/GJghQA5qRZs2bSu3dvgXlw1sPmjUhRiBs7dqwsW7aMoFvCyy2WNeybRD333HOPuEoIMD1G54Sz20qOR7oq9R6PDuvJ6UiPilEcHjlypCCV4toXwQT9+OOPEiwNGDDAb9N0uy\/28hFMExt12vF6Dh2BuGJgQocjPDXYPmBqpCSL7XFXMprksJIIT0taiyKgCMQDAiwx2ONYunSp7Nq1y1zCYLAMhCdatpFA94MlGxy7DRs2TKpWrWryOQ8s\/+AhF1f1MCZ33nmnLF682JnFhHEUh5fbcePGGSbo5ZdfziV9gTFgsod5wQU+16awhwNLPU4piYdsbqO3b98utkM+MjBuzk6yLEvQtznllFNMNP3p06ePoCND2ERG2YF7y\/IRDGelSpWirHex3R1lYCJ4\/46mrxOUdo\/uzRAr41TjeVelLxEEXKtWBGIMARiTG2+8UbC0sbuOK388taKnUb9+fbn22mtl1KhRwgSIQ7dp06YJlj\/siWSXcT2npKTIW2+9ZSQ0+\/fvN3XceuutxiX\/119\/bTZ8vPjiiwWz5\/POO09YkmI5yrKynKuhFwNT0LZtW8GUmfphdMjDnkQsixDnjlylJO7yOOOwpkpLS5Prr78+e0sE0sGkY8eOpm92e+CF0zwnowPjwgaK\/vSNevOS6Dd6Nywb4f2Xc162H0BbMZlVGZgI3DbbD8wJmRIYu\/qkiv+ZN9pxelYEFIHERgBpwkcffWS8zHpypjZ37lzDaCBpYJL2xrg40UQS07lzZ5kzZ44gXaEelG2xEhoyZIjcfvvtMmvWLIEhQflXHH84a0Ny88UXXxhrJ7tvkydPNlsJeJuILcuSbt26yTPPPOOo0XOQpRUYGOq22+G8fPly+fjjjwXGy24PvAYPHiy\/\/\/57Lod0lO\/atWtU6ZjQ70GDBpllPJb2PKOgKcEgoAxMMKj5KGMzMNWSt5mcR\/dlCCbU5kIPioAioAjkIQIFCxYUe\/nJsv6TsHjadToPu\/ZfUxpSBIJAQBmYIEDzVcRmYGxHdjAvRRu29lVM0xUBRUARiAgClpW1CaPTYgnfL0ho3OmaRKQTWqkiEGYElIEJM6BUt3ZHOiepcnShOcvRyllnPSoCISDw\/PPPmy\/pEKrQou4RSIhYlH8xO2YJpkaNGmbMMDHorLCMhMKwidSDIhAjCCgDE8EbhQIv1Zc8tycnJUVAEVAE8hUBy7IEXQy82aKUix5MamqqDBw4UBo0aGCUfVGI\/eOPP\/K1n9q4IuAPAgX8yaR5AkOAJaQqGRvl9IOLAiuouRMTAR21IpDHCKAXg6UTFj1YIqEUyx5G6MX06tVL1F9JHt8QbS4oBJSBCQo274WOFisvVY9s9J5JUxUBRUARUAQUAUUgaASUgQkaOu8FTz+wSJIqJZtMBYtVN+coPWi3FAFFQBFQBBSBmEMgYRkYvD6iFIlzKBxGnXjiicaREmvDuOwO9k4ifaEsWweYcybzos7rQEIpVASeeOIJ4w8k1HripTwO0JTWiWKgGPAMxMt7Hcg48p+BCaS3YcrLxmUostl7gFAtTMv8+fMF503sLxKqRn42A1NUpS\/gq6QIhAsBHJ81b95c8GCLd1Olc0UxUAx4H3gveD\/C9a5Fez0Jx8Bs2bJF8EbJuXHjxoJbbTw+wrzcddddUqBAARk9erSMGTMmqHt3tFg5U65GuZ3mrMtHBgY9KAJhQ4Af6JdeeknwYJvf9N577xkLHlzx53df4qV9xfQj82zjMfmSSy4xYU\/31jWe9yJsL1oMVJRwDMy3334r+D5gE7RXX33VaNtbliXsHfLYY49J9+7dhb01Ro4cKTA5gd7DjPInmSJVj2wy5wLFqpmzHhQBRSB8CMDEtGjRQqKBcMMfDf2Ipz4opi2kbt26Zl4K5L7yXoTvLYv+mhKKgdmzZ4+wtwe3hU3KatWqRTCbLMuSDh06SLVq1czeJLNnz85O8zeQUb6+VEveKqcfOO7Ezt+Cmk8RUAQUgZhCQDurCOQvAgnFwKxdu1ZWrVoleKSEq3UHfcWKFaV+\/fpGCsNGZ+7yeItjCem+1C8k+dQy3rJpmiKgCCgCioAioAiEgEBCMTD\/\/vuv7NixQ0qXLi2VKlVyCxu7sNputlevXi179+51m89dJBZIULvKM00y1kdFarQ3YT0Eh8DGjRtlxIgRwjm4GrSUEwFwjBc8nePKr7DiGX7kFdPwYxqvNSYUA7NpU5ZeStGiRQ0T4+mm2swN+dPTs\/Y18pTXNZ7lI+KO7s2QEqeOFJgYrpWCQ4AfMxQkgyutpVwRUDxdEQntWvEMDT93pRVTd6honDsEEoqBQQcGEJKSksSyLIJuqXLlrM0XMzIBQugtAAAQAElEQVQyzFKS20xuIpG+VE\/ealJgXNiF2lzoIQ4R0CEpAoqAIqAI5CcCCcXA2EAjYUEKY1+7nkuUKOEa5fd1teRtJu+aZavUwdS60B1MGTAzDzhqSnTKhEHWr18f0nNFHVCiYxmu8YMlFK76tJ51wGko0bEAhHBgQD3xSgnJwETqZp5QNlmK7N8rE+aKzP9HIupcKlEcV+GcifvFOVHG7Gmcc+fOlVGjRoX0XIGj4nluSBg674\/iGT4sbVwV0yxM09LSjM6mjUuwZ\/CMVwd3CcnA+NJtCURxl8nAJhiYx++4R+pfOVoa3v55QA6IXB0S6fVHit9HOTHo27evDBgwQHFxwUXflZzPieIRH3gMGTJE2C08HPczXh3cBcnA2FN2bJ1LlixpOuxLtwUlMjL60pUhjyvhSAgTbaXocDKm90Hvgz4D+gwk+jPAvOQ6V8XDdUIxMHjftSxLsCzatWuXx\/uHhIZEX7oy5FFSBBQBRUARUAT8RkAzhg2BhGJg4ELLlSsnMC82k+KK5MGDBwWHd8TXrFlTQlHopQ4lRUARUAQUAUVAEQg\/AgnFwGAejZM6dpr2tE3A5s2bZdmyZcbMumXLluFHXGtUBBQBRSD\/ENCWFYG4QSChGBh0YK655hpz88aPHy8rV640YfvAJo5jxowxpqrsk3TaaafZSXpWBBQBRUARUAQUgShCIKEYGHC\/8MILpV69esK2Ap07d5affvpJjhw5YszV+vXrJ2+88YaRvnTt2tVs6kgZJUVAEQgTAlqNIqAIKAJhQiDhGBgUefv37y8VKlQwDsFgYk488UQ5\/fTT5d133zWwdurUSTp06GDCelAEFAFFQBFQBBSB6EMg4RgYbsGpp54qEyZMkDvuuENQ6iWuYMGChol55513pE+fPmbHauKV4goBHYwioAgoAopAnCCQkAwM965ixYryxBNPyLx582TFihXy999\/y9ixY+X8888XmBnyKCkCioAioAgoAopAdCKQsAxMvtwObVQRUAQUAUVAEVAEwoKAMjBhgVErUQQUAUVAEVAEFIFIIeCuXmVg3KGicYqAIqAIKAKKgCIQ1QgoAxPVt0c7pwgoAoqAIpD\/CGgPohEBZWCi8a5onxQBRUARUAQUAUXAKwLKwHiFRxMVAUVAEch\/BLQHioAikBsBZWByYxJwzN69ewX\/MWeffbbUrl1b6tSpIxdffLF89tlnkp6eLvqXE4Fvv\/3W4ARW3mjSpEk5Cx6\/WrRokfHhc9JJJ5l6GjduLI8++qgxhz+eJe5PO3bskGuvvVZuvPFG4fnzNmDSg30+QynrrU\/RmOYvpjx\/p5xyinn2vD2\/b731ltthxiumjOuDDz6Qyy+\/XHAOCja8m126dJG5c+cKW7W4BSQzkrL6jGYC4fIPLsFgmijPqDIwLg9MoJfr1683XnvZhoDtCSjPi\/rPP\/\/II488Infffbfs2bOHaKXjCCxfvvx4KLATuPIj1759e5kyZYocOnTIVLBv3z4ZN26csM8VzJGJjOMDm5G+9NJLwo+Ur2GG8nz+V7af2XqDtrgH8fhsB4Ip7zkTC3gESvGKKX60eC\/T0tLkzz\/\/NNuzgA3v5o8\/\/ih4N3\/66acFnMXlLxRMQinr0o2ouwwF00R5RpWBCeGxRbqC115e2OrVq8uIESOMQ7zffvtNeJGLFy8u06dPl1deecXr10cIXYi5ohkZGeYHjo6zXcPs2bPFE51zzjlky6YZM2bIyy+\/LEePHpWbb75Z5s+fLzBD3333nTRv3lz4sQT3v\/76K7tMvAWYAIYMGSKffPKJz6GF8nyGUtZnx6IsQyCY0vUlS5ZwEiSu7KXm6fnlGTUZjx\/iFVMkV4899pjw3rFh7oABA+SPP\/4wElHw6dixo0Fg9OjR8v777+f4LQwFk1DKmg5F8SEUTBlWojyjysBwt4MkxKIwKCVKlBC+iJlw8eLL9a233mq2JLAsy2xbwMsdZDNxVWz37t3mh41BtWjRQvCI7ImKFClCNuHAj9WwYcOM1AURNV9zKSkpYlmW1KtXT9544w1hi4gtW7bIRx99lONHkvLxQPyoPfTQQ\/L666\/7Nb5Qns9QysYS1oFievDgwWwGnGWkatWqeXyG+R1wYhGvmPJhwUcb43377bfluuuuE\/vdBZ8+ffrIDTfcYJ7ZkSNHmj3obFxCwSSUsnb70XoOBdNEekaVgQnyCUaSgI4LX2\/nnnuu2UfJtaqLLrpImjRpItu2bZNEWNpwHb+7a7DYunWr8GNXq1Ytd1ncxi1dulR+\/fVXSU5OFpjDQoUK5cgHM3PbbbcZhmby5Mk5fiRzZIzBiyNHjsiXX34pPE9ff\/21FCtWTGrWrOl1JKE8n6GU9dqpKEoMBlO6j5QP8Tzhhg0bcvKL4hVTlhSnTp1qmBNPv4O8qzAwvPNgxxIkoIWCSShlaTuaKRRMGVciPaPKwHDHvZL7RL7cFi9ebBLPOussSUpKMmHnAXEqX2nEMfnyYBFOZFq3bp1h6CpUqCBVq1b1G4qFCxcaZdXU1FSjJO2uIEq95cqVM\/oa9o+ku3yxFgfzhsRp+\/btRsqEvg8TgrdxhPJ8hlLWW5+iKS0YTOn\/xo0bZdOmTQLD7IuJJL9N8Yrp\/v37BWaidOnSwvvn7ncQDKpUqWIwY3KGeSQuFExCKUvb0UyhYMq4EukZVQaGOx4E8ZAgSeDrwtsPWd26dU3tq1evNjoa5iKBD8uWLTNfazAiTCJMxPzwYbHQsmVL6d+\/v7DM5AoRa+rEwfSgW0TYlWBeypcvb+qnHdf0WL1mWfK8884zispI\/Vgy8zWWUJ7PUMr66le0pAeDKX1HgsDkWalSJfOc3nnnnYKlDc9v06ZNpVevXoJiKXmdFK+Y8i6+9tprsmDBArnnnnucQ84R3rBhg4CbZVnZm+WGgkkoZSXK\/0LBlKFF1TNKhyJIysAECe6uXbvkwIEDZkmDrw9P1fBDRxpWCzA8hBOV+PqydYFQ7mPJZ968eUavBUz4URo6dKhZKnFa2LCmu3PnTrJImTJlstfXTYTjULhwYZNOFAwj53igBg0aCJMEE6RlWX4NKZTnM5SyfnUuCjIFgyndRmGfMxK+m266SViutCWr4DZmzBi59NJLcy0Zk5aovxe897hE4DewcuXK2RLUUDAJpSz3L9bJE6aMK5GeUWVguONBED9G6L8gMuVrzlMVMDfobSBmtUWnnvLGezySlZUrV5ph8gJipTFnzhyj1IsS4PPPPy8su6GIiwm6\/SULzohVKYjEi7M7AueyZcuaJMqYQIIeQnk+Qykbz3DzDtuTA+\/yJZdcYhgYLOGQ+OH3BWtEGJpnnnlGWPa08UhkTPkY+fjjjw0U6HHVqFHDhEPBxM+yEq+\/v54wTbRnVBkY8yoFf0AxjWULTzUwqXqbdD2Vi8d4vsAKFChgFHhxPJeWlibowjBWcMTccvjw4YaJYVLA5wuMDuk2eVuug5kEbzuvnsVgHezzyT0Jtmw8Yg8TzQTBpMjS56BBg6RWrVpGcZx3HOeVWMARh7L6q6++msuRZaJhykcI7zp44OCTJTfLyilFDAWTUMrG6jPqDdNEe0aVgYnVpzgG+41JJQqoSFu6detmfvhdh4HVFh5micdEnWUlwkqKQH4jUKpUKUHKgr4H0kKYFtc+IYHp3LmziV6wYIGw1GQuEvCALgZ6MXyMoJ+Gqwl+AxIQirAN2RemifaMKgMT4qOFVMGbbost6gyxmYQpblmWcUrHgLG6ccV29erVJLklvo7B221igkaG8nyGUjZB4TbDxh8RkgE8cGOxZCKPHxIFU5aK77jjDuMtmmVhlPP5ODkOQ45TKJiEUjZHJ2LgIhBMfQ0nXp5RZWB83WkP6SxV8AXGpMl6uIdsYiubsbzhTVfGU\/lEjOcHD2ztCYAwvk\/AwptuC8wLTA\/5KMM5USk5OVnAIJjnU5\/t0J4anlUwpBZbAsN1sPeDemKJ0M9ACoW+EJIXPEeff\/75uYYQCiahlM3VkRiI8BdTf4cSL8+oMjD+3nGXfCiL8pXFpAmT4pKcfWl\/gZHXmz5BdoE4D2BRhJKuN6YPJoWJlx8p9A3w6mnrymCNRB3uYGJvJNJJ86YrQ3q8UyjPZyhl4x1Xnk2eX86exkqa\/XxjdUO+RMGUPcpwNIm\/J5bT2DqALRfAwJVCwSSUsq79iPbrQDBlLDx\/ifKMKgPDHQ+C+GHCRJqHxduyhv0FxoSKfX8QTcVmETe9Rn\/g5JNPlrZt27r1lWEXYZ0X5V2YPsymiW\/UqBEn46QOKw9z4XJAUZAlJ8uypH79+i6piXUZyvMZStl4RhlTYJ4rfPI4LYxcx4wUkKUNJC72R0u8Y8r7OmHCBLn77rsFyelJJ51k9oaz31tXjLgOBZNQytJ2LFAwmCbaM6oMTJBPMspSJ554oin9888\/G2+U5sJx4EVGYZWo0047TRKdgeHHjB91FHMRiYKLK7Hn0Q8\/\/GCi8dXBVxwX\/CDC0Kxatcps4EicKyGyhonB2Z3tQNA1T6Jch\/J8hlI2nvHlIwSGBKnrzJkz3Q6VSYdJhA8brJFgeMgY75iyoSrWRkhBzzzzTMO8MH7G7olCwSSUsp76E23xwWCaaM9ovDIwEX8W0Wlh0zIm5B9\/\/FHmz5+fq00mYr7U0Om44IILcqUnWgQ\/5phS8iP\/4Ycfmi81VwymTZsms2fPNhZK7dq1k6JFi5osMDMwgUweH3zwgTBBmITjB7x8vvfee8YLLyJrXJcfT0rIUyjPZyhl4xlsmOkmTZqYISJtYJnEXDgOMObsV0UUDu3spc94xhTnlGlpacYhZatWreTNN9\/Mdo8ADp4oFExCKeupP9EUHyymifaMKgMTwlN7+umnS4sWLcwePZgF4+adiRXx8YgRI+SJJ54wEyoTMdKHEJqKi6L8mN97772Cx1x2kgUzXlQYGqRV\/PA9+OCD5oeQHaeZAOyBw8jg+I6yTBA9evSQtWvXGnypo3v37sbigTa6du3qdm8qSbC\/UJ7PUMrGK8w8gyyRoJhqW4TgiBF9F\/SyeP9RXkUKiJUHYScW8Ygpumoo6aJzgYQU79r8Bm7evFk8Eb+PNi6hYBJKWbv9aDyHgml4ntHYmdOUgQnhCeaFxR8EyxtMwHiPRcrABo59+vQxEzHr5UzKlmWF0FL8FIUpefjhh81+KPz4c41Uhh98\/EQgggYzcEW65Rw5Uiy7LGJ68lGWOtiSgCU6vgT92SvIWW+8hkN5PkMpG694Mi4kMM8++6xZDoZx7tSpk7CUjG4X7z+\/A\/weDB482GxeSBmb4hFTGDmW0BkjjMntt99uPur4sPNEo0aNIruhUDAJpaxpPEoPoWKaSM+oMjAhPsQ4ZsJNNpu4oXtBdZZlCToYAwYMkDfeeMN4liVeSczSEN44kaLAkGBlJJl\/SFZ48fiae\/fdd91iZlmW4FuC\/WYwy6RMZlEzmSDl+uKLL+Syyy4jSuk4AqE8n6GUPd58zYwUNwAAD4RJREFUXJ54xtBP6NChg6ATwyBxkQDjAgONs0awI96ViI+n3wsMGFCcdx1nINeumFDWsvz7DQ2lLO1EI4UD00R5RpWBCcMTjEIZyyEzZsww+\/rgefL7778XdGQQ6YWhibirAikJWwXg1XTFihWCAi4ieF48JgNPA7YsS9jUECaHMpT9\/fffBelN7dq1PRWLu3iWzBg7ruv5EvU2wFCez1DKeutTNKYFgikT5wsvvCAshXIf\/v77b\/nmm28EE2Jf73w8YXrhhRea3zww8JfA2fX+h4JJKGVd+xEN1+HCNBGeUWVgouGJ1T4oAopAlCCg3VAEFIFYQUAZmFi5U9pPRUARUAQUAUVAEchGQBmYbCg0oAjkPwLaA0VAEVAEFAH\/EFAGxj+cNJcioAgoAoqAIqAIRBECysBE0c3I\/65oDxQBRUARUAQUgdhAQBmY2LhP2ktFQBFQBBQBRUARcCAQVQyMo18aVAQUAUVAEVAEFAFFwCMCysB4hEYTFAFFQBFQBBSBmEAgITupDExC3nYddCQQmDp1qtx4443GtTxO9dwRWx+w2eSTTz4puKJnH6hI9CVcdeIan72l8DL77bffhlwtdTRu3Ng4fKPukCuMcAU4SWSbC+7lGWecIWwXwB5n\/fv3FzYQjXDzAVfPM0j\/oI4dOwrPG31\/6KGHzL5hAVeoBRSBKEagQBT3TbumCMQUAm3atBE84zK5WVbW3lfsRM7u2eyw\/dNPPxmPwWwGOHr0aLn00kvlrrvuEjbCi9aB\/vLLL8Ju6+xRxQal7HcTbF\/37dsnjJszXquhYOvK63LJycnyyiuvCNuDsE9Xz549c+11lNd9ctcezyD9g0aOHCkXXXSRu2zhj9MaFYF8QEAZmHwAXZuMbwTY3M9274+bc76AK1asKLj2ZnsJtkx49NFHzb5QbEp57733SrRKI9ifqnnz5sJ42PuHDTODvXuUvfbaa80+V61atZJmzZoFW1Wel0tKShLuZZ43HEKDbIbKfQuhCi2qCEQ1AsrARPXt0c7FIwLs9YR4n13LGR\/76Xz++ecEo45SUlKMVOm3334ze3tZVpZkKdiOwsAsWrRIkA7YGyEGW5ejnAYVAUUgARFQBiYBb7oOOf8RgDFAumH3ZNasWXL48GH7Us+KgCKgCCgCPhBQBsYHQJqsCPhEIMgMlSpVyi65c+dOOXjwYPa1BhQBRUARUAS8I6AMjHd8NFURUAQyEcDiZv78+ZKRkZF5Fdr\/kSNHBOueDRs2hFaRllYEFIGERkAZmNi\/\/TqCGEVg9erV2T1Hwbdo0aLmGoXeTz\/9VGx9kb\/\/\/luuueYaYxJ766235lD4xRT7vvvuE0yTURZGgbhr167GRNtU5uaQnp4uKBJfddVVcu6550rTpk0F0+63335bdu\/enV2CJS2UjGnzzDPPNBZUBw4cyE4ngBk4Ojy33HKL0C5WVRdffLE8++yz8vrrr5Mlm9avXy8vv\/yyoMDLeBYuXJid5gzQ7tdffy033HCDXHDBBULbjI9xLliwwK05MJZc9P\/yyy8XdGwYB9ZgKAqDS4sWLQQTY2c74Q77iyvtIm2bPn263HHHHfLYY48JTB04ohvFWDF\/5v4vXryY7CZ9ypQpBhNM2rnP3bp1k3\/\/\/dek60ERSEQElIFJxLuuY853BJjM582bZ\/phWZaZ1FesWCGdO3cW\/I0wqS1fvlyY5Lt06WIYEpgFrslHGCbn+uuvl5NPPlkmTpwomGl36tTJmD1fffXVgs8V04DjADPExIgJ83vvvWfyon9Tt25deeGFFwzTtG7dOsFsGpPnX3\/91TAETLCOarKDX3zxhdx5553yf\/\/3f\/L+++\/L0KFDzZm+rFy5MpvZ+OOPPwRFZfq\/efPm7PKuAdrG2mngwIHy9NNPCwzUnDlzBFN0yrZr104efvhhsc25t2\/fLmlpadK+fXvTf8rD1MEIwMicd955UrZsWaHN+++\/34zFtc1wXPuLK21NmDDB+JOB2YMpgdlirPgG4t4\/\/vjjxmKN\/j\/wwAMC03bbbbcZppP7C9bFihUz2OCXxsaCupUUgURCIHQGJpHQ0rEqAmFAgAkHSQSMCNXBUFx22WXGAR6+VpjIiEfaAZOCxQ6+WJCYXHjhhXLCCScIX+tIObp37y58iVetWtVMekx+55xzjmFA+vTpIzAR1AXBNN19991SoUIFwxxwJh7JT9u2bQma\/F999ZUULlzYOJvD3BspiEl0OTCOMWPGGPPi1NTU7FSkSUhLLOs\/iyWYrHvuucdM3J5Me5EmIJGAcfvf\/\/4njRo1MqbmlmUZKdGgQYOMCfb48ePlueeeM8tZMCcwMPhnwVcLfULyQ\/qHH35o\/LaMHTtWwIe0cePGZfczXIFAcKVN7iM+ZbiXXE+bNk2wSPvmm28MPjhDfOmll4zpOs8I9xfsGBf3CUYMZhMzaZgbW0pDXUqKQCIhoAxMIt1tHWueI3D06FFBSoAEAGYCScIVV1whTMJ0BsYFp2gwEVxDTLacLcsSJiuWQGA2mMCRTDBRv\/nmm8JXOM7wMMsmP0Q9fMUTpk2WYggjsRk2bJgwITJBYgVFvE0tW7YUrKJgLmjPjrcsSwoUcP8zsX\/\/fmHy3rRpk6AfI44\/+lClShVHTFaQuizrP8YmKzbrCDOE9ARmp2HDhlmRjiP9Y3mKKCQ\/TN6EITBhQicMniyLWVZWOzVq1DBjI+2ff\/7Jlt5wHSoFiyt+Zegz7TOuK6+8Uuz+E8cyEctIhGFIwdOyssZDHPhghs5SFFIn4pRiDwHtcWgIuP9lCq1OLa0IKALHEUBRleUcdDCQZCAVYdJB9wRm5tVXXzUSkePZc5yKFCniNo1JmMl727Ztgu4JdTmJeu2KYFiYZDdu3CjogMAIMTna6fYZR3voxeDvxWYS7DRP59KlS5vlKxR7kdR88sknRleD\/DAvPXr0EBg4rn3Rrl27BJ0Q8iHBcec0zrIsueSSS4xUBunUDz\/8QPZcBJPkjHQyC\/72x1neWzgSuNIeTCnMKGF3BPPDuEijD5yVFIFEQ0AZmES74zrePEWAyRh9ExgJiOURthUYPny4UZx1fnX72zEkHij6nn766fLdd98ZPRaWmGxieYm2ICQ2lmUZHRAYHtqDMfK3LW\/5qAd9DJgidGZYvmrTpo2gUwPTxATMROytDjsNBgaJkX3t6VyzZk1hKwbSkdagOEs4v4g+hxvXvBuLtqQIxDYCysDE9v3T3icgAkhgGDbSHZZxCPsirHTQAYGxgHzl9zedjQ4\/\/vhjYwVFGZYzbr75ZkFnA9Nr4vwhGBj65ysvUh8YI\/IhhfGkXEx6XlCkcM2LvmsbikCsI6AMTKzfQe1\/wiGAxRCDxvkd0hjCvghJiGVlSWKQzPjKH0h6rVq1BCamX79+xuKHslhAYU2EjgzX7sgZB2OC\/g1xa9eu9UtPhWUvm5mhXH5QJHHNj\/Fom4pALCGgDEws3S3tqyKQiQCTPToQLCPZeiOZ0bn+kbRguYKVSvXq1c3SC\/oqLDuRlqtAZoTtg2Xfvn2ZV97\/Mf\/Fcoa6YCTw24JZMBIYy7KE5TJMp73XkpWKNRHLQ1zBwGzdupVgLkLqwhhIQNnXVd+F+LykSOCal\/3XthSBWEZAGZhYvnva9xAQiN2imCxjSs0I3nnnHeMnhLCTYCowGf7zzz+F\/CjV2pIbTLMxlXbmJ0wZdHMwZ8bCiThvBDOB4i7LKHY+lG\/xaYJDOeKQ9lAvYW+E9AXrLMvKkhKhcOwuP5ZcMDfo3eDjxbL+s8xxlz\/ScZHANdJ91voVgXhBQBmYeLmTOo6oRIDJGwpn55i8cehmWZbxytulSxfjtA7pCe2ge\/Liiy8az7k4QIM5gAjj34V8WA1hAYUFC3ok6K7gfwYHdDiFsyz\/GINVq1YZpV3atQlFYcx8ucarLGd\/CGstvPSS96OPPjIm2oRtot8wX0hgGHP9+vXtpHw7RwrXfBuQNqwIxBACysDk083SZuMXAfy+IJ1ghFioIDUg7C+h20Je6qA8YVfCIyvu+IlnKQmTZSZ0fLhgnQQDgDO30047jSyGMON+6KGHjBkyVkOvvfaa4P8Ft\/SYYeMt9\/nnnxenjxisfJB4UAFnd0rDOGXDIog8EPnwDAwz0rZtW9Me8ZATG5SQibOpZMmSZqsBtg5AcoMzPFuHBibr3Xffle+\/\/16I75LJtKF\/Ypd11uu6\/IVyMFIl8pIGI0Q4XBQMrjBh3F\/6wNhcmVziwJ508HTFnbLUQTpLeZyVFIFEQ0AZmES74zreiCHAsgdeU3ERb0+S+HxhjyDc7bPcw7WnDqxZs8a4ycdXDHmYoB588EHp3bu32NsOEA\/x5Y+3VnRcWBqyrCyJCfox7F2EbgqeXi0rK54ylmUZt\/9IMZh0cYSGRAZdEiQ2ePxFp4O8TJ64vMeRHlIZ4mAq0G+hPOmWZQnMEs74WDbCi+69994rOMrD7w0Mks0M4Wofb7lPPPGE2NjQZq9evQTGifohpEt40B0yZIgwqTMG6sI3De3TJxgYdG7ITz\/YEuGZZ57Jrhcvvng6RhK1dOlSgbljOwLyL1myxGx7YI+JuFDJsvzHlba4lzjbg8njGieAbI+APhPjgUlDQoZPHtIxw+c5IB1mDD8\/3bt3F5sBRIEaxhOMya+kCMQ5AtnDUwYmGwoNKAKhIYAPFDYURO+EydYmJhaYF5gYfKd4agW9Fibe33\/\/3XjMpTz74cDQsCmhazkkEOiNMOGhMEt+HNylpaWZbQVc83NtWZZxy09\/8BdDX\/HWy7IRTBF5IBgEXN4zHrtu6qct8pIOs4GfGZgQdGGQkMBQkYcJ2mZeqA8pD\/v2zJw5M3tstI1LfBgo8tjEuGCKcKwHc4PfnMmTJ5slsXr16uWQ6NAPlsaY3OkfBGYwBLTfoEEDGTFihFEoJg1iwse7rd1eOM6W5R+utMW95D7zXNAfCJzR6WE8MGtcO3HnmnTukc2gUg6C0YExBGPqV1IEEgUBZWAS5U7rOBUBRUARUARCR0BriBoElIGJmluhHVEEFAFFQBFQBBQBfxFQBsZfpDSfIqAIJCwC6J6gN4NybT6D4HfzKP+yDOV3Ac2oCMQYAsrAxNgN0+4qAopA3iGAHg0m6x07dhQsmVBWRt+kf\/\/+gpJw3vXEv5ZQJKd\/EPpJ6BfR9zPOOMO\/CjSXIhBDCCgDE0M3S7uqCOQ7AgnWAayysHDCysdJrkrK0QILiuTOftphLMMs6z+LtGjpr\/ZDEQgFAWVgQkFPyyoCioAioAgoAopAviCgDEy+wK6NBomAFlMEFAFFQBFQBAwC\/w8AAP\/\/nCskzgAAAAZJREFUAwAU6g6OWe99eQAAAABJRU5ErkJggg==","height":337,"width":560}}
%---
%[output:66a706c6]
%   data: {"dataType":"text","outputData":{"text":"Evaluating episode 50 \/ 100...\nEvaluating episode 100 \/ 100...\n","truncated":false}}
%---
%[output:0ee99c48]
%   data: {"dataType":"textualVariable","outputData":{"name":"idx","value":"77"}}
%---
%[output:5b2f7b92]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAQAElEQVR4AeydCbxN1dvHn40KIdcUMoYmyZSQIUqlolSIUuiNikqR0sQVKg3SQP9mmhTNI0VIKkk0qkzXmCFDCEW857uuddr33H2me8+99wyPj2cPa16\/ve9Zv\/2sZz2r0AH9pwgoAoqAIqAIKAKKQIIhUEj0nyKgCCgCioAioAhEiYAmL2gElMAU9BPQ+hUBRUARUAQUAUUgagSUwEQNmWZQBBQBRaDgEdAWKAKpjoASmFR\/A7T\/ioAioAgoAopAAiKgBCYBH5o2WREoeAS0BYqAIqAIFCwCSmAKFn+tXRFQBBQBRUARUARygIASmByAplkKHgFtgSKgCCgCikBqI6AEJrWfv\/ZeEVAEFAFFQBFISASUwOTosWkmRUARUAQUAUVAEShIBJTAFCT6WrcioAgoAoqAIpBKCMSwr0pgYgimFqUIKAKKgCKgCCgC+YOAEpj8wVlrUQQUAUVAESh4BLQFSYSAEpgkepjaFUVAEVAEFAFFIFUQSFkC8+abb0qtWrVk+vTpqfKstZ+KgCJQ0Aho\/YqAIhAzBFKSwCxatEjuvfdeOXDgQMyA1IIUAUVAEVAEFAFFIP8QSDkCs2TJEhk0aJBs3rw5\/1DWmhSB+EBAW6EIKAKKQNIgkDIEBm0L00Xdu3eXFStWJM0D1I4oAoqAIqAIKAKpiEBKEJi1a9dK\/\/795eqrr5YtW7ZI9erVpUSJEqn4vAu2z1q7IqAIKAKKgCIQIwRSgsA8\/PDDMnXqVClevLgMGTJEJk6cKGlpaTGCUItRBBQBRUARUAQUgfxGICUITJkyZSAuMnfuXOnbt68ULVo0v3HW+hQBRUARUAQUAUUghgikBIG5\/fbbDXEpVapUDKHTohQBRUARUAQUgVRAID77mBIEJj6h11YpAoqAIqAIKAKKQE4RUAKTU+Rc+dasWSMqioG+A\/oO6DuQN++A4po7XF3DVVJdKoHJ5ePkD2vw4DukZctzVOIQg7PO6iBnn322tG7dWkUx0HfA9w6celEfQfRvInV+E3AfwliVy+Eu7rIrgcnlI+GlmDt3ufTrN1bS019RiQEGnTvfLnv31owJppdeer1ccMEF8sorr6S0DBgwwLzpDzzwQErjkLP3wPvdSVRMy5w\/VHa2vEVGj3su7t6FRMU0lu9VrMsC03nz5pm\/\/2Q7KIGJ0RM98cQTpV69eioxwAAseSycc4tphQoVKEqaNWuW0tK0aVODA+f8xmJfueNk1ILD5JrpB5JKnt9ximw\/a7SMWFw5ofq1asueg+9C\/P1N8H7SOM75\/Z4ma31gCabJKEpgkvGpap\/8CDRu3NGnFbvHf5+IF4ne5knzf5e5y7YJA2eyyf7i5RKuX7xP1coUFYRrFUUgURFQApOoTy6J2122bFnp0KGDcE7ibuZr1ypWrChXXHGFcA5VMQRj0vz1MnraipgJ5IU6bz27hiy6s3nSyNs9Ksl1FX+Sb4Y0Scg+8UziTXg\/I3lP463d2p6CQUAJTMHgnkC15n9TIS4QmPyo+ccffxQkP+oqyDoYGHr27Bm2CRCX\/pMWy+hpGTETSBEVt6iVZr76+fJPBjnlhBrSu8t5Cdknnkc8SqTvaTy2XduU\/wgogcl\/zLXGOEHg33\/\/lXfffVfWr18fJy2Kn2a0qFVa0JjESsZ1P15a1i4dPx3UligCikDCIxD3BCbhEdYOFAgC27ZtkmefHSbXXttSjj++lhx99NHZpE6dOvLMM88USPvivdJLT6nkIzA1Yybdm1SM9y5r+xQBRSDBEFACk2APTJsbHoFdu3bK2LEDZOrUF+WPP9bJgQMHwmfSFIqAIqAIJBcCSd8bJTBJ\/4hTr4MbN66SVat+9WleTpFrrhkq3bp1k+XLl2eTX3\/9VXDwlHoI\/ddj7FOs3Qu2L9bg9r8UeqUIKAKJhsDWyemyaVxvI7UWPCmDa2ySZPxXKBk7Fa5P+AaZPXu2GdDatWsXLrnGJxgCxYqVkEMPLSrnnNNTSpcuG7T1hxxyiFx33XVSt27doGmSPSKTwGQIK48Q7ulz1bSinFQUgZwjoDkLBIF9mzJk65ThsmPWBL+cXXZngbQlrytNSQKT16Bq+QWLQLlylaVBg9aCHUy4lmzevFmQcOmSKf7zpdvEytxlW03XMNrF0BZ5t19DNbg1qOhBESg4BCAiu3+aJcEkWMuKlK9hojiX7\/+8LG98jdyfUd6EJduhULJ1SPujCBQuXEQuvvg6+fLLD2TNmhVBAdm9e7dMmTJFUmkVEk7lzh+\/UKywXBqAWNaMoS2SJKuF6JaKIpCQCKA9WdWvpvye3jaohOoYxIX4km16SYk2PeXjzSW4TToplHQ90g4pAj4EDjnkUEET89JLY+XVV1\/NtgKJVUlMHb344ou+1KnzH80LvYWwoHWx0r1JJYJVFAFFIA4Q2Lcxw7QCLUrRum3ES0yCIAeIS0kfcQkSnTTBSmCS5lFqRywCrEJ6\/PGbZc6cd2xQrs7YhXQct1AajPwycomztM0eXCgdJ66RKQv\/MFjcenZNea9\/Q7+o1sXAoockRABNRjgJ1e1weYmPdf60ruly9JQDUm38CqmcPtNTQtVJHGVwTmZRApPMTzdF+8YqpOXLf5Tq1Y+Tnj1vlksuucQYbAeuRIp0FRIrcxCITCLLuu37\/G+EGun6odCLJEcAW5JwEgqCcHmJz8v8ocpO9TglMMn5BiRkrzC6HT26r0yZ8ojs3\/9vrvtw8cXXS6VKVcVxHM+yft\/xrxxS\/WT5YuUu\/yocVuIEyudL\/zN0zc1ePrOvP05GN9wovQ6ZLj2cqTL8uJUy45raUe2js\/COZvL+FZXk4z41o8r31c0+bUvPKv59e1Tj4vlKaGCcIwBZ2OFaXWOvMXQN1nS0GOEkWF7Cw+UlnnTBhPhwEiyvhodGoFDoaI1NBQRWrfpNIA4jR\/aSMWOuk379WstLL42Wv\/7a7tl9wom\/+eZz5bHHBgrn++67Sn7\/fYVn+r\/\/3i0ffTRRrr++rdx5ZxeTfunS77KkzchYLHfffblkZPwsLVteIIUKFc4SH+7mwIED8ttvC+Xhh2+QYcO6y44dW+Whh\/r5+nWTvP\/++77zaJk\/f77xCTN9+nRTHAatj\/5SWsb+VFLwgRJMJs1fb9JjN5ITqVSysEx6+hEZ2PcKad6wroxKv1MOP7SQPHzf3dK\/1yWyb9u6sPvpbFj2gzz50Ajp3L619LjoXNm2dmnYPIFtrVyqiD+P6ZAeFIEEQ2DjQd8m1seJPWPsuryLI\/g\/CewSdiThJDCP+z5cXuLd6QOviQ8ngXn0PjIE8obARFa3pooDBGbPflNuu62TNG3a3kcuJsjAgY\/LTTc9KrNmvS533HGxbNq0NksrIQajR\/eRuXPfk5tuesxHSsb4iMdkk+auu7rKkiWLzLU97Nq1Q8aNGyzPPXe3tGvXXUaMmCzXXfegYKNy33195Mknb5f7fOTnhRfukQ0bVsmFF17r05rUsNkjOtOmsWNv8LW3s\/z44xdyxRW3S5cuA3xE6Hx58cUffOU\/KB9\/\/LGZSvr666\/9ZTIdVGTDD1K\/yErp3qRiWMFuxJ85iovJkyfLyy+\/LMOHD5cTTzxRihUr5mvrHdKpUydhP6Z9+\/6b2glWbJMmTaRr167BojVcEUh6BNC+7PlplulnppFqL8l2btvLxOshNRBQApMaz9mzl6tX\/2Y0LUceWU0aNGjlT1OnTgM566zLZO3aZT7iMVzQoBD577\/7ZNKkh2Tx4vnStm1nOeqoWgRL8eIlfIPxtfLPP38bQrJ16yYTzuHjj1+WL7\/8UOrXbyXt218ujuNIhQrVpHbt+nL11aN8co9PI\/OElC1bUY4\/vomPdFxAtogFgjR27AD54osPfO052kdW3pEzzrhEOnS4Uvbs+UsWLpzlI05nyeuvvy4tWrQw5U6f94OUGThTJn29Toqs+1Za1Txc8H8STtBomAKiOOzYsUPefPNNSUtL8\/W5tj9niRIlfNquMTJv3jw55phj\/OGhLipVquTDunioJBqnCCQtAmgxMGzlzDJhLyEu0QHQ9keOgBKYyLFKupQQC+xOqlSpLSVLpmXpX716LYw3259++kpWrPjZxEFovv56mgkn3gQePFAGhGblyl\/l228\/NaE7d\/7pG6Cn+a4dH+G5WA47rJjvWmTfvn9k796\/pXDhIuZ+0aLZhmhceGE\/3wBdwoRFcoBQMZX1\/fef+7QaJaRPn5FSvvxRRms00jcdtnTp975ppP5y3HFHS+PGjWXu3Lmm2MnPPial3\/4\/Kf1OHym68jMpV+JQE54XB3zMrF69Oi+K1jIVgZREAHuSlOy4djobAkpgskGSswDH8TYUzVlpeZ8LAsFKHWrC7b4lE9wjEIFSpcrK7t07fdMyXxLkIyYz5c8\/N\/u0JZWyTfOUKHGEWfUjckDmz\/\/EEJQ\/\/lhrpoWOOKKsVKt2nNh\/TEsVLXq4HH54KWOr8sYb43zakY5ywgmn2CTm7DihMV2y5Dv5\/PN3TdpGjdr6iEpjc12mzJFSrlwlX1v\/MPfhDidUOjxckhzHb9u2TXbu9Hbj\/e+\/\/\/rI1qag8TmuVDMqAjlGQDMqAomDgBKYxHlWMW3p3r17fVMsu0yZBw7slwMHDphre4DQFCqU+XqsX59h4plyIh6CcOihmdoU7q1UrFjdXK5evcQ3KP8phQsfYoxx0bxAcIiknm++mS4nndTCxE2f\/qqZourY8SozvUSaSOWLL943BIv0EJjCBzU6nE88sblQ5nk3zpZtHZ+XbZ2ezZSO\/5Ot5z1urq8c+4FcdNFFZon11VdfLVdddZVcf\/31csopp0jPnj0FexnaS\/mc16xZI0899ZRveu0sGTx4sI+ozZcuXbpInTp1fOTpOBk9erSvPbtJbohJt27dpH\/\/\/j6c98i6detMXa1btxakYcOGJl\/Tpk1903gvmTzuA16CJ06c6OtDR98UW+ZS8BEjRsiuXZnPzJ2W67Vr18qQIUOkXr16xmnfcccdJ9xv2vTfdN6WLVsEe5wLL7xQrrjiCvnhhx9Mn4\/zpaUPAwYMkK1bt1JcFtm+fbvpG+1u2bKl79mdJJRNne6ENh32OkcffbTpX+\/evWXFCm\/jbndevVYEFAFFIFoEMkeoaHNp+oRH4LDDikqpUmVMP9atWxF0xZFJ4Dvs3v2XbN683nclUthHFBzHMddeh507t8mWLRuMlqZu3aY+grLLaFpIy3TPhg2r5eST2\/kGtp9k6tQXfQN7f5\/GpDLREQvO6thxmgxs3li5ck0u\/cIU1\/HHnyLfrc0c8Ls3qZhp59LjJBl\/RSNzfUfH46RUqVIyduxY6dSpkzzzzDPy2GOPyVtvvWUIx2WXXWbCIC\/btm3ztXeFvPbaa7J06VJjFPzQQw8ZcvHEE0\/4puBKGnKDvQuNKF++vPEAfM8993ArlStXNrYwn332mSAzZswwRMBEBhywm4FQsVrqhRdekAcffFBeeeUVOfbYY33PYHNA9GNNuwAAEABJREFUapFFixYZA19sab799lv56aeffNNpfQxZufTSSwWiASFiKuudd96R7777zggk5OKLL5ZJkyb5NGTV5L333jP9p79y8N+PP\/4oF1xwgW\/a8FCZNm2aT+P1uekzRAjy99tvv5mU1IGR8d9\/\/y0zZ86UJUuWGKPlL774wrSNNpqEYQ4anZgIYGAbTsL1jPyyZY3I1jWCJ1rurYTLq\/GpiYASmBg892OOKSkzZoz1DQS3h5VDDxXfYJBVJk0Kn8+miVX+YsWK+KZtzvb13pFly76Xp5++MUvb33hjlGzf\/ocvXnwaim\/l9deHy\/r1y8x92bIV5IgjDvf3w7Zt2bIvTPyePbt8xORxmTJlmJQufYiPJJSQYcO6+rQbLX3TUXOkX7+7hfrfe+9JqV37RF\/5n2Wp+803h8unnz7iG4DvyhJOPbb\/Bw7sEqaoqPCff\/b4CMUTWdLOmvWU\/PrrJ3L0hnvl4l0vyRE\/vCg\/vPWoWWlUduv3UuvAKpkzZ44hBkcccYSvvk\/l9ttvNzJu3DgzoKOBQqty5ZVXygMPPCAfffSRoDmhzgYNGhhygwbl008\/lTJlyhgtFRoaiIEtCwJCejaMvPfee7k0cqivI8WLZxrkTp061dRr83Tu3NnY6xQpUsTUS\/hdd90lkAUMeU0BBw+QHTRHtJX4YcOGCZqalStXCuUvW7ZMbPshE6eddprJedRRRxnCdc455\/ie0xTTfiLeffddo12izptuukkuv\/xyvwaJcklDn8uWLWu0TKtWrTLaGMjeH3\/84Zu2+1MgbbQXMlWyZElDuq677jq59dZbyZ5NqCtSyZbZFxBpXtL5kmf7T3ikki2zL2Do0KFZnl+osnzJs\/0PlT4wLltmX0BgmlD3vuTZ\/odKHxiXLbMvIDBNqHtf8mz\/3emHDhsqfBBwdoff0udSg3G2zL4Ad7pw177k2f6Hy+OOz5bZF+COD3ftS57tf7g87vhsmX0B7niva\/6m0Zz6kibd\/0JJ16MC6NCePdt9A3E1Oe205mHFp+H3qfkli0SSz6aJZf7LL28jLVu2EL64f\/nlB6levZJpf716x\/rIwTrfwJU5HVKv3gm+dKdIWtoRBt20NO\/2n3jicSb+kEOKSKNG9UxZZ5\/d1vfFPlBGjRopt912q4wZc7s0bVpC1qyZ6tNkLJA777xB2rVr7SMGdX1aoK0+bcLX8ssv30vZsiV9BKuJKcP2nbPtv2\/WwxAoKqS+xo1PypK2Ro3KPlIy1Wg7flqyQqoc11CaN29OcmnVqpXwR33HHXf4tEN\/GwNfpkeaN29u0pDurLPOMlqT\/fv3+7RHO8y0EuGmAN8BDQsriXyXJs9xNMh3c9hhh5nySIvY8KJFi\/q0Tif7UmT\/X716dVMG6dGiQATKlSvnw6WdP5w4iAOkxl0C2pQ\/\/\/zTp\/HaImhsWC6OYLC8Z88ek5Tny9QSwjWBpUuX9pHQzOdJ2ZaYEcc1YWinKBtsEMKIp40TfdNbCCu7atWqZer\/66+\/ZPbs2T4y+bERNDHkJw9tOemkk7jMJpQbqWTL7AuINC\/pfMmz\/Sc8UsmW2RfAlFlu8keal3S+6rL9JzxSyZbZFxBpXtL5kmf7T3jDUv9IOCFdtsy+AMIR8tcrvksQrgOFNL7k2f4THqlky+wLiDQv6XzJs\/0nPFLJltkXEGle0vmSZ\/tPeCipVq2a0Shny5gEAUpgYvAQsW\/o0KGDdOzYMax4VRdJPpsmlvn5Oh4\/frzwpY1G4OGHHzZf3x9++KGPsLQUBuNDDjlEunXrJnypM6hS\/zbfdApTBVwjtm0MxNxTFl\/6Ntx9Jh680HLwdc+GiiwxZhqHqRkGxaefftqnFVomDRo0kPotzpSnVlXxS8dxCwXp9fIS2XygJMXJnr3\/yuurj\/CneWJZeXnyjRmGeDiFD5GSR6RJhw6Zz4YM+GHp16+f0RZwz9SMu41cs\/1As2bNiDZ2JwzghJuAgAPhxx9\/vAkFI7AiDMGehojDDz9czjzzTC6zCXlJi4Ah+EIKmLohzIpXfqZ4MAbGduebb74RKwsWLPARxKU+7dZyQyawewn2FUb5ge0kDEJCY6mXe4R7x3HkhBNOMEQQLGvWrCkYKp933nn++mkHbWAqiS0cuEdLQ\/5AodxIJTAv95HmJR3pA4XwSCUwL\/eR5iUd6QOF8EglMC\/3keYlHekDhXC3NF4wRoLJuvS2grjLIG\/3h16XcEI6dz57TThC\/i6jX5WWNz4snLl3C2lsHveZ8EjFnc9eR5qXdDaP+0x4pOLOZ68jzUs6m8d9JjyU8DvK36A7T7JcK4GJoydZEE1Bi8Dgx\/QCL\/kvv\/xibDf4coak4Hitfv36wgBcsWJF00QMSffu3Wuu3QemM7jny50pBq4DBQ0ANhcMfBAjBrhBgwYZsoEqnqkNCA1GpUx\/4C137rJtEihfrf5btjilM4v\/d6\/8tHSl2DRfL14l27dulh2nXCcVz7xWGtetI4E+XNBkQLQcx5GMjIzMcgKOkAmCGMgjcTZH2twK+zOBUbTlMGUUbZ5Q6SEkEE3SeD1rwgNl48aNPi3aX4HBep9gCOAsLpwkWJe0uUmKgBKYJH2wuekWpOKDDz4QtC+9evUyBqqO45ipEcrdsGGDf7UN91bsIIr2ABJiw93nRYsWCQagrM4hDWSGlTL1GzWRZVJFcNs\/ZeEmWbzxH5n5yx+Ct1zyY4T7br+G8p80ksG9OknhIkWIljNKrpa3rq5n4h++oIo0qVtLXrm9q1zW7CgTH3hAuwQpIxyj11CkAfsW0pM2vyQYxsHqB3tIh1c8BCxSEmLz8+yxoeEefDiHE4yEWYnklQ4yjHjFaVj+ILD7p1myY9aELOJVc6X0mRJOvPJpmCKQ3wi4CEx+V631xSMCaFFGjhxpDDRZodK+fXt\/M08++WRBs4KNBg7a\/BG+CwZPu1z29NNPFzQsvuAs\/0nDVBHTMUwxoeWB0JBo9eF1ZeBbGWZPogFTlsinP66Rpxb+bQgN8WhQWtYuLW7p06W9nHrQruXHb+ZK0W3LTHy7hjWleOF\/5cTyhaVqWlGy+4U20E4G5wMHl45DFuiTP9HBC0gBl6hgMfTlOq+F6SzHceT333+XjIyMsNXZ9EwleRENnidGxfQ5bGGuBBA2a7+D8TJLyF3R5nLhwoVm1RIrrNLS0gSD3i+\/\/NLEuQ+Qp3vuuUe++uord7Be5yMCkJfffdM\/du8ge\/ZqQrG6bSSceOXTMEUgvxFQApPfiMdxfQzurGLBALR79+7CShK+xG2T0ay0a9fOTPewDNiGc2bq4+effxYMPElDWKBgXMo0FfYnlMsXOb5JSpQoIcUq1jLJ0bR0qLFfjjp0l1zU8kSzaoiw7k0qSeA\/8g0cOFAwqKXtWOCjPeKeqSiWHmMfYvMxCGN3QzvefvttgYRBYsiLjYZNxxlyxXQahA0iR1h+CFN2TJ9h9MoKJgZ\/Wy9hgVNZNj19uPPOO81Sb5ueMJ4neFCmDY\/03LZtWwFjyM\/NN99slmPbvNi33H333dKhQwepWbOmWANhVloRZ9PRfuydIGTWzsbG6Tn\/EGA5MrXhar9km17+PYQIU4kBAlpEgSCgBKZAYI+vShnEFy9eLBhZ4qMEY14Gp0AtCnYjLIflyxyfIQxs9ISB8n\/\/+59ZzYQzNL7ICWf6B4PbBiO\/lJNufVcG3\/2QbK52tlz48noh7Mzxv8g6KSc7\/v5Xvl+zkyyGsNTf+51cffEZMu7y+mL3J0IDYxIEHLDPefXVV4Uz7cE3yahRowTNESSFvrC8+LbbbpPzzz\/f+F5h+TRaCcgAdi4sQR4zZox\/gAYP\/L1AyrAPwtCYakkPieCa1Umk4xoBA86Ec7Ziw915bZxNa9MQDtkAQ+xzmMa79tprBWKI0znCIQKku\/\/++wVtFs+IcNJjM4QBcd++fYV2Q0DQLGGk7TgO2fw2KrZuE+g72DYQTlt9QdKoUSPheTuOIzj1a9OmjTHubtasmVmaTT2QF9pw4403Gu0cy8W7desm+J+B9NAG+gGRIh3lquQfAhAXjG7RuFArmhX3HkKEqSgCiYpAoURtuLY79wjgdZXBH78jfL337NlTZs2aZQb6woULe1aAZoM8LDNmAGOQQqvBYP7GG2+Y1UrujBjWrtqyWzYteE\/+LlJSNpRpJBAbZPX2f2VrzXPlQKEicsjGH+WoEgfk50+nCF\/tTDO5ywl1zSDKZo2QDgbMTz75RPCNsnbtWsEuA40Acdu2bZMXX3xRMFimvazKmTJlinHgVr16dWGlDgM\/3mMhdBAgPNY6jmM81OLkbvr06aYpaKDwVwGheOmll0y5RDAlhs8Ypn9wagcxIpx0LNuGCFA2AzrtIo42sRqM58E95AvtC\/s3sSyZFQb4UKFdTBmhEcI\/DW1FQ+JOj8aJNlI2badcVpvhyI6l47SVOmw7wYgpIttODHcpmzY6jiP\/93\/\/J48++qjRslA20201atQwvmMgS5SFMM328ssvC9OHvDtMF4ER2jiw570hnUqeIBC0UDQuFfo\/748vWvc0\/7VeKAKJjoASmER\/grloP4MfS18hHgjaC8LCFVmqVCnjlAyigJdYPNfixZbpI\/JCTkZPWyGsIOK+aunD5KMnhsn0t16ShcNPl0V3NvfLt2Muk7feeV\/6nVZFOheeI\/jUGDBggKDtIW+kwqBJXgbbzz\/\/3HiCZbrqkUceMYQMjUWPHj3MkvD09HRhcGXJNhoP2k37IReQlAkTJgiaG8Jt\/dh4oMlgWoly8b\/CVFilSpWEcrknnPjhw4cLgzx4otEiHIGoMI3Csmm8+DLdRTh5KYs6qM9xHON3BnJFGjQrLG1HgwTZgCDhq4U+e6WnTEiaJTikQftBfpzLEW\/bCbGAiNh2UheY0UbyUQfvCHiRj\/ZAYN3YkA4hDBwpm7TYyIA1RtDEq8QWAbQrWyenS6AE1gKJYQNEhOmjwHi9VwQSFQElMIn65GLQbuxQ+DqPQVFZiug\/abGMnpZhhAinUCFpUKeKHFe1nDAVFCgNj6kqd95ykwwbNkwYmB0nc7qDvLEQVhuhZWJaDGNSNBNobRwntvXEoq0JWYY2ukAQ2Diut2ydMjybeDUGEoN4xWmYIpCoCCiBSdQnFyftRtsSKKu3ZnqA7d6kotx6dg2ztLmgmsv0C6SloOrXehMLAbQaVsK2PMi+PZHkt2lCncPVz35BpEGrktZlmFghTEURSAUElMCkwlPOoz6iacEYN1AgNFR569k1fQSmptG6cJ9HosUqAjFBYPdPs2RVv5p+CVfougF15MC9rYWzO5+9DpXfpgl1DpWfOMgP57SuPvLSNV3SDgphKopAKiCgBCYVnnIe9REDXYoOnBJy3xOvoggkAgJ4n6WdTLUgXIeUtCoiPiGtl4TK65U+MCxUfuLc6blXUQRSDYG4IzAYNLKChKW6Rx99tPEvMXjwYLOnS24eDkaFlIONBeXiF+Pcc88VVjEWkDsAABAASURBVIqwQiPHZSdwRrzeokXJqVhNC95x3Ya59jqBodGmpxACaDIwhN3982zT65JtegoGr+YmxKHyo0vEue0z4Uz6QAmR1ZQfmD7wPlR+4mx6rlUUgVREIG4IDMta2cQPY8tPP\/1U\/vnnH\/M8cIPOChk2tmMFhgmM8kA+8lMODsrIzpJQVkuwDJgVIHgsJTyVJHOl0HqByOREwAptC2cVRSBREdi7MUMwhrUamETth7ZbEUg0BHLb3rghMCx9ZWkpjrRYloofC5Z0Tp06VZo2bWoccLEkE6dk0XSa9OSDCOEtFKdalPv9998LGhkcgOFrA98X1oFXNOUnYlo0J58v3eZv+q1n1\/A7jLOO4yI9P97t+Li2ccHhHh56\/Z3Vi7hGAG0ItijBJFzjg+Wz4aHyF63bRnDyhi1JqHQapwgoAvGBQFwQGKZwnn32WaN1YVoHF\/b4xHAcx7imxxkXnlbZ9A\/38GhrIoUP513kY9ksLs3xb+E4jnGRfs011wg+RygL9\/J4cuU62YUpo\/PHLxSIDH3t3qSS8YDbvUnFqM\/sTUQZKopALBBgaTB79gSTcHUEy2fDvfLjnfboKQekcvpM42LfK42GJSMC2qdERyAuCAyu0nGwVbRoUcFHB\/5J3MBCZq688kpxHMc4IGNPG3d8sGv22mGaiHhsX6pUqcKlXxzHETYVpHxcrq9cudIfl8wX1vi2Ra3SgvZFp4GS+WknVt\/s0mC0IV4Srjdeedxh4fJrvCKgCCQOAnFBYHBrzl4seC9lw0Av+DDqZWM9XJ0vXbrUK0nIMKamDhzcfdidELKEt1F3WLJeo3FhbyLbv\/f6N\/QRmJr2NiXP7OKMpGTn47DTaENoFu7v0YgECnGhJDB94H2ovPkdp\/UpAopA7hCICwLDvit0g00A8ZrKdaBAXsqVKyeQEDbZC4z3uj\/ssMPkxBNPNFGsblq7dq25dh++\/PJLQftC2eyH445Ltmtc+1vti2pdMp\/uvHnzZP369Zk3eixwBLBBYTqHJcIF3hhtgCKgCMQ1AgVOYJjm2bZtmwGpdOnSAukwNwEHjG2JJziaqZ4uXboIWh3sW\/r37y+QJUgQ9bKnC\/vKOI4jnTp1ktq1a0si\/0PDEmo1EfH0r3uTimYvIq4TXdgAEfsljLO9hKlJ3hevOOyp2A8p0THIWfvzJ9eOWRMknORPS7QWRUARSDYECpzA7N27V3bt2mVwZTrHXHgcsI+xm8KRxyOJZxCb1TFQsSEdtjacITQY87I6BcI0atQoszmh4yT23jgY54YSyA0gJZP2hXdmwoQJwsaFXgJBQcvmFXfnnXdKpPZU4KYSPQKbxvWWcBJ9qZpDEVAEFAGRAicw7ocQagqH3YkhMe70kVyjbfnmm2+EZdP4fgnMg18YbHC2b98eGJVw9+49iNCyBJdKBd63WDWgRIkScvXVVwtEBjupbt26iVuaN29uNHDuMHvdrFkzSRX7p1jhHW057NMTTqItU9MrAoqAIgACcUVgaFAsBfLyzDPPyA033GC+tNu3b29WMeEHhqmkxx57TNDqTJ48WS677DLxspGJtD18yVvBn0x+CpqVMgNn+pdFDzqjqjzSpU5QqVyqiORn+\/K6rkaNGgnPdsSIEXL33XdnEcgKK9ACw7lHO9O9e3eB2OZ1G1O1\/LSrn5ZwkqrYaL\/3JdXvUDw9TzsWcY50DEvEdHFFYLBVCAYiL8eePZm7HAdLExiO3cvzzz9vBqhLLrlEHnnkEcEfjOM4xtaG6aQnn3xSMBBmuTW+aCA9geVEcn\/ppZdK69atjYwcOVJWr16db\/LcZyv8TYScOH9tyre686Kf\/NFhWBtp2Rhht2zZUpgqCsxj7asCw7mnnrp16wpemLlPZqGv0WAaLRarnrtZVnU\/RDhHmzdR0+c1pomKS27arZjGZtzgw92OR4xN\/gEiyS4KFXR\/UP0XL17cNCOUbQvkZcuWLSYdecxFmAMDGj\/aEJSePXuaaYbALA0aNDDaF8LxyMtgyHW0gjEwtjZIr169pFKlSvkmTK\/R3jf71pNvhjTJt3rzqo9HHnmklC9fPqp+dOzYUdhDK7BNGHGjbQkMt\/dsMdGhQ4eo6rJ5E+mcE0yD9a\/wm7dJoBRZtYBXUEr6pvSC5Uu28FhimmzY5LQ\/imlsxg3GIMYixDprNX+gsTjEURkFTmAwomWwAhO+llkdxHWgsDcS8YSHspUh3gpf1lxjtIsxL9deYpdaQ14gPF5pwoWx3UGzZs0EqVGjhmCvk19iCQzn\/KozL+spVqyY0ZDlZR2pVnYsMd331WuSTX793PyJJMs7GMn7EUtMI6kvFdIopkVjMnYwBjEWIYxN5o8zCQ8FTmDA1BIInNSxZxFhgbJ582bjr8VxHDn22GMDo0PeQ4qwcwiZKAEjWRaNJGDTtckxQIB9g8JJuGpykh9fLcFE9xEKh7jG5xIBza4I+BGICwLD6hFWk2RkZAgGtv7WuS6wUYHE4OwuUn8tFStWNCXYeVVz43Gwnn1xZmfzeCSLqyCIS4ORXwpindPFVQPjuDELFy6Ut99+W\/D+HMfNDNu0Vf1qSjgJV0hO8odaVRSuPo1XBBQBRSBWCMQFgTnhhBOElSTYubzwwgsSaAuDs7LnnnvOeOHFWJP51UgAaNy4sTB1BPHBqCmwXMqA3EyZMoVLadWqlUBizE2cH1ZtyTRoxqcL0qJWaUn1jRWZMpwzZ47gtI5dyL2eN48VuyeWT5966qly880352r1mRTgP7zVhpNwzctt\/nDlJ128dkgRUATiBoG4IDDMe\/bo0UPwtsvgg8dcLNlZEcRAdM011whbAWAr07t3b2GeXVz\/GISOPvposzHjxo0b\/TFVq1Y1m0M6jiNvvvmmUC7lUS6DG0a7WGizWgnNDoa+juP488fzxeqtmQQG4rLozubyXv+G8dzcPG0bz3L06NGGBPMMcVrHsmoI7NNPPy27d+\/OUr\/jONK2bVuBOPNesKQ+S4IY3uz+aZasS29rJFSxNo3XeevkdM+s1cavkHDimdEVmNv8rqL0UhFQBBSBfEUgLggMPT7jjDNk0KBBxrHY9OnTDRnB+JaBaP78+cIeSenp6XLMMceQPGLp1auX3HTTTVKoUCGhXMqjXOxoIENoYCBG48aNM0usIy64gBOu2pI5KKN9KeCmFHj1eOJ96qmnzHJ53pMzzzxTunbtKlWqVBGIDZqWJUuWZGsn70S2wBgHbJ0yXPb4SAwSqmjigwllLO\/iyO6fZoUqQuMUAUVAEUgpBOKGwDiOY5bB4lTu9NNPN9oYngQD0sUXXyzvvPOOnHPOOQRFJSy5RvPy7rvvCstljzjiCJPfcRxhNdOQIUPkk08+kfr165twPSQWAlu2bJG33nrLTC+2aNFCPvvsM8G3z3333Scffvih4JMHggqR\/eKLL\/K8cztmZd37x1aI0au99jpXSp8p4aRY3TZeWTVMEVAEFIGURCBuCAzoO44jeE3FXgWj3eXLl8sPP\/wg+Fhhiog0XvLggw8KaZkSqlChQrYkjuMIex89+uijggEnaTEWnjlzpvTt21dKlSqVLU+8B\/xnA1Ms\/5oahzWtXbvW2LBgBH7jjTdKWlpallZCgE8++WTzDg0dOlQ+\/fTTLPGxvgnc9wetCnVga8I5mEBOwkmwvBquCCgCikAqIhBXBCYVH4D2OTYIQFyweQpWGtNITzzxhIwZM8ZoZvbv3x8saa7CS7bpJV4COclVwZpZEVAEFAFFIAsCiURgsjQ8VW\/QvHQct1DY\/yhVMXD3m1VmiDss2HWdOnXM9BKGvUw7RUtiMKZl2TH2KAj3gXUxVeQlgen0XhFQBBQBRSB3CCiByR1++Z4bAmP9vmDAWzWtaL63IZ4qZDPOCy+8UDZs2CALFmS6sw\/VPsgOBr9Tp06Vb7\/9NlTSbHE7Zk0UHL8RwZRQUbVJAQoVRUARiHsEkrOBSmAS7LnOXbbVtLh7k4rC8umWtUub+1Q+YKDbuXNnGTZsmGAHdeDAgSxwQDrQllgpMnOc3HNWdWlUo7xJt\/PzSeYc7GDzUQ5pMLZl+bFOC4GGiiKgCCgCBYOAEpiCwT3HtX6+dJvJ271JJXPWg5hNOkeMGGGmh1566SVjmL19+\/Ys0LAU2S0HPhwtd5T4SpqU\/Et2zg1DYKYMF\/LaAg+pUMNe6lkRUAQiQECTKAJ5gUChvChUy8wbBCAvTB8xdaSal6wYO45jVrBh34K4V5Yx3ZPWZZgESpVud8n\/7rhWzutzS9bCAu7c+bBvobyAJHqrCCgCioAikM8IKIHJZ8BzU52dPsL7bm7KSca8TPNgWBusb2wyGEqC5SPcnY8VRoSpJBIC2lZFQBFIRgSUwCTQU8WAl+a2rJ3V1wlhqS67f55tILB2KuYmxGHdunWC8W+9evWMh+YQSTVKEVAEFAFFIA4RUAIThw\/Fq0l2+sgrTsPEuOv3wqFjx45yzz33ZIvCmeHPP\/8sf\/31l8yYMSNbfCwDtCxFQBFQBBSB2COgBCb2mOZJiaOnrRCrgUn1pdOhAI7UPqVp06YyfPhw6devnwwcODBUkRqnCCgCioAiEIcIKIGJw4fi1SS7+\/S47sdLy6iWTnuVljxhu3+aJUi0PWKPrG7dusnNN98s7JUUbX5NrwgoAoqAIlCwCCiBKVj8o65dDXizQvZ7eltBCI1U+0JaFUVAEVAEFIHERiDPCUxiw6Otj3cEWBWER1ykZJueETV34cKF8vbbb8vOnTsjSq+JFAFFQBFQBOIPASUw8fdMtEVRIIBflsrpMwUpcs4gmTNnjnzwwQfy22+\/yd69e+W9996T22+\/PUuJDRo0kMKFCwsbPDKFtHbt2izxeqMIKAKKQBwgoE0Ig4ASmDAAaXR8ILDON1W0aVxvz8ZAVEaPHi2NGjWSnj17yvXXXy\/t27eXxo0by6effmqIjDuj4zjStm1bOeGEE+TNN9+UxYsXu6P1WhFQBBQBRSABEFACkwAPiSbaFUh44eU+6WXrGtkxa4Jf6C\/3u3+axWUWmTBhgrBB47\/\/\/iuHH364nHnmmdK1a1epUqWK0cBMnz5dlixZkiUPN4UK6esPDiqKgCcCGqgIxDkC+gse5w8oVZt34LXBsvXJPoLWBdnjQVzAZsuWLfLWW2\/JgQMHpEWLFvLZZ5+ZPZHuu+8++fDDD2XkyJFmlREbPn7xxRdkUVEEFAFFQBFIAgQKJUEftAvJhsCWNSLL55leYaRrhT2JAneAXrt2rSAlSpSQG2+8UdLSsnopRiNz8sknywMPPCBDhw41U0qmYD3EOwLaPkVAEVAEQiKgBCYkPBpZEAgc8E0fUS\/LojHStcKeRIR7CcSlatWqXlEmDIPdJ554QsaMGWM0M\/v37zfhelAEFAFFQBFITASUwCTmc0vqVju1molz\/3KpcOcnYft51FFHCRI2oS9BnTp1zPQSu1Uz7RSSxPjS639FQBFQBBSB+EVACUz8PpuUb1mRCjXJZc0GAAAQAElEQVTCYlCmTBmzKeOGDRtkwYIFYdNDdjD4nTp1qnz77bdh02sCRUARUAQUgfhEQAlMfD6XLK2yK5CyBOqNHwEMdDt37izDhg2T2bNnCwa9\/kiPC7YOGD9+vLAfkke0BikCioAioAgkAAJKYBLgIdkmpswSatvhCM\/sazRixAgzPfTSSy9J3759Zfv27SFzYzMDiTnttNNCptNIRUARUAQUgfhEwJvAxGdbU7ZVVgNTNa1oymIQruOO40jDhg0F+xakVKlSJkvHjh3lnnvuMdeBh5IlS8rzzz8v7dq1C4zSe0VAEVAEFIE4R0AJTJw\/IG2eIqAIKAKKQOIgoC3NPwSUwOQf1jmuafXWPTnOqxkVAUVAEVAEFIFkREAJTAI91VSwgcHr7rrLi4l880a+PJkff\/xRkHypTCtRBPIcAa1AEUgdBOKOwHz33Xdy1VVXyXHHHSdHH3201KtXTwYPHizLly\/P1VNhw7\/33ntPzj33XMEfCGVjMzFkyBDjyTVXhedhZuxfPl+6NQ9riK+i927KyGxQ2lGZ5zw+zps3T9avX5\/HtWjxioAioAgoArFGIG4IDEtfMb5kOSw7CP\/zzz+mr3\/99Ze88cYbcsEFF8hHH31kwqI9bNq0Sf7v\/\/5PBgwYIL\/88ouw6R9l\/PnnnzJ58mRDaqiTsHiT\/pMWy6T5qTPA7tt4kMBE+CC2bt0qH3\/8sXzwwQdRyyuvvCITJ06MsCZNFgkCmkYRUAQUgfxCIG4IzOeffy4PPfSQ4B21R48exinZsmXLBIdj+OuAyKSnp8tvv\/0WFTY7d+6Um2++WSifVScPPvigLF68WJYsWWIGrypVqsiOHTvMShX21Imq8HxIPHfZNlNL9yYV5daza5rrlDiUqRJRN1lCPWHCBLn++uujljvvvFPWrFkTUT2aSBFQBBQBRSC+EIgLArN792559tlnBa0LUzx33XWX2ZTPcRw55phjBH8d9evXFzQpfDWjrYkUxpkzZxrywmZ\/7IVz0UUXyWGHHSaFCxeWVq1aybhx4wRiwxTVjBkzIi02X9IxfURF2L6M6368cOZe5T8EeK5XX321QGSOO+446datWxZp3ry51KpVK0uYTdOsWTPzHvxXml4pAoqAIqAIJAoCcUFgfv75Z+PWvWjRonLFFVeYwcgNYFpamlx55ZXiOI5AMiL9aoYYMf0E4YG4MJi5y+X6+OOPN0Tm8MMPFzQ+dnqJOJXEQOCUU04x04D33nuv0aTh98UKZKVJkybZwol\/+eWX5bLLLkuMTmorFQFFQBFQBLIgEBcEZtGiRcJUT40aNczXcpYWHrzBqLds2bKybt06Wbp06cHQ0CeIDuQIYnTeeecZAhSYo0iRIvL444\/LDz\/8IMOHD8\/VF3lg2bm9txqYqurALiSUxYoVEwgqU4MhEwZEOo4jbdu2Fd6PgCi9VQQUAUVAEYhzBOKCwNiBp3LlyoImxAszyEu5cuXMPje\/\/vqrV5JsYStXrpQ\/\/vhDKlWqJJCjbAniPGDusq2mhS1rlzZnPQRHgOlAtC3BU3jHsJVAy5YtvSM1VBFQBBQBRSA\/EMhRHQVOYP7++2\/Ztm2baXzp0qWNfYq5CTgceuihQjzBEBPO4cRqao488kjhKx0D4D59+pil2SyjtlMLGzduDFdUgcRbDUy1MsUKpP6CqLTa+BVS+cXdImmRGfEWRBu1TkVAEVAEFIGCR6DACQz+WXbt2mWQwBDTXHgcUPOXKVPGxJDHXIQ5bNiwwaSAwGALc\/755xsbGlY0EbF582Z55plnhP1y8D9DWDyJJTA6hRRPT0XboggoAkmHgHYoIRHwJDAM8DfccIO0bt1aWKnRuHFjs5Inr3tYvXr1oFVgqwKJCZrAI4Ll0QTj4wUDz2OPPVZee+01s4SaaSg28qtSpYpZ3cRS63hbRj13WaZmqlqZonRDRRFQBBQBRUARUAQOIuBJYFi1g+0IUzt33323fPPNN4KdAIa2TLcECsubI9WKHKw3X08QmapVq5rl2EwbsYQabQ\/2D48++qhgX8MKpClTpuS4XRgMW9m3b5\/kVqz2pXKpIoLktrx4zb\/7p1m5xipU33igoeI1LvfvqmKYawzz9G9An0\/qPR87FnHmNzBZxZPA2M6igTn99NP9q3dY0YPzL8LRzCDdu3cX\/KvgEdXmy+k5lG0Lf4R79uR8U8P+\/fvLUUdld0+PfxlWKNHm2bNnC955uY5WLr30UqOxApuRI0fK6tWrcyVf\/7zCNKFSySK5Kie37cjr\/L+\/MERWz5qSpY\/80eHePxZ1Q8IBMhZlJXIZscQ0EXCYP3++5LWwepJ9tPjAy+u6UqV8xTTy9zbU3yGmEYxFCGMTv4HJKCEJDEuXmbqxHcffxlNPPSWTJk0yzt+4f\/vtt2Xo0KFSoUIFmyyqM5qQ4sWLmzyhtDiQly1btph05DEXYQ525RGO6nBm5pXccRzB0y9xaJZySmAeeOABwcke0qtXL7PyidVPORVrsMy0WU7LSIR8RXZskANPXirlCv\/txwybpfLly\/vvc9MPHN3xDuemjGTIGxLTSpVignW84IQvpzFjxhgfP\/j5ySvBZ9WgQYOM76q8qiPVylVML4v4vb3lllvMtjhef3e9fGMQYxHCFjqMb8koIQkMP\/xenWYzRBzAMa0EOfBKE2kYXnEZrEjP1zKrkrgOFLz0Ek94KFsZ4q2kpaXZy5BnS4iYIsspgYEEoZFCIE4Qj9zIpt2OaTP2L7kpJ97zmk76DsWKFTX+WGhvsWLFzGo0rnMrF154oXFil9tyEj1\/LDGNdyyY\/l6wYIG4Pyr4IVd5xf+RpVgkNhaQEt7xYH+LjEGMRQhjk+8nNin\/hyQwoXpcqFAhCUZwJMp\/J554osmBkzoMiM1NwIEVQ\/wwOY4jGOMGRHveQnQgJ2h20OB4JvIFQlx8J0HrwZcq1yr5hoBWpAjkCQL8cPMDrtLMLMZQHJIHB97tPPmjSbBCc0xgYtlPpqpQ9WdkZBh3\/l5l\/\/LLLwKJwdld7dq1vZJkC2PaiPSQl7lz52aLJwCDZRuHjYydziJORRFQBBQBRUARUATiE4E8JTC46J8+fXrYnp9wwgnSqFEjgWi88MILgsbEnQkD4eeee8544WXaijk\/d3yw64oVKxpX8cSz7w0GYly7ZdGiRfLxxx8bQ2WMeSFS7viCuv58aaYXXqaQCqoN+VHvvk0Z+VGN1qEIKAKKgCKQZAjkGYGBhGChHwlezM\/36NFD8Lb7wQcfCCuGsLBGO4L33GuuuUZwNIetTO\/evbNNXeHDBc+6LIvGEFcO\/nMcR\/7v\/\/7P7K+E9gbfNpAV2oZQF2WzzJrl1Ti6O5i1QE+rtuyRSfPXmza0qBWZHY9JnGAHS16KlK8hSII1X5urCCgCikBKI1DQnQ9JYFiKBTm4\/fbbxS3Dhw83Uz3Tpk3LEm7TDBkyRNq3b280G5F28IwzzhAs+vHRgtYGMsIUEOWwhJA9ktLT0+WYY46JtEiTjmmh8ePHm3wsJYWwYEODXH\/99caJHVNYOLrLrUGyqTAGBwgMxaB9SeZ9kPZuzNS+FKlQg+7mibz33nvmHc2TwrVQRUARUAQUgQJDICSBYdnym2++Ka+++moWwS0\/zuvQigTGcT958mRZsWJFVJ1yHEeuuuoqIS++Z9DGUADE5eKLL5Z33nlHzjnnHIKiFlZNUS7ECsNex3HMlBHXhNHmmjVrRl1uXmeomuS7UO\/5aZaBsNgJp5mzHhQBRUARiBwBTZnqCIQkMPkNjuM40rBhQ7M\/EUa7y5cvlx9++MEsh2SKKFh7HnzwQSEtjuiC+aMpVaqU9O3bV2bOnGm0R3je5Zow4oKVreF5h0Ba13Q5esoB4Zx3tWjJioAioAgoAsmIQEgC06lTJ5kzZ4589dVXUQsO73DRn4ygaZ8UAUVAEcgvBNi\/DW+qaJL5kPMSPvzOOussueOOO4wHYmz8Im0fjv+YpsdcgKl7Ww9lov1mAYR1NeEuE59dTz\/9tFxwwQXGzpB2Me1Penc69zUfjUzdn3TSSUJ6hDrT09OFKX532miusWOkrW+99VY02TRtgiMQlMCglWBKBxsStBrRSrt27aRDhw4JDk\/+N\/\/zpdtk0vzf879irVERUATiEoFLLrnEOKC7+uqr\/e3DCzqOzNA8L168WJjq54MTu0TSs6pz4sSJxlOrP5PHBVP9nTt3FvKwVQwEiMUXS5YsMWVCMCAXzZs3F1aIsrDCFoMT0j59+ph0kB\/HccxK0REjRpgPX5vOfW7btq089thjQtuwOYTw4FSPOthY1502mutvv\/1WaDf72XmRrWjK0rSJg4AngcG7H38sNXNpF8I+DCxlThw4Cr6lo6et8BGYzBVIBd8abUH8IqAtSzUEGjRo4O8yntDTDnoah0jwW92vXz+ZNWuWXH755bJr1y5hscVFF11kFir4M7oucCvRtWtXs8IT+0I0J2hxKI\/FFJR5\/\/33C9syoNFJT08XyImbxFAcaXFvYV1Q4DX9tttuk1WrVhHtKRAj5OSTT5bcOg9lnzwIHG2cN2+e0UB5Vppggb\/\/\/rvgQiTBmp2vzfUkMHjYRX3I8ubA1qBuxICX5cpWUCUGpuMetm297HKvEh6Bucu2mUTjuh8viLnRgyKgCCgCESAAiWBvOqsRwYawZ8+e2UjM2rVrZfDgwcY5KFoQVpBaQuSuxnEco0lH00I4mhM0JlwHCsTniCOOMMF4VWdT2927d5v7wAOkh\/GFj2XGm8D4aO6xZ\/ziiy9MFsjV66+\/bnb3NgEJemBKjGfy008\/JWgP8qfZngTGq2pUldddd53UrVvXbH7YrFkzv3tqHNHB3PGrAsHxyq9h0SHQvUlFYRl1dLnyL3Vuato0rrcs7+LIjlkTclOM5lUEFAEPBCAHaGHQihDNgoiXXnqJSyNoLHAyysDvOI6ghcFUwER6HBznvzQQhHHjxklGRoZHSjF+t6ztI+4w2I+KPJ6JYxBI2e+\/\/76g6W\/RooUp8fPPPxem1cxNAh7oE8\/LkrIE7EK+NTksgYGQPP\/883LuuefKhx9+KKgHA1sH4EuXLhWMs3A0h4YmMI3eKwKKgCKgCOQPAmhirrzySmEvOGp8++23BY0519i9zJgxg0spV66cYERrbkIcsE\/ho5Uk69evl2DGskxz3X333cYpKWlDaWyIz61g+IuLDZyQYrNJf9FeBNMS5ba+vM7PWIqbkkceeSSvq0qK8kMSGMB89tlnBVWgJS6O4wgGvWeeeaZ069bNLHvmj8CiAftFZYmK0obF\/pycJWL\/kpw9E9k6OV3Wpbc1svunWcnaTe2XIhA3CNSrV8+sDqJBeDb\/\/vvvuRSmJf744w9zjeYF7YW5CXFwHEcgJzYJq5aCGcvifHTgwIHG1xZjyOjRowWDY5s3lmdWNVEejlAxXLYmC7Nnzw6qJSI9Qtu+\/vprGTBggKDtgOBhB4RjU+xzrPDxjk808iDbt2+XlWTKAgAAEABJREFUhx9+WNBwkYb0eHxnloJ4KxApPvoZD\/EuD14YQtt8PB+ICkoCm4dVXZSNPQ+aMqYCsSW94oor1B7GguQ6hyQwWHWjLuRB41COB\/3NN9+YJdVPPvmk3HPPPQJb5CXAeAq1JepLVJZPPPFEws9DunDK80u8746elqmWTcapox2zJgqO6xD3FgJ5DWzHjh3Ne5rX9Wj5ikC8IVCqVCmzVNm2Cy051zgg5YzwMYrtCtfhBMefaDhIh4Hutm3buMwmjuMImnhWNhHJwH3XXXdls8MhLjcCQUD7AmGqVq2asKqJJd2UiZZo6tSpXAYVlqdfdtllgkH0qaeeaj7MWa7OhzmZICaMd5CQMmXKEGRWOrFcG9sevHxDCrEPmjVrlpmlwO0ICdEMYVj80EMPmRVZ2AThc4yxElLCtB17\/0FgcKRKHoQ0\/\/vf\/wQNGrZB+Dj77LPPzAowLxsl8kQtSZQhKIGB\/bGVAC8J1ug8DAhMMBDZp2j48OHCdBNzoO+++66xbk8irPKlK5CXRXc2z5e6CqKS8v2fl0rpM40Uq9umIJqgdaYgArgnaDDyS4lX6T9pccyfCgNg0aJF\/eVu2LDBXPObbi58h0qVKkmkBAYDXVseGnkGcV8Rnv8hOqxEQttAAj5qWaIdKg\/pohGWTrNSh+kj+kpeFo6gVeKaJdXBzBn+\/PNPgcCg\/YD4kB5xHEdwAUL7165dKwjhCCQE8oHRM\/vqMRZCNFix27RpU2NekZ6ebogaU25oXrp06UJW42i1e\/fuZgl5p06dzGoutEZEQlDAk2uV6BAISmB4WLBPyAgsEudGkRTNC8uL+9dff4lV70WST9MkNwLVxq8wXndLtuklEBckuXusvYsnBFZv3SNoOeNZ4gmvWLQFYsAecwz4lHfQqNf4iuE+NwIRYrqHZdi2fMqDOGDewDW2Ppg0cB0oEBimjALDuWdZN2QNDQnpCENoP2WeffbZRkNCGAKJYfqKa+LdGi7CEEgVRsaO43BrbJPslBzTe2ipTIQeokIgKIFhnhR2i2GuBTrSkvEp0KpVK5k7d664X4BI82s6RUARUARiiUCLWqUFzWa8Sn64TECTnhtM9+\/fHzX5YOBmJRIfwtQ9ceJEYeqF69wITvcWLlwoaF9Yjm3LchzHOOWz9b344ouCzYqNt2c0SVZr8+uvv9pgc4a8UCZpuCYQVyFoSpiZwKYHuxS3MOMAeUKspot8KnmLQEgCw8uHSi7aJvDwYcGo34Kx3GjLTPb0fBnSx6pJvoEjfVRRBPIbAaZmjZQpKvF6jjUmfNVji0G5hx56qFgCU6NGDYKM8PXP4GxuwhyYjqFMkpUqVUqKFy\/OZVjhA\/jGG2\/0G\/WOGjVK8PQbNmOQBAcOHDAkaPPmzWZ\/Owxp3cIHNHFkRxviZUCMdsjiga8c94c2mheICmVCSCgHEoR2BdJj7VIgNF6CXQ15VPIegaAEBiOo+vXrSyXfHGlOmoE1OH80eITMSX7NowgoAoqAIpBzBPiAxNcLJUBa7FQLhAIbD8IxFWC6n+twgh2LTcPgbg1bbVioM0ar1qgXIoRmP6eaCkjZp59+KpgqsPLHSzCOdZzMrQ28HNvxkQ2pQlMDycHGE2JEHzDEZdyCiKQd9HZMHKSG1UHgSjqVgkfAk8DAsnmxMcyFceakmTx4Hjova07ya57ERoAl0\/HQA9TVeLSMh7aIiDZDEcgXBPjtxcEbmgjHcYQtBezHKB+mfGDSEAZjhOtQgpZm0aJFJgnkBwNUSIAJiOBAHggHNpIkZ3xBo8F1tGJtK2lDsLzUw4aRxENIIClcu6VBgwby9ttvC3gwLdSpUydhuTPkiFW2rDay6TF0toSNJeQ2PPAM7r\/99ptAdALj9D72CHgSGFsNxkz2Otoz84c5JT\/R1qXp4wsBlkmzXHpVv5rx1TBtjSKQIgiwAAOfI3SXVTWsgHGcTANSpk\/QLjiOI6xI+uijj0gWUvC8a0kA3thzYlpAvW6j3pAVBomkvSydxv6EpdNBkgkf0BdeeKGJ5oMcLYy5cR0gGbYs7Gm4xm0Ivs\/YLNNxMvEiC1NmtWvX5lI+\/vjjoH5tWLr91FNPqQsRg1TeH0ISmLyvXmvIVwTyoTLISz5Uo1UoAoqABwIsC0argFdaBvsmTZrIsGHDjI8Ud3LsRKwGg+XGgYas7rQM9LjUQJsDCWFLGfzHuNNgNwJR4OwOD7zGrtJt1BsYH+6eVUVoOFjRE+4DmaXNTBFR5uzZWR3boSnB5QdYQebCaZMcxzEGw5hF0M9rrrkmC4kBI\/y5sPFlr169JFx5tCmY8PGPxocy7WaOXOtS6+yIhSQwWFbzgNnjKFrhhceOJnuVGpKsCEBe1g1rK6p5SdYnrP0qSATsFA5tYBWONTxlcGMaCL8mOG7ECy5TNvgqQZtQuXJlsmQRBlimTVgtCjHBGRuOS7Mk8t1gvHrrrbcah6WQF1xqBGpfIAMYs0Jepk2bJtz7sgb9z9SNe7uBoAkDIhjM0W44jiNefQpIbrQw1tCYsWjChAn+tuGJGLxw7seyaPbyw8cL083I8OHDjQM6sLXlMi0Fto7jmE0w8fHCpsdog1hGjXYJzNFQ2TyQHa6xn2EVF9cIGNEmrrFBIp5rBM\/IVatW5VLGjh0rbIsAIYW8mUA9+BEISWAw2uKBYXAVrUBg3A9fRPyV6kVyIrB3Y4ZAYoqUryFIyTY9k7Oj2itFIB8RYKDFQyx2GbZa7DAYPDGmPfbYY4XpEjYAROPCID9r1iy58cYbs\/grsXntmWkWvL4OGzZMMFqlDFzWUx8frGz4yMCOHRn7JTElxXSU42ROrWAXg+v7Tp06yfjx402xeGYnDdoN96BsIl0HvOcytjhOZlmuKM9LfL5QD9NYkAKIwogRI4zTuMAMtl2kweDXxrOkmjLAju1vcDSHh3mIIEQGQ160KAjLvYmnnRkZGaYIx3EEojdp0iRp3Lix4HWevPST5dzsDwWBdBzHuP0HE+okM+WjucIeByIG2QEr4lgJNmjQIIGUcg9RJC1aLurmeUBgAokjaVNdQhKYVAdH+x8dAlunDDcZIC44rkvrmm7u9aAIKAI5R4DVO3yFs\/TYa8UNYbioh2igOYBA4FwtkhrRxDBQsxUM2hXKwdiWD1bsQdjjh32CICQYu7rLZJqDAR3bEVY70Q6EjSJ79+4toaZ4HMcxS6CHDh3qLjLoNdM8TANRPsI12xOw0CQwk20XRMTdLq5pKyTPcRzBpoUNKiFrkJfHHnvMeMrlnJ6ebrYGwA0IZMbW4TiOYB\/DtJt9HuDD0nDIpE0HOezXr5\/xRk97EdqDxos4tDws3yYcgeiccMIJNru0bdvWbNlj29yqVSuzDN2fQC8MAoXMMciBvSBgijzQaIV9kZgvDFK0BgcgMHfZVhOCjwpzkcCHorpFQAI\/PW16KiKANgHtgVsrgiYDrUIy4oHhMlNG7D3EXkhMa5133nliBU0UGqj7779f0JCARTLikOh9CkpgYOawedi\/fajRnGHHfAkkOkD51X72aqGu7k0qcUpIqZw+02wXoNsEJOTj00anOAKO45hNGN0rljAjQEODjU2ywINrD\/zElC5dWqxvnGB9I229evUi3i8qWDkanjcIeBIYx3EEdViNGjVyXCvqQ1R1WFTnuJAUyQh5mbtsm6B9aVm7dIr0WrupCCgC8YYAxr\/YlqB9sIakkBhsQZhGSga7xm3btsmWLVuElVfYl7ByK\/A50E+mldj\/CGPdwHi9jw8EPAkMhk3sHOo1vxhNs2HuWG5HkyeV01bVbQRS+fFr34MioBH5iYDjOMIyaxy6YWiKHQwfsywRxk4DY9\/77rtPFi9enJ\/Nilld9AXjZApkSTfTR1dffbVgl4LgwA4j3S+\/\/FKwC4pkxRNlqeQ\/Ap4EBitvXl6Mm6zAVO+55x5xW3V7NZe5xSFDhhgHSV7xGqYIKAKKgCIQ\/whgF8NKJz5mMRDG6BTDVexi+I3HdiT+e5G9hWiZMI9gBRVLoJkt+OSTT4Sxiz6yJBr3IZCb3H7EZ69dQ2KJgCeBoQIeIJboCMvxsE6HeYfzzovGBTfRvOBseU5ZKopAQSGAYSLEOzf1a15FQBFILgQgZ6effrpMmDBB8MLLSiDOEDUIGyYUydXj5OyNJ4FhCR7r2ukyngyZ+0SVBuOGvRIeTFjDzoDB\/CJziMHSaXgmAqu27BG7AikzJLGOO2ZNkK2T04VzYrVcW6sIKAKKgCKQyAh4Ehg6hNdHlkHjMRGtCmGRCluQswwN50MYS0WaLxXTTZr\/u4yelhGi6\/EbtfunWbJpXG\/B\/wvn+G2ptkwRUAQUAUUg2RDwJDC4Nv7+++8FF8tnnnlmjvrcpk0boRzrXTBHhaRAJlYg0c3uTSrKuO7Hc5kwwoaNNBa\/L2ldhnGpoggoAoqAIqAI5AsCQQnMypUrhaVzGDjlpCVlypSRSpUqidf+GqHKw1U0bpNxosc8JGvwBw8eLMxRhsoXbRwbnaElog7caUebPxbpIS92+TTkhWXUsSg3v8rY\/fNsUxXkRb3uGij0oAgoAoqAIpBPCHgSGPZ3wNaFqaDctKNQoUICEYqkjAMHDghW4Z07dxZWQNmdN9HisJTvggsuMFbikZQVLg11sW9IQW+ONXraCtNUtC\/mIsEO+zZmTn0dUiHn\/oLyussY5bE0Mq\/r0fIVAUVAEQhAQG\/zGAFPArNnzx7BdfL+\/ftzVT35cQgUSSGQCQyFydOjRw9ZsGCBsA\/E1KlTpWnTpmY6Kj09XX777bdIiguZBvuegtK62IZlGu9uM7ctaqWZc6Id2Lgx0dqs7VUEFAFFQBFIDgQ8CQzecyEwK1Zkaghy0lWMd9lDglVJ4fKz3PrZZ58VtC7nnnuusA1BWlqa2bzqmGOOMTudspEYbp3Z1AwNSrgyg8UzdTR69OgC91PD1BFtbFGrtCSy9112naYfKoqAIhBnCGhzFIEkR8CTwLB0uly5cjJjxgyJVIMSiNOsWbPk999\/lypVqgRGZbvH0Pfbb78ViBN2KUxfuRNBZq688kpDaGhTOGd67rzua4jPU089JV9\/\/bW0aNFCKlSo4I7O1+tXvv7d1HfpKYm791H5\/s8LoiTGPEo9KAKKgCKgCOQjAp4EBsJQt25dwTshe0FE2x4IxsSJEw0hOemkk8JmZ0oH7781atQIurkWRr0QK5zkLV26NGyZXgkgLrSrZs2acsMNNxToBl1zl20zTUQDYy4S8FCyTS\/RjRsT8MHlT5O1lhghgBf0vn37Cr+BLDrwEuJwzIYTNqbZ+ViLUfVxVQxY9OnTR+rUqWP26\/PCwisMzT6zAnHVGW1MrhHwJDCO40iHDh2EPwI86kJiuI6kNsgL3nuZfsLxHX9Y4fLZPTXYc4J9mLzSQ17QCtEOnOR5pQkVxvQT7qOZGsOzcPXq1UMlz7e4RFt5lG\/AaEWKgCJgELjkkksEzfGTTz4pVjtdu3Zt+ZvXkUwAABAASURBVOKLL8zqTH4\/cRpK2KuvvmpWj6KxZrrcFJBEB7BgscegQYP8vWrVqpXg9oOVqm5hnGDbA8wQ\/In1IqkQ8CQw9PDUU08VNrzij4CNrnr16iXz588Xr507SQ9BYItyNgFjKTRO8K699loJZwMDoWB3UMpge\/PDDjuMy2xCecQTEenKJtIikJ4nnnhC2FUVD8Pt27cnWCVCBDDWXd7FkVX9akaYIw6SaRMUgSRDANcUTLPTLdxb8JvINb+ZbLIIwbn55psJktmzZwtOSPntMwH5eMB0YOvWrXlaI2TNVgCpQ+y9PROG7eRtt93mJ342Ts\/JgUBQAsPD58HzAvBHMGfOHIH9olHBM+8ZZ5xhCA6bYeGrhZVCEBiWPTuOIyyHPu2008KihI3Nrl27TDrqNBceB\/5w+QMmijycIxVWOGH8y9TRgAED9GWOFDhNpwgoAgmDgOM40qlTJ7HaZXZTXrt2bb62nw9e3Bb89NNP+VpvqMqYCWDKyXGcUMk0LgERCEpg6AtTOqgu27Vrx60RNDDYoTBFhC0KU0aQFhPpO7BJ1k033STDhg2LmijYPzxfMdn+88UBickWESaAto4aNcqk4uskEqNikzjKAzhY2bdvn+zbt89TWD49af56f+nB0sVTuG1sPLUp0rbQ9kjTajrvdzbRcOGZp6qg8a5UqZLpPqs7rXbbBOTxgQ9d\/GsxtZXHVUVVPIs1HnzwQcG2M6qMCZA42N+mHYs4J0A3ctzEkASGUsuXLy9Mv\/Bioo1xHG8WC3FB4\/LBBx\/IddddFzV5oa5YCw93\/PjxxnfMRRddJG4iFuu6Lr30UkEbhYwcOVJYQu4lb3+1RPpPWmyqr1yqSNB0XnkLMsw02HfIjzbwR7d+\/fqYYGN\/wPOj3fFcRywxjed+0jbeHd+rmpL\/+cCEuNB5ppaKFy\/OZZ4L5AWHo2jh87yyCCugTVOmTDF2QhFmSbhkTNfxzgfKM8884x+PGJsSq2ORtzYsgaEoyAk2MRhEYd+Cwdhjjz0mVj766COzZcDzzz8vuTGYCmXbAhnBwR7tiVRYyv3666+blU39+\/fPU1L1wAMPCNNUSK9evcw2CnwJBcpHS\/aY5uN99\/LmVYKmC8xXkPflypU3bUYLlh\/tOPLIIwXiHIu6unTpYmwBYlFWIpcRS0zjHQdrK2de2hQ7YMyKUS\/dbt68ufl94doKNocff\/yxXHjhhYLBL1u9YBPIyh1+IzMyMuT+++\/3r3hiWxfykB8zgmuuuca\/Asgdh2Htww8\/bNxu8FuNtpuPOdxiuO1htm\/fLqTDDIE6MUn4v\/\/7vzwhGdTF7z9aeNrvFox977zzTmF1F6YSYIBZBKtULQF0p8fGk9mI8847z3wQ80HPClumpsDRpoVA8hHfsWNHQyBOOeUUYezE9xgrbW06e6auF154wZhj1KpVy2DLs8GRq00T6hzsd5IxiLEIwWwiVBmJHBcRgXF3sESJEtKgQQPhQVo59thjc7wkGbsX+5UQyrYF8mKXwZHH3Sava+Z+eWmIGzhwoBx11FFc5plgA9SsWTNBatSoYZaQM+Xllm\/W7JF5K3cKK4\/Y++jWs2t6pnPnKejrvV+9KpsGHe\/HLT\/aU6xYMfM+5UddqVJHqmHqf2EPXmCIvmPWBAklB5N6nkLls3GeGV2BNp3XefdPs1wpo7+EZPBhyfQ9v6Nt27YVVurw3CkNJ6Fjx44VBmlICB+i33zzjdxyyy1itZQs0iAve8+x7Jh8bmG1z+OPP242+XWHcw0RwLs54wMfOkzZfPbZZ8LgbKduIEsXX3yxMGizxQdkiyXRfGhSHwSJsmIhkAW0EPTTXR5aGdoESYAwMLtw7733Cu2BdGD4zLiGiQT51q5dK+AFbvfdd59s3LjRrAh77rnnhLIgLKzSBX+IGqu\/qBev8vQfY2o+6jGw7tq1q0CEKBeBWKEdoY04cmWRyYgRIwQCStqXX36ZZCEl2O8XYxBjEcLYFLKQBI6MmsDEuq+oOWGRlMsfEi8C14HCHyDxhIeylSEe4SVgKwLy8WUB23cLDxa1G2n54iCOKTBeUMLyQlZvzdS+VE0rmhfF50mZO2ZN9JdbpEIN\/7VeKAKJhMBuH0HYNK63hJJQ\/QmVz8aFyk+cTed13jplOEkiFga7k08+2e8LBUNVtB58DKINYcDEhtEWyIqlG2+80WiJ0aARvnDhQmHKhwEYm0U+9Bj4HMcR9rETj3+QEwZNj6iQQQzWtA\/SgA+utLQ0geywwpUBlt\/p9PT0LAN8yAJdkeydR\/\/5DbeCdmTcuHGGZLiSGues99xzj2AOwUITZheI56OYtkBUMnxaKFyBQEj48IWYQWxIt3nzZmG7m3fffdeQnn79+gn5wGvo0KFm+xvKh7SQHgJ55plncik8M9rKDQQLzQ9l3XrrrVK1alUzQ8BCGYgU5GjMmDGCk1fSq3gjUOAEhmadeOKJnISX3G0QbAIPHnhx\/vjjD+ONF43PweCEOa3askes9100MInS8EPKZ5KWSukzpbJPEqXd2k5FwI1AEd97jOPFUOJOH3gdKp+NC8wTeG\/TeZ2LnWBXbAbm8r5nGfHUqVPlq6++EjQXaEb4AMMmgqmds88+20zrB+aGfEBCCGd6lSkQiETPnj2lW7duZhAlLtYCSUKrQbuoz5bPdaNGjcwt8WgjzE0UBxz48cHKtJAV7iFEEDdbFCuk0LhAlho2bJjNxQdkA82H4ziGOKDRsnlpJ9cYSffu3duQL8gepKxJkybGJw9Tc2ipwJS0Vi644AJBqwXZoa2EM0XEs2vTpk0Wj\/CO4xgv8aSBQKHF4VrFG4G4IDDMg\/KCwHzRmng1FfYKieGrgj9erzTuMAx27cvsdeblgfWSBzUqaVD3YbFOWKwFw13rfTeRCAxbBRw95YCox91YvxFaXn4iwPvLuxxKQrUnVD4bFyo\/cTad1zmtazpJIhZISLly5czgh5aAKRhsEBGcfkIGunfvbghOsEIZjIPFxTIcrToDMbYxTOtjG+MWtBmsDkU2bNgQk6rR7EMc8I9jC0TjzhjDfbDfeTQnFStWJInZcgayY24OHigXEnjw1n9iGozpNz7GeTb+CN8FYxs+0SA7drbBpp80aZKxlXHjgeYFLBDGPF8R+j8IAnFBYHjJYOHYuTA\/yYvgbi9M1M45YvxV6eAyQXeaRLm+9ewa0r1J5jLHRGmztjN1ENCeJjYC\/D4yDUEv0GZjjwGB4L6gBGNaCBUDu7WNgdB4yWWXXRazZpYqVUogMaVLlzZlouFnLDE3QQ74GrMfthAepnqCJPUHg3OwD29\/IteFOz0Gtl442DC2hnBl1csABOKCwKC669Gjh6Duw4IbmxVengMHDhiLb4yoUC3CXlHf8Yfg7gfMlrlPVKh5acPirjOn1y1qpUkiaWBy2s94yYdxHo614qU92g5FIK8RaNCggb+KH374wUzN+wMK4ILfcbQvfJiuXbs231rgOI5gJItWhEqxd3GcTDcgocYJ7FlIz9lxMtNzH0xs\/4gPtZKWeCTa9ORR8UYgLggMTcN4Cqt5XjLmSyEjGHy1b9\/ebGHAHknMaaLiI71KMiKgfVIEFIFYIuAeLGNZbjRlMe2CZoM8rHTi7CW0lY0oITpe8bkNY0qG6TXKwWQAUsV1MGGswSg6WLwNZ2yy0070788\/\/7RRWc6svsJHEVNQFg9WZoXS8qC5ChWfpYIUvIkbAuM4jmB8NnnyZMHQCW0Mz4OXg6V377zzjrDPEmGJJBjvYv9iVyDFe9tZbqr7HsX7U9L2KQLBEWCqxMZi64GtjL3P7RmSwcqZaMop5ZvKsXaLGLpiwOqVH6NkfK2EIxZeeSMJg8BgrkBaVmBBJrh2C9NdaP9ZlcTqKMiXO97r2nEcady4sYliafj7779vrt0HcMPeZdq0acJHutUKzZs3T+zKJHd6rhctWmR8WOUVoaOORJe4ITAA6TiOYB3OvC1Gu7BkVKA4iWOKiDRewrwqaaMxwuUPm\/TkY4qKcvNCMNxl6wCITF6UH+sy927MMEXqkmkDgx4UgYRCgCmaF1980bSZQZiPv9KlS5v7aA5Wo8DUvTV8Jf+3335rVj1xDdFwkxk0Cwz4DLjW1oRr5PzzzzcmAmgT+L11kxjicQaH8SoO2DApoPxYC+XiNI+PY0ieF9GgXcRBXjCMjrQNrCZijIKoMFPAUmr6Sn6wYJn6rFmzBI\/wjuOYj3RMIkjPIhKmurkmPb5lZs6cKSxrv\/zyy5NyCwT6GQuJKwITiw7FWxl26TSO697t11Ba1o7+xyQ\/+4QGhvrs8mmuVRQBRaDgEYBI2EERh2gMtLZVhOMRnZVHGJQySDNYY1voOP\/ZcbBQAuJBPpYVc\/aStm3bmqXCrILBsBY7Q5y\/sU0K0\/vkYfk2PlAgOdxDeqwBLE7z8AKLVp3NdDEuZkB2HEcokyXcfKyy+oYFHDiTwz9M3bp1KSqsoJ2wiSiPftn7UGfaAblAs4+fGNpm0zNd8+ijjwo+ZIYPH55lmTX4kg6DaK+6WAmGrR0ruyAgfIRTDqQG7QwYuctkeio9Pd2QOlY6YcxLevDAvw+rlljWzrJs6k0AKZAmKoHJY9jRwFBF9yYV45680E4VRUARiC8EXnvtNePyHl8i9iudQRutBgMkwuCHF16IC1qMDz\/80HjZRQtDbyAtlIP2A18xhDGAM+hit8G9WyAVaERwW4EHdLTVOP+cOHGi4IcLIoBPFZzm4RSOvAze7IOHdhuyRX0QGMiQ4ziC112mURjQmUbBVoR20Q+2qcGBm+P8R7YoM1AoE8NcHPXZOMgBmg282ELsbLjX2XEcowUBH\/IMGTLEbKuAFgRCwSIRiFfNmjVNdnyPQXggbgRA+ugje+xZHAlHMH1g7yXOPAfCwA+neGjFbJmEI5hEsITcpmd1EuVDWsAJvOzzI71KdgSUwGTHREMUAUVAEYgbBFgWjW2InVZn2ttLiGcBBB5hITXuDrByk3IYuG1epucZnHHE5k7LteM4ZvNbNBSkh+TgMRZ7FgZW3F2cddZZZssP0luBrOBjCy0QdosMxo6TSUocxxH2BmKQX7Jkidn\/iJ2rR40aZTwK2zJCnekDBMPmp23IjBkz5K677jJ7qIXKb+PQFKERoX+QJ7yxQ84gR0w12XTYD0Hy2HaBepDZs2cLTum83HmgWUH7wrMgLeVDLMHNluk+B6bHNgctENopx8nEzZ1er7MioAQmKx56pwgoAoqAIqAIKAIJgIASmAR4SNpERUARiGsEtHGKgCJQAAgogSkA0LVKRUARUAQUAUVAEcgdAkpgcodfyNx26XQied4t2aaXsPcR+7WE7JxGxg8C2hJFQBFQBFIQASUwKfjQU6nLrGzAUDGV+qx9VQQUAUUgFRBQApOHT9lqYKqmFc3DWgq8aG2AIqAIKAKKgCKQ7wgogclDyO32AYk0hZSHcGjRioAioAgoAopAzBBIfAITMyhSs6Cb5BWGAAAQAElEQVR16W2FvY92\/zQrNQHQXisCioAioAgkJAJKYBLyscWu0XsOEpdiddvErlAtSRFQBBQBRSDuEUj0BiqBycMnuGrLblN6vE4h2X2PipSvYdqZjAc2ScOTZjL2TfukCCgCikAqI6AEJg+fvjXirVamWB7WokUrAoqAIpCICGibFYHcIaAEJnf4hcxtCUy8rkLauzHDtL9IheTVwJgO6kERUAQUAUUg6RBQApN0j1Q7pAgoApEgoGkUAUUgsRFQApPYz09brwgoAimCwIEDB4Tdivv06SP16tWTo48+2pyvueYaWbNmTRYUuL\/jjjukWbNmJh1pA6VOnTrSsmVLueKKK+T555+XTZs2ZSkjkpvt27fL6NGj5YILLpBatWr56+rfv7\/s27cvkiJk9+7d0qtXL39e2nnuuefK3XffLfPnzzdn7gm3QtvZOfrpp5+Wv\/\/+O0s9r732mhBHGtLTLtoXmNa2\/dRTT\/XXfcopp8hZZ51l6jzjjDOE3aRt4TNnzpSbb745G6YnnXSSXH\/99RJYvs1HPbTp4osvFnaZpk0I1zjafPjhh2XFihXy8ssvC7gF9seWo+fsCCiByY5JzELmLttmympZu7Q5x9vBGvEeUr5GvDUtBdqjXVQEIkcA8vLMM89I165dZcaMGfLXX3+ZzJw\/\/vhjOeecc7IMtlWqVJFRo0bJ5MmTpVKlSibtIYccImPHjpVly5YZmTdvnknjOI6MHDlSmjZtKpdcckk2MmQyBzmUKlVKbr31Vnn99dcFkmGTMdhDtux9qPPPP\/8s3377rT\/JcccdJxMnTpShQ4dKkyZNzPmNN96Q1q1b+9NA2hjwIXOHHXaYP5wL+kDcwIEDuTXkiPa5065du1YgFK+88opcddVV8t1338mvv\/4q48aNk2LFismECRMMqVi5cqUpg0Pbtm3lwQcflEmTJkmFChUIEjAF58cee0zc5RP577\/\/mn60aNFCWEhAuZC977\/\/XpYvX26eY48ePQxxgSzddddd2cgY5agER0AJTHBschVj7V+qlSmaq3LyMvO+jRl5WbyWrQgoAjFCAE3A448\/Lr1795YFCxaYARACcvnll0vhwoUNoXnooYfkzz\/\/zFJjmTJlpHr16ibsiCOOkOOPP14cxzFC3GmnnWYGWQZ8SA8aj7a+gZp7SJPJGMGBgbxu3br+lHv27JE333zTfx\/sgjpYKbhz505\/ksqVKwvEyB\/gu2DwL1eunO8q83\/FihVNHzLvsh8dxxE0K2XLljXEh\/a5U030ESSIHBoQNFAlS5Y0ZATC9OqrrxoiR\/odO3ZwyiLly5eXGjVqmDAwhXCZG9dh7969MmzYMKPJ4fq6664TtFxod0qUKGFSgj9k65NPPpFWrVqZsPXr18uuXbvMtR7CI6AEJjxGOUox96D2JV4NeOlUWtd0OXrKASnf\/3luVRQBRSAOEWBK4cUXXzSD6uDBgyUtLc20koE0PT1d0EYQgCbjhx9+4DJqYaqJQZ3pFjQHaD\/QTkRb0OGHH26IAPnQDNEmroNJRkaG0UQwpRIsTU7DCxUqZEgOBM9dBiQPokYYhK5IkSJc+gWydMstt0j9+vVlw4YN\/nD3BWW7793XkDLICpoari+77DKBwASSKJuH5wk5Zfpq\/\/79Qh4bp+fQCBQKHa2xOUXgla9\/N1lvPbumOetBEVAEFIGcILBq1SrBjqJnz55+cmDLcRzH2GygQeBL3z3lYdNEeq5Zs6aZDjr00EPNIPrEE08IdUean3RHHnmkNG7cmEvZunWrvP\/+++Y62OGdd94xJIPpnGBpYh0OIdyyZYspFi2Wl60OpKJz586yevXqqKd1lixZIs8++6zBEE0N00TByItphO\/A87v22muzPV9flP4PgUABEZgQLdIoRUARUAQUAT8CpUuXNsSCqRV\/oOsC0kAaBsnixYu7YqK\/xF4DbQw5161bJx9++CGXEQvkh2ktO03y7rvvBiVBGA1PnTrVTPEw0EdcSS4TgpG1C\/rggw\/kueeeM2QjsNgzzzxTmE4LDA93j70NfSMd01hVq1blMqygDcLw2HGcsGk1QSYCSmAycYj5cfXWPabMeLaBMQ1M8gNz3Pfcc0+S91K7l8wIMFXEihXH8R7Y0CAw9VC0aFGpXbt2rqBg+oSB2xby2Wefids+xYaHOrMqp1GjRiYJJAgbD3MTcPjqq6\/k999\/l\/PPPz9fNQ+QqzZt2pjWMF1z3333mSkeNEYm8OABQ12mfwKNhA9Ge57c01MQSrRRgVNUnhl9gdSHkTDaH99t8P8a40dACYwfitherNqiBCa2iGppioAi4IUAK2o2b94sDRo0EKaBvNJEE3bMMccIZIg8lB2tUSkkCGNjBnDKwJbGaiS4R1g6zcoijIhZEk5YfgrEhCXkts6PPvrIaIKwA8IGyIZHewYvO+3mOI5\/BVi05Wj6yBBQAhMZTkmVyi6fTqpOaWcUgRAI7P3jK9k2vVXcyl8LB4dofeiouXPnCl\/5V155paBdCJ06fCzkxZIPiAZaBVeuiC7RwJx44okmLT5OWEVlbg4evv76a+PTBu0LhOdgcL6dsDkZP368YFdkjXxZkj58+HBjU8Sy6pw0BvKDRoy8rKRiFRTXKnmDgBKYvME1bkuFvKzqV1OWd3GEc9w2VBumCMQQgf2718j+XfEr\/\/ralpPuQg7QZHTr1s0sG85JGXmRB4KAlsNxMqe9WEWFITJ1McCzxJqVR0yxEFYQAtljqTN2MCxjdpzMtoLpRRddJDiYY4opp23DWJjl5DnNr\/nCI6AEJjxGSZXCv\/9R+RpSsk3PpOqbdiYFEMhhFw8p20xKt5sTt1Ki4QNR94xVR3h\/ZbkvnmCt1iTqgkJkwFdJ6dKlQ6QIHgUpwCiVFGg08F\/DNf5XWMYM6YoHew+mzHBc9+STTwr9pY0QF5Y2P\/roo9zmSCAvOdFe5aiyFM0UdwSGFx3PiDgHgqEzP4rvAzwX5vQZ8TIuXrxYbrjhBr8rZ\/wdYGEOy7ZfBjktP5HyoYGhvWzgiB8YrlUUgWRHoFDxKhLvEs0z4DcNXyP8ruHlFY1HNPlDpWUbAuvArVq1ajmelsL4uFOnTqYq2otzPLQS2JswvQLBMZFxcHAcR9q1aye0DX8sNIk2v\/DCCxLOlw1prRx11FGCcA\/BxCaGa5W8QSBuCAwvC18TrL3\/9NNP5Z9\/\/jE9Zl4SFSl7WfBymcAoDrxEuIdmrhWfBJYRUx8+E3ABjdU9xCmKYkMmjWcDXut9t9gJp4XsQ7JE4uWTH\/gY9UeLUQTiAgGWH0+bNk3GjBkjwZZX56Sh\/C7iG4W8juMYV\/65sVE577zz\/AP6F198Ifw98nvOb65dykxdkQjedyNJ507Dsu7A9r\/99ttBt0uAdLESyBpDszIpGueAaJTcHomx\/WHKzN0mvY4dAnFDYHjQuMJmOSCOf1A3omrkD5U9OiAy6enp8ttvv0XVe\/5g0LJgXAXDZh8RymXfi\/\/973+C+2ws5NmkK1ZsedL8TCd2VdPibxuBfZtWRoWfJlYEFIH4QoAPOT68MDi1A22sWpiRkSHTp083xaEBd+9vZAKjPPD7ClkhG1MqbDAJSaJcx8m0OSEuEnEvEbcaolD50CQxrWanhWxaPlZxNGfvA8+02WphiIuGgDiOI3yElzi4XQBkEJsaylGJPQJxQWB2795tPBeideHFZlOrtLQ046GR+UmsxZnnhWiwJI8\/gEig4CVHbUl6vgTQxPAH7ziO8TvAvhTMcWIpDqmZMmVKJMWGTfP50m0mTcy98JpSc3fYMWtC7grQ3IqAIlBgCKCdxjbj\/vvvF7vKx90YPsJYChzNoGvz8zvMbyS+WxiA2aTRTofYNNGeHccxWyDwG0teNOL4YLG2MYRFKjh6s+V88803IT3k0n982FBPoOYGgoI2no\/YYHWzEos4cMCcgetIBcNkNP6k\/+OPP+SBBx4QsOU+lDBeDRs2LOqP9FBlJntcXBAY5hjZjZSXho21YM1u4CEzLBF0HMfsmwGzdscHu8ZuBrfOwcolH8TIzsXSBjQ9hOdUmD5iHyQc2LWsnTPjt5zWHUm+onXbCFKyba9IkmsaRUARiBME0FLzwYVjxmDkhelSSAfLqqNpNsRnwIABZgNG9jNCI80OydGUAWng95OzOx\/2hnikJQxCwOAebfvISzlnn302l8Ju16+99prwcWoCXAfCmCZC285KqMApJDa3xG\/OTTfdJHwUu7KaS8LQnHBDu4899lgu\/YI23xIS6mLWwB\/pu6BvkD983PhujUaL5xLK1hJbJnYax2keH9nkUwmPQDQEJnxpOUyxaNEi4+2xRo0awkvqVQwsGPbN18HSpUu9kmQLg9FiLIarbf6osyXwBTiOI2h5fJfCHx4vJNc5FQgMeasW4PTRuvS2NMFTKqfPFKRI+Rqe8RqoCCgC8YcAU+oMuN9\/\/71ceOGFwvROoPAhxuaDJ5xwgr8D7PmDrR8BaAPcU\/AMqJTL4Nq+fXvzcciAjWt9PiQdJ\/IpHn5rIVjr168XBmPqs8KATnmQF8p3t8+m+fHHH+2loA2n3f6AgxeUAzFAS3\/gwAFhCo1Bf\/bs2bJx40aBhLE8G8PhESNGyKBBg4xzuoPZ\/SfGA4yef\/nlFznnnHMMacO4mDLBh8UexDVp0kTY1JJ2+zP7LmgbHoR9lwIR4kOZa7dQPjMHAwcOFOxw2POJbRrAmg922stYxnRd7969jT+a7t27CwtWAj\/g3eXqdVYE4oLA2BceYzTYf9YmZt5BXsqVK2cYdyjVX2bqzCOeFvmjgq1TdmZo1qN9aQnlD8RxIv+jJU+gjJ62wgQVpPZlz0+zjJ8X0xA9KAKKQEIjwODOjtMMluE6gr0JxrFoqbE3YYC3gy15WW5tiQ92HjfeeKMZhAlnMMVfCwM3aSMR6klPTxdIA87pmCJi52WEpdK2DEhL8+bNs9iHEIcWpU+fPsLUFfcIhAuSNnz48GwaEojBI488IqwOOv3002XlypUCAWD\/JjRG2La0adNG2L7gkksuMWYIlOkWxhIwYM8mtDFjx44V8vPxzKaSmDKMHDlSJkyYkMVAmnEEzRREAwJiy4RY9u3b17SJj2AbjuYHHL788ku57bbbhA90bDI7dOhg6gMz+tG6dWvBIBuSp+TFohfZucAJDMx327ZtprWlS5cWVGjmJuAAiyWeYF5azrGQjIwM4Q+PstjDIxiBIj4aaVErLZrkuUq7b1OGYNuydXK6ILkqTDMrAopAXCHAdBFkgC\/9cILWwXEcszhh1KhRwn5DwfKgZZgzZ47g\/4QBmKkLx4nuAw57knQfgWG6xtbDqh3sdNxEiMGcelhI4QYXksHqU6b6bX7OtBt7EFYFudNzjedcPk7J9\/XXXwvpEfqDUzoIBfsKkdZL+Jh96623jAaHFVHYyixcuNCUQ9sJu\/TSS4U2u\/O3bdtWWKFE26jPCvU+9dRTAgHhI9idh+u0tDSBWjZDtwAAEABJREFUpEFeKN\/mo+0QGAgYaUirEh0CBU5gYOx2r41Q7BM7FmtNTp7ouumdmnL4I0CVBytn4z\/vlJGFMn1UEPYvOKfbNK63bJ0y3Ait1SkiUFBRBBQBEFBRBJIRgQInMG5QUee5793XMFtIjDssN9dMHeEIChWm4ziGIVtbmJyUiyrV5kONmK\/i08CYutOqSMkL78iUi+4wNj352o59++KuTnBJdQxSrf88cxVFIBUQCPa3zXhkJZlxiCsCk19AQ16Y68UJFNeoMVHjOU506lN3e1E5Mpdpw1avXi35JX9UP+iUrlZT2dn8\/4xsq3VGvtUf637yh4cxYCzKtdOTsSgrkcuIJabR45B\/fwu0jXfH\/h3qWRFIZgSwb+KdD5RnnnnGGDAzJjE2JSsGcUVgQtm2wDRxhJTbB8ESOKaNsFLHWAuL9ttuu834hclN2az1f+CB+00RaIswpMtPYcoorXF7yc8686ouVgkw9x2L8llBUBDPIxZtj2UZscQ0lu3Ki7KsrZz5Y9SDIpDECAT7nezVq5fgMw1heXyyQlDgBAa7l+LFixt8sUkxFx4HyAvL14giD+dohbX79913n4wePVogMliBc49le7RlBabHW3DTps38wUx35VZ2v3uf7Hj2Wk8JLLva+BWS1u4qCQxPxHuM5zDmjkXbWc2A34xYlJXIZcQS00TAwf+HqBeKQBIjEOxvkRVPrKxCGJuSFYICJzAMVLBIAEbdz6okrgMFbQnxhIeylSHeS\/BTwBp7ltkRz4aREBm+0LmPN2FlEUa5rC7yknhrr7ZHEVAEFAFFQBHITwQKnMDQWZYJcmY1EJ4cuQ4UfCDgiMlxHAn0jBiYNvCeDbn69esnH374obAED2dISE41OYHl23tWIXFdNYsTO0Kil70bM0wmvOaW7\/+8BIqJ1IMioAgoAoqAIpCiCMQFgcHLLpoQfLLghdHrWbDWHhLDGn73pl5ead1haF6YA5w7d67xiHjvvfeaFUcQGXe6eLvGGR1tKnbCaVKyTa9sQpyKIqAIKAKKgCKQtAiE6VhcEBi8NDZq1Eiwc8GxT6AtDBoU3FuzYggHRhj+hemXiSb9ww8\/LHjjxRHeAw88IHhadJycrzYyBefDIa1ruhw95YBwzofqtApFQBFQBBQBRSChEIgLAoOBYY8ePYyGBE+K\/fv3N0uAISDsTYEb7e+++06wlWG5M6tK3Cjj3hnX0Gye5XbxzOaMr7\/+ukmKB0hIEht1kcZLiMO412TQQ1IggPdL9h9Jis5oJxQBRSA\/EdC64hyBuCAwYMQ+FrjBZmqHPTkgI+xNwSZjuNHGxT8uq6NxNvf++++bTSIpH\/sXNjvDKjuYdO7c2ewLQvqcyOqte0y2amWKmrMeFAFFQBFQBBQBRSBvEIgbAuM4jrAyaPLkycImXUz50GWIC9M+7ObJzqGERSI7d+6USDd9jKQ8TaMIKAKKQL4ioJUpAopASATihsDQSsdxpGHDhoIXQYx22fSKza+wXWGKiDRewgZbpJ09e7bYTbwwCsaJD+GRiju\/Vz3hwlZt2R0uicYrAoqAIqAIKAKKQAwQiCsCE4P+xEUROZ1CwvfL8i6OrEtvGxf90EakNALaeUVAEVAE4hoBJTAxfDzWD0y1MsVyVKr1\/ZKjzJpJEVAEkhoBFjUsXLjQuIGoV6+eoJXmzCIH9rpyd577O+64Q7D3I52X1KlTR1jVecUVVwgb27KIwV1GJNfUg20iNotedYQKe\/nll7NVsX37duMp\/dRTTzX9I\/8pp5wiZ511ltx9992CrSSrSm3GmTNnSt++fQVXHKQNFOwowYA+skDEvcI1XF7KYkagY8eO8thjjwmrYW297jPlsJCEeshj5aSTTpLrr79e2Lrmo48+8vfHxkd6pv9LlixxV6nXBxFQAnMQiHg4oYGhHYeUr8EptUV7rwgoAn4EIC9MrXft2lVmzJgh1uEn548\/\/liwD3QP7FWqVJFRo0YJNoXW7QSOO8eOHSv42kLmzZtn0jiOIyNHjhRczl9yySUCKfFXHOaCeiAwr776qlgP6dTz1FNPidfUPXVi6+g42V1ZrF271ri5YOqfNKw8xY5x3LhxwkrVCRMmyIoVK8S9Z17btm2FutwrDRnwycug\/9lnnwlOTH\/++WdDJpo3by4QDrpl89J37pG6desKeWj7okWLDJkqVKiQ4I6Dch955BHhWZDWCuVgxjBp0iS\/CQMYgD\/Ep0+fPoKHecdxhK1NeH60jTqw7bRb2UDCvvnmG4Mb\/QaHmjVr2mr07IGAEhgPUHISNHfZNkFykpc8kJcdsyZyqaIIKAKKQBYEICePP\/644EZiwYIFZpCDDFx++eXGuzhE5qGHHpI\/\/\/wzS74yZcr4icURRxwhxx9\/vDiOY4Q4NCcTJ04UtCGQEVZ8MiBzHzhQZyk44IZB2BIlotijh3Og4ArjpptuElaEBsbRDogVGg80JpQJEWjSpIlAkCBX5ME5KWe34ODU3mP\/yCIQVrQeddRRQllvv\/22QAa2bNkiODYFQ5u+bNmyBg\/u7aanXJcqVcpofl566SWjqcLFxqOPPmpIIfGBQt\/Yg4hwsIaQcI2wQzoe5\/EATztoG+HBhH6j0WEfN54DTlyDpU3lcCUw3k8\/6tDR01aInULKyVYCG8f1Fut9t0j56lHXrxkUAUUgORHg6\/3FF18UBnD2c0tLSzMdZcBE+8EUEgFoGVj0wHW0wmAJgWDKhYF66NChggYg2nIiSY82BZJUvHhxf3KIF+SJAEhWoK8v8txyyy1Sv3592bBhA8miEsgZWioysUIVbQnXSLly5QTSw7WXQKQgQY7jGO0LWhPK8EqLtsYrnLDGjRsbX2ZcRyqQIKaa9u3bF2mWlEqnBCYGj3t\/8XKSsaeEVCtTVMZ1P15a1i6d41LTugyTkm175Ti\/ZlQEFIHkQmDVqlWCbUjPnj2FL3N37xzHMVoCBlnsO1auXOmOjuoazQAaArQXfPU\/8cQTQt1RFRJhYvrCdIpNDklDO8I9miWvARvihq+u1atXmykZ0kYjDRo08Cdn+gZM\/QFhLtDMWJLDVNeuXbvC5Mgazf59aJ0cx8kaEeYOTc5FF10kpUuXDpMynqLzry1KYGKMdfcmFXNUIsSFjGzeWERtYIAiJoI6GjVsTArTQhSBAkCgdOnSArFwT5O4m8HgShrIjVur4U4T6XWLFi2M4S\/p161bJzgA5TpWAmlgu5hADQbttlNQGNvarWMC6z3zzDMF7U1geF7foyFiqxvqwTdZoIaI8FBCm5FQaYLFoXlj+ilYfCqHK4GJwdPfX7ysKaVqWlFzzsmhWN02AnnhnJP8mkcRUASSEwGmilgN4zjeX+9oK\/bv3y\/YnUSz0a0XWkzVQBJsHAatgWTDxuXkjIYIgvLPP\/9kyY52o02bNiYM7c99990n1113XbaVP\/j5uuyyy+Swww4zab0OwcIwyrVx7L8HabL3oc60Z+7cuYKGi3RgDE5cqxQsAkpgYoD\/vnLHmVIimTraOjk9qJ+XyukzTTl6UAQUgdgi8Ofqf+T717bGraz8YmeOO8yUBkaeDRo0MIaqOS7oYEa2a4EMcUvZ0U6XkM9LWIbMcm0vI1zSQ0xY1s01wtLj1q1bC7Y52OUQllNZtGiRvPbaayY7RrtMRUWiRaFeVnLRbjJXqVLFLNtWAgMaBS9KYGLwDPaVO9aU0qJWpnGdufE4sNJo65ThfmNdjyQapAgoAnmAwMov\/pL3b1idD5KzOuY8EL1hqoUJ7QCD8ZVXXhnSGNWmD3eGvDAdRbrdu3dnW9lEeChBU4HRK8anbsGIldVAwfJixzN+\/HjBPsau0mF11fDhw42dD0ujg+UNFo5dDcSFZdmQPFZe3X\/\/\/QLZ88qDzQ8aItqJHx2m1G677TaTFFsUpr90OsfAERcHJTAxeAyWwERalNq4RIqUplMEYoPAEVUPkZMuSYtbqX5qiRx1FL8ob7zxhnTr1k1w\/pajQmKcCfLDQI+hrFtYutypU6eQtTGVNGzYMIFEuI1e6ScEAn8sBw4cCFnGp59+apaLQ55OPvlk4+MG+xqcymHTE8oWhek4CCFtYKXSxo0b5d577xXIE75eatSoEbJujcxfBJTA5BJvViBRBCuQWoZYfbT7p1myqp86JQIrlbxDQEv2RgCC0OHRqhKv0mrwkd4NDxGKpgMvrywtZnCGOIRInqMoNBalS5fOUd7ATGlpacaPDZqWwLjAe6axcFz35JNPCm0gHuKCLxx8sXAfTE4\/\/XRZvHix8ZUDgWJp+XvvvSf4n8GGJlg+wiE9GP1jg8NqLMLGjBmTxXkeYSrxgYASmFw+B0tgqoYx4GX6iKrQvpRs05NLlXxAgB8ut5fOfKhSq1AE8hwBBnPsMhioeb8jIQWRNgpPvNZOpVq1ajGZlrJ1462X7QCY8rJhwc6O40i7du0EWxi84JKOfqPdwecN93kl7du3l4EDBxoHd2yxMGTIELOUPa\/q03JzhoASmJzhluNcrDJK65qe4\/zxnVFbpwgoAvmBwNSpU2XatGmCdiDY8uqctAOCgB8W8jqOI3jBjaXBaqlSpYwRLGfqsILNCcTJ3rvPrMJi+gY\/NYRjDIxWheu8Esdx5NJLLzUeeKlj0aJFwjYM4MO9SnwgoAQmPp6DtkIRUAQUgYgQQCPBHjsYt9pBPaKMESTKyMiQ6dOnm5RMp5x77rnmOi8O7AeEFgkndtiYPPvss0GrYfWP1cKQCFsVznkp2OMwnYR3YuphNVReeSemfJXoEUgqAhN993OfY+6yraaQUPYvJCjZppccPeWAlO\/\/PLcqioAioAhEjQAGqtiBsJLmxBNPzJZ\/7dq1ZtlxTgZ4VhyxcSIO7Bi8cZ7HXkLZKolBAPY7bMKIV11sTSAo77\/\/vrCJYbDiWR1FHG3DxT7XsRB86ATTrND\/kSNHmmk00owYMULmzJkTi2q1jBggoAQmlyDa\/Y+qlSmWy5I0uyKgCCgCwRFgQ0cMWNEKBCMv2MMw6EZiY+KuCeLDJodvvvmm4Gn25ptvFmxV3GlidQ0RYAqMXbSZonIcR7CNYZkzhrbYnATWRZid2mK1Fa75A9Pk9J6yQznrQ\/ODUz3HcQQHfKxKIk9gffiMgQQSTh8hRlwniCRkM5XAJORj00YrAopAKiHAEmQG9++\/\/17YQ4jpnUBh2TEbHeJl1mKDHxS833L\/xx9\/yG+\/\/calEdz6Uy6kB6PVGTNmmKXYuPHHj4vjOCZdJAfsUtCmkBbtCsbFnLm3wlQRxreQIwxkWV1kNSlsh4Ah8i+\/\/CJsugiRIj1EgDbfcMMNQhyEh40m0cLYcjmjNeKMQISs23\/uw4k7L8u1wcWdx3Ecs3oKl\/6E0w58wwSSHrD+\/fffSSK0gRVQ5ibEgSk7W862bdvUUDgEVl5RhbwCNUwRUAQUAUUgPhD48ccfhR2nGRTDtYhtAPB5gkEsjti6du0qdlAlL8utLfFBs3DjjVlWiawAABAASURBVDeawZZwbF\/Y9RqSQNpIhHrYEbtHjx6CFsfmQUuBlsTWxZldpjt06CBvvfWWoK2oU6eOVKxY0WQpW7askObyyy832pixY8eaPZmwP7n44ouN5oOpnAkTJojbaHnmzJnGKBitlCnId8Ce5vzzzxf6H4pEeOWFwLB\/GvZFbi0LS9TRwrC821eFMJWHJgjy9+677wqkrHv37oLfGOIRCGffvn2FVVOBU3o41yNu8ODBAkkj\/fr16wUPwZQ1f\/58glTCIFAoTLxGKwKKgCKgCBQgAkwXMaAxGIeTQYMGmaW\/2JSMGjVKvvrqK78\/lMC8aBKw58DXCoMpBsGOE7nWBUioBwIze\/bsoPUE1mvvsYGxK5wgJRCb4cOHC4752INp4cKFpkxWHBHGqiCbnroRnNJRDn2x5XKmPfQfUkQ6LwmWF8yGDRsmrH5y56ONTH1RPoI2DOIEWWKVFPkIt0KbaBvarMApPbQ5xJHGpuf8zTffCGVFQyLdbUy1ayUwefTE2fNoeRdHEOsDJo+q0mIVAUUgdwhobkVAEUhABJTA5MFDg7Cw5xFF47iOs4oioAgoAoqAIqAIxA4BJTCxw9KUBHnZuzHDXENeWDbN2QToQRHwQkDDFAFFQBFQBKJGQAlM1JAFzwB5Yb+j39Pb+hPhedd\/oxf5jgAGecxT53vFWqEioAgoAopAniKgBCaG8Lo1L0XrtpEE2fMohghoUYqAIqAIKAKKQP4goAQmhjhbbQvEpXL6TNE9j2IIrhalCCgCioAioAi4ECh4AuNqTDJcsl2AEpdkeJLah0RHAB8lKmtEMUg+DBL9bzNW7U8ZAoOXRGwhWF+PbwCcKHXp0sU4JMKpUqwA1XIUAUWgYBHAN0nTpk3NbsKtW7cWFcUg2d4BfOLwjvOuF+xfW8HWnhIEBs+MuKd+5plnjNdJIIe04Ea7T58+gtOiQLfXpFFJfATee+89wVtm4vdEexApAvyoP\/DAA\/LKK6\/kqeByf8yYMcbTal7XlSrlK6aRv7O845H+TSRruqQnMLiDZt8KzvXq1ZMPPvhAli1bJpCXq6++WgoVKiSTJk2SyZMnJ+sz1n4pAimHACSmWbNmxh19Xp7r16+f53XkZftzV3be4KuYRoYr73jK\/WEHdDjpCcxHH31kNgHDDfTYsWOF\/Tgcx5G0tDS55ZZbzB4jBw4cEPYAgeQE4KO3ioAioAgoAoqAIhCHCCQ1gdmxY4e88847BvZOnToJe32Ym4MHx3GEzc6OOuooWbJkidk35GBUVKdK+9ZL0\/srCT5gosqoiRUBRSDHCGhGRUARSG0EkprArF69WjIyMoSdRFGVej3qChUqyLHHHitoYb744guvJCHDKv+7QfpunxgyjUYqAoqAIqAIKAKKQGwRSGoCs27dOtm6dascccQRcuSRR3oid9hhh0nVqlVN3MqVK2Xnzp3mOtJD3z8nSse\/pkWaXNMlDQLaEUVAEVAEFIGCRCCpCcyGDRsMtsWKFTMkxtx4HCy5If3u3bs9UoQPKtmml1QePjN8Qk0RFoH169fLxIkThXPYxJogIgTAUjGNCKqIEymmEUMVcULFNGKoNKEPgaQmMNjA+PooRYoUEcdxuPSUihUrmvB9+\/aZqSRzE+WhaN3TJD83bYyyeQmVnB+xF154IaHaHO+NVUxj\/4QUU8U09ghoidEgkNQExgKBhgUtjL0PPJcoUSIwSO8VAUVAEVAEFAFFII4RyCGBieMeFVDT1qxZqy6718TGZbd9hGtiUB5lxaKcRC8DHJBE70c8tR88kXhqU6K3BTyRRO9HPLUfPJNVUoLAhLNtidZw1\/0y\/HXq\/8nktE5y5cgn1WV5jNy24yYbjDnn1gX4gAED5PPPP0\/5ZwOWscI0t88kWfIrprHfoiAlMI3R72Skf0dgmqzbDiQ1gSlZsiS\/2RLOtoW5bBKGs5UhTaDUaNhSegx9TB6eMDlP3Zaniitx7WfkrsQVK8VK3wF9ByJ5B5J124GkJjB433UcR1hZ9OeffwZyD\/89GhpuwtnKkMZLcOmMnxmVyFxgK06Kk74DBfYO6NYH+bDFRLy934xRXmNXooclNYHhoZUtW1YgL5akBD6wv\/\/+W3B4R3j16tVFDXpBQkURUAQUAUVAEYhvBJKawLA8Gid17DT91VdfeT6JjRs3yq+\/\/mqWWZ966qmeaTRQEVAEYoSAFqMIKAKKQIwQSGoCgw3MBRdcYKB6++23ZcWKFebaHtg+YPLkybJ27VqzT1KjRo1slJ4VAUVAEVAEFAFFII4RSGoCA+7t2rWTY445RthWoGfPnjJnzhz5999\/zRYD9957r4wfP95oX3r37i1s6kgelaRFQDumCCgCioAikCQIJD2BwZB39OjRUr58eeOnBRJTp04dady4sTzzzDPmMXbv3l26du1qrvWgCCgCioAioAgoAvGPQNITGB5B\/fr15b333pOrrrpKMOolrHDhwobEPP300zJ8+HCzYzXheSpauCKgCCgCioAioAjEBIGUIDAgVaFCBbn99ttl\/vz5snz5clmyZIlMmTJFTj\/9dIHMkEZFEVAEFAFFQBFQBOIPAa8WpQyB8eq8hikCioAioAgoAopAYiKgBCYxn5u2WhFQBBQBRSDfENCK4hEBJTDx+FS0TYqAIqAIKAKKgCIQEgElMCHh0UhFQBFQBAoeAW2BIqAIZEdACUx2TDREEVAEFAFFQBFQBOIcASUwcf6AtHmKQMEjoC1QBBQBRSD+EFACk4NnsnPnTsF\/TMuWLeXoo4+WWrVqyVlnnSVvvvmm7N69W\/RfZAiAF9hNnz49bIbvvvvO+PE57rjjDOb16tWTwYMHmyXxYTMnaQK2wli8eLHccMMN0rBhQ4MLeLZt21Yefvhh2b59e8ieK6be8IDr119\/Ld26dRPeM\/7GOffp00cWLlwoxHvnzAxVXDNxCHfcsWOHXHHFFea9\/d\/\/\/hcyuWKaHZ6PPvrIYMf7GUqC\/b4mA6aFssOiIaEQYN8kvPayDQHbE5CWH7SlS5fKzTffLP369RP+MAlXCY7AokWLBAzBLngqMYMFZLFz587y6aefyj\/\/\/GOS\/\/XXX\/LGG28Ie13xh2wCU+iwd+9eGTdunJx\/\/vny\/vvvy59\/\/ml6D54rV66Uxx57TM4880zhR8pEuA6kUUxdgLguwfWuu+4SvHNDYnjPiOY8Y8YM4T0knnSEu0VxdaMR+hqsXnrpJfn8889DJiSdvqveEC1btsw7IkxoMmGqBCbMw3ZHo13Ba+8vv\/wiVapUkYkTJxqHeN9\/\/72kp6fL4YcfLrNnzzZfv7wk7rx6\/R8COBEcNGiQbN68+b\/AIFf8wD300EOyf\/9+6dGjhyxYsED4w506dao0bdpUGFjA\/rfffgtSQnIG41kaLQv7erHfF4MruPz666\/C1yzv56ZNmwyphnS7UVBM3WhkvZ4wYYJMmjTJBPK+zZs3z2j5eO+uvvpqKVSokIknnUnkOiiuLjDCXPIBw3saJpkhOPr3nx2lffv2CeMQMXxQf\/XVVxJMWrVqRTK\/JNN7qgTG\/1iDXfwXzhcZBKVEiRLywAMPCC8GXny5RxU6fPhwcRzHbFuQagPqfygFv4LUoc7k63bFihXBEx6MgTA+++yzRuty7rnnCl++aWlpBuNjjjlGxo8fL2wTwUD9yiuvGG3NwaxJfULD9\/LLL5v+nnfeeUYTU7NmTYPLIYccYqYzH330UbNtBqQGj9MWEMXUIpH9zHuEVo\/3lHd06NChZg81UvLe3XLLLXLZZZcZ3N966y3ZsmULUUYUVwNDRAfeX\/an4xwqg2IaHB2mh\/EoT4pmzZoJnuaDyWGHHUYyI8mGqRIY81jDH2C82GygOm7durXZRykwFyr7Bg0aGM1CKk5rBOLhvkcL0L9\/f+Erlh\/+6tWrC8TPnSbw+ueff5Zvv\/1WihYtaubKGZzdaRhUrrzySjNwo4FYs2aNOzppr\/nhQosVDBc6DrGDYHMNhmiquFZMQcFbGFBLly4tlSpVMlOTRYoUyZLQcRzB7o1A3meEa0RxBYXwAjl86qmnhI\/BFi1amIE3WC7FNBgyYsaYP\/74w\/yG8vESPGXWmHzHNGv1Mb9TAhMhpFu3bpUff\/zRpG7evLkE\/rgRUbJkSTnppJO4NAOvHTRMQIofmO5g2qd48eIyZMgQM\/0GAQkFC2pmDKZr1KhhDKW90h533HFG04A9EnZIXmmSLYyBtlSpUnLkkUfKUUcd5dk9x3EELRWRkG8GDq4VU1DwFgwhX331VZk7d640adLEO9HBUP7+0b4evBXF1SIR+gxxYeqdQRfjc7d2IDCnYhqIyH\/3fKwxBV++fHmpXLnyfxFhrpINUyUwYR64jV6\/fr3AeNECoD2w4YHn2rVrmyAMKZXAGCjMoUyZMgJxYXDo27ev0aqYiBAHVtgQzR8o9kVcB0rZsmWlXLlyRq2P\/UdgfDLeowVgHnvmzJlBf7wgLHYak8HWcRwDhWJqYMjRAe3rtGnTTF5IddWqVc01B8UVFEILU3RMs\/\/9999y4403SqjfUUqKAaYUk5TCbx1\/47yHaFUuueQSOc73MQcJP\/XUU4UpOqaZAjufbJgqgQl8wkHuWeWxZ88eM\/AeccQRQVKJ+SomEs0BhIdrFZHbb79dIC5oDiLBgx+5bdu2maSlfWr9YF9qhx56qBBPQkgjZxWRjIwMo6YHC7SCEEDFFDSiF4jL\/PnzhelKbGR456699lpB40ppiisohBYG2yeeeMIYnrJyrn379iEzKKbB4QFL+3EyZ84c817yftoVmnxsP\/nkk9lWISYjpkpggr8nWWIgL\/yQ8TXrVh1nSeS7gdxgm4DanhUiviD9nwMEwHrXrl0mJ1ovc+FxAGu0O0SRh3OqCziw9JRpNTRUHTt2NJAQrpgaKCI+4Brh2GOPFb5w0R6iDZwwYYKw8ssWEhRXm+DgOZXfVTSGGNozdTRgwAAJ9TcNXIopKHgLmhW7CAIy414tx4rYe+65x5BrNF68v9ZWKxkxVQLj\/Y4EDcXwlCmLYAn4kQr3xxksr4Z7IxBK1QyhBHPvnKkXyg\/a888\/L6+99poxbsb5mrWFcaOhmLrR8L52f7HaFJBClu17+dchjeIKClkFzEaNGmUCGVBZ4m9uIjwoplmBQrvPcn7GIpx58j5iC0MqwnDACMlGQ8gqRD5m+F0g3kqyYKoExj5RPSsCCY4AP1IvvviijBkzBpsgozXo3bu3ITIJ3rUCaT5TRWPHjjV+h7A5wG8Jgy\/q+6uuusoY7hZIwxKoUjTRuDsAs4suuiiL5iqBuhFXTcVwn6lMtC1MyztOpn2bu5ENGjSQCy+80ATh+oNpJXOTZAclMFE+UNhvKNsWO9UUZbGaPAQCoWxb+IEE8xDZUyKK6Uq+tEaMGOH3m3PbbbcFVdUrpuFfC8dxzDJVx3EMjmwXYv3rsAIETRfvn7skxdWNhsisWbPk9ddfN6sIcaOQE+20Ypoa3nF8AAAQAElEQVQV00juHMcxjj5Ji9uKwDErWTBVAsMTjkCYpuCPjx8sBotgWayxL1MboWxlguVPmvBcdgSsWXJNMczdcvYSyAt\/oMSRh3OqCc6p7rvvPrPygHezU6dOwj0qZDcW4KOYuhGJ\/hr\/OjgPJCdLUnn3FFfQyC7YXrAahpiBAwcGXfJPfKAopoGIRH\/P3z844nZhw4YNhoQn29+\/EpgI3wsMRZlfZMCEpATLxotCHGlD2cqQRiU4Aqw6svO6rEbCHsErNZb3xBMXal6X+GQUfpyYB8djMf1jaoNBg\/ePe7copm40cnbtOI6gnic37x1\/74oraGQXluxig8HfKNoXlvi6BQ+yq1evNhnvv\/9+szHhaaedJhs3bhTF1MAS9MDvIUa6fLAES8SHHx\/cfHyzuCQZMU1WAhPsmeY4vGLFimaJNC9FKPWbdabGYMrS1RxXqBnlxBNPNChgBBjMpw6qfNSjjuMIq0VMhhQ54FyxX79+8uGHHwravltvvVUQvrqCQaCYBkNGhO0Z8LHToUOHLNsEBObgnSOMAYHBgWvFFRRiK4qpN57YYh1\/\/PGCphUtl3cqEX43sYvjY6Z06dImWbJhqgTGPNbwB\/yX1KlTxyT88ssvBWZrblwHvoYxrCKoUaNGZnNHrlVyhgCOmfjjy8jIMIaUXqWwoRkDCstbrRNBr3TJFsa7xnJUlvZibHrvvfcKK44gMqH6qpgGR4d36PfffzcbtP7www+eCfm7t3\/jGFOy\/wwJFVdQyCosNWfbi2DC5oPWGSD7TJEOg1PFNCuOgXeQED5SMMwNthqOaeVPPvnEZD3hhBPM5sPcRPae\/mK2KuDvId5\/U5XA8FQjEGxasKLnxfnss89kwYIF2XLxwjAvztzjGWeckS1eA6JDgD88iCDTdi+88IKg\/XKXgAbiueeeMytu+HJmDxt3fLJe81XF1gz41oC8sLHoxRdfHNFqI8U0+FvBwMBHCu8Z2hgGgcDU\/I1bb7xnn322oJonjeIKCrEVxdQbTzTNtWrVMr97vKd8zASmxHgagug4jvDbUKxYMZMk2TBVAmMea2SHxo0bC\/O2rERi+Zrd3JF79vfA2yyDCy8MP4aRlaqpgiHAHx1OmhikP\/jgA2EenTlzMGZZ5jXXXCN8gWAr07t3b8\/9qSQJ\/7E5Iys76BpfuZA85sOxHfAS4uxcuWIKat7Ce8TfteM4wq7paLh430jNIIEnWTQF2HRAmNmZmjhEcQWF2IgtRTG1SGQ9855ef\/31wu8ie0vxzvJ7yO+ifU9vuukm\/2pEt9fjZMNUCUzWdyPkHdMZeDlEDceLglMm2DCu2tnjgx82jNB4eRzHCVmWRkaGAJqsQYMGGRsPBhXw5euDP0rcZ2NnlJ6e7t+4MLJSEzvV+++\/L5BmeoH9S6tWrQyxhlx7SefOnY1KmPSIYgoK3nLBBRcYOyKm4uz7huEpq4\/QdGGLxUaPXqu8FFdvTHMTqph6o8fvn\/1dnDdvnnDP76J9T+1YxHjFrIG7lGTCVAmM+8lGcM28NzvWDhkyxL+RnuM4wlzhgw8+KDhtYgopgqI0SQQIOI4jrKyZPHmynH766earg2wQFzRd77zzjpxzzjkEpYRAXHCqlpvOOo5iGgy\/woULmz270PjxQ897Rlq+dhs0aCCPPfaYvPTSS\/6\/feKsOI7iarGI1dlxFFMvLB3HMTZv9j21U5n2PR03bpw888wzZkuBwPyO4yTNb6oSmMCnG8E9Br2o7bBBwPCMpYIff\/yxYCODii6CIlI+CYZ6GOyBH9MgoQBxHEcaNmxo\/iAx2iUPRpZ8EfN1HCpvssWhBWRPGTCIVMAZvN1YOI5i6sYj8JrtF3AMyHsGzrx3TBnjAybwi9ad13EUVzceoa55J3k3wZfp4GBpHUcxDYaNfU8XLlwo4GjfUz7q0CIGy+c4yYGpEphgT1jDFYECQECrVAQUAUVAEYgMASUwkeGkqRQBRUARUAQUAUUgjhBQAhNHD6Pgm6ItUAQUAUVAEVAEEgMBJTCJ8Zy0lYqAIqAIKAKKgCLgQiCuCIyrXXqpCCgCioAioAgoAopAUASUwASFRiMUAUVAEVAEFIGEQCAlG6kEJiUfu3ZaEVAEFAFFQBFIbASUwCT289PWKwKKgCJQ8AhoCxSBAkBACUwBgK5VKgKJggAbG+LQ7dxzz5WZM2fmuNmxKifHDdCMioAikHQIKIFJukeqHYoXBBjw8djM3ll4DA4U9i5p1qyZXHHFFYJLcAb5eGm7bceSJUvkf\/\/7n+Dh86GHHhJ2ALdx0ZxjVU6QOmMWzP5mPKd69erJddddJ7fffruRH3\/8MWZ15Kag7du3y+jRo02baOvJJ58stBeCuWXLltwUrXkVgYRDQAlMwj0ybXCiINC2bVt56qmnzGBj23zKKafId999Jwzon332mfTr109+\/vlnYXfZ5s2b50rLYeuI5blatWrSsmVLs6fK2Wefbc6Sg3+xKicHVecoC\/tsse8Rm+EhJ554Yo7KiXUmtjG59dZbhTax99qdd94Z6yq0PEUgYRBQApMwj0obGrcIhGlY5cqV\/SnYy4gN19in5KijjjLal7fffltq1qwpfEEPGDBAFixY4E9f0Be095FHHjGkC5JVpEiRHDUpVuXkqPIcZHIcRxzHyUHO\/M0Crvlbo9amCMQPAkpg4udZaEtSFIEqVaoIm6\/R\/Z07d8qkSZO4VFEEFAFFQBEIgYASmBDgJEiUNjMJEGjQoIG\/F8uXLxdsHfwBeqEIKAKKgCKQDQElMNkg0QBFQBFwI\/Dvv\/\/K\/PnzhdVIaIjccdFcx6qcaOrUtIqAIpC8COSewCQvNtozRSDfEFi0aJG\/rhNOOEGKFy8urEqCOPTu3VtGjRol69atk27dugmrlzp16iSbNm0yeQ4cOCBff\/219OzZU+yKJwxvn3vuOYE0mEQBh99++0369OkjrVu3Nka6J510krlfsWKFP+XGjRuNsSgrpS655BJ566235J9\/\/hH3P1Yl3XHHHaZdQ4YMEVbDXH755cIKGeJIG0k5pEPrBEmiH6z+ufTSS4W6KXft2rUk8Qv9whCaFTlnnHGGfP7550Lbr7rqKoMBGF1wwQVCP\/2ZYnhBe2gX+LVq1UrAD4x4Xu5qduzYIR9++KFceOGFcvfdd8vu3btl4sSJBnNWD\/G86Cvh9Onjjz8W2l2nTh3znFmhZnF0l6vXioAiIKIERt8CRaCAEYC8vPbaa6YVZcuWlc6dO8vkyZOlffv2hhjMnj1bli5dKjfddJOsWbNGICyLFy+WX3\/91ZCce++9V4YPHy5XX321sNz33XfflcMOO0xGjhwp1157rRk0TeG+A4MkK6N69OghHTt2FJZ6M\/hDOGbMmGFscebMmeNLKYZE9erVS1g5ZQICDmhjbrzxRlm9erXRztx3333y3nvvSdOmTY3GBgJGFshYqHJIQ7s7dOggYDF27FhDnF555RVhpQ3tgqS8+eabJDUybdo0efHFF80qL+qnz\/3795dGjRrJ0KFDpUaNGvLDDz8I7cM42mSKwQHsLSGBgNAO8GLFEm2HYL788sumpi+++MI8l9tuu80YQfMMIWWQTQyir7nmGpPu1VdfNf1kRRH49fYR1ltuuUUw0OXZDBs2TPbt22fS6iG5ENDe5A4BJTC5w09zKwI5RoCBFeKC1mDz5s1SpkwZuf\/++wV7GAY6Bmy+7KmAAZ5BD8LBAM\/ATLoJEyYIZTDgnXrqqcIqIZb8Mmg6jiPTp08XiABlIO+8846MGTPG+Dg5\/\/zzhdVQhLOEu1y5ckbD8tJLL5kzAygrqPCJQppAWbRokXz11VeGLJQsWdJEUx6aCFZY\/fnnnyYsXDloliBQaCHQCqWlpZl8HNAk0VfIEJoK6iMcTQ9+WjCAZnDftm2b0WywLB3sRowYIUWLFhXsiWKphYF8oHlBA3bllVdKsWLFaI6AEVofCA5G2GhNeB4PPPCAQL5IlJGRIenp6TJu3DgBo8GDBxuCShyk55hjjpGxY8cKZYPDXXfdRZR8++23sn79enOtB0VAEfgPASUw\/2GhV4pAniPw6aefyvHHH2+cj+GEDC1JpUqVjB8Yvuzbtm3rbwNkwA6Q7dq1M9MphxxyiEA8rGbljTfeMFMNDRs29OfjgsGwYsWKXAoaDLQlTDmhfWHQt4OqSeA71K5d22hs0MxAqFjq7QsO+R9SAXlAC4EvG5u4fPnyQj8gUzYs1BltEySjatWqBpfAtKeffrrgR4bpq\/Hjx8tff\/0VmEROO+00oV4bQX+4J8+yZctscK7OEKwnnnjCLK9Ge+XuH+QTLRj43XDDDVK6dGlTl+M4UqhQ5s8sz4QpIzn4z3Eco63iFv8ukDWeOffIsccea\/zuMLW2YcMGgmIsWpwikNgIZP5lJXYftPWKQMIgwGDM9A+aAYRpDqYNmB6qUKFC0H6gTXAPmCRkCgm7D2xBmH7BHsMKAymDIWRl\/\/79ZhrJpmd6BW0LZVhxHEfOOussY6fRpEkTGxzyjKaHQRZihN0GztUY5MmEBqFmzZpchhTICBoGEkECmPri2i0Qkbp165og+oBNjbnJ5wPTdxA1tEtIYPXghp0LODqOExjteQ8h9YxwBf7999\/m+bmC9FIRUAR8CCiB8YGg\/xWBRESAwRwtCDYnn3zyieDZ10uwsYAE2PSx6itlMuUFScK25plnnhEMWqmP+3D1EA+BWblyJZdBBeJmNRdMFWFAGzRxHkZAYJjqy8MqtGhFQBGIAgElMFGApUkVgXhCAPJCe7CP2LVrF5chJdr0IQs7GIkW5v333xdWHqHxwa4HWxWmoViBczBZ0JPjOMZuhwTkRdvAdaBAYghznP\/Sc5+fAinDxgUSY+178rN+rUsRUASyIqAEJiseepcyCCR+R62NC6tw0A4E6xHEAEPZ6tWrmyRMXbEixtx4HCiPwdojKksQZAPBfgP7D+x7WDnlOI6wcooppSwZPG7Ia6eamBrC3sMjmT8Ie6EjjzzSf5+fF2icMEhmymzRokVBq\/799991yicoOhqhCMQOASUwscNSS1IE8hUBVr2w+gdNB\/5TMKgNbABxrJqB4GDYiu3Lnj17zLJna6\/izsPAzJLgSDQ6LB9m6bTNjxEuK2xYUUQY9j2sxuE6mGDzgt0O8ZCshQsXcplF0Hpg5Esgq30sceM+P4X+MV1GneANkeHaLUxvQdzC9dudR68VAUUgZwgogckZbrnOpQUoArlFAEKC\/QvlMI3Dsls3KUGT0rdvX8Fol1VKaDrs6iOWV+ObBSNg8jO9xLLtgQMHCsuQIUaEhxNsbtwDueM4xkkb+VmCjfFxuDLYB8rauLAEmRVT7jyQLxzE4SPHvXTZnSY\/rjEyZgUYdf3yyy9mCTSEC4KFxgpCB6Ys8abvpFNRBBSBvENACUzeYaslKwIGATQL5sJ3wH4CDYjvMux\/BkVLSNCkBGZgiTVOztDEEIc\/GPzGsBy3WbNm1Xj8KAAAA5RJREFU0qZNG7MMlxVBrHbBjgRfMvXr1ye5cTYHocEhG6uJmAZiNZSNN4l8B7uEl+mdQM0MBOjxxx\/3e\/xlMP\/yyy\/N0uFu3br5\/aT4ipFg5TA1w47XLDOeN2+ePProo0I55KH\/LJ3mjIM6d9vA0WqdwIr0VtCA2LZ6YWfTRXu+7LLLjLdh8tH3iy++2Cxjx3MuOHfp0kVY8k48Qvtop71mRRjXViCOXDMVZ9Nxj3BPPFgghKkoAoqA+CFQAuOHQi8UgdgigNM5NCBMKdiSv\/vuO+PHBff72KLY8MAzGhI8637\/\/fcmCgd0OG7DtoRB0QT6DiznxRcMadFQMJD\/8ccfggbgoYceEogBBMGX1PwnfOLEicZrL+kJPPzww4Vl2FOmTDHeeR0ncwkw2gVIzeuvv04ys5UB3m5xd08b0K6ceeaZgo3NRRddZLYPwIswtjBPPvmk8QVDxnDlkAYCQD0QMrzPstycqSj6jDaH\/uN7xXEy24ZDOTDE3oT89BWBuKAJwbEfZJE4fLcw5QMJ4j43QltwTkc7wZKy8JnD6qsXXnjBbMcAWSSc5e041ONZcj937lzB2y7L6GknDgPxA0QcJOvWW281noyJAwv6B4kBa\/KxuisWfaA+FUUgGRBQApMMT1H7EJcI4MwNx3FMN0BWrMyePdvsbYTmI1jD+YpnwMIJG\/kYDNF04LANTYo7H4awDH5Ms5CW+pgOwqMrWhp3Wq4D02OrgtYDDQjxVph2YqAmnnIRiAR+TmgDmh6ICjYzhOP2HzJFu9l+wHEyyUa4cmx9GMiy9w8O\/SB\/lPfss88Kq5ogajYdZ8pnKwHahEAMBw0aJHjxhUzQHsIR4tCOeGFBWdEK5bBfE0SL8sEbUkibHCezz5QJKRs+fLgQTzqeISQLR4a0E189kBriEPoNSSMOIghRJBzhnQnUaFGHSgEgoFXGDQJKYOLmUWhDFAFFQBFQBBQBRSBSBJTARIqUplMEFIGUQYApnTjtbJZmYSOTJUBvFIEUQkAJTAo9bO2qIqAIRIYAU1BMOzF9xTQa0ziR5czbVBhSjx492kyrYSPEyrO8rVFLVwTiFwElMPH7bLRlikD8IZDkLcLuBrKCvQqGvxhgY8cSyl4pPyGx9ku0i7Z+8803Qnuxn2GZd362RetSBAoaASUwBf0EtH5FQBFQBBQBRUARiBoBJTBRQ6YZChABrVoRUAQUAUVAETAI\/D8AAAD\/\/zg6IVQAAAAGSURBVAMARqrUcR6k9YgAAAAASUVORK5CYII=","height":337,"width":560}}
%---
%[output:4fafaeca]
%   data: {"dataType":"text","outputData":{"text":"\n================ RESULTS SUMMARY ================\n","truncated":false}}
%---
%[output:604fe58a]
%   data: {"dataType":"text","outputData":{"text":"Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n","truncated":false}}
%---
%[output:943faa00]
%   data: {"dataType":"text","outputData":{"text":"DQN Agent     |          0.33 |        0.17 |                 1.15 (Steps: 24.3)\nDQN Agent     |        165.68 |      171.25 | ","truncated":false}}
%---
%[output:445abb86]
%   data: {"dataType":"text","outputData":{"text":"Brute Force   |         13.53 |        8.22 |                48.65\n","truncated":false}}
%---
%[output:0755fce3]
%   data: {"dataType":"text","outputData":{"text":"2-D MUSIC     |       1438.93 |     1588.71 |              1930.50\n","truncated":false}}
%---
%[output:9765fc69]
%   data: {"dataType":"text","outputData":{"text":"2-D ESPRIT    |       1439.02 |     1588.71 |              1930.51\n","truncated":false}}
%---
