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
Calibration_tilt; %[output:87a42bb6]

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
    fprintf('Trained agent successfully loaded.\n'); %[output:0b044ab9]
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

fprintf('Running %d joint evaluation episodes...\n', N_eval); %[output:1d610a2e]

% Preallocate main arrays to satisfy parfor slicing rules
K_factor = 15;%15:-4:7;
num_K = length(K_factor);

err_dqn_ref = zeros(N_eval, num_K);
err_dqn     = zeros(N_eval, num_K); 
steps_dqn   = zeros(N_eval, num_K); 

parfor k = 1:num_K %[output:group:47ed160a] %[output:60394cfb]
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
        [obs, LoggedSignals] = resetFunction_nav_CST_Aligned_tilt(EnvPars_local);
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
%%

% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
num_curves = length(K_factor);
leg_labels = cell(num_curves, 1);
colors = lines(num_curves); % Define distinct colors for curves and patches

figure; hold on; box on; grid on; %[output:02f7580b]

% 1. Plot the CDFs
for i = 1:num_curves
    h_dqn(i) = cdfplot(err_dqn_ref(:,i));     %[output:02f7580b]
    set(h_dqn(i), 'LineWidth', 1.5, 'Color', colors(i, :));
    leg_labels{i} = sprintf('DQN K = %d', K_factor(i));
end

% 2. Set limits first so xlim_1 and xlim_2 are accurate for the annotations
xlim([0, 1000]); %[output:02f7580b]
yticks([0, 0.2, 0.4, 0.6, 0.8, 1]); %[output:02f7580b]
xl = xlim; 
xlim_1 = xl(1); 
xlim_2 = xl(2);

% 3. Patches and 90% Confidence Markers
confidence = 90; % confidence level
conf_val = confidence / 100;

% Draw the horizontal 90% line once across the whole graph
plot([xlim_1 xlim_2], [conf_val conf_val], '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off'); %[output:02f7580b]
text(5, conf_val + 0.02, strcat('$', num2str(confidence), '\%$ Confidence'), 'Interpreter', 'latex', 'FontSize', font-2); %[output:02f7580b]

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
    patch(x_coords, y_coords, curve_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off'); %[output:02f7580b]
    plot([x_val x_val], [0 1], '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');

    % Add the X value text
    text(x_val, 0.6, strcat('$', num2str(x_val, 4), '$ cm'), ...
        'Interpreter', 'latex', 'FontSize', font-2, 'Rotation', 90, ...
        'VerticalAlignment', 'bottom');
end

% 4. Final Formatting
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', font); %[output:02f7580b]
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', font); %[output:02f7580b]
legend(leg_labels, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', font); %[output:02f7580b]
title(''); %[output:02f7580b]
set(gca, 'FontSize', font); %[output:02f7580b]

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
for i = 1:N_eval %[output:group:3c42e191]
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval); %[output:37900f64]
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
%     %% ================================================================
%     % MUSIC AND ESPRIT USING THE EXACT DQN CHANNEL
%     % ================================================================
% 
%     % Static digital front-end response, if enabled.
%     h_digital = c_digital .* h_episode;
% 
%     % The channel returned by generateChannel is normalized to unit
%     % average element power. Therefore this uses the same controlled SNR
%     % convention as the DQN receiver.
%     snrLinear = db2pow(LoggedSignals.SNR_dB);
% 
%     % Unit-amplitude transmitted symbols. The channel remains fixed for all
%     % K_snap snapshots in the current coherence interval.
%     symbols = exp(1i * 2*pi * rand(1, K_snap));
% 
%     % The same element-space data matrix is passed to MUSIC and ESPRIT.
%     digitalNoise = (randn(EnvPars.N, K_snap) + 1i*randn(EnvPars.N, K_snap)) / sqrt(2);
% 
%     X_digital = sqrt(snrLinear) * h_digital * symbols + digitalNoise;
% 
%     %% ================================================================
%     % 2-D MUSIC
%     % ================================================================
%     R_music = (X_digital * X_digital') / K_snap;
% 
%     % Enforce Hermitian symmetry against numerical roundoff.
%     R_music = 0.5 * (R_music + R_music');
% 
%     [eigenvectorsMusic, eigenvaluesMusic] = eig(R_music, 'vector');
%     [~, musicOrder] = sort(real(eigenvaluesMusic), 'descend');
% 
%     % One transmitted MU signal:
%     % first eigenvector = signal subspace;
%     % remaining eigenvectors = noise subspace.
%     E_noise = eigenvectorsMusic(:, musicOrder(2:end));
%     musicDenominator = sum(abs(E_noise' * A_music).^2, 1);
% 
%     P_music_visible = 1 ./ max(musicDenominator, 1e-18);
% 
%     % Place the spectrum back onto the complete rectangular grid so that
%     % your existing parabolic peak-refinement helper can be reused.
%     [psi_x_music_signed, psi_y_music_signed] = pick_peak_local( ...
%         P_music_visible, PsiX_grid, PsiY_grid, visibleGrid, Refine);
% 
%     psi_x_music(i) = mod(psi_x_music_signed, 2*pi);
%     psi_y_music(i) = mod(psi_y_music_signed, 2*pi);
% 
%     pos_est_music = estimatePosFromAngles(psi_x_music(i),psi_y_music(i),EnvPars,pMU);
%     err_music(i) = 100 * norm(pos_est_music(1:2) - pMU(1:2));
% 
%     %% ================================================================
%     % 2-D ESPRIT
%     % ================================================================
%     % Static, element-dependent CST gains break exact shift invariance.
%     % A calibrated ESPRIT receiver removes them before forming its
%     % covariance matrix.
%     if useCSTDigitalFrontEnd
%         X_esprit = X_digital ./ c_digital;
%     else
%         X_esprit = X_digital;
%     end
% 
%     R_esprit = (X_esprit * X_esprit') / K_snap;
%     R_esprit = 0.5 * (R_esprit + R_esprit');
% 
%     [eigenvectorsEsprit, eigenvaluesEsprit] = eig(R_esprit, 'vector');
%     [~, espritOrder] = sort(real(eigenvaluesEsprit), 'descend');
% 
%     % Principal signal-subspace vector.
%     e_signal = eigenvectorsEsprit(:, espritOrder(1));
% 
%     % x-shift selection:
%     % every element except the final x element of each row.
%     J1x = find(EnvPars.n_x(:) < EnvPars.N_x);
%     J2x = J1x + 1;
% 
%     % y-shift selection:
%     % every element except the final y row.
%     J1y = find(EnvPars.n_y(:) < EnvPars.N_y);
%     J2y = J1y + EnvPars.N_x;
% 
%     assert(all(J2x <= EnvPars.N), 'Invalid ESPRIT x-shift indices.');
%     assert(all(J2y <= EnvPars.N), 'Invalid ESPRIT y-shift indices.');
% 
%     % Least-squares rotational invariance factors.
%     phi_x_esprit = (e_signal(J1x)' * e_signal(J2x)) / max(e_signal(J1x)' * e_signal(J1x), eps);
%     phi_y_esprit = (e_signal(J1y)' * e_signal(J2y)) / max(e_signal(J1y)' * e_signal(J1y), eps);
% 
%     % generateChannel uses exp(+j psi n), so the phase of each rotational
%     % factor directly gives the corresponding electrical angle.
%     psi_x_esprit(i) = mod(angle(phi_x_esprit), 2*pi);
%     psi_y_esprit(i) = mod(angle(phi_y_esprit), 2*pi);
% 
%     pos_est_esprit = estimatePosFromAngles(psi_x_esprit(i),psi_y_esprit(i),EnvPars,pMU);
%     err_esprit(i) = 100 * norm(pos_est_esprit(1:2) - pMU(1:2));
% end
% 
%     % % -----------------------------------------------------------------
%     % % GENERATE EQUIVALENT RAW SNR FIELD FOR TRADITIONAL ALGORITHMS
%     % % -----------------------------------------------------------------
%     % % Compute direction cosines
%     % dv = pMU - posS;
%     % d  = norm(dv);
%     % u_true  = dv(1)/d;  
%     % v_true  = dv(2)/d;
%     % 
%     % % Path loss & element patterns scaling
%     % cosT   = hgt / d;
%     % ampRel = (dref/d) * cosT^2;
%     % snrLin = db2pow(EnvPars.SNR_dB) * ampRel^2;
%     % 
%     % % Steer vector
%     % kappa  = 2*pi / EnvPars.lambda;
%     % a_true = c_cst .* exp(1i * kappa * (xe*u_true + ye*v_true));
%     % 
%     % % Generate multi-snapshot digital covariance
%     % S = exp(1i*2*pi*rand(1, K_snap)); % Random phase symbols
%     % X = sqrt(snrLin) * a_true * S + (randn(N, K_snap) + 1i*randn(N, K_snap))/sqrt(2);
%     % R_cov = (X*X') / K_snap;
%     % 
%     % % -----------------------------------------------------------------
%     % % 2-D MUSIC ESTIMATION
%     % % -----------------------------------------------------------------
%     % % Spatial scan grid
%     % u_ax = ((1:T_x) - (T_x+1)/2) * (2/T_x);
%     % v_ax = ((1:T_y) - (T_y+1)/2) * (2/T_y);
%     % [Ug, Vg] = ndgrid(u_ax, v_ax);
%     % vis = (Ug.^2 + Vg.^2) <= 0.98;
%     % A_vis = exp(1i * kappa * (xe*Ug(vis).' + ye*Vg(vis).'));
%     % A_est = A_vis.*c_cst;
%     % 
%     % [E_noise, lam] = eig((R_cov + R_cov')/2, 'vector');
%     % [~, ix]  = sort(real(lam), 'descend');
%     % En = E_noise(:, ix(2:end)); % 1 Signal subspace, remaining are noise
%     % 
%     % P_music = 1 ./ max(sum(abs(En'*A_est).^2, 1), 1e-18);
%     % [u_mus, v_mus] = pick_peak_local(P_music, Ug, Vg, vis, Refine);
%     % 
%     % % Map back to position
%     % w_mus = sqrt(max(1 - u_mus^2 - v_mus^2, 1e-6));
%     % pos_est_music = [posS(1) + u_mus*hgt/w_mus, posS(2) + v_mus*hgt/w_mus];
%     % err_music(i) = norm(pos_est_music - pMU(1:2)) * 1e2;
%     % 
%     % % -----------------------------------------------------------------
%     % % 2-D ESPRIT ESTIMATION
%     % % -----------------------------------------------------------------
%     % Es = E_noise(:, ix(1)); % Signal eigenvector
%     % 
%     % J1x = find(nx_idx <= N_x-1); J2x = J1x + 1;
%     % J1y = find(ny_idx <= N_y-1); J2y = J1y + N_x;
%     % dxs = (xe(J2x(1)) - xe(J1x(1)));
%     % dys = (ye(J2y(1)) - ye(J1y(1)));
%     % 
%     % Fx = pinv(Es(J1x)) * Es(J2x);
%     % Fy = pinv(Es(J1y)) * Es(J2y);
%     % 
%     % u_esp = angle(Fx) / (kappa*dxs);
%     % v_esp = angle(Fy) / (kappa*dys);
%     % 
%     % w_esp = sqrt(max(1 - u_esp^2 - v_esp^2, 1e-6));
%     % pos_est_esprit = [posS(1) + u_esp*hgt/w_esp, posS(2) + v_esp*hgt/w_esp];
%     % err_esprit(i) = norm(pos_est_esprit - pMU(1:2)) * 1e2;
end %[output:group:3c42e191]

% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
%plot limits
xlim_1=0;
xlim_2=150;

figure; hold on; box on; grid on; %[output:3a471d79]
h_dqn = cdfplot(err_dqn_los);    set(h_dqn, 'LineWidth', 1.5); %[output:3a471d79]
% h_dqn = cdfplot(err_dqn);    set(h_dqn, 'LineWidth', 1.5);
h_bf  = cdfplot(err_bf_los);     set(h_bf, 'LineWidth', 1.5, 'LineStyle', '--'); %[output:3a471d79]
h_mus = cdfplot(err_music);  set(h_mus, 'LineWidth', 1.5); %[output:2049e81f]
h_esp = cdfplot(err_esprit); set(h_esp, 'LineWidth', 1.5);

%patch
confidence=90;%confidence level
[F,X]=ecdf(err_dqn_los);
[~,idx]=min(abs(F-confidence/100))
x_coords = [0  X(idx) X(idx) 0];
y_coords = [confidence/100 confidence/100 1  1];
patch(x_coords, y_coords, 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
plot([xlim_1 xlim_2],[confidence/100 confidence/100],'--','Color',[0.4 0.4 0.4]);
plot([X(idx) X(idx)],[0 1],'--','Color',[0.4 0.4 0.4]);
text(5,0.9,strcat('$',num2str(confidence),'\%$ Confidence'),'Interpreter','latex','FontSize',font-2);
text(X(idx),0.6,strcat('$',num2str(X(idx),4),'$ cm'),'Interpreter','latex','FontSize',font-2,'Rotation',90);
set(gca,'FontSize',font);

xlim([0, 50]);
yticks([0,0.2,0.4,0.6,0.8,1])
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', font);
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', font);
legend({'DQN Agent', 'Brute Force', '2D MUSIC', '2D ESPRIT'}, ...
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', font);
title('');

% Print Summary
fprintf('\n================ RESULTS SUMMARY ================\n');
fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n');
fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_dqn_ref_los), median(err_dqn_ref_los), prctile(err_dqn_ref_los, 95), mean(steps_dqn));
fprintf('Brute Force   | %13.2f | %11.2f | %20.2f\n', mean(err_bf_los), median(err_bf_los), prctile(err_bf_los, 95));
fprintf('2-D MUSIC     | %13.2f | %11.2f | %20.2f\n', mean(err_music), median(err_music), prctile(err_music, 95));
fprintf('2-D ESPRIT    | %13.2f | %11.2f | %20.2f\n', mean(err_esprit), median(err_esprit), prctile(err_esprit, 95));
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
    % % CORRECTED inverse. psi in [0,2pi) (computePsiFromPos / estimator convention).
    % % Convert to a SIGNED direction cosine, then project along the true drop.
    % ux = EnvPars.lambda/(2*pi*EnvPars.d_x) * wrapToSigned(psi_x_est);
    % uy = EnvPars.lambda/(2*pi*EnvPars.d_y) * wrapToSigned(psi_y_est);
    % 
    % s = ux^2 + uy^2;
    % s_max = sin(deg2rad(72))^2;          % FoV guard (room corner ~70.5 deg)
    % was_clamped = s > s_max;
    % if was_clamped
    %     sc = sqrt(s_max/s);  ux = ux*sc;  uy = uy*sc;  s = s_max;
    % end
    % uz = sqrt(1 - s);
    % 
    % drop = EnvPars.pos_SIM(3) - pos_MU_true(3);
    % pos_MU_est(1) = EnvPars.pos_SIM(1) + drop*ux/uz;
    % pos_MU_est(2) = EnvPars.pos_SIM(2) + drop*uy/uz;
    % pos_MU_est(3) = pos_MU_true(3);

    %Including the tilt
    u1 = EnvPars.lambda/(2*pi*EnvPars.d_x) * wrapToSigned(psi_x_est);
    u2 = EnvPars.lambda/(2*pi*EnvPars.d_y) * wrapToSigned(psi_y_est);
    s  = min(u1^2 + u2^2, 1 - 1e-12);
    dg = EnvPars.R_tilt * [u1; u2; sqrt(1 - s)];      % global ray direction
    R_cap = 30;                                            % hall-scale containment [m]
    slope_min = (EnvPars.pos_SIM(3) - pos_MU_true(3)) / R_cap;
    if dg(3) > -slope_min, dg(3) = -slope_min; end
    t  = (pos_MU_true(3) - EnvPars.pos_SIM(3)) / dg(3);
    pos_MU_est      = (EnvPars.pos_SIM(:) + t*dg).';
    pos_MU_est(3)   = pos_MU_true(3);
end

function u = wrapToSigned(psi)
    % (pi,2pi) -> (-pi,0); [0,pi] unchanged. Maps DFT phase to signed cosine.
    u = angle(exp(1i*psi));
end

function [psi_x, psi_y] = computePsiFromPos(pos_MU, EnvPars)
    d  = pos_MU(:) - EnvPars.pos_SIM(:);
    d  = d / norm(d);                     % global unit vector, SIM -> MU
    ul = EnvPars.R_tilt.' * d;            % LOCAL direction cosines [u1; u2; u3]
    psi_x = mod(2*pi*EnvPars.d_x*ul(1)/EnvPars.lambda, 2*pi);
    psi_y = mod(2*pi*EnvPars.d_x*ul(2)/EnvPars.lambda, 2*pi);
end

% function [psi_x, psi_y] = computePsiFromPos(pos_MU, EnvPars)
%     % Geometry: vector from SIM to MU. SIM on the ceiling, array in the
%     % horizontal (xy) plane, z-axis pointing down toward the floor.
% 
%     delta = pos_MU(:) - EnvPars.pos_SIM(:);   % column vector
%     rng   = norm(delta);
% 
%     % Direction cosines along the array's x and y axes
%     u_x = delta(1) / rng;
%     u_y = delta(2) / rng;
% 
%     % Electrical angles (assuming d_y = d_x)
%     psi_x = mod(2*pi * EnvPars.d_x * u_x / EnvPars.lambda, 2*pi);
%     psi_y = mod(2*pi * EnvPars.d_x * u_y / EnvPars.lambda, 2*pi);
% end

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
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     x_max: 20\n                     y_max: 20\n              calSpacing_m: 0.5000\n      sectorBoresights_deg: [0 120 240]\n            selectedSector: 1\n       sectorHalfWidth_deg: 60\n                 MU_margin: 0\n                     N_cal: 400\n                  aimPoint: [3×1 double]\n                    R_tilt: [3×3 double]\n              channelModel: 'rician_los_nlos'\n                    fc_GHz: 28\n                    V_hall: 72000\n                    S_hall: 18000\n                   mu_lgDS: -7.2781\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 7\n                sigma_K_dB: 8\n                        KR: 14.9595\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 1\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         M: 196\n                       M_x: 14\n                       M_y: 14\n                         T: 3600\n                       T_x: 60\n                       T_y: 60\n                    SNR_dB: 34.6606\n                 theta_min: 1.7657\n                 theta_max: 4.5175\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 24\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [10 10 4]\n                    pos_MU: [85.4628 56.2422 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 … ] (1×3600 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 … ] (1×3600 double)\n                      h_MU: 1.5000\n                    L_hall: 120\n                    W_hall: 60\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 180\n                 tolerance: 0.0262\n         StopTrainingValue: 58\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9800\n"}}
%---
%[output:41f1396b]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 2.366%n","truncated":false}}
%---
%[output:87a42bb6]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nFull local grid: 40 x 40 = 1600 candidate positions\nSelected sector 1: boresight = 0.0 deg, angular interval = [-60.0, 60.0] deg\nCalibration positions retained in sector: 570\nCalibration complete.\n\n","truncated":false}}
%---
%[output:0b044ab9]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:1d610a2e]
%   data: {"dataType":"text","outputData":{"text":"Running 100 joint evaluation episodes...\n","truncated":false}}
%---
%[output:60394cfb]
%   data: {"dataType":"text","outputData":{"text":"Worker K=15 dB: Evaluating episode 50 \/ 100...\nWorker K=15 dB: Evaluating episode 100 \/ 100...\n","truncated":false}}
%---
%[output:02f7580b]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7sXQnYTdX3XuaZEBlDJDSgiJAxiQyRMSFjQsg8j5EpTUhFhkJkHjJUlGgwhDKHkjGZ5yn+\/3f7nc\/97nfvd86995x7zzn3Xc\/TU313733Wftc+d7937bXXSnD79u3bQiECRIAIEAEiQASIgIMQSEAC4yBrUVUiQASIABEgAkRAIUACw4VABIgAESACRIAIOA4BEhjHmYwKEwEiQASIABEgAiQwXANEgAgQASJABIiA4xAggXGcyagwESACRIAIEAEiQALDNUAEiAARIAJEgAg4DgESGMeZjAoTASJABIgAESACJDBcA0SACBABIkAEiIDjECCBcZzJqDARIAJEgAgQASJAAsM1QASIABEgAkSACDgOARIYx5mMChMBIkAEiAARIAIkMFwDRIAIEAEiQASIgOMQIIFxnMmoMBEgAkSACBABIkACwzVABIgAESACRIAIOA4BEhjHmYwKEwEiQASIABEgAiQwHmvgypUrMnr0aJkyZYokSJBAJk6cKM8++yxXCREgAkSACBABImAzBEhg\/meQzZs3S\/fu3eWvv\/6KMREJjM1WK9UhAkSACBABIvA\/BKKewFy\/fl3GjBkjkyZNUpC88sorMnXqVPXfJDB8T4gAESACRIAI2BOBqCcwO3fulOrVq0v27Nll7NixUrBgQXnsscdIYOy5XqkVESACRIAIEAGFQNQTmF27dinvy+DBgyV16tRy8eJFEhi+HESACBABIkAEbI5A1BOYW7duScKECWPMRAJj8xVL9YgAESACRIAI0AMTdw2QwPC9IAJEgAgQASJgfwSi3gPjbSISGPsvWmpIBIgAESACRIAExmsNkMDwpSACRIAIEAEiYH8ESGAsIDDlWw6Qffv22d\/6DtEw0fnDkujcYYdoSzWJABEgAvZCoGTJkjJz5kx7KWWCNiQwFhCYHOVekrFj3zHBPBwCCDycLbU8nDUVwYgyBB544AE5cOBAlM2a07UzAiNX3k10amc9vXV777335OhX7tuTSGBIYGz\/HpLA2N5ElihIAmMJrBw0SARqTtgq6\/adCbJ35LudHlsh8kqYrAEJDAmMyUsquOF+\/naplKxU3WdnEpjgMHV6LxIY6y24fv9Z6x\/ikifA+wIC07NKHsfNiB4Yx5ksOIXNCOLlEVJg2CMXz6g3XpZe7\/k+o\/324wHy0UcfBTYoWzseASSXHDhwoOPnYdcJZOiyxq6q2VqvJe2LSum899haR2\/l3PpjgB4YemAsfRFv3rwus8a\/JX9s\/VmuX7kst\/+7Ibe9n3j7liRMkEBGfPmTT11IYCw1kW0H\/\/PPPyVPHuf92rUtoB6KwZswcuWf6i9l8qV3gsq20XFxuyK20cWoIiQwRpFyeDt6YMw14Lu9Wsnx\/TsMDUoCYwgmVzZCfIG3XL16RZInT+HK+UZ6UlosB45DelbJHWl1+HyLESCBsRjgSA3frl07Wb9+fazHX7hwQf1\/smTJJGnSpDGfFSlSRKZNm6arKo+Q7kLUq0EZSZgoqVRu2EZyF3hU0qSN7Xq9feuWnD55Qma9P1AGTVpKD4zu6nJfA09vgPtmZ+8ZuTGw096IR0Y7EpjI4G75U5s1ayY\/\/PCDoec88cQT8uWXX+q2JYG5C1HvRuWldPVGUr3xq\/HitmbJF1KhRkMSGN3VFXgDu1\/9RCApPAI4yvD0Bhw7dkyyZs0a+ITZwzACTovlMDwxNoyFAAkMF4RhBEhg7kL1drdXJHvegtLwtZ7x4nf07wOS7f4HSGAMrzJjDZ109dP7OIMxMMZszFZEQA8BEhg9hPh5DAJOIDDnzp6VRYsWSfUaNSRDhgyWWe\/A7t9lxjsDpPu7MyR5ipR+nzN5RE9p2WukaQTm+PHj8v3338tff\/0l\/\/33n+TPn1+ef\/55SZHCeEwFbkf9\/vvvUrhwYcvwsXpgjcA44eqndywGCYzVq4PjRwsCJDDRYmkT5hkfgTl96pSsWLVSbv93S86dP6fibB5\/\/HEpWvRxn0++eeOGrPrmazl08G+1+f5z4oQ8+sij8swzlSRxkiQ++2zauFF+3bJFud\/PnDmtNu6MGe+NaYvsprNnz5YcOXJIkyZNgprx3r17ZPPmX+XA\/v1y7sJ5SZ4suSJCGTKkl9y5ckvmzPfJipUrpHPnzrJkxkT5dc0yue\/+fHGflSCBXLl4Xv75c5eMmBM7FklrHOgtpA8++EB27NghPXr0UPgOGjRIvv32WylXrpxMmTJFd74gPkuWLFEECCRm8+bNun3s1sDb8+LEq58kMHZbVdTHqQiQwDjVchHQ2x+B2btnt0ydNk0a1G8ghYvcuYq3f\/8+leOkePEnpW7dFyVBgoQxGv9386bacP89eVK6du2qAoqvXLkiY8aMkXvuSSevtX0tDomZM\/sL2bRpkyIO2bLnkMOHDstnn38mBQsWlP\/+uymJEiaSpMmTyc8\/\/iTduneTtGnTBYTQ5SuX5fPPPpfdu3fJfZnvk6rVqkq+Bx+UZEmTypEjR+SPP\/6QlatWCYhX4kSJpOrTxWTl5x+I3L6l+xwzbiHNmDFDRowYIWvXrpX06e9cD71+\/bogWPvy5cuG6oGcO3dOeadAfDCGGwiME4M1SWB0Xxk2IAKGECCBMQQTGwEBXwTm2tWr8taIEZImTWrp2rVbLKBWLF8hq75eKQ0bNpInn3wy5jOQkZ9\/+UXat2svefPd9V6ACE386COpWKGSVK9xN3vtb9u2ydRpU6V8+fJSs2atmHHmz58vderUifn\/jz+aKAUKFpSyZcsFZDCQp3fffVf+\/feEFClcRBo2bCBJkyWPM8aJE\/\/I5MmfqnZpzuyRK+dPS6oMmSR95uySOEmyWO0TiMi1q5flODwws9f51CcQD0yZMmUEL+v06dMDmpt3YxxBlSpVKiQCU\/vDrfJfnKQ3IalluLN2TdaJnhdtkiQwhs3NhkQgXgRIYLhADCPgi8Cs\/f57WbhooZR9uqy8ULt2rLH+OX5cRo4aqbwh\/fr2VV6VSxcvyuDBgyRp0mTy5ptviiTAVn9HcKzRr18\/9d\/wEmhXvcd9ME7FfPTr30\/uuefudeWZs2bKS41eUu1BclatWiVdunaVhAnvenuMTG7+vHmybv06SZ8+g\/Tp01sSJUrstxvm9PbbYyThwZ\/kwcfL+I1v0QaYOqafvNLtTZ\/jGS0lAA\/Q008\/Lc8++6xMnDjRyJT8tvn333+lRIkSIREYO2Q6daLnhQQmpKXLzkQgDgIkMFwUhhHwRWBAIhCbUrNGTSlfIW5RrR49usvNmzfV0c\/99+eSNWtWqziMRx99VJo3bxHn2ZMnTZIdO3dI\/XoNpORTJUVu35Y+fftImjRppXfv3jHtcRX1j717pWy5cnL92lUZOWqUNHm5ieQOMMPpwb\/+kvfef0+N2\/ilxvJEsWK6eHw6ebIc+H62lKhcW2o0eS3e9udOn5J0GTKGRGDWrVsnTZs2VTE\/iIPRBLiePn1aTpw4Iblz55bUqVPr6h4fgTFaP6bG+C3qOfCCREKcfkWWHphIrBo+040IkMC40aoWzckXgZkwYbzs27dPnqlcWapVrRbnyUMGD5az584qT0mx4sXl008\/le3bf49zTKR1XLhwoaxd+72ULFFC6jdoqAgMiEueBx6QNm3axIyP46myZctIylSpFSG6cvnSnfYByvTp02Tr1q2COgCjRo3yG0DsOSwCltcsmC77d\/wqPd75TH3098GDKu9OgkQJ5cQ\/JwR+pQIFCkjCSyek8otNYxEveIsQjPzewDckwdm\/FSaIR7n\/\/vtl6NChUrToXWLw4osvKoICL0yaNGkUUYGn6tq1a3L48GG5evWqGnvx4sXyyCOPxJr9jRs3ZMKECWp+iRPf8SrhyK19+\/ZxPDDwqiS4fEqS71ooiS7\/KwkvHJebabLKzdxl5XrOp+6Oe\/mMJD2+VZIe+UXGD+ignglSBZKFYGzE5DRo0CCOFc6cOaPaHTp0SJBQ8fz581KjRg1p2bJlrKSKOM57\/\/33Zfv27bJ7927JkiWLOjrs1KlTzBwCNLHtmpPA2M4kVMihCJDAONRwkVDbF4GZM2eO\/PzzT\/LIw49Ii5Yt46g1ZvRoOXrsqFStWk0qV64s777zjvx96G\/lTahU6Zk47VetXCErVq6U\/A8+JG1fa6s+R8zMzp27ZMDAgep46Ndff5XkKZJLoYKFBEc64ydMkJ49ekgqAx4I7wcO6N9fLl66qDb0\/v0HBATru73bSMEnSstVSS6bNm2WRo0aSub77lNj\/LF\/n8ybM0cSnfpD+oybI+nuuUeOHzuuApFXr\/5WwHAaPvuUXDm6Wx0P7dy5Uz7\/\/HO59957FRHCLSNPUtelS5c4HhiQAMTGoEzEypUr5cEHH4zpg89at24tmTNnlrFjx0qSJEkUccDfNmzYEIvAwPPywqBZkvK3mfJQ7a6SPHNuuXXjmvy9dKxc3L9R7i35omQp01j+u3ZZTm9dISc3LZH\/rpyTl19+WXDzq1KlSopIgXiAWHmTKbSBBwnHgs88c8fmw4YNk8mTJ8eaE7xDaNehQwf199u3b6sjs9GjR6tjr88++8wVJIYEJqDXjI2JgF8ESGC4OAwj4IvAaAG2uAWEox7EkcTI7dtq0zp\/4bzUffFFKVW6jAwaOPDO\/9etK6VKlY7zbHhf4IW5N1Mm6dO7j\/ocgcKLFi2Uw4ePqA2+UKFCypsDgQfo8aJP3DluClBwG6p7j+6qV94H8kr7Dh0Mj9DnpQpqk1euGx3J\/1yrWORu6NAhAo\/Eaw2qSd92L8f0fuGFF+S3335TZR1AanwRmBlJ7gYt4\/PUq3pK4ssn5fwzw+RW6iwxfVL8NkOSHvpJzj47WhIkuZsjJuHpA5J27TC5lTS1nK925+hMbt2UtKsHyP1lG8racZ1ixti7d68899xz6v9xXVsrQNitWzdBAHXVqlUF5ew1787w4cNl0qRJ0rZtW3XVGwKPSu3atRUhef3112PGRkwTCG3KlCmVhyhBggSCcUG03nrrLY8ldFsFgJ86dUr93Zd3Rw9\/u31OAmM3i1AfpyJAAuNUy0VAb3\/XqHH7Z\/eePZI5033qBs99WbLI0aNHVD6VDb9skFu3\/5M2rV+VAgULiLZ5161XT0o9VSrOLNb9sE7mL5hnyCMCTwyuFXfu1EkFAyMWZuu23+SvPw9IxnvvlfLlykui\/x2d+IIL7Xv9L64mX9680q69cQLzXt+2cmzvNoEr5db\/EzXlUrkbj6yqUN+69Z\/69\/WcJaVH956SJesdgoGNGDeZPh7RT6o9VShGte7du8u8efMERKBhw7vHYSB08MCkfai0\/F0wdtxQmq97S6JLJ2IRmISX\/pU03\/SRG9mflMvFWsee+tVzcs+KLrEITJIjmyTVxg8lWaY88miuu\/E68IBoV61xvAbSCenbt6\/MmjVLkVN4TDQBqQEJqVWrlrzzzjvqzzg2wn\/Dq5Q9e\/ZYumzZskWSJ0+ursLD+wKigmO0+\/7nxdIa4ygJXiYcf+GqvdOFBMbpFqT+dkGABMYulnCAHv4IDPKwIEfKtq3b1MacNk1aFUyLGktTpnyqfqG\/+eYwFeuA68p\/\/31QxT9UqFAxzqxxk2jFiuXq1\/7rr3f0iwq8MiNHjpQWLVpKjpw51AY45dPJkixZcmnarKmcO3tOeTNq1rp77drXYL179ZJr169JlixZY7wGRkyx4bvlsnbpbLmQ6n51NNO\/f\/\/Y3ifkabl2Vfq3ekFuZCoojRs3lieeuBMgrBGY5wukUblyNPFHDDQCc0+B0vJXgRYqeFYLZK1QoYIcPHhQvvnmG3XNGrJgwQKVXwexLvi3p\/gK4oVNcPyDOJZs2bLpTt+fnohFQqwKPDPjx49X49SrV0\/ZASQEXhZ\/8vPPP8tLL72kCIrn1XhdZRzYgATGgUajyrZEgATGlmaxp1JGSgkgwFS7xrxwwQJZ+8NaKfbEE\/JS4ztHJbjBs33HdhULUa3a83EmunjxIvnuu+9UPpamzZr5BQJjw\/OBzQ7J5UaPGSMXLpyXnj16qngTiOc1a38DTfxwouz9Y48kTJBIhr81PFZAaXxWAGlbt3KRLPp6nfK8dOrYSXLlzh2ny5iBPeTohRuxYn6CJTDXshWXK0+2Fc8rxL4IDIgdiJFRAqN5fuD98Qwg9jf\/QAgMYnSOHj2qSEx8t6Q07w10ee21+G922fPtMK4VCYxxrNiSCMSHAAkM14dhBIwQGG2w8+fPyYgRI+XG9evSs2dPFdMC+XH9Opk7b54UKVJEmjaNS1CmTp0qv\/22TerUflHKPF3Gp244nvr440+kV89eKpj3uzVrZPGSxVKpYiV5vvrdBHggQ56J73wN9uNPP8rc\/1Xi9nesFR9A2pEY5oI5QXZv3SAFitxJ3Dd58iSV\/h9XxnF1HOJJYJr2uFsn6dN3h8maZfOkaYeeUrnW3Zs867\/9SiaO6CcgMB37j4xV2Tg+AuPryMWXB+btt99WHpP4yAOy+KZLdye7cSAEBsdJqLuEaufwyPkT7ao4Yn8QA+RLcIyEm06JEiUyvGbt2JAExo5WoU5ORIAExolWi5DOlZt1k3wemXP9qYFjg7eGvyV79u6WV155RSpWrKRqFEGuX7sm33z7rSRJklgqV342zhA4Qrp9+5Y60kidOk2sz7UxfvrpJ8mZI4fkyJlTff7LL7\/IyZP\/ytNly6rjKy3Qc9asmdLof4nutL7eD0ScynfffS9XrlyWpMmSSbmy5fzWUdLGuHbtqspts3H5Fyob781098v9+QpI4cJ3CEySmxdl09qvpfOQsfLBuAnqiKdSxYoxRR+\/W\/OdXLp8UaoWzi7\/Jsoco9Kh7b\/IqcP75PJjL8n1ByrF\/D3JoZ8k1eZJku6++6VJvZqxpoCbOSAXOKLSSgwgjgRBtLiBhNgTBMbiJhLk0qVLqowDYk9atWql\/oYbUKtXr1bkoFmzZnGO0kB6PvzwwxgPypo1axQpK1u2rDz22GMx+iDoF\/bzPELCERaOtHAkiEBeb0EwL4gRjhk14rJs2TIVF+MpiMdBfxA2PQKDeCFv0eZv5NXx1R\/9jI6h93wENsdXfNPq5+thwOfHXT9m2p\/4x49AIOsPR8+4Ces2SXAb33gUUxEoVqyYbs0d5B7BdeAvvvhCxV\/gSiwEm5sm8ECgoCCCOz03KvxShxcAt3F8bRYY4+uvv1ZXhj2DOZFL5OzZsyoAFoIKzfiVi3a4EeP9fG9QcM0XwaeoKVSyZElFnnDbyVvwfMRyfPLJJyoXy65du1QTHJnhH6T5R+FHPB\/64UYONiuQOBAMTUAckMMFt3C+T1FRSue9U9to16rpcnzHj5K1UivJWPRuTp3T29fI0RUfSOb8T8j094fFUguE459\/\/lFHRrly5VKfZcqUSRV4hMeiV69eKn+Ohv+ePXtUnAo2UBzbIC4F827evLkiQriuDL1TpUqlxsLxD3K7wGYacQA+X331lfp7zZp3CRVIEIJ9PQkMAnXhCcJzYFfMXbu1hC8f\/IMkhxB4dhA0jFw3uGKt3XoChsjQjCKdvkiQt52Avy\/bGX0ZfPXXW0OeY+s9H0kYUZDUn1j9fD0c+Py468dM+xP\/+BEIZP0hbg6pKdwmJDAWWFTPXYdNEpsRyMSQIUNicn54q4LssfXr11eeAWyi2Eyx2eK4AZskjhu04wrPvshtgnT6IAqeixyxHhs3blT5TTQZMGCAIiVp06Y1hAQSxQ0ePFgFw+LZ2PQLFy6scqugaCLmhiBV1BJCjAn0wGfwdMCLgPmWLl1aJY7DfJAPBeQMcR\/r169XSeg0AREEBiAwD9TpHXMkhM0dJAB6gxxogkR3KLuA8eFx8RTErIB4ABPEm2gCb5GWubhRo0aK0ODqNq5DY44QkJU33nhD3f6BlwTPBe+HzhgX\/w1M8WztBhL6Ae\/ly5era9Hor8nHH3+sCk5CD8+aTUjOp1XL1m4dYY0g1w1uM2m2BiHEFxLmg7WBJHkghLhmDf3dcAMJWPEIydAryUZEQBcBvT1JdwCbNiCBscAwvhYLNuqZM2eqwFtseFWqVFGbnWciNl+qYJNC3AWOg+A5wJc64h+wcfsjHSAl2AD79LmTH0YTbHz4lY9rvdg8cRSB7K0oWhiogEBgM4fnAEcnmAc8GtjQq1WrJhUr3rk5haMTBKZqgsyxOGb5+++\/lQcBHocVK1aof8PrAYHXZdy4cTEuTxAYHLdh3vg7vCg4mkL22Y4dO6qr1JgLPD7whGAsJKJr0qRJDGnR3KewDTwruN2lCeJKMCY8WzheAglAf3isqlevrghjzv8dw6EPvCFIHIe5wKMEAgY9tNgV2BokBXYDqYOXCh6uFi1ayNy5c5X35eTJk4p84O94lha4C0zhlcNxFfrBSwNbewf2gkhiHOAPkpM3b151VRvem\/huMQVq50i2J4GJJPp8tpsQIIFxkzUtnouvxYKNDJsOSEigRRQDVffHH39URMJX\/MAff\/yhPDcgHNjssPFZKfBavDjU\/9nrjWtXZHKnamrTffXD1bFUQeZbVFWud+mLWNeordSXY9sHARIY+9iCmjgbARIYZ9svrNq7dbEEA+L9RcrJtcyPyo1HXvTZPfXqQZL4\/CG5lTiFnK8+zmcbEphgkHd+HxIY59uQM7AHAm7dk3iEZMH6cutiCQaqe9t8IWm\/7SdZH3xMitdoIVnyFpJL507J37\/\/LJuWTpUr58+oYR8q\/byUfzl2Mjn8vVbhTPLOwC70wAQDvsP7kMA43IBU3zYIuHVPIoGxYIk5ebGg2rLZkvj4b5J+8wTBzStvwS0lBKEuXbrU72NfffVVEhizjeKA8UhgHGAkqugIBJy8J8UHMAmMBcvPqYul5oStKubECtnd62F1EwcBvbjuixgcBKkiQNUNhQetwCzaxySBifYVwPmbhYBT9yS9+ZPA6CEUxOdOXSwjV\/4lI1f+GauGUBDTZxciYAoCJDCmwMhBiICq\/4Y8Xm4TEhgLLOrUxaJ5YDxrCFkAD4ckAoYQIIExBBMbEQFdBJy6J+lNjARGD6EgPrf7YsH1ZF8CDwyOkDyrOAcxfXYhAqYgQAJjCowchAjQA8M1YBwBOxOYy9f\/kxy91sY7GRIY47ZmS+sQIIGxDluOHF0I2HlPCsUS9MCEgp6fvnZeLFqcC1Qvk+9ObSFvWdzuTrFFuwhvIdnFEuHVgwQmvHjzae5FwM57Uiiok8CEgp7NCczk9Uek+7y7xSE91e1ZJU9MbSELIDB1SBIYU+F0zGAkMI4xFRW1OQIkMDY3kJ3Us8tiie9atJMCdUlg7LS6w6cLCUz4sOaT3I2AXfYks1F2lAcGRRBR7G7hwoWCwoSouowKvaj7g8J9nlWGAwEKdYpQQO+rr75SBQVR8RnFEO+\/\/35VQLBZs2aSNWtWw0PaZbG45Vo0CYzhpeeqhiQwrjInJxNBBOyyJ5kNgWMIDLK4onLvmjV3MsUmTZpUMmTIIKdOnYrJ8IqkaN26dQsII1RSBkHZvXu36pcnTx5FVs6cOSP79+9X1YRTpkypKgujSrERscti0bLqOsnb4gtfEhgjq859bUhg3GdTzigyCNhlTzJ79o4hMMOHD5dJkyapDK5DhgxRlZSTJEkily5dkk8\/\/VTGjh2rKhqPGzdOqlWrZhin1q1by7fffisZM2ZU4zz66KMxfU+cOCEdO3aUDRs2SJo0aeS7776T9Ol9B756PtAui8VJBKZXr14yYsQIn3YjgTG8nF3VkATGVebkZCKIgF32JLMhcASBgZcERznwhgwdOlQaN24cB4e+ffvKrFmzJFeuXLJ69WpFZvQER1BFity5cTNo0CBp2rRpnC7wwlSuXFn9\/e2335batWvrDWuLO\/eeNY2c4IEB6cQRHj0wussrahqQwESNqTlRixEggbEY4PiGnzlzpvTr1095QTZu3KiOj7wFX3aVKlVSf0acTOHChXU1Pnr0aEzczJQpU3weEd28eVMefPBBRYgGDhyojpv0xA6LRSMwuCodrmvRwB91jgKVq1evyunTp+Wvv\/4igQkUPBe3J4FxsXE5tbAiYIc9yYoJO8ID06VLFxW4W7FiRXWM5E9KlSolx48fl969ewuOhvQEQcEIAIYnxl+fvXv3ynPPPaeGQqDvk08+qTdsRD0wuHkE0YoyhtP7Ur9+fdm0aZMuPv4a+KvVwSOkoCF1dEcSGEebj8rbCAESmAgao0aNGrJjxw5p2bKl4KjInzRq1Eh++eUXqVu3rowaNcqQxoh7efPNN5V358MPPxSQIE1Ahjp16qS8PiAuIDBGJFKLBSUCaozfEkvFcBKYQ4cOKS9WgQIFpGDBgoaO8UAiQVy2bdvmt9gYCYyRVee+NiQw7rMpZxQZBCK1J1k9W0d4YDTPCgI927Rp4xcTkI0lS5ZI2bJlZerUqYax++STT2TChAly7tw5yZ49u\/oHXhnEv0BAoBAjkzp1akNjRmqxaAQGx0Y9q+SW0nnvMaSvmY0aNGggCLjOmzdvQMNWrVpVli9fHlAfNnY3AiQw7rYvZxc+BCK1J1k9Q0cQGMSzXLhwQTcGBQRnzpw5Urx4cZk9e7Zh7H7\/\/XeZNm2azJs3L47XAM9u2LCh1KtXTxImTGhoTCwWT0HcjK8AYUODBdDoiQ\/uxJAUy5FcPqqdJYCe5jX9559\/VBA1vGGBCOKcXnrppUC6sK3LETh8+LDkyJHD5bPk9IiA+QhMnz5d7Wme4u+I3vynh29ERxEYfzeFNLh69Oghc+fODYjAILamZ8+egqOMV155RR0\/5cyZUwWV4jgK17MR7IsA1YkTJ0qiRIl0rRMJtut5fBTOYyNdMNiACASJAD0wQQLHbkTAC4FI7EnhMIIjCEzp0qXl2LFjoneE9Prrr8uyZcukQoUKMnnyZF38jhw5oogJrmcjtgYxNt6CXDC4Rg0PEGJljHgJIrFYInHrSBdgNiACISBAAhMCeOxKBDzfbTB2AAAgAElEQVQQiMSeFA4DOILAIPcKgjybN28u\/fv394sL4i8QcIvbMP6Sonl2\/vjjj1W7xIkTqyBhJMbzJdotKKOBvJFYLCQw4Xhd+IxwIkACE060+Sw3IxCJPSkceDqCwKA8wPz581UyO+9zPU+QihUrpo5+kDOmRYsWuvghr8tnn32mztnXrl3rt\/3IkSPlo48+UiUG1q9frztuuBeLVvMIivH4SNc8bOAQBEhgHGIoqml7BMK9J4ULEEcQGCSm69q1q6RKlUql9U+RIkUcfLZv3y41a9ZUf1+xYoXkz59fF8MxY8ao20dIjIfikP6y92rBwaiThLIDehLOxeJJXpxIYFatWqWOBnHsh0zHvoTXqPVWnDs\/J4Fxp105q\/AjEM49KZyzcwSBQfwJrlKj7pG\/WJUOHTqoVPSFChWSpUuXGsIQmycKRELiy8Rbvnx5FciLoyx\/m6znA8O5WDQCE86Mu4bANdioVq1agltgII\/atXXvriQwBsF0WTMSGJcZlNOJGALh3JPCOUlHEBgAgiRzo0ePVt6SAQMGqGvNiFkBuXn\/\/fdV0C5uEs2YMSNWMjr0RQwLvgyRdRd9NUGZgCpVqqjPsmXLJvDIlCxZMubzkydPyuDBg1VgMDbYRYsWySOPPKJrn3Aulu1HL0rZMRulZ5U8KveL04QeGKdZLHz6ksCED2s+yd0IhHNPCieSjiEwICc4RsK1ZwiqUqMyNEgGiAjEX60iFArcvXu3zxgafEkiRwtuJEFAZJDIDknt8NmNGzcUaUJytjp16hiyTTgXi+aBWdK+aEQS1xkCJMRG9MCECKBDu5PAONRwVNt2CIRzTwrn5B1DYDRQ4A1BkjrcGrp48aIiMbgdhCvQ\/go4xkdgMC6OpjAmvAF79uxR44IgIbi3RIkSKj8M4l+MSrgWC+oeaTWPnOqBMYIpCYwRlNzXhgTGfTbljCKDQLj2pHDPznEEJtwABfO8cC0W7eo0dLTb7SPkz4HXCvEtZ86cUaTwv\/\/+k5QpU0ratGmlSJEigttlRoghCUwwq9D5fUhgnG9DzsAeCIRrTwr3bElgLEA8XItFIzB2Ii+oIYXSCVu3bjVUzBFeLsQtIfuxPyGBsWCROmBIEhgHGIkqOgKBcO1J4QaDBMYCxMOxWOxYOgDk5amnnpIrV64o8pIxY0bJnDmzqvSdLl06VYbh7NmzyhsDDw3qJkEQjI36SYg98iUkMBYsUgcMSQLjACNRRUcgEI49KRJAkMBYgHo4FotGYOwU+4KbYVu2bFHB1q1bt1YZjuOTW7duqarhSBSI6+\/I90MCY8GCdOiQJDAONRzVth0C4diTIjFpEhgLUA\/HYrFj6YACBQqoRH\/+PCn+oIYnply5cuqmGAmMBQvSoUOSwDjUcFTbdgiEY0+KxKRJYCxA3erF4nl89Hbd\/NK8lO+jFwumFu+QuA2GTMnBCAp2+ivTwCOkYBB1fh8SGOfbkDOwBwJW70mRmiUJjAXIW71YtOvTdsu+W7BgQfn+++9V3EsggvpVIDAo50AhAhoCJDBcC0TAHASs3pPM0TLwUUhgAsdMt4eVi8Uz94vdCAxiYFA1HLWNkDsnYcKEulgh\/w4SEIL8+IuB0R2EDVyJAAmMK83KSUUAASv3pAhMJ+aRJDAWoG\/lYtEy79opeFeDELeQUJQReV9wC+nee+9VmY2R9+Wee+5RzZDhGO2OHz+u\/oEg2HfNmjUBx85YYDoOaSMESGBsZAyq4mgErNyTIgkMCYwF6Fu5WLTgXbuWDsAVaRRoxOZjRLJmzSrTp0+XvHnzGmnONlGEAAlMFBmbU7UUASv3JEsV1xmcBMYC9K1aLJr3BSrbKXmdLwhRWXrYsGGyfft2Varh6tWr6kgJJRqQjRdFMzt16kTiYsH6c8uQJDBusSTnEWkErNqTIj0vEhgLLGDVYtG8L3Y8PrIAxpgheQvJSnTtOzYJjH1tQ82chYBVe1KkUSCBscACViwWJ3lfAClrIVmwsKJsSBKYKDM4p2sZAlbsSZYpG8DAJDABgGW0qVmLBaRFE+R+QeVpu3tfWAvJ6CphOz0ESGD0EOLnRMAYAmbtScaeFr5WJDAWYG3GYvH0uHiqaOfYF9ZCsmAxRfGQJDBRbHxO3VQEzNiTTFXIpMFIYEwC0nMYMxaLRmCQ66V03jtXkHtWyW2BtuYNyVpI5mHJkUTdZMuTJw+hIAJEIEQEzNiTQlTBku4kMBbAatZiQdCu3Y+MPOFjLSQLFlMUD0kCE8XG59RNRcCsPclUpUwYjATGBBC9hzBjsWgeGLvme\/EFG2shWbCYonhIEpgoNj6nbioCZuxJpipk0mAkMCYB6TmMGYvFiVemraqFxGvUFixSBwxJAuMAI1FFRyBgxp5kx4mSwFhglVAXi2e1aTsH7XpDZ1UtJBIYCxapA4YkgXGAkaiiIxAIdU+y6yRJYCywTKiLRSMwdivWqAeVVbWQSGD0kHfn5yQw7rQrZxV+BELdk8KvsbEnksAYwymgVmYsFhwhOY3AACTWQgpoqbBxPAiQwHB5EAFzEDBjTzJHE3NHIYExF081WqiLRfPAOOkGkjeMrIVkwcKKsiFJYKLM4JyuZQiEuidZpliIA5PAhAigr+6hLpaaE7aqrLtO9MBYACeHjFIESGCi1PCctukIhLonma6QSQOSwJgEpOcwoS4WjcBsH1hasqVLaoGG9hjyhx9+kHfeeUdatmwpzz\/\/vD2Uoha2QYAExjamoCIORyDUPcmu0yeBscAyoS4W7Qq1k3LAeML46aefysqVK+X06dMqk2r79u2lcOHCPpGePXu29O7dW1KkSCE7duywwBoc0qkIkMA41XLU224IhLon2W0+mj4kMBZYJpTFonlfoJaTrlBrMD7zzDNy4MCBWKjeunVLHn74YZkyZYpkzpw5DuIPPvig\/Pfff3H6WWAaDukgBEhgHGQsqmprBELZk+w8MRIYC6wTymLRCIwTvS9du3aVBQsWKEThUSlUqJCkTJlSjhw5ouraJEqUSGbOnCnFihWLhTpKEFy\/ft0vgeE1agsWqQOGJIFxgJGooiMQCGVPsvMESWAssE4oi8XJx0ePPPKIXL58WUBIli5dKgkTJoxBF3\/v1q2bfPPNNzJq1Ch54YUXYj4jgbFgEbpgSBIYFxiRU7AFAqHsSbaYgB8lHEVgbt++rX7hL1y4UHbt2iVInJYuXTopWrSoNGnSRMqUKRM01jjCmDVrlhp\/3759cuPGDcmaNas8\/fTT0qJFC7n\/\/vsNjx3KYtEIjBOPjzBvHBf9+uuvkiFDBp94bdiwQZo3by59+vSRxo0bqzYkMIaXVlQ1JIGJKnNzshYiEMqeZKFaIQ\/tGAIDQtG2bVtZs2aNmnTSpEnVJnnq1ClFNiDt2rVTv\/IDFSRfw6a6efNm1TV16tSSLFkyOXnypCRIkED9P+I3nnjiCUNDB7tYNPKChziRwOTLl08dE+3ZsydenA4dOiS1atWS1157TVq3bk0CY2hVRV8jEpjoszlnbA0Cwe5J1mhj3qiOITDDhw+XSZMmKWIxZMgQdQSRJEkSuXTpkuDWy9ixYxXZGDdunFSrVi0ghLCJfvvtt5ItWzYZMWJEjCcHGy0I0caNG9VnIE94pp4Eu1icTmBQjfrMmTPyxx9\/6EGkbig999xzygvz4YcfMgZGF7Hoa0ACE30254ytQSDYPckabcwb1REE5t9\/\/1VHOQj0HDp0aMzRgycMffv2VUdAuXLlktWrVysyY0TWrVsnTZs2VR6dJUuWCG7EeAq8MMhTkjNnTkGQKq4F60mwi8XJx0fAZNmyZfL6669Lly5dpEOHDnowqbIDzz77rBw\/fly19b69pA3AIF5dKF3ZgATGlWblpCKAQLB7UgRUDeiRjiAwuLnSr18\/SZMmjfKGgGx4C77sKlWqpP6MOBZ\/eUe8+yFHyfLly6VRo0YybNiwgMDz1ziYxeJ074uGxdtvvy0TJkxQHjLgmTx58ngxvXr1quDq9dGjR0lgTFl97hmEBMY9tuRMIotAMHtSZDU29nRdAtOqVSu5du2aYKOBBwSbEm6bhFPwix6BuxUrVlTHSP6kVKlS6tc8EqPhWEhPELj76KOPqrlNmzZNeXnMkGAWi9O9L564YeNBht1z584pXPXk5s2bUrNmTfnqq698NqUHRg9Bd35OAuNOu3JW4UcgmD0p\/FoG\/kRdAoOJ41d0r169pHr16n5vlwT+aOM9atSoobK04igHR0X+BF6UX375RerWrauu6uoJjizw6x+CfokTJ1beGwTzIpYjY8aMUrJkSalTp47Ka2JUAl0sTk9eZxSXYNuRwASLnLP7kcA4237U3j4IBLon2Ufz+DUxRGDeeOMNFdugCTwyDRs2lH\/++SfW6LiBAvLQqVMnU+eveVZAotq0aeN3bDwXcSxly5aVqVOn6uqAWBl4mKD3nDlz1Ni41eQtCOCdPHmyPPTQQ7pjokGgi4XFG+OHlQTG0LJzXSMSGNeZlBOKEAKB7kkRUjPgxxoiMP5iSvbu3auuwuKLBjd\/cIMHV47NFsSzXLhwQQYOHCjNmjXzOzwIDohI8eLFBTV29GTx4sXSuXNnFVOTNm1awTVgkDUQFeScWbt2rbrxBFKDnDArVqxQcTh6EuhicXLyOj0szPicBMYMFJ03BgmM82xGje2JQKB7kj1nEVcrQwQGG3mOHDl8zmn+\/PnqqjGuIRu5oRMMMBqBGTRokLox5E969Oghc+fONUxg0BZ9IEWKFJEvv\/xSeWM8ZevWrVK7dm11q6l79+6KsOkJFoungHT503vzkavSZv6dWzibX8+tNzQ\/JwJRg8Dhw4f9fu9EDQicKBEIAoHp06fHiT\/0d8sziOFt08UQgcFVYxyj+BLcCmrQoIHKjIscLVZI6dKl5dixYyoOJ74jJBxz4SpvhQoV1JGPnmjXftHuvffeE8Ta+BKQD2Bg1LMTKNuFB6ZMvvSyuF0RPZX5ORGIGgTogYkaU3OiFiMQ6J5ksTqmDW+IwKxfv14dofiSnTt3quBeK9kdPCDbtm1T2XL79+\/vd\/IgUiBU9evXV8dZeqLlgEG7RYsWqRtJvgTHSIipAYlDHz0JdLGQwOghys+jEQESmGi0OudsBQKB7klW6GDFmCETGHhenn\/+eV0Cg40\/2FpFOKLCURWuOcd3LRdVjpHhFTljUL9IT5Agr0SJEqrZ559\/LggW9iVaFmAks\/v+++\/1hg0oiHf9\/rNSY\/wW6Vklj\/SswiMkXXDZIGoQIIGJGlNzohYjQALjxwNjlMDgeOeDDz4IykwIIkYW3FSpUgmKAfq60rx9+3aVSwSCYNv8+fMbehaS3+GLEvohgNeX4Po2ygiA4IDo6Emgi4UeGD1E+Xk0IkACE41W55ytQCDQPckKHawY05AHJr7MtkYIDILxcEvpt99+C2oOuIEE8oC6R8gDA0LhLUhdj0RohQoVkqVLlxp+DkgVkq6hMOSqVavi5LnZv3+\/qtmDpHd6MTjaQwNZLCNX\/iUjV\/4pQ2rmkw7lcxrWO5oa8hZSNFn77lxJYKLT7py1+QgEsieZ\/3TrRjREYHDNGMcn3jd0oNaVK1cERQ99eTxu3bqlPkd2XBCAUOJkUPBv9OjR6srzgAEDpF69eqqwIsjN+++\/r4J2cfV5xowZcY6CkMkXX4ZFixZVfT0FpAj1eBAkXLBgQUVmtLkgvgd9cV0cSe1w0wrXrfUkkMXCIyQ9NEXCSWBeeukl+fnnn\/WVYgsiEGUIIKknyrpQnIdAIHuSk2ZniMCYNaFQCAzICY6RUFIAghtP6dOnFxRbRCp6iL88MfD+7N69228MzZ49e6RJkyZqLEiWLFnUmNr\/p0uXTlW8BgEyIoEsFi2JHWNg\/CMbTgITiO2MrAW2IQJuQYDvhnMt6VbbOYbAaEsHV5+RpA6lBVDNGCTmySefVMdK\/go46hEYjI26PR999JF88803giMveI+Q+wb1l5CtN3PmzIZXbyCLRSMwS9oXldJ57zH8jGhqSAITTdbmXO2KQCDfa3adQ7Tq5Vbb6RIYJKfr06ePIBcLgmeR0M2owGuCIxrUGXrzzTfVMU40iNHF4lmBmgSGHphoeDc4R+ciYPR7zbkzdK\/mbrWdLoFBzaMvvvgiZMsiGRyyA0aDGFksWuwL8GASu\/hXBT0w0fDWcI52R8DI95rd5xCt+rnVdroEBkFbCGwMVRBc27hx41CHcUR\/I4uFBRyNm5IExjhWbEkErELAyPeaVc\/muKEh4Fbb6RIYPdhQmRo3gXANOWHChHrNo+JzvcVC70tgy4AEJjC82JoIWIGA3veaFc\/kmOYg4FbbBUxgENeyfPlyVXMIJQbOnz+vEMYVawS6Il9LrVq1VMxMIPEy5pjJHqPoLZYf9p2RWhO28ujIoLlIYAwCxWZEwEIE9L7XLHw0hw4RAbfaLiACgyBcZKs1kpAOaf9HjhypriRHm8S3WLTEdcCEsS\/GVgYJjDGc2IoIBIsAbnT+8MMPUrVqVb9DuHUTDBYzJ\/Vzq+0MExjvXClGjIfkb6hhhCR40SRGCAzJiz1XhFtedCR+RN0ulN7wJUjIiPxGuGWI\/Ebly5f3m4bAV3+kGYAnFmkHfv31Vzl16pTyuN57771qvMqVK6sM1r68sGPGjFFZr\/ft26eGhh5LlixRaQt8CZ6D75HVq1erZJUQFF7FM5CB259gU0YdMyQm\/Ouvv2KaIRnmM888o5JhlitXTv0dFwxQgsQziWGuXLlUrTSUGcmePbstF+zRo0dlwoQJKsFmjx49dHXU6rr5avjWW28JCuL6E7e8G7ogubCBW21niMAgzsWzYCO8Kqg7VLZsWVWhGblY8GWBxG\/4Mvv6669jvgiQ1Xbu3LmSOnVqFy4L31PSWyy4Ps3EdfZcDnq2s6fW\/rVq166d2pghjRo1UuUwEidOLGfPnlUZpkFy8H7i\/QUp6Nmzp9+iptpTUBm+e\/fuioDA0woS8cgjj6gxkDASGypSJxQoUEBGjRqlPvOW69evS\/v27VV2awiyYEMPX3XOtL6LFy+Wzp07S6AZYW\/cuKEIGrJtQyZOnKiyb3sLCsGiICzwQckS3Jy06zE45oK8VbghCixfeeWVOFnGvefnWZLF+zN8h6Pgbnz4u+3dcNq7HIq+brWdIQKD9PpaIUa48zt27BjvQgfQcEf27t1b8AsB7fHFEy1ixAPDvC\/2XA1ue9G1Wl9A++OPP1aeB29BVXZUfMc7Cw9H27ZtFZHxJfC4gHiAFODfOFL2Fbyvlf5AxmwQBs3T4TkmvDB4liYvvPCCjB071u\/CwDMfeughgQenTp06AS2g1q1bx5AlzNOXRwWZvOfMmeNX34AeaGFjEEj8UMQPSZQ6wQ9MIwQGxBIeuUGDBsXRDt7yNGnSxKu1294NC01ku6HdajtdAgN2j4BcuIiRjC6QK9WImalbt676ssMvsvjYve0sHoJCRggMPTAhAGxhV7e96J988ongaAAyadIklVnal6BW2WuvvaaOhCAgJjg68ZQ\/\/vhDateuLZcvX1b\/fvvtt+O1hFZgFRvjokWLJHfu3LHag8DAk4MjHXy\/QPr37y\/Nmzf3Oy68RKh9VqFChYBWAeaCiwcQfBdlypQpVn\/E66HSPGqqIbO3UwSkDzGJegQGJAfeMhDLJ554Iqjpue3dCAoEh3Zyq+10CYz2K6lZs2aq1lCggvNr\/FIbP358vAFigY5r5\/ZGCMzpsYF9Adt5vm7SzW0vOrwuI0aM0CUwaIDCq\/DQ4HgCtwpR3f3BBx+MMS82ybVr1yovDY4b9OJCQHiqVKmi+uPIBp4YT8F3y5AhQ1QB1Zdffln90MFzkTPKH4nA5vvee+9JmTJlAlp2\/ggM5oIfZl9++aVMmzbNcL2zgB5uYeP69evLpk2bdAkMMAWRhTccMUqPPfaYqicXiLjt3Qhk7k5v61bb6RIYBH3BrfrTTz8F7UFBLSKcW3tXgnb6ovCnvxECQw+MPa3vthc9EAIDi2Aj146PcEyD4xoIgvi1GyogEWhnRDQPAYgCgnARNKyJRmBAhhDLgZIlEAQCw2OTNWvWOI8wk8DA64S5Ig7ns88+8xmrY2SOkWxjhMAg4LpSpUpy8ODBGFURk4gga3jdPG0S31zc9m5E0m7hfrZbbadLYBCV\/vDDD4dEPnD2inNbBJ1Fg8S3WFh9OvAVwGvUgWOm9QiUwKCo6eOPP668LAjs3Lhxo4px0WJaMC6CgYcNG2ZIKc9bL97HQ54EBoPBwwsiAUFhVhRtxfGSp5hFYHBrB96ILVu2qGcitsaJYoTAHDlyRMUj4phu\/\/79KuhXE3hhYEsjMUVu3QSdaPdAdXar7XQJDFy1iMiPLz+AHpgImsMZ89KlS\/WauuJzIwSGR0jGTU0CYxwr75aBEhj0xzHSgQMH1FBIVglPSJs2bWLiY3CTCf9vREBCsHlCcJNRuwyA\/\/cmMDdv3lRHSdrVb2zO2vGX9iwzCAw8LiBTP\/74oyJJxYsXNzKVeNssWLAgpFpvLVq0kBo1agSshxEC4zkoYmFw8wzHdvCqQXDTCvFRenFFbt0EAwbdgR3cajtdAoNfQnAX4zp0sKIF827evDnYIRzVzwiB4S0k4yaNJIHxrBhuXOPwtdQjwsEQGHhd4XmB4OoyrkEjaBdeVEgg8WyeN42QU2XWrFkx4HgTGHyAq8xI0YDbixB4cFBQ1kwC4\/l+4vsNwbupUqUKyWieHqpgBurXr5+AxAQqgRIYbXwcK+HYDkQO3jbk4MERH66Q+xO3boKBYu7E9m61nS6BwfkofoUh30uwgmua+PLSftUFO45T+hkhMHobj1PmGg49SWD8o6y3joIhMLhpqCV0Qw4Z\/HiBB1b7xY5gV9xoMSKeBAZkAZ4KTXwRGHy2a9cuefHFF+Xq1auSJEkStdEi8BRihgcGMXn4TsNxGQRzgQcCz3KaBEtgtHl6ro8pU6b4vO6utXXrJug0mwejr1ttp0tgMHEwc+8rkIGAiAR3uFVAAiOilRKgB8b4CookgTGupT1bBkNgPMkKYkSQKRdeEO1oB8npkB7BiGBTHDp0qGqKXDD4fz0Cg89x5RnXsHG8gRpryNSLq89mEBhco\/7nn39USggk34Mg2BjXwu2auM4f1qESGIwLQocEhLBT48aN\/ZrVrZugkXXs9DZutZ0hAgOXMoK84nMv+jMwAsaQWwL5FUhgRLQjCb1fzk5\/YczUnwQmeDQDJTC4yowg3kuXLqlEaYgTgSAOTjv+8ZUjxp+GnkG8+B7RctKgvT8PjDYWCAWOqyCIU8FVYHhyzbpGDXKGbLuYK6RVq1YxN6GCRzy8Pc0gMJpt9WKb3LoJhtdikXmaW21niMCYBTkJDAlMMGuJBCYY1O70CZTAgLAgkNZ7Q0dOGK3uELwxICZGREtmh7YgHp6BqnoEBrEZCBbWyg0gFxVicswiMNAJXiWQGO1mDgKOkbXXKWIGgfn0009VLhwQRsQ6+RO3boJOsXUoerrVdiQwoawKP33jWyz0wAQOOAlM4JhpPQIlMFrKfVxfRhJKLUcIbgghGR2KIuJvGqmITzMEij711FOCGDgEySLfC46jNNEjMGiHIx54f7XCjyA1uPZsViI7PGPNmjWCNYY5YnyUM4hvI\/c1Z6cF8XrOQbu+DpvGlxPGrZtg8G+Xc3q61XaGCAy+fPLly6e+hAI5I8aXAdyzyMiJ9OPR7oHR4l+w7HmEZPzlJ4ExjpV3S6OlBNAPFZ9REwni61YM4lA6deqkPsfNRL2U9EifAK8JBDV7vCtHYzxk4tVuPPmbJUhTrVq1BMUIIagcHSiB8fQE+SolgODirl27qvGRDRg5q\/yVXfClJwpRajlsgrFWy5Yt1e2rQCVUDwyODEFM77vvPhUsHZ+4dRMMFHMntner7XQJDM6eEcPi+cspUAOeOXNGvSR6X1SBjmvX9v4Wy\/r9Z6XG+C2sRB2g4UhgAgTMo7mRYo7wPOAYATlX8KMDZMO7DhKGxGc4PsJ7XKRIEZVDxd\/NHfxwgRcDnhNUeMbm7p26HkQER1E7duxQpCE+ARlCKQPoEAyBQX0l5D+B4N85c+aM8zjMHx4rCDxQU6dOVRnE7Sza9fb4aiGNGzdOecEQg1SoUKFY08H6wA2shQsX6mbkdesmaGf7mqWbW22nS2DwReaZfCpYQJH1EkXYokH8LRYtCy+9L4GtAhKYwPDybN2uXTvBVWgIglQRqAkSgKMZ5GdCDAiICLyjuE6MI6T4rkijH8ZBP6SnR4LKDBkyxFIQ4yJF\/d69e1VGXZAjZPX1FJCmJk2aqMKK3rEx\/marxWoESmA8C9JibGzouHnjLfBGgGxpnp7kyZOr7yxfFbyDt4h5PWELFNqFvpgP5uUt+PGIoGx4zvEPcs0grgh1r5D\/BjdMkSndSJ4vt26C5lnEviO51Xa6BAZfQkYWt57pzBpH7zl2+NzfYuEVajtYJ34d3PKiw5OA2A4QBE8BecFGljFjRpXbCQUZkSIB16IxdyOCbK4YH\/lgsOnj2jWKA0JQGRlXoFF+AGQIRMbbuwKvCzZO7UgZ+uC4Bj9yUG06PsERF648GzlCOn\/+vApMRewNSJUm8K5UrlxZeSS0cXCcBbxwlOR9TI7r39j4jea+MYJhKG3+\/vtvmTdvnspsrs0LdkXuHJR9qVevnqDWkSYIwIaXBWUEYDvYHB4otIPtYCsj4pZ3w8hc3dbGrbbTJTChGBJBfEZfjlCeY7e+\/haLFsDLHDB2s9hdfdz6oluBOMgLYkUQ9KpJypQpFbnBdWe9YyErdOKY1iHAd8M6bK0e2a220yUwcEsiABdZMfFvMHicb+PXElKMxydwEaPCbMGCBa22j63G97VYGP9iKxP5VcatL7qV6CP41TPDbvXq1dWxUCAB\/1bqx7HNQYDvhjk4RmIUt9pOl8C8++67MbErcFMiUyNcjzjb1hMkioILGV9uKAgXLSYvWAgAACAASURBVOJrsVz\/77Zk6f6dlMmXXha3KxItUDhunm590a00BDytqKnjWecIwaWDBw+OdZRhpQ4c23oE+G5Yj7FVT3Cr7XQJzIkTJ1QkPn5N4UpmIFcLYQz8EsNNBDMCga0yrtnj+losDOA1G2VrxnPri24NWrFHxU0WZM5FrAUkV65cgqy9VapUiXMDKRz68BnmIsB3w1w8wzmaW22nS2BwhRpR64GkD\/c0DI6cEPyGiHczgoHDafRgn+VrsTCAN1g0RSUZQ6xFOMStL3o4sMMz4KVFoUQED2\/atEkOHTqkHo1jZCRJa9u2bZxbS+HSjc8JDQG+G6HhF8nebrWdLoEZNGiQ4CgISa6CDcpDHAyOkHzlloikUa16Nj0w5iJLAmMunpEYDcdMCPrFDSDGxkTCAqE\/062bYOjI2H8Et9pOl8AgQRKOjVAvJFhBnglc+5szZ06wQ6h++HWHeBq4qnft2iW4JokEe0WLFlU5JYxcrTSqALJ\/IrcCgpfhQcKVUaNCD4xRpIy1I4ExhhNbEQErEXDrJmglZnYZ26220yUwSOKECrLIyBus4BgKnhzkYwhW8OsN7mfkaoDglxwSaJ06dUr9soMgaZeWCj3Y56Affi0iR8TmzZvVMGYQGC0GhleoA7cMCUzgmLEHETAbAbdugmbjZMfx3Go7XQKDs2vUyDBy68if4VAUrnPnzrJnz56gbYvkV0jGhHTkqJ+CZFZIY44r3cjQiVwUcE37y7IZyIO1wmx4lhbDE6oHhkG8gVggdttwEpiXXnpJfv755+CVZU8i4FIEcJlj5syZLp2du6cVtQQGGTbHjBmjahkFK0hsBWKBLJ3BCOp4wAuClOBDhw5VV7m9pW\/fvuoaJ24+IMtnsOfsOJoCOQJ5wVVQ1HAxwwPDJHbBWP5On3ASmOC1ZE+zEUCW2fiqI5v9PI5HBNyKQNQSGBwhlSpVSnk9ghX8qv3nn38E5dqDEbB+VMdNkyaNKiSH4yNvwZcdarNAECcTjMcIBAnkZffu3TJq1Cg5fPiwyoFjJoFhHaTAVwAJTOCYuaEHCYwbrMg52AGBqCUwvXv3FtTSQBxLpkyZArYFYlZQKr5OnTrKkxOMoDouAncRTIxjJH8ConX8+HGBzqjDEqigMB2u64IIIeeNlsTPDALDI6RArXG3PQlM8Ng5uScJjJOtR93thEDUEhgcx6D6bIUKFdTmnjhxYsN2QXwKklgdPXpUkYGaNWsa7uvZsEaNGrJjxw5FhHBU5E8aNWqk8k+gMB08KIEIAnbr168v99xzj6reC7JmJoHRjpDogQnEKnfaksAEjpkbepDAuMGKnIMdEIhaAoOryyAQO3fuVAXakDK8UKFCujY5cuSIdOjQQbZt2yZZsmSR7777zufRj+5AIuoIC56VXr16qaR6\/qRTp06CqrJly5aVqVOnGhlatQHRQv2WgwcPqozBzz\/\/vPo7CYxhCC1tSAJjKby2HZwExramoWIOQyBqCQzsBO8E4lhwXRmEBnEiKMOeO3duFTSLgFdcPT558qRs3bpVVq1apYiEdr0ZR0c4QgpWEM9y4cIFGThwoDRr1szvMCA4yDWDK9\/IPWNUtABgEDWUPtAkFALj+Wzo\/N65supPm1\/PbVQttiMCUY0AYtBy5MgR1Rhw8kQgGASmT58eJ3fZgQMHghnK1n10r1Fr2iMTL6rOet\/uAaFBcC0Ihq+bPzjWGTZsWEggaAQGuWTiS6jXo0cPmTt3bkAE5ocfflCkCEdGIF5IjGcGgfFcLFolaozLI6SQlgI7RxEC9MBEkbE5VUsRiGoPjIYs4mFQEwlkRU9AbHCEhADcYK80a88oXbq0HDt2TPcICaUKli1bpuJ1Jk+erKeinDt3TsXo4IYUjpzKlSsXq08oHhhvtosYGFai1jUJGxCBGARIYLgYiIA5CJDA\/A\/H06dPy8SJE1VpgDNnzsRBF1ecEYOCxHVGYmWMmAf5WBBL07x5cxWD40+QPRfXrBGMO2LECN2hO3bsKEuXLlVBoj179ozT3iwCoxVyJIHRNQkbEAESGK4BImAyAiQwXoAi5uX3339Xga8gMqlSpZL77rtPHn\/8cfXfZgrKA+AIS+86c7FixQQECzljWrRoEa8KCDLGeJB8+fL5vF2FmB78g\/nkzJlTtUUcDoKZ4xPvxaIRmJ5V8kjPKoyBMXNtcCz3IkAPjHtty5mFFwESmPDiHetpSEyH+BsQiQ0bNkiKFCniaLN9+\/aYa9q4Bp0\/f\/54NfZMfBfI1JBVGMn9AiEwx89fl0KD1gsJTCBIs220I0ACE+0rgPM3CwESGLOQDGIcxNzgKjWuO+PGEPLBeAvibZBwD8dWOBYyQ8w+QiKBCc4qvEYdHG5O70UC43QLUn+7IEACE2FLaAUWEWMzYMAAqVevnirmCHKDdP8I2kXg8IwZMxTZ8RQEEuPLsGjRoqqvUTGLwGhJ7EhgjCIfux0JTHC4Ob0XCYzTLUj97YIACUyELQFygmMklBSAIPdM+vTpVYzKzZs31d\/85YmpVq2aqm+kF0PjPUWzCQyvUAe3iEhggsPN6b1IYJxuQepvFwRIYGxiCVyTRpI6lBa4ePGiIjFPPvmkOlbyV8CRBMYmxgtSDRKYIIFzeDcSGIcbkOrbBgESGNuYwv6KeC4W7QYStKYHJjjbkcAEh5vTe5HAON2C1N8uCJDA2MUSDtDDF4Fh\/EvwhiOBCR47J\/ckgXGy9ai7nRAggbGTNWyui\/diQRAvCUzwRiOBCR47J\/ckgXGy9ai7nRAggbGTNWyuCz0w5hqIBMZcPJ0yGgmMUyxFPe2OAAmM3S1kI\/08F4tWyJEemOANRAITPHZO7kkC42TrUXc7IUACYydr2FwXz8VSc8JWWbfvDAN4Q7AZCUwI4Dm4KwmMg41H1W2FAAmMrcxhb2V8HSEtaV9USue9x96K21Q7EhibGsZitUhgLAaYw0cNAiQwUWPq0CfquVi0LLy8Qh06rhwhuhAggYkue3O21iFAAmMdtq4bmQTGdSblhCKAAAlMBEDnI12JAAmMK81qzaRIYKzBlaNGFwIkMNFlb87WOgRIYKzD1nUjk8C4zqScUAQQIIGJAOh8pCsRIIFxpVmtmRQJjDW4ctToQoAEJrrszdlahwAJjHXYum5kbbFoOWAwQQbxBm9m3kIKHjsn9ySBcbL1qLudECCBsZM1bK6LN4Epky+9LG5XxOZa21c9Ehj72sZKzUhgrESXY0cTAiQw0WTtEOfqfYREAhMaoCQwoeHn1N4kME61HPW2GwIkMHaziI318fbAsIxAaMYigQkNP6f2JoFxquWot90QIIGxm0VsrA89MOYahwTGXDydMhoJjFMsRT3tjgAJjN0tZCP9tMUycuVfMnLln8IjpNCMQwITGn5O7U0C41TLUW+7IUACYzeL2FgfbwLDI6TQjEUCExp+Tu1NAuNUy1FvuyFAAmM3i9hYHxIYc41DAmMunk4ZjQTGKZainnZHgATG7haykX4kMOYagwTGXDydMhoJjFMsRT3tjgAJjN0tZCP9tMWy89glKTN6g\/AIKTTjkMCEhp9Te5PAONVy1NtuCJDA2M0iNtbH2wOzpH1RKZ33HhtrTNWIgP0QIIGxn02okTMRIIFxpt0iorW2WBZt+1eaT9tOD0xErMCHOh0BEhinW5D62wUBEhi7WMIBemiLpeaErbJu3xnWQXKAzaii\/RAggbGfTaiRMxEggXGm3SKiNY+QIgI7H+oyBEhgXGZQTidiCJDARAx65z3Y2wPDGBjn2ZAaRx4BEpjI24AauAMBEhh32DEss+ARUlhg5kNcjgAJjMsNzOmFDQESmLBB7fwH0QNjrg15jdpcPJ0yGgmMUyxFPe2OAAmMDSx0+\/ZtWbBggSxcuFB27dol58+fl3Tp0knRokWlSZMmUqZMmaC1XL16tXzxxReydetWOXv2rCRLlkxy5swpZcuWlRYtWkjmzJkNj00PjGGoDDUkgTEEk+sakcC4zqScUIQQIIGJEPDaY2\/cuCFt27aVNWvWqD8lTZpUMmTIIKdOnRJ8BmnXrp1069YtIE3RF32WLFkS0y9NmjRy8eJFAWGC4P+nTp2qiJIR0RbLwCX75YM1fwtjYIyg5r8NCUxo+Dm1NwmMUy1Hve2GAAlMhC0yfPhwmTRpkvKMDBkyRF544QVJkiSJXLp0ST799FMZO3asJEiQQMaNGyfVqlUzrO3IkSPlo48+Uu1btWolbdq0kXvvvVeRolWrVsnAgQPl9OnTki1bNoGXBsRJT7TFkqHLHbJ1emwFvS78PB4ESGCic3mQwESn3Tlr8xEggTEfU8Mj\/vvvv\/L000\/L9evXZejQodK4ceM4ffv27SuzZs2SXLlyKaIBMqMnOIJ68skn1biNGjWSYcOGxemyfPlyad++vfr7tGnTlB56QgKjh1Bgn5PABIaXW1qTwLjFkpxHpBEggYmgBWbOnCn9+vVTRzkbN2706QXBl12lSpWUloiTKVy4sK7Ge\/fuVR6WM2fOyJgxY+SRRx6J0+fKlSvy8MMPq78PGjRImjZtqjsuCYwuRAE1IIEJCC7XNCaBcY0pOZEII0ACE0EDdOnSRQXuVqxYUR0j+ZNSpUrJ8ePHpXfv3tK6dWtTNL527ZoUKFBAeXRGjRoldevW1R3Xk8BkSpNU9gwurduHDfwjQAITnauDBCY67c5Zm48ACYz5mBoesUaNGrJjxw5p2bKl4KjIn+AY6JdfflEkA2TDDPnmm29UXAzkhx9+kOzZs+sOSw+MLkQBNSCBCQgu1zQmgXGNKTmRCCNAAhNBA2ielV69esWQCV\/qdOrUSd0mwtVn3BoKVRAjU6tWLTl48KA0aNBA3nrrLUNDYrF89vWvUmP8FtWeQbyGYPPbiAQmNPyc2psExqmWo952Q4AEJoIWQTzLhQsXVLxKs2bN\/GoCgjNnzhwpXry4zJ49OySNEReDW0lbtmyR\/Pnzy9y5cyV16tSGxvT0wJTJl14WtytiqB8b+UaABCY6VwYJTHTanbM2HwESGPMxNTyiRmD0gmh79OihiEaoBObAgQPquAqeF5CX6dOnB5zI7ua9BeRime6S+OQeaZ\/3H0PBv4YBibKGI0aMEJBTSnQhcPjwYcmRI0d0TZqzJQImIIA9C7dmPQX7mtskwW0tW5uNZ1a6dGk5duyY2sS0eBRf6r7++uuybNkyqVChgkyePDmoGSFRXufOnZXHB5l9x48fr24\/BSKeR0g9q+SRnlVyB9KdbYkAERARemC4DIiAOQjQA2MOjkGNUrt2bdm2bZs0b95c+vfv73cMxKngmnX9+vUFv9oDlY8\/\/jim3yuvvKIChhMlShToMMIjpIAhYwciEAcBEhguCiJgDgIkMObgGNQoSPU\/f\/58lUTO2y3mOWCxYsVU1lzkjEH9okDk7bffVt4WZPdFsG6dOnUC6R6rLRbLqx+ulpEr\/xR6YIKGkR2jHAESmChfAJy+aQiQwJgGZeADITFd165dJVWqVLJhwwZJkSJFnEG2b98uNWvWVH9fsWKFil0xKu+++668\/\/77kjJlSpVnpmTJkka7+mzHI6SQ4GNnIqAQIIHhQiAC5iBAAmMOjkGNgngUXKVG3SMc6yDA1ls6dOggX331lRQqVEiWLl1q+Dmod4QikYkTJ1bBuqGSFzzY0wMz79XCUuGhDIb1YUMiQATuIEACw5VABMxBgATGHByDHuXDDz+U0aNHqzICAwYMkHr16qnjHpAbeE8QtIt45BkzZiiy4ynI5IsvQ1STRl9NUCagfPnyglpLHTt2VMG7ZgiPkMxA8e4YvEZtLp5OGY0EximWop52R4AEJsIWAjnBMRJKCkBQlTp9+vRy8uRJuXnzpvqbvzwxqE69e\/fuODE02tEU+uL4SC9gt0SJEoJAXz3xDOJlDIweWvqfk8DoY+TGFiQwbrQq5xQJBEhgIoG6j2fimjSS1KG0wMWLFxWJQUVpHCv5K+Doj8DAWxPfrSbvx8Oz8\/nnn+siQQ+MLkQBNSCBCQgu1zQmgXGNKTmRCCNAAhNhAzjp8fTAmGstEhhz8XTKaCQwTrEU9bQ7AiQwdreQjfTz9MAsaV9USue9x0baOU8VEhjn2cwMjUlgzECRYxCBOxdLmImXK8EQAjxCMgST4UYkMIahclVDEhhXmZOTiSACJDARBN9pj8ZiOfvCnVIGrEQduvVIYELH0IkjkMA40WrU2Y4IkMDY0So21YkExlzDkMCYi6dTRiOBcYqlqKfdESCBsbuFbKQfCYy5xiCBMRdPp4xGAuMUS1FPuyNAAmN3C9lIPxIYc41BAmMunk4ZjQTGKZainnZHgATG7haykX4kMDYyBlVxLAIkMI41HRW3GQIkMDYziJ3Vuf\/JanKxTHelIoN47Wwp6mZnBEhg7Gwd6uYkBEhgnGStCOuqeWDK5Esvi9sVibA2fDwRcCYCJDDOtBu1th8CJDD2s4ltNcpW7Q25WqCmkMDY1kRUzAEIkMA4wEhU0REIkMA4wkz2UFIjMCzkaA97UAtnIkAC40y7UWv7IUACYz+b2FYj7QiJBMa2JqJiDkCABMYBRqKKjkCABMYRZrKHkvTAmGsHXqM2F0+njEYC4xRLUU+7I0ACY3cL2Ug\/7RYSPTDmGIUExhwcnTYKCYzTLEZ97YoACYxdLWNDvTQPTO\/n8kj3Z3PbUENnqUQC4yx7maUtCYxZSHKcaEeABCbaV0AA8+cRUgBgGWhKAmMAJBc2IYFxoVE5pYggQAITEdid+dDMTSfKzXsfEh4hmWM\/EhhzcHTaKCQwTrMY9bUrAiQwdrWMDfXSCAyz8JpjHBIYc3B02igkME6zGPW1KwIkMHa1jA310gjMkvZFpXTee2yoobNUIoFxlr3M0pYExiwkOU60I0ACE+0rIID50wMTAFgGmpLAGADJhU1IYFxoVE4pIgiQwEQEdmc+VAvipQfGHPuRwJiDo9NGIYFxmsWor10RIIGxq2VsqBc9MOYahQTGXDydMhoJjFMsRT3tjgAJjN0tZCP9GANjrjFIYMzF0ymjkcA4xVLU0+4IkMDY3UI20i9jp5VyO1FS4RGSjYxCVRyHAAmM40xGhW2KAAmMTQ1jR7UydFmj1OI1ajtahzo5BQESGKdYinraHQESGLtbyEb6kcDYyBhUxbEIkMA41nRU3GYIkMDYzCB2VocExs7WoW5OQYAEximWop52R4AExu4WspF+JDA2MgZVcSwCJDCONR0VtxkCJDA2M4id1SGBMdc6vIVkLp5OGY0EximWop52R4AExu4WClG\/jRs3ymeffSabNm2S06dPS4oUKSR\/\/vxSu3ZtqV+\/viRMmNDwE0BgWMjRMFy6DUlgdCFyZQMSGFealZOKAAIkMBEAPVyP\/OCDD2Ts2LGSIEECSZQokWTMmFEuXrwoly9fViqULFlSpkyZIsmSJTOkEghMmXzpZXG7Iobas1H8CJDAROcKIYGJTrtz1uYjQAJjPqa2GHH16tXSqlUrpQv+3b59e0mXLp3cunVLVq5cKT179lRk5uWXX5YhQ4YY0hkEplvl3NKnah5D7dmIBIZrIC4CJDBcFUTAHARIYMzB0XajVKtWTXbv3i3PP\/+8wBPjLUuWLJFOnTopz8yaNWskR44cunOgB0YXooAa0AMTEFyuaUwC4xpTciIRRoAEJsIGsOLx+IKsVKmSGnrBggVSuHDhOI+BJ6ZMmTJy\/Phx6dWrl7Rp00ZXFRIYXYgCakACExBcrmk8ePBgGThwoGvmw4kQgUghQAITKeQtfO7s2bOld+\/ekjp1atm6davfQN033nhDFi1aJBUqVJDJkyfrasQgXl2IAmpAAhMQXK5p7NYvXdcYiBNxDAJufZcS3L59+7ZjrGCyosOHD5dJkyZJoUKFZOnSpX5Hf++99wT\/4Pho7dq1ulqQwOhCFFADEpiA4HJNY7d+6brGQJyIYxBw67sU1QSmS5cusnDhQilXrpy6ZeRPvvjiC+nTp48kTZpUxcvoCQgM6yDpoWT8cxIY41i5qaVbv3TdZCPOxRkIuPVdimoCg43x66+\/lueee04mTJjgdyXOnz9funXrpj7ft2+fbk4YEJjU60ZL4pP6ZMcZyz+yWlauXFnZiUIEiAARIAKBI4BUIDNnzgy8o817kMB8\/bVUrVpVxo8f79dUCPDt2rWrYQJjc5tTPSJABIgAESACjkcgqgmM0SOkGTNmSP\/+\/SV58uSyc+dOxxudEyACRIAIEAEi4HQEoprAjBgxQj7++GMpWLCgLFu2zK8t33nnHZUjJnfu3ILEdxQiQASIABEgAkQgsghENYGZO3eu9OjRQ1KmTCnbtm1Tyep8CbLzLl++XJ555hlFeChEgAgQASJABIhAZBGIagJz5MgRefrpp5UFkBOmePHicaxx\/fp1KVGihJw7d04GDRokTZs2jazF+HQiQASIABEgAkRAoprAwP4NGjQQVKJGRt5PPvkkzpKYPn26Ii64Qv3DDz9IpkyZuGyIABEgAkSACBCBCCMQ9QQG5AUkBtKkSRNBYC+KOd68eVPliOnXr5\/AC9OxY0fp3LlzhM3FxxMBIkAEiAARIAJAIOoJDEDALaMBAwYIkhIjDgZeljNnzsi1a9fUKkGhR2TiTZgwIVcNESACRIAIEAEiYAMESGD+Z4Tff\/9dlRX45ZdfFHlBfaSHH35Y6tWrJzVq1LCBqagCESACRIAIEAEioCFAAsO1QASIABEgAkSACDgOARIYx5mMChMBIkAEiAARIAIkMFwDRIAIEAEiQASIgOMQIIFxnMmoMBEgAkSACBABIkACwzVABIgAESACRIAIOA4BEhgTTIaMvigxsG7dOjl69KgkTpxYcubMqZLjtWnTRtKkSWPCUzgEEbA3AshWXbRoUUNKDh8+XBo2bBinbSjvUih9DSnNRkTAIgSuXLkio0ePlilTpkiCBAlk4sSJ8uyzz+o+DXnMPvvsM9m0aZOcPn1aUqRIIfnz55fatWtL\/fr1\/ab+QMqQBQsWqFxnu3btkvPnz6v8Z3h\/kQ+tTJkyfp996dIlpefKlSvl4MGDKt3Ifffdp\/q0bt1a8uTJo6u3WQ1IYEJEEtl5X331Vbl69aoa6d5775UbN26o0gOQLFmyyBdffCH3339\/iE9idyJgbwT+\/vtvKV++vFIyWbJk6ovYnwwZMkTq1q0b6+NQ3qVQ+tobVWrndgQ2b94s3bt3l7\/++itmqkYIDAoMjx07Vr1nyF+WMWNGuXjxoly+fFmNU7JkSUU08C56Cvantm3bypo1a9SfkWU+Q4YMcurUKbV3Qdq1ayfdunWLAz1+oOOHx+HDh9VnSDcC0vTvv\/+q\/0+SJImMHz9e1Q0Mh5DAhIAyjFaxYkUBIwX7fPPNN2OIyvbt21VW33379slDDz0kS5cu9VssMgQV2JUI2AYB5FKqVauW0mfr1q2SNm1aw7qF8i6F0tewgmxIBExGABnex4wZo\/KPQV555RWZOnWq+m89ArN69Wpp1aqVaot\/o+AwPCi3bt1SnpGePXsqMvPyyy8Lfix4CryfeCaIDT574YUXFPHAPvbpp5\/GkKJx48ZJtWrVYrrCa1OnTh1V+Dh79uzy9ttvy5NPPqk+P378uAwdOlQVPca433zzjWpjtZDAhIAwjI8FlyNHDrVowEQ9BS5tEBywWixUGJ9CBNyKAI5QtWKn+\/fvj9cD441BKO9SKH3dagvOy\/4I7Ny5U6pXr642enhSChYsKI899pghAgNisXv3bpUlHp4Yb1myZIl06tRJ\/WiGpwV7FARkHwWMQZ5AOBo3bhynb9++fWXWrFmSK1cuAVHSPKmrVq1Snhv8P4gKjqo8BfscfsBALxxfjRgxwnIjkMAECTHY6FNPPSUnTpyQ3r17q7M\/X4JFhMVUrlw55c6jEAG3IvDVV19Jhw4dlOcFHhijEsq7FEpfo\/qxHRGwAgHEnsATMnjwYHUUA4+JEQLz559\/qvhKCOJYChcuHEc9eGJwKgDPSK9evVQsJmTmzJmqvh\/iMhE\/g+Mjb\/E3PuoB4iShQoUKMnnyZJ+QIFyiT58+aj44GoNnx0ohgQkSXZxXwrsCWbRokTz66KPxGjR58uSyY8eOgH6VBqkauxGBiCCAmmL9+\/dXAezff\/+9YR1CeZdC6WtYQTYkAhYgAJLhWV\/PKIGZPXu2+tEMkoAfCv5q9L3xxhtqb\/IkHAhrQOAu9i7t6MrX1EqVKqXIj+ePc+1vnoTIKPmxAD41JAlMkMjijE9jtb\/99ptaTL5kw4YNMbctEGgYjnPBIKfEbkQgJAQ+\/PBDdZMCNcRGjRqlgtfxbpw9e1Z5ZR555BEVuOt9UymUdymUviFNlp2JgMkIGCUwWgxLoUKFlEfEn6AAMf7B8dHatWtVM9T1ww\/pli1bCo6K\/EmjRo1UXUC8r3iXPXXDjVt\/QbogZQUKFJCbN2+qGBnchrJSSGCCRFdjwQhYgivQn3i64+bPny9FihQJ8onsRgTsjQDOvPHlBm+jdivPl8YIVoSnRjtbD+VdCqWvvdGkdtGGgFECo3lR9MIStOMcHBMhLgVixIuCdlroQ9myZVWcp6en09+xlWavEiVKqFibHj16qJgZK4UEJkh0Ec+CIChEfm\/ZssXvKLh2pt2p\/\/zzz9UCohABNyIA1\/KcOXPU1PDlivQCOFqFixu38hBsiEBfCK5o4qomJJR3KZS+brQB5+RcBIwSGLxXX3\/9tTz33HMyYcIEvxPGD2btKjRuw+I9RLzMhQsXZODAgdKsWTO\/fbV3uXjx4oIfCfiRjoBhCAJ4cbPWn+DdP3TokLoZ1bVrV0sNQgITJLzaF2f69OlVsJI\/wU0kRH1DSGCCBJvdHIEArlEfO3ZMUqZM6TMRFtzLzZs3FxylwksDFzWCCUN5l0Lp6whQqWTUIBAogalatarKueJP4CnRCIQ3gRk0aFDMjUFf\/eE9mTt3rvgiMCtWrIhzA8lzKAmdGwAAEoJJREFUDHhtkCfm9ddfF8ThWCkkMEGia9R1feDAgZjzQj3XW5CqsBsRcAwCyCGhnYtrZ+mhvEuh9HUMaFQ0KhAwSmCMHiFpQfX4sYAr25DSpUurHxnxBeKiHcjHsmXLYgKAPZNU6u1jID1Iihff7VyzDEoCEySSuFuPQChIfEm7fvzxR5VMCIL\/RmZeChGIVgSQK0JzP2tu7FDepVD6RqsNOG97ImCUwGixZsgbA5LhT9555x11bJs7d26VzwWCHw\/4EQFPKOLQ\/EmDBg3UNWstnwtKHSBoGHFr8QXxIngX7zfSGyC3DZLkWSkkMEGi63k0NG\/ePL81YKZPny5w1+GWEm5kUIhANCOAuin44oVo9ZBCeZdC6RvNduDc7YeAUQKDox0c8eCoFmQEyep8CWJQEK+CG0MgHRDExCA2BmEN06ZN8wtCsWLFVG0l5Ixp0aKFaodYTsR0ouzBa6+95rPv3r17VWwOBORKe9etQpsEJgRktWAlnPPB5eZLkODu22+\/Fb3zyhDUYFciEHEE8OWLLyy4p6tUqeL3iwu\/6vDrDoKkWqjXAgnlXQqlb8SBowJE4H8IGCUwnqQdR6g4svEWZNrFbSDU5POMd9HiYlKlSiVI8eGdPR7jIOC+Zs2aakjPeBctLgbHUCgg6Us++eQTeeutt1RNwJ9\/\/tlvjhqzjE4CEwKSmosOx0JIs+ydCwaR20gVDXdafG63EFRgVyJgCwTgOsavNlS19fzF562cRuhReG79+vUxmUBDeZdC6WsL8KgEEfh\/74hRAgOwtCMeZOQFafAWzfOPK9QIms+UKZNqghtIuAmLukfIA6OFQXj2RzZtZNX2zjOjhUNgP0N2eeR18hSMCe8LCJZenhmzDE4CEwKSWAxYQCdPnlRFrcA8tVLi+KUJzwxcbvgMd\/IpRMDNCGiJ7DBHJMBCsKEW84X3APXAkAUUoh0faXiE8i6F0tfN9uDcnIVAIATG05PZpEkT9a4hpQd+SOAdw9EPvDBI\/9+5c+dYQGjvKcjNgAEDpF69eirlP96j999\/X5UJAElBELB32g9cvQYhypo1q0paqX2OW0cI2sWPEuiBUwdUuLZaSGBCRBgBvDAqjA+jw3WGQEX8EoXky5dPucrxdwoRcDMCuCaNKriICdMEX2LIPwGSD8E7giRZ3l+q+CyUdymUvm62CedmXwSQBwkbvqdgH4EgQapnnSIkQPWOWQHBAAHBO4U4GHhZzpw5I4gzgyBvCzLxepcaQHtcr9Z+TOBZSAeCdxQECOIvTwzGRwFILTEe0iCgPxLXIcAXpxBIbfDEE0+EBXgSGBNghvFQ\/vy7775T7jMsPFTyRNpmkBsYmEIEogUB\/ELD2TwSPGrEJVu2bOqsHtWqvV3PnriE8i6F0jdabMN52gcBzZthRCMQgi+\/\/DJOU+ReQk0j5FQCuQCBQCkPeFWw\/8QniFnDe4rSAvD+gMTgtADHP74KRGpjgSDhiAplDJBpHp4evN\/ly5dX5XXCedOWBMbI6mEbIkAEiAARIAJEwFYIkMDYyhxUhggQASJABIgAETCCAAmMEZTYhggQASJABIgAEbAVAiQwtjIHlSECRIAIEAEiQASMIEACYwQltiECRIAIEAEiQARshQAJjK3MQWWIABEgAkSACBABIwiQwBhBiW2IABEgAkSACBABWyFAAmMrc1AZIkAEiAARIAJEwAgCJDBGUGIbIkAEiAARIAJEwFYIkMDYyhxUhggQASJABIgAETCCAAmMEZTYhggQASJABIgAEbAVAiQwtjIHlSECRIAIEAEiQASMIEACYwQltiECRIAIEAEiQARshQAJjK3MQWWIABEgAkSACBABIwiQwBhBiW2IQBQj0KtXL\/ntt99k2LBhUrRo0ZCQ+Pzzz2XKlCnSunVradiwYUhjsTMRIALRjQAJTHTbn7M3EYFly5bJ7NmzZd26dT5HTZo0qaRLl07y5MmjiEDt2rUlf\/78Jmpg\/lB\/\/vmnVKpUSQ1ct25dGTVqVEgPKVGihPz777+SI0cOWbt2bUhjhaPz77\/\/LrVq1VKPgv0effRRSZ48uaRKlUomTpwYDhUCesaQIUPkjz\/+kBs3bsiWLVvUvyHvvfee1KhRI6Cx2JgI2B0BEhi7W4j6OQ6BL774Qvr06aP0TpIkiSxYsEBSpkypNu4DBw4IPt+2bZv6vFq1ajJ06FBJnz69Led5+\/ZtefXVV2XXrl0yevRoKVmyZEh6fvjhh8oD07ZtW2nRokVIY4WjsyeBmTdvXsgeqHDorD3j6NGjUrp0aUmQIAEJTDiB57PChgAJTNig5oOiBYH9+\/dL5cqV1XRz5swp33\/\/fZypv\/POO\/LBBx+ov5cpU0amTZumNhqKvRDwJDAgn04TrC0QGXpgnGY56msEARIYIyixDREIAAFsGNg4ILly5ZI1a9bE6X3r1i157rnnZN++feozkJnnn38+gKewaTgQcDqBqVChghw8eJAEJhyLhc8IOwIkMGGHnA90OwLHjh1Trvv4CAw+6969u+BYAtK8eXPp37+\/26Fx3PxIYBxnMiocRQiQwESRsTnV8CBglMC8++678v777yulEAszbty48CgYxFMQv4OjsVBjYPDoK1euyPr16+WZZ54JQpPwdiGBCS\/efBoRCAQBEphA0GJbImAAAaMEpmvXrirAF\/LGG2\/I66+\/rv4bN3\/mz5+vbjPh819++UUGDx4sJ06ckI4dO0rTpk1jtEBw7WeffaauOeM4KmPGjFK+fHnp0aOHuvHkS65fv65ibr799ls5efKkXLhwQR588EFp1KiRIlJaLA6Ixtdff610gC44jvj444\/jDLl8+XKZOnWqpE6dWhImTKjGK1KkiOTLl0\/dXILgyOznn39WY61YsUI9Azr7EgQOz5kzR8UOnT9\/Xvbu3SvZs2eXQoUKSatWrdQtLm\/BzZuvvvpKFi9eLJMmTZIsWbLIhAkTZOHChXLmzBnVt2\/fvlK4cGEDFrzbJBACYxRXjA6MgP+SJUvUTbSePXuqo8aPPvpIBXgj+Bt4I8A7bdq0Cj8Ef3\/55ZfqllHmzJnV7ahOnTrFOx8eIQVkbjZ2GAIkMA4zGNW1PwJGCMylS5fk2WefFbSF4Po1NjVsYBs3blR\/wyaM2BgQGxCNmzdvxgoKnjFjhtqscRRVvHhxtVHDi7N06VK1yeMz780em2Pnzp0F15lBoDJlyqSIEbwhFy9ejHVVGoGfhw4dUsdcIBxo401g5s6dK\/369ROQGO1Z6FOvXj1FhgYMGKDmAkK2fft2QXs8B2THF4EBeevWrZtcvnxZxo8fLw888IC6Crxq1Sq1yQMDzLdly5ZqXHw2aNAgRYyuXr2q\/jZ9+nR58803JUWKFJItWzbl7QERwjXo7777TuFqVIwSmEBwha1B+Pbs2aPUaNKkiSJ+33zzjRQrVkwSJ04sixYtUnNFMPhbb70l7dq1ExA7EDEQVcwJ0rt3b5VTx5+QwBi1NNs5EQESGCdajTrbGgE9AoNf6th4NO\/Lyy+\/LMjfAcFnr7zyivJWYJOvUqWK+gxjYiN7+umnlQfmxx9\/FPTDGJ5eBfSvWLGiunmCzQ6\/8DWPCo6AkHsGwcPe+VxGjBgRQ06wkYI4aFK1alW12foiMMgRkyhRIkUwPAXPxRyQ\/M5TBg4cqDxGvggMyAhyleC2z+rVq1WuGE\/B3+CBgSAHCwigJtC5TZs26n8LFCigsNJwAaGC7hi\/ffv2irgZFSMEJhhc4VGBLWFXYAFSCbuDyEA++eQTNQeQlscee0z9N+ypCTxxIKr+brlp7UhgjFqa7ZyIAAmME61GnW2NgCeBgasfXhUkPzt+\/LiKI8HxDTZVCDZskAd4CzQBYcEvdBAPkAB4STwFm1\/16tUlWbJkMSTI83PkoMFxA8Qzd8lLL72kjqNwHJQ1a9ZYYyLGBZlxoQc8BEjUpgn+vmHDhjgEBpvrQw89pDwFSOJXsGDBmD7Xrl2T4cOHq6MvT9HifnwRGHibcL3cF1HSxgABg7cDR2U4gsHxCgTenZo1a6r\/Br4gB56C4zHMPb6xfS0qIwQmWFzr1KkjW7du9amT5xqCRwoeGE+BxwtkDLJ582a\/eYRIYGz9VUHlQkSABCZEANmdCHgj4Ln54DOQARyJ4Nc1SMd9992nfk3Xr18\/zkaL9tj4cfwD8gMC4y07d+5UBAZHIp5EQ2uHZ4FAaGOBgGgZdfV+sfuypj8Cg7aYw6ZNm1S8DfSGt0aT\/\/77T3lnjBIYeIYQ74Ikd4jh8SXwOGkeFCTEK1eunGqGWCDtGvrMmTPjBBtrHgt4ZTTPl5GVq0dgQsFVww6Y4bjMU0BS8+bNq0gsSjlo3iWtDcgYSBlk5cqVKobJl5DAGLEy2zgVARIYp1qOetsWAb0jJD3FNQKDWA0cFXkLAlVx5ABigbZGRMsOjFtE2OADkfgIDIhDgwYNVFwLBLE1iHvx9MYYITA4+nr44YcFpKdLly7SoUMHnyrC+wIvDASxMK+99pohAoMxEdCLUgCILzEqegQmFFzjIzDQD6QEePgiMCgT8OKLL6ppxJchmATGqKXZ7v\/au2OcyIEgCqBcgARxAK4BQlyEgJCzIMghIUPiIARcAYmEgJSAG6yepULG65kxO54d4\/klrVbCnnb379n1p6r+79+IQAjMb9y1zHnSCGyawFSpxcvp\/v5+EBZVutHbol\/kJ7GMwBgHiaGGKVM+WYOLi4uGYLRLY+5dVEJ6f3\/fOzs7a6Yls9Dtnan5asalcBJ6gTTw1hyWZWA2RWDWwTUE5iffwtwbBP5GIAQm34ogMDICmyYw1EdM75SYnp+fBx1BQJnjZU+eK6ug\/DQ0VhEY42iQ1bdDBUVNJZRrZHvaJGYRgZFpkIGRiVEWUh7qC6Wxyu5YT0nKV5WQNkVg1sE1BGboNzD3BYF+BEJg8s0IAiMjsGkCowm3Xtx9Dau1HE221Ebkx6Vacg354fzbF7xfkCJKpoplBIZXS\/WhuF+GxPhUSIKPyfn5+ddYy5p4+ZogV4eHh41MGNnqxtvb29fcHh8fm5KV2BaBWQfXEJiR\/+FluJ1DIARm57Y8C940Au2zkP6laXZVDwySQGUj0+GsJS\/yrrcJ8mIcaiOnPutR8ZnPz8+m4VZmpNunIosiU0Hx0r62jMDoR+lrii3VD9lzncwN92UEpjJL7ru5udmj0ukGYqRcRSqtF4hnyjYJzDq4hsBs+l9ixp87AiEwc9\/hrO+\/I1DKFA8m9y1juqET0f+ht2XZZ29vb\/eur6+bId3HZZaZHXLCs8V1sm0NnpRP4uHh4ctYzs\/IcI+Pjxvy4zN6a2RTNAi3o6TLelSUidrBvE5zbNfhtk7b7p6CfHV11cjKlbBeXl6+jYV0aQimauLiSyrcVTG5Ds9u42q7ubdPhYSUcQCm\/uKfMjRWNfGug2vJqPtUSMYtL552s3LNm3SaWaBIE+\/Q3cx9c0MgBGZuO5r1bB0BL3SZjAp+JX3294smWi\/bRT4wPqcXxDO85PsCAfAiV46p0GdCISRj0xccYbu+LTxekJyPj4\/ek7W9ZD1Lzwq7f8ER2BECsj9KXG0Scnl52RxPIJSJun40SBcJtTKZMpYS2MHBQXN+EnddkmF\/k1y3gw9NHcXALE8TcTv4vzDIQ\/ZIkMswbtWXZQiB+RdcyaRPTk4aF2RlsO6e8OWp8phMDa+gdlQmys+4Iy86VyoqpFU7nOu\/GYEQmN+8e5n7pBDwG77fhvWFePFX8GqR2WCt78+ioA5SRvH5iqOjo+Zl7bftPhLEdM5zGbl5JkKhl0QJp6+HxLhIj7KPrIWXr3IRYnF6evr1XOUkmZKnp6dvGST3mA\/zNoH0aL6lQEI0ECskRLmKxX1lfxwhYF2IRoW5Ug4hbHWfa3UWEuLH82Z\/f7\/5mRe+e9ukzP2yPQiA4xYEkoJAuRdpcWCmc4Yqam7t9S7akyEEpj47BFf3OlYBDjUna7O\/cEXaZM84G9sfYR9dZ3rIldd6fM9qvTJoJNX6orqmhyEwk\/ovIpMZGYEQmJEBzXBBIAjMB4GfEJgprjoEZoq7kjmNhUAIzFhIZpwgEARmh0AIzOy2NAuaEQIhMDPazCwlCASBcRFoExiNzovKcuM+dbzRlMmo4rrN1OM9ISMFge0hEAKzPezz5CAQBCaOQJvAMM7Ti9J1F57iEvTV8PNxYrkIgZniLmVO6yIQArMugvl8EAgCs0WAJJ5yqxsas+\/u7ia3bieZv76+\/jUvknnnYCWCwJwQCIGZ025mLUEgCASBIBAEdgSBEJgd2egsMwgEgSAQBILAnBAIgZnTbmYtQSAIBIEgEAR2BIE\/E7iAsCTYwnYAAAAASUVORK5CYII=","height":337,"width":560}}
%---
%[output:37900f64]
%   data: {"dataType":"text","outputData":{"text":"Evaluating episode 50 \/ 100...\nEvaluating episode 100 \/ 100...\n","truncated":false}}
%---
%[output:3a471d79]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7tfQl4VdW59gcqwxWtWLgZQGpARCMXkp8CKQZB49BAEaHEYqVwbxRoFa9etFBBZfARRDSKEQWlKEOQQCooOECRoeIU65VbK2gNcaKiYgkXpYmi5H++5d3Hk+Scs9c+Zw9rf3n38\/RpS9bwfe\/7rbXes8YW9fX19YQPCAABIAAEgAAQAAIhQqAFBEyI2IKpQAAIAAEgAASAgEIAAgaBAASAABAAAkAACIQOAQiY0FEGg4EAEAACQAAIAAEIGMQAEAACQAAIAAEgEDoEIGBCRxkMBgJAAAgAASAABCBgEANAAAgAASAABIBA6BCAgAkdZTAYCAABIAAEgAAQgIBBDAABIJA0Aq+99hr94he\/oN69e9O6devilvPAAw9QSUkJTZw4kaZOnZp0fVZG3XpTrggFAAEgYCwCEDDGUgPDgIA9At988w2deeaZcRN26NCBKisr7QtKMsXf\/\/53Wrt2LaWnp9Po0aPjlsI2vPTSS\/TjH\/+Y8vPzk6zt+2y6Auavf\/0rLVq0iF599VX63\/\/9XzrllFOU2Lr66qupf\/\/+qsB77rmHFi5cGCm8VatW1LFjR\/rJT36iBFe3bt0ifysqKqLXX389pv0\/\/elP6cEHH0zZNxQABICAHgIQMHo4IRUQMBKBaAFz4YUXUuvWrRvYefLJJ9Mdd9wRmO180Tf\/p2XLlq7aoCNgtm7dSr\/5zW\/o6NGjSpCw0Dtw4AD97W9\/o+OOO47uu+8+Gjp0aETAnHbaadSrVy86fPgw7d69m\/7xj3\/QCSecQA8\/\/DANGjRI2W8JmJycHOrUqVMDn3Jzc6m4uNhVP1EYEAAC8RGAgEF0AIEQIxAtYHiWgQfqWN\/OnTtp7NixxLMEPPNw\/\/33q4H917\/+NQ0fPpz+8z\/\/k9566y3igZkHdp5R4RmTMWPG0HnnnUd9+vShFStWqFmM888\/n+6++25q164dNRYSVj2FhYVqgF++fLnKx7Y1XkLasmUL3XvvvbR37176wQ9+QJznpptuUuXyt2PHDmXLO++8QyeeeCINHjyYbrvtNjrppJOa1NvYZ\/Zt4MCB9Nlnn9EVV1xBs2bNouOPP14l27BhA11\/\/fV0zjnn0FNPPaXs4hkYTmeJva+++opmz55Njz\/+OPEsFtvStm3biIBZsGABDRs2LMSRA9OBQPgRgIAJP4fwoBkjoCtgWEDwAM0Ch5dQevToQbwvhT+eObjoooto8+bN9D\/\/8z90+eWX05133qlEB+fhJZVLLrlEDdiPPPKIEg9XXXUVTZ8+vYmQsPJkZGSofAMGDFDLNU8\/\/XQDAcP1XHbZZUq48N95xuO5555TdixevJiqq6uVoPmXf\/kX+t3vfqdsWb9+fcQ2uxkYXrLiJS0WQy+\/\/LISQNHfhx9+SF26dFH\/ZC0hRQsY\/ncWMbzcxTMxS5YsoQsuuAACphm3NbhuHgIQMOZxAouAgDYCdntgfvvb36plFGvA54GcB3eeTWAB8Ze\/\/CWysdYa9M866yx65plnInmsmRZenuI9JZdeeimdeuqp9Oc\/\/7mJgLHq4dkOnrVgIcNf4028PPPDgolnOXiWh8XCLbfcQt9++62aBfn4449V2SwyWATxXhueUfnRj35E27Zts52BqaiooClTptC\/\/du\/0ZNPPpkQz3gChjONGzeOXnjhBWUbLw8lWkLi2S1eksIHBICAPwhAwPiDM2oBAp4gYLcHhpeHeG+MJSzOPvtsNRvCH29Q\/eMf\/6iWaViU8KwEL9NkZmYSLwVZeXhfCM9+WLMSXAZ\/vOTEgib6FJKVhze+ctnW11jAFBQU0Hvvvac2APPyVOPvyy+\/VLNA27dvV\/tWWrRoQV9\/\/bVazmGhZTcD88QTT6jlqOzsbNq4cWPSAuaXv\/wlvfLKK3T77bfTlVdeGREwsQrkZbgbbrjBE55RKBAAAk0RgIBBVACBECOgu4QUa8C\/5ppr1LJNaWmpmjnYt2+f2u\/SWMBEiwDe4Mr7ZOwETONj1Y0FDO+j+eCDD6i8vJz69u3bhIEZM2aovTMsvv7rv\/5L7b1hMaErYPikEM+W8BIUCxBrX41VEe+\/YRt5SS3eDAyLKF5CYp9XrVpFeXl5WEIKcVuB6fIQgICRxyk8akYI+CFgeOmI96DwiSZrY288IRFvZqSxgBk\/fjw9\/\/zzxEKFl2n4pNKECRPo0KFD6igy\/2\/eJ8P\/m5dmeNbouuuuox\/+8Idq9sVuBoaXolgksSjjGSJequITRfzxJl6eLWHxwstcXEfjTby8Cfjmm28mnsnhzcicjk9SWUtI2MTbjBoZXDUWAQgYY6mBYUDAHgG7JSQugffBfPrpp00unNOdgeFTP7wvhpd9ysrK6KOPPiLOy0s0jYWEroCx0nHZvB\/m7bffVsKC97ksW7ZMiRUWLVwnbyB+6KGH6MiRI2o5iZeWTj\/9dNsL9HipicUR769hscI+fP7557Rnzx51jJpFyJAhQ5oco66rq1PLY5988gm1adNG2WPNEkHA2MckUgABvxCAgPELadQDBDxAwG4TL1fJN+Ty\/pHGN+bqChheMuIlJj4d9MUXX6jTQbzRljcCJytg2C7eKMwzM1VVVeo00sUXX6xu6eWZHhZJN954o9pkzGKFj09zujlz5qi0nE\/nBuB3331XiR+eOTp48GDkIjue4bFESeOL7Ng2Fjy8fHTttddS165dI8xBwHgQxCgSCCSJAARMksAhGxCQjoDdMo10\/+EfEAACZiMAAWM2P7AOCASGAARMYNCjYiAABDQQaLYChtfSp02bpjYS8pp44yvYNbBDEiAgGgEIGNH0wjkgEHoEmqWA4fV13tw3cuRIdTsoBEzo4xgOAAEgAASAQDNDoFkKGD5dcOzYMaqtrVVXl0PANLOoh7tAAAgAASAQegSapYCxWONH5CBgQh\/DcAAIAAEgAASaIQIQMBozMNZ14s0wPuByAgTqzhpOdWddSm3eforavJ34vR0ACQSAABAICgG+RZpvk5b2QcBoCBi+B4Jfx5X2NWe\/Xtx7KGU65216n3ZW1dDUS7Jo6iWnp1yeXQHNmS87bEz9OzgzlZnYdll81e3eYWt4m+xBCdPYlZEofyp5YxklNQ4hYCBgbBtq2BLYNdZTJ29z1SUImNTgtOMrtdKDzS3Vt7D7VV3UgrqurW8SHJZf\/He7L1b+6Dx2ZSTKb5eX62l\/+UxqXzTDzkz197DzFc9JCBgIGK0GEKZEiRorz7wMW\/iGcif\/jPauuPXUNd89buj1J7UTkuqX5IEjzJzVrJ1FNWtmUuas7dR4FsTya\/\/M822bc8bMxD+E7MpIlN8uLxtnV3+0A2HmKxERzVLA8NXi\/A4KPyDHj7a1atVKYbR06VIaMGBAXFVuG9EhSzBr1iz1mJ60L5FfloBh8eKX8HAL3+bIl1vYBVUOOAsK+Yb1smixvrq3tlPtW9tjzmBI5QsCxow4DMQKqeS\/9957lJWVFQimXlaayK8wC5jmyJeXceJH2eDMD5Tt64i1JNN1zbdELVo2yCyVL6ljWLOcgbEP94YppJIvtbFCwDiN8GDTS41DRlWqb2HzK3oGxor2WPtHwuaXbsuVOoZBwGhEgFTyw9xYL31wV1zm6upqqU2btnH\/zieHwriEFGa+EjUzqX6FTcBY+y509mbU1dVRmzZtYtKqkz9RPHiZ3667lxqLUscwCBi7iBa8gzusjZWPL8\/b9J4Gc\/GTQMCkBJ+rmcMahzoghMU3a2Mr+5Tq6RiT89txFha+7Pxo\/HcIGKeICUovlfywNlZLwLAIiXX\/yv79+ykjIyNhBJ7b7ZTQRWhY+bIDWqpfYZqBsQRM23MGJzzdYt1PkqiNpXK\/CWPmZf7mGotSxzDMwNhFNGZgNBBKPUkyd7PEu39F6oAIv1KPM79LSMSZ3V0fXs5kMA6x6te9WwSx6HckpVYfBExq+IU6t1TyTeqEkhEwB0ti39Vgkl9uBj78chNNf8pKxBnPZnw8Y3BcQ4IQMHaXs1nGIhb9iR+3apE6hmEGRiNCpJIfVCcU6xp\/63K5eKJEg6ZIkqD8cmJjMmnhVzKo+Z8n+hp4a6nF7tp5\/61MrUbEYmr4+Z1b6hgGAaMRSVLJD6IT4tNDfAoo3gcBEz8gg+BLo3mknESSX\/GWhXRnNlIG06cCJHEWDZlUv6SOYRAwGg1eKvlBNFZLwMS6xv9nvTrShPxOGowkThKEXykbrVEA\/NIAKcAk0UtCvBmWP+u4sZNr3wN0QbtqxKI2VEYklDqGQcBohJdU8r3shPikUKyPl494BmbDtbnk1UkgL\/3SCBfPksAvz6B1pWBLwESf5AFnrkDrWyFS+ZI6hkHAaDQNqeR71VjtlokYcjeWiuJR55VfGqHiaRL45Sm8rhTOS0gQMK5AGUghUtuY1DEMAkajmUgl36vGagkYPuYc64t1d4sGDdpJvPJL2wCPEsIvj4B1sVi+TyX6inpw5iK4PhQllS+pYxgEjEajkEq+F401evbFy2WiRLR54ZdGmHieBH55DnGTCqzr9aP\/wC8Z86dzZwo485+zVGqUypfUMQwCRiPapZLvdmNtvHTk5TIRBIxG4IYkidtx6Jbb0dfrxypT52SRqb6lihH8ShVBf\/NLHcMgYDTiSCr5bndC0UtHXi8TQcBoBG5Ikrgdh267HX2vS3TZOne7mO5bsljBr2SRCyaf1DEMAkYjnqSS70YnFOsG3aCWjiwq3fBLIyx8TwK\/\/IX86w\/\/Qq269EqpUnCWEny+Z5bKl9QxDAJGo4lIJT\/VxspHoq0bdKNhDGrpCAJGI5gNTJJqHBroUsQkqb7BL5OjrqltUscwMQKmvr6eSkpKqLy8XF0e1bt3b5o7dy517ty5CZtfffUV3XXXXfTss8\/SkSNHqF+\/fjRnzhzq2LFjzKiUSn6qnVD0q9BPXZNjTItO1S9jHGlkCPwylZn4doGzcHEmlS+pY5gYAVNWVkZLliyhFStWUFpaGs2fP58qKytp\/fr1TVrQPffcQ1u3bqXHHnuMTj75ZPrtb39Lhw8fVv8\/1ieV\/FQaa\/TSEd+qCwHjfUedCl\/eW5d8DSb5xfe46GzO1fXWJN90bdZJB790UDInjdQxTIyAKSoqouHDh9OYMWNU1NTW1lJOTg5t3LiRunfv3iCShg0bptL94he\/UP++b98+GjhwIL322mvUoUOHJlEnlfxUOiFLwJgmXpi8VPwyp8tpagn88pYd6yZdnePRupaAM12kzEgnlS+pY5gYAcNiZdGiRZSXlxdpCQUFBTR58mQaOnRog9bBQueKK66g0aNHq3+vqamhPn36qOWnvn37QsDE6UuinweYt+k9lSro\/S6xTJXaCcEvZ4Nc40vlGufmvzf5tzUzG9yk66xGiM5U8Qo6v9Q2BgETdGTZ1N+jRw9avXo15ebmRlIOGTKEiouLadSoUQ1y33fffbR582ZatmwZtWvXjubNm0erVq2ipUuXUn5+fkwBE\/2P48aNo7FjxxqOiL15PPMUa49QrJx9SmO\/bfT6dafbV+RzCid++WxaStXBLwfwLf4l1e99hVrcVR03U\/2UrjH\/1qJbHtHEVQ4qi58UnLkCo2+FSOFr+fLlanyL\/qqr47cF3wB2uSJRMzClpaVqKcj6WIxMnz6dCgsLG8BmbeJlEXPiiSfSf\/zHf9CMGTOooqKCevbsGVPASCTfya8Na8ko+nmAIO96SdQOnPjlcnvytDj4pQ8v36DLN+ZmztpO8e5riTUDwzVEPwWgX2PslOAsVQT9zS+VL8zA+BtHjmvj5SBeMho\/frzKy8tCfLqIRUpWVuw3eaxK9uzZQyNHjqT\/\/u\/\/prZt20LAxEDfEjAmLhk1NldqJwS\/9LsF3ozLXyIBo19a8inBWfLYBZFTKl8QMEFEk4M6efZkwYIF6hRSRkYGzZ49m6qqqtS+Fv62bNlCmZmZlJ2dTY888gi9+uqrxDM2R48epYkTJ9KZZ55Js2Y1XRPnvFLJd9JYIWAcBKNHSZ3w5ZEJnhTrhV+WgHHzRFEyznvhWzJ2uJ0HfrmNqLflSR3DxCwhMf0sYFauXKlOIPFmXL4HJj09XUXGiBEj1AzNpEmT1JHp3\/3ud\/Tyyy8T3x\/Dm3xvu+02at26dcwokkq+k04IAsbbDkandCd86ZRnShov\/IKA8ZZdLzjz1mK90qX6JXUMEyVg9ELUeSqp5Os01tqjx6jT1B0R0LCE5Dx+3Mqhw5dbdflZjtt+WeJFzZ6urffTlSZ1ue1boM5EVQ6\/TGFCzw6pYxgEjAb\/UsnX6YSs23YtmCBgNALGoyQ6fHlUtafFuu0XBIyndKnC3ebMe4v1apDql9QxDAJGI66lkm\/XWPmtIxYwO6tqiE8fmXrqqDGFdn5pUG5kkubmV7xXoGvWzLQ9YWQKgc2NM1NwT9YOqXxJHcMgYDQiXSr5iRrrpQ\/uUsLF+iBgNALF4yRSO9d4fkXPpMSCNugTRjp0NzfOdDAxOY1UvqSOYRAwGq1JKvk6AoafCuDPpLeO7CiT2gk1N7\/4npa6t7bHpLt1jwF06hV32IVC4H9vbpwFDniKBkjlS+oYBgGjEfBSydcRMBuuzaVzu52igZI5SaR2QvDLnBjTtQSc6SJlRjqpfEkdwyBgNNqNVPJ1BEwYNu1iD4xGEBucROqgwZBL9Q1+GdygYpgmdQyDgNGIQ6nk6wgYzMBoBIhPSTBo+AS0i9WAMxfB9KEoqXxJHcMgYDQahVTyYzVW3rzLn7WBFwJGI0B8SiKtc+X3ivirq6ujNm3aqJNF\/LW\/fKar7xH5RE\/MaqRxZjkJv4KMKud1Sx3DIGA0YkEq+Y07ocZ3vjA0WELSCBCfkkgbNOKdMupa\/g1Ry+N8QtXbaqRxBgHjbbx4VbrUMQwCRiNipJLfuHO1jk7zySO+8yVsm3fRuWoEs0FJLAHT4tePq\/fLrC\/e69EGma5tCgSMNlRGJJTKl9QxDAJGo9lIJT+6sVpvHTEcYbrzJRZ9UjshaX7xMWm+lK7FXdW2L8ZrNFMjk0jjDD8SjAwzW6OkjmEQMLbUy3+Nmm\/cHbbwDYUEz76E6c4XCBiNADY4Ce+DqRu3FALGYI7QxkJGTgxzIWDCz2HSHkgl3\/p1aAkYCeKFScav3qRD3fWMPMtife2LZsQsXypfiEXXw8nzAqXGotQxDDMwGk1CKvmNl5AgYDSCIcAkYetceXbFOlnEsMV7GTpsfjkJAam+wS8nURB8WqljGASMRmxJJR8zMBrkG5QkbIOGJWD4WDR\/mIExKJhSNCVssajrrlS\/pI5hEDAakS2VfKuxjln6Jj3z189F7H\/BtL1GQPuUxDplZPfootRBA7HoU6C5WI3UWJQ6hkHAaAS\/VPKtxmqdQMISkkYwBJgkTJ1r9PJRvKUjC8ow+eWUfqm+wS+nkRBseqljGASMRlxJJb+xgAnjpXWx6EPnqhHUHiexBIzd7IvkWQrJvqGNedyAXC5e6hgmRsDU19dTSUkJlZeXq6vJe\/fuTXPnzqXOnTs3CYWjR4\/SnDlzaMeOHdSiRQvq2LEj3XbbbZSdnR0zbKSSz51Qn9L3Iz5DwLjca7hcnImDBi8TxZphqdu9gz6eMZggYN4TeUTcxFh0o7lJ9UvqGCZGwJSVldGSJUtoxYoVlJaWRvPnz6fKykpav359k7h+4IEHlHhZvnw5tW3bVv334sWL6cUXX4SAcaMXCLgMqZ2QaX5Zsyx2S0R24WCaX3b2Ovm7VN\/gl5MoCD4tBEzwHCS0oKioiIYPH05jxoxR6WprayknJ4c2btxI3bt3b5B30qRJ1KVLF5oyZYr6d26MBQUFtGvXLjr55JOb1COV\/OgZGCmzL5i297ah8syK9fEtunxMWmeWJZFVUgdDxKK3sehF6VJjUeoYJmYGhsXKokWLKC8vLxLXLEomT55MQ4cObRDrvMz02GOPqdmaDh06EM\/IPP\/887Ru3bq4MzDRfxg3bhyNHTvWi\/bja5n79u2j4eu+UXW+ft3pvtbtZWXsV6ylQy\/r9KPsoP2qn9I1ppv8lhF17Z80BEH7lbThGhml+ga\/NMgPMAmvKixbtqyBBdXV1QFa5E3VYgRMjx49aPXq1ZSbmxtBasiQIVRcXEyjRo1qgB7vl7nuuuto06ZNdOKJJ1KbNm1o6dKlzW4PTMXOPTThiU8UNpiB8aaBuVlqkL8OrXeL2J+25wxu4FbGzG0puRmkXykZrpFZqm\/wS4N8g5JgBsYgMmKZwjMwpaWlNHDgwMif8\/Pzafr06VRYWNggy+zZs+mjjz6ie+65Ry0ZPfvsszRt2jQ1C3Pqqac2KV4q+TeXv0GLXz0k5v4Xizh0roY31kbmSeWL3ZTqG\/wKVxuTOoaJmYEZPXq02scyfvx4FVk1NTXUr18\/2rx5c5NTACxsbr755gZLS7169VICaNCgQc1CwFz64C7aWVWjfJVy\/wsETLg6Vel8QcCELx6lCjMIGMNjsaKighYsWKD2tWRkZBDPslRVValj1fxt2bKFMjMz1TLRhAkTqHXr1moGplWrVvTSSy+ppSaegenUqVOzEjBTL8miqZfI2f+CQcPwhhrDPKmDBmIRsWgKAhAwpjCRwA4WMCtXrlQnkPr27avugUlPT1c5RowYoWZo+ATSgQMHlMDhU0fHH3+82gdzww030IUXXhizdInkW7fvbrg2l87tdkoI2NU3UeqA6KdffESav1T3t+iw5qdfOva4mUaqb\/DLzSjxviyJYxijJmYJycsQkEZ+9PKRpM270pck\/Bo0ojfsMqb8GGO8hxjdaHd++eWGrU7LkOob\/HIaCcGmlzaGWWhCwGjElTTyLQHz8Mh0GpV\/tgYC4UqCzjU1viwBw6eNWLy0yW66Lyy1GhrmlsoXeynVN\/jlZgvwvixpYxgEjIOYkUa+tXwEAeMgCAxI6tegYb0i7fXMi\/QZMwgYAxqNQxP8amMOzUo5ubQxDALGQUhIIv\/FvYdo2MI3lPd8eV1WVpYDJMKRVGon5JdfloD50e8P0HEnd\/CcdL\/88tyRGBVI9Q1+BRFNydcpaQyLRgFLSBoxIYl8S8Dw0ekFhT+AgNHg35Qkfg0aloBJ9Y0jXdz88kvXHjfTSfUNfrkZJd6XJWkMg4BxGC+SyL\/j2ffonj++r+5+gYBxGAgBJ\/dr0ICAcY9ovzhzz2K9kuCXHk6mpJI0hkHAOIwqSeRbG3j59BE6IYeBEHByv\/iCgHGPaL84c89ivZLglx5OpqSSNIZBwDiMqrCTP2\/T+xGPeQmJb+Dl+18yW9ZgCclhLASZHINGkOgnVzc4Sw63oHJJ5SvsY1i8eMAeGI2WEmbyo+98iXYVAkaDeMOSSO1cpfrF4SPVN\/hlWOdgY06Yx7BErkHAaMRhmMm3BAw\/GWB91tMB6IQ0yDcoCfgyiAxNU8CZJlCGJJPKV5jHMAiYFBtHGMln4cKf9WBjrCcDpDZW+JU44K1nAqxUtW9tV\/8zc9Z2zy+ti2WZVL4wA5NixxtAdqmxGMYxTId+zMBooBQ28mMtG8V6MkBqY4Vf8YO68TMB0Sn9Ojbd2DqpfEHAaHSuhiWRGothG8N0wwICRgOpsJEfvWyUf8YpcR9rlNpY4Ze9gLGeCbBSev1cQKJmJpUvCBiNztWwJFJjMWxjmG5YQMBoIBUm8q1nAtgtu5empTZW+KUR1AYlkcoXBIxBQaZpitRYDNMYpkmVSgYBo4FWmMiPFjB2L01LbazwSyOoDUoilS8IGIOCTNMUqbEYpjFMkyoIGF2gwkS+JWDsxAs6V132zUnntHP1+0K6ZJFy6ley9QSRT6pv8CuIaEq+zjCNYU68xAyMBlphIh8CBndvWCENAaPRuD1OgoHeY4BdLl4qX2Eaw5xQCgGjgVaYyIeAgYDhkLZOGx33g3+lHy35VCPKg0siddDALGdwMZVszVJjMUxjmBPuxAiY+vp6KikpofLycqqrq6PevXvT3LlzqXPnzk3wGDFiBO3Zs6fBv3\/99dcqb9++fZukDxP50W8d2QWC1MYKv4hq1sxUIqb95TOpfdEMu1AI9O9S+YKACTSskqpcaiyGaQxzQpwYAVNWVkZLliyhFStWUFpaGs2fP58qKytp\/fr1tni89tprNG3aNHr66aepVatWoRUw\/M7RsIVvKPuxB+b7m4dtAyAkCXQ717rdO+jjGYOVVxAwwZKry1mwVjqvHX45xyzIHBAwQaKvUXdRURENHz6cxowZo1LX1tZSTk4Obdy4kbp37x63BJ55GTJkCN1xxx3Uv3\/\/mOnCQr4lYPjZAOu5gETQoRPSCCyDkujyZQkYvuslY+Y2gzyIbYquX8Y7EsNAqb7Br3BFY1jGMKeoipmBYbGyaNEiysvLi2BQUFBAkydPpqFDh8bF5aGHHqLdu3dTaWlp3DRhId9aPso\/oz09dU2ObSygE7KFyKgETvjiDbwQMMHT54Sz4K3VtwB+6WNlQsqwjGFOsRIjYHr06EGrV6+m3NzcCAY8s1JcXEyjRo2KicuRI0do8ODBtGbNGsrKir\/kwORHf+PGjaOxY8c6xdrT9A9XHqLFrx5Sdfy4cxtaPCLdtr59+\/bF3CNkm9HwBPCLqH5KV2rRLY9o4irD2SKSyhcDL9U3+GV2s1q+fDktW7asgZHV1dVmG52EdWIEDM\/A8CzKwIEDIzDk5+fT9OnTqbCwMCY0TPKWLVuI\/zvRFwb1Om\/T+zRv03ukO\/vC\/uJXVBItJsAsTvjiZaQgnwdwApMTv5yUa0Jaqb7BLxOiS9+GMIxh+t58n1KMgBk9ejTxktH48eOVdzU1NdSvXz\/avHlz3NkVnkW58MILbWdTTCc\/+vZd3f0vEDDJNJdg81iDhnW\/i2VNUI8wuoWG1MEQbcytCPGvHKmxaPoYlizDYgRMRUUFLViwQJ1CysjIoNmzZ1NVVZU6Gs0fz7RkZmZSdnZ2BKuzzjqLHn\/88QbLTrGANJn86JNHbLvO6SPLR6mNVbpfEDBMKcJmAAAgAElEQVTJdnf+55Mei\/4j6m2NUvkyeQxLhVExAoZBYAGzcuVKdQKJ73Phe2DS07\/bC8J3v\/AMzaRJkyIzNH369KEXXniBOnXqlBBDk8m3BIyTpSMImFSaTHB5G8\/AhH3mRXocYgYmuLaSbM0QMMkiF0w+UQLGKwghYLxC1ptypXZCEDDexIuXpUqPRS+xC6JsqXyZPIalwjMEjAZ6JpOPGZimBErthNivU\/68XN2ySy1aUNc1xzSi1\/wkUvnCDIz5sdfYQqmxaPIYlkqUQMBooGcq+d8cq6d7\/viB49NH0qfupXZC0QImDDfsajQtlUQqX5J9k8qZVL9MHcN0+4h46SBgNBA0kfzGm3exB+Z7IqV2QvBLo7EalgScGUaIjTlS+TJxDHMjMiBgNFA0kfzopaNzu52i9XRAc5kuldoJwS+NxmpYEnBmGCEQMOEixMZaCBgNOk0UMGw23\/+SzMwLlpA0SDcwCQZDA0lppgMiYjFcsWjqGJYqihAwGgiaSH4yN+9iBkaDbIOTYNAwmJw4poGzcHEmlS8TxzA3IgMCRgNFE8m3BIyTm3chYDTINjiJ1M5Vql8cSlJ9g18GdxQxTDNxDHMDQQgYDRRNI996dZpNh4BpSqDYzvXhGyhrwn0aERuuJFL5goAJVxxK5su0McytyICA0UDSNPKjBYyTpwMwA6NBtqFJ+HHGj2cMVtZJuYHXghoCxtCgS2CWVM6k+mXaGOZWxEPAaCBpGvnW8tGGa3OJTyAl+0ltrFL94jeQ2p4zmDJmbkuWciPzSeVL8i96qZxJ9cu0McytjggCRgNJ08i3Xp+GgIlNntROCAJGo7EalkRqLMIvwwLNxhzTxjC30IOA0UDSJPKjL7BLZfkIvw41iDcoibWEJOkGXiwhGRRgDk2BgHEIWMDJTRrD3IQCAkYDTdPIT\/X+F+kDh7TO9dATd1D9t9+oN5CwhKTRYA1KIi0W0XcYFFwOTDFtDHNgesKkEDAaSJpEvhv3v6AT0iDdkCTRm3fZJAgYQ4jRNAMCRhMoQ5JJ5cukMcxNqiFgNNA0iXw37n+BgNEg3bAkLGT271yHY9SG8WJnjtQBEX7ZMW\/W300aw9xEBgJGA02TyIeAsScMnas9RialkMoXYyzVN\/hlUguyt8WkMczeWv0UEDAaWJlEPgSMPWHoXO0xMimFVL4gYEyKMj1bpMaiSWOYHhN6qSBgNHAyiXwIGHvCpHZC8Muee9NSgDPTGElsj1S+TBrD3IwIMQKmvr6eSkpKqLy8nOrq6qh37940d+5c6ty5c0y8tm7dSnPmzKFPP\/2UTjvtNLrtttsoLy8vZlqTyIeAsQ9\/qZ0Q\/LLn3rQU4Mw0RiBgwsVIYmvFCJiysjJasmQJrVixgtLS0mj+\/PlUWVlJ69evb4JAdXU1XXbZZbRo0SIlWtauXavSrVq1ilq0aNEkPQRMuEJewqAR694XCX7FiiSpfmEJKVz9hmS+TBrD3IwKMQKmqKiIhg8fTmPGjFH41NbWUk5ODm3cuJG6d+\/eADOeeTl8+DDdeeedWliaRD5mYOwpkzAgWgIm+ti0BL8gYOzjNwwpEIthYOl7G00aw9xEToyAYbFizahYABUUFNDkyZNp6NChDTAbPXo09erVi9555x3au3cvderUiaZNm6aWnWJ9TH70N27cOBo7dqybPGiX1af0fZV2Yv9TaEK\/5N9B4jL27dsXd4lN2yADE0rxq35KV2rRLY9o4iqFshS\/GoeMVL\/AmYGdg41JUmJx+fLltGzZsgbe8sqDtE+MgOnRowetXr2acnNzIxwNGTKEiouLadSoUQ14438\/evQoPfroo5SRkUFLly6lhx9+mLZt20bt2rVrwrEp6tXNZwTYSfyKMrc5YwnJXG6cWIY25gSt4NNK5cuUMcxthsUIGJ6BKS0tpYEDB0Ywys\/Pp+nTp1NhYWED3K644grq378\/3XDDDZF\/59mXhx56iAYMGGCsgLn0wV20s6qG8s9oT09dk5NyLEhtrFL8avx4oxS\/GgeuVL\/wIyHlLsr3AqTGIgSM76HkrEJeFuIlo\/Hjx6uMNTU11K9fP9q8eTNlZWU1KIxFzXHHHUezZ89uIGB4FoaFTePPBPIt8cK2QcAkjg0JnVDN2llN3j6S4Fcs5qT6BQHjrA83IbXUWDRhDPOCXzEzMBUVFbRgwQJ1ComXhVicVFVVqWPV\/G3ZsoUyMzMpOzub3nzzTbXZd+XKldSzZ0967LHHaPHixWoJqW3btkYLmKmXZNHUS053JRakNlYJflkCJvr1aQl+QcC40nQDLwSxGDgFjgyAgHEEVzCJWcCwKOETSH379lX3wKSnpytjRowYoWZoJk2apP4\/Cxtecvriiy\/ojDPOoFmzZikxE+szgXx+gZq\/Ddfm0rndUtu8a\/mITiiYONWpFXtgdFAyPw3amPkcRVsolS8TxjAvIkHMDIwX4FhlBk2+25t3IWC8jBb3yuY9MF3X1kcKlNq5SvWLiZPqG\/xyr537UVLQY5hXPkLAaCAbNPmWgHFr7wsEjAbpBibBoGEgKTYmgbNwcSaVr6DHMK+iAAJGA9mgybeWjyBgNMjCr149kAxKJXXQwAyMQUGmaYrUWAx6DNOE33EyCBgNyIIkP3r56OEx2TTq\/6VpWKyXRGpjhV96\/JuSSipfEDCmRJi+HVJjMcgxTB995ykhYDQwC5J86+kAt2df0LlqEG9YEqmdq1S\/0MYMa0Aa5kiNxSDHMA3Yk04CAaMBXVDkW0tHbCIEjAZR\/5ckzJ0Qb9zlL3rzruV5mP1KxJ5UvyBg9NusKSmlxmJQY5jXvELAaCAcFPmWgHHz7pdod6U21jD7BQGj0SBDlCTMsdgcRadUvoIaw7xuqhAwGggHRb4lYA6WnK9hpfMkUhtrWP369vMP6YPf\/AgzMM5D2dgcYY1FO0Dhlx1CZv09qDHMaxQgYDQQDop8CBgNcmIkCWvnas2+YAkpOd5NzBXWWLTDEn7ZIWTW34Maw7xGwSgBU19fTwcOHKB\/\/OMfyu8OHTqo\/7Ro8d2+gKC+oMiHgEmO8bB2romWjxiJsPplx6JUv8CZHfPm\/V1qLAY1hnnNsBEC5rXXXqPly5fTSy+9pB5hjP7at29P5557Lo0bN4769OnjNR4xyw+KfAiY5OgOYydkN\/uCwTC5WAg6VxhjUQcz+KWDkjlpghrDvEYgUAHDYmXKlCn0pz\/9iYYOHUoDBgygs846i1i08Hfw4EF6++236YUXXlCvSp933nk0b968yN+9BscqPyjyIWCSYzhsnav1cGMk3qKeD4hGIGx+6bIn1S+ITt0IMCed1FgMagzzmtlABUx+fj4NHjyYrr\/+eurYsWNCXz\/77DO6\/\/77afv27bRz506vcWlQfhDke\/X+EQZEX0PH1cqkdq5S\/YKAcTX8fSlMaiwGMYb5QVigAmbLli104YUXOvKTZ2IuvvhiR3lSTRwE+dF3wOAUkjMGpXZC8MtZHJiQGpyZwIK+DVL5CmIM00c9+ZSBCphos1euXEljxoxp4smRI0forrvuolmzZiXvZYo5\/Sbfun2XzfbiAjsLDqmNFX6lGPA+Z5fKF2ZgfA4kF6qTGot+j2EuUKFVhDEChjfonnPOOTR37lzq1KmTMp43995000104okn0jPPPKPlkBeJ\/CbfEjBeXWAHAeNFlHhfptTOVapfEDDetwm3a5Aai36PYW7zEq88YwQMb+jlmZann36apk+fTnv37iWelbnuuuto\/PjxdPzxx\/uFSZN6\/Cbf6xt4IWACC6UGFfPJo1hPBsSzTmrnKtUvCBgz2pkTK6TGot9jmBPMU0lrjICxnHjuuefoN7\/5jZp1+cMf\/kA9evRIxT9X8vpNvtenjyBgXAmLlAqp272DPp4xmNpfPpPaF83QKktq5yrVLwgYrbA2KpHUWPR7DPOLVGMEDF9it2rVKjULw0eqP\/roI\/rggw9ozpw5xKeV7D7OX1JSQuXl5VRXV0e9e\/dWy1GdO3duknXZsmVqT02rVq0ifzv77LNp3bp1Mavxm\/wf3riN6uuJvNq8CwFjF03u\/33\/zKbPQdS+tZ3anjOYMmZu06pQaucq1S8IGK2wNiqR1Fj0ewzzi1RjBMzIkSPp008\/pTvvvJMGDhxIlqDh\/19YWKiETaKvrKyMlixZQitWrKC0tDSaP38+VVZW0vr165tkW7BggRJHLHh0Pr\/JxwyMDivx05jYCUVfVBdtOQSM3BuGIWBSa8dB5Dax73ADB7\/HMDds1inDGAHz29\/+lm699VY6+eSTG9i9b98+ddkdz84k+oqKimj48OGRk0y1tbWUk5NDGzdupO7duzfIOnv2bPr222+1Tzb5TT4EjE7ohkvA8JJRrK9N9iBtZ6V2rlL9goDRDm1jEkqNRb\/HML8IDVTA8FIOPxFg9\/FsjPUeUrw8LFYWLVpEeXl5keIKCgpo8uTJakkq+rvxxhvpww8\/pH\/+85\/q7aUzzzyTpk2bRtnZ2TFN8Zt8CBi7iEj8d6mdEPxKLS6CyA3OgkA9+Tql8uX3GJY8A85yBipgfvazn6nlHj4qzXtQEn179uyhu+++Wy0z8axK4483+65evZpyc3MjfxoyZAgVFxfTqFGjGiTnd5dYuLB44hmfhQsXqhme559\/vskMEGdk8qM\/zjd27FhnSGumfv3vdTThiU9U6tevO10zV3LJeHYr1h6h5EozJxf8MocLHUuk8sW+S\/UNfulEdnBpeIzjH\/vRX3V1dXAGeVRzoALmq6++Upt0H3\/8cbXcw28hsRA59dRT1YwLv0r9t7\/9TT3yuGvXLrryyitp6tSp1Lp16yZwcP7S0lK1f8b6ePMvH8nmPTSJPp7h4fz33XcfnX9+082WfqpX6wkBLy+ws7CQ+msDfnnUW3hUrFS+GC6pvsEvjxqDR8X6OYZ55ELMYgMVMJZFvJzDMyD8qCPf\/3L06FH1pxNOOIHOOOMM9Ygji5dEswWjR48mXjLiO2P443tl+vXrpx6BzMrKauD8G2+8ocqy3l86duyYOrW0ePFiJaIaf36Sby0fQcAk3wyC7lz5xBGfMOIvc9Z2crLPJZHXQfuVPCOJc0r1CwLGq4jxrlypsejnGOYdO01LNkLANDbr8OHD6p8ab+hNBExFRQXx6SI+hZSRkUG8Ubeqqkodq+aP313KzMxU+1yuvvpqatmypVqSatu2rcrHp5U2bdqk7p8JSsD48YBjtG9SG2vQflkCxskJI51GH7RfOjYmk0aqXxAwyURDsHmkxiIEjEdx9cQTT9All1zSQDjwfhcGPNZSUSIzWIjw7b18Aqlv377qHpj09HSVZcSIEWqGZtKkSfT555+rE0j8qjULmZ49e9Itt9zS5LSSVZef5PMMjB+zL+hc3Q3omrXfv9VV99Z2NQPj5uwL+HKXL79Kkzogwi+\/Isidevwcw9yxWK+UwGdgGFheOopeHurWrZt6+8iEW3gZRr\/It95AgoDRC954qYLoXGPd8+LkmQAdj4PwS8euVNNI9QuiM9XI8D+\/1Fj0awzzmzEIGA3E\/SLfr0ccLZelNtYg\/IqegWF8dZ8H0Ai\/SJIg\/HJiX7JppfoFAZNsRASXT2os+jWG+c0cBIwG4n6R\/7OFb9BLew+R169QQ8BokB4niTXT4vbsio5FUjtXqX5BwOhEtVlppMaiX2OY32xCwGgg7hf5fl1gBwGjQToETPIgOcwpddCAgHEYCAYklxqLfo1hflMIAaOBuF\/kQ8BokKGRxMtOCDMwGgQ4TOIlXw5NcT25VN\/gl+uh4mmBfo1hnjoRo3AjBMzPf\/5zOumkkyLmPfroo+rU0CmnnBL5t9tuu81vbCL1+UU+BIw7FHvZuULAuMNRdCle8uW+tc5KlOob\/HIWB0Gn9msM89vPwAXMsGHDtHzesGGDVjovEvlFPgSMO+x51bnyg4wfzxisjMQeGHe44lK84ss9C5MvSapv8Cv5mAgip19jmN++BS5g\/HY4mfr8Ih8CJhl2mubxqnPlk0Y1a2aS2xfU6XrtlV+69XuVTqpfksWZVM6k+uXXGOZVHxGvXAgYDcT9IN8SL2zOwZKm7zFpmOk4idTG6oVf0fe8QMA4DrWEGbzgy10Lky9Nqm\/wK\/mYCCKnH2NYEH5BwGig7jX50U8I+HWJHX4dahDfKAkvIfEMTMbMbc4zu5ADg4YLIPpcBDjzGfAUq5PKl9djWIqwJ50dAkYDOq\/J9\/MF6mh3pTZW+KUR1AYlkcoXfiQYFGSapkiNRa\/HME14XU8GAaMBqdfkQ8BokOAgidROCH45CAJDkoIzQ4jQNEMqX16PYZrwup4MAkYDUq\/Jh4DRIMFBEqmdEPxyEASGJAVnhhChaYZUvrwewzThdT0ZBIwGpH6Q7+cr1JbLUhurG37tn3m+elGavyCOTMcKSzf80gh335NI9QtLSL6HUsoVSo1FP8awlMFPogAIGA3QvCYfMzAaJDhI4kYnFH3qCALGAfhJJHWDrySq9SWLVN\/gly\/h41olXo9hrhnqsCAIGA3AvCbfEjB+PeKIGRh70oO8cTeedRg07HkzLQU4M42RxPZI5cvrMSwoliFgNJD3g3wsIWkQoZkklU4oeubFpOUjLEdokm9YslRi0TBXGpgDv0xmp6ltfoxhQSACAaOButfkz9v0Ps3b9B5tuDaXzu32\/ftPGqallASdUFP4IGBSCqmkMkuNQ4jOpMIh0ExSY9HrMSwo0sQImPr6eiopKaHy8nKqq6uj3r1709y5c6lz584JseU3lq6\/\/nqVr2\/fvjHTek2+JWCwhOROM0ilEzJx6chCJRW\/3EHWm1Kk+gUB4028eFmq1Fj0egzzkpNEZYsRMGVlZbRkyRJasWIFpaWl0fz586myspLWr18f1\/9PPvmELr\/8cjp8+DA98sgjgQkYNpCXkCBg3GkGUjsh+OVOfPhZCjjzE+3U65LKFwRM6rHhaQlFRUU0fPhwGjNmjKqntraWcnJyaOPGjdS9e\/eYdY8bN474Nez77ruP7r333sAEDGZg3A0NqZ0Q\/HI3TvwoDZz5gbJ7dUjlCwLGvRjxpCQWK4sWLaK8vLxI+QUFBTR58mQaOnRokzqXL19OL7zwgpp5yc\/Ph4DxhJVgCpXaCcGvYOIplVrBWSro+Z9XKl8QMP7HkqMae\/ToQatXr6bc3NxIviFDhlBxcTGNGjWqQVkcpL\/61a9o3bp11LFjRy0BE10Az9yMHTvWkX2JEj9ceYgWv3qI7r80jc79UVvXyrUraN++fbZ7hOzKMPHv8MtEVuLbJJUv9liqb\/DL7DbGP9CXLVvWwMjq6mqzjU7COjF7YHgGprS0lAYOHBiBgWdWpk+fToWFhZF\/+\/bbb9W+l4kTJ9LFF1+s\/h0zMElEjsFZpP6Kgl8GB10c08BZuDiTyhdmYAyPw9GjRxMvGY0fP15ZWlNTQ\/369aPNmzdTVlZWxPp33nlHCZiTTjop8m\/79++nU089VeWdMGFCE0+9Jp838PKHTbzuBFmiToifCGj8WU8GZM7aTm2yB7ljhAelSO1cpfrFISDVN\/jlQQP3sEivxzAPTU9YtJgZmIqKClqwYIE6hZSRkUGzZ8+mqqoqdTyavy1btlBmZiZlZ2c3ASToGRhLwBwsaTq4ehkYzbETanzPSzS+pjwZEI\/z5siXl\/HvR9ngzA+U3atDKl8QMO7FiGclsYBZuXKlOoHEd7rwPTDp6emqvhEjRqgZmkmTJkHA\/B8CUhtrIr8sAcOzLdGfyTMvlp3NkS\/POgufCgZnPgHtUjVS+YKAcSlAwliM1+RjBsa9qDD5IrpUvZTauUr1C0tIqUa8\/\/mlxqLXY5j\/TH1Xo5glJC8B9JJ86w4Yth9LSKmxWLd7B308Y7AqxPTloGQ8ldq5SvULAiaZKA82j9RY9HIMC5IxCBgN9L0k\/9IHd9HOqhrfN\/BK61xZvPBm3Jo1M6lFtzzKuvNlDWbDlURq5yrVL2ltLLq1SOVMql9ejmFB9qIQMBroe0W+tXTEJvh9AklS59p4Yy4EjEZQG5RE6qAhqY01DhepnEn1y6sxLOhuBAJGgwEvyH9x7yEatvANVXv+Ge3pqWtyNCxxN4mExhq9bNT2nO+Wj+rGLW1wdN5d1IIrTQJfsdCT6hcETHBtJdmapcaiF2NYshi7mQ8CRgNNL8i39r4EJV7QuWoQb1gSqZ2rVL\/QxgxrQBrmSI1FL8YwDTg9TwIBowGx2+RHLx1BwGgQ4DCJ1E4IfjkMBAOSgzMDSHBgglS+3B7DHEDqaVIIGA143SY\/qJt3G7sqtbHCL42gNiiJVL4wA2NQkGmaIjUW3R7DNOH0PBkEjAbEbpJvnTriav0+Ng0Bo0G2wUmkdq5S\/YKAMbgxxTFNaiy6OYaZxCoEjAYbbpEfvXEXAkYD+DhJrI278d4uktoJwa\/kYyaonOAsKOSTq1cqX26NYcmh6l0uCBgNbN0i3xIwQe57iXY3rI21Zu0sdd8LnzrKmPndQ5gS\/LILxbDy1Vz9wgyMHfPm\/V1qG3NrDDONMQgYDUbcJJ\/3v0DAaIAeJ0lNxWyqKZ+h\/goBkzyOJuWUOmhAwJgUZXq2SI1FN8cwPST9SQUBo4GzW+RbR6cXXZlNl\/dJ06jZ2yRhbKz7Z56vbtxl8ZJ2UwW1bPdDzMB4Gyaelx7GONQFRapv8Es3AsxI59YYZoY331sBAaPBiFvkWwImiFt3Y7kZxk7IWj6Kt\/8Fv3o1AtqwJGGMQ10IpfoGv3QjwIx0bo1hZngDAeOIB7fIN+X4tOV8GDshawYGAsZRCBudOIxxqAuoVN\/gl24EmJHOrTHMDG8gYBzx4Bb5loAJ+vi0BAGT6LVpdK6OwjvwxFL5wmxg4KHl2ACpsejWGOYYUI8zYAlJA2C3yIeA0QDbhSRSOyH45UJw+FwEOPMZ8BSrk8qXW2NYivC6nh0CRgNSN8iPvgMGMzAaoKeQRGonBL9SCIqAsoKzgIBPslqpfLkxhiUJqafZIGA04HWDfEvAmLKBF9PbGsQblkRq5yrVL7QxwxqQhjlSY9GNMUwDPt+TiBEw9fX1VFJSQuXl5VRXV0e9e\/emuXPnUufOnZuA+tVXX6m\/Pf300\/T111\/TWWedRbfeeiv17NkzJgFukG8tH5lyB4xpnWt1UYuEwZ9oz0vjjFI7Ifjle\/+YcoXgLGUIfS1AKl9ujGG+EqFZmRgBU1ZWRkuWLKEVK1ZQWloazZ8\/nyorK2n9+vVNoJgzZw69+eabtGjRImrXrh3dfffdtGHDBtq5c6cnAiZ6+ejff5JJJUU9NOnxNpkpjdV6GiCRtxAwRKbw5XZUSvXLtB8JbvImlTOpfkHAuBn9HpRVVFREw4cPpzFjxqjSa2trKScnhzZu3Ejdu3dvUOOOHTvUzEy3bt3Uv7\/zzjtUWFhIe\/bsodatWzexLlXyTXtCwHLQlMZqCZh4N+s6DRdT\/HJqt116+GWHkHl\/B2fmcZLIIql8pTqGmcqimBkYFis8o5KXlxfBuqCggCZPnkxDhw6Ni39NTY1aTjp48KCawYn1MfnR37hx42js2LHanPYpfV+l\/XHnNrR4RLp2Pq8T7tu3L+YSm9f1el0+\/PIaYXfLl8oXoyTVN\/jlbhtwu7Tly5fTsmXLGhRbXV3tdjWBlydGwPTo0YNWr15Nubm5EVCHDBlCxcXFNGrUqJhAjxw5knbt2qXyLFy4kNLTY4uLVNSrdfsuG2DK6SMLDKm\/NuBX4P2KIwOk8sUgSPUNfjkK8cATpzKGBW58AgPECBiegSktLaWBAwdG3M3Pz6fp06er5aF435dffklr1qyhxYsX03PPPUft27dvkjQV8k17PiDauaA6IX4OgL\/2Rd89yuj2F5RfbvvRuDz45TXC7pcPztzH1MsSpfKVyhjmJd6pli1GwIwePZp4yWj8+PEKE14a6tevH23evJmysrIa4PTEE0+ov0WfUOrVq5cSQIMGDXJVwFj7X24d2o3+q6BLqny5mj+Ixhp92sjJxlwnjgfhlxP7kk0Lv5JFLrh84Cw47JOpWSpfEDDJRIOPeSoqKmjBggXqFFJGRgbNnj2bqqqq1LFq\/rZs2UKZmZmUnZ1NV111FbVs2VIdu+ZTSOvWraNp06bR1q1bVZrGXyrkYwbmezSthxh5s26bcwZjBsZh+5DauUr1i+mV6hv8cth4A06eyhgWsOkJqxczA8NesoBZuXKlOoHUt29ftTnX2tcyYsQINUMzadIkOnDgAN1+++308ssvE98J06VLF7XZ94ILLogJVirkm\/aAY7SDfnZC1iOMXH\/7y2d6Jl4waJjc3cS2zc849Bsdqb7BL78jKbX6UhnDUqvZ29yiBIxXUCVLfvT9L+Xje9FFZ\/\/QKxOTKtevTqjxPS+d579BrU7PScpmnUx++aVji5tp4JebaPpTFjjzB2e3apHKV7JjmFu4elUOBIwGssmSb+r9L5bLfjXW6HteePalTXbTfUYaNGgn8csvbYNcSgi\/XALSx2LAmY9gu1CVVL6SHcNcgNTTIiBgNOBNhXxeQjLp+YBod6U2VvilEdQGJZHKF0Ms1Tf4ZVAD0jAllTFMo\/jAkkDAaECfLPnWBl4IGA2QXUyCztVFMH0oSipfEDA+BI\/LVUiNxWTHMJfhdb04CBgNSJMl3+QTSOhcNYg3LInUzlWqX2hjhjUgDXOkxmKyY5gGZIEmgYDRgD9Z8k0+geR158r3vXh1z4sdZVI7Ifhlx7x5fwdn5nGSyCKpfCU7hpnOHgSMBkPJkB99Asm0JwQsl71qrNZ9L14fl45HnVd+aYSKp0ngl6fwelI4OPMEVs8KlcpXMmOYZyC7WDAEjAaYyZBv+gkkL2dgIGA0giqJJFI7V6l+ednGkggfV7NI5UyqX8mMYa4GjEeFQcBoAJsM+aZv4PWyc7UurcMMjEZwOUgitXOV6peXbcxB2HiSVCpnUv1KZgzzJHBcLhQCRgPQZMg3fS4fzh4AABpNSURBVAOvl52rJWCwB0YjuBwkkdq5SvXLyzbmIGw8SSqVM6l+JTOGeRI4LhcKAaMBaDLkm76B18vO1XqwMXPWds8vrYtFn9ROCH5pNFbDkoAzwwixMUcqX8mMYWFgDgJGgyWn5Edv4L1lSFeafOGPNGrxP4kXjdXa\/8LeYAbGXU694MtdC5MrTapfXv5ISA5p93JJ5UyqX07HMPcixduSIGA08HVKfhg28HrVuQa9gdcrvzTCxPMkUjtXqX4hFj1vEq5XIDUWnY5hrgPrUYEQMBrAOiXfWj4y9QZey2WpjRV+aQS1QUmk8gUBY1CQaZoiNRadjmGacAWeDAJGgwKn5FsCZs5l3enX53XWqCGYJFIbK\/wKJp6SrVUqXxAwyUZEcPmkxqLTMSw4BpzVDAGjgZdT8i0BY+oFdm7PwPBr016\/MK1BUySJ1E4IfjmJAjPSgjMzeNC1QipfTscwXbyCTgcBo8GAU\/Kbk4AJ+sh0LPqkdkLwS6OxGpYEnBlGiI05UvlyOoaFhTUIGA2mnJBv3f\/CxTaHGRgIGI0AcimJ1M5Vql9YQnIp8H0sRmosOhnDfIQ75arECJj6+noqKSmh8vJyqquro969e9PcuXOpc+eme1C++eYbuuuuu+jJJ59Uabt160YzZsxQeWJ9TsgPwwV2bi4hWQImqDtfMAOTch8QeAFSBw0ImMBDy7EBUmPRyRjmGLQAM4gRMGVlZbRkyRJasWIFpaWl0fz586myspLWr1\/fBN4HHniAnnnmGXrssceoQ4cOtGDBAqqoqKAXX3wRAsZBMFrihbMEdecLBIwDwgxNKnXQgIAxNOASmCU1FiFgDI\/FoqIiGj58OI0ZM0ZZWltbSzk5ObRx40bq3r17A+t37txJJ510UmTGhYP2ggsuoN27d1Pbtm2beOqE\/OY0A2Pi7AsGDcMbagzzpA4aiEXEoikIOBnDTLFZxw4xMzAsVhYtWkR5eXkRvwsKCmjy5Mk0dOjQhFhwvq1bt9KaNWtSnoE549addPDIUZp6SRZNveR0HQ4CS5PqwAEB4y91qfLlr7X6tUn1CwJGPwZMSSk1FiFgTImwOHb06NGDVq9eTbm5uZEUQ4YMoeLiYho1alRc63mJ6c4776THH3+csrKy4gqY6D+MGzeOxo4dGzNtn9L31b+\/fp3Z4oVt3LdvX8w9QtpUL\/4l1e99hVr8+nGirv21s3mdMGW\/vDYwyfLhV5LABZgNnAUIfhJVS+Fr+fLltGzZsgYIVFdXJ4GI2VlEzcCUlpbSwIEDI4jn5+fT9OnTqbCwsAkLx44do7vvvpu2bdtGixcvpi5dusRlyol6DcsRavw6NLthxrJO6q9DqX6hjaGNmYKAkzHMFJt17BAjYEaPHk28ZDR+\/Hjld01NDfXr1482b97cZGaFTyxNmTKFPv\/8c2LR065du4RY6ZL\/5VffUpeb\/6TKMv0ItZPOlV+XNmmTrl1gSx0Q4Zcd8+b9HZyZx0kii6TypTuGhYstIjEChk8R8WkiPoWUkZFBs2fPpqqqKnWsmr8tW7ZQZmYmZWdn06pVq9R\/1q1bRyeccIItZ7rkX\/rgLtpZVSNSwLS\/fCa1L5phi5UJCaR2QvDLhOhyZgM4c4ZX0Kml8qU7hgWNv9P6xQgYdpwFzMqVK9UJpL59+6p7YNLT0xUmI0aMUDM0kyZNIt4b8+6779Jxxx3XAC8+it2nT58mGOqSbwmYDdfm0rndTnHKhe\/pYzVWfhag8ffxjMHqn8IyCyO1E4JfvjeRlCsEZylD6GsBUvnSHcN8BduFykQJGBfwiFmEDvnRsy9hFTDR97rEAgICxqsI0ytXaucq1S9mVapv8EuvzZqSSmcMM8VWJ3ZAwGigpUO+JWDyz2hPT12To1Fq8EnidUIsZBp\/GbduIWo0YxW8B7EtQOdqKjPNiy8ImHDFoWS+dMaw8LElaA+Ml+DrkG9dYBeW2RfJjRUCxsvW4H7ZUvlCG3M\/VrwuUWos6oxhXmPrRfmYgdFAVYf8MB2ftlyW2ljhl0ZQG5REKl8QMAYFmaYpUmNRZwzThMioZBAwGnTokA8BowGkT0mkdkLwy6cAcrEacOYimD4UJZUvnTHMB3hdrwICRgNSHfJDKWB+9xNq06ZNBIGMmds00DA\/idROCH6ZH3uNLQRn4eJMKl86Y1i4mPrOWggYDdbsyH9x7yEatvANVVIYLrCzXOYL6qK\/sJwysqNMaicEv+yYN+\/v4Mw8ThJZJJUvuzEsXCx9by0EjAZzduRbAiZMJ5DYbUvAZM7arlBokz1IAw3zk0jthOCX+bGHGZjwcRRtsdQ2ZjeGhZU1CBgN5uzID5uAkTrzYlEptROCXxqN1bAk4MwwQmzMkcqX3RgWLpYwA+OILzvyIWAcwel5YqmdEPzyPHRcrwCcuQ6ppwVK5ctuDPMUVA8LxwyMBrh25IdNwGCmQoN0A5NI7Vyl+sUhJNU3+GVgB5HAJLsxLFzeYAbGEV925FsnkMK2BwadkKMwCDwx+AqcAscGgDPHkAWaQSpfdmNYoKCnUDlmYDTAsyPf1CPUNWtnUc2amRTvJWmpjRV+aQS1QUmk8oUZGIOCTNMUqbFoN4ZpwmNcMggYDUoSkV9PRD+c\/N39KSYcoWbRYn11b22n2re2Q8BocByGJFI7V6l+QcCEoVU1tFFqLELAhC8WXbM4EfnW7IsJAqZu9w76eMbgJn7Hu99FamOFX66Fvi8FSeULAsaX8HG1EqmxCAHjapiEqzAdAWPC7IslYNqeM5janPOdkGlfNCMu2FIbK\/wKV\/uSyhcETLjiUDJfEDDhi0XXLA6LgGGH9888X\/mt8yyA1IEDfrkW+r4UJJUvyQOiVM6k+gUB40tXZmYlYRIwjOAndxRS+vRnbcGU2ljhly31RiWQyhcEjFFhpmWM1FiEgNGiX2aisAkYXRakNlb4pRsBZqSTyhcEjBnx5cQKqbEIAeMkCgJIW19fTyUlJVReXk51dXXUu3dvmjt3LnXu3DmmNQcOHKBp06bR888\/T3v27KHWrVvHtToe+XVHj1Hm1B0qnwl7YJzCLrWxwi+nkRBseql8QcAEG1fJ1C41FiFgkokGH\/OUlZXRkiVLaMWKFZSWlkbz58+nyspKWr9+fRMrPvroIxo3bhyNHDlSiZ5kBcy8Te\/TvE3vQcD4yLNOVVI7Ifilw75ZacCZWXzYWSOVLwgYO+YD\/ntRURENHz6cxowZoyypra2lnJwc2rhxI3Xv3r2BdZ988gkdO3ZMpbnoootSFjBTL8miqZecHhgCfPqIP6evSUttrPArsFBMqmKpfGEGJqlwCDST1FiEgAk0rOwrZ7GyaNEiysvLiyQuKCigyZMn09ChQ2MWsHfvXm0BE10Az96MHTuW+pS+r\/55Yv9TaEK\/U+yN9CBF\/ZSukVJb3FXtqIZ9+\/bFXWJzVJBhieGXYYTYmCOVL3Zbqm\/wy+w2tnz5clq2bFkDI6urnY0PZnv4nXVibuLt0aMHrV69mnJzcyO4DxkyhIqLi2nUqFEpC5jG5FvLRy1aEP3jnu+OLvv9WU8FcL1894vO0eloG6X+2oBffkdiavVJ5YtRkeob\/Eot5v3OjRkYvxF3WB\/PwJSWltLAgQMjOfPz82n69OlUWFjomYAJcvnI7q0jOwjRCdkhZNbfwZdZfOhYA850UDInjVS+IGDMibGYlowePZp4yWj8+PHq7zU1NdSvXz\/avHkzZWVluS5gLn1wF+2sqqG\/3zmI2rZqGQg6loBJv3kj\/cv\/i71MlsgwqY0VfgUSjklXKpUvzMAkHRKBZZQaixAwgYWUXsUVFRW0YMECdQopIyODZs+eTVVVVepYNX9btmyhzMxMys7OjhToZA9M4yUkS8BsuDaXzu0WzP4XzMDEjg2pnRD80usLTEoFzkxiw94WqXxBwNhzH3gKFjArV65Up4v69u2r7oFJT09Xdo0YMULN0EyaNIkeeughJXb47pijR49Sq1atVJqlS5fSgAEDmvjRmPzoBxwhYAKnvYkBUjsh+GVerNlZBM7sEDLr71L5goAxK858tSaa\/Bf3HqJhC9+I1B\/GC+ws46U2Vvjla\/NIuTKpfGEJKeXQ8L0AqbEIAeN7KJlTYSwBk39Ge3rqmhxzjEzCEqmNFX4lEQwBZpHKFwRMgEGVZNVSYxECJsmAkJAt1hISBIy5zErthOCXuTEXzzJwFi7OpPIFAROuOHTV2mjyrftfhvTsSCuLe7paj9+FSW2s8MvvSEqtPql8YQYmtbgIIrfUWISACSKaDKkzloAJ8v4Xt2CR2ljhl1sR4k85UvmCgPEnftysRWosQsC4GSUhKwsCJlyESe2E4Fe44hACBnyZggAEjClMBGCHaQIm1ftfLAgxIAYQTClUCb5SAC+grOAsIOCTrFYqXxAwSQaEhGwW+b9ZtYfK\/\/yJcinIJSR+ffrjGYOp\/eUzqX3RjKQhltpY4VfSIRFIRql8YQYmkHBKqVKpsQgBk1JYhDuzRX70BXZB3v+yf+b5VPvWdgiYOGEltROCX+HrR8BZuDiTyhcETLji0FVrGwuYIG\/fZccsAZM5azu1yR6UtK9SGyv8SjokAskolS\/MwAQSTilVKjUWIWBSCotwZ24sYIKaffnyTyvps9JfRcCEgIkdV1I7IfgVvn4EnIWLM6l8QcCEKw5dtZbJn\/jQVpq36T1VblACxpp5sZzrurY+JT+lNlb4lVJY+J5ZKl+YgfE9lFKuUGosQsCkHBrhLSBawAS5edc6fZTqzIvFhNTGCr\/C1dak8gUBE644lMwXBEz4YtE1i5n8njc9QTuragI9feSaQ\/9XkNSBA365HSnelieVL8kDolTOpPoFAeNtH2Z06dECJqjlIy8AktpY4ZcX0eJdmVL5goDxLma8KllqLELAeBUxISg3WsD4eQKJl4zanjM4pZNGieCV2ljhVwgaVZSJUvmCgAlXHErmCwImfLHomsVBCJhD6+bSwVXTlA+pbtaNB4TUgQN+uRb6vhQklS\/JA6JUzqT6BQHjS1dmZiVBCBhrwy7PwGTM3OYJMLNmzaIZM5K\/ydcTo1woFH65AKKPRUjliyGU6hv88rGBuFAVBIwLIJpSRH19PZWUlFB5eTnV1dVR7969ae7cudS5c+eYJjL5hy77vfqbH0tI\/\/j9dfS\/zz2g6kv1uYBEmEsNavhlSkvTs0MqX2r2tGtXqq6u1gMiRKngV4jIEhyHLep5NG9mX1lZGS1ZsoRWrFhBaWlpNH\/+fKqsrKT169fbChg\/NvFWF7WI2OHV8hE61\/AFPQYNcGYKAohFU5jQs0MqX81SwBQVFdHw4cNpzJgxiv3a2lrKycmhjRs3Uvfu3ZtERPQMjJ8CxkvxAgGj1\/BNSiW1E5LqF9qYSa1HzxapsSjVr2YpYFisLFq0iPLy8iJRXVBQQJMnT6ahQ4c2ifRf\/vKX9Fz6ePXvp6y\/Sq8lpJBqS5\/vbvy98PWsFEpBViAABIAAEAACpMa6VatWiYOiWQqYHj160OrVqyk3NzdC6JAhQ6i4uJhGjRoljmQ4BASAABAAAkBAGgLNUsDwDExpaSkNHDgwwmd+fj5Nnz6dCgsLpXEMf4AAEAACQAAIiEOgWQqY0aNHEy8ZjR\/\/3bJQTU0N9evXjzZv3kxZWVi2ERflcAgIAAEgAATEIdAsBUxFRQUtWLBAnULKyMig2bNnU1VVlTpWjQ8IAAEgAASAABAwH4FmKWCYFhYwK1euVCeQ+vbtq+6BSU9PN58xWAgEgAAQAAJAAAhQsxUw4B4IAAEgAASAABAILwIQMOHlDpYDASAABIAAEGi2CEDANFvq4TgQAAJAAAgAgfAiAAEThzs+mXTrrbfSn\/70J2rZsiX99Kc\/VQ+ztW7dOnRsHzhwgKZNm0bPP\/887dmzp4EP\/P\/5+Pi7775LJ554ojqZddVV3l\/W5waI33zzDd1111305JNPqjetunXrph6n5Let+Aurb1999ZXak\/X000\/T119\/TWeddZaKxZ49e4bar2jON2zYQNdff73aOM970MLM17Jly1Tf0KpVq4iLZ599Nq1bty7UflnObN26lebMmUOffvopnXbaaXTbbbdFLgENaxsbMWKE6h+iP25rVjyG1a+jR48qrnbs2EEtWrSgjh07Kr6ys7NFxGLjcQMCJs5Iet111xEPJPzoIz8XxQN7r169lBAI0\/fRRx\/RuHHjaOTIkcqXaAHDDfb8889XTyqwfx988AHxEfN58+bRBRdcYLybDzzwAD3zzDP02GOPUYcOHdTGbD5h9uKLL6qBP6y+cQf05ptvqtui27VrR3fffTfxgL9z585Q+2UF1CeffEKXX345HT58mB555BElYMLMF8cdtx1uX42\/MPvFvvBDlJdddlnk5vK1a9eqN+P4VlceLMPaxhrz9Nprr6m+nX808BdWv7hPZPGyfPlyatu2rfrvxYsXh75PjDcYQcDEQIZPJvGveA5m622kl156Sf1i5EAP08eDxbFjx9Rpq4suuqiBgOEBkZ9PePXVV5Va5++ee+6h9957j7ghmP6x\/SeddFJkxoXtZuG1e\/duev3110PrG3dA\/DI6zyjx984776gLFll8cvyFmTP2hwX1sGHD6L777qN7771XCZgwxyJfw\/Dtt9+qWZjGX5j9Yl9YTLPQvPPOO8X5ZjnEIpNvYr\/jjjuof\/\/+oY7FSZMmUZcuXWjKlCnKPe4T+c6zXbt20V\/+8pfQ9x2YgdEYlXmg+NnPfqaWVXj5iL\/PPvtMTZv++c9\/plNPPVWjFLOS7N27t4mA4anvZ599Vj2rYH1PPfWUEi98qV\/YPp6x4OnuNWvWkBTfeCmTl5MOHjyoXlAPu1\/8i\/CFF15QMy98+7UlYMLs14033kgffvgh\/fOf\/yRerj3zzDPVr3metg+zX9z+eUaWZ55ZRHMf0qlTJ+Ub\/8ALu29W\/\/bQQw+pHz18Ozt\/YfaLl8B4RprvOONZae7LeesAL2eG2S\/MwDgYjfnXOz\/gyI3W+r744gvVaHlPDP86DtsXS8A8+OCDavaFA9v6WLjwL0r+5Rimj6e1+Vfi448\/rm5TluAbL\/vxLyd+s2vhwoXqnqIw+8W\/Bn\/1q1+pzpTX5qMFTJj9YlHGwoVnlk4++WTFFS+x8MDBd02FuY3xzAQvFT366KPq0s+lS5fSww8\/TNu2bVPLE2H2jfu3I0eO0ODBg9WPHusW9jDHIm934O0PmzZtUnsa27RpozhjMR1mvyBgHIzGPAPDr1K\/\/fbbkY15+\/bto\/POO08tTbRv395BaWYkjTcDwxtgn3jiiYiRvIeEf+k\/99xzZhhuYwUvj\/EeEe5Qea2Xp0+tX1Fh9439+PLLL1Xnyr4xJzxDFka\/eImF971MnDiRLr74YsVR4xmYMPoVKzx5EOH31niJjGdmwuzXFVdcoZZVbrjhhoir\/EOOZy14hjrMvrFDLMK2bNmi\/tv6+AddWP3iH5+875G3ArCY5hl26wAH76MLq18QMA6GY968y9Omf\/jDHyInPzjIORAqKysdlGRO0lgC5pVXXlEDCouy448\/Xhl7++23q1+T999\/vznGx7GEBwpe6\/3888\/V9C9veLW+MPvGgpLf5oqe6eN4ZB\/5FFwYOePZTBYwvGfJ+vbv36+WY60N8mH0i3154403FFc8q8Qfi2oe5Fl08hJ0WP1iX\/iE4nHHHadmZa2PfeNZGG5\/YfaN\/Rk7dixdeOGF6r8l9B38o+Dmm29WP8CtL+x9R6KBCJt446Bz00030aFDh9TJAhY03Mny69W83h3GL5aA4WPIvLH35z\/\/ueqIeJDhKX4eKLkhmP7xND3\/h5ckTjjhhAbmhtk3PsbOAx\/HHosy9o\/FM+\/v+dd\/\/ddQcxZNUvQMTJj5uvrqqxVfPBPIJz\/4VBIvafI0PgvOMLcxPg3HpxR5KYyP8fP+ChZmPOPJbS7MvnEs8hUFvOzMy7TWF+ZYnDBhgoo5noHhY\/18+KS4uFgtZ6alpYWer8ZjEgRMnFGad97fcssttH37dtU58VFC\/v\/WTIXpg7tlH0\/1cofKv5Z4Ldu6q4LXRQcMGKCmgXlw\/Otf\/6p+DV977bVq\/08YPl6fZ\/v5F2L0V1ZWRn369AmtbzwDxjNhL7\/8shLPvCzGJ4+so+1h5iyegOF\/D6tfPAPIJ5B43xj3FTzQc19hnWAMq18WV7wxlH\/U8D7AM844Q\/lq3UkUZt94gzz3E7ypnDcnR39h9Yv7Dp4t471zPFbxPhhe\/uNZpjC3sXjjEQRMGEZq2AgEgAAQAAJAAAg0QAACBgEBBIAAEAACQAAIhA4BCJjQUQaDgQAQAAJAAAgAAQgYxAAQAAJAAAgAASAQOgQgYEJHGQwGAkAACAABIAAEIGAQA0AACAABIAAEgEDoEICACR1lMBgIAAEgAASAABCAgEEMAAEgAASAABAAAqFDAAImdJTBYCAABIAAEAACQAACBjEABIAAEAACQAAIhA4BCJjQUQaDgUD4EaiuriZ+CoLf1Rk0aJByiF9v5ocs+QVdvgIdHxAAAkAgEQIQMIgPIAAEAkHg97\/\/PS1btkw9evjpp59SYWGheiywf\/\/+gdiDSoEAEAgXAhAw4eIL1gIBMQgcO3aMrrjiCvWg3ltvvaUeCrz11lvF+AdHgAAQ8BYBCBhv8UXpQAAIJEDg\/fffV0tJ6enpaumodevWwAsIAAEgoIUABIwWTEgEBICAFwhUVlbSVVddRS1btlQCJjMz04tqUCYQAAICEYCAEUgqXAICYUDgyJEjavZlwoQJ9Oabb9L+\/fvVnhh8QAAIAAEdBCBgdFBCGiAABFxHYObMmbRnzx5avXo1ffHFF1RQUEA33HADXXnlla7XhQKBABCQhwAEjDxO4REQMB6BV155hf793\/9dLRtlZWUpe5955hmaMmWK+rfTTjvNeB9gIBAAAsEiAAETLP6oHQgAASAABIAAEEgCAQiYJEBDFiAABIAAEAACQCBYBCBggsUftQMBIAAEgAAQAAJJIPD\/AbtFRZJZm6+oAAAAAElFTkSuQmCC","height":337,"width":560}}
%---
%[output:2049e81f]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"Unrecognized function or variable 'err_music'."}}
%---
