%[text] ## Evaluation of DQN vs Brute Force Localization Precision (Episodic Navigation)
%% Evaluation of DQN vs Brute Force Localization Precision on Random Positions (CDF)
% Executed until Done/Terminate conditions are met
clc; clear all; close all;

fprintf('=== Initializing Random Deployment Environment ===\n'); %[output:2a25c3ad]
% 1. Add required codebase folders to path
addingPathParentFolderByName('code');

% 2. Load Parameters
Parameters;  %[output:5a0597c6] %[output:79ace775] %[output:1cd895fd] %[output:1912f58b] %[output:2928e912] %[output:702a069b] %[output:7aa31890] %[output:5759a5d2] %[output:455a99a7] %[output:1a2bfe05] %[output:6b43b9a7] %[output:66d2d09a]

% 3. Run Calibration (Requires Parameters to be in the workspace)
Calibration; %[output:3a1c9308]

% 4. Load the trained DQN agent
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_144_neurons_1_relu.mat');
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_4_hidden_916_phases_CST_Nx_4_Mx_25_Reward_Mixed.mat');
% agent_path = fullfile('..', 'Dataset', 'A_Training_noCST_FC_2_ReLU_2_Neurons_128_Nx_4_Reward_Mixed.mat');
% agent_path = fullfile('..', 'Dataset', 'A_Training_noCST_FC_2_ReLU_1_Neurons_128_Nx_4_Reward_Mixed.mat');
% agent_path = fullfile('..', 'Dataset', 'A_Training_noCST_FC_1_ReLU_1_Neurons_128_Nx_4_Reward_Mixed.mat');
% agent_path = fullfile('..', 'Dataset', 'A_Training_noCST_FC_1_ReLU_1_Neurons_64_Nx_4_Reward_Mixed.mat');
% agent_path = fullfile('..', 'Dataset', 'A_Training_noCST_FC_1_ReLU_1_Neurons_64_Nx_4_Reward_Mixed_no_bias.mat');
% agent_path = fullfile('..', 'Dataset', 'A_Training_noCST_FC_1_ReLU_1_end_Neurons_8_Nx_4_Reward_Mixed_no_bias.mat');
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_4_layers_Nx_4_Mx_12_Aligned.mat');
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_4_layers_Nx_4_Mx_4_Aligned.mat');
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_Nx_4_Mx_5_Aligned.mat');
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Aligned.mat') %[output:01d3f9e8]


if isfile(agent_path) %[output:group:804547fe]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:594fc1ea]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:804547fe]

%%
%[text] #### Training results
S = load(agent_path, 'trainingStats');
trainingStats = S.trainingStats;

% Pull the relevant series, defending against missing fields
epIdx       = trainingStats.EpisodeIndex(:);
epReward    = trainingStats.EpisodeReward(:);
epAvgReward = trainingStats.AverageReward(:);
epSteps     = trainingStats.EpisodeSteps(:);

if isfield(trainingStats,'TotalAgentSteps') && ~isempty(trainingStats.TotalAgentSteps)
    totalAgentSteps = trainingStats.TotalAgentSteps(:);
else
    % Fallback: cumulative episode lengths
    totalAgentSteps = cumsum(epSteps);
end

if isfield(trainingStats,'EpisodeQ0') && ~isempty(trainingStats.EpisodeQ0)
    epQ0 = trainingStats.EpisodeQ0(:);
    haveQ0 = true;
else
    epQ0 = nan(size(epIdx));
    haveQ0 = false;
end

% %% Figure 1 — training quality vs episode
% figure('Name','Training quality (per episode)','Position',[100 100 1100 700]);
% 
% plot(epIdx, epReward,    'Color',[0.6 0.6 0.6], 'LineWidth',0.8); hold on;
% plot(epIdx, epAvgReward, 'b-',  'LineWidth',2);
% xlabel('Episode','Interpreter','latex');
% ylabel('Reward','Interpreter','latex');
% title('Episode reward','Interpreter','latex');
% legend({'Per-episode','Moving average'},'Location','best','Interpreter','latex');
% grid on; set(gca,'FontSize',font);


%% Figure 1 — Enhanced Training Quality vs Episode
fig = figure('Name','Training quality (per episode)','Position',[100 100 1000 600]); %[output:37ee272b]

% --- Styling Configurations ---
% Modern professional color palette (Deep Blue and Muted Cyan/Gray)
mainColor = [0.00, 0.45, 0.74];       % Deep corporate blue for the average line
shadeColor = [0.30, 0.75, 0.93];      % Light sky blue for the deviation band
rawColor = [0.70, 0.70, 0.70];        % Soft gray for the raw background dots
alphaValue = 0.3;                     % Transparency for the shaded area

% --- Calculate Deviation Band ---
% Calculate the standard deviation using a moving window (e.g., 20 episodes)
windowSize = 20; 
movingStd = movstd(epReward, windowSize);

% Define upper and lower bounds for the shaded area
upperBound = epAvgReward + movingStd;
lowerBound = epAvgReward - movingStd;

% Clean up bounds to ensure they align cleanly as columns for the fill function
xPoints = [epIdx; flipud(epIdx)];
yPoints = [upperBound; flipud(lowerBound)];

% --- Plotting ---
% Plot raw data as subtle background dots so it doesn't clutter the view
hRaw = plot(epIdx, epReward, '.', 'Color', rawColor, 'MarkerSize', 6);  %[output:37ee272b]
hold on; %[output:37ee272b]
% Plot the shaded deviation area
hShade = fill(xPoints, yPoints, shadeColor, ... %[output:37ee272b]
    'EdgeColor', 'none', 'FaceAlpha', alphaValue); %[output:37ee272b]
% Plot the crisp moving average line on top
hAvg = plot(epIdx, epAvgReward, 'Color', mainColor, 'LineWidth', 1.5); %[output:37ee272b]
% --- Aesthetics & Formatting ---
grid on; %[output:37ee272b]
% set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'Layer', 'top');
set(gca, 'Box', 'on', 'TickDir', 'out', 'LineWidth', 1); %[output:37ee272b]
set(gca, 'FontName', 'Helvetica', 'FontSize', font); % Fallback if 'font' variable isn't defined %[output:37ee272b]
xlabel('Episode', 'Interpreter', 'latex', 'FontSize', font); %[output:37ee272b]
ylabel('Reward', 'Interpreter', 'latex', 'FontSize', font); %[output:37ee272b]
% title('\textbf{DQN Agent Training Progress}', 'Interpreter', 'latex', 'FontSize', 16);
legend([hRaw, hAvg, hShade], {'Raw Episode Reward', 'Moving Average ($\mu$)', 'Deviation ($\mu \pm \sigma$)'}, ... %[output:group:7256a28f] %[output:37ee272b]
    'Location', 'best', 'Interpreter', 'latex', 'FontSize', font); %[output:group:7256a28f] %[output:37ee272b]


%% Numerical summary
nEp = numel(epReward);
[bestR, bestEp] = max(epReward);
fprintf('\n=== Training summary ===\n'); %[output:39ad2195]
fprintf('Episodes completed   : %d\n', nEp); %[output:4c12ad7e]
fprintf('Total agent steps    : %d\n', totalAgentSteps(end)); %[output:84a9ec2f]
fprintf('Final episode reward : %.3f\n', epReward(end)); %[output:506d6127]
fprintf('Final avg reward     : %.3f\n', epAvgReward(end)); %[output:3f385c4c]
fprintf('Best episode reward  : %.3f (episode %d)\n', bestR, bestEp); %[output:6f6a62bc]
fprintf('Mean steps/episode   : %.1f\n', mean(epSteps)); %[output:3f138209]
fprintf('========================\n\n'); %[output:37083b11]

%%
%% ========================================================================
%  BRUTE-FORCE POSITIONING CDF VERSUS POLARIZATION ROTATION
%  ========================================================================

% Rotation angles to evaluate.
% Linear-polarization orientation is periodic over 180 degrees.
rotation_deg = [0 15 30 45 60 75 90];
rotation_deg = [0 90];

N_angles = numel(rotation_deg);
N_eval   = 10;

% true  -> include polarization-induced SNR loss
% false -> useful only as a noiseless regression/debugging test
add_awgn = false;

% "relative":
%   Applies g_s(psi)/g_s(0) to your existing nominal U_func response.
%   Therefore psi = 0 degrees reproduces your existing model exactly.
%
% "absolute":
%   Replaces the nominal coefficient with the absolute CST coefficient
%   g_s(psi), including nominal insertion loss and phase.
orientation_model = 'relative';

% Choose which nominal first-layer model is used.
%
% false -> preserve the ideal EnvPars.U_func baseline
% true  -> preserve the existing CST amplitude-aware U_func_CST baseline
use_cst_nominal = false;

% Reproducibility
rng(2026, 'twister');

%% ------------------------------------------------------------------------
% Load Jones/state data
% -------------------------------------------------------------------------

jones_file = fullfile('..', 'Dataset', 'T_jones.mat');

if ~isfile(jones_file)
    error('Jones data file not found: %s', jones_file);
end

J = load(jones_file);

required_fields = {'t_co', 't_cross'};

for f = 1:numel(required_fields)
    if ~isfield(J, required_fields{f})
        error('T_jones.mat must contain the variable "%s".', ...
            required_fields{f});
    end
end

t_co    = J.t_co(:);
t_cross = J.t_cross(:);

if numel(t_co) ~= numel(t_cross)
    error('t_co and t_cross must contain the same number of states.');
end

% Preferred: explicitly save the nominal state phase.
% Fallback: use the phase of t_co.
if isfield(J, 'phase_state_deg')
    phase_state_deg = J.phase_state_deg(:);
else
    warning(['phase_state_deg was not found. Using angle(t_co) as the ' ...
             'nominal state phase.']);
    phase_state_deg = rad2deg(angle(t_co));
end

if numel(phase_state_deg) ~= numel(t_co)
    error('phase_state_deg, t_co and t_cross must have equal lengths.');
end

% Compute g_s(psi) for every state and every requested angle.
%
% Size:
%   rows    -> physical/CST atom states
%   columns -> rotation angles
g_state_table = ...
    t_co    * cosd(rotation_deg) + ...
    t_cross * sind(rotation_deg);

assert(isequal(size(g_state_table), ...
    [numel(t_co), N_angles]), ...
    'Unexpected size of g_state_table.');

%% ------------------------------------------------------------------------
% Construct the nominal coefficient applied by the existing model
% -------------------------------------------------------------------------

N = EnvPars.N;
T = EnvPars.T;

U_nominal = complex(zeros(N, T));

if use_cst_nominal
    if ~isfield(EnvPars, 'U_func_CST')
        error('EnvPars.U_func_CST is not available.');
    end

    nominal_function = EnvPars.U_func_CST;
else
    nominal_function = EnvPars.U_func;
end

for t_psi = 1:T
    u = nominal_function(1:N, t_psi);
    U_nominal(:, t_psi) = u(:);
end

% Your original operation is diag(v0'), where ' is a conjugate transpose.
% Therefore, the actual coefficient multiplying a_psi_x_y is conj(v0).
V_nominal = conj(U_nominal);

% The state lookup is performed using the phase that is actually applied.
phase_command = mod(angle(V_nominal), 2*pi);

%% ------------------------------------------------------------------------
% Interpolate the complex state responses versus nominal phase
% -------------------------------------------------------------------------

% The 0-degree data are the reference for the relative model.
idx_zero = find(abs(rotation_deg) < 1e-12, 1);

if isempty(idx_zero)
    error('rotation_deg must include 0 degrees for the relative model.');
end

F_g0 = buildComplexPhaseInterpolant( ...
    phase_state_deg, g_state_table(:, idx_zero));

g0_command = F_g0(phase_command);

% Avoid an unstable division if a nominal co-polarized state is effectively
% zero. In that situation, use orientation_model = 'absolute'.
relative_floor = 1e-6 * max(abs(g0_command(:)));

if strcmpi(orientation_model, 'relative') && ...
        any(abs(g0_command(:)) < relative_floor)

    error(['Some nominal co-polarized coefficients are nearly zero. ' ...
           'Use orientation_model = ''absolute'' or inspect the CST data.']);
end

% V_rotation(:, t, k) is the complex coefficient for every atom,
% snapshot, and orientation angle.
V_rotation = complex(zeros(N, T, N_angles));

for k_angle = 1:N_angles

    F_g = buildComplexPhaseInterpolant( ...
        phase_state_deg, g_state_table(:, k_angle));

    g_command = F_g(phase_command);

    switch lower(orientation_model)

        case 'relative'
            % Orientation-only modification.
            %
            % At psi = 0:
            % g_command/g0_command = 1, so the old model is recovered.
            orientation_factor = g_command ./ g0_command;
            V_rotation(:, :, k_angle) = ...
                V_nominal .* orientation_factor;

        case 'absolute'
            % Full CST response, including insertion loss and phase.
            V_rotation(:, :, k_angle) = g_command;

        otherwise
            error('Unknown orientation_model: %s', orientation_model);
    end
end

%% ------------------------------------------------------------------------
% Regression test: psi = 0 must reproduce the nominal model
% -------------------------------------------------------------------------

if strcmpi(orientation_model, 'relative') %[output:group:29264c47]

    relative_error_zero = ...
        norm(V_rotation(:, :, idx_zero) - V_nominal, 'fro') / ...
        max(norm(V_nominal, 'fro'), eps);

    fprintf('Zero-degree coefficient regression error: %.3e\n', ... %[output:0d37e64c]
        relative_error_zero); %[output:0d37e64c]

    assert(relative_error_zero < 1e-10, ...
        'The zero-degree orientation does not reproduce V_nominal.');
end %[output:group:29264c47]

%% ------------------------------------------------------------------------
% Paired Monte Carlo positions
% -------------------------------------------------------------------------

% Use exactly the same positions for every orientation angle.
% This gives paired CDF curves and avoids differences due only to sampling.
target_idx_eval = randi(EnvPars.N_cal, N_eval, 1);

err_bf     = nan(N_eval, N_angles);
outage_bf  = false(N_eval, N_angles);

% Unit-variance circular complex noise:
% E{|noise|^2} = 1 per complex sample.
snr_amplitude = sqrt(db2pow(EnvPars.SNR_dB));

% Used to represent an invalid/nonphysical estimate as a positioning outage.
max_hall_error_cm = ...
    100 * hypot(EnvPars.L_hall, EnvPars.W_hall);

fprintf('\nRunning brute-force evaluation:\n'); %[output:03453dd8]
fprintf('  Episodes: %d\n', N_eval); %[output:7c62b38b]
fprintf('  Angles  : %d\n', N_angles); %[output:9c646e31]
fprintf('  SNR     : %.2f dB\n', EnvPars.SNR_dB); %[output:4a043ad8]
fprintf('  AWGN    : %d\n\n', add_awgn); %[output:842d130e]

for i = 1:N_eval

    if mod(i, 25) == 0
        fprintf('Episode %d / %d\n', i, N_eval);
    end

    target_idx = target_idx_eval(i);

    pos_MU_true = EnvPars.pos_cal(target_idx, :);

    psi_x_true = EnvPars.psi_x_cal(target_idx);
    psi_y_true = EnvPars.psi_y_cal(target_idx);

    % Preserve the steering-vector convention used in your existing code.
    a_psi_x = exp(1i * psi_x_true * ...
        ((1:EnvPars.N_x)-1))';

    a_psi_y = exp(1i * psi_y_true * ...
        ((1:EnvPars.N_y)-1))';

    a_psi_x_y = kron(a_psi_y, a_psi_x);

    % One independent noisy measurement for every phase snapshot.
    %
    % The same noise realization is reused across orientation angles.
    % This is a common-random-numbers comparison: differences between
    % curves are caused by orientation rather than different noise draws.
    if add_awgn
        noise_all = ...
            (randn(N, T) + 1i*randn(N, T)) / sqrt(2);
    else
        noise_all = complex(zeros(N, T));
    end

    for k_angle = 1:N_angles

        % Equivalent to:
        %
        % G * diag(V_rotation(:,t,k_angle)) * a_psi_x_y
        %
        % for every t simultaneously, without constructing 1600 diagonal
        % matrices.
        weighted_input = ...
            V_rotation(:, :, k_angle) .* a_psi_x_y;

        r_all = ...
            snr_amplitude * EnvPars.G * weighted_input + ...
            noise_all;

        % Find the maximum over:
        %   output DFT bin and phase-scan snapshot.
        power_all = abs(r_all).^2;

        [~, global_idx] = max(power_all(:));
        [~, t_best] = ind2sub([N, T], global_idx);

        % Recover the DFT output-bin mapping using your existing convention.
        r_best = r_all(:, t_best);

        R = flipud(fliplr( ...
            reshape(abs(r_best), ...
            [EnvPars.N_x, EnvPars.N_y])))';

        [~, linear_idx] = max(R(:));

        n_psi_x_max = ceil(linear_idx / EnvPars.N_x);
        n_psi_y_max = ...
            linear_idx - ...
            (n_psi_x_max - 1) * EnvPars.N_x;

        t_psi_x_max = EnvPars.t_x(t_best);
        t_psi_y_max = EnvPars.t_y(t_best);

        % Kept equal to your current index-to-angle convention so that the
        % orientation study is directly comparable with your earlier CDF.
        best_psi_x_est = mod( ...
            2*pi * ...
            (n_psi_x_max + t_psi_x_max / EnvPars.T_x) / ...
            EnvPars.N_x, ...
            2*pi);

        best_psi_y_est = mod( ...
            2*pi * ...
            (n_psi_y_max + t_psi_y_max / EnvPars.T_y) / ...
            EnvPars.N_y, ...
            2*pi);

        pos_est_bf = estimatePosFromAngles( ...
            best_psi_x_est, ...
            best_psi_y_est, ...
            EnvPars, ...
            pos_MU_true);

        % Do not silently discard invalid estimates. Count them as outages.
        invalid_estimate = ...
            ~isreal(pos_est_bf) || ...
            any(isnan(pos_est_bf)) || ...
            any(isinf(pos_est_bf));

        if invalid_estimate

            outage_bf(i, k_angle) = true;
            err_bf(i, k_angle) = max_hall_error_cm;

        else

            err_bf(i, k_angle) = ...
                norm(pos_est_bf(1:2) - pos_MU_true(1:2)) * 1e2;
        end
    end
end

%%

%% ------------------------------------------------------------------------
% Plot empirical CDFs
% -------------------------------------------------------------------------

figure('Name', 'Brute-force CDF versus polarization rotation'); %[output:1142eb43]
hold on; %[output:1142eb43]
grid on; %[output:1142eb43]
box on; %[output:1142eb43]

for k_angle = 1:N_angles

    h1 = cdfplot(err_bf(:, k_angle)); %[output:1142eb43]
    set(h1, 'LineWidth', 1.5);
    % [F_cdf, X_cdf] = ecdf(err_bf(:, k_angle));

    % stairs(X_cdf, F_cdf, ...
    %     'LineWidth', 1.6, ...
    %     'DisplayName', ...
    %     sprintf('$\\psi=%g^{\\circ}$', rotation_deg(k_angle)));
end

% Reporting guides
xline(30, ':', '30 cm', ... %[output:1142eb43]
    'HandleVisibility', 'off'); %[output:1142eb43]

yline(0.95, ':', '95\%', ... %[output:1142eb43]
    'HandleVisibility', 'off', ... %[output:1142eb43]
    'Interpreter', 'latex'); %[output:1142eb43]

xlabel('Positioning error [cm]', ... %[output:1142eb43]
    'Interpreter', 'latex', ... %[output:1142eb43]
    'FontSize', 14); %[output:1142eb43]

ylabel('Empirical CDF', ... %[output:1142eb43]
    'Interpreter', 'latex', ... %[output:1142eb43]
    'FontSize', 14); %[output:1142eb43]

legend( ... %[output:1142eb43]
    'Location', 'southeast', ... %[output:1142eb43]
    'Interpreter', 'latex', ... %[output:1142eb43]
    'FontSize', 12); %[output:1142eb43]

set(gca, ... %[output:1142eb43]
    'Box', 'on', ... %[output:1142eb43]
    'TickDir', 'out', ... %[output:1142eb43]
    'LineWidth', 1, ... %[output:1142eb43]
    'FontSize', 12); %[output:1142eb43]

% Keep the same displayed range as your previous figure.
% A CDF that does not reach 1 inside this interval indicates a large-error
% or outage tail.
xlim([0 200]); %[output:1142eb43]

%% ------------------------------------------------------------------------
% Numerical CDF summary
% -------------------------------------------------------------------------

p50_cm = prctile(err_bf, 50, 1).';
p90_cm = prctile(err_bf, 90, 1).';
p95_cm = prctile(err_bf, 95, 1).';

prob_below_30_percent = ...
    100 * mean(err_bf <= 30, 1).';

outage_percent = ...
    100 * mean(outage_bf, 1).';

cdf_summary = table( ...
    rotation_deg(:), ...
    p50_cm, ...
    p90_cm, ...
    p95_cm, ...
    prob_below_30_percent, ...
    outage_percent, ...
    'VariableNames', { ...
        'Rotation_deg', ...
        'P50_cm', ...
        'P90_cm', ...
        'P95_cm', ...
        'ProbabilityBelow30cm_percent', ...
        'Outage_percent'});

disp(cdf_summary); %[output:71af41bb]

%% ------------------------------------------------------------------------
% Save results
% -------------------------------------------------------------------------

save('BF_orientation_CDF_results.mat', ...
    'err_bf', ...
    'outage_bf', ...
    'rotation_deg', ...
    'target_idx_eval', ...
    'cdf_summary', ...
    'orientation_model', ...
    'use_cst_nominal', ...
    'add_awgn');


%%
%[text] ### DQN Agent and Brute Force
%% Evaluation Setup
N_eval = 1000; % Number of episodes (random positions) to evaluate

fprintf('Running %d evaluation episodes...\n', N_eval); %[output:56d44986]
err_dqn = zeros(N_eval, 1);
err_bf  = zeros(N_eval, 1);
steps_taken_dqn = zeros(N_eval, 1);

for i = 1:N_eval %[output:group:9ba08430]
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval); %[output:54b30dce]
    end
    
    % ---------------------------------------------------------
    % DQN AGENT EVALUATION (Episodic Loop)
    % ---------------------------------------------------------
    % Reset environment: automatically picks a random pos_cal position
    % [obs, LoggedSignals] = resetFunction_nav(EnvPars);
    [obs, LoggedSignals] = resetFunction_nav_CST_Reward_Mixed(EnvPars);
    
    % Identify the true target position chosen by the reset function
    target_idx = LoggedSignals.pos_idx;
    pos_MU_true = EnvPars.pos_cal(target_idx, :);
    
    isDone = false;
    step = 0;

    % Let the agent navigate until it triggers the native termination (IsDone)
    while ~isDone
        action_out = getAction(agent, {obs});
        action = action_out{1};
        if iscategorical(action) || iscell(action)
            action = double(action); 
        end

        % Step the environment using the native nav function
        % [obs, reward, isDone, LoggedSignals] = stepFunction_nav(action, LoggedSignals, EnvPars);
        % [obs, reward, isDone, LoggedSignals] = stepFunction_nav_CST_Reward_Mixed(action, LoggedSignals, EnvPars);
        [obs, reward, isDone, LoggedSignals] = stepFunction_nav_CST_Aligned(action, LoggedSignals, EnvPars);
        step = step + 1;
    end

    % Record the number of steps it took to terminate
    steps_taken_dqn(i) = step;

    % Extract the final position estimate after the agent finishes
    pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x, LoggedSignals.psi_y, EnvPars, pos_MU_true);
    pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pos_MU_true);
    err_dqn(i) = norm(pos_est_dqn(1:2) - pos_MU_true(1:2))*1e2; % cm
    
    % ---------------------------------------------------------
    % BRUTE FORCE EVALUATION (Optimal Benchmark)
    % ---------------------------------------------------------
    % Retrieve the exact DOAs for this target index from Calibration
    psi_x_true = EnvPars.psi_x_cal(target_idx);
    psi_y_true = EnvPars.psi_y_cal(target_idx);
    
    a_psi_x   = exp(1i * psi_x_true * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y_true * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);
     
    best_power = -inf;
    best_psi_x_est = 0;
    best_psi_y_est = 0;

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
            % R_2D = reshape(abs(r), [EnvPars.N_x, EnvPars.N_y]);
            % [~, idx] = max(R_2D, [], 'all');
            % [n_y_max, n_x_max] = ind2sub(size(R_2D), idx);
            
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
end %[output:group:9ba08430]
%[text] ### Figures
% CDF
figure
hold on
h1 = cdfplot(err_dqn);
set(h1, 'LineWidth', 1.5);

grid on
h2 = cdfplot(err_bf);
set(h2, 'LineWidth', 1.5,'LineStyle','--');

grid on

%patch
x_coords = [0    30   30 0];
y_coords = [0.95 0.95 1  1];
patch(x_coords, y_coords, 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none');


% Formatting the plot matching your script's style
xlim([0, 200])
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', 14);
title('');
legend({'DQN Agent', 'Brute Force'}, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 14);

set(gca, 'Box', 'on', 'TickDir', 'out', 'LineWidth', 1, 'FontSize', 12);

% Boxplot
figure
g1 = repmat({'DQN agent'},N_eval,1);
g2 = repmat({'Brute force'},N_eval,1);
g = [g1; g2];
h3 = boxplot([err_dqn, err_bf], g) %[output:38a767b5]
set(h3, 'LineWidth', 1.5);
grid on

%%
%[text] ### Random Way Point mobility model
% Parameters
time_simul=1;
t = 0:delta_time:3*time_simul;     % time vector in s

EnvPars.tolerance = pi/25;
%% Initialization
N_Steps = length(t);
EnvPars.NSteps=N_Steps;

pos_RWP = zeros(N_Steps,3);

% Initial position
pos_MU_init=[1 2 1.5];
current_pos = pos_MU_init(:);
current_pos(1) = min(max(current_pos(1), 0), L_hall);
current_pos(2) = min(max(current_pos(2), 0), W_hall);

pos_RWP(1,:) = current_pos;

% Initial mobility state
state = "move";   % either "move" or "pause"

% First target waypoint
target_pos = [L_hall*rand; W_hall*rand; current_pos(3)];

% Pause handling
pause_end_time = -inf;

%% Time evolution
for k = 2:N_Steps
    
    dt = t(k) - t(k-1);
    
    % In case of non-positive time step, keep previous position
    if dt <= 0
        pos_RWP(k,:) = pos_RWP(k-1,:);
        continue;
    end
    
    if state == "move"
        
        % Vector toward target
        dvec = target_pos(1:2) - current_pos(1:2);
        dist_to_target = norm(dvec);
        travel_dist = MU_speed * dt;
        
        if dist_to_target <= travel_dist
            % Reach waypoint within this time step
            current_pos(1:2) = target_pos(1:2);
            
            % Switch to pause state
            pause_duration = pause_min + (pause_max - pause_min)*rand;
            pause_end_time = t(k) + pause_duration;
            state = "pause";
            
        else
            % Move toward waypoint
            direction = dvec / dist_to_target;
            current_pos(1:2) = current_pos(1:2) + direction * travel_dist;
        end
        
    elseif state == "pause"
        
        % Stay still during pause
        if t(k) >= pause_end_time
            % Pause finished, choose new waypoint and move again
            target_pos = [L_hall*rand; W_hall*rand; current_pos(3)];
            state = "move";
        end
        
    end
    
    % Store position
    pos_RWP(k,:) = current_pos;
end

% Enforce constant z-coordinate
pos_RWP(:,3) = pos_MU_init(3);

% %Evaluating the electric angles
% psi_x_MU=zeros(1,Nsteps);
% psi_y_MU=zeros(1,Nsteps);
% psi_y_MU
% for k=1:Nsteps
%     [psi_x_k, psi_y_k] = computePsiFromPos(pos_MU_traj(:,k), EnvPars);
%     psi_x_MU(k) = psi_x_k;
%     psi_y_MU(k) = psi_y_k;
% end
% 
% EnvPars.psi_x_RWP=psi_x_MU;
% EnvPars.psi_y_RWP=psi_y_MU;
% 
% EnvPars.pos_MU_RWP=pos_MU_traj;

% Preallocate calibration tables
peak_map       = zeros(N_Steps, EnvPars.T_x, EnvPars.T_y);
psi_x_RWP      = zeros(N_Steps, 1);
psi_y_RWP      = zeros(N_Steps, 1);
global_max_RWP = zeros(N_Steps, 1);
best_tx_RWP    = zeros(N_Steps, 1);
best_ty_RWP    = zeros(N_Steps, 1);

for i_Steps = 1:N_Steps
    pos_MU_k = [pos_RWP(i_Steps,1); pos_RWP(i_Steps,2); EnvPars.h_MU];
    

    [psi_x_k, psi_y_k] = computePsiFromPos(pos_MU_k, EnvPars);
    psi_x_RWP(i_Steps) = psi_x_k;
    psi_y_RWP(i_Steps) = psi_y_k;

    a_psi_x   = exp(1i * psi_x_k * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y_k * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);

    for t_psi = 1:EnvPars.T
        v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
        r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;
        peak_map(i_Steps, EnvPars.t_x(t_psi), EnvPars.t_y(t_psi)) = max(abs(r).^2);
    end

    % Global maximum and best action for this position
    slice                 = squeeze(peak_map(i_Steps,:,:));
    [global_max_RWP(i_Steps), idx] = max(slice(:));
    [best_tx_RWP(i_Steps), best_ty_RWP(i_Steps)] = ind2sub([EnvPars.T_x, EnvPars.T_y], idx);
end


% Store calibration in EnvPars — available to step/reset functions
EnvPars.N_Steps          = N_Steps;
EnvPars.peak_map       = peak_map;
EnvPars.psi_x_RWP      = psi_x_RWP;
EnvPars.psi_y_RWP      = psi_y_RWP;
EnvPars.pos_RWP        = pos_RWP;
EnvPars.global_max_RWP = global_max_RWP;
EnvPars.best_tx_RWP    = best_tx_RWP;
EnvPars.best_ty_RWP    = best_ty_RWP;
%[text] #### Including the channel coherence time
%[text] **Description**:In this block we include the number of transmittions along wich the channel is coherent.This is similar to assuming the user keeps the same position.
% pos_MU_traj = repelem(pos_MU_traj, 1, floor(N_packets_coh*delta_time/T_coh));
% pos_MU_traj = repelem(pos_MU_traj, 1, 5);
%[text] #### Ploting the mobility of the user
figure; %[output:7e6b773d]
plot(pos_RWP(:,1), pos_RWP(:,2), 'b-', 'LineWidth', 1.5); hold on; %[output:7e6b773d]
plot(pos_RWP(1,1), pos_RWP(1,2), 'go', 'MarkerSize', 8, 'LineWidth', 1.5); %[output:7e6b773d]
plot(pos_RWP(end,1), pos_RWP(end,2), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5); %[output:7e6b773d]
axis([0 L_hall 0 W_hall]); %[output:7e6b773d]
% xlim([0 L_hall]);
% ylim([0 W_hall]);
xlabel('$x$ [m]','Interpreter','latex'); %[output:7e6b773d]
ylabel('$y$ [m]','Interpreter','latex'); %[output:7e6b773d]
title('Random Waypoint Mobility with Random Pause'); %[output:7e6b773d]
grid on; %[output:7e6b773d]
%axis equal;
legend('Trajectory','Start','End'); %[output:7e6b773d]

% %% Animation of the mobile user trajectory
% 
% figure;
% hold on;
% grid on;
% axis equal;
% xlim([0 L_hall]);
% ylim([0 W_hall]);
% xlabel('$x$ [m]','Interpreter','latex');
% ylabel('$y$ [m]','Interpreter','latex');
% title('Mobile User Random Waypoint Trajectory');
% 
% % Draw hall boundary
% rectangle('Position',[0, 0, L_hall, W_hall], 'EdgeColor','k', 'LineWidth',1.5);
% 
% % Plot full trajectory as dashed reference
% plot(pos_MU_traj(1,:), pos_MU_traj(2,:), '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
% 
% % Initialize animated trajectory and current user marker
% h_traj = plot(pos_MU_traj(1,1), pos_MU_traj(2,1), 'b-', 'LineWidth', 2);
% h_user = plot(pos_MU_traj(1,1), pos_MU_traj(2,1), 'ro', ...
%     'MarkerFaceColor','r', 'MarkerSize',8);
% 
% % Optional text showing current time
% h_time = text(L_hall*(1-0.2), 0.91*W_hall, sprintf('$t = %.2f$ [s]', t(1)), ...
%     'FontSize', font, 'Color', 'k','BackgroundColor','w','Interpreter','latex');
% 
% set(gca,'FontSize',font)
% 
% for k = 1:2*N_packets_coh:length(t)
% 
%     % Update traveled path
%     set(h_traj, 'XData', pos_MU_traj(1,1:k), 'YData', pos_MU_traj(2,1:k));
% 
%     % Update current user position
%     set(h_user, 'XData', pos_MU_traj(1,k), 'YData', pos_MU_traj(2,k));
% 
%     % Update time label
%     set(h_time, 'String', sprintf('t = %.2f s', t(k)));
% 
%     drawnow;
% 
%     % Optional real-time pacing
%     if k < length(t)
%         pause((t(k+1) - t(k))/2);
%     end
% end
%[text] #### Agent and Brute force evaluations
N_steps=100;

R_max_per_frame = zeros(EnvPars.T_x, EnvPars.T_y, N_Steps);

[obs, LoggedSignals] = resetFunction_nav_CST_Reward_Mixed_deployment(EnvPars);

t_psi_x_dqn_frame = zeros(1,N_Steps);
t_psi_y_dqn_frame = zeros(1,N_Steps);

psi_x_dqn = zeros(1,N_Steps);
psi_y_dqn = zeros(1,N_Steps);

for i = 1:N_Steps %[output:group:7bf985ee]
    if mod(i, 10) == 0
        fprintf('Evaluating step %d / %d...\n', i, N_Steps); %[output:664e78eb]
    end
    
    % ---------------------------------------------------------
    % DQN AGENT EVALUATION (Episodic Loop)
    % ---------------------------------------------------------
    % Reset environment: automatically picks a random pos_cal position
    % [obs, LoggedSignals] = resetFunction_nav(EnvPars);
    
    
    % Identify the true target position chosen by the reset function
    target_idx = LoggedSignals.pos_idx;
    pos_MU_true = EnvPars.pos_MU_RWP(:,target_idx);
    
    % isDone = false;
    % step = 0;

    % % Let the agent navigate until it triggers the native termination (IsDone)
    % while ~isDone
        action_out = getAction(agent, {obs});
        action = action_out{1};
        if iscategorical(action) || iscell(action)
            action = double(action); 
        end

        % Step the environment using the native nav function
        % [obs, reward, isDone, LoggedSignals] = stepFunction_nav(action, LoggedSignals, EnvPars);
        % [obs, reward, isDone, LoggedSignals] = stepFunction_nav_CST_Reward_Mixed(action, LoggedSignals, EnvPars);
        [obs, LoggedSignals] = stepFunction_nav_CST_Aligned_deployment(action, LoggedSignals, EnvPars);
        step = step + 1;

        %estimated electric angles
        psi_x_dqn(i)=LoggedSignals.psi_x_est;
        psi_y_dqn(i)=LoggedSignals.psi_x_est;

        %estimated phase shift
        t_psi_x_dqn_frame(i) = LoggedSignals.t_psi_x_max;
        t_psi_y_dqn_frame(i) = LoggedSignals.t_psi_y_max;

    % end

    % Record the number of steps it took to terminate
    steps_taken_dqn(i) = step;

    % Extract the final position estimate after the agent finishes
    pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x, LoggedSignals.psi_y, EnvPars, pos_MU_true);
    pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pos_MU_true);
    err_dqn(i) = norm(pos_est_dqn(1:2) - pos_MU_true(1:2))*1e2; % cm
    
    % ---------------------------------------------------------
    % BRUTE FORCE EVALUATION (Optimal Benchmark)
    % ---------------------------------------------------------
    % Retrieve the exact DOAs for this target index from Calibration
    psi_x_true = EnvPars.psi_x_RWP(target_idx);
    psi_y_true = EnvPars.psi_y_RWP(target_idx);
    
    a_psi_x   = exp(1i * psi_x_true * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y_true * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);
     
    
    
    best_power = -inf;
    best_psi_x_est = 0;
    best_psi_y_est = 0;

    % Exhaustive search over all phase configurations
    R_array_k = zeros(EnvPars.T_x, EnvPars.T_y);%for storing partial results
    for t_psi = 1:EnvPars.T
        v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
        Upsilon   = diag(v0');
        G_Upsilon = EnvPars.G * Upsilon;
        r   = sqrt(db2pow(EnvPars.SNR_dB)) * G_Upsilon * a_psi_x_y;

        power_vec = abs(r);
        current_peak = max(power_vec);

        tx_loc = EnvPars.t_x(t_psi);   % 1-indexed value
        ty_loc = EnvPars.t_y(t_psi);   % 1-indexed value

        best_power = current_peak;
        R = flipud(fliplr(reshape(abs(r), [EnvPars.N_x, EnvPars.N_y])))';
        [R_array_k(tx_loc, ty_loc), linear_idx] = max(R, [], 'all');

        if current_peak > best_power
            n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
            n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;
            % R_2D = reshape(abs(r), [EnvPars.N_x, EnvPars.N_y]);
            % [~, idx] = max(R_2D, [], 'all');
            % [n_y_max, n_x_max] = ind2sub(size(R_2D), idx);

            % Map snapshot index to temporal coords
            t_psi_x_max = EnvPars.t_x(t_psi);
            t_psi_y_max = EnvPars.t_y(t_psi);

            best_psi_x_est = mod(2*pi * (n_psi_x_max + (t_psi_x_max) / EnvPars.T_x) / EnvPars.N_x, 2*pi);
            best_psi_y_est = mod(2*pi * (n_psi_y_max + (t_psi_y_max) / EnvPars.T_y) / EnvPars.N_y, 2*pi);

        end
    end
    
    t_psi_x_max_frame(i)=t_psi_x_max;
    t_psi_y_max_frame(i)=t_psi_y_max;
    R_max_per_frame(:,:,i)= R_array_k;

    % pos_MU_true = estimatePosFromAngles(psi_x_true, psi_y_true, EnvPars, pos_MU_true);
    pos_est_bf = estimatePosFromAngles(best_psi_x_est, best_psi_y_est, EnvPars, pos_MU_true);
    err_bf(i) = norm(pos_est_bf(1:2) - pos_MU_true(1:2))*1e2; % cm  
end %[output:group:7bf985ee]

%%


figure; %[output:7658e487]
h_img = imagesc(1:EnvPars.T_x, 1:EnvPars.T_y, R_max_per_frame(:,:,1)); %[output:7658e487]
axis square; colorbar; hold on; %[output:7658e487]

% Common color scale across frames
clim([min(R_max_per_frame(:)), max(R_max_per_frame(:))]); %[output:7658e487]

% % --- Brute-force optimum (red) ---
% % Trail of past optima
% h_trail_brute_force   = plot(t_psi_x_max_frame(1), t_psi_y_max_frame(1), '-', ...
%                      'Color', [1 0 0 0.4], 'LineWidth', 1.2);
% % Past optimum markers (dim red)
% h_past_brute_force    = plot(t_psi_x_max_frame(1), t_psi_y_max_frame(1), 'rx', ...
%                      'MarkerSize', 8, 'LineWidth', 1.0, ...
%                      'Color', [1 0.6 0.6]);
% % Current optimum (bright red)
% h_opt_brute_force         = plot(t_psi_x_max_frame(1), t_psi_y_max_frame(1), 'rx', ...
%                      'MarkerSize', 16, 'LineWidth', 2.5);

% % --- Agent (white) ---
% Trail of past agent actions
h_trail_agent = plot(t_psi_x_dqn_frame(1), t_psi_y_dqn_frame(1), '-', ... %[output:7658e487]
                     'Color', [0 0 0 0.4], 'LineWidth', 1.2); %[output:7658e487]
% Past action markers (dim)
h_past_agent  = plot(t_psi_x_dqn_frame(1), t_psi_y_dqn_frame(1), 'ko', 'MarkerSize', 6, ... %[output:7658e487]
                     'MarkerFaceColor',[0.6 0.6 0.6], 'MarkerEdgeColor','k'); %[output:7658e487]
%Current agent action (bright)
h_agent       = plot(t_psi_x_dqn_frame(1), t_psi_y_dqn_frame(1), 'ws', 'MarkerSize', 14, ... %[output:7658e487]
                     'MarkerFaceColor', 'w', 'LineWidth', 1.5); %[output:7658e487]

set(gca,'FontSize',font-5); %[output:7658e487]

xlabel('$\Delta_{\psi_x}$','Interpreter','latex','FontSize',font); %[output:7658e487]
ylabel('$\Delta_{\psi_y}$','Interpreter','latex','FontSize',font); %[output:7658e487]
%h_title = title(sprintf('Peak 2D-DFT, $t=$ %.1fs / %.1fs', delta_time, time_simul), ...
               % 'Interpreter','latex');

xlim([0.5 EnvPars.T_x+0.5]); ylim([0.5 EnvPars.T_y+0.5]); %[output:7658e487]

for k = 1:N_Steps-1 %[output:group:85f4f077]
    set(h_img, 'CData', R_max_per_frame(:,:,k));
        
    % % Optimum: trail, past markers, current
    % set(h_trail_brute_force,   'XData', t_psi_x_max_frame(1:k),          'YData', t_psi_y_max_frame(1:k));
    % set(h_past_brute_force,    'XData', t_psi_x_max_frame(1:max(1,k-1)), 'YData', t_psi_y_max_frame(1:max(1,k-1)));
    % set(h_opt_brute_force,     'XData', t_psi_x_max_frame(k),            'YData', t_psi_y_max_frame(k));
    
    % Agent: trail, past markers, current
    set(h_trail_agent, 'XData', t_psi_x_dqn_frame(1:k),          'YData', t_psi_y_dqn_frame(1:k)); %[output:7658e487]
    set(h_past_agent,  'XData', t_psi_x_dqn_frame(1:max(1,k-1)), 'YData', t_psi_y_dqn_frame(1:max(1,k-1)));
    set(h_agent,       'XData', t_psi_x_dqn_frame(k),            'YData', t_psi_y_dqn_frame(k));


    % set(h_title, 'String', sprintf('Peak 2D-DFT, $t=$ %.1fs / %.1fs', k*delta_time, time_simul));
    drawnow;
    if k < Nsteps
        pause(0.05);
    end
end %[output:group:85f4f077]
%%
%[text] ### Functions
function pos_MU_est = estimatePosFromAngles(psi_x_est, psi_y_est, EnvPars, pos_MU_true)
    % 1. Shift phases back to principal domain [-pi, pi]
    psi_x_est = mod(psi_x_est + pi, 2*pi) - pi;
    psi_y_est = mod(psi_y_est + pi, 2*pi) - pi;

    phi_est = atan2(-psi_y_est, -psi_x_est);
    % phi_est = atan(psi_y_est*EnvPars.d_x/(psi_x_est*EnvPars.d_x))
    theta_est = asin(EnvPars.lambda/(2*pi)*sqrt(psi_x_est^2/EnvPars.d_x^2+psi_y_est^2/EnvPars.d_x^2));

    pos_MU_est(1) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*cos(phi_est)+EnvPars.pos_SIM(1);
    pos_MU_est(2) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*sin(phi_est)+EnvPars.pos_SIM(2);
    pos_MU_est(3) = pos_MU_true(3);
end

function F_complex = buildComplexPhaseInterpolant(phase_deg, coefficient)
%BUILDCOMPLEXPHASEINTERPOLANT
% Creates a phase-command -> complex-coefficient interpolation.
%
% Real and imaginary parts are interpolated separately. This avoids phase
% unwrapping artifacts and artificial jumps close to +/-180 degrees.
%
% Extrapolation uses the nearest measured boundary value, matching the
% conservative behavior of the existing CST amplitude interpolant.

    phase_deg   = phase_deg(:);
    coefficient = coefficient(:);

    if numel(phase_deg) ~= numel(coefficient)
        error('phase_deg and coefficient must have equal lengths.');
    end

    phase_rad = mod(deg2rad(phase_deg), 2*pi);

    % griddedInterpolant requires a strictly increasing grid.
    [phase_sorted, order] = sort(phase_rad);
    coefficient_sorted = coefficient(order);

    % Remove exact duplicate phase entries.
    [phase_unique, unique_idx] = unique(phase_sorted, 'stable');
    coefficient_unique = coefficient_sorted(unique_idx);

    if numel(phase_unique) < 2
        error('At least two distinct state phases are required.');
    end

    F_real = griddedInterpolant( ...
        phase_unique, ...
        real(coefficient_unique), ...
        'pchip', ...
        'nearest');

    F_imag = griddedInterpolant( ...
        phase_unique, ...
        imag(coefficient_unique), ...
        'pchip', ...
        'nearest');

    F_complex = @(phase_query) ...
        F_real(mod(phase_query, 2*pi)) + ...
        1i * F_imag(mod(phase_query, 2*pi));
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
%   data: {"layout":"onright","rightPanelPercent":44.5}
%---
%[output:2a25c3ad]
%   data: {"dataType":"text","outputData":{"text":"=== Initializing Random Deployment Environment ===\n","truncated":false}}
%---
%[output:5a0597c6]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:79ace775]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:1cd895fd]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"4"}}
%---
%[output:1912f58b]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:2928e912]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:702a069b]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:7aa31890]
%   data: {"dataType":"matrix","outputData":{"columns":6,"name":"zeta","rows":1,"type":"double","value":[["0.9850","0.9860","0.9870","0.9880","0.9890","0.9900"]]}}
%---
%[output:5759a5d2]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:455a99a7]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:1a2bfe05]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"40"}}
%---
%[output:6b43b9a7]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:66d2d09a]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     N: 16\n                   N_x: 4\n                   N_y: 4\n                     T: 1600\n                   T_x: 40\n                   T_y: 40\n                SNR_dB: 36.5005\n             theta_min: 1.8485\n             theta_max: 4.4347\n                    fc: 2.8000e+10\n                lambda: 0.0107\n               Ptx_dBm: 23\n               Gtx_dBi: 14\n               Grx_dBi: 8\n               txArray: [1×1 struct]\n                   cdl: [1×1 nrCDLChannel]\n          var_noise_dB: -110.9794\n                     r: 0\n                   d_x: 0.0054\n               pos_SIM: [5 5 4]\n                pos_MU: [2.7850 5.4688 1.5000]\n                   n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                   n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                   t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 … ] (1×1600 double)\n                   t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 … ] (1×1600 double)\n                  h_MU: 1.5000\n                L_hall: 10\n                W_hall: 10\n                 N_cal: 100\n             MU_margin: 0.5000\n           MaxEpisodes: 5000\n                 psi_x: 0\n                 psi_y: 0\n    MaxStepsPerEpisode: 120\n             tolerance: 0.0393\n     StopTrainingValue: 114\n       episode_counter: 0\n           delta_moves: [9×2 double]\n             n_actions: 9\n        DiscountFactor: 0.9500\n"}}
%---
%[output:3a1c9308]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:01d3f9e8]
%   data: {"dataType":"textualVariable","outputData":{"name":"agent_path","value":"'..\\Dataset\\dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Aligned.mat'"}}
%---
%[output:594fc1ea]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:37ee272b]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtUAAAGzCAYAAADg9uuxAAAAAXNSR0IArs4c6QAAIABJREFUeF7snQmYXFW171dVz91JZw6EJpAwhIQphJAQJpHhXa8o9+GVJwpPkFkZ5CLeiyByIfpEhIugoDJdfOqVyfEiyEMNggEyQAYIJCSBBMg8p5Meqrur6n3\/XVmV3bvPPFSd6l77+\/LRVO3p\/Pc+p357nbXXTuXz+TxJEgVEAVFAFBAFRAFRQBQQBUSBwAqkBKoDaycFRQFRQBQQBUQBUUAUEAVEAaWAQLVMBFFAFBAFRAFRQBQQBUQBUSCkAgLVIQWU4qKAKCAKiAKigCggCogCooBAtcwBUUAUEAVEAVFAFBAFRAFRIKQCAtUhBZTiooAoIAqIAqKAKCAKiAKigEC1zAFRQBQQBUQBUUAUEAVEAVEgpAIC1SEFlOKigCggCogCooAoIAqIAqKAQLXMAVFAFBAFRAFRQBQQBUQBUSCkAgLVIQWU4qKAKCAKiAKigCggCogCooBAtcwBUUAUEAVEAVFAFBAFRAFRIKQCAtUhBZTiooAoIAqIAqKAKCAKiAKigEC1zAFRQBQQBUQBUUAUEAVEAVEgpAIC1SEFlOKigCggCogCooAoIAqIAqKAQLXMAVFAFBAFRAFRQBQQBUQBUSCkAgLVIQWU4qKAKCAKiAKigCggCogCooBAtcwBUUAUEAVEAVFAFBAFRAFRIKQCAtUhBZTiooAoIAqIAqKAKCAKiAKigEC1zAFRQBQQBUQBUUAUEAVEAVEgpAIC1SEFlOKigCggCogCooAoIAqIAqKAQLXMAVFAFBAFRAFRQBQQBUQBUSCkAgLVIQWU4qKAKCAKiAKigCggCogCokC\/heqenh565ZVX6NFHH6W5c+dSd3c3tbS00DnnnEMXXnghjRo1ynb0W1tb6YknnqAnn3ySVq1aRTU1NXTGGWfQRRddRNOmTaN0Oi0zRxQQBUQBUUAUEAVEAVFAFCgq0C+hevv27fSd73yHfve731kO9X777Uf\/8R\/\/Qccff3yf75cvX07XXHMNrVy5ss931dXVdPXVV6t\/+FuSKCAKiAKigCggCogCooAoAAX6HVR3dHTQt771Lfrtb39LgOfbbruNPvaxjykIXrx4Mc2cOVP9Fxbn+++\/v5fFGjB+3XXX0ezZs+m0006jW265hcaNG0e7d++mp59+mu68807K5\/N077330llnnSUzSBQQBUQBUUAUEAVEAVFAFFAK9Duofumll+jyyy+nYcOG0UMPPUSTJ0\/uNdQrVqygiy++mNatW0f33XcfnX322cXvn3nmGQXVkyZNUmXhLsIJMP3zn\/+cbr\/9dpoxYwY98MADqg1JooAoIAqIAqKAKCAKiAKiQL+C6kwmo6zUv\/71r+n6669XbhypVKrXKMPX+tvf\/jY999xzykf6qquuUj7SsEZ\/\/etfpxdeeEFZqC+55JI+s2PDhg105ZVX0tKlS+nhhx+mU089VWaQKCAKiAKigCggCogCooAo0L8s1R999JGyQu\/cuZMee+wxOvLIIz0PMTYkfulLX6Jt27Ypi\/SUKVP6lO3q6qJbb72VnnrqKeVXfcMNN3iuX8+4Zs0a+s1vfkOf\/exnaf\/99w9UhxQqnwIyfuXTPoqWZfyiULG8dcgYllf\/sK3L+IVVsLzlZfzs9e9Xluo5c+bQ+eefT1OnTlXuG37cM7jsQQcdpIB87NixlqrBD\/uee+5RbiPf+973qKGhwffs5rZ+9atfKVcSSZWlgIxfZY2X2VsZv8oeP\/RexrCyx1DGT8avshUYIFCNEHg33XRTEXg3b95MP\/3pT+nPf\/4zbd26lcaPH0\/nnXcenXvuuTR8+PBeqjz77LN07bXXKh9shOEzv+fM3EYQcOc65IFS2beTjJ+MX2UrUPm9l3uwssdQxk\/Gr7IVGCBQzVbkyy67TLl+wL96165dfa7+kEMOUZE\/JkyYUPzOKyzPmjWLUL8bfDtNGHmgVPbtJOMn41fZClR+7+UerOwxlPGT8atsBQYIVCPk3YMPPqhC6QGmjzvuOLX58LDDDqNcLqdC6SF+Nf47ffp0+uEPf0ijR49W6phWbju3Dn4YCFT311vC\/brkB8FdoyTnkPFL8uh465uMoTedkppLxi+pI+OtXzJ+AwyqcbkmNLMEONzlK1\/5ijopETGscbpinFANh37809PatWvpX\/\/1X1UkEbiRSKosBRAFBhtWZfwqa9y4tzJ+lTlueq9lDCt7DGX8+sf4IQSxeYgegi8M5AAM\/WqjIluqMV0feeQROv300\/vMXMSbhuvHD37wAzrzzDPVpsNBgwYVLdVuvtJ+3T8QCxv\/JIkCooAoIAqIAqKAKNCfFQBo499ATf0Kqp944gm6+eablfsHIngceuihluOKExNhoZ44caLalDhmzBiKa6OilaUaB9TATeWuu+7qdcDMQJ2ElXbdnZ2dtGPHDho5cqQcV19pg7env1u2bKH6+nq1oJZUeQrgXAHch7gHJVWeAjgvAvegPEMrb+zQ47lz5ypjId7YgqP0JJZqmG77SWIwdoNq9gfS882fP58uuOACOuCAA2IPqfeXv\/yFrrjiCpKQepU58fBjvn79ehV2sbq6ujIvYoD3GjHtAdR+wm4OcMkSdfnbt29XB3bZhT5NVGelM30UAFTjHoRBC4tbSZWlADMUQhfjjb+kvQr0K0v1woULlQUaN6nT4S9WLhx8cMzGjRtjP\/xFoLqyb0GB6soeP\/ReoLqyx1CgurLHT6C6ssdPoNp+\/PoVVON1EizAixYt6rUJUb98GObhdoH41foBLqU8plygurIfKALVlT1+AtWVP34C1ZU9hgLVlT1+AtUDBKoBzIBlQDMOevnJT37SKxY1ZHj33Xfp0ksvpU2bNtG9995LZ511VlGdZ555RjnYT5o0SZ3I2NLSUvwOdeP48ttvv52mTZumNjuOGjUq0J0hUB1ItsQUEqhOzFAE7ohYqgNLl4iCAtWJGIbAnRCoDixdIgoKVA8QqMZlApa\/+tWv0rx589QBLQhdh\/B6SJgIM2fOpJUrV9KnPvUpuuOOO3ptVMIJjIBq5ENZOOHjv21tbfT444\/T3XffreoxYdzvLBeo9qtYsvILVCdrPIL0RqA6iGrJKSNQnZyxCNITgeogqiWnjED1AIJqXCp+MG+88UYFx1bp5JNPVkCtW6I5H+JYX3PNNQq8zYRNaVdffbX6F2aDmkB1ch4OQXoiUB1EtWSVEahO1nj47Y1AtV\/FkpVfoDpZ4+G3NwLVAwyqcbmwLj\/33HPKwrxkyRKlAI4u\/8IXvqBcPpqammxV2bZtG\/36179WsatxSExNTQ2dccYZdNFFFynXj3Q67XcO9sovUB1KvrIXFqgu+xCE7oBAdWgJy1qBQHVZ5Q\/duEB1aAnLWoFA9QCE6rLOOJfGBaqTPDrufROodtco6TkEqpM+Qs79E6iu7PETqK7s8ROoFqhO1AwWqE7UcPjujEC1b8kSV0CgOnFD4qtDAtW+5EpcZoHqxA2Jrw4JVAtU+5owcWcWqI5b4XjrF6iOV99S1C5QXQqV42tDoDo+bUtRs0B1KVSOrw2BaoHq+GZXgJoFqgOIlqAiAtUJGoyAXRGoDihcQooJVCdkIAJ2Q6A6oHAJKSZQLVCdkKlY6IZAdaKGw3dnBKp9S5a4AgLViRsSXx0SqPYlV+IyC1Qnbkh8dUigWqDa14SJO7NAddwKx1u\/QHW8+paidoHqUqgcXxsC1fFpW4qaBapLoXJ8bQhUC1THN7sC1CxQHUC0BBURqE7QYATsikB1QOESUkygOiEDEbAbAtUBhUtIMYFqgeqETMVCNwSqEzUcvjsjUO1bssQVEKhO3JD46pBAtS+5EpdZoDpxQ+KrQwLVAtW+JkzcmQWq41Y43voFquPVtxS1C1SXQuX42hCojk\/bUtQsUF0KleNrQ6BaoDq+2RWgZoHqAKIlqIhAdYIGI2BXBKoDCpeQYgLVCRmIgN0QqA4oXEKKCVQLVCdkKha6IVCdqOHw3RmBat+SJa6AQHXihsRXhwSqfcmVuMwC1YkbEl8dEqgWqPY1YeLOLFAdt8Lx1i9QHa++pahdoLoUKsfXhkB1fNqWomaB6lKoHF8bAtUC1fHNrgA1C1QHEC1BRQSqEzQYAbsiUB1QuIQUE6hOyEAE7IZAdUDhElJMoFqgOiFTsdANgepEDYfvzghU+5YscQUEqhM3JL46JFDtS67EZRaoTtyQ+OqQQLVAta8JE3dmgeq4FY63foHqePUtRe0C1aVQOb42BKrj07YUNQtUl0Ll+NoQqBaojm92BahZoDqAaAkqIlCdoMEI2BWB6oDCJaSYQHVCBiJgNwSqAwqXkGIC1QLVCZmKhW4IVCdqOHx3RqDat2SJKyBQnbgh8dUhgWpfciUus0B14obEV4cEqgWqfU2YuDMLVMetcLz1C1THq28paheoLoXK8bUhUB2ftqWoWaC6FCrH14ZAtUB1fLMrQM0C1QFES1ARgeoEDUbArghUBxQuIcUEqhMyEAG7IVAdULiEFBOoFqhOyFQsdEOgOlHD4bszAtW+JUtcAYHqxA2Jrw4JVPuSK3GZBaoTNyS+OiRQLVDta8LEnVmgOm6F461foDpefUtRu0B1KVSOrw2B6vi0LUXNAtWlUDm+NgSqBarjm10BahaoDiBagooIVCdoMAJ2RaA6oHAJKSZQnZCBCNgNgeqAwiWkmEC1QHVCpmKhGwLViRoO350RqPYtWaIKvPbaa9Ta2krNzc10wgknBO5bJpdXZevSqcB1SMFgCghUB9MtKaUEqpMyEsH6IVAtUB1s5sRUSqA6JmFLVG05oBog2N7eTo2NjaFAsEQSJbqZv\/71r9TR0UENDQ10xhlnBO5rXFCNegXUnYdFoDrwtE1EQYHqRAxD4E4IVAtUB548cRQUqI5DVes644DRckB1VCBYOuW9txTHGDm1zpbq6urqkkC13+tb256llsYq7wIOwJwC1ZU96ALVlT1+AtUC1YmawQLVpQPgOGC0HFDNYAblwlhXE3Uj7OlM2DHyC61oFj7Vq1evpq6uLkvrv5c6YVHOZImaa5zdP\/xen0C1+ywVqHbXKMk5BKqTPDrufROoFqh2nyUlzCFQbS22X\/jwMmRxwGg5oNrLtUadxwtYRtFm2DEKMm\/e\/3ANrVj6NnV3d1u6gXip04Tq1u68JWD7vb73d2fpoEFiqXaaWwLVUdx53uuwm9vea+idU6A6qHLJKCdQLVCdjJm4pxcC1c6W6qRbY8sN1aWCXS9gGdWNFeaa\/EIr+gyo\/uCDDyjX1akuwbT+e7l2E6q3ZHK0YsHc0L7vAtXus0qHavFBd9crbA6B6rAK9q\/yAtUC1Yma0QMJqsPAUtBBi7vNckO1F+ALqp1eLgisBm3XzzXp\/UJ7QTZvAqqXr1pNNT0ZR\/cPE7h1gMPfu7rzNLIurS4bbhvLXvub2gSJhI2QgfomlmrXaaRDtbjLuMoVOoNAdWgJ+1UFAtUC1Yma0AMJqv3AUlSDFHeb5YbqUsBunAsTq7rdrkkvgygoDK4Mr379zAHVS5Ytp3RXu68oIG5Q\/eHiecpSrYO13jcvcJJUS3Wcc8LvvV+pUJ0kDf1o7mXe+qlP3D\/8qJW8vALVAtWJmpUDCardYCmOgYm7zXJDdRyamXXGuTAJUrdeBtZfgKuegkD1ex9+ROlMoR6v4OsE1ToM281BL3CSVKgOMm5xzdVKheokaehnbLzMWz\/1CVT7USt5eQWqBaoTNSsHElQnSviIOjMQoDrOhUmQuoOUsRtuAMIHa9fRkKYGGjp0GL39+hwF6VX1DfTxk04kEyBMkEa9iCNtun94gWEvcOKlnoimsq9qohwDXw1bZK5UqE6Shn7GwMu89VOfQLUftZKXV6BaoDpRszJJUF2pryPLOaADAapNfeOaJ37r9Zvfap4AEFasWa+gevSwoTT\/5VnKXaOqroE++T\/O8AXVWzpzxZjSXmDYDU4A6st2ZmnysOpyTvHEt12pUJ10Ya0ONMJn+jyP4hoEqqNQsXx1CFQLVJdv9lm0nCSortTXkeUc0LBQHQUYlvr645onfuu1y2+lqV1UiGdffYO2726nqqoqGtlUT4PyGWWp7soRnX7G6b02H0JnvR5AcV3VXkt1HFC9eHsPTR9R43uI3YDdrcK4Toh0azfI9\/0BqhEthje5BtEgjjLm2xee\/2vbc57CPHrdNCpQHcfola7Ov7++iC763D\/TQw89RGeeeWbpGq6AllL5fD5fAf3sV130CtWlgK9yvI6M+7rirj8sVPsFSavJH\/c12lmq8bkX\/2WvETr8zj\/WDv3Qo2uYmjoB4uN\/fZXau7rVJY6qr6J\/Ou1k9Tcgpzadol3de63PJlQjz+CaVNH9ww9UA3rNuk2d0e+gUO0VaOwepugfktthNkl4GPcHqA47XnGMg0B1HKr2vzqfe20BXXPBuQLVFkMrUF2G+e4VqqOArzJcnmuTcV+XU\/1RwGhYqPYLklaC+rHYlgPKdfhlAGYYDzMGdvWamgJYBtekLQHxD68tpC27O6m2KkVDUj2eoJpPTtShGhC6tiNLk5qrlcsI\/+0ErQLVro8HTxkEqj3J5DsToNq0SvtZ6HldKIil2vfQJKqAQLX9cAhUl2GqeoXqKOCrDJfn2mTc1+VUfxRAHxaqXQXykMHuGr1en9d8HrpimQX927p1q\/oOFmUkhuowbZsWcL1e7gjyrO\/M07CGGjpx+nHF\/mFzIb5b2ZGiTKqGjjp0PB08aqiyTsM6C2BGymTzRT9ptt525fIq36vzX6dcZ4eq+4jjZvSC6lW77X2h+aCYOKF6aWuPekWP6wySvCwMgtQbR5n+ANXwwYf7R9xvBvwsYq0A2i9UY7xbGp1PBBWo9ndX6G5n\/krGk\/tPb7xD1197Nd0\/85vi\/mFILFAdz5xzrNUrVJeha7aABJ\/TIAdZJOUadODicGxe3Bis+m8H1V5+vLzkCaOZ1wWL13xh+uJmTbcC4jDtcVm0uyaTVq4dJ59yCr3xxuvUuauVhg4qhOJb3VVNPTUNdMzEQ2j\/4UOURRsRQDa0dVFDYyMddcyxRShg0Oa6Z702n3JtrcoPO93YTDvqhtD5J01WQA5IsvOFZjh3Am8F9CHcP+A2Mn5QVWBIQx\/d+hfF+ERRR3+BagB13H7VfhaxcUK1\/uybNm0affTRRzRmzBiqr6+PYkr06zrwfNEX++W+WIFq+xEQqC7D7Kw0qPbzUPYiZ6nAMq5FgB1Ue9HJSx4vGgbJU2rd3cA9SH\/0MsceP4MWzC2Ew9PHmq3RcO048cQTad4rs2lbZzeNqSucvriso0ptSmSorqtK0eLZL9Lmzhylm5rpuOOOK27KMqH6lQWLKdXZRum2HdSZqqYdjaPostOmukI1XoujHbxad4rsUQqottvAaQXVSbOQ8by3g2q3zZpJOtIci7BSQLXbfag\/S+KEav3Zd+qppwpU+3iI4zmEuY03UUlIAtUC1UmYh8U+mFAdBC5KeUF+HsqmRdgKbK3AMowGZtm4wZWhes2aNdTV1VUEOi86eckTx9ia7hh+rfRexsev7n7zQxfdp3rIiFHU1b5bhcPjTYsM2I0Tp1FtFdGoujQten0+bepO0z7ZVjr5tNNVyLq1mzbTiKY6ZakG7K5YMFdZqrur6+moY6cqP2kk\/jHjV\/QMbZx\/a9UgT1AN14zmariY5D1B9cTmat8WZ6+WajtQjguqvfrZ+pn3DNWjW\/ZX49nSmFYWXzeoDtqXOCJ1lAqq\/egaJ1Trzz6BavdR0TdbxwHVbveKUw8FqgWq3WdwCXOYUB0ELkrY3UBNedksiIqj8LM124obXBmqly9fTt3d3b6OuQ4kZgSFdI103b1W7XU8UZ8OuSeccEKvJnQ4xxd+XXH0xQHcL+DbrJ+uyG2PPPZU5UrBlh22EHMcaEB1c2M97TesuWgt5B8uLgffZP6sLk0KvvmHCACnu3zg7+def5tGZrarPsGKrvs2A3hH1nmH6paGtHJB8QNzfqAa2pu+vHZQDX\/yMC4KQUHWaW4yVDeMblHWf4ZqtDWyPm3rVx60L0HLOV3DQINqXQvxqXZ\/8uqRWPAc2JzJFRf77qXdc4SZ0wLVAtXuM6yEOews1UFgx4sFsYSXVmzKL9iGWViYbVmBW5SuIKalOsi4lXpM\/I6H2T+n8ubYOY2ln3HW20R\/eAzx+Za2TspX1ajIHZwPQI2Ur2+ifaYUwuQxVL\/w+lu0T+cWytcPouaJx9KWLVtsoXr+kqU0uH2bguNDjz1egbQO1fibgZf9qPGj9\/tXF9LQ9s3UVFuj4l1HCdVefgD9QLUVKOtQzZpCr4Mm73WHCTJv3TbkBXmG2UE12uLxseqr3henCDFmWeRFctuA50cft776qSuqvE6Wai9vT7zqJFDtPmL6oTtxQLWXw6rseilQLVDtPoNLmCNKn2o\/kFLCS\/TdVFjo0xs0rbJsvfTr8mB3EUmI\/uFb4D0FggCMW1t2ixori7VuaR4xYgSZlmyrceTP4OLBY8g+yrCg6i4hPZSmjoahdPAx03pBNQNvXV2dAm6G6hFDBiu3DIZkWD0Xv7mYutt307jaHpp88mkEP+qu9jZqbGigSUcfoyzgcA+xs1Rnc1k69dSP97IEs6V6bUeOnOCEoYYt1QzTYaFaH3dELbHakKhDNWuKBcoh008JZSFzg8cgzzAdqhkOMBfc2tK\/92Mp9gqLbveK\/r1bX\/3UFVVegeqolAxfjz4WcUA1XNLgHhfkLZRAtUB1+BkeYQ1RQnWUMOp2iXEAGdqMul5dE1g3\/boYuOlQCVBtp2kQgHHTw+57u7bc+mBaqPX6Gar1HwQ9\/66ODG2taqKa5uE0ceLEoqUarhmDunZRTTbTC6oHDxqs3DIA1QAn+Dwve3cZZTs7qHn3RgXo7ZnCQTGNdTU09fgTFExjs+Gu7nwx4gcDNoAZCf+vbyrSoRqf2\/2Q6VANNwY+stxLuDwnS7Wu+bSPne4K1axpez4dO1TbPcOcngsDGaqjfl7q95cTVDvNW67D6+JDLNXuT1WBaneNkphDon+UYVQA1W+99RYdffTRJQ9TF+aB7BYiLaiLhRtkBRmiMNfp1p7dRkW3cub3cfbRbazQl6gs93YLIztYchtvLxtZGVLN1\/HPzXqZPsqkqaq+Qd1fsPjCF5pf+wNmfz9vifL7HtxQR0ccfoSC6g8Xz6M17VnKNQ2liYdNVJscP3p9Nm3qzFImVU21dXVUn++hI6fNUDDNwMzuH4AJtkIjFjVCXkcB1Xy6ohfXDqc8+lh4gWqeq0FjV+tze5+jpju6ZHhZlPHimJ8xAxmq3e4fv88hr1ANdyanyDWoR6A6jPq9ywpUR6dlKWsSqC6l2nvaAlSvW7eOhg8fbrvJLS7gCvNADgpJbhLHYW0Pc51u\/Y1qo6LeRxMa7EDVrW\/8fdSamtZ\/023Dj9527iK6z7T5dsGsf97WbmXtNUNM\/eWVuSoONRKgevO7b1J7Rwd11A+hkycfrso8NXsBZTIZlWfw4MF0xpTDadlrf6P1GVJQPfnoycrFI7N8AW1o76bWroI\/7cihzTThiKP6QDVAA8eVe4VqHHPO0UXM8TQt1VFBtd6OXTxqu42KbidFWs1JfbzGH\/\/xQFBtvrHQ3bgGMlRHfW8LVHt9qpY2n0B1afWOqjWB6qiU9FGPbqlGMSuLoR9I8dF00dXCrl0\/dcUFcEH6YJbx+sMTZPES1UZFJ2jA9cQ1B+z0ddJC91vWfZujmANertMcT0D1+uVLaFD7NvW2h6NtvDhnPr3Xkaba2lrl\/vHBorkKoDsbhtLp045RUA1L9a5du1TX4WONzz9cPJfWdxK1pesUjCPBlQNW5xfmvEG5TAc11tbQMdOmF6EahzEApGENR4oaqt9f\/Dq921lNR9R3EUIEuh3s4sWajX76geqgvpz6eAWFan2emuM\/kKE6iuejXR1O7h9iqY5T+b51C1SXVu+oWhOojkpJH\/V48an2CoU+mi171iAAG0en9X7AIup3I6PuUz1\/\/vw+h48E6bOTy0MUCyAv2vsJmxfEdcSuD17nul6+asJx9O7iN2hY+xb1tgcuDQgR998vzlbuH4BltlS3dmQULJ80dbLamAOXjbfffpuy2YIFGp\/D4o3P57+1VMUeB5SfddwR6hSzlxYuoWymg6rzOVeoRj0o4+T+4dVS\/facv9OKrlqaWNNBCBFYSVCt3wNxbMgbCFDt5Z4N8qwxF8K6214pofrvf\/875XK5krtAhtGslGUFqkupdnRtCVRHp6XnmrxAtefKIs7o50HuJy+66cUi6XQ5ftuzq8vK7cIPuOpQ\/dJLL3mGctMyrf+YBdXGqyZe6reDW6d+W2ls1ycvfXAafy5f2ziIhk4+iVYue0eFycumquiIGaeozYbPvzpfHVGOBKjmDYGwuAJmEekDFmVE\/8DxyIMGDVKftzRUqc178xctVpZtQPk5J05RgIzNi\/Cxhi818rJPNSy+HEcan3F867lvLKS2rh4aW5crRjfRNyp6heoVC+fSyq46Gp\/b0QeqrTROkqW6EqDaT\/SDOBYGbnWGvV\/cfhqs6o8Cqs0FpVU\/sFER7VdKnH83LeP4XqA6DlXjr1OgOn6N+7SQZKh22ySGi+FT6\/xaeZ0sklaQYH4W1Y+MV8uo3dQICtW6CwXq1t0ogvbJqyZB69cXQ6yHlfuHrlWYTZJOiwT+LpOqoZGTC4fKwEUDMZcBv9hsiFMRccohEqzNkw+fpEAY8IvX19iU6ATVi98pWKqRGKqRn2Fah2ocxgCY5oNgUAYuKbu2b6MspWn\/ulzRtSsIVCP6h51PtZXGDNVvv9736HZ9fErh\/iFQ7f7D4gbVYe5Z99b3Rl3SDQqlhGq2VPsxaHi5rv6SR6C6MkdSoLoM48ZQfeedd6rNikGjZgTpuptl0+rHWv8MbZrHQkfxUHRrF+4Gcf\/IeNXTDqqtNhvqdZoW3yh0K4UmfvvtBtVO890sazWKWMfRAAAgAElEQVRfOdIG9GN\/Zvy94NWXqS3To0LqIcHaPO2Yyb2g+r23F9G27pRy\/cDiAH7XgG24VsBSzUcD63XbQfVLC5bQ9s5uGl6Tp92pOgXjNR2tVEU5gjV9RHZ3bFBtNe4M1fNfnuX49kSgunCIS9It1V6fR1HmKyVUf\/TRRzRmzBj1xkhSXwUEqitzVghUl2HcGKpvu+02xwggcXTNzbJp9WOtfxZH3Gdcp1u7QXx4g+pnXq8Z6ULfqLhz504FZ5z8+mdb9dFt4RP0ukpVzg70vcA2+qhH\/tCt+3xYDGAIFmIG39aewt9r33qDdnV0Fi3V+KyxtppSnW3UXT9YWa3fWTCPthXCThf9rgHVOOYalkM9AdjnLH5bgXNzTZrGTTqyl\/vHC6+9sQeqCxsc4TYyqr6KUtluddrj5z5+fLG6qC3VVmNpWqqRx+q+EagWqLZ7FghUl+op6d6OQLW7RknMIVBdhlExLdV2P35xdK0Ulk23fntx9XCrI87v3SJd2IXUi0pbt4WPfu1BATxouTCLADfYRt1YoOiWbJTZunWrapbdTnSohusFfnwWvlU43AUW6F0Nw1V+WI4BuvX5bnU8OXys1yxdXLRUV1VVqc8A1euXLlS+2Bw1hIH9pXkLClBdm6bDJk9VedGeDtxNuYyCdrTXnOqhk6YfF8lGRSf3DyeoxoZNpyRQLVAtUB3nL0g0dQtUR6NjqWsRqC614kQUlU91lGDkJkOUbVm94jfBya0\/cX5vujuY1r6wIfXctPQD534AXNcsaDkr3b3WZXfdOjjr8MxtoX6clFhNOQXWW+uGK6sx0uqlS6itq5u6Ml0KnuvyPbS7sQDVvOEQpyP2pNIqJjViVcO\/Wt+oiLxrF75K63pqitZrhmq2VNfluxVUY3Nke1ePch9AH0yfakB+XZocoRp9HpEphAI034IkLU510JB6+jxx8x0Oci9HEf1D3D\/6Kl9KS\/VAjv7h9hugnl+5fHFPRRT3oTnafua\/WVaOKbd\/aglUB3mihywTFVR7hZmQ3VXFrV7D29Xr9sAwodHNMhyk\/259CFInlwl7THmU4+YHwPVrDlLOCYq9HAXvtAkWbjN60i3WOlT3UJrWVg9V8AurcirTRu2ZHnWCYmMqRw097cr9A9\/xhsPaNCko55jUVlC94Z2FtKmnSnWB41RjUyKSHvXj7cULFVSPqSOaNLUQs1rfqOgFqvVQgOaCTaDa251pQvW2dxcpf\/addUPo+GOnqEWPVdIB3w9UhFkY2N03Yer0ppL\/XGGhml2ozEOZzJ4M9OgfXn4DkgzVf3htIX3w3kqadvD+fQwD\/mdd\/yohUF2G8YwKqoOAUdDLdQNf\/YcjSFSQqC3VXh5a0MINFK2siV6h2q1utF8KX\/GoFhheNXVbbOnXbRURhQGb3T2QBy4YiBONBKhGAiTDZxqQi6THodb7sGrFuwqqkXDQixVUw53jrwvfKYI48gLSs7ks5euaaNpRkxRAL1u2TOUZTpnAUP3R8qU0tH2z6o9AdbAnkgnVeNNQ07lLuf587ITjEwXVdvfNQIfqgRz9w8tvd5Kh+qm\/zaWu9t3KuFCK37BgT4nylBKoLoPuUUF1Kbtu5xLBn+sgpG9m9Hr8tpeHjJ\/rtarPCi7tfvCcANIrVIeFUPN6g8JxVP2Ieoz0RY2VpZqhE\/3f3Jmj5oY65QLyYWpwURq2KivINjYcMgADvA+ZeLgqw1Zl0\/0DUP2nuYuKx5cjL9xJkOCPjSgiHG4PVnAcbY4NkkEs1V7jVItPtf0db2ep3kQNiYNqu\/tmoEN1JUX\/CPrs9fObZeZNMlQ\/+sxfqG3ndjpizHCBamPgBKrDzPqAZeOA6rhverv6zXB7pvXNtHDje9MSaSejl2tyymN+Z+XCov\/g6QsA9IldGsyFgVeojhpCvcCxufhB3\/VrCWJV8DIOAW+FYjG0wW8r8KEZw3t9ZyHCx5knn0DLWguWaYZm3lzIcajZ4gyfaiRA9RGTp6i\/eaMi\/j7ooIPU4S9IbKnWjy+HHzVSa1dOWcW5Pvz9yeOPsYRqxKnuam+jhkHN9D+mHaVAHynq6B9WY1KKw1+wydFtI6Q5F+KAx0r2qWYN49Al7H1YSvePSoJqL8\/esNpbQfWTry2hls5NlKltovGTj6NJzdWRNePH\/clsVHyq7YdBoDqyKeq9Ii9Q7Rdk4r7p7ep3swgz0OmWSA5Bp8e7NjdsoZyXa3LK43VDpJW1XYdPsx6vUO19RljnNOeAqbWT5V2vkUPR+e2Pmy5mfW5z1u17XWeckji4oa64sOHNiXycONp+8803e4Eu8g\/KZWh1awGmdRDGJkXEpH5v0XwVUq+6upoOPfTQXlCNeNRcJ8rCnaQrS71OWeRrBlTjpEVAiO5TjfpxpDniVJ9+wjSqTacIh7Es66ii5oZ6GjuhcBiN3Y+jV59qq3kPqN78zhuU6tztGPs+TPQPgKCbv2x\/hWq1EBtk7avt9d7C+C7bmaXJw6pVCEdsarXz\/7ar0+0+8toXq3ylhOpK2qgYtYHEyxhhLH7x0gLat229ii40cfpJ6rlhNf5B5oRAtZdR8J9HoNq\/ZqFLeIFqL0CpdyTKm94N1sywZ6YgpnXatJYiP6DV7Rrtrknvnw7terxoOwutVZ16Pxj+dRC1gmrUk8vlPB3cE+SBZy4qrNxo7BYN+qIF9bidgGg1j7g90y3DCdCdxlO3RFudJGmOF6B059aC3zHS9saRKvoGLMqAXyS2VLMFmaF6fYGp+ySA9aB8htbvLpyYyKCNvxmMUSdbq\/cdMki5jeinLLK7CfphBdVsqUac6qOOnUqj6tK0ePaL6rjxaspSTfNw5Z+94+35xZNJ9QWlG1TzaYn6xfECEFD94Rt\/V3G5rTTGZs6Pn3QiuUF1+7K9fTtg8nRlkedFQH+DahxbD6h1u0e9bsBz+3HA+LL10W1TpV1dbs9Nq3Jo9++vzXNdcMUJ1ag7kyX1pqPSNyq6zRe3eYDv3erQ50prvroI1VbjH2ROCFR7GSX\/eQSq\/WsWuoQXqI4Skr122MkyabWhzM6VwGteL9foBPg6OHhtk7WwAnP+zjzAxavF2w1OTbh1e6jq+qBuvV86pOI71G36tcOdgj93Wwhx3\/Vr1X3jvbjs6H0y4dt0E+K5Y7UAA2Q+N+tlyrW3qm6lG5vpo0xaRfiASwaOKdfjSRfdQNKkLNV2UI26htdQn8NfUL6+u506axqVNZut1RzfGuXM9hATu7O2SUG3DuSop74qRUdOmaqs2IDqFQvm0ps7sgqqO1M1Kub1sPYt6tp4XNjVCBZ6+IxjI+SgfJf6e3xuB22sH0mfnn406aclYuGBzUK8mVa3VKNuXWNs1MRJk+Nqe5R+uxqH03kzCmEJeZ7x5yPbNqq5hFMic0NH0xGHH6Esq7iete25PtZac56iTvQJkVrQPy\/wCNC3Ol4dbbLfun5\/wf1j0aJFtCObVpsTzzruCDUObP3FkfXQ1NxojKgFCLE4JNdB61JNNLKpXrnyWLmG6e2hXCbTRWPrcqGiHejWR3NTpd3zwPwc\/7+lrVNtpPXqzgVdFs+ZrTZzOi2y\/UC1Vb9Wt+dp8KBB9Onjj+7zs4MxXtuRVQs0QHWUGxXdnqVeINbr7yTyBYFYs363Oux8qvXfaf354ffgMUD1B0sWqqg5fM9avTG20kXcP+xny4CB6t27d9O\/\/\/u\/0+9+9zu644476LzzzrNVpbW1lZ544gl68sknadWqVVRTU6MeXhdddBFNmzaN0um0n\/uvT14vUB2qgYCFTfAxf5j5puXq+YFuPtD8Aq5Td+0gj8vo4GfW4\/SD49edhf19AYt6dBNdI27ftMric6sHnt21WUUcMRcgpsYMaLoG+njZPXCtfhitwuPZQYe5ODH9ovlhzZqZ\/eR+6RZxwOK2jm4aVZ9WunWmqmlrepCKNY0IIPgvElw09ARLNX4gANW664eeB9Dd3dCsDonhw18A0Xosa+SHtRpQDQjmtjhMHyzjgDK8koWvNqzT2EiJz1F\/Y10NTT3+BAWh\/KPFlmrUh3r3ye6ynPYcMhD1oy5EOjmwZxutrRpKE4bVUa6ttbh4AvRCi6G5DgVKAPBxjSk6bcY0+vtrc1U+aIK0cev2Yl2s5yG1BZM+a8+ft2R3qM93pBuos2Go2qQJ6\/Wa9qw6KRLHr5vzzOpiNlYNpqZcF+Xrm9Qm0\/qOHQpykXTgfXFOoW5eaOhzpK12ME2dOpUAyXCh2Yc6aVhDjYIynGYKDRCXfP+6nBp7bFSES0Vt27Zil\/Q3WDjgB8fIQzP0b1h9DY2pT\/Xy5+f29b6iHBLasXru8TXpOli9ocPC8EMaTMO3f9CrfZSzunf08WEYxj3HWrDfv9MzA9\/tc9R0Wv7mgiJA8TUAdJFgPUa9WHxtqBtBF550VPFSMJfe7axWcwkLMTvjC54Ruk7mnMCG31GHT1ULNIzfex+uofamkTRln8K+BqSgcOwGqKjbSx7Lm9LiQy8GIbfrcVvIWUE1v+Ey5wX\/v5+3koDqZfNeUQstnvP6b6aV4YnnmUC1\/UwZEFCdz+fpqaeeoptuukkp4QTVy5cvp2uuuYZWrlzZRzX4YV599dXqH\/4OmpIK1U4PCqvVMa9qrSy5DGb6zc56Wf0A6FpaWb74eztrrZ1LAffRDvz1lb5Tv\/RrBFzh+jAHrMDdtL7aPfCsLNFWDzereWa1AEI+8wcfn1lFY9H15L+xYEDSN2jq+ln96OsWbP1B7+feYOjhsHcAT4ZF1ANwgpXVLQG4m\/IZ2pnee2y8WQZ5ADb77bcfbdy4UcE1u49wXhPIrQAd0NuWqlN1wYeZ4ZuhuraxiVo7MlTTsUu5nACK2VIdFKpH5HZTfb6wSRMJ0JulNI3Itan\/RxvIA1AE3OAgHLQNfaEtA7oJz1yf+fnWdBPtThfCFo5q39SnPV1bLAYQmUVPDNWZVHURZK1ck7AgaUvXWi400AfAHCCZr0\/XgOfGiGybulbkZ2i2mi\/69+gfNAJguyWUQwJU27lGOdXB+uC\/W5r2UX6yXtpHHsw1vmZ+82SlBT\/L+B7WF6pwn2pK5VRdeoK1fkiqh3AqKPLzou7YIYUNtngW8NzBvYOFmOkShnzcL9aJ56TeFhZXB0w9RS3QUG9nNk\/ra4apNzGcrPbe4Dt+Nlu9ocP3VtGnzEWNbmRwMsa4zQW9P3peKwi16xeeq1bGJ70+fmsFfeD+gXtRX3ja9ZPvMV0z860N+rqyI1V8PnFd+hzSw9yyxvw93ki8u+RNOvmIQ+jMM8\/0ItmAyTMgoPrdd9+lSy+9lNatW6cG1g6q8Urxuuuuo9mzZ9Npp51Gt9xyC40bN45g5X766afpzjvvJAD6vffeS2eddVbgSZIEqPZiEbC6wCD+XH4t1yaUWoGi6ZONvpouBVbuIfxQ0AFRf6DYWbd1AMbDCm8zmpubLV8Fm7Cs\/wibDzxu2yxj9xaA8we1ulhZuPXrN3+grCwXVj+qgW+GPT\/IKI8fb8A0AFWHRYY7L20AGpygmuvAggjWsqCJoRr91S3agGpAHfqP7wBt+Bs\/iGxl16EaIMvXi\/yod2NVs\/qvaak2oRpghTLIh8UErPmcB5rhO7aIMywBPvE5FgRskWYNTKgG0KFv6DfyAuIByBwvXAcnK5hFeeRFHeifHbwCjHWoxv9DI0C6fh24XuiCxQlDphNU45r5zQD0h06YG7qlWodqXB\/6Czg3E1+LCYtWiwmzLHRFu9AeCYubfbKt6jMrqEedSLh+hmfWHJ\/he8wRjPXeOeU8lxngoUFhzhby88KHr5nnCd6OcOLPeB7o14fv9PqcoBpj1dEwRC3QuJy6jj0LIrd7UXdxc8vr9H3QesyxdlrEuPUPb+M2dWYdF3T6WJj3iFX9yGPOXSu3QH5raI69W5\/17\/l+P3hY31Nh\/dTTH\/P2e6gGEMNC\/eyzzxbHzw6qn3nmGQXVkyZNooceeohaWlqKZQDTP\/\/5z+n222+nGTNm0AMPPEDDhg0LNCd0qEa9Vr5\/gSr2USgolFlZs03gtFoV65ZrdNPJNcMOMM3Ls7Osu\/UR9ZgWEasNinYLDyu3DdPKbWpiWibcXtPZvWJlDZw0d3LpcLIsm1ZEtKW\/OrezFnmddm4\/ZvgRgUWaQZQBJg6o9tpnu3xuUA34RPq3V3K0rTNP959Z3weqcb2f+Vs97TcoRTfP2OvK8pvlPXTJId22UM1uH1+claLh9Sk6ZFiaPnVQNY1oSClYAzCxZgAWXmR8b34PfX16bQHY9oAy\/gZMch7988VdQ+iWV7tVG\/93RpvKd92raXU9SIcOS9MNx1YrAAa0IqE9ADL+y4f0IDeAGv8YdFlXABkvOgB867praXhtvhgjHN\/\/9+oU\/fOB2WIbWzvyNKG2vdguA6tpqbZ7w4GxY\/cPhlrOi7r3qc8rcGed+M0J+oc5ydCLzzFf8TkSYJ3LsIsS2tH1hS7LuxrVWCFxXxi8e6gQXQTtw3UG1wZ912aqqakGUWuqivNIX6SxNRv1MDDriwp+0zN7fUH5M\/btUfWifrSJecMLEH6jURyjPaeYmnMG\/6\/K71lYYgGHRQDqhU7oC+rGuKAvmCPvZ+rpoLpOVTXuEX7DwzrwfIRuXB716eCOz5HMNxb6wkxfmCCv+f\/6Z+gX5iDGD2PJCxf9vzw+GBO+nuLbKW0Rxu0UxrCwAOK\/+Xp4bmFBwW\/XeJ5w+9w\/HgsTqvktFeZvQYtudb9hfulvxEyNdOi2gmpe5OsLYL4Gc\/EEveTwF75L9v63X0O1DsLnnHMOLV26lGC1toJqwPfXv\/51euGFF5SF+pJLLumj1oYNG+jKK69U9Tz88MN06qmn9lXUwyc6VLe1tVn623qoJlSWoFDt1mhc9aJdJx8vL5Z3K1A1ra6636JuzdY33mGDDbt\/oF9O\/tL4Xq9TX1w4+WZavQb1sjHUtEzoFn1z7DivWcbOEm1l+XebD\/r3VtCuf8+WGT50BT\/SbJVjSHFrz6ul2q0et+\/doBqwcevsTBFAUd9JY1L0hSMKVlhc29l\/K\/gXW6VrJvYo8OEfSkARA7NTuesmddOZ+3Q71s3tPfPxdrp0TgNt70pRt+a5wZ9v6iyA3+j6PN1xTCfdu6yW3trhHlIOED6qMUXvbttbKep4dEYBlO5c2kBt3Xna2JanIXUpQivv7+ztOuKmP77\/PyfV0JCGQn\/Y2glt0NaDMzJqwWKVuC8XzWmi+04s+Fc7acp1HDU0S989JqPyfurganr2vb3WYVwzLzbw98yTC4sMQNJVLxO19xQWImibdcX\/63XqfUUdAG9APtfrpAnynzexhp5c1k0ntlTRCfumlDa8SPjyXwsRb+zSjDFV9MmDqmnNzh56eEmWMAeQAMGoZ+HOGpo0lIpvN762eJBabM1dX1h8II0bkqbWTJ7uPpFoVK5NzS3M9xnNHfTkhzX0+\/ecx\/iRM6oo29ZOG6qaafHmLI1pStPjS7vVImfpDqKfnZ6irZ15qmusV9eF+2i\/+myvN1P6m6IPMrWqLD9PAM9b2npoV4Zo1gddNH2\/Gpq1Jq\/qQtq3KUVfPaaaPtqWoadWF7TjcVy+PUe1aaJ3tubon8anCXpiofzYW930z4ekqaWuh3Z2ZOmRlTXUWE30xSNqaNW2Hnrk7SxVFW4jNZ4rtueKi1HUMWO\/KpqzLkvQf876LN318Tpq7yZqqe9RCxI89\/DWwITqD6qH0+JNWZo8uoq++tdO+vaJNfTNVwoLYE7oO6Ad1w8AvnFemr4woUq1D\/B+bBnRNRMKiwTUD1C\/Z0GPuu6LDy\/kQ9m\/bCg4dmF5gP5CY2iJZ6BAdd87ql9D9ZIlS5TbxwEHHEAzZ86km2++mRYvXmwJ1diQ+KUvfYm2bdumLNJTphQOi9ATjie+9dZblX82\/KpvuOEGxweV3ZdWlmrk9bqT2wtAunUsLvh1qzdM300\/Yh1m3dqFHla+zvjctN5agagOlHqc6vnzC\/6B5vjZuYDoFm0T8vUx0y3EbnPDqh4u7wTV3J4Ou1Y+rzpk8+JC19vJx9JtHurf61DN1jN8zxZPL3WVG6rxI4QfpRtf7QsQDF6Ah1TbbgUd+Gx0Y4o2tfeGJx2q+bpR93Pv99BvVxV+OFH2uqm1NG9Dtgh4AHHAz\/3LqtX3OpABYHd15SlXYAh64IQeuvq1vntDvqsAuq4If\/hxnXlMN33hxQLAov+PLc3TK+u8gXBtFdGgmpSCajvQ9TK2TnkePy2rAA\/gf\/yYKvr4mBzduWDPhRoFocs1k3po5sIqpRU0g15uCXn\/eXye\/s\/CveDiVsbte0D2+eO6Vb+DJIzpzoz1dVrV5zU\/5sDSnVX0i1WFjbpIPz2jVkH1zYvqHBdXvKi5cs7eORTk2qRMQYFfnJ5XVn0kt8W4X80wp79\/TLunRaVZN+6bm6ZUeeYWv32r1Pz9FqrZP3rhwoV0\/\/3301FHHaUA2w6q58yZQ+eff746Ze2xxx6jsWPHWo4p6rrnnnvo7LPPpu9973u9Xo97nQRhfaq9AKRbX9x2LweFXzfXiDB9t3J5MCOS4Lq9+EVzPjv4tdIP9XJ+RICBXy4WWm4bL1GXmy+6CfZO12E3tk6+3Cijg7rZHq7Bys\/cbIsXF+Y4m5tauJyT\/7XZHx2q8Wq1pWeHstTgNanXpPs3ey0TJJ+dpVq9hu2spn9\/xSZYNhHdeGyKHnsnp6AVAPLACd3KHxpp1fZuBYP3nZCjA+q6bCEUcHf6wQUfYE73vdFFwxsK1vBr\/tJJl00i+sS+BRcC9AmWMuT\/jz3WKAD5L9\/pVhAKcMbfsKT978NrFLwDyBnMD2xO0wethbbYionXx2xN4z6wJe7HC7voqim1ynoHCxoSoAzWuQOaU4RQ4d89qbAhFRb97Z15euz0lPK95vFmCyO7RLD\/OOpaso3o7kUFmEQfsUBgqNbH8\/aTChZjWCw3dqaK43LWQdXqGnHdSOg3PoOuSHitzv7jsMpBTz1ddGQNzdgnRdk9rg\/oI1\/7tu4U3fzy3vGHvrBcQvs\/f5SnFz\/sUYshzBH0He1Ce7T9mfGk\/N1xrW9tydHTy7qVNR3uCegT3lhg0YaxxBxEHzG2Vfkc\/XJZlj55YIre2Z6i+RsKFmR9UXXD1Go6Zki3qkv3NUe\/bn6lu5el\/fsnpWnR2kyvxQZcfT7e3Ep3L6unlzakCdrCooq6MM6rd+botytzNHEY0Zcn5ej+pdVFSzZb79EWrKSY47iOqftW0WcPTtMvlmXplbWFPo+oT9HXplbTq+vzyrUJ4AcrLsZLt\/LzeFhZ9c0FJeedNCJN+zbkKdOdo2y6mj5xQIr2G5RWFvA\/vJ+jldsLc\/z6Y6tpdH2OrpxV6BPqG9ucpne3ZWns4LR68wCLMtyu9Htl8ug07duUptc3ZNX1feekGmrcE0VoZ0cPrctUq8+5DOqGjk+8naHPH1FHP1rQRVv2WM65z7gfYSFWbjXZtuIzgd9kYP5g7mGe7d+Up0Oa89SWr6H7F3apz7n\/qRRRPl+YE3ju4A0V3i6gHOYfrvHyI6to\/JA0\/fjNHlqzq3AvoJ3DR6TpDyt7++3j3vnJjKxAda8nA1G\/hGq4ffz0pz+lu+66i7785S\/T1772NbWxzAmq4XN97bXX0uTJk+nRRx+l4cOHG1IV\/hdh9uCjjTBP8LsO4lcdFqrdgNiy4xYfOoFzUPg1wc50jXDquxUUwhc5yn66LRbcrpu\/R5hFpO7ubsuDNkzQtqrXapGAOu02KZr68JDahdPSQVf3GUc5jmGtt2f1JgDfW1nN0Rer+u3cZ6zqMdve3tmtgIpD5SFeMuBiR+OoPhE6+No5L0fwcINqu1B7Xu8ZzucE1Qu3VdEDC7uKr3mXrd3dyxIJoMKPKn7w2BWB\/RZ\/80EV\/XJVjfp8n\/qcrUXwXyZ20TH77fFjzrUp2IDlW0\/IA2sS+9YCyLBQuXVRTa96T903pyy3gCDAkp7wg28uEBiqkQ\/uHAAl1I3\/wpcWr5HhUsEbC697Ld0LhuAGA0s8QzOgTPeH1X1m9Q2QZsQLtANrM3ScPipPq3dRH+jC9V82KVXc0OlkKT+9hej6QwtvnfSE\/vyvF3tbkvlNAvLpvresCS8MeGMo4FffqJhp76QbXu3dDvqKMePx4qgx+gZVfYMfuwPxZ\/oCC31il4FCn2qK\/tv4f950x361AF3APLRkSzzeLkweWXBR+enStIIuQDS7AGEhs19dAbQ48su\/LWpUrhqc2L0FfUHCAg\/Wbl5A4zN23cF4Yk6gH7i\/OISmri9vlsUiBgsw5W9NVUX\/7IJvecEHW8H+ns2n\/FnB57gAmrq\/M\/t843Pdn1j3XWb\/do6gwi5phYVCwX8a\/+X5y59bzSe9bxhHTuyjjmt6\/O3OPvcjXE54gQd3GN7cjLZ4Ayz8qHnfBfuK793omlV64XmB+1JPeA58feLexSNvGGZfat40zG\/i8Bw7e3RGoNoY4H4J1bBOw5VjwoQJ9KMf\/Yj23Xdf5dbhBNVeYXnWrFl02WWXucJ3nyez9kFYqHaq2893TgAZBbj7rcMuSkiU\/fQKzbyzW990qYMkdNZhk63YOmhaRc5AOa9uPmZf7SJ3WG16dMqLPlj5gZuQbwXe+qJEjz2t62FGZtGt4k5vEdZu21mEapxcuHXBy7SzK6fiECPpoe\/0GNXIywe26HGsne4FvGVAnGoznJ7X+8cOqnGwy4XPFcDshP2q6KqJhZBt8MOFZYctRxOGpQl+mjqgokxVXQPdsqhaWf3YBxOWJFhhYVWCxeid7dX0o8kFP0s9EgYDDVvpnj6tS4XX00OlwT\/zg45qumZuLY1sSFFHT56eOKnQX8DAla\/VFS2WsBDihxM\/qvxDPm5YvWrbS9JD6s3dVkM\/WVTw633i5A5qqt7rsmAV2YDvLT2utFUYOSwkAAjoK\/oJfQEe+OE\/sK6rGG6Q+wsw+NumavrvlQEm0y4AACAASURBVAVLfE2a6LDhaRpWk6d\/O6wQncMq8cbPzizR91\/rUK4sbgmQhXCAiGqjgxxvPoNPt25JNucC6re6ZqfP7fpkF77PKvqHufAA7P91Q1\/3GKv+mr7pDNXoF8aZoZr\/H4sxPdKIm6ZO35vROcLUZVXWarNjmDZ48WlXx6yNNfSDpXvdb8x8pv5+r9\/L\/LPq27rOKoJrD9JfPpn3\/HsWRqtKKtvvoHrz5s0qgsfbb7+tInScfPLJajy8QrWbWwe7ibhZtJ0mQRxQ7WaBteqPX+gNM7HtIlJY+Rjr7eguF\/icw9N5cbngeswNjvpBLvoJUgzNpluC6fLA9Zowa4Ks05HedlqafdX9te0s1ajLhFXTz9qM4mHlB673yW7xYefXzmXd4mRbjSfKoj9sqcb\/A5qHtm8mHBiBH2LEBufjwxmo+chw5OdTFavyWWrvKljPnKzSgOpDDz2U3n\/\/fUtYd4Jt1IuDRjjihQ7y+Js3heG17BcPzKhDbHAwy0OLOtSmJD29+tmGXhb\/rhzRvy1upKOH99CLH\/TQ+ra82uD4yvoChMI6CLDFoR+mC486ibGhhjbsaOsTwk4Pz8Wh82DpQ3\/RPzUG+TSt664pRg\/AhiQ9OgMAaFR9lTpV0FxQWc1nABvyt+XS1JUphHLjaBv65liGapz2aC7A\/vvF2dTe3qE2U3HMZiwUEA4Rb4k+zNTStxbW0H5D62ndjoKV7Z4TC1E10HczxBhb29APWERxkEy6bbsqp8eC5uvh\/nA5RH9obGwoxnXWr9squo25R0EfBwZ1xBoZmd1pGcoPeRAbG0l3pQJsYxz41FHzbZLZfx6LdJtmRt4Duhy+EGW4f7pLyHOfyNNZ\/6\/gRz6sPqVcdeBzfdTQva5H3D4O3+murqOO9kLoQ\/gA83e4v3Ev62EeYdHn0I5u0YHsnpnmRmtTK7OcW34rNzn9mW8XS1tv160Nt3nDY4GY5jiYSnfPwXf6YsVOF7s+43P0DzHK1X1JVeo+wKFOTtGh9HaweKrt2U3Pnd0kUG0MQL+CajxoAdL33XefcuXAPz6kpVxQvWbNGsI\/PQEAsHES7iMIzxdFmju3cIIa3BJOOumkKKqMtA6zf0H7G6Sc17bd8uENCGACB4YgwdKpa83f48ceKchYBLk+p4HiPiGPn3lhV07\/HPdWkDnH18gaoR4cAFJfnVKWao4frV6d19TT2MMnq0sEAMOHHQnHiVuluqoULV3xXjGflTUa7WEMEYMeB8CgTm4T36HuFStW2FqxkacxnVMWaViFv39qPfWkqlQdOlQ\/ec4wqu7YqQBsyNEnUm0qT59+eAmt37n3Feuu26aTPubTTjiJRsyc1+vSYBXtbhpCv18O94CCry4WlThRUJ9nS3bl6YD6PL23ZDFt7+hW4K3PRRxDrmCcqmhjrl7BDLsxYW7s6iF6fsE7Ko4wPkf+rjxRa3XBbWR7qp5y6Wr6pxmF8dDnArejdxzwPrg6RS1TTqSVy96hho4dvfrD18BQvX+6Uz2v9Tmlt6HXfcQRRyjDybzNpNxZOI0eVEM\/OilfPORE7xeuaXOqkainqxhSD\/0bkd67QNH1QJ08NoDbmlqECiycJgm9rO537j\/3B\/XpB41051PqNf247HbVF+zNQJ1uh5Ho7aHurVWDVDm3exB9xDxRh+jUpNQiR583qEufIx3d2V6HBDXUVNMJM2bQBb9+n55\/Z0tRZ8zJlsHVvcqijzgUBGnjW\/OKFnp+Dr48bwGtz+QJ48xt6iEAkU+\/Hqv7W++r\/repg\/6M0ceC\/9b1Rr\/1e1B\/VlqNsTlWVr8BfI+y8cIcP\/PauM5e17dnroxNd9LFs7rVGxnsxxg\/rIbGpQtvVfT8ep3mvLeaX6OPOYlWv\/UGpTrbqKGm8FtmPvfN6+c2MJ\/WbW+l9JplfaKg7b\/\/\/oR\/AzX1K6iGFRkh7xBn+oc\/\/CGNHj26OK5eodrNV9qv+wcAH\/+sEvoIi3cUafXq1UWQgNtL0pLZv6D9DVLOa9te8yH6B96I4FQ+QJuZgvSR6whTttRjHrSvejn0GVbozT1VlMp2U93YCcWTDodThtryaRrRcmDgS+NTE80KGurrqa6+vnhstp5PP23RruHWXR10w8sFwP\/l+QfTsKZa+uIvltHdpzfRJc8VwOtvV06k4TV5GlqVU8dK16aIPvdf79Gm3d307OWT6KjGgkVd1+PAQw6l5Z01dPnTq9SmIgD4y\/+zhlZ216kjuo86dHyxS6b+aAPWy6b0XtcKqzHanSVauHYb7dNZgCR+Xlh9jk13W7JVNKG+hzZ2pZQls6W2t7XdvIY+Wh84kbDYGV2d7XWtaBf9A8jhDQPPBV44OT3HAIa4B3dk0\/TPz2dpzJB6Gj2omn58dkuxLat+NRw4kbasX0tNXbuUpQ6nDI6qzhafnboeenlYYHGvn3Dwfo73Ox9lr2fSrwNHT2N8MfYf4dTLPboEneBe7kHkWdNVRU3pHE055EDLMWDNcQz96nxTr1MOeU5e9tQq+uwRQ6klv42mDS0s7vTrxXXimtKpFHWv27uwZU1Xrv6QtvcUNlhiMYE2zZOJzfqcnq16XrOc+YzheuzmlJuObt8HHT+n62PdB29aSVsyOTXP8SzQ7w+7fpnXb3Xd73dWUduGD6muqwDoTvebWd\/SrW3U3p2ln36rbwQ0eArg30BN\/QaqN23aRF\/96ldVDGnd7YMH1g2q49qoaGWpfumll+jBBx+kX\/3qV5FZqgfqBC7Hdesh9cIcV1+OvielTcBFXXpvaLK17QWoWtux95VyS0OatmTyhLxRpy1btlB9fT0NGlSIuuE3rVjXSv\/7\/y5WxebedArN\/ONyevatjXTsAUNo\/c4MHTaynn58wdE0qi5NI+vStHh7DyG03K7uvLIY4toPGtR3QYZrRV49TR9Roz4bP6iKmmvsw7l5yYN6W7vztGp3liYP6+0na\/U5fszXtudUXowRIBDX4ye9vztLKNLS6B7j2mu9iO6EswUaRrdQof6CLgVt7dvS+7K0tYeaYUn30C+UQ7IaM699Rj4eX4xpHLrY9QXj6GXc9P5xXfwZNDbnjNmeV53wVuejjz6iMWPGqPtQUl8F9LHA+G3O5GhSs3voR69aYv7z88lrGc73pzfeoeuvvZq+ccVFfd4aiqUaoTL6QWJfZz+XovtFI97wBRdcoGJaJz2knp9rlLzRKyBQHV5TE6r5xxg\/HpwASF3ZAohEnaKEavStqbaK2tDZPenjBw+lR784WUGwFVTb\/TjyDynK4N+y1h5KElR7AVAr0Iobqhl2sTBIMlRDGyxOoGMpodrr\/SNQ7VWp+PNVAlTfP\/ObdOaZZ8YvRgW10G8s1WGhGqvmiy++WL16LuXhL1H5VFfQnKv4rgpUhx9C\/GBkskR1VaSsjLCaIMGSq0O1\/v\/hW91bQ9RQbfbtvdtOVVZNXjyYlmo3qIaVfnCNQLXdmJuWamiNRYgbqJbbUq1fj1tfo5zvXusSqPaqVPz5BKrj1ziOFvoNVLuJ4+b+Ua5jygWq3UYued8LVIcfE\/xg6K4Q87Z2K7jWrdLm\/4dvNTxUH3\/H35W7x1f+601a8GFho6CZPnXUPvTDz07s5SoQBKpH1hfcRpJiqfbqQmDqEQc8ClRHeTfsrUugOh5dg9QqUB1EtfKXEajWxuCZZ55RDvbY6IjIHC0tLcVv4SUDC\/btt99O06ZNU6c0jho1KtAIxhFSL1BHpFAgBQSqA8nWqxC7ebB\/MaC6lCmIpRqbBs\/58XzXbtpB9ci6lPIZxzV7sVQnBarZl9N02XEVYk8GgWprpeLQxeuY2OUTqA6rYHTlBaqj07KUNQlUa2pzjGu4ksDf+tZbb1X\/bWtro8cff5zuvvtulfvee++ls846K\/A4CVQHli4RBQWq3YfBDcDgVwr36ZbGtHIDge9wKVOcUP37q6bRUaMb+1iqKx2qg45PHPAYlaXa60Ytrxvw\/GgUhy5+2rfKK1AdVsHoygtUR6dlKWsSqDbUXr58OV1zzTW0cuXKPuOASA9XX321+hcm6oNAdSmnePRtDVSodgNlXWlsGHOKVMGQgjyZbL5X1I\/oR6xvjX6hGm4fZkKkD7iALPrmx4puK+yyAv9ePVIE3DgEquOL\/hHUp9oPVEe92VKgWqJ\/OD3rBKpL8UsQfRsC1Raawv\/617\/+NeHo8lWrVqmDEHAS3EUXXaRcPxBjM0wSqA6jXvnLDlSodgNlfWRM\/1sTyLExEQDK0dn0UHqlGGE\/UI1QeQiZZ6abPnkonXPMvr18wQWq+45eHPBYDku1QLV7SD1+A+UWelBC6rk\/5QSq3TVKYo4BA9VJEl+gOkmj4b8vAtXumnHYMM7pBNVwA9FD6bnXHj5HFFB966cnEPyn4SPNUUoEqgWqvc7OOBYbXtu2yxfW\/UOgOuwI7C0vUB2dlqWsSaC6lGrvaUugugyiR9ikQLW7mDpUc0QP\/bAXtlSjJo4E4l5rdDn8QDUf7GK2jiggSDpU89\/i\/rFXrTjgUSzV0d0Lek0C1fHoGqRWgeogqpW\/jEB1GcZAoLoMokfYpEC1u5heoBq1sGU3rnjUdj31A9VW7h9nHTGa\/v2fDhOodp8KrrGjPVTRJ4tAdRDV3MsIVLtrVKocAtWlUjradgSqo9XTU20C1Z5kSmwmgWr3ofEC1TiAEJbdpFuqv\/3scvrjmxt7XfRt\/zCezpk2VvVdt1TDQg3fW7i0yEbFgmRiqba+X+LQxf3OdM4hUB1WwejKC1RHp2UpaxKoLqXae9oSqC6D6BE2OVChGhDgtgGJZea85smJ+B6f4Xs+1bu2qvdJihEOlW1VfizVVpE\/BKq9j1Ic8FhqSzUWiUhBjmm3UyoOXbyPinVOgeqwCkZXXqA6Oi1LWZNAdSnVFqgug9rRNylQ7a4pQzVvQKxNp4oh9hBFZG1HAapNS697zdHk8ArV5oEvXzpxLP2\/tzfT\/BuOpy2ZfL+3VPNY2R1W42U04oBHgWovyvvPI1DtX7O4SghUx6VsvPUKVMerr2XtYqkug+gRNjlQoRqbC2Gp1jcc2smKvAAxhmrEomYrH0ANh72wP7XuPhHhMDlW5RWqUYluqcbBLmOG1FNLQ1qg2uNgCVRbCxWHLh6HxDabQHVYBaMrL1AdnZalrEmgupRqi6W6DGpH3+RAhuqWhirHQ11YbYZqfm2OzxmqAdp8+As+TzJU3\/n8SvrtwvXqsgDTgGp1LRpUw4+aFw\/9zae6P1uqMQdx+BDGzC2J+0dhETx5WLWjVBJSz20mef9eoNq7VknKKVBdhtEQS3UZRI+wyYEK1TgVEMeKu0EI+0wDwOHmwZZt9sfGD69+2Es5oBrW58tOPoAuP+VAx5nxlf96U52aiHjUiEvNSaDa+w0Vh0U2CvcPgeq+YxiFpVpfQNvNEjn8xf3+Eah21yiJOQSqyzAqAtVlED3CJgcyVAOA3TYrsnWToZo3JE4cUqUOSTGPJY8Dqh\/++weWwHzOj+cT\/KSRrKCaXT04BjX\/v0C1s4XS6fYSqLZWJw5dwj7mBKrDKhhdeYHq6LQsZU0C1aVUe09bAtVlED3CJgcqVM\/b2q1cNZw2rXG0D1ioAdW67zReHSvrYHWql6WafaujGiK2LvOJh3q9un\/0sQcMoZ9ccHTxaz0eNSB66oFDiseTC1QLVHuxwPqZwwLVH9GYMWOovr7ej2wDJq9AdWUOtUB1GcZNoLoMokfY5ECGaje\/SiuoZuntoDrCoVFV8QmIblCNvGyR\/pcnl9Br728vdoVdPVAXEuAbEM7Jr\/sHfMx5MeG0MOEfUtQ\/sj5NcLmZPqJG\/Xf8IGd\/di950H+0sWxnto9\/LN4wrNrd93M\/oRStxhL17urORRqOTtw\/or5rCvWJpToeXYPUKlAdRLXylxGoLsMYCFSXQfQImxSotrda6lBtWqRLBdVsqdahmYffjDnN1mq9DPLCMv3po\/chfB4FVLNVEr7k5YZqXA9vJNVvi7igGnNiS2c8UD1k3\/3V2xC4JMHX321Dod4X8anu+1AUqI7whyJkVXFDNe6VuipvG3XNS\/nTG+\/Q9ddeTffP\/CadeeaZIa+0fxUXqC7DeApUl0H0CJsciFDND3g3SzVHwVjbnqORdb3dPCY2V6uNiyZsRzg0qirdb9q0MPuB6jFD6uiR2R+qOtmizX31a6mGLuxLngSotrI+20E1+u62OdVtDKOoQ2+DLdVjx45VVnzeQMs62x3SEhSq0bZ+Sqjb9Xr5Xtw\/xP3DbZ7o8f43Z3KOrndudZnf437vyuUD3dsC1fZqC1T7nYkR5BeojkDEMlYhUG1vqWZLIQ5GMaGaQ8\/FsTFRnw46OAeFaliw8a\/cUM1uGoBF\/QAdq+nv1f3DDhDtoBog6iU2eSlvSTuo5jclCJVnl3Cd+B7\/rasiz9cmUO0tpB50dzt5UqJ\/eLtb4oRq3CvYOB5kwSxQLVDtbQaXKJdAdYmEjqmZgQjV+oEtdrFqOZQeW6L1+M0YinJANR\/WwlPB6shxWKFN9w8ANdw\/4FM9bkQjPXnF1F6zqVSWalj84doQNVRb3Rp2UB3TbRSqWjuoDlVpiQuLpVos1W5TjhdyuP+jtlS7te30vUC1QHWY+RN5WYHqyCUtaYUDGaohNDbOWSV9kx18h02oZgt11NE+zL7o4KxvVuQjx3GIywFDa2juB7tUUVizEYIP8ag56ZZqq9B7gOrWnr2WHvPwF9NaB2BFwkZAWPCd3BOUO8OejYoM1dAWycli7MdS3Z+gGv7ho+rc46eX9CHhobFKg2rcz27hNN182lkWsVR7mCBaFoFqf3qVM7e4f5RBfYHqMogeYZMC1eGgOsKh6FMVgzN\/AajmCB78GaD6u2ftTxc\/vrJYHp9x\/Gp8CKg+77gWuvG371jGs\/YL1aiTLcFuB+hwpBBE\/2Co9qIZIA3g4+T64FaP1QZGtzLl+F63VFcqVIfZKBaX5k4bFQWq41LdvV6BaneNkpJDoLoMIyFQXQbRI2xyIEK1frS4m6Xazs0jbgs1hhjWZrhyDG2ooR0d3epocR2WkccKqs3pARh\/fskmmrd6R59wesgbBqpxCI6TxVm3YPrxZ\/biIuJ2G4QNn+dWf1Tf9weoDuPTGpWOZj1W4Rb1N1BuvtJiqY5nZASq49E1jloFquNQ1aVOgeoyiB5hkwMRqvWjxeFTbQWF\/OPLbh7mhsRSQrXTcAOqH\/ncePrUw0tts+kwbhXv2g2qB9dYW4y9QGtQtwCB6nSEd3n8VSURqnHV5hy1sl47qeNlQ6e4f\/ibXwLV\/vQqZ26B6jKoL1BdBtEjbHKgQzVC45kuBhxJAX695YRqPvjFabjhI\/2ZSU30n\/M20W8Wby1mBUjD7QOWbd2\/2gyn58VSbWfRixOqo5jiHBkjirrirKM\/WKqhTxL1tpqjXuYtj7dAdfQzP45Y72F6OW\/BIrrwmuslTrWFiALVYWZWwLIC1QGFS0gxgeq+UA1LCmA6KVA9fdxQ5bphlXCwy1UzhtMvX99Ejy\/sDdWIFuJ0lDnX52SpdvJr9gIcQS3VCbk9StKN\/gLVJRHLZyNWMcX9QLWXmORiqfY5KHtOu0xKaEuBavvxE6j2P7dDlxCoDi1hWSsQqHaGanbzMN09SuH+wUAMcH72rY295gncOL79x+U056ZTaMuWLZ6gGtZrgLaZTKhGvFdYkwDUTlDtxTLp56S\/st4IZWxcoLq04vuBai89E6j2olLvPH72V\/iv3V+JOXPm0Bev+Ar95J675ERFQzqBan9zKZLcAtWRyFi2SgSqnaGaByZOiAY8m24ZvEkR7dtBNT5HAlT\/dUUr3fO3tcV5ZHVkuRNUZ3Kk4kcDoqOGarfQZWWb\/AlpWIdqWYTEPyheFoN+eiFQ7UetQt6kQfX5559PDz30kEC1QLX\/yRx1CYHqqBUtbX0C1dZQjRP\/lrX2FAcjLqiGBZrD5OlgrX9uBdV6vGlAdX19PS3fllXRQpAYqnW\/bCt\/auSFpRoJ8bjhY47402ypdgJiL3AStVWwtHdHaVrTodqLS01petV\/W\/Eyb\/1cvUC1H7WSlxeWaoFq63ERS3UZ5qtAdRlEj7DJ\/gLVfo5p1qN\/WG1UhMU2k80ryOQUF1Tr0KtH5tChGoCsbzZEn6ygGj7VfBQ5QFy5iDy7nP74ZsF1xIRqviYdqhENBceJA6rxuVPYMS9wIlDtfrMKVLtrFGUOL\/PWT3sC1X7USl5egWr7MRGoLsN8Fagug+gRNlnpUM2vEf1ANQ7YgIsDkh1Uoz4+WRD5ygnV+qEvDNg6ILOlWodqtlTjdEWAtpXrhw7VdVUpFX4McbuxQRO6wkoNdxC75AVOBKrdb1aBaneNoszhZd76aU+g2o9aycsrUC1QnahZKVCdqOHw3Zn+AtUcscNpRzkDuBeo3pzJFcE7TqjWo3OwdRnt6ZZqhmo7n2iG6l3Zajrnx\/PVHHjtGydTOpVSR5a7QTW7eDBU47\/Q02rBoU8wL3AiUO1+SwpUu2sUZQ4v89ZPewLVftRKXl6BaoHqRM1KgepEDYfvzvQHqMZFb+nMEY7CdoJq\/jF1g2q4h2zJFCJgxJ3soFrfqAirNPK5QfVLq9qUf7aZz64sx+A2oZrdYwSq4x79Qv0C1aXRmVsRqC6t3klvTaBaoDpRc1SgOlHD4bszOlS356v6HITiu8ISFwD4ZrKFyBUtjWn1t3mYC3eJo1voUG3l4qD7XMd9OTpUoy1274DPNCzMuu+0XV9MS7UdfJvlBarjHl1v9etQHTXweevBwMoVtcZiqa7s+SNQLVCdqBksUF2a4YgrBFGlQLXVDyFbkhmqAdOI2mEF1cgLV4RJzdWUFKi2OjGRjxTX4fryUw50nGRRQDVHO4FPtVdLtZc5Ke4f7s8HHardc0uOsAoIVIdVsH+VF6hOCFS\/+OKLlM1mB3xcQ4Fq6wkZ9YM76vq41yZU11UVNuUlLenXz9Zp9BVAjbS2I0vN1SkaXJO2hWpEtZg4pErBNW9UtLJUs09x3Bqwv7PezqjBtbR5V5dy4cAR4\/d\/4SiaNm5oSaAaofQQ\/cMrVHvRR6DaXSWBaneNoszhZTHopz2xVPtRK3l5BaoTANW5XI4effRReu+992jmzJlUW1ubvJlSoh5VKlRH\/WA15faycc7PELlBddDrMaG6a89Jen76Voq8evxehmq0i\/7CyuoVqscPqlJ5kwDV2FQIcLYKmcdHk\/tx\/xg0aJDyvf7lpcfSoaObXIdFd\/9gDWHJx9wFDLv5VLs2AF\/3PQfKeMk7UPMIVFf2yAtUV\/b4CVRHANWzZs2iyy67LPRMOOigg+ixxx6jsWPHhq6rUisQqCayAl5ACXx8o7L6OkE1u0EEaStpUG23ONAtntBiV3eOEAYOCTGlW3vyylKNz6zCwKEMDnMBKCYFqtmfWg+ZZz4H\/EK1l+cIh9Jzguq17TnCAsTOP91LO2pscvnI7gGvbVZaPoHqShux3v0VqK7s8ROojgCq169fT5deeiktW7Ys9Gy477776Oyzzw5dT6VWIFBtDdXw221pCA8lPC\/soJrjMyNfFFANy6KfY6Wjhia76zShGv1k4GOoxvXjMyeoxrXp4fLK6f7BUG11YiKPu90piPrzgn2qYan2khimdajG3+xzjjGAK4hAtRc1w+cRqA6vYTlrEKgup\/rh2xaojgCqu7q66NZbb6WhQ4cquK6qqlK1wq3jd7\/7HQEUr732Wjr88MMtW9u8eTN9\/\/vfp3PPPZdOOeUU8vpjFn74k1dDpUK1n8NC3FS3esUdJVSzuwOA0YTYqKGa\/Wrdrpm\/D2Mlt2rD7phmHaqhtw7f+Jt1wVklVqcAsqUaEA0rLPfb6tRAfSOjVx2C5PNiqf79VdOUf7VTihqq0Rb0xuIkrKU6iC4DrYxAdWWPuEB1ZY+fQHUEUI0qnnzySWppaaGTTz65WOPSpUuVj\/Tdd9+tvnNKS5YsoW9961sq78EHH1zZsypE7wWqSW3sMkEuSqgGECLZQTVcIdxiNNsNsen+USlQDWvzqD2n\/eFvJFiqo4DqeVu7Q9wR3oqacahRin2s9RrKCdV+3lh4u2rJZaWAQHVlzwuB6soeP4HqiKDaqhpYqQHWN910E6VSzhEQtm3bpqzc5513Hn3+85+v7FkVovcC1dZQjaOeo3h9jqEBVPMGQitLdVRQvbErpQ49QQQIr0nvm9cyTvmwGAHMsSuLfgoiNtEh8eEshbjUBX9qr1ANy7R+sItpqUZ7GLu4E0M1HyeODYt8GmIpoLorS1RbRWrDJvTW3T\/QvkTtiHsG7K1foLp0WsfRkkB1HKqWrk6B6oRA9YoVK+jiiy9W\/tQ33nhj6WZAwlqqVKiGC4FdTGO\/ElsBSJRQjfrZV9iEalwHwNJug57bteiWakD12o4cIVax18TtW7lceK1Dz2da+Nm9BeHwGPYZqgGF2JwISMbfXizVJlTDxYEtsuxmgw2NcSd2\/WCo1i3X3LbXQ1yCuH8IVMc9wt7rF6j2rlUScwpUJ3FUvPdJoDpGqIaV+uabb6Y777yTJkyYYNtSJpOhe++9lx588EG65ZZb6JJLLvE+gv0sZ6VDdRQh5EoB1ezWYAXVcH8AUHp9Xa\/XERaqAbhIcUE1hyY0oRrwD+sqrhvWVkA1J7Zo67ca+1QDqlGWkwnVqAvjGWfST1E0LdVeQVrvn1+oxjXjOkfWpZSVH240+Ex3Y7LzbY9Tl4Fat0B1ZY+8QHVlj59AdYxQjZvjnnvuoT\/96U\/K+nzqqadSQ0NDsUV8v3z5cgXUgMnBgwfTz372M5oyZUplz6oQvReotn5VDks13BOsIlH4lRuQZwfVgB+AURRQ\/WEnrL7+LNVoHy7NXoGer90qagjAF+Hu9KgpqB\/+4nZQjfpgdUVisNbdR\/A5u6ioMIdlhmrzwJdyQTX0h5UfY8ehH\/UNt25x0f3OYclvr4BAdWXPDoHqyh4\/geoYeyTvTwAAIABJREFUoRpV7969m77zne\/QU089pVoaP348NTc3Eyx6OOwFNxAnRAjBv+pq7z6olT39+va+UqEasMZxjsOCL1wWTOtoXFBtRhphVwhYbb2CrQ5MuqU6bqg2I3aYkSVwbbC6A\/bY8h0Eqs1QhmgXdeOfE1Rz\/Gvdkh3F\/QrLNIfGM6Ea4fQQpxrp5Dtn0+wb926c9tp2EEs1oBo66SEUzVMrg4Ro9NpnybdXAYHqyp4NAtWVPX4C1TFDNapHaL1XXnmF7r\/\/fpo\/f36fFseMGUPf+MY36FOf+hSl0+nKnlEhez9QoVoPJVcqqIbFtvDafu+c4yO3\/UC1DuZWUA3fZa9AhWv3aiV3gmq2JiPcHdwSvEC1uZmRrdUmVLM1H9qZUA3deEEUB1TP\/ONyevatjcTwjL\/xGScdqoPeilFBtf72IOr440GvbSCUE6iu7FEWqK7s8ROoLgFU603Acg0L9a5duxRAjxs3jvbdd98BD9OsUSVDNa4h6Aa\/UkI1wBXWW0D1ls5cL\/9lDv+mw6HdLcJQW0lQzRsX9VB\/gGTTmsyADRcQHaoxTtCMI364QbV+2mLYn4qv\/NebhA2ISAzPDNnlgmroVPBFJxpcg0go\/g77CauJlO+rgEB1Zc8KgerKHj+B6hiheufOnfSLX\/yC2tra1ObDUaNGVfZsKUHvKx2qIVGQTXb6gSxxW6p1qIYlV3fzYKgGLMHC7GRhZJj2C9VOdcLNxcpKblVGb9d0Y2HXD\/hH65ZqK6iGdR759aQfvW1CNTSDlbrg8tB7o6JpqY4SqvUNiYDqz0zZly77+eJe\/YbrB74Lk\/xYqgWqwygdT1mB6nh0LVWtAtWlUjqedgSqY4TqZ599VvlII\/3oRz9S7h2SnBWoZKgGl9kdFmJ11ebrcYCaGd+XywF2rY7ADjKfGKqtLIteoRp9Z9cRtWGvMa3cNry4f7hBNazDpk+5VfSIoFCN6BQAYw6pZ3XioR1U85Hb0N0Kqnkxgu91N5Eg46SXsdqQuH5nhhCPWk\/lgGosWvCGBnPXXKSFvW4p718BgWr\/miWphEB1kkbDf18EqmOEaoh74YUX0ic+8Qm64447XI8fx3HnSLW1tf5Hsp+UqFSoZl9k3SrqNiTmRi47qOYDRKyOwHZrw+p79lvm7\/S4ynxQCeBw4pAqymTh0lKI2awn9IkBSj9gRYfqFe2Fw0AmNlf3Op7aDqr5Oq1cTwCoWATomxGdoJo3XDL48tsDvnY9KolfqNbjTpuWahOq9YNhgowVl9Gt1PiMjxrXoTpI+DyrPvmxVGOs+CRK+ObLIS9hRjmasgLV0ehYrloEqsulfDTtClTHCNUc+QNHlF9zzTWupyriBEZsWpwxY0Y0o1uBtVQ6VAfZ4Mf+1AzVAFv9wBSGTT0GcpihBUTCLQJ9RbKCanyu3D+0EMs60OpQjf4CwE1LddRQbfqrxwnVrC+uST\/JEm3qcacxJrrrCEO1HiEkzFihrNXpiABoHah\/csHRhHB6USSB6ihULF8dAtXl0z6KlgWqo1CxfHUIVMcI1ai6o6ODHn30URo2bBide+65VFdXZ9kiHoS33norXXDBBQLVV1xBv\/rVrypKB7bwevUHxiRgKGSoxgY4bB40oZoPGvGyedBqcpmWYbbMctQPJ6hGn2AhxqE2fDgK2mBoRFm4jHCEDz+WarNfTtfJIQv1SCW6S4jpU637SesWfq+Wal1H3e3GhGpogsWQDuFYYEA3uAOZvtpBHvV2R47rdXGIvSD1m2X8QjV8zvnQI7FURzEC4eoQqA6nX7lLC1SXewTCtS9QHSNUs\/uHHovabbgqDSbdrsfv95VqqQYIAxKtrMl2B1\/whkR8j7SrO9lQrR\/DzgerwPc5KFSbujCw8uJBh26ro9U55jSsw\/rf0NIrVKMNdt2xm6sM1exHrluq7aAarjG8cPJ7D1jlN90\/zDxJgWpzcRPFtUsd\/hQQqPanV9JyC1QnbUT89UegOkaoxsPt6quvJojsNQlU\/4WuqFBLNUMgb4DjMbeDaoA48jJUq8NEGqO1VOuh+rg\/fizVDIdw\/dChmn2W0X+vUM368FHhcC0xfaQBrAzVum5WUK1vkDShWveTtrNU8zjxW4YgUM0bGvWyqDdKqNYt1XDxQFg90\/2jnFCNRQdvWhWo9vqkjy+fQHV82paiZoHqUqgcXxsC1TFCNarGgS9Lliyhb37zm9TU1GTb2ubNm+mWW26hG264oaLcHqKempVuqdY3qrE2VqABwOSjshmqEX4NgBKl+wfXrcOrX6hWh5w0phVU86ZFhmq4OvDigH2qP1i3gQ4+YH8yfar1uNYAZzeo1nVDn7EhDm8CGM71DZImVPObA4yB\/vaAQ\/ahbmygxPW4QTVDudfDXADVbM3WXUOC3ismVKMejlfNdSYFquU48qCjHF05gerotCxHTQLV5VA9ujYFqmOGagA1QPFf\/uVfXEftiSeeUIfByEbFyvSpZquwvskQgw6A0628+MwPVJtuEa4TSctg1bYO1egHh6\/jDZFcnC2uBSskQqYVTl4EoLN7BfJ4hWqGZAZgN6jWfaatoFrfIOkE1bo7Ccrw5kI\/UM2nT2ay+T4HxZjjwVCNzaA8J\/yMmZmXoZqje+jh9aI4QdFsz69PNeYPj61AdZiRjqasQHU0OparFoHqcikfTbsC1TFDtddhev7552n58uV0+eWXU0NDg9di\/S5fpVuqMSBeoJrjHXPYOpSzs1QnCar1DWnolxVUv7NmIx09roXe3lWASg6pp0M1InmYCw3zOvVNbxxbG6Hx2FKtx+7WXUGgpW6pdoJq5NVD5FndUIBwWOqxuOANmk43XtRQDas0TlOE6weifOhHk39myhj6xj8eEulzQKA6UjlLXplAdcklj7RBgepI5Sx5ZQLVCYHqxYsXK1\/i22+\/nf7xH\/+x5BMhKQ0OJKgGzMHdA3AJWN2854hn0\/0jDFSb8Z3ZQo7\/ckg9O0s1YBhts6UaIIzE8Yj9QjVbkxEZA3UBUPXII2hH96lmqNbjdMNijARXE0A1u2boriBuUM1+0Lg+L1DNYfWw6MHfbhE9ooZqtlQzVDNko+9RhtLjZ4AfqGbXGrFUJ+UJSiRQnZyxCNITgeogqiWnjEB1zFCdyWRo1qxZ9Ic\/\/IE2btxo2Voul6PVq1fTrl276Morr6Qbb7wxOTOkxD3xA9VOJ\/OVuNu9LKNulmr0G64PgGpACf6VEqr18H9wUeANexzSjrUzobq1J1\/0a2b3D0Ax\/tZD6rGlesHOQlQTtlQz+KpNh9UpFXKOoRptQ4OgUK1vliw3VAP0oVUY9w9YpgHP8JXmyB9WluoofanDQDXfi0m6J0v9DEhKewLVSRmJYP0QqA6mW1JKCVTHCNWIUf2tb32Lfvvb33oa7+rqarr33nvprLPO8pS\/P2aqVKjm470xJm5QbQIkx\/ktlaXahGo+uMWEagBvYXNewaoNYOLNggzVfKKgDtWLP9xIRx64H729uzBDGarZBxpl+YRGuFUg8YY+N6jW\/bp1S7UJ1fp4mO4fbKlG32Ex10PkWd1TbKletScyiZulOixUL\/xoJ335l2+qrlx28gH0yOwP1d8mVE89YAj9+IKjI38MBLFUC1RHPgyBKxSoDixdIgoKVCdiGAJ3QqA6RqheuHAhfelLX6JzzjmHTj31VNXSz372M\/X3YYcdVmwZvtSAye9+97t04IEHup68GHi0K6CgV6i2ChVXzstzgmrTBcOEaj7mWYdqhlRck1f3DysroXloir4ZEbAJa6oTVCM0HLuKIC9HAUGfVAjAhrTauKdD9aurN9Px4\/ctQjW7aDBU82ZBXJsfqGarPspx1A6uW4dqc8OlCdU8T1AWCf13SgzV\/GbBC1TjeHKkIBsVz\/nx\/OJpidiICB9qHarxN1uvo7RU82JDoLqcT5LwbQtUh9ewnDUIVJdT\/fBtC1THCNVPPvkkrVmzhq677jqCFRrpmWeeUa4e+rHl+Xyefv7zn9O2bdvo2muvLeYNP7yVV4MfqDajR5Tzap2gWo+xzJCM\/7JV1g2qAcYAP7cTFa0iL5QDqv\/8\/hY6\/eB9HKGaD03ho9Bx8A0DLoclNH2qAdV1BQ4unjwZBqrZp9wt7J1fqOZrs4ph7WWO6lCt5\/\/9VdNUfGokuIcgwac6quQE1fxmwVwkmIcdiftHVKMRvB6B6uDaJaGkQHUSRiF4HwSqY4ZqVH\/eeecVW9mwYQPNnDlTxa1uaWkpfo7Qezgo5oc\/\/CFNnjw5+IhWeMn+CNV65AoFju3ZIkACwADVCNUGP1yOU61bqpMA1egwwAqh9eAyAau6il9tWKpbOzrppVVbaPq4feiDjj0AXJcuXhfHtXaCapSCC40TVMMNBJZjHarZzcSrpTqpUG13eiJcQS4\/5UBlxQZ4c4i9qG55geqolCxvPQLV5dU\/bOsC1WEVLG95geoYoXr+\/Pn0+9\/\/nm699Vaqq6tTLbFVeseOHQqi2YINqL744ovV4S+f\/\/znyzsryti6V6i2OtSkjN1WkSg46UCMz7xCNVwGzMNUGMKtLNWmVdDKUs1WcgAkQ5N+2Ak+c3L\/MP2NYZlEnOqgUD1+UJWCYa9Qze4h7DICSzU2OKIfJlSz1dT0DWdINz\/HtddW7fXntps\/uqXai\/WZ83jJa7apH\/Rifqe7esBSffbR+9BZR+0T2bTnMbFy\/2BXIdNSrZ9WGVlHpKJQCghUh5Kv7IUFqss+BKE6IFAdI1TjlES4eeBHCq4d5557Ln3ta1+jnTt3qsNgPv3pT9NnP\/tZwk30gx\/8QPlbxx39A1AP95Nf\/vKX9MILL9DatWsV2B955JHK9xv9cTr5sbW1lXBIDVxbVq1aRTU1NXTGGWfQRRddRNOmTaN0es\/7+YDT0itUWx1qErDJSIrpUM0WU67YCqoBhnxcN1uqdajW67CzVNtBtf45H5oCSzhiPFuBpR+oZsu6FVTjelvbMzTnQ3tLNSCZNyqin7r7B2vCEMyxp\/VIKQx12NzJUI0wewzdsPSzD7o+sLB8m5\/HBdVuEwqHt8DizEn3j9bD5en1\/OxLx9CkMYPdqg71PR+K4wTVaEAHa4HqUJLHUligOhZZS1apQHXJpI6lIYHqGKEaVc+ePZuuv\/562rp1q3LreOihh2jUqFHqc1iqEUZPT3fccUcvd5EoRx03K2AY7ifd3Xstq3obhxxyiDpafcKECX2axoZKLBJWrlzZ5zuAOa5Ht74H6XvSodruxDg\/UA1YhOsEbwAMAtUoW4gfvXcRwycQWkE1x4S2suKyZd38jl079HEEiDIY43Pd\/cMLVMN9BFZ3tuI6QTUWBIBnJN4kiGvDhkm2eKN9r1DNixO+nqih2svmweufeptefW+bCpWHZBVz2sr9Q\/enDnJfeSnjBNX4jn3PeX7Byo\/QiFisSUqOAgLVyRmLID0RqA6iWnLKCFTHDNWoHlbqZcuWqcge7EcNi\/Ef\/\/hHBbgAbqTPfe5zdMstt9CgQYNimSGIl\/3lL39ZWca\/8IUv0FVXXUX77bcfZbNZmjdvHt11112EQ2imT5+ufLtHjx5d7Ace1NhwicXAaaedpvqJI9V3795NTz\/9NN15553KtSVsSMCkQ7V+dLY+SE5QXbCi7oUPL1ANiysDs2mpZkshDlHRTxisVKjGdcBKjcSRNWBZhm6A5yigmmNum5E72OXB7YbjeNxOLh28wdApIgfHn2ZI5v9H+9h0CCs2QNtMUUb5sLtWN6jmBY1AtdtsKe\/3AtXl1T9s6wLVYRUsb3mB6hJAtdMQ4waCtRpuFHHBNNoH\/H79619XLh8I83fzzTf3iTICS\/RXvvIV5dZx22230YUXXljsOqKWAKonTZqkrO36Jkv2E8dpkDNmzKAHHniAhg0bFmhm+4FqNKBbagM16LMQnwrI0RC4uA7VOhDj+7BQzTGhURdgk6F62c7CoSsMOdw35OP+oW1Ylp0s1exqYlqqGbJ0iUxLNecpHiDj4v5hZakOAtVsLfdqqbaDaq9+z3ydUUE1QuXd+ukJxdB4lQjVBT99hEUUS7XPx0is2QWqY5U39soFqmOXONYGBKpjhGrA6csvv0wXXHBB2cPkAZixEbKrq4see+wx5UNtJsAx3E8eeeQR+sxnPqPiZmODpQ7ksFBfcsklfcoiqgn8wZcuXUoPP\/xwMS6339nrB6rZT9hvG37ym+4eDK6664Xpq+sHqhlScS26T7VehwnV7IsNqIY\/NEM0+yDrUA3Y50Nc+L9wpdATQ7V5HVZW3CRANbRny3GpoNqLRduL+wdbs\/kgF93VA5D9xzc39rFURx3lw27+s4vH2k2bqb6+vtciH9+ZlmqOWY6Nq5KSo4BAdXLGIkhPBKqDqJacMgLVMUL1ihUrFMjCqnvppZfSKaecQg0NDWUZfRxEAyAGJD\/44IPKr9sqwef6pptuorPPPpu+973vqf5icQDrNtxYEE97ypQpfYoC1hHl5KmnnlJ+1YhiEiQlHaoZXNn1Atdo+uoGgWrAO6AWFl9Yl71CNcCGFxfsg4zjvtmCz6cO8pHgVj7VYaCaYRN14EAWt42KVpZqHDDDycn9Qw+BFxVUe52jUUM1QBmRPvQEF4+Zf1yuDnuBJRtuIMgTNVTbWdv9QDX0wBgkKVa817Hs7\/kEqit7hAWqK3v8BKpjhurvf\/\/79A\/\/8A\/05z\/\/mQC2iAACf+axY8cm7uREO0s1T5KDDjpIWbnRd6uEDY733HNPLyD3e3skBarZpQKQp7uYWEG1bknG9TpBNepFfmy24zYAmhz5AnAKS7JeB2CZN4nB\/YNPaMSx2bprB0f6QB\/QZwZo7r9fS7XV2JmWaoZNtIG+rNuVoSVr7aN\/mFDNbhleoFrfWKj7OHMYQg6pZxX9w879w+\/8tMsPEAYQI8HiDCi2SnYHu3BePkFRh+offO5IOvHgYO5UdmNoddIjz5NlazdRc2M91Tbu3dthWqoFqqOaOdHXI1AdvaalrFGgupRqR9+WQHWMUA2XiPfff59OPPFE1Qp8p59\/\/nn6z\/\/8T2pubi679dq8dITXu+KKK5QLh+7m8eyzz6qTHhG95NFHH6Xhw4dbqsZW7qlTpyq\/6yB+1SZU\/3\/2zgPMjqr8w9\/23fROQgmEmtBSgIRmqIKI8BcRe0EBRURBBUVQEKWJgCAgVYqAICIgRbCAgBAIJQmQSjGF9F63l\/\/znpvv5rtnZ+6d23Y3yZznyZPde2fOnPlm7t33\/OZ3vi8s2wbgVEz7hx43CKoTit6mRYK68FCD4kO1qsW6qNBCNftgYcD6AehEgepESjoW9ZEmb5NfWlP3UZyFMaqCHhWq\/clB2EfDeq1zhWrtOwyqUeyxt9iFinY86aDaf3LAfsWGalWY84VqPcdzjtpZHn5zoVOqC535I0ip5jV9EvDRoiVSXVMtJTWJFH76noK4Fv0h64q+X\/g\/TXGPuUYghupcI9c19ouhumtch1xHEUN1EaE6rGs+NG+++aaDawrEUOyls9VrxnTjjTe6f8OGDXPwTHYPWlRYJrvI6aefnhG+092sXQWqFaaxYugiPMatGSksVNtiKmyTDqqB9QV1m5RqttfH7lGhGnimoXZbqFZ4zxWqrSKe7hpZKLO2COKkSvXIodvI4oZEL6piA8mqVGv\/qp7r72r\/UBsMMEzzfeA23Z9uq0p10ORAJxZ+9o9cvzjtfve9Nl9u+s\/s5Ev5KNXaCVaQk299Q+avqk+m3yvEWOkjzCuvT0yWLlkiJVWboJrtSZ1Ho+qnQrXNPFOoscX95B+BGKrzj2Fn9hBDdWdGP\/9jx1AdHsOSNvwQRWp0vXDhQrn33nvlvvvuk4aGBlc85YILLgj0LBdpGK5bxvLII4+40um0yy+\/3NlUSkoSf0iDfNZB49GbKZOine5cLFTvud84sf5gu1+xlWq7GNBCNeAK6EWFai2ZrUUy6BcPsX38HgbVWgXRwi5j0SqHQVBNXywcC1Oqw6wR7BMVqu11CIPqAQMGSFlZYhFlMaDaquVRoVpzgxf6s2RVavrWcuL+cWz6vExjKGYKPXu\/6fXTEvTYkIKgWnOpx1Cd6cp1\/vsxVHf+NchnBDFU5xO9zt83huoiQrVv\/+BQWEDICMKCP1RqbXvssYfLqvGJT3xCevYsbuU0e8qaLxu7B2PD5sE\/LZ9eTKieP3++8M828nmTu5v47L3\/gVLfkupp1m1XNLZJfXPCghHWGttEKrNMTIClBKV3YT2L\/UqcMn1Av4RaSn\/8PqJXuaxpxM9clnzNjmHb6k05qdc3J8qU62tA7ewNmxbm6X6q\/tI32w\/rXiq9KkulrbVN5tS2On80bd\/eZe53GovEtq0pSabLe2Ml+bATUN2\/MnEOWCH4vbI08Rrd+KXHUYt5b9a6luRxon419SgX4RxpwO2ijZ7q\/gaq6Xv77mUudsSMMWnTsXEfMolTJVnjQNx6V5a5mNjGfhoThWqOw7kEnYcFyajnFnW7y5\/50C0u1HbRcbsEeqoPvnpC1C5lwo8TlrFiNBsLnZxwHQdVl7kJ9gcLlkhZVbXIRvsH7+3Qrdx9FhvbStz1s\/d4McYY95l7BFavXu2+y8PWvuTec7xnR0QghuqOiHLxjoEL4Utf+pLLhnbYYYelHGj77bcX\/m2tLW+lmuwfDz74oCtNzs8ovo899liymiHwTFnwU045RYDqfEt8Z3uhWltb5a9\/\/avzT1NhkVLjZO3w82UXy\/5xww03CP+CGgsed91nlPsj3resPYSuaimVplaRQRXt39P+2LeyJLuHDfTL8eY3lrm+36svl71rEtUn6Y\/fh1U1y4aWEve+vmbPYVB5S3JcG1pLZHZDufQpa5XtK1tkaVOpLG0OnwjQN9tvV9ki3UsTY1\/QWCb0Q+P9pU2J\/YF83Ya+p9ZVuOPQ7LHYhm05r4rSRH+26dg4rh4n6r1UUdImTW2JsTHmNfXNMnflOuenL92oVNO\/xpLY2PNnbPShbXVLwqercWD77mVtLia2sZ+OdffqZndd9Dg2XroP72nfUc8t6nY3\/HexPPf+2uTmR+3WS8752OCU3d9dVCsXPZM6gdQN9h5cI1MX16XdP+pYomxHTN3np63ExYy4EM9BFS1CdryZi1dJSWW1lHRLTO71Pe51PnNcP3uPRzlmvE3HRWDt2rWyYcMGGTJkSMcdND5SwSIAVCPIkaGLbF1x27wiQAE9mC+oUeuDf1trKwhUk1IPm4dt2Dw6O8UedhMyeQCvfIjxQlNOPSjlX7EWKgYp1S+++KJL+YdSvduYcU45s9k3NI5RlGpVndnHX3AYdlPTL4onajJq3DtrWpJKNf3x+x49y5L2D9RSlFHbrIqn76Ossp+qx2HHZxv6ow89b6tUo+Diyaah1FeXlzoQQqFFqWbsKMf79tl0LJTGplaUanHb+0o1+3CsRfWbFPGoH3rrr0aFXl\/XIO8vXiG+Uo39hdhxXsRAm+Y4bm1rk+oyFmwmJgU2DkFjtuNDvadvzoP+uIaqYut21i4S9dyibucr1eSgvukLe6Xs\/uz05cnsIH6\/bLt0fVPy\/aD9o44lynbEVq+1PjkgPlwbntLMmrtA6iq6SVN1Aqp5vWdFqbNi8TSB+ydWqqNEunO2QammtsDWrIh1TuQLc9RYqS5MHDurF+wfFM4jxfDw4cNThhEr1Xl6qjVPNVCNKk2e6g8\/\/NAVVjj33HPlkEMO6ZSiMHzhXnvttc7Pjc3j\/PPPd\/m0reXD3gnYVChgM3To0A5Nqbfr6LFuGEFQzSNozb0c9uFR3zLgF1Ze3N\/XLlDUVG2ksaOpP9p6qoPyPqt\/mn30fbyrDohrW2RBXbi6rh5XzcSgWS40pR6\/L9sInrymvmtex++tj\/axRNjy5vivsbMATT5Ua3w5Pz1O1C8kP5NEr5JmB9VBnmpn\/6gpTTl\/xq\/pBa0\/W7Og2MwoYWPSbTkPsqKo79duX0yoVq+0psPTwi72+GTxIJ1eUMODTbvz5Xnu\/9MOGSrfGr9j1EuQ9Xbcz+qf1+uhRXS4nv+bN1\/WlG2Can1P7w2XVrKGFIpxJcWsg98BO8Se6g4IchEPEUN1EYPbAV3HnurwIBdEqUbqp\/T30Ucf7VRgPjCvvPKK\/P73v5elS5e6RwHHHntshxWF4dEgJcgff\/xxVxr9sssucxaUdNaTjz76yEH3kiVLOrT4SyaoBp4A1ShQDQgAnplaFKhOFL0gnV1ZEpptv\/lAtfajOZkzQbVuRxwUqjkHBX8AXmGVcQVBtYKtThwyxci+70N1dVuTzF26MiuoVlgrFFRresKOgmrNPU3qO34Ogupr\/\/WhS5EXBarDFjpmc13CttXJnQ\/VWrwnCKr1PTz8NDKxxFBdiKtRnD5iqC5OXDuq1xiqOyrSxTlODNVFhmqKvgDVmklDD4efedasWS6F3aRJk9wiRdLqFXORIgr1JZdc4nzdHOc3v\/mNfPzjH89YhKazypTvsO8BDgKDlGogGWU1ClQTczJudAWojpoLWmF5u5oyZ\/dQ8FTltrJMkq8pKFmoVtU2ClQTH47XEVDtf9wsSG\/uUE0qPTKBBFVA9DOE2Dj4SnVHQbU+veAesVDNJBqluraqlxumVtzk8wZYx1BdnD\/Gheo1hupCRbJz+omhunPiXqijxlBdRKhOd5GCsoD0799ffvWrX7kMIIVuOFluvfVWB9IA9e9+9zsZP358RqDWcTz55JNOVR8xYoQr7ELpdW30jQf60ksvdWkBqawYVgY903nZlHoK1WqFUPCjj2yhmhzJNjVe2DiiKNVaXQ6gD0pRl06pjgrVjI\/zDoJqC8p6HjzSt1CtuYU7GqopV88f9TD7R0dAdZC9ppj2D1Wqb\/nyvoIVJB1UA95PvbPElSDXRvq8O\/47N2n\/KGY6PZ2A6eeHVHn8rJ8NPmNRoNrPxZ7pcx2\/33ERiKG642JdjCPFUF2MqHZcnzFUFxGqsVrwTxeM8GF5772ORzfFAAAgAElEQVT3HIDaLCB4mQ8\/\/HCnVu+\/\/\/5F8VlTJfHUU0+VZcuWuSwfZ511VjKPcFAIUNZ79+6dtIWwH1DNDUMeakz4\/M8qczKcXHPNNa6b66+\/Xj75yU\/mfAcrVN\/2xwdkzzFjnVJNOjg\/X3VnQzXKMSCSDVQDMHisoxYgUTC2HmH1wPoFPDStnPVkU7AjhurErdhVoBrwxh5y4FUvy2sXHJryORl35X8LXuglaBLDZDAIqnXSClTXV3SXVRWJhYqqVPOzrhGwr+X8YY93LEoEYqguSlg7rNMYqjss1EU5UAzVRYRqFiqSYeOMM84QbCAsDFy0aFHyiKQ8AnBPOumknJXdqHcF6jGZPqK2oAIuTAjOPvts+eCDD9p1w8Tgu9\/9rvsXtuAxyrEVqn939wMy+oCuCdXqQbaLAe25qVJtH5ezD1DNIsOoiwF1EaJdQJgJqu041O9sPdW8H6TkFsr+sTUq1cAwTT3VmZRqFjR2VvOVap6EzDZKNeOKAtXc+\/bJUWedT3zc9hGIoXrzvitiqN68r18M1UWGaj+lHosDgWhguqNyU9fV1blKjVg4orawqogrV6501RfJXT179my32PGoo45y54P1I99c2wrVv\/nD\/XLguHF5K9WaISOq\/YOMGQOqS4Xtw7J\/RIVqTeuGB1Wh2q+mmO56KFRbZTsMqjUDRiaoDsqMwT5dAartosd8sn\/ka\/9QSCYuUawYUaBaLSLpSphH\/Wzmup1d1KpKdRhUt1V3l1XlCU\/1sB5lLlUhTZXqGKpzvQrF3y+G6uLHuJhHiKG6mNEtft8xVHcQVHdWxcTi30KFPYJC9eV33CfjDzrQda45cu2Cxaj2j+X1rUlIjuKp1mp\/ZJDYrlvCb+qn1MsHqv10drlCtZ91Q7OExFAdrMRHsX9YmLZxzATW2UB1pr4K+2lK7U3XJlj7RxhUV3brIYtLewoLYtlGoVrTSupnopjjjfvOLQIxVOcWt66yVwzVXeVK5DaOGKqLDNWkrCMnNcpvvipubpd489rLQvW4cQmopvFHPReoRhkGjqMq1UAvuY5ROxVcc4VqAJ0JgVWqOxKqNXbW\/rE1KtVA7\/SLx6e13bBwkEWGQQ1bB5YOv1kIV8uHvmbhWRchBtlCOvLTyVMOze2u9zn3Z5D9A6heWdHTfQb43MVQ3ZFXKr9jxVCdX\/w6e+8Yqjv7CuR3\/BiqiwjV\/kLF\/C7V1rE3UH3m2d+XX918h4w6YFwSbPEj5wLVauPIBao14rlCtSt2slHtVvtHoaDavxuClOrNGaptUZKgvNP2\/P3iL9b+oYVZ7v\/6SNlt24SdIahlC9V+irx0UK2g3dlQbfOrK1Sr1ck+xcFTDVRvqOrlJpi2yEusVHf97+EYqrv+NUo3whiqN+\/rF0N1EaHadk1easrHrlixQoYNG+YW8+FJJnMGxVd23333yOntNu9bLv3oFaovu+M+p+6rWpwrVAO2QJevxoWNwirV+UI1Ke4UdrdmqNbUgCj2frNZTKylxarr+UC1Aq2F6snz1siZD7wjn9p3G\/n58bu7IYVZP3S8vm3Dh2ot+EI\/Pjxr3yxQxFPdWc1CtT5FqSqTdk9xgOoePXok81THUN1ZVyy348ZQnVvcuspeMVR3lSuR2zhiqC4yVAPTTz\/9tFx11VUu88cJJ5zgfqa6Io2MGiwiPPLII+W0007rsMqKud0uxd9LofqKe\/8i+4zYIxSqqQhHy1T8JVuopl\/gzi4ORKn2M3lQOCNT9g+g2mbg0JzAUaOYbqGi30ehleqo6d18b3dQ9g\/GGrSQkteLCdU297OFagVihdx0JcQ1zpoKT3\/XhYf6u4VqXrMQrlAdVGkx6r1QiO1cCsh67FBl7v4mVSVQzUTSfo4Uqlu69U5WDrXH576OPdWFuCLF6SOG6uLEtaN6jaG6oyJdnOPEUF1EqKYoCpkyLrroIleenOZDtYI1VRdPPPHEvFPSFec26bheo0I1sMwiqnRQjRqn1eKiKtXpoBoLCXAddaHi5grVaoXwQTLoLsgE1cAnbeYFBzlvud\/CoFr71QqSxD2shdk\/LFT\/5tN7yPgRg1wXVmVW+M2kVGeC6uP2HiS\/OGGPpOLt99vZ1g\/O24dqXbgYBtXdevdJQriNPZ+RdJ+7jvu2iI8UFIEYqjfv+yKG6s37+sVQXUSonjNnjlOf58+fL4cddpiQlxoLiFWqOTzwTR7p+++\/X+655x5XtXBrbTFUb7rynaVUf+GOt2T28lrJxa7gK9UKq2uuPKLDofrcP0+VV\/+3ygXUQrX6rHkdWKaFLVLUq+HbP3ylWhczat\/0+9bc1ckqibnEstDfAajLTDRRqkmNp4sPgWqedGhTpbpv377J7e1YtI9Cjy\/urzARiKG6MHHsrF5iqO6syBfmuDFUFxGqn3\/+efnBD34gN998sxx66KGuGiEeah+qGQJ5n3\/605\/KFVdcIV\/4whcKc3U3w166MlSjjtPSKdVA5E59q2X2RQe7suGFtn+EpYYrlP1jzqo6UXWZc82UAu7pd5c45Veh0Ydqhc\/5vzwssOhMsZTqoIWHei5hUL3H4B4ya\/H6dp+aof1q5C\/f3j\/5epCq7UM1v1OO\/M6X57n9OhuquQ9RqhWmo0K13U4DEEN11\/5ijaG6a1+fTKOLoTpThLr2+zFUFxGqAeXnnnvOVTJk4U86qH7ooYfkwgsvlG9\/+9vyk5\/8pGvfNUUcXSGhGgjGNhB1oaIWtvDBVT3VmaBavbmFgmot1hGlAmO+UM0l1QmAhc6wdHJ6C6jFIgyqFUA7GqoV9u2tGgbVQ3pXpUwkgm7v0w8dKmd8bEcJ818HQbWdnESx0hTxY+WubaGgms9UXE2xmFcrv75jqM4vfp29dwzVnX0F8jt+DNVFhGqC+6c\/\/UmuvPJK6d69eyhUr1u3Tn70ox8JQEmZb37eWlshoRqlmJYvVGu6sUxQzbEKqVQXE6rfW7JevnrXZBcfhU2FaqvE5gPVz05f7lRsbUF9FUupDoJfPU\/rqQaWUbX5p82vevj529+So0cMcFD90vsr5PxHprtNOZ\/vPPCuA22r6NP\/GR8bmgLqCuWd9blWqFYgtgq0D8nW\/hE03hiqO+sqRjtuDNXR4tRVt4qhuqtemWjjiqG6iFC9ePFiOeecc+R73\/teqP2DEuK\/\/e1v5c4773Qj4X8ygWytbXOF6omzV8n3H5qavGxt1xyZt\/2jmFBtwVmBj+Pd9OKcpGUhU7YKqwYHKdUK1SjBi9Y0OAj1i6gEQTVj02Nns1CR8dMfmS1UQf\/E3oPk2alL3XVRWP79C3Pk3lc\/cq99bLf+sqGh2UE1YwOQLQBbOAecrYofVnLc91vbiUuxP9dh5eYLCdXFPoe4\/\/wiEEN1fvHr7L1jqO7sK5Df8WOoLiJU0\/Wjjz4qV199tZx99tnSv39\/efbZZ+WSSy5x2UBefvllue++++Ttt992o8B3fcMNNwgLhLbWplB9yV0Py\/57Dw9NqRcl+0euSrUFPa5DFKXaz1s8+8KDZWlbWV6e6kJBtd5LACoVFR+atDhFQVYg9qE6k7pqAVMh2HqqD756gju0wmoUqP7aPVPknQVr3X6M67ZThkumPNVB1heFas7hb28vlmXrGh1U43O2qrT9nP32c3vJDx6elqI6W6hmPOzLa+lAOZ3nutifa4VqMuNY25BCdZTjZ1Kqo\/QRb9N5EYihuvNiX4gjx1BdiCh2Xh8xVBcZqsns8dRTT8kvf\/lLV\/glrB144IHy61\/\/WnbYYYfOuxu6wJELAdU8ngYotHphtvaPrgrVCrFBiwfTeaqzgWq7UNFCddCxrSJ7yn7bynnH7CIK1fNqK+Tsh6al3FFB4\/ZjbYE0KlRr9UV7MB0v54BKjqoOVN\/x33lJKPZv96DxBXmzdb+gSUImz3WxP2KaJs\/PtR5DdbEj33X6j6G661yLXEYSQ3UuUes6+8RQXWSo1u7xTf\/rX\/+SJ598UrCF8G\/o0KGy6667ykknnSRANVUWt\/ZWKKimyIWWq+4MqP7Pd0ZLtz49CqpUK3AWGqqD7B9q0wAcaQqpFiQtVPtKtYVqhVnty97jFqr9jB0+VIdZMYIWQeqTA4754OsL5M9vLnRQbT3e+UJ12OLDIKU6UxaVQn3uNTVeQ0tbSraVGKoLFeGu308M1V3\/GqUbYQzVm\/f1i6G6g6A6ym3yxhtvOIvIzjvvHGXzLXKbzRWqbaERLkyhoVr7\/\/Gxu8rJY4a0u\/bZKNUvfrjaQXL3qjLZ0NCSslBx1OUvub7VKmHzOAO9Fqp9eAQcValuqughJ982yfXFPv+atlS+dvBQ93sQjPP6LS\/OkXsmJLzOQD1e7CdOH5W0f\/hQrccPKqyiCrPNFw2k83pYCwNf\/zw53v2njZEelCMMaHb7jsz6AThv1600WaVSn9QwxBiqt8ivy8CTiqF6877WMVRv3tcvhuouAtV8kLB\/HHXUUU613lpbIaAamKgqlRSlmmp+mUora0q9XOwfvk2gWFDNffHYdw6QbftUp9wiQRYI\/x5ST\/V1zycWI6rXWWFy6vw1ctp977jdUK\/ZRv+3fbF9kM0BtfqGU4YLf9Qfn1krd02Y74qrsC0KsSriYVCtr+u4fKXawup93xwjX70rAe3+2Phdt\/WPzzmph5xt+nWvkJUbElliwqDaP9d01RGBV52YaBzJGtIRjWMP61HmSo9b+xPH5p6OWgUx9lR3xNUq3jFiqC5ebDui5xiqOyLKxTtGDNVdBKpZtEg6vdtuuy2G6rO\/L\/ksVOwMqLbV+7il7v78CNlz5wFp7R9AnQU5\/Z3\/Abdnzh7rSrEDSHZRYNACwmygWhcqKjCrdcMCIeqyzbNsPyZsj5Ltvx8E1fQzdf5a+dkTs5JQHaRw07+fNQPgfeP8g102j+uf+5+zcWRqfnnwoAmAzWqix8y0KDMoW0rQWPy0hJn6zXQ+2bxv1WhitqC21e3OGoMYqrOJ5Oa9bQzVm\/f1i6F6875+MVQXGKpbW1vl\/fffl5deeknq6+tl1KhRsv\/++0tNTU3gkdj+iSeecBlB8F2T1zpWqjdBtQZtYFWpUJRFW7rsHx0N1W8urk0C5mdGD5FHJy8KhGrGbjMyAHTH7zPI5T9WD\/BPjt1Vfv2PDxxUA6Sqml\/x9\/ddFgtaUHW+bKB6z18mLB4AJ7YSbB0oulg0LPy+OWeNfPfBhHId1hgnkIpSz88Pnz7SKdUn3pXITw18L1xdn1S9Udh9X7NaJD5zyxuyYHV9UkHXYzLOIJ9y0Jg+t9+28s8Zy2R1bVMyLR\/bBWUq4XWF6kz5uP0sIJxXOqim35UbGuWl8w\/psL8QPjjrZ6SxJaFU21Lk6QYVK9UddsmKcqAYqosS1g7rNIbqDgt1UQ4UQ3UBoXrp0qVy3nnnuVR5to0cOVKuvfbadl7pDRs2yE033SR\/+MMfXIo9Fio+8MADcsABicVhW2PD\/vGdH54vF996n0upp62rQTVwTKo9FkTe\/vqiZKluxgtg+kp10LUEvKrKS2XOilopLy2R5tY2B4Ka8g2YVKjOpJSGQbVCI32p\/eOoG15PFiyxUK1KdZBNI+xe9JVhfzuA+Zjd+souv3jRTQb227F34GJBtXywvw\/R6bzQdj\/\/2FaRzheqrWf+e0cMk68cuH1gSPR6RYX1Qn7GfahmcolK3au8xB1mu27BHnB\/DDFUF\/KqdHxfMVR3fMwLecQYqgsZzY7vK4bqAkE1H4QrrrhC7rnnnsAejznmGLnmmmtcuXIaf7goR84FoLFAkcqLeKpLShJ\/BLfGtqVAte+p9q\/llc+8L49PSSjPtLKSEmlpa0tRaS1Uq9UCxTSoKEsYVNssGPsM6uYgC6im3fLlfZIVAfl9ykXjnR84CKqBY8DUb1Gh+uYX58oLH65y56eLBW02DgVnVejtJCIdOHP8z9\/6hsxZlcgdTdOJiVX0M0F1oRYU+lDdUVk\/OG8fqhfUtiRjUlVWkvKkJ913SwzVm\/c3bwzVm\/f1i6F6875+MVSHX7+SNpJMR2wzZsyQU0891T36Pv744+XYY491e1L85YUXXpCqqioH3GPGjJGJEyc6RXvhwoVumzhH9aYgbw5QraNVpfrJWasccAJxtChKtV8sRvvs063CWRdoFqrpH1hkoV0QVGONwUfrNwuTvDf94vGC\/SMIXnVfBUHrcQ6yYKjvG1APs2dYpTrKRynduHgPELdwHwTVehwL1fOWrJNT7pqSjKtuo+dYaKiOcq6F3saHau6HytLEBL2xtW2Lguq\/\/vWvSUGi0HHc3PsDyrAeqoCzuZ\/P1jj+9evXS3V1dZxmtwtffLjt5JNPbjfCGKoLBNWPPfaYU54vv\/xy+exnP5tUm+HyRx55RC666CI566yzZODAga4QTFNTApy+9rWvyQ9\/+EPp1atXF759Om5omxNUk\/96XVOr\/OKfs5OeYS00ctspI2TUromFihc9McuBtlUtg0pZh0X58k8Pl4sen5lUsTNBdV1ji9SwwtFkwdC+LztxD+ebVngNgnsdpwXyoIWLT509VrbvXe3U75NvfVPmr6pLnoLNLALwY\/\/QpkpyGKhrPuugVHYWqn998gg5fPcB7ZRqPY5dJNhU3yiH\/naieyso1\/aWCNVks6Fp5j+ttpjp07w5KNXnn3++EyfGjRuX6XTi9+MIxBGII1DQCOh3z29+85sYqrOIbFZKNenwKDd+8803tyszzqwTZfqf\/\/xn8vA9e\/aUn\/3sZ67wS1z0ZdNV6YpQDTwDJiz8so2FXxTZsFA9qGeVXPHM+\/K7k4fLuN0HOqg+7qaEh1lhLqzqXti9qRBq09tZQP\/n9GXypdHbOKVaC6gEqbr0rzYLVXF\/9dR78pSXu9nPRe2nkFPYZbud+tY4qKZZCAd+VbkHqu+bMM\/FiTZ6aG+59cv7pkC\/fwxUV82nrXGxqfDYHvsK\/0\/\/33J5ZPrKpIqtnnQLygr2\/nHOfXiqvPrhqtB0ell8X7hN\/XSM2e6fz\/ZBGT702kQFao6\/uUA1Yw36o5ZPDON94wjEEYgjkCkCTOrDvn9ipTo8epGhuqGhQS688EIZNGiQU6uD2l133SWXXXaZe4sqivwxYAFj3FIj0BWheuWsKVJav0EWV\/WX4cM3LZ5UqFb7h4KkrXxoleowa0NQLmgblU\/uXC5\/\/19zSs5ogHbNojny9ceWuU3H79RNThvdXR7+oNSp4oA4+132woqUAI8aXClTFjfK3oNr5A\/f2D8J4Xajq47sJf1rSqSysjLlfHUbPb97TxooLQ11UlZV47azqvdZB\/aTrx+xl9sFoP3iba\/J64sa3O9BKe2IzVVH9nTFY2iDpE6umNQiExdt8gUz5vMP6uG20bHNnDlTJs9YINPaBslX9qyQ37+2MrnPaaN7yIHbJhT77q0NsqG0yv1sz4v99Zi5fhbpj0Y\/VW1N0lBSkXKcQhwjyth6lTRL\/5b1UTZNu40ums67I9NBt27d5KCDDmrX5auvviq1tbUS9n7YGNL9USvkuOO+4gjEEYgj4Ecghurc7onIUF1XVycXXHCBHHzwwfL5z38+8Gg6e\/nkJz\/pFOrBgwe32+7vf\/+7DBkyREaPHp3biLeAvboiVC+YPEFK6jfIgvI+zhuvUKZQ\/b0H35THZmyQz4\/s60BWQRc12S8GAvy+NKc2eaX6VSc8ryvrw+37bMP7wO4Fz6912\/\/44J5y9YR1yX5261sq5+xXKfdNa0pCpd3evzUUqhX46Ov9VQlPNv3Qn56rQqyC4xlPr3Hb3XF8b2mpr5Oy6hoXE6wvOr6bjq52+9OAve\/9e70srU+cK30zfva54Pl1TsXnHH95aGJ72jYta2VJWS85+9\/17jgcU89Rt6F\/JrS6La8\/+l6zPD8v8UQByFao7tHaIOs3QjXv6djYv5At6DiFPkbYeDl2\/9YNhTydgvbFYmwfrJ977jnh+5OUo4B1VMCOobqglybuLI5AHIEsIhBDdRbBMpsWFKopQU52kNtvv935qv2G9xr1evz48Vt9nuqglHqkBdOUYKhbM+vKpFxaZdu2De1UrnzyVD\/++lQpb6qX2rZETmxg7qejS2RMnwYHeTRAD8C76uBKaWltlXtmtTolGaDmH+\/TgME1dc1y3vObQGfckLIU9ZXtFJoz3aaXHlIll7ySgEC\/HwXOG95qTMKx9se2vH\/\/9ISPn6aqL7nRV9S1JfvlPaCa97Wlg8LytlZpLtmUP\/zGSY2yrLYtBZCBvR9PaHVQzVi+uldCyaVd\/HKDmzAA4bZZUM4UF7utPRfbpw+7mfrM9f1iH6e6rUmapSwl5jrWrgLVADKgTPN\/JruRbapU62sK2P52\/vWIoTrXOzTeL45AHIF8IxBDdW4RzBqq+YPw5S9\/WcrK2ueDXbBggQPqb33rW7Lddtu1GxE5rq+77jqXdm9rLv7y3\/\/+V2bM+UhKt99DelWUyh577SMfzJzubAb6aJs4Lyjr42LYu7VOAI1t+vdNqmD5QPXDL0+Skvr1Ul9S4UCZf7R7jixxSqcPoHceVSYTlpbKXe82tYNqoG5NXYtc9MommFUY3rVPqfzkkJ5OaVWw9G8KH7aB6lsmN8jiWnHH0rHpfmRitPlqdP8gqPZVX50I0BeTAcalSnC6j48P1UHbAnvXTmqWd1eXyTf2qZD9ttn0+eDaEWu\/5QrV9MO5+JBebNi1YGsV8dy+esL3Il40P2a8XtXWLH1aNy0WLdSxtXBVGCgHHUdVZ96zP6eDZQvYmaD6m9\/8pktDGnuqC3WV437iCMQRiBqBGKqjRip1u6yh+sknn8ztSGavrb2iIo+DV9U3OVW4V2Wp7DFyP5n29mRpbGh0j\/lRp\/kj\/0FjVVKt69+yQQZWl7oc37SoUO2rZHhhZzeWu+Oe+3xdiiXj2oNFqrpVJ1VojqNg+trCFqcCq21CIRkI3qa6Tc58rjFFWVbLh9odgqAaKER1BuLVGkJ\/QNM5r25ShoNuON9SouqwBecbjqyWMtONnoOvJGe6oaNCNZMf7DN+KwZUh4F9MWG3o6CayQHNP5eoUG2VY\/oJU5VtDNmGz1aQVYPtMgEw2+TqnQ67\/2KozvTJjN+PIxBHoFgRiKE6t8jGUJ1b3PLaS6F6dbeBUtbW4hbB1a1d4wAaqK5uSyjHKNW8hpraTxqcYp0tVCsk6IDrS8odzAMopz+3aYEc719+SIU0S2mKTUI9uwqk+ruF6tmrmuWe6S2yz8BSeXfZpjzSADKLAWlq2fDVY14HclGk1SKxdMUG+eXkTUrvkN5V8smdK+QPkzctUPMVblVtVXn3j8MY9L3OhGoL6Lko1enU6K6sVEd5IqD3KPd5i5RKU01qCs6KurVOqe7R1t4j7oMzfanNgp99VXnt2rXtMhLx2cpGSfa\/BOxnTf3TQQsXo355xFAdNVLxdnEE4ggUOgIxVOcW0ayhet68eW6h4vbbB5cwDhsGKfcmTJggf\/7zn+Xee+\/dqu0f\/PFdVt8q67v1k3333VfeeecdtxhOfbs7Nq9MQjUL5GjHjRuVUoJZlernJk93GRl2qmyWXsPHyNj+qTYDX6lWhVwh2V4vIJhMFNZyoYvrFKKDoPqhaQ0yY3V7uwYLFj+3awKyFarVk62L6OziQbYDjmfOXy83zSxPDg2rBvtPXbzpsf\/B25bJqoY2mbGiNWlJSQfv2lmQbSLTR6eQSnUQVKfzEOvYFMABZ2wR1uOt2xQKqjMBMMcBeNP50PX62tiyaJN71e7nL6YkPkA157ehpEpqevV2XbBfj9rE54InNmrT0P7tIkB9Ld3iwGKk1NPPmrWQRFG4w+6\/2FOd6ZMZv886pZdeekn+8Ic\/uL\/LJAnYmqsVx3dE4SIQQ3VuscwKqinogp967733zulopLH61a9+Jccdd1wM1fWtsqKse9LT68MWajVKNVANeJOyjUwc2hSqn5k4xUHK0PIGWVUzQLarX5pc1Bj0OPrfr0yUWetaxIdqhWdVc4FfMmXwjywWL86udcCtqeh+8tzapGVDPdQXHYQfujH5OgsBsYswviCo5rxOu\/vNFFgGqn2lmtcenNEkryxIKOuDu5fIzw6qSirPCuq8p\/aPIKU6p5tWRIoN1UAkABkEykFQXSatsqa0pp0fvFBQnSlOUbN\/2Cwytk8\/\/Z5OLLlPmGAMqSqR2rpaWVHaIyUm25Y3uXUHfDbUA2379ZVm+56\/ODBbqM7G2pGP2m3HvCVB9bPPPivPPPOM8L8WBfPvs2HDhsluu+0mJ554ohxxxBGB1zjTvdkZ73NOgC0VMMPOLWhchx56qNx00015FUWjujFrmN566y2Xxvbuu+8OXM\/U0XHhb\/38+fPdNWcNETUudthhh4IOY0u+pwoaqI2dZXtNYqjO7SpEhmr+4JGHmtlwv379cjsaj+DjlHrOtzm\/obSdZ1SDCmThKz7ttRqXTUJLeY\/otUm9DYJqbB2DNyxyf4zC\/KHqqbaLEVGo1S+tsA2osg1Q\/eujerl0egA3UH3EuJEyd8pE+dwziXzLqNfs\/5MxJXLH9E2p804a0V2O2i4Bwg9Mb5JXF7akLK7z1WoFYc7\/q88nbCM2DZ2m0lNft0K0XbAHAOPv9hfxaWyjALJ\/c0fZB9CM4qkOUqrzgWrGqspvEFQHpdXLpESn+3CzL+Nd2rTJrK7wTJaVoBYG12xLOkaKtwDa0twoNc11zurUVt1D5q6td6o8\/v\/uJa3Soy1xvzXWrk\/e4\/7xrCdaVeIgyM0WqoP6zflLMOKOWxJUc8qoqvwNoSIv7Ytf\/KKreUCRsNWrVwvZo2655RZXYIwFmldeeaUcffTREaPV+ZvxFPanP\/2pG8hnPvMZJyD5kz\/AZsaMGa7iMO\/lC9X098c\/\/lFuuOEG+cY3viHf\/va3O30ysmzZMldhmUJwXMudd97ZwX6hoXpruKcKdVfnck1iqM4t+pGhOrfu472CIsAf6N\/PFHl0domcOapCbp3S5CBw7kmhrycAACAASURBVJpW2bF3qWjaMIVq7ePfx7UlVeh0SjXb+1DNa6h1eKrxciskA6hLatvkoRmJjAsKyIzH5oPWMQDg2\/apduBz8tOJtHragGo8r+qHtgqyA6iNOZftPrx2y7viCrSosg20Xfhas3ywqjUlD7Oq0BaqtS9dDJgJgDO9718vxmetOWF3dBBU6\/nahYocXy09WDrmNScg1SrVQdaJPrXLnBee43SrqnBKtW+\/CFOqbdyzAWp\/HHq8IVUiKyWRP1uvq1Wb7X7+Nu3iuxGqXT+lpJQsdU9lAO1\/vzZJqKVDUZvjxx8kvSpKMnqeo6rE2UJ11H4L+Y23pUE1sXnsscfkRz\/6kQvTD3\/4Qzn77LNTQkaGqO9\/\/\/vy+uuvO9gGGDeXAmJap4ET+tznPufAWYsm+fcFyvLvf\/97+e1vf5uXUl3I+63Qfb388svyta99rahQvaXfU515TWKozi36MVTnFre89vrLX\/4i9y8dKE\/M3rSoTzsEao8e3OwKXPhQfe7wRvnUsAoHzH97dbI0NDTKsvqEEoz9Y5vRh8rIvuUOTmhB+XEVqlGkWfinHmkFVs0Tzeuo1LbaH+nsbjwqkWvZqsk69gc+1VNKN6yWL\/4nYVOxCxV1Gx\/qulWWyzf\/nliAqMVasLkcceNEZwvR8bGfWk6CVGgF10wWCIXqqHDJdiwmrW1MLd\/u3wBhUM0f1aa1K53iqn21lJQ5Sw\/X6r4XJ0l13ep2UM37apOgjwENq1zWFo6D\/aO+pk8gVGfyOgfduGHwzOuMQxvefyCZaw\/U0zSO9n9e1\/2s1cP2lbwfPKge1qPMwTNtQW2LLKhrle1qSmVAdWnyvs7nw6efidLSUhk1apT07ds3n+6Kuu+WCNUWPIOgmoBaxRflNayCb1GDn0Pn2UD1ypUr5Y477pCzzjrLTR62xKbxKKZSTdy25Huq0PdFNtckhurcoh9DdW5xy2svPGZ\/+6jN5TX2G+ruicNKnbf01OcTFQixRQC4X9ypSU4bUe6g+uEXJkpdbW3SQoLqiQI9tG2dU6NR97Rp5gNVqm\/9X408NzcB4wqttkqffR341qIvWrmP9wGrKasrkudAP4cPaZXurY3OKx61AdWvLBIH+FqFkFzdP5mYsJ741o4w33G2UB11fMAiZbnXNrafAFkwt1BtLRf8TE5w7Ay0A0aNlLenz3AL77hW7zdWpijVuq+v\/FIAyEK1eqrpM539Iwmv5imBr1yzjS4i9ONi4V6Pxb3G8XWiYMfgg7jtr2dFiaxrSq2qqfYPtuNnC9XLG1plQW2rU6\/ZVyeLUa9d0HZq46ioqJCxY8fGUJ1PMHPYNwoA2W1OOOEEueqqqzrd0hDlVLOB6ij9be7bZANw+ZzrlnxP5ROXoH2zuSYxVOcW\/Riqc4tbXnvxh33i2sqUginaobVM2Ep8\/Hze8DoZO6TKWUDeW5WwXpDLl\/fOHtEsg\/p3Ty6q2655tct3rU1TjgHcp\/ynMvm6n4pO3+B1VF2yerDIkGYBF6h+bnG5S6VHY7HiGXvyqD4YqsOUYbVDWCsDObmDwDyddUOh2qqoQUCZi\/2jb3WFLF6zKZ1f0MX3lWp7vowNqAZcWWinqjfXiLzW1v5hodSqvGR3mdNYLqSVU6WacVj4VpXeQr2NgQVnm4nDvzZhCrV6o\/utmitLynomJwpBUG1V9uHDh7thYOsAlG2zoA00o95rY9u1TW0pC3TDPnhhCwn91\/V3vKiFgOpsFjBm+6WxtSrVTz\/9tHzve99z4cpko8g2psXcPipU4x3v3r277LnnngUZDl517EzYLUgC0FWevmQDcPkEIgpUb673VD5xCdo3m2sSQ3Vu0Y+hOre45bUXUH30M4lH3H6zC\/Pumdok\/1vdKj89sFLOf6FBThhWKecMTyjQLHSk3TKj1Fk06E01QODXh2o9zvqSKrl4SoVbhKiFWXjPVlZEMf7RmE1ww8I\/P5sGIHjTtBKZuCTRM+\/jqe5XIbKirEfadGv2nK3HWME6DKrTWTssVKfLpBEFqn0l95D9Rsob786QsIV4nE+6hYostGOBKA1\/Np5qztVCdVN1T8EWolUegV6FUfYb3qtcZq5tlg+nvOHAHPuHtViwyK9tw9rkk4ugSQyPmW2fgK+ek3\/O1qqhgMzYeLIwsHaprCjt7tLqMU76sBYSX2XXvooJ1WELCcNez9ZTHfaBL+YCxq0Rqpns4EW+\/\/77Xch\/\/vOfy6mnntouTRwLr7CJ\/O9\/\/5PBgwe77Bfch9\/5znfk+OOPF+w9wOsVV1zhFsvRPv3pTwu5v4cOHSoPPfSQs1+sWLHC5Ss\/5ZRT5LOf\/ayMHj3abUv618suu0yoEsx1+MIXvtAur7l\/T0SBagCYCpl77bWXG6c2Xl+yZIk88cQTLsXqL37xC5k0aZLceuutMnXqVKfUf+pTn3KTDc6XtmHDBrfwH9\/5tGnTAr3LfGaxG3I+u+yyi8yaNcvFkorHp512WrtMIWTsIIsJ3nYSEhDXlpYWl\/Xr2GOPTfmc69hbW1vllVdecePYdttt3cucC5NWYhhm\/yA9L+l1OWeuA9lfOA4LWIOy+4R9BjNBddR7inhy7sSLePJ9SYrC7373u8kUwoW+p6hMzbG437kP+T7lO4VMLtiedt9995TTJtvLv\/\/9b3nqqafk0ksvdftyjy9fvlyuv\/56N6mi5XpN7MFiqA6749K\/HkN1bnHLay+Far\/ktnbas7JErhxf5dLDAdlf3avCqcX79GmR247qIbW1te2gWi0i9IGXed+KNSlKNa+jUvPI\/h+LK521gn61AZt3b1SdeZ3ftQVZLrRABxaVQT2r5OJxJc6yAlRvN\/pg94chXR5j+lZrhQKnbk\/qtIXN7ct6qwodpKxq2XW2adh4nkEXKUgZ17HYRXUKi7ynUJzufAZWl0nPpvXyv5aE19g2Vap5TRc9cg5ANdszZqB6r5Gjk3Hz1eJ0UE2\/KL4z3pkiK5tKUhYQWutKmALtjzfIwqEATvx2rWxwCxXVV60TgSAQp29VxQeVt8jgPRPQos2CNudgM9z857U3ZFVdkwypLpFMRVTCFhKGvV4oqC7mAsatDaq5jwBlQAwQIvMHmUIGDhyYcs98+OGHLo0cSi8ZQnr06OG2v+666xyEXnDBBe592vPPPy9nnnmmAC9U8lVo5j0gEHgF5sjzvNNOO6Uch8wcfI9de+21kXzPmaAa0AFSGR9+coXq2bNnu\/OYOHGimxjsscceDoDnzp0rpN177733XFo6zhHYuv32291YsZBR\/wEQZH8fXtn+5ptvdmtrfve738mgQYNcBpYXXnjBxZhYkcKQxuuAGqnvzjvvPJfWkIkJfRA3tt9nn31cPwC5NiyFLLYENtlXIZDPF+fI+frj0mPdc889blHqIYcc4qCQa42izDmTzSSq4p4OqqPeU0zOLrzwQmet1AnUww8\/7MYE4DNWvXcKeU89+uij7n4gUwxPZZjwLF682E2euO66UJdJ5I033ij\/+Mc\/hJ+5Z4855hi3oJfJCZMSfarDJCjbaxL0tzKG6qCoZH4thurMMSr4Fr5Sve\/AUnnHVCLkgFox0EL1dj1K5Nb9N7jxoBSWVtfIzZMbnFJN+joyetB8qEadBowVghTWrVIN+JGNBMVarR\/0xetBpa8BQQDL2jQUqtfV9Gun6obZP4KANUypDrJ26MWxSnUmqE6XD1r7Q6XQQiVBZcZRFfiDQ+PcepU0y\/+NH+sWHvrwrfv7ix736ikyT3pK6ZqlKYVOVD22yrJC9aL3psq6VSuTeap1vMAz1wM4pzF2Gir7K2+9nVTAwxRoPQ\/tz26HLeONKYk+sMLgqR5+0OFuEWHYQkTrldbiRv0qRXYcNS75eWLfPo1rZHVlb6egz5kxVfo3rHT2JtqClWvcOfWsqYpUJjybD2qhoDqbY2a7bbGhupjWlbBztQAEVJKiFTBElQWoFi1aJEOGDJGvf\/3rDm569Uqtqkm\/QC6w+PGPf9z9DFTTNNsE2UKAZJRWlD2URo7L\/5p5hO3nzJnj1FqgFlC16fsYE7Bz5JFHuvR4UZo9t0zbA0hWqWZ71HPAjs89KqSCrRZ4AUD5bvAtMWGP9LnHSbM3ZswYB23WFsbTAKvM0weLQokBx9aYMi4+92z\/4IMPysc+9jGXBpAxMi7AE+DmOgB5tj355JNyzjnntINqjnXuuefKRRddJHjmtb3\/\/vtuvAsXLmx3rdLFM997CkhlnExYSImoKjkTBu4BzmO\/\/fZz9wigX6h7yuYYZ3JHphQacWWSdOedd7prYhfqAvSnn366247\/f\/CDH7jxkLP7wAMPlBEjRuR0TYLiG0N1pk9x8PsxVOcWt7z2slBtfcqagcN2jlf5OyNa3aJFAPu6g8VlBtHH72c8vSZwLGQKOWpwswDUwLSCpKbJu+6IaqncVEvG2RfUSsKYtMpf97aGwMIkmkt7bvmmnOXsk85TrV\/QdsBBwFoIqMbDHKQsR7F\/6Pj0j1C3klapbduUlxklQBtgzXZU+aupW+MWHvrNniP2CfVU71xWJ8u6DUpCtRb6UZXfKsYK1Si7U96Y6FLN+XEknaG1hfA+27\/29tTkMYMycAC3dWsT9xEVDK1FhNcAZBZX8gedc9mzT6UMHTk2BarVrqKQTcYU7CzJlGLNjU7J5ymGNs6TKolaWXTW229J39rlyT9qVP\/sVVOdzEsNbGdSrKN+MGOoFveY2S+KEzV+uW5nAQhVELvA448\/7u4tFGmADUWQSWtYA54BMsAbENFtw+BSFWnua0BFLQrWFnDSSSe5x+j6mUdhvfrqqx0sRq0enKtSreepWU+CisIAWsQGBZLxk\/dZVeaw82aygHWGHOCc9wEHHJAMKXCGck0hNyYQqNPklbZwZ+PPMYg1MSMmxAtFFeirrq52Y\/OfKASNi+9kLD0ffPCBU16t6s04ULexN9iJUaZ7Ld97CmhmMkNs\/bzo5FVn0kBDsQdcaYW4p\/RJAuozExl7fVD9b7vtttAJlH8PaIxyuSZh8Y2hOtOdF\/x+DNW5xS2vvRSqg1LO+ZUODxvcKgDySS9UJ8txoxTiG354TllKSXF\/UE8eTlW67ilKs\/bv53r2cyUrVPdvXe8sI6py6zGyheqwgBUTqjmmD9bpoDrMh3zAPiOcp1rVX\/oFrBWoNc0cExMWHvr9+FBdVlXj+lJPtabUSwfVLOB7e1Wzg2RsHnPWplI1CnJNc62zzeiCQiCC7cm4Qd7ndG3a25OltaHOLT70wVtVZyAYmw2LJlWptjYfq+7b1IX0h7VjYFWpkF9dmyrVK9oqZczoMfLBzOlCPm4aAL2ursGlM6TlC3++KhtDdWrKzXzKqWfzZeg\/qkc9plAIkMznKdtH\/xyb\/fCWUtUQMPLtBmGKNK9jEwHwKDYDqAKZ9AfUoNYyvnSA74Pnl770JfdS2ALLME81+6SDat7HW42Sy6N+xnrYYYe5Y4VBtVWYOYevfOUrTiH21X+Fbz4TQQo6x+AJAqo+n1nO8ZJLLnEQjk0Bxdsq4RqToHGpeo46jIUFi0lQY4xqzcl0f+VzT\/E9fPHFFws2DCw3TBDC2hlnnOE81rRi3VNMLlkDALQzuYj6VELHrAsys7kmYecbQ3WmOy\/4\/Riqc4tbXnulg2o6tor1nUeVyTYt6+SEF7rJp3crl6N3LHdqIdCiUK2ArKXAdXB\/OLBO2rr3cFBsG\/37UA3gYeXQbRWqt2tZLatLa4oK1YzNjjFbpVrT1tEHsN9W3d0tDLQ5ovX8Lezpa9ZLzWs+iGNRCUqpx7aqbAHGCtX+zWE91UAi3mkaoPznV6dKj8Z1sqat3Cm7YYv8LFRXlYo8N3l60p5CX0D1MQftJ49PIH95gxuXlref\/O40QfX1Fz+yXzJLR6lIRXODbChNzU3tztFUPaxuqhVSHgZBtcI8f6gUhvWcyDVdVVaSAtWaQm\/2+pbkUxPrqWYiwDaF8C37qmwM1Xl9heW8c5D\/FYgF5vhHAxx5DJ8JZlHlAHIADOUUsAb4grzFQDKWEFWk+SzgKZ48ebIrLf7iiy\/Kz372M7eQEWDisTqgCGRHbZmUau0H\/zHNKpP8ngmqAVEe+b\/77rsp8JsuowP7MGEB0GhMHgBhbDf63WXHHQbVLOLDCqGeZ5RpwA8fe9gEImhcnDuLEVF8860mqfHM555SCwbfg5yP9dynu+468SrEPYXXnmvKgs3DDz\/cPY3jaUA6pTps8ScxzfaahJ1nDNVRP\/mp28VQnVvc8trLh2qrnmr5cM0CApApVOtB1TKiEK25rbF40NeS+hK55JUG+fXBpdK9W8KOQL+XTmgQFSzJ1LFDv6rkeQDVeKdVke7KUO1nqgAifU+1prAradjggFj\/gNiFgkAn6uiqdQkvut\/0OBaKFbj5g8\/iJxp\/oLutXuBij1LtN+up7lVRKjuNSPyhVlAGOJc3bFKTg3zKPlTjZ6bptlhUgOq\/vzktqagrVL\/y+hvJGPgqNAo1dhR8y588cHQK9Op5ECO24TwPHbmnS3H39IS32llQ2F5VaUCeLCVtVd2dnSQIqnltu25lMmPtRm96aUmk9Hm5fPh8MC80VBfDn1xsT3Uuccx3n7BFZepr5X0+W2QyUFXQPyZPLYCZ\/\/znP07NZAEdLR1cYufACoESicqLNxbVFo+1Hlt9s9gBeAITVGY83flHheqwPjJBNQVjUItR1q1VId15cyy+swA27Cyo3DSeCFxzzTXOAqKgCyiG2T+sv1h91VSEBPzCcokHjUtfw7\/MdbD2j1zvrXzuKY0p6jCTLN8Xnm5MhbinyC6D9QPrDBYcXZyZyf4RBtW6XzbXJOwcY6jO7Y6MoTq3uOW1Fyl7Pjexf7LioAJsZVVlu8p9CtUnvtAtmTJPqx7qIIBsTTcH2L28qE3un96U7N9XsAH264+sSVFfM0G1X8GvkPYPziOqUs04tFlFOWiholtAWFEqa5sSAErhlZmzZsry1Wvd79gVFLrtBbXp4QBJlGSyXWghFP7AKECyuAZlFttCJqhmH1TvPUbu5\/ZveG+SzKovdyXfUaqxhfh+ZqAZdXhQRYt0G36As3NYpVrVZ7VXKFSrl7mifp3UNjYlC7X4UJ0E5lKRIVUlMreuLaloK7DbMu079+0mx40bJY\/+Z4KDauv71u2JeUNjgzQ2NCTtJIA4zdo\/8IlTPRGoRpHmvIDsjmiFhupi+JO3JqjmmpPtgJR42BH4bNqMC3pPqA+V7BB+ur10cGl9w1g6gBgF5zVr1jhvMGohC8SwNfCoX\/2zUe\/HbKGac8FPy2ce60EmqFYLxtq1ayN5qukfm5p+n2EtIIUdajTqvKr2pM\/j6QAZMJh4oGyrYKDnbqEatRwAJA0cfuMw\/3PQ9bAWFutR9mPM8QDeKNAdBtVR7inr405XvVNzgeNl1ico+d5TOpkjmw2TRPtUJFeoVg94Ntckhuqon\/Bo28VQHS1OBd2K9Ejj\/5aA3m2q26RcWqRZEjDhZ6ZQqCYd3pKyXinWELZXX7ZaJgC7C19pkpX1be49UucB2Nqoikj1Q1VfFUyBasBWs3n4SrXCrM0l3aOtQVioqEp7uoWKYQFkH5q1nYQtdgTkGYdNd6f9ZoJq9hm55wi3uWay4I8NlSdXbgpPStltBVYUVVWG2R+I5rHhgAED3CNnYoLPvXtlRTKlngVzm9ZOoRoI\/ujNlwMrKlqbBn\/4KfgyqKLVLWoEsKvLSmRp86ac1qoOb1dTJg+\/nMg+osdnX+4prX7IxEL91VqhUG0Wc6e8Jh81lCZBWf3S9knKkCqRzxxxcIpSraCu25NWsVu3Gllb1+BsOLwPQDe2tqWFagCbSUNHtEJDdSEsKv55b4lQrRk6ONegMuWarsxPH6exsd5e36qQSbHVbBRAUe\/evZ3ai2JrFwHyuRk\/frxTcbMtH54tVJMzGm8yFSPxF2eCalVGyRdtVfSw82bCj4ADACcXDIs4q8vZZ5\/tsqygFvM9xgQFKw3ZIwA8zYWtcVfrCeCtmT7oh8kH18rPnsJ+QeMidR4+9ilTprTL3mInTqjGpOeLohznc0+px53jMdFiIkcM\/MYxyMXN0w373Z7PPaWZPIJU51yhOpdrEkN1Yf\/ixFBd2HhG6g1V63P\/KXXFV4DBmppuUldX68A6DKqbpdRZC6zfmswgmms6CKrtYPziLf4CwR2bV4oeg\/0UqslJjCLpL1QEIqvbmgOhmmwOtqXL7xwFqtWGEea11vGqp9qm1LPWC1+pDoNq+rO+5GWz3nExUOuGQjV\/IMgZThtW2SwlLU0OfHVBo8ImAK1p7YBqLBEo5ANaNrhrqgsV9dpbbzJ9o2SXbkhUX+SJRLeqCqmr7p08Dn8w6W\/XmjZZKN2df1pjDuDSr+YCJ8Welgq3Zb\/JBDN36uRk\/nNV5VG3LFTvUNUqJx5xqKuMqKqztaswXiY\/+407yMG7HguorioTt9hS29j+Ca\/\/gtrEYsQB1ajwqUWRimGr4FiFhupIH\/wsN9oSoVrBMQyqfX81hUBYSKYgY9OuoZjivSa3L\/c7lhHsCJkyI6BIo9La1HEzZsxwKi2TZSA3aho9e0mzgWrUZqwWKMlahl1jM27cOGdFAPwtaJKdBBUdqGUbbemgmnRsbE9eY21qeWDSoPmg8ZZz\/nzeUevxXNumsIatQONGrEhFB3CijDLJ0Uwpmm6P62OvB69zbhTAoVGQhzjo4kmOr0o61hyb2i\/s45PvPWXPnTjhS1aw5n7817\/+5RR+4g8A26bZNnK5p+y48XMzwXPfhwsWOHhnEuXbODJNHHO5JmFxje0fWX5hb9w8hurc4pbXXkC1giqQVVaaUKkBa8DQQowq1byv6esUrC1UA7ko2UDXjye0ytL6TXAyrHep\/OiA1FRvYVC9vPs2bixkeQDywxYqpoNq1G79I2jLYQcFLQpU634+VNu8q6q8A6BApVW+8VfTSGe3bu265MSF\/QHApU0JddZfsMg+wOXcKRPdNgDqLqMOSCrV5FPVXNXEY0CfXs6iQbNlum32EFRrAJf4cj5r++4g\/UsaZWl9IgUdTbdRsMduMe21\/zpVm+tLmfLS3gNk1+GJMsc2K8e2BxyWrP7IfcT9NeqAsUm4tWXBZ78\/y00K+vboJnsMH94uO4eeB\/YXTeEXBNV+BpCjRifGtba5PVTPXNPiYNuWJAeqWcQYpFIXw1bB2LoqVNtJBBkJaAogeX3pdIGdrSLMcPw0djpEHquj4D722GPuJRTts846y2WKsI\/ceQ8QoTrhm2++6R6fU8AE3zDQCeABfRb0ACbSywGaNn2apnoD2i0cZhM2TbPGPhyfY2j6Pl7j\/Elvh4f5lltucVkerIfZQhYTBsZOOXMtYMK4sGZo\/modm2Z88JVWnYAQI+Kp+ZeBYBYdAtxUaWRSwtj4rAHBNJRnXbQH5AG4gC\/9WEsGfbGok5hzrqTdo3gN9zHZWPgepO2www4uLzfWG45lry9PDrC\/cH3xizNe8mJHsX4U4p6iD5uBhvFybBR8Kkzyt4EiMDzBIFa26fFzuad0IsdTgIqKCjn55JNd\/3w3bbPNNq6oD7EhbtzDLDAl3twbYRNHxpbLNeE+81sM1dl8+jdtG0N1bnHLay++vJbVJwqnAIOVVVXOf0qeYRRhC48WqheU9XFWkfveE3nxoxZXpEWbQjWKM21KYy95Zm6bzFzRmlKOXLf3oXp4RZ00toqzL2g2jahQrbmXi23\/0LhYANZHtE1rVzqQxvNLFT7AmgYIa0YLFuU1NjS6MuE0MlQglmKl0BZUHVCVapf1Yuwhsr6+URYvX+mqgKE0WftH\/zHjXVc217RVh4Fa0vMxXhagLu4+xHmUdaEisIlqbFP48f77kyfKB41VzgoCVGsFRo7FH67ypnpnQRk27nB5\/o0pbkwsPqxprpO9DziwHVS73NPvvC11tXVuuwP3H+OgllLofsP+otlGBkldilKtOa6tys54geS1TW1O0aZZ\/zTqtQ\/VQSo1+xXDVkG\/XRWq7SSCgg60LQGqOZdnnnnGFanAz6uNDBhAA2XCbWlqrZqIv5oGqFEMhkwTqIOoqVQGZB8sAuSsRmVETcTyABACIVTrsyAEyDzwwAMOKn0VFDUWxZEKjJkyj9jPCOcEQAJB9twy\/ZHwfeMK1SyYBHbJ311fX+\/GT0YIzh1LhJ4P9zDHBeiovkhDMcZeQdEaJv3YGUhPyeeUxYGo1Pwjywmx8VPaaUYVvN7YQzgeJcfpj3\/W+qDnR0U\/Fi1yfVGameSgdAPSqNKo\/mS1sHms+X7iGJSKx0IHVDIRQbnG3hIEeX48C3lPsYCVcXAefAa5jnjdubdIT5cuV3mu9xRAjgqODQkrEPcvE00tiESxH9YZ8Br3N7CMD91ea7ZlcsgEwLZcrokf3xiqM32Cg9+PoTq3uOW1F1\/6Ld36JKFaOwOqW6TUwTWAivIMRANf2oLS2\/GeD9UAeFDlQIVp31MNjPfo28+Bm2bTAKqt\/cMqw3iwgbHZpX0cnLLIDlBUP7Sfq5kx+q8pvPOeFmtRMFcLiVWPgVq\/fLkulKNwCFk+8PzS1+I1613IgGoAlEwULL4BgjXNG2C3XbdNuZPDqgOqp5oMHAtqW2VFbYOD6uHbDZJVzSXuD9aAhlXt7B+VpeImKj5UA5Va6MSHasBWbROayo7jA51YJ6ioWFK\/QcjtrKn5OE8t8Y0l4+W3pyfPl4weyxpak5COkl9f2V322nMvt2gTpZrjZIJq9ZSz4FBLi3MsVcn1aQDnfNQBo6RnBRlNWpNQTew4DosSfahmO8bvWz\/y+pBl2LmrQrWdRGxJUF3Ma7ml9J3JU72lnGd8HptHBGKozu06xVCdW9zy2ouFigvW1gdCNR0DlviVVZm2UM37Csya8SMKVKtHmiwSgKn1GgOuQDVgjCdYK+dZpVoXKqrXdnDDCgfTPUeNT+YxBhQr6ta5Cnna0vmp9VwV9fIc3AAAIABJREFUqrGKoLjaIiY20J8+eHQyD3NyIrIxHzMlrsnyATSO2G9sEiJZiIhSjd2iolsPGblvYqEeIAxoO+uFJJRrXSBoF\/DxulYzBAwbWkRmrapzUD12p21kbiIRiNsGz7CWKQf2dVGgQihj5JhAPh5oqgdi6WEiAOhi58gE1bzPgj5A2XqjgVKA92+vTnZPQXh6ADgP61EmC+pakp5uLfLTs1dP974777JELup0SnUYVBNHfUqA8sZi0AFVJQ6q1zW1Jhd4+lDNsTQjCFDdUQsU9b7pDKiO4g\/fku0feX1pbgU7x1C9FVzkzegUY6jO7WLFUJ1b3PLay7d\/aGeovP1qKlxZZoVnX6nmdVWr00H1krKeSdUYby3e6MpuPWRDa4msKOvhIFMzQtDn4A2LHFRjX1D10Ydqm46NdHCMc0H1ICmXVpcKkFzJFogzAbUP1WFZLNSXvX1Vqyxtq06pbEgfuh+QiaprvbzAGsotENtWVpGEas6RVHFUjOSJAE1Vb9LM2RzLmiOa\/2nvLM8OqoFHVN0Pp7zhbD5NNb1csRbS8M1sqkkWrGEBI9k9pLzS5bJW\/7NVqhVGVfF14y4tcfAMbD\/8wkRZVt\/ivNT7jt5fhvcuc+o6MWABpS6KJH0jVQwVqnuVJ0AY+4f1g2uOaYVqtXHYhYr6lEAnEkA1yjoTEPpTFZ1jEQeuk04CeG1rgeoo\/vAt1f6R1xfmVrJzDNVbyYXeTE4zhurcLlQM1bnFLa+9sH+saS2Tht6DU7J\/ANWkqaPhFcSKEQTVvE\/5cRoFW2i+\/WN9SZXrC1W7e1uD9Gmtc30eMP5Iefz1qSn5mYGhYa2rnVUhDKrxeqNEah7l1W+\/4qDaptQLquqngbK5jvU1jsuiPJoWaKGPj++\/jxujLQsOoG9b3iTbjT44MPaq8AKHPlQrtAFzWB4of+1gtK3JxQWo1gkA5whQ25LdPlTPWlknc5emKtUK3EFKNe+xQG\/6pNeTUI0lhOqBf3zlXZcHG+sxijqe6dLqGpfLWhXuIKgGTtWvbKFalWoU+j3HjE1CtW6LP5xFkTSgGrhFuScdn2bnmDn1HRcjnWTYlIIK1eyjmTz0KQF9cl5cA4VqqiVagGZRIv5x+5ouXAz7UEVReLP9QHamUs1Yw8qCx\/aPbK\/klrN9DNVbzrXcEs4khurcrmIM1bnFLa+9KBn73Qt+Lt\/4yS9kYI\/qZLYPC9UcIMz+oVDNgjVNdedDtQ5Q4VpBHahW9VDTvPEeCxVZwEf2iiClmuPYQh8WqrEa4FMOq+oH9E2aPKldYRsU9G5V5dKtplsyDRy2FIqQLCzp3q4ICZ7qTFCtqrBaIxRMed15iye\/nYRqAJ78z72Gj0lZWIhXnAmD5liunfmGm+DsUd0sY8eNldmr6+X9xStS7B9qbwCSgXLiofYP3gOC35v2rqxelahciZK8bdsGWVI9QA4bjSpd6nJMA9UsXCXLiMKs76nuUbtS1lT1lu1HjExMDoxSrbDNeRMDlGr+1\/R3Wr2R\/bB96CRErRiAMllBdOLBU4BMUK0xpk+1pzhYbxFnPWF82r9CNWq2FnrJBNVRFN5sP5CdAdXZjnFLTKmXbQy2lu0pA07u6Ycfftgt0iNX9NixY7eW04\/PswtGIIbq3C5KDNW5xS2vvfBUz5jzkTQN3Mkp1ZpCz4dq\/M0ACcCsTav5oVQD1VrpMAyq7UBVqVaoJsME\/mmaeqpRqnn8j12itqHZ2Ubqavok4V0tINg\/1qxY5pRq598dOdqpjwOrNi38o1\/1LqP82qqJvKfp4dqqu7vczerlJsuHlvu26rgCc1DwValG\/bXWCAvVlaUl8sxb0xwwokxz\/H6VItuMPjRpeaBvYBjgVDBfMekllxVlt8pGOezww2ThugaZumB5IFS\/vqIpqTDrQkNd4EjfpP7CnqFZXbjGxxy4n4Nq1HnsH+VtLbK+sqf0aVwjqyt7CynqdKEiFpKedStlXU0\/B97q\/wae+TkIqtWGkfBNlzilmPOjAdU0AFfVZ99TbqFaJw9WqbZQreXIdeGhwrz1Tze0tIWm0Au6tsXIABJDdV5fYfHOBYwABVX+9re\/BfZIejkypMQtjkBHRyAMqvX7GPFtn332SUlN2dFj7IrHi6G6E64Kyhsp37AdKFTzyL93SbOr8KdtXklPl5UBUKRxM5MPFDhmcR3F5zR\/sIVqm5pKIVz7LO3WS5ZX9RVyJzs7xMyZzmZh7R9sCxS9M32mDNiwxJXo3nFUotiAbo9qi5d6Wn2lW7AYBtXqXQZeVVVXqwVQiyqM9QFA1NRwfZvWplQmBOT9TB023zLj0oVuwFsYVLMNSqwuVKSENxYZoNo2P63d0LZ1wrXg\/zCo1kImQLVCaBBUv\/vuO24hIZ52FqBaqGayo2XIn5k4RVCkWfRJirp0UE0syGISBtWcG+dtoVptH0C15oj2oZr9eE0XarK\/wvsLr0xw6j1+90NH7pm0okSBavplgoMHvLNaDNWdFfn4uHEE4ghsDhEIg2p9ckhqRvJl23zvm8N5FXuMMVQXO8IB\/eOpXt9SIi3bDHNwhVINVB93yNiULAiAEIqiQrXezHTZ2GewjBszWoAvIHVoeYODdOCYpp5N++gcwKbc+epuA1PSvLF9y3tvOgV3cVV\/55sGuhpbRBIqa0syi4PmX8YuwjGASIXHMKUa1ZtsJgrmc96d5NRasl7QRh0wLgll6rX986ubPNVBUK02Bi25HQTVqpbiJ9ZUcArVHJf0dGQrYZLBOet5KHxqbFGo+4w8xMWCFqRUK1TT\/\/wZb7vJDvaSHXYfsTGGiXzNNMajKfUUqt97Z5J8WFcqvUqapaK53i1Gxe\/NolIL1YwZ2GZSxUREzzEMql3Gko3+Z52YsHCR\/WgKtqr0K3xz33H92Vc94fZe\/OfzL8jshnJnGTryAPJsJ87Ph2peJ1+1KtX8rC2G6vRfPrH9oxO+nONDxhGII+AikE6pprAQLYbq9jdLDNWd8AHCU\/2dH54vF996n5Q11klT7XoH1QOry5ynlmT9JPv3oVqVaobM4\/\/xB41LFuVAqUZxVbjT07KPzul3+YZ6B982d7Lrb8pLDqqpqKjZNIKg2lXIam4U8lRHgWr6BuBQRDWDxNqZk2ROY7lLf0fCf3hMoUwzRVj4VeVYrSXan2ar4BhBUK3+XiwIQVCNlaKifq2bZCi4q78XSP37m9Ocio9CnQ1UvzfpNVeBEajedZ9RgVD90XszXPYPFPATDtlfXn\/lZZndWO4ygaDgr+o2wL1ux67WErXBaH5nYqv+ZN\/+4UO1ptgjpjzpQAHniYVCNftz3dUewntBUP3fVyfKRw2JnNqHjd0vBaq5LgrMvA\/EW6juTJjWz0WsVHfCF198yDgCcQQ2mwjEnurcLlUM1bnFLa+9gqCaQi3qqca+AbD6UM1BFZJRKoFqm+oM+4AP1f5AUQrVU62eYbZBqV69vtZ5mRWqFdasUm0VbPazSrWFWHtcH6rVTgBoYQN4ZdLbTg3HSoAtBWVeoVr9vfSheZRVEWdfrQBIX6qIqv2D7XUfv6If4wPumdDoOWse6dLuvZyPWf3ANvsH41m4rl7e+ijVU22V6rlTJ8mKJpHupW2y\/\/77O2sGYKlNYVjPE9B9\/fXXkxONyg0r3SLVIdXiFGmsNh87aJyLtabUU3jW2GYD1RwfoHaTkQCotteOe4Rz8+9FYFkBHJ+2zWOt6rneq409+ju1nZZpUWJeH6wsdo6hOotgxZvGEYgjsNVFIIbq3C55DNW5xS2vvRSqL7\/jfilrbXHZItbV1TuoHlidoJ0wqNYDAzSAkYVq0pepRSFsgGFQDTjZxWcW1nyotsqohWqbpcKHaruA0UI1kPjofyY4uwTZRTTVnIVqvziJQjX\/a1q3dFBNXDSPs\/Vb66I7zoGGJYNqhfjOjzlovxSoJiWe+okVqj+282B5b0PCzmChGsgkrR9xQoWPAtX0wQSBGM589QUH2KjWPFXAfnLMkYc7qFbFXaFa7Raq1Ou10uwfvlLNOSyv3wT4mvoO9VhVZb12WtkxV6hW61Fj937ymSMSqRA7CqozpeGLoTqvr7B45zgCcQS28AjEUJ3bBY6hOre45bUXUH3m2d+XK+79i+wzYg\/n5Z387jRnRWChovqhg5TqYkO1epl7VZTKIWP2dbYCC9WqRivE+VDN9ta6wXh9r7UP1c9OeEPmNyQmE0FQrf5fX6nmd4XkQkD14umTpbm52Y1j\/MEHFgSq6YuY6CRIrS6qOBMrgN9C9by3X5cP6kqSObSxnyhU60RAoVptFXo9tCiLFsFRP75eJ4VqrXioWT\/UT44yzcSLCZvm+1aotmnwGC+xp1lrj0I\/ryvYskC1o6HariXA9kQ5drVVMbZ0UJ0JyPP68Gexc+ypziJY8aZxBOIIFDQCMVTnFs4YqnOLW157BUE1Ch5qrsIRBygmVNvCIqjdqlRr8RLyKB954AEdAtW2Op96qhWWgb10UG3hklRtQKK1fwB5Cq6osemUalV97cSFn4kN+9GXtX8cucs2Mm19Ik+0PiHgWFap1swaTEz4GWDVYjTqjdZrrkq1noOO3WYWUajWiY4P1fokgtc1HpwDUK2LQAFnrDOUEbdQzXa6sJBYKVzrRAnF3d6fjEEVee4hXQhpt6FPtlN7Sl4fnCx2tmsJ2I1FumqrygTVxciLncXQk5vGUJ1L1OJ94gjEEShEBGKozi2KMVTnFre89upKUK2qsUK1KtVYDw4+6KCcoNpW+yNQmZTqTFANpKLwAorqKwb6gDcLl7lCNcDM4jyg0baEupwA5lygmkmSpo5TqGaMAKj1RnNczSVt7R28DqTnAtVcV4VmC9Uo15qBIwiqFfqJL2NvbE0sYuT8\/Ukf21LcRZVqvc4+VHdGGXJ7HYPyXJMrnkWoVr3WfYqRFzuXL4wYqnOJWrxPHIE4AoWIQAzVuUUxhuqAuK1du1YeeughoWzs7NmzpaKiwlkyvv71r7tE\/KWlqfCVbei7ElTrIjzfU22VzmztHwrVYan2fPuHqqsK4LaAi6a34z3UznRQzTaqwDMGBVeAkDLcVqlW1Zb+g6Ba\/dphUP3O\/GVy9G5DZNIarBKpSjUwTPVGC9Wa3SQIql2sN6a9U6h2KnFZ4mlFEFTrRIRYKvhy\/toPr9M004Yq1cSWbYB43c\/6nK2SzvHZjj6Ip05k9H5nP\/zZKPAab38btu1sqA76fHYVNTrdd0cM1dl+s3bM9m1tbfLSSy+5qoef\/\/zn5ZOf\/KSUlHRezvWOOev4KFtbBGKozu2Kx1Dtxe29996Ts88+Wz744IN2ES0vL5fvfve77h8\/59o2Z6jWxYialSPIU50OqrWACLCoAGkXSNK\/2h9QjxMwjZKbsHakg2rfV+yrwQrQ2B\/UZkL\/FqpVnfWhmokFi\/roY21dvUyYkxtUczwL\/JoyLx+oBn5tFURimytU6+JEtdykg2oFZj0nf9GkhW8dX66fmULvp0o1\/eoahkIfI9\/+tiSofvbZZ4WYP\/XUU8kqruSGv\/POO12u2yjtySeflHPOOSe56Y477iif+MQn5Mgjj+zQqoOrVq2Sb33rW\/LWW2\/JrrvuKnfffbdst912UU6hy2zzxz\/+Uajk+LOf\/Syvv2Vd5oQ2g4EsXrxYbrzxRifO7b777nmN+Pnnn5fJkyfLWWed5WxtxWgxVOcW1RiqTdz4suRL++WXX5YjjjjCfeHstNNOsn79eqFgy69\/\/WtBpbj++uudOpFr62pQTWYLmyWC87L2AatU24VobJcNVGtaORbLdQRUK6BbT696qi1UA7aANvDO65q32Xra84VqFGCU4ahQreox8c2kVNu8zwrn7npuVJk5Juei6rwq1bzu54zW9\/R1xs3PnL\/NP633vlpJ8INzb2g6vVw\/Gx21X5z9o6MinXocYODMM89MLgjmO\/ab3\/xmxsHwHXzeeefJP\/\/5T7dtz5495Z577pHRo0dn3LfQG7CYGSi94YYb5Bvf+IZ8+9vfLhrYFHrs9MffOYShefPmuQnBbrvtVozDxH2aCEycONHdL4BqIe5ZfVpy1113yW9+8xsZNGhQweMdQ3VuIY2h2sRNlZARI0bI7bffnqI+cBPzRXrppZfKgQceKDfffLP07ds3p6h3BajWUtrAF4\/wdcGapqgLg2rN96wnngmq2Y6UeFghokC1Qruqx+wfpFRrFhLrqQ7KgMF5KRjSl7+IUSEXkNaUebq9hWrNLe0r1dPWJSJhFyoyXu1DPdUKn4A1EK\/AH6ZU2\/haqNafrf0jW6imbx1PNlCtSr296bWipaYc5L2upkoHfUhjqM7pqyvvnajEBoguXLjQ9bXffvu579pM36Wowqeeeqps2LDB7bfzzjs7INxhhx3yHtPW1sGLL74oZ5xxhpvYANc\/+tGPtrYQdOj5ItL94he\/kMsvv9wVOytUg0keeeQR9\/QHoS\/TZyjb48ZQnW3EEtvHUL0xblYJCVNPeHyDKjFjxgy544475LDDDssp6j5UA5zqwe2o7B8Wqm1FvUJAtYKuFooBqvE0ky+a5ivVvKa5oi1U87pdQEhs1P7hQ7WFfZtWLgyqFWoVqoFDTT+nKjHAq68rPCpUvz1vqRy0y7YSBtXYVVB2aZpJw53PRqjW81QV3bd\/6I3F9bDArj\/nAtU2JV4YVCtw24qInLNV6n2o5n07zpw+FB28UwzVHRzwjYcDqlGqWbeyYsUK9+q1114rJ510UuiAgL8rrrhCUPuWL18uy5Yti6E6x8vX0NAgP\/\/5zx2Mue\/iLC04OR52q90NO+l3vvMdZ\/n46le\/WnDvvX42uK4XX3yxq\/VQqBZDdW6RjKF6Y9xYkIgSsnLlSqdIBz2iIVsAN+7DDz+c1wy\/q0I1oVAlNx+lOgyqyRQBJAdBtea2ttkvAE2FalV8tQCK2gz0WOmg2i7E0\/PzoZpz1\/R0+lHyoVoVWDzVMz5aIvvtvJ28X5vYWoGc89AsIoxZ\/c668JAJDP3qeO0iPlvcxY5Bx2XBVRd3+k8OdIKiC091zJrqT1PbZVuEJQyqg+A\/t6+ijt0rhuqOjbceTaEaHzv2DaCAn3\/7299Kjx49AgeFiAGI42MGBt9+++0YqnO8fFOnTpUf\/vCHcuyxx8ptt93m4n\/llVe6BZdxK2wESON5wQUXuEkg1o+BAwcW9gBmosrTn5\/85CdywgknFOwYMVTnFsoYqjfG7bXXXpMvfelLGb+sb7rpJrnuuuvczXvVVVfl5KXrylCtsGZtIdZT7UOcenWBQ+thJm2cr1QD1Qqv1lNtYV6hmv5Q7xWqbbEZOwbGtryhLVkxkb58pdoCpM1rrX1qYRY\/l7KWKdfsHGFQzesWfJk0aC5o35vM2LDBaIVHOzYFcvu0wkI3EwKrpitk+3YL68O20Ovnmc7mKyNTrulYqc4mmtG23ZIWKvpQjTjxwAMPCN+FLPoG8FjH4jcecfOdS5pDHqEDDl0BqhkXEzMe7R933HEFf\/Qe7Q7JbisAmnVBeKp\/8IMfOH8vf\/eiWnCyO1q8tdpsePL9ta99rWgBUbHvww8\/jGSlijqQGKqjRip1uxiqN8bj6aeflu9973sycuRIlyqpX79+gRElzd5Pf\/rTvL6IFKqvve8vsvseezi\/cWfYP4BRBUAFOQvVBEAf\/dty6NaHq\/mLrU9YIRzI5hhq\/8gFqjm+TdMGMPpQTQESTZkXBNX2QoZBdVAquHRQ\/cGCJbLvTgml2odqXfSp8eN\/hWedLNjx6vjCckFrvC3Y2gWJUaFaPeO5fFVkgmr1hufSd2fsEyvVnRF1EVWqr7nmGpk\/f34ymwf2Dywe\/uNrtdx9+tOfdkIGXuB0UA3s8sj9r3\/9qzQ1NUl1dbUDx169esnpp58uhxxyiEuJyuNyvodRadXfzTFYNLn33nu74KAw4ve+99573fc9i9j32Wcf+fvf\/+6eZk6bNq2dCAPgvPvuu24b7jEglu1Yg4MvnEwNAFZQ1gbGzrZ\/+9vf5FOf+pTzizOZuPDCC5MZUxjXoYce6iYanFM2bc6cOe645557rhxzzDHuHJio0DjPo48+OqU7LDqPP\/64s+esW7fOTX6wMSA+DRs2zG3LE14yWrDd8ccf7zJn7bHHHu49\/O9cBxb5c14sLmWBPz7u7bffPnksznvJkiXyxBNPyDvvvOPGxM+ou5WVlW5sPDnW\/t544w0h8wtPMMjSxZi+\/OUvt7t3co1n1HGni73abABrnsiwTitdw37KvTh9+nT5\/ve\/nzLBXLp0qbPs8D+TStZ0+U3Xg2WyUmVzv8RQnU20Nm0bQ\/XGWESFZVav8+WcCb7TXQ6F6hsfeER22m33ToNqCqqgXqqnWuHPV0BtcRZfqbZQbVVmhWygGsXaFWrxlGrbly03zra2sIjNOgFUa1VBxqvbRYVqW3HQKtVh+ZVtVUK9ptg\/5i1cLMN33F7m1adCv8bOt1f4UK3VGe19EpYLWqE5KlQHqcb5Qm+mXNP59p\/b11fue8VQnXvs8tnTQjWZlQAsoBdgQ7mmDoBtwAIZDgA3CvWcdtppoVCNEsviRWCRjAgsCiN\/NI\/hsZeQvg+oA14APBrwC2Sy76233upg0zbGyxhR1oFZoBkAAhbpxy6Y5HXyV9Mn\/\/gbAYQDaR\/\/+McdvN9yyy3OS44FAwC17ZlnnnFj+eUvf5m0Y6hSz\/hzhWk9BnEB8ogNohGQTTwB47AnrxyfuLAPMaMPzss2rt9FF13kzk1TxZGuj8kA1p4vfOEL7vpim2SxHhMLzdzCsQFB0i0C7kxe9txzT5k1a5ZLGcfEiFhhT2FSw+\/cC4MHD3aZuOgTkesrX\/mKy9YFhGvLJZ5Rx53pM8D3C5YMJj6MN1PKRV3AC4xzD+vEjuMQXyZiQ4cODV2cS6zYhvs3aHKaabxB78dQnUvU4oWKyagpVGeydahNJCpUo8bwz7aZM2fKZVddLbc\/9FcZsvNuDqqD4M2mQfMvr1oWrILMQkBdyBZ2OwTZOywwqQKqWSbox0K1BVp9D++wplSzxVdQqrUBxj5U2758qGYcDjLLS1JKXAdZG2y1Qzte65vWcQR5kbXaYVAlQPbzX6+vr5e5CxfLLkO3lyWN+KYTkwZaGFTb4xOHQkA1NhLfB27HYO+BTEpzpq+PbD3Ymfrr7Pc3V6i+9J+zOzt0aY9\/yTEJBTOsWahGfbRqKVBg8yZr6jfgGLBFOU0H1Y8++qjzsLIwjKeOtpaATZdqAcy+7h+fcwDq+Qf4KYjzejq7oL7H9pwPKjxjAQKZIACWH\/vYx5zarH0C5MAjggsQhuqrTYGpd+\/eOafA0\/NEjf\/MZz7julY7CE9mwyY1bAd842dHFfYX8Sv0r169Wn784x87tRiFHwAmhzfAq3mU1WNMPH3LCXG57LLL3Dg4Bmo4gMskgInQ66+\/7rKUoPgyXqCaxlMJ4JXJk00PmEs8cxl32H2OLYj7KepESMW6oO2jWE4Vypl4FipvehSoJnGDn7CBpxD2SUSX\/sIqwuBipXpjUIsF1TzC4p\/fSiur5bpbbpcBOwyTnuVUrmuTbSoTeYy1zawrk34VIoPKN1ZBMe991FDqYA6oo+1c3SK8Nrym\/ba2zw2tJfK\/+jLZp1uzLG0uc32vbimVPmWtbrOmthLhuLyvTffhd\/qvKEkdp77PGLqXtglJPmY3lLvzqSxpS26\/sqnE9b9DVas7hu2LMdEP7zEWtuN8+lW0JcfG8XU\/e07sy346Lo6zoKnMHd+PnY7VHpvtWVDI2G3jPPiy9l9HTeBRHEUrlreUu+ug8Qsan9\/nhpYSd139OHK+gys3xcu\/Z7jWeo8Qn1UtpYH3RtAY7DXO5XtkS4PqRYsWSffu3bN+hJ5L7HLdB8WJxXuohNpKzns+1+46ZL+2a45MexwfqhcsWOCAjUf5LOSyj8oBKh53ozCj3LGIPAyq1SYCiKHq+lYGBqXf8T5A6qNz7BYcX60NgBmQTnEZBVE9uShQHZT2L2w\/7kfODcHFh2pVPbFIhC2iz3RxUc6J4+9+97sU4AHYWaCPShw0qaBfq5b7MEzcAWg82mpLIJ6o1EHXQeGZfv\/0pz8l99Frw5MKINJf1Ac8o+IDndgjdFGrwuSaNWtSYpNLPHMZd1jc9Xw+97nPuScPVkH39yG+fMZ5IkCGEKweOiGMkpWM\/pgQ8ASd+z\/Xe8QfF1CNgMRkyW9YsLgOQY37wRZpynRvbmnvx1C98YoWy\/4RpFTzx+KOu++Vu+\/\/k2yz484OyhrbSmSIlw1n6ro26V\/Z\/nWGPKeWMtYlsrAuAcO7dRdnRdi7Z\/pyuWubWpPbrWwSB3frmsWBPQ2457j79dmkMrPP+4n0sK5\/jmubvr97j1LXT2OryLtrW2VotUhpaYn0KBN3fisa2xwsD61uf4z31re6cezUrcSdM43X6NM2xuafI7HYtrpEKjduquPZtqa0XUz1\/Ox5EAfA2T8vzkP7tGNAceEPHI\/0lrdWuImDjjlofHZfzhFRO6hfriVjDmv++4sapN35sS\/XivvBP65e41y+xMJikUtfXWEfYA6o7tOnT1cYTuAY9I+rheotTam2wEYQNG+yelJRclUBTQfV+GzVVxsGFWS+QNXEfgEsA\/M0BXK80FaJxQN99dVXO5XaV94KDdX23ABUlHZt+h7glEuxFgUzVF6\/GrBNsedPauxNyaQH+FafuU5afCVfF83x1GCXXXZxnvawhj9ei6jp39+oym5ra6tTzrkOXB9+t9c923jmOu6wc8NLz+LbKFCdTlXnnmWyxZMGzi\/IT80Y7PkG2Zhy+ZIDqnma8atf\/ard7sSdzw+2KNIy2hYr1XyrxU06eqEiN+TDT\/zd2T8U5nybgZ8GzV4ma\/\/As6x5oKPYP7AfkEVCC5TY4ihB9g+1THB83\/7Ba9ZSgv9X+8AzTV5m9Uiva2oV9XFzbtZiYlP56cK8IJ+uzYCh8fCtDTrrOlteAAAgAElEQVSeIPtH0AI\/e\/42xmHqLLN3lBCUrdq2smTaPPYNGl\/Uj1cmi4bvaw7bfnPzN0eNTyG321ztH4WMQWf05SvVjMEWhFF1F9DAc4xVQkEiHVQrlDFRCoNqVfOAZws7QUosky0mM9gZfBBlzIWGagu3fopBVV2BFd+GEuUa+oVz0u0TVgzGjk8XlQKyKNJYWVTJ1xLuKO7ZKKZRoRpFnYWPwN6JJ57oFHYmSr5SnW08cx13IaBa73\/uQyZN6kvXHNQ8PQFcsb0MGTIk8JD2s+E\/6YhyjwRtE8X+EbTANdfjbSn7xUr1xiupSke6xQBsGsXflOnmwDcHVPP4a4d9D0jx5Np900G1+oDJGgLIAq5RPdWZoFrLlutYsoVq9tMsHQrVACoLIumLxZFRoDoIGgsB1Vr9L5kib2Mpbv+6RYVqPxuKpr3LdB\/472cL1WGLB2Oozhz5GKozx6gYWwRBtfX2ckx8uNir5s6dmwKR6aD6scceS1YGBEyCCnPZ\/f2FbXbRHqCA3QSPM9BoF41pTAoN1fSrhUJYvEfa1v\/7v\/9z9jOsGyyK5MmFv5Ay0zWyafRQHNXfbPezSn26YjAK56jPxLisrMyNC5uSKvm5KqaZoJrzIMMI9hEsQePHj3exCbN\/ZBvPXMcdFn8mP2R8iaJUh\/mprTUnUz8xVGf6JHTc+zFUb4x1FN9aIYu\/RIHqsPzJDFkBTLdhIWC2UB0EjbymZcujQnWQ+lsIqA5SkKNAtY4nTKn2F\/iFKdVhH0OrVLeUlqWU5c6UJSPdRzsTVPvjDDtWPmPouK+ezj1SDNWdE\/8gqGYkFiBQh\/GgsnDNFrNIB9W6mI9MG6RkC8oLbPf3t7FgjxI7duxYQWgJA9FiQDVxQE0nkwbVI\/fff3\/3NAzVFWvLXnvtlXVFPj+NXtBVRyFFVb700kvd22Hxs\/5eFG2uEWq1VfKtlYFFbABw2DH5DLIuBf9wJqjWRah4jtUORL\/poDqbeOY67rBPkZ7PZz\/7WXcPhVU6tH5qPNFMKJksoMBfcskl7nPB5JKJC1lUwpre21hisnlCkO5bIFaqc\/uOjKF6Y9yiLAgoZJnyQkI1KjUL7XyFOeiWCIJmu10uUM3+fqo74BdLiq9U42keUB1NqQ6C\/iAVNih9nV9cRs+RbX2oznYRnoVqm2GAY2QL6Db2mfaNCtWZ+sntq2LL2iuG6s65nmFQ7ZfPxvLhV6FLB9WaKQTYJX0dSqFfoVEzRQDQPEr3FWhVYhkLmTZ4jB7mYS0GVDMuKkYi3pBCrhAlpwGsZ599NmNFP6vUpysGo4s6+d4jJSJp9PBOa7OQmM6jTXaMV155xS0+5DzTQbUFXj8VYTqoziaeuY477FOkT765f9LlFLfnpjmmGTf7MKannnrKPbXhOvIUgc9AUHo+rQjd0tKSk+8+6DxiqM7tOzKGahM3\/cJgQQePAO3Na2fzYSuUo16CqPaPdEq1ghNAqC0KVLNtOkU0DKqxjGDfCPJUh0E1NgirmvIzDQ94FPtHEOwGjT1bqLZ5uRlPIaE66j0QtF0mGPbHGaZIZ+onnzFuKfvGUN05V1LBFoDwgVUr0AEVWB1YGIdqpy0dVLON5pzm56BFXZq+j0WAfso99rHCCgvx0vmXCw3V\/H0BqFFksQ2EFR\/L5qqheuNLR3UnG4MvANi+\/ElNmFfWWkXwMlvVWPuzTx3IpAIgavETru2\/\/vUvV1AH9RUPPS0dVNvrTqpBKhnjnUclf\/DBB929QrPZRHKJZy7jDrse6oPX3OlheartegKsNKeccorzjFMEhycVpBEcNWqU\/P73v3f3B8WLSEXpN12EO2bMmMAJZTb3jW4bQ3UuUYvzVKdETfNU8oVJHmpWtvI\/jxT58FIFjHb99dcnVy3nEvZCQLUetxhQHQSdAH46qPZzRSv8+lBNTmtU9VyhOsyyYqsK2oWS1u+sAN1VoToT3EeF6kz95HLPbmn7xFDd8VfUFusgwwXAZ6FZoZZrA4hpajsdqc1AASD64AzAAGekMEM9JfsC9gka+3I8skugjvbt2zcwAID9mWee6RZI+mn07A66sD1IjVWPLONHEUfR1abCDbYHm8lDF8qhlgNgeJSBKU3FVlFR4YCKYjLp4FiPY0WgTNYB3UczVvA7Ex7gPihOxJ0JB\/ENUvIVaLGxcE1onNOAAQNczQaUaeKrvmje11R7QfUffM8923Bc0rrtttturlQ8Si1KLr9joyB+PA3OJp65jDvsU6RZM5gkpbNj6L1CP2S6IUbcU6Q+vO+++9w14J7dd9993X108sknB1qAtJ+gyWiun\/QYqnOLXKxUe3FDSeHLF2+S3\/gywz8WtBo8m\/BbqN519Fin3PqlpukvnVKdD1Rn8tz676tlQqE66Fx9r7P2EQbVvqqu5cP9io1R4hqmVAf1pQsm\/UwrUY6j26Szf2TTT77bZrqO+fa\/Je8fQ3XHXV3sDJSdpmoeVgSq4vFdeuyxx7qFd6RVY9EbDegk1679juV31Dv2xV+qDQihnDd+Y6rw0QAjrAz333+\/Ox4Wj0GDBgnFSdj24IMPdmXKwxpwS9YPjh+kLnLfUDWRnM86FiAPgCMLBsfENkL2CxqFa4B0VMd\/\/OMf7j3dD1hicoHyqI\/6mRCQ8i+s8ZSU\/M9AeVjTXN0cD7jr37+\/O3dUUI2T3RerAio\/VgN7bMqNc15k2dDrw37E94477nBpCW1BHNunlghHYX3uuefcNQd6ud74jHVhI4ouqjUVKvXYxOWLX\/yim0RoKXYEL8Qs1FoaC1HxzaPCM3HhPVRvyrAfd9xxbrycT7bxjDruKJ8eYJ9CRHjL\/eqZeq9qfmrGPm\/ePLc4lcWxTGYmTJgggC0LQ7kfuQ5BEyoFeI7HEwZrx4kyzrBtYqjOLXoxVAfEjcdNfHhRPZgBoxKQ5ujrX\/+6K6Ob7ks5ymWwUD167LhAoKafbKGa7aNknsgEY0GQ+v\/tnQnUVtP+x3+VpEmSiAaVRFdukmhdyk2KW7il6JYrpVQaJEWj5lHzRIMMSUUTjYRcZClJhmgQhSJKrso\/lfRf3+3ud533eZ\/p7H3Oc87zvN+91rsa3rP3\/p3v7zzP+Zzf+e3fxuLFQ7+fijl+ZK5zLKjGA4Q+N6etXkK1Hh\/gHBmpxu8SnX8iHxKqEykU\/t8TqsPvo9xmISKvuOegfjMgFLm0ugH2P\/zwQwWKuA9Fg7Tcplei8w1aT6TUAOrxlgSwiyi0s2GHUPjxyy+\/zFZKL9F5Rf5e58NjISNScpJ5k5HMHITqZFTKeQyh2kw3q15uoDoWGGoDnOkfkWkNsYx0C5U6uovyfbGgPRKqo9XAxrwaqiMrjCBdBFuux8rZjid4rAWNmQ7VTPMw\/xgSqs21Y0\/vFUCUEWka2GwmWgk\/PSOO0xHYeLv0eW9heo0YFj11mUQ8CEVWpNH51IhSx1vMGE95RKlRfhFv1pFSE7kTpY3XCNVm6hGqzXSz6uUXVB87KVEjs5HGmkA1xka\/itgeMUqLXECogc+5aE5DNVJdIo8nVLu\/pAjV7jXTPQjV5tqxp7cK6MoliDAmgiukx2CjE9TZZouuQNj0BOAjVQapP85Fhjov31lKz41Pkary8ssvq\/VeWOSoF326GSPesYRqMyUJ1Wa6WfXyA6qj5WTHMtItVGOcRKX4IitORAM+HINFirA12u6AppHqaOeJyHmmR6qtLsJc3plQncsvgBCdvt4jATm1qGmM3OfIFENEJJHLDRADoHkZkQyRFJ6YEjY9Ab8AaxQ6QA41dkzUedBYjGiyAyLGxPWAtQPIwY5c1OuFkIRqMxUJ1Wa6WfVyQnWsOqiYIB4YagN0+offUI354pXiS6aMm9PWyON1pNq5dbmNyNAOOzdG0yUZW+PNHZacaht9cntfQnVuvwLCc\/4ALFR5wI9eWIhFnFhgiLZ9+3bB1upYxIZ861iVS8JzRsFaElY9kQqCqh49e\/ZUArVt21b9GVkhJhn1AOkffPCByq\/363ogVCfjiZzHEKrNdLPqFTRUm0Kl31B94NifdbC9aH5u1U2o9sJDwY5BqA5Wf86eXQHUXEYVDlTBeO+992Tv3r2q+kilSpVUBRFU79AVM6hdYgXCrufRo0cFP0j5QQUVZ2nJxGeXmiMI1WY6E6rNdLPqlSxUA2Kx\/Xi0ChbagFTm1XoJ1ZF2Y+x41UXcCk6odqtY7jqeUJ27\/M2zpQJUwJ0ChGp3eumjCdVmuln1ShaqkXeMzVLSAaqTEcQJ0oTqZBTjMX4pQKj2S1mOSwWoQCYoQKg28yKh2kw3q15eQrWVIS47myxwjDWF31Dtpa2R58D0D5cXTggPJ1SH0Ck0iQpQgdAoQKg2cwWh2kw3q17pCtWmudjRxIq25Xa8OthuBSdUu1Usdx1PqM5d\/ubZUgEq4E4BQrU7vfTRhGoz3ax6uYHqWFuYWxlg2NnP\/G1AMJrN9uHO0yJUGzo5l3QjVOcSR\/M0qQAVMFKAUG0kmxCqzXSz6pWuUG110gk6ew3VXkbVI01n+oefV0JqxiZUp0ZnzkIFqEB6KkCoNvMbodpMN6tebqDaq8itlcEp6Ow1VPsZVSdUp+CC8HmKdIVqVMkJcytdKPqOq2G2mbZRASqQUwFCtdlVQag2082qV7JQ7We01eoEfOjs3G3Rh+E9HZJQ7amcgQyWrlD9\/k8nAtEr2Um92rwp2fl4HBWgAv4oQKg205VQbaabVS9CdU75THaGtHKCRWdCtYV4IelKqPbHEYRqf3TlqFQg1QoQqs0UJ1Sb6WbVK1mo9jOFweoEfOhMqPZBVA4ZUwFCtT8XR6ZBNXY2HDt2rJQqVUo6d+4sRYoU8Uc4EbWLYqrm8u0kOHDGKECoNnMlodpMN6teyUK11SRp1jmdHiAYqU6ziyuKuYRqf3wYC6pfeeUVWb16teDPEyeip7BgW25sxY1tuRs0aCCVK1dW2zgH2caNGyfTpk1TJkyfPl3Z5VdL5VxenQO2A1+5cqV6IGjbtq3kz5\/fq6HTZpw5c+ZI3rx5pWXLlurPTGmEajNPEqrNdLPqRai2ki\/wzoTqwF1gbQCh2lrCqAMkilQ\/9dRTMmzYMNW3UaNGMnz4cDnzzDPl+PHjsn\/\/fnnnnXdkxowZ8vXXX0uJEiWkX79+cttttwUGK5s3b5aHHnpIKlasKIMGDZKyZcv6I5yIpHIuL07iyJEjMn78eClcuLB06NDB1yi+F\/b6NQYeLBYuXCgbN25U14ifbzP8Oodo4xKqzdQmVJvpZtWLUG0lX+CdCdWBu8DaAEK1tYRGUL1+\/XoV0UMDrHbp0iXHOEePHpXZs2fL5MmT5ffff5e6deuqtIjixYv7Y3SKRz1w4ICcfvrp6mEiXRuAeuDAgeocevTokTEgaeoPXKdTpkwR+HbAgAFSoEAB06FC049QbeYKQrWZbla9CNVW8gXemVAduAusDSBUW0voG1RjYB39e\/TRRxVYX3fddTJp0qS0B2vAKCL1zZs3l+rVq\/vjBJ9H1QC5bt069eCDtB02kcOHD6sHDPi1Y8eOkidPnrSWhVBt5j5CtZluVr0I1VbyBd6ZUB24C6wNIFRbS+grVGNwRKwB1UuWLFFz4dV6q1at\/DE8BaOeOnVKXnzxRQXVyMNNV6gGTGPRJiLVt99+ewqUiz8FHlSQPoS0C7wBCLJBm+7du6v8+xo1agRpivXchGozCQnVZrpZ9SJUW8kXeGdCdeAusDaAUG0toe9QjQk2bdokrVu3ll9\/\/VWqVaum0kLOPvtsf4yPMSoikG+88YZaOPmXv\/zFaG4A9WuvvSYAFUThY0G1F3MZGZhkJwBsz5495ccff1TpDmGIUo8ePVrl4cMe5OkH2Q4dOqRSmkqWLCkjRoxI6zQQQrXZlUSoNtPNqheh2kq+wDsTqgN3gbUBhGprCVMC1RpSEAFEmzdvntSqVStrbsD24sWL1UKxzz77TIoWLSoNGzZUkVRUEkHbsmWLTJgwQd58803170svvVS6du0qN9xwg4KeY8eOyfLly2XixInq93h136xZM\/n444\/l2WefVUCNiiXRoA2LK1944QX56quvVNk9PAQAjO+\/\/34FeLoaBOx7+umnZefOnWqOiy66SM444wz1J6Lx27ZtSzgXwHzHjh3qfGEP+iNHHXnN7dq1k2uvvTbbgk4AMBY\/Llu2TC0mBCStXbtWZs6cqbTCQtBu3bq5qlqhH3JatGih4DpRZBh+Q1WTiy++WPr3758tjxy6LViwQC6\/\/HLp1auXstGk+QHVSHH54IMP5Pnnn1ca\/\/TTTzlMu+KKK5SW55xzTtbv4KOpU6eqawV9a9asaXJKoehDqDZzA6HaTDerXoRqK\/kC70yoDtwF1gYQqq0lTAlUA26GDh0qzz33nJpv5MiRKh8ZDSDbt29fqVevnvzrX\/9S5feQXoGKIgULFpRnnnkmK8Vi9+7d0r59ewW1WCiJ1AVn+TekmvTu3VvKly+vgBsw\/Msvv6jjEXkEPEdC9ZdffqnGRPQadiH9APaiIgZe\/2M8\/F63L774Qtq0aaPGdUaqEblONBfGBZSj35gxY1TZQeTswm48MDz55JPqYQJ24MFi37598v777yvdAMI33nij0gRR\/tq1ayvYBhDGeliIdXUAGHF+yaTiADBhK7S4++671cODLpGo9cbDTKwFq8leoV5D9aeffqpSdFDNI1678847ZciQITkeLFBiENcQHuyQY52ujVBt5jlCtZluVr0I1VbyBd6ZUB24C6wNIFRbS5gSqMYkGprwdw1ggFxEWStVqiR9+vRRwIjmhDXktAIcUTUEgAe4A+QhUg0IveCCC7LOAdCNXFgAfNWqVbP+\/+DBg6r+MqLWkVCt60rXr19fRWN1KTVEZ5H7HZmuEguq9WTx5kJeOSAdEXAAm7N+988\/\/6y0wLz\/\/ve\/VURYR5ARDYY+OFc8bNSpU0fBOCAdQDh37tyoDxnRnIu3ArAB0JhMqgUi5fAX7neRx+PzhweMb775Rj0oON8+uL0yvYRqnS+Otw3QDBpdcsklKsqP6Pwtt9yScAGirnCDhxc8hOAhJx0bodrMa4RqM92sehGqreQLvDOhOnAXWBtAqLaWMFCoRoQTUWpEaRGFdTZnLWxnugjAGYC8a9cuBcFNmjTJ6gaw++STTxRUa0DHL+OBLgAMdbTvueceBdEadDVUobY1osu6trUpVCPqjDrQiMxHO1\/YqeEZNjjTDvT\/o3oKAM9Zxi\/e76I5V2uB6H0yCy31+SK9Bjo4H1beeustue+++9RDEfLkzz\/\/fOML0iuoRmoNHlpwfeCBCNcIfIiGtBk8dCGCnQisMQ4eGHAdOf1vfIIBdSRUmwlPqDbTzaoXodpKvsA7E6oDd4G1AYRqawkDgWqU1bvppptULWBEb3VucqyzAbghLQIN0VkAGCAOKSMAVESXEel98MEH5a677sqxY2I8qI6cE+MjV\/ntt99WY3sF1UhDgG3I\/44Fs8gbB8gh99eZduIlVCd6KIjUAyCKXO9IoIdOWMSH9Jxbb71VRo0ale1BJpovnW8rTK7cRJF15xsORJax8NEZPf\/+++\/VAxm+N5544gl1TrGa1gnRbvgLgJ6OjVBt5jVCtZluVr0I1VbyBd6ZUB24C6wNIFRbS5gSqHamHAB2AGLIe0auMhb3JRMxdRqqF9oheqojuogsP\/744yoyiaoNzpYMVCOSvGjRIhUFRvQbYI28ba+gWoMxFvLFOl+kwwBgkQ\/szPUNCqqd+dSwCykoum4zHgAAqLAZqSr33ntvwosRqTuvvvpq1OP27NmjHiYqVKgQc0Md5DdHvtGIdl3geoNtWDjpTLFxXgeJcqXdPnwkPPmADiBUmwlPqDbTzaoXodpKvsA7E6oDd4G1AYRqawlTAtXOlA1EDhGpzpcvX1aeM2CrQYMGSZ8MYBoL5gDBSNkA7GG3RuTPonRf5IYd8aBa7\/yIqiJYuIcqFmhep38sXbo0a8Eb0gmuv\/76HOfrtNOZV+0lVCMtAhphrkQPM7HyqaE\/8pTnz5+vqn0kGicZx3qR\/qEXYKIiSmSqCmzQkWo8yCV6ECBUJ+O1zD2GUB2AbwnVAYju4ZSEag\/FDGgoQrU\/wl9dIn\/cgZPZplwPgGgnoGvw4MEqaoiFdih1h2iiXgCHXGNEFaM19IefAczOqKPO50UeLwAbixYxNlJJIlssqEYKw7Rp0xTkYwwnkHsN1ajUgQcAnHesqhtOO53HeAnVeg6AJXLV421eo8ESPgCkosY3GlJCkIqBhZ94CIH2kW8H3F6ZtlCNjWOQToTKMc7FrU474FP4IF4Kjj5enztKHjKn2q030\/94QnUAPiRUByC6h1MSqj0UM6ChCNX+CO8lVH\/++ecqzeO7775T6RTIFUYOtDO1AECGlJAqVarkOCEsJHz33XdVvjRgSDfkUOMVPkAJ0VKkJ+DfTvDWx8aCamfkMjJf12uodtobWWlE26kXxwH2kTOuFwV6CdWIMmNxKCLniXKUdT61E1JhIyLCqMaCUn86TQVpICj151wg6ubqtIVq53lFy\/F25oCjAggA3Hk9Rdqqc+Dx0IHztX1ocKOFl8cy\/cNMTUK1mW5WvQjVVvIF3plQHbgLrA0gVFtLGHUAL6AadZv\/85\/\/KHgBUGOhIepKO+EE0VtEh7EYDLm0gBcN1oAg7F6IjVuwIE5XcHAajAg4IroY0wmhkScVC6p1NBL2OXOGAWjYRAYL3RAhR6QSG5+g6TJyqOKh0zhgKx4SUDM7XqrJqlWr1MMBWrQSdPp8UG7PWXLPS6jG3LqySmQNbqduzocevUGKBldUSsFDEB54UFMb4AlIj3zwcXN12kI15tLpH9iwBwsnnRvR4M0GapXDVqQKnXvuuXHN0w8UuC6S2SDHzbmm8lhCtZnahGoz3ax6Eaqt5Au8M6E6cBdYG0CotpbQCKqd5e4AywAYvWkKIrJYSIjUAkDXhRdeqHKnmzZtmiOKCXBDXjRK2gFM0bBlNna3w8I1RBKddZkjjdVl6gC8kWX0nMc6QdiZVqG3616zZo06HNUgLrvsMrULH6LEK1asUIvnsEkLdnZEHemzzjpLQRb6IIKLBXrYrfGOO+5Qx8WaC+PjHAHIyEdGmgqgXZfq27p1q4I+2AA4RSRYN603xkf+ebFixdSv9M5\/qFLiZvt3XWUE1VOibXyCsZ351Pg3HnrwwIAo95VXXqnOG+eKFJ6vv\/5a1YBOBKrxrla9iDHRYsR4Y+iNfAD\/eMhCbWpotGHDBmUf3hBAW2dJwljjQXPoioWvbvL9\/flEmo9KqDbTjlBtpptVL0K1lXyBdyZUB+4CawMI1dYSuoLqV155RVavXi34E7v4RWuAr3LlyingvOqqqxTsxUsJAPSg0gYqd+itxLGxC0AGudd6m\/Joc6EvwBR1kqNVhcD26NiREAvq9PbmWMSGSCt2bwS8Y+Eeoq2IqsNOzIvf4zwQIcfW5IhuInKMLcSxCBKpAYjAA+KwOQhgGJCcaC4Nwli4iQ1b3nnnHQXvgNH\/\/ve\/qnby3\/72t6xtymE\/otuIwCKajoYoLIAW82F78FmzZmVtv924cWNlJ2yP13TpOegeK19YR\/HxIHDy5Em1EQ3y4gHiyAuHNqgz\/ve\/\/12l9OBhKAwNG9EgEo20I2ySg4e8QoUKqVxq5IRHLmKNZrN+oMCfyLdP19QPnBuh2uyqJFSb6WbVi1BtJV\/gnQnVgbvA2gBCtbWErqDan9k4ahAK6NQbRJ71lvFOO\/Q23ZGl9GxsjVdSL5lxH3jgAalbt24yh1odgzctHTt2VA9OyM9O50aoNvMeodpMN6tehGor+QLvTKgO3AXWBhCqrSUkVPsjYehHRQQaCxWRO4xt4J27ITrzqRMtZnRzon5v\/uLGlljHIjqNEo14i9GjR4+sbeu9GDuIMQjVZqoTqs10s+pFqLaSL\/DOhOrAXWBtQLpC9dZDf+YPh7VVOfO0sJpGuzxUAACJxaMoG+eshoG0E6S0IL3FWUrPw6lDORQeNJBOgyg+cvnTOe1DC0yoNrvUCNVmuln1IlRbyRd4Z0J14C6wNiBdodr6xDkAFfBIAQA0FnmiyomuOKIXMqISC\/K5k1nY55E5gQ2DajXIn0fqB0A0E4AaYhKqzS4pQrWZbla9CNVW8gXemVAduAusDSBUW0vIAaiAoFoGSheioVILFo2i8kX37t1VxDqZxX3pLuOSJUvUYlHUsDattR1GDQjVZl4hVJvpZtWLUG0lX+CdCdWBu8DaAEK1tYQcgApkUwAR219++UWVokOZRFT9YEtfBQjVZr4jVJvpZtWLUG0lX+CdCdWBu8DaAEK1tYQcgApQgQxWgFBt5lxCtZluVr0I1VbyBd6ZUB24C6wNIFRbS8gBqAAVyGAFCNVmziVUm+lm1YtQbSVf4J0J1YG7wNqAdIFq7OiG3fDYqAAVoAKpVEB\/94wZMybHtOvXr5eWLVuqkorRNk9KpZ1hm4tQHYBHCNUBiO7hlIRqD8UMaKh0gOrFixcLbl5sORXAZxANJd3Y0k8BlKCDD5F7zRZeBWrVqiVNmzYlVLtwEaHahVheHUqo9krJYMYhVAeju5ezpgNUe3m+mTYWtpBGreSyZctm2qnlivMBVOMziI1j+GCUfi5npDq2zwjVAVzPhOoARPdwSkK1h2IGNBShOiDhPZqWUO2RkAENQ6gOSHiPpiVUE6o9upS8GYZQ7Y2OQY1CqA5Kee\/mJVR7p2UQIxGqg1DduzkJ1d5pGcRIhGpCdRDXXcw5CdWhcodrYwjVriULXQdCdehc4sogQrUruUJ3MKE6dC5xZRChmlDt6oLx+2BCtd8K+zs+odpffVMxOqE6FSr7Nweh2j9tUzEyoToVKvs3B6GaUO3f1WUwMqHaQLQQdSFUh8gZhqYQqg2FC0k3QnVIHGFoBqHaULiQdCNUE6pDcin+aQahOlTucG0Modq1ZKHrQBXKZlEAACAASURBVKgOnUtcGUSodiVX6A4mVIfOJa4MIlQTql1dMH4fTKj2W2F\/xydU+6tvKkYnVKdCZf\/mIFT7p20qRiZUp0Jl\/+YgVBOq\/bu6DEYmVBuIFqIuhOoQOcPQFEK1oXAh6UaoDokjDM0gVBsKF5JuhGpCdUguxT\/NIFSHyh2ujSFUu5YsdB0I1aFziSuDCNWu5ArdwYTq0LnElUGEakK1qwvG74MJ1X4r7O\/4hGp\/9U3F6ITqVKjs3xyEav+0TcXIhOpUqOzfHIRqQrV\/V5fByIRqA9FC1IVQHSJnGJpCqDYULiTdCNUhcYShGYRqQ+FC0o1QTagOyaX4pxmE6lC5w7UxhGrXkoWuA6E6dC5xZRCh2pVcoTuYUB06l7gyiFBNqHZ1wfh9MKHab4X9HZ9Q7a++qRidUJ0Klf2bg1Dtn7apGJlQnQqV\/ZuDUE2o9u\/qMhiZUG0gWoi6EKpD5AxDUwjVhsKFpBuhOiSOMDSDUG0oXEi6EaoJ1SG5FP80g1AdKne4NoZQ7Vqy0HUgVIfOJa4MIlS7kit0BxOqQ+cSVwYRqgnVri4Yvw8mVPutsL\/jE6r91TcVoxOqU6Gyf3MQqv3TNhUjE6pTobJ\/cxCqCdX+XV0GIxOqDUQLURdCdYicYWgKodpQuJB0I1SHxBGGZhCqDYULSTdCdS6D6lOnTsnu3btl7ty5smbNGtm7d6+cdtppUrVqVWncuLE0bdpUChcuHFOVQ4cOyYIFC+SFF16QXbt2Sf78+aVevXpyzz33SM2aNSVv3rxWlzah2kq+wDsTqgN3gbUBhGprCQMdgFAdqPzWkxOqrSUMdABCdS6CanxYAcNDhgyREydORD3zSpUqydSpU6Vy5co5fr9jxw7p0qWL7Ny5M8fvAOadO3dWP\/i7aSNUmyoXjn6E6nD4wcYKQrWNesH3JVQH7wMbCwjVNuoF35dQnYugeu3atdKxY0fBh7ZFixbSqVMnueCCC+TkyZPy\/vvvy5gxY+Tjjz+Wq6++WiZPniznnntuljr4ou7WrZusW7dO6tatK\/3795fy5cvLkSNHZOHChTJ69GhBFHzixInSsGFD4yubUG0sXSg6EqpD4QYrIwjVVvIF3plQHbgLrAwgVFvJF3hnQnUugWrAb8+ePVXKR+vWraVv3745IsqIRN9\/\/\/0qrWPQoEHSqlWrLHWWL1+uoLpKlSoyc+ZMKV26dNbvANNz5syRwYMHS61atWTatGlSvHhxo4ubUG0kW2g6EapD4wpjQwjVxtKFoiOhOhRuMDaCUG0sXSg6EqpzCVQDmNu0aSPHjx+Xp59+WuVQRzbA8ciRI+XJJ5+UJk2ayIgRI6RAgQIqGq2BHBHqe++9N0ffffv2SYcOHWTr1q0ya9Ysuf76640ucEK1kWyh6USoDo0rjA0hVBtLF4qOhOpQuMHYCEK1sXSh6EioziVQvXnzZpWyAUieMWOGlCxZMuqZI+e6T58+cuutt8qoUaOkYMGCKnKN6PbBgwdVRLp69eo5+gLWBwwYIC+++KLKq+7Ro4fRBU6oNpItNJ0I1aFxhbEhhGpj6ULRkVAdCjcYG0GoNpYuFB0J1bkEqpO52mJFqvVFUrFiRRXlLlu2bNThsMBx\/Pjx2YA8mXmdxxCq3SoWruMJ1eHyh4k1hGoT1cLTh1AdHl+YWEKoNlEtPH0I1YTqLAVQXq99+\/YqhcOZ5rFy5Urp2rWrVKtWTWbPni1nn3123Ch3jRo1VN61SV41oTo8Xw4mlhCqTVQLVx9Cdbj84dYaQrVbxcJ1PKE6XP5waw2hmlCtFMAHecqUKeqnQoUKCp5R3QNNp4QkgmVUF2nXrl1C+I53kRKq3X6Ew3U8oTpc\/jCxhlBtolp4+hCqw+MLE0sI1SaqhacPoZpQrUrhLVq0SPr166fUGD58uDRr1kzy5MmTDaqdedbRZNMXU6KItu67Z88ewY+zbdu2TdXRRqWRa665JjyfFFqSlAK4IRw4cEDOOeccq3rlSU3Gg3xRAP4744wzpEiRIr6Mz0H9VQALy\/FTqlQpfyfi6L4pgIX\/Z511lvocsqWXAnjj\/\/DDD6vCDZEFG8qUKSP4ya0tzynQZoY3nOKKFStUusfhw4dVmgd+nBu4RFu86AVUT5o0SfDDRgWoABWgAlSAClCBTFYAwUL85NaW8VD9xx9\/yOLFixVQY4dFbDWOqh2RESq\/0j+iRao3bNigQBvRakZa0u+jt2nTJlVdhv5LP9\/BYvovPf3mtJo+TG8f0n+Z4T9spufczwNnxUh1mkSqsZshQCZawyuIXr165fjVsWPHVCUPVOvAK3vkQnfv3l2V0ItsqVyoqFNI5s2bpzaSYUsvBei\/9PJXpLX0X3r7D9bTh+ntQ\/qP\/ktvBWJbnzaRardQjXy7cePGybPPPqvSPJD\/g41hnCkfTlk2btwod911l5QrV873knr8QknvjxP9R\/+ltwLpbz0\/g+ntQ\/qP\/ktvBTIAqt044NChQ2oL8pdeekny588vw4YNk6ZNm0revHljDoNqAIDuH374wffNX\/iF4sab4TuW\/gufT9xYRP+5USucx9KH4fRLslbRf8kqFc7j6L9cBNWIUA8cOFCWLl0qRYsWFeT81K9fP6vKRywpUrlNOS\/IcH5RJGsV\/ZesUuE8jv4Lp1\/cWEUfulErfMfSf+HziRuL6L9cAtVID58+fboCaQD15MmTpU6dOgmBWsuzfPlytWq1SpUqamMXZwI+xsb25YMHD5aaNWsKdlaMtQ16oosTixexeBLR89xceiaRTmH9Pf0XVs8kZxf9l5xOYT6KPgyzdxLbRv8l1ijMR9B\/uQSqsUti69atZf\/+\/arKR6dOnSRfvnwxzx41qosVK5aVFoJ+gGo8haEO9YABA9Sfv\/76q8yfP1\/Gjh2rxpo4caI0bNgwzNc8baMCVIAKUAEqQAWoABVIoQJps1AxGU0QPUalj2RbtA1cduzYIV26dJGdO3fmGAaLHDt37qx+Yi14THZuHkcFqAAVoAJUgApQASqQOQpkDFQfPXpUevfuLUjhSLbF2hXx4MGDavdF1K7etWuXWuxYr149Ff1G6ke8BY\/Jzs3jqAAVoAJUgApQASpABTJHgYyB6sxxCc+EClABKkAFqAAVoAJUIN0UIFSnm8doLxWgAlSAClABKkAFqEDoFCBUh84lNIgKUAEqQAWoABWgAlQg3RQgVKebx2gvFaACVIAKUAEqQAWoQOgUIFSHziU0iApQASpABagAFaACVCDdFCBUp5vHaC8VoAJUgApQASpABahA6BQgVIfOJTSIClABKkAFqAAVoAJUIN0UIFSnm8doLxWgAlSAClABKkAFqEDoFCBUh84lNIgKUAEqQAWoABWgAlQg3RQgVKfIY9jxceXKlfLss8\/KZ599prY5r1GjhrRt21Zq164tBQoUSJElmTvNqVOnZPfu3TJ37lxZs2aN7N27V+lctWpVady4sTRt2lQKFy4cU4BDhw7JggULjHbSxPb2Tz31lKxevVoOHz4sJUqUkDvuuENatmwpZcqUiTknbP7ggw\/k6aefljfeeENOnDghFSpUkObNm0uzZs3k7LPPzlyHJXlmR44ckYEDB8rSpUtl5MiRSptYjT5MUlQfD\/v999\/l3XffldmzZ8uGDRvUNV26dGn1GWzVqpWULFmS\/vNRfy+G3rNnj8ybN0\/tUIzvUewqfM011yj\/Jbpf8TPohQfcjYHP3IgRI2Tz5s3qcxfvvmFzz7HhGJu+NteUOyXtjyZU22uYcIQff\/xRevbsKevWrYt67J133in9+\/eXIkWKJByLB0RXAF8q2FZ+yJAh6iYerVWqVEmmTp0qlStXzvFrQHGXLl1k586dOX4HMO\/cubP6wd+dDV9QgL0+ffpEnRdwPWHCBLnuuutyjAubp02bpn7w98gWz97cch1A3xdffFHpixYPqunD4K+Kn3\/+WYYNG6Y+E9HaBRdcIOPGjVOAFtnov+D9Bwtwn+revbv89NNPru9X9GHqfYjvyEWLFkm\/fv3ksssuiwvVNvccG46x6Wt6TaXeE3\/OSKj2Wfljx44p0Js\/f75Uq1ZNBgwYoP7E\/7\/++uvqd\/jyevjhh6Vjx46SJ08eny3KzOHXrl2r9MOXRosWLaRTp06CG\/jJkyfl\/ffflzFjxsjHH38sV199tUyePFnOPffcLCEAAt26dVM3k7p166oHnPLlywsipAsXLpTRo0cLvrgmTpwoDRs2zCYgIgOtW7cWPIV37dpV\/R0PR99++62MHTtWVqxYIYDjmTNnqjGdbdWqVfLggw9KwYIFpW\/fvnLbbbepNxaff\/65ijqsX79ewfikSZOkePHimem4BGe1fft29Tbnu+++iwvV9GHwlwc+A48++qgsWbJEffYGDRokderUUQ+i+Ozhuw5\/1qxZUz3cOiPW9F\/w\/oMFeNPXvn17FVxwfhf+9ttvsmzZMvW9hDdx+L6MDDLQh6n3Ie5Lr732muIH+AVsES9SbXrPseEYm74211TqvUGoTonmmzZtUqBVqFAhdbEjFcHZNAyWLVtW\/T4SvFJiZJpPAvjFmwCkfEBrAGpkRBlPu\/fff7\/s2rVL3ezxGlM3vOLETaJKlSoKfvGqWjd8ac2ZM0cGDx4stWrVUlFlDbj4sgBEIEoQbV58yfXo0UM9PCHyg0i4fmjClwVuSgBnpDbAHucDFV654ua2detWBdW33nprmnvJvfnwKyLUSJvSLVakmj50r6\/XPd566y2577771OcDnyPc4J3tiy++kDZt2qgHpMhrmv7z2htm4yGFDW8akJo4ZcoUKVWqVLbvQv3WqGLFiiplDfctfo+aaW3bC\/cfpJPizY9+OxsPqm3uOTYcY9PX9HvBVlub\/oxU26iXoC+ipohyApbvvvtuBWCRsOcEwkjY89G0jBoawIyb9fHjx9UXfeSDC04WcAwge\/LJJ6VJkyYq4oKosFN\/RKjvvffeHNrs27dPOnTooAB31qxZcv3116tj8G\/A9B9\/\/BFzXg0aAPYZM2Zk3aT0\/yOKjevj\/PPPzzEvYB7XRIMGDVTUOzelBzkfZpCLC60RtY4G1fRh8B9n5wNm5AOktg7fh0OHDhVEy+655x71Nilv3rz8DAbvPmUBvj\/xJhXgjO+7Xr165bAMQQl85+FNHHKuEWhA42cwdU7Ed+Mnn3yi7mEbN25U+e54CEKAJh5Um95zbDjGpq\/NNZU6b+SciVDto\/p4MkS0EU9qeOpv1KhR1NnwKnT8+PHZYM9HszJuaKRgAIgByQDXWAuhkHONyCeivqNGjVJpF\/omcfDgQRWRrl69eg59nDcbRJcRfUZD3ij+jhQN+PDMM8\/M0Rc3HwD\/N998I88\/\/7x69Y2G6AKi3sinx2vx008\/PUdfnBci2Fh08swzz6gFjLmlbdmyRaV9lCtXTumDtw9IHYgG1fRh8FeFvs5\/+eWXmA+Ysayk\/4L3XyRUt2vXTn1XRqYjaj9\/9dVX2aCaPkydD3GvwnejTqXCdyNSr7AoPh5Um95zbDjGpq\/NNZU6bxCqU6q1ft2JG00sYINBeL2NfNxE+VApNT7DJosVqcbTPb6Mor3OdEqgH3ycQI63EID4eGCMVctI+0C+tgZCfAH27t1brax\/6KGH1O+jtVg3sAxzTY7T0Xl0eKiA7pdffnnWTSQaVNOHwV8R2geImCH1w80aAPoveP9pC\/TbsWjpHzhGv46P\/L6kD1PnQ3w\/Yl3QzTffrII0eNuj9Y\/FEDb3HBuOselrc02lzhuE6pRqrSONxYoVU9Gbiy++OOr8+uJBflpui0imyiHOHGVnmkeyDzQ6yq2hAaX5Er0qxbk5v8x0lDsaaEfTwRmRQJQBaSuZ3vDwM336dLWwFAtP8dABvXRkJhpU04fBXxWRb4H279+v\/IhFVFiIHa9MJP0XvP+0BajS8MADD6jF3c6FisjZRclPfOchSIQ1LMif15Fs+jBYHyaCapt7jg3H2PQ1vabcPND74TWmf\/ih6v\/GTBaW8aobKQJIX4gH3z6amtFDI68L6Tf4wc3duSA0EpZjfSCxoBSvRHUkAKkjOtrsTAmJFFLnkT733HNZeYrJwjJyygCVWOiYqD5zpjhQV1NB2UO9UMqpVzQd6MPgva\/f5OAzgjUNWD+ChbqRLVqZSPoveP85LUAkFJ89pKtFlie98MIL5ZFHHpGbbrpJRUh1ow+D9WEiqLa559hwjE1f02sq6L0dCNU+fhYSXeh6av2KBP8mVHvrEGcNT4w8fPhwtamKjrBEy7OOZkGkL51QnQh4dZqIXvzj\/IJzLvaJnNcZ5U40h7eqBTMaopuowoLNkZBvrmt7JwvVztQc+jC1PtTXOErpAaavuuoqFc285JJL1EJe5H+iqkS0spb8DKbWV\/Fmw\/clNu7BmzH4KrJhURzWCSFK7VxDQh8G68NErGFzz0k0djyOselrek0RqoO9Fn2d3eaC8tWwXDI4bhCoE410D9zokbeOH2cFFtMPLqHa24tIb0qAUmuRfiJUe6u1H6NpqMbY0WrB4\/9jlbXkZ9APj7gf07mRFf6Oev8AaDwoIWKN3TFRvQU1rPHAi4pEut4\/feheby97JGINQrWXascfi5FqH7W2efXho1m5YmhExxYvXqyAGjcElPBCpY7IsnSmr5iY\/uHtZYTPCiL5KD0YuTlPslCdaJEcU3i89ZlzNCdUo2zlDTfckGMygBrSRLDD6I033qgqHuHzyM+gf35xMzK2JgdEb9u2LWoAIvLByLlhGX3oRmnvj3UD1fHW50RLObThGJu+ptcUI9XeX1+hGdEmST80J5GGhqBmLtJocNNGBBR5nqidCxCObKaLIbhQ0bsLQy+OQi1qZ9qHniERVNOH3vnCdKQFCxaosoeIasZLYUMVHJSJvPTSS7Pqs9N\/pqp720\/7IXLdSeQszgohutILfeitL9yOlgiquVDRraLmxzNSba5dwp7J1FnEIMl+ISWckAeoTQjwJI6dppDmgWgKFoFGbrqjpULx\/LvuukvVQ47cIcwpZ7SSesnU\/Yz2ZQboB4CgzjVL6klWOSg3l6+zdBR96EY5f47V32GJoFrf\/J3H0X\/++MTtqDoyiP0UUMcfgYNojT50q6z\/xyeCapt7jg3H2PS1+V7wX\/HYMxCqfVTfCVTc\/MVHof83NPTGDoQvvfSS2mUKC6OaNm2abZV6pBW6FvQPP\/zgevMXDRImm79oSOfmL\/ZQTR\/6\/9lKNIN+K3fGGWfE3fwlMgUHr2rpv0Tqpub3GqqdqTnJQjV9mBofxZolEVSjn+k9x4ZjbPraXFNBeoNQ7aP6Nlt0+mhWRg6NCPXAgQNV9Ldo0aKqznH9+vVz7AgWefI2W6Fym\/LUXEqJ0j\/ow9T4Id4sBw4cUPm4H330kXqwRYpHZENONT6XqF\/trNRC\/wXvP1igH3gSpX9Ey3WlD4P1YTJQzW3KU+MjQrXPOmOL8tatW0uhQoVUDiFquDobvsiwyQVWUbOcnpkznBuGAKix0K1OnToJgVrPpncJwyI55AiWLl06yxCMjRzCwYMHq92r8LSvt0HHKzXU4120aJHyMVI6nGkmqDiCxZGoM41a1igXp3+PWrD4P3wZ4mEAEOLcEti5WU1uKKcXz\/OJoBp96UOzz45XvZyfQUDZE088Iag17mzbt29Xm\/ggh37ixInSsGHDrF\/Tf155wnycffv2qcXCn376adbGS5FpczgG1XlwX8M6FewGq7+36ENz7W17JgPVNvccG46x6Wt6TdnqadOfUG2jXhJ9UWsY4LVkyRK1iGfIkCEK+FCR4tVXX1VABfhyrqROYlge4lBAR4xR5xhVPjp16iT58uWLqRFuAtjlUm9eoOsj44sJubrYNQx\/\/vrrrzJ\/\/nxVOgotEgTwf7pqBXyIElS44Zx33nnqlfZjjz0mq1atEmx4AVgvX758NptwTWADGTT0A5gjj\/Hzzz9XEI8vI6SWoMxc0LtEBXnBJQPV9GGQHvpzbudufPj84DsN5fX05wTffSjHhpxdPCg6K\/HQf8H7DxY4v5PuuOMOBdnY6ffkyZPZao1HC0DQh8H5MBmojvSvm3uODcfY9LW5poLyBqE6BcrjZoONELDyPVpDXi1Kv0WWe0uBaRkxhc4VS\/ZknIvcdB\/U0EXUBTf9yIZoDaLK+ImM3Dhru0buPoZxSpQooUqI6Y1MnGPr2syoeIG\/R7Zou88le46ZdFwyUI3zpQ+D9zoeJnv16qUeNqM1fA4A1M63QfwMBu83bQG+h5DegQegaN9nOC7e9xI\/g8H4Mlmotrnn2HCMTV\/TayoYT4gQqlOkPKKeiFoilQA7xgHOUFcXr0Nr166ttihnc6+Ac9fBZHtHg2r0BbwhlQM3FaxaxmLHevXqqeg3Uj+c2\/I65wJY44OP9J3Vq1erNw+AaUR6WrZsKWXKlIlpGuppY5UzqpW88cYb6kaG1+fNmzdXOz8GXXMzWU39PC5ZqKYP\/fRC8mPr7zq85dmyZYvqiLQ3vMlBykesqhL0X\/Ia+3kkvs92794tc+fOlTVr1ghS0XC\/og\/9VN1u7GShGrPY3HNsOMamr+m92U5Vs96EajPd2IsKUAEqQAWoABWgAlSACmQpQKjmxUAFqAAVoAJUgApQASpABSwVIFRbCsjuVIAKUAEqQAWoABWgAlSAUM1rgApQASpABagAFaACVIAKWCpAqLYUkN2pABWgAlSAClABKkAFqAChmtcAFaACVIAKUAEqQAWoABWwVIBQbSkgu1MBKkAFqAAVoAJUgApQAUI1rwEqQAWoABWgAlSAClABKmCpAKHaUkB2pwJUgApQASpABagAFaAChGpeA1SAClABKkAFqAAVoAJUwFIBQrWlgOxOBagAFaACVIAKUAEqQAUI1bwGqAAVoAJUgApQASpABaiApQKEaksB2Z0KUAEqQAWoABWgAlSAChCqeQ1QASpABdJAgZ9\/\/lkef\/xxwZ+9evWSkiVLpoHVNJEKUAEqkHsUIFTnHl\/zTKkAFbBQYN++fbJw4UJZu3atfPzxx65GmjJlijRq1MhVn8iDly5dKj169FD\/3bt3b2nfvr3VeF51PnTokNJj\/vz58te\/\/lU6duzo1dAchwpQASqQVgoQqtPKXTSWClCBoBU4cOCAAtqPPvpIChcuLLNnz5arr746h1mIKAPCx44dKxMmTLCG6l27dimYPn78uAwaNEiqVasWtBTqAeP111+XxYsXy4kTJ+Shhx6SLl26BG4XDaACVIAKBKEAoToI1TknFaACaavA0aNHFdwuX75cQfWcOXOkevXqUc\/n2LFj8uijj8r1119vDdVhFezXX39VeqxcuZJQHVYn0S4qQAVSogChOiUycxIqQAUyRQE3UI1zXrZsmRQqVEhuvPHGTJEg23k49WCkOiNdzJOiAlQgSQUI1UkKxcOoABWgAlDALVRnumqE6kz3MM+PClCBZBUgVCerFI+jAlSACriA6iNHjsibb76pItQFCxb0RDsA7HvvvSf58+eX2rVrezKm7SCEalsF2Z8KUIFMUYBQnSme5HlQASqQEgWSjVR\/8cUXMmnSJBkxYoSceeaZWbZhoeGHH34oTz31lDRp0kQtOJw1a5asWLFCfvrpJ7nsssukU6dOctNNN0nevHnl1KlTsmPHDnn66adl9erVcvjw4ai5y19++aWa7+yzz1b9UJGjSpUqcvnll0vz5s2zaYNc73fffVeNd95556kyfRs3bpSrrrpKVe8oV65cVC3379+vcsi3b98uF110kWBOjP\/pp5+qBYvR0j8w15o1a2TevHmyadMmyZMnj8ox7969u7KPjQpQASqQKQoQqjPFkzwPKkAFUqJAMlAN8EUZvW3btsnUqVMVVKM6BsrO4QdQigaoBvyiFN1ZZ50lr776qnz\/\/ffqdw8\/\/HBWeTqMh5J+ANGtW7fmgFfALSqSAIibNWumwBWl7vr16ydlypRRda11Axjj\/9GGDBkipUqVUn\/\/9ttv1biwp3\/\/\/nL33XercXTbvHmz9OzZU+644w5p06aNFChQQH7\/\/XcFy8OGDVN\/j4RqPVfFihWldevWcs4558hrr72mFjZCx4kTJ0rDhg1T4jdOQgWoABXwWwFCtd8Kc3wqQAUySgEnVCc6seuuuy4Lqp1Q265dOxXdRWqIE2wBwiiX99JLL+WoLBIvzQLgPm3aNAW4zkokiAy\/\/PLLCpJPP\/10BbKoRgKwxfGwz9kAzoBfHDd58mS5+eab1a8B9F27dpXixYvLuHHjpGjRolndEOXu3LmzrF+\/PhtU67nw+zFjxqgIOhoi7\/j39OnTpWzZsvLMM89IhQoVEknJ31MBKkAFQq8AoTr0LqKBVIAKhEkB00i1PoeDBw9K27ZtVUQ42qYwu3fvVr9HXeoOHTpkRZnjQTVAF5B85513ysCBA7NyuBHhBlTj\/wHVb731ltx3331StWpVNXfp0qWzSeuco0aNGjJz5kwF0kj5AOyPHDkyRypJLLsA9AB0RNxbtWqVbR7Ut8aDBVq0McPkb9pCBagAFUhWAUJ1skrxOCpABaiAi4WKsXKqE0E10iiGDh0qzz33nNSvX19FhosUKZKt6khkmoWOMAOiL7zwQhk9erTUrFkzW\/oGnKfhO1oEXTsXud5I50A0GjCNKDI2dMEciCoDtp0tFlQjej5+\/HjV35lTHnkRIXecuzDyo0UFqEAmKECozgQv8hyoABVImQLJRKphDKp\/YCHgP\/7xDwXFyUaqcdyCBQukb9++Kj1D52THi1QjpQILHZHmAbBGq1u3rhoDCwrRnP3jQfW6deuyIsuIZiPfGznUP\/zwQ9SNbqLZhcWJmBtbq3uxRXvKnMuJqAAVoAIWChCqLcRjVypABXKfAslCdSxlEkWq0e+FF16QPn36JB2p1nN98803KsqMShxoKL2H9AosiETVEQ26qLqB7dX1IkWnrciNbtmypZx22mkqrxpQDqj+7rvvVM52rVq1\/hiIKQAABDpJREFUsp1aNKh2\/h8WJWIRJRsVoAJUINMVIFRnuod5flSACniqgAlUo7IGcoxR6QIR7Hg51TBWp2BgAWC3bt0U4MaLVGOrcNTCRim9P\/74Q+Vrjxo1SpXJQ9\/nn39epYPolIySJUuqVI5oJe00VOtjUJUE9qKSSbI51c4UlltvvVXZEqtW9969e9UiRq9qeXvqbA5GBagAFXChAKHahVg8lApQASrgFqoBmMgtRo4ycodRDSMeVCN1AhU6UF5vxowZWZHhRAsVb7jhhmyVPwDviHavXLlSlbBDtFgvHgSEY+Fh5AJCeFennqAyCXKwUVYPpfRQa7pevXoyYcKEbOkssezS0XZAPc4D6SiRDfW38QCBEn\/OiiK8yqgAFaAC6agAoTodvUabqQAVCEwBN1CNXOe3335bVfAYO3asypF2pn9gsxZEcp1ty5YtCroR1e7Ro0cWwMaDaixMBLzieGcD2ALQdT1oADtK+KFW9hVXXKGqe6B2tG4AcQA0dm10ltxbtWqVPPjgg6oWNcZDVQ9dw1qX2wOwO6uVoIoJQH7nzp0KmJFKUqdOHdUP0XQcjwj2I488Itdcc01g\/uTEVIAKUAGvFCBUe6Ukx6ECVCBXKACIBDyizjQaYPm2225TUKsb8peR1gCoxU6I2CVRl7BzQjUqYzz22GNy5ZVXKthEGT1AJsrYOetXY1zUsEYVDiwkBNQiupsvXz41JaAa+c5PPvmkSvNAAyCjvB4WLg4fPlyQzoGGSDnyrrGIEJFqRLOxkQtAd+HChSoSDbBu3Lhx1jlhLPR58cUX1RiNGjVSP7\/99pv6v6+++kp+\/PFHlcMNQEZNa9gBW5HCohdPlihRQm1Gc+DAAXU+WFiJfG+ndrniIuJJUgEqkJEKEKoz0q08KSpABbxWADAN6ESNZeQsu2ktWrSQAQMGKHh1QjVK4+3Zs0dFhlF2DrnF\/\/znP9UW5YULF1ZTILqMqO6SJUtk2bJlKloMeMXiQWw\/DjBHCgV2L8QPQBebqnz00Udq23GAswZqbTPGQO40yvYh3\/vaa6+V\/\/u\/\/1Mwj5rW0bYphx1IScGW6p999pmKPjdo0EAdD9sKFSokt9xyi9q23AnJWDz5+OOPZ22xjtrYul\/lypVzlP1zoyuPpQJUgAqESQFCdZi8QVuoABXIeAWSqf6R8SLwBKkAFaACGagAoToDncpTogJUILwKEKrD6xtaRgWoABWwUYBQbaMe+1IBKkAFXCpAqHYpGA+nAlSACqSJAoTqNHEUzaQCVCAzFCBUZ4YfeRZUgApQgUgFCNW8JqgAFaACKVIAJfY2bNigNnTBosLbb79dhg4dyo1PUqQ\/p6ECVIAK+KkAodpPdTk2FaACVOB\/CqDMHqp3oNpGZCtWrJiqolGkSBHqRQWoABWgAmmqAKE6TR1Hs6kAFaACVIAKUAEqQAXCo8D\/A77+tauXq\/DBAAAAAElFTkSuQmCC","height":435,"width":725}}
%---
%[output:39ad2195]
%   data: {"dataType":"text","outputData":{"text":"\n=== Training summary ===\n","truncated":false}}
%---
%[output:4c12ad7e]
%   data: {"dataType":"text","outputData":{"text":"Episodes completed   : 10000\n","truncated":false}}
%---
%[output:84a9ec2f]
%   data: {"dataType":"text","outputData":{"text":"Total agent steps    : 536741\n","truncated":false}}
%---
%[output:506d6127]
%   data: {"dataType":"text","outputData":{"text":"Final episode reward : 41.306\n","truncated":false}}
%---
%[output:3f385c4c]
%   data: {"dataType":"text","outputData":{"text":"Final avg reward     : 40.646\n","truncated":false}}
%---
%[output:6f6a62bc]
%   data: {"dataType":"text","outputData":{"text":"Best episode reward  : 51.929 (episode 3996)\n","truncated":false}}
%---
%[output:3f138209]
%   data: {"dataType":"text","outputData":{"text":"Mean steps\/episode   : 53.7\n","truncated":false}}
%---
%[output:37083b11]
%   data: {"dataType":"text","outputData":{"text":"========================\n\n","truncated":false}}
%---
%[output:0d37e64c]
%   data: {"dataType":"text","outputData":{"text":"Zero-degree coefficient regression error: 8.548e-18\n","truncated":false}}
%---
%[output:03453dd8]
%   data: {"dataType":"text","outputData":{"text":"\nRunning brute-force evaluation:\n","truncated":false}}
%---
%[output:7c62b38b]
%   data: {"dataType":"text","outputData":{"text":"  Episodes: 10\n","truncated":false}}
%---
%[output:9c646e31]
%   data: {"dataType":"text","outputData":{"text":"  Angles  : 2\n","truncated":false}}
%---
%[output:4a043ad8]
%   data: {"dataType":"text","outputData":{"text":"  SNR     : 36.50 dB\n","truncated":false}}
%---
%[output:842d130e]
%   data: {"dataType":"text","outputData":{"text":"  AWGN    : 0\n\n","truncated":false}}
%---
%[output:1142eb43]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQmUFcW9xv8galBBVoOACNEBdx7iwiMSXKLGF9E8waBgVHYXEBeWAQUVUYZFI4ILCKKoCAomETUucSH6yFPccDmCIKIs4RkQFzi4RHnnK9M3d+70na7bXbe7\/3e+PoejMNVV\/\/6+qurfVFVX1dq5c+dO4UUFqAAVoAJUgApQAUUK1CLAKHKLoVIBKkAFqAAVoAJGAQIMKwIVoAJUgApQASqgTgECjDrLGDAVoAJUgApQASpAgGEdoAJUgApQASpABdQpQIBRZxkDrgkK7NixQ8rLy2Xx4sXVPu6gQYNk5MiRsUoyceJEmTFjhnTr1k0qKiqkbt26ecvPfo5ixfq\/\/\/u\/0qtXL2nevLnMmTNHysrKqtUjn7bz5s2TTp06Ze4N8qB9+\/Yye\/ZsadSoUeaeVatWSZ8+fWTjxo15YyiWDrFWAhZGBVKgAAEmBSYwBCqQq0DQy9NLn8TLUDPAeLCTr8Zl62nrQTb4EGDYlqlAfAoQYOLTmiVRAWsFbF+eSQCM9UPElNB2BMYGLhDyhAkTpGfPnmLrQfbIj00Z9CymisFiSl4BAkzJW8wH1KhAoVMvCxYskFGjRgmmNaZMmSLDhg2T5cuXm0f3XsheGvxb7nRL9qjKmWeeKf3798\/I5t3v\/UPuCAxi7devnylvzJgx8tZbb5mpL5Rx5513yqxZs8zfc1\/c2fEgb78pmc8++yyTt1d+bjpbgPHiRj7VxeJNjSGdN43nBx3Z+XkaZQNM7pSUxnrImKlAmhUgwKTZHcZWYxUICzD5BOvSpYu89NJLlX6cDQLZL2O\/PLJf4NUBTPa9AIGxY8fKuHHjqgBMvvKywcoPXvwgxgZgsvPKNwKCmLB2xVvXE+RB9s896Fm\/fn1mDQwBpsY2Xz54TAoQYGISmsVQgUIUsJm+yH7ZZ49meC\/o3OkM74Xqpc2+P9\/ohPfvfmm9l3b2CEzuyI4fBGSvQ\/EbufDi9+LMXiycfa\/3PDYAE2ZkJAhg4Gf2yBcW9G7ZsqXaRbw2C58LqSdMSwVqsgIEmJrsPp89tQqEBZhsgPAbIcAXQ34vcw9UcqdnskcuPNiobgQmd3TDDwL8FgF76Tp37mzWn+RefnoQYFJbfRkYFYhFAQJMLDKzECpQmAJhASYbQPKNIFQHMLkjBDYAkj0Ck7texub+fJ9hB2lAgCmsTjE1FSg1BQgwpeYon6ckFLCZvsh+0NypDOxNoh1gsqe1PLDCQmHs+YKrEIDhGpiSaBZ8CCpQSQECDCsEFUihAkkBTNQppEJGYHLLyl5EC0u8L4Cy8wy7Bgb52X6F5MWFkaHqvkIKWnfERbwpbFgMqaQUIMCUlJ18mFJRIGj6xHtO72X77LPPZj6j9naHDTMCg3z9vjiyXcRrAzB+i3izR0hyv17y+6y50BEYpLfZowXpXO0DQ4ApldbI50irAgSYtDrDuGq0AkkCjJ\/wtp9R2wBM7mhIbnm5X0vlqwiFTCF5eQTtxJu9BsjWg+xnDvO1U42u6Hx4KhBBAQJMBPF4KxUolgK2L0\/XIzB4gfft21cuvfTSzHk+hWxkZwsw0C13Izu\/s4yyp33wrNOnT5dJkyZV2lfG5jPqbJ\/yaev6LCSOwBSrdTBfKvCjAiUNMNV9eskKQAWowL8VKOR8I+pGBagAFUiDAiULMB68QGRvZ03vNzX+ZpSGqscY0qQAASZNbjAWKkAFbBQoWYDBXPSIESPMcHNZWVlGi9ztwoNEwtbgixYtku7du0vLli2DkvPnCStAv8IZkBTA0K9wfiV5Fz1LUv3Cyy5lv0oWYPLZjHn3pUuXZkZlgqoDR22CFErXz+lXOD+SAhj6Fc6vJO+iZ0mqX3jZpexXjQIYb1oJiwVHjhxpVRNK2XwrAZQlol+6DKNfuvxCtPRMl2el7FeNApgwRoa5R1f1Lq1o6ZcuP+mXLr8IMPQrTQrUGIDx9mfAZ6LVjb5gvhB\/vGvDhg0yfPhws7lXx44d0+QdY\/FRYNOmTTJ27Fj6paR20C8lRmWFSc90eeb5NXToUDn22GMzwWNNp\/Z1nTUCYGzhBc5OnTrV\/OFFBagAFaACVKBUFQDQ4I\/mq+QBphB4gZG5IzBLliyRGTNmyOTJk6VFixaavQ4d+8qVK+Wxxx4zG5w1btw4dD5x3PjPf\/5TNm\/eLE2aNJE6derEUSTLiKjA559\/bnJo0KBBxJx4exwKsI3FobK7Ml555RXzSzlGpg866CCOwLiTtrg5FQovftH85S9\/kYEDB2ZOvi1uxOnMHQADgMNUWrt27dIZ5L+iQue6bt062W+\/\/QgwqXbq38H94x\/\/MH9p2rSpkohrdphsY7r899aZzZw5U375y1\/qCj4g2pIdgfEOh+vUqZP1F0cEGP11m52rPg8JMLo8YxvT5RcBRpdfJtrcc1ayH8E7P6ZRo0aBT8YRmECJUpWAnWuq7LAKhgBjJVNqErGNpcYKq0AIMFYylWYiAowIp5BKs26n5akIMGlxwi4OAoydTmlJRYBJixMJxEGACRbdayC5KQsZ6QouxS4FO1c7ndKUigCTJjeCY2EbC9YoTSkIMGlyI+ZYCDDVC+6tNcLeOlhvlPTFzjVpBwovnwBTuGZJ3sE2lqT6hZdNgClcs5K5gwATDDBXXXWVjB49utKhmUlVAHauSSkfvlwCTHjtkriTbSwJ1cOXSYAJr536OwkwwWtgsGB67dq1kb72clVR2Lm6UjK+fAgw8WntoiS2MRcqxpcHASY+rVNXEgHGDmBGjRpVxTuugUlddU5lQASYVNqSNygCjC6\/CDC6\/HIaLQEmeAqpX79+ZvSFa2CcVr0akxkBRpfVBBhdfhFgdPnlNFoCTDDAcA2M0ypX4zIjwOiynACjyy8CjC6\/nEZLgAmeQkIDeeihh6SiokLq1q3rVP9CM2PnWqhiyacnwCTvQSERsI0VolbyaQkwyXuQWAQEmOARGEwhLV++vEpCroFJrNqqKpgAo8ouIcDo8osAo8svp9ESYJzKWfTM2LkWXWLnBRBgnEta1AzZxooqr\/PMCTDOJdWTIQEm2CtsZpe9DgafVS9dujSRKSV2rsF+pS0FASZtjlQfD9uYLr8IMLr8chotAab6NTA7duyQ8vJy6dy5s\/Ts2TOjfVJ7w7BzdVr9Y8mMABOLzM4KYRtzJmUsGRFgYpE5nYUQYKoHmNzRF8\/FVatWyU033SQ333yz2Jz67cp9dq6ulIwvHwJMfFq7KIltzIWK8eVBgIlP69SVRICp3hJvBObcc8+ttA8MGs3EiRNl9uzZBJjU1ep0BUSASZcfQdEQYIIUStfPCTDp8iPWaAgwwXJjugg78c6bN89AjNdgJkyYUGlaKTin6CnYuUbXMO4cCDBxKx6tPLaxaPrFfTcBJm7FU1QeASZ4HxjYhSmjPn36yMaNG417HszEbSU717gVj14eASa6hnHmwDYWp9rRyyLARNdQbQ4EGF3WsXPV5ReiJcDo8oxtTJdfBBhdfjmNlgDjVM6iZ8bOtegSOy+AAONc0qJmyDZWVHmdZ06AcS6pngwJMHq8QqTsXHX5xREYfX6xjenyjACjyy+n0RJg7NbAOBU9QmbsXCOIl9CtHIFJSPiQxbKNhRQuodsIMAkJn4ZiCTAEmDTUw1KOgQCjy10CjC6\/CDC6\/HIaLQHGqZxFz4yda9Eldl4AAca5pEXNkG2sqPI6z5wA41xSPRkSYKr3CjvxZp9GncQJ1NkRsnPV07a8SAkwujxjG9PlFwFGl19OoyXA+E8heY2iOrGT2AuGnavT6h9LZgSYWGR2VgjbmDMpY8mIABOLzOkshABT1Rdv1GXkyJGVjg\/ITsmjBNJZn9MYFQEmja7kj4kAo8svAowuv5xGS4DxB5irrrpKRo8eLWVlZb568zBHp9WwpDMjwOiylwCjyy8CjC6\/nEZLgPGX0zv\/yO+8o+p+5tQcn8zYuRZbYff5E2Dca1rMHNnGiqmu+7wJMO41VZMjASb\/Z9S5C3g9U5NcyMvOVU3TygRKgNHlGduYLr8IMLr8chotAYb7wDitUMysigIEGF2VggCjyy8CjC6\/nEZLgPGXM3v0pVu3btK3b1+59NJLM6dRDxo0SLDIN+6LnWvcikcvjwATXcM4c2Abi1Pt6GURYKJrqDYHAoy\/dRMnTjQ\/AKR4a14AMhUVFbJjxw6zN0ynTp1ihxh2rvqaGgFGl2dsY7r8IsDo8stptASYqlNIGH3J\/gop9+8wgF8hOa2GJZ0ZAUaXvQQYXX4RYHT55TRaAkxVOQkwTqtYjc+MAKOrChBgdPlFgNHll9NoCTDBU0h+KbKnmJwaEpAZO9c41XZTFgHGjY5x5cI2FpfSbsohwLjRUWUuBBh\/27DOpby8XDp37iw9e\/aslAjwsnHjRrMepm7durH6zs41VrmdFEaAcSJjbJmwjcUmtZOCCDBOZNSZCQGGn1HrrLl6oibA6PEKkRJgdPlFgNHll9NoCTAEGKcViplVUYAAo6tSEGB0+UWA0eWX02gJME7lLHpm7FyLLrHzAggwziUtaoZsY0WV13nmBBjnkurJkACjxysOb+vyyouWAKPLNwKMLr8IMCnzC4Y89NBDgYtE853V43cAYb5HJMBwCill1b\/kwiHA6LKUAKPLLwJMivzCBml9+vSRjh07BgIM0o4YMUImTZokZWVloZ6CABNKtsRuYueamPShCybAhJYukRvZxhKRPXShBJjQ0rm90duyHrl629ZX95kujMMnvbNnz5ZGjRqFCoYAE0q2xG5i55qY9KELJsCEli6RG9nGEpE9dKEEmNDSubvRg5d58+bJkiVLrPYZwT1r166NdB4PAcadh3HkxM41DpXdlkGAcatnsXNjGyu2wm7zJ8C41TNybjYbpXkbrS1evLhSeYWsf8GNBBiugYlcYZlBtQoQYHRVEAKMLr8IMCnzywZgvAW8rVq1yqyV8dbPDBkypMrusfkekQBDgElZ9S+5cAgwuiwlwOjyiwCTMr9sACZfyJhWmj9\/ft51MevXrxf88a4VK1bIuHHj5J577pFOnTqlTAmGk6sAOlf417JlS6lTpw4FUqDA5s2bTZRNmjRREC1DZBvTVQdee+01Of\/882XQoEHStWvXTPDoI\/FH81Vr586dO7U9QBSAAY0OGzZM5syZ4\/tl0tSpUwV\/cq9bbrlF2rdvr02qGhcvOtdNmzZJs2bNCDBK3MdoKa6wC+2VPGbJhMk2psvK5cuXy5VXXlkl6KFDhwr+aL4IMDnu5Y7AYMHwjBkzZO7cuTV2BGblypUCgEMjaNeuXarrO4e3U22Pb3AYgcHvUU2bNtUXfA2MmG1Ml+n4pR0jMGPHjpWDDjqIIzBJ22czAuOtd5kyZUol8MAU0tKlSwP3kPGekWtgkna7sPLZuRamVxpScw1MGlywj4FtzF6rNKTkGpg0uJAVgw3AIDnSwTxvHxjPSHyKbbuehQCTMvMDwmHnqssvREuA0eUZ25guvwgwKfMrH8Dg33GNHDkyEzH+DVNA3lUIvOAeAkzKzCfA6DLEIloCjIVIKUpCgEmRGRahEGAsRCrVJAQYfkZdqnU7Lc9FgEmLE3ZxEGDsdEpLKgJMWpxIIA4CDAEmgWpXo4okwOiymwCjyy8CjC6\/nEZLgHEqZ9EzY+dadImdF0CAcS5pUTNkGyuqvM4zJ8A4l1RPhgQYPV4hUnauuvxCtAQYXZ6xjenyiwCjyy+n0RJgOIXktEIxsyoKEGB0VQoCjC6\/CDC6\/HIaLQHGqZxFz4yda9Eldl4AAca5pEXNkG2sqPI6z5wA41xSPRkSYPR4xSkkXV550RJgdPlGgNHlFwFGl19OoyXAOJWz6Jmxcy26xM4LIMA4l7SoGbKNFVVe55kTYJxLqidDAgzXwOiprTojJcDo8o0Ao8svAowuv5xGS4AhwDitUMysigIEGF2VggCjyy8CjC6\/nEZLgHEqZ9EzY+dadImdF0CAcS5pUTNkGyuqvM4zJ8A4l1RPhgQYPV4hUnauuvxCtAQYXZ6xjenyiwCjyy+n0RJgOIXktEIxM04hKa8DBBhdBhJgdPnlNFoCjFM5i54ZO9eiS+y8AI7AOJe0qBmyjRVVXueZE2CcS6onQwKMHq84haTLKy9aAowu3wgwuvwiwOjyy2m0BBinchY9M3auRZfYeQEEGOeSFjVDtrGiyus8cwKMc0n1ZEiA4RoYPbVVZ6QEGF2+EWB0+UWA0eWX02gJMAQYpxWKmVVRgACjq1IQYHT5RYDJ8Wvnzp3y1VdfmU9Wa9WqJXvvvbfUrl1bl6uW0RJgLIVKSTJ2rikxooAwCDAFiJWCpGxjKTChgBAIMDli7dixQ8rLy2Xbtm0yePBgad++PQGmgArFpMVTgJ1r8bQtVs4EmGIpW5x82caKo2uxciXA+ADMmDFjpHfv3tKhQwfZvHmzzJ8\/X+6++25p0aKF9O3bV0455RSpX79+sTyJLV+OwHAKKbbKVkMLIsDoMp4Ao8svAowPwNx4441ywQUXSFlZmfkpKvUNN9wgXbt2lRNPPFGXw9VES4DRZSU7V11+IVoCjC7P2MZ0+UWAsQAYJJk4caIBmE6dOulymABTMn6xc9VnJQFGl2dsY7r8IsBEBJg1a9aYHH72s5\/pcl5EOAKjyzJ2rrr84giMPr\/YxnR5RoDxAZjx48ebKaS2bdtmfuo3AoPKft9998mhhx6qcmSGAMM1MLq6K33RcgRGl2cEGF1+EWB8AAZfIS1evNjayXnz5hFgrNVKV8KVK1fK5MmTZfjw4dKuXbt0BZcTDTvXVNvjGxwBRpdnbGO6\/CLA5AGYN954Q5o0aVKtm19\/\/bV8+OGHMnfuXAKMrnqvMlp2rvpsI8Do8oxtTJdfBBgfgJk2bZr5XDoIYHArRmqaNm1KgNFV71VGy85Vn20EGF2esY3p8osAE3GYHvvE4LKBnbRVDa6B4RqYtNXJUouHAKPLUQKMLr8IMAF+bdiwQZ577jn54IMPzHTRUUcdJT\/\/+c\/Nf+vUqaPL7ZxoCTC67GPnqssvREuA0eUZ25guvwgwefzyvjCaNGmSfPfdd1VSHX300TJhwgSVn097D0OA0dVY2bnq8osAo88vtjFdnhFgfPzCgY4LFy6Uq6++2hwfcN5550mXLl2kYcOGsnbtWnn++ecFXx4dfPDBctttt8k+++yjy\/V\/RUuA0WUbO1ddfhFg9PnFNqbLMwKMj1+AlIEDB8phhx0m1113ne+5R9jA7qqrrjLnIl100UXm5GptFwGGa2C01Vlt8XIKSZdjBBhdfhFgfPx64okn5P777w8cXcFIzL333ivTp09XebgjAYYAo6u70hctAUaXZwQYXX4RYHz8wq67rVu3lp49e1br5pdffimjR4+WoUOHZg5+1GQ\/AUaTWz8eKrpu3TrZb7\/91C8g16V8+GgJMOG1S+JOtrEkVA9fJgEmR7tvvvlGrr32WgMvHTp0qFZZ75Tq0047jfvAhK+DvNNSAXaulkKlKBkBJkVmWITCNmYhUoqSEGByzNixY4fceOON5iyksrKyQKs0n1LNERhOIQVWcCaIpAABJpJ8sd9MgIld8kgFEmB8AGbcuHFy5plnVjrM0U\/l77\/\/Xm699Vbp1q0bR2AiVUPebKMAO1cbldKVhgCTLj+ComEbC1LI7c8xYPDFF19Is2bN8mb80UcfyW677Wa+CM69CDA+AMPDHN1WUubmRgF2rm50jDMXAkycakcvi20suobYhuTJJ5+UFStWmK1HPv30U7nkkksyH7rg4xd8uQutcQzP7NmzzRe\/aCuY0UB6fN179tlny6uvvioPPPCAVFRUmLwIMAH+gAgBMO+++27gl0U\/\/PCD2RdmxowZHIGJXu+ZQ4AC7Fz1VRECjC7P2Mai+\/Xmm28aEMGZgoAOzGjsueeeZtsRHIA8efJks01J9ogK1p4i3QEHHCC\/\/e1vzTsVa1Dffvttad++vRx\/\/PG+W5VwBMZnBAbiYQ2MH\/Hl2svDHKNX+CRzWLlypWlQw4cPl3bt2iUZSmDZ7FwDJUpdAgJM6iypNiC2seh+YVsR9KsYNQG4LFiwwIyi3HPPPfLJJ5\/IpZdeKr\/61a8EH79gR\/vatWsLpokAONjdft999zUbyQJaHn30UZO+bt26voERYHJkKbQCp+Ewx1WrVslNN90kN998szRq1Mi6BnIRLxfxWlcWJgylAAEmlGyJ3VRo\/59YoCkuGKMvH3\/8seAYnr322ktefvllGTlypMyZM0d23XVXAzcAkxdffFEuvvhiGTJkiPz97383IzB4j+F9huN7XnvtNQM5hxxySN6nJcDkSIOhLAxb4b8QFYc5YkFvmzZtKqV8\/\/33ZdmyZdKrV69E9+T47LPPpF+\/fiY2zCUSYFLcsiOGxs41ooAJ3E6ASUD0CEWyjUUQ71+3YiNYgAiApW3btvLMM8\/I9ddfn\/k7kmH5xSOPPGI2gcV7C1\/8Yt3M8uXL5fDDDzfLN3CAMt6v+G+DBg3M3my5O94TYHz8mj9\/vtmgrn\/\/\/oYO69WrVyUVKjrEb9WqlZx11lnRXQ+Rg2cebsU8IQEmhIiKbmHnqsisf4VKgNHlGdtYdL+2bdtmdrHHyEunTp3kvffek1122UVuv\/32SssysBHs4MGDzXqY4447LlPw1q1b5fe\/\/7307t3b\/LdHjx5mQGH\/\/fc3i32zLwKMj18AE4xsAGLq1KmT19FNmzbJ+PHj5Zprrqn2M7DoVaJqDp5xmDPEBegiwBSuNNfAFK4Z77BXgABjr1UaUhJg3Lrw1VdfmbUtRxxxhPkSCetdvAsfzIwZM0bOPfdc6dixo\/lnfMH08MMPG9DBgt\/HH3\/cLI0ABMGbbNBBegJMjl\/eamiImkt7udZCUAyVeYuR3FpvnxsWSRFg7PXSmpKdqz7nCDC6PGMbc+eXN02ET6Gxuz3Ww2DPl5\/85CdmUS7WuuCDGUDM3nvvbQrGV73z5s2Tyy67zIzYNGnSRPr27SuLFi2So446ykwjcQSmGo8wfIVRlauvvtpqPYntuUnuqkXVnGwBZv369YI\/3oXv9LFwChUGQ3280q0AO9d0++MXHQFGl2dsY278Qr3HewVTR\/iiF8swMLWEdyumlvBvWKh7+umnm2UYuKA9NoY9+eSTzZKIp556ypz9hlGXN954w4zUZI\/gZI\/ADBo0SLp27ZoJvmXLloI\/mq9aOzEeVeAFgAG8jBo1yhyaV93lnYWEVdJBBz8WGEZByW0BZurUqYI\/uRfmK1FheKVbAez8vHHjRmnevLnpGHilXwFMRWPhoc2WDOl\/mtKPkG0susd4h27fvt2MnmC0xfaC9phW2mOPPQyoYAQHozT4cgnrX\/z6PCz6xWhN7oUDlvFH8xUKYDCFhLUvp556qtkNsLoLXyhhARJgJ3duLk7hbAEmdwRmyZIlZghv5syZmTnIOONOQ1mrV682i7GxmOzAAw9MQ0h5YwAwewBT3dqsVD9EDQsOAIPfoxo3blzDnlzn47KN6fLt9ddfN+\/gsWPHykEHHZQJvsaOwECBuXPnms13MDKBnQH9LoAOXnx\/\/etfDQRUd5ZDsauELcDkxsF9YLgPTLHrZk3Pn1NIumoAp5B0+cVFvD5+4SwGDEt9+OGHZhgKozH4DQrDWpjHwzEDd9xxh5nLww6uONch9\/v0OKsBASZOtZMri51rctqHLZkAE1a5ZO5jG0tG97ClEmDyKLdmzRrz+Rfm2PJdWIiENFhdneRFgElS\/fjKZucan9auSiLAuFIynnzYxuLR2VUpBJhqlMRCJHyejE+48MUOLiwownqXAQMGyDHHHFNlVbQrYwrJJyrA4BM3wBgu7IuCyzsXqNT\/ju2s77rrLrNTJJ45zc+LzhVTlj\/96U\/l0EMPrZF+pdkfv\/bzt7\/9zaytwqm7ubF76fO1Nf78x\/4oTn2wXmn33Xc3H3BgnVnc5Wf3vfQ\/2H\/s+ovNZrGO85e\/\/GUhr83Upw21iDf1T+UwQG8NDE79BKThwu7DuGbNmsW\/p0wPAMz5559v9lDApoX0K\/31FX5hIy4ATG7b8vzL19b48x\/9jVMfAAw2JvUAJu7ys\/te+h\/sf\/fu3QWnXxNgHIKBlqw8gMGx57\/+9a\/5G33KR6A4AqNvhJAjMPGOoEQdwUAbw2e8HIH58S2W9hGo++67z4yeE2C0UIfDOPkVkkMxY8iK8\/MxiOy4CK6BcSxokbNjGyuywI6z5xoYx4Jqyo4Aw8+oNdVXjbESYHS5RoDR5RcBRpdfTqMlwBBgnFYoZlZFAQKMrkpBgNHlFwEmol8QEJfGs4QIMBHNj\/l2dq4xC+6gOAKMAxFjzIJtLEaxHRRFgMkREdvt\/\/GPfzQHTdlcOGkTm90RYGzUYpooCrBzjaJeMvcSYJLRPWypbGNhlUvmPgJMju6bN282Zyu89dZb1o5oPc2ZIzCcQrKu5EwYSgECTCjZEruJAJOY9KEKJsDkyOYd6Y3Pig8++OBqRcWeAX\/605\/MOUgcgQlV\/3hTAQqwcy1ArJQkJcCkxAjLMNjGLIVKSTICjI8ROGW6Xr16Ur9+\/UCbMGKDC0eHa7s4AqPLMXauuvxCtAQYXZ6xjenyiwATwa+tW7fKwoUL5dxzz038PKQwj0GACaNacvewc01O+7AlE2DCKpfMfWxjyegetlQCjIVyqNQ4C+nzzz+vlBqnVt9yyy0yatSozE62FtmlJgkBhmtgUlMZSzQQAowuYwkwuvwiwAT4tW3bNsFhh3\/4wx98U+6\/\/\/5y++23yyGHHKLLeREhwBBg1FVaZQETYHQZRoDR5RcBJsAvHBQFeBk5cqR8+eWX5hNrnESNk0qXLVsmr7zyilx00UXm79ouAowux9i56vIL0RJgdHnGNqbLLwJMgF8Q6OOPP5aePXsKKvf06dPNdFFZWZls377djM787ne\/k\/bt2+tyniMw6vxi56rOMgKMMsvYxnQZRoAJ8GvdunVmnUsti+h1AAAgAElEQVS3bt2kTZs28vXXX8sDDzwgw4YNk02bNsngwYPNvjEAHG0XR2A4haStzmqLlyMwuhwjwOjyiwAT4Bf2enn88cdl3Lhxcvrpp0t5ebksWLBAxo8fb0Zk9txzT7n33nulY8eOupznCIw6v9i5qrOMIzDKLGMb02UYASaEXz\/88INZ\/4JjBDp06CCdO3eW2rVrh8gp2Vs4ApOs\/oWWzs61UMWST88RmOQ9KCQCtrFC1Eo+LQHGwgOvUu+zzz5mxAXX0qVLpVWrVtKyZUuLHNKZhACTTl\/yRcXOVZdfiJYAo8sztjFdfhFgAvz65ptvzPTRQw89JJdddplcfvnl5o4dO3bIHXfcIW3btjVTS7Vq1dLlPKeQjF8rV66UyZMny\/Dhw6Vdu3ap9pCda6rt8Q2OAKPLM7YxXX4RYAL8WrVqlVx\/\/fVmt93jjz8+MwKD29auXWs2sZswYYK0bt1al\/MEGAKMuhqrL2ACjC7PCDC6\/CLABPiV\/Rl1blJ0Tv379zdfIeHTam0Xp5B0OcbOVZdfnELS5xfbmC7PCDAWIzCPPfaYDB06tMpmdUuWLDGfUVdUVBBgdNX7TLScQlJqnJKwOQKjxKh\/hUmA0eUXASbAL1Tou+66S\/A59XnnnScNGzYUHC\/wxBNPyJQpU8zfZ86cySkkXfWeAKPUL21hE2B0OUaA0eUXAcbCLwDL3XffbUDmu+++y9yBc5AmTZokRx99tEUu6UvCKaT0eVJdROxcdfnFKSR9frGN6fKMAFOAXwAZLOrFEQL4pBoLd3fbbbcCckhXUgJMuvwIioada5BC6fs5R2DS5wl\/SdDlSXXREmAierl582aTQ5MmTSLmFP\/tBBh+Rh1\/ratZJRJgdPnNXxJ0+UWAieAX1sU8\/fTT0qBBA+nUqVOEnJK5lQBDgEmm5tWcUgkwurwmwOjyiwCT49eHH35oPos++eSTzYGNH330kfTp00c2btyY19l58+YRYHTVe5XRsnPVZxsBRpdnbGO6\/CLA5Pj16aefyg033CD\/+Z\/\/aTavw+nTOMARi3fLysoqpcaZSC+\/\/LKMHDmSAKOr3quMlp2rPtsIMLo8YxvT5RcBxsKvp556ypx5dNhhh1VJDQFxcQrJQsgUJuE+MCk0pYRCIsDoMpMAo8svAkyAXzjz6MUXX5QDDjjAnHtUShfXwHANTCnV5zQ+CwEmja7kj4kAo8svAkyAX8uWLZPevXvLFVdcIRdddJHKQxvzPSIBRldjZeeqyy9ES4DR5RnbmC6\/CDABfqEDGjNmjFx55ZW+IzBvvvmmyaFDhw66nOdhjur8YueqzjICjDLL2MZ0GUaACfBr69at8tJLL8kzzzxjzjuqV69e5o7vv\/9eFi9eLD169OAaGF31PhMt18AoNU5J2ByBUWLUv8IkwOjyiwAT4NemTZtk0KBB8s477+RNyc+odVX67GgJMHq90xA5AUaDS\/+OkQCjyy8CTIBfqNBTp06Vo446Sg4\/\/PBKqTEC88gjj8iRRx7JERhd9V5ltOxc9dlGgNHlGduYLr8IMBZ+4bgAnHlUv379Kql5lICFgEziRAF2rk5kjDUTAkysckcujG0ssoSxZkCAiSg3F\/FGFDDh2zmFlLABJV48AUaXwQQYXX4RYHL8QgV+9913pWnTptKiRQvBIt733nvP11VMIf35z3+W3\/zmN5xC0lXvM9ESYJQapyRsAowSo\/4VJgFGl18EmBy\/Xn\/9denVq5ecccYZMn78eAMwcSziXbBggYwaNcpE07x5c5kzZ06VowuyQ\/3ss8+kX79+snz58kpPMGHCBOnZs6dVLeQ+MFYypSYRO9fUWGEdCAHGWqpUJGQbS4UN1kEQYAJGYFChb775ZunYsaNZrJt9eZ9RH3LIIZFGYAAv8+fPl9mzZ0ujRo0k9+9+bq5atUpGjBghkyZNqhZ0qqsJBBjrdpKKhOxcU2FDQUEQYAqSK\/HEbGOJW1BQAAQYC7kAC3Xq1JE2bdpUSb1mzRrzbz\/72c8scqqaxBtJOeecczIjJzi+AAdIdu7cOe9oCoybOHFiBnrCFE6A4VECYeoN77FXgABjr1UaUhJg0uCCfQwEGAuttm\/fbkZIFi1aJCtWrDCb2XXp0sUcLXDooYdGOl4g30gKRmGWLl0qFRUVUrdu3SpR4udr1641J2GHvQgwBJiwdYf32SlAgLHTKS2pCDBpccIuDgJMgE4YDcFRAo8++qg0btxYTj31VPPfdevWyQsvvCCjR482O\/HWqlXLTvGcVPlGUqqbRvJGaLALcPZVyPoX3EeACWVZYjexc01M+tAFE2BCS5fIjWxjicgeulACTIB0H3zwgfTp00fOOussGTx4sOy+++6ZO9A5TZ48WQYMGBB6HUoYgPGmnVq1apUZocFIDuIcMmQIF\/GGbg7pvpGda7r98YuOAKPLM7YxXX4RYAL8wifVN9xwg9x6662y7777Vkn9\/PPPm3878cQTQzkfBmDyFRS0+Hf9+vWCP96F6bBx48bJzJkzzSLlmnitXr1apk+fbuD0wAMPTLUEWDS+YcMG83n\/LrvskupYGdyPCmzZssWMzmJxPq\/0K8A2ln6PsiPEV8MDBw40Xwp37do186OWLVsK\/mi+au3cuXNn1AfYtm2bXHfddeaT5YMPPrhKdg888IDst99+GfGefPJJMxqDPzaXS4BBXsOGDcv7CTaORMCf3Ou2226T9u3b24RbcmkAMLNmzTINwG+RdpoeGL8d\/v3vfzef2RNg0uRM\/lgwWgqAadiwoY6Aa3iUbGO6KgAA5sorr6wS9NChQwV\/NF9OAAb7wDz22GMCobp3717pxYGfLVy4UPAFERb2fvvtt2atDISzBZiwi3j9jAkCmNwRmCVLlsiMGTNE62GUmitnmNg5vB1GtWTv4RRSsvoXWjrbWKGKJZvem0IaO3asHHTQQRyBybXD5jTq7HtsNqHLTh\/mM2pvvcuUKVMq7T8T9OVS7rNxEW+yja\/Q0tm5FqpY8ukJMMl7UEgEbGOFqJV8Wq6BCfDAO4365z\/\/ubRt27ba1EiL6Yizzz7begQGGQI8pk2blpn6CVrLgnuwBwzM8za\/84wsZDSFAMPPqJPvgko7AgKMLn8JMLr8IsBY+IWFk5gi8juNGtNGuHBaNa7q0lZXVNBRAgAWXNn7vuDfMAXkXYXAC+4hwBBgLKo\/k0RQgAATQbwEbiXAJCB6hCIJMBHEw63PPPOMAZtOnTpFzCn+2wkw8WsepUR2rlHUS+ZeAkwyuoctlW0srHLJ3EeACdD9hx9+kKefflruvvtuc0o1KnjuVejIRzJWVy2VAJMWJ+ziYOdqp1OaUhFg0uRGcCxsY8EapSkFASbAjffff18uvPBCad26tRxzzDFSu3btSne8+uqr5qsjjsCkqVrbx7Jy5UqzGeHw4cOlXbt29jcmkJKdawKiRyySABNRwJhvZxuLWfCIxRFgAgRctmyZ3HTTTWazt6ZNm1ZJHfUwx4j+RbqdIzBcAxOpAvHmQAUIMIESpSoBASZVdgQGQ4AJkAh7veCMoSuuuMJ3J14CTGAdYwJHCrBzdSRkjNkQYGIU20FRbGMORIwxCwKMhdiYZrj33nvlV7\/6VZUdUJ999lk57bTTOIVkoSOTRFOAnWs0\/ZK4mwCThOrhy2QbC69dEncSYAJUx1EC48ePl4cffjhvSi7iTaLquimTa2Dc6Mhc\/BUgwOiqGQQYXX4RYAL8whqYSy65xCzUPeWUU6ROnTqV7sDZRzgEkIt4dVV8L1oCjE7ftERNgNHi1I9xEmB0+UWACfDrzTfflIqKCrnzzjt9T5TdvHmzyaFJkya6nOdGdur8YueqzjIhwOjyjG1Ml18EmAC\/vvrqK7n22mtlwIABvqdRcxGvrgqvOVp2rvrcI8Do8oxtTJdfBJgAv3BUwBtvvCGLFi2S008\/nYt4ddXvwGg5hRQoERNEUIAAE0G8BG4lwCQgeoQiCTAB4q1bt0769Okj3kiLX3Iu4o1QAxO+lQCTsAElXjwBRpfBBBhdfhFgLEZgpkyZYj6V3n\/\/\/auk5iJeXRVec7TsXPW5R4DR5RnbmC6\/CDAWfq1fv14aNGgge+21l0mNSr5z507ZddddhYt4LQRkEicKsHN1ImOsmRBgYpU7cmFsY5EljDUDAkyO3AATLNxFRa5Vq5bsvffeVc4\/+uKLL2TWrFmyfPlyOf744+WMM87gV0ixVlt3hXEKyZ2WzKmqAgQYXbWCAKPLLwJMjl87duyQ8vJywQZ2gwcPlkMOOUS2b9+eSQWoqVevnlnM++KLL8pll11mTqrmPjC6Kr4XLQFGp29aoibAaHHqxzgJMLr8IsD4AMyYMWOkd+\/e0qFDBwMvzz\/\/vEycONGcVgyoOfzww82GdvgZYAdpCTC6Kr7GaNm56nONAKPLM7YxXX4RYHwA5sYbb5QLLrhAysrKzE8xrYQDHX\/xi1\/IcccdV+kOgE3Xrl0JMLrqvcpo2bnqs40Ao8sztjFdfhFgLAAGSfKBCgFGV4XPjZZTSLr9S3v0BJi0O1Q5PgKMLr8IMAQY0bqPjYumRoBxoSLzyKcAAUZX3SDA6PKLAOMDMDh9GlNIbdu2zfzUb6QFC36xXqZHjx6cQtJV71VGy85Vn20EGF2esY3p8osA4wMwWJi7ePFiaye1jmD85S9\/kYEDB9boERhrk1OQkJ1rCkwoMAQCTIGCJZycbSxhAwosngCTB2CWLl0qLVu2rFbOr7\/+Wj788EOZO3cuR2AKrHhpSc4ppLQ4UZpxEGB0+UqA0eUXAcYHYKZNmyZ9+\/a12pxuwYIF5ogBfkatq+J70RJgdPqmJWoCjBanfoyTAKPLLwJMjl+FVmAeJaCrwmuOttC6qflZSyV2AowuJ9nGdPlFgNHll9NouQbGqZxFz4yda9Eldl4AAca5pEXNkG2sqPI6z5wA41xSPRkSYEQ4haSnvmqMlACjyzUCjC6\/CDC6\/HIaLQGGAOO0QjGzKgoQYHRVCgKMLr8IMLr8chotAcapnEXPjJ1r0SV2XgABxrmkRc2Qbayo8jrPnADjXFI9GRJg9HiFSNm56vIL0RJgdHnGNqbLLwKMLr+cRkuA4RSS0wrFzDiFpLwOEGB0GUiA0eWX02gJMAQYpxWKmRFglNcBAowuAwkwuvxyGi0BxqmcRc+MnWvRJXZeAKeQnEta1AzZxooqr\/PMCTDOJdWTIQFGj1dcA6PLKy9aAowu3wgwuvwiwOjyy2m0BBhOITmtUMyMU0jK6wABRpeBBBhdfjmNlgBDgHFaoZgZAUZ5HSDA6DKQAKPLL6fREmCcyln0zNi5Fl1i5wVwCsm5pEXNkG2sqPI6z5wA41xSPRkSYPR4xTUwurziGhidfhFgdPlGgNHll9NoCTCcQnJaoZgZp5CU1wECjC4DCTC6\/HIaLQGGAOO0QjEzAozyOkCA0WUgAUaXX06jJcA4lbPombFzLbrEzgvgGhjnkhY1Q7axosrrPHMCjHNJ48vQM88rcd68edKpUyfrAAgw1lKlIiE711TYUFAQBJiC5Eo8MdtY4hYUFAABpiC50pMYxg0bNkzmzJkjZWVlkvt3m0gJMJxCsqknTBNeAQJMeO2SuJMAk4Tq4cskwITXLrE7d+zYIeXl5dK8eXMZOXJkJo6JEyea\/8\/+t+qCJMAQYBKrxDWkYAKMLqMJMLr8IsDo8stE+9lnn0m\/fv0MqGRPGcFMQMzs2bOlUaNGgU9GgAmUKFUJ2Lmmyg6rYAgwVjKlJhHbWGqssAqEAGMlU7oSrVq1SkaMGCGTJk0y00feVeg0EgEmXb4GRcPONUih9P2cAJM+T6qLyKaNLVq0yEzZ8yquAvjlvHv37tUWQoAprgdFyT0swKxfv17wx7tWrFgh48aNk7lz5xa0+LcoD5VQpqtXr5bJkyfLlVdeKe3atUsoCrtibTpXu5yYKi4FNm\/eLDt37pSmTZvGVSTLiaCATRsbNWqUvPLKK3LsscdGKIm3VqcA9O3YsaPpm6u7XnvtNenVq5cMGjRIunbtmknasmVLwR\/NV62d6DlK8AoLMFOnThX8yb1uueUWad++fQkqFfxIH330kcyaNUv69+8vbdq0Cb4hwRToXDdt2iTNmjWTOnXqJBgJi7ZVANO9uGymdG3zZLriKWDTxjDyvddeewW+XIsXZennPHz4cNm2bZuZaajuWr58ufnlM\/caOnSo4I\/miwCT417uCMySJUtkxowZZgTmqKOO0ux1jYgdnSs8xG8WBBgdlmMEBleTJk10BFzDo7RpYxiBQfsLGh2o4VJGenwADLyYMGFCtflgCqlv374yduxYOeiggzJpOQITSf7i3sxFvMXVN6252wxvpzX2mhoX18Doct6mjeHliosAUzxvbTXmGpjieVC0nPkZtTtpV65caToiNBiugXGnK3P6UQECjK6aQIBJh18EGJGSnUJCFfPI09t9t9AvkJAHv0LiPjDp6K5KNwoCjC5vazLA4NmxvhLTnl26dEnUOAJMiQNMNsR4NY1HCSTa5opeuE3nWvQgWEBBChBgCpIr8cQ2bcz25Zr4wxQQwPbt2+X555+X8ePHy1lnnWW9GWoBRWRGJLF7fO3atc1O8vkuW405hVSoAyWUniMwusy06Vx1PVHpR0uA0eWxTRuzfbnqevJ\/b5CK\/VdsdnP\/4IMP5LnnnpOLL77Y6lFfeOEFWbduncycOVO6detWbRm2GhNgrKQvzUQEGE4hlWbNTs9TEWDS44VNJASYfmZPsCCA+fTTT+Wyyy6TDh06BKbN1t37ACWoDAJMDZhCsmmQ1aUhwBBgotYh3l+9AgQYXTWkpgHMO++8IwsXLpTDDjtMli5daqaRsDEcAGbr1q1mtGTfffc1+09h9ASfKzdo0EAefvhhue2228yWDscdd5yceeaZsvvuu8vtt98uRx55pGCDufr16xvIwZ453kWAsW8PJb2I116G\/CkJMC5UjC8Pm841vmhYko0CBBgbldKTxqaN+Y0OPLRsU3oewieSc49uVuVfMQUEIKmoqJDWrVvL2rVrzRl7p5xyigGY6dOny7PPPms2+sSFzT4BKth3JRdEvv32W5MX1tIgvw0bNkifPn3khhtukBNPPJEAE6J2EGACRCPAhKhVCd5i07kmGB6L9lGAAKOrWti0MT+AaXTlC6l+0M9uOaFSfHhOHPz71VdfmeNkdttttypQ8tZbb8m7775rziPCrrgAmM6dOxu4yQUYbHqPNS5ffvmlnH766bJmzRoDMBiB6dmzJwEmRO0gwBBgAqsN94EJlIgJIihAgIkgXgK3hgWYSx96P4Fo7Yu8\/dyDKyUGaAwePFgOPfTQzBoWv+kdjKg8+eSTZpoJQIPRFz+A8TJHHki7ePFiQd+KERgCjL1P2SkJMASYwJpDgAmUiAkiKECAiSBeAreGBZgEQo1UpAcrRx99tOBohFq1alUZVcFC3csvv1zOOOMMOemkk2TgwIGZBb5+sIMpKaTHCFWLFi0M7AwZMoQAE9IpAgwBJmTVSedtNp1rOiOvuVERYHR5b9PGbL+QSfOTY2SlvLxctmzZYhbeNmzYsBLA4BBFrIEBlGBNyzfffGPWx3hfD+UCDH4+ZswYs8AXQLR69WozhUSACV8LCDAEmPC1J4V32nSuKQy7RodEgNFlv00bKwWAgSsLFiww0HHjjTdKjx49DKwAUrDoFtNEWJT7ySefyLRp0wzcYAQGIzF4\/u+++878vU2bNnL11VfLxx9\/bKaLmjVrZg5gXLZsmVn\/csUVV0jv3r3NF0q4+BWSfXsgwBBgAmsLp5ACJWKCCAoQYCKIl8CtNQlgMGry4IMPyp133mlGYI444ghzlMAJJ5xgpo3ef\/99ue666wRfGA0YMEDWr18vf\/jDH+T66683wHPffffJrbfeahb2YtQFn2BjYTBOXh80aJAsWbJE3n77bfNvgCIsGH700UdlypQp5uRoANLBBx9sTvbOvWwhkRvZJdBI0lIkv0LiPjBpqYulGgcBRpezNQlg0uwMAYYb2QXWTwJMoESpSmDTuaYqYAbD06iV1QGbNmb7clX26KkK11ZjjsCkyrZ4gyHAxKt31NJsOteoZfB+twpwBMatnsXOzaaN2b5cix1rKedvqzEBppRrQcCzEWA4hVSDq38sj06AiUVmZ4UQYJxJGSkjAgynkAIrEAGGABNYSZggkgIEmEjyxX4zASZ2yX0LJMAQYAJrIgEmUKJUJbDpXFMVMIPhGhhldcCmjdm+XJU9eqrCtdWYU0ipsi3eYAgw8eodtTSbzjVqGbzfrQIcgXGrZ7Fzs2ljti\/XYsdayvnbakyAKeVaEPBsBBhOIdXg6h\/LoxNgYpHZWSE1GWDw7NgHZvPmzdKlSxdnmobJiADDKaTAekOAIcAEVhImiKQAASaSfLHfXFMBBkcLYCO68ePHy1lnnZU54NGVAdD1scceM8cWfPTRR+YQyWHDhskvfvELcw5T7kWAIcAE1j0CTKBEqUpg07mmKmAGwzUwyuqATRuzfbkqe3Trbf6958LRA88995xcfPHFgY+KHXyxE+\/ZZ58tX3zxhTm+4NVXX5V7773X7MZLgKkqIY8S4BRSYMPSlMCmc9X0PDUhVo7A6HLZpo0RYERwUjXOOurQoUPgaM2XX34pd911lwGdevXqmQrx+uuvy4UXXijXXHNNpdOqvdpiqzHXwOhqX06j5QgMp5CcVihmVkUBAoyuSlHTAOadd96RhQsXymGHHSZLly4100i9evUyULJ161aZOXOm7LvvvrJp0yZZt26dOb8IJ04\/\/PDDctttt0nLli3luOOOkzPPPNMc2IgpoiOPPFJee+01qV+\/voGcPfbYQ3bs2CF77rlnpjJgrQ1Oqx43bpw5J4kjMByBKbinIMAQYAquNLyhIAUIMAXJlXjimgQwmAICkFRUVEjr1q1l7dq15jTqU045xQDM9OnT5dlnn5VZs2YZX\/r3729ApW\/fvlWmm3DgI\/LCWhrkt2HDBgMoOKHaD1BwWvW1114rd9xxhymbAEOAKbjxE2AKlizRG2w610QDZOEcgVFeB2zamN\/0xj9u75PqJ2966ZxK8eE5cUo01qVgFGS33XarAiVvvfWWvPvuu9K9e3fZtm2bARicPA24+eyzzwzsdOrUyfx9586d8sILLwimi04\/\/XRZs2aNARiMwPTs2bNK2VOnTjWnVp9\/\/vlcxJun5nANTECTIsCkus+pEpxN56rriUo\/Wo7A6PLYpo35Acyas6t+SZOmJ\/\/ZIzsrhQPQGDx4sPkaCACCKxdK8G8YUXnyySfNNBOABqMvfgDjZY48kHbx4sWycuVKMwKTCzBvvvmmLFiwwKx\/2WuvvXxl4hoYfoUU2H4IMJxCCqwkTBBJAQJMJPlivzkswHz14r2xx1pIgfWOv7BScg9Wjj76aBk1apQZBckFGCzUvfzyy+WMM86Qk046SQYOHJgZcfGDHUxJIT3go0WLFgZ2hgwZUglgkCfWyVx66aWyzz775H0EAgwBJrB+E2AIMIGVhAkiKUCAiSRf7DeHBZjYA41YIEZWysvLZcuWLQYoGjZsWAlgRowYYdbAAEqwpuWbb76pNGWUCzD4+ZgxY8wCXwDR6tWrzRRSNsAAXjB1NHTo0Ay8YKHwLrvsYhb8Zl8EGAJMYBUnwARKlKoENp1rqgJmMNwHRlkdsGljti\/XtD86pnEAHdiTpUePHgZWsK4Fi24xTYRFuZ988olMmzbNwA1GYDASg+f\/7rvvzN\/btGkjV199tXz88cdmuqhZs2YyYcIEwSJdrH+54oorpHfv3gaAsOamW7du0rZtWyMNIOr+++83IzW4jwBTucZwDUxACyLApL2LqRyfTeeq64lKP1qOwOjy2KaNlQrAACoefPBBufPOO80IzBFHHGGOEjjhhBPMtNH7778v1113neALowEDBsj69esFG9Jdf\/31Bnjuu+8+ufXWW83CXoy64BNsQAoW5w4aNEiWLFkib7\/9tkmPDe9wb+4FwMFanNzdeG015j4wutqX02gJMJxCclqhmFkVBQgwuipFTQKYNDtDgOEUUmD9JMAQYAIrCRNEUoAAE0m+2G8mwMQuuW+BBBgCTGBNJMAESpSqBDada6oCZjBcA6OsDti0MduXq7JHT1W4thpzCilVtsUbDAEmXr2jlmbTuUYtg\/e7VYAjMG71LHZuNm3M9uVa7FhLOX9bjQkwpVwLAp6NAMMppBpc\/WN5dAJMLDI7K4QA40zKSBkRYDiFFFiBCDAEmMBKwgSRFCDARJIv9psJMLFL7lsgAYYAE1gTCTCBEqUqgU3nmqqAGQzXwCirAzZtDC\/XV155RY499lhlT6cnXE\/fyZMnVxs0p5D0eOo8UgKMc0mLmqFN51rUAJh5wQpwBKZgyRK9waaNLVq0SPDi5FVcBXBQJA6SrO4iwBTXg1TnToDhFFKqK2gJBEeA0WWiDcDoeqLSjpYAU9r+Vvt0BBgCTA2u\/rE8OgEmFpmdFUKAcSZlLBkRYGKROZ2FEGDS6Uu+qNi56vIL0RJgdHnGNqbLLwJMSvzCwVo4TwJX8+bNZc6cOVJWVpY3Ou800OXLl1dKg4O0evbsafVUBBgrmVKTiJ1raqywDoQAYy1VKhKyjaXCBusgCDDWUhUvIeBl\/vz5Mnv2bGnUqJHk\/t2vZBy6hSPPJ02aVC3oVBc1AYZTSMWr1cyZIzD66gABRpdnBJiE\/fJGUs4555zMyMmOHTukvLzcnPKZbzQFxuHkTw96wjwGAYYAE6be8B57BTgCY69VGlISYNLggn0MBBh7rYqSMt9ICkZhli5dKhUVFVK3bt0qZePna9eulZEjR4aOiwATWrpEbmTnmojskQolwESSL\/ab2cZilzxSgQSYSPJFvznfSEp100jeCM3ixYsrBVDI+hfcSICJ7l+cObBzjVNtN2URYNzoGFcubGNxKe2mHAKMGx1D5xIGYJjOtBQAABDXSURBVLxpp1atWmVGaDCS06dPHxkyZEjeaaf169cL\/njXihUrZNy4cTJ06NAau6vkli1b5J577pEzzjhD2rVrF9rHuG7ctGmTNGnSROrUqRNXkSwnggKff\/65ubtBgwYRcuGtcSrANhan2tHK2rBhg2Bn5EGDBknXrl0zmbVs2VLwR\/NVa+fOnTvT\/gBhACbfMwUt\/p06dargDy8qQAWoABWgAqWqAH4pxx\/NV40DGMDQsGHD8n6CnTsCg\/MmADQYhWnWrJlmr2tE7K+\/\/rrMmDGDfilxm34pMSorTHqmyzPPL5yZ1KJFC47AFMs+v31b5s2bJ40bN\/b9HDpoEa9fnEEAk3uPN3+IOHDuBK90K0C\/0u0P25cuf\/L1ob169RL2iTq8LOU+UcUITJjPqL31LlOmTKkEHoVCTymbr6P5FRYl\/SpMr6RT06+kHSi8fHpWuGZJ3lHKfqkAGJgP8Jg2bVpm6idoLQvuwR4wMM\/bByaMkWHuSbKy1vSy6ZeuGkC\/dPmFaOmZLs9K2S81AONBTHVHCQBYcGXv+4J\/w5oI7yp02BNrYnA0PI4s175iW1ezCxct\/QqnW1J30a+klA9fLj0Lr10Sd5ayX6oAJgnzWSYVoAJUgApQASqQPgUIMOnzhBFRASpABagAFaACAQoQYFhFqAAVoAJUgApQAXUKEGDUWcaAqQAVoAJUgApQAQIM6wAVoAJUgApQASqgTgECjDrLGDAVoAJUgApQASpAgMlTB3JPs+7WrVvmUEhWm+QV8Nu1GVHlnjbu7YHgRVzoZ\/TJP6n+CODBQw89VKX92LQxmzT6FUrfE+TzDPtveVtZeFG3b98+s9cW\/o2exeNnbh\/YvHnzKkfk2HhhkyaeJyq8FAKMj2aeoagQ2FMm9++Fy8w7XCuAnZZHjBghkyZNkrKyMt\/sc4+NKPQYCdcx18T8vB2xO3bsWAlgbNqYTZqaqGmxnzmfZyjXb6+t7HjoWbHd+TF\/D15wvI2371nuZq82XtikieeJwpVCgPHRze9FZ\/PCDGcB7wqjQL4Tyr288kFnUAccJhbe469A9m\/ruSOYNm3MJg21d6tAdZ55bapz587Ss2dPq18akIh9p1uPkJtf\/5fb59m0H5s07qN3lyMBxkdLv\/OSbBqvO1uYU5AC8Gjt2rWVdl3Ovsf7DQW\/nWQfwhkEPkHl8ud2CngvQkzZLVmyRDZu3FhpBMamjdmksYuGqWwUCPIMbeqqq66S0aNH5x31pGc2ShcvTfYvaDZe2KQpXrTRcybA+Gjo91s6p5GiVzZXOeTO2Xr5Zq9\/yfdbH6eRXLlgnw\/aUy7A2LQxmzT2UTBlIQr4eZa7ngz55a5\/oWeFqOw2be6hxzZe2KRxG6Xb3AgwBBi3NSqG3LyG2qpVq8xv9d68\/ZAhQ8zwNgEmBiMsiyDAWAqVomR+nmWP0HijmrkH5mp\/IabIgoJDyV0DY+OFTZqCA4nxBgIMASbG6lbcorJPKN+yZYvvIl+OwBTXA7\/cCTDxax61RD\/P\/PIM81t\/1Nh4f1UFvNGx7FFoGzixSZNmvQkwBJg018+CYsuGE9zo95USAaYgSZ0kJsA4kTHWTGwBJndqXfsLMVaRHRXmBy\/I2sYLmzSOwixKNgQYH1m1L2wqSk1RkGk2nDRu3Fj69etnFvlyEW+y5uWbjli6dKnvp9XeVy5sh8n5FhZg6Fm8nuWDF0Rh44VNmnifqLDSCDA+emn\/tKywKqAvtbfeZcqUKZXgJLsx4qnKy8vF28vHe0p+Rh2\/3\/kWhA4bNqzSxlu565bYDuP3Krud+C28zv233K\/96Fl8nlUHL4jCxgubNPE9UeElEWB8NPOGRfGjiooKk8LvZVi43LzDlQK5iwe9xpy9027uv3H6yJX6heXjBzA2bcwmTWGRMLWtAn6e5S6Uz\/Wnbt26mU0\/2XfaKh0uXa4XfrnYtB+bNOEijOcuAkwenTVvrxxP1Um+FHSyM2bMyATid0wAjxJIh0+5v7kjKps2ZpMm+ScsvQjyTSF5L074icvviBV6Vvz6kNv3ZZeY7YmNFzZpiv9E4UogwITTjXdRASpABagAFaACCSpAgElQfBZNBagAFaACVIAKhFOAABNON95FBagAFaACVIAKJKgAASZB8Vk0FaACVIAKUAEqEE4BAkw43XgXFaACVIAKUAEqkKACBJgExWfRVIAKUAEqQAWoQDgFCDDhdONdVIAKUAEqQAWoQIIKEGASFJ9FUwEqQAWoABWgAuEUIMCE0413UQEqQAWoABWgAgkqQIBJUHwWTQWoABWgAlSACoRTgAATTjfeRQUqKfD+++\/LfffdJw8\/\/LDUq1dPfvOb38iWLVvk\/\/7v\/6SsrEwuuOACadu2rdSqVSuScv\/85z\/l9ttvl8cee0xmzpwpBxxwgG9+tunCBFPMvMPEk\/Q93nEVv\/3tb+W0006TBg0ayKGHHip16tRxHtq6devk448\/lg8++ECmTp0qvXr1Mieu86ICNVEBAkxNdJ3PXBQF8FLp06eP7LPPPjJ79mxp1KiRfPnll3LdddfJ448\/LjfeeKP06NEjEsTs3LlTHnroIfnb3\/4mY8aMMWV514YNG6RJkyay++67S3Xpoj58MfOOGlsS9\/sdJFrsOLxToDt16kSAKbbYzD+1ChBgUmsNA9OmgHfQXdOmTTMAg2dYtmyZ9O7dW9q3b29GTRo2bOj80TZt2iSTJ0+Wq6++2oATr\/gUIMDEpzVLogLZChBgWB+ogCMF8gGMNzKDqYU5c+ZUGjVxUbQ3yvPRRx9VAicXeTOPYAUIMMEaMQUVKIYCBJhiqMo8a6QC+QBm8eLFMnToUPGOuf\/hhx9k7ty5AuBo3LixvPzyy\/Jf\/\/Vfcv7558uee+4pWGOCNS5YV3PsscfKU089JQceeKAMGjRIsAbi0UcflUceeURuvvlm6dixoyD\/u+66S7744gs588wzTdojjzzSTFt56TDVgGv79u15y95jjz1k9erV8sQTT8gLL7wg\/fv3N9NVeEF36dJFbrrpJmnevHmVGBBj0H0tWrQw5W\/dulXuvPNOsy4I010YNdp1112ldu3acuqpp8pFF11Upe5AryVLlpi4dtttN3nrrbfM2o+zzz5b8LPXXntN7r\/\/fqMV4vjjH\/8o48ePFzwzdMR021lnnWXK2n\/\/\/c3akR07dsj06dNl7733NpqsWLFChgwZIl27djV5vvPOO7JgwQKBJhjRmjZtmlxyySVy6aWXVlnbkg9g4OOf\/\/xnEzueFXEgZvzZvHmz8XXhwoVy4YUXmjrwzDPPZDQB7N56663y4osvSrt27YzXWEPlXZxCqpFdDB86RwECDKsEFXCkQC7A1K9fX\/7617\/K2LFjzboUvLgBANdee615cY4ePdq8DDFCc\/HFF8t\/\/Md\/yPXXX28AYdiwYeYFhhf92rVr5d5775URI0aYl+u8efOkoqLC\/BcvabyMy8vL5ZNPPjEjMJiiwks5N922bdsCy4YUkyZNkvnz50u\/fv1k8ODBBhDw39\/97ncyfPjwKnkDHFBe0H3ff\/+9TJw40SxCveWWW2SvvfYyMAXYwH89yMq1A8D20ksvGYCqW7euATYAIWLByx9gA20Ac9ANIPCLX\/zCLHBGmtdffz0ztfbee+8ZUEI6TLedeOKJZr0QQAI+jRs3zkDgK6+8YoAFa4zw72+\/\/bbsu+++0r179yprmPwABvACvwFsWGQL8AIw\/f73v5fbbrtNTjrpJOMP1kWde+65Bp5QF\/Dfd999Vy677DIDtJ9++qkMHDjQTD8iNuSDiwDjqNEyG9UKEGBU28fg06SABzD\/+Mc\/zCgIXjY\/\/elPzegFvk7Bb\/L4bXzAgAFyzz33yHHHHWfCxwsULzf8ln\/33XcLwAcvL7zY8KIG\/ABq9ttvP\/OSw8jAqFGj8gKMtwYmN51N2RiBAGQAEjDdBYDyXpatWrUy4ASIyM0bzxF0H0ALUNSyZUuTD0ab8LLGwmcARc+ePavYiekxwBM0OOSQQ8zPMXoBqIGuACHkgREZAM4555yTycMDOwDgrFmzBGuTABaI89VXX838mzcyhNEVgNiMGTPkq6++MnFhZAxaV\/f1mB\/AoC5gNGnKlCnSoUOHzOgTRlqgMTy20dB7ho0bN1ZaP0WASVPLZyxJKUCASUp5lltyCuSbQsp+UEwF4DNob\/TE+xlebIAWvEQBLUiDP5h6wQgAXnqYZsEVFmBsyr7qqqsCQSQswGAE5sorr5T169ebkSKMaEAzTFVhdAHPmHt5mmJkwg9wkN4DiAkTJlRKkzsyBbDDiAhGNAAy3pdiHkTifox0Pfjgg+ZTaAAMpv2CPlP2AxiMCiE\/DwL9KnshAOONrnlwSoApue6DDxRCAQJMCNF4CxXwU8AGYDDSglEDjLb8+te\/zmTjvQTxgseIA6aK\/ud\/\/sekXb58uWCPkWuuucZMu4QFGNuyg0ZSwgIM7gOoXXHFFWakBFM0GEnBehPv2fIBDNJDF7+rEIDxRnQwogG4wKiWd+G5H3jgATOdBZ2jAAw88qbGvBGY3NgJMOxHqEA0BQgw0fTj3VQgo4ANwHjTONjYDr\/Ze5ud4Td2vNjx8sR0CaYyMMX0zTffmNEa\/DaPkYGjjz46NMDYlI11KMUEGIyKYDqndevWZq0OFtVilMkbXcqtTpguwojJLrvsYqCvWbNmJglGUKAV1rDgE3JMIdmMwHhTSNh0ENN13qjPt99+a0a6sB4Jn7pjpCYKwACqMKKG0SVAqecz1rRgTQ6mFAkw7DyoQDQFCDDR9OPdVCCjgN9Gdrny4AWODejwgsMUBr4wweJarLMAuNxwww1mxAUjE\/h\/\/Bvyvfzyy82XKAcffHCVF5\/38sV+M4ADwA\/W3jz\/\/PNV1soElY1RkmIBDL42wrQYYAUvdpQVdGF9EL6wwhdEJ5xwgoE8LIDGcx5++OFmUS0W3NoCDMr78MMPDRRhTQ2gB6MtK1euNP+G6Tt8seTBaNgpJM\/Tp59+2izQxggavjr705\/+ZBYWY3EwASbIff6cClSvAAGGNYQKOFAg+ygB\/LaNlyFefgAUP4jBWgsslMWIBxb94mWKL2rwUn\/zzTcNvBxzzDFy1FFHmc9t8fLGglJ8JozPgJ988knzVRDKwQiGNzWDhcOAIZTrlw4Ala\/sn\/zkJ+ZzYoxEAKKwsBaf\/GLkxtv1F19QYRQEX9Jkx4Cpmeruw9dVGD0CMGDkJPvC0QtY34IvbwATuRdGoTD6hK96cDwDtunHc2MUA4ub8Tz4jBr6YIE0wAb34MslwBhGcfDF18knn5zZ5A+7FmMhMfI77LDDDFycd9555usl+IEjIfCM+PoH64IwDQSY9LvyfUaNURyMGiF2XP\/93\/9tvjKCX1gHhPKhIdY9YUQOseCrJOQH7bFrMz4Zx78BiLCjM0acEAfXwDhotMxCvQIEGPUW8gGogA4FsK4HI0tYzAsgW7NmjZkKwggS9qzBp8XZ64J0PNW\/FxHnLswuZvwEmGKqy7y1KECA0eIU46QCyhXAaANGkzAKg0+osy9Md2F0iABjZzIBxk4npiptBQgwpe0vn44KpEYBfAWFhbPYrwXTOZjuArRgUSvW72D6x28KKTUPkCcQTPnhM3Dsk4PN9LBZIRYHY82P6wt73mB6D9Nc2CTxlFNO8d292HW5zI8KpFEBAkwaXWFMVKAEFcA6GWzgt2jRIsEaFKx9wRogrOXp3Llz3i+RSlAKPhIVoAIOFCDAOBCRWVABKkAFqAAVoALxKvD\/i6AOZPY6S9oAAAAASUVORK5CYII=","height":337,"width":560}}
%---
%[output:71af41bb]
%   data: {"dataType":"text","outputData":{"text":"    <strong>Rotation_deg<\/strong>    <strong>P50_cm<\/strong>    <strong>P90_cm<\/strong>    <strong>P95_cm<\/strong>    <strong>ProbabilityBelow30cm_percent<\/strong>    <strong>Outage_percent<\/strong>\n    <strong>____________<\/strong>    <strong>______<\/strong>    <strong>______<\/strong>    <strong>______<\/strong>    <strong>____________________________<\/strong>    <strong>______________<\/strong>\n\n          0         643.83    799.19     918.5                 0                         0      \n         90         618.57    1065.1    1414.2                 0                        10      \n\n","truncated":false}}
%---
%[output:56d44986]
%   data: {"dataType":"text","outputData":{"text":"Running 1000 evaluation episodes...\n","truncated":false}}
%---
%[output:54b30dce]
%   data: {"dataType":"text","outputData":{"text":"Evaluating episode 50 \/ 1000...\nEvaluating episode 100 \/ 1000...\nEvaluating episode 150 \/ 1000...\nEvaluating episode 200 \/ 1000...\nEvaluating episode 250 \/ 1000...\nEvaluating episode 300 \/ 1000...\nEvaluating episode 350 \/ 1000...\nEvaluating episode 400 \/ 1000...\nEvaluating episode 450 \/ 1000...\nEvaluating episode 500 \/ 1000...\nEvaluating episode 550 \/ 1000...\nEvaluating episode 600 \/ 1000...\nEvaluating episode 650 \/ 1000...\nEvaluating episode 700 \/ 1000...\nEvaluating episode 750 \/ 1000...\nEvaluating episode 800 \/ 1000...\nEvaluating episode 850 \/ 1000...\nEvaluating episode 900 \/ 1000...\nEvaluating episode 950 \/ 1000...\nEvaluating episode 1000 \/ 1000...\n","truncated":false}}
%---
%[output:38a767b5]
%   data: {"dataType":"matrix","outputData":{"columns":2,"name":"h3","rows":7,"type":"double","value":[["11.0002","18.0001"],["12.0002","19.0001"],["13.0001","20.0001"],["14.0001","21.0001"],["15.0001","22.0001"],["16.0001","23.0001"],["17.0001","24.0001"]]}}
%---
%[output:7e6b773d]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnX9wHVd59x8XU0mUllRUtWKZRPlDAYZOnUmZkfC8reV00sB0DDN0WtluK1mVg8sUqg6JfiQmtU2bWD\/sgEjaicdRZImiYGgZGnfKFDq13XbcakoymBmmU5u+UfJaRooahyEwRsWgd54V5\/povXvv7t797u7d\/e6MJpF19nvOfp7nnvO955zd3bC6uroqPEiABEiABEiABEighghsoIGpoWixqSRAAiRAAiRAAg4BGhgmAgmQAAmQAAmQQM0RoIGpuZCxwSRAAiRAAiRAAjQwzAESIAESIAESIIGaI0ADU3MhY4NJgARIgARIgARoYJgDJEACJEACJEACNUeABqbmQsYGkwAJkAAJkAAJ0MAwB0iABEiABEiABGqOAA1MzYWs+gafOnVKHnroIU+hrVu3yuTkpDQ2NlZfURkF04YjR45IV1cXtC4j7lXn1atXpa+vTy5cuCA7d+6UkZERaWhocE4ZHR2V48ePy+zsrHR0dCTSxnKVXLp0SXp7e+VXfuVX1rWz2oYZ3StXrjhS9vXafPRvQeNlztNz\/PLJfT3Xrl1zYmHO0f\/av6Nz0otjkDwNUsatXdTPoP25cjPZvHmzTE1NSVtbW7UpzfMLQoAGpiCBti+zXOep5dwDOQJRlE6\/2nb8+7\/\/u+zZs0f2798vQ0NDjpz5N\/1\/uwPVwXR4eFief\/75zHSqUQ2MXsujjz4qPT09noOD28DYfNx\/y4KBUWO5ffv2REylV566eUbJ5aJ+BssZGPdnsNrPO8\/PPwEamPzH+KYr9Otwzbfm5eVl+KAdpdOvNlReswLugcTMPkQ1C9W2Me7zg8TUXGtTU5No7O0ZHjefOA2M+1orzdoYU3n69OnUZsW8eEbJ5aJ+Bm0DY8\/02bENmmNxf1aoV3sEaGBqL2ZVt9iv8\/QaINzfwLVyu4MxyyxPP\/20\/O3f\/q3o4KKH\/S1ef7cHQp3hueuuu+TP\/uzP1mm567KXs+y2PfLII86yhC55mDIXL150Zle86jbA3LMqW7ZsKc2y6FKFtse02z1bY67TaJl6dbnJa6bGZnz33XeXln7MdfvNdLnrsVm7TdXly5dLuh\/4wAdk3759TvPMTNJb3\/rW0vKYabc7LvrvRtdM3evvOpVv8+ns7HRMg90et7mxtW0zcv\/998tHPvKRdW3TusIsIT355JMyNjZWyi\/D7xd\/8RedXLAHQ6+ZNnP9Xn8zzE377TzR5c1PfvKTznXfe++9njxbW1udJVkto4dZnvVibdoR5jNo5745354lraXPoJ+BsfsIwy1o32PH3s3Vi125vknbkZUl46o7+gII0MAUIMjuS\/TrPE2HoeV1APMaAO0BUgch94Br12U6gnLT5WZAtJdybA0zGJvB1BikSmHz64TsazcDkmrpktKDDz5Ymn349Kc\/Xdr\/8tJLL3nuGTIdoV+naZafVF\/3rpg9JnbbzUCk\/6ZGyOv63B26mSExBsZPd3Bw0DEOur8niIFRXWOwlN+dd97pDNi33XbbTYbTL+7mesx+Frtut\/l79dVX1+3pKbcHxs\/AfPCDH3Q07EGp3N4l9yyPtsm9B8puh17Pn\/\/5nwcyML\/6q78q\/\/Iv\/7IuNf1mE4J+Bst9xox2rX0GveLjnoHxMot+fY+fgXn\/+9\/v+5ky7Pz6JpqYSj1sNv5OA5ONOCTaikrr737fHL1maExn5DUQaydhdyKm07A3hrrLeA1E+m9\/\/Md\/XOqM3LMk9rcm0x6\/gcP+Bq77KMyeGKOvpuMv\/uIv5JlnnvHd\/2I0zDXbMyG6CdjvdzUa7iUq8296DdoWex+O21AaI+RlYIyuaZuZITIDdLllQXsmxMzkKGObjz3LYGaU7OuxY2qbHzUwJhbuQcqemVJulTbxmtkuewnJz5DodXttHnbPwqmJMjN3hr35N2VgX7fOxpRbQrJnDM1nzO+zFPUz6DVTUWufwXKGy28jb7m+p5KB8dvH5hXLvCwdJzqgpFgZDUyK8NOqulzn6fXNw2sa1pTzMgx+sxz2gBKkjL03w\/4G7jYBunfDaFfaj2APeL\/+678ujz\/+eGmANedqXSdOnHBmHuy7ktyzRG7TZjrKF154obSkoIOeX6dos9NcMMsQ9l1Zdhn3gG+Mkn39fgN6UANjZm3sWRflbWah1IzcfvvtN22Gdg+sumzkdQeRPbCb2RNjyKIYGK3X\/kZvjGDQ5RvDXcubO87OnTt30+yb23zbPL3MSrllLJuVVx9QafbQPZtWa59BPwPjdQdkkL6n3BKSu6\/z+oLgNYOZ1N2YaY0BeamXBiYvkQxxHe5B3l5r9tvHoAbB6xuw13RwEHMSpExQA+O16dRvBsbdIdodmtug+C0RuWdglIu5Ht2f8\/Wvf33d7A3awNjXX62BOXjwoBw+fLi0lGX42KYsawbGNgv6Mah067t7z893vvOddUuIL7\/8svNp0pz\/6le\/us5YBt3EG9TAmDz1+wy6DZrezu82TLX2GQzyeAJ3Hpfre7z25Lk\/\/177tYyBpoEJMXhkrCgNTMYCkkRzvGYp7A7UvT5sBnL3MoF2ppU6T8QSUrk7hSrNwLi\/\/drftNzPPLGXQry+cdsbKd0bDv3+ZtruZlnNElI5A+O3ydjOM7fBeu6550p7fsx1mH\/T3EhrCUmfA+OVb+64Vfr27Dax7iVKZRN0f5PuUfHKubAGRuv0+gzaJt7MMro3HdfaZzCIgXGbtHJ9j1f\/5PcFxuY5Pj4uAwMDzp13fP5MEiNP\/HXQwMTPNPOKfoO8+Xf3XoBy09yVOk9dDol7E2+1BsaeabFnnOyBLcjmYdukuAdFr7uHUJt4gxgYv7vD7IHT6Oi+FfcdXe6cSXoTr3vw1nZ73Yljm49yH0Q7J702dLo3oHvt4zF1uffJ6L9HMTC2uS53J5nfElKYpRSbTZSN9NV8BoMYGL9N\/druoDcH+G0E9jKo7lwptwSZ+Q6+QA2kgSlQsM2l+hkYexA2g4O5G8d86O2NnXrnThAD4571qPY26mo6T22LexOxvefEsPGbQdGBxW+Tr+l03RsRvTbJugdgE5uot1GbvTqVnnXj9ZBCv9uZ1cj4DeTumLqNg99t1PbsSJjbqI2BsQc2Ly17Y3G5j7Z7g7TOpLg3QOuMj9dnxTY\/yvPd73636NKbbVqjGhivz6BtKPWajx496twxp4dy0f1a7mWzIO2u5lEG1XwGgxgYvTb7s+DeVG4eRGmX8epXKt2CXimPCzg81NQl59LAmKTdvXv3uqd1en3rqqlosbGZJuC1N0YbzDsb8GHzWmrB18oaSIAE0iSQOwPj97RO7eD0Dgt9GJYe5v\/53o000y8\/dZd7kigNDD7O7n0h+BpZAwmQQNoEcmVg7IFC7yTQaUbzEj6dfTl\/\/nzptljt8HTtOqkXCaYdaNaPI+C3p8bUSAODY29vfE3iHV64K6EyCZBAWAK5MjALCwvO9eudF\/oMCtvAqGHRw147tX\/3A6fP2tAfHiRAAiRAAiRQiwT0Seb6k7cjVwbGBMdsIHQbGHvGRWdk5ufnS4bGK7BqXPQ2u7m5ubzFnddDAiRAAiRQEALt7e2it43nzcTQwAwN+aawWRrQwLe0tBQk1ZO5TDWFExMTzoeKbONlTrbx8rTVyBbDllwxXFXVsM3j+50KZWDCLiEZA5PHwOM+LsGUyTYYpyilyDYKtWDnkG0wTmFLkWtYYsHL55ltYQyMe8koyCbePAc+ePpjSpIthquqki3Z4ghglJmzGK557w8KY2Ci3EbNDxU\/VDgCOGXmLdniCGCUmbMYrjQwOK4wZa9NvFpZ2AfZ8UMFC5GzgXp6elp6enqc29l5xEeAbONj6VYiWwxbcsVwpYHBcc28Mg0MLkQ\/+MEP5Nvf\/rbceuutUl9fj6uogMpkiws62WLYkiuGKw0MjmvmlWlgcCFih0W2OAI4ZeYthi25YrjSwOC4Zl6ZBgYXInZYZIsjgFNm3mLYkiuGKw0MjmvmlWlgcCFih0W2OAI4ZeYthi25YrjSwOC4Zl6ZBgYXInZYZIsjgFMuUt4m+RqV69evy\/\/8z\/\/ILbfcwj1xAdM36OsB8jyO5fI26oDxr1gsz4GvePHgAkUaCMAob5InWxzxorDla1RwORSXctDXA+R5HKOBKZNNeQ58XB+iqDpFGQii8qnmPLKthl75c4vClq9RweVQHMphXg+Q53GMBoYGJo7PU2iNogwEocHEcALZxgDRR6IobPM86OGyIznlMPEJUza5K4inJhoYGph4MimkSlEGgpBYYilOtrFg9BQpCts8D3q47EhOOUx8wpRN7griqYkGhgYmnkwKqVKUgSAklliKk20sGGlg9uyRWn2Rrb7r7vjx4zfFcOvWrTI5OSmNjY2BkiTIO\/P8hPSJ8A888IA8\/PDD0tbWFqi+oIXCmJIwZYPWn5VyNDA0MKnkIgdZHHayJdtqCeRl0PN7rUy1fIKcTwMThFJ1ZWhgaGCqy6CIZ3OQjQguwGlkGwBSxCJFYZtnA2Ne7GtSQGdkLl68KHv27CllhZl5smdgrl27JsPDw3L69GmnnD07ZYzShQsXRGd5nnzySRkbG3PKbt68WaampkRve\/Y6390ePf+d73yndHV1OfXoO\/z0ML\/r\/4eJT5iyET8WqZ1GA0MDk0ryFWUgSAMu2eKoF4VtXgY9rxkYNQy9vb1y9OhR6ejoEGMg1HDoUo8ahvPnz8vIyIh8+tOfdl42q+ZBzYweQ0NDjoF48MEH1xmT3bt3O3p6vr6c8v7771+3hKTnX7lyxdFVo2POV027Par97LPPOuX0ePTRR52X3trLUGHiE6Ys7pODUaaBoYHBZFYF1aIMBGnAJVsc9aKw9Rr0Dh8WOXkSx7Za5akpkc7O9Sp+BmZwcNCZIfHam2IbCGNg7r33Xunr63PMi5oUMxujpuWtb32rPPbYY3Ls2LF1e2vsJSQz+2JMjjl\/27Ztcvfdd4vdHj3v8OHDcvDgQXn11VdlenpaDhw4IA0NDaWLC2NKwpStNgZJn08DQwOTdM459RVlIEgDLtniqBeFrdeg19ubbQNz5kxwA+M2HO5Nvzt37lw3A2MMjM6c2MeRI0fk9ttvd2Zn3JuDbQOjJsc2QKphlqfUwHi1Z\/v27fLSSy851dnLR\/p7GFMSpizuk4NRpoGhgcFkFmdgUuFKc4jFXmQDMz+PZVutemvrzQp+MzC2YdAB3jYgfjMwfncU6RJUtTMw7vO1DX\/3d3\/nXJB7+YgG5kacaWBoYKrtNyKdX5SBIBKcKk8i2yoBljm9KGzz8q09rIHRZRrdaKtHuT0w9j6aO++8c93sitlDo8tCH\/\/4x0u3UZfbA+M2MKbdt912m9MOe\/mIBqagBsaeJty\/f7+znlnuyMuHGNedR1cuykAQnVD0M8k2OrtKZxaFbV76viAGxr67SO8Y0ue2fOELX3D2tJw4caK0idd9F5IuH5mlHWNodJOuedaMMUPPP\/98xbuQvGZwyj2DJkx8wpStlP9Z+3thZmDUFX\/uc58rrVMGeUBRngOfdiIWZSBIgzPZ4qgXhS37vrUcCjJOILKt0jNkwsQnTFnEtSA1C2Fg7B3ftmP22t1tw85z4JFJFUS7KANBEBZxlyHbuIne0CsK26L3fWbmRiMf5sm9cWSemc356Ec\/etPmXaMfJj5hysbR\/iQ1Cm1gvKbtaGCSSb+iDATJ0FxfC9niqBeFbZ4HPVx2JKccJj5hyiZ3BfHUVAgDY6YCNZDGTevUoD4lUZ+Q6PeeChP4\/v5+aW9vLxFvbm4W\/eERnYAOBIuLi7Jp06abNqhFV+WZSoBscXlQFLZf+9rXpLu7u2bfhYTLgGwom7HpmWeecZ5L4z60b9UfPRYWFmRgYCCXsSyMgXFvwHrkkUfkueee832YkQbeJIk7OfS2Nv1w84hOYGVlRZaXl6WpqUnq6uqiC\/HMmwiQLS4pisJWn3fysY99LPZBb17m5ayclWmZLgWpUzrloBzEBS2HymZsevzxx51Nw+5jZmbGeQCefdTqiznLha8wBsYNwX3vvxckkyTj4+PS0tJSKsIZmOp7BDWUS0tLzkxWfX199YJUKBEgW1wyFIWt9n1\/8Ad\/EKuBOSyH5ZAc8g3OGTkjambiOuz3E6mm\/SbqSptk\/doQ9by4rsnomLFJjcq73\/3um+TtGZi5uTmZmJiINZZxX09UvcIaGPt9F+577N1JkkfnGjVh4jqvKHsJ4uIVRodsw9AKV7YobOPeN3FSTkqv9Dqw98pe2S7bpVXWnjynszH6dz3iMjHGvOzatWvdSxHNnahal9+D6cplRNYMTJCxKe5YhvvEYEsXxsCYF2zps1\/cL\/PyQ5znwGPTqrJ6UQaCyiTiL0G28TM1ikVhG2ffp8tGd8gdDkKdgfFaLrINzovyYsncRI2k+wWNqmNMzZ\/8yZ\/IF7\/4xXVvijaP+jevCjDPCbPfFP3jH\/9Ybr31VvnKV75SesO03\/7JqO0Oel6Y+IQpG7T+rJQrjIEp9xAiGpjk07EoA0HyZPmeKSTzouRtnIOeWTrS5SGdYfE7dsgOZ3\/MlEw5szTVHHZ\/7zVL4fWiRX2xoj5mwzY\/2gb7TdGcgakmKvGfWxgDEwVdnB\/iKPXn+ZyiDARpxJBscdSLwjbOvs8Yk0rLQ2YWRs2Lmpg4DveNGMbMlDMi9t+0De43RUdZeorjWmyNMPEJUzbudqL1aGDKEM5z4NGJVUm\/KANBJQ6Iv5MtguqaZlHYxtn3BTUwZqlJ98boMlLchz2zoktGthFxGx19pYA+YkMP+3lhnIGJOyrV6dHA0MBUl0ERzy7KQBART1WnkW1V+MqeXBS2cRoY3byrsyuVloZ0+UjNTqWlpiDR1fafO3du3fvuzLLS7t27RV\/AaAyM2f+i+yP1mSruGRgamCDE0ylDA0MDk0rmFWUgSAMu2eKoF4VtnAYm6NKQmanx2+gbJqpeL3HUa3rwwQedmRV7BsZtYPSGjyeeeIIzMGGAp1SWBoYGJpXUK8pAkAZcssVRLwrbOA2MRmODbHCC4jcLYzb66vKR7pUxt1hXE0n3c2DMspDeOWRmY8ybol944QV56KGHnOr0Iadf\/\/rXRWdq1NzYMzDu83gXUjURqv5cGhgamOqzKIJCUQaCCGiqPoVsq0boK1AUtnEbGLM8pGB1iahHehyTovtezsm50nNgKi0z4SJbW8ph4hOmbG1REKGBoYFJJWeLMhCkAZdscdSLwhYx6NnPenFHSM2M3n3EVwoEy90w8QlTNljt2SlFA0MDk0o2FmUgSAMu2eKoF4UtatDTGRd98q7OyJiD70IKn69h4hOmbPiWpHsGDQwNTCoZWJSBIA24ZIujXhS2eR70cNmRnHKY+IQpm9wVxFMTDQwNTDyZFFKlKANBSCyxFCfbWDB6ihSFLWzQm58XOXtWxH5TcmenyEG+jTpM1oaJT5iyYdqQhbI0MDQwqeRhUQaCNOCSLY56UdhCBr3Dh0UO+b+NWs6cEVEzE8PhvgPJSNp3IgWtJsiLf4NqxVUuTHzClI2rfUnp0MDQwCSVa+vqKcpAkAZcssVRLwrb2Ae9kyf1pUJrgdm7V2T7dpHWtbdRO7Mx+nc9YjIxXs+BiZoVNDBRyeHPo4GhgcFnmUcNRRkI0oBLtjjqRWEbq4HRZaM71t5G7czAeC0X2QbnxRdvmJuIoaxkYPS1Avp8l5\/7uZ9z3kqth\/3SR3P9OmPT2dkpr7\/+uoyMjEhDQ0PEFsV7Wpj4hCkbbyvxajQwNDD4LKOBSZRxUQbZRKH+pLKisI110DNLR7o8pDMsfseOHWv7Y\/QdRDpLU8URxMDoW6Y\/+tGPOm+gHh0dlStXrjgm5fLly6U3UG\/dulWGh4edltDAVBEQ0Kk0MDQwoNQqL1uUgSANuGSLo14UtrEaGGNMKi0PmVkYNS8\/eZFi1Ej67YHZuXNnyaTYb5nW63322Wedv124cKH0\/zrjYv+NMzBRI4I5jwaGBgaTWRVUizIQpAGXbHHUi8I2FQNjlpp0b4wuI1VxBJmBsV8RYJuU5557Ts6fP1+acaGBqSIQ4FNpYGhgwCnmLV+UgSANuGSLo14UtrEaGN28q7MrlZaGdPlIZ2sqLTUFCG81BoYzMAEAZ6RIoQyM7iY3L+wyU4nlpgRj\/RBnJOBZaUZRBoI0eJMtjnpR2Mba9wVdGjJLTX4bfUOEtRoDoy9s7Ovrk6GhIeEemBDQUyhaGAOjH0jdqDU5OensJNeNWbrDXJPU74j1Q5xCcLNcZVEGgjRiQLY46kVhG3vft2HtbdS+szBmo68uH+leGXOLdcRQ+u2BUTm928j9lmn3MpG5fi2vb6f+7\/\/+bzlw4ADvQooYD9RphTEw7nv5g9zbH\/uHGBXFGtQtykCQRmjIFke9KGxj7\/vM8pCGRpeIenrWTIruezl37sZzYCotM+FCW1PKYeITpmxNQZACvY3aawZm27Ztzi10nIFJPm2LMhAkT1aEbHHUi8IWMujZz3pxh0jNjN59xFcKBEreMPEJUzZQ5RkqVJgZGGWuDy\/Se\/\/1fn\/7oUWVDEx\/f7+0t7eXijU3N4v+8IhOQAeCxcVF2bRpU2amZaNfTbbOJFtcPIrC9mtf+5p0d3cH6idD0dYZF33yrs7ImIPvQgqFUAsbU\/LMM89IR0fHTedr36o\/eiwsLMjAwED8sQzd6vhPKIyB0SWjz33uc84emMbGRmc\/jB5B9sC4sff09Dgfbh7RCaysrMjy8rI0NTVJXV1ddCGeeRMBssUlRVHY6p04H\/vYx8T95Q1HlsphCBhT8vjjjzsbjd3HzMyMTNsvzHQ9aThMXVkuWwgDo7vKddOuvWSkszH2g4y8gmRc7vj4uLS0tJSKcAam+pTWmCwtLTkzWfX19dULUqFEgGxxyVAUtvo0Wt28Ojc3h4NJ5aoI6KrAkSNHPFcD7BkYjeHExARnYKqineLJ1RqYIMtNKV5eTVZdlL0EaQSHbHHUi8RWTYz+JHEo1+985zvyC7\/wC7Jx48Ykqqz5OrZs2SL6U+ngHphKhGrg715LSObdF37Pgslz4NMOWZEGgqRZky2OONli2JIrhquq5nkcK8QSkkkN3fdy\/Phx51c+yA73gQmizA4rCKVoZcg2GrcgZ5FtEErhy5BreGZBz6CBCUoqZ+XyHPi0Q8UOCxcBsiVbHAGMMnMWw5UzMDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45p5ZRoYXIjYYZEtjgBOmXmLYUuuGK40MDiumVemgcGFiB0W2eII4JSZtxi25IrhSgOD45qY8rVr12R4eFhOnz69rs7NmzfL1NSUtLW1ebaFBgYXInZYZIsjgFNm3mLYkiuGKw0MjmuqyqOjo079Q0NDvu2ggcGFiB0W2eII4JSZtxi25IrhSgOD45qashoTNTCTk5PS2NhIA5NCJNhh4aCTLdniCGCUmbMYrjQwOK6pKJvlpG3btklXV1fZNpgZmP7+fmlvby+VbW5uFv3hEZ2AdliLi4uyadMmaWhoiC7EM28iQLa4pCBbDFtyjZer9q36o8fCwoIMDAzI7OysdHR0xFtRymobVldXV1NuQ6LVB519sZ2ru4E9PT3S3d2daLvzVtnKyoosLy9LU1OT1NXV5e3yUr0essXhJ1sMW3KNl+vMzIxMT0+vE6WBiZdxKmpB9r6YhpkZmPHxcWlpaeEMTIwR05mwpaUlZyarvr4+RmVKkS0uB8gWw5Zc4+Vqz8DMzc3JxMQEZ2DiRZy8mlk+2r17d6CpNG7ixcWIa95kiyOAU2beYtiSK4arvZLAGRgc40SUL126JIODgzI2NuZ767TdEBoYXFjYYZEtjgBOmXmLYUuuGK40MDiuiSurIXn22WdlZGQk0MZRGhhciNhhkS2OAE6ZeYthS64YrjQwOK6JK586dUrOnz9PA5M4+ZsrZIeFCwLZki2OAEaZOYvhSgOD45p5Zc7A4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMXSkJkfAAAgAElEQVSwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxTVzZGBKteOvWrTI5OSmNjY2+7aCBwYWIHRbZ4gjglJm3GLbkiuFKA4PjmqjypUuXpLe3V44ePSodHR1y6tQpOX\/+vIyMjEhDQ4NnW2hgcCFih0W2OAI4ZeYthi25YrjSwOC4JqqshmV+fl6GhoYC10sDExhV6ILssEIjC3wC2QZGFbog2YZGFugEcg2EKVKhPI9jG1ZXV1cjUamhk65duybDw8Oybds26erqCtzyPAc+MARQQXZYILAiQrZkiyOAUWbOYrhyBgbHNTFlY2Duu+8+OXHihFy4cCHUHpj+\/n5pb28vtbe5uVn0h0d0AtphLS4uyqZNm3yX8KKrF\/tMssXFn2wxbMk1Xq7at+qPHgsLCzIwMCCzs7PO9ok8HYWagXn55ZdLG3dHR0flypUrgfbAuAPe09Mj3d3decqDxK9lZWVFlpeXpampSerq6hKvP88Vki0uumSLYUuu8XKdmZmR6enpdaI0MPEyTkzNawlJN\/UODg7K2NiYtLW1ebbFLCGNj49LS0sLZ2BijJjGZGlpyZnJqq+vj1GZUmSLywGyxbAl13i52jMwc3NzMjExwRmYeBEnq6YzLq2traU9MGpgHnvsMTl27JjvrdTcA4OLEde8yRZHAKfMvMWwJVcMV1XN8zhWiCUkE0Q1MebZL\/r\/epS7KynPgcd9XIIps8MKxilKKbKNQi3YOWQbjFPYUuQalljw8nkexwpjYGwnqv+\/c+fOsvtf8u5cg6c\/piQ7LAxXVSVbssURwCgzZzFc8z6OFcrAhE2RPDvXsCziLs8OK26iN\/TIlmxxBDDKzFkMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxzbwyDQwuROywyBZHAKfMvMWwJVcMVxoYHNfMK9PA4ELEDotscQRwysxbDFtyxXClgcFxLSnrY6S\/8Y1viP436PGzP\/uz8q53vUs2btwY9JTQ5WhgQiMLfAI7rMCoQhck29DIAp9AtoFRhSpIrqFwhSqc53EsE7dRX716Vfbt2yf6OOkgj5XXZH\/Pe94jDz\/8MA1MqFTOTmF2WLhYkC3Z4ghglJmzGK6cgcFxLSmrgdEXTf3hH\/5hIEOi5T\/\/+c87poczMAkECFAFOywA1J9Iki3Z4ghglJmzGK40MDiuJeXXXntNvvzlL8uuXbvkp37qpyrWGLZ8RUGfAnmeeovKJK7z2GHFRfJmHbIlWxwBjDJzFsOVBgbHNbTy9evXZXV1Vd74xjeGPjfKCTQwUagFO4cdVjBOUUqRbRRqwc4h22CcwpYi17DEgpfP8ziWiT0wXqFQs\/L66687hsUc+lpw3SOzY8eO4NGromSeA18FllhOZYcVC0ZPEbIlWxwBjDJzFsOVMzA4rr7KX\/ziF+Whhx6SH\/7whzeVefrpp+Wee+5JpFU0MDjM7LDIFkcAp8y8xbAlVwxXGhgcV09lvRNpeHhY7rrrLtmzZ4\/U1dWVyv3rv\/6rY2o4A5NwUADVscMCQP2JJNmSLY4ARpk5i+FKA4Pj6qmsS0ZPPvmkvP3tb5ff+I3fWFeGe2ASDgawOnZYOLhkS7Y4Ahhl5iyGKw0Mjquv8iuvvCK6VNTT0yMNDQ2lcv\/0T\/8kjY2NXEJKISZxV8kOK26iN\/TIlmxxBDDKzFkMVxoYHFdf5S996Uty8OBBZxOv++AemBQCAqiSHRYA6k8kyZZscQQwysxZDFcaGBxXT2WzB2br1q2ye\/fudTMwuqlW\/x5lD4zRPX36dKne\/fv3y9DQkO8VchMvLvjssMgWRwCnzLzFsCVXDFcaGBxXT2XUHhh9eu8DDzzgvH6gra0t0FXRwATCFKkQO6xI2AKdRLaBMEUqRLaRsFU8iVwrIopcIM\/jWOaeA6MG5lvf+pZ85jOfEZ0hsffA\/Nu\/\/Zvze5TbqC9duiSPPfaYHDt2zNlHE+TIc+CDXD+yDDssHF2yJVscAYwycxbDlTMwOK5lZ2A++clPev496h4YNSOjo6MyOTlJA5NwTL2qY4eFCwLZki2OAEaZOYvhSgOD4+qr\/Pzzz8vy8rLcd999smHDhlK5ap4Dc+rUKefheObQPTaVzIyZgenv75f29vbSuc3NzaI\/PKIT0A5rcXFRNm3atG6WLboizzQEyBaXC2SLYUuu8XLVvlV\/9FhYWJCBgQHnhckdHR3xVpSyWiaWkH784x87m3N\/5md+JhCOsOVVVGdfrly5IiMjI86A6f7dq2JjYNx\/09u7u7u7A7WVhbwJrKysOCa1qalp3cMKyat6AmRbPUM\/BbLFsCXXeLnOzMzI9PT0OlEamHgZl9R0g+3nP\/952bdvn2zcuLFiLWHLewnqnpjBwUEZGxvz3dRrDMz4+Li0tLRwBqZiZIIXUMO6tLTkzGTp+614xEeAbONj6VYiWwxbco2Xqz0Do+8QnJiY4AxMvIhvqKkh+dSnPiUf+tCH5E1velPFal566SXRh9rp0k4Qw+NnYCpt6uUm3oqhiFyAa96R0VU8kWwrIopcgGwjoyt7IrliuKpqnsexTCwhqYHp6+uTCxcuBI7i7\/\/+78sjjzwSyMCYZ8Bs27ZNurq6nOUqfd\/S5s2b+RyYwMTjLcgOK16ethrZki2OAEaZOYvhSgOD45qosvtBdjt37izth\/FrSJ6da6LwPSpjh4WLANmSLY4ARpk5i+FKA4PjmnllGhhciNhhkS2OAE6ZeYthS64YrjQwOK6ZV6aBwYWIHRbZ4gjglJm3GLbkiuFKA4PjmnllGhhciNhhkS2OAE6ZeYthS64YrjQwOK6ZV6aBwYWIHRbZ4gjglJm3GLbkiuFKA4Pj6qusT8h96qmnRDfa\/tIv\/ZLzRN6gD7mLs7k0MHHSXK\/FDotscQRwysxbDFtyxXClgcFx9VV+5ZVX5NVXX5V3vOMd8r3vfc95yN3P\/\/zPy2\/+5m8m+tRWGhhc8NlhkS2OAE6ZeYthS64YrjQwOK6BlPX254sXL8pXv\/pV+Y\/\/+A95+OGH5Zd\/+ZfXvSMpkFCEQjQwEaAFPIUdVkBQEYqRbQRoAU8h24CgQhYj15DAQhTP8ziWiQfZuWOhD7b7yle+Ip\/97Gflm9\/8pvMY\/3vuuUfe8573OI+f12PXrl3w2Zg8Bz5E\/kOKssOCYHVEyZZscQQwysxZDFfOwOC4+io\/88wz8pnPfEb2798vnZ2dzhuL7bdS\/+d\/\/qecOXNGent7oW8ypoHBBZ8dFtniCOCUmbcYtuSK4UoDg+Pqq6zvOfr+97\/vbOL1Oq5fvy6f+MQn5F3vepfzagDUQQODIstZAhxZsiVbJAGMNg0MhisNDI6rr\/L\/\/u\/\/yo9+9CPf2RXd2KtvkdYlpfe9732wFtLAwNBymQOHlmzJFkgAI00Dg+FKA4PjmnllGhhciNhhkS2OAE6ZeYthS64YrjQwOK6ZV6aBwYWIHRbZ4gjglJm3GLbkiuFKA4PjmnllGhhciNhhkS2OAE6ZeYthS64YrjQwOK6ZV6aBwYWIHRbZ4gjglJm3GLbkiuFKA4PjmnllGhhciNhhkS2OAE6ZeYthS64YrjQwOK6ZV6aBwYWIHRbZ4gjglJm3GLbkiuFKA4PjmnllGhhciNhhkS2OAE6ZeYthS64YrjQwOK6pKV+6dEkGBwedZ8m0tbX5toMGBhcidlhkiyOAU2beYtiSK4YrDQyOayrK+nLI4eFhef7552VqaooGJpUo8GmxSOwcDHB0yRbDllwxXGlgcFxTUdZZldHRUaduzsCkEgKnUnZYOPZkS7Y4Ahhl5iyGKw0MjmviyvqW68OHD8vu3bsdExPUwPT390t7e3upvc3NzaI\/PKIT0A5rcXHReVFnQ0NDdCGeeRMBssUlBdli2JJrvFy1b9UfPRYWFmRgYEBmZ2elo6Mj3opSVtuwurq6mnIbEqv+1KlTTl133313qD0w7gb29PRId3d3Yu3OY0UrKyuyvLwsTU1NUldXl8dLTO2ayBaHnmwxbMk1Xq4zMzMyPT29TpQGJl7Giarpxl0N6IEDB+Ty5cuhDMz4+Li0tLRwBibGiOlepKWlJWcmq76+PkZlSpEtLgfIFsOWXOPlas\/AzM3NycTEBGdg4kWcrJouGW3fvt2ZQuNdSMmy96qNa964GJAt2eIIYJSZsxiuqprnu2kLsYSke1\/6+vrkwoULN2VJuWm1PAce93EJpswOKxinKKXINgq1YOeQbTBOYUuRa1hiwcvneRwrhIFxh5ozMMGTH1WSHRaKLO\/wwpElWxRb9gcospyBwZFNSZkGJiXwVrXssHAxIFuyxRHAKDNnMVy5hITjmnnlPE+9pQ2fHRYuAmRLtjgCGGXmLIYrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa+aVaWBwIWKHRbY4Ajhl5i2GLbliuNLA4LhmXpkGBhcidlhkiyOAU2beYtiSK4YrDQyOa6LK165dk+HhYTl9+rRT7\/79+2VoaKhsG2hgcCFih0W2OAI4ZeYthi25YrjSwOC4Jqo8Ojrq1Kem5erVq9LX1ye7du2Srq4u33bQwOBCxA6LbHEEcMrMWwxbcsVwpYHBcU1V2TY0fg2hgcGFiB0W2eII4JSZtxi25IrhSgOD45qaspmB0dmYjo4OzsCkEAl2WDjoZEu2OAIYZeYshisNDI5rKso683L8+HHZuXOnjIyMSENDQ0UD09\/fL+3t7aVyzc3Noj88ohPQDmtxcVE2bdpUNgbRayjumWSLiz3ZYtiSa7xctW\/VHz0WFhZkYGBAZmdny35hj7cFyahtWF1dXU2mqmzVokbmypUrZU2MWUJyt7ynp0e6u7uzdUE11pqVlRVZXl6WpqYmqaurq7HWZ7u5ZIuLD9li2JJrvFxnZmZkenp6nSgNTLyMU1W7dOmSDA4OytjYmLS1tXm2xRiY8fFxaWlp4QxMjBHTu8KWlpacmaz6+voYlSlFtrgcIFsMW3KNl6s9AzM3NycTExOcgYkXcbpqak50FmZyclIaGxvLGpg8Otd06YtwzRsXAbIlWxwBjDJzFsNVVfN8M0phlpDsu47MM2E2b95c9lkweQ487uMSTJkdVjBOUUqRbRRqwc4h22CcwpYi17DEgpfP8zhWGAPjfpBdmE28nIEJ\/mEJWpIdVlBS4cuRbXhmQc8g26CkwpUj13C8wpSmgQlDK0dl8xz4tMPEDgsXAbIlWxwBjDJzFsOVS0g4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeWUaGFyI2GGRLY4ATpl5i2FLrhiuNDA4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeWUaGFyI2GGRLY4ATpl5i2FLrhiuNDA4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeWUaGFyI2GGRLY4ATpl5i2FLrhiuNDA4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeWUaGFyI2GGRLY4ATpl5i2FLrhiuNDA4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeWUaGFyI2GGRLY4ATpl5i2FLrhiuNDA4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeWUaGFyI2GGRLY4ATpl5i2FLrhiuNDA4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeWUaGFyI2GGRLY4ATpl5i2FLrhiuNDA4rplXpoHBhYgdFtniCOCUmbcYtuSK4UoDg+OaeeVaMTDzMi9n5axMy3SJaad0ykE5mFnG7LBwoSFbssURwCgzZzFcaWBwXDOvXAsG5rAclkNyyJflGTkjamaydrDDwkWEbMkWRwCjzJzFcKWBwXFNVPnq1avS19cnFy5ccOrduXOnjIyMSENDg287sm5gTspJ6ZVep\/17Za9sl+3SKq3O7zobo3\/XI4smhh0WLv3JlmxxBDDKzFkMVxoYHNfElK9duybDw8Oybds26erqEvP75s2bZWhoqCYNjC4b3SF3OG3XGRiv5SLb4LwoL5bMTWLgy1TEDgsXBbIlWxwBjDJzFsOVBgbHNVXlU6dOyfnz58vOwmR5BsYsHenykM6w+B07ZIezP2ZKppxZmqwc7LBwkSBbssURwCgzZzFcaWBwXFNVDmNg+vv7pb29vdTe5uZm0Z80j\/fVv88xJl\/+wZfL7nH5q41\/JfdvvF9+9\/rvytPXn06zyevq1g5rcXFRNm3aVHYZLzMNrqGGkC0uWGSLYUuu8XLVvlV\/9FhYWJCBgQGZnZ2Vjo6OeCtKWW3D6urqasptSLx6sx9m165dzpKS32FmYNx\/7+npke7u7sTbbVe4+9bdMlc\/J7PfnpWOH3TI5csbZc+eW2V29tuyZcv1UtHLGy\/Lr73t12TL9S3yz\/\/vn1Nts135ysqKLC8vS1NTk9TV1WWmXXloCNnioki2GLbkGi\/XmZkZmZ6+cVeqqtPAxMs4FTWz\/0UrD7qJd3x8XFpaWkrtzcIMzL6N++SzGz8rJ66fkP\/zrd+Te+\/d6JiYAweuy8c\/fsPA6CyNztboUpPO1mTl0DgsLS05M1n19fVZaVYu2kG2uDCSLYYtucbL1Z6BmZubk4mJCRqYeBEnrxbGvGjrsrwHxmzQ7ZzfK\/M7pmR+XqSzU+SMazuM2QPjt9E3+Sis1cg1bxx5siVbHAGMMnMWwzXr41i1V12YJaSgdx7ZQLNsYLSdG2TDWnN7p6T17F558cX16WA2+uqt1brR19xiXW3SxHE+O6w4KHprkC3Z4ghglJmzGK40MDiuiSqPjo7KlStXKi4b1ZKB6T15Vk7u3eE0WZeIeqTHMSl6i\/U5OVd6DkzW7kDiDAw29TkY4PiSLYYtuWK40sDguCam7H6Inal469atMjk5KY2NjZ5tyfIMzOHDIocOibQeOinzB9ceZuc+1MzordNZfKUAOyxc+pMt2eIIYJSZsxiuNDA4rplXzqqBOXlSpLdXpLVV5OBBkc69886Td3XDrjn4LqTMpxesgRwMYGi5dwuEljkLApvxvZzVXnVh9sBEAZVFA6Obde9YewCvTE2J7M3Os+lCIWaHFQpXqMJkGwpXqMJkGwpX4MLkGhhV6IJZHMdCX4TPCTQwZUhmLfBqXnbsEOeOIzUuamBq9WCHhYsc2ZItjgBGmTmL4aqqWRvH4rxSGpgaMTC2efG6XTrOpEhCix0WjjLZki2OAEaZOYvhSgOD45p55aw4VzUvuufl7Nm1fS\/u26UzD9KjgeywcFEjW7LFEcAoM2cxXGlgcFwzr5wVA1O64ygn5kUDzw4Ll\/5kS7Y4Ahhl5iyGKw0MjmvmlbNgYGzzontedPkoDwc7LFwUyZZscQQwysxZDFcaGBzXzCunbWDct0vX6h1HXoFmh4VLf7IlWxwBjDJzFsOVBgbHNfPKaRoY+3ZpfWCdPu8lTwc7LFw0yZZscQQwysxZDFcaGBzXzCunZWDsO45q+Vkv5QLMDguX\/mRLtjgCGGXmLIYrDQyOa+aV0zAweXrWCw1MOinOwQDHnWwxbMkVw5UGBsc188pJG5g83i7tF2R2WLj0J1uyxRHAKDNnMVxpYHBcM6+ctIGx7zg6c2btmS95Pdhh4SJLtmSLI4BRZs5iuNLA4LhmXjlJA5PX26U5A5N8mnMwwDEnWwxbcsVwpYHBcc28clIGJs+3S9PAJJ\/mHAxwzMkWw5ZcMVxpYHBcM6+chIHJy9ulwwaTHVZYYsHLk21wVmFLkm1YYsHKk2swTlFKJTGORWlXHOfwZY5lKKIDb99xlMdnvZRLUHZYcXx8vTXIlmxxBDDKzFkMV87A4Limpjw6Oiqtra3S1dVVtg1IA2Oblw9\/WOQv\/zI1HKlUzA4Lh51syRZHAKPMnMVwpYHBcU1FWc3L8ePH5ciRI6kZmCLdLu0XZHZYuPQnW7LFEcAoM2cxXGlgcFwTVb569ar09fXJLbfc4tT73ve+NzUDU6TbpWlgEk1zpzIOBjjmZIthS64YrjQwOK6JKquBee2112Tz5s0yPDws27ZtC2xg+vv7pb29vdTe5uZm0Z8ox+hoveh+ly1brsuJE9dz83bpsCy0w1pcXJRNmzZJQ0ND2NNZvgwBssWlB9li2JJrvFy1b9UfPRYWFmRgYEBmZ2elo6Mj3opSVivcJt5r166FNjDuGPX09Eh3d3fo0P3N37xZBgaaHPPS3\/+a\/NZvfS+0Rl5OWFlZkeXlZWlqapK6urq8XFYmroNscWEgWwxbco2X68zMjExPT68TpYGJl3EqalEMzPj4uLS0tFQ1A6P7Xt75znpHQ2dghoZ+kMr1Z6VSjcPS0pIzk1Vfv8aFRzwEyDYejl4qZIthS67xcrVnYObm5mRiYoIzMPEiTkctioGp1rkW4e3SYaPJNe+wxIKXJ9vgrMKWJNuwxIKVJ9dgnKKUQt5NG6U9cZ7DJaQyNOMIfFHeLh02KdlhhSUWvDzZBmcVtiTZhiUWrDy5BuMUpVQc41iUepM4hwYGaGB4u7Q\/XHZYuI832ZItjgBGmTmL4aqqNDA4tokrJ7mExNulaWAST3DeRg1FzoEWg5dcMVxpYHBcM69cjXOleSkfXnZYuPQnW7LFEcAoM2cxXGlgcFwzrxzVwBTx7dJhg8kOKyyx4OXJNjirsCXJNiyxYOXJNRinKKWijmNR6kr6nMLtgQkDOErgi\/p26TBctSw7rLDEgpcn2+CswpYk27DEgpUn12CcopSKMo5FqSeNc2hgylAPG\/giv106bPKywwpLLHh5sg3OKmxJsg1LLFh5cg3GKUqpsONYlDrSOocGJiYDY5uXvj6Rp59OK6S1US87LFycyJZscQQwysxZDFdVpYHBsc20ctDAq3k53Dsvcvas9Mj0jfcbdXaKHDyY6WtMq3HssHDkyZZscQQwysxZDFcaGBzXzCsHNTBibjnyu6IzZ6Swb230YcIOC5f+ZEu2OAIYZeYshisNDI5r5pWDGJj5wyel9VDv2rXs3SuyfbtIa+va7\/oyLb0lSQ+amHXxZoeFS3+yJVscAYwycxbDlQYGxzXzypUMzNmT89LZe8fadegbGr2Wi8w91VrmxRdvmJvMXz22geywcHzJlmxxBDDKzFkMVxoYHNfMK5czMLrvZfqOw3JQDq0tD+kMi9+xY4ezP0amptZmaXjwNmpgDnAwwMElWwxbcsVwpYHBcc28sp+BMXccTc3vkE45u255SP9mVpBKF2hmYdS8qInhQQMDzAEOBji4ZIthS64YrjQwOK6ZV\/YyMPbt0mfkhoGZb+109vKqV7lpu4t5up06G11G4kEDA8wBDgY4uGSLYUuuGK40MDiumVd2G5ib3i7d2es4lsOtU3Jofm1pSD2KboVZt1Kky0e6jFRpqSnzROJrIDus+Fi6lciWbHEEMMrMWQxXGhgc18wruw2Mfbe07tlV83JwvldOyl7HxKhp8Xzsi9kD47fRN\/Mk4m8gO6z4mRpFsiVbHAGMMnMWw5UGBsc188q2gfmHf+hwbjRyH6uyYe2f\/Dbo8rXUnnFmh4VLf7IlWxwBjDJzFsOVBgbHNfPKxsDcd9+sPPVUx7r26lKRM+Oy\/SfLQ\/pXXSLq6VlbR9L1pnPnbjwHhncgrePHDguX\/mRLtjgCGGXmLIYrDQyOa+aVjYF58cX\/W2qr5x4X+1kv7qsqOR2+UsBGww4Ll\/5kS7Y4Ahhl5iyGKw0MjmviyqdOnZKHHnrIqffIkSPS1dVVtg22gdHZFp1c0UkWz8N5MMz02vNezMF3IfnynZ+fl+npaenp6ZHWm+47Tzw1clUh2eLCSbYYtuSK4UoDg+OaqPKlS5dkcHBQxsbGnHrN\/7e1tfm2o9KTeBO9gJxVRra4gJIt2eIIYJSZsxiuNDA4rokq6+zL+fPnZWRkRBoaGmR0dNT55l9uFoYfKlyIyJZscQRwysxbDFtyxXClgcFxTVRZDYseQ0NDzn\/dv3s1hh8qXIjIlmxxBHDKzFsMW3LFcKWBwXFNVNk946IzMrruagxNOQPT398v7e3tibY375UtLCzIwMCAkG38kSbb+JkaRbLFsCVXDFdVNWxnZ2elo2P93bS4WpNR3rC6urqaTFXp1hLFwFy+fNkZZOfm5tJtPGsnARIgARIggYgE9Av4+Pi4bNmyJaJCNk8rlIHREIRZQtLyamL0hwcJkAAJkAAJ1CIBNS55My8ah8IYGPeSUZBNvLWYqGwzCZAACZAACRSBQGEMTJTbqIuQALxGEiABEiABEqhFAoUxMBqcsA+yq8WAss0kQAIkQAIkUAQChTIwRQgor5EESIAESIAEikCABqYIUeY1kgAJkAAJkEDOCNDA5CygvBwSIAESIAESKAIBGpgiRJnXSAIkQAIkQAI5I0ADk7OA8nJIgARIgARIoAgEaGDKRFmfFXP8+HGnRB4fw5xGguvt7L29vXLlyhWn+v3795d9nUMabWMqWoUAAAYCSURBVMxDnfpuGc3fyclJaWxszMMlpX4N9l2MO3fuLL0YNvWG5aABdl\/LPiGegHo96+zq1avS19cnFy5ckK1bt9Z8\/0AD45Mr9gBw8eJFDgYxfKbMh0efhqzv5DC\/79q1q+xbwWOoulAShqteNA1MPKHX\/uDBBx+UqakpaWtrC\/Qy2Hhqzr+KGsPz5887hvDatWvOAMs+obq4G0N45MiRdX2r\/RLjIC80rq4V+LNpYHwY28HVD9Xw8LDs3r07dy\/DwqdY+Rry8CFKm6G7fh0Q\/v7v\/16++93v0sDEFBw+uTsmkB4y7j6AfUJ01ubLyy233OKIvPe97y0ZGPcXSJ0Nf+yxx+TYsWM1O0tLA+ORK8awbNu2zQm++\/fo6cUz3QTYWcWbE9opTU9Pyz333CNPPPEEDUwMeN0dfwySlLAIeM3AmFlaggpHQHP1tddek82bNztfus0Ypir20+h1FtH9e7iaslGaBqaMgbFnXPgNLP6ENfthjh49ypmtmPBqnm7fvt1R4x6YeKDqoPDAAw\/Ib\/\/2bzvfWHX\/FvfAxMPWqOgS3Z49e5yB1yzTxVtDsdS8vnS7Z1xMXj\/88MPOsmgtHjQwNDCp5K35Vqt7YcwbwlNpSI4q1UHg3LlzDk9u4o0vsCZXb7vtNmefhh767VYHW+Zu9ZzVaKsptNnaMwfV11A8BRqY4sW8dMVcQsIGn+Ylfr6as48++qj09PQ436ZoYOJj7LWERL7x8CXbeDi6VfwMzODgoIyNjTl9BJeQMOwzoWovGXETb3wh4Z1H8bG0ldy3p5u\/cUq+et5en381MM8++yxvpa4SLw1MlQB9TvcyMO4lI27ixbDPhCpvo44\/DOZDxan3+Nm6FTlDEC9je6OpWULiMkc8jL2WkNhHVMfW78YT3kZdHdeaOpsPsos3XH6zBNwQGS9nVaOBiZ+p\/SA7PmwtPr5msD19+rQjSrbVs\/UzMHyQXfVsqUACJEACJEACJEACVRHgXUhV4ePJJEACJEACJEACaRCggUmDOuskARIgARIgARKoigANTFX4eDIJkAAJkAAJkEAaBGhg0qDOOkmABEiABEiABKoiQANTFT6eTAIkQAIkQAIkkAYBGpg0qLNOEigwAXOLp750bu\/evdLS0iJvf\/vbAxNZWVmRb3zjG86TRPVRB\/ouF33pKg8SIIFiEaCBKVa8ebUkkDqBuB5oyKc6px5KNoAEUiVAA5MqflZOAsUjQANTvJjzikkAQYAGBkGVmiRQQAIXL16Uv\/7rv5b\/+q\/\/kkOHDsnLL78sTz31lHzoQx+SHTt2lIi4Dcz169flhRdekC984Qvyxje+UW655Rb5x3\/8R6f8Jz7xCecN22fPnpUf\/ehH8uSTT8qdd97p\/I0zMAVMMl4yCVgEaGCYDiRAArERMK+LuOuuu+Qd73iH7NmzR9785jfLT\/\/0T\/saGP3D97\/\/fRkeHpbFxUX51Kc+JW9605vkj\/7oj+SHP\/yhPPHEE1JXV+f8rroDAwOyYcMGGpjYokYhEqhNAjQwtRk3tpoEMkngu9\/9rnzkIx8RnVWZmJiQpqamm9rptYRk\/k0Lj4yMOOeoofH7vaGhgQYmkxnARpFAcgRoYJJjzZpIIPcE1Ljoss8rr7wiR48edWZf3AcNTO7TgBdIAokQoIFJBDMrIYFiENC3YB84cEDe8IY3yNTUlHOLNA1MMWLPqySBpAnQwCRNnPWRQM4IrK6uOntYXn\/9dWfTbkdHhwwODsqxY8fk8uXL8ju\/8zvrZmI4A5OzBODlkEBKBGhgUgLPakkgLwT0gXQf\/vCHHROjpkX3vfT394v+u\/5u7hoy1+s2MGqAXnrpJWdzrh5jY2POf9UEef0+Pj4ut99+u6Pf19cnu3bt4oPs8pJMvA4SCEGABiYELBYlARKongCfA1M9QyqQAAmI0MAwC0iABBIlYAzMW97yFvnTP\/1T2bhxY6T6l5eXZd++fc6t2nyVQCSEPIkEapoADUxNh4+NJ4HaI6B3Kn3zm9909szooUtOUd6FpO9E0kOXk972trfVHgi2mARIoCoC\/x9a3d5GiHe3rgAAAABJRU5ErkJggg==","height":337,"width":560}}
%---
%[output:664e78eb]
%   data: {"dataType":"text","outputData":{"text":"Evaluating step 10 \/ 31...\nEvaluating step 20 \/ 31...\nEvaluating step 30 \/ 31...\n","truncated":false}}
%---
%[output:7658e487]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7snQecFEX2x9\/k2byETWQkByMKeKLcqad3CqeeiopKFhDBRBYkKzmjICJRUYxnOFG8P8oJp4AIIgssOeyyyy5sDpPn\/6lee2d75tXshN7dnpnX\/w8f799TXV31qrv6u6\/er57K6XQ6gQ6yAFmALEAWIAuQBcgCIWQBFQFMCI0WNZUsQBYgC5AFyAJkAcECBDD0IJAFyAJkAbIAWYAsEHIWIIAJuSGjBpMFyAJkAbIAWYAsQABDzwBZgCxAFiALkAXIAiFnAQKYkBsyajBZgCxAFiALkAXIAgQw9AyQBcgCZAGyAFmALBByFiCACbkhowaTBcgCZAGyAFmALEAAQ88AWYAsQBYgC5AFyAIhZwECmJAbMmowWYAsQBYgC5AFyAIEMPQMkAXIAmQBsgBZgCwQchYggAm5IaMGkwXIAmQBsgBZgCxAAEPPAFmALEAWIAuQBcgCIWcBApiQGzJqMFmALEAWIAuQBcgCBDD0DJAFyAJkAbIAWYAsEHIWIIAJuSGjBpMFyAJkAbIAWYAsQABDzwBZgCxAFiALkAXIAiFnAQKYkBsyajBZgCxAFiALkAXIAgQw9AyQBcgCZAGyAFmALBByFiCACbkhowaTBcgCZAGyAFmALEAAQ88AWYAsQBYgC5AFyAIhZwECmJAbMmowWYAsQBYgC5AFyAIEMPQMkAXIAmQBsgBZgCwQchYggAm5IavdBmdmZgL7RwdZgCxQ\/xY4f\/48\/PDDD5CVlQVWqxUGDRoELVu29KthzZo1A\/Yv2KO25ga52hds\/+j60LMAAUzojVmttZhNUOPHj4e9e\/fW2j2oYrIAWaBuLdCjRw9YuHBhUBBTm3ODHO2rW4vS3ZRiAQIYpYyEAtrx888\/Q\/\/+\/eHqoGfA1qixR4vUtii0lfEW\/LzepEfL6yrw86ww7zd1OX5NbIkGv0ep3atFy4v2Qv7FVZDSbh7oDE3BEo\/XUxqL1+OItnDrt0WZ0d8sUVb8vBGvq0RXgZZ3aPHzDTVlaPkYTWV7HMdOg+1f\/wHdM\/1A1bgBGDjl9WoTt29RKvy3aM75KMDLGwHvsw5wG2nAgbZJA06v45ybXghHProAPZ5rDzFJRrCDilveDmr0NxPgz54VdGj5CjCi58ud+PkK5Hzxr6eg\/VkrXH\/99R51\/fbbb\/Bb8wYQfWMXj9+K7VGgPn4C9F98DeYhT4PqylXhf2\/duhV69uwZ8Cwjzg1z55mgaVPvNvfnJvv2aWDVSn3Q7fPnnlQ2fCxAABM+Yxl0T8RJ6vLYyWBq39GjPo01Dr1HI1Mset5Yjk\/Y+lIDt62GUhyGNMV4XYkFOHjoC21e7VFRtA8yjzwNzbpugaiE7mBuqEXLFybiAGOPxyGCVWKJxX8zx+JgY4rGP\/L5BhxI7LoStK2pumL0fIKmvBJgjp8By9y3QD95BKg7XgPRWry8kQM2rI5YNd6mOFXlPdyPWMDLR3PAxsABGy3g46DlgI3Yjtz0Itg54zDcOeM6SO6SADYOpLDyNsCfpXIOkJg5YFMKMagtSpzR6PlSh2f5I6NWwYLh47jP8MR\/vQ9pk0d5\/H7VFguajJNgXLAMTBNeFH5n\/1sugNm82QTdu3v\/48CfiYgBzIABxqDb5889qWz4WIAAJnzGMuieRBLAWM1ZUHz5U4hP+afggYkEgHFeKQD7j7+A5vabBQ9MJABMWZ4Jzn5\/GVr\/JUXwwIQKwJx66W2YNXAM952eMP91aLJxIQowzOui3fMz2G7rCeqr+bICzKZNZujeHfeGBTIB7dunhoEDDQQwgRiPrgECGHoIqiwQSQDjPuyRADDufY4EgHHvc8gAzOytMKvvEO7s9PKaFdBi+XQUYKqfFL0xcnlgNm20yg8wg3QEMPQdCsgCBDABmS08LyKA8RzXcFpCIoBhy0R4nIvSlpCKfj0F1\/14FXr37u3xUO7atQv+XXBJiOZRF5aA2WSChL\/1hoS\/\/xnYElJtAszG9TbofouMMTD7VTBoiJYAJjw\/KbXeKwKYWjdx6NyAAIYAhlkgnGJgQtUDw9p9bOaH8LfElhKIYfDCvCks2L463IhQYxg\/oXYB5h0H3HKzfACz\/xcVDBqqJoAJnc+EolpKAKOo4ajfxogAUzTmNbC2vdajMXFmPAjRwFEV6crwYF19GR6Qy26oKeYonYrwAEtjMR5QqC7kKGnUuArFEY+3tSIRv29JAj9I2B6HB\/FaYvA2WWJwRY6Zo2YqNuABs2U6PGA2VleKPliN\/gjudf8xRsMPUDZwruEpl\/QqPHDZwDmvV\/mrQvIvoNTOCdRlNuCpkCxOXG1kduLPjNmBP8MWB\/7c886X2KPBfOh3yJ+\/ArQFRaD5y23ARFcPNEzhema23tAL4LqbqobUeOIopC2dHTQgiHPDxnUMYOSbp\/b\/AjBoGAGMfBaNrJoIYCJrvL32lgDG0zwEMFKbEMC47FEXAMPuVrTxfSje+D40\/+ELyHtpCiwfOZr7Ho\/85Etwvjy19gBmrVN+gBmuChqwaBqPTAsQwETmuKO9JoAhgGEWIA+M9DmoTw8Ma0n+vOVQ9s3\/CQBTPGk2LBg8jDtrvTRvAZSvfq\/WAGbDWoBbusk3ae4\/ADB4OBDAyGfSiKqJACaihtt7ZwlgCGAIYDyfASUBTP785bDkwX58D8yba8E5\/43aA5g1KvkBZqSTAIa+QwFZgAAmILOF50UEMAQwBDDKBhgWE3P3vnR+DExeEahUKjAU5oPZbAbDLbdC2v\/9O2hAEOeGDasZwPB3M\/Z3Ztx\/wAmDnyWA8dduVL7SAgQw9CRUWYAAhgCGAEbZAMNad+n1N+Cxxo18Uid9+eWXkJ6eLh\/AvKmGW26SEWB+dcLgUY6g20fTeGRagAAmMscd7bUIMPZhS8F5zQ0eZfQmfLt9bYV\/aiN1CV+FFMfJbRRVhO\/+qS7GVS5QhCtvQMWZfBPwdAj2eDwPTnkCfz+RsnhcoWSP5amQ8PM2Tr4liwFX6piMeD1FOvy8VYurmaI0\/FxIDTgKpWg1Pg56Nd5WnQpXXqnVuO3UnJQBKpV\/u8I6nfxxc3D2iHE48Ofe6sSfDZMDfx9K7fj5Ek75Cvsf78mi1wAOHwTY\/LHwTups0eD8\/RdwbH0LDAYjmBMaCucfT4718MxcuHBBgIO5c+fCY489FvBsV+WBeUMjP8A8ZyeACXhkIvtCApjIHn9J7wlgPB8GAhipTQhgXPaoM4AZPxrgco4EYNyfVPvkZ2D1c894PMAiwNx7772wevXqgGe7KoBZpYVbbpTRA3PQCYNH2whgAh6ZyL6QACayx58ARrQAeWCqngXywEgnhWA9MLacXKHCMrvUY2PPuSycr\/jDk+PMyZHcuCp4eMt6gMvZAE9VphZQ51zxmLUMJ4\/AslcmcgEmPj4eDh06FPBsVwUwK2sBYMYQwAQ8MBF+IQFMhD8A1btPHhjywDALhCLAmLOvom+yJVv6sTdn56Pl2PVOYXP+ykNy3R\/LTpacPMm1bMnJmi09xwpY3crV9hSjSmkCkJsNa9as4QJMhw4dYPv27QE3pQpgljOA4S\/D+XuD\/QcdMPgFAhh\/7UblKy1AAENPQpUFCGAIYLwBDPMQJCDxMczDYHSLdbH+4XXQqlw75Vb\/2KtVdrC6wYUADn9cV30kLDkMTqTb11s4wFJbr7M+tbFH1dq0ZI9z+tQksDuluzdrU5MqochZGUujSU2RXGf6I8ZGlZoqOW926AFSUgFYDAxbQlq4EiAlTYiBcT9YTEy\/w99zY2DkSua4YZkObrlBRoA55IDBL1ppCam2Htwwr5cAJswH2J\/uVQeY0ounwHxsL2hL8sBsMkP0HQ9Dgx6Po9VREK\/ULMEE8VqvXqyqzB7lCnS1XsmsOm\/TVQa62vJc59j\/b9VZwZGbJWmMPfcSWNXS7fadly8JZZzV4EL4QP5xqP6ABWe1c\/48R4GWFSDBLbxCn9pIqK76aX1a5bnqPxj+KCeelpSp1iAGGO6H4Y\/6xCBefZoUVoJdQhLvF3AQL4uBYcfCVcJ\/MIBh5+1Lp3sE8sqtQtq4VH6AGfQSAUyg70ykX0cAE+lPQLX+iwBTGN8a\/t4l1UOm+fVpJzR5snISrX5oynFVkaYMV2nEFeP5hVidxhKO2qgEV61AMa6kcZQUoCOrUqmhwiz98AsfyNh4qKhwwYN4cYVTGpfAzleUnQen3tWHitJzknvZdSowF0vPCde5lWPnLIXn6\/QJ1CQ1q7qfU1Xp1dAkN5W2X+UAVXITj3axpQpbNehRpaS5ylRXD1U7r\/njvIp5EqodMWmVyhn3Q18dqqr9qOWojTR+qpDsXlRINs5vFjePitisco4KyerAcyeBWwyMWI+Bcz7WVqlayh9+tzBGCXM2Cf+\/wYKrmdhv9kO\/QNFnS8FgNII9LgkMBgPEndodtIejKhfSEj3ccr2MHpjfHDDoZUvQ7avTl4huphgLEMAoZijqvyHiJNWlSxfo27evR4NYxts9jR6HmHa3SX7DAMZScB405Z4AYyk6B9Fl0gnQVOz6iOtMlQBTUSr9sJsLzni0h4EE2KXAU1FxobKcw3W+wiz1StS2pQ3xrTxuYYxvCXY3btMnthTKObUuD4muQeU54bzOBrqGLTzqsultoGvsAhGxgKqJFES0f5SxcmTXZj0ufS7VcqTpAGDWcECSc17HkVFHc2TUBDCu4Q4EYHRmN3g6exBUG14IGhCqAGaxAbrLCDD7GMCMNQfdvtp+p6l+ZVqAAEaZ41IvrRInqf79+0OLFp4fTtaoMa+8Bk5pOAKonCpgwFKXR1TMHx\/6aqASFeVqs9NWucwSZZR+1NliRJRBei7K2AwgyjOuQKgvWvrXrnhfe0xlPENUrCesmOLxv1BL4vDMyY4YHBjsnGzUlmi8vM2A108A43oyw90DU+sAs7AWAGY8AUxdzp3hdC8CmHAazSD74gvATJi9BKxGaYyAIaG1x511DVqA2iJ1OegTKz\/2xgrXB555Jqof+nInGOOk5xgkqGVcQsLMpIpPxK0X7wk2rKA9lrNMAAAEMC5Tkgem2mNVB0tItQ0wmxYYoPt1\/CVgf6egfYftMHCC7wBjs9lg5cqVYLFYYOJET9l4VlYWzJs3D3bs2AFGoxH69OkDY8aMgVS3AGm5y\/nbbyovjwUIYOSxY1jU4gvAvLxoC7Qc87mkv6EWA0MAU2kBWkKSPglKjoHRde0Occ+\/LjTYWwxMbQPM5vlG2QFmwESTT0tILLfThg0bYMmSJTB06FAPgDl37hwMHz4cYmJiYODAgVBWVgbr1q2DtLQ0WL58OSQlVQZwy10uLCb\/EO0EAUyIDlxtNNuXGBgWyMsOTXkumExmSOj+ODS6djDaHKUG8RLAEMBgz4BSAebKg53AcOdDygCYeVHQ\/VoZPTC\/22HApAqvAON0OgXoYOkQ\/vOf\/whDN2LECAnAsDKrVq2Cb775BtauXQtNm1YuE2dkZAiww8BmwIABIHe52piHqU7fLUAA47utwr6kCDDF0W3h3uuTfEoWxwJ7v\/1dB63vX+9hn9hSfMtxXRkeq8Eq4C4VlVSg9neWFqHnbWXZ6HmVCp98NdGee3qwCrhLS7FR3OfBwVlessTh9y6NxZVX1mg8L5CDExtjM+IBtjZOEC8vZsZeLajYvZNWPZ7byKbB22rm5DYyccrbOSokp59qo0BeVrUDHx81J0eS0Y7nSDJwcifpbPiyo9bKyTFmrWzP2QHNIfb2RyHpmSWVfzy4Lc1W76vWLYjXcfEAWLc965OHw5vNxLlhy2vR0ENGgNn7ux2enlLutX35+fkChGRmZsK4ceNg9+7d0KxZMwnAFBcXw+jRo6Fr164wfvx4ISM3O9hS07Rp06CiokJYWrJarbKWi4rizwOBPIN0jX8WIIDxz15hXVqcpIwPvgMmUwXkfTdPSBbniE4WAnfva+\/02CiLGYRBzAEYBnEtekvsQwAjfVwIYFz2IIBx2SLkAKYrDlyBTI57j9hqBJiCggJ488034emnnxaWgSZNmgRNmjSRAMzZs2dh0KBBMGrUKI+klevXr4fPP\/9cWE4qLS2VtZy4LBVI3+ma4C1AABO8DcOmhuoAo2l6s6Rf59bcD4vHP8Xt65Q3\/get7pN6YQhgCGDIA+N6BoLxwCQ+9BI0eOjlevfAvDs7BnrIDDBPvVrms4eIeVIwgDl58iQMHjxY8LL06tVL8uJt27ZNCPxl8TPskLNcu3btwmb+D8WOEMCE4qjVUpu9AUz2hofh9ecf4d554oxl0HbgEfLAsC1oaAmp6jkggAkOYGxXMuHiy7eCUgDmvVmx0KNL8B6YzFwHZOU5gP134irvS0jVJxUewBw7dkzwrLBg3Z49e3IBhqmY5CxHAFNLHyMfqyWA8dFQkVDMG8BkbXsW5g27g2uG8a+\/D+2f+D8CGAIYyTNAABNmADNTHoBZ8aEJ2D\/x8DVXE3lgIuFL5HsfCWB8t1XYl\/QGMGWnf4Rbi97jxsCwQF4WN6cy5QCTOzbsOgBatR6E2oyCeKVmoSBelz0oiNdlC61VA4rzwEyPk8cDk+eArFwH7D1qhRUf+SajZpbhAQzFwIT95wntIAFMZI472msRYGLvehe0KT08ymT9ayjcfb3VZ3XSDz9roeuf1nnUoyrjb1UPZbjayMFRG9nLL6N9sVXgKiTecOuiPbfmZ2W56qSYOP6TE4dvfueMwXND2aJx9YspGldxlcfgKi67EVcCOTkqJDtHteRNheTQce7NOe\/Q4Aorh9ptO+c\/rMkrL+Ztcjc67zzbHdrfg3eN2o7vrMw978DvrflDVeTeLjXnvNaiBWv+BTg540ZI+vsESPp75cZtagt\/E0W1SfqM2bP3QfnXA32OMeHZTJwbtjKA6cy\/v782ZwDTf2aJz+3jAUxNKiS2JwyLj7Hb7V5VSP6WY3vO0FF\/FiCAqT\/bK+7ONQFMVIUaii7+AJk\/zxJ2uXTqWYI+pwfUiB1j6qQTeYOhYYpUnUQAIx16AhiXPQhgqnlglAgwr8bLDzCzi4MGGLn3d\/G1PsVN4hHWIAKYCBtwb931BWDcr0\/\/6E5YMPUJbrWzFuyBLrdKvTAEMAQw5IFxPQO+eGBYFvjEHpXvWb16YKYmyA8wc4qCBhhml9OnTwsb1rEM3Oy\/4k68KSkpsGLFCkhOrtzrSe5y9AmpPwsQwNSf7RV350AA5vSX98GscQ9x+zJ56jK4+d7fJL8TwBDAEMD4BjBlJ\/fA+ZX\/AMUAzJRE6NEJXwoNZELbe8wC\/V8rlAVg2P0vXLgACxcurDEXktzlAuk7XRO8BQhggrdh2NQQCMCc+nYIzH7+Vj7ATN8KN9\/9HQEMW2yjGJiq54AAJkQB5pUG8gPM6wU+A0zYTLbUEVksQAAjixnDo5JAAIbFxFyvW8dVJ1EMjOvZIIBx2YIAJkQBZnJD6NFRRg\/McQv0n5tPABMen5A67wUBTJ2bXLk3FAEm7q53QZfsqUKKLsPVGKe2D4a\/3OypTtr7+zXQ5rpXPTrsdQmppAw1kKOsGD1vr8BVSNaKS34ZWhfVBC2viUpBz6vjEvn1x+D5UZwxBrwPMfjGYOYYXM1SEcXJnRSFK4ScPLURL6+RHlczscY7dBylE0dtZNfiaiOn2j91EleFxFEz8QZHxVEIsfJcFRLnGpUDfx80Nnzc1DZOriVOLiSmWio7tVtYQmo55guIaVu5w6zGiwpJhamQtg8IGhCqVEiTGskPMPOuBt0+v152Khw2FiCACZuhDL4jgQKModwJBdm74Oyvs4QAOpUuFRLTekOzZk+ijSKAkZrFTgBTZRCeZ4YARiEAM5EBDA7igcxAe4+bof98AphAbEfXsD86mF6MDrIAAAQDMJgB1eX4X+sEMAQw5IFxPQNqLx6Ywn3vw6X3nlOOB2ZCY+jRQUaAyTBD\/wVXyANDX6CALEAAE5DZwvMiAhjPcaUlJKlNaAnJZY+6WEJSHMCMT5IfYBbmEcCE5yel1ntFAFPrJg6dGxDAEMAwC9goBkbyIKjrMQZGaQDz7rhk2QHmqUW5BDCh85lQVEsJYBQ1HPXbGAIYAhgCGM9ngADGtby8Zaz8APP0YgKY+p35Q\/fuBDChO3ayt5wAhgCGAEZZAJO3fT7kfTMf2k0\/BLqGLYTG1acKafPYZOjeXr4YmH0nzDCAAEb2uTxSKiSAiZSR9qGfIsBEPfAOaJre4nGFqhzf\/8FYjsuAY0pxOam2FJf7shuqSzmJHkvK0R44igvQ8zx5Nag40tdonly6AW65OFwqzQo7eBvWxeIy2rJYPI7eFMuRLHNk0bYo3HY2XjJHPT4ONk5iRgFuODJqqxZvq1XNO4\/f26LGzzsAl13z1Em8x91bkkc14M+G3oGPm8GOP\/c6B35eY8fr0Vk45W0aKPhsCRT8awm0WPwTaBs3rwQYM16e\/aYzSRMtOjIPgPmjEUEv0Yhzw8axSbIDzKDFFAPjw\/RMRRALEMDQY1FlAQIY5K\/vOAKY6lYhgHFZIxIBZsPYRnCLjB6Y\/SfMMHgxyajpMxSYBQhgArNbWF5FAEMAU+ll4XvICGAiG2DWjU2Em9vLtxPvLycsMGyx77mQwnLipU4FbAECmIBNF34XEsAQwBDAeD4DSlhCumZTZlXD6nMJae3YBLi5vXSZKpiZ8JcTVhi+2Pds1MHci64NPwsQwITfmAbcIwIYAhgCGGUBTN7bL0HJ7o9AKQCzZmwcdGvPj8Hxd\/I5cMIGIxeXBB2j4+99qXx4WIAAJjzGUZZeEMAQwBDAEMBgk4k4N6weGwM3yQgwv56wwbOLywhgZJnBI68SApjIG3Nuj8VJSvPkG6BqcZNHOX05Lp9UV+DntaX4Wnl8Ca72YDfUl+DxF+oSC97uYk7yR446CdT4vdV+Bus64vlSUlMcfo\/SWFxJY4vF+2aPNqF9tkbh5W1GK16eszGd2YDXY9JwbA0AZVpc6eTQ4Oe1arxNOhU+zgYVrlrSc8r7+\/raOUojVo\/diavmzE7c42B14qoim53znjjwpZcYG14+2maA4pWToOL7TyHl0xNVXdV5SeaoN7vd49xBcG58PmhAEOeGN8ZGw00d8H77Oxas\/K8ZdnhucXnQ7Qvk3nRN6FuAACZExvDcuXMwfPhwmDVrFvTs2dOj1WazGb788ktYvXo1nD17Flq3bg3PPvss9O3bV0iw6MtBAINYiSOXJoCR2ooAxmWPcAaYVWOj4EYZAeZghh1GL64ggPFlgqYyHhYggAmBh6K0tBSmT58On332Gfqi22w2WLlyJaxZswYGDx4M3bt3h59++gk2bdokQMyYMWNAq6153ZoAhgCGWYA8MNLnoL49MJb0vdB4zfeK8MAsH2eQHWBeWGQmgAmB75ASm0gAo8RRqdam4uJiWLJkCWzevFk4u3XrVg8PzO7duwVQmTp1KvTr1w9UKhWwJOMffvghzJkzBzZu3AjdunWrsacEMAQwBDCez0B9AkzBq0+BPS9LMQCzbJwBbujAXwKucZJxK3AowwEvEsD4azYq\/4cFCGAU+igwADl8+DC8\/vrrcPz4cbjhhhvgxx9\/9AAY5n2ZP38+7NmzB9555x1IS0ur6lF2djYMHToUbrvtNpg4cWKNXhgCGAIYAhgCGGxKFOeGJeP0cL2MAPNbhgNeXmQhD4xCv0NKbxYBjEJHKD8\/X4CPvLw8WLRoEZw\/fx4mT57s8aIzD83o0aMhISEB5s2bBzExMVU9qqiogEmTJkFBQQGsWrUK4uPjvfaWAIYAhgCGAMYbwCwcr5MdYMYvtBLAKPQ7pPRmEcAodIQYdHzwwQfwyCOPQFJSEmzbtg0FGNHL0qtXL+F3tnwkHsyLM3fuXNi5cyds2LABmjevzKXCO0SAcQxbBs7WN3gU05k4uV8qcLWRrhTPF6Qp5QcVxxXjCgdjMUedVMzJnVRYgneTo0KChFi0PC9YtyKB70YviePk84njqIr8VBtZonBlj8mI26JEV4H2zaTFz+u1eDtZJXFq\/B6xnPPRnPMajtqId16lwhVcvPO8Z9zp5I8b7zc7V4WEq4pMDvx9KHXgz32R3Yg218bOTxgFcDkbYNNnVWVirNHcdzjWKq1LffowGN8cHzQgiHPDggk6uE5GD8zhDAdMWEAAo9DPkOKbRQCj+CGqbCAPYE6ePCkE7jK1EVsmcj\/Y8hJTJzGAadeunW8Ac+cggGtuAGdiKkCD1KprCGBc5iOAkT5KBDAue8gOMKzqBW\/6DTCq\/MugOX0Y9B8skg1g5k3QwrUyAszvGQ6YtMAWdPtCZBqnZspsAQIYmQ1aW9XVJcCIfXDcOQicdw0igEEGlQCGAMbqrAMPzMCHAFLSAgIY3bdbQLfjXWGgsOB\/f+Yq0QPzGgOYjvIF8f5+3AFTCGD8GQoqW80CBDAh8jjUJcA4Hp4keF7IAwNAS0jSF4SWkFz2UDrAMA+Mdv8OAWLkApjZE7TQVUaAOXLcAa8SwITIV0h5zSSAUd6YoC0KJgaGyazdFUrYTSgGxtMqBDAEMPUaAxOEB4aNnNwxMLMmaqGLjACTftwB0+bTElKIfIYU10wCGMUNCd4gHsCIKqQGDRoIKqSoKFfgbKAqJAridY0BAQwBDAEMgPjHzfRJOtkBZuY8CuINkc+Q4ppJAKO4IfEPYGpjH5iS0XPB1vZaj4bEmHFVkYGnQirDVRf6Mlx1wW6oKcZ\/SyjC1Un6Io46qQhX2EA1lVb1Djri8fuaEvH7Fifg92V12uPxe1tiOCqkGFzZY+bkPCoxlKMPSYkezwul56iNGmvw8gkavH52U70a74NOg583qHFbaDkqJKMKt4UGcHtrAFcn8V5rb7mQLJyYFivgsS42jjrJ7MDfEytHbWTilC+xR8HlJ4aA4YZrIXHiS1VdumJ3bZXg3k+rVfqb8cRRSF02U7YlpFcn6aGzjB6Yo8cdMHse7QMTIp8hxTWTAEZxQ+IfwLDSvJ142e5LKC4+AAAgAElEQVS9bA8Zf3fiJYBxjQEBjPR5JIBx2aMuAObSX\/pA9N\/uUgzATJlsgE4yAsyx4w54ba73VAJsOwi2Uecbb7wBBw4cEPa8evzxx2HAgAHCFhPVj6ysLMETvWPHDjAajdCnTx8hlUpqqktNycr7Wi5EPg8R20wCmBAZet4SEmu+mAuJJXIcOHAg3HrrrUHlQiKAIYDhvRYEMJENMJMnG6FjJ\/myUR8\/Zoe5c01ePUSffvqpsMdV586dhfmNHSzPG0tSu2LFCkhOThbOiQlv2WaerFxZWRmsW7dO2J18+fLlVbDja7kQ+TREdDMJYEJk+L0BDOsCy0b90UcfCfu9BJuNmgCGAIYAxmWB+lxCUpoHZsLkKOggI8BkHLPDgrn8bNRsQ8\/nnnsOGjVqJGzKGRtbueFkbm4uPP\/880KaFLYTOTvYbuPffPMNrF27Fpo2bSqcy8jIEHY0Hz58uOCxYd4cX8qFyGch4ptJABPxj4DLAGKgHgEMAQwBjHIAJm5gf4gb1L+qQfUZAzP2lWjo0KnmzPa+TqsZx2yw+PVyrgdG3KiTeWDuv\/9+SbXr16+H7du3C8Ci0WgEkOnatSuMHz++akdyi8UC06ZNAyZoYEtLVqvVp3LVxRC+9oXK1b0FCGDq3uaKvSMBjOfQUAyM1Ca0hOSyR23HwBRmFcPlJ4aCkgDmhSlx0K4THtQcyMR28pgVlr9WUiPAzJo1C+68804PgGEQw7zOWq0WBg0aBKNGjYLHHnvMo9znn38uLCeVlpb6VM49tiaQvtE1tW8BApjat3HI3EEEmOyXp4CpfSePdsdY4tC+xJvx3CzGco4KqZSvQtKX4AoOLUedlFiI7wqqK8TzBfFUSJZE\/K\/Kwga4+sUez88XZI7FlTdWjtrIFI0rbwqMpai9zXr8fEMtfr6xFs8LFcVRIUVp8HpYY2LU+G+xKly5FM1RFUUBbiMD4ONmAEutv0dmwHMYmTkqpArAn9VyJ\/7cFzvx96fMgauKSjLL4egj4yB1yIOQOvTBqv5XeFEhXbFJ76HJOAExC5bKpkIaMzUB2soIMKeOWWHlnCJu+3JycmDEiBHQqVMnmD59etU2ESUlJTB27Fg4evSoADDsYClVmJeF5YWrfrDl95UrV\/pVrqa0K7X+MNINfLIAAYxPZoqMQgQwnuNMACO1CQGMyx6RCDCjpiZCm0446PkzS+bn2aHgih3y8xzwwVvFXIBhMSsff\/wxTJkyBW6\/\/XZ49NFHwWQyAVNYsriYY8eOCWDChAzMA8OCdXv27MkFGF\/LEcD4M5r1V5YApv5sr7g7E8AQwDALkAdG+hyQB8a1kd2IqQ2hTefgAea7T0qB\/RMPb6kOHA4H7Nq1C5YsWQLp6enQpUsXePnll8Fut8OMGTP88qz46qkhgFHc5wltEAFMaIxTnbSSAIYAhgDG8xkggHEBzLCpjeCazvgSmT+TVAHzwOTZ4OwxC\/zfJ\/wYGG91svgX5p1haVKYV4ZiYPwZgfAoSwATHuMoSy8IYAhgCGCUBzAtpgyDhve54jrqMwZmyKvJ0FoGgBGtfPaoGdbPzuUuIbFUKePGjYPrrruuSi7NrhXTpLD\/7Yu6iO0Jw8oxr403tZJYju0lQ4fyLUAAo\/wxqrMWEsAQwBDAKAdgrv5yAU6NngdKAphBr6ZAq878IHx\/J6tzR02wcfZlLsCwmJXZs2cDk1OznXhZzjd2sN3H2f4wc+bMgb59+\/q8vwvtA+PvCCm7PAGMssenTlsnAox1wvPg6NjO49451ni0PXoLfr6BqXLTKfcjyksuJH0p7p7WFeN\/EUUV4uXj83HTOVX4+aKG+HlzIid\/URw\/X5A5Dr+mgpML6SpHbeTUF6GNStMVo+cTOOqhaC1ePl5TiNaTyFEascIJgNfVkHM+xonbgneepzYyOHF1ktbJz0mFdc6m4u8ia1bh8mCeOqlMhX\/Ii1T4s1oM+PkiwN+f0\/uvwMFn10CnaY9BWp+bq7pTasfVTKxAuU1al\/3YWTDNXSebCunpaWnQsjOuvgpksjp\/tAK2zMr22r6DBw8Ky0M9evQQUggw5RHbifeOO+6AmTNnVm1ud\/r0aWHDOrZDL\/uvuBNvSkqKZMdeX8sF0h+6pm4tQABTt\/ZW9N0IYJC\/vglgJEYhgHGZIxIB5slpzaCFjABz4WgFvDcr0yvAMK\/J4cOHYf78+UJWbKY+GjJkiLCzrvtSz4ULF2DhwoU15kLytZyiJ2xqHBDA0ENQZQECGAIYZgHywEifg\/r2wNy4eiQ06NZGER6Yx6c3h+ad8X2fAplKLx4thw9mXgzaQxTIvema0LcAAUzoj6FsPSCAIYAhgEGeAc4Gd3XlgVESwDw6vSU07yxfgOvFo2Xw0czzBDCyzeKRVREBTGSNt9feEsAQwBDAKAdg0r84CcdmbQMlAcw\/p18DzbrIBzCZ6WXw6cwzBDD0HQrIAgQwAZktPC8igCGAIYAhgMFmN3FueGB6W2jaBQ\/OD2RWzEovhc9nniKACcR4dA3FwNAz4LKAOElFvTIENJ1ae5im0E3hIBbIsiSiZowy4+cTK\/gToJGjQjLwciQV4n8NJuTjihInnjoJihviKhdbfBnaN3McnsuHFa7gqJDyo\/CcRDYDrgZqqsNVSIla\/HysrgBta0M1Xn+y6ipenqMoYoWTHXhd8U7cTvEOTo4kB57\/ycBRFemdNlleVTtwHgAAsKvw38wc5VK5mpPzSI3HiOSrcbVRvgpXFf30VTbsnvEt\/O2tRyG1W7Oq\/l914u8VK3DF3khiJ8vRi5A\/+8OgAUGcG\/pObwdNuvBVUP4O0qX0Evhy5smg2+fvfal8eFiAPDDhMY6y9IIAxtOMBDBSmxDAuOwRiQBz34yOkCYjwGSnl8DXM44TwMgyg0deJQQwkTfm3B4TwBDAMAvw9nQhD4z0+ahtgPlu7XE4tPZneOSLoRDbxOW9qU8PzN9mdIbULrgnKZCpNCe9GL6ZcZQAJhDj0TW0hETPgMsCBDAEMAQwns9AfS0hKRFg\/jqjC6R0SZBt2rycXgTfzUgngJHNopFVEXlgImu8vfaWAIYAhgCGAAabJMS54c4Z10GyjACTm14EO2ccJoCh71BAFiCACchs4XkRAQwBDAEMAYw3gPnzjBsgqQs\/iNjfmTEvvRB+mHGIAMZfw1F5wQIEMPQgVFlABJhG0x4BQ2eX6kEsUGqtTKTmfuRZ8URClyx4+UQTfwKML8MVSgaOOslYhKuQ9AV4vhYHJxeStQGuljEn4OoaUyyuomG2KYzB1UalRlw91EyPJ25qqOUofnS4eihFg59PBvx8Eyd+PsmBq5lY3xrYS9FnoIEdt1+sw4KWj3I40PM6J\/5CanjngTOgnPfaDpyKBBUSfpGJc75CjedVKtXw1El47qQrGnxJ5sN1mfDD20dgxv7HJQ3LVeHvFSt0yZkkKVuafhlOz9wRNCCIc8PtM2+Cxl349\/d3Or2SXgA\/Tv816Pb5e18qHx4WIIAJj3GUpRcEMJ5mJICR2oQAxmWP2gaYtbMz4NBXZxUFMH+aeTM0khFgrqYXwP+m\/0IAI8sMHnmVEMBE3phze0wAQwDDLEAeGOlzUF8eGCUCzM0ze0LDLtK9ZoKZQvPTr8Iv038mgAnGiBF8LQFMBA++e9cJYAhgCGA8nwECGBCyQPfv3x9umvknaNClsWyzZkH6Ffh1+v8IYGSzaGRVRAATWePttbcEMAQwBDAEMNgkIc4NN8y8HRJlBJjC9CtwaPqPBDD0HQrIAgQwAZktPC8igCGAIYBRFsCcO5ALL37RV9Ko+gzivW7mHZDQRRooHMxsWJSeB4en\/5cAJhgjRvC1BDARPPi0hOSyAKmQXLagGBjpm1FfS0gLRv0KhZfKFAUwXWb8GeK7JMs2axan50L6jB8IYGSzaGRVRAATWePt0xJSm+n3QGyXFI+yl+34xFXCkVFnc87nm3HZNbthcgUusY4uxWXRUYV44jxtESdhpAqX0VoTcXmwKR5P2lgah8uGWR\/yonAZcjJHLp3COR\/PkUunafLQcWwCuej55g78fBInMWOqDZd7s8oT7Cb0HvF2XBYdY8elxganFq1HB\/h5Lae8nK+vTYUnjLQCfr5cjScALVfjtijV4Mkir2rxZ3vcmDNwJdsEiz7rKelmtoYfRJutkv4mqHxmBK\/yEb2zHWfcJTvAHJ\/xfwQwcj7IEVQXAUwEDXZNXRUnKQIYl6UIYKRPDQGMyx6RCDDtZ9wNccgfNzXNLbzfS9Ivw4kZ\/yGACdSAEX4dAUyEPwDVu08A4\/kwEMAQwJAHxqVCajuDeWdTZZs1S9Nz4NSM4Dfak61BVFFIWYAAJqSGq3YbSwBDAMMsQEtI0uegPgGGtWTSmzcoZgnpmhl\/gxgZAaYsPQfOzPiGPDC1O7WHbe0EMGE7tP53jACGAIYAxvMZqC+AGfTIUWicZlQUwLScfh\/EdEnzf3LhXFGWng3nZ35NACObRSOrIgKYyBpvr70lgCGAIYAhgMEmCXFuaD69D0TLCDDl6dlwceZXBDD0HQrIAgQwAZktPC8SJ6keM7pDwy6eSqGLTnzt+7LNU7HELFRkxdUSZy38nTzVJjxRXONSPOFdVDGu4OAleXRyVEi8nEflCbgK6Wp0MfchUBnxJIktOGqjRD2uKkrmqI2aq3LQe7d2ZKPn0+yc5I82vA9JNlxpxCqPs+NKmjgHnsDQ6MTHx6DiqI3w6kHFUfYAZzy5g+PkJ390cPpm5+R\/NDtxdZJJhT8zJWo8AWihBlct3ffEWUhN08OiFW0k3bmsjed276JWqhS8lF4CX808GTQgiHND0+n\/gKguTWSbACvSL0HWzC+Cbp9sDaKKQsoCBDAhNVy121gCGE\/7EsBIbUIA47JHJAJM2vQHwNi5qWwTkeloFmTP\/JwARjaLRlZFBDCRNd4+LSGRB8ZlJgIYApj69MBcf2MsjH+luWI8MCnTH5IdYC7P\/IwAhr5DAVmAACYgs4XnReSBIQ8MswAtIUmfg\/oCmG5\/yYB7\/t5QUQCTNP1hMHRuJtsEaD6aCXkzPyGAkc2ikVURAUxkjTd5YCgGpuoZSKEYGMn7oLQYGCUCTKNpj8oOMFdnfUQAQ9+hgCxAABOQ2fy7KCsrCy5evAhdunSBuLg4\/y6uw9LkgSEPDHlgPJ8B8sC4NrJrOK0f6DtLl7SCmaIsRy9C\/qwPawSY33\/\/HRYuXAhsjlKpVHDPPffA+PHjoUWLFpLbs7l23rx5sGPHDjAajdCnTx8YM2YMpKZKBQi+lgumb3Rt7VuAAKaWbWyxWGDp0qXCy9SpUyc4ffo09O3bF5o1k88NK1cXRIC5d3onSO3iqXQ4p8L3f7jgwFUJ+VY8a22uFxXSJTOuUEopx\/MnxZTguZCMnBxJPNVKRSKe24iX8ygvKp9r9uZGXFXUWH8FvaaRFs9V1FJ9CS3f2omrkFrZcBUSz9OSasXVRol2XCHEGhPniEHbFKXCVUg6La6wAR2u4AGOIsfJOQ9qjkSINzoOvgpJxVEhAe+8FbeT1YZLqcqcuDqpTI2fb\/7XdPjH3xJg1kTpxzdPa+Q+e1k6qYrv3FETbJp9uUZAqGkOEeeGBq8+LjvAFMz+wGv7Dh48CIMGDYJrrrkGBg4cKDR106ZNUFxcDGvXroU2bSpVWufOnYPhw4dDTEyMUK6srAzWrVsHaWlpsHz5ckhKqpyPfC1Xk03o9\/q3AAFMHY+B2WyGjz\/+GMrLy6F\/\/\/7Cy6aUgwDGcyQIYKQ2IYCpZo9aBJjMy1a49akTMHJgIxg5SAr19Qkwia\/2B11nqdcjmPnLevQCFM7e6hVgFi9eDDt37hRgpWnTSgUUg5ChQ4fCQw89BKNHjwan0wmrVq2Cb775RlIuIyNDKMfAZsCAAT6XC6ZPdG3dWYAApg5sXVFRAb\/88gt8+eWX8P333wvemN69e4PBYIAHH3wQrr322jpoRc23IIAhgGEWIA+M23NQDx4YpQJMwtQnQde5Zc2TiY8lrEfPQ9Gc97wCzPz58wXP9ZIlSyA2tjLLfH5+vgAmPXv2hIkTJwreGAYyXbt2FZaW2DITO5gHfNq0acDmYLa0ZLVafSoXFYXvX+Rjt6hYHVmAAKaWDS2+QOwvgWHDhsGf\/vQnaNCg0s1rs9ngk08+gebNmwvn6\/sggCGAIYBB3kICGCH2hHmM46Y+LTvAlMzZ4hVgmPflhRdeEMDkscceA4fDAR9++KHgcVm5cqUAMWfPnhWWmUaNGiWUqX6sX78ePv\/8c2E5qbS01Kdy4nJTfc\/JdH\/vFiCAkfEJYbDCDr1eL6n1gw8+gOuvv16IgXE\/Tp48CXPmzAH2V4Z7oJmMTfOpKgIYAhgCGAIYbLIQ54bYKYNA26mVT\/OJL4Vsx85B6WsbvQIMWx7as2cPjB07FvLyKuPLGGCwuJYePXoI3hY2jw4ePFjwsvTq1Uty623btgmgs2HDBuG8L+XatWvnS\/OpTD1bgABGxgHYv3+\/sFbLXjStlh8IWf2W7IVkL+e9994L9e22JIAhgCGAIYDxBjDRUwaDtlProGdNR14hOK4UgDOvECrWet\/Ibvfu3fDSSy\/BddddB48++iiYTCYhiJcF6bJlJbZsdOzYMcGzwqCGeWSqH9UBhnm9fSlHABP0ENdJBQQwMpqZSaVHjBghvEDPPvtsVdS7jLeo1apEgOk\/rSm06Oy5BnxKg28hfhZwRVWWDc+dlG\/BcyexzvHyJBnKcXVSg1I8L0wUR4XEy4XEUyHlx+L5gqxReH4h1oc2nNxGiXpcbdRCm4WO6zWAn29lx9VGLa24yinViiusGtlwyE5w8HPtRKvxa9QGK9oHpx5XG3HPa3BVEU+F5OSokFRe1Ea8l4inQlJbOQmabBq0KpUFt5HDrEPLlzo8cyRduGyG6wb8BkvGN4VH70mUXJer5Si4ACBbLxUFnDxmgxWvlcimQop+ZShoZAAY86c7wfLZzqp+bd261QM82I8FBQXw3HPPCUqimTNnVsXAsKWgyZMnC7EvbCkpJyfHJ88KeWBq9RNS55UTwMhocubq\/Oqrr2Dq1KlC0FiHDh3grrvugvvvvx\/atm1bFVjm6y1PnDghvJy7du2CkpISIQKfrUOzaHp39RJTN7Eg4dWrVwvrwa1btxYgikm2WbCwLwcBjKeVCGCkNiGAqWaPWgSY3YeLoc\/444oDGOMrw2QBGOeVAmBeGPuxM2D9bCcXsI4cOSKACVtiv\/POOyUP47\/\/\/W+YO3eusDTEPN4UA+PLLB9eZQhgamE82bIQAxkGFOwFZMfdd98Nr776qvCXhC8Hc5uyvzwaNmwoBP+yfWN++ukn4WVlHp5FixZVeXiYW5St8a5Zs0Z42bt37y6UZW5WBjFsIydflrQIYAhgmAXIAyN9DurDA6NUgDFMfgY0na7xZQrzqQwDGPPct4MGmJSUFK\/qIrbcxOJj7Ha7T+WUtL2FT4aM0EIEMLU88MxzwvYmePPNN4Gtq7I9DWrajZddwySBbK8YBibVg3tZjM3IkSMFLw\/zxLCDwQ4DFXauX79+gqeHeYNYpD4LEN64cSN069atxp4SwBDAEMB4PgMEMK6dePWTR4BaRoBxHDsDlrlvcQFGXEJq1KiR4G0RZdTiEtLVq1fhjTfegMTERNoHpsYZPvwKEMDIOKYsEn7ChAlCsBnzuNx8881Vgbls62omA2QyP\/coefcmsCWgF198UZBWsz0Oqh\/i\/gdsC232F4VOpxPcqywQ+J133pF4eLKzs4W9Em677Tahnpq8MAQwBDAEMMoCmA8XtYJbr5fGtdRnDIx28khQd6zc+VaOw3H8NNjmrvEao\/Of\/\/xHEEYwJScL4mVz3kcffSTsrcXgRZxP2V4xbMM6tmTO\/ivuxMu8MytWrIDk5GShyb6Wk6N\/VEftWoAARkb7Mrhgu0WytVkGLAwYWIQ8i4G54447BAngoUOHBIgJ9HAHGHFjpoSEBAFoqrs+WRzOpEmThEA4FksTH88PzmTtIYAhgCGAIYDB5iZxbtBMHgUqGQHGefw02Oe+WaOM+vDhw4L3WsyFxGIL2TzK8suJm9axdl+4cEHImVRTLiRfywU6T9N1dWMBAphasDNbvrl8+TIcOHBA2ECJBeEy0GB\/Gdx4443C0g\/bv0Dc0M6fJrA6WbDa008\/LXh0WPQ987Kwv0JYVH71l5m1g7ld2bITi51hG+Z5O8RJasTUhnBNJ+leNsJfLjo859EpFUed5MDvd9WLCinTUvlXkvtRYMJVSKll0twv4nXRRZydNDmpcMoScKVOTgye8yjJgCt+2P2bGHC1UWMdnsOojfoi2ue2DlyF1MaG50hKs5ag9STjAiFoZMcTi8aq+UHfamPlXkfuh9OI38Sht6Pl7ZzzTk7uJCdPncTJLs57zlVOb7mQ8N94S0hqK65CUls46iQTrkJymDzftU07LsPQhSfgi4UdoNd10nEq0JRxX+Nct1scPe6AOfMssqmQVJMYwLT1Z8ryWtZ5\/BQ453kHGNluRhWFnQUIYOpgSFmQLVteYp4ZticBW7dlR8eOHYW\/KrAN7rBmieu+\/\/3vf4W4FgZD4gZOTG3kvtzE6mDLSyyYmAFMTXsbEMB4Wp0ARmoTAhiXPSIRYJyTRgPICDBw\/BSo5q0KGrDqYBqnWyjQAgQwMg4KW7J5\/\/33Yfv27cCCzlgMy6233ipkS2WekW+\/\/RaOHz8OQ4YMgd9++w2+\/vprePzxx4W13ZoOJpNmy0BszZd5Xlggb\/UdKOUEmLv\/GQttOuuhQWMNNEhy\/TVJHhjXKJEHRvrEkgemGtgoxAOTd8UJx447YM06a9CAIP5xY584Bpwd5dulVnX8JGjmrwy6fTXNn\/R7eFqAAEbGcWU78bLloSZNmkBmZqawbMQOBjMs\/oSdY6qie+65x6+7MjBaunSpkMuDSaKry6JrwwMjNo6BzF8frkyexg4CGAIY8sCEjgfmk3\/ZgP1jB2+jOF8nIhFgrBNfAIeMAKM+fhJ085cH3T5f+0HlwssCBDAyjufBgweBbT7HkomJGahZBD0LQGMyP1ERVD1Opabbs50mZ8+eLcTSsKA1JpeuvjGdqDTyFgPDZNbuCiXsvuIk1W9EguB5IQ8MAC0hSZ8UApi6AZhZm8\/DrC0X4NDma6FFijQmydcYGOaB+e9uuwAxcgGMacKLsgOMccGyoNtX0zxKv4enBQhgFDyuTLU0ZcoU+OGHH4Q9XtguvO5SaDGNPAsIZiqk6vmUAlUhURCv66EggCGAqY8gXjkAho2c3EG85RNeAnuH9rLNmpqMExC9YCkBjGwWjayKCGBqabyZjJp5Sho3xtUzNd02NzcXxo0bB3v37hU2o3v44YdBrfbMycIChOXeB2bs5Gjo0Mkzn8sZPa4Q4uVIOulsgXYz04bnTmKF88x4nqTzZtyOSZwcSfGF0r0zxIY4OWltihNwBc\/VaDznUWtDZVZc7Gisv4yeb6HLRM+3Bvx8R9sFvB4r3qYmVhNaPsmGK7ISAZfVazlKI1a5MwpXIdmNeH4entqIp04CrX+5kMBPFRJ4VSHxch5x1EmcWBcN77wJz5GkqvBUIc3ckAns34ktt0DLFKNkXIsAf1ZZocvaCknZ3zMcMGWBfB6YkgljwSYjwGgzTkDcgsUEMDV9EOh31AIEMDI+GEy2zJaQ3n77bfjiiy9g8+bNaIKymm7JoOT111+Hd999F1577TV45JFHvOZR4u3Ey+7PUg74uxMvAYxrhAhgpE8rAYzLHjy5dDgDTNH4cWDt0KGmKczn33UZGZCwcBEBjM8Wo4LVLUAAI8Pz4HA4YN++fcJujyyORDwCXXdmCiUWDGw0GuGBBx6QLAuJdbM9Xf7xj3+AXq8HMRcSS+Q4cOBAQfkUTC4kAhgCGN5rQQBTtwBj3nG7x1DUpwcmf\/x4sMgIMPqMDGi4cCEBjAzfoUisggAmiFFnMSZsk7rly5dDRkaGR02BAgzbK4ZtSuftYLLp6jEvTGbNttdm+70Em42aAIYAhgCmmgVsdb+ENHjuadi0PQ+UBjBXxk0Ac4eOQcya0ksNGceh8aIFBDCyWTSyKiKACWC8WbJFtkTEEjQyFVD1g23lz3JwsCNQgAmgSbJcIqqQCGAIYAhgCGCqPwPi3JA7dpLsAJO8eF7IzZWyTLhUSdAWIIDxw4RMFcS8IyzGhUGMeLDkYg899BCMGDFCWEoSvScEMBTEW\/3xoiBelzUoiNdlCyyIV6kemMtjJ4NJRg+MMeM4pCyeSwDjx3eIilZ7d5ws8pQOrgWYec6dOwdvvfUWfPbZZ1Wb07EL0tLShJgTFmTbsGFDoY7qyz+hCjCvTDJAp46eioyLejzvUIYWz3l0QtUStet5O55TiRXON6eh15w04wqo2PIktHxiMa6wcXJUK\/nxRWg95mg851EbvRcVkgHPYdRKg5\/v4MTVRm1snHoseH6mNAu+1NHQkYD2LVaPl3dGm7nvg4OjNrJF+5cLyalz4Pfg5EJSaTjl\/Z27vKiQnA5OniQbrk5SWfHzao7aSMs5r67wzJE0ZPZZ2PTvq+DYeZtHD0utfFvkaQol5X\/LcMDYRfLlQsoeOxVM7Tv5a3VueeOJY5C2eA4BjGwWjayKyAPDGW8WmPv7778LW\/ezfVhYoKx4sK3\/2YZyt99+u0eALQGMy6AEMNKHiwDGZQ8CGJcteACz60ApnNl6s7IA5mUGMJ1l+0oaTxyFtCUEMLIZNMIqIoBxG3AGKnv27BHiW1hqAPFgG8jdfffdwjb+HTp0QPdkIQ+M1JgEMAQwvKUiAhjvAHPnqAw4f8miPIB5aZr8ALN0FnlgIgw85OouAYybJfPz82HBggXw6aefCl4XFt8yePBgeOqpp6BZM\/4GbGI15IEhD0xjWkKqeggIYFzvgz9LSEoFmJwXp8sOMKnLZhLAyPVFj7B6CGA4A149YJclZ3zhhRegdz2vXJ4AACAASURBVO\/e6J4s1asggCGAIYBxPQMEMOEFMJdfnAGm9l1k+0waT6RDyrIZBDCyWTSyKiKAqWG8mST666+\/FpaUSktLhYSM\/fr1qwradb+cAIYAhgCGAAabVsLBA5P74iwwt5MPYAwn0yF52TQCmMjiDtl6SwDjoynZchKTSK9atQoOHDhQJZtu1aqVZJv\/cACY2RO00BVRIZ3Xx6HWOqHDl9aOq3EZ9VkHrlpileeam6L3OM3JkaSuwFVIjYtx5Q1PhJIbX4DeVxuFq41aG\/B8R6ySJL10byCx4nbqc+g9OjpwFVI7K65Camap3GfI\/UixSrMWi783gkRO33C1kSOGr0KyRXFyHnHUSU6DHb23So+fB47aSKXhlJcxF5LTrsFnAzuuNnJych6pzHg92go8F5K23FOFdOeIkwAONXy\/wlPxY0VyJ4kNzwepmu7XE3YYvbg8aEAQ94G58gIDmK4+zpo1FzOcPAKNlxPA1GwpKoFZgADGz+fCPd8RUyINHz4cbrnlFiGwlwDGZVACGOnDRQDjsgcBjMsWGMC06ZsOLVONigOYq8\/PAYuMAKM\/eQQarZgaNGD5OY1T8TCxAAFMEAOZk5MD7733npC0kcXJsI3sioqKYObMmUKtoboPDHlgqn1cyAMjeUPIA+MyR216YJQKMAVjXgOrjACjO3kEGqycEnJzZRCfDbpURgsQwMhgzOpxMufPn6+qkQCGlpCqP17kgSEPDDbdhJIHpnjM62Btd60Ms2ZlFbqTv0P8ylcIYGSzaGRVRAAj43i77yFDAEMAQwBDMTDiM+BPDIxSPTAlY+aCra18AKM99TvErZxMACPjdyiSqiKAqYXRZnEy6enpwBI7tm7duhbuUDtVioF6tIRES0i8J4yWkOpuCan3DQmw4ZVrPIaiPoN4y0bPkx1gYlZNIoCpnSk97GslgAn7Ifa9gwQwnrYiFZLUJgQwdQMwmpsPwsC\/JykOYMqfmw\/2ttf5PqnUUFJz6jBEvzGRAEY2i0ZWRQQwkTXeXnsrAszcCVq4tgOWzDEavf6MLgU9f0SDe59OO\/ky6hwzLsk+z5FRl5nwJI+pxXjiSV4yx+w4PEFiIieIt7khh2vLJvqL6G9tVfj5jnaOjNqCy7GbW0xo\/ck2XOaeoInC2xpbgZ63x1i4fbNFcZI2GjkyZ45cWqXjLC1pcZk2T0atUvmX5NHpxCXRrMM8GbXThsufnVZcLq0y4+XVJo68GpFR667\/DQbe1xg2TPV8hxxlRu74FDqlEvsDJ2wwcnFJ0IAgzg2mUQtkBxjjmxOCbh9N45FpAQKYyBx3tNcEMJ5mIYBx98AQwIgWiUyAWQiONvJ5YNSnD4PxzfEEMPQdCsgCBDABmS08LyKAIYBhFiAPjPQ5IA8MgDg3WJ5dBI4218s2AapP\/wb61eMIYGSzaGRVRAATWeNNS0icnVtpCcn1aBDA1C\/AsCzUbf9+DKYPbQLTh3ruTF2fS0jWkYtlBxjdmrEEMPQdCsgCBDABmS08LyIPDHlgyAPj+QzUtQdG0QAzYgk4ZfTAqE7\/Brq3XuYCzMmTJ2Hw4MFw6dIldNJlG4hu2LAB2rVrJ\/yelZUF8+bNgx07doDRaIQ+ffrAmDFjIDU1VXK9r+XCc6YPn14RwITPWAbdEwIYAhgCGAIYbCIR5wYbA5hrbgh6rhErUJ05BFovAMN2O\/\/000\/BbJbm52KJdb\/44gsBXJYvXw5JSUlw7tw5Ia0L275i4MCBwDYYXbduHaSlpVWVYff1tZxsnaSKas0CBDC1ZtrQq1icpBaO18H1qArJM+kc6+VJvfSvG7HnxzX4RnYZzlZc42Rb8WSOWSb8HlcrcAVUWklD9B4OTrLAnBg8mWNqFK42SvOmQtLhSRg7qs6ibepow1VIra14IsnmFjyQNskWj9Yfr+eoaKLxpI1el5CicZWQg5PMUWXgqIp0nPMyqZC8qY14D5\/\/KiSOXbkqJF4yR+n581kWaHdvBkwblgbTn2ni0VyVFxVSsVX6bPxywgrPLCkMeolGnBvsw5fKDjCatS\/51T62z9aHH34Ia9euhdWrV0P79u2BnWOJdr\/55hvhfNOmlfNIRkYGDB06VACbAQMG+Fwu9GbvyGwxAUxkjjvaawIYT7MQwEhtYiOAqTKI0xqBAPPMMvkB5u0X\/QKYI0eOCFDy8ssvQ79+\/UClUkFxcTGMHj0aunbtCuPHjxfOscNiscC0adOgoqJCWFqyWq0+lYuK4mw9QN8LRVmAAEZRw1G\/jSGAIYBhFiAPjPQ54MfA1C7ArJ\/WCgbe30hRHhjHsOUAreVbQoKzh0C97gWfAYaByKRJk6CgoEBYFmrQoHK\/p7Nnz8KgQYNg1KhR8Nhjj0lstn79evj888+F5SS29ORLObYkRYfyLUAAo\/wxqrMWEsAQwBDAeD4DdQ0wu\/aXwV8HnwElAoxzKAOYG4OfkwpyAAqzAQpyQPXpXJ8BZteuXTBy5Eh444034M4776xqhxjsy7wsvXr1krRv27ZtsHLlSiHYlx0sKLimcmJQcPAdpRpq0wIEMLVp3RCrmwCGAIYAhgAGm7bEuQEGywQw328AYP\/+OHxJfMsCeV999VVggb3VvS+simPHjgmeFXa+Z8+eXIBhCXd9KUcAExofLwKY0BinOmklAQwBDAEMAYw3gFENXgHQSgYPTGGlB8Z59iDADxt88sCIkPLcc88JAbnVD\/LA1MknQnE3IYBR3JDUX4MCBZizOny9+LgWVyEdd\/IzdGda8TxJPBXSFU4upLRSXIXEzYUUexU1fJoxFz\/vRYXUXIerijqpzqF1deCpkCy4CqmZBc\/\/k2RPROuP5aTOcXJUSDavuZBw9ZCTp0LineeojdRaXGHFy4Xk79viVZ3kqAz8dD8cNlx957TwznNyJJk4KqQK6fn\/7iuDvw46CzvXtIPe3TzzW6lK+bmQSi1OSfN\/OWGBoUvyfQIEb7YU5wbVoBWgkgNg\/riZ89xBcG583qf2bd68WVAYVd\/3RWwzxcD4+yaER3kCmPAYR1l6QQDjaUYCGKlNeNmoCWBcdnJawhhgBq6UH2A2jakRYNjy0SuvvAJsCYjFr7irhGpSIbE9Ydh1drvdqwpJLMf2kqFD+RYggFH+GNVZCwlgCGCYBcgDI30O6toDs+VfBTDslSxFemDUDGBayrCEJHpgzh8Ehw8Ak5eXB8OGDYO\/\/vWvAoC4H7QPTJ19JhR1IwIYRQ1H\/TaGAIYAhgCm\/peQFA0wA1bJDzCbR9fogTlx4oSgHmJemPvvvx+dKE+fPi1sWGcwGIT\/ijvxpqSkwIoVKyA5OVm4ztdy9Tsb0919sQABjC9WipAyBDAEMAQwBDDYdCfODZqnGMDcJNuM6Dz\/K9jfrRlgxPvXpFa6cOECLFy4sMZcSL6Wk62jVFGtWIAAplbMGpqVEsAQwBDAEMB4BZgn35AfYN57rkYPTGjOqNTq2rYAAUxtWziE6q8JYC7p8ODEU3o8HxGpkKSDTyoklz1U9aRC8vY68nIh1XUMzOw3cmHOG7lw+ouu0CpN79Hk+lQhaRnAtJDRA3PhV7ARwITQV0JZTSWAUdZ41GtrCGDIAxPuHhgCmMCmGHFu0PZ\/U36A2TqKPDCBDUvEX0UAE\/GPgMsABDAEMAQw9b+EpGgPzBOrQS2jB8bBPDDvP0sAQ9+hgCxAABOQ2cLzIgIYAhgCGAIYbzEw2scZwHSTbQJ0XDgAtg8IYGQzaIRVRAATYQPurbsEMAQwBDDKARj7fjzWpD5jYHSPrQF1cxkB5uIBsG4bSR4Y+g4FZAECmIDMFp4XEcAQwNQVwLy3rQh+3G2By5e1YDKZ4akno+CpJ2OApRLYsqUCfvzRCjmXdWA2meCpp4wwYJBnMKvcb6FSgniHvZIJW\/5VCEoEGH2\/t2QHGMuHIwhg5H6YI6Q+ApgIGWhfulkTwFzU47lfTupT0eqPa\/BcSBnOVtzmZFubor\/xciFdrcAVUKllDdB6eLmQcmIK0PKpUTno+TQvuZCa6LLQazqqzuLnObmQ2louo+WbWO3o+RQb3udYPZ7jxxlrQuuxR+H5iAS4icZzITk416j0nm19bX4BnMu8D3r37l11\/127doFa9Tm0bKWCs2f\/4fGbRvMpvL0u3qO9KhWeF4r3gHnLhcQDGKcNz2HktHLOm\/Hzal4upHJp+WFTMmHz5wVg34t7OlRl\/FxIxVbp2P1ywgrPLCkMGhDEucHw6FrZAcb80fCg2+fL\/EZlws8CBDDhN6YB94gAxtN0BDBSm\/gLMJrom0Gj61lVyfkLNvjtSCto1coTYtPT04HloKn+26VLlyA7OxsY4Awa\/DXccYfUE0MAIx2fWgeYR9aCptnNAc8x7hfaM38B88cEMLIZNMIqIoCJsAH31l0CGAIYZgE5PTD6hNGgi3oh4LfswIEDwP6xY8+eF2Ht29LszAQwdQswxofflh1gTJ88Qx6YgN+QyL6QACayx1\/SewIYAhglA8yyZWPh4KFoySCFK8Ds2l8GZz67Fp2d6nMJyfjPdfIDzKfDCGDoOxSQBQhgAjJbeF5EAEMAo2SA2bp1DHy7IzHsAebuwWfgfJZVkQAT9c93QNNUxiWkrF+g4tOhBDDh+Ump9V4RwNS6iUPnBgQwBDBKBRgWA8MCeVUqFeTk6MBsNsFTTxthwAD\/1EmhEMSraIB5YD1omt4i26Rmz9oPFZ8PIYCRzaKRVREBTGSNt9feigAzd4IWru2g9ih7US9134sFzug4uZACUCHlmJuhbcy0JKPni3kqpGL\/VEjZcflo\/Y2iOEogQy7Xlk30F9Hf2qrw813suDrpGgt+j+YWXD2UbJPGh4iNSNBE4W2NrUDP22Ms3L7ZOGojhxFXRukSnwN9zPMBv2Us\/mXJkiXCB65\/\/\/6IOukzeHutpzqJd8OAAMaO5wBzWnC1EVjw8poKvLy2XKruu3voaTh3yQJnP74e7YaznK9CKnCUSa45cMIGIxeXBA0I4twQ9cAG0DSREWAuMYAZHHT7An7A6MKQtgABjIKHj6V8f\/PNN2H79u1QUVEB3bp1g1GjRsFtt90GarUUMMxmM3z55ZewevVqOHv2LLRu3RqeffZZ6Nu3LxgMBp96SQDjaSYCGKlN6hpg3n\/\/fdixYwdYLBZIS0uD9u3bSxokqJMGbvdQJxHAVFpAdoDpu1F+gPlyEAGMTzM0FXK3AAGMQp+JEydOCADicDhg2LBhwuT91VdfCf9ee+01eOSRRwR3OjtsNhusXLkS1qxZA4MHD4bu3bvDTz\/9BJs2bRLqGDNmDGi1nL8Wq\/WfAIYAhllAaR4YUYXEnueBAwd6DNKe3S\/CWh+9MKHigWGd\/H5FJ8V5YKL7MIDpLtusab+0D8q\/IoCRzaARVhEBjAIHnHlTXn31VUE+yjwq4l+d7PysWbOAgcY777xTtV\/G7t27BVCZOnUq9OvXTwAbp9MJH374IcyZMwc2btwoeG9qOghgCGCUDDDsWWbPuPuxbOlYOHgwpqbHW\/g9FACm7d+PQasmeoUCzCbQpMkIMNkMYAaSB8anp5cKkQcmBJ6BvLw8mDJlCjRr1gxeeeUVifdk27ZtMHny5KoXnnlf5s+fD3v27BGghnlqxINtADZ06FBhyWnixIk1emEIYAhglAwwLBbm5Zdf9hikre+NgW+\/xWOe3AsTwAQ2AYpzQ\/R9tQAwXxPABDYqdBV5YELoGWBelVWrVsHmzZthw4YN0LVrVyguLobRo0dDQkICzJs3T9jJVDxY3MykSZOgoKBAuC4+3nuwIwEMAYxSAYbFuvBiYDSaz4AtplbPnfT003jgMgFMYBNeFcD8fbP8HpjtA8gDE9iwRPxVBDAh8giUlpbCv\/\/9b3j99dfhoYceErwwLDhX9LL06tVLOCfGxVS6y50wd+5c2LlzpwA8zZs399pbcZKaPUELXTt6qpDO63GVyxmdy+tT\/QbH1XgupJMOfi6kPAte11kzrnRyluPnG5UkoH3l5ULKiytEyxs5KqTmhjyuLVP0mehvbdS4Cqmj4wJavp0Vz6nU0ixVmogXp9jwj3Yi4OCqjcHVTI4obyokPBeS3Yif1zZ+FvSxtadC4qmT1Kp\/wdo1DT3t6sTzQgnvi8PzmRcqsPFUSPh5lZmjQuLlQnJTJ7XpcxRapenh+2Vd0OfCWsGXjudDkeSaX0\/YYfTi8qABwQUwW0CbKt8Ski1nH5Rvfzro9oXINE7NlNkCBDAyG1Tu6tgS0ezZs2HLli1C1SzGhcUBxMbGCv\/\/yZMnhcBdpjZiy0TuB1teYuokBjDt2rUjgFE5URsQwLjMoiSAqa5CYt5G9rwzcGfPP4sN86ZOGvDUd3DH7W4KvBABmN7dYmHjBPx9rVeAubcWAOZbAhi5vxuRUh8BjMJHmk3Q+\/btA7vdLiiLGIj06NEDFi1aBMnJybUCMI89oIGuHVSQ3Ljyn3iQB8b1sJAHRvri1KYHpiYVEk+dtPvHl+Gt1W6xMSEAMNqbDsGAvg2DBpjsqw44eMIOczaagvZwiB6YmHsYwPSQbda05eyFsh0EMLIZNMIqIoAJoQFnS0LfffedEPPC5KTM48L2fJHbAyOahIHM4w+43OEEMAQwtqi6X0ISAYanQuKdX7p0HPy6320pMYIA5p0vzfDOV5XLgWyprWdPV1Zwf6c9EWBi\/\/ouaFNkBJjLe6H0u6eCbp+\/\/aHy4WEBApgQG0cxaLekpERQHTFpNVMaeYuBYTJrd4US1m1xkhozVAvJjYA8MABAMTDSJ6U+AYanQuKdf+\/dF+CbrxtLOxBBAMM8MF\/\/zypAjFwAE3e3\/ABT8h8CmBD7DCmmuQQwihkK3xoiKovYLr0MStgGdcwj06BBA0GFFBXlCuQMVIVEQbyusSCACQ5g1PHdQK33DPpUqfFYpC+2N6va34jd+dKlS0KgujcVEk+dFKoxMHItITH7yR3EG3fXe6BLls8DY83dCyX\/92TQgOXb7Emlws0CBDAKHNHffvsNxo8fD48\/\/jgMGTJE0sIrV67A8OHDoXHjxkKOGKPRKPs+MFMn6aEzokI6p0dUHQBwWtsUtWKGClchnbHz1VD5ZlyFdNKM50KKLU9C751YjCtveCqk\/HipekOs1Bx9Ba2\/rQHPkcQKN9Jno9e00eBqow5O\/Hwb6yW0nlaWAvR8mhVXvzRw4IqsaI6YxRlt5r4V9mgr+hsvBsahx3MkOTnnR7x0GZzqvh45j3hqI975Pbs\/hKO\/IXm1HF5USHaOCsmOX6Pi5DxS83IhcVRImmq5kM5lW6DNg7\/DtGFpMOPJ1qitSy04\/LHC+Wrpc3wowwEvLTYHDQiidzbhTvkBpmgnAYwCP0Mh0SQCGAUOE9u35bnnngOr1SqkCEhNTRVaKe6uy+TSTB792GOPCed5O\/Gy\/WJYsK+\/O\/ESwLgeCgIY6QtS2wADGif8uKcC5i4qAIPRCClJdnCy\/1P3rUqn4a5C4qmTBjyxE27v5Zb4kAAmoBlPBJjEP2+V3QNT+EP\/oAEroE7RRSFvAQIYhQ4hgxIGMQ0bNqzKhfT111\/DF198Af\/85z8lUmoxFxJLO8CCe2+99dagciERwBDA8F6LugAY93vf99AleHLAMu6bylMh\/bhrHLz1RiPpdQQwAc14IsA0+PNW0CfJt4RkydsLBQQwAY0JXQRAAKPgp4AldGQ76LL1fxa026VLF3jmmWfg3nvv9cgwzYJ5P\/roI0FmHWw2agIYAhglAcxDT+TDI\/3mcd9UrgppyXg48LPbsicBTEAznggwDe9gABO4msn95pa8nyH\/vzV7YJhXev369fDBBx\/A1atXhbmQpZXo3bs3qNWupb+srCwhFpBlMGfL63369BGS2YpebPH+vpYLyFh0UZ1ZgACmzkyt\/BuJkxQBDAGMkgBm5PN5cPtfFnJfIK4KacuLsP1Lt52aCWACmojEuaHR7VvBICPAmPN+hqs\/egeY3NxceP755+Hy5ctV3uivvvoK2L9ly5bBfffdJ\/Tp3LlzQnwgS6fCPNFlZWWwbt06Yelx+fLlkJRUGS\/na7mADEUX1akFCGDq1NzKvhkBjOf4UAyM1Cb1sYTEYmK2fPRnSWCv2Cpv6qRQjoFZP60VDPpLE3TCqM8g3sa9toKhsXweGPOVn+HKbu8Aw2L52DIhWyJnuy+zg3mcX331VSEXHIvzY9DCvNXffPMNrF27Fpo2rRQWZGRkCNtMMLAZMGCAEEfoSzllz9TUOtECBDD0LFRZQASYsZOjoUMnrYdlzuhxJdApDa5COunEVUiXbPjEzG54xYLnNjpjwu+dVO62z8cfrY4vdCW1rN4RhwZXcJTEl6JPwtXoq+j5Nt5USJzfmmtxVVE7wFVIbW14TqXWVjwPUxMLnsOosR23RTxUpqNwP7RRfBWSk5MniadCshtwFZJD60Dv7dTh50eMzQanxnd10u4fP4Sj+\/g5t9Cbc1RIKiuuTlLb8PMantqIc15VLbfRDweL4c4XjsKGyW3gyb\/g71UhFHNnrTxtheS33zMcMHmBLeggWXFuSL5NfoDJ3cMHGHHfqz\/\/+c8eiszqHRXLsQS3TMEp5oRjO5lPmzYN2JYSbGmJCSPYthM1lau+HQV9IpRrAQIY5Y5NnbeMAMbT5AQwUpvUF8CAxgE\/\/q8C5i5m6iSDS52k4auTnn54F9z+JzzBJQGMf9NLFcD8aSsYG8nngTFd\/Rly\/8cHGDHXG4MPFgfz5ptvwunTp4VdhceOHQvXXXedACss7m\/QoEEwatSoKnWm2EMWO\/P5558Ly0ksKa4v5cTlJv+sRKXr2gIEMHVtcQXfjwCGAIZZQIkeGAYw7sd9D1+CJwcu5b5RP+4cD2uW4Z47Ahj\/JiJxbkjtwQAmeBWSrSITbOVZwP575fAErodo\/\/798OSTT0Lbtm1Bo9EIy0FMRs+2hjh+\/Di89dZbAsxUBx22K3n1Y9u2bcJ2FEzgwA6WeoUBkbdyNSW+9c96VLq2LEAAU1uWDcF6CWAIYEIJYB568io88hhfnbR08Xj4ZVdoAszO5Z2hVyd8ebQ+l5DSGMA0DB5gCk4uh8JTK6peOF6qA3FO6tatm2RPLOaNeeGFFyA6OlqIgbl48aLgWWHBuu45n6oDDNtywpdyBDCh8QEjgAmNcaqTVhLAEMCEEsCMfDEXbr+Tr056b9NL8PUn\/Hgrj9FWUAyMYgHmlq0QJQPAMM+LtSILTPk\/Q8HpFVwPjDgnMck0i12pfjBJNQvsZd4YBia+eFbIA1Mnn5I6uwkBTJ2ZWvk3IoAhgAklgGExMVs+6c1VJ4ViDMym7XkweO5pUCzA3PyeLAAjvmkV+Xsh+xd+KoEjR44IYDJu3DiP2JbqnhWWE86X2BaKgVH+d8ifFhLA+GOtMC9LAEMAE0oAw9raufs56HV7P4\/cSaGqQlI6wDTp9h5ENQh+CakKYAr2wqUDfIBhiTxZ3AtTIVVXF7HrWXDuxx9\/LCS1ZTJqb+oiticMi3ux2+0+lWP10aF8CxDAKH+M6qyFIsCMmpoIbTrpPO57Wou740+pcbnnaQeetPGKtTK3E3Zc4iRtvMKRUaeWNUDriS7iqE84ufzKEsrRenJi8tHzacZcbh9SDPhvyboc9Jo2qovo+XYOXEbd0oYnkmxmweW1yRwZcKIDl1HHaQzcvqk4EmungZPk0U8ZtUPHSf6IyN9\/\/LkcNn\/Zi5sjaUDf3XB7z2hJX1ROfjJHwBXcoOYkyfRXRq0yeb5TrHFOsyur5qZvL8OQhSdg5+Lr4Kaubnmc\/ujJVU0Jd3xy3W5x9LgD5syzyCajbnqT\/ACT9SsfYNjS0Pz582Hnzp2SfWBycnKEHXZbt24Ns2fPBr1e79P+LrQPTJ19TurkRgQwdWLm0LgJAYznOBHASG2iJID5+xMX4cnBS7gv14\/fTYS3FkphmQAmsLlInBua3vguRCXK6IEp3AtZB5\/yClgspcqzzz4LDodD2ImX5YdjcS9XrlyRQA2TV7MN65hKif1X3Ik3JSUFVqxYAcnJlQHdvpYLzFJ0VV1agACmLq2t8HsRwBDAMAuEigfmwcG58MgTc7lv1dJFE+DAjrSQ8sDM3HweZm2+AKffvQUaJeHeovr0wDS7QX6AyTzkHWDYALLcRUwKvX37djCZTHDXXXcJ+8C0adNGMr4XLlyAhQsX1pgLyddyCp+yI755BDAR\/wi4DEAAQwATSgAzYnwO3P7X+dw3+L0NL8P296XLmEr3wCgeYK7fIrsHJvO3p4Ne4qJpPDItQAATmeOO9poAhgAmlABGjIFhGYndD5YjKRRjYBQPMNdtgagEGZeQivZC5mECGPoMBWYBApjA7BaWVxHAEMCEEsCwtna+\/TTcdsejHiqkPf\/9CI7+KF1eYOXJAxPY1CXODc2u3QLRCd0DqwS5qrxoH2T+TgAjm0EjrCICmAgbcG\/dFSep\/tOaQovOniqecxppPIFY1xnAVUgXbJxkdBb+7qinLZUp790PXUUj9HyjkgT0vJGrQsKTOVYkSJPgiZXmxRWi9auMeJJHVri1AU+22ECPq4daavAkj9cAJ5mjLRttUwsr3qZUqwkt39COq2LiHHHcxyRG45nkUwADA55I0qm3oXU5OWojByeZoxNJJfDffWWwYcetXBXS4Ht+gju6S+WwKocXFRJHoaTmqLhUHHWSyoLbqLraqLpRyuwuG83dchHmvpsJxd\/eCkVqXFV2VYvblNWZo5Oqrk4cs8HS10uDXqKpApgutQAw6QQw9BkKzAIEMIHZLSyvIoDxHFYCGKlNlAQw9ww6B\/2HLua+i3u+mQRvvyaFaAKYwKYuF8Bshuh4GT0wxfsgM31A0IAVWK\/oqlC3AAFMqI+gjO0ngCGAYRYIFQ9M35HZ8PCTr3PfgGULJ8ChL9yCeBXugXl20Sl477s85XpgOm+SH2CODiSAkXEej6SqCGAiabRr6CsBDAFMKAHMM1Oy4La\/fIuBfQAAIABJREFU8ZM5bn1nLOzY2EoyqEr3wCgeYDpulB9gjg8igKHvUEAWIIAJyGzheREBDAFMKAGMGAPDUyFpyr8GlUoFOVfVYDaZ4ekHE2DAPxryX14FxMCEBMDEybiEVLIPMglgwvODUge9IoCpAyOHyi0IYAhgQglgWFuHvZIJ9uj7PFRIW7duhf79+3uc15Zth3WvNcNfSYUAzI+Hi+HI5psUGcTbvP1GiI67RbYprbxkP1w8QR4Y2QwaYRURwETYgHvrrggw903vCGldPJUoZ1V4DqPzDjxH0lUbrja6YmnMbcZFE65CSqrA\/3KOK8aTrhkLpWqMqhuqOCqkRDwXUkl8KdrWK1F4jiRWuLkBVwM11l9B6+LlSGqlykLLt3LiOZWa2\/AcTE1suJIqyYorh3jqJNaYGAdu12gVnrdHq+MoZnjqJERtJBiBdx4Adh0ogVlv54DBaITUhg4AJ4A1\/l5uluphd+yH3t0QpZVdjdqbpzYCmwYtb7PiKqRyJ64GK1GXVdXTb9w5uJhjgZ\/ebQ+FGtx2eTpX7iT3BlzSJkpOnT1qhvVz8oJeohHnhubtNsgPMCcHB90+msYj0wIEMJE57mivCWA8zUIAI7WJEgHGfdTuGnkSnhi+iPtm\/++rV2D9tJaevxPAcG3mApj1EB0rowemdD9cPDmEAIa+QwFZgAAmILOF50UEMAQwzAKh5oFxH7X7XrwI\/xwwh\/uSLlswEY58cA0BjB\/TWBXAtHlHfoA5PZQAxo+xoKIuCxDA0NNQZQECGAKYcACYIbPOw5\/68OXV768dB\/+3ph0BjB9zXxXAXLMOomNu9uNK70XLy36Bi2eGEcDIZtHIqogAJrLG22tvCWAIYMIBYFhMzLr\/3hKyMTBsDD5c1EqZMTCt3pYfYM49QwBD36GALEAAE5DZwvMiAhgCmHAAGNaHITPPewTysgSPTJ007Zk0OJ9jgZx8Jq82wYA+DWFgn0YACoiBufWpE9A8Va9cgGm5FqJjusk2AZaXHYCL54cTwMhm0ciqiAAmssbbJw\/Mn2bcDI26NPAoe9aJ5zbKtePKoUJOXqMLFv5eHE4TnvOocSme8yiqBFe\/GAtj0b46OSokUyKuNqqIx5UjuTFFXFtqDbhCqZUeVycl6nH1UIoGP99ChauQWjvwHEmpdrw9KTY81w5PncQ6HGfHlTexTs44cM7rVbhSR6PGVWIqjR23N6c8K\/zDoWKYtTETjAYDpDYA6H1DPAyeewqVV+vy\/wPrx3VA72Hn7N5rceIqIZMKf2ZKOecLta6+3ffEGWiSqoN1S5tDvhZXG13WxnOfvUyN9F3MOloKn888HTQgVC0htXgLoqNlBJjyA3Dxwoig20fTeGRagAAmMscd7bU4SRHAuMxDACN9VEIJYNwfcgY06\/Zez19a6nYEel\/vCcoEMABVANN8jfwAc3EkAQx9hwKyAAFMQGYLz4sIYDzHlQAmfADmzhePwhMjF3Bf3v99Ou3\/2zsT6CiqrI\/fzt5JSMjOkogMgixREJRl4FNxRNkVHdTDKMgikLCJLGEnLEIQCLsgIrIMDCCKis5x\/BRFBXQUEQUV+FCEQALZQ0Jn7f7OfdghnX6VpdNJuqr+dQ4nh+5a3v29qlf\/fu8utHVaK7vvIWBuCZjIyI3ka3TiDIzpOCUlxUDAaPOVUutWQcDUOmL1XAACBgKGCWhlCal8b\/aNO0+Dhi9SfCBXJ8ygU69H17uAubeDLy2Ma+SSS0iRTV8lX2NHpw1qN0zfU9LlWAgYpxHV14kgYPTV3xVaCwEDAaNlATMi4Tx1e1xZwPxrYxx9uuKuehUwHXqeoYG9A11XwDTZ4HwBc2UcBAzeQw4RgIBxCJs2D4KAgYDRsoCpzAfGM+1TIgNRShZRQUEBDe0VTsMejaC6XEJyeQHTeD35+tzjtAHwRv4JSkoeDwHjNKL6OhEEjL76u0ozMK3iH6YG7SLs9k1WiDbKKZJHDl1ViDa6VkEUUpjJPvqJG+J\/XV6DxyfHKLXJs5pRSEUKUUj5ClFIuQ3kUUvcmDQfee2hJgo1ksI85VFCAZ7yqKUmCtFJTUghasks\/zysRN5Opegkti2wpEDKO6BEHj3kZ5ZHG3lbPKXn8bDI9\/cg+f7VfXy52rMlrFeViz9aUv9Da6ZFSS9zwyCvJZWnEDGV52aQnifD49Y93LvHCerVJ5imzG5Gae7ySLpkd\/nzxie\/5GZbfyz9dCYdjf+uxgLB+uMmstE65wuYlAk1bl917wPsrw0CEDDa6EenWGEdpCBgbuGEgLG9tdQuYNgarvac8M+b4dXhXPvQYrETNVarOXfM4Hv+S93b24sJXQqY8DXOFzDXJkHAOGUE199JIGD01+eKFkPA2KOBgNGegCnfy\/2m\/0zPxipHJ3329ixaO9V+FsbZAuZqciENG3yanh3RiJ4d0dg1Z2DCV5OvdwenjZo3Cn6gpGsvQsA4jai+TgQBo6\/+xhKSQiI7LCHdujW0vIQkewCenHue\/j5S2bk3MSGOvnztdrtDdSlgwlaR0YkCxsQCJnUyBAzeQw4RgIBxCJs2D8IMDGZgmIDeBEzMyvN0\/5PKAmbHq1PpwPIWEDBDhlBkSCIZvds7bQA0FZykpPSXIGCcRlRfJ4KA0Vd\/YwYGMzCl94AenXhlDwD7xOw6oZyh15z2EXF4Umq2gfLzC+jpXkH0zCPBVNEMzLv\/yaZvT5ooPctd1Ft67NEA6tWHHW7sN6sTryqWkEJWkNHLiQKmkAXMVAgYvIccIgAB4xA2bR5knYEJm\/cEebeNtDPyerE8Qii9SF7bKEkh2sg\/X17XiC\/YMK+BFK5Prrf0c+9sP+nnnpnyqCWLQu2c4oY3pOfJD8yTf+4vj8bhnTP9rkuPMXnLo36ivDKl+wd5yj8PVIhOauSWJj1PBMmjmRpb5J+Hl8ivyycPMsujrxqWmKTX9i8pkn5uNMujljzlH5Onxa1WH7rpr1wh79B+VY5Oykv\/kGbMaiJt06Zt6VRU8LDduZKzPqYJc\/5id0ym200H4bTkfIobdJQGjmpOj41qTsnuodLzXzPIhRDvfIVso5Cun75KZ+M\/qbFAKI1CCn6FjF53O60vTIU\/UlLG9Bq3z2kNwolURQACRlXdVbuNhYCx5wsBY8tEqwKGrTzyYy5t2JFO3t4+FNqwhFhLGUNsRY2VBkcnde9yhDp2sBXK3\/9wg458012x3lLrv35P0R1tizGWFzAj5rah7v0au6SAaRqU4HQBczlzRoUChnPyzJo1iw4cOGD3gI4ZM4bi4uJKP798+TIlJCTQxx9\/TD4+PtS\/f3+aMGECNWrUyObYqu5XuyMuzl5TAhAwNSWooeMhYCBgmIAeZ2DY7iKD2eYGGDblEo2IWaX4hL\/\/bhzNmdHY5vtxL16kMWMTFY\/Z+\/4cGj+7uc33VgFz5vtMeiX2BLm0gGm4lIxe9tmKHR0GTYU\/0eWsmRUKmIyMDBo5ciSFhoZS27ZtbS7Vpk0b6t27t\/jswoULNHr0aPLz86Nhw4ZRXl4ebdmyhRo3bkxr1qyhsLCblbqrup+jNuG4uiMAAVN3rF3+ShAwEDAQMLfugdGzUukfI5YqPrfLlk2nXW\/aLiNNibtKw4YrH7NkWRyt+qdtwUhVCZjAl8noaV8vytHBzVR0ii5nz65QwJw9e5aGDx9Oy5Ytox49ekgvZbFYaP369fTRRx\/R5s2bqWnTpmK\/M2fOCPHDwmbo0KFU1f0ctQfH1S0BCJi65e3SV4OAgYCBgLl1D8x6JYUefUI5P8xrm16iDatvs7lpFick08DHlyk+5+tem0YL17dW7QxMk4BFThcwV3LmVihgvvrqK5oxYwa9+eab1LJlSynbnJwcGj9+PEVHR9O0adPIYLiZ9biwsJDmzZtHJpNJLC0VFRVVaT+jUZ7h26UHcB02DgJGh52uZDIEDAQMBMyte+C\/J2\/QR9\/9j6I\/S3bGh0QGA2VmulF+QQH17R1IycmFZCp8RPEYduTll2tOJkc05VPPvqHUof\/NHDPWJaTpr95Dd3YMckkfmCYNFpDRs53TRk1T0Wm6cn1+hQJmz549tG\/fPnruuefE7Mr58+epa9euNGXKFLr77rsFz99\/\/52ef\/55io2NpaefftqmfVu3bqX33ntPLCfl5uZWaT\/rcpPTDMWJaoUABEytYFXnSa0Cxm\/2MPJoY5+4K7NIHoV0uUgeVeRRII+WCDbJI42YmlEp2ui6\/BeRR7a8XkxAhrx2jllejoZyg+V1bYoD5FFIBQ3kUTdsw40G+dIbIM0oj04yKEQnRXrKo5YCPLKl52\/gIY8eCnGTfx5hkEchhZH8unzREHOO9NoBZjmnBmY5J1+LPIrL21wiPb+XpVj6ubtFHrZU8ucv8Oo8iYUG+zpMy5dcotCg3lWOTtq9e7e45JAhQ6p8zE8531D\/+O508XgK7RrzMf3jtUfotk6N6BrJn590i3IUUqrZtk5Swc9JlL5wf42jfKxjQ2P\/uWT0sPVDqQ7j8vuain+m5NxFiu0rLi6mRYsW0c6dO6lnz540ePBgIfy2b99Ov\/32G23YsEEsK507d04sM\/EsS\/llpr1799K6devEDA5vVdlPaaanJrbiWOcTgIBxPlPVnhECxr7rIGBsmehNwLD13528Qf\/amiKikwKDzCI6KaLho4qzLD4tTlBoYx96\/40L4hh\/zjJgIWoW2FPxGMMDaZSdnEcfxB9xbQHjN5uMHm1qPMYVm1OpyJxG\/DfVtFlRwLAj7uLFi8XST3x8PPn73\/zBkp2dLZaKeEaFRUxKSoqYWWFnXZ6dKbuVFTAsiKqyHwRMjbu4Tk4AAVMnmGt+EX7wEhMT6dixY\/TGG29QcLBt7hUONTx48CBt3LhRTKc2b96cYmJiaMCAAeTtLc+hUr5VEDAQMEwAMzC290GBm7vNB7MmnKNxY1YoPtQ7DsbTiDm2L\/nl407Q5NEJisds+GA53dYpwvUFjO9M8vGw9eFxZHTLLDhAWQXvlh7KM1flhUdl52VhsmLFCjGzwmNcVWZWMANTGVV1fQ8Bo5L+4rwT7KR2xx132AkYFjc8Rbpp0ybxEHfu3FkIHZ5mZRHDeRA8POynxyFgbhHAEtItFhAwFQuYBVMv0ajnFyuOHItfmUnxu2wLHq596ReKHRaveEz8K3Oo03OtVCBgppOPe80FTDHPvljSyFR8hrIK36twictsNhP\/Kz+GsXMvRxax+ImIiKiSbwt8YFTywqtiMyFgqgiqPnfj6VEWIcePH6f27dvbCRh+kFmozJkzh5566inh1Mbhguz4xtOv27Zto06dOlVqAmZgMAODGRj7e6D8DMzql\/+gwY8tUXyeVm2eQdM23GPz\/dbFv9DQAcoCZsnr8+mu\/i1cXsA0Mk4jH\/c7Kx1LqrpDfskZSjEtVxQwVt+WESNGEP8ru7FzLv9I4\/EtJCSkwugiXopi\/5iSkpIq7ce5ZLC5PgEIGBfvI+vsytGjR8Wy0bVr12wEDH\/P+RGOHDkiPuekTdYtOTlZ5EDo3r27yFZZ2SwMBAwEDARM5QLmpxPX6aej9yn6s0T2+EVEEZXdOMIo6as2isewI29OSh5dPZVFEdENhZiJ6C\/\/0VGfTryNfKaQj7ttHpuaDKH5JWcpJX+looDhGZOpU6dSVlYWrV27lsLDb5ZKsP6o46VydvL18vJCHpiadIRKj4WAcfGO49kVfoBZpLDA4H9lfWCs+Q8CAwPFL4yyvxw49wHnT8jMzBQPd0CAbQrz8qZbBUxJ3ASytL7Djkxykfx4Q6E8CikkXx4hZMzzUaTulSv\/zitHXtvIO0u+f2CG\/BJKJXWyg+XRLIWB8oiiwgqikAoUopBMfvKInDRveXSSu7c84qexh\/zzhgrRSb4K+we4y88TYpBHOTHRhiQ\/JpDkUUgBFvnnfhY5C2+LvHaSh0UeneROttlzK3ucS0i5plKxwdbXxXquGwb7e+xfC3+gOwJsQ6x5mTfF7yR1GSMXHu\/E\/5c6BXSscnTS4ZxfKXr+3+1MyipRjkK6UWz7jJb88juZlmx1WhRShPeLThcwVwtWV5oHZty4cdS6dWuxTMT+fhxOzX\/Z569Vq5uCisOrOWEd+8PwX2smXl5eKit+qrpfZfcSvq9\/AhAw9d8Hii1ITU2lSZMmiQeXa4GsXLnSTsBYZ1k4dHDmzJmlCZz4pLyMtHTpUjp06JBwdIuKiqrQWggYezwQMLZMIGBu8Tj1\/XX67PWfydvHh4zB7nR7xzDq2P92ypMIHj4qgxrQleOX6fjr35KPtw95BnvxQ2onaqxXYEF07n5fCu5kWwCyfgXMRPJxkyeTc2QozTefo6sFaysUMDyO\/fjjj6XjHyeZ69Onj1hWt2bctV774sWLtHz58kprIVV1P0dswjF1RwACpu5YV+tKvDTE4YFclMyaGts6C1N2Bsa6RszRRmWLmlkvxsdwdFJFWSyt+1oFjPmx3mRp3ZIsocFE\/O\/PDTMwt7oQMzC2t7PeZmDY+jyDPDdRRQKm\/CBwcOy7NP+FeYpjw8IPNlP0vCdtvq+qgDGnZRHPwBRsfsdpMzDhXhPIx81+drZag1uZnfPN\/0fXCtfVuH2OXh\/HqZsABIyL9t+JEyfENOiCBQuob9++opV1JWCsSFjImB\/vAwEjuUcgYCBgnCFg\/nfSRxQ3dLriKDTzlQV0357xDgmYwncOUeGBz8SxjoQpl72o9cdNuGes8wVM0as1bp+LDuNoVi0TgICpZcCOnJ59VnjpiJd8uI6HNY9LXQmYkpH\/EDMvWp6BKTYl0fXL71CDpk+QhzGS9OADU5KaQ6bDp8j4QDS5hwWQHnxgslKL6YfDudThAX9qGOZBzvKBcdYMzOcLDtFL\/V9UFjCbl9F9m0ZVS8DwzEvxF9+Te5vmYgaGRYyzBEyAey\/yMbRwZFiTHlNMmZRRvLfG7XNag3AiVRGAgHGx7uL1Xs7nsn\/\/frF01KLFrcFCJmCq4gPDjsDlI5RkZpdfQiq\/T3qx3JHWUCwPOQwolE+xe+d7KVL3MMm\/81Rw\/PXMle\/vJ\/eLJfqzlAALmNRTcdSwxUQyBneh3AC5E2+xn7zEQJGvPBU+G1ZklB9ToPB5tucNKQ93T7kDbLC7fH9\/d\/n+Pn9+zgIme9NH5P9kN\/JqG0W+bvLzBBpyFfvHn+TH+JHc2dlXwVnXSHJGnoolA+TOum6VOPGygHlvUzo98GQg3d7Wh8wVOPGWGOQOvvkG+T2WT\/IEkUr755D985B8\/Ao1\/C1ApEcov508eZIu\/MVCQR1tfWByzXLneD6+oMSXWMDwspHXoJ5kCAtyyhJSUlKSyHz7zTffOH3E7NKli\/BbiYyMdPq5cUJtE4CAcbH+zcjIEKHPPHhVtFnzwXBoNCe4CwoKElFIZauoVjcKqTYHKRfDjOaAgG4IOEsg8PjA\/5y9sXCBeHE2VX2cDwLGxfqZQwPZ457\/lt84dfbPP\/8swqpDQ0NFJVZ3d3en5YHh69XWIOVimNEcEHApAn\/88YdIyObp6Skiax588EFq1qyZU9oIgeAUjDiJCxKAgHHBTlFqkmwJifdVysS7Y8cOUSukqpl4VYQCTQUBEAABENA5AQgYFd0ASgLGmq2XkzoNGzaMunXr5lAtJBWhQFNBAARAAAR0TgACRkU3gJKAYRN4yemtt94S+V4crUatIhRoKgiAAAiAgM4JQMDo\/AaA+SAAAiAAAiCgRgIQMGrsNbQZBEAABEAABHROAAJG5zcAzAcBEAABEAABNRKAgFFjr6HNIAACIAACIKBzAhAwOr8BYD4IgAAIgAAIqJEABIwaew1tBgEQAAEQAAGdE4CA0fkNAPNBAARAAARAQI0EIGDU2GtoMwiAAAiAAAjonAAEjM5vAJgPAiAAAiAAAmokAAGjxl5Dm6tFIDMzkyZNmkT9+vWjp59+2u5YzmJ88OBB4lIMas5ifOXKFXr99dfpgw8+oPT0dAoJCaHHH3+cRo8eTWFhYTZ2a8Xm1NRU4ppfe\/bsETa3a9eOXnjhBXr00UfJ29tbkzaXv4EtFgvt27eP1q1bJzJxt2zZUhd2V2sQwM6aJAABo8luhVFWAtY6UTy4L1261E7AWL\/ftGkTDR8+nDp37qzKOlJnz56lmJgYysnJEfWwOnToQD\/88ANt376dgoKC6LXXXqPmzZsLLFqx+dq1azRx4kQ6f\/68sPmuu+4ShU3ZZmYxYcIE8vDw0JTNsif7zJkzNHLkSPFVeQGjlb7GiAYCMgIQMLgvNEuAZxl4QE9MTBQvbZmAUarkzb9oFy9erIpK3oWFhaKthw4dojfeeIPuvPPO0j49deqUeLkNHDiQ4uLixAtdCzbzrMP69etpy5YttGHDBurRo4ewmT9nMfrqq6\/a9J0WbJY9qLm5uTRz5kz68MMPqUmTJnYCRqt2a3bQgmHVIgABUy1c2FktBC5evChe6p9\/\/jk98MAD9Omnn9oJGBY1XCDzyJEj4sXfuHHjUvOSk5PFi7979+6lL35XtZ2XyKZOnUr+\/v6UkJBARqOxtKkmk4lmzJhBzINtDAgI0ITNeXl5tGTJEsrOzhY2s+3W7euvv6YhQ4aU9rdW+rn8\/cdibf\/+\/ULI\/fWvf6Uvv\/zSRsBo1W5XfQ7RrronAAFT98xxxVomYH1ps3hZsGABhYeH07PPPmsnYHi5Zfz48RQYGChegn5+fnYvfhYH\/ILgF78at\/IChmdgtG4z+zOxzxPPzjz00ENiWU2LNluXDdnHibfyPjBatVuNzyHaXDsEIGBqhyvOWo8E+KW9c+dO6t27N912221U\/he5tWnWWRZefuBpeIPBUNpq\/nXLS068LMPLUFFRUfVokeOXvnDhgphJio6OFiItKytL\/F+LNvNS2hdffEHx8fHUunVrWr58ufD\/0WI\/8z0+d+5cIc5WrlxJ\/\/73v+0EjBbtdvxJwJFaJAABo8VehU02BJQEzLlz54Tj7oABA8QyUfmNl5f417wsskMNiHkJgf1\/eCZi9erV1LdvX9KqzVu3bhVLhrzxcsqqVatKI6+0aPM777wjhAs7Z7M43bt3r52A0aLdanju0Ma6IwABU3escaV6IqBHAWM2m+mtt94Sv9IHDx5M8+bNE2HFWn2pcR+zYCsbecVLf61atdKczTyrxstGgwYNorFjx4qZQwiYehpccNl6JQABU6\/4cfG6IKA3AcMv8t27d4sZCZ5dYj8gq5OrVgVM2fvo5MmT4gV\/zz330IoVK8QSklZm2jiybuHChXTp0iVas2aNWCLjDQKmLkYSXMPVCEDAuFqPoD1OJ1ATHxgOQy0foeT0BjrxhPyC44R8HEb82GOPiRmYsg7IVfGLUJvN5fGxLwzPOLEdvPzH4q0yvx+12My+LvPnzxfLRV27di01XSZg9NDXTnx0cCoVEoCAUWGnocnVI6AkYKxRGvwrVin8WE1RSJwThP0iOJHbqFGjaPLkyTYh1UxNazYr3Qll\/ZciIiJEFJLa+9kaUcZ+WRVt1nwwWrG7ek879tYTAQgYPfW2Tm1VEjBaypPB4oWXjNi5MzY2VmSiLZ9Kn7tfKzZzXhvOb8PRRrNmzSrNuMs2Wl\/0vFzGs2dcRkEL+X64706fPk3Xr1+3e5I5+urAgQMiH1CzZs2obdu25Ovrqwm7dTpswewqEICAqQIk7KJuAkoChq1SylTK9XXYf2Lbtm3UqVMnlwbAId\/cXvZ14fT5ZVPoyxquBZt5qYyXx44ePWqXffjw4cNixuX5558X+WAqyj6spn6u6CaULSFp5f526YcPjatXAhAw9YofF68LAhUJGGutGPYb4Xo63bp1U10tpKSkJOG0yo6dHJlidewsy5aT9T311FPCH0QLNrNt5es\/cS2kY8eOCb+XLl26CAHKSQytM0\/sN6LmfnZEwGilr+tinMA11EcAAkZ9fYYWV5NARQKGT8W\/5jnkmF98aqxGbbWvIizt27cXMxXBwcFiN7XbbLW1fAVuLljJZQSeeeYZm8zKWrJZ1s9KMzBat7uaQwF21xgBCBiNdSjMAQEQAAEQAAE9EICA0UMvw0YQAAEQAAEQ0BgBCBiNdSjMAQEQAAEQAAE9EICA0UMvw0YQAAEQAAEQ0BgBCBiNdSjMAQEQAAEQAAE9EICA0UMvw0YQAAEQAAEQ0BgBCBiNdSjMAQEQAAEQAAE9EICA0UMvw0YQAAEQAAEQ0BgBCBiNdSjMAQFOpb9r1y5KTEwUmXexgQAIgIAWCUDAaLFXYZNuCVhrBO3fv582b95MDz\/8sG5ZwHAQAAFtE4CA0Xb\/wjqdEThx4oQoYsgViwcMGEAJCQlkNBp1RgHmggAI6IEABIweehk26oIAF+5btmyZqHnEG1dh5qWk++67Txf2w0gQAAF9EYCA0Vd\/w1oNE7hw4QLFxsZS165d6e233xazMFyBeuHCheTl5aVhy2EaCICAHglAwOix12Gz5ghYLBZav349HTt2jJYvX05r1qwh9oMJCQkRVbajo6M1ZzMMAgEQ0DcBCBh99z+s1wiBlJQUiomJoeeee46eeOIJ4kikF154gXhZafjw4TRz5kyxpIQNBEAABLRCAAJGKz0JO3RNYO\/evbRnzx5at24dRUZGUmZmJk2aNIm++uorioqKom3btlHz5s11zQjGgwAIaIsABIy2+hPW6JAAi5Vx48ZRz549xWyLdabl4MGDQsTwxt9PmTJFh3RgMgiAgFYJQMBotWdhl24IHDp0iF5++WWR96VFixaldvOy0pgxY+inn36i1q1b05YtW6hJkya64QJDQQAEtE0AAkbb\/QvrNE4gNzeXpk6dSk2bNqXp06eTt7d3qcVWx95Vq1aJz+bMmUMjRozQOBGYBwIgoBcCEDB66WnYqUkCx48fp5deeok2bNggjTTi0OqRI0fS77\/\/Th06dBCzNKGhoZpkAaNAAAT0RQACRl\/9DWs1RIDLBnCOF\/67aNEiacbd8sntVq7Kc4rBAAAD3UlEQVRcSYMGDdIQBZgCAiCgVwIQMHrteditegK\/\/PILjR07lmbPnk2PPPKIoj08S8PlBfLy8qhHjx4iX0xAQIDq7YcBIAAC+iYAAaPv\/of1KiVg9W9hB12eVWnQoIGiJWULPPJOZYs8njlzhtLT06lly5YUFhYmzmE2myk7O5vc3d3FeQ0Gg8gn89tvv1FGRga1bdsWAkil9w2aDQJaIgABo6XehC26IWD1bencuTP169evUru\/++47Wrt2rdivV69eQvT4+\/uLcgMcXn316lURpcQi5tKlSyIc+\/7776dZs2YJ0cKOwixiONoJ+WQqxY0dQAAE6oAABEwdQMYlQMDZBHbs2EHx8fEOnbZ8kUcuAMllB6wlBzgse9SoUaIcQZ8+fcTfffv2CQHUpUsXh66Jg0AABEDA2QQgYJxNFOcDgVomkJqaSuPHj6dvv\/3W4SuxI++SJUvEMhE7AHPGXq5i3axZM1FL6f333xeChmdpOIqJs\/tWtlTlcGNwIAiAAAg4QAACxgFoOAQEtEIgJydHiCGuVp2YmCjM4v\/7+PgIwXL69GkaOnQoTZ48WdRWQj0lrfQ87AAB9ROAgFF\/H8ICEHCYwNmzZ4W\/y9\/+9jeaN28e\/frrr+L\/nPCOBcumTZvE0hH\/feihh0SNJS5R8MUXX4jEeadOnaLPPvtMFItEll+HuwEHggAIOEAAAsYBaDgEBLRC4Ouvv6YhQ4aIKtZz586lt99+mxYvXkwbN26kdu3aiRpKRUVFokhko0aNhNlpaWk0evRo4tmbiRMn0oMPPiiWmtzc3LSCBXaAAAiogAAEjAo6CU0EgdoiYI04uvfee0WpARYvHJrNodaffPKJWFZinxiOXOJwat44Gon9Zjg6KSEhgfz8\/GqreTgvCIAACCgSgIDBzQECOibAYmT79u20detWiomJoV27dlFUVBTdfffdwkmYP+NQ7bKzK1x\/iZeMOFcMkuLp+OaB6SBQzwQgYOq5A3B5EHAFAiaTiQ4fPkyxsbGiPMHAgQNLk9iVbR8n0Nu9ezdxpesDBw6ImRhPT0+R4RcbCIAACNQlAQiYuqSNa4GACxPg2ZSdO3eKcOro6Gi7lnJoNZctmD9\/vnD6nTRpkohe4nDs8PBwF7YMTQMBENAiAQgYLfYqbAKBahLgZSGuas1lBzh8GhWrqwkQu4MACNQ5AQiYOkeOC4KA6xE4f\/68iELiiCIOoUa+F9frI7QIBEDAlgAEDO4IEAABEAABEAAB1RGAgFFdl6HBIAACIAACIAACEDC4B0AABEAABEAABFRHAAJGdV2GBoMACIAACIAACPw\/CMqDvMWyMAQAAAAASUVORK5CYII=","height":337,"width":560}}
%---
