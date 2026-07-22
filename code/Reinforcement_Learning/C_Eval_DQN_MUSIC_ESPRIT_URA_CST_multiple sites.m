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
EnvPars_base = EnvPars;
%% ================================================================
% SIM 1 calibration
% ================================================================

EnvPars = EnvPars_base;

EnvPars.pos_SIM = [10, 10, 4];
pos_SIM = EnvPars.pos_SIM;

% SIM 1 points toward SIM 2
EnvPars.sectorBoresights_deg = 0;
EnvPars.selectedSector       = 1;

fprintf('\n=== Calibrating SIM 1 ===\n'); %[output:785415ac]

Calibration; %[output:83c07206]

EnvPars_SIM1 = EnvPars;

% Store plotting variables before the second calibration overwrites them
X_cal_SIM1 = X_cal;
Y_cal_SIM1 = Y_cal;
pos_cal_SIM1 = EnvPars_SIM1.pos_cal;

%% ================================================================
% SIM 2 calibration
% ================================================================

EnvPars = EnvPars_base;

EnvPars.pos_SIM = [30, 10, 4];
pos_SIM = EnvPars.pos_SIM;

% SIM 2 points back toward SIM 1
EnvPars.sectorBoresights_deg = 180;
EnvPars.selectedSector       = 1;

fprintf('\n=== Calibrating SIM 2 ===\n'); %[output:7f3b5680]

Calibration; %[output:9239e280]

EnvPars_SIM2 = EnvPars;

X_cal_SIM2 = X_cal;
Y_cal_SIM2 = Y_cal;
pos_cal_SIM2 = EnvPars_SIM2.pos_cal;



%% Optional visual verification of the calibration sector
%% ================================================================
% Visualize the two SIM sectors and their common illuminated region
% ================================================================

sim1_xy = EnvPars_SIM1.pos_SIM(1:2);
sim2_xy = EnvPars_SIM2.pos_SIM(1:2);

boresight1_deg = 0;
boresight2_deg = 180;

sectorHalfWidth_deg = 60;

% Use the inter-site distance as the visualization range
D_sites = norm(sim2_xy - sim1_xy);

% Slightly larger plotting window
plot_margin = 3;

x_plot_min = min(sim1_xy(1), sim2_xy(1)) - plot_margin;
x_plot_max = max(sim1_xy(1), sim2_xy(1)) + plot_margin;

y_plot_min = min(sim1_xy(2), sim2_xy(2)) - D_sites/2 - plot_margin;
y_plot_max = max(sim1_xy(2), sim2_xy(2)) + D_sites/2 + plot_margin;

% Fine grid used only to identify the overlapping sector region
n_plot_grid = 500;

x_plot = linspace(x_plot_min, x_plot_max, n_plot_grid);
y_plot = linspace(y_plot_min, y_plot_max, n_plot_grid);

[X_plot, Y_plot] = meshgrid(x_plot, y_plot);

%% Sector mask for SIM 1

dx1 = X_plot - sim1_xy(1);
dy1 = Y_plot - sim1_xy(2);

range1 = hypot(dx1, dy1);
azimuth1 = atan2(dy1, dx1);

delta_phi1 = atan2( ...
    sin(azimuth1 - deg2rad(boresight1_deg)), ...
    cos(azimuth1 - deg2rad(boresight1_deg)));

sectorMask1 = ...
    abs(delta_phi1) <= deg2rad(sectorHalfWidth_deg) & ...
    range1 <= D_sites;

%% Sector mask for SIM 2

dx2 = X_plot - sim2_xy(1);
dy2 = Y_plot - sim2_xy(2);

range2 = hypot(dx2, dy2);
azimuth2 = atan2(dy2, dx2);

delta_phi2 = atan2( ...
    sin(azimuth2 - deg2rad(boresight2_deg)), ...
    cos(azimuth2 - deg2rad(boresight2_deg)));

sectorMask2 = ...
    abs(delta_phi2) <= deg2rad(sectorHalfWidth_deg) & ...
    range2 <= D_sites;

%% Region illuminated by both SIMs

commonSectorMask = sectorMask1 & sectorMask2;

assert(any(commonSectorMask, 'all'), ...
    'The selected SIM sectors do not overlap.');


%%
 %% Plot

figure; %[output:3bf0b593]
hold on;box on; %[output:3bf0b593]
grid on; %[output:3bf0b593]

% Plot SIM 1 sector
contourf( ... %[output:3bf0b593]
    X_plot, ... %[output:3bf0b593]
    Y_plot, ... %[output:3bf0b593]
    double(sectorMask1), ... %[output:3bf0b593]
    [0.5 0.5], ... %[output:3bf0b593]
    'FaceAlpha', 0.15, ... %[output:3bf0b593]
    'LineStyle', 'none'); %[output:3bf0b593]

% Plot SIM 2 sector
contourf( ... %[output:3bf0b593]
    X_plot, ... %[output:3bf0b593]
    Y_plot, ... %[output:3bf0b593]
    double(sectorMask2), ... %[output:3bf0b593]
    [0.5 0.5], ... %[output:3bf0b593]
    'FaceAlpha', 0.15, ... %[output:3bf0b593]
    'LineStyle', 'none'); %[output:3bf0b593]

% Highlight the common illuminated region
contourf( ... %[output:3bf0b593]
    X_plot, ... %[output:3bf0b593]
    Y_plot, ... %[output:3bf0b593]
    double(commonSectorMask), ... %[output:3bf0b593]
    [0.5 0.5], ... %[output:3bf0b593]
    'FaceAlpha', 0.35, ... %[output:3bf0b593]
    'LineStyle', 'none'); %[output:3bf0b593]

% Plot calibration positions
scatter( ... %[output:3bf0b593]
    X_cal_SIM1, ... %[output:3bf0b593]
    Y_cal_SIM1, ... %[output:3bf0b593]
    22, ... %[output:3bf0b593]
    'o', ... %[output:3bf0b593]
    'DisplayName', 'SIM 1 calibration positions'); %[output:3bf0b593]

scatter( ... %[output:3bf0b593]
    X_cal_SIM2, ... %[output:3bf0b593]
    Y_cal_SIM2, ... %[output:3bf0b593]
    22, ... %[output:3bf0b593]
    'x', ... %[output:3bf0b593]
    'DisplayName', 'SIM 2 calibration positions'); %[output:3bf0b593]

% Plot the two SIM positions
plot( ... %[output:3bf0b593]
    sim1_xy(1), ... %[output:3bf0b593]
    sim1_xy(2), ... %[output:3bf0b593]
    'kp', ... %[output:3bf0b593]
    'MarkerSize', 15, ... %[output:3bf0b593]
    'MarkerFaceColor', 'k', ... %[output:3bf0b593]
    'DisplayName', 'SIM 1'); %[output:3bf0b593]

plot( ... %[output:3bf0b593]
    sim2_xy(1), ... %[output:3bf0b593]
    sim2_xy(2), ... %[output:3bf0b593]
    'ks', ... %[output:3bf0b593]
    'MarkerSize', 11, ... %[output:3bf0b593]
    'MarkerFaceColor', 'k', ... %[output:3bf0b593]
    'DisplayName', 'SIM 2'); %[output:3bf0b593]

%% Draw the sector boundary lines

boundaryLength = D_sites;

anglesSIM1 = deg2rad( ...
    boresight1_deg + ...
    [-sectorHalfWidth_deg, sectorHalfWidth_deg]);

anglesSIM2 = deg2rad( ...
    boresight2_deg + ...
    [-sectorHalfWidth_deg, sectorHalfWidth_deg]);

for k = 1:2

    plot( ... %[output:3bf0b593]
        [sim1_xy(1), ... %[output:3bf0b593]
         sim1_xy(1) + boundaryLength*cos(anglesSIM1(k))], ... %[output:3bf0b593]
        [sim1_xy(2), ... %[output:3bf0b593]
         sim1_xy(2) + boundaryLength*sin(anglesSIM1(k))], ... %[output:3bf0b593]
        '--', ... %[output:3bf0b593]
        'HandleVisibility', 'off'); %[output:3bf0b593]

    plot( ...
        [sim2_xy(1), ...
         sim2_xy(1) + boundaryLength*cos(anglesSIM2(k))], ...
        [sim2_xy(2), ...
         sim2_xy(2) + boundaryLength*sin(anglesSIM2(k))], ...
        '--', ...
        'HandleVisibility', 'off');
end

%% Draw the sector boresights

plot( ... %[output:3bf0b593]
    [sim1_xy(1), sim2_xy(1)], ... %[output:3bf0b593]
    [sim1_xy(2), sim1_xy(2)], ... %[output:3bf0b593]
    ':', ... %[output:3bf0b593]
    'LineWidth', 1.4, ... %[output:3bf0b593]
    'HandleVisibility', 'off'); %[output:3bf0b593]

plot( ... %[output:3bf0b593]
    [sim2_xy(1), sim1_xy(1)], ... %[output:3bf0b593]
    [sim2_xy(2), sim2_xy(2)], ... %[output:3bf0b593]
    ':', ... %[output:3bf0b593]
    'LineWidth', 1.4, ... %[output:3bf0b593]
    'HandleVisibility', 'off'); %[output:3bf0b593]

axis equal; %[output:3bf0b593]

xlim([x_plot_min, x_plot_max]); %[output:3bf0b593]
ylim([y_plot_min, y_plot_max]); %[output:3bf0b593]

xlabel( ... %[output:3bf0b593]
    'x position [m]', ... %[output:3bf0b593]
    'Interpreter', 'latex'); %[output:3bf0b593]

ylabel( ... %[output:3bf0b593]
    'y position [m]', ... %[output:3bf0b593]
    'Interpreter', 'latex'); %[output:3bf0b593]

title( ... %[output:3bf0b593]
    'Two SIMs illuminating a common sector', ... %[output:3bf0b593]
    'Interpreter', 'latex'); %[output:3bf0b593]

legend( ... %[output:3bf0b593]
    'Location', 'best', ... %[output:3bf0b593]
    'Interpreter', 'latex'); %[output:3bf0b593]

set(gca, 'FontSize', 14); %[output:3bf0b593]

%% Find calibration coordinates observed by both SIMs

xy_SIM1 = EnvPars_SIM1.pos_cal(:,1:2);
xy_SIM2 = EnvPars_SIM2.pos_cal(:,1:2);

% The calibration grid uses regular coordinates, but rounding prevents
% numerical matching issues.
matchingTolerance = 1e-9;

xy_SIM1_key = round(xy_SIM1 / matchingTolerance) * matchingTolerance;
xy_SIM2_key = round(xy_SIM2 / matchingTolerance) * matchingTolerance;

[xy_common, idx_SIM1_common, idx_SIM2_common] = ...
    intersect( ...
        xy_SIM1_key, ...
        xy_SIM2_key, ...
        'rows', ...
        'stable');

fprintf('SIM 1 calibration points: %d\n', size(xy_SIM1,1)); %[output:0f4a3f66]
fprintf('SIM 2 calibration points: %d\n', size(xy_SIM2,1)); %[output:4c69ad73]
fprintf('Common calibration points: %d\n', size(xy_common,1)); %[output:854001db]

assert(~isempty(xy_common), ...
    'The two SIM calibration regions have no common positions.');

scatter( ... %[output:group:91c8c013] %[output:3bf0b593]
    xy_common(:,1), ... %[output:3bf0b593]
    xy_common(:,2), ... %[output:3bf0b593]
    45, ... %[output:3bf0b593]
    'filled', ... %[output:3bf0b593]
    'MarkerFaceColor', [0 0 0], ... %[output:3bf0b593]
    'DisplayName', 'Common measurement positions'); %[output:group:91c8c013] %[output:3bf0b593]
%%

%% ================================================================
% POSITION-PROJECTION FIELD OF VIEW
% ================================================================

fovMargin_deg = 0.5;

EnvPars_SIM1.positionFoV_deg = ... %[output:group:49a2f20f] %[output:442fc62e]
    calculatePositionFoV( ... %[output:442fc62e]
        EnvPars_SIM1.pos_SIM, ... %[output:442fc62e]
        xy_common, ... %[output:442fc62e]
        EnvPars_SIM1.h_MU, ... %[output:442fc62e]
        fovMargin_deg); %[output:group:49a2f20f] %[output:442fc62e]

EnvPars_SIM2.positionFoV_deg = ... %[output:group:0659fb78] %[output:1de16e8d]
    calculatePositionFoV( ... %[output:1de16e8d]
        EnvPars_SIM2.pos_SIM, ... %[output:1de16e8d]
        xy_common, ... %[output:1de16e8d]
        EnvPars_SIM2.h_MU, ... %[output:1de16e8d]
        fovMargin_deg); %[output:group:0659fb78] %[output:1de16e8d]

fprintf('\nPosition projection FoV:\n'); %[output:85ebda03]
fprintf( ... %[output:group:5cc7c73f] %[output:9948d3ea]
    'SIM 1: %.2f degrees\n', ... %[output:9948d3ea]
    EnvPars_SIM1.positionFoV_deg); %[output:group:5cc7c73f] %[output:9948d3ea]

fprintf( ... %[output:group:00c60b45] %[output:82bc02fc]
    'SIM 2: %.2f degrees\n', ... %[output:82bc02fc]
    EnvPars_SIM2.positionFoV_deg); %[output:group:00c60b45] %[output:82bc02fc]
%%


% Load the trained DQN agent
%agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Tx_50_Aligned.mat')
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_ideal.mat');

if isfile(agent_path) %[output:group:4e64244d]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:4e00c158]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:4e64244d]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining the DFT as ideal or with CST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CST front end for both SIMs
EnvPars_SIM1.G      = EnvPars_SIM1.G_CST;
EnvPars_SIM1.U_func = EnvPars_SIM1.U_func_CST;

EnvPars_SIM2.G      = EnvPars_SIM2.G_CST;
EnvPars_SIM2.U_func = EnvPars_SIM2.U_func_CST;
%%
%% ================================================================
% FIND COMMON MU POSITIONS
% ================================================================

xy_SIM1 = EnvPars_SIM1.pos_cal(:,1:2);
xy_SIM2 = EnvPars_SIM2.pos_cal(:,1:2);

% Avoid numerical coordinate-matching problems.
positionTolerance = 1e-9;

xy_SIM1_key = ...
    round(xy_SIM1 / positionTolerance) * positionTolerance;

xy_SIM2_key = ...
    round(xy_SIM2 / positionTolerance) * positionTolerance;

% idx_SIM1_common and idx_SIM2_common point to the same physical
% coordinate in the two different calibration tables.
[~, idx_SIM1_common, idx_SIM2_common] = ...
    intersect( ...
        xy_SIM1_key, ...
        xy_SIM2_key, ...
        'rows', ...
        'stable');

assert(~isempty(idx_SIM1_common), ...
    'SIM 1 and SIM 2 have no common calibration positions.');

% Use the unrounded coordinates from SIM 1.
xy_common = ...
    EnvPars_SIM1.pos_cal(idx_SIM1_common, 1:2);

% Verify that the corresponding SIM-2 coordinates are identical.
xy_common_SIM2 = ...
    EnvPars_SIM2.pos_cal(idx_SIM2_common, 1:2);

maximumCoordinateDifference = ...
    max(abs(xy_common(:) - xy_common_SIM2(:)));

assert(maximumCoordinateDifference < 1e-8, ...
    'The matched calibration positions are not physically identical.');

fprintf('\n=== Common calibration region ===\n'); %[output:5c5e676e]
fprintf('SIM 1 positions: %d\n', EnvPars_SIM1.N_cal); %[output:09e13d5a]
fprintf('SIM 2 positions: %d\n', EnvPars_SIM2.N_cal); %[output:872ada1a]
fprintf('Common positions: %d\n', size(xy_common,1)); %[output:2ac47e4f]
%%
%% ================================================================
% SHARED EVALUATION DEPLOYMENTS
% ================================================================

N_eval = 500;

% Reproducible selection of common positions
rng(1, 'twister');

commonSelection = ...
    randi( ...
        size(xy_common,1), ...
        N_eval, ...
        1);

% Calibration index used by each SIM
target_idx_SIM1 = ...
    idx_SIM1_common(commonSelection);

target_idx_SIM2 = ...
    idx_SIM2_common(commonSelection);

% The true MU coordinates are taken from the SIM-1 calibration table.
pMU_eval = ...
    EnvPars_SIM1.pos_cal(target_idx_SIM1, :);

% Verify against SIM 2.
pMU_eval_SIM2 = ...
    EnvPars_SIM2.pos_cal(target_idx_SIM2, :);

positionDifference = ...
    vecnorm( ...
        pMU_eval - pMU_eval_SIM2, ...
        2, ...
        2);

assert(max(positionDifference) < 1e-8, ...
    ['The two loops are not using identical physical ' ...
     'MU coordinates.']);

fprintf( ... %[output:group:71d155c1] %[output:74d398a5]
    'Prepared %d common evaluation episodes.\n', ... %[output:74d398a5]
    N_eval); %[output:group:71d155c1] %[output:74d398a5]
%%
%[text] ### DQN Agent and Brute Force
% Benchmark Configurations
N_eval = 500;              % Evaluation episodes (random positions)
K_snap = 16;               % Snapshots per point for digital estimators
Refine = true;             % 3-point parabolic peak refinement for MUSIC/DFT


%% ================================================================
% RESULT ARRAYS
% ================================================================

pos_est_SIM1 = nan(N_eval, 3);
pos_est_SIM2 = nan(N_eval, 3);

psi_x_est_SIM1 = nan(N_eval, 1);
psi_y_est_SIM1 = nan(N_eval, 1);

psi_x_est_SIM2 = nan(N_eval, 1);
psi_y_est_SIM2 = nan(N_eval, 1);

steps_SIM1 = zeros(N_eval, 1);
steps_SIM2 = zeros(N_eval, 1);

err_SIM1 = nan(N_eval, 1);
err_SIM2 = nan(N_eval, 1);

%%
%% ================================================================
% DQN EVALUATION: SIM 1
% ================================================================

fprintf('\n=== Running SIM 1 evaluation ===\n'); %[output:9e4a146d]

for i = 1:N_eval %[output:group:4e5a1438]

    if mod(i, 50) == 0
        fprintf( ... %[output:98b9dedc]
            'SIM 1: episode %d / %d\n', ... %[output:98b9dedc]
            i, ... %[output:98b9dedc]
            N_eval); %[output:98b9dedc]
    end

    % SIM-1 row corresponding to the common physical MU
    target_idx = target_idx_SIM1(i);

    % True global MU coordinate
    pMU = pMU_eval(i,:);

    %% Reset SIM 1 at the prescribed calibration entry

    [obs, LoggedSignals] = ...
        resetFunction_nav_CST_Aligned( ...
            EnvPars_SIM1, ...
            target_idx);

    % Sanity check
    pMU_from_reset = ...
        EnvPars_SIM1.pos_cal( ...
            LoggedSignals.pos_idx, :);

    assert(norm(pMU_from_reset - pMU) < 1e-8, ...
        'SIM 1 reset selected the wrong MU position.');

    %% DQN navigation

    isDone = false;
    step   = 0;

    while ~isDone

        action_out = ...
            getAction(agent, {obs});

        action = action_out{1};

        if iscell(action)
            action = action{1};
        end

        if iscategorical(action)
            action = double(action);
        end

        action = double(action);

        [obs, ~, isDone, LoggedSignals] = ...
            stepFunction_nav_CST_Aligned( ...
                action, ...
                LoggedSignals, ...
                EnvPars_SIM1);

        step = step + 1;

        % Safety protection in case of an environment error
        if step > EnvPars_SIM1.MaxStepsPerEpisode + 5
            warning( ...
                'SIM 1 episode %d exceeded the expected step limit.', ...
                i);
            break;
        end
    end

    steps_SIM1(i) = step;

    %% Refine the DQN estimate

    [psi_x_hat, psi_y_hat] = ...
        refineAngleParabolic3( ...
            LoggedSignals.h, ...
            LoggedSignals.best_t_x, ...
            LoggedSignals.best_t_y, ...
            EnvPars_SIM1, ...
            struct( ...
                'm',     10, ...
                'n_avg', 12));

    psi_x_est_SIM1(i) = psi_x_hat;
    psi_y_est_SIM1(i) = psi_y_hat;

    %% Convert to a global MU position

    pos_est_SIM1(i,:) = ...
        estimatePosFromAngles( ...
            psi_x_hat, ...
            psi_y_hat, ...
            EnvPars_SIM1, ...
            pMU);

    %% Horizontal error in centimetres

    err_SIM1(i) = ...
        100 * norm( ...
            pos_est_SIM1(i,1:2) - ...
            pMU(1:2));
end %[output:group:4e5a1438]

%%
%% ================================================================
% DQN EVALUATION: SIM 2
% ================================================================

fprintf('\n=== Running SIM 2 evaluation ===\n'); %[output:517a6f59]

for i = 1:N_eval %[output:group:0ebf00bb]

    if mod(i, 50) == 0
        fprintf( ... %[output:6d7e5519]
            'SIM 2: episode %d / %d\n', ... %[output:6d7e5519]
            i, ... %[output:6d7e5519]
            N_eval); %[output:6d7e5519]
    end

    % SIM-2 row corresponding to the same physical MU used by SIM 1
    target_idx = target_idx_SIM2(i);

    % The same global MU coordinate used in the first loop
    pMU = pMU_eval(i,:);

    %% Reset SIM 2 at its corresponding calibration entry

    [obs, LoggedSignals] = ...
        resetFunction_nav_CST_Aligned( ...
            EnvPars_SIM2, ...
            target_idx);

    % Sanity check
    pMU_from_reset = ...
        EnvPars_SIM2.pos_cal( ...
            LoggedSignals.pos_idx, :);

    assert(norm(pMU_from_reset - pMU) < 1e-8, ...
        'SIM 2 reset selected the wrong MU position.');

    %% DQN navigation

    isDone = false;
    step   = 0;

    while ~isDone

        action_out = ...
            getAction(agent, {obs});

        action = action_out{1};

        if iscell(action)
            action = action{1};
        end

        if iscategorical(action)
            action = double(action);
        end

        action = double(action);

        [obs, ~, isDone, LoggedSignals] = ...
            stepFunction_nav_CST_Aligned( ...
                action, ...
                LoggedSignals, ...
                EnvPars_SIM2);

        step = step + 1;

        if step > EnvPars_SIM2.MaxStepsPerEpisode + 5
            warning( ...
                'SIM 2 episode %d exceeded the expected step limit.', ...
                i);
            break;
        end
    end

    steps_SIM2(i) = step;

    %% Refine the DQN estimate

    [psi_x_hat, psi_y_hat] = ...
        refineAngleParabolic3( ...
            LoggedSignals.h, ...
            LoggedSignals.best_t_x, ...
            LoggedSignals.best_t_y, ...
            EnvPars_SIM2, ...
            struct( ...
                'm',     10, ...
                'n_avg', 12));

    psi_x_est_SIM2(i) = psi_x_hat;
    psi_y_est_SIM2(i) = psi_y_hat;

    %% Convert to a global MU position

    pos_est_SIM2(i,:) = ...
        estimatePosFromAngles( ...
            psi_x_hat, ...
            psi_y_hat, ...
            EnvPars_SIM2, ...
            pMU);

    %% Horizontal error in centimetres

    err_SIM2(i) = ...
        100 * norm( ...
            pos_est_SIM2(i,1:2) - ...
            pMU(1:2));
end %[output:group:0ebf00bb]

%%
%% ================================================================
% TWO-SIM POSITION FUSION
% ================================================================

% Equal-weight average of the two independently estimated positions.
pos_est_fused = ...
    0.5 * (pos_est_SIM1 + pos_est_SIM2);

% The MU height is assumed known and should not be altered by averaging.
pos_est_fused(:,3) = pMU_eval(:,3);

% Fused horizontal error in centimetres.
err_fused = ...
    100 * vecnorm( ...
        pos_est_fused(:,1:2) - ...
        pMU_eval(:,1:2), ...
        2, ...
        2);
%%
psi_x_true_SIM1 = ...
    EnvPars_SIM1.psi_x_cal(idx_SIM1_common);

psi_y_true_SIM1 = ...
    EnvPars_SIM1.psi_y_cal(idx_SIM1_common);

psi_x_true_SIM2 = ...
    EnvPars_SIM2.psi_x_cal(idx_SIM2_common);

psi_y_true_SIM2 = ...
    EnvPars_SIM2.psi_y_cal(idx_SIM2_common);

figure; %[output:9ed4957a]
hold on; %[output:9ed4957a]
box on; %[output:9ed4957a]
grid on; %[output:9ed4957a]

scatter( ... %[output:9ed4957a]
    wrapToSigned(psi_x_true_SIM1), ... %[output:9ed4957a]
    wrapToSigned(psi_y_true_SIM1), ... %[output:9ed4957a]
    25, ... %[output:9ed4957a]
    'o'); %[output:9ed4957a]

scatter( ... %[output:9ed4957a]
    wrapToSigned(psi_x_true_SIM2), ... %[output:9ed4957a]
    wrapToSigned(psi_y_true_SIM2), ... %[output:9ed4957a]
    25, ... %[output:9ed4957a]
    'x'); %[output:9ed4957a]

xlabel('$\psi_x$ [rad]', 'Interpreter', 'latex'); %[output:9ed4957a]
ylabel('$\psi_y$ [rad]', 'Interpreter', 'latex'); %[output:9ed4957a]

legend( ... %[output:9ed4957a]
    {'SIM 1 true angles', 'SIM 2 true angles'}, ... %[output:9ed4957a]
    'Interpreter', 'latex', ... %[output:9ed4957a]
    'Location', 'best'); %[output:9ed4957a]

title('Electrical-angle regions seen by the two SIMs'); %[output:9ed4957a]

figure; %[output:5111c754]
hold on; %[output:5111c754]
box on; %[output:5111c754]
grid on; %[output:5111c754]
axis equal; %[output:5111c754]

scatter( ... %[output:5111c754]
    pMU_eval(:,1), ... %[output:5111c754]
    pMU_eval(:,2), ... %[output:5111c754]
    28, ... %[output:5111c754]
    'filled', ... %[output:5111c754]
    'DisplayName', 'True MU'); %[output:5111c754]

scatter( ... %[output:5111c754]
    pos_est_SIM1(:,1), ... %[output:5111c754]
    pos_est_SIM1(:,2), ... %[output:5111c754]
    24, ... %[output:5111c754]
    'o', ... %[output:5111c754]
    'DisplayName', 'SIM 1 estimate'); %[output:5111c754]

scatter( ... %[output:5111c754]
    pos_est_SIM2(:,1), ... %[output:5111c754]
    pos_est_SIM2(:,2), ... %[output:5111c754]
    24, ... %[output:5111c754]
    'x', ... %[output:5111c754]
    'DisplayName', 'SIM 2 estimate'); %[output:5111c754]

plot( ... %[output:5111c754]
    EnvPars_SIM1.pos_SIM(1), ... %[output:5111c754]
    EnvPars_SIM1.pos_SIM(2), ... %[output:5111c754]
    'kp', ... %[output:5111c754]
    'MarkerSize', 14, ... %[output:5111c754]
    'MarkerFaceColor', 'k', ... %[output:5111c754]
    'DisplayName', 'SIM 1'); %[output:5111c754]

plot( ... %[output:5111c754]
    EnvPars_SIM2.pos_SIM(1), ... %[output:5111c754]
    EnvPars_SIM2.pos_SIM(2), ... %[output:5111c754]
    'ks', ... %[output:5111c754]
    'MarkerSize', 11, ... %[output:5111c754]
    'MarkerFaceColor', 'k', ... %[output:5111c754]
    'DisplayName', 'SIM 2'); %[output:5111c754]

xlabel('x position [m]'); %[output:5111c754]
ylabel('y position [m]'); %[output:5111c754]
legend('Location', 'best'); %[output:5111c754]

title('Position estimates before fusion'); %[output:5111c754]
%%
%[text] ### Figures
% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
%plot limits
xlim_1=0;
xlim_2=600;

figure; hold on; box on; grid on; %[output:7430f943]
h_dqn = cdfplot(err_SIM1);    set(h_dqn, 'LineWidth', 1.5); %[output:7430f943]
h_dqn = cdfplot(err_SIM2);    set(h_dqn, 'LineWidth', 1.5); %[output:7430f943]
h_dqn = cdfplot(err_fused);    set(h_dqn, 'LineWidth', 1.5); %[output:7430f943]
% h_bf  = cdfplot(err_bf);     set(h_bf, 'LineWidth', 1.5, 'LineStyle', '--');
% h_mus = cdfplot(err_music);  set(h_mus, 'LineWidth', 1.5);
% h_esp = cdfplot(err_esprit); set(h_esp, 'LineWidth', 1.5);

%patch
confidence=90;%confidence level
[F,X]=ecdf(err_SIM1);
[~,idx]=min(abs(F-confidence/100)) %[output:5163ba95]
x_coords = [0  X(idx) X(idx) 0];
y_coords = [confidence/100 confidence/100 1  1];
patch(x_coords, y_coords, 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); %[output:7430f943]
plot([xlim_1 xlim_2],[confidence/100 confidence/100],'--','Color',[0.4 0.4 0.4]); %[output:7430f943]
plot([X(idx) X(idx)],[0 1],'--','Color',[0.4 0.4 0.4]); %[output:7430f943]
text(5,0.9,strcat('$',num2str(confidence),'\%$ Confidence'),'Interpreter','latex','FontSize',font-2); %[output:7430f943]
text(X(idx),0.6,strcat('$',num2str(X(idx),4),'$ cm'),'Interpreter','latex','FontSize',font-2,'Rotation',90); %[output:7430f943]
set(gca,'FontSize',font); %[output:7430f943]

xlim([xlim_1, xlim_2]); %[output:7430f943]
yticks([0,0.2,0.4,0.6,0.8,1]) %[output:7430f943]
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', 14); %[output:7430f943]
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', 14); %[output:7430f943]
legend({'SIM 1', 'SIM 2', 'Fused'}, ... %[output:7430f943]
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12); %[output:7430f943]
title(''); %[output:7430f943]

% Print Summary
fprintf('\n================ RESULTS SUMMARY ================\n'); %[output:8e05dc72]
fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n'); %[output:59d82c0c]
fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_SIM1), median(err_dqn), prctile(err_dqn, 95), mean(steps_dqn));
fprintf('Brute Force   | %13.2f | %11.2f | %20.2f\n', mean(err_bf), median(err_bf), prctile(err_bf, 95));
fprintf('2-D MUSIC     | %13.2f | %11.2f | %20.2f\n', mean(err_music), median(err_music), prctile(err_music, 95));
fprintf('2-D ESPRIT    | %13.2f | %11.2f | %20.2f\n', mean(err_esprit), median(err_esprit), prctile(err_esprit, 95));

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
%% Divide completed episodes into validation and test subsets

rng(10, 'twister');

validEpisodes = ...
    all(isfinite(pos_est_SIM1), 2) & ...
    all(isfinite(pos_est_SIM2), 2) & ...
    all(isfinite(pMU_eval), 2);

valid_idx = find(validEpisodes);

randomOrder = ...
    valid_idx(randperm(numel(valid_idx)));

validationFraction = 0.30;

N_val = round( ...
    validationFraction * numel(randomOrder));

idx_val  = randomOrder(1:N_val);
idx_test = randomOrder(N_val+1:end);

%% These are the previously mentioned "_val" variables

pos_est_SIM1_val = pos_est_SIM1(idx_val,:);
pos_est_SIM2_val = pos_est_SIM2(idx_val,:);
pMU_val          = pMU_eval(idx_val,:);

%% Independent test arrays

pos_est_SIM1_test = pos_est_SIM1(idx_test,:);
pos_est_SIM2_test = pos_est_SIM2(idx_test,:);
pMU_test          = pMU_eval(idx_test,:);

residual_SIM1_val = ...
    pos_est_SIM1_val(:,1:2) - ...
    pMU_val(:,1:2);

residual_SIM2_val = ...
    pos_est_SIM2_val(:,1:2) - ...
    pMU_val(:,1:2);

sigma2_SIM1 = mean( ...
    sum(residual_SIM1_val.^2, 2), ...
    'omitnan');

sigma2_SIM2 = mean( ...
    sum(residual_SIM2_val.^2, 2), ...
    'omitnan');

w_SIM1 = 1 / max(sigma2_SIM1, eps);
w_SIM2 = 1 / max(sigma2_SIM2, eps);

alpha_SIM1 = w_SIM1 / (w_SIM1 + w_SIM2);
alpha_SIM2 = w_SIM2 / (w_SIM1 + w_SIM2);

fprintf('\nFusion weights estimated from validation data:\n'); %[output:91d40384]
fprintf('SIM 1 weight = %.4f\n', alpha_SIM1); %[output:867db4b5]
fprintf('SIM 2 weight = %.4f\n', alpha_SIM2); %[output:6ccb57e5]

%% ================================================================
% VALIDATION-BASED DISAGREEMENT GATE
% ================================================================

% Weighted estimate on the validation subset
pos_est_weighted_val = ...
    alpha_SIM1 * pos_est_SIM1_val + ...
    alpha_SIM2 * pos_est_SIM2_val;

% Keep the known MU height
pos_est_weighted_val(:,3) = pMU_val(:,3);

% Distance between the two independent estimates [m]
disagreement_val_m = ...
    vecnorm( ...
        pos_est_SIM1_val(:,1:2) - ...
        pos_est_SIM2_val(:,1:2), ...
        2, ...
        2);

% Determine which SIM is globally more reliable on validation data.
% This SIM will be used whenever the gate rejects fusion.
if sigma2_SIM1 <= sigma2_SIM2

    fallbackSite = 1;
    fallbackName = 'SIM 1';

else

    fallbackSite = 2;
    fallbackName = 'SIM 2';
end

fprintf('\nMore reliable validation site: %s\n', fallbackName); %[output:40ef520f]

%% Search for the disagreement threshold that minimizes validation P90

% Candidate thresholds are obtained from the empirical distribution of
% the disagreement between the two position estimates.
candidatePercentiles = 5:5:95;

thresholdCandidates_m = unique([ ...
    0; ...
    prctile( ...
        disagreement_val_m, ...
        candidatePercentiles).'; ...
    inf]);

bestValidationP90_cm  = inf;
bestValidationMean_cm = inf;
gateThreshold_m       = inf;

validationP90_perThreshold_cm = ...
    nan(numel(thresholdCandidates_m),1);

validationMean_perThreshold_cm = ...
    nan(numel(thresholdCandidates_m),1);

for kGate = 1:numel(thresholdCandidates_m)

    currentThreshold_m = ...
        thresholdCandidates_m(kGate);

    % Reject fusion when the two estimates disagree too much.
    gateMask_val = ...
        disagreement_val_m > currentThreshold_m;

    % Start from the weighted estimate.
    pos_est_gated_val = ...
        pos_est_weighted_val;

    % For rejected measurements, retain the more reliable SIM.
    if fallbackSite == 1

        pos_est_gated_val(gateMask_val,:) = ...
            pos_est_SIM1_val(gateMask_val,:);

    else

        pos_est_gated_val(gateMask_val,:) = ...
            pos_est_SIM2_val(gateMask_val,:);
    end

    % Horizontal validation error [cm]
    err_gated_val = ...
        100 * vecnorm( ...
            pos_est_gated_val(:,1:2) - ...
            pMU_val(:,1:2), ...
            2, ...
            2);

    currentP90_cm = ...
        prctile(err_gated_val, 90);

    currentMean_cm = ...
        mean(err_gated_val, 'omitnan');

    validationP90_perThreshold_cm(kGate) = ...
        currentP90_cm;

    validationMean_perThreshold_cm(kGate) = ...
        currentMean_cm;

    % Primary objective: minimize P90.
    % Tie breaker: minimize mean error.
    improvesP90 = ...
        currentP90_cm < bestValidationP90_cm - 1e-9;

    tiesP90AndImprovesMean = ...
        abs(currentP90_cm - bestValidationP90_cm) <= 1e-9 && ...
        currentMean_cm < bestValidationMean_cm;

    if improvesP90 || tiesP90AndImprovesMean

        bestValidationP90_cm  = currentP90_cm;
        bestValidationMean_cm = currentMean_cm;
        gateThreshold_m       = currentThreshold_m;
    end
end

fprintf('\nValidation-selected gate:\n'); %[output:025a72b1]
fprintf('Disagreement threshold = %.3f m\n', gateThreshold_m); %[output:107d62ce]
fprintf('Validation gated P90   = %.2f cm\n', bestValidationP90_cm); %[output:29531e26]
fprintf('Validation gated mean  = %.2f cm\n', bestValidationMean_cm); %[output:7c40fed1]
fprintf('Fallback estimate      = %s\n', fallbackName); %[output:89026cf4]

pos_est_weighted_test = ...
    alpha_SIM1 * pos_est_SIM1_test + ...
    alpha_SIM2 * pos_est_SIM2_test;

pos_est_weighted_test(:,3) = pMU_test(:,3);

%% ================================================================
% APPLY THE VALIDATION-SELECTED GATE TO THE TEST SET
% ================================================================

% The gate uses only the difference between the two estimates.
% It does not use the true test position.
disagreement_test_m = ...
    vecnorm( ...
        pos_est_SIM1_test(:,1:2) - ...
        pos_est_SIM2_test(:,1:2), ...
        2, ...
        2);

% Episodes where weighted fusion is considered unreliable
gateMask_test = ...
    disagreement_test_m > gateThreshold_m;

% Begin with the weighted estimate
pos_est_gated_test = ...
    pos_est_weighted_test;

% Replace high-disagreement fused estimates with the more reliable site
if fallbackSite == 1

    pos_est_gated_test(gateMask_test,:) = ...
        pos_est_SIM1_test(gateMask_test,:);

else

    pos_est_gated_test(gateMask_test,:) = ...
        pos_est_SIM2_test(gateMask_test,:);
end

% Preserve known MU height
pos_est_gated_test(:,3) = ...
    pMU_test(:,3);

fprintf('\nTest-set gate operation:\n'); %[output:9c6ac2fa]
fprintf('Gated episodes = %.2f %%\n', ... %[output:group:651dc82c] %[output:9ea5755c]
    100 * mean(gateMask_test)); %[output:group:651dc82c] %[output:9ea5755c]

fprintf('Weighted episodes = %.2f %%\n', ... %[output:group:79b712ac] %[output:51205de6]
    100 * mean(~gateMask_test)); %[output:group:79b712ac] %[output:51205de6]

err_SIM1_test = ...
    100 * vecnorm( ...
        pos_est_SIM1_test(:,1:2) - ...
        pMU_test(:,1:2), ...
        2, ...
        2);

err_SIM2_test = ...
    100 * vecnorm( ...
        pos_est_SIM2_test(:,1:2) - ...
        pMU_test(:,1:2), ...
        2, ...
        2);

err_equal_test = ...
    100 * vecnorm( ...
        0.5 * ( ...
            pos_est_SIM1_test(:,1:2) + ...
            pos_est_SIM2_test(:,1:2)) - ...
        pMU_test(:,1:2), ...
        2, ...
        2);

err_weighted_test = ...
    100 * vecnorm( ...
        pos_est_weighted_test(:,1:2) - ...
        pMU_test(:,1:2), ...
        2, ...
        2);

err_gated_test = ...
    100 * vecnorm( ...
        pos_est_gated_test(:,1:2) - ...
        pMU_test(:,1:2), ...
        2, ...
        2);

figure; %[output:8afcbbdf]
hold on; %[output:8afcbbdf]
box on; %[output:8afcbbdf]
grid on; %[output:8afcbbdf]

h1 = cdfplot(err_SIM1_test); %[output:8afcbbdf]
set(h1, 'LineWidth', 1.5); %[output:8afcbbdf]

h2 = cdfplot(err_SIM2_test); %[output:8afcbbdf]
set(h2, 'LineWidth', 1.5); %[output:8afcbbdf]

h3 = cdfplot(err_equal_test); %[output:8afcbbdf]
set(h3, 'LineWidth', 1.5); %[output:8afcbbdf]

h4 = cdfplot(err_weighted_test); %[output:8afcbbdf]
set(h4, 'LineWidth', 2); %[output:8afcbbdf]

h5 = cdfplot(err_gated_test); %[output:8afcbbdf]
set( ... %[output:8afcbbdf]
    h5, ... %[output:8afcbbdf]
    'LineWidth', 2.2, ... %[output:8afcbbdf]
    'LineStyle', '-.'); %[output:8afcbbdf]

xlabel('Precision [cm]'); %[output:8afcbbdf]
ylabel('CDF'); %[output:8afcbbdf]

legend( ... %[output:group:64aa9e97] %[output:8afcbbdf]
    {'SIM 1', ... %[output:8afcbbdf]
     'SIM 2', ... %[output:8afcbbdf]
     'Equal average', ... %[output:8afcbbdf]
     'Weighted average', ... %[output:8afcbbdf]
     'Weighted + disagreement gate'}, ... %[output:8afcbbdf]
    'Location', 'southeast'); %[output:group:64aa9e97] %[output:8afcbbdf]

fprintf('\nTest-set P90 results:\n'); %[output:7db090fb]
fprintf('SIM 1          : %.2f cm\n', prctile(err_SIM1_test,90)); %[output:1584fd48]
fprintf('SIM 2          : %.2f cm\n', prctile(err_SIM2_test,90)); %[output:2832d795]
fprintf('Equal average  : %.2f cm\n', prctile(err_equal_test,90)); %[output:7b36806b]
fprintf('Weighted average: %.2f cm\n', prctile(err_weighted_test,90)); %[output:1455ce39]
%%
%[text] ### Functions
function thetaFoV_deg = calculatePositionFoV( ...
    pos_SIM, xy_region, h_MU, margin_deg)

    drop = pos_SIM(3) - h_MU;

    assert(drop > 0, ...
        'The SIM height must be greater than the MU height.');

    horizontalDistance = hypot( ...
        xy_region(:,1) - pos_SIM(1), ...
        xy_region(:,2) - pos_SIM(2));

    rho_max = max(horizontalDistance);

    theta_required_deg = ...
        rad2deg(atan2(rho_max, drop));

    % Add a small geometric margin but avoid reaching exactly 90 degrees.
    thetaFoV_deg = min( ...
        theta_required_deg + margin_deg, ...
        89.5);

    fprintf( ...
        ['SIM at (%.1f, %.1f): maximum horizontal range ' ...
         '= %.2f m, required angle = %.2f degrees\n'], ...
        pos_SIM(1), ...
        pos_SIM(2), ...
        rho_max, ...
        theta_required_deg);
end

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
    % s_max = sin(deg2rad(72))^2;          % FoV guard (room corner ~70.5 deg)
    % was_clamped = s > s_max;
    % if was_clamped
    %     sc = sqrt(s_max/s);  ux = ux*sc;  uy = uy*sc;  s = s_max;
    % end
    % uz = sqrt(1 - s);

    %% Site-specific geometric FoV

    if isfield(EnvPars, 'positionFoV_deg') && ...
            ~isempty(EnvPars.positionFoV_deg)

        thetaFoV_deg = ...
            EnvPars.positionFoV_deg;

    else

        thetaFoV_deg = 89.0;
    end

    thetaFoV_deg = min(thetaFoV_deg, 89.5);

    s_max = ...
        sin(deg2rad(thetaFoV_deg))^2;

    %% Protect against invalid estimated direction cosines

    was_clamped = s > s_max;

    if was_clamped

        sc = sqrt(s_max / max(s, eps));

        ux = ux * sc;
        uy = uy * sc;
        s  = s_max;
    end

    uz = sqrt(max(1-s, eps));

    %% Project onto the known MU-height plane

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
%   data: {"layout":"onright","rightPanelPercent":49}
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
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     x_max: 20\n                     y_max: 20\n              calSpacing_m: 0.5000\n      sectorBoresights_deg: [0 120 240]\n            selectedSector: 1\n       sectorHalfWidth_deg: 60\n                 MU_margin: 0\n                     N_cal: 400\n              channelModel: 'rician_los_nlos'\n                    fc_GHz: 28\n                    V_hall: 72000\n                    S_hall: 18000\n                   mu_lgDS: -7.2781\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 10\n                sigma_K_dB: 8\n                        KR: 2.8512\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 1\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         M: 196\n                       M_x: 14\n                       M_y: 14\n                         T: 3600\n                       T_x: 60\n                       T_y: 60\n                    SNR_dB: 34.6606\n                 theta_min: 1.7657\n                 theta_max: 4.5175\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 24\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [10 10 4]\n                    pos_MU: [111.9022 36.2006 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 … ] (1×3600 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 … ] (1×3600 double)\n                      h_MU: 1.5000\n                    L_hall: 120\n                    W_hall: 60\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 180\n                 tolerance: 0.0262\n         StopTrainingValue: 58\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9800\n"}}
%---
%[output:6228f180]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 2.366%n","truncated":false}}
%---
%[output:785415ac]
%   data: {"dataType":"text","outputData":{"text":"\n=== Calibrating SIM 1 ===\n","truncated":false}}
%---
%[output:83c07206]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nFull local grid: 40 x 40 = 1600 candidate positions\nSelected sector 1: boresight = 0.0 deg, angular interval = [-60.0, 60.0] deg\nCalibration positions retained in sector: 570\nCalibration complete.\n\n","truncated":false}}
%---
%[output:7f3b5680]
%   data: {"dataType":"text","outputData":{"text":"\n=== Calibrating SIM 2 ===\n","truncated":false}}
%---
%[output:9239e280]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nFull local grid: 40 x 40 = 1600 candidate positions\nSelected sector 1: boresight = 180.0 deg, angular interval = [120.0, 240.0] deg\nCalibration positions retained in sector: 1600\nCalibration complete.\n\n","truncated":false}}
%---
%[output:0f4a3f66]
%   data: {"dataType":"text","outputData":{"text":"SIM 1 calibration points: 570\n","truncated":false}}
%---
%[output:4c69ad73]
%   data: {"dataType":"text","outputData":{"text":"SIM 2 calibration points: 1600\n","truncated":false}}
%---
%[output:854001db]
%   data: {"dataType":"text","outputData":{"text":"Common calibration points: 570\n","truncated":false}}
%---
%[output:3bf0b593]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7snQnYVVP7\/1ejohnJEBFKpHpLoQypVKJfKGPGTGUMGfJKhkJIk+I1ZEwk81AyVqLCq5CK4jVEoiQNmp7\/\/3M\/1tN+dnufs8+49znnvq+rqzpn7zV81zp7ffc9likqKioyKoqAIqAIKAKKgCKgCOQQAmWUwOTQaulQFQFFQBFQBBQBRUAQUAKjG0ERUAQUAUVAEVAEcg4BJTA5t2Q6YEVAEVAEFAFFQBFQAqN7QBFQBBQBRUARUARyDgElMDm3ZDpgRUARUAQUAUVAEVACo3tAEUgzAn\/99ZepUqXKVq3+9NNPpmLFimbHHXcs9Z3f52kelm9zfuPNVv+rV6822223Xba6034UAUUgTxBQApMnC+mexgknnGA+++wzU65cOVO9enU5OH\/55Re5rFatWqZs2bJm5cqVZsOGDfIZ11arVi00NH777Tdzzz33mGnTppn169eb8uXLm5133ln+PuaYY8xZZ50lYxs\/frx55513zFtvvSX\/hwycdNJJ5vjjjzdvv\/22mTRpkvnvf\/8r391yyy2mZ8+eMed05513mgceeECuadGihTnxxBPNySefnDQOTz75pLnpppvMfffdZzp16mTWrl1rHnroIfP666+bBQsWmPvvv98cffTRvp8n3XGSN86bN8+wV04\/\/XRz4403JtlK8rddcsklsp78qVOnTvIN6Z2KgCJQcAgogcnTJefQP+qoo8yll15qttlmG5ll8+bNzYoVK8zEiRNNs2bNDCmAOPT79Oljpk6dGtoBAon6v\/\/7Pxnnf\/7zHyElixcvNqNGjTIvvviiufDCC821115bslKbN2+WuUDAbr\/99lKEg8+YG1K3bl3z7rvvClnzEsjFIYccYv7880\/5GvK06667prQjXnnlFXPrrbeaESNGmIMPPrikrX79+gnulsDYL\/w+T2kQCdz87bffCnk5++yzzQUXXJDAnYlfOmPGDHPooYeWuhGSyR5knWvWrJl4o3qHIhAHAa99p6DlBwJKYPJjHbeaxSmnnGKefvppU6ZMmZLv3ATGeYhCYvbcc89Q0HjzzTfNRRddZIYPH26OO+64rQ64P\/74wwwdOrTU5+3atTMcvhCe9u3bl\/qucePGMm9MIyNHjjRdunTxnNdTTz1l3njjDcMDDvnyyy9N5cqVM4IB47zjjju2IjB+n2dkECE2+sMPP4h2a9asWSGOQrsuNAR03+X3iiuBydP1RRPgJgN+BGb27NmirQhLhT969Ghz9913m4svvthcddVVpVYELcnll18uRMUplsBgnkHT5BTmyWHJd02aNDEvvPDCVquM9qlDhw7muuuuK9E8fPXVVyXaqnRvi0ImMMuWLTNnnnmmaNUwo6koAtlAQPddNlAOtw8lMOHin9Xe\/QgMJhxMTWhCcD5F24FWA\/8QTB5oM\/Al4RrMO\/jL9O3b1+ywww5m3LhxpkKFCubvv\/8Wkw8OmWhMcEylP8iHNWH5TRayxXX4u+AH4yZen3zyibSVCIF59dVXzRFHHGE2btwofjMtW7Ysdf\/7778vc2OurVu3lu+cBAYz1V133WXWrFljdtppJ\/kOn5z+\/fvHXTMOae6zpixuCEpg8N+57bbbxI\/nsssuM1dccYVgyRwgehAviACCBgpfIOY6YMAA+fzRRx81vHUefvjhMv5169aJOQsTIaayc8891\/Tu3bvUHH799VfzxRdflBBB\/j9lyhTDuuBfVKNGDRk\/GIAjZBO\/KivMFz8i9gNjoE\/Gs++++8olQ4YMkX1C\/507d5bPMP3hc7Vp0ybz4YcfivYP8x3\/h1CjGZs7d6557LHHZE8wT\/bhNddcY7p27Vpq\/Ow5+gAb9grtYBYDA6cG0mvhMDnee++9hvVmzRYtWiT44KcUT9jzDz74oPnxxx\/l3v\/973\/mjDPOMN27dy9163PPPWc+\/fRTGQvt77LLLgbTIfsJwf\/rvffeM6+99pqYNNnrzJk12W+\/\/cQkWb9+ffPwww8LLmAMaUejx2+LedM+GOEDxp9hw4aZyZMnixby1FNPld\/rwoULRSP50Ucfye8c891hhx1Waqzgx15jzVlXxsw+tL8\/2uO3Qx\/40mF6ZJ+hxQR39gZjjiUQC+7hfsbx+eefm7Zt2xq0xlYwdbMuYDN\/\/nzTsGFD8dGymNnrwAwTJGZixstvnvHi8xdr33F\/vLnaazD9opnt1q2b\/B7ZK5g8nb+BeHtFv88MAkpgMoNrJFv1IzAMlgd5q1at5CHEA8EKfhE8GN3+IbxRoy1Bw8FDDSfaY489VogI8t1338nht\/3225vnn38+pmkGAsXDgQMSweRz\/fXXy4PeT+JpYHg4MT60Lzzs0cY4BadgDpv999\/fk8DgjMuDy2pvIBE8vCZMmOA7Jh5q9INjrCUf9uKgBIbrcWg977zztmoDPyEe9pbAcNBDBCCUHAD43HBoQ1b4HPw5WPkbEvbvf\/9byAGHD+SCf3MAs74QEw4uhHWALHE4tGnTxjRo0EAcqSG4EBXGZokcBwxkFyyt9uz888+XNpgHBwnC\/gAXq4Hh8OdAHjt2rPn999+F4DB+e0BeffXVEpnEZ+wryA7khf44LDn8EA5vDr4DDjhADjjIFySJA7BevXpm9913F1LnJ9Zh+5lnnpFLGNOgQYNKMPK7D6IFSULT16NHD7mMPUs7TrPllVdeKfMDZ7BAo3jOOecIFuwlsIVYQNQw+fIbxQQKiQcLsD7wwAPNHnvsIYcz83ziiSdkTtZJHSz5feInxrzB2l4LCWAd0W7yHWPFSR6TMQKZwdEfYf15CYFwWW0sJImxQzhw9uZ38PLLL8v\/GRf4gj+4Q8oaNWpkXnrppZjPPxy30fhavzb2I6ZFSC8CweH3CY4QN\/7PMwH8INXWVwqSR2ACY2QO\/P7A+7TTTpOXAK99ZwcWZK7sM\/Ynz0OeN7Vr15b1Y36MeZ999ok5T\/0y8wgogck8xpHpIRaBYZC8pfHj5C2PByvCv3n4cfjx9o7wI4bA2MOcN0QeejyEnNqWRx55RB4k+Ldw+MQSDiYOAEueaAcSAIGyD1jn\/UEIDIcoDz4OOQ5f+8DhLdgSMx6AXhoYHoQctjwwrRMwD1S0ULGEA4x5pEJgpk+fLvi624CIfPzxxyUEhnFw4KBF4m3Tvvnzhti0aVM5qGbOnFnypmh9jZxr+f3335sjjzyyFIGhXQ495gpOmNkQtFkcoOwNe0hBFDkUmTPEBeHtH60Pbdi3cTeBsRiCMwePJTD2c0gRmg20M\/bAAn8OZKfZ0M4JAmC1CfY6Dh8Ocj9hPpAFSBqHNMKBzgEIGYJo+An+WozNkj6ug5QwzxtuuEHWD2IIftZp3raFZgUtEho6vkMgk5ApSAJaDCtgym\/COZdVq1bJiwMYOU2raEk5dJ0vG7Yv9gN9WY0U5AcSZNfo559\/Fq0TxMapoYMcgStaNcaBBhZCwcsObfIMsL9PSOwHH3wgJDtWWDyO3B07dpRoPSvO3xaf8\/t3ajt5xoCBNfmyTmBLf1WrVi15LmEWZlz8LhCvfZfIXJkLLw4QfrBirmGnHYj5ACqwL5XAFNCCxyMwPPx4CDrfsO1bDZoKDnOEhxZv9ja0mfBjNDfuNy9MBjxM+I6HahDhQEJ7gDoeoW0OLHeIdxACw\/2Mkb453DnkEQ5wHkgcMjzMvAgM5iWu56HOw5MHNw\/zeOYwP\/KRiAYmEQLj1y74cNBghrGCWQqNgZNQcoiDhVMDw\/VexJXPOewhhBwcznbZH7whc7hygGPWch7cfgSGt2hCzt0EhhD0JUuWlBo\/+463dmfkmSVLaMxshJM9tJ3+TX57D+0HawtJAovHH39cSDfaBLepzdkGBzimGbRyTkGbiEkVQdNA+LzXgY5WCcJjyYY9KDHJWE0EbYAlmhknGeRzSCSYo920wu8WrdecOXNKDnVLNtxkZ8yYMaJFscSPeQ8cOLAUObTtopFCMwWxgmChyeFZ4m7Tajwxh6GZ8RPmiBkK7MCYfWN\/W5BvtDiYjJwRgZhLMSWhCYScoOldunRpKQJJf078+b\/Xvktkrmg7mScvD2h6VKKFgBKYaK1HRkcTj8BwMGGC4GHC2yVvGxx4aC6effZZsanzbx7cHPDYgJcvXy4kA3UyZMctHIw8RDlUgiYrQ3PAw5WHpj00eeg4JSiBsaSMQ4XDAgKCiQEfD8bjR2DAgActGgze8Dg4mXc8n4qoEBgvAuB1SGIKYU0TITAcMjZyy64JmhxrqsEciekt3QQGLR+aPCeBsVoOp7bKEhgvB2+vHxhEm\/2FeQSzJVqeWAQGbSF4oSmwJN6rXbQBaPu8otvQbkFK7Bj9CAykgrG5CQyaMEgCBCkWgfEjG27iC2mCBHphRrQeGin7YuPXJmY\/CBUkCvOdn2BeRsuF9hPfJwiSTTmA1o1nUKzoQdoFW8xLPIdiiReBSWSulsCgHWPfqUQLASUw0VqPjI4mHoGhc6f6Hcc1DiLysOBjwdsSviq8AWEiQKyWBV8X1OBu4Y3p66+\/lrdN+2bqvAaiwJu2Vwg3b+VoSxDrt2HvDUpguN6+7TIPSBcHrLW\/+xEY7uNtDu0AD0lIFX3yf+vX4bVYhUhgUO2jHQEb1tHuoWwQGNYF8gKhwqyDyWrw4MFCOjkY4wkEFc0B12JusqacWASGvcyecmoqvfph76M1gGTttddepS6xJhyreQqbwNjxoG3Et8cp1kxnSWKqBIa2eamhT4gZL07gjfnKks94GjBMaBA4qxX2W2cvApPIXJXAxPsFhfu9Ephw8c9q70EIDG+h2Lx5QEMueKviDZ03EN6OMKlwkGNKsILqHiKAD4w7TT7fbbvttiWZc90Tts6QTl8C5zW86fLGywHpjESKRWDwLbDZeGkLp0PU1UQ8QGDwU7EOwn4Ehgcpb7kI8+aNGdW8V94Z53gzSWDwNcHnxDrx0q+fCSlbGhjrcIyWwO4JPwKDTxIHulMSMSF5aWDsYXjzzTeLSYXDkEgeZwSY34+MQxQHbzRt1gE5CIHBBwKNI1oGNHluHy27d6x\/j5c2wX4HgeI3EzaBsb5b+LGApVPsd3YeqRIY528LXxaIEY7ONlILs7PTZO0cC9f\/61\/\/Es0wewm\/HLe5CrMga8OLBgTGve8SmasSmKweUQl3pgQmYchy9wbeWvBRcJMB94zwDYGMQGSsc6P1P+DBwEPDaUrhzZc\/PJSx+1tBs8KhFk\/VDhlBu+MOP6UdzFOMmUgbZ5I5v0R2mDfwD0DrYwXfBhxVGQ\/EzGqP+N4SNv7tVPXjlMrbKKHd9pDE78E9Rzd26SAw1jEWB2QODcQ5B2e4d6oExs8HxjrxOh1+GQfr6TQhWb8H555yaxe4DwLMnvrmm29KHfh+BAaijInB6cPjRWDQpqEJgUAlmoTQ7mnnmrq1DX6\/dutQjdYAkm8FTRDmNCJzrEMwJB7zjFPYv7wYWBNI2AQGUoKvB\/sB\/yZnLS98iYhss2ZXazZ2+8AENSFxndNR2fpb2T1kNabuUiDse\/yB+NzuG\/Yj\/jzWPM3vmSgt63jute8SmasSmGifd0pgor0+aRsduTJ4a+QNNZ4WweZlcTpW8taJNgSi4Y7EQVPDg4KHB2YD+0YECcAZlweKVySRnRzRHjw0OSxRX3Mt48RUhT+CW63tLCWAwyX2dCs4+3E4uB0JbUQUJjGInBXr2Mr\/cSwkvBMh8oo3vV69esn\/aZfICWz8VjPjtTj2AHSbF6yzqTvbsNfnmOV4MGMewdRFCC2EgvwuRCExNkgmb+5W20E7zvwomPwgbLylWrJptQtOe76NaCEax+mEbckCkWl2vVkT3o4hdZAshIMC\/yh8GJgzJBASR1+8zaMNQRtkDzecUmkD8xz3WLLj9r1g\/mhIaM+SSBvijF8I80csCdl7770FJ0xHaPwIG+eAjZWTxDqoo40j4gXc8ZPiQEXbx74iAsWrFAWEGhyZB\/sJog3hwrmd35fd75B39r\/zN8ce4gAGN+uoiiYGHyuIDyYwKzY0G63BQQcdJB+zDpB07qVUhhXGC2niDyG\/iNUw4qjOOKzY6DUi2DALI\/zuIaRopGwEEC8CtEtOKJsw0pJ+d5s2Yspt7nX\/TpiHTTLJd2hf2ceY2iCh7B+eCWhn+S2Dr9XwMgdCvHkeWS0jONicP2hx+K3bMHC\/fRd0rjaS0fkykbaHsjaUMgJKYFKGMPoN8IDA2Y8HA8IhzcMXk4Tb5MP3HJy8ldpQRDtDDhsiWOzD0Tlz7uEhxMGFHwTqW5JO8VCLRV5og3Z5KJJLAjLEIcTBirmHN1xnsi0OVsiJdV7E9wbSw0GDkzHmIg5tDi4OBFvMkQceh7HVKNEvb78cJmgFEA5x3qxpj0ME8kU0BILfA5i5yxY4MYBkcMhyuIERZitU2ByyYMOBDIkEEzQMNmeI83Nb9gDTDH4ZqNY5zDnIyM3CAQmRgqBAljiAGCcHKP0xXjCyERMcRhAJDmeu5YAm4RwHFX4ZjAEMIDl8hhYMosTBy5whBZAOtE9orqxjNSSKebBO9AvRIJoJwse1HMQc7hx8aCBQ63M9hxLfQUA4RCB0mAgZP3sLwsPhBnFFGD+fs0YQXBxAmSOOmBAgQvrZI\/xNVArrDFnn\/8zJGV7t\/qWibcDXAs0ChIfcK8yL8Fw0P\/hMxaoPBT7gbAuhstaMxa0JgjRDVvitQZbY42BmE6Hxm+G3xtqwn1lron1YG9aM\/cFvgLHx24UMMS+IDNeylyDf7FkwZ5+S8oDvOczpm98jZBjTCwSHvcU6Y2rjd2HNf8yFNeE3CylEwwWZsEkJuQcHe6LM8GmD6PAb4xnDywRzw\/eHveRXmoT2cKbnBYE1g5wwBmdeFQgdewACgXYFggJmzmzhaILoE60XawlGrKczf5R73zk1xPHmCk6sC+0zXiKf+H3al5zoP\/nzf4RKYPJ\/jXWGikDeIsABDXmG5DjFZvSF0Do1GnkLhE5MEShABJTAFOCi65QVgXxAwJr\/8M\/YbbfdtpoSztf4a8UKdc4HHHQOikChIqAEplBXXuetCOQ4AjYCCnMapgFMEJg+MLmg9ifaBVOYV\/h+jk9dh68IKALGGCUwug0UAUUgZxHAjwm\/HHyFEBx+yeSK\/wpOnvESD+bsxHXgioAioARG94AioAjkNgL4uxABhAMpDpaqccnt9dTRKwJBEVANTFCk9DpFQBFQBBQBRUARiAwCSmAisxQ6EEVAEVAEFAFFQBEIioASmKBI6XWKgCKgCCgCioAiEBkElMBEZil0IIqAIqAIKAKKgCIQFAElMEGR0usUAUVAEVAEFAFFIDIIKIGJzFLoQGwtF0UiGAKbd9op2IUpXNV41Y5m0ILDzbXdfzdzd\/s7hZbCvXXHYwaYP\/p3zvggymzYYMosX57xfvKhA0pBUG9NRRFIFgElMMkip\/elHQFq81D9NdsSVr+pzHPFhg1m+Yb1qTQR6N51x75s5jWtYuZf0iDQ9clcdP+Y+81FvUuXAkimnVj3fLVyO9O4SjVz0q5l0930Vu3VqlDR1KxQIeP9pLODMH4DYfSZTsy0rfARUAIT\/hroCP5BIKwH2s033ywZW3NFskVewOPt5b+Zpeszq3n5YPoHpnWb1hmHHwLTuErVjPdDB7lGYsL4DYT1e8\/KBtBOsoKAEpiswKydBEEgrAfat99+61s5N8i4k7lGzWXJoKb3KAKKgB8ChWiSUwKjv4fIIFBIBCasuUZmsXUgioAikFYECvGZogQmrVtIG0sFgbB+gGFoYMKaayrro\/cqAopAdBEoxGeKEpjo7seCG1lYP0AlMAW31XTCikDeIRDW8zNMIJXAhIm+9l0KgbB+gLlMYP5et6kUhu7\/b1OpXKnv3f\/XLagIKAL5gUBYz88w0VMCEyb62rcSmAT2wJ9\/FIdNQ1LWrduYwJ2lL61Uqbx8AJmxf5JuzHHjihUrzLRp08zUqVPN3XffnY4mfdv47rvvzPDhw03nzp3N0UcfndG+tHFFIBcQUAKTC6ukY8xbBML6AUZZAwNpSZWwBNkwkJpqNSoKoUlWyOEzcOBA8+mnn5ovvvgicDMbNmwwFRLIm\/Luu++aV155xbz44ovmvvvuExKjoggUOgJhPT\/DxF01MGGir32rBsZnD0BcVv4RO\/9KUcUyZnOFMtLC5orFf8eS8qs3y9dl1hfFvC4VMjNq1Chz\/\/33ByYw06dPN8uXLzddu3aNN\/xS30M627VrpwQmIdT04nxGQAlMPq+uzi3yCIT1A4ySBiYecdm0XXEm2Y3bxScssRa87AZjIDSxyEz1GtuIViYRGTlypHnggQcCEZiFCxeaU0891dx4442mW7duiXRjMCEdddRRSmASQk0vzmcEwnp+hompamDCRF\/7LoVAWD\/AqBCYWOQF4hKPtFQu511aYO0mfxISj8jEIzFoT4YNG2Zq1apl1qxZY77++msze\/bsEgKzbNkyM2TIELPPPvuYL7\/80lSvXt0MGDDAlC9fXu4bMWKEOeKII+T7vn37msqVK5vnn3\/ezJkzR9qcOXOmueqqq0zz5s1L7RUlMPrwUARKIxDW8zPMdVACEyb62rcSmH8QSJa81Krwl7RQs\/yqmLtp3eZtzPKNVYwfmSm\/usiU+8fE5G7Ij8SsX79e\/E9w2G3WrJnc1qtXLyEd1gfmwgsvFP8WTEt\/\/fWXadGihRAaTEarVq0yTZo0MUOHDi3RwHz11VemS5cuZvLkyUJquJZ\/v\/3220pg9HmhCMRAQAmMbg9FIEQEwvoBRkEDs+yXtZ6RRfi5rK\/hXYAQ8hKPuLiXc8XGqmb5hiqeq7zNr6VDsu1F+MTsWKfyVvc8\/vjjZvz48eb1118v+c5tQnrhhRdE64K5Z\/PmzaZNmzZiNrr00ks9CQyRTGPGjCnRxlCtmDpVaHacohqYEH+o2nUkEQjr+RkmGKqBCRN97Vs1MP8g8MN33hqUbBKYin\/4+8TUrbd1EcRLLrnEoIX5z3\/+40tg+OLPP\/80EydONEVFRUJOTj\/9dHPFFVd4Ehjb0IwZM8zcuXPNokWL5F53lXIlMPrwUARKI6AERneEIhAiAmH9AKOggYllQvq7tn9o867bLDeVygarFh1L+4IvTIUViWlgTj75ZFOmTBnRwlhxa2BmzZpl7rnnHsPntWvXFg1M9+7dfQkMIdVXX321adq0qTnnnHPMc889Z6655holMCH+LrXr3EAgrOdnmOioBiZM9LVv1cA4EPDTwnDJhprlzOYK3hvGOu\/WKl\/sD+OUtZuLHXj9zEZ8F8v\/he9r19nWMz9Mv379xD8FbUmVKsVmKYgKWpZ58+aJxqVVq1bm4osvNmeddZZ837p1a9OjR49SBAY\/F0gN8tRTT4lPzX\/\/+1\/5\/zPPPGOuv\/56JTD6rFAE4iCgBEa3iCIQIgJh\/QCjoIEBdhLWoYnxy7KLOWnjdmV9iUyiSwdxQfycd+Plg4FknHDCCfJn8ODB0tb5559vyO0yYcIE06BBA3HaPfzww81dd90lzr2Yjjp16iSkZPvttzcNGzYUk9J1111nfv75Z4PPDCQI4rLbbruJ\/8tbb71lJk2aZOrUqWOqVasm\/Xz22WfSL5FMieaQSRQnvV4RyAUEwnp+homNamDCRF\/7LoVAWD\/AqBAYC0a8XDAQGSQZMhOPtNgxxAuftte98cYb5t577zU43x566KFm9913NwsWLJCooo4dO4p\/zCOPPCJk5fLLLzeff\/65aFn69+9vMEENGjRIyArXkw9m6dKl5sorrxSn3bZt2wohOvvss029evUkQV7NmjXNJ598YkaPHm3IyHvQQQcZfHEOO+ww\/TUpAgWNQFjPzzBBVwITJvratxKYGHsgkTICltR4NRcv867znqDERbeuIqAIRAsBJTDRWg8dTYEhENYPMGoaGPey2wrTscxLyW4VCAuSzqKOyY5F71MEFIHkEQjr+Zn8iFO\/UzUwqWNYMC2QVZXieajuUfXvtNNOkkWVnB477rhjCQ7k+xg7dqx5+umnzY8\/\/igOnvhBEE2CH4OfhPUDjDqB8SM0ltjYv+11Th8aW3nakhR7TaIlAgpmk+tEFYEcRSCs52eYcCmBCRP9HOp75cqV5vjjjxdHyzPOOEOypBJpQqIxSAlOlqSBR2699VYhMPg1EDYL2XnooYeEyFBF2Dpiuqcf1g8w1whMDm0bHaoioAhkCYGwnp9Zmp5nN0pgwkQ\/h\/omMgRnTTQwpI+3Qop4UsHbaJAffvjBHHnkkRIqe8cdd5RcR8TKiSeeKA6aOF16SVg\/wGwTmHWbN5tGe++9VWhwDm0HHaoioAhEDIGwnp9hwqAEJkz0c6jvadOmSX2bCy64wJQrtyWxGqGxpIYnDJbvHn74YYkseemll0zjxo1LzfDoo4+WIn7O1PPOC8L6AWabwCz5e51ps18jJTA5tP91qIpA1BEI6\/kZJi5KYMJEPw\/6tonGCJdt3769ZFGlmjChtBTxcwral1dffVVMTxAZt4T1A8wmgVmxYYNZvmG96XBAYyUwebD\/dQqKQFQQCOv5Geb8lcCEiX6O941T73HHHWdq1aolvi1oZs4880zz5ZdfSq4Ot6CZQUOD1sbp9GuvC+sHmE0Cg\/Zl7aZNSmByfO\/r8BWBqCEQ1vMzTByUwISJfg73\/euvv0qtml9++UWyrvLjQUhORqE9SIpbSBlPMrKpU6dKltVC08BY8sK806WB8YtEcmJLiLQV579zePvp0BUBRcCFgBIY3RKKQAAEMAGdd955pmzZsuaxxx4z9evXL7kLDQzZVm0tG2dzNjoplgamSdV1Zs6qSlI7h7bM4n+I0F6ttjTl9dmU4cZ0uDz2NT5toUnasdWxSd0rNznH5jOOv\/dsaZZu3CCXV\/p2ljm1900Jm5DIA4NAWvzKDQRYvuIxVCo24RFOnS5SQzZefKUgqNQzyoQQoo+58tlnnzW\/\/fabOfDAA80NN9xg9ttvv0x0p23YsyffAAAgAElEQVQqAjmDgBKYnFkqHWhYCLzzzjsSRbT\/\/vtLOne3KQgfmIkTJ4oPTMWKxYUErZAvZsqUKWJicjoC2+\/tD3DFhJtNzR43Gfs33wf5LMg1Xm1hQqrx8eNJ9enVntc4Fo+7wRR162\/KvDhY\/g6qgUkkG28qe8LWPaKNZAnN4sWLzcCBA82nn34qDt9BhQrUbn8pv3spFEkoP3mFvvnmGykcyT57++23fcPzg45Dr1MEchkBJTC5vHo69owjQJE+NC84695zzz1mm22Ks7g6BY3MzTffLCSmWbNmpb6jXs0OO+wgBfu8xPkDXNyjjNlrQnGxQStBPgtyDe05r7M+MMnc6zc2Z1vWcbfMWVVM0WPFFaPjEZh49ZBoY9N2ZUuw2fxPfSR3xeqyxUofkbLri0zZDUUmXmmBeEUcY200wuoxEwYlMOyp5cuXByrIuHHjRiFIt912W8kQ8L2ixpI7vD\/jPwbtQBGIGAJKYCK2IDqc6CDAIXPUUUcJKSEpnZcGhdHydkweGJx7nWYEKgoTZk11YcxDsQhMspqUZO\/LtAYG35d1E28NpIGJR1wgLZAVN1FJdKdAbMqv3hyTzCRTF4l8QQ888EAgArNw4UIJwaeII0kP4wl+V6tXrzZ77rlnyaWYkVq2bGluv\/128b9SUQQKFQElMIW68jrvuAhQPXj8+PGSiA4tiltwyiXrLkLCOw6yY489VlT9lBOA9Oyxxx4SYu2lueE+foDzXh1rKjU6oqT5dfPel3\/H+8xpbuL6oPdxHaRrz3anJNynVx\/ucSyZM8Ws27TJFDXcUi25zPxppn33Plv5wMQiLxCXjdsVV6H2k8rlin1k3LJ2U2lTnvv7in\/4E5l4JAZiSxJDItHWrFkjVaRnz55dQmDwL8J5m8zNmA6rV69uBgwYIGH03DdixAgpR8H3ffv2lWzO7JE5c+ZIm\/hLXXXVVaZ58+aec2PtWrdubV577TX1g4n7K9YL8hkBJTD5vLo6t5QQOOaYY8z8+fN928CshHMlUlRUZB5\/\/HHzxBNPGDLzcmh16NBBcsTUrFnTt418JDA\/z5kiYdPxCEyy5KVWhWKTVM3yq2Ku77rN25jlG6sYPzJTfnWRKbd6s2cbfiRm\/fr1kpUZTZs1F\/bq1UtIhzUhXXjhheLfgmnpr7\/+Mi1atBBC07VrV7Nq1SrTpEkTyeRsNTBfffWV6dKli5k8ebKQGq7l3\/i4eMnLL78sDr1PPvlkSvtbb1YEch0BJTC5voI6\/pxGIN+ceH9+ZoBZc9y1sibWedfPiXfZL2s9I4uKKpYx62ts8XVxLjDkJR5xcW+IFRurmuUbqnjuk21+3eT5OT4xO9YprnPlFEgqWjlnZmW3CQl\/Jwgs5kciiNDSYTbCoduLwBDJhGOu1cZQawuzI5odt9De6aefbgYPHlzKrJTTPwIdvCKQJAJKYJIETm9TBNKBQL458VZ6cq1oX6zEcuL94TtvDUo2CUwsU1LdelW3WmKi0dDCWM0bF3j5wPz555\/i1I1mDnIC6bjiiis8CYztZMaMGWbu3Llm0aJFci8RTm6hX\/YM2j8VRaDQEVACU+g7QOcfKgL5pIEx3fqbFRMGiuMuEk8DE8uEtKFmOV+n3US0MLG0Lzj1VliRmAYGp9kyZcqIFsaKm8DMmjVLItb4vHbt2qKB6d69uy+BIaQaU2PTpk0lUeJzzz1nrrnmmq0IzLvvvmuWLFkiZEhFEVAEin0IvYh+PmOjmXjzeXVzbG755ANjs+7isIvE84HhGj8tDN\/FIjF8jwNv5bLFf6ys3bzFedfPbMS1sfxf\/MxH3NevXz\/xT0FbUqVKsVkKooKWhWSHaFxatWplLr744pLIMxxuqVTu1MDg5wKpQZ566inxqbGJEG2tLeeDme8++OCDUlXN8a\/BiRiSpKIIFCICSmAKcdV1zpFBIF8IDNoXCjaK5iUBAkOGXTQxfll2MSdtJIy6dI3MpNcP4oL4Oe\/GIi\/cB5E44YQT5A9+KMj5559vyO1CeYkGDRqI0y6RaHfddZc490JcOnXqZK6\/\/nqz\/fbbm4YNG4oWhWrmRBThMwMJgrgQ2Yb\/CyH4kyZNMnXq1DFENRERd9FFF0kmaATiwvcQIb8It6RB0hsVgRxBQAlMjiyUDjM\/EcgXE5I7664QmX8y8AbJxBsvFwxEBoHMIEEJDWYiktnFIi12Z8ULn7bXvfHGGxI2j\/PtoYceanbffXfJwkxUUceOHcU\/5pFHHhGyQsI5ykygZYGEYIKiwCdkhevJB7N06VIJ1cdpt23btkKIzj77bFOvXj25lhxCEB238DlkR0URKFQElMAU6srrvCOBQD448Xpl3bXgJpKJl3viaWTci2aJjfvzeJl37fVoXCgjQH0kFUVAEcgtBJTA5NZ66WjzDIF80MCs7XrtVll3E9XAuJfVVpy2xRxTLeTobB9NC6KkJc9+TDqdgkNACUzBLblOOEoI5LoPDNoXZ+SREJcEfGCCroUlNFZL4\/y\/XxvOAo38O9mCjUHHqNcpAopAdhFQApNdvLU3RaAUArlOYJw1j+zEMkFgdNsoAoqAIuBGQAmM7glFIEQEctmE5Mz74nbYFU1MAk68IS6Bdq0IKAI5ioASmBxdOB12fiCQ6068RY8V1yUSwnJWFeP8v\/uzDgc0LrikU\/mxS3UWikA0EVACE8110VEVCAK5qoGxNY\/cWhb7f9XAFMgG1mkqAiEioAQmRPC1a0WAH+And55lavbYks9jxYSbBZh4ny3uUcbsNaE4xwkS9D6u+2PFCrPnBcOSupebiDyyNY\/cmhdIDGJLClgy0\/7fT6sGRre8IqAIpA0BJTBpg1IbUgQSRyAXNTDrNm82S8bfKAQl2xqYP37\/u5is\/b4uLtg1t69Uck2N7YtDp1UUAUUgfxBQApM\/a6kzyUEEctEHpqTmkcvnJd0+MJAVS1QWL1yZ8upaQrPXvtWlrXSQGrLxTps2zUydOlXqGWVKyNxLhl8y8h544IHm9ttvN3vuuWemutN2FYGcQEAJTE4skw4yXxHIRQ2MM3Q63RoYS1rSQVji7RknoUmWzFBwceDAgebTTz81X3zxRbwuS76nAnWFCsEKPL3yyitievu\/\/\/s\/8+OPP0pNJQo4Tpw4MXB\/eqEikI8IKIHJx1XVOeUMArmWB2bJnClm3aZNpStN\/xMubUFPJg\/Mt\/9oWOIRl3W7lpNuNlYtrom0sWpxjSS3lF9V7BtUftVmw7\/L\/7k55p6AzKCZSYbIjBo1ytx\/\/\/2BCQyFH5cvX266du0aaJ9CYI477riSa+nvscceM7Nnzw50v16kCOQrAkpg8nVldV45gUCuEZif50wR592ihoeV4OuMPOLDRAkM5CUWcYG0rNulmLikIhCZSks2xSQzkJg9\/zExBe2LStIPPPBAIAKzcOFCc+qpp0oRR4o5JiNUwf7zzz\/NHXfckczteo8ikDcIKIHJm6XUieQiArlkQrKh00JSXEnqkk1k9+mHv\/o65AYhLjtW\/NPsUGGLf8xvG4r9W5atrxZzO0BkKv20yfOaeCQG7cmwYcNMrVq1zJo1a6SKNNoQa0JatmyZGTJkiNlnn33Ml19+aapXr24GDBhgypcvL\/eNGDHCHHHEEfJ93759TeXKlc3zzz9v5syZI23OnDnTXHXVVaZ58+ZbjY++uP+ee+4xFStqAcpc\/M3rmNOHgBKY9GGpLSkCCSOQS068lZ5cWxI6LSQmRSfeWJqXWORlv+1+MDu4iIsX8JCZ+avr+pKZZEjM+vXrTefOncVht1mzZtJtr169hHRYAnPhhReKfwumnr\/++su0aNFCCA0mo1WrVpkmTZqYoUOHlmhgvvrqK9OlSxczefJkITVcy7\/ffvvtkmmtXbvWjB49WkxHtHneeeeZ\/v37J7zf9AZFIJ8QUAKTT6upc8k5BHJJA7N43A0luV3SoYF5+9XvPddrY7Wy5q8G5T2\/g7w03O6HhNYZEvPV6rqe91RZsNHXpNTu2N23uufxxx8348ePN6+\/\/nrJd24T0gsvvCBal6OOOsps3rzZtGnTRsxGl156qSeBIZJpzJgxJdqYcePGmZtuukk0O05ZvXq1wQSFNgfNDuNo2bJlQljoxYpAPiGgBCafVlPnknMI5IoPDFWnl29Y7+3fkqQTb7YIzPQ\/DvDVwvgRGJx6\/3VI7a320yWXXGLQwhDSbMXLBwYfFaKEioqKhJycfvrp5oorrvAkMLadGTNmmLlz55pFixbJvUQeecnff\/8tWpyLL75YSJGKIlCoCCiBKdSV13lHAoFcITAluV\/mTxPc0uHEG8uE9FfDCr4RRoloYWJpX3DqrTJ\/g+c+8CMwJ598silTpoxoP\/wIzKxZs8RHBWJDuDMamO7du\/sSGEKqr776atO0aVNzzjnnmOeee85cc801MbMW9+zZ0xx66KGmT58+kdjHOghFIAwElMCEgbr2qQj8g0CumJCWd7m6xHGXoafDhEQ7sZx4Y5EY7rUOvPjDIL85HHfxf4nlyBvL\/8WPvNBHv379xD8FbUmVKlWkX4gKWpZ58+aJxqVVq1aiHTnrrLPk+9atW5sePXqUIjD4uUBqkKeeekp8av773\/\/K\/0laR64XPw0M15AT5t\/\/\/rc56KCD9LekCBQsAkpgCnbpdeJRQCAXnHglcV3PyjErTQupSbIadSwSgz8MIdR++V4SXUOIC5JsBBIk44QTTpA\/hDMj559\/viG3y4QJE0yDBg3Eaffwww83d911lzj3Yjrq1KmTkJLtt9\/eNGzYUExK1113nWTWxWcGEgRx2W233cT\/5a233jKTJk0yderUEcddiMrBBx8s\/b3\/\/vvm3XfflQR6KopAISOgBKaQV1\/nHjoCuaCBsc67sSpNJxtGbReADLzkgolV48iSGe4JSmgkiV3AZHbxwqftWN944w1z7733GpxvMePsvvvuZsGCBRJV1LFjR\/GPeeSRR4SsXH755ebzzz8XLQtRQ5igBg0aJGSF68kHs3TpUnPllVeK027btm2FEJ199tmmXr16Enl0yy23CGHp0KGD+L5AcnAQVlEECh0BJTCFvgN0\/qEiEHUfGJLWLa\/fqgQjzyR1STrxegEfhMi474PYOCVe1l3ntbYuUqLJ60LdNNq5IqAICAJKYHQjKAIhIhB1ArNiw3qzZp9Ds0ZgnEthyQyfBak+HWQZ8W+p+U9laiUtQRDTaxSB6CKgBCa6a6MjKwAEom5CWrRm9VYOuyxLupx4E1liCI2TzKz45\/9ebViSYr+DuCRT5yiR8em1ioAikF0ElMBkF2\/tTREohUCUnXht7hchLHEcdINc0+GAxjEja3RrKAKKgCKQCAJKYBJBS69VBNKMQJQ1MKZb\/+Lkda66R2FpYNIMvTanCCgCOY6AEpgcX0Adfm4jEGUfmMWfThJwSyWtS2Miu9xeOR29IqAIhI2AEpiwV0D7TxgBQk5TFcJVjzvuuFSbSfn+qBKYdZs3myWfvakEJuUV1gYUAUUgUwgogckUstpuxhBg05YrV85ss802SfVBUbyLLrrIXHvttUndn86bompC+vmZAWbNccX4qAkpnSuubSkCikC6EFACky4ktZ2sIXDaaaeZhx9+2FSuXDmpPkn5\/sorr0SKwDCRxT3KmL0mFJWaU5DPglzjbv\/bb781e+65p2+fRY\/9VWoc6sSb1FbTmxQBRSCDCCiBySC42nRmELjzzjtTJh+JtIHGhrTxTz\/9tDn11FMlk6pb9tlnH7NpU3GaercMHz7c11wVRQ1M5RNvNEvG32iKuvVXDYzHelJpmuy4lBUg2+6yZctMjRo1JMvuKaecIhl1X3vtNcO6U\/iRPUPdI4gzWXp\/+OEHKdpIFl4v+eqrr0yXLl3kKzLynnHGGUI2\/eS7776Tvjp37myOPvrozPzojJF5kRGYMgjbbbed9PPggw8KBpRW4HcydepU8\/bbb5sbbrjB1KxZM2Njcfed0Y4y2DiFPCkJwb5p3Lixb0+Uj\/joo49Sfu5lcCqhNK0EJhTYtdNUEKA2jC2Ul2w7QdvgkMLnZv369VK3xovA8ODm4UP9G2reuMWmm\/caaxR9YGz4tGfW3SSdeHvWq2KWVtkzL8Ko2Q8\/\/fSTefLJJ02FChXkYIeQULCR0gFWSP0PWYGQQGQQCMwdd9xhqlataj744IOSgpDOvUGRRipSI857vfYPhAJt4osvvmjuu+8+ITGZEGo6UT6BStuffPJJCTnhd0jJg5tvvll+H1Thfv7556UG1I477pjWoXDYg7cVZ99p7SiLja1cuVLWjLpYXbt2LenZPddRo0aZ119\/XYix3UtZHGZku1ICE9ml0YGlE4GFCxeafffdN+EmDzzwQNO+fXtz1VVXmcMOO8yTwPzyyy9SE6dv377m0ksvTaiPKBIYijdSQiBdBGbE4MFmxKDBxSarxYsTwidqF1Ntmj1x4YUXmksuuaRkeNRCevzxx0tp56gYjalu7ty5JddRE2n8+PHmyy+\/lDfvM888s9QU\/\/rrL9G4bNy40fzvf\/8rda8fFvTRrl27jBIY+qboJL8DJ4Fxj4lDFlzSTWDQbqLVgfzluxTSXFNdSyUwqSKo90cKgb\/\/\/lu0JU7hUBg6dKhUB05UKNzHGxJmg6ZNm3oSGN7AKeJHFeFENUNRNCEt73J1ieMueKXqxHt6585m5tRpGSEwd07+rmRJP1j0h7m2Yz3Tun6NRJc58PWbN282++23nxRUREviNJOgCXFGtvkRGEgQZshddtlFqk4736jR6tDHxIkTtyI\/foPEhERxx0xqYOib+aFhikVg+L1cfPHFaSUw4MVva9q0aWLCymcppLmmYx2VwKQDRW0jVARQ599+++1if4es+Ekqb\/+xCAxmphNPPFHU58cff7y8PfMgcqq7\/cYUtUy8Ej59cjmTqhMvWpeZ06YJcbGSbg2MJS+QFiten6V7c1599dViJtl1113N3XffbVq12lLs0tmXH4GpVq2a+DPgU4UpE82eFfYRn+Fn4tbepIPAQPAxY\/GWz+8GE8Ztt91mateuLZ8xH6L71qxZI6YxfMX4zovALFmyxLz88ssGAmU1I5bAQMRGjBghfj\/gA2GjHSpzQ87o94gjjpCXCjROVOD263vOnDmiffn+++\/FV6RFixbmgAMO2Kpvxgi5wswCsfziiy\/MHnvsIVoj5jRjxgwhnevWrZMXEeb2448\/ionYrQmjLfCYNGmSmOfatGkjeL300kti9uMe66fEtRMmTJDxIWjcjjnmGPGJsgI+mNR4Jrz55pviO0QQAmOkzZ133tmce+65xmuuaHftOKhyXrFiRWkW\/B966CFTp04dg4aZeYETbbF3xo0bJ32h9ePzzz77TLTJ4GxJs9+40v2byVR7SmAyhay2mzUEIA38OHkztg8JZ+f8sD\/++GMzf\/78pMcUi8DgYIcPBIcPpgEe\/Dz8GjZsKG+jzgedewBR08DY8GmrdWG8iWpgTnugNHHJBIFB2zL9m2KNi1syTWLwecJciPYEoorvC34rEJOgBKZBgwbiL0U+IiLqEMwumGDwJ\/EiP+kgMP369RNtIQcZe7R169bmkEMOEf+WsWPHmtGjR5vZs2dLVzjmsodxYPciMJAwDkbmjXkJsQQGTSS4cMhC+NA2QSz4DUJWOEBxYoYU7LDDDua3336L2TdO02hfrAbGq2\/ICxqiKVOmCDmArPFsQFsGaeP\/PXv2lD6vv\/56wQASA5HkXoiJU8CHzyFNBx10kKw5pHXAgAHmvffek7XimTNmzBghHvfff7\/cjt8T\/k\/XXHONkJLp06cbfFgwHSJgjXM2ZIZ5YI684IILShx03XPFD4p1AT\/+DRnjeUQfjz76qIwB6d+\/v4wL0lKpUiV5oWJMpItgPcDliiuuEFMnhMxvXJl2vk76IexxoxKYdKKpbYWCAD4J\/Oi7devm2z+HRK9evZIeXywCY1XrPKR5UNSvX18cGumTt1PeeDgMvIQf4Jm7\/GEeX1JD7pU3wSnDiy\/tsMUh1Ouzomv2MmWGOHxKAt5HW8yn2omOKJh\/7v3+yN4lw6z+zij598qjtvh6eH1W94aG5odBxeTw0VGjzKMji+9zSzo1MF1Hf2Ze7tPUdz0hMV7kJukN4HEj6472gigk3vTROnDAWYmlgcHUxGGKI+8777wj5rXLLrtM\/Efw1coEgeEw5hBGU2mFz3ijxwmdQxgnXQgGwnhWrFhhnnjiCfm\/lwkJwoDmwU1gMPdYLNA2oCHlAG\/ZsqXsc\/afvYe24\/XtPtS5x903WhVwtISLa9CUWY0ZJmDmz2\/S9o1WhnWAHFgi4F5qfs+QA\/v8YL0hftwHOaBdiJDTCReiiNaEFytIF75x4ABx4NmAE7clTGioeD7YvFRecwU7xmAJDFFnaL+I+LICMUObB3Eiz5X7HgjZ3nvvLb5XrAEEPNa40vlbyVRbSmAyhay2mzUEOPSxkfOg8RO3V3+ig4tFYHgj50FPOKkzNw1vlbwlEXLK246XRE0Ds3jcDRI+nawGxvq7ZJrAxCMo8b5PdP39rscUgpaNg5CDiLf5oASGA4Q3b97Ge\/fuLQct5iMkEwQG8wNv51YT4DUnDjkORg5pDkc0JfZ6LwKDKQVzhZvAOJ14rY\/YrbfeKqYx6ydm52rHEatvr0Pd3XeTJk3MSSedJFohK+6+3fegbcLUgxnHL4yZFAlobNCmWEF7VrduXSFExx57rGh40OhYgcyiqUHrQZQWGlpIHVop2nJqOTARsd6xCMwzzzwj91kCw75Bu\/Xqq6+WWkY0RQcffLAZOXKkcd\/DhcwFcsX94B1rXOn6jWSyHSUwmURX284KAjyUeaviR+kn2LFjaWjiDTQWgYl1b58+fUrexNwmBu6Lkg8M\/i8\/rVsbqPI0Y3cnt+P\/9Yujfz0lnRqYeAQlnoYm3nr7fQ\/p4C3XmQUa0wQHx6pVq8Tnw5LYeBoYnHXxA4EEcbCjSbRh0JkgMGgJICF+ZPrXX3+VN3cIQPPmzYVQ4SOSKoFZvny5+K1gziBPjReBidd3EALDmCEDHN5WeLHgc+YOeUgXgaEtNGXMBTJjyZnt15rSMC9DdHiBQgODKYkXHcw4NrdPMgQG0gz5QmPmFMay\/\/77i\/koHoHhvljjSvY3ks37lMBkE23tK2MI8IBCvHJPoCHByc6pOk90ILEIDG8yv\/\/+e4mzo7NtHnD0i+3fq\/RBlDQw5H9ZMWFgShoY8r18tCw7BIZe\/Hxg2uxdIyPRSDiBsp\/c0WaQZ7QOmAysg6QXCeHggshaMm3NKzhectiVL19ewMPEFDSMOmgUkg2DdjsO8wIAscCcwF7G3wLhsOctP1UCg9aA+TI\/HE69CEy8viEwjNNJvtxkBE0Wzqz83iyONikghAK\/o3QRGLQc5G7Bvw2CBBF1EicwhqxAMDATop3D5wX\/GExdOPliEkK8CIx7rm4yAhkcMmSIaHicmudmzZqJVgitdDwCA5axxpXoMzKM65XAhIG69plWBHg4oX6NFYFEh5mKQsLpkgcW5gNnRAqqdd6IiJgg8sJLopQHZsm6tfErT8dJZGdzvnjNNZ0aGNoPIwqJt16iZnCe5BBDeMvHMRY\/Bg41K5gU0GA4k9ENGzZMCIw1R0CMaQdNnTOvDHld8GmIl8iOviBN9E3bTj8M9xqsXbtWTJr0iTMnpg0OexIwEoaNKYU+IfsQcnxMeEPnIIRg2QMRcxlEBIF4QKDQ7CBW8zB58mQxVyCYUnixsDmSKAUCUaI9K\/H6hhwwP\/DHNIv2w903ET3gQGSN9TnjxQZzGH5oCBom8jahkUWsDwxzhoh4CfOAENjMyZALIqwwtZUtW1a0KWhgrLkIzRoaGpuBGdLLy4sNsYdE8VuweOAXBMGzhMZrrkQS0T8RTvjOQKLZczhg2\/QQEESIHs7F9McziWcTgQVoBa0PDFFZ7LV440rrQzpDjSmByRCw2mz2EOCgIBqEaAEerPbty44AFT8PAB5SiQiHj33jow0eBvZhQzs81HnzwomRByYPex7OvBHxpsWDjXBU3mB5M\/ISJTCJrMjW1xKNBJFx5n7JpPMupg4OWw4ETAMcJuyTI488Upxf2XsQV3wTyD2ENoY9QYQa5iUOPpxmOUSsFoZ9BYHhkEfjwcFu3+a5F+dRv1ICOOGiMcFRFDMWB5MzLNuNGHuViClC\/\/mtcJhasxVtcBCTR4nSB40aNRJ\/CbQFmLg4HInmO++882SutMGhigkM0sABzcHKvyENjJmiq\/weaA\/BWRZfDvrA2Za+wTJW3zgAgyljwHcEExcvK+6+6QsyB+6YUapXry4pDfAv4jt+y2gnIHAQGogjpAFfH9YCPzrucQsEht853xEKjqmQFyYcca1ABlhzfudEPTrNgRAK1pS5QnbZL5A6MIAEsR4414I9Y3LPddtttxVM0ebgWA2ZqlWrlmDMmuBjxxjZO6w\/GNEGa4tzNOYmIp3YJ2hu8PVh\/vgHeY0r2SK5qf2Sk7tbCUxyuOldEUKABzdOc6iI\/YSHFxEAiQgP21iZdTkorCMiDwwSiX344YfyYOEtG20MDxS\/6AbGEhUTEvWPxP\/lxcEpmZCy5cSbyDrqtYpAKgh4OfGm0p7emz4ElMCkD0ttKSQEeCtBZRvrzZM3nyjmN4iKE6+tf8QSBqk87XVdNp14Q9pq2m0BIqAEJrqLrgQmumujIwuIAHZ7vO7541fojDDHWFFKAbtK+2VR0cCs7Xptcf2jJDQww+cZM+s\/g32ddy1o6faBSftiaIOKgAsBtKloUvGdIb+KSrQQUAITrfXQ0SSBADZenB2xd7uzadIc\/iuYeFJx4k1iWIFuiYoPzOJPJ8l4ixpuSWsfpJhjLJORGwAlMIG2hF4UEQTws3n22WfFJIxvE1mJY2XVjsiwC2oYSmAKarnzc7I20yYPGVsnxDlTCAwe+EpgilFZN+99yQa6Z7tTSmBKhsDEijjy2mlKYPjYUlIAACAASURBVPLz96ezUgTCQkAJTFjIa79pQ4DkYkuXLpWwRcIa3UJYIynf8fyPmkTBhGTrH4FNIiYkv5pHfhgrgYna7tPxKAK5jYASmNxePx39\/09mRn4LQjltbgovUDAh+YWihgliFJx4a45bb5ZvWF8CQ1An3r23K138Lh6OSmDiIaTfKwKKQCIIKIFJBC29NhIIUFMkVsRRkEGmo40g\/cS7JgoaGFv\/SDUw8VZLv1cEFIEoIaAEJkqroWMJhADZNW3hs0A3eFyUjjaS7dt5XxSceBetWW2COOwKwXFk4lUfmHTsAG1DEVAEkkVACUyyyOl9oSFAOCN1QJIVogqozJsqCUq2fyUw6UAuvDbI5EoGVDLRUpiP\/VSjRg3J0Es2aDKckgRx+PDhEtZvU8qTiZdwfrI0UwXYpqZ3z8TW7+Fz6vtQusDL\/IlvF+0RKUN6fbK\/kqU2VuLEVFBjXmTMJSMu2V8RajmBAZmoycJLaQIy2zKOTOddcvadyrzCvJfs3QMHDpR941cNm\/FRJuCjjz6KxDMrTLzcfSuBidJq6FgCIUBiKbLveoVMB2mAminO8vVB7snUNWGbkMjAu2T8jZJ9F1En3vgrTYp2cg+R\/ZkCfRzsEBKSKVI\/yAp1hyArznpGEA7SwpOGniJ\/XnuY1PKkpkdi1UIaM2aMRJNRy+ibb74x\/J8oPAiEV+Xz+DPzv4JClffee6+ks6d8gSUnFGakphLlEBgLuZief\/55KWzpVVg1lTFw2IO3FWffqbQb5r347lFigBpazjpW7rlSGJKSBBBjv1xXYc4jrL6VwISFvPabNAKQl1SFeiKxygSk2n7Q+6PgxFv02F+lhqtOvP6rR80tNB3UlnEWX1ywYIHUvho0aFDJzV7VqKnJRW0sCuzx5s0+dAq5R9C4kNMoVjVqvud+ouusUFARAkVJC1vfKOg+DHKdrWbtJDDu+zhkwSXdBIY0CGh1IH\/5LoU011TXUglMqgjq\/YpACgiErYEx3fqbFRMGJqSBwfclSOZdNyzpjkIin02lRkeU6obPEPfnKSxRqVsx22Ci2W233URL4jSTQCBsxWFu8iMwkCCIzi677GJIAeB8o0arQx9ULydyjurDXkJRSUw2TtMSZiSKjVL8EHNWusUSpFgExlajTieBAS8KLeJ4jwkrn6WQ5pqOdVQCkw4UtQ1FIEkEwnbiXfL3uuISAg7nXDsVr8961qsSt2SAHxSZIDArnh1odh5YfKhBXpz\/T3JJ4t5mEyfuuuuuUtWXVPNe4kdgMO\/gz\/D0009LMVBnRB1Vq\/kMP5NYBMarP0w4rVu3FjODnx8MSR0xY\/GWjxkMEwZaHKos8xnzoRoxVdQxjeHszneIm8BQ\/Zhqyt99912JZsQSGIgYlbfx+wEfCBvtfP7550LO6JcKz3fddZdonKhu7dc3FZXRvlBJG1+RFi1amAMOOGCrvhkj5Ir5QywxFe+xxx5S+Zs5zZgxQ0gn1aLxS2JuVIbGJOjWhNEWeEyaNMm8+OKLUggWvF566SUx+3GPMysvle4ZHwLppIK3k0SiOcKkhgnszTffFN8hqnAzRtqksv25554r1aPdcz300ENLxoHvnk3WCf4PPfSQpI9YuHChzIt7aYu9M27cOOkLrR+fU6m7ffv2grMlzX7jivsjiMgFSmAishA6jMJEIGwCQwQSEoTAJBp15F7RdBMYJ2mpedLArJAX+kTz0bdvX9Ge8MaM7wt+K26\/k1gEhsrpnTp1Mm3btjUPP\/ywQIXWAhMM\/iRe98b7hUAmcOiFPPhJv379TMeOHeUg44CG8BxyyCHi3zJ27FgzevRoM3v2bLkdx1zS5w8ePNiTwEDCOBiZN+YlxBIY\/FPAhUMWwoe2CWIxf\/58ISscoDgxQwp22GEHcUKO1TdO02hfrAbGq2\/ICya0KVOmCDmArB1\/\/PGiLYO08f+ePXtKn9dff71gAImBSHKv2x8JfPgc0oTZmjWHtJIQ87333pO1gijiewTxuP\/++wUD\/J7wfyLYAFIyffp0gw8LpkMErHHOhswwD8yR1GmzQQXuueIHBTbgx78hYziS08ejjz5aQlb79+8v44K0VKpUSfyRGBPtsh7gcsUVV4ipE0LmN65MO1\/H28eJfK8EJhG09FpFIM0IhG1CWt7l6hLHXSEyMYo5JlL3yAumTBAY+lkx4WYhL3tNKErz6sRuDo0E2guikHjThzhwwFmJRWAwNXGY4sj7zjvviCmIqur4j+y7774JExjMTmhtIBt+CRs5jDmEiRSywme80RMBwyGMky4EA2E8VHF\/4oknPAkMH0IY0Dy4CQzmHosF2gbMWhzgmLg4TDmA7T20E69v96Hu1TdaFeZuCRfX4FBsNWZNmzaV+aMxsn2jlWEdYmmt6tevbyAHvXr1EhxYb4gf90EOaBci5HTChSiivUHrAenC3w4cIA5oynDitoQJDRVk0Y\/A0CfYMQZLYIhwg7DisG0FYoY2D+JE8Un3PRCyvffeW3ynWAMIeKxxZfXHlGRnSmCSBE5vUwTSgUAuOfHWLw6MSVoyQWCs2SibGhgnAJhCKCbKQchBxNt8UALDAcKbN2\/jvXv3loMW8xGSqAYGDQN7Ca2Cn2B+4O3cagK8ruOQ42DkkOZwRFNir\/fygcGUgrnCTWCcPjCYotD63HrrrUKyODwRO1c7jlh9exEYd99NmjQxJ510kmiFrLj7dt+DtglTD2YcvzBmoh7R2KBNsYL2rG7dukKIjj32WNHwOLGHzKKpQetBlBZRapA6tFK05dRyYCJyRkV6zfWZZ56R+yyBYd+g3Xr11VdLLSOaooMPPtiMHDnSuO\/hQuYCueJ+8I41rqR\/6Fm8UQlMFsHWrhQBNwJhamAqzRpr1hx3bWANTCr+L8w73QTG7fOSDR8YSAdvuajxrWCa4OBYtWqV+HxgvvAjIfgjYHJBA4PWBD8QSBAHO9FNNnooEQLDGz6HGW3EErQEkBBMB16CYzBv7hCA5s2bC6HCRyRVArN8+XLxW8GccfTRR3sSmHh9ByEwjBkywOFtBQ0SnzN3yEO6CAxtoSmDjEFmLDmz\/VpTGvlbIDqERaOBwZRE3hzMOFZTlgyBgTRDvtCYOYWx7L\/\/\/mI+ikdguC\/WuHLhaa0EJhdWSceYtwiE6QNT4\/ATS2og5aIPTBhRSDiB4gNjtQh2Y\/JGi9YBk4F1kPQiIRxcEJhu3brJrda8guMlhx0V1REITqwwatsvyfQwQzlDugnFxgnXOt\/aa20YtNtxGK0MxAJzAm\/l+FsgHPYQo1QJDFoD5sv8cDj10sDE6xsCwzid5MtNRtBk4cyKicziaJMCQijwO0oXgUHLQe4WHHkhSBBRJ3ECY8gKBIP1QTuHzwv+MZi6cPLFJIR4ERj3XN1kBDJIMk80PJi4rDRr1ky0QjglxyMwYBlrXLnw0FUCkwurpGNMGQGc5XjoRE2UwERtRWKPh7deomZwnrT7ibd8TCT4MXCoWcGkgAbDmYxu2LBhQmCsOQJfENrp06dPKRLSrl07cTaNlchu0aJFcgiiNbFV2CEu+F5wuDm1RIyJoqc4ftInzpyYNjjsSYR31FFHiSmFPomo+f3338W\/hTd0DkIIlj0QMZfZwqkQD3xK0OwgVvMwefJkMVcgmFKIwLF5l0477TQhSrRnJV7fkAOwA38cftF+uPsmooc1ILKGvxE0L5jDiLxBwOqXX36RyCLE+sAwZ4iIlzAPCIHNnAy5IMIKUxu4Q0rRwFhzEZo1NDQ2AzOkl7WwIfaQKLQvFg\/8giB4ltB4zRXNHf0T4YTvDCSaPYcDNpFcCAQRoodzMf1BonAIJ+cQWkHrA0NUFoQ33rhy4ZepBCYXVknHGBcBHmr8uHlYuQUVP2+R\/LCjJrliQlo\/\/2hz5m29zOyiRUlDmG4TUtIDSeFGTB0cthwImAY4TCApRx55pDi\/8uaPTwi+CUOHDhVtDAc24dGYlzj4cJrlELFaGA4ZCAyHPBoPDnb7Ns+9OI+6HXPZ59yPQ6hb0HCQN8VLcLglYgrNDSSEw9SarTBFcRCvX7\/edO\/e3TRq1Ej8JdAWYJ7icPz444\/NeeedJ3OlDQ5VTGCQBg5oDlb+zfgYc7ly5URDQHsIzrL4ctAHzrb0DZax+sYBGEwZA74jmLjQMrn7pi80YOCOGaV69eqSEBD\/Ir5D44B2AgIHoYE4Qhrw9QFLMOMet0Bg0LDwHVotTIU43OKIawUywJqjASGc2WkO5LnDmjJXyC77BVIHBpAg1gPnWrBnTO65brvttoIp2hwcqyFTtWrVEoxZE8o6MEb2DsQEjGiDtcU5GnMTkU5o1tDc4OvD\/PEP8hqXm\/im8HPJ+K1KYDIOsXaQaQR4m+KBwttiLFm8eHGmh5Jw+2E68W6zy85mXc\/KJl4m3s3Tl5i1XV4yTTb2S3h+zhvygcCkBIDenJMIeDnx5uRE8nDQSmDycFELbUqENPKmy5sxb5XuWiG8IRIVEMUsnmFqYIo+fEgy8NrQafaNO4ya3C8fDn4mJc2L3Y9KYArtl5kf81UCE911VAIT3bXRkQVEACc41MaxfFywdXNd1CRMH5jNhxxXAke6s+564awEJmq7T8cTDwHMNDi64jtDfhWVaCGgBCZa66GjSQIBbNdoYYik8BN8F9xRGUl0lfZbokpgUs26qwQm7VtFG8wyAvjZkNkYEoNvE1mJneUDsjwc7c4DASUwui1yHgGc6nAMxAnOKw02jnw4DOI4FzWJqgkp1ay7SmCittN0PIpA\/iGgBCb\/1rTgZkS9E\/Ir8MZkC505QSAKCQdfdeItU5Juf\/7ixaZCnZ1MmbOqlDjxonWZOW2amTl1Wkb2kJqQMgKrNqoIFCwCSmAKdunzZ+LU9iCFOwXj3CGA5GQgNwYhhlEmMNTzqdnjJqnrw99IkM+CXOPV1meLvjHVZo4tceI97YHMERe70zJJYMiBQairiiKgCBQOAkpgCmet83amOOgSaWTzWXhNlOJnJOaKmvAD\/OTOs0pIiyUb\/G2JjN9ni3ts0agkch+k548VK8zmnoMEjkz4u3jhnEkCQ+QZlaFVFAFFoHAQUAJTOGudtzMlwyRZRm11V6+J8r2tURMlIMLygXFqYDLh75JNAoP2hURyaOL8ErhFac11LIqAIpAeBJTApAdHbSUCCHCQkQ2TDJOVKlWS7JZkMiWjaFQlrER2EJiqO+8sPjCpVpkOim2mNDBt27Y17733XtYIDJlcyYBKJloK8xGlUqNGDclDdMopp8j+I+MsWj80QzalPJl4qVqMvxZVgG1qejd+tn4Pn1Pfh9IF7ky8zntI5U9faCBjReIFXSe9ThHIFQSUwOTKSuk4YyJAFJKztorzYlJpk7o9ihIFDUyqVaaD4popAuNMXJgNMxIp2qkZRHJECvRBWCAk1L9xmimpOwRZcdYzgsCQFp409BT589Iaklqe1PRIrFpIfE9yRuoQkY36vvvui2lGDbpOep0ikCsIKIHJlZXScfoiAHGhiB4HxkknnWTY1NSVoSAdtVE4NDh0KK4XNQkrD8z45583h105WODIZR8Yaz6y65ppMxIEiTo31JZxVoBesGCBFPUbNKjYrwjxqkZNUT7qclFgj7FS18YpRNKhcSH0P0g1au6l7g3FH5XARO3XrePJNAJKYDKNsLafcQRQ3XOgUFHXSzgUKAjnPFwyPqiAHYRFYNr2us48+fZbOUFgICmYiLyEz53f4QvDHy9Jh38MUW377bef2W233URL4sw7hCbEVhyORWAgQexFoubeeuutUqUv0OrQx8SJE4WYUKA0nmBCYu8rgYmHlH6fbwgogcm3FS3A+VxwwQWiZYklVPxNxwGWbnjDMCHNrV18yI8b1t+0PL+\/yQUnXremJdF1SKdm5uqrrzbPP\/+82XXXXaWqL6nmvcRPA1OtWjXz0UcfSej\/Y489Zg477LCS26lazWdUXlYCk+gq6\/WFhoASmEJb8TycL1l2+\/fv7zszIpAgOU888UTkZh+GE691er3shv7m8gWDc8qJ14496EKmk7jYPlevXm369u0r2hO0Kfi+4LcCMXFKLALToEED06lTJ8N8Hn74Yblt5syZ5vXXXzeQba97\/easGpigu0GvyzcElMDk24oW4HwgL127dvUs5jh79mx5S65bt678nYxwYEGSeGMmosTLFIXaf+zYsXLNjz\/+KM6Zhx9+uBSAo0K2n4ShgTlhdGmzSzKYJHNPupx4OeAhJvEkE+TF2Scmo9tuu02ikPbYYw9x6kUrYyUWgcHU1LNnT3HkfeeddyTK6LLLLhO\/mn333VcJTLzF1e8VAWPE3zGKCUozuThlirIRqpDJGWjbpRAgyy4hpE2bNjW82SI\/\/\/yzwbHym2++kVBX3mxx7E1UCJXFAXj9+vXSph+BoQ4TBKZbt26mTZs2ZunSpeahhx4SIsNB5347t+PItg9MqqaYRPFzXp8uAkOb8eaBX0w2MvOSQJEoN5IpYkqCwAYlMGhw0AwSKt27d2+DaQrzEaIamFR2mt5bKAgogSmUlc7zeX7xxRdS0JHoDqccfPDBUuSxfv36SSFAxAnRS4Rh46vgRWAIlcWnBFMCIbJWID\/4NECAnBErzoFkm8AE1V4kBVacm\/KBwEA62AfOkhXU2jrooIMMRUXJ9WITJsbTwKC1g2RBgvB5Ya\/ZbNJKYDKxA7XNfENACUy+rWgBz4cDAY0Lf8qXLy+q+Hr16qWEyBtvvCGHCsnL0PB4ERh8GDArvfTSS6Zx48al+iOxGGNBA+Ql2TYhJepDkhJ4rpvTSWDizSNT5iMijzApnnXWWaVmhyYFHxbC9m1eGi8SQqg12jg0dciDDz4oEXI777yzaJXYKwgmpqBh1OoDk85dqm3lEgJKYHJptXSsSSPw66+\/mtq1ayd9fywCY6NSMFmR2MwpaF\/IDsybuT2c3BoYa8N11zbiuiCfOa8JU8MSD9xMERgbOu0Oqc6EpRifKvK0PProoyU+VytWrDAdO3Y0J5xwguQjsoLmDn8oZzK6YcOGCYE599xz5TL2FVrCPn36lNLSkdeFZHnxEtnRBqSJvmkbXzAVRaBQEFACUygrXeDzHD16tBwSyUosAkMyMkxXn3zyyVbNo5lBQ8PbuZcPTiY0MJNG32zunBCOo24sfNNFYJz+L25Ni5PAZcIPBiJM3iHMRDiG4+MESYFEnX\/++UJSCX+GtA4dOlS0MZSzwJQIiR0xYoSpWLGimCStFoYxszfZH\/hzkZhx5MiRAiX39urVy7eUAHuOvU1GXsxYmCqdYdnJ7ne9TxHIBQSUwOTCKukYSyGAT8tOO+0kDpAIURyx8sAQRo2PzKJFi5JGMhaB4UBDjQ9JccuQIUPM\/fffb6ZOnSrJz9ySSR+YqGlj0kVg4tU+0uKOSW9zvVERyCkElMDk1HLpYMmq27x5c7PDDjtICCrCWzFq+HiSSrhdPA3M559\/LsX93GKjk2JpYJpUXWfmrKokfhWSWn7xP0RoL0eCNK\/Ppgw3psPlW7r0uGb45aeZ4a98FA+arHyfLgIDQYkXYcQ1aGfQTKgoAopAfiKgBCY\/1zWvZ2UjPZwVetHGkEyM9Ozu4n5r1qyR5GD33HNP0rjE84Eh9Ts+MJgHnHLppZeaKVOmiImpXLlynhoYiNWKCTebmj1uKvmbC4N8FuQae5j7peNPGpQEb0wXgUmwW71cEVAE8hQBJTB5urCFNi2cK\/EB8BKik\/AV8Ps+CFaxCAy5OyBIkJhmzZqVag5\/BLRFL7zwgmc32crEGy9qJwgGqV6jBCZVBPV+RUARcCKgBEb3Q84jMG7cOHF29BO+P+SQQ3wdIYMAEIvAkOAOJ05CX53Zfm2iMmowucNubZ+ZcOL10so4tVJB5puJa5TAZAJVbVMRKFwElMAU7trnzczJhEolXj\/BgZcEc6R6T0SILpk+fbrcQrIytCwtW7YsiR4hd4f1xbj33nslcuTYY4+VEgLcSyZeUsxT+M+Z+Mz9BjHv1bGmUqMjSj5eN+99+Xe8zyxRsTf63YfpqPPF8VPvJ4JNMtcqgUkGNb1HEVAE\/BBQAqN7IycRwGT08ccfy9hJIEfSMC\/B\/wVNCOGpc+bMSWiur732msGHxU8wD9nU7+QcIUkZBSPJzFu9enXToUMHSQ9fs2ZN3zYyGYVEp5AayIvT\/+XgHY35aFlCUKTlYiUwaYFRG1EEFIF\/EFACo1shJxHYsGGDZDAloVgQod7MgAEDglya1WuyYUJymo9e6HOk6Xbfu+acyy41X82ZY2ZOnZa1+SqByRrU2pEiUBAIKIEpiGXO30miAUFTghnJS8iMixnHWSU4Smhk2onXmRPljC8Gmr0mFMn031+4QPLSlDmriqn\/XHYQUQKTHZy1F0WgUBBQAlMoK53H8ySsulGjRjk5w0xrYE4Y\/Z44GF\/WyJQK04bA1P34cVPUrb\/pWa9KVkxK6SAwELJEwsGZe7ycMYluHBy6b7nlFsn7Q6XzZcuWmRo1akiG3lNOOcV8\/fXXQqqHDx8uIf3Uz+revbtk4iXhIibGc845x9x4442eXVM+oEuXLvIdmkNKFzhTBtibiK6jvWeffdb89ttvUgzyhhtuMPvtt1+iU9LrFYGcREAJTE4umw46UQTI1nvUUUclelvGr8+GDwyTcDsEf7fkJ1Ph0GK\/oRGDB5sRgwZnfK7pIDCJZhfORFFH6ltRpwincDR8EBYICdXIL798S2JBfKAgK856RhAOHMqrVq0qiRgpReAW8hlRNBKJVQtpzJgxhgg4nMYpYMr\/yUP09ttvS70lFUUg3xFQApPvK5yH8\/vll18kqsc6x3KYcFD4CY68pPPnTTVqogQmsRUJm8DgrI2m48ILLyxVfJEkhjhxU\/vKilc16qeeesqMHz9eEhtCriTzskPINI3GZePGjTGrUfM99992220ld7\/yyitCoIjIo4K6iiKQ7wgogcn3Fc6z+a1evVoS0lH4DnMCQqiz+yDwmnYqpQQyBWOmTUhk90XcuWE+W\/SNqTZzrJiQTu\/cOSvOvPmggcFsg4kG\/yG0JM4IMwgEuYDiERhIEESHrNFEyDmdrNHq0AdJESkKOXfuXM+tR\/kMfgtO0xJmJML8cW7HnKWiCOQ7Akpg8n2F82x+PPwJbaaYo\/Uh2LRpk9RCIqzZy1mXBz3VgWfNmhU5NDLtxOuc8OIeZUqceCEwVXfeOeeceMPWwIAnofHk9mGvkbiwVStHzSoH4H4aGMw7H330kXn66aclDN9ZPZqq1Xx2+umnxyQwXhsZc1Lr1q3F\/0b9YCL3U9cBZQABJTAZAFWbzD4Co0aNKqXSd4+AN9uePXtmf2BxeoyCBiaXnHijQGAgxH379hXtCYQa3xf8Vtx+J7EITIMGDUynTp0MJR4efvhh2SUU\/Hz99dclYaLXvfE278svvyxm0kQTNsZrV79XBKKKgBKYqK6MjishBMiU65ftloaIHImiYyM\/wE\/uPEsihKxg7kHifebUqHB90Pu47o8VK8zmnsX+GurEm9BWK7kYkxE+KEQhEaoPcXBqAGMRGExNEGoceXEwxxR02WWXCQnfd999EyYwmJ3Q2gwePDilkhnJIaF3KQLhIKAEJhzctdcMI7B+\/XozdepUM3\/+fIk+imqYdRQ0MGVeHGyGzzMZj0TKBx8Y97ZduXKl5CCaMWOGmJIwC1mJR2BsrSxCpXv37i2mKZvZOVENDNFN7KX27dtn+JelzSsC0UFACUx01kJHkiQCOER+9tlnEk56wQUXSCgpfxNOihDqSmRGFB\/uUfCBKXrsrxJNzMxp0zLm0JsPBAbSgc+KU9uH9g\/H8lWrVkmul8qVKwue8QgMWhNy1ECC0J4Q3WSjhxIhMO+++66UyqANFUWgkBBQAlNIq52nc8Wp9\/rrr5eoDoRQVRx8mzRpIom+CE1Ftf7ggw9GDoGwNDBEuBR9+JBEIaGB4W\/E\/jsTfjH5QGCIPMIHxl1dHMKMDwtE2kYVeZEQQq0xZXbr1k3wZk8SNURhUKLqypcvL59jYvrf\/\/7nG4VkNzLJ9DBDYXqywn4ndUDt2rUjt991QIpAOhFQApNONLWtUBAYPXq06dOnj\/SNPwKaFt6KeVsm3BWhWjSOl1GTsPLAELGy+ZAtIb9l5hfXRCpqeJj8nQm\/mHwgMBQRJU8LNbiIfENWrFhhOnbsaE444QRz3XXXlWwx9iFVyZ3J6IYNGyYE5txzz5Xr8M2iHfavk4S0a9dOkuXFSmS3aNEi079\/f3PRRReZsmXLSnsQl0mTJpkhQ4bE9AmL2u9Ax6MIJIOAEphkUNN7IoUARRpJ7Y5KnoPgzTffFIfIK664omSc99xzj7nqqqsiNW4GE1UCw9hGntnZDJ+YvmKP+UBgyL9CjhXMRHXr1pVMupAUShacf\/75okFBu0XY\/tChQ0Ubc9pppxnCozEvjRgxQkyc7EWrhSHqiH1LbiNMQc8884wZOXKk7FXu7dWr11aOuRB17oeIugXt0E03bXEKj9ym1wEpAmlCQAlMmoDUZsJD4M4775S8Gjju8sbKGy3OkPi+ULWat+VPPvlEsvFGTaJqQrKmJDQx6fKLSQeBiUItpKjtIR2PIlCoCCiBKdSVz6N5o3kZN26c+fTTTyUElbo0OFl+8cUXEtqKcyXhrajboyZhOfGiJdhml53Nup6VjXXitdhQodr92VmvHW4+OOnTlOBLB4FJaQB6syKgCOQVAkpg8mo5dTK5hkCYGphKs8aaNcdd6+nE63bsfWH\/nc2tp\/c3K79clTTESmCShk5vVAQUAQ8ElMDotsgbBPALmDZtmuR+2Xbbbc0+++wjOWBsWGsUJxqmD0yNw080yzesF1jcTrxen6Xq2KsEJoo7UMekCOQuAkpgcnftdOQOBKhNQ3VeQkidQnjqrbfeKkQmiqIEJoqromNSBBSBXEBACUwurJKOMSYCOOiedNJJZvvtt5fkYfXq1ZOIG2KsUAAAIABJREFUDsJbyZNBQjt8ZNDIRE1yxYSUjqrVqoGJ2u7T8SgCuY2AEpjcXj8dvTGSVIzqu+R58aqH9PXXX0vBvDvuuCNyeIXpxCuEokeZQE68OPbWfy41+JTApIaf3q0IKAKlEVACozsi5xE488wzDRlOYwnkxZlkLCqTDlMDU+Pjx83yLlcHcuJFA5Nqdl4lMFHZdToORSA\/EFACkx\/rWNCzwMeF0gGxhGRgaGGiJmH6wOzZ7hSzaM1qgUSdeKO2M3Q8ioAiEA8BJTDxENLvI48AVXyvueYaz9ovOPWS\/ZS060pgipdy3bz3JYMrBGbFhg0SiRSEwHBvKtl580kDU1RUJAkSKaRInqGddtrJ\/PHHH6ZBgwaCLbW5mK+KIpCrCFCnC79CSmT4yXfffScJQtnv1atXz\/pUlcBkHXLtMN0I4MR7+eWXSzVeDhCEQ2ThwoXmxRdfNGvXrjUTJkyQ4o5Rk7BNSKZbf7NiwsCtijl6FXhMNTtvvhCY5cuXS5kKii1SLNTuOfYZmkAi4l577TXxy1JRBJJFgCziZBMPS\/AtJCCCUhdWNm3aJHW3bMHSGTNmyG+B8hdhEHYlMGHtDu03rQiMHz\/e3HbbbVLMzik1a9aU8Gqq+0ZRcsmJ152dd+\/tqiQEaaYIjH3AUo\/oiCOOSGhMyVxM9fMpU6ZIdBsZnt1y4YUXSoHFZs2aJdO83qMIGGpuPfHEE5Gr30ZNOepzkZ4iCqIEJgqroGNICwJoXThYvvnmGymqx5vx0UcfbSAxUZWwNTCVT7zRLBl\/Y0IaGLBEG3PaA9PMzKnBiz2mm8BQFwnS4hT+D2HNFJGhGjXFHE899VQzaNAgz21FXS7EVquO6t7TcUUTAUySBCY0bNjQ3H777ZEZ5CuvvCJFctG6KIEJb1nKFGHAVslbBMj\/UqNGjRI1Z5QnGrYTL9jgyBvUB8Z5XaKZedNNYKwa22t9M\/UT5w30vvvuk0rTtpp0rP2FWWn06NGmWrVqYtaEXGPubN68uSRdfPrpp83rr79uzj33XDNz5kwzdepUs91225lhw4ZJIdJ7773XfPzxx0LU6LNcuXJionrjjTfM3nvvLX5fmEcxa1Hrq02bNmbw4MEGcod2CBMX1yCxxvLnn3+ap556SkxfROsxpsmTJ4sJAR8H8iq55fvvv5f8SoxlzJgxMtY5c+bIwfvAAw\/I\/bRJGoOLL75YqnVboZ\/33ntPzHCYJHDEt3mayN0ELszvgw8+MB07dpS3fmTlypVmyJAhYp77\/fffzS+\/\/CKH\/GeffWauvPJKKeg6ffp089NPP5lHHnnEjB07VtaLFxn8lV544QXzr3\/9yyxdulTMy\/jHHXTQQWbUqFGyPnb8aPVYs2Sx5hl09913m3Xr1pnPP\/\/cHHPMMbLukJN4OEMU6B\/MDzvsMHP88cdvZY5krzB+9sihhx4q+IMN+bDAAUwRzOtgzUscteH22GMP0erYdBPMjz3J52Qxhzi1bNlSqqK\/\/PLLBh8XIjiZD\/uCF0TIO9cTGAHWjIPPuC9d+8xvXO49qBqYKJ9uOrbACPBDHT58uDzseWjYUgI8NDt37hxZMqMEJvASl7rQS\/vivAAtzE033ZRc4zHuwjzEQxybPwdfPDn77LNN27ZtJVcRwqFAviIe+vvvv78cbhxQ7du3l\/HyZksyRkyhHPr8m7med955UmGdA41DkMzSO+ywg7nllltkHPz93HPPCRFijL\/99ps4X\/bo0UMc3JF4Y4FQ0A\/m1htuuEH6oQ0ONQ49t2zcuFEIAgSCsdIvv73WrVuL+axfv37idwbxgkzMmjVLSMGkSZPMO++8I0QE4cCFuNE\/v1vmA0Y9e\/aUQ5Vx8\/cuu+xiIJCQOPwuEDCzJkT+Zm04VBHIDYc7BAa86J91oH0c\/7kW7J999lnTuHFjmStzYr34XYJ3MlhTXJZ2R44caXbffXdD4dR27doJHmAUBGcID9j5aWBok3WBqF977bWmRYsWQiaZK5gwT+sbyDwpp\/L333\/LuHbbbTchthBasKAIbsWKFUWzAgnieYkWkT3AekH6kDfffFNMo5BK9ikk8aWXXhKixj6wWs9U91mscSmBMUY1MPGeujn2PQ8sKlDjYFalShVTv359+eFRG4m3Px5IvEXat5IoTS9sE1LNHjeZn58ZIEUdEeuoG8uJ114XpgmJwwqS4ieZIjA8nDlsyTuEtiOWcGCecsopchhYLQgHDtoUtFFEMaGFOfDAA+Uw7969uzSHaQqCNHfu3JLmMYdCRHjrRSAZfMbhgbz11lvmggsuKOU8zCHGwUQ0SZCx2IPWeRhhLuPtnd+Pl9h+IRjWH+jYY48VssEhiaBROfHEE4W8HXDAAaKNYN\/XrVtXvv\/yyy+FdBAlyG8VksJByzV2TJYwoqmBAIA\/\/S1evFiuQ9A8oTmyBIbff6tWreRQ51BG0A7xrODQR9BcHXLIIUIGMDvbg5p20VigDUsUaw56nLmd0TusNX1DBoLgHI\/AME4IHsSE9bXSqVMnec6BA1oR9hm4WEGzAXnjb75DG8X\/ISYIGhfrjIvGCC2bH4Hhevdc0rHPIFGxxuXch6qBifkI0i9zAQEedvPmzZM3Dx6Uzmy8qIU5EBo1amT69OkTuelEwYm35rj1JUUdAYisu26HXa\/PwnTijUdgOOQy4QfDYfDQQw\/JoQA5iSVoHXAsh4hArK1ccsklQij4s3r1ann7dxIY3tw5lJwEBpKDdsVqHtDMYHKxBMb65jijnyA0mIZwcA8yFg4vNBVOAnPGGWfI74k5ewmaFLQ2TgLDPRyslsB89dVXpkuXLqIhQFPAb5EDFDLjJxyMmKCsGY178SnCtIGZBOKBPwYaVjQyQQkMpA8Me\/fuLfewBpBCNGF+kijWaIIhP05i4Ww7CM5BCIwbZ\/qAuIMtewcNDlihSbHCCx0mOYggUZuMlT+YNHlO7rvvviXXohljHWIRGPdc0rXPYo1LCUymDOSROx4LY0Duh7t71mhmIC\/Y5aMmqoFJfkX8fGDQcODvkAnB\/MFegijfddddMbtAS8CBggofraAVPkNzgaYgWwQmyFiCHKzuCSdKYDAtoYnA98QvMpBDEK0MJBFTGGYzS2DoH\/LCdxMnTpTvIAqQrCAaGDeBgXihqcKnxMvPh\/4SJTCQSvyCiFLzkiA4J0tg0F6hfaFvSAkmNAixFXxZ+PzOO+8U8yLCnoZo41PEtZgzkWQITDr3md+4lMAogcnEsz20NlH\/oq6N5VTJQwqbthXe7H788cdQche4f4DzXh1rKjXaEv5Lojkk3mcrJtxsMAFZCXqfM5GdvXfxp5Pkn0UNDytpL55jb9hOvH5+MJnSvgg+RUWiecFvAA2BNV841xQVOE6i+LGwL+3brr0GdT1aA0wb2SIwOLnGG0uQgzVVAoMWBdKB+QfNkBUwg+hBcDp06CAEALKBCQNCagkMhMlWlrf+MRAYPoPA4JMBGUHw0aAvpwnJTWDsNWhzrHaLezF74cvCYZ8ogbFmGifpok3mi3kNUhZP0wWBQVNlNWxeD1cvDQxmH7RfON5i7iQXFiZPax6z2jDwrVWrluxTNGH4nbAvWQf8s2IRGNrDjwZx75l07DNMf7HGpQRGCUxoZCMTHRM5APPn7cNLUDvz1mtV2lzDg5FDKEgkSSbGbNuMghMvY0mGwHBfIiQm3VFIFkOIDKQFyYTjrnv9UatjdiDihQPGhktz4GGS4OBCHV+1alVxbp0\/f768FaMlQHvAAU3kEGuPkyiqfucbMepzDmXMEFYwM3EgWWdaDjgiceyet74HHOBci2BewYSE\/wgSbyz2cMNcZEkCPjCM26\/WmHXsZA2sTwu+F\/iOWLMT8+jatat58sknRSPA3HBOhRjQPi8SkEHMaIwB3w4cXok8IhcKTsBcjyMs\/2Ye1t8Gh2EitnBG5QUF0yK\/cyJiiPZBS8b1HOz4hkBgOKidTsngBDHCcRifGQ5h1sximyjWkIHDDz9cCBAZajns0YjgB0VfQXBm7uwNzDf44zhNO3ZPQGBYX65hbpjLeZ5BoMAH3CGtkBnrj8M+gyCwbyFvRMjZdADsk1dffVUwR8h3BEEhKgqBGLK23MMeI\/MuRIP9DOZWc5PqPos3LiUwSmAyeSZnvW1+oB9++KF4xtepU6dU\/yw1b2r4C9jcBWhfIC9EBESBwPCAstoUp1YlyGdBrgEQ93UcwhRzRIPDd2TklZICLw6WnDBBnHjtdcPnGTNzWvycMJkiMFnfcMZIRAcPbfYWqncODA4s\/G6s8yzjguQQXgxh5qFPgjIObQ5SDh\/a4EDAIRjHT6JgBgwYIBEkHOIcUrwR8xn3c9hwuHItad7Z+5inCAMm9Bh\/MKv652+igjj0OVzijQVNEWYZGxHFiwEOnmSDpX20GU5hD0EsMIdBrvDtwZfEkgPIJAQDIsDYOEQZE79R5syBCQaQDULA2R9gCNmAEBGlxbxph5cQNCkQHQ5qxgIORGLh64PgEM3hyW8bJ1DugzBAQCBLmCTAgn5o1\/pIQRS4lj4hAlwP9vgtEdmVDNb4mtAHJAJNB\/jgPMx8g+CMLxOYoJFCs4RTtFvYG+w9NDX0QSkL\/JGcZAcyxpzBEsLB\/sL\/Bw0gmIIXPjFEf+E3gyaKyCnriIxGhj2GuY+1gVhBMCCBaMZwvIbw4DzMurMXU91ntOE3LjcG6sQbxtNP+0wrAjw4ecNNVILm8gjSLgQJXxsv4Y3az94fBSfevSYUldREYvxBnXjd1\/H\/+s\/5o5VPBCbIntBrFIFMIuBlQspkf1FsWwlMFFdFx5QQArwB8pZqfQvi3QzRwMkTlXc6NDDWjwG1MW8ibkFtzluNl0TBiTdVDYxTY9OzXhXz0TLvFVACE29n6veKQHAElMAYMcOiwS4k0TwwebbamIRQ1SZSewY7NfcESUYWDy6bMAsbOnbjRCQqPjA49i5ZtzYhJ147T6e5KZZPjBKYRHaGXqsIxEYAnyDMQvi8FKoogSnUldd5pw0Bm1vBZsBMpGElMImgpdcqAooACKBxXrBggYBBuQf8dnAaLzRRAlNoK67zTTsCNtOozSCKoxzOwzg\/xpOomJDEmXfDBrNiwsCEnXidJqTTO3f2LfCoGph4u0G\/VwQUgUQQUAKTCFp6rSLggQDRC0QYkNyMBFxoZPCzIWEX3vRkIfWTqDjxMj4IjEQiBczEq068+nNQBBSBMBFQAhMm+tp3XiBAngRCPwl1JISWMEBCPQkxJI8CORecdVGck46SBoZxLR53Q0oamFhOvOSQoC6QiiKgCCgC6UCAHEzkXSokUSfeQlrtLMyVKCRSdJOXgyyYVsi4SZInknrZAnPu4UBgztzlD\/P4khpCfsgqbKYML76sw+VbLvf4rOiavUyZIQ4P\/ID30T75KKqdeONW7X9\/ZHGNGKT6O6Pk75VHXRLzs7o3NDQ\/DJov1zw6apR5dGTxfSrxEajbork56YEx8S8sgCvemLLWHFF3W3Nyi\/T6clQqW9bULldcqNEpJNCzGWUzDS+JAG0m8EKLmsk0toXWvhKYPFtxEm6RzTSKQt0cEmiRUIoK2W6JmgbGVqZOJJGd0weG+0hsN2LQlgq4UVyXqIxpz4NbmXOeejwqwwltHD\/+tN4s\/uxvc3+vhmkfw66VKhtIjFtIxGcrL6e9U58GC9HkkS1sC6UfJTB5ttJk8sTXhHTlNWvWzPrs8HchIyapwt2CVoXaIWQNdVbJttdFyQeGMYkj72kVA1Wj9vKBsVWsCacOkp0364sVwQ5vWVQcTVLI8twLv5l\/d9rL\/GvP9GpfwFQJTCHvrPybuxKYPFtTUonvvffeUgyOlO6kDcd0QyrwbAj1V1APky6dWipWeMMjsR3F0kjR7iVR08CQ1G55l6sTKiXg1sDwf8RqcWL5xWRjfaLch2pgilenfa0dTe2KFdO+VJXLlTO7bFPJs13VwKQdbm0wCwgogckCyNnsgtoctrgbhdKoI0JxNvIjkMKfmh1O35R0j406NzjpklCPWiE48f7www9SAI+KxFSh9UuyF6U8MOBCBeySaKT50wSquBWq\/6mfZHF1V7FOpOBjutcm6u21vfxS0\/ayLT5GUR9vJsa3U8VtTLtaO2SiaVOrQkVT0yedgRKYjECujWYYASUwGQY4Ks1TMv66666TImRoZCAzFHCjYFu6hYchxeYoKkm1V\/xd0MZQp4mqwX5SCASGuY88s7MZPrGYEKlsQaDQCUwmyQso+5mP+E4JjP4ScxEBJTC5uGoxxkxlYOtfAll57733RPtBfpYyZcqIRgQCg6YEDQ3VgMnbEiTRXKahiqIJqfKJN5ol42+UcGokSIXqINeoX8zWu6nQTUiZMh1ZpJXAZPoJpu1nGwElMNlGPMP9kW8Fvxdqgjz55JOG8Egy4WI66tWrl2nTpk3JCAhtHjhwoBQAw7TjFRmU4eGWaj5qTrwMbnGPMik78dpJ+iXFi1W1Opv4R6GvQnXixXH3+d6Zix6M5f+iGpgo7HwdQzIIKIFJBrUI37PPPvuImQjSgiaGjLjnnnuuVCr1Ekw8mHfQwtx4oyMXSghzjKIGhrICNpw6nRoYZ1vNW+5sVn65KgTEo9VloWpgMhk2HUT7ogQmWr8DHU1wBJTABMcqJ66EBNSpU8dQXh4nWiq0xpI333zTXHTRRaZ9+\/bmP\/\/5T6hzjKIPDICs27zZLPnsTcEmVSdeIS4uh2B17C3edoXqA5PJsGklMKE+0rTzDCOgBCbDAGe7+UsvvdQMHTo0sE8LfjD\/\/ve\/zWWXXSZOvWFKVAkMmCz+dJISmAxvjkIkMB\/NWmXWLzMZSVpnlyue+Ug1MBne2Np8xhBQApMxaMNpmDT+YSSwS8dso2pCIh+M6da\/uLjjP2HSsbLzBrlGNDH\/tBWranU6cM2VNgrRhHTvqJ\/MA732y0jSuqDaFyUwufIL0XG6EVACo3siMghE1Yl3rwlFJflghHgEqFAd5BrbljrxbtmCheTEi\/alcZWq5vyjds3obzBW9JHtWMOoM7oE2niGEFACkyFgtdnEEYiyBgZn3kVrVmdEA6PZeYv3SiFpYHDcnfDCMjP7tpaJ\/1ASuCOI+Ug1MAkAqpdGCgElMJFajsIeTJR9YNbNe9+s2LD+\/7V3LtA6Vvkf\/7mdk7\/j6JBLMU1n5FITRoVGyBIhTCakToVq6IqSNGVyGUTGjIomGi2XZsRySybVULNQIYRRKo1LDU5lhlzK9ZzzX9+t5+05b+97znt5nr2fy3evZeE9z7Mvn9\/e+\/2e3\/7tveW7ei0jRooOxlUelVJO4lXPMIg3ZkcPUwwMAnc7NzjPde9LSafv2o1AD0y4516\/tp4Cxq+WC2C9vS5gjhcUyMG6P9zv5JSAgSl5Om94diHp2DZtTQ+JLB\/RAxPAyTQkTaKACYmh\/dBMry8hgeGuucPVqbxOBvFaeYX9dN6wLCHhyoCcw5muBu6irya6fEQB44fZkXWMRYAChv3CMwS8HMRrQcLljofyMko9nTeZIN6i2ceK2QDvhjWwNwxBvHm13A3aTdb7QgHjmSmQFUmSAAVMksD4uHsE\/OCBsU7ldcMDo+Jjvo+hCWNgbxg8MI2ystXOIx0p0eUjChgd1mAZbhCggHGDKvNMiYDXY2DQqHMuvUb2nzwhiIdxMgaGgb3Bj4Fx+7Zp+6BLZvmIAial6YoveYAABYwHjMAqnCXgFwGDZSR1qF3UbiK7B8WyacxnYr3HnUmBvkoAgbv9GtWWGhkZWoZ7Mt4XChgtJmEhLhCggHEBKrNMjYAflpBwKi\/OhLGCee2iJfoE3pKWmWK9Z\/8sjKfzBnkJScd9R9aoS9b7QgGT2nzFt8wToIAxbwPW4HsCfgjiRVV39SojOXNPKS9MxNMSdTovg3hT69ZBDOLVuW0a1BM9+8VuIZ4Dk1p\/5VtmCVDAmOXP0m0E\/OSBgSfmYJehqvbxPC\/peGAYxBucoaHT+wJqyS4f0QMTnL4WtpZQwITN4h5ur19iYCyEMW+oTuEkXiWCGAMTyBgYHbdN24d0KstHFDAenhRZtRIJUMCwg3iGgN8ETP7WFWo3UlHD1j8sJTkkYJBh2E7nDdpVAtZ9R27fNm0fwKksH1HAeGYKZEWSJEABkyQwPu4eAb8tIcW74DF6SSnWMlOin4XpdN6gBfHquu\/IGpGpel8oYNyb05izuwQoYNzly9yTIOCnIN6fLShSLYtsqXY4iDesp\/MGJYhXd+Au+iIFTBKTDR8NBAEKmECYMRiN8KMH5kRhoeyf90Sx+5Gc9MDYPTVBD+wNkgdGd+Au+kkqwbvWzMFdSMGYQ8PWCgqYsFncw+2FgNn0VF91zoqVsNsHqbTPsLXZ8oooz0iC7+G5bw4dktwBTyddplWGdH88sqU6evs0xAwSLoC0UqqfYTnp2XFn8wtiCoqA8Zv3BX2JAiaIIyr4baKACb6NfdNCP3pgLLGELdVOHmQXbwv2M9slsCImKAJG123T9oGdjveFAsY3UyQrGkWAAoZdwjME\/BgDY8GDB8get5LuQXaxYmCsz4Ic2BuEGJj2VatruzIA\/S+d2Ber\/9ID45lpkBVJggAFTBKw+Ki7BPzsgcEy0qEFo2LGwoCaG3ExedPXyPrVa9w1isbcg+CB0S1eYJ5Ut07bTUsBo7GjsyjHCFDAOIaSGaVLwG\/nwJzYvko1GTdUI+387tuIULFYpHqZoxI9pVz6GLSYmCCcA5NXq3a6wyCp953wvqBACpiksPNhjxCggPGIIVgN\/9xGbdkqWsBgS7XlhaGASb5H+13A+NX7QgGTfF\/lG94gQAHjDTuwFnJWwOzatUvtIMKuI+tvwEnks0SeiZUXfvs8d+OclMqMzi9WMK\/ypnx\/Qm869yNFv8slJO8MGxPbpp3yvlDAeKcfsSbJEaCASY4Xn3aRgN+DeLGNO97BdkrERB12l+5nF1fKctEaZrL2YxCv7vuOLMs4Efti5cUlJDP9naWmR4ACJj1+fNtBAkHwwMBztGvu8GLBvPTAJNZJ\/BrEO3nqPtF53xFoOul9oQcmsf7Jp7xHgALGezYJbY38HsRrLScVO9iulEBcy9ilBezGeo5BvOaHiu77jtzwvlDAmO9HrEFqBChgUuPGt1wgEBQBY13yqDwvLgoY5B8kEeO3IF7rtukNY5u7MBriZ+m094UCRqv5WJiDBChgHITJrNIjEJQlJHXFgO1cGLeWkHA9AQJ7cTrv+jX+PxPGb0tIQfG+UMCkN2\/xbXMEKGDMsWfJUQSCEMRrNSn6ZF4lYhwO4v3RvUt9s6TuQn93K78E8Zq47wiWdcP7QgHj7zET5tpTwITZ+i61vbCwUGbOnCkvv\/yy7N27V7KysqRNmzYybNgwqVWrVtxSg+SBwTJS\/vwR8l23R1V73dhGbXlg7H\/7+cZqP3lgTGybRj9ycueRfSByF5JLkyGzdZUABYyreMOZ+ZgxY5SA6d69u7Rq1Uq++uormTFjhhIyy5Ytk+zs7JhgghQDE\/HEfPCG+mdRw9aRNjseF\/P9GTMowM8xMX6JgTG1bdot7ws9MOGcp4PQagqYIFjRQ234z3\/+I23btpVevXrJhAkTIjXbvHmz9OjRQ4YMGSIPPPBAaARM\/tYVcryggAImgT7qFwHTKCtbTh8QuTy3cgKtcu4Rt7wvFDDO2Yg56SVAAaOXd+BLe\/HFF2XcuHGydOlSadSoUbH2XnfddVK+fHlZvnx5XAHj95N47ScIWycD2+9IcvIk3lhLSLd27uzbCx79soSk+74jDBY3vS8UMIGflgPbQAqYwJrWTMOGDh0qixcvlk8\/\/VQqVKhQrBLwvvz973+X7du3KyETnYIWxIuTeZF0nc6LoF4G8brb703cd4QWuel9oYBxt88wd\/cIUMC4xzaUOffp00c++ugj2bRp04\/aD88MPDTr16+X6tWrxxUwqd5plOp7Tt6FFMsDg4ZaAb1ue2AYxOvusAui94UCxt0+w9zdI0AB4x7bUObcu3dv2bNnjxIp0WnixIkybdo0Wb16tdSpUyemgGlS+YRsPXqO9O3bVyCGZNf3+fysxQ\/Px\/psxTMiHQaX\/EycvA4cOCDVW3RN6V31kr1ucepxMre5fHXmtHr8nN3vq79P5P5wAFo6n1V5e6ocbnc2rmjW1Kkya8pUX\/a9lnf3l18O6O\/Jun\/5VYHceFG2VCtXTnv9apavIJllyrhaLnYLxhqTbhQ6Z84cmT17tsoaS8ZMJJAqAQqYVMnxvZgEIDq2bdsmCNqNTtbupNI8MLrR6tpCai0lud0+P+5E8nr8i6lt027Hvlh9UdcYsPd9+5Kx22OC+QeTAAVMMO1qrFWIgVm0aJGKgcnIyChWj4EDB8qKFSvUElO5GL\/JmprQdE7eCOjVkSBirNN5W7RpLScvzpCcypfKnvXvy+516wWC4aIWZz1ATn0W+TJMMv\/zmzSWTsOG6sCSUhlB3DYdDULnGLDKNjXeU+oEfMmTBChgPGkW\/1YKruHRo0crEdO0adNiDWndurWcd955smTJkpgNNDWh6Zy8dXlhogH3fa2NdLziZU92rMPfHJYq51bxZN2s+4503zYNGG4H7tqB6xwDFDCe7Oq+rBQFjC\/N5t1K5+fnq3NgunXrJpMmTYpUdOXKlTJgwAAZOXKkim+JlcIgYNDu\/SdPqLNhdKZXPpsp244dlbrn99BZbEJleVnAmLrvSNfSUcRrtnu35ObmJmQvpx4yNd6dqj\/zMU+AAsa8DQJXg8mTJ8uUKVOka9eu6goBBAjiJN6f\/vSnaot1ZmZmqAXMicJC2XfiuHa7wwtz8QU9PSdivCpgTN13hI5R+5yKck7Zstr6CD0w2lCzIAcJUMA4CJNZnSVQVFQk2Gnw0ksvCU7mrVKlinTo0EEQH5OTkxMXk6nfyExM3iaWkuCFWZW\/XprVf8JTXdWrAibogbv2TmBiDJga757q\/KxMWgQoYNLCx5e6aL4PAAAUsUlEQVSdJGBqQjMxeYObroBeu428uJTkRQEThsBdChgnZy\/mZYIABYwJ6iwz1EtIVuNNeGE+ObhFxq8d5KmAXi8KmMlT90nQA3cpYDgR+50ABYzfLRig+ofNAwPTmRAxXvPCeE3AhM37gn5owgtparwHaMoMfVMoYELfBbwDwNSEZmLytlM3sZSEHUnbjh3xhPG9JmBM3Heke9dRtOFNjAFT490TnZ6VcIQABYwjGJmJEwRMTWgmJm87LxNeGJQ\/98t9Tpgt7Ty8JGBqZmTKtVXPS7tNyWage9cRBUyyFuLzXiRAAeNFq4S0TmEVMKaWkrzihfGSgAmj94VLSCGdcAPQbAqYABgxKE0Is4CBDcO6lOQVAdMoK1saZVXWOpxMLx1ZjTXhhTQ13rUamIW5SoACxlW8zDwZAqYmNBOTdywuYV1K8oqAMeF9Mb10RAGTzAzFZ71GgALGaxYJcX3CLmBgehPXDJi+J8kLAsbEtmmveF+4hBTiSdfnTaeA8bkBg1R9ChgRE9cMmN5WbVrAmLjvyEvihQImSLNouNpCARMue3u6tRQwZ81jYikJXpjmDUZITtYl2vuISQFj3Ta9YWxzre2u+3+VtJZXWmEmllFNjffSWPDn\/iFAAeMfWwW+pqYmNBOTd2nG1C1iTHphTAoYE\/cdVa2QITkVKpTWBbT+3MQYMDXetYJlYa4SoIBxFS8zT4aAqQnNxOSdCBfd8TAT1g2W05l1td9WbUrAmLht2mtLR1Y\/NDEGTI33RMYen\/EHAQoYf9gpFLU0NaGZmLwTNajOrdXwwizZMVP7PUmmBIxu74tXxQv6ookxYGq8Jzr2+Jz3CVDAeN9GoamhqQnNxOSdqFF1B\/WaWEoyIWBM3HfktbgXex80MQZMjfdExx6f8z4BChjv2yg0NTQ1oZmYvJMxqu54GN3bqnULGCtwV+dt016Me6GASWYU8lkvEqCA8aJVQlonCpj4htcpYnR7YXQLGN3bpr28dMQYmJBOtgFpNgVMQAwZhGZQwMS3ou6lJJ33JOkWMDpP3PWDeGEMTBBmz3C2gQImnHb3ZKspYEo2C0TMwdOn5HhBgRb76RIxOgWMzvuO\/CJeKGC0DCcW4gIBChgXoDLL1AhQwJTOTedSEmoz98t9pVcqzSd0Chid3hev3HOUiHlMxIGZGu+J8OAz\/iBAAeMPO4WilqYmtNGjR8vIkSN9w1iniNHhhXn3nXfl6lZXu85fp\/fF60G70bBNjAFT4931jsYCtBGggNGGmgWVRsDUhGaq3NJ4lPRzXSJGR0DvtOenyT333pMOjlLf1blt2m\/iBfBMjAETZZbaUfiArwhQwPjKXMGubF5enqxbty7YjXSwdYU1azqYW\/ys6gw5Jbvfv0ZLWW4UUijnycmCX0r22sfdyL5YnmVOn5YyBw+6Xk4QCrjqqqtk7ty5QWgK22CIAAWMIfAslgRIgARIgARIIHUCFDCps+ObJEACJEACJEAChghQwBgCz2JJgARIgARIgARSJ0ABkzo7vkkCJEACJEACJGCIAAWMIfAslgRIgARIgARIIHUCFDCps+ObJEACJEACJEAChghQwBgCz2JJgARIgARIgARSJ0ABkzo7vkkCJEACJEACJGCIAAWMIfAslgRIgARIgARIIHUCFDCps+ObJEACJEACJEAChghQwBgCz2L1ESgqKpI5c+bI+PHjpXbt2vLWW2\/FLHznzp0yceJEWb9+vRw\/flzq1q0rd955p\/Ts2VNfZQNY0qlTp2T27NmyaNEi+fzzz6VKlSrSuHFjeeihh+SSSy4p1mLawJ0OcOzYMZkxY4YsW7ZM9u3bJ9WqVZOmTZvKoEGDpH79+rSBO9iZq8sEKGBcBszszRL48ssvZdiwYfLBBx9I2bJlpXr16jEFzN69e6Vr166SlZUld9xxh1StWlVef\/11WblypYwaNUr69OljtiE+LR3iceDAgbJ8+XK54YYbBPfffP3110pQHjlyRF555ZWIiKEN3DHymTNn5Pbbb5eNGzfKTTfdpMRjfn6+zJw5U06fPi1Lly6VevXqqcJpA3dswFzdIUAB4w5X5uoRAv3791dfmE8\/\/bTcf\/\/9cvLkyZgC5pFHHpFXX31VCZaf\/OQnkdr37dtXNm\/eLGvXrpVKlSp5pFX+qcaGDRukd+\/e6gt09OjRkYpv2rRJevXqJTfffLM8+eST6nPawB27vvHGG3LffffJY489JhgPVlqzZo2gf+PPyJEjaQN38DNXFwlQwLgIl1mbJwBBcs0110iFChXk+uuvjylgCgsL5Re\/+IU0b95cudnt6bXXXlMehGnTpsl1111nvkE+q8GOHTtk1apV0rFjR7nwwgsjtYdnBksXrVq1Up4A2sA9w1oC\/JZbbpGcnJxIQRDzWMLr3LmzPPfcc7SBeyZgzi4RoIBxCSyz9R6BeAIGbvM2bdooD83DDz9crOJ79uyRdu3ayeDBg9UfJmcIfPHFF9K2bVu56667ZPjw4WrpgjZwhm2iuWzdulV+\/etfK4GOeCTaIFFyfM4rBChgvGIJ1sN1AvEEjLWcESvWBcGPiBnAb6\/jxo1zvY5hKADeln79+gmWl1asWCF16tQR2kCP5Q8dOiTffPONfPjhhypgPTMzUxYsWKA8M7SBHhuwFOcIUMA4x5I5eZxAPAHz3nvvyW233aZiMRCTYU\/YQdOwYUO58cYbZdKkSR5voferB56IxUDw7pQpU9SyHhJtoMd28Hj985\/\/VIW1b99exo4dKzVq1KAN9OBnKQ4ToIBxGCiz8y6B0jwwCGREQKM9YacM4mPy8vLUZM+UOgH89n\/vvffKli1bVFB1p06dIplZv\/3TBqnzTeRNLBthqQhbqefPn692gv35z3+WZs2aRTwwtEEiJPmMFwhQwHjBCqyDFgLxBAwm89atW8eMgcG5JB06dFAxAogVYEqNAGJe4OVC4OgLL7wgTZo0KZYRbZAa13TewvIoRCSWkXA2Em2QDk2+a4IABYwJ6izTCIF4AgY7Yq688kq57LLL1IFr9rR48WIZOnSo2imD3UxMyRPYv3+\/On+kYsWKiu8FF1zwo0xog+S5JvoGdoFBiON8ozJlyhR7DduqIV4+\/vhjycjI4DhIFCqf8wQBChhPmIGV0EEgnoBB2SNGjJB58+bJm2++Kbm5uao6BQUF6hRe\/Gb6zjvvqAmeKXkCiCvCssWSJUvUQYLxEm2QPNtE3pgwYYLyeuEP4l6sdPToUbXDDqLm\/fffVx\/TBokQ5TNeIUAB4xVLsB6OE8ByBb40rYSgUQSRWluly5cvH7kmAIfddenSRcqVK6e29mZnZ6tAU1wrgPfwM6bkCeBwwAcffFC6d++uztmJTli+wFZeJNogeb6JvIHTqBGEjt1HOFSwQYMGgnikhQsXyu7du2XMmDFy66230gaJwOQzniJAAeMpc7AyThI4ePCgconHS\/CofPLJJ5Ef48wXbC2FtwVCB4d8IeiUB9ilbpXf\/\/73MmvWrLgZVK5cWRBYaiXaIHXWJb0JL+L06dMFp+9iSQ\/3UeEgQSwhRS+N0gbu2IC5Ok+AAsZ5psyRBEiABEiABEjAZQIUMC4DZvYkQAIkQAIkQALOE6CAcZ4pcyQBEiABEiABEnCZAAWMy4CZPQmQAAmQAAmQgPMEKGCcZ8ocSYAESIAESIAEXCZAAeMyYGZPAiRAAiRAAiTgPAEKGOeZMkcSIAESIAESIAGXCVDAuAyY2ZMACZAACZAACThPgALGeabMkQRIgARIgARIwGUCFDAuA2b2JEACJEACJEACzhOggHGeKXMkgdAT+Pbbb6VSpUoJcTh27JhkZWUl9CwfIgESIAGLAAUM+wIJkICjBB544AF5++231Z9atWqVmPf27dvVRYO4TPCJJ55wtB6JZtasWTN10SEulaxWrZq0bNlSWrdunejrcZ+bPXu24CLFLVu2qEtBcaHln\/70p7TzZQYkQAJnCVDAsCeQAAk4SgAXOL711lvqNu+cnJxI3qdPn1Zf5hAMVsJtyBAv\/fr1kwEDBjhaj0QzQ33q1asnc+fOTfSVpJ47fvy4NG7cWLp160YBkxQ5PkwCJROggGEPIQES0EIAwqZ69erqhm8vJQiYpk2bygsvvOBatS677DJ1qzk9MK4hZsYhJEABE0Kjs8kkoJvAnDlzZNSoUfLII49QwOiGz\/JIIKAEKGACalg2y\/8EXnzxRXn22Wfl6NGj0qpVKxk\/frx8\/PHHcv\/99yuPwcMPP1xsOcbeYizXrFq1SpYuXSq1a9eWzp07q9\/+N2\/eLHXr1lXvIk972rRpk8ybN0\/OPfdc+fTTT6VMmTIyaNAgueKKKyKPFRYWyh\/+8Af57rvvpGbNmqo+559\/vjz++OORZwoKCmTt2rWSm5urykacC8TLxo0bpX79+qr8q6++WvLy8tQ7X3\/9tXz44YfSrl27YvXJz8+XqVOnqmDgPXv2yJEjR6RHjx7Sq1cv9RzK2bBhg7z++uvyr3\/9SxBz8sc\/\/lFWrlyp6j5s2DD51a9+VWpHiOWBwbLPu+++K2+++aZq6\/Dhw2XChAmqXeAzZMgQxfQf\/\/iH\/O1vf1NcL774YpkyZYpqc3SiB6ZUM\/ABEkiaAAVM0sj4AgnoI4AvyHvuuUeuvfZa+ctf\/qK+KPFF+te\/\/lUqVKgQtyJ4DgJo+fLl0qhRIyUaIFi++OILef755wVCBGLFEif4N8TCwoULI4G3KAdlQrAg0BYJ5S5atEiWLFmi\/r9v3z558MEHZcGCBXLy5ElV5syZM+V\/\/\/ufiim56qqr1HMQU3fccUcxDwzEB\/KH4GjevLmqj5UgaBAbM3369EgeCAq+66675Oabb1ZiDruXkO\/QoUOVyEFZXbt2lezsbCVe\/vvf\/8p7770nVatWLdFgsQQMgm8Rx4PA4osuukiaNGmiGNSoUUPuu+8+OXDggKpflSpVlG22bdumRGGXLl0URwoYfWOEJYWXAAVMeG3PlvuEwN133y0rVqxQ4gEehkmTJinvRmkJX6o33HCDdOjQQQkBK0FY\/O53v1OCBks78HQgPgNfzPb4FAgS7MY5ceKE+jJH\/Ao8D\/DOLFu2TMqWLauyhNdh4MCBkfzxDAJ4SxMweAGCqm3btsUEDMQVdgTB0wGvij0h0BeCB59bO4Xat28ve\/fuVd4RK2gYdZo8ebLMmDHjR56daG7xYmBQj4YNGypBB6EErw4SRBcEFOzy6KOPRrK7\/vrrVT3gDaKAKa138uckkD4BCpj0GTIHEnCVALwcECFFRUVKQPTv3z+h8rC8A49Anz591BKOlSBM8KV95swZtbxjxafE+rIfN26c8qpANMEDMW3aNJk4caISPGPGjFGiBvllZmZG8ofIeu655xISMKgDlpXsHpgdO3ZIp06d5De\/+U2xpSkUAPECEYOlJHiGkPDs\/v37iwkHeIQgLiA0evfuXSKvkoJ4sfRz4YUXKk+WlSDeBg8erETgnXfeGfm8b9++smbNGrWsZueBB7iElFCX5UMkkBQBCpikcPFhEjBDwBIOjz32WNoCBi2Ah2Pr1q2CuBd4KuDdiSVgEN+BZRRLTECsYMszzjWpXLmyEgm33HJLxDuBvJMRMIhjwRZmu4CBWMBZMrEEzM6dO5WYw7ZkeHniCRgshWEZyQ0Bg5gbxCFFCxjUF8tc4Ao29kQBY2bcsNRgE6CACbZ92bqAEMDuHSwjnTp1Sv0dK1A0uqnxPDB4Dt4CiBcsM8GTMmvWLPU34jrsyYrBQTAvYl2QECAMDwtEFeqDGBD8PyMjQ\/08XQFjlQnvEZaC7AkBzYhHsQueWB4YCpiAdHw2gwRKIEABw+5BAh4ngPgL\/NaPnTtYuoiOaYlX\/ZIEDOI1sMyBYNz58+cLPDu33367jB49ulh21s8gJCAoEFwLbwIS4j0Q+wKPA85QQSyKEwIGO46wIwm7eiBm7Mn6GXYwjR07Vv2IAsbjHZjVIwGXCFDAuASW2ZKAEwSw0wa7bl566SUVoAqRge29dsFQmoCJFibYYYMA2GeeeUYgZA4dOqTEB+JRkLf9XqLf\/va3snr1auX1wU4fCB14asqXL6+KxW6cFi1aqNgcLPvEEzDYDXTbbbeppRfs1rFSrBgY\/AzLUlimgoCyn9yLZSOU9eqrr0aEFOJx0CZ78Cw9ME70PuZBAt4mQAHjbfuwdiEngPNVfv7zn0eWdnD0fseOHdV2XgST2o\/qj0ZleWCwDRixLDivBTEnWApCLAt2Jlk7a5AXhAXiW6wzXT777DN1VstTTz0V2ckDAXL55Zer7cxIVn0WL14cERQ4cRdLUvaYGgQiQzRhVw+CgCE2IMxw6SOWhLDVG2fWWAmxLgi+xfZv5FWxYkW1bfqmm25SS1Z2EYTdVBBSH330UURYIfAYAcgIXkYQc0mppF1IDRo0UEG82IVlJUtEPfTQQ8V2X0GgQahBBIK1PTEGJuQDmc13hQAFjCtYmSkJpEcAX9Y4eA5f3k8++aT6skdat26d2r6LWBAEso4cOVIdahcrWQLG2sFk7RbCe9gybW2Dtt7FPUXwypQrV04t3xw+fFid3YJdQlZCXT7\/\/HMlRJCwiwlCAx4cHDQHIYQ8cAbLlVdeqc6wsQ6og7DBGTIQZBAXeAYeFogfCCmIkp49eypxhgRRAvEE7woCfXG4HPKCxwUJ\/4dIQhAyEsQXyvv3v\/+tAmyx3IR6jhgxInKWTCxOsQQMhBm8XuCPBM8T6gaRhMMFcShfnTp1lDcIS2t4FsIMMUHYIYWdYnZuFDDpjQe+TQKxCFDAsF+QQEAJlBQDE9Amp9Qs3oWUEja+RALGCVDAGDcBK0AC7hCggEmMKwVMYpz4FAl4jQAFjNcswvqQgEMEsGMIdwEhIBZLNkyxCUDA4FoGxODgYD6nE5bZsMyEs3d4G7XTdJlfmAlQwITZ+mx7YAkg3gU7lRAfgjgQBOwisNTaPRTYhqfQMGwRx4WNVmrZsmXkmoIUsou8gusOEL9jpUsvvVS6deuWTpZ8lwRIwEaAAobdgQQCSAAXJWLnjz1ht08idygFEAebRAIkEEACFDABNCqbRAIkQAIkQAJBJ\/D\/NAfkcISkzxoAAAAASUVORK5CYII=","height":337,"width":560}}
%---
%[output:442fc62e]
%   data: {"dataType":"text","outputData":{"text":"SIM at (10.0, 10.0): maximum horizontal range = 13.79 m, required angle = 79.72 degrees\n","truncated":false}}
%---
%[output:1de16e8d]
%   data: {"dataType":"text","outputData":{"text":"SIM at (30.0, 10.0): maximum horizontal range = 19.75 m, required angle = 82.79 degrees\n","truncated":false}}
%---
%[output:85ebda03]
%   data: {"dataType":"text","outputData":{"text":"\nPosition projection FoV:\n","truncated":false}}
%---
%[output:9948d3ea]
%   data: {"dataType":"text","outputData":{"text":"SIM 1: 80.22 degrees\n","truncated":false}}
%---
%[output:82bc02fc]
%   data: {"dataType":"text","outputData":{"text":"SIM 2: 83.29 degrees\n","truncated":false}}
%---
%[output:4e00c158]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:5c5e676e]
%   data: {"dataType":"text","outputData":{"text":"\n=== Common calibration region ===\n","truncated":false}}
%---
%[output:09e13d5a]
%   data: {"dataType":"text","outputData":{"text":"SIM 1 positions: 570\n","truncated":false}}
%---
%[output:872ada1a]
%   data: {"dataType":"text","outputData":{"text":"SIM 2 positions: 1600\n","truncated":false}}
%---
%[output:2ac47e4f]
%   data: {"dataType":"text","outputData":{"text":"Common positions: 570\n","truncated":false}}
%---
%[output:74d398a5]
%   data: {"dataType":"text","outputData":{"text":"Prepared 500 common evaluation episodes.\n","truncated":false}}
%---
%[output:9e4a146d]
%   data: {"dataType":"text","outputData":{"text":"\n=== Running SIM 1 evaluation ===\n","truncated":false}}
%---
%[output:98b9dedc]
%   data: {"dataType":"text","outputData":{"text":"SIM 1: episode 50 \/ 500\nSIM 1: episode 100 \/ 500\nSIM 1: episode 150 \/ 500\nSIM 1: episode 200 \/ 500\nSIM 1: episode 250 \/ 500\nSIM 1: episode 300 \/ 500\nSIM 1: episode 350 \/ 500\nSIM 1: episode 400 \/ 500\nSIM 1: episode 450 \/ 500\nSIM 1: episode 500 \/ 500\n","truncated":false}}
%---
%[output:517a6f59]
%   data: {"dataType":"text","outputData":{"text":"\n=== Running SIM 2 evaluation ===\n","truncated":false}}
%---
%[output:6d7e5519]
%   data: {"dataType":"text","outputData":{"text":"SIM 2: episode 50 \/ 500\nSIM 2: episode 100 \/ 500\nSIM 2: episode 150 \/ 500\nSIM 2: episode 200 \/ 500\nSIM 2: episode 250 \/ 500\nSIM 2: episode 300 \/ 500\nSIM 2: episode 350 \/ 500\nSIM 2: episode 400 \/ 500\nSIM 2: episode 450 \/ 500\nSIM 2: episode 500 \/ 500\n","truncated":false}}
%---
%[output:9ed4957a]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7tXQe4VsXRXkAUYslF7KKRGGPBxBBrjCWx11giaixoTMTEEo0tsbfYsWDvEWLvLZofxYoNUexYkUTs5aJGwQb\/8+7NXPbu3XPOnr5nvjnPwwN8357dnbK77zczO9Nj5syZM5U8wgHhgHBAOCAcEA4IBxrEgR4CYBokLZmqcEA4IBwQDggHhAOaAwJgRBGEA8IB4YBwQDggHGgcBwTANE5kMmHhgHBAOCAcEA4IBwTAiA4IB4QDwgHhgHBAONA4DgiAaZzIZMLCAeGAcEA4IBwQDgiAER0QDggHhAPCAeGAcKBxHBAA0ziRyYSFA8IB4YBwQDggHBAAIzqQiQNLLrmkQgqhiRMnqjnmmCNTH1lfKnrsnXbaST3yyCPq73\/\/u1p77bWzTquQ90Kai03QOeeco04\/\/XS1xx57qL\/85S+F0Mupk6L1Moo3IesIJ3kKLeFzQABM+DKqdIZDhgxRTz75ZOSYd999t8JGXfRm\/d\/\/\/lettNJK6ogjjlA77rhjLM0jRozQ4GnvvfdWs802W27+hHQghDQXm7Hjxo3TQA9yWmONNXLznVsHRa8J8Me1LsrQkTTrr0i53Xvvveryyy9XL730kvrkk0\/UPPPMowYPHqz+9Kc\/qeWXX14Pddppp6lzzz1X\/eY3v1HHH3985\/\/x3UEHHaT++Mc\/dk7p8ccf1+3wrL766uqKK64ocrrSV2AcEAATmEDqng4BmJ\/85Cdq0UUX7Tadww47TC200EKFA5h\/\/vOfap999lHHHXdcJID59ttvVa9evQpnURkHQtZJFj2XsniWlT7O75UBYFzromgdgUx81l\/RsrvvvvvU7373O9W7d2+12Wabqf79+6tnn31WASjPPffcasyYMWq++eaLBDD4EfPjH\/9Y3XrrrZ1Tw\/5x2WWXqR49egiAKVpgAfYnACZAodQ5JQIwsHJsvvnmkVOxN+sPPvhAnXDCCQqb0vTp09WgQYPUkUceqVZYYQXdBzabCy64QF111VUKbRdZZBG1yy676D9HHXWU+sc\/\/tE51mKLLaYeeOABte2226rx48ers88+W7su8A7a2WN\/+umn6sQTT1SwDn3++edqmWWWUfvvv79ac801dZ\/4dYnv77nnHv0r7\/vf\/77+fr311tPf+xwISX3Qr8TDDz9cvfXWW+r666\/XrrUddthB\/fnPf9bjYJ5\/\/etfFX51YoOGBQlzhlXj5ptv1ryy5\/LNN98oyOKmm27SfANv8N5WW23llA3NA+M89NBD6oknntC\/bpP6gXzw7pVXXqm+\/vprLXscHvjle8ghh6jdd99duVxIOGzwHg4ePPjVvO+++3ZaaHz4grlBvrfddpv68MMP1Xe\/+121\/vrr63HnnHPObnT6tH\/uuee0zCdMmKCtdHANQs\/mn39+3V+cvgL0LbXUUnoesA5gHpMmTVLLLrusGj58uNY\/10N6ec0116ijjz5av4N+8M7SSy+teYhDGetk++231128+OKL+vBeeOGFtbx69uzZ2XXUuiAdwboAzx588EH9owI\/Lkink9ajOX\/XOKB19OjRun\/IFHPDWsUD\/V1iiSX0eho2bJj6\/e9\/rw499FANPOJ0weYZeDRq1KhuLsljjjlG7yFDhw7VPI+ywPzwhz9Ur7zyip4b\/djCmu\/Tp496\/fXXuwAYrK+LL75YvfHGG3pdrrzyynrOAwcOrHO7lbFzckAATE4Gcns9K4DBhowNDKZfbG7YtAEmsLnMNddcevPAgQJwgcP3lltu0fEzcBnB2nPyySfr9\/EdNuGNN95YH\/6PPfaYBi4wK+PXFg4BG8DgVxyAE95Du0suuUSDluuuu06DggMOOEADhC233FKtuuqq6pRTTtFzA3gAIPABMEl94DA544wz9Fx\/9rOf6Y3xrLPOUl999ZU2Y8OcjQPm6quv1vMHn++66y690WKu+AWMzdqey5lnnqn72WSTTdSmm26qARx4AoC04oordlM\/cx7Y1HGA\/u1vf1NJ\/dx+++0aeEBWOJRefvllfTB+9tlnWka\/\/e1vuwEYgIPttttOH7q77bab\/htynjFjhqYHB4wPXy666CJ10kknafrWWmstDToAAn7961+rU089tRuNSe0BUgFYAMpwSEHW0C\/IHgcmniR9BciFtQ+HNwDG\/fffrx5++GF98F177bWxAAbgfYMNNlBwZwCcQq7gBx34cL\/RPABO8ecPf\/iDOvjgg7v0Cz641gXpCHQMY3388cd6HMhu7Nix2g2TRJ85kGuc999\/XwFIkEUU84DugZdYx+ifgMWll16q2traEnXBZhrtCaADff70pz91uoSjAAz4gLWFHw3QPwKD2DfwQ4lcSE8\/\/bTeV7B\/wL3U3t6u1yrWBwClCRq57efc6REAw13CKemLi4HBr5uRI0fqHk0QgV9BW2yxhfrRj36kQQMebPL4hXXsscfqQxkgBRaI\/\/u\/\/9OHKt45\/\/zz1YABAzTAoMPddCHRRo0NCYcwPebY+JWLgw\/WgkcffVQfOnRY4BADaAFYwi86gABs7gceeKC2aOBXP773ATBJfZB1AqAC4AIP0QQLzF577aWWW245DWjoVy1+Jf\/85z\/X1hGAGfxKN+eCwxybLgDOU089pb7zne\/oX\/WgF7RgTPuheQA83HnnnXpzJlN7XD\/4dQ2wiTkDEOLBGLDe4Bc6vrctMGRRgLUHoAcPtQFfwd8kvsBtSOAQ+gJZQ4awvAEMutyYSe0J4IAOyBoP3gE\/\/vWvf6kvv\/wyUV9JxwA0ADggKwAguDvAE7go7IfewRrBWsE4ADzgO8bF9zhUATgA1vv166ctXS+88IIG0y7LTty62HnnnTXIgHwx3ttvv63XHSwQSevRnrs9Dn5cQM8IRAIAwGoI6yJAHUAXxgeYBgCCjgMMxOmCPSb4gvgVAEM8sIzgBwf4jXFhlcITBWDwQwT7AtYN6DYBFWRPAAbrEUHn0EnsRwB6ADuQJXguACblIRFQcwEwAQkjhKnExcDgAKaAORNE4PCFS8b14ODDO6uttpreMPDL3vXEbdQ4BHGYugAMANF+++2nNz36VWv3D3cULBCwdgBAwEWCTZ\/AkgkayL1EfeCAgSk7qQ86qHfddVftOsMDgAYLAkzs+AMe4OB79dVXOzfNddddV5u1XQAGIAQAx\/XAkoUDOQrAgCZs1njeeeedxH5oHvjlinniIZlEARjMDX3j8MBBjQcWAPALunLHHXd0ApgovsBCAusZ+AOZAKThlzjmg1\/5rhtuSe3h8omykpx33nkazMbpK+gl\/X7mmWf0wY0HPIf+wD3lcm3RO3Cn4ZDEg0MT\/4dVcJ111tHWP7hSYXEAwIa1DuAeFsK06wK0bLTRRvo1AqAYBz8Ukuizx7LXH2SBYG243G644QYNLABS\/vOf\/2hrFIALrJ1YL5i7jy5E7W\/PP\/+8Bs8A6QB2sPqBvzfeeKO24kUBGFh58aMF7fAedB5gBG5P6A8BGOgo1jGAI1nVsF8AgC2wwAIhbLsyh4wcEACTkXFcX8viQiIAg02ODm\/iD35l4hDCBodYBFhe0m7U9vVmF3hC\/2YcDY3x7rvv6oMCwAEgBu\/CfIxfxC4Ag1gCuK\/owcEK83RSH674EDKRUx\/YUDGP1157rfMXfByAwS9LvAO+2QcywCDd0jD56ZoHeJDUDw7XyZMna9fNKqusorsEuMD\/owAM+kTfJoChWAnwGVaFJL5gDDzQC7ixYHnBL3oABbhhcNi7nrj2sALAEghrDn7Jmw9chgBZOOCj9BUuUFdALkAZwI8JUMy+6R0T4JCFhXQYPP7lL3+pfvGLX2jwAbAFyxNAX551QW5UWJ9g2UiiLwnA4Hu4tRAHA2sL3IuQ85QpU7Q1C+Bpzz337HR9+eiCz54JqxWsZrCi2reO7P8DwMB1hfWF20j4sYB5\/upXv+oCYDAuYqsQ7Au3HoASwAwsfNgHCGz6zE\/ahMUBATBhyaP22WQBMLAoYNMwAxFhVXjvvff0QQD3Dn5l4pcVWRpgioblZMEFF9SHHP0CNDfzKNeOebjQ2KYLCZsuwAMOCsQfYHMniwV+WcKaA0sQubeSXEh0WyKuj6SDGocquZAo3sXHhYRDFgcSzPOIFYD\/Hr9Y4XpzBSC65gGak\/rBr1GY8imeAIro60KiIF+8QzEdtgvJzB1jAjsAGNIVsvzAgoAg3qlTp+o4KdvEn9QeMRmwcMCNArCKB6AIsTmQASxxcfoKq0MeAEMxT9OmTdNgEHEjJD\/MBQcx5gMXISx7sGYA6McBmKR1YQIYgPAk+qIAjDkOXGFwUcESA52DNQo6C+sFLG4IECd3GbkT43TBHBM6CYCJPgG6zGv5FDeF2CPEf8VZYNAG88MPA+wvACQA96YFBnPGeod7t2\/fvgpB2gBnkIkJvmvffGUCqTkgACY1y3i\/kHSNGocCgmXtDR6bMn7dwG8O0zKuMiIQEL96ACLoYAUI2GabbfSvbWxeCFzEZkKmdcTR4PBD0KgPgIF1B79e8asa88JmBjP6Rx99pH+F4xcaDsPZZ59dW1yw6eIXGGJJYFXBgQ0LQ1wiOxx4SX0AmNlJ3uyDGr8SYe4G72AZgPsL8zCBnU0zAQLwBe8A\/IAGAl+2NkYlm0vqBzFB+OULdwkOI8R5wJqCuUVZYJAvCG4eHBiIgYHVBHLH4YSYIVPucQAG7iUc5NADxEfBqoP54n30Yz9J7SmIF8AB40JHcCjCXQCrEA6xJH3NAmDgTsFDgergH\/QS4B38pQcxUADv4BOsTBdeeGHkpuK7LkwAg3WQRJ89oGscioNBW9NFC+AFSwwsg1jD4KePLthjwooDKwr0BxYcAEfIHkAaYBNAFJbBOACz9dZba2sT9ASAHqAEANcEMLDigT7wGn+++OILbdmDFQZ6RzfTeO\/sPKkTAMNTrpmpSkpkR7+wXNeoARDgH8dBhsMHpmb6ZYVfPThcEVAHYANrDQ5rHJZ4YFrHv2GZgesEfnVfAAOrBIL5sHnBxI\/3YUrG5ocHv\/AAJmCehkkfhwdAD8ADLD+w2CRl4k3qA1aDJACDDRM8gd8eOS8QwIq4HYAFCuKMukYNvoFOBLXS9XOXkKMADF2jjuoHBwZ4iO\/xaxZBmzhgYU0gV5urb\/ANVg64VRBfgPgV0AgQiyfJMgULDOhCgjLceqJkZtAbACpXEK9PexysdI0aQa2whMC9CRcSHvwqj9PXtAAG\/PvBD36ggTIANEAfDnlYvnCDx7SWQQ8xH4BDHKQ4VKMe33VhA5gk+uzxXONA\/rC0QG\/NhHHkojMD1tFfki64aMQ6h5sSPxIgV7hzELiOHzC0fpMADN3ugksLOmMDGMgGlhz8mEKgM\/QBt7cQ00OxW5k3THmxVg4IgKmV\/TJ4K3EAGzQCCgHe4DIAqIDFCOAHN1HwS7auB9YKbO6YA9xTeMgtEEKJhbr4Usa4b775po6BwS9\/WGlggZBHOCAcSM8BATDpeSZvCAcycQDuKtzywa9yuNHgcsNNHbjd4OKo86FMrDhU4XYB2IKJHxYQuLrqBFd18qXIsWGRQcwFeA2LByxeiAORRzggHMjGgZYEMDDhwryMjQTuDviskazLdasjG1vlLeFAdw5A7+BOQKAhzPIAC7iJAlcS8tPU\/cCdhZtcuCqLLLQwr8MkL9lKi5EMEqohZgguDLgKKUdNMb1LL8KB1uNASwIYpPLGVUf4n+FzRSAZgkoRcCePcEA4IBwQDggHhAPhc6AlAQwiz+Hnp8yXuGKH3B+IunclzgpfjDJD4YBwQDggHBAOtBYHWhLAmCKGrx\/uJJj0cXtAHuGAcEA4IBwQDggHwudASwMY5BCAXxpXPpF+GgmgXA8VFQxfnDJD4YBwQDggHBAOdOUAkkTiAgG3p6UBDIRJVYuRTArBla6MmEhQhZwh3B6hq1kSFXk1S16YrcisWTITeTVLXi0JYJAVE4mkKN8FRIbkSbjKiuys9iNK3SylFnmJvELhgOhiKJLwm4fIy49PobRqSQCDrJWor4LMqbiFhGyQyAh677336gJfAmBCUc9s85BNKBvf6nqLq7zEAlOXRmUfl6sucqWrJQEMpdlGSnfk5lh88cV1PQ1KXd0qAAaF2pDynNsjdDVLolzlBSlwpU3oatYaEwDTLHkVOluuwkfNEI5JyoSuQtW\/9M64yguM40qb0FX6sih0AK5nWEtaYNJqBlfhyyaUVhPqbS+34erlf5bRud7+kL0jizbU9w7XM0wAjIdOcRW+bEIewg+oCVc9DIjFhU+Fq8xk7yhcVUrtkKseCoDxUBuuwpdNyEP4ATXhqocBsbjwqXCVmewdhatKqR1y1UMBMB5qw1X4sgl5CD+gJlz1MCAWFz4VrjKTvaNwVSm1Q656KADGQ224Cl82IQ\/hB9SEqx4GxOJuU\/nkk090Ze6sD1eZyd6RVSPqeY+rHgqA8dAnrsKXTchD+AE1idPDk\/9vcudMH359qvrLhkuony\/ZFjv7J554Ql188cVqiSWWUC+99JIupXHSSSepu+66Sx1wwAFqrbXWUgcffLB6\/PHH1RFHHKH22Wcf9ec\/\/7lLn6eddpouw4E0BHvssYfq3bt3l+8fe+wxdcYZZ+jK764s199++6166623dCqDkJ6ZM2eqyy+\/XJ133nkKfMr6yN6RlXP1vMd1T+SqhwJgPNYJV+FzXaxc6YrSQwIvAC30uD6zVX2LLbZQRx99tK4F9umnn6rjjjtOnXrqqbrZGmusoQ488EC15ZZbdv7\/888\/V4888ojq27ev\/gw5lJAUEp8988wzau655+4yBL7\/7LPP1M9\/\/nMFIOMCMMcff7zOgr355pt7rMRqm3z44Ydqww03VE8++WTmgevaOwBikwBsZqLkenge1tXybl16WDaxAmA8OMxV+FwPeq50ufQQB9XY1zosLvaTBGK23XZbteiii6oTTjhBg5IXX3xRLbfccp2A5S9\/+UsnsADQ+Oc\/\/6mtLLvssotuc+2116pevXppK81zzz2n5pxzTudqWn755dVDDz3UDcAgoeRuu+2mfvSjHynM5bXXXlPXXXed2mijjdQVV1yhzjrrLAWQBfDz3nvvqV133VWdc845CleT77nnHp1jZezYsWrnnXdW6623XpexYUUCKJs8ebKm8Te\/+Y1+F2Msu+yy2soEi9IGG2ygYG1BVu4FF1xQ04j+N954YzX\/\/PPrfgnAuMa8\/fbb1TzzzKPniHmvsMIKXeZR9d5hWuJO\/r831F82HOjUDY9tL7YJ1zXGla6q9TCvfvm+LwDGg1Nchc91sXKly6WHvzrvaXXbnj+J1GIcaC5wgxeef\/55NWzYMF1WA9aXX\/7yl539wAJjApiTTz5ZzTvvvGrUqFHq\/vvv18AF1hvkpsHBnQXAYLC99tpLvw8LzJQpUzRgQEkPWG+QZBE1ywAqACaGDBmiXVvf+9731Pnnn6+OPfZYDWD2228\/7ebp0aNH5\/xhGTr88MN1qZBf\/\/rX6sEHH9TFWuHKuvHGG9UDDzygrrzySnXppZfq8e688041fPhwnTkXdALctLe3dwKYd955xzkmwBPcTOAHwFGdAAayNi0v+DceWGKgA0VaZbiuMa50cT3DBMAIgPHgQLOatNImFAdQILWk7xGkijiWf\/zjHzqOBQc3HheA+eMf\/6g\/P\/HEE1X\/\/v21hQPxM0UBGFhkNtlkky4xJyuvvLIGFwAwsNJgjgATAB2YCz1bbbWVmn322bsoKtxaAD+odQaAg3euuuoqdckll6iJEyfqeB6AGlhRAGauueYadcMNN6jXX39dgzcTwKAP15gAQLAaIXZo1VVX7bZQqjo4CLyMfa29Q34\/6Kfwb\/qbPjMBTZ5VzXWNcaWrKj3Mo1NZ3hUA48E1rsLnuli50uXSwySAEmWhAViYY445tPsDz2WXXabdKDjo4U5yARgc6nA3IagXlpFDDjlEWx2qBjAAGPhz5JFHdq5eWGxADz0ITv7iiy+02wlWnTgAM23aNPWnP\/1JW6AA6GCxWWCBBboAGAAc15iwvNxyyy0KLjbED9murCr2DtPyQm4j0\/rS4WbsADOwxpAlZo0ftGWOk+G6xrjSVYUeehylhTcRAOPBUq7C57pYudIVBWCgwlExMFGH1Pvvv69dIlTME9YKuEOeeuopNdtss3UL4qXifW+\/\/bb6xS9+oQ477DAdCwNLxqabbuoM4qWlhbiahx9+2BnEC5cQrCzbbbedQtAs+ho3blznqlx33XV17MpSSy2lA2rhtmpra1NDhw5Vd9xxh45vQSwOCrHCSoNnxowZCnE3cC8BnKDPp59+WsfNwMoCCwzcZ7DmjB49Wo8LKw\/6NJ+PP\/5YYfwJEyZoF5lrTLwHPqDvMWPGaOuU+VSxdwCkApgAvACk4LEtMaYFhsAMtcsSJ8N1jXGlqwo99DhKC28iAMaDpVyFz3WxcqWr6FtIa6+9tgYqiy22mHr22Wc1iFhzzTX1YQxXEg5v3ERC1Xa4ShAHA7AB0HPQQQdpoICYEvyBOwYBvqYb55tvvtHuG1huEK8CywbdYKJlh2Daiy66SIMJyA19w3pClgxYPuCiAWiaNGmSWnrppdXee++tzjzzTHX11VdrALPnnnt2s3ygDaxMoAkAaKeddtIBvQA1cJfBVQSaYD2BKwwgBLeo8AdjYc4jR47UMTFwOyGw1zUmYm0QvwOwA94gILlKAANrCiww9o0jgBkCLXFghuZKcTIe26FuwnWNcaWL6xkmAMZjxXIVPtfFypWuOD10HWRRwbseKt9STW677TYNPGDZgQtp\/PjxatCgQfq2Ut6nzL2DXEGmm3De\/e\/rBlyIBlhaCNiY4IbiZPA9LHZ4kq5gc11jXOkqUw\/zrpE87wuA8eAeV+FzXaxc6eKqhx5LsNQmuDKNGBhYn5BUD9emERRsW4uyTKIMmdlJC2FhAfjAQ\/EuUXM1g3sBcKkv00rj41riusa40lWGHmZZD0W\/IwDGg6Nchc91sXKli6seeizBUpsg\/w1uIiEuCEn1cNupCPCCSRctM1duH7K6YLwkK0scI01LDAGhqPgYrmuMK11F62GpCzJF5wJgPJjFVfhcFytXurjqoccSbGyTomXmulUGUAPX0Men\/\/J\/FhjExXQksTNdRsREM9DXt43tjuS6xrjSVbQehrIgBcB4SIKr8LkuVq50cdVDjyXY2CZFyiwq6zI+3\/zcCZ3XpMEs+0ZSFFBxfW7njiH3kmmN4brGuNJVpB6GtBgFwHhIg6vwuS5WrnRx1UOPJdjYJkXLLCrvD9xIZIHpiGHpKC8Bi419bdoEKCZj4z6na9oU6LtIz3adKZnbI3tHsyQqAMZDXkVvQh5DVtKE62LlShdXPaxE2TMOgltJ3\/3udzO+XV4MjJnfhxLZYZJ0e4gS2ZkJ7Myr1R0gpyNrb9xjgxrz\/1nyxySNV\/f3snfULYF04wuA8eAX14OD62LlSleUHk5\/8QHVZ7m1u2gyPsNjf242QnZa5FxBOYCXXnpJLbTQQjolPnKzIMHcWmutpQs1IvPuEUccoXPDIN+L+aAMwbnnnqvzuCAPTO\/evTu\/RnAsEt4hgy36QqZaOzj222+\/1Td\/Fl98cY+VWF0TFHi8\/PLLdZ0j8CnrU\/Te4SoZYIMW+r8NWEyAQy6mJBCT5GLiBmJabe\/IqtehvCcAxkMSRW9CHkNW0oTrYuVKVxyAab\/uaLXw0R05QABezP9HKRMqPSOz7eDBg3UyNxR0BMjAgwR3SGK35ZZbdv7\/888\/V4888kgnCEH6fiRyw2fI5IskcOaDDLmbbbaZmjp1qi76iLT+v\/3tb7u0QQp+3PxBMrjQHmToRfZfqkadZX5F7h32DSSKicG88G8q6mnmhwGYgbUGLiX63Lxh5Lp2HVdHCYDn9r0Gq2Nunaj69OmrWZI2CV4WPlb1TqvtHVXxtaxxBMB4cLbITchjuMqacF2sXOmK00MCLf22PdoLvEDJcF0YmWxR3wiWEVhMkPafAIxZjRpAA4URYWVB1lo8ACioBQQrjV2N+quvvtJJ4Si9P95HfaHdd9+9U7+RKXe33XbTSeQwF9RVQtZd1Fa64oor1FlnnaUAsh577DH13nvvaQCErLrIiotswZAzMuuiBIJdgwhWJIAyZN8Fjb\/5zW\/0uxgDCerwPSxKG2ywgYK1BXWgFlxwQU0j+kduGMwd\/RKAcY2J69eoJ4U5Yt5lVqOOqmvlulptghtyN6EdgRkt4\/\/VQkL8DCwpptspyybEwRrTintHFlmH8o4AGA9JCIDxYFJATVp1E2q\/\/hgNXr5\/\/UwvaaAe0LBhw1TPnj219QXFDOlxFXOcd9551ahRo9T999+vgQusN7CsJBVzRJ9I9w+LDtaS+ey11176fVhgpkyZogEDqj7DuoMgURSNBKgAmBgyZIh2bX3ve9\/TdZxQngAAZr\/99tNunh49enR2DcvQ4Ycfruaaay5dwuDBBx\/UladR9uDGG29UDzzwgK5AjXICGA81jYYPH65LB4BOgBuzGjUqYLvGBHiCmwn8ADgqC8BE3UACwS4AQ5+blhlqR5YYYpavO8lHqZoOYlp17\/CRbYhtBMB4SEUAjAeTAmrSiptQFgsMRAYrCeJYUB8IcSw4uF0WGNRB+uMf\/6hdSyhY2L9\/f23hQPxMEoBBDAwAxvbbb99NS0wAA4vMJpts0iXmBPWFAC4AYGClwRwBJgA6MBd6kDnXrMOEz+HWAvhBPaa4atSwogDMoNAjaiRhvrA+mQAGfbjGBACC1QixQ6uuumo3+orcO6JuIFExRwxu5mvB53jgWjLdSq64F9+bSa6gXptozCGpFEFA20WXqbTi3hGqLHzmJQDGg0tFbkIew1XWhOti5UpXkTEwAAtzzDGHdn\/gueyyy7QbBQc93EkuCwwOdbibENQLy8ghhxyirQ5xAAZFDlE12q70TEqeBcAAYODPkUce2blWYLEBPfQgOPmLL77QbidYdeIMANVjAAAgAElEQVQADCpWo5QALFAAdLDYwN1lAhgAHNeYsLygICRcZIgfsl1ZRe4dLkuL+RkF+JrgwRXjQleh0c4EPz6WmSig49q0mmiNabW9o7LDpqSBBMB4MLbITchjuMqacF2sXOkq8hYS0ubDJYLqz2StgDvkqaeeUrPNNlu3IF64VtD27bff1tWacbsIsTATJ05Um266qTOI97PPPlO33nqr2nHHHbV7BzE2cP\/MOeecnToOlxCsLKgajaBZ9DVu3LjO71ERG7ErSy21lA6ohdsKRRcBiACMEN+CWJx11lmnM94GVbKXX3557V4COEGfTz\/9tI6bgZXlkksuUXCfwZozevRoPS6sPDbIAvjC+BMmTNAxPq4x8R74gL7HjBmjrVPmU\/Te4QIpZHUxg3Q78rZM1lPB9\/QdxbvQZ5QvxgZCRAPiZJAkLyoRnr1Z2QCnaSCm1faOyg6bkgYSAOPB2KI3IY8hK2nCdbFypatoPVx77bU1UFlsscXUs88+q0EEChriMIYrCYc34lYeffRRHSsCNxLABoDMQQcdpAAUEFOCP7hejQBfcuPAIoL4mH\/\/+98avOAPgmdHjhzZRbcRTHvRRRdpMAG5oW9YT8iSAcsHXDQATZMmTVJLL7202nvvvdWZZ56prr76ag1gEF9jWz7QBlYm0AQAtNNOO+mAXoAauMvgKgJNsJ7AFQYQgltU+IOxYG3CXAHcrrrqKh3Y6xoTsTaI3wHYAW8QkFwmgKG+TWuJXdyRbgXZQb9mYC+BHnJL2QCGShFgvKQCkSa9ADAY33yHPmtCdXTZOyo5egobRACMByuLPjg8hqykCdfFypUurnpYibLHDHLbbbdp4AHLDlxI48ePV4MGDdKAK+9Ttsyi3EoEcGzQQO3Nm0kEUmzXE9GelCtmlrXGDV5cpQjy8rWs92XvKIuz5fQrAMaDr2VvQh5TKKUJ18XKlS6ueliKcqfoFFemEQMD6xOS6uHaNIKCi6hIXbbM0lytBkvMwo\/EIio3QFepATjw7472s4pCmu6hlQb00XlgKNOvL8hBuYOQH9k7QpZO97kJgPGQV9mbkMcUSmnCdbFypYurHpai3Ck6RWwObiIhLghJ9XDbqQjwgikUJTM7wJasJhTDYpNrxsqYVhhX2QG8S3WTzCy+VLUa32Ec84mrdh3H+tDdSbJ3pFg4ATQVAOMhhKI2IY+hKm0S8mJFTpN+QzoCTPHY\/49jVMh05REwVz3Mw5PQ380rs6j4FqI77mo1rk\/7BP2iL9sVZV\/NNmNlqESBabFxycF1Y4k+CzW4V\/aO0FdU1\/kJgPGQV95NyGOIWpqEvlgJtKQBL2Bk6HRlFTaCYpHxVZ7mcADBvwgCzvIkXZt2AY+oz1wWHLQ188OYpQjIskNVrm0gFeUysos9Ethx0R+iO4nr3sH1DBMA47GzcBV+aIvVZXVJk1mWRBkaXR4q5tVE6PJiU1CN8sgsLr7F5RYyg3CTbvz4WnaibieZVhjXv30sMiG6k\/LIKyjFsybD9QwTAOOhdVyFH+JiNa0uEA3cSGKB6VDSEOXlsXwSm3ClK4\/MkkoH0C0ik7lRVhZbAC7LjpkfxrTidM8t0xHUO3Vqu3rhox46dgbWmjhLC\/qLq2odkjuJqy5yPcNaEsB888036pRTTtFJtqZPn66WXHJJnX\/CrmNCC5+r8ENYrFFWFxQllBiYrkdPCPJKRCMZGnClKw+AIRDhsqREWWbiWG+CG9f7SbeTqG\/Mh9pSkK8Z+EvtTFcStbPdTmabUNxJXHWR6xnWkgAGia2QQfPyyy9X8803nxoxYoRObPXwww879wCuwg9hsQLAmJaWrFYXU3Ah0JXhHE98RehKZFFwDfLIzCcGJolg21WE9pTozmWZsa04ZhZfAlVoM6j\/TNXW1k9bYMwxyBKTpuQA+g3FnZRHXkmyqPN7rmdYSwIYZONExk2yuEBpkYoc1yld1ye5Cj+UxUpVlPNYXQTA1Lk95hs7FD3MR4X77by0xd0iSppvWgBElpm4gF\/0iWDfQ66doAEMHgIxeA9PmmKRNg11u5PyyitJJnV9z\/UMa0kAYysRUqGj0ixSlrsersKvc7GarqMoAJN1sddJV9Y5+7wndPlwKaw2RcnMN77FpD5NkjsX2KG+6DsqD0AxMARgCLiQdYfqJ5lzcVlkzGvYppupTndSUfIKSwuLy0cUGl0tD2BQC+Wkk07SdVUGDuzIPmk\/ADDmg7opUdV1QxNw3HymTJmiBgwYUP2U7x6h1Pr7KoW\/6TH\/j3\/neGqjK8ecfV4Vuny4FFabumT25FvTFf4MW6WtG0MuGjdVf4dnxUX7dH4f1RYN6Du8iwf1o1DJHP2gD\/ob7ah\/fH7h41MVsvbiGT+lY8yoB+2ozR6rtjnnXrZ065JX0XSNGjWqW90x1BLj9rQsgEEhuuHDh6v77rtPXXjhhWrxxRePlK1YYIpRe4p3MXvDNWnTdZT2xpFrZlx\/RQldxehhlb3UKbOkJHfgQ5Jlx7bikDUGt5BO3G6wZiUlvTNdSaYbKSqI15SDbaGpKyamTnmVqZdcz7CWBDAzZ85UBx98sPrwww\/V2Wefreaaa65Y3eEq\/KoXqwlgAFzwALzov42su3kXctV05Z2v7\/tCly+nwmlXtsziAEjaGBiba1FuJfMWEt1AMsGLfSvJBjA+Ab51ZewtW151aSbXM6wlAQwyY+LPzTffrHr37p2oU1yFX+VidYEXAjBZcr3ECa1KuhKVp8AGQleBzKyoq7JklpSIjsjLEgRs3ypyBdZucPqj6qgtltU3mugha41ZdoDiY9AuKVcM+jGBC8XcVBkTU5a8KlK3yGG4nmEtCWA22WQT9eqrr6pevXp1EfiVV16pVlxxxW5KwFX4VS3WOPBCzBYLTPIWV5W8kmdSbAuudIFLZdCWxbKS5CoyAQ\/+TflnyNpighh8ZrqQ0B7943MAFTPY13YrkTUG77jywhDQsUsSJGUWLkojy5BXUXPL0w\/XM6wlAUxaReAq\/CoWq5lZl9xGZHkx5SAAJlkrq5BX8iyKb8GVrrIAjG+JgSySiktyBxBDz7Y\/nNnl0gNZeqiekivZHVWwxi0ln+y9dYAYrrrI9QwTAOOxyrkKv+zFalpeAFAmDemhuU1xL\/rfBca+kCjLpstDZUppInSVwtZSOy1aZllKDPgSmNS3aZmx6bKrV9OY9LnranXSvGwAY46f9G7W74uWV9Z5FP0e1zNMAIyHpnAVfpmL1b5xRLeNbHYLgPFQwP81KVNe\/rMoviVXusqywPjcLvKVku1a8u3blJkJfOyYG3xH9ZLMuko0v6iq1vQ9gRj6u+x4GK66yPUMEwDjsdK5Cr+sxeoCL8Tm718\/UxdnLMv6Utah4aEmpTcpS16lTzxhAK50laWLWWJgbBFEBQH79k0yM5PcYQwqU0BghVxStoWGgns3P3eCnlrUzaSoBHhlxcRw1UWuZ5gAGI\/dm6vwy1isceCljCvTLvGVQZeHmpTeROgqncWFD1CWzLLcLiLikkCKT9+g67pXOlzCdj0k+zO4j+wAX8rzgriYKPBiVrC225RVcqAseRWuWCk75HqGCYDxUASuwi9jsVZ940gAjIcCB96kDD0MheSyafO9XWTywzcIOK5v0LXvXZ\/oukgmMDKvTZtABp+72sKFlFSt2nYjlelOKltedekl1zNMAIyHRnEVftGL1efGURkxL7YIi6bLQ0UqaSJ0VcLmQgfxlVkWIJJlokmBurCWmLldosa4YexE9frnfTuvW5vtYG0x+4kL8DUBDPqIi4kxLTJoe\/teg73mmoZPvvJK02cIbbmeYQJgPLSLq\/CLXKw+riOwWgCMh8JFNClSXtlnUfybXOkCp5Jo801IVyTXfQN148YkF5IrFsW08JiACZ\/jIYAUly\/Gl96iXUlJ8vKdV2jtuJ5hAmA8NI2r8ItarFElAsybR1UAFxJlUXR5qEalTYSuStldyGBxMkuKRSlkAo5O8o4L4LFIz\/YuMTA0jKtv0wJDLia4k+zPzWy9rtpIZK0xrTSUm6aooF6ua4zrGSYAxmOX4Cr8ohZrVNxLFTeOXOIrii4P1ai0idBVKbsLGSxOZr6xKIVMxOrEJ1DXHte0Fo154R217qCFdRPb\/WWCCXMcM7uvy2VmWmTo3\/YcyI2Ev802RV2v5rrGuJ5hAmA8dgeuwi9isSYF7VZpeRELjIcyB9ikCD0MkCw9pSjaiopFiaLbN6bGt51tWbFvIUX1Y9ZHMttQUC+5lcwMviYwibuhFAecsuoDV13keoYJgPHQdK7Cz7NY42JewNKys+3GiS0PXR7qUFsToas21mceOMmFlBRDknbgsmJqbGsR0RUVT0OWGZQMMGkk0EGBvub1aqqjhHfNf0cF9pZRaoDrGuN6hgmA8dghuAo\/z2KNsryY4KUO60vcr14PUQfdJI+8QiaMK11Jupg3FiXKxWO7cNAuT4yIy1pkApi4m0txAcOUoRd\/U2wM5krJ7Yg++\/ZR1OdFBPRy1UWuZ5gAGI+dnavw8yxWAjBmgca6LS\/iQvJQ5gCb5NHDAMnpMqUk2rLEokTRXGZMjQ1EiK6oMWmOFKgbBapcgbxkvcHfPqUGyFpDt5vygLUkeYWub1Hz43qGCYDx0Eiuws+yWM0yAFSckVhYp9vIFGMWujzUoPYmQlftIkg9AV+Z+caiRE2g7JiapBiYKGuQCUbM\/DKUvZfoNgN8KTsv4mJoXHxGj53Yzv48T0Cvr7xSK0LNL3A9wwTAeCgWV+FnXawAMXRF2rTACIDxUKYcTbLKK8eQlbzKlS4wr0raisjvEidw01o0dWq7amvrF5nIDv24biOZ2XijAnwBVuiWEVlhXLePzBgYKk1gA6K0ClylvNLOLU97rmeYABgPreAq\/CyL1XYdAbTUle8lSnRZ6PJQg9qbCF21iyD1BKqUWdExNVFWIcoDM3DgQCc\/fFxZLosRxcEgBoay7KIvV\/6XqNtJNKGs8TBVyiu1MuV4gesZJgDGQym4Cj\/LYiXri8m2uvK9CIDxUN4GNMmihw0gS0+xatqKiKnxuckURVcaV1aUxWje\/e\/TFhh64go+Uhuqp4T\/U0K8LCCmanlVpcdczzABMB4axFX4aRZrVNAu2AcrTF03jlziS0OXh\/iDaSJ0BSMK74nUJbOsMTW+Vpw4unxdWVFjmTEwYDRZYfBvs4q1KQQzLoZADP5OG9Bbl7y8FSpjQ65nmAAYD4XgKnzfxRpSnSMPcVX+q9dnTkW08ZVXEWNV2QdXuuqwwOSVm4\/7J4kuHxBEbShpHQET9E3ga1bOmDc0cLFvJNmghb4326Yt+MhVF7meYQJgPFY8V+GnWaxJ16bFAuOhSDmbpJFXzqEqfb2JdPlaOJpEWxr3TxJdca6sOICDnDJm8juzLfUJq4qdK8ZUWNxCIqtNWjdSEl2VLowCB+N6hgmA8VASrsL3Waw+16ZDAi9Jvw49xB1sEx95BTv5mIk1iS6f+BCT1CbRhnn7un\/iYmDM69IuoJdk5bHnQP+nsgNmUK99E4l4T23IneTrSmqavHzXO9czTACMhwZwFb7vYk26Ni0AxkOJCmjiK68Chqq0i6bQ5eMasRnXFNpo3r402nT5AjsfKw8sMHgow69pzQEQQZBvVC6YqOvWAFU+IKZp8vJdqFzPMAEwHhrAVfhJi9VVLsC8Nh0acCFRJtHlIfIgmwhd9YolyXLgml0emfm6qYrmis9NJpMuX9BjgqS4GlA0vh3TYuZ6ob7MLLxmOQJ8T1aYNBl688iraDkU2R\/XM0wAjIeWcBV+3GIFeAFAcVlfQrs23fRfvR4qqJtw3VybQJeP5cB0neQB077WDF+9ydouDkCZMksL7OIAD82VAA7xHZ9TzSSKkXFl6Z31\/kBFt5kIzPgE9DZBF7PIk+sZJgDGQxu4Cj8JwBBrKNsuZdrF\/wFiQn24bkJCV70a5xsfYs4yrczSWjPq4gjRlRXYUTyLXV4gCgzBbUQlAuwbTGaxR7tCNd1y6gBAAxPdSGnlVRf\/047L9QwTAOOhCVyFH7VY46wvoZQLiBMb101I6PJYrCU2yQIu0sosrTWjRHJju7ZdSHEuIbMj27qE76i8QBIYMq0w6Md1bRptKKkd1VIiS4xPHExaedXF\/7Tjcj3DBMB4aAJX4ccBGLDFrnNEJQNCjX3JY7b3UIPam3DdXJtEl098SFYLTNIBTkGteRSxqLiaLDEwPgAwzspFriCTfoAVABWqam0WfaR2dpbeuGDeJuliGj3geoYJgPHQAq7Cdy1Wsr7YlabBJop9EQDjoTQlNOG6uTaRLl8gkJa2LG4qH1UrOq7GpssH2PlYl6JADhV4pCrWJs34zMzWawIX8zo1\/k0lCqJATFp5+fA+hDZczzABMB7axVX4UQDGtr4QiyjuhUCOB+tqacJ1ExK6alGnXIOmlZmPlSLthMroM4quuAKQZoI6kwbMz7QuucAQ+jUrWRPQM9uS9cW8Yk23lMjdJAAmrfaE3V4AjId8Wg3AmK4jkz2h1TyKEl3aQ8NDBYJoInQFIYZUk8giMx9rRppJ+Fg+0vSHtlnp8o2VoSvRswDI1C4BuKZFyb5t1NU6M1CXJjAfgJmoG0lZ6ErLuzracz3DBMB4aBNX4duLNa5gY1PcR1k3Vw81qL0J182VK115ddHXTRWnmGXF1WSRmY8lKMrV5XKt2W4jiofBjSX7AWCBBYhuJUXdSMpCV+0bg8cEuJ5hAmBaWPguANN060veQ8NDHWprwnVz5UpXKLpYZFwNgaqsMouzLqXJD4O2FBdD7iXqG3w3XUd21l76P13LNhd0Vrpq2xQ8BxYA48kojs24Ct9crHHWF7o6HXrwLuke101I6Gre7hKCzHwsH0mctS0jg\/rPVCduNzjptcjvs9ZIMitUkxXFpI\/KDGBgVyZfM5AXbWyXVgjyyszUmBe5nmFigfHQFq7C9wEwFPcSeuBuK\/yK4rq5cqUrFAsM5pEnrsYFgA65doJqa+uXmBjOY3vVTXxdXXZmXuqf6CPQ4hrXLEVAQEYAjK+EwmwnAMZDLtwBTJL1pSmWF7HAeChzgE0EwFQnlCxxNa4gYMjsuld6FAZgCGRFBfm6KlybbQlkwa0E15DrWjXdQKKgXlcwL1dd5HqGtSyA+eCDD9Shhx6qxowZoyZOnKjmmGOOyF2Eq\/CxWNvGj+qk245\/aaL1JaRfvUUfS1w3V650cdDFKMsIARif5Hq+oMknB4xpcTFdQJTIjsYy\/zavUNtXre04GK66yPUMa0kA8+abb6pddtlFbb311ur0008XAGNl3aVNoinXpm2gwHUTErqKhoTl98dBZq4gYNC1712fdMnNYnMzS\/I8nxwwttvIrEhNie7MGkjUnmomkSvJTI7H3XorAKb8tV7ZCO+++66aMWOGmjZtmlp\/\/fVbGsAMHDiws+K0KYCmBe6ac+dwaLgWg9BV2RZR2EAcZJYlBiZv4HBXC0rXHDAkHLK6dLif3ugs1kggiMCMK5gX71CWXrNGEgd5uZRXAExhSzqcjl5\/\/fWWBzBwIbmuTguACUdPuf865HpocHAhuawd+Gzq1PbYW0hFJs9zWYDoGrVZodq8Vk1WGczV\/DcF75oFHgksAfBw1UUBMOHt57lnlAbAmIPB\/TR06NDc49fdwac3Hqfmfvzv3abRY\/19Oz6jv+ueaMrxp0yZogYMGJDyrfCbC13hy8ieITeZPfnWdLXion1UHF1ogz\/DVmnrJrCLxk3V7+OP\/VDf9ud4B4\/Z34pnT1Z7rNrW5TO0u\/DxqWqlAR39oz8846d0\/E0P3qN2+O7JfZZQeBf9c5HXqFGj1MiRI7vQPWnSpOYtoIQZt2QMDPEkDYDhJnzcPJra3q5m3j3CqSJNjX8BMVx\/RQldzdt\/W1VmaZLn+cTK2LExdm0kaIaZyM7UFFhWNj93QudHlMiOPoAVB+8iIHmRnu0KbnVuj1hguElUKSUApl219evXzYVERRubKvJWPTREXvVwIO6WTavqom8MjG87kuysRHaTnbWRTFmYN47wvp0jBkG9ZqwNikVylZcAmHr2hlJHFQDjtsA02foiFphSl0wpnTf10PCxHDSVtiRB+9DlkzzPN1bGBok28DEDes2bSESHHciLz01LDNVR8qEriTchfi8AJkSpZJzT+eefr0aMGKFmzpypvv76azX77LPrni677DK1+uqrd+uVo\/DhQooK3m1a4jpbYFw3IaEr44Iv4TVfy0GRMvPNp1ICud26TENX1Lx9su+iAKNtfaEEdiZAMgNx0d50J5ngBQG9ZJkxiaKr1dv+cKa4kKpQoILGaOkYGF8ecgMwceCFeNJkEJNmc\/XVgRDaCV0hSKFjDr6WgyJk5mPpqZozRdBFQCMp+64r4675mQ2ETH7RzaSOsd7oZJOZE4auU+NvBPRKDEzV2pR9PAEwHrxrFQADVjTdfQQaitpcPVSj0iZCV6XsjhzMx3JAqe\/zyszX0lM1Z\/LSRfONo88VqGuDHjMmBt8B2BC4dFlhovhEgbxJ18Or5nNR43E7w4gvAmA8NIST8JPqHmkQM+QoD66E26SozTU0CoWucCTie8smr8x8LT1ZOZPVLZWXLnO+rlgZ3AiC+8hlnXHdSCLASHWO7NpJUUUe7UDevFW2s8qh7Pc4nWEmrwTAeGgOJ+EnAZimgxexwHgodGBNijwMqyLN1zKSh7Y0lp60dOd1S+WhK2qurkDdOPdSlGvJBH2mNcaOfTFLCmBO+D8AzGYrDdTJ7zg9nM4wATApNZOT8KPiX8CSpl+fJrGWsbmmVJlSmgtdpbA1c6c+t2zyyszX0pOGCF\/wFddnXrp85hs1zzjXElluOgDJLCuO6U7Cdel5979PT8G+iQRLjV3g0WeuobfhdIYJgEmpbZyEHwdgOMS\/iAUmpXIH0LyKw7BMMsvMA1ME2LBpL8ItZcosqxvKRyZR7iLbMmPfSKLAXIxBpQQIuNiJ7FyJ7Xzm1qQ2nM4wATApNY+L8JPAC9giLqSUylFh86Yf9FGs4kpXUWDax9Ljq4ZFuaUgs+te6dE5rH2N2Xc+vu26JqiLTmIHkIKHgJ8ZE0MBvia4MWNjEA8z5oV31LqDFnbG3vjONcR2XM4wm7cSA+OhbVyEzzn3iylGrgei0OWxWANrUqTMirJ0FOGWOuTaCaqtrZ8zG64rbqVIsSQlsSMAY2bZJUsMFXHE\/+3kdnAdbXD6oxrA4CmbjiJ5ktQXlzNMAEySpB3fcxF+HIBBUjuJgcmgHBW+UuRhWOG0E4fiSldRFphEBqZsUIRbCgf96P1\/1m3kKHBkNiwCiCUlsaPxEAfTkbxuss4DAysLng5LVLuOgQFQwY0nfIYgXgCzqDiblKwOpjmXM0wATAaV4iD8uNtHYAmX+JdQD40MatftFa4HPVe6QtbFPG4pHO53jH9DnbjdYCeAIdBgf5n35pNrDblcYhTjY95AMkFLh3VloDMjLxLZ7XvXJxr0RNFRxFquug8OZ5iLZ+JC8tAkDsKPi38BC7hYX0I+NDxULbYJ14OeK11N0MWs1hC4kFwAJi5AuAM4LNGp4y5LEH2ZZl5mPwRo7IVk1kcyY2DQDv+nfDDTp0\/TLiSxwOTdrap5XwCMB5+5AxhO1pcmHBoeKudswvWg50oXZ11MGwPje\/Mpq5UmyqVk54QxayGZt49MANOnT19tgZEYmKw7VXXvCYDx4DV3AAPrCyw0HG4gcT40uB70XOnirou4hWRbSlyHvu\/NpyJic0wLjJ0HhixAFA9DriSKhzGPAm65YDicYeJC8gArriZNF37S9WkuwIVkx\/VAFLoyLuAaX2sFmfm4e3xuPvlaaXzHM69Qmy4hE8CYqmEWeBQLTI2LJsXQYoHxYFbTAQxInDRkVs4Gk2Ru1hfuv3o5VsrlesiLLs7aaZKsKz5WGtwUosc374zpWqI4GNdVavRLpQRe+KhHZykBLm4kDmeYWGA8wEorWWB6rL+vmnn3CFYBvHJoZFTyGl8TAFMj8zMOnUVmSTef4qw0VJsoLgg4yjJjjuu6Sk0sAIChIF58ximQVwBMRkXn8FrThd9KN5AEwDRvxWU5DJtCJVfa8tAVBzQgVxdIiat\/ZOqCaZmxs\/cSeKGMvHZAL\/oxbyFxukrd9DMsar0H6UJ65ZVXUu1PCy20kJpnnnlSvZOmcZOF32rgRQBMGs0Oo22ewzAMCqJnwZW2suhyWWnMYFyb04ibseNV8BkeqoFE39s3lcw2dJWaKlbTrSQqTRC6niXNr8lnWBxtQQIYMLt3795JMtHff\/311+qUU05R22yzjVf7LI2aLvxWqEBtyrWszTWL7hT5jtBVJDer6Utklo3PtpUmyr2EqtLmjSHzGjZZcsz4G7MqNbmlCLSYM4U7iVMgb9PPsEZZYM4880y13377eWn+Bx98oB544AEBMBHcigMviIFp69ePzfVpYoEcGl5LJ5hGXOUl1sDiVCwqCNh2LZnZd8lyg1nYN5IomZ1dD4lmLACmONmV2VOQFpgbbrghFSBJ2z4tQ5uMXuPqH01daahqGz9KAExahaipPdeDnitdAmCKXSgu9xJGIEuLeZOJXEv0vWl5oVmZIMYsLUDuI3IrcbiJ1OQzLE6LggQwPmr\/zjvvqIUX7qgaWvbTZOFHXZ9G9l1OBRxNHeB6IApdZa\/04vsXmRXPUzs41wQxdiZfimExY2Xo364Ednus2qZLJFAbuJc4JLVr8hnWKADz9ttvq3HjxsVqPcDLp59+qv7yl78UvzocPTZZ+BLEW4mKVDKIHIbRbPZJblamkKLGF5mVyfWOvl3BuVRh2qyBBDBjx8AQiIHLCN9DXsgwjIdkyqGoY5PPsEYBGMS0rLrqqmreeedVPXr0UF999ZX64osvVFtbWycdn332mVp22WXVzTffXP7qQKHD739fTZo0qZKxih4kygKDcXqcMklJYrSiOV5ef3IYdp49IQcAACAASURBVOdt1to5RUkpafwyZFY3WAPvyqArr0yILy6QQgG55g0luk6NcSkWZqUBfXQxR3rQJ9pRwG\/eOdb1fpPPsEYBGEzWjGk54YQT1G9\/+9su7qK33npLXX\/99d6BvnmVpsnCj3MhIQZGAExe7aju\/RAPjSKoz0pXUnbXIuYW14fP+Flpc42bBJby0psGGBVJV955R71v3z6CJYU+I2sM5YYxbyJRAG+HdUdcSGXJp4h+g4+Bueiii9SwYcO60frrX\/9a3XjjjUXwILGPpgKYpBpIU9vb1cBhZybS37QGTdhcs\/BU6OrKNd\/aOVl47fOOz\/hFycwHLPnMuShg5KIrDQDKOte079nBu2ZgLq5g4zGDdu3+8Z0E8ablenXtgwcwxxxzjNpzzz3V\/PPPr7kyc+ZMddVVV6lLLrlE3XdfhwKW\/TQVwIAvLgsMAnhRwPGNi\/YTAFO28hTYf1GHYYFTKqSrLHT51M4p0+zvO34W2lxM9QFLWYSRFRiZdGW1DFUJeEheVAeJktyBZ2ZGXriQxk+Z3slKATBZtKq6d4IHMG+++abadddddUxM3759dSwKXEinn3662mqrrSrhFDcAQwUcxYVUifoUNkhRh2FhEyqoo6x0+VQ4LmiKzm58xs9KmzmgL1jKQmtWYER0ZQFAaQBPGSAn6ko1+IdbSFTM0Ry76VaYJp9hcXodPIDB5L\/88kt1++23q4kTJ6q5555brbfeemr55ZfPsl4zvdNU4cOFhAfXpemB9YUeATCZ1KG2l4o4DGubfMzAWenKcngWSb\/P+Flps+fpA5bS0pYHGBFdaQGQD89Ahw\/IyQNu0L9ZlZp4N6j\/TNXW1q\/zBhIlwGt6SYGmnmFJOt0IAGMT8c0336gRI0aoAw44IIm+Qr5vqvCj3EeU\/6WozbUQJhfYidBVIDMr6CqPvFzJzar8tZw0fh7aTNb7HvxpxZUVGIGut2f0U2Nf67il4wJcruvHPoAnidY4cGODmiSQY1tjCMCAHjN7b9OvUjf1DEvS5+ABDKwuACvt7e06\/gXP1KlTFa5bT5gwIYm+Qr5vsvBNEEPJ6zpjYN54Q24hFaIh1XRS1GFYzWz9RymCrqSDyn822VpWkQcmCSxlmXkSWIjq03QhuQCMC6j4WnziQA7Nx65YbfLfBB4EROzK0nYyPLLGmACGbiihrVhgsmhX+e8ED2CGDh2qlltuOfX++++rZZZZRueGeeaZZ9R2222n1lxzzfI51OA8MOYtJAIvYJgAmErUpvBBijjoC59UAR1ypQusKYO2osFaFmCUNQYmyeKTBHJcYILmTyDDpMdMZAd5UPkAs1I11UwCYKE8MAReXICpAJWvvIsm\/wiPY1bwAOaKK65QO+20k3r33XfVq6++qkELXEiHHHKIOvXUUytRhKYKn6wvLvACcCMxMJWoT2GDlHEYFja5HB1xpassAJOD1bGvpgFG9i0k+90oN56PxScO5LgqRFPaf3LzmAUdTQBi1kcyM\/C6Cjri9hEX8AI6mnqGJel68AAG7qPFF19cbbTRRuq8885Ta621lnYnoYyAuJDixQsAY4IXsr7ov3GNWlxISesjqO9FXkGJw2syrSQzXwCUZPGJAzlgugmOyGJDlhnTgmNbZpD35fa9Buu4HdPCQi4nAjK2YDlcpRYA47Vci2\/0+OOPq8MPP1ydddZZar755lPbb7+9Pni33nprNXz48OIHdPTYZOHbMTBEngCYSlSn0EFa6TAslHE1diYyi2Z+HOCJAjkucGMWasRoZMExP8dYm587QRdmxOcAKwAz1J8JXsj6YmbnbTqIafIZFrd8g7fA4Pr0oEGDtAkMz4wZMxSKOS666KKVbUtNFb4dAwOG4QYSXaUWF1JlKlTIQHIYFsLGSjsRmeVjtwvk2OCG2pBlxv7ejIOhfxOAIWsMygcQiCErja9bLB+F1bzd1DMsiTvBAxgUdoSlxQ7YnTZtmk5sl+XBbSYkwrv22mvV9OnT1QorrKBOPPFENWDAAGd3TRR+XAAviBQLTBbNqfcdOQzr5X+W0UVmWbjm9459k8h1E4l6MrMym7ldyBpjAhiy0pjZetFPk4s6NvEM89GC4AEMLDCoRG0DGIAP3ETK8lx55ZW6FME\/\/vEPteCCC+pg4HHjxqlbbrmFDYABIa4YGHyOTLx4ZHPNoj31vSPyqo\/3WUcWmWXlXLb3XHlgYGWhzyl4F70DtOAhoAILjFmNmoAOzaTJV6kFwGTTp9xvDRkyRD3\/\/PM6A+9ss82m+8MtpA8\/\/FCXFcjyoM8ttthC327CA2vOT37yE3XHHXeopZZaqluXTRa+ncyO3EdigcmiOfW+I4dhvfzPMrrILAvXynknKvcLRjNjYKgatcTAlCOHInsN3gJz2mmnqTnmmEMttNBCnXR\/++23avTo0erSSy\/NxAuAlQsuuECtttpqne+vu+66av\/991ebbropGwBjlxIwywiASImByaQ+tb0kh2FtrM88sMgsM+sqeXHWDaSOm0mup+kBvKCpyT\/C4xQheACDBHb9+\/dXvXr16kIHCjpmDeRdeuml1TXXXKMGDx7c2ecmm2yidtttN7XNNts4AYz54S677KKQYC\/kZ+bBHUHPeHqcMknZ\/8fnU6ZMiYz7CZm2pLkJXUkcCut7rvKSNRaWnvnMZvg9\/1HzzDNPZ9Nhq7T5vBZcm1GjRqmRI0d2mVdWj0VwxBkTChLA3HTTTfqatO+Ttj0sMGeffXaXuJo11lhDHXbYYWrjjTd2ApgmCl8sML4a1Ix28mu+GXIyZykyC1tmYoEJWz5JswsSwJxzzjlq7733Tpq7\/h6WmEcffdRpOYnqALlk4DLafffddRMkxltllVW0W2rgwFkZGOn9JpvfJAbGS40a0UgOw0aIqcskRWbhyMwVAwP3kO06khiYcGSWNJMgAQwAQ58+fVTPnj2T5q8DcE8++eRUAOaGG27QBSJxC2nhhRdWxx57rHrttdf0tWrX01QAI7eQEtWnUQ3kMGyUuPRkRWbVyizqFhLNwgQrdAsJ31EQr3kLyZ55lVXOi+ZaU8+wJD4ECWBQgTrNs8gii6jvfve7aV7RAAZ1lgCAVl55ZZ0HxgwUNjtrovAlD0x3S1oqBQmwsRyGAQolYUois\/JkZltUaCSzGjVVmTbzwNBnuBaN8gIEZAjEIA8MZeg1r1JLHpjyZJm15yABTFZiynqviQAGvJBMvGVpRD39ymFYD9\/zjCoyy8M91Zm\/xeyFwAU+M8GKKxMv1Uii7LxRmXhNV5JZL8m00kgemHyyLONtATAeXG0qgAFpUgvJQ8ANaSKHYUMEZUxTZBYts6RaSKZFhapQu2ohwYpiXnW2q1EDtJi1kNAHrDAEVDCOnfPFTHiH75t+lbrJZ1jcqhcA47EnNlX4Uo3aQ7gNaiKHYYOE9b+ptpLM0lSjdoET+iyuGjVZVMz3kWm3A2R0ABX8nywt9Dn+jqpGTS4lqUbdvPUlAMZDZk0GMCAPCexQxJH+jSy8cC9JIjsP4QfUpJUOw4DYnmsqTZKZLwABQ0y6bJcOWUtcjIsDJ+QCIguK\/b7r3Q7ryWTddI0ftOmyAHY1agIxVJ0a\/ycLi1kDCZ8jiLdPn76d5QXiaMmlGBW\/3NQzLIlNjQMwqFe0xBJL6NT\/VT1NFX5cIK+UEqhKe4obp0mHYRqqudJlH\/RpeBLXNg3Q8BkzDQCh\/khmPoDEnEMcOLEtKGkADKwoCL4lQGNWqKb4F1hYCLDAJUQPgA9Vpd5j1Tb1wkc99K0ks02TbyCBzqaeYUn62zgAc88996gVV1xRX3vu3bt3JUCmycK3Y2BgiYFFRgBM0tII73uuB30RdBV9qKeVftT4RdBGc8kCNJLoSAtAbACTBEjM8U33jguc2BYUu41pQTEBBQEUtKfbRmZwrwlUzO\/p3xQTA3Azffo0te6ghf8HhN7oBDECYJI0qZ7vGwdgZsyYoZ599ln1yCOP6AR2uPp80EEHqQUWWKA0DjYVwNhJ7MAgciehInWRm2tpzM\/QsdCVgWk1vpJHXmUc6mlYkTR+HtrMeWQFGkm0pAEgZl+g6+0Z\/TrjTeIAiU2HCwyY80iilQCLeTWa+nTlgTHbueZpBuzCAtPW1lGl2nwIXCXxM9Tvm3qGJfGzcQAGNYtQw2jttddWK620kpp33nkV8sYg8Z0ri24SA3y+b6rw7VICBGCIZomB8ZF+OG2KOgzDoahjJlnpSjroyqbTZ\/ystNlzzwo04njgaxFx9WG6kJIASRYgFgdSqL88VjeyulBf5C6aOrW9C4AhcEPuqbJ1qqz+m3qGJfEjeACDhQIrS9++fTUtTzzxhE48Zz9HHnmkzqhbxtNk4busMLC+SBBvGZpSbp9FHYblzjJ971npKuNQTzN7n\/Gz0mbOIw\/QSKKHAl59ARO1yxoDg\/d9wEkRICWKdgKedHXazAFDMTAdwcBv6JgZDoG8TT7D4nS4EQDm73\/\/uwJAmW222SJpQdXqstxITRZ+lBtJx8BctJ8aOOzMpD2ucd8XcWiESLTQNUsqZR7qPrL3Hb8omWUFGkm0+FiRXH3Yt5Bsa4hPzEgeC0oSXfb3JC8CJfjezs6Lz3ALafyU6Z2vNz3\/CxHS5DOs0QAGk0dhR8S7\/PKXv1RwIcF91KtXr7Q6nLl9U4Vv3kKyiUcszNT2dgEwmbWi+heLOgyrn3n8iFnpKutQ9+WPz\/hZabPnkBVo+NCSxiJiW2DM\/qsEJD50oY1pbTFvIAFgIS8MHoAUO3Ed9S8AxpfT9bQL3gIDtsycOVONHz9e3XrrreqOO+5Qc845pzrvvPPUCiusUAnXmgpgwByXBQafawCz0tDS4oYqEUzEIEUdGnXS4Bpb6OrKlTIPdR\/Z+4xfpMyyAA0fOqhNGgBSJF1p5pimrSkfuPsAWjY\/d0JnVl2Kg7ErUpPbqAMAzbqenWbs0No2+QyL42XwAObLL79UkyZNUssuu6ymA\/8fPXq0OvPMMzWgmWuuuUrXlSYLPwrAgGk9TpkkAKZ07SlugCYcGlmozUNX2Yd6Ej1J4+ehLWrsNEAjaf5Zvy+DrqxzsQEYARezEKNdigDvULZe0\/riqkbd9BtIoLXJZ1ijAUx7e7t2GSGId80119T\/\/vnPf67w+QsvvKB+9atf5dX7xPebLPw4N5JW7OtnJtLftAYhbq5F8FDoiuZi3Yd6FXlgitChovoISRfNq+wU4wJw4iopYOZ8IV5QgjsUawRd173So5NNkGuTq1ATIU0+wxoNYDD5b775Rk2YMEE9+OCDauzYser5559Xyy23nFpsscXU0KFD9bVqJLUr62my8ONcSEhqJwCmLK0pvt+QDo0iqeNKF3jElbY66TLBou3GM5PdkZUFf9vuJHxm3jSiOki4hXTidoM7b0rh86ZfoRYLTJG7VQF9ffLJJ+rhhx9WF154oXrvvffU559\/rs444wy13nrrFdB79y6aDGCiLDAUA9M2fpTOysvpqXNzLZOPQleZ3C2nb5FZcXx1JQ00izu6CjkSiCFXH\/5PZQLMG0n0Of5GDIzZzudGVXFUltNTk8+wRltgkHn3mGOOUf\/+97+1H2\/JJZdUAwYMUM8995xOYrfDDjuol19+Wc0+++ylxXM0WfhxLqQe6++r2vr1EwBTzp5ReK9yGBbO0tI7FJkVw2KfgGmytFBsC9U4sssKuAo52kUdO4DPQO0+4vA0+QxrNIB57LHH1GWXXaYGDRqkgQrcRx988IFaY4011Omnn67mnnvu0vWr6cKPAzHiQipdfQobQA7DwlhZWUcis2ystmOKopIG4iq06eIxA3gR00KgxrbEmMUaKaGdOVMuCeyIpqafYVFaFPwtpI8\/\/ljNM888sUnssi0R\/7eaLHwJ4vWXc+gt5TAMXULd5ycySyczl5uILCkua4jLMgOw46om7XIjdQCcN\/QkYXEZ88I7OpEdlRYQC0w6+VXdOngAUzVDXOM1GcCAnlYDMXJohLBq\/OfAVV7gAFfa8tAVdWMrzk1EFhRbqyho15UJ2A72BVAht5Cd4I76tYEPhyvUoK3pZ1hjLTD+22B5LZsu\/CgAgxiYmXePYHcTKc\/mWp4W5e9Z6MrPw6p7EJnN4rjLumJaOOJqS1EvZnsb8MQBI\/qOQAz6s7Pvwm00ffo0te6ghfVwXJLYCYCpetUHNl7TAQzYGXWdmgo7crqJJIdGYAsoYTpc5SUWmO7gJQqA+NSWQl6XtDWXzESDFNxLt4\/o6jTNEgBmUP+Zuho1jSMupLD3EnEhecin6QAmqSYSJ\/Aih4aHQgfWRABMYALxmI4pM58kgj6Vu31qS5HlxCzE6JqunY3XrChNyezs98gC06dPXxYVqE36mn6GiQvJY1FGNeEg\/KSbSPieC5DheiAKXTkWcU2vcpaZnbHWBAkmu32sKx1J5Sbr1+LcRElipLHQzgz+Nfs2AUxUIUcOyesEwCRpS4t8zx3AIKkdF\/AiFpjmLUquhzxnXTzk2gna1eILNnytK0m1paK02y4nYAbr0hypb3IdUe4XujJt5oeJAmPNW10dM+Zwhrl4Ly4kD43kIPxWuonE9UAUujwWa2BNQpeZj\/vHxdINTn9Ujd7\/Z92+igIqaa0raeZl9m1aYMzJ2YUbzWy7Hdaajuy7ZhAvl\/gXATCBbQpVT4cLgAHfUP\/I9XCywoR+aGTVX6ErK+fqey9UmSXdCIrjGEDCHePf0DWD7Af9Rl09zmpdSZoL+qWkdWhLIMr8m7LxUuFGtLODePHZk\/ss0VnMkcsVagEw9a3\/IEbmAGA0eLn+GCeAAXjhVNgx1EMjrzILXXk5WP37IcosrTXExTW4kFwAJipY1+wjjXUlSmJmkC7lbjFdRfQegRCKe6EEdeQuwv\/RhvqjW0icrlALgKl+3Qc1YisAGImBCUrlnJMJ8TAsgmtc6QJviqStiIMfc\/K5EZQk17QxMEn9pfneBmCmNckEMXbOF1f1aYxLsTAI3AVdL3zUQ24hpRFIjW0lBsaD+dwBDFgAK4z+m0Fl6iIPDQ\/1qKyJ0FUZqwsbqAiZ5XH32IT43ghKYgDowi2ktHlZkvqN+t4cxwZgUQAGfZmFHAFuzDIDcYnsOMW\/iAUmq9YxeY8LgIE4WiEnTBGHRoiqK3SFKJWOQzIqL0lemRXh7rG55nsjKI7bafPAZJWcDd7Qj+uGELmI6EYRWV9sAIPij67K02ZQL7cr1AJgsmofk\/daBcBwqUyd99AIVW2FrrAk42MZySuzItw9LgCDz3yvQLu4npcuH0mmBW\/gFWhCxl48dh4YAjMI+AWQ6WjTrzOYlxLZuW5X+cw35DaczjCTz+JC8tA6TsKHBUZbYhy3kbjcRKpic\/VQm8KbCF2FszRzh76Hax6ZFeXucRGZ90ZQHrqimG5bspJqI0UBMPM98yYSVZ2m8cnqYs4HQbybrTQw0qKWWWFqfpHTGSYAJqUycRJ+EoABa5oeB1PG5ppSZUppLnSVwtZMnfpaRvLKrAh3TxyBWQOD89JlzsllyTKtJ\/b8CXzhc9t1R1YW8zs71sXsjzLxUj9jXnjHmd8mk5IE9BKnM0wATErF4iZ87nEwRW6uKVWl1OZCV6ns9e48jWUkr8x8LT3eky+oYV66aBpx9KGNK5jWBI+zqkzPKkNg539x5XsxrTD4HnEv6BcWGNf18ILYVls33M4wYqS4kDxUipvw4\/LBEDuabIUpanP1UI1KmwhdlbI7djBfy0gRMsvr7imDa0XQhXkluYlsEOMCPAAxZjI7l0UH\/ZALyQzYpRgYWGLwPRLZDRw4sAyW1dontzNMAEwKdeIo\/DgQ02TwArEWtbmmUJFKmgpdlbDZaxBfy0iRMsvq7vEiKGWjNHRFzdvHkoWA3Kir2knJ7ABIbLBCuWBscun20rY\/nCkAJqUu1Nm8ZS0wH3zwgTr00EPVmDFj1MSJE9Ucc8wRKQeuAGZqe7uaefeIbnQ3PZg3zeZa5+JLO7bQlZZj5bb3sYy0ssx8bmn5WrJsEJOUzI5kYxZuhDbY7iQzkBe3k7jKi+MZBnm2JIB588031S677KK23nprdfrpp7c0gGnr16\/bjaSmX6fmugkJXeUCkqy9l5kHJuucyn4vSRd9LVS+7Yge4nVcMjuzLdU+csXB2AG8iLdJoqtsvpbVvwCYsjhbQ7\/vvvuumjFjhpo2bZpaf\/31WxrAuCwwEEmTrTBcNyGhq4bNIueQrSoz31taYK+PJcvXmkNxLgRO0D9VojYtMHbiOvSPm0+L9GwXF1JOna\/y9Za0wBCDX3\/9dW8AYwoF1puhQ4dWKadSxvr0xuPU3I\/\/vVvfPdbft+Mz+ruU0cvrdMqUKWrAgAHlDVBTz0JXTYzPMSw3mT351nS14qJ9VBxdaIM\/w1Zp68a5i8ZN1e\/jj\/1Q3\/bneAeP2d+KZ09We6za1uUztLvw8alqpQGz+h4\/ZXqX7ug7fE7\/vnCrhdQeN7+r8DcXeY0aNUqNHDmyC+2TJk3KoclhvioApkUtMFBH\/DpsGz8qMqkd2jQxoLdVf\/WGucUkz4qrvGiNcbjVYltAkq4b+8a2JGuH+6YSlQ6gtP\/0f6o2jX5N1x7lhyGXEo2LoF5qJy4kH2mE1YY9gMHmiFgXekaPHq3mn39+\/d80FhiO6JUODteNpCYXd+R6IApdYW2ePrPhIDNXnIqrGrXJj7SxLTYvCVQk3VSalQfmDQXwQu4ifG67juwaSORGMmsrcZCXSy8lBsZntQbY5ttvv1Uff\/xx58z69++vevbsKQDGsMCAGZxKC3DdhISuADeYhClxkJkrngV0oRp1XNVmn9gWm31ROVyiEtqRBQWWFRPA2GUDMI5544iuU9M7NA8O8hIA07x9ItOMxQLT4UKixwYxFMgLC02TXElcNyGhK9Myr\/WlpsssygJCAAaBr1GVuInxvvlroqw2VFk6rvikmROG5uNyG9FtJEpcZ1eebrq8opRdLDC1bgPFDn7++eerESNGqJkzZ6qvv\/5azT777HqAyy67TK2++urdBuMqfNOFxMkKw3UTErqK3Qeq6C00mfmCCdsdZFtAQNe+d32ikDulqCcpK68d02KOTQAGYIfKAthXp+34GMwbdJkALDR5FcVbrmcY+xiYIhSAq\/DNxRpV5LGJVhium5DQVcRqrraPUGTmcw05ijNZYmDScjkp1oUsPdSOwAeNYyeuixrfvF6NNjYAC0VeafmX1J7rGSYAJknySimuwvcBMGBP0wJ6uW5CQpfHYg2sSQgyyxtQC5ba8SxTp7bnKnrosgQl3VwyQZjpVjLpm3f\/+3SsiysDr1n3yLx5ZKpMCPIqQ4W5nmECYDy0havw7cXKpUo1101I6PJYrIE1CUFmaZLKJbGPDv6sdMVZguKAFs2LXFnmtWnMCZYUE2TZ8S+mO4mAjB3\/gjGy0pXEt7q\/53qGCYDx0CyuwncBGLDDdSMJ5QWaEszLdRMSujwWa2BN8sgsS7yKTb6vayYt27LQ5WMJirq55AJh+MwGJgAysMLYz+17DdYAJ+r2EbXPQlda3tXRnusZJgDGQ5u4Ct+1WDlYYbhuQkKXx2INrEkWmeWJV3GRn+SaycKyLHSlsQSRBWXWNemOvC702O4kV44Xulrtuo1kB+8KgMmiBfW\/IwDGQwatBmDirDD6u8CvVWfZXD3UoPYmQlftIkg9gbQy87FSpJ1EGX1G0RVlNUpjCfLJBUNgCH\/jMfPBmPwxg3ZNQBN1eyqtvNLKoq72XM8wATAeGsVV+FEWGOR8mTSkRzfONMWNxHUTEro8FmtFTXzdO2lllsZKkYbULEnl4vq36fKxGvlYgnxywdhjAYy43Em2e4mADv6OSsCXVl5pZFBnW65nmAAYD63iKvyoxeq6Uo2bSIiNaUKVaq6bkNDlsVhLbuJzUJtTSCOzNFaKrGT6Aq+k\/k26fC08Pu3iAFwHf9q7TI0y6eI9im+xbyGZ9Y\/MsgEuGtPIK4lHIX3P9QwTAOOhZVyFHwdgYIWheBgCL2AVXanW\/x5ylAf3qm\/CdRMSuqrXJXNEnwPYnmFamflYKerlQsfoJl1prEa2JYhuEKHPOABHAAWWE7RDP66r0uiHSgbQzSRcuTY\/S2NZCoHXRcyB6xkmAMZDO7gKP25zJSsM2EO3kgi84P9wJ4X6pD00QqUj72EodBXLgTQHNY2cVhezgKRiqfTrjejKajUiQELAhSwjUQAON4vMytOzAE97l7wv5uzN2kdmrSQBMH4ybkIrATAeUmpVABNlhaFYGLAuRCtM2kPDQwWCaCJ01SeGrAd1FpkVHa+SlWtx7ibbhRRVbNEVLJsm3wvxnSw1ZH2huBdyGxGNZoVpKuhIn0XdPDL5k0VeWflb5XtczzABMB5axFX4SYs1ygoTeixMEl0eIg+yidBVr1iyuHfyyKyoeJW0XPOJ88kSA0PzSLJkEYCzXUTmTSOAEvp\/En3UzqduUx55Jc2jzu+5nmECYDy0iqvwfRerKxZGW1+2PVpzLzQrjC9dHqIPqonQVa84srh3miYzXxpdt5BswOWyyvhYssa+NlULmuofmRYp9ElXpyk415Vl13Qfoa+k4F3SrKbJy3dFcD3DBMB4aABX4fssVrLCuK5WC4DxUJ4Cm\/jIq8DhKuuqSXSlde80iTYIPMk6knTQ2yAmS80jew70f9MyY8e3dACefp06a1pvfMELXm6avHwXKdczTACMhwZwFX6axRpXrTo0K0waujzEH0wToSsYUeibMDgYk54myczHOkI0J9EV54aKs\/LA6gILjFnzCDzG\/wnA4N+bnzshkvUI9qW8MHTNOklOScDM9\/1Q23E9wwTAeGgcV+EnbULEGjMWBp+ZtZJCvFbtS5eH6INqInQFJQ6vyTRNZr5xPnF0+bihTEsWVZaOAoV0DdrlKuoANwO1LMygXXIv+QTumoJsmry8lFApxfUMEwDjoQFchZ9msUZZYLT1Zdujg4qDSUOX83\/0YAAAH\/tJREFUh\/iDaSJ0BSMK74nUJTNfC5FNiA\/4wDtxdPm6oeKy7sIdREG3ZpZdchNFgRk7uNcncFcAjLc6B9dQAIyHSATAzGKSq9hjaNeq6zo0PFQpVxOhKxf7anm5apn53CBKYoRPnE8UXWncUFFABzlfqIaRnV3XLtpItNhBu755X2xeVC2vJFkU9T3XM0wAjIeGcBV+lsVqW2LMEgPaGhNAdt4sdHmoQe1NhK7aRZB6AlXKzNd64ktEXGHGRXq2q4EDO1w3LitOUl4YF9ChqtGIb7l9r8E6xohuHGEMu4yAa2wCOGljX6ivKuXlK4ci2nE9wwTAeGgHV+FnXaxJ16rrBjJZ6fJQhVqbCF21sj\/T4FXKzNd1k4kQHWMyufPVMS+8o9YdtLCzKGIUkDLLBqAjirexrUZkPelo84YeE4DELAlgAxrTdZTm1pHNiyrllVUOWd7jeoYJgPHQBq7Cz7JYfa5VC4DxUKoMTbLIK8Mwlb\/ClS4w0pe2rDErJKw0rpssArZBCei67pWOivUua4vLDYW2RKd5q8gEHHiPgEpUtl0CM+jPVbiRSg5kodNXXln6rvMdrmeYABgPreIq\/DyLNeladZ0gJg9dHupQWxOhqzbWZx44SWZFxKzQ5HxvEGUhxrbuEF1RY9IYZhFGc54EfBDvQtYTE4yZWXbjXEdm7Atd8XYBKl+ak+Tl209o7bieYQJgPDSNq\/DzLFZXmQFiZd0J7vLQ5aEOtTURumpjfeaB42RWdMxK0f3FWXdMAEMZc11MinNrmTlfKP4FfRAdBFxMi4s5hv151rgXs0+ua4zrGSYAxmNr4ir8PIs1LjeMtr78r8xAHZaYPHR5qENtTYSu2lifeeA4mZURs+JzgygLMbalheiKogFjRAXqdtwsmqxLBVDZgFnWmTe6ZNTF566SAfic3EdkhUl7ZdrFB65rjOsZJgDGYzVzFX4RizXKElOnFaYIujzUovImQlflLM89YJTMyo5Z8Y2p8W2XFAMTnYRucmcW3a4WnXZFGXPxuZnzhUCLCVJsQVDyOrLc5AncFQtMbjWvrQMBMB6sFwATz6QoEFNXfhg56D2UOqAmXOUFFie5kJKuG5clpiyxN6Z1Z+rUdtXWNqv2EFlcbCBhvmOWByDAE\/e3HftiJrGzE9rlCdwVAFOWlpXfrwAYDx4LgEkPYOrMD8P1QBS6PBZrYE2SAAyma4IYVxxL0STljZUB6EAeGNctJFffdqAuXac2QRTRmFQ2wE5kR5aYPIG7AmCK1rDq+hMA48FrATDJTAqpXpIc9MnyCqkFV3klWWDwfVkxK3HyLSL2BjLb965POl0\/5nhmvIzpKqMbSWStodgWstrQFWof3cybsC5qDK66yPUMEwDjsVq4Cr\/oxQoQg0y8drmBqgN6i6bLQ0UqaSJ0VcLmQgfxlZlvLEreyRUVe3PD2Inq9c\/7OnPAAKiYRRRNS4tpKaEaR2YSOteVaTtYl9pQtt68PBELTJEcrLYvATAe\/BYA48Gk\/zVJCupFs7LLDfgeGv5UhdFS6ApDDmlmUbbMsgAf33wxcX2DLriQXO4vcgOZlhVXFt4OMNXe5dZRVJFG8NwEMmhXVNyLAJg0Gh1WWwEwHvIQAOPBJAeAwUft1x3d+XJVN5PKPjT8uVFsS6GrWH5W0VtZMssShEv0JsXA+PRNAAZ9UlZdk5\/mZ7gujfpGZGkBwKFijSaAiQvaNQFMx5gDndafvDItS15555X3fa5nmAAYD83gKvyyFmtcPEwVN5PKostDVUptInSVyt5SOi9DZkkAxIeQqNgb376JLurHrF8E8ELWG7L22G4ks3BjVIVpG7QQXWWBF\/Rfhrx85FF2G65nmAAYD83hKvwyF6sLxJixMMT2MtxJZdLloS6lNRG6SmNtaR2XIbMignCJYNtN5Nu3SRf6AEBBLhfbegNgg1gVJKwzxyL3URRIMQViApwy3EbmWGXIqzTlStEx1zNMAIyHEnAVftmL1QQxACqThnQUfys7qLdsujxUppQmQlcpbC2106JlVlQQrovopL7xDsW82HTFBeoiFobcSGmYbRdqLCpZXdwcipZXGnrLbMv1DBMA46E1XIVfxWL1uZmkQc2Qozwk4dekCrr8ZlJsK6GrWH5W0VsZMvMNws1Cn6tvut5McSsAEtv+cKYaOHBg5xDkSqKMuq53KPaFSgjg\/3GPCWDKdBuJBSaLpoTxjgAYDzkIgPFgUkyTqm8mlXFo5ONAMW8LXcXwscpeypCZb5yKSafvbSW7bxOImJl0kYn3xO0Gdw6B\/l2BungHbik8Zt4XF3gxg3zRvqxcL2KBqXIFlDtWSwKYb775Rp1yyinq1ltvVdOnT1dLLrmkOuqoo9QKK6zg5LYAmPxKGAdiKHdMUVaYMg6N\/BzI34PQlZ+HVfdQlsx8E+D53CiyeWL2TVYTO9PtIddOUJutNFDBGkMPWW8ojobGdmXXjZKDCVooTqbsuBexwFS9KoobryUBzDnnnKPuvPNOdfnll6v55ptPjRgxQt1www3q4YcfFgBTnG5168kFYsq4Wl3WoVEia7y6Frq82BRUo7JlFmdZyWKpMZnneh\/f4\/MxL7yj1h20cGdgLl2bNudDVha4f0yLi\/1\/09pC49dhfcHYZcurLuXk+iO8JQHM2LFj1dxzz91pcYHSrrPOOurFF19Uffv27aZjXIVf9WK1byaB0cgTAxBD1heKmcmz0KumK89c07wrdKXhVhht65SZz42iJNeSHRNDoAYAZvT+P9NMphIBJogxc7q4AItLOnbMC9oUVePIVxvqlJfvHLO043qGtSSAsRXgggsuUPfee6+67rrrIi0w5he77LKLGjp0aBY9CuqdKVOmqAEDBlQ\/p7tHKLX+vkrhb3rM\/+PfOZ7a6MoxZ59XhS4fLoXVpi6ZPfnWdIU\/w1aZ5eIhzlw0bqr+bsVF++iP6N9RbdGGvtvj5nf1e59++qmaZ555Opl94eNT1R6rtul2Zv\/0Of5OelYa0EeNnzJdN6O+kt4p+vu65FU0HaNGjVIjR47s0u2kSZOKHqb2\/loewNxyyy3qpJNOUldffXWXqHpTMlzRa52\/NkxLC9VOMi0xeVZGnXTlmXfSu0JXEofC+74omSVZSlyUx91Wsq8kR7mL0K8ZE0PzQBBvW1u\/zmEpMV3UlWlXsjozGy86IktNlTEvNt+Kkldomsj1DGMPYKCQW2+9dac+jR49Ws0\/\/\/xqxowZavjw4eq+++5TF154oVp88cUjdY6r8ENZrFEAJqs7KRS6it7EhK6iOVp+f3llliUIl6iKioGx6xKZ7c0sujZ3zDwxG5z+qI6B6QAeHTeNAF7MBHXm+3HZds12VV2XjpJ8XnmVr1HZRuB6hrEHMN9++636+OOPO6Xev39\/1aNHD3XwwQerDz\/8UJ199tlqrrnmitUKrsIPYbFSXAzdRIIg8t5KCoGubNtM\/FtCVxlcLbfPPDLLG4RrW0\/wf7KguGJLCITY7boG5k7WfQzqP1O98FGPyOy7PlytI1Fd0rzyyCup7zq\/53qGsQcwLqW56qqrFP7cfPPNqnfv3ol6xVX4ISxW28pShDUmBLoSlSpDA6ErA9NqfiWPzHyCcH3Js0GIDWAo34vpvqH8LWRZMatLI4i3T5++nVYXfGfeNLItLkk3j+p0G5k8zCMvX1nU0Y7rGdaSAGaTTTZRr776qurVq1cXXbryyivViiuu2E2\/uAo\/xMVqZu7Nao0Jka4iNi2hqwguVttHVpklpfVHNlszB4tpNUmi0GXZmXf\/+7pUeDZdV9QfuYcARhADAwsMgBDqHPlk1UU\/dDsJAIf+X7fbSABMksaE+31LApi04hAAk5Zj2dpHWWNQwTrNk\/XQSDNGHW2Frjq4nm\/MPDLzLRmQJU7GToRnx8WQ9cd1RToKiCTFudjfU+bdqq9Kx0k0j7zyaUq5b3M9wwTAeOgNV+GHvlhNa0yaLL2h0+Whcs4mQldWztX3Xh6Z+cTA+LSJs86Qa8kES6b1B5YZuHdskGTmeSHumlYVfJaU\/yUUt5FYYOpbH3lHFgDjwUEBMB5MKriJyxrjC2LyHBoFk1Fod0JXoeyspLO8MksqGZAUJxNnnbHjYjpAxxKaL\/QeWWZMiwyBF\/satM1Ql0Wmrgy7vsLOKy\/fcapux\/UMEwDjoUlchc91sQpdHkodUBOu8gKLi6LNlQcmKU7GdePI5Tqi4Fw7aBfzp7wuiHMhSw7+BohJsrC4AA31GZLbSCwwAW0GKaciAMaDYQJgPJgUUJOiDo2ASNJTEbpCk0jyfMqWWVycDGZ3254\/6TJJAjDm5yZwocKLZk4Xcg2ZHU2fPk3fQqIK0+bfcVwJ0W0kACZZj0NtIQDGQzICYDyYFFCTsg+NukgVuurifPZxy5ZZXLI6O9suqKCgXLrF5LptRJYW+2q0DxcoMNd1Kymk20ZRtJQtLx8eltGG6xkmAMZDW7gKn+tiFbo8lDqgJlzlVabVzJVczrxWTYUVTVcNuZzMG0cU22JaZlxuJlfQrkuFTGsNWWVoXqG6jcQCE9BmkHIqAmA8GCYAxoNJATXheiAKXQEpmedUipaZT1CuebMI0zTLA5jXos0YGvNzsywA3t383Akq6Yo0scOVwK4JwIXmX7S8PNWk9GZczzABMB6qw1X4XBer0OWh1AE14Sqvoi0wSVemXeDGVZsIrhwfF5JPXEscsGmCy8heBlx1kesZJgDGYyPnKnyui1Xo8lDqgJpwlVfRACbuyjSJ07R2UHAuAnbJKoPPbHdQHAjxLQlgqhPmYGcJDkjdYqfCVRe5nmECYDxWFlfhc12sQpeHUgfUhKu8igQwPlem7RtHplsIc4kK2KWAW183UZzqNNHqIjEwAW0GKaciAMaDYQJgPJgUUBOuB6LQFZCSeU6lSJnFXZm2bxwR4ME0yV0EQEO1i\/A9WUqoDpLL3eRJpm7WdPBSJOBMw7cq2nI9wwTAeGgPV+EXubl6sLGyJkJXZawuZCCu8ir6QLRjYEyQ0gEgOrLomkG8BFSQiG7W52\/ooNyoa9JRGXTRN9xPt+81WB1z60SdBwaP67p2IYpRQydcdZHrGSYAxmORcBU+18UqdHkodUBNuMqraACD\/uiKs1nVmeJNzKvVZE0hMEKAhcROCeuSrkfb2XZd9Y6adMsoSe256iLXM0wATJJGK6W4Cp\/rYhW6PJQ6oCZc5VUWgDHdQiaoIZGagMZVtyhNCQBut4yS1J6rLnI9wwTAJGm0ABgPDoXVhOsmJHSFpWc+sylaZlE3kahqNCwvBGoQ1GuXCegAP\/263USKosVOTEdXsBfp2a4GDhzow4JGtSlaXqEQLwAmFEnUMA+uwue6WIWuGhZJjiG5yqtoC0zUTSR8jmRzABd4CMSYMS5RVpc0YMYM0uUqM650cT3DxALjsfFyFT7XxSp0eSh1QE24yqtoAEOWFTvmBHExiGlBoUQCOfh\/HGhBX0kVpQncuG4XcZUZV7q4nmECYDw2cq7C57pYhS4PpQ6oCVd5lQVg0K8JYuA+ItdQlgKMtioAsFAQcNTVaK4y40oX1zNMAIzHRs5V+FwXq9DlodQBNeEqrzIADFlhuhZz7LC2kPvIdbPIjmVBW1yHhvXGLhlAlpu4vC5cZcaVLq5nmAAYj42cq\/C5Llahy0OpA2rCVV5lARgSnVkegLLwwhoTFddif266iMjlhKR3eJJKAXCVGVe6uJ5hAmA8NnKuwue6WIUuD6UOqAlXeZUNYMjiYgfwAnxEZdU1QYvpJrLdUknqwVVmXOnieoYJgElaqXKN2oNDYTXhugkJXWHpmc9sqpAZZeil5HRmHIyZeC4qw26WTLpV0OXD36LbcKVLAEzRmtKg\/rgKn+tiFboatLiUUlzlVYUFxo6JIVdQ1FVqstoAtFCNpCzawlVmXOnieoaJBcZj9XIVPtfFKnR5KHVATbjKqyoAY4IY\/NssM2D+v8NN1BHsmzf9P1eZcaWL6xkmAMZjI+cqfK6LVejyUOqAmnCVV5UAxrbEmBYYAizmzaW84ucqM650cT3DBMB4rGSuwue6WIUuD6UOqAlXeVUNYFwgJkt8i49qcJUZV7q4nmECYDxWK1fhc12sQpeHUgfUhKu86gAwJNYirS0uVeEqM650cT3DBMB4bORchc91sQpdHkodUBOu8qoTwJQtXq4y40oX1zNMAIzHSucqfK6LVejyUOqAmnCVlwCYgJTMcypcdZHrGSYAxkOxuQqf62IVujyUOqAmXOUlACYgJfOcCldd5HqGCYDxUGyuwue6WIUuD6UOqAlXeQmACUjJPKfCVRe5nmECYDwUm6vwuS5WoctDqQNqwlVeAmACUjLPqXDVRa5nmAAYD8XmKnyui1Xo8lDqgJpwlZcAmICUzHMqXHWR6xkmAMZDsbkKn+tiFbo8lDqgJlzlJQAmICXznApXXeR6hgmA8VBsrsI\/5phj1FFHHeXBgWY1EbpEXqFwQHQxFEn4zYOrvLieYS0JYL788kt14oknqn\/+85\/qq6++Ussss4w64ogj1PLLL+\/Ucq7CF7r8NrVQWom8QpGE\/zxEZv68CqGlyCsEKfjPoSUBzAknnKCee+45dcEFF6i55ppLDR8+XN1+++1q7NixAmD8dSfYlrIJBSuallpfIFZ0UXQxBA5w1cOWBDAPPPCAGjBggFpyySW1br388stq4403VhMnTlRzzDFHN33jKnyhK4StxX8OIi9\/XoXSUmQWiiT85iHy8uNTKK1aEsCYzG9vb9fupI8\/\/lhdcsklTrnssMMO6rHHHgtFZjIP4YBwQDggHBAOeHNgtdVWU1dddZV3+6Y0bGkAs\/XWW6unn35aDR48WJ177rlqoYUWaorcZJ7CAeGAcEA4IBxoaQ6wBzC4FgegQs\/o0aPV\/PPP3\/n\/\/\/73v+q6665TF154ofrXv\/6l+vXr19IKIcQLB4QDwgHhgHCgCRxgD2C+\/fZb7R6ip3\/\/\/uqWW25Rq6yyio6DoefHP\/6xOvvss9Xaa6\/dBLnJHIUDwgHhgHBAONDSHGAPYFzS\/d3vfqd69uypTj\/9dH0L6eabb1aHHnqouvfee9UiiyzS0gohxAsHhAPCAeGAcKAJHGhJAPPBBx+o4447Tj366KMKOWEWX3xxtf\/++6t11lmnCTKTOQoHhAPCAeGAcKDlOdCSAKblpS4MEA4IB4QDwgHhQMM5IACm4QKU6QsHhAPCAeGAcKAVOSAAphWlLjQLB4QDwgHhgHCg4RwQAJNCgCg3sO+++6prr71WrbzyyineDK9p2npQ4VEQPaNvvvlGnXLKKerWW29V06dP1xmXUbRyhRVWaBIZzrkifgsB52PGjInMHN0UIpFEEjXIHnzwQR1Uv9FGGykU03Nlw24KTTTPZ555Rh144IGdlwSaNv+o+b7\/\/vvq6KOPVo8\/\/rhugn0Q8YRmaoom0jplyhSte+PHj1ezzTabQuI3\/H\/eeedtIjnOOWPfwNn1+uuvs6FJAIynKN9991217bbbqk8\/\/VRdfPHFjQcwaetBebIpiGbnnHOOuvPOO9Xll1+u5ptvPjVixAh1ww03qIcffjiI+WWdxJtvvql22WUXndcIN+iiSl9k7b\/q9\/bZZx8dRA9aZs6cqXbffXeFdAbYaJv83HXXXTolw09\/+lP1wgsv6FuOXJ6ddtpJLbjggupvf\/ubQoqKvfbaSyE1BWTY5GfzzTdXP\/vZz9RBBx2kpk2bpvbcc0+d2BR18jg89913nzr22GPVf\/7zHwEwHASalgYcHFDyM888U51xxhmNBzBp60Gl5Ved7VGUc+655+60uCCZIW6Yvfjii6pv3751Ti3X2ADRM2bM0Bvs+uuv32gAAxpgEUNF+KWWWkrz5ZFHHtEWzieeeCIXn+p+GcASNXUAmvGHE4C56aab1BprrKEWWGABzeZrrrlG\/6pvMo0AYrDWYk1h38AzcuRInS+syXTROoClc4stttAWpd\/\/\/vcCYOreIKoef9SoUeqhhx7SlhcsXg4AxuShTz2oqnle5HioOo4cP8i4zOGBCbjpAAaH\/GabbaZeffVV7T7CA\/cETPcw43Mw3V955ZXsAIy5fmA1w4G43HLLqQMOOIDD0tKWwMmTJ6v99ttPbbrppmrYsGGNp2vvvffW5XI23HBDnahVXEiNF6k\/Afj1vvPOO2skDj8vNwDDvR4UfkWddNJJ6uqrr1YDBw70F3zALTkAmCeffFKhSCoqwdPz2WefaasMYmLMLNkBiyJ2apwBDCyBcEkg1gdWGA5xS59\/\/rk+6L\/++mu1\/fbba\/p69+7dVPXT88b+B\/lg\/3vrrbcEwDRamgmTh7n3+OOP162WWGIJ\/esJcS977LGH2mCDDfTnTQQwNl22WbTJ9aCiaMMGC\/81fL+oc4VkhU164mTGAcDAAoNfuC+99JKaffbZtWgQSLnWWmspgBsONcm4AhjEASJ+Ce6WU089tdFuWdeeAFctLgF89NFH2pXU1Oedd95R2223nQYviy66qF5fYoFpqjQ95g2\/PA5zPL169VK48QEAQ35RfA6lgHkbAYdNMS\/adGH+8GVzqAflog1m4IMPPlh9+OGHOpgS5SKa9rjoIho4ABgE7yJg98Ybb1TLL7+8Ju2ee+7RAbzjxo1rmric8+UIYD755BO14447qnXXXVf9+c9\/ZiEn0HTHHXfowx43kPA8\/\/zz6le\/+lWj48ygfwh36NOnj6YJtzPhpkW5HC5hEHILKeUSbKIFxkUi53pQV111lcIfWJqabgJ2yY4DgAFduGY8depUfYMFgAY\/CtZcc0028RQcAcyf\/vQnbR1DQCiXB7qH2CvsiX\/4wx\/07SpY4h977DE1evRoLmSKBYaNJHMQwgXAcK4Htckmm+jgUFjRzAcHyoorrphD+vW+ev755+sr4bAwwU9PrpfLLrtMrb766vVOLsPocEUcfvjh6v7779eBvFtuuaX+P\/0KztBlEK\/suuuu+vDDQYhfvRQfgivVtk4GMWHPScBSgRgR\/Cjo0aNH51uQHW74NflBLM+JJ56oXZqgDdZB5I7CbTIuj7iQuEhS6BAOCAeEA8IB4YBwoNEcEBdSo8UnkxcOCAeEA8IB4UBrckAATGvKXagWDggHhAPCAeFAozkgAKbR4pPJCweEA8IB4YBwoDU5IACmNeUuVAsHhAPCAeGAcKDRHBAA02jxyeSFA8IB4YBwQDjQmhwQANOacheqhQPCAeGAcEA40GgOCIBptPhk8sIB4YBwQDggHGhNDgiAaU25C9XCAVYcQGJGJI1beOGFWdElxAgHhAPRHBAAI9ohHBAOVMoBZA4+77zzdC0dFNlE+YA8DwpfovYVShIgmy\/+j6yjZ511lvrXv\/6lfvjDH+bpXt4VDggHAuWAAJhABSPTEg5w5QAAzMMPP6wuvfTSwkhcaaWVdBkCABh6ll56aXX77bcLgCmMy9KRcCAsDgiACUseMhvhAHsOAMA88sgj6pJLLimM1pVXXlkddthhAmAK46h0JBwInwMCYMKXkcxQOFA7Bz7++GP1yiuvqFVXXbWzkF97e7v6zne+01ms8I033tDzHDhwYOx8bQDz\/PPP6+rhKBKIwoe33HKLdgO98847aty4cbr68d13363++te\/qmWXXVb3DRcR+kHMy0cffaSuvvpqXSFZLDC1q4pMQDhQGQcEwFTGahlIONBcDiBAds8999SAAUAB\/\/\/pT3+qrr32WrXMMsuoffbZR7W1tWkrSN++fVMBGFTW\/t3vfqdBCdxKqAy83nrrqa222kqPib9POukkNXnyZHXBBReoL7\/8UqHiOEDLAgssoIHOWmutpU499VQBMM1VMZm5cCA1BwTApGaZvCAcaE0OjBgxQk2YMEFdfvnl6oknnlB77bWXtpDg33\/4wx\/U448\/rmabbbZE5rhcSAceeKD66quvdOAtPRMnTlRLLLGE+uyzz9TJJ5+sAQ4AE6wz+Pv666\/vbCsupES2SwPhADsOCIBhJ1IhSDhQDgdgDVlooYXUkUceqU477TRt+Rg+fLgOnp0+fbr+t88TBWBmzJihbxLRg6vRo0aNUgjQffnll9U999yjrrvuOnXcccepV199VX9HjwAYH85LG+EALw4IgOElT6FGOFAaB+Cm2WOPPdSOO+6o3Tq77bab2nDDDdUqq6yigcc666yj3nrrLXXbbbepnXfeWQfqLrXUUt1iYnwADMDM+uuvr11HACcXX3yxjoMBgDnzzDO1Feahhx7qjMcRAFOa2KVj4UCwHBAAE6xoZGLCgbA4gABaxKD85Cc\/0VYRuIxwHfrEE0\/UYKJXr156wmeffbaaNm2aGjJkiFp00UXV7LPP3oUQF4DZf\/\/9FWJh8C4eACHkh7nooovUL37xC7XffvvpIOI777xTvfTSS2rzzTdXxx9\/vNphhx1023XXXVe3gSuLHrlGHZb+yGyEA0VzQABM0RyV\/oQDTDmw++67qx\/\/+MdqySWXVBdeeKG2ugwbNky7kwBq6Bk7dqy2liDY1\/XYAAZBuwgChhsKfa6xxhr6tQMOOECNGTNGW2I222wzte+++6qddtpJIV7mpptuUojJ6d+\/vw7gfeCBB9Rqq62mhg4d2pmNVwAMU0UUsoQD\/+OAABhRBeGAcMCLAwimhXsIbpwBAwZoYIHbQnPNNVfn+3D9oA2uRY8cOVJfi+7Zs2eiBcZrAikbCYBJyTBpLhxoGAcEwDRMYDJd4UDdHIClA7eFEPtiPgiyfe655\/TVZ1hP4PrZZpttuk0XFhjEsNx8882dOWSKpgkBwMhZI6UEiuas9CccCIcDAmDCkYXMRDgQPAdwGwig5KmnntKJ57I8uEEEtxGeBRdcMHctJHsOAEf0bLDBBmqeeebJMk15RzggHAicAwJgAheQTE84EBIHkG339ddf164jeYQDwgHhQJ0cEABTJ\/dlbOGAcEA4IBwQDggHMnHg\/wFbuyEX4b3BiwAAAABJRU5ErkJggg==","height":337,"width":560}}
%---
%[output:5111c754]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7sfQmYVMW59geiQFgcNAjEBVARxIURRVBQQXBBE0QjETcwXpUYgwv4i0ETR9SruLDEi7jFOKCI4g+IGkXcAxFcGSMqN4iouLAoiBsEkP9\/a3jbmjOnu6vOOd1d3VQ9zzwz3V3rW3Wm3v7WOlu2bNkivngEPAIeAY+AR8Aj4BEoIgTqeAJTRLvlp+oR8Ah4BDwCHgGPgELAExh\/EDwCHgGPgEfAI+ARKDoEPIEpui3zE\/YIeAQ8Ah4Bj4BHwBMYfwY8Ah4Bj4BHwCPgESg6BDyBKbot8xP2CHgEPAIeAY+AR8ATGH8GPAIeAY+AR8Aj4BEoOgQ8gSm6LfMT9gh4BDwCHgGPgEfAExh\/BjwCFgi89tprctppp0mnTp1kxowZoS1N6lgMmdeqxTz3IFBnnXWW\/POf\/5S\/\/e1vctRRR0XGsbKyUiZMmCBff\/213HjjjXLKKadE7su24Y8\/\/ihXXHGFzJ49WzV9\/PHHpU2bNrbd1KqfFDaxJ+I78AjEQMATmBjg+aZuIDB37lwZNGhQjcnssMMOsscee8jAgQNl8ODBst122yUy2U8\/\/VSmTZsmLVu2VH2j3HrrrfLII4\/Iq6++ql6H1Ulk8Bx08o9\/\/EPhM2fOHNlrr73yNvfguDlYmiRxSf\/www9y4IEHyubNm+Wqq66SI488Utq1a5eL6Yb2SZxatWolQ4cOlRNOOEGaNm0ae\/zp06fLxx9\/LCeddJK0bds2dn++A49AIRDwBKYQqPsxE0WABKZRo0bSs2dP1fe6detk\/vz5snHjRvWP\/7LLLkt0TL2zY489VtauXZsiMDkbKAcdX3PNNTJ58uQUgcnBEKFd5mPcJAgMyOgRRxwhzZs3lwULFkSGBwQoCol+9NFHlQTmV7\/6lYwfPz7y+L6hR6AUEfAEphR3dRtbEwkMvkk+99xzqdU\/88wz8rvf\/a7G5QMpyW233SZvv\/22qrf\/\/vvLJZdcIj169FCvN23aJGPGjJFZs2bJ6tWrZccdd5RjjjlG\/vjHPwoIkq5i+etf\/yqHHHJIDbQhjdl9991rqZlwEUL9gLniW\/1uu+2mpEaQfqBwDccff7wcffTRMm7cOKWygOrjpptuksaNG4fu6r\/+9S\/V71tvvSX16tVT9UEOcOGiQM11zz33yIcffij169eXLl26yMiRI9W3bhCvJUuWpPr9zW9+I7\/+9a9rzB0XLyQOwGHixImq7eeffy69e\/eW\/\/7v\/5aKigp56qmn5Be\/+IV63bVrV9Ufxrvhhhvk9ddfF6hBoHL705\/+JPvss0\/ouFjjqlWrVB8vvPCCrF+\/Xvbbbz\/585\/\/rNqirFy5Uq113rx5iqBizDPPPFP+67\/+KxQbEhjsN\/bzlVdekZ\/\/\/OdqL0888cTUfoMYQCKB8bF3f\/jDH+Tkk09WBPiMM86o0fd1112nxpw6dapg\/yHFaNCggRx22GFy5ZVXptQ7wBJrv\/3229V5wlxBFLOtUR8MZ+Avf\/lLjfEffvhhtT\/YD+w5yldffaXOof4ePrvlllvknXfeUWe6ffv2cumll6ZUaWHkznRNDzzwgFo71HO77LKLkkzhLPniEcg3Ap7A5BtxP17iCKQjMLhEcdHim+\/\/\/u\/\/ysKFC9U\/\/7p168q5556rfuNyxwX75JNPqsv17rvvVoQBFxzUBbgI8I8dFzsuBJ3A4P0HH3xQrr\/+ekUwcNkeeuihsmLFihokAJdt37591cXfv39\/RQjQ7rPPPhNeiPh2f\/rpp6sLATYOuBDuv\/9+Wb58eVoJEgkO0pmBWHz33XcyevRoRSImTZqk1ouLGCoQ9L1mzRoZO3as7LrrroroPfvss+ry+fLLL+Xyyy9X7YBF0MZnzz33VBiC\/PTq1UvZg2BNHTp0kG7duinp08yZM9W8n3\/+edUHiBRI27BhwxTOIHYkmCCWwXEPPvhgpZIDwbz44otVX2iDNUGNAnx\/+9vfyosvvqgIBgggyBnqo16YXQovaZAHqF4wZ6j6oF5EnyB5JAn4HHsOkgHiAjUhVJBYF0hTkyZNFAHr3Lmzwg1nBP2C4HzwwQdqLsAVn4Eo4n30gzoHHXSQ2oPzzz8\/6xr1h+Pdd99V+4g5l5eXqz6BE850JgLzn\/\/8R+3V9ttvLxdddJFaL8gnziX2HesKEhiee5M1YW\/69eunCP6UKVMUsQeZAUa+eATyiYAnMPlE24+VEwTCCAwuK1w8+MZKg1tcIPgHjm\/KF1xwgZrL\/\/zP\/6hvyCAW+D18+HB1GUGygAsDFze+SeMfOy6ooJErvlHj4sc3e9rABOvg2yqkEbjUYVCK8sYbb8iAAQMEtg2QKLANLj\/0g8sABpuQDnXv3l1drMHCSwcSCBAQFMz\/73\/\/uzz99NNSVVUlI0aMUGsbNWqUIgG4FHGxwd4FxAKXIYgebWDCjHhRFyQJWOJihJTk3nvvTeG6YcMGdUFDXQfSVKdOHUUIIZnA2CiQcIGwYa1Yc3BcSJJgj3HAAQeoCxsF42EfMHdcuOgDUhiQC+wpCBzm3rp1a2nWrFktfHhJQwoHNQwKpF44L+j37LPPVvP+9ttv5c0335Sf\/exnsnTpUkVkQGhwNkAgQWRBLEFIIJECicD5wjmhdIj94syBAHJsnCEQXBSTNQYXQRUSsAH55HnLRGA++eQTddb23ntvRYJxdvHeN998o0hkw4YNaxAY4GqzJpAinDMU7iPPRk4ecN+pRyANAp7A+KNR9AiEGfFyUbhEcfnjHzSIAKQg+j9btu3YsaM88cQTSn1x3nnnqQsbFxq+ceOfNKQDIBdRCAykEPgmj9+QHqDgsofEB5c91Fnvvfeeuvgg1QABQYH0B5IfXOqPPfZYrX2CKgRrCSt33HGHulxhOwEVA4gY1GW4rHBx40LWLyATAgNyAgNSkDBIjoAJyAwKsYVkA0QPpO2hhx5SpAUSGZAcrJXjBAkM8QlbC9RsUItB0oJ1oey8886KOP7yl78UqN3CCkkEiF6fPn1UFZIv7DEkOph3WOE+BAkMVEaws6JUD2tCgZoKkimQSUiXODZIEMgQiskag3OJQmBwdqHCAklGAWmBpAxSOJwBFF0Cg89t1nTnnXemVEbYG+w5CC1Un754BPKJgCcw+UTbj5UTBMKMeCFlgKj81FNPVeoGlMMPP1y++OKLGgSGXh6QMuByRYG6CdIPSF5AIiCSh0oH\/7iTIjDoE3YJuAAhcYEkIai6gQQFKiDYgmA+wQJJEqQV+JYPoqMX2HJAKgQxP8gPVFSQMoDM4Bs5JDSQyNhIYECyQOLgVnzttdcqIoTfKJSwvPzyy0rSgG\/pUDVA7VZWVqYkHyCP2QgMSBdUcXqBdIWuw9hrqGlwOS9atEhVA7mhLZHejpe0frmSwAwZMkS1wZmA7VCQCOL84LIPEphly5apixptFi9erPYP5eabb1bnA2NSYhR04SaBybZGfQ3pCAwkdDgfKMAVREyXyuB8QQqG8w2sIIGBKokSSZ3AQIIVdU0gbCD9OknMyUPuO\/UIhCDgCYw\/FkWPQDobmODCqEKC5AJ\/o8CAEz9UIYFIwFYA31hRoCqAES\/sPHCB4zLQiQZF+rhk+Y03nQoJ9iOQTKCAtECCAXdsXHRhqptsBAb2O1BZUL2AfkG4IPGARAmqEVyykD5BbQD1B9QpUKNRCkUCA0IDiVAmFZIpgYGRJy40Sk6IEeaHeCawAQqOC2NT2FVAvYRLF+ot7gXIJUjQv\/\/9b0UY9t13X4Uh8QEJwZjBwkta90Lje1DrQCIBMgGcgAkkEbATwlxAevE6TIUEqRzUMboKCX2BJAZVSHoMmmxrpOF1JgIDA3DsLQokYiAtMKIGYSSB4b7jbEEahkJ1I6WAOoEB+Yy6Jk9giv7fZ1EvwBOYot4+P3kgYEpgQDBAGvDtGjYw+JZ63333KXURvh3jYjznnHPkpZdeUhc9LlpIbEBw8BnqBC94qEZ4oeISwTdZeH3oJAe2Gscdd5yy34BoH9IEGGeib1ykkKBEITA04oWhKyQKkI7A6wXqIUg6ICmBZADSI\/x8\/\/33SkoAKQzWiAsTxq+4CGn3AalNUBJEGxhTAoO+YfCKdcJ9HXYYIIAgJJgnLj3gr48LI2eSAMwFhq\/YG2AGCRKkabBFAQkDIQFhBAmEBArqIBgxBwsNabEm\/E2jU6gV4ZGEC58EFmo6SLEgtcBeUIoSJDA6GYCUCx5J77\/\/vjobkGRA\/Re0MdGD6GVaI89RJgKDz4AP9gJ4wMUbqk+QOUplQGKxr+gPki9IiyDJAV6URqUz4rVdkycw\/n9wIRHwBKaQ6PuxE0HAlMBgMEg7YAwJuxPYMeCbJwwScWGi4Bs4DG6hCgFBgM0HvqHCSDbMiBdtYP8AyQrqQq2DekESABE+3ahBekAKcPHSeyYKgaEUgm7UuJjhBQUVDC4iSGLghgsCAFsUfA51FEgFjHFRoI6BRAo2OZgLiERcAgNiBBzQN+yIfv\/73ytjWZAWrB2eKyBv+riYMyQ1sK2BpxHIJS5g7A1d3CHBgEoKlzXcrDEObGCwHqhHggXrAK4gKfD6AmGC+uzqq69O2WuAbOJzGAZj77F3kBxRJRVGYDAOJD6QrmBfQRxAJmAwDakHSroYNNnWGFxDUIWEz0FeYJQMVScwwv5D1QgiDokbCogYyCpII84ByCRsfmA4nm5+UdbkCUwi\/8J8JxER8AQmInC+mUfAI+AR8Ah4BDwChUPAE5jCYe9H9gh4BDwCHgGPgEcgIgKewEQEzjfzCHgEPAIeAY+AR6BwCHgCUzjs\/cgeAY+AR8Aj4BHwCEREwBOYiMD5Zh4Bj4BHwCPgEfAIFA4BT2AKh70f2SPgEfAIeAQ8Ah6BiAh4AmMAHONJGFT1VTwCHgGPgEfAI+AUAgjMifAFpVY8gTHYUWTjRZI3F4pLc3EBD9M5eNxMkfqpnsfMHjO08Lh53KIhYN\/K9KyZ1rOfQWFbeAJjgL9Lm+\/SXAygc6aKx81+Kzxm9ph5AhMNM49bNNxMn1HTetFmUbhWnsAYYO\/S5rs0FwPonKnicbPfCo+ZPWb+Io6GmcctGm6mz6hpvWizKFwrT2AMsHdp85H9F9l3fbFDwONmhxdqe8zsMfO4RcPM4xYNN9Nn1KU7LNpKw1t5AmOApkubj9wmyJLrix0CHjc7vFDbY2aPmcctGmYet2i4mT6jLt1h0VbqCUxk3FzafNMDG3mxJdrQ42a\/sR4ze8z8RRwNM49bNNxMn1GX7rBoK\/UEJjJuLm2+6YGNvNgSbehxs99Yj5k9Zv4ijoaZxy0abqbPqEt3WLSVegITGTeXNt\/0wEZebIk29LjZb6zHzB4zfxFHw8zjFg0302fUpTss2ko9gYmMm0ubb3pgIy+2RBt63Ow31mNmj5m\/iKNh5nGLhpvpM+rSHRZtpZ7ARMbNpc03PbCRF1uiDT1u9hvrMbPHzF\/E0TDzuEXDzfQZdekOi7ZST2Ai4+bS5pse2MiLLdGGHjf7jfWY2WPmL+JomHncouFm+oy6dIdFW6knMJFxc2nzTQ9s5MWWaEOPm\/3GeszsMfMXcTTMPG7RcDN9Rl26w6Kt1BOYyLi5tPmmBzbyYku0ocfNfmM9ZvaY+Ys4GmYet2i4mT6jLt1h0VbqCUxk3FzafNMDG3mxJdrQ42a\/sR4ze8z8RRwNM49bNNxMn1GX7rBoK93GCMzKlSuloqJCFixYoFbepUsXue6666R58+ZSWVmpwqTvsMMOKVT23XdfmTFjRihKLm2+6YFN8pCUQl8eN\/td9JjZY+Yv4miYedyi4Wb6jLp0h0Vb6TZGYM466yxp0aKFXH\/99bJ582a56KKLZOedd5YxY8bI+PHj5aOPPlJ\/mxSXNt\/0wJqsa1uq43Gz322PmT1m\/iKOhpnHLRpups+oS3dYtJVuYwRm+vTp0qNHD9lll13UyqdOnSoPP\/ywkrKMGjVKkRpIYUyKS5tvemBN1rUt1fG42e+2x8weM38RR8PM4xYNN9Nn1KU7LNpKtzECoy93y5Ytct5550nHjh1l+PDh6ufjjz+W77\/\/XlatWiX77LOPjBw5Un0eVrD5wTJ48GAZNGhQknth1Nfy5ctlt912M6rrK\/2EgMfN\/jR4zOwxQwuPm8ctGgL2rcLO2qRJk5SZRLAsXbrUfgDHW5R8Nuoff\/xRSVyqqqqUFKZ+\/fqCDQZxAQlp2rSpTJgwQaZMmSLPPfeceh0sLrFXU8bt+LnL+\/Q8bvaQe8zsMfOShGiYedyi4Wb6jLp0h0Vb6TYogVm3bp0MHTpUmjRpIrfccos0bNgwFAVIaMrLy2XcuHHSq1cvT2CSPGGO9GX6oDsyXSem4TGLtg0eN49bNATsW5meNU9g7LEtaIuvv\/5azjzzTOndu7dcdtllNeby1ltvKTUMPJJQIKXp1KmT3HXXXXL44Yd7AlPQncvN4KYPem5GL85ePWbR9s3j5nGLhoB9K9Oz5gmMPbYFbXHxxRdLs2bNQg11YQ9Tt25dufXWW5VUBl5JM2fOlNmzZ0ujRo08gSnozuVmcNMHPTejF2evHrNo++Zx87hFQ8C+lelZ8wTGHtuCtYD05aCDDpLtt99e6tSpk5oHSMu7774rq1evVsRm7ty5isjsv\/\/+cvXVV0u7du1C5+zS5pse2IKB7+jAHjf7jfGY2WOGFh43j1s0BOxbmZ41l+4w+1Wmb1HyRrxJgOXS5pse2CTWXUp9FAK3b\/55ujQ5\/CEF46YvF8gPi8elXhcDtoXArBhwyTZHj1s2hMI\/97jZ42aKmUt3mP0qPYGJhZlLm296YGMtuAQbFwI3kpaG7S8tOvLiJQnRH4JCnLXos3WnpcfNfi9MMXPpDrNfpScwsTBzafNND2ysBZdg43zipkteflg8XpGXnfp9WHSo5hOzogMnw4Q9btF20+Nmj5spZi7dYfar9AQmFmYubb7pgY214BJsnE\/cgpIXL4EpwQPlCUzim5rPZzTxyReoQ1PMXLrDkoTK28AYoOnS5pseWINlbVNV8o1bUPLibWC2neOW77NWKsh63Ox30hQzl+4w+1V6CUwszFzafNMDG2vBJdg4H7hRdQSysm7eQNn+590UkjTkLTZY84FZsWFiMl+PmwlKtet43OxxM8XMpTvMfpWewMTCzKXNNz2wsRZcgo3zgRulLBtXz1fkBcSlGCUv3P58YFaCR827UUfcVH\/e7IEzxcylO8x+lZ7AxMLMpc03PbCxFlyCjXOJWz6MdtdMu1bWL3oxtTMN9uspzQZck9OdyiVmOZ14gTv3uEXbAI+bPW6mmLl0h9mv0hOYWJi5tPmmBzbWgkuwcS5xy7XRLsgLik5Ywt5LettyiVnSc3WpP49btN3wuNnjZoqZS3eY\/So9gYmFmUubb3pgYy24BBvnGrdcGu1+XtFLWlW8UGtX0r2f1PblGrOk5ulaPx63aDvicbPHzRQzl+4w+1V6AhMLM5c23\/TAxlpwCTbOBW75MtqFtCVMXeQJjJsHNRdnzc2VJjsrj5s9nqaYuXSH2a\/SE5hYmLm0+aYHNtaCS7BxLnDLh9EuSNK3z3yhJDBBg2BPYNw8qLk4a26uNNlZedzs8TTFzKU7zH6VnsDEwsylzTc9sLEWXIKNc4VbriPtgrR8Nb2\/1NlcLg0OLEu5ZAdtYPB6zSMV0nC\/nmr3flj0ojT7TUUsQ99cYVaCx6vGkjxu0XbY42aPmylmLt1h9qv0BCYWZi5tvumBjbXgEmycJG75Uh1xG0iSNrzZRuB9BG8k3QuJ5IWEhR5LIDEoUYlMkpiV4JFKuySPW7Td9rjZ42aKmUt3mP0qPYGJhZlLm296YGMtuAQbJ4lbPlRH3AKThJBQJaFAzaRLZvg+yI4iMpZu10liVoJHyhOYhDfVnzd7QE0xc+kOs1+lJzCxMHNp800PbKwFl2DjpHHLteqIW6DHmEkXFC9IWuixpBObKPYySWNWgscqdEket2g77XGzx80UM5fuMPtVegITCzOXNt\/0wMZacAk2ThI3E6lIGIQmZCQK9EEJDCUtugRGt48xDYKXJGZR1lWsbTxu0XbO42aPmylmLt1h9qv0BCYWZi5tvumBjbXgEmycJG5RiUhU4pNtO3QbGNjHUJUE0oICGxi+j9emQfCSxCzbGkrpc49btN30uNnjZoqZS3eY\/So9gYmFmUubb3pgYy24BBu7gluuVE8kMdg6eCLRC4kGv9xSpiMw8VJyBbNiO44et2g75nGzx80UM5fuMPtVegITCzOXNt\/0wMZacAk2joNbOomLrSQmCQmMyZj0QtI9lrilunopm3FvHMxK8AgZL8njZgxVjYoeN3vcTDFz6Q6zX6UnMLEwc2nzTQ9srAWXYOM4uKUjHraExIR8ZIPedkwa7waNeNO9r48fB7Ns6yjlzz1u0XbX42aPmylmLt1h9qv0BCYWZi5tvumBjbXgEmwcFTeSDqp+tv95t1QwOcCUK5VQpi2wGZP2LuiP0pd0XkvBMaNiVoLHx2pJHjcruFKVPW72uJli5tIdZr9KT2BiYebS5pse2FgLLsHGUXADeWnY\/lL5YfE42bh6voC8oDQ5\/CH121YakgSsUcYMGvnqXkiZ3KujYJbEGou9D49btB30uNnjZoqZS3eY\/So9gYmFmUubb3pgYy24BBvb4hYkL4Bkp34f1shHlIRKyBbqqGOGeR7p79FuhvMByVl7yCBp27at7RS3+fq2Z22bB2wrAB43+5NgiplLd5j9Kj2BiYWZS5tvemBjLbgEG9viRklHvZ27KQlM0+5T1W9KX5KCKCohiTJ+mHEv1ErpyM3aNWuk7QXjogy1TbexPWvbNFja4j1u9ifBFDOX7jD7VXoCEwszlzbf9MDGWnAJNo6CG21NSF6oTkqSxERRCSW9PenUSB9eeZi0vemVpIcr+f6inLWSB8VggR43A5ACVUwxc+kOs1+lJzCxMHNp800PbKwFl2BjU9wgEUEBWVk3b2AKiVxJYDCAjVFuLrYGEpiwPEmewERD2\/SsReu9dFt53Oz31hQzl+4w+1V6AhMLM5c23\/TAxlpwCTY2xe2rWTVtPmC4mwvJCyF2QQKzdEAdFfyOhUa+nsBEexBMz1q03ku3lcfNfm9NMXPpDrNfpScwsTBzafNND2ysBZdgYxPcaLirS15guJvLEtcGJm573S5Gd7NGxN71v+jkbWAibL7JWYvQbck38bjZb7EpZi7dYfar9AQmFmYubb7pgY214BJsbIIbpC+QuMBlmiUY98U1aOJKcGj\/EjTwRaqBOnt1kwYNGqglmyaAdA2fQszH5KwVYl6uj+lxs98hU8xcusPsV7kNEpiVK1dKRUWFLFiwQK2+S5cuct1110nz5s1ly5YtMmbMGHn44Ydl\/fr10qlTJ7nxxhtlt912C0XKpc03PbBJHpJS6CsbbkHpi05kXCcxcWxowuxfGDemzs1LlRs1yQ1IDVRNnsxkfiKynbVSeJ5ysQaPmz2qppi5dIfZr3IbJDBnnXWWtGjRQq6\/\/nrZvHmzXHTRRbLzzjsr4vLggw\/KvffeK5MnT1Z1brnlFnn11Vdl5syZnsAkeboc6ivbg05JBqUvubZ9yQSNjVooKQkM56MnhYQEZssH81OkRc90jfphhr8ObXnBppLtrBVsYo4P7HGz3yBTzDyBsce2oC2mT58uPXr0kF122UXNY+rUqUriMmPGDBkwYICcdNJJApKD8sMPP0h5ebk88cQT0q5du1rzdmnzTQ9sQcF3cHAT3GjAC48j2MEUSvKik5I5T1bImQuvUIhOaj5Lfnj3RfX3UUd+Ia9+OVD9ppFxmLHx6NnLZPTsajueHns3k+57ldV4fcG6Spm7ZI3c3XSw6h\/l4A0LlZQFcWC2zBmv3mOG6z2nbVGvM0XwdXD78zolk7OW1wkVyWAeN\/uNMsXMpTvMfpXpW9TZAn1KiRcs8bzzzpOOHTvK8OHDFVm58847pVu36tDwKL1795Zhw4bJiSeeGEpggm8OHjxYBg0alHfkli9fnlbVlffJFNGA2XCr912VNFhZKet3GVzj97dtxxRklZhL\/RWVsveL96nxQTRQQDRQupYtlspW18q8xW2kxwktJWyeb3y6Xi6Y\/kXW+aPvgzdUycHrF8obDcrlkF0biOzZVdatWydNmzYVWbpAZMgUkbvOSPUFyQwkNKgnx1ySdYxtqUK2s7YtYWGzVo+bDVrVdcMwmzRpklRWVv+\/0MvSpUvtB3C8RckTmB9\/\/FFGjRolVVVVSgpTv359ad++vfr7oIMOSm3PCSecIOeee66ceuqpoQTGlc03ZdyOn7u8Ty8bbjZqm1xPnhKYvyw7STpvqlQSmLtWDZMhzWuSqYvbPCb4Acn5akyvWtPqd8dCJV0xLSAyIEivf3K0krisH3yfNKg8VzWHRGbNIxXq72a\/qVB\/0\/Xa28TURDjbWTPdj22tnsfNfsdNMfMSGHtsC94C3yCHDh0qTZo0UXYuDRs2VHOCBOb222+XI444IjVHqJuuuuoq6du3rycwBd+55CcQ9qC7RFr0FXNeOw17QUlaQFJeerlFSvqCunx\/\/dtrpcGBZdJ30FO1QEN7lAfLb06pofT+IHVheaN+JyWFAUkCWYI0hl5IMN7VSQteo5DA8HNvE1ONpumlkvwpL+4ePW72+2eKmScw9tgWtMXXX38tZ555plINXXbZZTXmMnDgQPX++eefr95fs2aNHHroofLMM8+EJq9zafNND2xBwXdw8DDc4hrA5nqZJCAYJyiBISnB+\/\/bd4QiOUhxACPcZ2f9ZIwOYlK1Rzf1OSQ6JEPok+oo\/A3pywVfV8rdO1arqEBm6gyZIr0\/f0RgvItC4oK\/QWhIWGAPg+IlMZ7AxHkm\/P82e\/RMMXPpDrNfZfoWJatCuvjii6VZs2Zy7bXX1lr9o48+KuPHj1deSK1atVIqpiVLligj37Di0uabHtgkD0kp9JVOApPrZI1xsNMNcIN3fceZAAAgAElEQVQ2MCQdikwMqJARx7VR5AUGu0FiosjFgWUpdVOYOookiZIYzhuSGLpO62qjVhXV0h0UEhj8rb8fZ+3F3NY\/o9F2z+Nmj5spZi7dYfar3MYIDKQvsG\/ZfvvtpU6dOqnV161bV9599131GgTmgQceUB5IiBGDODAtW7b0BCbJ0+VQX8EHHWoakhd67+Q66q4tHEEbFhrbkmTgN8kKbGCmnX1QLTsZjPnXuiMVgdElMDrJ4byCxIbj9enXPyWFUWRov54p6QszWVNK4wmMVyHZnnPWN72Mo\/Zfiu1MMfMEphR333BNLm2+6YE1XNo2Uy2IG9RHcJUmeUkyWWNStjW6CinbRoHAXHneOXLUkStq2bvALXqX8z9XXWDdX03vr14H+08nmdENhBknBiokkBaQGRbGick211L\/3D+j0XbY42aPmylmLt1h9qvcxiQwSQKEvlzafNMDmzQGxd6fjhsJBiPYIt4LCmxIkihxbGt0tZHpXBDfZdbvy5UE5r52Q2rYu+geTIwDs2ZatTdRmKoJ7wWNfh\/p81INbIJpB3QS4w15vQTG9NwG6\/n\/bfbImWLm0h1mv0pPYGJh5tLmmx7YWAsuwcY6bnrUXZIXm4zTJhIW2\/D+6PPtFhPlVxPeSnkYMYBdtu346NS7Uwa8Ly54WgW3Q4F7dZjtDD7LpI6ipxJUTkHywrmEkRhPXqrR8c9othMb\/rnHzR43U8xcusPsV+kJTCzMXNp80wMba8El2DiIG6Puwu6FhMZUApNNwpLt8yC8IAObvpwvW7ZbqALTwV7FlLygLxAOEA2QMKjFNq1Yn+on6H5turWMMUO7IFuMTMcpxXr+GY22qx43e9xMMXPpDrNfpScwsTBzafNND2ysBZdg4zAJjI3UJQhJJgmLiYRGl2Tgb0gvKv5yqVL\/3DzrKPWxruKBHQpsVsJiuoDsfDrkbflh8bga6Q9mXH6YdG+\/TF56udo4HUa\/KMH4L0GDXhKitU1Pk7J1D6fshEwJXgkeH6sl+WfUCq5UZY+bPW6mmLl0h9mv0hOYWJi5tPmmBzbWgkuwcZgNDJYZRbJgK2HJBCc9h6i26Vr2vixY20G+nfOFVO3RdWvclpbSY68yGb+8dSqmy7zFrVOf1WvRQPqc0iZFNFo\/ekGo6ghGuih6RN8wFRNI0sRvb5RBB9aXE5s\/o4iRqYcWVUtc87YYG8Y\/o9H+gXjc7HEzxcylO8x+lZ7AxMLMpc03PbCxFlyCjZPEzUbCkgnKeR+slc8qeqmot42PbakkJSAwjY9pqdRAb9QvV6qkTDYuIBs99i6TTZ92UGqo139WPSIIUDD1QBiBQV16HwWlOw+Wj1YSHVNJFV2qdVuYsPdK8HjVWFKSZ63UsdLX53Gz321TzFy6w+xX6QlMLMxc2nzTAxtrwSXY2EXcnrjxMun45jgV\/ZZqHEhEoEaCHQxUSZfs9pHUa9kgZR8DA11IaBjThb97tOmpJCW6uiioGqLhbhixwXu68S7mgAICBfWViaQqXYbqbS1ztYtnrRgeaY+b\/S6ZYubSHWa\/Sk9gYmHm0uabHthYCy7Bxq7hxngqhJokhlmmmeMIUXBhe0KbG9QHsQE5oaEtDH9h6\/LctLKU1AUJGVGQXVr93prrCH+nIzD4bEnP6uSN8GAioQnLsxR2RLCmME8kT2BK8IHKwZJce0ZzsMTEuzTFzKU7LEkQSjaVQJIgubT5pgc2yfWXQl8u4Ua1CsL+syAPEcgGJCzfPlPtBn3RAROFapyNq+cryQtUTFAvDf78mhrxXiCJqWx1rXR4b0rKRRqqqUN2f171xVxHQVKTIlBNB6cIC+1w8BlsYRBjxqR4CUw1Si6dNZN9c6WOx81+J0wxc+kOs19l+haewBig6dLmmx5Yg2VtM1Vgs7K61X+rRJ0mqpBcA8OLHga8KJSI0BaF2aCZWPGKfi+pemsnL1PqJNjIoEDiAsIDtRHISlXrbimvI2aVVoRlq\/QFddAnvJDwHkkTx6cNDCU7IExNuz8k3fcqM4LE28B4AmN0UNJU8v\/b7NEzxcylO8x+lZ7AxMLMpc03PbCxFlxijUFa1lTdJM06XansRArtDkxVC9VIVPOAYAQLCAvsYahSqrO5XBnqwgupe\/uPBHFedCICEtL+8acVUYEKiDFlshnwctzhh6yQzpsqUzY2puojtsea6u36vjIqRmqBjV\/OV3PY\/GkHVWVb8Eryz2i0fyAeN3vcTDFz6Q6zX6UnMLEwc2nzTQ9srAWXYOPPF1RI\/RWVxu7ASUKA9AAI3w\/Jh05SQFz4GrYuPyx6scaw+BzSFhjSQgX0+4YPq9e0daG9DBpB7QR1EaUnkMroAfE+OOxleXbWzIz2L0g1cGHjP6rx8Pe5e70vvZs8liIjnFw2EkIpl2wqF6m30Lp9ktgXoi\/\/jEZD3eNmj5spZi7dYfar9AQmFmYubb7pgY214BJqDPUR3IAhgWm8a7WnjmlMExsY0rlWw1X68RsvrdEViYxOYFABr3XVD8gN3gtTAVHtA2JTtUc3Zf9CY17YyIC8wBAXBWRk3AsH1fB24oTSJXDk55Oaz5Lue5fVMM6lqgiSFkqzgqo5Gh3X2XCO6ko37oUKLRsJssHetbr+GY22Ix43e9xMMXPpDrNfpScwsTBzafNND2ysBZdQY2ad\/m7PsTmNKpsuuF2\/OxbK2a\/8NkVMwggKbVPgMo1gdSiMlqt7DFENhM+DdjOVrUYp+5czF45QUhiqnKBiQl8MhJcugWO6LceYAya\/VetjkJDmQ29ShDAYJ4ZYYA5Ij4DM13qAO0qa9py2pYRO2k9L8c9otG31uNnjZoqZS3eY\/So9gYmFmUubb3pgYy24RBqHZZ3Opf1LWHoBhP+HlATEQf+th\/MnIdEJipLIbA39T9IRRmDQJ21d4IkEryWmIoBkBmSGsVxAIqBGokQHv4OxYoJbj\/5vuvf+UALTquKFlHu3LtUi7iA59TsvU8HwYBPT5OjjFeGhl1WpSmH8MxrtH4jHzR43U8xcusPsV+kJTCzMXNp80wMba8El0ljPOr25cbk0aNAgZwa86SQwIDC6dxElMDqJgBHupb3eUpIaFqqPlDRmqyqJkhpKaNgHPZHwGnYyUB8xoB3qwr7ml1eOrbWrmFu2sqjpKAFRCZZMEhjW1ZNU7nTKTEVeQGRQYOCLEtZ3tjm5\/rl\/RqPtkMfNHjdTzFy6w+xX6QlMLMxc2nzTAxtrwSXUmFmnvz7gedm96cqceSGls4GhAW8QUnoOkZxAxaMb8YYFoDOVmGAsXbIyaZdZyhWadii6OmfuB2szJnmEBGbEcW2tbWB0EgPiAlK14c02yvaFBSTGE5gSethiLsX\/b7MH0BQzl+4w+1V6AhMLM5c23\/TAxlpwiTQGqUChEW8uJTDpIIMNzNwla1LB5VhPl6qQrOAzSlWC0XLxWTqDW5CMYV1n14gHQw8kkI8Rx7WRlfe0StmibFo9Xxp0KlP2MphbtiSP8GAC2QD54O+wiLthGEAy9dX0\/somp9np\/ZX6iKqjUo3Q65\/RaP9APG72uJli5tIdZr9KT2BiYebS5pse2FgLLoHGJC+FJjBhahoSEeYm0r2M4ApNWxl9G5CPiJmr9feZTfr60Rcq6RINZ7cv+10NqcmqiX0VaVnz0ExFJGALxLllixGDsaMW7APVRrSBwdilnOTRP6PRTovHzR63Vc+fLM2PnqEaZgrS6dIdZr9KT2BiYebS5vuH3Gwr+TDX27mbutihQkIk3lwXGspynDC1TxhBYah\/ulbrCR4RnA4B5UBg7ms3JGWYG5TKMAYMDGqD0g28Lju9v8ICRr6U2KCPei0aKA8mpCHQC4jN5MP+ZpxKIBO2VFtFkeLkes+S7t8\/o9EQ9bjZ4\/bJOzONvCtdusPsV+kJTCzMXNp8\/5CbbSW++ZO80M03F\/Ff9Nngkp63ZK0MWtUv9TbTAgRtWnQVEe1MoJ7Zb92fU3mLSGKmn7ZBDt5cKeur1srrjSQVJZcB7mDvwsSLkMCAjEBVA+NdFNjWQPqxpd7ClHu1nksJMWQYCC9IYH5R8UKNVAK6\/QzqlqonkdkpC6\/ln9Fo6Hnc7HEDZi3\/MytrfCuX7jD7VXoCEwszlzbfP+RmW0mXZpIXxoGxcaNOZ5ibbgaQcgzZZWzK5kU31CUZoaSFryFdKV+6XhEB3Tal45vjVIoA1EN8GKh\/EFPl7RZ3qn9WIC8o7+97hpKcoB+mF8BvSGOgRlrzSIUiMo2PRUTfESqgHXImwUZmwdr2qh48loISGMxzyKFl0vaCcanl+lxHZmfPP6NmOAVredzscfMSmC1bSjOalP1ZSNvCE5gEwcxTV7oEpmn3qSoSL3XFplNI5xqdrj0u+L1eOVJJUFAQ6yUYuE5PGYC\/ob6hbQpzBkFiwnp6HUqUKEmCDQvcppETCTFgGPNFl+jAy4fqJMakwbxAWPQcS0hPgEJ7nD79+svaQwbVULvpaimd3K26s6\/qy4Ycmu5BMdbzF3G0XfO42ePmbWA8gcl6ajyByQqRcxWC5IPZqG0nGhacLpMEBiog3UgX6h28btixpxy8YaFyGwYRgPsy8yCBwIAAnLN8hvIY2nPyySmX6mnnfSgHrrhQqX5AOpguANIVkCUU2r7oJOSoI1fI5s86yKKDLpF95pxeK6kjcioFJTCQxIDAvNv5UmX3ErxQmIQSY+r4wssI0XZ9qUbAX8TRToLHzR43U8xcusPsV5m+RZ0tnsBkxdOlzTc9sFkXVeIVguqffElgRs\/+MJUGgAQF9i9Lz5ohv\/34MKXW2bj2ToENCmxPGPIfxISuzwgehwJJDL2SQEi+nfOFkuggY\/T5310mYxYcp\/IggcBAAoP+aIjLRI8ch+9ThXXRARNVfbRjVmtGC2bcl+BZCxoGk9whvkspxnOJ+oj4ZzQach43e9xMMXPpDrNfpScwsTBzafNND2ysBRdx43R2K1Fws7WBAWy45EE8YLsStIEZfvAKZUi7YG0HJU1B0sU36pcL8hWBcCCiLj4DUela9r4iNCAekOBAMqL3xy2CbQuIDMgH1Um0j6FkBl5HDGxH122MBcNdkKveA9aqcdGOxAcSlTAJDMZFDJhUvqOqtWpNXgLz00MT5awV8SOX2NQ9bvZQmmLm0h1mv0pPYGJh5tLmmx7YWAsu4saIvIvcOzDeXTdvoPobthn5wg2uznoOI909GqSBmaJps4LXuhQGxOanRIwtUzY0euoBbA+kJDDQhYSGnk54n27Y9Vo2UASIkh0mddTdummro+dSYiLIPqe0kTC1G9RIyEINkrb50w7KhgcEBiUYe6aIj1GsqefrrMWapIONPW72m2KKmUt3mP0qPYGJhZlLm296YGMtuIgbM\/u0ulC3khf8nQ\/ccLmDVNBlWo+2izm0un6c8iAieYHkBPFYUNZOXqbICKUmeE1PJUpgUA8GuIzNopMlvE\/ihL9h4wLSggKCQtdqzI0kRs\/RtKjLUVuNgEco1dI9jcbKBS0fDjV81qUvsN+h8W4pB6ezeSTycdZs5lMsdT1u9jtliplLd5j9KrdRAlNVVSWXX365NG7cWGbMqI5WiFJZWSnXXnut7LDDDqn39t133xp1dMhc2nzTA5vkISmmvnJJYDLFQNHJC8gDY7QwuSLUNVD3UAIDNREkJCiQYuiSGF0Co6uOgh5N3BdKXUicWC8scSTtXPQ9pQRGN+oNpiIInoHPrm+m5h+MrVOq6QFsngH\/jNqg9VNdj5s9bqaYuXSH2a9yGyQwTz31lNx+++3SuXNnWbRoUQ1yMn78ePnoo49kzJgxRli6tPmmB9ZoYSVYKVcqJF26QNsYZlvGRQ6CggISsRG5hv7\/5U7DWtqanDyuOoYLAtJR7cJ2iM0CFdPeL96n+vlr3ZFbs0q3qGX7gjHg5ozAd1DlkMBwOyG5oRSm\/eNPK8kO+6N9jO4pBSmQrtLC3HWPppvuvb\/GScnkXu4JTH6kfSX46OZFSlpquJneBy7dYUnugVNeSBs2bIi0tvr169dq99577wk27dFHH1U\/ugRm1KhRsnnzZiWFMSkubb7pgTVZVynWSdKIV8dHv5jDXIhJcBgHBqogEAGogkhsjr+wg1K3oC9IXRBVF5IYFF1qgtfMT6SrhXSVFOLD6K7YekJItEddehrREwkqKxS6dmNM2sHAy4mB8MJiyujJG4ExiBc8j4L5V6LGhCmlCL\/+GY32n8XjZo+bKWYu3WH2qywSCQxAjlKWLl2attmDDz5Yi8AMHz5cPv74Y\/n+++9l1apVss8++8jIkSOlY8eOof24tPmmBzYKjsXeJpPXUFzc9Bgoo2cvU5IUkBRIKhBuH7FbUBAHhp48JAKIr0KygKi6KPRUIuZhpAJeR3Sjxt\/Bokte6PXE34qo7DhYzRFEip5I9EJiX7pRL6L5UkLD1AQTv71R7lp5WS03aV0iRRIjm8pF6i20DmhXahF+4561Yn8Oo87f42aPnClmLt1h9qssEgKz\/\/77y+TJk63Wd\/bZZ8s777xjRWAmTZqkiMvgwYOladOmMmHCBJkyZYo899xz6nWwhBErtB00aJDVXJOovHz5ctltt92S6Krk+qj3XZU0WFkp63cZrH5\/2\/YnFWFs3O46Q2TIFHnj0\/Xy1+dfqhV7RU8RwJgtIAe0gWEcFkpWAD7zIb3+ydEpw1+8T1KB90Fg8JtqIbTha6YagLpIL4w\/E\/REIjmh1IWu1WwblmRySNcyueCt36u11ypzxossXSCyZ1epv7JSESUkzbQuW7Gt1S7wfuMPh6X2lHut77H1uDlqEPus5WhernfrcbPfoTDMcL\/BzjNYMn3Rtx\/ZjRZOqZCOOeYYmTNnjhUy2dqESWCCAyCWX3l5uYwbN0569eoVSmBc2XxTxm0FYglVThc5Ny5ulBIMXnWSXNj4jykj3epMzh8pjx9IVyBZUSSkQXkq0i5f4zfVQDqBoVQk6OKspyKgBAfvkaBw20Bu9AIXa9jHLNyzQQ1j4kVdeiqVVViGbLSnXQz+1qUxi\/e5OWOqgHQ2MaZxdHTplr6OoD2NbWqHQh3ruGetUPMu9LgeN\/sdMMXMS2DssY3dYvXq1fLJJ59ImG1Mt27djPoPIzBvvfWWkmI0b95c9fHjjz9Kp06d5K677pLDDz\/cExgjZN2rlOmCM33QM60KF+2zs2am8gWRCFCioRu7Lh1QR+UzGr+8teqSbs4MRqfHddElKMF4L8G28GSi4S8lLFBRkdQwAzWSQzLVQJhqiNIYfb26PQzVX7CLGXHi1xnVQumIiinhSGf4G\/a+TWqHQp3QJM5aoeZeyHE9bvbom2LmCYw9trFa3HPPPXLzzTcrY9uwYioRCSMw5513ntStW1duvfVWadiwocAraebMmTJ79mxp1KiRJzCxdq4wjXGJosBINmWTsfU13jd90DPNHmO0fvQCVYU2IiACQRWQMrBdskZ5D0EFxBJM8qiPlY5kUDXFuvftM0RJfJgbCYa5JEXMtwTjXl3Kwrnqxrnp1onxoPaCtKbTx9XeVEFXaZsdJuGos+EcJRViAcGiYbCpDYwpIbKZXy7qJnHWcjEv1\/v0uNnvkClmnsDYYxurxaGHHirXX3+9QNIS5mUU9p4+4DnnnCPz589XBGjTpk2pPuBSvWbNGuWBNHfuXEVkYHtz9dVXS7t27ULn7NLmmx7YWOAXWWMQC0TehWEtfwczIyeBGy7QOU9W1Mr8TAIB2ILqHQaOowRGN9bVYZ5+2gbBZR\/0ANJVRuxL93DSY7\/oBrqYBwPhYRwGyKObdrYtZv23W0yUgzdXWhvmov9M6QaCpIVeSCA2IDo6weFcTVVS2daW68+TOGu5nqOL\/Xvc7HfFFDOX7jD7VaZv4ZQNjD7N3r17K6NaF4pLm296YF3ALV9z4EVZb+duisSESQySwA2X7NxlL6qotvA+0smKHkyOweewftqngLhQygJbk\/vaDdka+fYKlRYAWaif\/fu10nnT\/SoWjG5oy79BlBb\/6niV1ZrB7hgED2olxpJBUDy85th6wkcQE3gWKQnRuspU4knMlVIbSmw2tBgsvZs8lpYUZttfEg6ogpoPvUntjU4s06mOioWopFt\/EmctG7al+LnHzX5XTTFz6Q6zX2UREpgbbrhBBaHr27dvkuuN1JdLm296YCMttIgbUVXRtPvUWhcllhUXNxW0bvV8FYCOGZwZw0WP1aL\/TVJAY1wEkwP5QX4i\/NZzB1G6Q9UQg9\/RJgaB65CmADYwKDqBUdKOFetTEX1pI6MHtOs76Clh4L0t2y1MJXKkJxT60G1gUB+Y7d50ZSieNkfF1EiXfRaLqsgTGJtTkL1u3Gc0+wilV8MUM5fusCR3wVkJzOLFi2XgwIHKJqVFixZK1aOXadOmJYlDxr5c2nzTA5s3cBwYKHjhhamR4uIGaQHIw9MT31fuz4z1ArLAhIokLIzZoudEYnyWRYf2rCHBadL7+FRKgRvuqw7ISIkJ0xEE+wP50YPkIR8R0xJAesPcSSBOiO573sanpLL5Y4qAMXfRyntaqSzWwfHQFrmWUH\/tG09LgwYNVJ0wlY7p1tsY6aZUSZ3KBEQLtjN6ED3TMQtZL+5ZK+TcCzm2x80efVPMXLrD7FeZvoWzBOb4449XMVkOPvjgUBuYSy+9NEkcPIHJG5rJD2SicjB90NPNjlKEK887R0WyRdFjv9CTSFcf0YCX2aKh4mGKAahy4OGD34gRAzXSgHvb1kgESRsWGAkrctSgXLqWvR\/oo40iKZTALFjbQfT0ASA7MCpe89BM1Q4ZpEFGNn05X17\/2U+SHD3FAIyBu+9dJpt2XphK5vjNy6MjBanDvE2NdFmvydHHp+yZvprev+iyXMc9a8k\/IcXRo8fNfp9MMfMExh7bWC26dOkir7zyitSrVy9WP0k0dmnzTQ9sEusupT7i4kYpAjJAU+3CoHS0hQlGxtUTJ9J+hVIVeBKBXDDrNLCmqilIjhiwjn1AdYU5LNx4gso6TZuXsrPbqC3Da6YPgISIUh\/aBlHKgfdpc6OPib9BejbtKFLWtWXKBobpA6KcCxMjXWIcJKQgMbuc\/3mUYQvSJu5ZK8ikHRjU42a\/CaaYuXSH2a8yfQtnJTCnnXaawJU6LDJukgCY9OXS5pseWJN1lXod\/SL85J2ZUrbu4UjeNLoUYfTsDxVs9CYihvT6obqH9id6VN09p20RSHCY\/ZnJHvU4LvQ0IvnRUwgEVVL0UCJxQlwY3f0Z5IixYmAz0+z0\/qn1I2eRLoFBID4SJawJMW0+vPvSVIRdkB8SDD1v0Xa7vp+SkATzItmeL1tbGdv+81XfP6PRkPa42eNmiplLd5j9KouQwCD54tSpU6Vfv37KBiZY+vTpkyQOGftyafNND2zewHF4IN02Zk3VTSl1SNQp4wJnpN1gH3oyRd0wlvX0dACQmlASw1QDutcQ2+gkKUiQ9PExNtRI6IP2MJDCoLxRv7zGWCQxsIEhcYJqC9IgvEbBWO8efKlsOqirHP7xZbLTKTOVSkf1v19PVQd2KXRfh4SkzubyjNF6TTC3sZUx6a9QdfwzGg15j5s9bqaYuXSH2a+yCAkMAK9Tp07amX\/wwQdJ4uAJTN7QzO9A9E5Cjp62bdtGHpz2GSAVa6ZV1OhHVxXp5IN\/012ZaQToWcTXzG\/ExI16ZF4QE9jIMMAc7FjoZaQIytaUBQhwB3saulXjM3o74TfaUcrybudLpeOb4wS\/IUGBzQwMlFmPtjkgP1BXDZj8lqya2Fd5YMGGBlmoUdK5r5vYJIVtBEgVPbPY96ZPO6QIU+TNy3ND00slz9NyfjiPm\/0WmWLmCYw9trFaID9RJgITq3PLxi5tvumBtVxiSVZPUgKjSwfw95BdxioSEAxix9grDFynB7ALGv2SyAB82tHQcJeeRMy1RHUTCAQKJEGU+sDodtCqfrKk57nqs38u6i+dN9+vSAs8jUBs8JsSHYwLo+F5S9bK\/G9\/UJ9DYsP6zO00rOtsJZXpsVeZIkaQ7oBQ6F5BJIjKOLhTWY1IyOmCCqY7bNgvSnPggUSDY++FVJKPZ61F+f9t9vtsiplLd5j9KtO3cMoGZujQoXL77bdbrS9KG6sBRMSlzTc9sLZrLMX6SdnAjJ69TEldQAB67N1MuRijIC4LStDbSI+AGwwWp0fI1ZM4KhITSNRIexga\/G54s40iLYNW9lNB8SBRwXwm\/Ke9PFg+WuCBBC8l\/EZZsLa9Iic3zzoqZd9CwgXJDsgJpD\/v7HRKyg0b7fA+PmdBbJ118wbK9j\/vpqQ\/QQkMiAqIB1VNCFYXJWcR9ovBCNPF8ymGc+qf0Wi75HGzx80UM5fuMPtVFgmBQULFefPmWa2ve\/fuUlVVZdXGtrJLm296YG3XWOr1o+I27\/\/nFfrVhLdqZGqGdARSCRTkHQrzStIj8\/JzXZWk4w31jR6TpbLVtepjEAkQEtjMgDygBGPFwDiXbtSozxgxkKhABcTUBFAF6eSJcWyoNsI4wULXbLyP8UFMdJdoEkS8BwmN1FsYSwJD0kPJTZwcTIU8z1HPWiHn7MLYHjf7XTDFzKU7zH6VRUJgAHKUYprYMUrfaOPS5pse2KhrLdV2UXHrd8fCVNh9YKPbu1yy20cq23TQYwiEBHYrIDpVe3RVyRchLWF7vE+7F0hrcPlDUpIiG1vTAoCM4DOQB6hu7tjYXjpvqlR2KWhPldGZC0eo9iAhIB0gJSQ1IDGM7kuywpgvlLxAQnPUkStUe73okhgSGHyezSU6qg2Ml8CU6tNntq6oz6hZ76VZyxQzl+6wJHfCKRVSVEkKJDe5LC5tvumBzSUexdi3DW66m7Ce2VlX+ejuzUE86M6spwVQ5GXHwSk7FBAQxpFB5Ns7v+ql1DhQE8G+BXWZtJGZnPdb92c1FNpBKoOcSDTaVRKQrTFgMC7cqVFIivZ77UX1HskLJTCsC8Nd2OdP\/BUAACAASURBVLwEC9RRx5xYkVIhBZNkRj0LYSQnW0LOqGPlu53NWcv33Fwez+NmvzummLl0h9mvMn0LpwhMkgtLsq+KHjvJkD4\/kaQ4YdXjzsv0wMYdpxja23zTN8WNKhKsH1mRQWBQGM+FuDC6Lo128b4eA4Z2K3pof7aFrQsICKQyIA0gGfjN\/EaQsOBzkIdDvhPl\/QPvHMagwTjMiUTjW\/StS0wQA4aFrtV6Jmo9sSTVVKhPOxq2Rf8Ym9Fx8T5IDAxuaRcTJDUm+xKW78ikXTGcS9OzVgxryeccPW72aJti5gmMPbYl0QIX2vjx46Vi7lep9YSFRs\/XYk0PbL7mU8hxbJL+meIGDyM91gkMeEEc6BXENAJ8recpoifRRQdMVNITnZAAJ0pdQIYgAYGKiASEden6zOi6sKWBdKXPKW2UAS3j0OB92rvQaJcSn2CGatSjpIUSGEW41lWq7cOaoI5iP1Ql4XXT7g\/JnpNPVoa7OmnZuHp+yi4meAa+mtVWfQaJSjqSgzZRDH0Led5MxzY9a6b9bSv1PG72O22KmScw9tiWRAtcaN0nfSRBO5t0QbdyvWjTA5vrebjSv+klaIob1Uf0tME6Ycj7WUWvGksOehPBhoRxVhDSH5c\/JCnVbtA1g8RRakMpDckLf9NrSJGLHQcryU6z31Qorye2RcJHGPvqsV5g9wLJC8jNfe2GqM9RYCAMex3M5fVGkNS0UB5PLFgLI\/ZSFQVpzX3HrZDeTR5TpImh\/Ik32sLINp0qCMQFRbed0QFMRz7DovkWm2TG9Ky58gy5Mg+Pm\/1OmGLmCYw9tiXRAhfawSMqPYFxcDdxsaHQdgJ\/p7PRyPagk7hQwgHCgAI1EgrfX\/yr45UdiW6gC6ICwkCpCwgLg83pLtUfHPZyyvWaiR07fTy\/RnJGqIFUsLo5XyhDXHonXT\/6QpnzZEVq7HTRfPVUBJg37WzoNQViBtxgFLzXK0cKIgHTyJg2N2gHCdCGbztIo72qpUQgMGESGB174s96mQhMJhuY4B7aSNpcOKbZzpoLc3RxDh43+13JhJn+jA04dk+5r6Jr5FQq9jPLTwtvA5MFZy+Byc9BjDJKUgQG5AVqonT5h3D5k8DouYUYFZcu07qtCYkDXagZP4Z2LJDYgAjBFgUGtiAkIC3oMyiRQWJGxHl5blqZUkP9te7IWlIdPU8SsdSTS8JmhlF0P7u+mRqH3kuYh27Ui7mwgLwwoFy9Xau9lII2MIzdElQbZVMhoS\/+k6VkJ53ExlTSFuUcJd3GX8TREPW42eOWCTOd+L9QebocM2yp\/QCOt3CWwKxZs0buv\/9+WbJkiaxfX53XRS9\/\/etf8wKtt4HJC8yRBrFxuw0+6HoMk29evmlrqP2WKlaKHsMFEwtmm2Z2ZxrKQsrCnEYkH3peIfQHEgTbGpCkPR84uUbsGPavg0CidMcPpynJDgK7IVhcMNw\/JDTBhI9h2bHZH4PNkXRAXYU5gUgxVgyJGOZD6QuSP4L0ZJKOhKmNsiV45Oe0qSFB0rHwEphIj0fRNfIExn7Lsklg+JwP+Z86XgJjD2\/0FoMHD1Zqm27duskOO+xQq6MbbrgheueWLemFhAsIKgXvhWQJYI6qw1gUhWqMdN\/eUSf4oPNSfHb6MiXNYKh9Jl3UPYr4NyLuQvqBixxSk+r8QtXRbBkZF3WRlwhqGSZspA0LQ+LrEp\/39z0jlVSRJIPqJ9jSoDAqLS55FAas07NYE2LazOB1UAJDg2EQFSR1hJ1O5033C72TaJh8Rb+XqlVIh02URksvU3YskEDpkXapckMuJUhoOPdMxr3pjgH3ETY1QcKDcTauvVONgcJxSKSyEaQcHb2M3fqLOBrqHjd73DJhViqBITOh4qwEpkuXLvLCCy9I48aN7Xc14RYuGUD5h\/ynzYUUhZc6I9Xa2MBQlULVDHoOhv6n5xCzPetJD+nijKB1KHStZmoB2LyA8Op2NTqJeXbWTBU8DvFZKLmhl9DifW5WBImEgOtkPZAOehHRIDdMkkPVUu8B1e7gus0KCAziv+iB9tAX0wh81+ef0vI\/s1QmahSqiCAJYtJFvE+Sgb91MoPXYfthav8S5u2H9zZ9WW03ZJtrKeF\/C2m7889oNKQ9bva4eQkMsiY6WI499lh55plnnJiZJzBObEONSeASpI0FP8gUer6WCunl0epiphePrvLRBwIZgVQmzPYF9UhWSH4gbWFWaSRMBGGBHRUD4sEWBu9136tMdhr2giI9i7oclcpZRDICEvHqlwPVRQ0pCQolL1QjMSKvnqIABAsSHBAXkCPa6TS\/bGCK7HF9IG5w6Q5Km+jFNKnHbNnu24UpOCgJWvPQTCXBQWG2aMSJAbFpuF9PkU3lqbQCYSfH1AMpnacf3i87vb\/aPxfTDfiLONr\/C4+bPW7eBsZRAnP33XfLdtttJ+eee27Bs1J7AmP\/YOW6RdDTJZvqQn\/QKTEIXvRBuxXdHgbEABILukTrnkZBA1ravMDjR3fBpjQHv0FuUE5\/Yt8ahrv0OqIqiikGICVBoZ0KCRVtYKgugncT5oYUAojUiwKpjW6Yi\/d0A97gXpEQUWoDyUtYkkVmj6ZqCXiSWCCisO6KjjF0yYuJUS72KSwTNe1xvAQm109Zfvv3BMYe72wSGEpAvReSPbaxWsAG5s0335T69etLy5Yta5GYxx9\/PFb\/No09gbFBK391M9lOBGehP+hLB9RJxVNRkpOvqwO6ocCGJEwlQ8NWqnD0SLe6rYk+Llyx5y1ZKx3fHFcrjQBIAiPq0uaF5AOvMQ94KrV\/\/Gkl0dE9j6jOwlg6kcG8IVGhOzTsWPRIvXqgOvzNNAP0hsJreDuB3OA3Cur12Ls6caXubcR\/jFTDQRKiEwtIYxg7hpgEJS\/ZCEg6CczKe1ql+vY2MPl73nI9kmsEphjiD5li5tIdluQ5ctYGZty4cVKvXr20a\/3DH\/6QJA4Z+3Jp800PbN7AKdBANi7UmOLUqVNl4MDq4GrTzj4o5QWk27zo3jsgAkzWSLsQ3QiWqpng8sPIDIPPoR99DHgmQUoBSQOC1OmSHKqmaFSMcdAe0ptgHiNKZSAlYloCqsb0+TG6L2PQ0H0ahGfB2vY1EkqSvKDOR6ferbqhlAs2PXp6gbA4PKsm9hWQvqANTFDywqi9dM2GWoht0tnAYC5hkpkCHcVaw\/pnNNpOuIZbMXi\/mWLm0h0W7XSEt3KWwOjThZYLP3Xr1k1y7cZ9ubT5pgfWeHFFWtGWwNSpU0edoSCBwWtmh+ZvvEcJCeEJ2sBA3QSSQ1JC8gGCobdlYDkd5qCEh3YrqEMpDA2DQWDQB\/tl\/JjKVqNSqQjQDlIbzDGoKuK4UAPNXbI2FS1YD7AHAqNnw\/5JMjNCHunzkiIUQWkXJCwcF7+b9D4+5WadLtVG2IWQ7ZLIlvnaxePrn9Fou+IibiaqzmirTaZVNswoRcId9r+vPaTsxpJKyJrMCuL14iyB2bRpk9x+++0yffp0Wb58uVIhtWnTRn2LPv\/88\/NiF8N\/nvMXLJBuXbsW1H2a25ztwMY7DsXV2vSfy0svvSQ9e\/aUiooKueaaa4QqJOY10o1gaYAbjK2SSW2je\/+QsARJih4hl0Hu0I5eTcxXRMkO50G1EOaKdkxNwHxHsHVhNF89mWNwJyF90Y1\/8TnIDsgKSArVQyQv\/H1P\/Rdk+CEr1D8+XdISNN6lmzP6TRdmIJ1I3nQfi+V0+mc02k65hls2ch1tlcm2yoYZ13BuxQIfByZZ6DP3BhXSlClT5PTTT5fWrVurytgsqAIuvPBCZdyby6J\/i6QEppBJHD2BqbnbNhKYXr16yYsvvpgiMLrKRpeiYAQ995CejZpkg7FXIO2A+zPUNyjB3EggQLSn0TNZM2AcJCoLN56g+oCxLdVT8Ayi8TDbUQrDcRgfRrdxoW2OjtJPrtnVkhRIafZb92dh7Bka6bINM1HrEhhIZmADg29txByqJIwNEhWMvUO7FZJ\/9I1YMXS7DtqsFMMlEfZ\/JpN9RLZLJZf\/t4q573zhZmrbYlqvkJibYFZqXxB0vJ2VwBx11FEyYcIE2X\/\/\/Wucj4ULF8rll18uzz77bE7PjW5AqKuQCpXE0ROY6AQG0jsWqpGCqgm9d0oQUGf+D\/MVSQHhoNqGIf9pfKtH7tUJkW77wpQCNMaljQptVjA++2XI\/76j7lTGtIiWizKs62xFGkAe6BGF94P2LvgMdZSn1JwvFEmCSzXeoys2JD0kKugDbWDvQnUSvZQoodHx0d3Xg27MzObNHFJoB7K3ZbuFUmdzea1ovsVwSYT9owHxevb\/3iQLPi9PSadYD1HEmzVrVqsZpID4v+ZLOAIml3ES2BUraQ5bezbMvAQmiRMToY+OHTtKVVWVbL\/99jVab968WZGa9957L0Kv5k10F05PYMxxy2dNk28WVB9xXk0GjJVPL10gq8ZOrRErhZF4Eadl3AsHpaLY0vaFJIGh\/SEpYY4k3c6FKh9KYJhfSVcp0aMJ5AJqHBALSkvgkgz7EvxG8kaSDHwOQgICQ1URJSZYmy5tYRZqtkX9qj261rBz0QmQTmCW9KyWbOpeSiA\/KLo7NYPaBXXqIDAougs1niXUA2lyIW5LUqTp2muvVVI900IVpmn9ba1etss4STxM\/nckOV6u+sqGmbeByRXyWfrt27evDB8+XPr06VOj5vPPPy\/XXXeditKbrYAAQVqDaL4zZsxIVce38DFjxsjDDz+s8ix16tRJbrzxRtltt91SdbwEJhu6hf3c9FsU1Uec7SUdRYYd3UBdpkiSCIkDvXhQR\/ciojpJJxx6skVISuCRQwNeSl\/4m1IX3dOJweWoMqLkA\/Ytx58\/UAWAw2eQWMD+ZM20ChUpF+QFpIlB79gOfdNtmpmsoZLSSQvWhddUGVH9Q0xIhPTf+Ixu1XqSRhKZoCcSVEy6VEv3EmK8mDqbykM9k\/J9kkzPTqZ5oY+rhx4vox+qjnBsUjyByYxStsvYBGOTOknsv8k4+ahjiplLjihJ4uKsCgnGuyNHjpTevXsLwEf597\/\/rYgLDDHPOOOMjDg89dRTygi4c+fOsmjRohoE5sEHH5R7771XJk+eLC1atJBbbrlFXn31VZk5c2aqT28Dk+QxS74veMXw2\/yfLu6rbFxgiBoseD9YujWvVmug4NLWjXAX9RwrC7dcpj4DMaHRLIPBIf4L4rtsWj1fttRbqGw7EKwOkpa5S9aodgzxTyNh3bsJ9ieU5uiJHzEHRLHFeCBMUNNAAoOcS1ABgUxAsgKjXf4NTyWMBdUSg\/CRIAWTSqp+DyxTqiS9gMygT7pR4zOkKaAqiTYuwaSLmB8kRcxUzRxhaI+\/dQkMvgVS3dZ86E1OeELE\/QaONY2Z08FLYBJ8tE0v47hDJiWBizuPJNqbYuYJTBJoW\/YB8f8DDzwgy5YtS3khnX322XLEEUdk7QkqJmzao48+qn50CcyAAQPkpJNOkrPOOkv188MPP0h5ebk88cQT0q5duxokBv+M73q2Sob06eS9kLKinr8KwW9R\/a76QpGYqGX40Q3kioHVyRkhQQEpAEGA0S0JA0iBHkIfUgZKFyiNIfHRDXdBMkBmvhpTrV5BIDaQFaixqKLCaxrK0s2RNjDMe4Q5odBlGsTmpZdbpPq+8rxz1Lyp5oIdC0gP6tNVW880TbKCTNpop0tgYLirRzemG3U61RFxx9pA6lCQXoAqJtrGuBC\/Jalv4F6FFPVpC29nehknO2px92aKmScwRbrPkLYECQzIyp133qkyXbNA0jNs2DA58cQTa63Upc03PbBFul1W0w5+i7a9UDjYiNPLZFivBqkcQpCGBG1fkJcI70OyELwA4VKMxIwoeqA6eibBHoZqJtSB7UzjrZc7w\/DTLRlEZ\/TsD1M4MEM03tDtaTgOI\/b2HfSUCtBHFRPtX2AATJdwxpih+ooqJzXvDQsVYUOQPBgOg6g8981JcuSGP6q\/s6Vq4IS\/eXm0UoMhHxKySNfZcI6SyEACBG8kSs2yRdDN5bfkpPq2PW9ehZT58fb\/26z+\/anK2TDzNjD2mEZucdttt8lvfvMb2X333QV\/ZyqwjzEpYQSmffv2yh37oIMOSnVxwgknKNfsU089NZTABN9EqoNBgwaZTCHROoiJo9vqJNp5EXVW77sqabCyUtbvMlj9\/rbtGDX7BQsWyPjx42X+\/PlZVwNV0hWnl8lRvcuEqQF0G5hJR1yiVCmQgIBIHLJrA5Eh1ZmnMWb9FZXy9Rcnyev\/eCkVaA6SDOYzAhlIlzOpzl7dlKqofqP35esDnheZM162zBlfg+hwASRCwQXRYPii\/T5Sfc39+xdKTUR1kC6BQVtIYWgADGmSngWbnlbBMahKwvtqnllK4w+HyaZGnRQ2G77roNb33Qdt1Py4V8E9C+uS+8vPsL98j3udbS65\/hznDD+m5ZJLLhH8+BKOgP\/fZn8ywjCbNGmSVFZWp0c5eG+RC44Xuftp8XFg7OG1awHVDuxb4GWEvzOVadOmGXWeTgID+xhdFdWjRw+56qqrBMbDweIlMEZQ57VStm\/R2b4d68a8tBfRo9\/qEhiQAqhqkFoALta0f4FbMIxt6cEEknNfuyGK9NBriLYfekwYSETQP9RRDA4HCQwkFrRrAfnBuHBtRhwYvTAyL8ZbetYMGfTKb1NJFN9ucadS21ACwzg1VGkFow6HRSHmWLo0xlQCQ6kY18XflLzY2J6wbja1VV4PnjZYtjMWnJeXwGTeqWzShELts8vjmmBm88y5vNawuTlrxJsUkGEEBtF8oTJCRF8UxG049NBD5ZlnnpG2bauzBOvFE5ikdiN\/\/QS9j4Ij9+hcLv\/3d1+kbF5gB6IHpkPkXRIaXOQgJeVL1ysCA6LByLMbv5yviAaJB9Q8lFrAa4iFHkl4DdKA\/pnskEkQn52+TElsSKh0Q9w+\/frXGJexVdAH26+vWqvUNMgEjTL\/2+pYMCBYkNbAZRtEDIUECQQH5Cao5sJ7V\/R7SdUF+eA\/Qb5Ot5MglozqS\/LStPvUGpF8syVxRN9UMXFNeA\/rShflN38n66eRPIFJFnWTyzjZEYu\/t2yY+TgwBdrjU045RaURCBaQjf79+wsMfE1KGIGBTQxEv\/BCatWqlYwaNUqWLFmi3KrDiicwJki7VScbgcFsHzmrXHnl0BiWl3gwNYD+Gu0oASEZUe9tzYuUTgKDNiAQkKjAqBeECRIYuh+jD9i+gAyBAAXVP\/icUpvfPHuUMs6FOggSmN9+fFhKbdWze6vqHEmfdlAJIlkwPtdK6RCIDe1jmBsp6PKNOdvYwOj2QevmDaxh8xJ8nSkvC4gQ1gBpF9RPlADRVsgFY2BPYJJ95rNdxsmOVhq9ZcPM28DkeZ8Ru+Wdd95RsV7+9Kc\/1RodHkkgJe+++27GmZ1zzjnKDgKB75BXqX79+qo+XKq32247RWDg4QQPpC5duqg4MC1btvQEJs\/7nYvhgsHrMAbsXeavqjna49ceL4e2Wphy8SU5yZTbiGQFv2lUqxMf2pmgj\/v2qVYngWhAAgPSgpgu8xa3Vu\/TBZlEA147rzeSVG4j2uVgLHgpIfYKvYjwWTCAHqU8I45rKyOOayNP3HiZdHxznFo0bXv09iQtVCOFuX+DZHXedL\/qwyQIXTbVns1+w3MJeNIri+QIajndTdumzyTrbisEJsk9zYR\/tss4yb1zvS9TzE0xc+lLeJLYO6dC+uc\/\/yn33HOPcolFjJZg+dnPfqbyI1H9kyQY6fpyafNND2w+cHF1jKD0BbYHg1qtkflTxstf3v2JyMD76N0eL6jIu4qQ7DhYqVpIUiidIFHQo+vqsV3wt+5xRFzoeaTnNmLQOXjnBAvSFpDcUH2EyxrzAil5ccHTNVRM+CwYeI+xZz469W55emJ1zBfGstGj+kLSQ9sdEBdkuQ56InF+6fIe5Xr\/9WjY+liFTufBuWwrBCYpt\/Ns58X\/b\/sJIVPMTTFz6Q7Ldg5sPneOwHDyiPcCFY8LxaXNNz2wLuCWyzlk+oYSTN6Ieei46QTn75XHS\/vHn64mLQ3KU6ogvNZdlZkNGu9D0qFLLXSig88hLUFQO7RBagKllkHm6a2u07ABgasxCMikw\/6mVEdM2BiW9Rr1EDyPeZl0N28GsAu6VVNNRCIEssKAdxiDOZUQVA+vSVIYCwZqI0h8WA\/u3lAB6YHt9Ne52Ot0RCWdZIaSmlzMJaxPSPrCYg+VYi6kfBiC+v9tNU+ZCeammLl0hyX5fDpFYKDqgXoHyffwd6ZSr169JHHI2JdLm296YPMGToEGSvcNheqjoMeHjpseRfWqkTXVPCAmOknRXZjptozfQbsYqmPQlnYkuuHupOaz5NCfT1WkRUXcbVkdGbf1oxek6lNSw9+KVNXvpOYDD6jxy1unbFbwnm63QhKF99k3470gFgwlS5gTEkrqpAX1QHBotIs+dLsXzJkRdGmLks0ryVQEnun4hGV\/53sMkmdiEBw2RhLzSzf3UntGTaUBpv8K0mFfariZ4hFWzxRzU8xcusPi4BJs6xSBAchTpkxRAeaYPiDdYpcuXZokDp7A5A3NZAbSvV3o5cJv4CAxway\/YQ\/6zIt6yaBVJ8nvmy5UxEBXIWGWuuqH5EURlK+rYywwVxIlN1TfgMDoUhl6\/0ANRKNUBr8jQaGEJ2h\/A8Lx+EUHScc3x6tgeXqgPHxGuxXGn6GLNsZEYfZpSmo4HlVZqAOCA\/scttF3CEQGnkCwnQG+pt5Ipv+As52GYNZw3QvJ5Btquv6Tml9Y\/6aXSra1u\/J50mQvHfalhluc\/TPF3BQzT2Di7IZh27lz58oBBxwgO+64o+DvTAVxW\/JVXNp80wObL2wKNU4w3kg2A9Mw3HA57vXKkWoJJB26TQneD77Ge5CgDNllrFITMU1A0BCWhALEB15BkLYoo92tRAD9IHIuik5EiKcu7YELNaLZwkWaNjEkHDTm5RgMVAepCpJVwluJQeugKsN64CmFaLssegoBtENfwQJ8cfFQbZRNAoP2cQhGtnOVBAHJ1fz8M5pt98LPhsctO27BGqaYuXSH2a8yfQunJDDBaW7YsCHlPYS\/Fy5cqKLQ7rrrrklikLUvlzbf9MBmXVSRV8gkgQlbWhhusKXYb92fa6hw2PaQ3aujzlKlRKIxaZdZ6v3ue5UJXHlBQmgTg\/cptaHLMg1nSQxABEbPXqbsXmj8S5KkS3Lw2eTD\/ib3bt83FatGzzoNkgH7FJAPGP4yTxLnjxg0mBdj0uiRhqE+YlJIncjAdRuv0SdyISGVQO8mj6VcmNk3JDE6mQmzPcH+KLLX\/lIVAwYlSRsV02+o6Y55EgQoXd\/+Gc38z8VLYJL752t61ly6w5JbvYizBOYf\/\/iHDB06VN58801lD\/PrX\/865QI9YcIEOfbYY5PEIWNfLm2+6YHNGzgFGsj2AkongYFaRo9SSxKiuxjjvbBItgMmvyWQ4lC1A9JDlY4ueUGwOEhgmPcIdi8kR7rdi25Hg6zVCI4HexVmkQaxYJoAEBLYuuhqH5AkGuWifxCS6pgyV9QiYrTTQd8qbsyK9SlChPqMwntP\/ReUS3Ywd1GmXEYkL7kkMHGPXVwClGl8\/4xm3h1vAxP39P7U3vSsuXSHJbd6hwkMkirCEwlRc2fOnCk333yzPPnkk\/Laa6\/JX\/7yF5U5Ol\/Fpc03PbD5wqZQ49heQOlwWzqgTg3vI51QKOLSoFxJVUhOdANeeAYNXnVSDTds2KAgdgnVN8wMDcKx57QtcsOUqdJ5U7UNTTAqLl\/js2rX54+U+gcSk+r6LRXZAEkB8QhKXkBCIFmBOzTao+A9uluTnNG9m+7VOulB3icUZOEGkUFZedXPrCLpktxAzYRC9VOmwHWFOke5GNc\/o9FQ9bjZ42aKmUt3mP0q07dwVgLToUMH+de\/\/iXbb7+9XHbZZdK8eXMZOXKkbNy4UTp16pQ1kF2SILm0+aYHNsn1u9qXDYlJh9uV552j3J1R4MocDO7GPER4H0UPXqfcpT9Yq96nxxEkG5BegHRs\/qyDIhowgt2y1W2apAjkAZ9BxQQ7FT2mC+dQ2WqUIiyQvKAe1DsgGJgTpS9BqQtJC92f8ZopEfC3buuDOYIUcS5Qe2EsSmAgvcG4j\/R5KaX+MbUb0Y19YQiMYqtCogEvz59LaQQyPRP+GY32H8PjZo+bKWYu3WH2qyxCAlNeXi5QIzVs2FAOO+wwGTt2rMBw95tvvpHu3bvL22+\/nSQOGftyafNND2zewCngQCZqJJIc4LZ705VKkqBfpLRhIYEJBp8Lev2QBPTYq5n88o9jaxni0l2ZRrXI3wOpCOqi7DTshZSaiWkDQEoQLwZRc+m9RFdotEXRSQVUUpjX4l8drz4juamuV23HQiID6YoutaE6SXfxRtA7qLUgdaHNjt4nDaRN8FbjfrkgJbGB0S8K48eYHpdMLtQupBHwBMZ0J83r+f9t5lixZibM9C94A47d02ejtoc3eosLLrhAtmzZomLCIG0AA0bddtttKtVAPoPceQITfR9z3TKbRICX6dqmp0nZuodrSQFwUcKgVg9UhznrxCXMnfpXfxwnB664UElOIIWB9ILRbav26KakKiQPkIAwvD8NeBduPEHlYCJheLvFRNnvzfGKxEACg5gsMMRlvJbqzNTVNi3MMI15kihBcoL3YROjq5BQh6onPWu2TmDgpj13yVoV6ZdSFxrzvlmvOjklSB9tW2jEm844l\/84dVWSLYHJFMTOhTQCnsAk\/2R7AmOPaSbMvprVVn1xgCF9roNO2s88mRbOqpBWrlwpo0ePltWrV8uVV14p++67r\/obiRyRagCv81U8gckX0nbjmEoEspEc3RBXd5vWpTGwayEBGH7ICnWpUzVEexfGWuk9YK2KrQJS9M5OpyhiAynLB4e9rAx+UUBM2B+zXesRganW6fRxN9IgrwAAIABJREFUdbZr9F1NYqqTQULSA3doEiV8phMYuk6DWKFgfHo94TU9npoNqFBGuv3uWKjcwum1RFUS6tIGRicuJgHkSGSA76Yv58uW7Ram8j9lk6K4nkbAExi7Z9WkticwJijVrJMJM3oKqud9SR05Zlj+YqfZryRaC2cJjL4cSGLwU7duXfnxxx\/V73wWT2Dyibb5WPiGQfVGOpKSTQKjjwaDXganC0a5ZWJGehLhMkf9RYf2VFIQEhb0R88h2rdADQQDXUhhznivRQ0pSjBwHdozLgzdppkOgAa2tNVB\/iI9uzWJDqUyIEgwBKaUh3Fg4FF10733Kw8qFJAJqLZIpFifRrxfjelVY1OyEUK9clRVkJfAmD8HpVLTExj7nfQEBszAwQLX6dtvv12mT58uy5cvV6qkNm3aKK8kJHLE63wVT2DyhbTdOCYi0mw2MPqIuDSD4fr5OSQWJAZIxAhJih7rBZIQEAZ6KZHYQA1Ebx+QnzlPVqS8j3S1FUiFHr2XbejFBOkKVE5UHzGiLm1uVK6lrd5PMLr95vmnlcQG4+u5k6DyurTXW1LZ\/DHZuPbOlPEwJT4kLSQzE7+9UWb9vtoOB8VU6sX6UYlIVOJjd4JyU9tfxNFw9bjZ4+ZVSI4SmHHjxqm0Asg83bp1a7Wz2KypU6fKhRdeKOeee679bkds4QlMROBy3AwERi+Z7CxM\/jlSlYQ+KQUBqQB50TM6Q6qBi153NYbUBQSHkXAhdWFCRXXxr1ivCIUeUA72KjTIpdEu2mNs9I3PoGqiBIY2NTpRYvoDEBrMkQkb+5zSRt7Y7hxlTIu5Mlkk5tKwY0\/pvneZwFan49vnpFy+qQJLkbb6nQS2Pgjax2Lj+YU2UVVB9EBCQkk15\/16ivdCyvEDVeDuTZ7RAk\/RueG9Ea+jBAa5bBCwbv\/9969xaBCN9\/LLL5dnn302b4fJE5i8QW01kO6qi4aZ0gmY\/nOEWgiFtjBUJSGwHKQXiAdDFZEet0UP\/c+4LSQeXFSdzeXKDoQu0zT8pb0KCAuICPvSEzRmSxbJMeh6TSkRSRI+B8HB3CCVQRA+xKS5sPEfVVOowDAeIxDjvbubPCy992+lVExRSxQJTDFLX4CT6VmLimmptvO42e+sKWYu3WH2q0zfwlkbmI4dO0pVVZWKA6OXzZs3K1Lz3nvvJYlDxr5c2nzTA5s3cAo4UNISGCwFcWFQKHkhoWBWaqp99LgvrKuIz9bs0ZDSQCIDN2rmMMLnVa27qfdBYlCQGykomcFYkDhQ+oCAebRTIdyQqOhkA+9D7UOjXkhgbrivvkrSCNXTfq+9qMaBwTFsa445sUJ5JjDGDNpR+oO+YPeCs9ag8lyJ4\/UThYxEIT0FPIa1hvbPaLTdSAo3WylhtNm60coUM5fusCSRc5bA9O3bV4YPHy59+vSpsd7nn39errvuOnnhhReSxMETmLyhmdxAJjYwHC34oAf\/yeEyhwQHcWHo3cPEibRr0fMj6W7I9Arie3CRbnZ6\/xoB7OjRhPnQcJdqI0pqdMkM2tNQGBF8UWBom4nAUO3UY68yaT70JmVvA\/URPYugxqJLNcgMPY0wfxAdGgGDPP3yyrGxCYypF5LphZOO2CR3opLpyfRSSWa00uklKdxs7bSKGUFTzDyByfMuw3gXkXd79+4tAB\/l3\/\/+tyIu11xzjZxxxhl5m5FLm296YPMGTgEHMo1LgikGcQv+k0slHdxULnOXvZjy7tGXxyi8eI\/Re\/E33JMpqcFrRrgNGs\/CyJZ2JrSpoSSGZIYeQMhcHbT\/mP9DdWh+kBKosTCO7u7MPEwIige7FcyRsV2YkkDPPI2+qAaDFEn3vILH1NpDBhlJYNIRkGwXSVi8GMwJdkIgUcAIEYy3L\/udkkCRwJgSnkIdTf+MRkM+SdxsPOWizdaNVqaYuXSHJYmcsxIY9c\/1pZfkgQcekGXLlqW8kJAf6YgjjkgSg6x9ubT5pgc266JKpIKpFCYMt+A\/Ob6m\/YjuGaQbyyJ2yrwP1qq4KSiUwBBSPZ8SJS\/oi1KOoMSF2ayZWJGGvlAdsaxf9KJs\/HK+ClTHxIuUqKBvuFSDjGBuiFPz1fT+KYNiSnZoA4M+0TbMOJjEbFHTUbL+F52krFmzrDYwmYhK2EWSjrhgXiBuIC2wFwIekELRoJdGvNmIUaGPtn9Go+1AUri5fj6ioRPeyhQzl+6wJNfvNIFJcqFx+nJp800PbJz1FlNbPViTjRdSOglMtgBtwfw8cLvGpU8vH5IeEBeochCll4QGuNIVmyQF76EupCkgGPAwgsoHf2M9mz7tIGse+YnI0GYG7fTYM8GEi2jHCMPcTxAoxoVhpF2SGD1AHtdQ55hLpO0F44yOQxhRSXeRpEs1AOxBvJoceaXynkIuKQa\/gzSKqjRMyOVv2P4ZNToytSolhZvrErpo6HgCE4aAswQGzlGPPfaYPPfcc4KovPXr15df\/OIXgizVhZLAuJBcLqmHPMmHqJB9RSUw6WxglHRiay4f5kwKRuqFemPQyn4p6QvXr6uVdKkMbV5owBuUwKA9CQylK3jvn4v6y9wP1iivJEhpdLURPJhQF2kKoDJCrBZe6oy6S1dwxorBGGHSGAbI4zpoS1NnyBTpv++6WvmjwrBjyHI911QmFR\/nykSPGBt4f3Z9MyV5occWpGG6+kjfn2xkM6lzaXsh+mc0GvIeN3vcTDFz6Uu4\/SrTt3CWwCAOzMSJE+Xwww9XxAXeRx999JG8+uqr8n\/+z\/+R3\/3ud0nikLEvbP4bo6uzEesupWEeFrmelOmBzfU8XOmfnkhNu09N5fvgZajPMSpuwVxJJAXom4QFNjAoumcSDX9VvR0HK68m2sAEbWM4Twa\/w2tIV0hcaNuCGC8oICEgQ7SBKV+6PmUvglgxTCugS37oHYU+IOWByokkSE8Yif4Z0wYeS3omas4zKFnR8Q6SvzApDN9jf8HcSvV27qZIE\/YUEhnYwECFRm8oW0IR96zaqiSinrW48yz29h43+x00xcwTGHtsY7UAcbn\/\/vtln332qdEPMlQjN9K8efNi9W\/TGJs\/b1DrUHfSfHtGmB5Ym\/UVc11cZvwmzosvLB5MVNz0bNUgEjppIWGhEa+OYzDKrk5GGN+FxAbtdEJCYsFcSVBPkQjp\/YLAwGYEtiFb6t+vDHphB6N7FAVJFY2H8RuFXkkkRsyRpGfKDqYSQN10KpwwchHcl3QEBM9S46OPF6m3MLWnIDd4v9BB7GxUVlHPWjE\/h0nM3eNmj6IpZp7A2GMbq0WvXr1CXaWRYuDggw9WMWLyVSiBCQvo5QlMvnYh\/Ti4EPVv7boag61MH\/TgKIgLk4606AQGfyNZI+xVdINfXRKj981AefjNyLvIIg2VEOK5wEYFKiKQGMaZ0aMDw8D3m5dvqnbXfnutshWhhxNj1nA8zAH2OLAjwXg0FmZkX9jj0EUcJA3jow49ovoOeqoGLGESiXQZqJkJF9IU7EsmtQ+kXfV2fV\/Z\/UDiotIbdCqTVWOn1rB\/yfeJ8xKY\/CAe9RnNz+zcHMUUM09g8rx\/f\/jDH1TOo06dOtUY+ZlnnpGnnnpKxo4dm7cZeQlM3qCONBC\/HfNyzCaBMVVBoN7TE99XBINB4igVgdRFVye92\/lSZYeiEx6qfkCAGMkXC6TLtE5O8L6u8tE9mQiKnikbBAZkqdkZ\/RV5QXRdEiome9QlPZfs9pEKqEdSwhxLZy4coYLfgTixYK58zczbtAdCnXS2LfiMhAV\/Yx9MJWSo72oAO9PzEpcsRzr8JdTI9DIuoSXHXoopZp7AxIbaroNbb71VJk2aJF27dpXdd99d2cDAnRqpBE455RRp0KBaBI4yYsQIu84ta3sbGEvA8lw9TALD+CqYCtQPiGnStm117iTTb9SoBxsM2pXonjphrtL0OgpKZvRAcbArYeJHRVo2VCniAgkIVVF6UkcSG\/ymS\/bSs2bIiOPayMdDG6oou8x6TdJBlRO3gZF8UZ+pBSBhoTExUxkwkSPJmXq2jmurxgqWoCcRyIpumBuUtJionL55ebRSH+lkqRB2ZnGPr+mlEnecUmufJG62pLNYsTTFzBOYPO\/wySefLPXq1TMaddq0aUb1olbi5uvxKCDiLoRe3vTARl1rMbYLEhKoIILG1mvXrKnhEmxq08B6kHDoeYr0oHYkIiAwJBr4\/OxXfqtek+zQhVrvS8eb3kokKunSFVBdhWzSlAjp+ZlIiqDKQgEhYmoCGgpDbQSjYiaGrNqjq3KxBonRCQzah9nA4H1iQ\/XQxtXzles3C0lM8Dc+J0kJ9kH1EZ6tQj1jcZ8B\/4xGQzBJ3Ey\/pESbqTutTDHzBMadPcv7TFzafNMDm3eQCjigrtJYdWdfZb+hX5L4+8MrD5O2N71ipdII+yeoh\/PXjXdBWHDRg+TqcVtIRkAQEN9Fd6Em2QnmWSJR0tMVYCxdxdSnX3\/Z65UjU\/Fn9M8YGI\/9ktDct88QRVKQFwmSG2bLDpuTPnYYgQliA\/KCAkkMXdvDXKvDPqP9ElymQVpYCvEFIYlj7J\/RaCgmjZvpl5Ros3WjlSlmLt1hSSLnrBt1kouM25dLm296YOOuudja80LlZayrIXQCwwvUxKg0TAzd+tELUtAEcyDR5gX2JjCY1b2MdBfqsrPbKANdqLlgDwMyoatvOEA6AkNJDevREBivOYfg\/tGNm6o1xqTBPED4mI0a7TBXPV4MEkMG8dSxCUpRUDebYXWwzfqqtSoC7y7nf56aejGqj9RZ+\/DDlLqy2J6jQs43Sdy8BKbmTrp0hyV5xrZJAlNZWSnXXnut7LDDDiks9913X5kxY0Yoti5tfpIPeZIHqdB9ZbswKYHBPON8Mxs9e5mKcItCWxM9UaMu\/aBdC3MjoQ1cpCkhImZQ41Tt0U1JaNo\/\/rR6GyQC8VlAbOAWjfgtdJPWsaYqC1IWRtBlTiPUo+QGKiREBdYjBcMeBoWGvfCCQqGdDsbHe4jD0uTo42sFtOM8wogeJTJUH+lRksPsZ2DAiwSUQQ8yEy8\/1+wd\/DMa7b9Bkri5diaiIZK9lSlmLt1h2VdlXmObJDDjx49XQfHGjBljhJRLm296YI0WVkKVgp5IlLAwlw5tYOJ+M+t3x0KVA0mPtEuVj27DohvmkrTQbRnSDaYE0HMS0SsJ2wLJDAiGnnwxaJ+i26rwb3gQMVAdiAkC0cmmcmUcC48jzh2EiAHsaASMcWmwjHnBZofjZ0rTEHaMcIGQxNAuhlKcsMslaLfEPlfe0yollQkGyWOduHua9GPgn9FoiHrc7HEzxcylO8x+lelbbJMEZtSoUcqrCVIYk+LS5pseWJN1lVId\/cLkt35cwgz0Ri+kuN\/MaAMDwkCPHpAV2JdAgkLVix6LBeqbSUdcoj6HqgbSFngigSxAOoOCcP60NaGx+OuNqnMn8XOQJsaEwXskSfgbyRdhM7Lpy\/nKrZqqIeQTYlJE3X5n4cYTlOoK5AoF84KrOMZjkDtKZvB5mGt6pvNjmuKBfaSTtKya2FeRvWxpA+JI1ZJ+DvwzGg1Rj5s9bqaYuXSH2a+yCAnM0UcfLfBEgsv0rrvumuSaZfjw4fLxxx\/L999\/L6tWrVLRfkeOHCkdO3YMHcelzTc9sIkCVgSd6Rcmp6tLDZLCjSQAZIJEhBc97UsYr4U2MIz7QukG5vftnOrkjSAXIBGQdCAlAGK1gHQx87QugUGUXaqK9AB57x58qSIzIDAIBEej2GBcnKABMr2iMB+MA3URpUQkMcCQ3kXBWDB8HSYZMc0Szr0Ks3fhe8AIqqV0JMpLYIrgATWYYlLPqMFQJVPFFDOX7rAkwXdWAnP77bfL3\/\/+d1m8eLF0795dfv3rX8txxx0nDRs2jL1+xJcBcRk8eLA0bdpUJkyYIFOmTFGJI\/E6WLD5wYK2gwYNij0X2w6WL18uu+22m22zkq\/f+MNhsqlRJ6m\/ojK11q8PeD71d1K43f3qWrlrwdqUq\/E7O52iSAgTKzLhIl4zMi5tYPQcSPgs7H0aIYNA6KkBqE4C8UFhbiMmakTfdfbqpqQV9Ru9n\/J2+u6DNuq9b9uOEc6d0XZBYDAebWEwZxRdMrS5cbl8u6R1db8rKoWYNlhZqV5\/t+dYwd\/rdxmsfmMc7IUiaW3HSL3vqtT7fJ3xIM4ZL7J0gcieXVO\/6x3es1b\/wT4wHsZC4Xh8XYiDn9RZK8TcCzmmx80e\/TDMcL\/BzjNYli5daj+A4y2cJTDE7YMPPpDHH39cnnzySVmxYoXKRn3qqaeqdAJJFWS+Li8vFySQRAqDYHGJvZoy7qSwKaZ+mNiRc44qgcmmZpr3wVrlLg3JBAPCMbEipSk60dBJhiIfDcpTBIRB6xBsjnFPIG1Av4jV8vlNa+Szil6hZEe3mUG\/e07bIrAZwbqhRsIc0A8kMbCBgS0M+n519UAl6YFNDdRGUG3pqqQr+r2kCBnJDW1YgvFcgukb0sV9CXow2ZypbHth01e+6vpnNBrSHjd73Ewxc+kOs19l+hbOExh96tOnT1d2K99884106NBBhg4dKn379rXG46233lJSjObNm6u2P\/74o0pZcNddd6ns157AWEPqRINMmalNH3QsxFQlAbJAmxddHQN7EhQSF0pL0kXYRV2om3rs3Uwa9zpOOr79W9nplJkqCvDf9nhF9nzg5JS6iRKaoNs1JDBIZwDVE\/MYwc6mzqZy5Z5M1dCrXw6UQSv7pfbrwfKbU15Ius3LxG9vlEEH1pcTmz+TUt+EeXqly0Hlkk1Kvg+nzVnL99xcHs\/jZr87pph5AmOPbSItPv30UwFxmTlzpnzyySdKQnLaaacJRGe33XabnHfeeYrI2BS0qVu3riBdAVRS8EpC\/7Nnz5ZGjRp5AmMDpkN1M+XdMX3QuRzTCxh2JTCIhRoGUguofvQkjOyPUXFJbOixRDsZJoCkYfCAyW8JDFhhyAtJCkgQJCUgJzD4RdFjvtAmhvFm0A8D1UEVBYKFCMBq\/PULFWFikkqQGBIiSmQonYEXUzDbN1\/T0yuYbdqUADp0dBKdiu1ZS3TwIu7M42a\/eaaYeQJjj22sFlAbIUXA3LlzVVCoAQMGKDsYSk3Q+WuvvaYIjG1m6tWrVytJDvoGkdl\/\/\/3l6quvlnbt2oXO2aXNNz2wscAv8sZh5MMGN5sLGHFhfvvxYamYLcFM0kGjWz0hI2BGrBgmX9TrItIuM1ujHg2GmbeIpAYkRE\/eiP5BXhBtV88oTaNc3duIaQjSbTeJDbyjaCStk5d18wZKWEDAYlT7JHnkbc5akuMWe18eN\/sdNMXMpTvMfpXpWzirQoKK6IQTTpCBAwfKoYceGrqCDRs2KAIzefLkJDHxEpicopnbztORD9MHHbNLl205nS0HbWIQfZeB4iAJQXJFJnKEyzJJh24TwyB4wRgyuuQGJEWXkiCwHVIC0CtJTxnAzNmQ1nT6uFqdhHHpMk1DXRgRM\/0A5kxpTNju0L07DNttnayE4WVz1nL7NBRX7x43+\/0yxcwTGHtsY7WAnUuTJk1i9ZFUY5c23\/TAJrX2Yusn3YVqi5uNFEbHCColuliDRDBIHD2SQBqYRBE2MQzpT0Kh90VyQ4KhB89DPf19vKaNDf5m0kbavkDthPngB8bBjBIMwkNVVBiJgV3OrN+Xq2mZkhXGseFaijWnUdSzb3vWoo5Tau08bvY7aoqZS3eY\/SqLUAKT5CLj9uXS5pse2LhrLub2YRft6lb\/bZ2fxtQORscKKiUY3YIYkDDQqBcEA7YnkKBAMqJH5lWERPNOSklgmg5ORf0NSkpIaILSm2CeI5Ajpi+gagnSobBIvkF7GNjAZIqgG5RKZYrnomcIL+bzlW3u\/hnNhlD45x43e9xMMXPpDrNfpScwsTBzafNND2ysBRd54zDpiS1uUSUwgA6X+IsLnq7hDQR7ExRIOUYc11Z5GsGwlkkUg\/mUUJcEA6QCOZIgNdGNbBmZF6QIqQGosqIbNFIF0JBXEaT65TUC6EEFRakLx1rS81wVfwaSowfLRyuX7GDAuqC7tE5i0kXUNclpVOTHLjV927NWKuuOuw6Pmz2Cppi5dIfZr9ITmFiYubT5pgc21oJLoHFQemKLm6m6JB1UcLFmGoNvXr5JkRlmWgY5gos0XaJ1Y1oa5IJYMF0B1VBIsgipDgrzIjGaLyQrsHOZt7i1qkPCBKkPiE9QWqN7IelkiUbIeA9B7JofXTPBaVjmaR0DkLcwSYsnMCXwUOV4CbbPaI6nUxTdm2Lm0h2WJLDOGvEmuci4fbm0+aYHNu6ai7l9EhKYJNcPtdKaaRUqLgxtYBg\/5r52Q1ReJEhPUEBg6EkE41t8rgeagys0SQulK1QHLepylKqLOlBhoSC5I1RIUD8x11HQ+0i3gclEYIhr0G1ax8pLYET8Mxrt6fG42eNmiplLd5j9Kr0EJhZmLm2+6YGNteAib0zpiR4XZm3T06Rs3cMpdUi+lggPpV9NeKuWoSySL8JrSY8PQ6kLovrSPgZSF6hyUGjLAumLbl+DdvRMorQGpAaqJ0hk4LHEIHWU3NBuBv2CMJHwQIUEtRHURHCT1lVIOp7pkit6GxhPYKI+W\/5\/mz1yppi5dIfZr9ITmFiYubT5pgc21oJLpLEuiVlTdVMtdUg+ltnvjoUyd8maGgazGFf3KAJ5gLqIpILJHxm7BTYpC9a2r852vTU9APpgTBiQEpAVulUjLQDVSSAvICGdN92vAuCFeRzp73106t0pkvfJOzMV6UOhnQuTNOK1\/ree0JFeSPA+YnqEbcWAF1j5ZzTak+Vxs8fNFDOX7jD7VXoCEwszlzbf9MDGWnAJNabNBpIQIiBivouevTropqwb4dJDCfODqgk2MzDyRWGmaEhRQEhQKEkhWenWsFsoWdBtVuY8WaEIEKUtOhaQCLWqeKEGPDxr6Qya4xg6\/7\/2rgXspmprj8+1UiJxOieXijh0wY\/q6CulIqSiQyJ0EZUculFULh2hXCuUUiG3081BLuWWcIhOdBGdJB25\/0j+yPV\/3qm5W9\/61t57zLVvc+015vN4Pt+355przneMtea7xxhzjHTLIZ33k2fUH9qCmzluXMxs2sPMVykEJiHMbBI+V2ETWnCWXGyDBUYTGOeRZSe8TssL\/g53jjPGBe4kNGd1a32cGteqa4pWj9Rdwu\/I4guLB9YP0qLdR043knMOyPXy8o4HacmGvZE\/nwzrSW5H+mtuVfW3aEfK\/Rw1zxL1iroMeUb9SVhwM8eNi5lNe5j5KoXAJISZTcLnKmxCC86Si50nibQ7JJHKyH5gQQDvoLkbo7puYHnRMSia5CDwFoG2OugWhAZWGFhdnKeJdFkC94ki9MFR7eUHluercaSzAev7Yk3adYXTSrppt1LJFn3o4drbVUFHd9yLLi\/gVU7AiVXYEtvJM+rnSRHXmx\/UuLpm0x7mZ53RrpFTSAw0bRI+V2EZywpVl0zhpmNgALYmH9pi4rSc6Iy4zrwssL7oDL7aVYR8MJq4OOsoud1TcAl1KjNMxd84422cRSR1iQKMiYYEeHBdOStdY17OuBhnrEus4plaubzIi\/4sW+NiMqVrQX+gBTdzCXIxs2kPM1+lWGASwswm4XMVNqEFZ+HFXNySbS3QLqRokLorSuv4FFhjQHCUheTXNRE3kftzXVfJHdcCAnPBvqfyWX60y0pn\/XVnBdbHuXFf7aLKrXiiijVKILiT2uH3aG4kjaUztkafUkJwrzvmJlvUjqtrmV5v69atafny5ZmehtzfhcBll11GkyadSKsQr3F1zaY9LN6aTD4XCwwDLZuEz1VYxrJC0wWWAl1KwGlBcAOQiiPAs8c3yufG0aeFFDk5uDpylNrt1tEWFmfBR008dNCvF4HRMS0gMO7YG02YcJ0mPV+e0VxZXpz5Y9AP2X07HJ6taiHp5HuIjXG6kmIF8n7XIkdBjGvQdE0k5IpBEwKT2UfQpvdaZpGw6+4mcuHuByZj2oVG7NkIgWFIyybhcxWWsazQdMEmi2PUJas\/pmI5osXBpCIJ26IVc9U9dQAtyAuadvk4TyLhb+4YF\/R1Bu063U+6v862qwU6vvR0urxSCXUPJNBzupe0VQV9deVqJNLTWYGdCfVeq9yJejT5iehIDbUGZBY+Xmi1Ijtn3LhR3S5axmKnJUsTFaf1RQhM5h8\/m95rmUfDnhmYyIW7H5iMaQ8S8WciBCY+RmST8LkKy1hWqLpsXdGHim4fpzbeWJtuKtLgaxcL8rB4nUZykhjtLgLp0Fl5taVGkZmTaiirjdMq4xYkPmsx4TP1Z1hBYAHRJ4ycCexg4dH5Y0CCkH8G+WMQ6KsJzrg\/9lPHuYFbxApTpRtRodUxkwI6rS9I2FeyZR91MkpbX7K5QnVQnlGb3muhehnFWayJXLi6ZjJmkGQhBIYhLZuEz1VYxrJC08VtgYlWjDAVFhi3i6X\/a0XzWES0ELxcPdrNEy0IV5cm0G4lp0UGhAHNmUgO8TiaLGkypMmRV20kXcbguiZ9lAUGDSUQYFHZOboRnVS9hCeJcSay0+vb848+ikg5yUy2KmBQntFkvdfatWtHS5cuVeI8fvw45eSccB2irV69mk477bSkinrIkCE0cuRI+uijj6hcuXKRsbdt20Z169alzp0708MPP0y9e\/emokWLUs+ePfPcPzc3l4YNG0Z16tRJ6rySNZiJXLi6ZjJmstaRjnGEwDBQtkn4XIVlLCs0XXQMzJlbe0biN7zq+cSLgfFT4NF9DYo4OgNloxEYZ44X9NFuoFhWGX0MGtfqwNvxf3k9UodJu5+cFhxFZIpWz0eqQHTqXbmdjm75M93w2LBI1l0QkDOaT1PFKPHTyyXnjHFxkhmQGLTz3jqe1boXlGc03nsNKQDQciuVoMt\/C+SOJzgEoI4ePZpq1qyZr+vRo0epYMGC8YaI+zkIzLvvvkutWrXa7htVAAAgAElEQVSiLl26RPqPHTuWxowZQy1bthQC40Ixnqzjgm5pByEwDMHYJPygvBwZsKa1C3A7\/Yv6+er8YBLOmJhYafCd1hTUCdJxILECg92LxPjIC+OOd4F1YsTmCnliY3CtM9gW7h595Fm7exBTo2soOWNlQGBgmWm348Y8U9AWGE12tAsJndaUv0yVK9DHqHVfWHN0Re3TrjwRRwRig1gYYMc9Kg1ik82uIw10UJ7RaO81Xb\/LqTgIDO\/R8Jy4RMZNYL788kvq3r07ValShbZu3Uo9evSgRx55hObPn6+G\/+yzzyK\/79+\/n5566illtQHRueOOO6hNmzb53hMgMDt37qRPP\/2UPvzww8jnzZo1o3POOYfKli0rBEYITFr3F6tvJgTGavGwJodNpVzxHapAIZouWKhdI9wEd87U\/F7J3TiTwYY\/b\/q0SCyLzpyr\/w7yAUuHM94F5EQfedYBtzpHjK6lhHvr\/2MMxL24j1dry47+O5LlgbToMTV5UWNdWoK6F5iuloT4Fb32nKM16HjB1YrAeVmtEP\/idmHBlYU1Zbv1BVgFncDEOvq\/e+iJE2TRmpvAfP3113TLLbfQoEGDqGnTpnkIi5vA9OvXj\/bu3UsgKPiJ\/q+88gpVrXoiG7Ru+ByuoVmzZtHgwYOpWrVq9MMPP9C9995LDRo0IFh6xIWUV0I27WGcdyS3j1hgGEjZJPygvBwZsKa1ixeBObxreZ5qy\/Em5I5n8XJDxRtDf+4VbwMyADcLNnlNDPC7DoB9rMMdqhYSgmpx5BmkRh+9dp5Owt9wDSw9aO5TSG6Xka5sjSBj3bBRaV3DXI7873JFWlDSAJYXnX1Xx8Q41+1ch\/67F9HhYhW0fkF5Rr3ea17WFyf+yPAMSwyXwKxfv55uuukmWrt2LRUoUCAmgbniiivo+eefj7ifnnnmGTrllFOoW7dungSmSJEitGvXLhXj8sILLyirzaFDh4TAeAjHpj0smc+zEBgGmjYJPygvRwasae3idCGBuOim3UCcyTjjWdyWGK4Fx7mhu088ufOjaCKAa+BiWr7\/BHm4YOUi9RPlBkBW3M0ZkOvO\/ou+TquMlwUG7gLkfnHqml67M\/uuu7SAcx7O00Zhq0gdlGfU672mS19Eex78EBgE+a5YsUIN6XQZuX+HJaV48eJUqFAh1ffXX39VVhi4lZxNW2CaN2+u4l0+\/vhjaty4sbLWvP3223kIDEiN+\/pLL72UXnrpJc84Hc57INV9TPYbrq6ZjJnq9SVzfCEwDDRtEj5XYRnLClUX4KaDeL3cSKYExE9Ar3uDdydy88pci2s0GSh49rpIaQFd8FHHwDgz+uIad2kBfW\/dT3\/uFQPz+R9Gq2\/ZHF1LxcmtoCsmBzcb1uiHwMzoXDNmHIzbhQQLTPv27SMZf9esWUMPPfRQJAYG5KNPnz7q93r16qkAYBCZWE0TmAceeIBuvfVW+utf\/0qTJ09Wgb3Dhw+PEBiMhTiZV199NTLcnj17qFatWurU1B\/\/+EcbxJBvDib7DVfXTMa0EpQokxICw5CWTcLnKixjWaHqonHbPf3cfOtGPIwpgUkUvFixI16WGZAdd10lZzZep1tIlyHQWXoxV\/05vkFf8NmIEzE2v1Wx9jqF5HQh4fpoJRbindxKFKcgXh+UZzTaey1aDIy2zMWSSTwCg6PO1157LS1btkxZW+D+gXUGBObpp5+mgwcP0t\/\/\/nc6cuQIDRw4kG6++Wa66KKLPC0wIDATJ05U7qP77rtPESUngdm8eTM1atSIEFtz\/fXX0+7du6lv37507NixPKTGNh0z2W+4umYypm14xJqPEBiGtGwSPldhGcsKVReNm3aBaCsMQNDxHH5JjF9rjPvEE+biPqXjJAjujQXWFG1JAeHQn2sC47TCbPjLYiVvJzmKFayJAo66\/MLPiwflS1znnFesk1uhUrLfFhuUZ9TkFBKWFs\/6gj7xCAz6ILYFhOXss8+mq6++msaPH08LFy4knEJC7pZVq1YpK8o111xDTzzxBBUuXDgqgYFFBfdcsmQJlS5dOg+BwUUgRyBCiME59dRT6brrrqNevXolPTdNMvXcZL\/h6prJmMlcS6rHEgLDQNgm4XMVlrGsUHVx4oZgXKcbyTSY1w1crHpApiDPHPhgxDrSteymPEer9VjucgM6r4u2pCBXjD4mjWv0N2e3uycWgUFszPjcuTHLL0RzH5muOdv6B+UZjfdeQ0Dvkm\/3KvHECtzNNvllej3x5OKcH1fXTMbM9PpN7i8EhoGWTcLnKixjWaHq4sQNbiS4jZzBvIlaYaJVZDYBOd4JEIzlrl6tK1br+3gVd9SBl27CES9gU59Oyvn1jjyWG21xgRsKwcVhyO1iIsegPKM2vddM8M32viZy4eqayZhBwlcIDENaNgmfq7CMZYWqi\/tEDU7QOK0wAMOvCylZFhhnjIvb2uIUls71gp\/OMgPoo60x7vwvcDG5CYzX\/fR99OkkFKFEsccy92xVHzldR3q8MB2R5jw0QXlGbXqvcXANSx8TuXB1zWTMIOEsBIYhLZuEz1VYxrJC1cWNm7bC6FwuiQTy+o2BcQvA6dLRWXB10Uane0hf56yBpGsp6fgX9HHGwGjLTdPHh0dOkcRyIeF0kk5qt+nmT1QMDHK+aKuL2\/oi7qTfpRmUZ9Sm91qoXkZxFmsiF66umYwZJFkIgWFIyybhcxWWsaxQdXHj5sxn4oyBMSkLkGwANaHQZMNZRsBtacG9YWVZV7W1ygeDvii+CGsJai258784Txppl1IsAqPXhviZcaX\/qYpComnigv\/DdaQbPncfC082PkEZLyjPqE3vtaDINh3zNJELV9dMxkzHGpN1j1ASGFRMHTp0KE2dOlUd26tevToNGDBA1dDwajYJn6uwyVKQbBnHCzdtOdFHq3VcTCLWmETw0i4dbU1xnjLCuAjORZI6fXwaf0P9IpCWpevPURl6QWai5YBxzg0upVguJN0XJ0\/Om9BMkRN3oj1tdXH\/PREMsuHaoDyjNr3XskHuyVqDiVy4umYyZrLWkY5xQklgkDsAyY0mTJhAf\/jDH+i5556jTz75hKZNmyYEJh1al4F7xHrQ3aeS\/MbCJLospwVGF2h0xrKAwLzVYSOd92azPLcCcUFm3men12ORF1zsPHYdbd66FhLKGSBYF9YXNF3awElgJJD3dxS5m0qi+pLo9bE2NR3XhHvAugaZn1StXtRbImFd\/\/79ad26deoINL4MomgjygOg4agz8rXUqVOHWrRooQo7IoldTk5OZMzp06ersgE4Vp2bm5vvXigbgBpHuPaDDz5IdPmR63EEG1j86U9\/Utl8N2zYoI5eJ6MhMzAS7Zk0E7LB1TWTMU3mmum+oSQweIBQn+P2229X+B84cIBq1KhBM2fOpPPPPz+fTGwSPldhM61Ytt3fhMDESpGfynU587joAF3cb+3\/dFOp\/Z1FEpHPBf0RbAsLTPutvSOVpPWx6XgJyeJ9vnFMNypRsqTawLQFBkRFExqNRViKNHJl7+Wu1KQ4ky5K9\/yjvdf8JCesX78+3X333XTbbbcpUjJ79mxFYJDxtmTJkvkIzJYtW1TOFhAa3e655x76\/PPPVTFHN4FBjhhUm0ZumAULFiSVwOC+999\/vyotgL3g8OHDKsleog2WfpQtwJdjk2ay33D3A5MxTeaa6b6hJDAgK6iFgW8FuuHBQIrrJk2aeBIY9x+R9RE1PtLdkF0ymqsr3XMJ0v1i4Xb6F\/Xp6Kk1qOD+1ZEl\/XTRgrQvb8wne+nlFXsjR6VhhYEbqdOZ64nOu5SOfziCcipeRtRpkpob+j+4r3HEbaRPDv3feUOp1tknqc8xnruNaX4W6\/Pj3c87cT8iOr7hRP2onOu6En23Qs0nT8PfpSkE3LpW6P\/W0Ek7xtHBMu3Vz\/3nDrUCKbzzvvvuuzxzObj2I2Vpc2eDRqdop82w4VepUkWVCyhTpkxkPGyu5cuXV0UW3RaYihUrEooxIksu2r59+1Q9I\/QHmfAiMDt37lTFG5GILpYFBpYebU3HOOiPe8G6MmnSJGUhQtK7YcOG0fvvv08jRoxQ83788ceV7LQF5uKLL1YkDPf6\/vvvVU0lkCdYm5Bc77XXXlPJ8VAeAQn3fvrpJ1UlG9l+69atSx07dqQPP\/xQfSl+\/fXXVaFJ3APrQJFKlFBAWQN3A9lAoj9O83qvwYI1blz+GmluWXPGt71PKAkMHrYpU6bkKeaFh+euu+7yNPfZxF65jNt2xUv3\/GLhpmNhdC6XTMXAABPkgkF+lmr\/Hq4CcxF8i4R22kXjlfW2\/c6baMm3exSk7mJ7GG\/GgG5qLLTciiXy5G3R93Nfj\/vs\/XQOnTvwX+o6Z2FJLTvJAeOtxV66low8Qcl+Zrzea\/FOk0EPvMgNNuvt27fTnXfeqTZvJ5HBvN0EBlYPbPooKYDijYhHRLZckAOUCPByIWGclStXxiQwIBsICUBdJJAElBj4y1\/+oizuyPoLixBIB1w7v\/zyi\/oS2qBBAxo0aJDaD5wuJPzeqVMnuvfeexXxATGZM2cOnXvuucrS1Lp1a1Vs8oYbblB7B4pLwg0GQgQCgtIFwAJuNTR8OW7bti21atVKkR5gBjcayJWzmew33P3AZMxk61kqxwslgYEFBixd+2fVi\/03po7aGV6M2Bb2ylXYVCpNEMeOh1uycrnYho0fdwDWgI0MAe6awOh1ScBufAm7dc1W3fLa1KIRFL3qaJ\/DCoNNftasWWpzxtidO3eOWLTdBKZ79+40atQotaHD\/YSfiH8B+UiEwDz66KPq3iAuaLCYjBkzRllA4M7p0aOHsvTAraVbLAIDYlW5cmVVqgAWk3nz5qnLYEmpVKmScpvB7QQSAkvTjh071F7yzTff5CEwP\/74oypj8OWXX1KBAgXUGCBVGMfpCcDfTchGvPeaXqPJmPE13J4eoSQwYMAwn+JbABrqaVxyySXKVAh2LQTGHgVN1kziPejJyuWSrPkmaxy\/1aKVBWbPHhUD4\/zGjfEkYDe2dIIcAxOPwMSz0AAZbOhz585VhRpxYAKWDC8Cg00dNZBgiWnZsqX6P97NiRAYEIp\/\/\/vfVKxYMSUkFG4sVaoUzZgxQ5EHhA4sXrxYnTxFTaZy5crFtMDgOrjs4R5DkDJ+R4Nb6pxzzlF7CP6GAyEgcShCCUsS3FBOCwxIHSw0zgrYwAkuNHfYggnZiPdeEwKTrDepRePAfAgzH5QOCgUl+vbbb5UZ06uZKFSql8lV2FTPI2jjhxW3aBtSvI1IWWDav0YlVo1XQbwgLfgpAbvxNT8ouhbNAoMVRouBUW5D12kkEBFYHOCicTbECcK60aZNG08CU61aNXUNrCWIC4FVJlEC89hjjymLCVw60RqIxvPPP69cO3AZxbLAxCMwcB9dddVVkQMgCE6GZd9NYPB3WPdBZOI1k\/2Gq2smY8abn02fh9ICAwGAwLz55pvq2wIi4ZEH5qyzzhICY5N2JnEu3Ac9ibe0YiiuBUbH1uhJg6ggYPfcjsMj65CSATyRBkXXknUKCe51uEPg\/gEZwCkkxKogxmPy5MlUtWpVTwKD926XLl1UX7h40C9RAgMXD97tuC9iXeDWgnsHQcOwvoC4INAWMTII4B07dqxyKcG1VK9evXwxMPEIzJVXXqliYXQsD2JpQIpghYE1BuEKOFkFixDIDjC58cYbVTAyvjjDCoR5OpsJ2eDqmsmYPC23o1doCYwJ\/DYJn6uwJusLQ9+w4saJgXGTF51hVx+X1tYXcR3xnpSg6Fqs9xpOI0H+zmzLXlYZjQiCUXEsGpYHEJgKFSqo00QgNGheLiQQGBCOZ599NnKqKBqBQfAsYmRwNBmneUBCMH\/E3Ljbiy++SO+9957qhz4gFQgqhgsIri3MD7\/jSyusNSA8IDLIMYNr9CkkuL7iERi4kHDdihUr6PTTT1euJYwH8vLOO++oIOEvvvhCETR8jpgX5LFBvAzcXbBOuZvJfsPVNZMxeVpuRy8hMAw52CR8rsIylhWqLmHGzevkEjYj58ki54kiTXqcp5BCpSwJLjYoumbTey1ByLPqchO5cHXNZMwggSkEhiEtm4TPVVjGskLVxQ9uqQzsTeXYHMFqkqLjWzShwbX4f7RTSJyxw97Hj65lAjOb3muZWL+t9zSRC1fXTMa0FReveQmBYUjLJuFzFZaxrFB18YNbKo+\/6mrYyPq7b2krSlfuGbfVRRdn1IUYneUBvI5Rh0ppfC7Wj675vFVCl9n0XktoIVl2sYlcuLpmMmaQ4BQCw5CWTcLnKixjWaHq4hc3XejxjBs3UjLTwGei\/pImL6hroy0tKE+g\/t+yT8TyossGHPxT9TxBvKFSmAQW61fXErilr0tteq\/5WkCWXmQiF66umYwZJFiFwDCkZZPwuQrLWFaouvjBTROWw7uWKwsJWrLqJGWCwDiT0GlLiyY1iIFxNgRv7q3dzjMvUqgUx8di\/eiaj9skfIlN77WEF5NFA5jIhatrJmMGCUohMAxp2SR8rsIylhWqLn5w8yoxANCSUa06Ey4k54kk5\/9BZlT9m5Z91MkTbY3xg1molCrKYoOCWzLfa+moRo3s6cjdhZNCyLWCU0TuI8h+9E+qUftBzY5rhMAw5JDMB51xu5hdgvJyTHSdyb7eL27aClOo1GV0YP1wgisJLVF3UjqCeN3HozWmOt7F+bkmMM6jsn4xS7bsgjZeUHBL5nst1dWoUd168ODBKrMvSAuS39WuXZu6dk28iKhUow7aE\/b7fIXAMGSXzAedcTshMImC5HG9300FREO7jdw\/k2GJScFS1ZDR8r84LSy6n\/tvek5+MUvVmtJB+pIxd9twi7amaO81kPNCpfJWG8ff0Nx\/x9\/SUY0aGWyRW0VXb0belq+++oqGDs1f2TsbqlFza+9xdc2mPSwZz5geQwgMA02bhM9VWMayQtUlEdycG6eNVYW9BBkrA68+eYTrvCwvthKYVJ4KS+bDkIiuJXMe8caKRWBgbdQEnWNtTFc1ar0mVL1u2LChytzrbFKN2lvqNu1h8fTS5HMhMAy0bBJ+UF6ODFjT2iUZuAVlA9WWFa\/MqfFqIDmFkgzMki3kIBBIG3HzkkOs95qprqerGjXWgYy\/q1atovHjx0cqO+v1STVqITDJfucEfjwhMIEXISVjUwmKCwPS4tZAiiXZZGCWTM0x3VSTeW+TsWzDLdrc473X\/JLFVFWjRhmBPn360Pfff0+jRo2KVJx2rk+qUQuBMXlWQ9E33oOeThCC8nJMJyace4UNN04NpHi42YZZUAikbbj5ITAmZDFd1ahRy2jbtm0q7qVw4cKey5Jq1EJg4r3XQve5EJjgizwom0oykY5WA4l7jzBixsUmSJYrUwLjjnmJFwOTjmrUy5cvp759+6riioUKFYoKv1SjFgKTjGc4q8YQAhN8ccpmbC5DwcwcM1wRFNySdQoJa051NWpUfJ42bVoey0ulSpVo5syZ+YQk1ajz661Ne5i\/p8r7KgniZaBpk\/CD8nJkwJrWLqnCLShuDT9gpwozP3MJ0jVBwc2m91qQ5JvquZrIhatrJmOmen3JHF8IDANNm4TPVVjGskLVJVW4uWMFklVqwAbhpAozG9aWyjkEBTeb3muplEfQxjaRC1fXTMYMEl5CYBjSskn4XIVlLCtUXVKJmz6tUfzyKSpbr80J7kyEnkrMTOYRtL5Bwc2m91rQZJzK+ZrIhatrJmOmcm3JHlsIDANRm4TPVVjGskLVJVW4RSs1kA3gpgqzbMAm1hqCgptN77Vs1wmT9ZnIhatrJmOazDXTfYXAMCRgk\/C5CstYVqi6pAo3r1IDYoEJlWrlW2yqdC3ZqHLeax999BHVq1cv2beW8WIgwJGLvpyrayZjBkk4QmAY0rJJ+FyFZSwrVF1SiZszkFdXmQaJiXf81HYBpBIz29eeyPyCghvnvZaTk0NIIBevpaMa9a5duwinkbZu3UooGZCtjSMXITAnEBACw3gKTBSKMVxCXYLyckxokSm4OF24mSQAS8EykzpkujBL6qQtGCwouMV7r8H6ctVVV6nst717946JbKqrUe\/fv5+aNWtG11xzDS1YsEAIzG\/S4OpaPFlb8Nj4moIQGAZsNgmfq7CMZYWqSzpx85uC3TaBpBMz29aeyHyCglu899rVV19NixYtiktg0lGNGgRm586dBCtMr169hMAIgRELDPclFe9B546TjH5BeTkmY63JHCNduIkFJplSC+ZY6dK1RNFxvtdgbQFZcTZYXnRz\/h9\/c1tk0lWNeuXKlUJgHELi6ppNe1iieuu8XiwwDDRtEj5XYRnLClWXdOGWTYnt0oVZtiliUHBzv9e0yyiePLxcSumqRi0EJq90uLpm0x4WT79MPhcCw0DLJuFzFZaxrFB1EdzMxS2YmWOGK4KCW7T3mnYduVfPiYXBNamqRo2xhcAIgRELjOF7SQiMIWAWdg\/KpmITdIKZP2kEBbdY7zUvEhPtNFK6qlELgcmvj1xds2kP8\/dUeV8lFhgGmjYJn6uwjGWFqkumcQuiaynTmAVVQYOCW7T3WjRXEmJkvHLCpKMatdYFscCIBUYsMIZvRiEwhoBZ2D3Tm0oQg3szjZmFasSaUlBwi+dC0i4jzmmkVFejnjNnDnXr1k3lpDl06BAVLVqUMP9Zs2axZBKkTib7DVfXTMYMElZigWFIyybhcxWWsaxQdbEBt6Adr7YBsyAqaVBwi0VgkP\/FedKob9++6jg1J6ldEGVm05xN9huurpmMaRMW8eYSSgIzbtw4wgNZpEiRCD5Vq1al9957zxMvm4TPVdh4gg\/b55nGTSww4dG4TOsaF2nT9xpcS2hSWoCLsL9+JnLh6prJmP5mnZmrQklgRowYQZs2baKhQ4eyULdJ+FyFZS0sRJ0yjZvEwIRH2TKta1ykbXqvcecchn4mcuHqmsmYQcI4lASmX79+dPToUWWF4TSbhM9VWM66wtRHcDOXtmBmjhmuCApuNr3X\/CGdnVeZyIWrayZjBgnVUBIYFAT74Ycf6JdfflHpqStXrkw9e\/akatWqRXUhuT9o3749tWvXLu2y3rx5M5UtWzbt9w36DQU3cwkKZuaY4Yqg4Ia6QjhBJM0uBEA25s+fz5qUl66NHz+eECbhbtko61ASGAgYxAUkpHjx4jRy5EiaNGmSUhr87m42sVcu42Zpf4g6CW7mwhbMzDETC4w\/zOSq3xEw2W+4z6jJmEGSRdYTGAi4efPmEZmgDHvp0qXzyAiR9TVq1KDhw4cTjgwKgQmSCvPmyn3QeaOFo5dg5k\/OQcEtmZva+vXrqX\/\/\/rRu3TrlnoeV+JFHHqErrrhCgXjZZZfRCy+8QHXq1KEWLVrQ1q1bCUevc3JyIiBPnz5dHZXGF8zc3Nx84OP6CRMmqGPUGHfAgAF06qmn+hOSxVeZyIWrayZjWgxNvqllPYHBw7R79+7IwkuVKkVr1qxRD5gmMseOHaPq1avTyy+\/THXr1hUCEyQNZs6V+6AzhwtFN8HMn5iDgpvXpoa4QHdRRycK7uPV+rP69evT3XffTbfddpsiJbNnz1YEZunSpVSyZMl8BGbLli3qCyMIjW733HMPff755zRkyJB8BAbjDR48mCZOnKhIy3333Ue1a9emrl27+hOSxVeZkA2urpmMaTE04SMwXsLo0KEDFShQQD0QJ598MuFU0rRp02ju3LlUrFgxITBB0mDmXLkPOnO4UHQTzPyJOSi4eW1q0eogaSRAYBYuXJgHGBRyrFKlCi1fvpzKlCkT+Qw4lC9fngoWLJiPwFSsWFGlscCBCrR9+\/ZR48aNVf\/7778\/H4HBl84jR45QrVq1VP+xY8fSV199xT5J6k+SmbnKhGxwdc1kzMys2t9ds94C4wXLrl271AmkJUuWKCJz4YUX0hNPPEHnn3++J4o2CZ+rsP7UIXuvEtzMZSuYmWOGK4KCW7IIDNbcsWNH2r59O915553Kiu0kMvjc7UKCtQXv3GXLllGhQoVo6tSptHbtWoIr6oEHHvB0ITmlgfs0bNiQWrVq5U9IFl9lst9wdc1kTIuhEQuMH+HYJHyuwvpZZzZfI7iZS1cwM8csrAQGVhgchEBqf1hL8M7s3LkzNWnSRIHoJjDdu3enUaNGUdu2bQnuJ\/xE\/Mtzzz0Xl8DA9bRq1SoVK4MvoNnWTPYb7jNqMmaQ8AylBcZUQDYJn6uwpmvM9v6Cm7mEBTNzzMJKYJxIHThwQLnjkZoCMSs1a9b0JDCoYg13FCwxLVu2VP+HRSWaBQaHLVDO4Pvvv1fkx8vd709idl1lst9wn1GTMe1CI\/ZshMAwpGWT8LkKy1hWqLoIbubiFszMMQsjgQER+eabb\/Kd4ESaigYNGlCbNm08CQzybiHmBgG5SGsBq0wsAoNTTtu2bVNxL4ULF\/YnnABcZbLfcJ9RkzEDAFFkikJgGNKySfhchWUsK1RdBDdzcQtm5piFkcAgQdpNN92k3D8gLDiFtHLlShUXM3nyZEKdOS8XEk4gdenSRfV9\/fXXVb9oBAYBwohbnDFjhoqZyeZmst9wn1GTMYOErRAYhrRsEj5XYRnLClUXwc1c3IKZOWZhJDBYM3K6IDZlw4YNisBUqFBBnSYCoUGLRmDmzZtHzz77LCE\/F1o0AoPs6Tgp6rS8VKpUiWbOnOlPSBZfZbLfcJ9RkzEthibf1ITAMKRlk\/C5CstYVqi6CG7m4hbMzDELOoHxmwfGH1JylRcCJvsN9xk1GTNIUhECw5CWTcLnKixjWaHqIriZi1swM8cs6ATG34rlqmQiYLLfcJ9RkzGTuZZUjyUEhoGwTcLnKixjWaHqIriZi1swM8dMCIw\/zOSq3xEw2W+4z6jJmEGShRAYhrRsEj5XYRnLClUXwc1c3IKZOWZCYPxhJlcJgfGjA0JgGKgJgWGAZHkX2YzNBSSYmWMWJALTui42OesAAA06SURBVHVrlf5fml0IIOAZSQE5jfuM2rSHcdbF7SMEhoGUTcLnKixjWaHqIriZi1swM8csSATG3+pSd5Xomzm2XMxs2sPMVxn9CiEwDDRtEj5XYRnLClUXwc1c3IKZOWZCYPxhJrj5w437jNq0h\/lbqfdVQmAYaNokfBxz7N27N2PW0sWJgOBmrg+CmTlmuEJwE9z8IWB+FVfXbNrDzFcpFpiEMLNJ+DbNJSFQ03yx4GYOuGBmjhmuENwEN38ImF\/F1TVuP\/MZZPYKscAw8LdJ+DbNhQGdNV0EN3NRCGbmmAmB8YeZ4OYPN+4zyu3nbxaZu0oIDAN7m4Rv01wY0FnTRXAzF4VgZo6ZbMT+MBPc\/OHGfUa5\/fzNInNXCYFhYC\/HDRkgSRdBQBAQBAQBKxEwOZpt5QKiTEoITJCkJXMVBAQBQUAQEAQEAYWAEBhRBEFAEBAEBAFBQBAIHAJCYAInMpmwICAICAKCgCAgCAiBER0QBAQBQUAQEAQEgcAhIAQmcCKTCQsCgoAgIAgIAoKAEBjRAUFAEBAEBAFBQBAIHAJCYAInMpmwICAICAKCgCAgCAiBER0QBAQBQUAQEAQEgcAhIAQmICLbs2cPPfnkk7R48WIqUKAAXX\/99apoXNGiRQOygvRMc+fOndSzZ0+aP38+ff3113nwwe+9evWi\/\/znP1SsWDG655576O67707PxCy+y5EjR+jZZ5+lf\/7zn3Tw4EGqWLGiKhhavXp1NWvBLb\/wfv31VxowYAC9\/\/77dOjQIfrzn\/+sns8LL7xQMGPq+owZM6hr1640depUqlOnjuAWA7dx48ap932RIkUivapWrUrvvfdeqHETAsN82DLdrUuXLoSX5tChQ+n48eNq87344ovVZi3tBAL\/\/e9\/qX379tS8eXOFk5PAYJO5+uqr6fbbb1fYbdq0iVq1akWDBg2i+vXrhxrCF198kWbNmkVvvPEGnXnmmTRixAh6++23aenSpWpzFtzyq8czzzxDX3zxBb300kt06qmn0uDBgwkb8pIlSwQzxtO0bds2atmyJe3bt49eeeUVRWBE16IDh2cS7yy819wtzLgJgWE8bJnucuDAAfVtGN\/2zj\/\/fDWdZcuWqW8vK1euzPT0rLk\/XorHjh0j4HXdddflITDYWB566CFasWIF5eTkqDkPGTKENm7cSNjAw9yAzWmnnRaxuAATkLq1a9fSp59+Krh5KMdHH31EZcuWVdYqtPXr11OjRo2UzuGZFF2L\/UThi0bTpk1p+PDhNGzYMEVg5BmNjlm\/fv3o6NGjygrjbmHGTQhMAHYuvBRvuOEG5fqA+whtx44dhPoWq1atojPOOCMAq0jfFDds2JCPwMAEO3v2bJoyZUpkItOnT1fk5YMPPkjf5AJwJ1gVFixYQP\/4xz9IcIsvMLh34U7avXs3vfrqq4JZHMjGjx9PH3\/8sbK85ObmRgiM6Fp04B5++GH64Ycf6JdffiG4yStXrqys79WqVQu1vgmBif9+yngPfAtGQUl8y9Pt559\/Vt+YERODb4LSfkfAi8CMGjVKWV\/wktQNxAXfbPANRtoJBKZNm0YDBw6kyZMn07nnnkuCW2zNgLty9erVVLNmTRo5ciSdddZZglkMyGDda9u2rYrdKF26dB4CI7oWHTiQPhAXWK6KFy+udG3SpEkq1u\/NN98M7btNCEwAdi5YYJo0aULr1q2LBHFt3ryZrrzySmXiL1myZABWkb4pRrPAIEj13XffjUwEcR74xjxnzpz0Tc7SO8H1hjiOhQsX0ssvv0zly5dXMwXhE9xiC23\/\/v3KWgXcoEuw7Alm+TGDCwRxL506daIGDRqoDm4LjODGe0EgDrJGjRrKBQfLTFhxEwLD05eM9kLwLgJ233nnncgph3nz5ikT4ieffJLRudl4cy8Cs3z5cvXiBOErVKiQmvbTTz+tvtU8\/\/zzNi4jbXPCy7B79+60a9cueuGFF1RQqm6Cm7cYQIQvueSSPNZPPKPADycDRdfy4wYLMggM4q1027p1q3KB60MJgpu3vn322WdK12C1QsMXDljgQZoRVhBW3ITApG2bSOxGjzzyCO3du1dFoYPQ4IG\/4oorCL5RaXkR8CIwOCqMwN5bbrlFPex4mcKUjQ0H3wLD3GCKxj+Y9QsXLpwHCsHNWzNw\/B4bB55HED5ghy8UiB0qU6aM6BrzgXJaYETXooPWoUMHpW+wkp588snqpCDcvXPnzlWEOazvNiEwzAct091w3PCJJ56gRYsWKUW++eab1e\/ampDp+dlw\/9GjR6sHGxaFw4cPR9xtr732GtWtW1cFQWOT+fLLL9W3vs6dO6vYorC3xo0bK2wKFiyYB4qJEydSrVq1BDcPBYHlDha8f\/3rX+oLBVxuOHmkj+SLrvGeKieBwRWCmzdusI7iBBLi9fD+R74hvP\/1qdSw4iYEhvecSS9BQBAQBAQBQUAQsAgBITAWCUOmIggIAoKAICAICAI8BITA8HCSXoKAICAICAKCgCBgEQJCYCwShkxFEBAEBAFBQBAQBHgICIHh4SS9BAFBQBAQBAQBQcAiBITAWCQMmYogIAgIAoKAICAI8BAQAsPDSXoJAoKAICAICAKCgEUICIGxSBgyFUEgWxFA\/op27drRN99845m7qEWLFqo4aSoSM65Zs4aaNWumMsCi\/AaSF5o0JAlDxlgU0kNZDyQOkyYICAKZR0AITOZlIDMQBLIegZ9++klt\/pdeeinl5OTQsmXLqFixYiodOhqSC55++ulUrly5pGOhCcznn3+ep0yCyY2Qyh1ZnIXAmKAmfQWB1CIgBCa1+MrogoAg4IEAUvEja22bNm1Sjo8QmJRDLDcQBDKCgBCYjMAuNxUEUo\/Ao48+St999x2h6jasHnv27KFrr71Wpbz3Ig5ws1x++eW0adMmZRE5dOgQwbXTtWtXdT0a6q+MGTNGVcBFzR9cc\/\/996syBEiv\/+STT9KKFSvowIEDVKlSJerRo4eq2eV0Id15553qd7hiqlWrpiqEu11Ise4zbNgwWr16tXIHYW1Is37BBRcQ\/u5Vmd1NYFBqomLFiqquzNSpUxVGpUqVUmUoUDB1\/vz59PPPP6t6Y\/iHJhaY1Our3EEQMEVACIwpYtJfEAgIAqif1ahRI0UwQFgef\/xx+vHHH2n8+PGeKwCJQJHLCRMmKNcO\/n\/jjTfSoEGDVO2thQsX0n333UcvvviiIg+IZ7nrrrtUPalu3bqp+BVdcBQF59566y167rnnFKFZuXJlnhgYxLt06dIlQqScBCbefXB\/VOHFPWHJwTpRzwkungcffDDf2rwsMCAwtWvXpldeeYVOOeUUuu222+jbb7+l\/v37q7HmzJmj5oe5o26WEJiAKL1MM1QICIEJlbhlsWFD4OOPP6YHHnhAEQlUNJ89ezadffbZUQlMiRIl1KauG6wlqLaMwFdYI4oUKUIjR46MfA6rByoxL168mNq3b6+sKqNGjYoE6h47dkwVn3MH8cYiMPHuAwLzxhtvKFKkLUOwNiHI1jk3PcloBKZfv34RAgWSNmPGDDVPNJCiGjVqKItMzZo1hcCE7cGR9QYCASEwgRCTTFIQ8I8AKnBPnjyZBg4cSLfeemvUgWAFueiii+ipp56K9OnVq5eyTMDV0qBBA2rYsGGek0Jw9cDqsWHDBlq3bp0iOXA9wW101VVXKQtQ4cKFjQhMvPuAIMFCMnPmzDzz3LZtG40dO5ZtgYEVBy41tOHDhxPIHggLGipMV61alSZNmqROR4kFxr\/+yZWCQKoQEAKTKmRlXEHAEgTatm2rNmC4ekBmojUQGMSk9O3bN9Llscceoy1btii3kxexQPwKXEeII4E15PDhw8rtAovM9OnT6ayzzlLkx8SFFO8+sLLMnTtXWUx0A9FKlMAsXbpUub2EwFiiuDINQSAOAkJgREUEgSxGYOLEiSpeBP8QIzJu3DiqU6eO54pBYBAPgj66ITakQoUKynrTsWNH5Q566aWXIp8PGTKE3n\/\/fVqwYIEKpoULqlChQupzBA3XqlVLWX9AbJx5YGK5kOLdBy4kITBZrLSyNEGAiYAQGCZQ0k0QCBoCCNi9\/vrrafTo0ZSbm6t+TpkyhWbNmqVysLgbCAwCcxEvc80116hcLXfccYeKN4FLCC6WDh06qHHq1aunTirBZYQ4mU6dOqnA3qZNm6qgYQTxgtR07tyZFi1aRBs3bsxDYHCEGu4bBAXj5JAziDfWfXQQsRCYoGmjzFcQSD4CQmCSj6mMKAhkHAEcFcbJIwTsgpCgHTlyRJ0qwukbBLB6ERi4kHCSCCeBELwL9xNIg26w6OAY9fbt29XYrVq1UieBYJlZu3YtPf300+onLC7nnXce\/e1vf1OuJ3cQL8ZAAHDp0qWVu8l9jDrWfcQCk3H1kgkIAlYgIATGCjHIJASBzCMAEgFyg9wt2dQkkV02SVPWIgj8joAQGNEGQUAQUAgIgYmuCHIKSR4SQcA+BITA2CcTmZEgkBEEsp3ASDHHjKiV3FQQSBkC\/w9Kv\/LKbyuYJQAAAABJRU5ErkJggg==","height":337,"width":560}}
%---
%[output:5163ba95]
%   data: {"dataType":"textualVariable","outputData":{"name":"idx","value":"451"}}
%---
%[output:7430f943]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7sXQe0FMXSLjJXsoASFRRBEAUEhAdIVIICAkpUQVBQUEkiIEhQDOQoUaJIRpBsJIMgIqBkETEQlBwvmf98ff9Z5u6d3enZnd2Z6a0+5x0fd6p7ur+q3f62uroq2a1bt24RN0aAEWAEGAFGgBFgBDyEQDImMB7SFk+VEWAEGAFGgBFgBAQCTGDYEBgBRoARYAQYAUbAcwgwgfGcynjCjAAjwAgwAowAI8AEhm2AEWAEGAFGgBFgBDyHABMYz6mMJ8wIMAKMACPACDACTGDYBhgBRoARYAQYAUbAcwgwgfGcynjCjAAjwAgwAowAI8AEhm2AEWAEGAFGgBFgBDyHABMYz6mMJ8wIMAKMACPACDACTGDYBhgBRoARYAQYAUbAcwgwgfGcynjCjAAjwAgwAowAI8AEhm2AEWAEGAFGgBFgBDyHABMYz6mMJ8wIMAKMACPACDACTGDYBhgBRoARYAQYAUbAcwgwgfGcynjCjAAjwAgwAowAI6A0gYmPj6dBgwbRlClTKFmyZDRu3DiqXr06a50RYAQYAUaAEWAEPI6AsgRm69at9Pbbb9OhQ4d8KmIC43Fr5ekzAowAI8AIMAL\/j4ByBObq1as0ePBgmjhxoljiSy+9RFOnThX\/nwkM2z0jwAgwAowAI6AGAsoRmN27d1Pt2rUpd+7cNHToUCpcuDA98sgjTGDUsFdeBSPACDACjAAjIBBQjsDs2bNHeF\/ee+89Sp8+PV24cIEJDBs7I8AIMAKMACOgGALKEZibN29S8uTJfWpiAqOYxfJyGAFGgBFgBBgBFT0w\/lplAsN2zggwAowAI8AIqIeAch4YJjDqGSmviBFgBBgBRoAR8EeACQzbBCPACDACjAAjwAh4DgEmMBIqq1mzJ\/36668SkizCCDACsYRA7do56dtvv42lJfNaPYhA2bJlaebMmR6cefApM4GRUGnu3HVp8eLFEpIswggkRaBuXbYfVe1iwoRXafz48RFf3n333UcHDx6M+Hv4BWoioKr9MIGRsFcmMBIgsUhABJjAqGscTGDU1a1KK2MC41Ft2nELiQmMR5XvkmkzgXGJIiIwDSYwEQCVh7QdASYwtkManQGZwEQHZ35LYAQmTJhAbdq0YYgURCBaBAaJOfv06aMggrykaCDABCYaKEfgHUxgIgAqD2kJgSNHjlCuXLks9WFhbyAQLQLzxx9\/UP78+b0BCs\/SdQgwgXGdSuQmxARGDieWihwCTGAih63TIzOBcVoD\/H4ZBJjAyKDkApl27drRhg0bEs3k\/Pnz4t9p0qSh1KlT+54VL16cpk2bZjprjoExhYgFgiDABEZd82ACo65uVVoZExiPaLNFixa0bt06qdmWLFmS5s2bZyrLBMYUIhZgAhOTNmBGYLZu3Ur4ngm38RFSuAjGTv8BXx9KstgBX\/9Bp4ZWUQ4E5a9R26ExJjB2oBi7Y7AHRl3dByMwt27doi5dutCQIUPCBoAJTNgQemYAIwKiTb5CgcxUZ\/S2kNbCBCYk2LzfiQmM93Xo5AqYwDiJfmTeffPmDfr66xm0YMFwypAhA505c4ZAWPTtypUrdO3aNVsS0DGBiYwe3TDqnZ1X2T6NJa+XoPUHzvjGHTFiBB1ZPsz29zg9IHtgJDTABEYCJBYJiAATGPWMY9KkvvTVV59JLcyODLpMYKSgjppQMC+J0SSsek661Qh846xbjXyW18kxMJYhU6cDExh1dOnESpjAOIF6ZN\/ZqlVJypgxK3Xp0otKlbqP7rjjjkQvhDfm7Nmz1KtXL\/r888\/DnkysEhirRMEMaKtEwmw8q8\/TpEpOV67dTNLNjuOd+H0jEo0bv294on\/fWfcPq9N1vTx7YCRUxARGAiQWYQ9MDNlA+\/bVqFmzLtSgQS3KmTPwwtevX08VKlQIGxkVCUwgcvJgjnTUctrOsDGL9AD+xzQy7wvFe6KNqycoqbKVpXMbmsi80ifDBMYSXOoIM4FRR5dOrIQ9ME6gHtl3jh\/fkwoUeIRefLFxUAJz\/Phxyp49e9iT8RKBATHZ8PsZWn\/gdNjrxgChEAWzF4dDJMzGNnt+\/eRmy+TDbEztecbys+naiU0+8bhCHcT\/5yMkWQQVlGMCo6BSo7gkJjBRBDtKr\/rvv79p8uT3aPDgMXTvvbdzS\/m\/fuzYsdS2bduwZ+UFAmM1GDUYMXGSYISjrEvb36bLf82XGiJZirR068ZlKVkzgmI2CBMYM4QUfu4lAnPy5DGaNu0DeuGF7nTXXXmU0wrWt337Gjp69BDhJkjevAWpXLmnKU2aOOm13rx5kw4e\/JUKFCgm3SccQSYw4aDn3r4\/\/LCCpk7tRS+88ILhJE+fPk1ffvkl7dixI+xFRJrAWIk1QU6RYE0jJl4lIcHWpj\/GuX5yUyJvh6yScfyTMmtZ0rwjsv3CkWMCEw56Hu8rQ2Dwi2zu3BFiUz116l+Ki0tHFSrUpfLl6xiu\/urVK\/TFF5\/QgQM7KF26jHT48O\/02GPVqX79dpQ6dRrDPmvWLKD165fQPfcUouPHD9Pzz79Nd999j092z54tNHZsd7rvvqLUsWPigC5ZFezYsZ7Wr19Eu3f\/SKdP\/0d33JGesmfPI\/5XsGAJyp37fpo7dzh9\/PFC2SFtk5s\/fxT98ccuev75rpQqVRqaPLkv\/fTT91S8eCXq2XOK6Xu2bVtDGzYsEQQIJGby5K2mfewQYAJjB4ruGmPHjnU0ZMjrFB9\/wXRibriFpCcokQxktSMY1RTQKAucWmytBpUbY02YwETZaNz0OjMCo32ZtW3bn\/73v6fE1Hft2kwffNCCKld+llq37kfJkyf3Lenatas0eHBb4UUYNGip8B5cvHiOunR5mrJmzUF9+nwuNmh9AzFZs2Yh9e+\/kPLlK0IHD+6k4cM7UIkSlenGjWuUIkVKSps2HX333SwaPHg5ZclylyUIL1w4SyNGdBSbO0hKkyZvUdGi\/6O0ae+gP\/7YTTt3bqR580bStWtXKEWKVDR79j5L44cr\/M03M2j69P40ZsxaypAhixgOOA4Z0o4uX75EffvONH0F1rhu3SJBfDAGExhTyFggAAK9ezehPXt+pKxZs1K9evUobdq0SSQvXrxIixcvpp9++ilsHGU8MFaPcPSTqlAgC5W\/P7P0PL3gXfG\/lWO0OKvBsIFiTKSBc0iQCYxDwLvhtcEIzKVLF6hjxycoU6Zsgozo25w5wwheg3btBlKVKs\/5HoGMrFw5V2y6Dz1U1vd3ECGQnmeeeZVeeKGb7+9wVQ8d+jrVqfMKNW\/ew\/d35KJ4+eW+vn+jLwjN00+3tAQbyNM779Sno0f\/EASsbdsBwoPk3+AlGjCgjZCbM+dAIlJm6YUhCLdtW4Fy5bqPevWSy70R6BU4gnrttXJMYELQAXe5jUCLFsWoXr3X6Pjx7TR+\/PiA0Hz66afUunXrsKELRGDqjtkuHSzrn1vECyQkFODid\/Wj+N8nh9LVsA9IS8qsZWwbz4mBmMA4gbpL3hmMwCxdOlnEnDz11EvUsmXvRDP+++\/fqHPnGsIbMnr0GuFVOXfuFL366v8oTZo7aMqUnylZsmS+Pjh+atnyUfHvCRM2+eI6evduTPv2\/UyjR6+lbNlu39n85JMu9MYbg4X8pk0rBFkaOHAJJU+ewhJykyb1oa++mk7ZsuWmUaNWUsqUqQL2x5q6dq1NkyZtFcdL0Wg4LmvX7nFxxPb22+PCeuWZM8epdesyTGDCQpE79+jRgGrXfpl27lwclMBcuHCB0qcP\/3OiJzDBbvmoeISjWZvVa8Tpivamm9fOSRlrNONRpCZksxATGJsB9dJwwQgMSARiU1588R2qWzfpL61mzR4URx2IGUHQ6KJFE+jzz\/sH3Iz793+Ftm5dSa+++hE98UQTkZ4cv\/YyZ85OI0d+74Ptr7\/20S+\/bKDatVtRfPxFQZQQ91KokLXCcfv3b6OePZ8V47755hCqWLG+qWrghXn11Q\/FnKLREJfzwQfN6X\/\/e5o6dx7le+WNG9cFIUSsTo4c+aQIFROYaGhM\/XesXfsl7dq1iVKkOB2UwKxZs4YqVarkA8RKsCw6ycSr4PhncbviSoAOkhJqcKweADfGoTipICYwTqLv8LuDEZi+fZuJL7IGDV6npk3fSjLTV18tR6dOHROekkqVGtDAga\/Sli3fJjkm0jpOmdKPli+fQlWrNiLE1IDANG\/+MD34YOlEgao4noLXB7EciA25cOGMkLfahg17kzZuXCbeM2vW3iSxN0bjIWAZXiV9nM5vv22nFSumCe8PjprgWSpevCLVr982kRyIF47EEEzbp88M+uef3wherH37toqAZMQLIVhYaz16PCsIyokTh+mOOzJQzpz5RAAugqCPH\/+Hrl5NuIY4YMBiEbysb9evX6MFC8YQ5oYYITi7oAMEXxrFwMDTA1yxPqwhT54HqFq1RolIHY6gfvrpOzH\/GjVeoPz5i9IXX4wikCzEMjVo0I6qVWucaB4I4s2QIU54yDD2xYvn6dKlcyLAG7\/iU6W6fQ33ypV4EWuEGKc\/\/9wrYqJwLNioUQexBm7uQQCf1cOHf6bJk42PK1ALqWnbt6nma\/3I7OaO1VXhOEilIyCZQNkMj02g62d3J4JKdc+JVbsIJM8Exi4kPThOMAIzfnwP+u672VSq1BPUrduEJKvr0uUpsRE1adKZnn32DerevR79\/vsv1KzZ22Jz92+4yTRv3gh6+OHy1Lv3dPEYMTM\/\/7yKxo\/fKAjC+vWLxWb+6KNVCEc6IFHDhn1NGTPeaRndl18uJbwY2bLlorFj11vujw7z538iyAviU\/LlKyzGwHwHD24nxgVRwUb811\/7ae3ahfTll+MEwUGsz7\/\/\/kXFij1Ohw7tpq+\/\/lzEEo0duy4R6cGv3VGjOifxwCB2B7ExuAkydOjXlDfvA77549mAAa0pc+a7qH37oeJY7NKl89S\/f2sRfOlPYEBCEVPUvv0wsQYQCQRJg6xo5BTvwVHb0qWTBGYgMEeOHKSSJasJIqUFOfuTqe3bN9H48V2oVau+VLr0E2KO06Z9KMbBFfBOnRK8SvAO9evXXNgJ\/g5SCaxmzhxERYqUEfbAJCYkE7W9U\/v2VcVNQJBks3am3qREIsHq3BiNBaIiE8RrNg+nn2tHQGaBsyrEnDiNtf\/7mcC4TSNRnE+VKs9T06ZNDd+4b992WrhwsiAWn3yymrJnz+2TwwbUsmVx8Yu7evWG9Oijj9OoUT3Fv2vUaEQlSiROMV6yZG1atmwKTZ3aT3gaRo5cKcbauHEerVz5JR07Bs9HNrr\/\/ofo4YcTgspmzhxJDz1UiooVK0fo79+2bk0cWKx\/jiOYQYM6iz\/lzXs\/DR36reEag43xzz8H6fPPh1PVqvWobduhifrPmDFQbMAFChSl555r43s2ZkwfOnfuNFWrVp9Kl67i+\/u8eRMFuXv33WmC1Gjts8\/60JIl0+nBB0tQvXqJA5THjn2Pzp49SW3a9KQnn3zZ1wdkBGTp1VffTZQj5siRQ\/TZZ0NFkHKHDh8LeeAwaVJ\/ev757lSxYj3fGCBcb71VU\/y7TZt36c47E252LV06nXbu3EIPPlic6tZt4Ys52rXrF1qyZKII7sRVb7RNmxbQ1KkgICWpfPmEsdBOnfqPJk78SHhfOnUaQETJ6IcfVgui9dprCfNC09tQrVpNhJ4DNav69x\/HqD9kgulfP0YsvX\/lyoX044+rRCD7zWQJMWc36HbsWfLkyejW1XhKnoyoUfv3acPvp8mMuNSunfTzm2BvSwk5ZbJkSbh9F6gZ9Udf2Rbs\/TJjBHq\/f00ew7GSpaKGnVcYPpJdg1vXL4MdZCKFP8ZGNepvvzX+fpednxvluBaShFaefPJJ6tAhISWzURszZgzt3btXpGseMGAAPfDAA7Rnzx5atGgRzZ07Vxx5vPbaa1SkSBHq06eP+DJq1KhRkhopMODPPvuM+vbtS7ly5SLUUdG+wIzei+uZOGPv3Lmz8GhUqVKFVqxYIa5t3nvvvfTyyy\/TN998E3Dely9fpq5dEzbaAgUKBJQN9gWirR1zeOONNxK968SJE\/S\/\/\/2Pbty4Qe+88w7l\/P+iMf369SOkWNf\/DR2xli+++II++ugjatLkdp0PYDZ9+nQqUaIEtWyZmMC8\/\/77hPf07NlTrBft77\/\/pqpVq4ovhGrVqiWa07lz5+jdd9+ldOnS0ccfJxCFbdu20ZQpU4R+9EX5QB62bk3IFdOsWTMqWzbhxtjs2bNp48aN9Nxzz1HFihV941+9epW6dOlCzzzzDA0bllC6HpgsX75c6B1XbvXt0KFDlCpVKsqdOzdhXljDPffcQ3fffXciuZ07dxJ0Vbp0aXrxxRcD6pO\/wJMSANnND6Ba2UAWrf+Vfti0icoXykG7UyccXe5KlfgI898BFcQPn5deeimgzvQPVCIwWizLVz8c9y0xrlDHoDhYwd9oILb\/wPbPBEbqI6imkJn77dq1azRy5EixUcHVi9onJUuWFPkhQFxSp04tNsm4uDiqX7++yMzZvXt3atPmtldCQ27UqFFi80P\/efPmBQQUtxuqV69OEyZMoKJFi4r3vvrqq+LGA8Y4duwYffXVV2JTDNbQ99KlS1SwYEEhb7U99thjgkCsW7dObMT+DQQCcxs6dKjAA037G96H92oNc501a5YgcM2bN\/f9HdlMQZCefvppsTZ9A2n7888\/6bvvvhMEEm3hwoX01ltv0euvvy7+q28gTmXKlBG\/ZjVyMnz4cKE\/EEYQR7MWaJ5LliwRRLdWrVo0evRoMUzDhg2Fvvft25foxpn\/OzZt2iRI0uDBg6lBgwZmU+DnDiCgBeCKwNpPfqbkJ\/bT2y\/UDBqL8uOPPxI+I+E2tx8hgbAEOhrigNpwtR9+f7M9LPw3ODMCe2AkcLeifHgbUqRIcCXDOzB16lSxcWMDRwNpwWbbrl078Wvdv8H7MHHiRHrqqafok08+CTg7jA3PDjZ7BAtCHpszPC45cuQQ\/TA+NsRgDb\/oN2zYIOb8yy+\/CJIl20DcQEDg\/YHnBB4S\/waPCTwrb7\/9tq8mTKQJDLxgyM0hS2AwN8w\/0Br812SFwKASMYJ4gW2w67QLFiwQ+tLjJKsHloscAsGSw+H2z6K2xZIQ099\/\/53uv\/9+WyflRgID0hLoeMiJdPm2Aq7YYFb2MC8tnQmMhLZCUf5\/\/\/1HOHqKj48XpCJfvnziTZ9\/\/jn17t3b0JuA5yA28Ez4eyH008TxFIgBzjQzZMhASJaF4xB4e7QjIciDDPXocTvxndFSZ86cKY5U0D744APhBbDStA0anhF4SPwbknh9\/\/33NG7cOOExQosWgYEnw5\/AGXlghgwZIjwmwcjD2bNnKVOmTGL+VggMjpN+\/fVX4U2DVy1Qg\/cHXqfHH3+cpk2bZigGrxsIpkaQreiJZa0hEIi4aHV+OlXJJbyC8DziSLFw4YTgdbSVK1eKzya8rJrNWHt7UmknCYzMDSHMWDsi4ptB4Wrb\/v6h7GH2z8L+EZnASGBqVfnwwoAIbNmyhRDv8fzzz\/vecurUKXGEkTlzZvFc3xBz8eijj4ojHRwpGAXtQaZx48YiRkQ7amjRooX4IvU\/kpEhMPCiPPHEEyJuBDEaIBsZM2YMigpIQMqUKcX84OVA3M2zzz5LgwYNStIP59K7d+9OdMQUaQIzf\/58QeTuuusu8V7EmWjNiMBo8tmyZRPyadIkLuOAPqgqDOJplcDgCAtHWsAYx33+bdWqVSLuBanoQV7Qli1blmhDxN+gd6wJOtWvR8J8WSQAAoFysmhXntOlSUEXr9wgo+Rw8I7CqwoyiTg3f+8jvKx4jmd2J7KLhEL90+6b3RTS5gDSwoQlEhqxd0yre5i9b4\/caExgJLC1onwEciJgE7\/KsHn5B7bide3btxc3C\/yPLHBeDmKiP3Lynx42W\/yanzNnju8RCAHiUPRVb\/GL7euvvxZeGbMGjw5IEX7hYyx4c7CZGzXE8uA54nQQ87J\/\/34R\/IhjJGz+IA1aO3z4sNiUcUz13nvvJZov5udPuPCLFV\/4\/t4n7XjFKAYGQbT\/\/POPWCuCp9EQJI3kYViPf6wRMEIcEoJ14RnBvCFXuXJlArlE8C8C3hDki4bjH3jFsGbtV7ZZrI4+BgZ4gWjiPfCaYT4aAcEmB8KoBRMj2HPt2rXCWzdp0iTKnz+hiBy8ePCSgdzqybCZXvl5YgTenLOXZmw+Kg1LsKy20DG8j9CH5l31HxifJcgZHRVLT+L\/BSPpgTHzsKQr9iGludeaZ9bq+lg+sghY2cMiOxN7R2cCI4GnrPKxmXfs2JHOnDkj4l\/wq9uoYaPELSRsZNiccSyATRTHDfhFB4Ji5HrGTRUcw+Cmkj74FV4QeHNAgLQGbwG+OM28KXqyAZKBTRXvRqxOsWLFBCkAKcPaEKSK4GDEmOTNm9f3LhyLYb3ly5cn3ErCehCXg3mdP39exPTgqEtrpUqVEmQB\/cqVu30t+JVXXhHud8wbpEFrSBSG4y2Mj9tI+oZfvjjeASbYULQGgodbTmi4CQJCA2IDwoA1osET1qlTJxFkCS8J3gtPB34xY1z8f2CKd+PGkdY0r9Obb74p+msNHpb+\/fuLeWA+WsO40DMaPC0gQrAReHoQtKzpGkQSnjusB7aBAOs777yTtm\/fLuZvFs8kYcoxJwIvS6BstovalaCNv58xxMQsSRxsCroL1kBcoUvYXLjNbgKjeVz0MSxGt4TYuxKu5tzRX3YPc8ds5WfBBEYCq2DKx0aNOJLVq1eLDa9GjRpis\/M\/hvB\/Db7YEHfxww8\/iCvP+IKCtwIbdyDSAVKCDdA\/rgUbH74s4bnA5okYCngU9ORAYplCBAQCmzk8Bzg6wTpwqwobOgKF4aEwarjqi2OWv\/76S\/wihccBfXC8pVXihqcErnd4WdBAkEAGsG78HYG3169fF0HI8FLBG4W1IMYHnhCMiZgaeHQ00qKNBR3hBlCdOnV800NcCcaEpwXHXSAB6I\/r1vAagTDqiRiO7RCrg7VgziAamIcWuwJdg6RAbyB18FLBw9WqVSuCZ2zgwIHCEwbygb\/jXSBD0C1ICLxyOE5DP\/wyh679jxfgtcI4wB8kB8GgiI2BV05fN0tWn7EsZxTHYletIOgWdombf0bFHHE0+8gjjwhbQAB3uC0cAnP5t9F0aU\/wYH4+CgpXQ+7uzwTG3fqJ6OyCKR8bGTYdkBBto47UZJB7BKTA6KbQb7\/9Jjw3IBzY7Oy+BRGpNcXCuOFsPrGAj51rhMfFP22\/Fnhr5lWxMg94HPEDAZ85IwKj3UCEdw\/ENdwWjg0ZHRGlyVWLkmcoJKbFXpZwteP+\/kxg3K+jiM1QVeVHDDAeOBEC4Ww+DKUxAloArlmNIbs8Lv6zwHEkvH3wiMJLiPxBFy9eFPl+QGi0Y0q78vrI2lCwis2cjyV2P02q7mF8hCRh06oqX2LpLGIDArKbjw2vUnqIYDlZ9AuPVqFD5HvBMSRuDeKoCMdGWsNxMo4We\/XqZYtOzGwoWE4WTIDJiy1q8Owgqu5hTGAkTFJV5UssnUVsQMBs87HhFUoNob\/eHCgAFwvWagvZeTRkFUjEUxUvXlzcHjt69KhIRYCyHC+88IJhYker42vy\/jak97T4J5PzD8blI6JQUVenn6p7GBMYCRtVVfkSS2cRGxBgAiMHopmH5dfe5Sh35sQ5euRGjpxUoCBeu994dHNfypw5S8B0\/XgfB+Lajbo646m6hzGBkbBRVZUvsXQWsQEBJjBJQQwWw6IF3Wq9nPSwmKk\/0gTm+snNdG7D7cKm+vlkLD+brp3YJP7EXhYzTcX2c1X3MCYwEnatqvIlls4iNiDABIZow+9nqM7obUHRBHEpf39mGxCP3hB2E5hARREv5R9AWVMnJOFjshI9\/aryJlX3MCYwEhaqqvIlls4iNiAQqwTG7EjIDTEs4arXLgITKBsuvCwps5YRuYS0zMzhzpn7xx4Cqu5hjhIYROojadqXX35JSMaGTLPISopcJ0hWps+sasXkkJsFuReWL18urjUiyy2uO6LmDJKmIblazpw5pYdUVfnSALBgWAjE2uYTiLhE6kpzWMoJs3M4BMaItAQ6Foo1GwpTLdzdDwFV9zDHCAyuHCJbKYrZoaVOnVqkTT958qTvOiIylVqtI4LssSAoe\/fuFePiVwvICvI24NojyA3q4CCbKjKzyjRVlS+zdpYJHwHVN59A8SxePBKyqm1ZAnPtyFI6\/9ObAYdPlb0cZfjfjIDPVbchq7izvDUEVN3DHCMwqKqLGjnIHIuslcgei1wKSAaF2jeo5IrU6UgxjxT2sg0pvlF7BNcZMc7DDz\/s6\/rff\/+J1PCob4PaPEj\/b1Tx2f9dqipfFlOWCw8BlTefWPK2hGIFwYJwUfE5Q7ng9ZS0d6psQ6Hgyn2sIaDqHuYIgYGXBEc58Ib069fPsMKuVvEXKfpRn0emDgyOoJCTAc2\/orGmbnhhnnzySfHPIUOGiMrEZk1V5Zutm5\/bg4AKm49ZbpZIpOu3B\/3ojYLaSDi6xg8o\/+MhK2TFaMYq2FD0NMFvipUf4Y4QGBQ\/fPfdd4UXBFWUcXzk3\/CBRTl6NMTJoPCfWUPBPy1uZsqUKYZHRCgWiArLIER9+vQRx01mjQmMGUL8PBgCXtl89CRFWw9uD60\/cDqoglWMbQnFolGlHWUEfhp+y9c9XOKiDeQVGwoFN+4TeQRU3cMcITCdO3cWgbuobIxjpEANxdKOHTtG77zzjqjsa9YQFIwAYHhiAvXZv38\/1axZUwyFX0sotmbWVFW+2br5uT0IuGnz+XzzUWo\/JyE+zEp79tG7qUD2O3xd3Jybxcq6ZGRRaTxYu3bsG0qR\/n5aOn8crd+djPo0vSnE05VIqACN6vThNjfZULhr4f7RR0DVPcwRAoMiaLt27SKk4cZRUaDWtGlT2rx5s\/gCGDg3z+APAAAgAElEQVRwoJTWEffywQcfCO\/O2LFjRcVYrYEMdejQQXh9rFSJVVX5UoCyUNgIRHPzMfKiaAswK3xYoUAWwzwssURWjJQN7+\/58+dDtoODBw+G3FfrGE0bCnuyPIDrEFB1D3OEwGiele7du1ObNm0CKhtkY8mSJVSxYkWaOnWqtFHgLHrMmDF09uxZyp07t\/gfvDKIf0EDgUKMTPr06aXGVFX5UotnobARiPTmY5ZvxX8Bs155hGoUyRr2umJlgMOHDxO8xviv5k25fnKTyIL73f776YmCv1Oae56jjRs30c+7joiLAvrWsWPHsKGKtA2FPUEewNUIqLqHOUJgtF80ZjEoIDhz586l0qVL05w5c6QN5NdffxUl7r\/44oskwb94d5MmTahhw4aUPHlyqTFVVb7U4lkobAQiuflM23SUOs29fSQUyIuiLSLWvSmhKvPmzZv0yeAe9PO6udSz8Q3KmjHhu+OdZdVF3AuaFgNjh8fFf56RtKFQMeF+3kFA1T3MUQIT6KaQZhZdu3YlnD9bITCIrenWrRshHuall14Sv5jy5s1Lp06dEsdRuJ6NYF8ECI8bN45SpEhhaoVQvr4h8Ld58+am\/ViAEQAC\/\/zzD+XJk8c2MLYevkxtFhxLNN7WN\/PZNj4PdBuBtP9NozT\/TvP9YcfBWzT0S6IXW7Shx4tcpb6TjxJ+aKFNmDBB\/NDCLSS7m902ZPf8eDx3IfDZZ5+JH\/H6Fgli7fSqHSEw5cuXF6XnzY6Q3nzzTVq2bBlVqVKFJk2aZIoVXLwgJriejdgaxNj4N+SCwTVqnGkjVqZZs2am46rKXk0XzgK2IGDnr+e6Y7YnuRWElPzsWbFFVb5BjLLkajeKcBwN73HGjBnFjyEcWaOxB8ZeHfBo9iGg6h7mCIFB7pUdO3ZQy5YtqVevXgG11LhxYxFw26hRI+rfv7+pNvELCHIpU6YUQcJIjGfUtFtQsoG8qirfFFAWsAWBcAmMUYxLLGS5tQX8EAbxJy9xhToaFlBEeof33ntPeF0KFSrEBCYErLlLdBBQdQ9zhMCgPMCCBQtEMjt\/N5denaVKlRJHP8gZ06pVK1NN41fR9OnThbt+7dq1AeW1X0ooMbBhwwbTcVVVvunCWcAWBEIhMLhNFOjWEOddsUUthoPoycuddf8wfdELL7xAyC1Vq1YtkfIB8TCRcNWHYkOmk2eBmEFA1T3MEQKDXy5vvfUWpUuXTqT1j4uLS2JIO3fupLp164q\/f\/XVV1SwYEFTYxs8eLC4fYTEeCgOGSh7rxYcjDpJMufVqirfFFAWsAUBK5vPwRPxVOqjTUney6TFFlUYDhK\/bwRpt4o0ARnyAlnUQkJdteHDh4sfTziaZgITOV3xyKEhoOoe5giBwYccV6lR9yhQrMobb7whqkkXKVKEli5dKqW1b775RhSIRAuWibdy5cri7BpHWSgnYNZUVb7Zuvm5PQjIEhhkva0zepvvpRzbYg\/+\/qPcOP8bnV1VPeDggY6MjDroizlCz2fOnBHJNO1usjZk93t5PDUQUHUPc4TAwCSQZG7QoEHCW9K7d29xrRkxKyA3I0eOFEG7uEk0Y8aMRMno0BcxLPhA44sCfbUGV26NGjXEs1y5chE8MmXLlvU9P3HihDizRmAwvDOLFi2iokWLmlqoqso3XTgL2IJAsM0HR0UVCmRORFzwUva42AJ9kkECFVe0Qlr0g8pWow53NUxgwkUwtvuruoc5RmBATnCMhGvPaKhKjcrQIBkgImiB8sSgOvXevXsNY2jwQccVZ9xIQgORQSI7JLXDs2vXrgnShGrYDRo0kLJqVZUvtXgWChsB\/eYTiLBoL2GvS9hwGw5gRFxkj4mCzSgQgUHJkg8\/\/FDExiDvVLiNCUy4CMZ2f1X3MMcIjGZO8IYgih+3hi5cuCBIDG4H4Qp0oAKOwQgMxsXRFMbEkdK+ffvEuCBICO4tU6aMyA+D+BfZpqryZdfPcuEhgM3nyM0sSbwsetKC\/89XocPDOVBvfWBuxvKzRQbduEIdIvOy\/x8VP5BQ5y179uwi\/1S4jQlMuAjGdn9V9zDHCYwXzEpV5XsBey\/PMVCKf1yBXn\/gDBOWCCv3\/MamgqxozQ6Pi+yU4YEBiUHhWPbAyKLGcpFCQNU9jAmMhMWoqnyJpbNICAj4B+NqQ3DulhDADKOL1SvRYbwq4l3ZAxNxiJV+gap7GBMYCbNVVfkSS2cRiwhMWHeYui\/c7+uFYFzefCyCaIO4Rl6i6XWxYdoBh2AbiiS6ao99efcaOtKnMt0375ZyC2UCI6FSJjASILGIQEA7NtLfIuLNJ7rG4ZTnBSUFZs+eLZVbyioibENWEWP50\/Peo8u7VlP8rtUCDCYwMWoTTGBiVPEWlq0\/NvK\/ScSbjwUgQxT1T\/\/vhOeFayGFqDzuZisCBxsmSzLeW\/tz0sIdR2x9jxsGYw+MhBaYwEiAFIMigdL9++dwYQITWeO4vG8YXdo30vcSreiiXW+dP39+wKGmTp0qbjWirVixglatWkUDBw5MJP\/cc8+FPRW2obAhVHoAeFtOz+2bZI253ltNaYtUIlX3MCYwEmatqvIlls4iARAwumEUKPkcbz72m9H1Uz\/RufUNEw1859O7iVIkLUsS7tuRzgEJNkNtdpQWYBsKFX31+gUiK\/qV+h8XqbqHMYGRsG9VlS+xdBYJQmAqFMhC5e\/PbHodmjcfe80oUknpAs0SSTGR\/Rv\/9femLFmyhOrUqSO6bty4kX766Sdq3759oqE6duwYNgBsQ2FD6NkBQFjQjDwswUhLomf33ReRGl1Og8oERkIDTGAkQFJcBMdFaPoK0bLXonnzsc849LldQk3\/H8psbt68KQrF7tixQ+R3QYI6NH0mXo6BCQVZ7uOPgAxhiXuoMuXsu0oaPFX3MCYwEiagqvIllh7zIoGS0cH7srhdcSl8mMBIwWQq5J9RN2XWMqZ97BbYunUrffDBB\/Tmm29S1apVmcDYDXCMjaeRFTMPS5ZGCfEtWRr2CQkhVfcwJjAS5qCq8iWWHrMi\/sSlZbncdFeG1AIPqyn\/mcDYY0Zuye1y7tw5UactY8aMoqo9rk8L79yAATR+\/PiIuOrZhuyxoWiPoico+nenzvsQ\/TskcQyX\/nm4hMV\/naruYUxgJCxaVeVLLD3mROZu\/Zdem7E70brDrQzNm0\/4ZqQdHTlxPTrQ7BcuXCiq26PuWqFChZjAhK9mz48AwoLjHSSOk20aWQnHw2L2LlX3MCYwZppHAiBFA6Aklh4TIoGOiXo9fR91qnZv2BgwgQkPQqcS08nM+oUXXqDr16+LqtPHjh1jD4wMaArKGOVe0ZaJq8xaMrlEXpYQj4NCgU\/VPYwJjIQ1qKp8iaUrLRKIuITrcfEHjQlM6GbkZvKCVSGId\/To0TR8+HCaPn26uG5tx7VptqHQbSaaPY\/2rUrxuxIH08KjEmqsSqTmruoexgRGwmJUVb7E0pUV8ScvdpMWPXBMYEIzo\/h9Iyh+33CK5m0jqzPV30KCns+cOUMlSpSwOoypPNuQKURRFTjat0oSr4qbU\/WruocxgZEwe1WVL7F0JUUGfXOIPv7qD7G2SBIXDTzefKybkf66tFNxL1evXqX4+HjKlClTwAXgCKlcuXJ06NAhunjxIt1zzz308MMPU40aNShFihTWFx6gB9uQbVBaHijYtWYt063lQaPcQdU9jAmMhCGpqnyJpSsnsvPIBao4eItY19GBlShNyuQRXyNvPtYgvvLnLLq4o4fo5BR5wbtBSCZPniyuTBu1IUOGiJgXxMD4t\/z58xOeFy8ud9XeDCG2ITOE7H9ulvHWzR4XfzRU3cOYwEjYvarKl1i6EiJIQlehQGaqM3pbovVEw\/uCF\/LmI2dGN878QmfXPuMTdsPRUevWrX3XpPWrQAbeDh06iD\/hyKho0aKULVs28W8E865Zs4auXLlCixcvppw5c8oBEESKbShsCIMOcG7FKDoxOXEGZX0Hu681R3Y1SUdXdQ9jAiNhSaoqX2LpnhUJRFq0BUWLvDCBkTMh\/2rSbiAvmPlrr71Gb7zxhiAo+takSRPat2+fCNz1fwa5W7duCeKDoyVk7g23MYEJF8Hg\/QN5W9wYkBsKEqruYUxgJKxBVeVLLN2TIkZVorvVyG85AZ1di+fNxxhJLUjX\/6mTx0b+c0GOl0mTJhE8LmnSpPE9fuSRR6h3795JaiP590ddpJEjb1fKDtWm2IZCRc64X6B8LV46FrKCiKp7GBMYCStQVfkSS\/eUiP\/NItQqWn\/gjGPERQOPN5\/bZhS\/fyTF7x1maFduIi7aBBEHU6VKFapYsSJ9\/PHHlCpVKvHoscceoxkzZtADDzwQ9DPy7LPP0hdffBH254htKGwIySzJnNX6QuHPKHojqLqHMYGRsCFVlS+xdNeL+JOWe7PGUf5scbTg1WKumTtvPrdVoT8qSpWtLKXMWpbiCiXEkri1zZ8\/n7p27UrFihWjTz75hHLnzi2OlVq0aEETJ04UgbxGbfv27fT222\/Tt99+G\/bS2IasQ6jdHgqUGVeV4yEZZFTdw5jASGhfVeVLLN21ItFKQmcHALz5JKCokRcQlwzlZtkBbdTG+PDDD8VREjww8Mgg7gUEBQ0kRt9QKwlHTij62K5du4C3mKxMnm1IDq1gGXHjilahtEUqiYHclmhObnWhS6m6hzGBkbAJVZUvsXRXiRiRloHPFqRXyud21Tz9JxPrm49\/gK4bj4rMDAhBuUOHDhUemGTJkvnE06ZNK8oIpEuXjk6dOkX\/\/vsv7dixQ1ytxhXqWbNmJYqdMXtPoOexbkMyuF3evSZJDSKv3x6SWbeMjKp7GBMYCe2rqnyJpbtGxIi8RPMmUThAxOLm44UA3VB0unHjRho0aJAgKYFa1qxZqXHjxsLzog\/8DeV9Wp9YtCEZvIyOiVQNxJXBI5CMqnsYExgJq1BV+RJLd4WInrx4hbTogYu1zcff4wIsvOh1CWb8uEK9du1acY0aN5KQrRexMQ8++KAI+E2ZMqWtn51YsyEz8Iy8LeijciCuGSbBnqu6hzGBkbAKVZUvsXRHRfy9Lk5ehQ4HiFjZfK6f3EznNjTxQeWWXC7h6M6sr74WkplsOM9jxYZkMNLHuWRvN5mun\/hLdIu1uBYZrDQZVfcwJjASVqCq8iWW7pjIs+N30Kp9p3zv96LnRZt8LGw++tpFYjOptZ2SpQpcQ8gxw7L5xUxgbAbUb7hAV5\/haUn7UGUmLZLwq7qHMYGRMABVlS+x9KiLNJ74C327+6QSxCVWCIz+yChj+dmUMmuZqNuNUy8EgWnYsCEtWrRIZN1F2YCyZcvS888\/b1v8C9YWCyRY06FZvhbIcZyLNYtXdQ9jAiNhB6oqX2LpUROpO2Y7rT9w2ve+aS2LUp2Hs0ft\/ZF8kaqbj\/+RkWpxLnqbmDJlCuF\/\/\/33H1WuXJn69etH2bNnp0qVKtHff\/\/tE82YMSOdPXtWxMPMnDlTVKe2o6lqQxo2wa4\/802i8C1I1T2MCYyEbaiqfImlR0VET14GP1eIWpXLFZX3RuslKm4+KlyNltW\/lsgO8unTpxdFGvPkyUPvvvsuvfzyy2KYl156Sdw6ypIlC124cEFk312wYIH4rx0BvSrZkHZzCLgZJZnL9d5qit+1WuDKcS2yVhpcTtU9jAmMhH2oqnyJpUdFRAvW9XKcSzCgVNp8\/K9Hq+x10XT69NNP04EDB0QiuwoVKtCNGzdo3rx59P7779Ply5epZs2aNGbMmCQmAAKDnDH169cP+3PkdRsyOxZKeWduyvBEayYsYVuK8QCq7mGOEhgkh1q4cCF9+eWXtGfPHkIGS1xHRHn6F198UXxZhNrwJYMkUhgfXz7Xrl0T59OPP\/44tWrVypJrV1Xlh4qtHf1UuWEkg4XXNx9tjXqvSyzcMNLWXaRIEapduzYNHDgwkbpnz55NPXr0oM8++8zwuwrJ7FCCAAnwwm1esyG9l+X03L6Jlp8i092UsUbbRH9jT0u4FsIemMgi6Dc6CAVK1a9atUo8SZ06Nd1555108uRJQTbQkIa7S5culucFF27Lli1p69atoi\/cvkgodeLECfGLCP\/GeXbJkiWlxmYCIwWTtNB3e09RowmJE4Gp6n0BKF7bfIwUqb9lFPdgZ4or+Ka0vr0uiB89SEyH+kf+Dd6VCRMmiHgYo\/bUU0\/R8uXLw4bAzTZ0tG8V35FPsIVqR0NMVsI2B8sDqLqHOeaB+eijj0QNERALuGLr1asn6oyg+uvkyZPFrxaQDaTuxpeAlda6dWv6\/vvvKVeuXNS\/f3\/fryME24EQbdmyRTwDedKqywYbX1XlW8E0XNkBXx+iAV\/\/kWgYlUmLfqFu3nzM9Iojoxvn99DVI18L0Vg4MvLHBN8Z8BYPGTIkCVzwHteoUYPi4uKSPFu3bh3hu2jv3r1mMJs+d6MNBUomh8VkqtGOkme6y7cuJi2mKo6ogKp7mCME5vjx4+Io5+rVqyKaH1cO\/VvPnj3FEdC9995LK1euTFR\/JJim169fT82bNxceHRRU8y93Dy8MAu\/y5s1Lb731FuXPn9\/UcFRVvunCwxTY8PsZqjN6m+EosUJesHg3bj4yqo2FjLoyOBw+fFj8wMKR0f333y\/TRRAeHDvhR9Mvv\/wi1SeYkJtsyOjGEF9rDlvFER1A1T3MEQKD64WI4M+QIYPwhoBs+Dd8YKtVqyb+jDgWlLKXaa+\/\/jqtWLGCmjZtSqgga0dTVfl2YBNojHUHztAzY26TlwoFslD5+zNTtxr5IvlaV47tps1HBiB\/4hJL8S6B8Pntt99EdWkcI5UuXdoUxiNHjgjPb5kyZcQPsXCbG2wIcS3+8Sw4FtIqPIe7Ru4fOQRU3cMcITCdO3cWgbtVq1ZNUoper8Jy5crRsWPH6J133hGuWLOGwN2HH35Y3AyYNm2a8PLY0VRVvh3Y+I+x5Jfj1GLqTt+fl7xeQhCXWG5u2Hxk8Y+l69GymGhy8Kps375dXDLQWrBMvDNmzKDy5ctTvnzhk3anbcj\/uIg9Llatx1l5VfcwRwhMnTp1aNeuXeIoB0dFgRq8KJs3b6bnnnsuyQ0Aoz4HDx6kJ554QjxCP+RfgPcGwbynT58mVIlFlswGDRoYnlkHmoeqyrf7I+V\/ZBRLx0TBsHR685HVs568xGKsiyxOejkVSwnobxBpa9V7Xpi8hGIpzvZRdQ9zhMBonpXu3btTmzZtAmq2Q4cOIo4F1V2nTp1qagGIlXnllVcoRYoUNHfuXDE2bjX5NwTwIqdDoUKFTMeEgKrKl1q8pJD+WjSOixa3Ky7ZU30xLxAYJi+h2aFqBCZYRlwmLqHZiBt6qbqHOUJgEM9y\/vx56tOnD7Vo0SKgfkFwQERw5jxnzhxTO1i8eDF17NhRxNQgpXeBAgWoU6dOgqjA\/bt27Vpx4wmkBjlhvvrqKxGHY9ZUVb7ZumWf628Y8ZFRUtTcSmD8j4vS3NOI0hUfIKt2liMilQiM\/jq0PhsuFM23iLxt7qruYY4SmL59+4obQ4EakkAhjbcsgdGn\/C5evLjIlglvjL7hDBu5G3BF++2336a2bRMnVDKaC5SvbyBdwebtbVO3PvuSow6JTlvfDP+s3\/rb3d\/jn3\/+Eann3dYy\/VrVN6WL9w2j6+nkAuXdtg4n54M0DfihFekWaRu61fX2d1yygQcjvRweP8IIILki4kD1DSEWqjVHCAwC244ePSo++MGOkFBbZNmyZVSlShVx5GPWIIs+aCNGjCDE2hg1kA9ct5YlRqqyVzM8ZZ\/j+Ghgg4L0SoXcsl1iSs6NHhjN+8KxLuGZohc9MMHyt6A2Uc6+CclFuamDgKp7mCMEBh6QHTt2iGy5vXr1CmglyH6Ja9aNGjUSCenMmpYDBnIob48bSUYNx0iIqUEsDPqYNVWVb7ZumedaIUYO2A2MltsIDMe7yFi2nIyXCIxZxlyOcZHTuRelVN3DHCEwyGyJQme45uzv5tIbR6lSpejUqVMiZwzqF5k1JMhD3gW0zz\/\/nBAsbNS0LMBIZrdmzRqzYTmINwBC+sBdJjDuJzB8Rdr0o25ZwAsERrtVxDeJLKtXmQ5MYGxUJa42IwtuunTp6McffzS80rxz506qW7eueCuCbQsWLCg1AyS\/wy9eHCUhgNeo4fo2ygiA4IDomDVVlW+2brPnGoHpViN\/TCaoM8NHe+4GD0z87v4Uf2C8b8p8dCSrveBybiYwgW4UsafFHt17aRRV9zBHPDC4gQTygLpHyAMDQuHfkPESRdBQCXbp0qXStjJq1CgaNmyYKAz5zTffiP\/q2++\/\/041a9YkJL0zi8HR+qmqfGlQDQQ18tKjVn7q8iQH7wbD0kkCw16XcKzcvK9bCYyevMQVreLLlsu3icx1qqKEqnuYIYHBscqVK1coefLkvsRwdit17NixNGjQIHHluXfv3tSwYUNRWBHkZuTIkSJoF1efkc3S\/ygImXyxKSAjJvrqG0hR9erVRZBw4cKFBZnRvDe7d+8m9N2\/f79IaoeCj7hubdZUVb7ZuoM91wgMHx2ZoxhtAoMCjKmylaVzG5r4Jod\/ZygXfkp789XGloSbCAzXKIot27OyWlX3MEMCg8W++OKLooQ8PCCILdE3VJCW2fiDAQxygmMklBRAw5hZsmQhFFu8fv26+FugPDGoTo0Kr4FiaPbt2yfmj7HQcuTIIcbU\/p0pUyZR8VqfEjzYXFVVvpUPgF5WIy\/vPnUfdX7i3lCHiZl+0SAwIC3x+4YbYsrHRd43Nb0NBbtFpF8pHxV5X+92rUDVPcyQwOCaM27nIFcKGuJEUJBsz5499PTTT4vq0UjJb0fD1WckqUNpgQsXLggS89hjj4ljpUAFHM0IDOZ19uxZGj9+PH333XeEHAo3b94UuThQfwnZeu+663apd7N1qKp8s3UbPefAXeuoRZrA+JOXjOVn07UTmyiuUAfrk+UerkMAQbhnTp+mnBXq05E+lYPOj0mL69TnigmpuocZEhh4Xvwz36IsPP6+YcMGH7FxhWaiMAlVlW8VOu3KNPrx0ZE8enYSGJAVfdN7XUBcUmZNuIXHzfsIBEvrz1Wgva\/faK5A1T3MkMDg+GX69OmJ8IUH49lnnxXFEWOtqap8K3pkz4sVtBLL2kFg\/INx\/WfDMS6h68cNPc3IytH1CylzlixiqhyI6waNeWsOqu5h0gQG6kICOiMCg\/T8SN2valNV+bL64kKNskgZy4VDYPyJC4hKyqyJj2\/5qCg8\/TjROxhh0eaj97KEY0NOrI\/f6S4EVN3DDAkMMt+iiKJ\/C0RgUNMI\/1O1qap8GX1p5KVJ6Rw0pmlhmS4s44dAKJuPkceFg3HdZ1pmt5C0JHKYuT6RnNFKsjTqG9C7EooNuQ8tnpFTCKi6hwW8hYQF44qzvhl9iHDt+fDhw6RioSht7aoqP9iHSR\/vUqFAFlrcTl0PW6S\/VKxuPuc3Pk\/XTmwU0+KjoUhrJ7zxNQKjJyraiGaExUrArVUbCm9V3Fs1BFTdwwISGKsKZAJjFTF3y3PMi336kd18rp\/cnCh3C3tc7NOBnSMJsnI1nk5\/OYAG3NmG+ubcT\/G7Vgd8BY6CtOehxq\/I2pCd6+Sx1EEgpggMjpBwJIRcL9pVaiNVIpfLmTNnqF+\/fkluLamjeoq5Wkia94VLBNhjxTKbz+mlhejWzavsdbEH8oiMYhS3AgLT7dQE8T49UdEmECph8V+AjA1FZNE8qBIIxBSBQQp\/5FqRbcjlgvwwqjZVlW+kL74qbb8VB9t8ONW\/\/XhHYsREqfkfqkxpH6os4lXMYmDsmgsTGLuQjM1xVN3DpGshoSr0n3\/+KRLN5c6dW6T9j5WmqvKN9MclAuy3aqPNxyhzLh8Z2Y99qCMGCr71j1thAhMqwtwvmgiouocFJTDx8fE0cOBAgodFS8MP0HG0VLt2bVEKAIRG9aaq8v31xnEvkbFkIwKj97wwcYkM7qGOaqWKMxOYUFHmftFEQNU9LCCBOXTokEjnjy9ftJQpUwrPy5EjR+jatWvib9mzZ6f58+dT3rx5o6mLqL9LVeXrgRy9+m\/qtfiA+BPHvthrYv4EhsmLvfiGM5r\/7SH9zSGZ4FsmMOGgz32jhYCqe1hAAoN6QStXrqQmTZqIwogFChQQx0Y3btygv\/76izZv3kxDhw6lbNmyEWJmVG6qKl+vswFfH6IBX\/\/BJQIiYMiBsuiy5yUCYJsMKZNADkPk\/mADpSlUznSCTGBMIWIBFyCg6h5mSGA2bdokSMvIkSOpVq1aAeFH\/heUFxg8eDBVqFDBBWqKzBRUVb4eLY59iYztMHmJDK6hjGpEXnJ0W0xX\/vg50XBWbg4xgQlFE9wn2giouocZEpiuXbtShgwZqFevXqY4z549mzZu3CjIjqpNVeVr+tK8L3x0ZK8F81GRvXiGM9rRvlV8uVisJJAL55129uVbSHaiGXtjqbqHBayFNHr0aBGsa9auXLlCLVu2pJkzZ5qJeva5qsr\/dP1h6rZgv08vXGHaHhP1v2F09uGVlD9\/fnsG51GkELASiCs1oMNCTGAcVoDHX6\/qHmZIYBo3bmwpMV3r1q3p008\/9biKA09fVeXrbx2NaVaYmpTKoawOo7kwf88Lbz7RRJ8oEHm5u8sXlK5Mg+hOxqa3sQ3ZBGSMDqPqHmZIYJo1a2bJo4LbSpMmTVLWNFRUPl+Zjoy56r0vWpAubz6Rwdp\/1Mu719CRPpV9f\/biUVEgpNiGomNDqr5FxT0MujIkMAjIxRES4mDM2sWLF+n111+ntWvXmol69rlKytfiXTRl8LGRfWYZKOaFNx\/7MA40kn+m3Jx9V0X+pVF8A9tQFMFW8FUq7WF69XAxRwljVUX5N28RZXvr9hc7kxcJ5VsQYQJjASwbRLUcLv65W9IWqWTD6O4aggmMu\/Thtdmosof54x6QwCDoUCaI99y5cyLZHcWD05cAACAASURBVFejdq9Js9clcrrBkVHKzI\/Q+c2txEsylvucUmYrn+iFvPnYi7\/+RpF+ZCeOjPgatb265dEig0BMERhUl5a5Qq1BbVU+MiqK3KheV74+3qVCgSy0uF3xyIEVIyOf39iUrp3YlGS1RsnpmMDYZxT+AbpatlwruVvsmw1xMUc7weSxIoaA1\/ewQMAYemD27NlDhQsXlgbTqrz0wC4R9LryOUmdvYZ0\/eRmOrehSYLHpfxsQWTiCnUI+BImMOHjj+Mi\/VERRnTC4+K\/EvbAhK9bHiHyCHh9D7NEYFDvCPldUMwRQbooI2BUtHHLli306KOPUooUKSKvAQff4GXlT9t0lDrN3cv1jWy0Hy3WRbYUABOY0ME3Oi5yA3HRVsQEJnTdcs\/oIeDlPSwYSoYeGFyLXrVqFeXMmVO4SOvXr294I2nbtm00b948+uijj6KnCQfe5GXl1x2zndYfOM01jmyyG733hQmMTaAGGMb\/WjSOi9wWoMsEJrI2wKPbg4CX9zDLBGbs2LE0Y8YMWrJkiaHnRT\/gqFGj6JFHHqFKldSL\/NfW6WXl8\/GRPV8AGEV\/y+iOwl0o7QOvSw3OHhgpmISQURK6LI36klMxLmYzZwJjhhA\/dwMCXt7DLBMYlAZo164dlS5d2hT7Q4cO0ZAhQwhERtXmVeVrt484cDd8y9TIS6rsFSjlnaWCxrz4v40JjDH+2jVoPL28a7WvVpEm7UaPi\/9KmMCE\/9niESKPgFf3MDNkDI+QGjRoQAsWLDDr63v+1FNP0fLly6XlvSboVeWz98UeS9PfOJI9NtK\/mQlMYj34Hw3pn97dcRalK58QIO2FxgTGC1riOXp1DzPTnCGBeeaZZ2jRokVmfX3Py5cvTxs2bJCW95qgV5XPBCZ8S9OXBoh7sBPFFWxveVAmMEQ3z5+gQ62yJ8EOx0Nac+sxUTCFM4Gx\/HHgDg4g4NU9zAwqQwJTvXp14YFJnz69WX86evQoPf\/887Ry5UpTWa8KeFH5WvDu2GZFqHGpu70KvaPzDpRZ1+qkmMAkjW1x000iq\/rUyzOBCQc97hstBLy4h8lgY0hgOnToQA8++CC1bdvWdIzevXvTf\/\/9R+PGjTOV9aqAF5XP3pfQrU1PXDBKqJ4XbQaxTGD8g3JVIS6abpnAhP45457RQ8CLe5gMOoYEZtmyZfTmm2\/SgAED6LnnnqNkyZIlGev69ev02WefEbLwjhw5kurUqSPzPk\/KeE35Gnn5u39FSpda7Rw9dhqU\/rgI46bKVpYylJsV9itihcAY3SDSg5d32C5KladI2HjG4gCxYkOxqNtorNlre5gsJoYEBp1bt25N33\/\/PRUsWJCefvppypcvH2XOnJmQ5O6ff\/4RQbv4UKFyNYiMys1LyteOjha8VpwqF8yislpsXZve62IXcYkFD8yxj2rRpW1fBdWFal4XWw1PcjAmMJJAsZghAl7aw6yoMCCBQRbeDz\/8kGbOnBlwvCeeeELIZM+eNDhPZhK3bt2ihQsX0pdffkkoR4DCkJkyZaISJUrQiy++KMiRXQ3XvXFb6vLly\/T444\/TtGnTpIf2kvL56EharT5Bu2JdAr1Z1c3HqC6R2xLNWbcGd\/ZQ1YbcibZ6s\/LSHmYF\/YAERhtkx44dNGvWLNq\/f7\/wuCA7L+okgbzUqlXLyrsSyV67do1ee+01kfEXLXXq1HTnnXfSyZMnCc\/QkIumS5cuIb9D63jz5k1q3Lgxbd26VfxJVQKz\/vczVHf0Ni4bIGsxNy7TqWW3a36FckVa5lVe3XyCXXfWr5s9LDJWEJ6MV20ovFVzb7sQiFkCYxeA\/uOg\/MDEiRMpTZo09P7771O9evUoVapUovbS5MmTaejQoSL25pNPPhGek3AaMgsPGjRIvAs1nlQkMPqK06eGVgkHrpjpq+V3QUHGlFnLRGzdXtt8ZIkLAGPyEjGzSTSw12woOqjwW2QRYAIji5SE3PHjxwWJuHr1qggCxjVs\/9azZ0\/h+bn33nvFFW2jQGKJV4mjKZAjkBfUdJo+fbpyBGb2T8eo3cw9Ao5uNfJTtxr5ZKCJWRn\/W0aR8rxoAHth80FW3Ov\/HqTzaxLHszFBCf4x4VtIMfs14qmFM4GxUV2Iq3n33XdFgUhUtMbxkX\/Dl361atXEnxEnU6xYMcszAEECedm7dy8NHDhQBB\/jxpQqHhgtYFcDZupLRanuI6HFI1kG18Md9AQmrlBHS2UBQlm22wmM0e0hJi5ymmYCI4cTSzmLABMYG\/Hv3LmzCNytWrWqOEYK1MqVK0fHjh2jd955R9yKstpwDXz8+PGCCH366ac0fPhwpQgMHxtZswh9JeloEBe3eWDMrjm7uWiiNU1HT5oJTPSw5jeFjgATmNCxS9ITOWN27dpFL7\/8MuGoKFBr2rQpbd68WeSigQfFSkPAbqNGjcTV76+++krclFKVwHDMi7llXD4wli7tvm1DkT420s\/IDR6Yo32rJCmWqM3RC0UTzTXsjAQTGGdw57daQ4AJjDW8gkprnpXu3btTmzZtAsoiI\/CSJUuoYsWKNHXqVOkZIBC4du3a9Oeff4oq2chjg6YSgdG8L5OaF6X6xfnYyMw4In1VOtj7nSYw+qBcPhoysxRrz5nAWMOLpZ1BgAmMjbgjnuX8+fPUp08fatGiRcCRQXDmzp1LpUuXpjlz5kjPQAsAhqdnxIgRvn4qEhj2vpibhT7DbjQ9L9rMnCQw+mOjnD1XUFzxmuaAsYQ0AkxgpKFiQQcRYAJjI\/gagenbty81b9484Mhdu3al+fPnWyIw69atE6QIR0bffPONSIyntXAIjH6SGD\/YvG2EynCorYcvU5sFx2jrm3zbSAbrTL9WFWJnH3am4CiCx\/PkySMzVVtlbnW9zzdestdmEd0Xuavitk7cQ4P179+f8EMr0s0pG4r0unj8yCCA7Pj+yVoPHjwYmZc5OKppIrtIzK18+fKiirXZERLqMaEuU5UqVWjSpEmmUzl79izVqFGD\/v33X3HkVKlSpUR9wiEwblL+gK8P0YCv\/yD2vpiahBDQjo+c8L7g\/dH2wPjnceFjIzk7CUWKPTChoMZ9oo0Ae2BsRBz5WJDht2XLltSrV6+AIyN7Lq5ZIxgXv3TMWvv27Wnp0qWEL5Vu3bolEVeBwPDNIzMrSPzcafISKQJjdqNIQ4HJizV7sSrNBMYqYizvBAJMYGxEHeUBFixYYJqPpVSpUnTq1CmRM6ZVq1ZBZ3D48GExHlqBAgUoZcqUSeRPnDhB+F+6dOkob9684jnicMqUCe5ad4vy9eRFeBY4425Qm3AycFc\/MTs9MDLEBbeK4netpiwN+9j4qeWhjBBgAsN24QUE3LKH2Y2VI0dISEz31ltvCSLx448\/UlxcXJJ17dy5k+rWrSv+jmvQqIodrOkT31kBacKECaKuU7DmBuXrk9YxcTHX8OXfRtOlPYOFoFNHR9os7SIwevLCOVvMbUAlCbtsSCVMeC3yCLhhD5OfrbykIwQGN5BwlRrXnXFjCPlg\/Nsbb7xBy5cvpyJFiohjITuaV4+Q+NjIuvbdcHRkF4Hx97rwsZB1e\/B6DyYwXtegs\/NnAmMz\/lqBRZQR6N27NzVs2FAUcwS5Qbp\/BO3eunWLZsyYIciOviGTLz7QJUqUEH1lmxcJjBawizVynSNZTTsfuKufaTibj568cMI5ef2rJhmODamGBa\/HOgJMYKxjFrQHyAmOkVBSAA3FFrNkySJiVK5fvy7+FihPDKpTo76R1ZpGXiQwmveFj43kDfDW1dN0+qtHKd0jH1CafEkLhcqPZI9kqJuPnryw18UeXXh1lFBtyKvr5XnbiwATGHvx9I2Ga9JIUofSAhcuXBAk5rHHHhPHSoEKOMYageleMz91rc45X2RMUB+4m7HcTEqZ7X8y3SIqY3Xz4SOjiKrDk4NbtSFPLpInHTEEmMBEDFr3D+yU8jf8fobqjN5Gs155hGoUyep+oBycob5QI6bhdOCuHgorm4+evMQ9VJly9l3lIKr8arcgYMWG3DJnnod7EHBqD4s0Ao4E8UZ6UXaP75Ty+fhITpN6r0uqbGUpQ7lZch2jJCW7+fCRUZQUYuNr+Bq1jWDyUBFDwKk9LGIL+v+BmcBIIOyE8rXgXQ7cDa4gt+R6CTZLqwSG410kPpQuEWEC4xJF8DSCIuDEHhYNlTCBkUDZCeWz90VCMS4oEyAzSxkCo3lfmLzIIOoeGSYw7tEFzyQwAk7sYdHQBxMYCZSjrXyNvFQokIUWtysuMcPYE\/GC50XTSiACY5RVlwmMt2yZCYy39BWrs432HhYtnJnASCAdbeWz98VYKXrSokm4MebFf\/Z6AnN63nuE4NwjfSonWSSTF4kPo8tEmMC4TCE8HUMEor2HRUsNTGAkkI6m8jXyMvuVR6g63zzyacefvMQV6khxhTpIaM95ERCYnPF\/JSEtGaq+TNnbTnR+gjyDkBFgAhMydNwxighEcw+L4rKICYwE2tFSPpcMMPe8uOl6tITpCBH\/oyIutiiLnPvlmMC4X0c8Q6Jo7WHRxpoJjATi0VI+Hx0lVYbmefHCUZGRKfHVaIkPmIdFmMB4WHkxNPVo7WHRhpQJjATi0VA+k5fA5AVPvO554fgWiQ+aB0WYwHhQaTE45WjsYU7AygRGAvVIK19fsJFrHiUoRPO8pC\/+EaW+p6mEltwlcmH9LPpvRDMxqWRPdaP8Lfu7a4I8G1sQYAJjC4w8SIQRiPQeFuHpBxyeCYwE8pFWvuZ9Wdi2OFV6IIvEjNQV8Q\/W9aLn5fLuNb6AXXheZPLAqKtRtVfGBEZt\/aqyukjvYU7hxARGAvlIKr\/umO20\/sDpBK\/D0CoSs1FT5Nb1i3R6edFEi\/MieTEqxMgERk2bjeaq2IaiibZ674rkHuYkWkxgJNCPpPI178u0l4pSnUeyS8xGTRHN85Kx\/GxKmbWMJxd5\/vuJdHxcazH3LI36UpaGfcT\/583Hk+p01aTZhlylDs9NJpJ7mJNgMIGRQD9SyteqTfepfT91qHqPxEzUFNHIi5dyu0AT+qMivWb8A3Z581HTbqO5KrahaKKt3rsitYc5jRQTGAkNREr52vFRLB8dXT+5mc5taCK04OYjo6N9q1D8rtUS1kLEBEYKJhaygAATGAtgsWgSBCK1hzkNNRMYCQ1ESvl8dfr2bSO3kpdgqf\/9j4oCmRJvPhIfMhYJigDbEBtIOAhEag8LZ0529GUCI4FipJQPAhOrBRvdftvoxrnj9OfLdyWyjqwthlCm2p0lLCaxCG8+liHzTAe+heQZVcX0RCO1hzkNKhMYCQ1EQvna8VHv2vdTx1iLf7lxmU4tKyyQT52zOqUvPV5CC5EXMaoOnbluF0qWNr14uRaUa3UmTGCsIuYdeSYwCbpq1qwZbdq0yTuKU2CmZcuWpZkzZ0qtJBJ7mNSLIyzEBEYC4EgoP5aPj9wYtHth1RT6b0yrJNZgRwZdJjASHzKPijCBSVBcJL4jPWoSUZu2FcytyEZtATa8iAmMBIh2K\/\/GzVuUvctq+rDeA9S2Yh6JGagjoj86ckPci39wrh2ExV9bTGDUsV\/\/lTCBYQLjlHVb2ZesyDq1nlDeywRGAjW7lT\/o2z\/p4xUHYzJxnUZg3EBeoPpoFFtkAiPxIfOoCBMYJjBOma6VfcmKrFPrCeW9TGAkULNb+bF4fKS\/Lh33YCeKK9heAvnIimjel7iHKlPOvqsi9jImMBGD1vGBmcAwgXHKCK3sS1ZknVpPKO9lAiOBml3K1xdt7FYjP3WrkU\/i7d4XuXp0BV3Y0s63ECe8L0YButqEInFspNcaExjv23CgFTCBUYvArF+\/nvr3709ZsmShkydPUrp06ShHjhw0atQoOn36NC1ZsoSGDBlC58+fp5deeomeffZZ+vXXX8Xzo0eP0tChQ6levXqG5vLkk0\/S77\/\/Tk899RS99tprVLRo4tIpWqdr167R3LlzaePGjTR69OiAHx4r+5IVWS99WpnASGjLLuVrnhe8MhaS11059Bld\/CUhnT6aE2UCghEXzCnS5AXvYAIj8SHzqAgTGHUIzPHjx6lKlSqCjOC\/t27dokGDBtGGDRto0aJFPgtt3bo1ff\/994KMJEuWTPz9o48+ookTJ9JDDz0kSI5\/AzFq164dXbhwgZYtW0aFCyfcwvRv586do3Xr1ok5xMXF0cKFC5nABPluYAIj8cVpJ4GJhbwvt25cotPLHkqEbLS8Lkg8h4ZjoSN9KvvmkKn6a5S19VgJbdsvwgTGfkzdMiITGHUIzDfffCM8I\/B8wOuitQ4dOtCIESN8\/3799dfpu+++o3379vn+Bq\/Mjh07CERl1qxZVKZM4npu6JMtWzaaPn266Is9JVjr2bMn7d69mwmMyQedCYzEN6EdBEY7Plryegkqf39mibd6U8SpBHVOHhGZaYoJjBlC3n3OBEYdArN\/\/36qWbMmVaxYkT755BNKnz4h\/9OePXsSeUwCEZhixYpR27ZtqWrVqjR+\/O3cVocPH6YBAwaIMeDRYQJj3+edCYwElnYQmFgI3I3fN4Li9w0XiEbL44J3+ZMXVIJGCzXxnIRJWBJhAmMJLhY2QMDtNmTHd6QbFP\/KK6\/QypUrKVeuXNS7d2+qXr16kmkFIjCIbRkzZow4Ilq9ejXdc09CgV7E1FSrVo1++uknJjA2K5kJjASgdnw4Y4HAnN\/YlK6d2BRV8nJyakc6uyzBvQvi4hbSojcrt28+Eh8BFnEYAbfbkP93JDzObm0VCmQO6AW\/fv06jRs3Tnhgrl69KjwyICAZM2b0LScYgbly5Qo1aNBABPiCAOHfIEU4Oho7diwTGJuNggmMBKDhEhitbECvp++nTtUSWLmKLdo5XqKRw8UOPbl987FjjTxGZBFwuw35f0fqLyxEFhnro8vEIf7zzz\/UpUsX+vHHH6lkyZI0Z84cSp48uXhZMAKDYyLcTMJxFIJ\/V6xYQSlSpKDnnnuOCYx1VZn2YAJjClH4abJjwfsSzQy7J6d0oLPLR\/o0F42bRBJmElDE7ZtPOGvjvtFBwO025CUPDDRmlMLit99+owceeMCnUHhjmjdvLmo8zZ49mx577DEpAoMjpDfffJPeeecdcaPo008\/pTRp0jCBicBHhQmMBKjhemBUrjqtHRsBxlTZylKGcrMkEA1dRO91yfXeakpbpFLog0Wpp9s3nyjBwK8JAwG321C435FhQGNb165du1KfPn1E7hetffnll9S5c2dRNBHFEwN5YAYPHiyOm5Db5caNG1SpUiXClWgQIHhy0PgIyTZV+QZylMDgnj3uucNIEOkNhWfKlIlKlChBL774IlWoUCHkFSMQC6x5+\/btdObMGcGA8+bNKyLMW7VqRXfddZf02OF8OFX1vugDdjUgIxm4G42aRdIGYVHQ7ZuPxeWwuAMIuN2GwvmOdABOw1e+\/\/77dOzYMRo2bJjYL9Bwe+irr76i5cuXi7wsaFoemAMHDviOld59912RoK5cuXJCBl6XgQMH0po1a0RAMBrGRX6XYHlgtIl16tRJ5I\/C3hioWcHciqxb9CEzD8cIDLIN4s79qlUJKdxTp05Nd955p8h+iGdoSPyjsVeZxUAGfdFHn0woQ4YMIoEQCBMa\/j116lRBlGRaOMpXkcDoywIAv0gSF4yP3C6n5ybcLEJz+5GRv025ffOR+QywjDECfI36\/z+T991HBw8e9LSZfPbZZ+IHNchLoUKFRKK6O+64g3r06EH58uWjs2fPipiWjz\/+WGTiffnll0XWXWTihQemQIEC9NZbb1Hp0qXFj3Ekt0MAMBpICwjN33\/\/TcjIiyMmo0y8CPpFPhoQovj4eJH1Fz\/kkRnYv1nZl6zIekmJjhEYLXMhjAXMF4aQKlUqunjxIk2ePFmkZEaWQ0SDg9nKNjBm7Q4+or\/btGkjEgiB2MAw4CI8deqUYMXw0oA4mbVwlA8Co1rZAC3eJdKZdY1yu3iNvMC2mMCYfcK8+5wJjDoExmtWaGVfsiLrJRwcITBI2fz444+La2r9+vWj559\/PglmyESIjIb33nuvIBpayuZg4IL1ItAK4zZt2pQ+\/PDDJOJg0IgiR5s2bZqYh1kLR\/kgMN91LEWP3pPB7DWufx6tJHVGxMWtV6RllMYERgYlb8owgWEC45TlWtmXrMg6tZ5Q3usIgUFAFFxkOMrZsmWLoRcEX\/pI\/oMGtx6yHJo1XF2DhwVFt+DSM3LRwS2HehVoffv2FUFWZi1U5WvZd3\/oVoYK3X2H2Wtc\/1wjMHGFOlJcoQ4Rma8\/efGix8UfGCYwETEVVwzKBIYJjFOGaGVfsiLr1HpCea8jBAZR3QhOQsplFMAK1BAQhaAqXEdD4JQdDWeMDz74oPDo4EwS9\/PNWqjKVyn+RSMvGR4bR6ly1DCDLKTnl3evEfWLUMcoZ9+E2CgVGhMYFbRovAYmMExgnLJuK\/uSFVmn1hPKex0hMHXq1KFdu3aJICgcFQVqOAbavHmzIBkgG3Y01KFAXAwa7ujnzp3bdNhQlL9ox3FqOW2nEvEv0cjxog\/UzTtiL6XKVchUL14RYALjFU1ZnycTGCYw1q3Gnh5W9iUrsvbMLjqjOEJgNM9K9+7dfWTCaLmoAorbRLj6jFtD4TbEyDzzzDP0559\/UuPGjUU0uUwLRfna8dGpoVVkXuFamWiQFyxeOzrycqxLICUygXGteYc9MSYwTGDCNqIQB7CyL1mRDXE6jnRzhMAgngXX0BCv0qJFi4ALB8GZO3euuJaGVM7hNMTF4FbStm3bqGDBgjR\/\/nxftVGzcUNRvtePj678NYcubu\/ugyaSV6Vvnj9Jh1plE+9SIebF356YwJh9wrz7nAkMExinrNfKvmRF1qn1hPJeRwmMWRAtMiOCaIRLYJCfAMdV8LyAvOC+v9VEdnpwQbqCBf9O+PEMjd98hgplT00zmyQkMfJSS\/vfNErz7zQx5WtZqtOlPLeJjN3ruNX1Pt+QyZ7sQIT\/KdZQVyVPnjyKrYqXAwSQ5wM\/tCLd3G5DuHDh9Twwkdah3eODlHz\/\/feGw2KPwy1bfVNRP44QmPLly9PRo0fFB1+LRzHSApL9IAFQlSpVaNKkSSHpH4nyOnbsKDw+SAg0evRocfvJSrPKXrXijV49PtJn2Y2k50Uf96Ki50WzMfbAWPm0eUuWPTAJ+rL6HektLbtztlYwtyLrztUaz8oRAlO\/fn3asWMHtWzZknr16hUQL8Sp4Jp1o0aNfBkNrYA7YcIEXz+UN0fAMCqDWm1Wle\/l46NokBfttpGmB9VuHfnbFxMYq58478gzgVGLwKxfv17sGch8i6zwqIuUI0cOUQIAYQiIyUR2XPwgxp6CytPIxIvn+FGOBKxIymrUkIEX2X2RmBVZ6I3SfMDThpMJVMFGWZ1mzZpR27ZtDcezsi9ZkfXOp4\/IEQKDVP8LFiwQSeT83Vx68EqVKiWy5iJnDOoXWWkwMnhbkN0XwboNGjSw0j2RrFXle5nAaEG7kfK8+Od5UTFolwlMyB817hgAAbeTYKvfkW5UNBKswtsPMoL\/ovTMoEGDaMOGDbRo0SLflLVaSCAjWoJVLbM8cozpy9honUCMUBoHJW0C1UJCig8QG\/xwz5w5M82bN0\/kQMP+hb\/5NyuYW5F1o24CzckRAgOloGYE2C2YplYkSz\/JnTt3Ut26dcWfUEwLsSuybfjw4TRy5EhRxwJ5ZrQqorL9wzEU9PVq+YBIJ6rzYiXpUG1G38\/tm48da+QxIouA221IhQ0SpWZAIDZu3Ci8LlrDbdgRI0b4\/o1M7kjHsW\/fPt\/f8IMZpwogKsggX6ZMmUQGgT4oaTN9+nTRF3j5N7wfsZnFixcXj0CgatWqJYoQozhkOPuSCvox+oQ5QmDgfsNVatQ9wrEOAmz92xtvvCEqgBYpUoSWLl0q\/e2gGWHKlClFsG645AUvtqJ87fr0pOYPUf3i8hWvpRcYIcHzm5rTtf\/WidHt9r54vRhjuJC7ffMJd33cP\/IIuN2GrHxHRh6t0N6ATO41a9YUaTtQgy99+vRioD179lDhwoVNCQxu1+K4BwlatXp86HT48GFR1RpjwKMTiMDgCCpnzpyJJo+q1JcuXUo0niZgBXMrsqGh50wvRwgMljp27FihTBRT7N27NzVs2FAc94DcwHuCoF0w0BkzZvhKlGsQIZMvPtCoJo2+WkOZgMqVKxNcge3btxfBu3Y0K8r34vHR+Y1N6dqJTREhL0f7VqH4Xat9alA5WDeQrbl987HjM8JjRBYBt9uQle\/IyCIV3uhItYHaeyj2i72levXqSQYM5IFBbMuYMWPEEdHq1avpnnvuEX0RU4NbWj\/99FNQAmM08yZNmog4G+yP\/s0K5lZkw0Mwur0dIzAgJzhGQkkBNFSlRuDUiRMn6Pr16+JvgfLEwFD27t2bJIZGO5pCXxwfmQXsws2HQF+zJqt8zfuC8bxyAynSiepUTlBnZjfac7dvPrLrYDnnEHC7Dfl\/RxoVZHUOvcRvDnZpAHvPuHHjhAcGRYHhkQEByZgxo2+QYAQGcSyIt0SALwgQ\/g1ShKMj7Ud7IA+MPz7w3OBIC\/saThSYwCS1IMcIjDYVsFUkqUNpAQQ4gcSgojSOlQIVcAxEYOCtCXaryX\/5OMb6\/PPPTT9XsgTGa94XJi+mqrdFwO2bjy2LjNFB+BZSguJVITCaGeM2EC6bIEazZMmSYo9Knjy5eByMwOCYCB4THEch+HfFihXihzTK4VglMMiDhiOp\/PnzG366ZPclI\/2o8nF1nMB4AUgZQ9HIS7ca+albjXyuX1a0yIv48My75Xo8IjlBJjCRRNfZsZnAGBMYZ7US2tt\/++03euCBB3yd4Y1BwtJNmzbR7NmzxQ9rGQKDH+XIYYYixKi3hwBcnDBYITAIoUAwL8hToCazL2l9rciGhp4zvZjASOBupvwNv5+hOqO3eaJwoz7e2ZWbogAAIABJREFUBUu3O2D35JQOdHb5SB+qsU5eAAQTGIkPmUdFmMCoQ2Dg8UDYAm7Hag0hDoi5nDlzpu9CiJEHZvDgweK4Cbldbty4QZUqVSLU3gMBgicHTZbAIPt81qxZxVVurSG0AreY9M1sXwpV1ksfRSYwEtoyMxSNwHgh7iWSnpdYv20UyJSYwEh8yDwqwgRGHQLz\/vvv07Fjx2jYsGHCY4KG20NI44EbsVq6Dy0PzIEDB3zHSshVhtAGhCWgwesycOBAWrNmjQgIRsO4yDETKA8MZPCe7du3U506dUSfmzdv0tatWylfvnzidlOopMRsD\/Pox8+ZRHZeA8tM+V4kMHZ6Xvwz67LXJbGFM4Hx2idefr5MYNQhMEi7gYBZkJdChQqJrLm4DNKjRw9BIM6ePStiWpBYDrdlEaeJrLvIxAsPTIECBcTFFNTug\/cFye0QAIwG0gJC8\/fffxMy8uKIyT8TL\/LPIDv9tWvXEhkgPC8\/\/PBDkkspZvtSqGRH3vqdl2QPjIQOzAzFK7WP7M6ya3TTgMlLUoNiAiPxIfOoCBMYdQiM10zQbF9iAuM1jUZovmaG4ubbRzcv\/U1nvquYCJlwvS\/+R0UYPBZKAoRqXkxgQkXO\/f2YwDCBccpKzfYlJjBOacZl7w1mKF9s+49aT9\/lygDe66d\/pnPrnvWhmSpbWcpQblZI6BqRFgzEHhdzOJnAmGPkVQkmMExgnLJdJjAOFXN0SuGhvjeYoWjJ69wWwKsP1sW6w\/G6+Me45Hpvtcium6Vhn1Ahjal+TGDUVTcTGCYwTlk3ExgmMFK2F8xQ3HZ8ZCdx8S8DwB4XKXNJIsQEJjTcvNCLCQwTGKfslAkMExgp2zMjMBUKZKHF7RIqiDrZbpzZQWfX1vNNIRSvS6AU4HxUFLpmmcCEjh33TEDA7TZkZTNlndqDgBXMrcjaM7vojMK3kCRwNiMwi9qVoMcLZJYYKbIi4dwyChjjMvcmUbJkkZ244qO7ffNRHH4llud2G1J1g3Sz8VjB3Iqsm9fsPzcmMBLaCqR8N12ftlJROhBZARR8m0jCICyKuH3zsbgcFncAAbfbkKobpAOqln6lFcytyEpPwAWCTGAklBBI+a6Jf7l1nU4tSajhkaXGFkqWJnHKaf0Sg1WJzfriIMpUNyHtNTf7EHD75mPfSnmkSCHgdhtSdYOMlD6DjYtEdlu2bBFZeCtUqBBQ1ArmVmSdWHOo72QCI4FcMALjhvgX2fIAevKSb+opSp4ui8TqWSRcBNy++YS7Pu4feQTcbkNe3yBRQgCZeMeNG0cZMmSg6tWrU8qUKUVWXNQhQibcJUuWUMGCBSOubNRfQiHIl156ibp168YEJgjiTGAkzNGtBCZ+3wiK3zdcrCCuUEeKK9Qh4Gr05IUDciWUbqOI2zcfG5fKQ0UIAbfbkNcJjKa2hx9+mEqVKkVTpkxJpElUh0Y1ajyPRkNdJRSEZAITHG0mMBLWaPTh1OJfutXIT91q5JMYxT4RPXHRRjW6cYRYl7iHKtORPpUTSM5DlSln31X2TYRHkkLA7ZuP1CJYyBABvkadAIsqBKZkyZJUokQJmjhxYiJ9Hz9+XPw7e\/bsUfkkMIGRg5kJjARORh9OJ+Nf9EdGGcvPppRZyyRZhX+sy92dZlO6co0lVssidiPABMZuRN0zHhMY9QnM\/v37KUuWLHT9+nV6++23RXzKvn376PDhwzR8+HD64osvRMVqHC8dOXKExowZQ8WKFaPdu3dTkSJFqGHDhgKkTZs2iWOof\/75h65evUqDBg2iPHnyiGd\/\/fUXTZ48mXLkyCEKQS5evFhUpGYPTPDPOhMYie9CNxGYYPEu\/hlzsTRkzU1bpJLEKlkkUggwgYkUss6PywTGmMDAS+zWhpIqRj\/6MF9\/D8zFixepX79+1KZNG+Flmj17NvXp00cQGLQ9e\/bQ008\/7SMwffv2pcKFC1Pjxo1F9eqZM2dS27ZtRWVrjDN16lTR74033qC9e\/fSt99+S3gHxpg\/f77w8Bw9epQqVqxIr7zyChMYEyNiAiPxKQtEYEY1KUzPP5ZDYgR7RIKRF64MbQ\/GkRiFCUwkUHXHmExgjAmMf0Zwd2grYRbBasKBwCRPnlzEu9y6dYsuXbpEmzdvpmXLlgkCA5LRs2dPH4GBd6ZmzZo+AvPRRx\/RTz\/9JGJoMmXKRIcOHaJ8+fJR9+7dCYHCxYsnJDz9+eefaf369bRy5UpasWIFbdu2jSZMmOCDiY+Q5CyGCYwETv4EZuA3h6j\/V3\/QktdLUPn7I5\/ATv9lkDpndUpferxv1kZ1itjjIqHUKIowgYki2FF+FRMY73lgMONAFx6MYmDmzJlDpUuXliIwOAp65plnxA2mDz\/8UNxmQqtfvz61atVKHAv5t9atW1Pu3LkJ3hutMYGR+yAzgZHAyZ\/ARLOAo3\/Arn+wruZ54ZtFEop0SIQJjEPAR+G1TGCMCUwUoI\/IK4wIDGJSUqVKRXFxcaYeGEwKMS7wuGzcuFH8F8dPOCKqVauWODryb\/Xq1RPkaOjQoUxgLGqVCYwEYP4EJloBvGb5XfhqtITyXCDCBMYFSojQFJjAqE9g9KaDI6QePXqII6RkyZKJQN3atWv7jpDWrVtHjz\/+uOgyZMgQmjZtmjgeat++vYh5+e6770Q\/NAT8YhwEAe\/atUscJ2nP2AMj94FlAiOBU7QJjP\/5sT7HC8e6SCjMZSJMYFymEBunwwRGLQKDW0PIA4OkdkYNcSvNmzen0aNHU9WqVWns2LE0cuRIkQAPx0VIQIf\/ZcyYUQTuNm3alH788UcCsWnRooU4XoJHBrExiKuB12XDhg308ssv+7w1uN0Eb03ZsmVp8ODBYiyjZuXquhVZGz8eER+KCYwExEYEJlIZeIPVNDK6ZcRHRxIKdFiECYzDCojg65nAqEFgEGA7Y8YMQUzuuOMOQUJANtKnT5\/IepDev0OHDrR69WoR6NuuXTvq1asXwWPSsmVLQUguX74sSBCCeRHzAiKChvGRXwaZffE3xLwg9gUNhOnTTz8Vx1Q4bkKQ7\/33309NmjQJmP3XCimxIhvBj4vtQzOBkYDUiMBEKoGdUUVp\/+KLXHBRQmkuEmEC4yJl2DwVJjBqEBibzSIqw1khJVZkozJ5m17CBEYCSCMCM+75ItSo5N0SveVErp\/cTOc2NPEJ64N1tWMjzqQrh6XbpJjAuE0j9s2HCQwTGPusydpIVkiJFVlrs3BWmgmMBP565V+5fpNydl1DU18qSnUfsSettP7YCNMBefH3uuDvfFwkoSwXijCBcaFSPDYlt9uQqhukm83ECuZWZN28Zv+5MYGR0JZe+VoNpFNDq0j0NBcxumnE5MUcNy9JuH3z8RKWsTpXt9uQqhukm+3NCuZWZN28ZiYwIWhHr3y7rlD73zTSjozOfTueTkx4zTdL9rqEoDCXdXH75uMyuHg6Bgi43YZU3SDdbIxWMLci6+Y1M4EJQTt2E5hLuz6ky7\/frnZqFO\/CNYxCUJRLu7h983EpbDwtHQJut6FmzZqJYoXcoocAbjKh1pJMYwIjg5KiMv4EJpwbSP5HRkZ5XQAje17UMSa3bz7qIK3uStiG1NVtNFbGBCYaKLv0HZryp\/1wlDrN20uhEhg9eTkz\/ZDhavmmkUuNIIxp8eYTBngu78q3kFyuIJ6eQIAJjMcMYcuWLTR9+nSRTOjUqVMiQVDBggVFUa1GjRqJiqOyTVN+OPEvevJy5ed8FL9rte\/17G2R1YQ35ZjAeFNvMrNmAiODEss4jQATGKc1YOH9o0aNEhkRUVciRYoUlDVrVrpw4YIojY6Gs0OUO0+TJo3UqHoCYyUDLwoxXj+5ia6duH02fPmXM3R5x5kEVjzvltT7WcjbCDCB8bb+gs2eCYy6ulVpZUxgPKJNFMR65ZVXxGzx39dff50yZcpESAH99ddfU7du3QSZeeGFF+j999+XWpWewAxr+CC1+F\/OoP38bxhpwvpjIyYvUtArIcQERgk1Gi6CCYy6ulVpZUxgPKJN1KRA1U\/Uk4Anxr8tWbJE1LKAZ2bVqlWUJ08e05VB+R0mrqb3lv5OZvlf\/DPq6j0u7HUxhVpJASYwSqpVLIoJjLq6VWllTGA8oE1sFNWqVRMzXbhwIRUrVizJrOGJqVChAqF4V\/fu3UVlULMG5Z+pN0mIBSMwes+LP3Hh+kVmKKv7\/L333qM+ffqou8AYXlm0CAzbUAwbmQ1LZwJjA4iRHmLOnDmiiigqiG7fvj1goG6nTp1o0aJFVKVKFZo0KYGYBGv5Cz1EZ5\/+JCiB0ZcDuPDtMbp+7LJvSD4uMkNY7eeqfnmorTW51UWLwLANyemDpYwRUNV+lCol8NFHH4ly5UWKFKGlS5cGtOURI0YQ\/ofjo7Vr15rafLDbR\/51jDjOxRTOmBNQ9csj5hRpsGAmMGwFXkBA1e8gpQhM586d6csvv6RKlSqJW0aB2uzZs6lHjx6UOnVqES9j1kBg9LePcLvo\/Nr+lPLutIm6grzcN+8mESUzG5KfxxACqn55xJAKAy6VCQxbgRcQUPU7SCkCgy+Tb7\/9lmrWrEljxowJaFcLFiygLl26iOcHDhwwzQkDApN272Iqc3gOjW4YT2kfyZxo7M0brtDmDZfpsyOJ\/+4Fw+Y5MgKMQOgIPPnkk+I7hxsj4GYErJQdcPM6\/OemJIGpVasWjR49OqAeEOD71ltvSRMYLymU58oIMAKMACPACMQCAkoRGNkjpBkzZlCvXr0obdq0tHv37ljQM6+REWAEGAFGgBFQCgGlCEz\/\/v1pwoQJVLhwYVq2bFlARQ0bNkzkiMmXLx8h8R03RoARYAQYAUaAEfAWAkoRmPnz51PXrl3pjjvuoB07dohkdUYN2XlXrFhBTzzxhCA83BgBRoARYAQYgf9r7zyAqyq+MH7IgAgKf+pILwIiUqNGFIkgAgLSISAdRZpICAIqVdpIb9JEeqRZ6EW6NEEQBullEEEQpTiEJh3+853hvnlJ3iOBfe++9p0ZR5J39+7ub0\/e\/e7Zs7skEFgEgkrA\/PXXXxIZGakjgD1hIiIiEo3GrVu3pHTp0nLp0iXp27evNG\/ePLBGjK0lARIgARIgARKQoBIwGM+GDRsKTqLGjryTJ09ONMSxsbEqXLCEevPmzZI1a1a6AQmQAAmQAAmQQIARCDoBA\/ECEQNr1qyZILEXhzneuXNH94jp1auXIAoTHR0tMTExATZcbC4JkAAJkAAJkAAIBJ2AQaewyqhPnz5y\/\/59zYNBlOXixYty8+ZNHXUc9IideMPCwugFJEACJEACJEACAUggKAUMxmHfvn16rMD27dtVvOB8pKJFi0pUVJTUqFEjAIeKTSYBEiABEiABErAIBK2A4RCTAAmQAAmQAAkELwEKmOAdW\/aMBEiABEiABIKWAAVM0A4tO0YCJEACJEACwUuAAiZ4x5Y9IwESIAESIIGgJUABE7RDy46RAAmQAAmQQPASoIBxMbbY0RdHDGzZskXOnDkjKVOmlNy5c+vmeG3atJF06dIFr0ewZw4C58+fl5kzZ+p5WX\/++afcvn1bMmTIIMWLF5f69etLlSpV3NIy8SGTshw+\/ybQs2dPmTt3rjZy\/vz5Eh4e7rLBJj5gUta\/6YVm644ePSrTpk3T59GFCxckffr08uyzz8q7776rK2rdHZlj4gcmZe0cJQqYBLSxO2\/btm3lxo0b+kmWLFn0wYWjB2DZsmWTefPmSZ48eewcJ9ZlMwFsiNi6dWu5fPmy1pwqVSo9vfzKlSuOltSqVUtGjBiRaD8hEx8yKWszIlb3iAQ2btwo7733nqOUOwFj4gMmZR+xO7zcBgJLlizR8\/2w+SqESubMmSUuLk5\/hmFPs9GjRycSMSZ+YFLWBiTxqqCAccKBN+4KFSrItWvXpGzZsjJw4ECHUNm\/f7\/u6nvs2DEpXLiwLFu2zK3ytXsQWZ9nCZw7d04qVaqkYiV\/\/vzSr18\/ee2113S88WYyZMgQHX\/YgAEDpEmTJo4GmPiQSVnPEuDdPE0AL0CVK1dWQWxtqOlKwJj4gElZT\/eX9zMn8Ntvv0mDBg10F3mc2depUyfJmDGj\/oyDi7FZK\/4dyt9BFDBOfta\/f3+ZMWOG5MqVS1atWiVp0qSJ54V4eEHgICIzfPhwqVu3rrmX8g5+RwBRlfHjx+t5WWvWrNHpQ2e7e\/eu1KxZUw4dOiSlSpWSBQsWOD428SGTsn4HkQ2KR6Bjx46yfPly6dChg\/oWzJWAMfEBk7IcLv8jUK9ePdm9e7fUqVNHI70Jbdy4cbJ161YpX768pjZYZuIHJmV9QZAC5gF1HDuAt2y8fXfv3l2nD1wZVPDSpUulXLlyMn36dF+MGev0MoGhQ4fKtm3bpFChQoJ\/uzJEYSZNmqRvRLt27dJLTHzIpKyXcfD2hgQgXCBgEM1bsWKFFClSxKWAMfEBk7KG3WNxLxBA3ouVY\/fzzz9L9uzZk1WLiR+YlE1W47xwEQXMA6gnTpzQ6Aps8eLFmqjpypD\/0qNHD82HOHDggKRIkcILw8Jb+jsBROAmTJggOXLk0OQ6mIkPmZT1d1ah3D68EGHqCNORiLggYocETFjCCIyJD5iUDeXx8de+T5w4UYYNG5YowptUe038wKRsUu3y1ucUMA\/Irl271hGG27t3r56d5Mp27Nih2d8wJDvlzJnTW2PD+\/oxgerVq8vBgwd1jnrw4MHaUhMfMinrx5hCvmktW7aUTZs26dRRly5dlIc7AWPiAyZlQ36Q\/BCANeXYuHFjzcXEcwd5dydPntSX5gIFCkjt2rUTvWib+IFJWV8hpIB5QP7bb7\/VqaPUqVNrboM7++OPP3Q5NQy5D3ijooUWgdjYWOnbt6\/myKxcuVLy5cunAEx8yKRsaNEPnN7Onj1bevfuLc8\/\/7xGdbGS7WECxsQHTMoGDtHQaSnECV6kO3furFs4IFrnylq1aiVYmm+ZiR+YlPXVyFDAPCCPfBZkc\/\/vf\/\/TxCl3hn1hsEIJNmvWLClTpoyvxo71+oDAokWL9E0a88XIj8F+MJaZ+JBJWR9gYJVJEMBDp1q1arpKBOIFKxctcxeBMfEBk7IcTP8jgJdkvCxjihor2KKjo+Xtt9+WZ555RqMwX331leC7CAYBAyEDM\/EDk7K+IkgBk0DAOCdluhoUrESKjIykgPGVx\/qoXggW7LcwduxYFS9Ywui8p4fzl8fj+JD15fE4ZX2EhNW6IXDv3j2dZt65c6fu4dGuXbt4VyYlYB7HB+g\/weWOeEnGyzJsypQpjvxM517i+wd7C2FjVUwxYfbAxA9MyvqKPgXMA\/LJDZ8dP35cKlasqKUWLlwoJUuW9NXYsV6bCFy9elW6du0qq1ev1i8JJPBiA6mEZuJDJmVtwsBqkkkAb8eIzkVEROiuu2FhYckSMCY+YFI2md3iZTYSwDMGzxqsXFu3bp3LmrH6MSoqKt7LtIkfmJS1EU28qihgHuD46aefHGE4bCCE7ZpdGdbdN23aVD\/Cv7EzLy14CZw+fVqX1B85ckTDuXg4FStWzGWHTXzIpGzw0g+8nsFPsEMz9pDC8mn4TEJzF4Ex8QGTsoFHOfhbjMUBiOBhZSwiMK4MK9usF2hrOtvED0zK+mpEKGAekHeeGnrYGSVWAidWKSHJiha8BE6dOiWNGjXSUO7LL7+s4iVTpkxuO2ziQyZlg3cEAq9niNQhuR+5dO727jh8+LB2DMnf2I4B+w2NGTNGd3m2pqcf9TvIpGzgUQ7+FltnZiGKh8iIK0NkuESJEvqRtbGqiR+YlPXViFDAOJHH5nR4aCHzG8vYXBnexhHSq1q1qmNHTV8NHuv1HgH8MTds2FDFCzaUQv4LVh0lZSY+ZFI2qXbxc3sI4LsDSbuPYkWLFtXNMWEmPmBS9lHay2u9TwD+gE1TMWWNc9lcbeuxZ88e3aUXBpEDsRNqPkQB4+SLo0aN0iRNTAsh3yGh02B5Nfb\/QBInTqu2cmG8786swU4COCoCIVx8QbzxxhsyderUZJ97ZeJDJmXt5MO6zAi4m0LCXU18wKSsWY9Y2tMEcB4fdoZHlAUrkGJiYhJVYe0K75zEG2o+RAHj5BaYU8TyNRxZ\/sorr8igQYM0iQoGFYy3K7yR4zPsyEsLTgI4uh6bR2EaYP369XpcQHLNxIdMyia3fbzO9wQeJmBMfMCkrO+psAUJCcycOVMPkoVhjzIc6IiIzPXr1zUvZuTIkbqpXbdu3aR9+\/aO4iZ+YFLWFyNIAZOAOhJ4W7RooVt\/I9KSJUsWPbwRp8jCChYsKHPmzNHf04KTAKIuSN7F6dNp06ZNspM4ADQ8PNxxnYkPmZRNsqG8wC8IPEzAoIEmPmBS1i\/gsBHxCGDDTORdwpAvhefO2bNn9ZkEw4Z3yH9JuNLNxA9Myto9fBQwLojjWHokbG7YsEET65D7kDdvXqlRo4aKG6hgWvASQIQNUbjkmvP8s1XGxIdMyia3zbzOdwSSEjBomYkPmJT1HRXW7I4AVgfhpRlT2nFxcbpCFishsdeQdeCjq7ImfmBS1s6RpICxkzbrIgESIAESIAES8AgBChiPYORNSIAESIAESIAE7CRAAWMnbdZFAiRAAiRAAiTgEQIUMB7ByJuQAAmQAAmQAAnYSYACxk7arIsESIAESIAESMAjBChgPIKRNyEBEiABEiABErCTAAWMnbRZFwmQAAmQAAmQgEcIUMB4BCNvQgIkQAIkQAIkYCcBChg7abMuEiABEiABEiABjxCggPEIRt6EBEiABEiABEjATgIUMHbSZl0kQAIkQAIkQAIeIUAB4xGMvAkJkAAJkAAJkICdBChg7KTNukjgMQkcOnRIvvnmG5k3b57e4YUXXpCnn35a7t27J2nSpJFatWrpfzhB29u2fv166dGjh0yZMkUPlUuOjRo1SpYtWyYrVqyw7TDUnj17yty5c\/XQu+zZswtOGS9ZsmRymvtY11y9elWmT58u\/\/33n44TxmXr1q2PdS8WIgESSJoABUzSjHgFCfgNAeuk7O3bt0vWrFlVwHz\/\/ffy2WefSXh4uOBk7FSpUnm1vRs3bpT+\/fvLuHHjpEiRIsmqC9euWrVKfvjhB1sFDE7y9YWI6NWrl0Do+aLuZA0ILyKBICBAARMEg8guhA6BcuXKyalTp2TXrl2SMWNGR8dbtmwpmzZtkpEjR0rt2rVDB8hDeooIDAQERIzd9sUXX2jEiQLGbvKsL5QIUMCE0mizrwFP4M0335STJ08mEjB9+\/aV2NhY+fjjj+Wjjz4K+H56ogMUMJ6gyHuQgP8SoIDx37Fhy0ggEQF3AgZ5Hjt27NC8lOLFi8vSpUt1amnmzJmC6Yw9e\/ZoXkb+\/Pn1usWLF8u5c+fkn3\/+kVatWsWL2ty9e1dmz54tx44dkzt37gjybzp06CAVK1bU9pw5c0ajC5g+ioyM1N9dvHhRhgwZInny5NGfUcfnn3+u9cEOHjwoixYtkvfff1+yZcvm6NfatWtlw4YNkjZtWjly5IjkypVLPv30U0mfPr3cvHlTfvzxR1m5cqUUKlRI81fGjx8vv\/\/+u9SoUUMGDhwoKVKkcOslDxMwmzdvluXLl8uTTz4pe\/fulUqVKkm7du20v6gP9SLChXqGDRumLDB99+WXX2r\/Bw8eLL\/++qv2D9NjOXPmjNcORmD4x0sC3idAAeN9xqyBBDxGIKGAuX37tkyePFmGDx8umF6aNm2aCoJBgwbpQxfCA6ICggS\/u3Dhgia2jh07VhN+cf2AAQNkwoQJUrVqVW0n8luQX9O+fXv9GVEdCB6IDZRH2S1btuj9GjZsqNdAJOGB36VLF\/0ZD\/h33nlHxRSEy5gxYzRyhLyQfPny6TVIeF2wYIEjL+b69evSuHFjiYuL02Rf2M6dO6VFixby3HPPSd26daVKlSraZkSbpk6dKuDhztwJGAiU+fPna5+RL4S8oe7du6sAQyI0BBvqArcGDRpInTp15Pjx49K0aVN566235MUXX5T69evL2bNnpV69evofBIuzUcB4zOV5IxJwS4AChs5BAgFEwBIwpUqVkvv372s0IF26dPrgbdu2rTzxxBPaG0Q\/sGoJkRKsWLKsWrVq0qdPH3n11Vf1V+fPn5fSpUvrfxA2yNno1KmTbNu2TVKmTKnXHD16VEaMGKERDwgbJBA3atQonoBp1qyZhIWFqbiAMDpx4oRAkFhJvhA9WIlkCRgIofLly+tqJogWyxAZgWCB8IIYQh8LFCigggIiDYaoUZkyZZKcLnMlYBApQr2IRlltQzu7desmzZs31ygLDPcvWLCgCiXLIiIiJEOGDLJmzRrH7xD5unz5skNwWR9QwATQHxWbGrAEKGACdujY8FAk4G4KKSELRFFmzJih0YTUqVPHe\/BXr15dnnrqqXhFIIIgJvDg3b9\/v8yZM8ct3n379qlgco7AIMLTu3dvnebBPRKuTvr66681KmMJGPz\/gw8+0CgMIkeWQbCgbIkSJeS7777TX2P6CBEP1AfDcmV8jvJosztzJWBQL6aKwOVhS87Lli2rwglTcJYhEoNpLSs6hN+3adNGp6B++eWXeM2ggAnFv0722W4CFDB2E2d9JGBAwETAIGcDUz5Llixxu38Loh\/YxwT5M+7MlYDBtZiSGT16tCCHBg\/2rl27OkRCQgEza9YsjQTh+po1a8arCrk2N27c0GkqVwLm2rVrOjX1OAJm4sSJmtPiLOxc9fNRBMzu3bs1H8bZKGAMnJxFSSCZBChgkgmKl5GAPxAwETBIkkWeC3JemjRpEq87t27dUtGAiAaWY2OaCBuxORuiD4jmuBIwmIbB9agD01dI4o2OjpaYmBi9RUIBY00VQeR8+OGH8epBnkmOHDl0CszTAgZTQlixBYH20ksvuewffkkB4w\/ezjaQwMMJUMDQQ0gggAhgN9nTp09rcmumTJncttzVFBJEBvI4kMesmQeIAAACr0lEQVSCVUrYydcyJLAi7wR5MJiq6dixo3Tu3NnxOaZIsEIHD3ZXAgYrcazl24jAQCBB8CxcuNClgLly5YquYMIOuVjxYxnaiA35kP\/SunVrlwLGZAoJq7EwFfT6669rvo616R9ygdAO5MFQwATQHwSbGtIEKGBCevjZ+UAjgIf7pUuXZN26dY4lyq76YO0Lgwc28lsss5JpsaoHybPIOcGDG6ttkCiL6ZnKlSvL33\/\/rdEaJLUePnxYi1s5KNZUFOqwHvhI6oVoQr4KDCuXsDQaib8wq16sAELdMERBPvnkE12GjBVLMIgKrBDCyiWIC4ghJNNCdGCTPpiVeIzN+zAN5c7crUJCDszq1as1XwdJzf\/++68cOHBAML1kiTokNSMHxjkXCEuqIbCwGssyLEGHmARnZ+MUUqD9ZbG9gUiAAiYQR41tDjkCCc9CwgMWoiFh\/gjA4OE\/dOhQXa2DnBcIFeSMwCBYsFcMlg5jBRMe0oi8YHmyZSgHcYJpJIiQqKgovQaCAjsAT5o0SR\/iRYsW1T1bEJXBKh483CF4ECFClATTURAEEEgQH9i\/Bcm\/EBCFCxfW6rBLLpZD582bVyNDiNpg6gnTUbgH8lUwlZQ5c2YVK4icYIoKe7hg7xUIJOckYGfHcCdgUAf44GgD1FGhQgVdRo36ERmCsMI0G9qAOiHssIwcScpYto768TswwOe4B8qDNfavgVHAhNyfKDvsAwIUMD6AzipJgAS8T4A78XqfMWsgAV8SoIDxJX3WTQIk4DUCEDDYswUJxXZbv379NMLDs5DsJs\/6QokABUwojTb7SgIhRAACBknJ2Gwud+7cmjRcrFgxrxHAVBKmuzBFhXqxzwwFjNdw88YkIBQwdAISIAESIAESIIGAI0ABE3BDxgaTAAmQAAmQAAn8H86kxUZnLg+AAAAAAElFTkSuQmCC","height":337,"width":560}}
%---
%[output:8e05dc72]
%   data: {"dataType":"text","outputData":{"text":"\n================ RESULTS SUMMARY ================\n","truncated":false}}
%---
%[output:59d82c0c]
%   data: {"dataType":"text","outputData":{"text":"Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n","truncated":false}}
%---
%[output:91d40384]
%   data: {"dataType":"text","outputData":{"text":"\nFusion weights estimated from validation data:\n","truncated":false}}
%---
%[output:867db4b5]
%   data: {"dataType":"text","outputData":{"text":"SIM 1 weight = 0.8687\n","truncated":false}}
%---
%[output:6ccb57e5]
%   data: {"dataType":"text","outputData":{"text":"SIM 2 weight = 0.1313\n","truncated":false}}
%---
%[output:40ef520f]
%   data: {"dataType":"text","outputData":{"text":"\nMore reliable validation site: SIM 1\n","truncated":false}}
%---
%[output:025a72b1]
%   data: {"dataType":"text","outputData":{"text":"\nValidation-selected gate:\n","truncated":false}}
%---
%[output:107d62ce]
%   data: {"dataType":"text","outputData":{"text":"Disagreement threshold = 26.093 m\n","truncated":false}}
%---
%[output:29531e26]
%   data: {"dataType":"text","outputData":{"text":"Validation gated P90   = 337.14 cm\n","truncated":false}}
%---
%[output:7c40fed1]
%   data: {"dataType":"text","outputData":{"text":"Validation gated mean  = 165.90 cm\n","truncated":false}}
%---
%[output:89026cf4]
%   data: {"dataType":"text","outputData":{"text":"Fallback estimate      = SIM 1\n","truncated":false}}
%---
%[output:9c6ac2fa]
%   data: {"dataType":"text","outputData":{"text":"\nTest-set gate operation:\n","truncated":false}}
%---
%[output:9ea5755c]
%   data: {"dataType":"text","outputData":{"text":"Gated episodes = 1.43 %\n","truncated":false}}
%---
%[output:51205de6]
%   data: {"dataType":"text","outputData":{"text":"Weighted episodes = 98.57 %\n","truncated":false}}
%---
%[output:8afcbbdf]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFRCAYAAABqsZcNAAAAAXNSR0IArs4c6QAAIABJREFUeF7sfQl4jcf3\/4k9JXaNoNSShqgltYUmRSlFCSpKKd+mlrYoVaqW2lpULZWqrVV77D97KbVvRanWvgRtqaWWEDRByP85k\/97e+\/NXWbufd\/7vvPeM8\/zfeqbnJk553zmnfnkzJmZgPT09HSgQh4gD5AHyAPkAfIAeUAiDwQQgZEILVKVPEAeIA+QB8gD5AHmASIwNBDIA+QB8gB5gDxAHpDOA0RgpIOMFCYPkAfIA+QB8gB5gAgMjQHyAHmAPEAeIA+QB6TzABEY6SAjhckD5AHyAHmAPEAeIAJDY4A8QB4gD5AHyAPkAek8QARGOshIYfIAeYA8QB4gD5AHiMDQGCAPkAc89sAvv\/wCb7zxBlSpUgVWrlzptJ1vvvkGJk6cCN27d4cBAwZ43J9SkbdfrzuiBsgD5AHDeoAIjGGhIcXIA+49kJaWBs8995xTwcKFC8OBAwfcN+ShxN9\/\/w3Lli2DokWLQrt27Zy2gjrs3bsXqlevDlFRUR729l81XgJz7NgxmD59Ouzfvx\/u3LkD+fPnZ2SrS5cuUKtWLdbghAkTYMqUKZbGc+TIAUWKFIHatWszwlW2bFnL72JjY+HQoUMO9X\/11Vdh6tSpXttGDZAHyAN8HiACw+cnkiIPGNID1gSmYcOGkDNnThs98+bNC6NGjdJNd7zoG\/+XJUsWVXXgITBbt26F9957Dx49esQICRK969evw5kzZyBr1qwwadIkaNasmYXAPPPMM1C5cmVITk6GEydOwM2bNyF79uzw7bffQt26dZn+CoGpWrUqFC9e3MamiIgIiIuLU9VOaow8QB5w7gEiMDQ6yAMSe8CawGCUARdqR2X37t3QqVMnwCgBRh6+\/vprtrC\/++67EBMTAx988AEcP34ccGHGhR0jKhgx6dixI7z00ktQrVo1mD9\/Poti1K9fH8aPHw958uQBeyKh9NOkSRO2wM+bN4\/VQ93st5A2b94MX331FZw7dw7y5csHWKdfv36sXSw7duxgupw+fRpy584N9erVg6FDh0JQUFCmfu1tRtuio6Phn3\/+gfbt28OIESMgW7ZsTGzt2rXQu3dvqFixIqxZs4bphREYlFPI3oMHD2DkyJGwaNEiwCgW6hIYGGghMPHx8dC8eXOJRw6pTh6Q3wNEYOTHkCzwYw\/wEhgkELhAI8HBLZSwsDDAvBQsGDl45ZVXYNOmTfD7779D27Zt4YsvvmCkA+vglkrjxo3Zgv3dd98x8vDOO+\/A4MGDMxEJpU5ISAirV6dOHbZd88MPP9gQGOynZcuWjLjg7zHi8eOPPzI9ZsyYAefPn2eE5qmnnoJPPvmE6bJq1SqLbu4iMLhlhVtaSIZ+\/vlnRoCsy19\/\/QUlS5ZkP1K2kKwJDP4cSQxud2EkZubMmfDyyy8TgfHjb41MN54HiMAYDxPSiDzA7QF3OTD9+\/dn2yjKgo8LOS7uGE1AAnHkyBFLYq2y6JcvXx7Wr19vqaNEWnB7CnNKWrRoAQULFoSDBw9mIjBKPxjtwKgFEhks9km8GPlBwoRRDozyIFkYMmQIPH78mEVBLl++zNpGkoEkCHNtMKJSqlQp2LZtm9sIzPLly+Hjjz+GSpUqwerVq1360xmBwUqdO3eGXbt2Md1we8jVFhJGt3BLigp5gDzgGw8QgfGNn6kX8oAmHnCXA4PbQ5gboxCLChUqsGgIFkxQ\/emnn9g2DZISjErgNk2xYsUAt4KUOpgXgtEPJSqBbWDBLSckNNankJQ6mPiKbSvFnsA0aNAALly4wBKAcXvKvty7d49FgbZv387yVgICAuDhw4dsOweJlrsIzIoVK9h2VHh4OKxbt85jAvPmm2\/Cvn374LPPPoMOHTpYCIyjBnEbrk+fPprgTI2SB8gDmT1ABIZGBXlAYg\/wbiE5WvDff\/99tm0zefJkFjm4dOkSy3exJzDWJAATXDFPxh2BsT9WbU9gMI\/mzz\/\/hCVLlkCNGjUyITBs2DCWO4Pk68MPP2S5N0gmeAkMnhTCaAluQSEBUfJqlI4w\/wZ1xC01ZxEYJFG4hYQ2L1y4ECIjI2kLSeJvhVQ3nweIwJgPU7LIjzzgCwKDW0eYg4InmpTEXmdEwllkxJ7AdO3aFbZs2QJIVHCbBk8qdevWDW7fvs2OIuO\/MU8G\/41bMxg16tWrFxQqVIhFX9xFYHArCkkSkjKMEOFWFZ4owoJJvBgtQfKC21zYh30SLyYBDxw4EDCSg8nIKIcnqZQtJEri9aOPjEw1rAeIwBgWGlKMPODeA+62kLAFzIO5du1apgvneCMweOoH82Jw2ychIQEuXrwIWBe3aOyJBC+BUeSwbcyHOXXqFCMWmOcyd+5cRlaQtGCfmEA8bdo0uH\/\/PttOwq2lZ5991u0FerjVhOQI82uQrKANN27cgJMnT7Jj1EhCmjZtmukYdWpqKtseu3r1KuTKlYvpo0SJiMC4H5MkQR7wlQeIwPjK09QPeUADD7hL4sUu8YZczB+xvzGXl8DglhFuMeHpoLt377LTQZhoi4nAnhIY1AsThTEyk5iYyE4jNWrUiN3Si5EeJEkfffQRSzJGsoLHp1Fu9OjRTBbr8dwAfPbsWUZ+MHJ069Yty0V2GOFRSIn9RXaoGxIe3D7q0aMHlClTxoIcERgNBjE1SR7w0ANEYDx0HFUjD5jdA+62acxuP9lHHiAPGNsDRGCMjQ9pRx7QzQNEYHRzPXVMHiAPcHjAbwkM7qUPGjSIJRLinrj9FewcviMR8oCpPUAExtTwknHkAek94JcEBvfXMbmvdevW7HZQIjDSj2MygDxAHiAPkAf8zAN+SWDwdMGTJ08gJSWFXV1OBMbPRj2ZSx4gD5AHyAPSe8AvCYyCGj4iRwRG+jFMBpAHyAPkAfKAH3qACAxHBEa5TtwPx4dhTW6W+jkcjp\/gVL+mPWez363\/5m1hG3jq8sg465inLo8MtW8sfE\/8PF54rLmrkGf3OMh245Q7Mfo9ecDigU7FbrN\/dwpJsvFKmWXppvMSERgOAoP3QODruP5UtLI5atl\/794oi7Qo0Sh0tjzUih\/gkpwQAXC9uJN\/1PdPwevfqz5FrHk\/49kGtYtW37faeqrdnp52Jy0bobY5mdpLWjo8088CK9aDffv3Q6sN\/2rev687IAJDBMbhmNPiQ7cmL9ippwRGqXu26Wo423QV\/HnivUw2FDxbnv3sVmjGX68DGpfOJFOlRBDkyZnV5ucYbds44lf2s5v\/v64jByGJcifj7GPmqcsjo2b7aDe+98NTXOl2d2971oS9\/521GxjWGwJyFISsQc9ZRHhs55Hh8Y8zu0XbjyiS+UFKHl\/qJcP7fV8ZXl8vFTXpFxfyyFq1NGnbVaMpx7f7rE8kLLkq1mP9FYgdxv7Li7fPlFSpIyIwRGAcDiW1B3yP7V3h9+sZxABLxNPVITJ+gOX\/7+s9FvYk2oY87RV7sVwBy4+w7qXuWbg+gwGNn4W0m\/u5ZNu3bw+LFi3ikvVWKOX0JHh0Y5+3zUhZP+BxVQgMc\/1y8+VhGZMwFfKAWTxQoG3mCInatimkxbpdtedztXX2tD2\/JDB4tTi+g4IPyOGjbTly5GD+mzVrFtSpUyeTL80KvqtBM2LECPbQnjdFibiErm\/JIiVK2R17iP1z7MY\/LD8bu\/GCy65uTeT7SzDldHymdpAoGLFkLxxpSgKTdi2VkZOkZY4n67SrqUaEg3Ry4oHio\/ZCetpDU\/gHX1\/Hx0H1KLnC6+rRLevTrGuYXxIY0VFkVvBd+eHChQtQunTmbRelzqwT3zqsPuv4jEw\/V7aK9vcey7ZlkMAU7LvNYf2ocgXA1b7\/w8vr4cEf8x3WdRfNQMLgruBDfviAn69KtkKRgNsoviz\/HlwDd9Z9ZdOlp3bzhMYxpO1pCRnueJx42p59PXfjXK1+jNYO2W00RLTVx6xrGBEYjnFjVvBFCEzC6bnw4PEDwL1+\/J99PourtpDAFAs7BksaJmcSiyqXH15Imwu18qt30sLR1gQvSdBjYk89sYNjFAIYeUvFWWjcUTiby1gfCemBt49Mc9kN2W0EFHyng1nXMCIwHGPIrODzEJheO7rD4X8O2ogiIVESaF21gYmQL0z9GHpNbcnhZTGRvC8udlghWyHvEvR8NbErpEVPUlJsxH+JhVeuXIGQkBAxEP6\/tJ6hcY8UtqrkK7y91VPt+mS32h41dntmXcOIwHCMO7OC747AfHvlG9j1d+bseWVL6FboaacndULXx0D1FouhevP\/EmLdbeEE1fFN8qw7u11tnXEMF7cieLLDfutFZJtFi20VWtDcwmYqAcLbVHC6NcasaxgRGLfQmzcBypnpuD00IcsSOHr2PMwpO9BGLDL+Eyh4Noz9bMkrtltCtfKfhg+eXc1+V6pObpt6RiAnHFCDlhN72rVz8FfPchY1FNKiBSHhsdVaRku7RXXxpTzZ7Utv69+Xv+JNBEb\/saebBmYFX3EobhNhwdwWJQlXISrKTbaVPvsOOq4sY8Gg+ZTDmfD4s8237FRNwRauTxTpBiRHx55McMp20JP7t+Hql3zbZUa7FdMTuzncaXgRstvwEKmqoL\/ibdY1jCIwHJ+H2cA\/fD3jGDOWXtu7OfUAbhXhyaHH2z6AvZUc3+K4tkcEq1\/heMZ\/sfgTgXG0HeRuSBmNvKC+\/jqxk93uRqu5fu+veJttDVNGJREYju\/TTODznh5C8qJsEV18+hHg8Wb7Uih3dpjduSLcWvPfcWvMc5Flu8gR9DwTnPXtpEoui3UOixG2hDiGtY0Ij92ibcogT3bLgJJ6Ovor3mZaw6xHAxEYjm\/DDOBvubgJCuYqBBh9cXRXC24ZWSIoZ8Pg7tePIX\/+DNKCN9k6K9bkRebIi2KfuwnufGxAJlfgEWKjHxd2N8zd2e2uvqy\/J7tlRc4zvf0VbzOsYY4QJwLD8R3IDv7Ew2NhReJSZileIpfQ+jxszpdgeScIf55zQ2\/YXOM+k8FoS3yTfC4vsrMmLljHDOQF7VAmOHx4zdHDaMpwsT6CLPMxYl7ixvGZSCnirwsa2S3lcPVYadnXMGeGE4HhGBIyg2+\/ZdRj6XK4sPNuJqvHd7jJfqZc2e9sgnuS8jfcP9zP5gp8s5AXawLjKNKiOM2IOSwcw9ilCC1o3npQrvqEt1x4eautzGuYK9uJwHCMDFnBd5TvYv0CNJqOeS6Y44KvNVtvFTma4O7tfxseXvvvXhgzERfrSESuuXGWe1rMSFYcDXla0DgmAhOJEN4mApPDFFnXMHemEYFx5yGJH8Kyv0XXmrwcqpAK2174b8vI\/v0h+wnu\/m\/94cFfy5m3ZE\/UdQY5Hoe2vhnXDLktHMObidCCxuspc8gR3ubAkdcKIjC8njKhnMzgx4zuAE+yPobaEwfZIINbRsoR6BfL5s+Emv0EZ7ZkXcVgzHXBknp8u83tuJjjYobcFt7PkRY0Xk+ZQ47wNgeOvFbIvIa5spEiMBwjQFbw8cXoq\/X\/e4H5YrE02FvxX7Zl5O7VZ5zgij5cY\/FOyulJ7N9m2jZylOcSUDYSSn\/xM8eoMJcILWjmwtOdNYS3Ow+Z6\/eyrmHuUCAC485Dkm4hKfkv1lf\/K4m6P\/WpBtVK5nVpOU5w+Y6+bCODrzzzvurM4VZdRazJi\/KScr5Xe8KfN5Jdnr7SVWkNO6cFTUPnGrBpwtuAoGioEhEYDZ1r9KZlA98+eXfMH+shun8wFOy7jblaOWnkyu8Xj62C3Oc\/ZPku2QplRHHMQF7soy72Sbo0sRv9a1RXP8JbXX8avTV\/xVu2NYx3HFEEhsNTMoFvT16a9J4JDQaVgKbnjzFLW1UNhu87hbu02qz5Lmi0NYFxdMLIXyc4sptjIjCRCOFtIjA5TJFpDeMwxyJiGgKTnp4OEydOhCVLlkBqaipUqVIFxowZAyVKlMjkjwcPHsCXX34JGzZsgPv370PNmjVh9OjRUKRIEYe+kwl8awIT+fUAKHimvOWoNE\/0xUzPAtiD6Y68oDxN7CLTh\/yyhLf8GIpY4K94y7SGieBpGgKTkJAAM2fOhPnz50NwcDCMGzcODhw4AKtWrcrkjwkTJsDWrVthzpw5kDdvXujfvz8kJyez\/++oyAD+vJPfQ6XCVW0eZ1SOTdtfUudsgKRd3wPJP3dk20Y3QkabJhfE\/ni0q7td\/HWCI7tFpk35ZQlv+TEUsUCGNUzEHkXWNAQmNjYWYmJioGPHjsy2lJQUqFq1Kqxbtw5CQ0NtfNO8eXMm98Ybb7CfX7p0CaKjo+GXX36BwoULZ\/Kj0cHH00b27xu92uc7yJKWDZTj0o6OStsbmrzjNUi7c5ydNDLLBGef8+Lubhez2C06GZDdoh6TW57wlhs\/Ue2NvoaJ2mM6AoNkZfr06RAZ+d+x4QYNGkDfvn2hWbNmNv5BotO+fXto164d+3lSUhJUq1aNbT\/VqFHDFAQGoy8T3rwJ6QF8Sbvw5AHcWlee2W4GAvM4+Tokb5xqec8IX4vmeSWaJnZPpxI56xHecuLmqdb+ijcRGE9HjI\/qhYWFweLFiyEiIsLSY9OmTSEuLg7atGljo8WkSZNg06ZNMHfuXMiTJw+MHTsWFi5cCLNmzYKoqCiHBMb6h507d4ZOnTr5yDL+bmasWwRpS0Kh4NkwuP\/UE5jWKgkO9XL+krR1y8qR6ftlJkBa7ggWlXKUP8SvjY6SM96E9HP7LAoEvNIbAP\/HUaS2m8M+ZyJktxfOk7Aq4S0haAIqz5s3j61v1uX8+fMCLcghapotJIzATJ48mW0FKQXJyODBg6FJkyY2aChJvEhicufODW+\/\/TYMGzYMli9fDs8\/\/7xDAmNU8DFpt+3Bz+DeHNtkZd68F8VYTN61fiJA1r9UrLeMMOqSq2I9KBA7jPtrlNVubgOdCJLd3npQrvqEt1x4eastRWC89aDG9XE7CLeMunbtynrCbSE8XYQkpXTp0i57P3nyJLRu3Rp+\/fVXCAwMlIbAWJ84sn+kEQkMz30v1gQmqPY8yF4kgwDKOMFZkxdPnwKQ0W41Pi2yWw0vytMG4S0PVmpoSgRGDS9q2AZGT+Lj49kppJCQEBg5ciQkJiayvBYsmzdvhmLFikF4eDh89913sH\/\/fhaxefToEXTv3h2ee+45GDEi410c+2JE8O3ve0ECU+rFPPDlU5fZUwFxLxaH8a8\/x+Vx5eh03hcXQ7ZCtaQkMPimUdLS4Ux3b16Qpomda8iYRojwNg2UXIb4K95GXMO4AHMjZJotJLQTCcyCBQvYCSRMxsV7YIoWLcpc0KpVKxah6dmzJzsy\/cknn8DPP\/8MeH8MJvkOHToUcubMKSWByXO1GLz0+SgYdK2y0G27aOy\/Rz6F1D8WQLZ8FSFv3XUW+2X50K8Mr890Tjm+3Wvygg3IYrcaH791G2S32h41dnuEt7HxUVs7IjBqe1Si9owGvv2xaSX68uy4gtB8ymG3DzVau16Jvtg\/0mjkCc7+Xhdre7yJvhCBcb3VKtEny62qkcc5txEeCJLdHjhN4ipGW8PUcqWpIjBqOcW+HSOB32hlNLQLewtyDmkIm\/MlQOj6GKbuwh734PLtB+zfa3tEAM+9L3f3toNHN\/azOjIQGNwmSj2+3RJtUXAq9L9JkLN0VcgVXtfrIUATu9culKoBwlsquLxW1l\/xNtIa5jWIVg0QgeHwppHAV3JfbJJ282eB8c2uM0uiyhWANe9X5bAKwFn0xYiRCPsL6XjvdeFyhJWQv05wZLfoSJFbnvCWGz9R7Y20honq7kqeCAyHN40EvkJgQte3ZNGXSm0LQOfsicwK0VNHWCcwrI\/DV6aNNMGlHt8Gl4e\/zGz09HQRB8xMxEh28+qshhzZrYYX5WmD8JYHKzU0NdIapoY9ShtEYDi8aRTwrU8eKQTGk8TdtJv7IXlPxi3E9ltHijuMNMEp0Rdv81s4oCYCw+MkE8kYaZz70q1kty+9rX9fRlnD1PYEERgOjxoB\/ImHx8KKxKUWbZUtpKzz8sLYjRe4IzApp+Mh5fQkJu8s+mKkSATPC9IcEHKL0MTO7SpTCBLepoCR2wh\/xdsIaxg3SAKCRGA4nGUE8K2jL5Hxn7DnAqyjLwMal4YBjd0\/G3DvQBd4eHWLS\/JiBAKDx6OVo9Fa5bs4gt5fJziym2MiMJEI4W0iMDlMMcIaxqGmsAgRGA6XGQF8+1t3X+ofDC0un2Daq5W4a+0KPSc4+4RdX2wdKbbraTfHUNRMhOzWzLWGbJjwNiQsmillhDVMC+OIwHB4VW\/wf79xGHps68I0VbaOBl6tDIU+2sZ+xpu8a7195Cz3Re+F3PpGXdTFl+TFCJEnjuGoiQgtaJq41bCNEt6GhUYTxfRewzQxCgCIwHB4Vm\/wMfpS6Gx5uBl6ihGYY6MD4MfjN4TICyM6azIuKstR9BXIU\/Nbl5brNcH5MmHXkQP0sptjGGoqQnZr6l7DNU54Gw4STRXSew3TyjgiMBye1Rt865t3kcAoL03z5r2giY\/vHIU7O1rYvDjtynS9JjgiMBwDUgMRvfDWwBShJsluIXdJL+yveOu9hmk1cIjAcHhWT\/CP3TwCzxeqDKODjzBNFfLCIioTM94B4ilK9CV74UgIqrPIbRU9PnTliYACbYdDgdhhbnXUQkAPu7WwQ7RNslvUY3LLE95y4yeqvZ5rmKiuIvJEYDi8pSf4uH2knDqyJjAi5CXt9hFI3pnx5IC73BfFHb6a4B5ePAZP7t6Ey8PqWZDQ+rI6V5D7ym6OYedTEbLbp+7WvTPCW3cIfKqAnmuYloYSgeHwrp7gK6ePFBKDERgR8oLmuXoywJn5vprgfPVEAAfMTMRXdvPq4ys5sttXnjZGP4S3MXDwlRZ6rmFa2kgEhsO7eoJv\/XTA6gJRTFsRAqOQF5Hoi68WcuWuF7znJVfFerptG1kPAZrYOT4IE4kQ3iYCk8MUf8VbzzWMAxaPRYjAcLhOT\/BjRndgp4\/C1rSBlUUiYUnXyvBKhUIcWmeIeBJ98RWB0Tth15ET\/XWCI7u5PylTCBLepoCR2wg91zBuJT0QJALD4TS9wFeiL3iEulb8AJbAKxJ9eZJyFW7\/VJtZyJv7orhD6wnuYs8y8OhaxhMIvr7rxRXkWtvNMdx0ESG7dXG7bp0S3rq5XpeO9VrDtDaWCAyHh\/UC3\/r4dPKaeDgyP2MLibco0RdXbx45a0vrCc6I0RdfRZ548fOlnNZ4+9IWkb7IbhFvyS\/rr3jrtYZpPWKIwHB4WA\/wtyVehNenJkJ04ZmQljMVjiV+AH9Oq8uhbYbI3b3t4dGNfezfotEXrRdyXz\/QyO00SuIVcZUpZP11QSO7TTF8uY3QYw3jVs4LQdMQmPT0dJg4cSIsWbIEUlNToUqVKjBmzBgoUaJEJvc8evQIRo8eDTt27ICAgAAoUqQIDB06FMLDwx260tfgd1ozHM4\/WAvZN\/WGV9ZUZTrhw40iRSEwQbVmQvbgBiJVmaxWE5xy1wv2YaStI8VBWtktDICPK5DdPna4zt0R3joD4OPufb2G+co80xCYhIQEmDlzJsyfPx+Cg4Nh3LhxcODAAVi1alUmX37zzTeMvMybNw8CAwPZf2fMmAF79uwxBIFx9vK0yKDwNHlXy4XcOvKi52V1rvxIE7vIKJNflvCWH0MRC\/wVbyIwIqNEB9nY2FiIiYmBjh07st5TUlKgatWqsG7dOggNDbXRqGfPnlCyZEn4+OOPLdGGBg0awG+\/\/QZ58+bNpL2vwbd\/ebpcw7zQNuFZbq8++HMh3P99MHiS++ILAqPnRXXunOivExzZ7W5kmOv3hLe58HRnja\/XMHf6qPV700RgkKxMnz4dIiMjLb5BUtK3b19o1qyZjb9wm2nOnDksWlO4cGHAiMyWLVtg5cqVTiMw1r\/o3LkzdOrUSS0MbNpZ8fdyWHllMftZ016zoFzLnFBjQE6hvvIdfZnJ36m0VaietfClS5ccbr953OCMNyH93D4IKBsJ0H2hx81oXVF1u7VWWKX2yW6VHClJM4S3JEB5qCbuKsydO9em9vnz5z1szbjVTENgwsLCYPHixRAREWHxdtOmTSEuLg7atGljgwDmy\/Tq1Qs2btwIuXPnhly5csGsWbMMkQNTf3ldeJR+L4PA9JwtnPuS\/uAmJG2szup7kryrOErtv9AuD64NqWf2GTLvxXpwqG23cT99W83IblmQUkdPwlsdP8rSCkVgDI4URmAmT54M0dHRFk2joqJg8ODB0KRJExvtR44cCRcvXoQJEyawLaMNGzbAoEGDWBSmYMGCmSz1JfjW20e7Yw8Je92S+9L8LEBANuH6WhEYzH\/BG3dDhm\/zWCdfVKSJ3RdeNk4fhLdxsPCFJv6Kty\/XMF\/gqPRhmghMu3btALeMunbtymxLSkqCmjVrwqZNm6B06dI2PkViM3DgQJutpcqVKzMCVLdu5qPKvgLfmryU2foqzJs2SngseJu8qwWBMfKxaXsH++sER3YLf2pSVyC8pYZPWHlfrWHCinlZwTQEZvny5RAfH8\/yWkJCQgCjLImJiexYNZbNmzdDsWLF2DZRt27dIGfOnCwCkyNHDti7dy\/basIITPHixXWJwEQvqw7pkG7p25Pto0f\/bIO7++JYG95sH2F9NSc4o15a5+jbUdNuL79Nn1Ynu33qbt07I7x1h8CnChCB8am7PesMCcyCBQvYCaQaNWqwe2CKFi3KGmvVqhWL0OAJpOvXrzOCg6eOsmXLxvJg+vTpAw0bNnTYsS\/At751F8kLFtG7X1JOx0PK6Ulekxe1CMythYPg9soxzJb8LQdAwQ5feAasD2vRxO5DZxugK8LbACD4UAV\/xdsXa5gPYbR0ZZoIjJbO8xX4o4OPwL+Fr8NTN4oIkxe0X63tI7UIjAz3vtiPG3+d4MhuLWcQ47VNeBsPEy018tUapqUNjtomAsPhcV+Av2v8Ndg17hrT5u1NoRBSJZBDs\/9EFPIKYk00AAAgAElEQVSCP\/F2+0htAmPke1+IwGR4gBY0oc9NemHCW3oIhQzwxRompJBKwkRgOBypNfiHrx+CXtu7Qej6lhC6Psaj6IvydED2wpEQVGcRh1WuRdSY4DAC81TVxlB08I9e6+OrBtSw21e6qtkP2a2mN43fFuFtfIzU1FDrNUxNXUXaIgLD4S2twbc+fZRQfCeUqpObQytbEYzAZCsYAXmjVgjXdVTB2wlO2T4y6pMBzpzkrd2qOF+HRshuHZyuY5eEt47O16FrrdcwHUxiXRKB4fC8luBbJ+8GQABMefwTVG5XgEOrzARGregLtuzNBCfTsWl7R3tjtzBoBqpAdhsIDB+oQnj7wMkG6kLLNUxPM4nAcHhfS\/DLT+oPhYtnXPnvydFprKecPsr74mLIVqgWh0XuRbyZ4BQCI1Pui+IRb+x271XjSpDdxsVGC80Iby28atw2tVzD9LSaCAyH97UE3\/7hRtGj06h+0vqKkJ72ryrJu94u5Ap5yd\/qEyj4ZsYRapkKTewyoeW9roS39z6UqQV\/xVvLNUxP\/InAcHhfS\/CfGzQLno6YwrRo9ck8+OhcRQ6NbEXUPD7tDYGReevIG7uFATNgBX+d2MluAw5GDVXyV7y1XMM0hMtt00Rg3LoIQCvwMf8Fj07jySMsnkRfsB4SGDXzX7BNTz50hcA8M\/kMZC8ayuFZ44l4YrfxrBDXiOwW95nMNQhvmdET112rNUxcE3VrEIHh8KdW4OP2UaGz5dnx6YJnw6QmMEnLRkDS0uFSPNjoCnKa2Dk+CBOJEN4mApPDFH\/FW6s1jMPlmooQgeFwrxbgn7x1HLpu6cR6D\/+\/DlD+cGPoe1p8++jOT1HwOOVvCAzrA4FhvTms4RMR\/dD\/\/qQGPDh3EMos++89J76ejCUlarextPdcG7Lbc9\/JWJPwlhE1z3XWYg3zXBv1ahKB4fClFuD\/76f2kHj7DOsdIzBxFbtBdL9gDm1sRZT8l\/yN9kOWXE8L13dWQXSCk+nBRorAZPaAKN6qDTSdGyK7dQbAx937K95arGE+hs5hd0RgOFDQAnzr00ftv0iAHofKc2iSWUSLBF7sRfRDJwLjEXyGqSSKt2EU91IRsttLB0pW3V\/x1mINMwL0RGA4UNACfIXAYA7MhOrToFyjvBya2IqknvkG\/j01gf1QjfePrFsX+dAvD64DqWd+hpAhGyGwSiNhO4xUQcRuI+ntrS5kt7celKs+4S0XXt5qq8Ua5q1OatQnAsPhRbXB77WjOxz+5yDrGS+v67iyLJT08PkALciLSATm7rbZcH1qHLNF9vwXEbs5ho1UIrSgSQWX18oS3l67UKoG1F7DjGI8ERgOJNQGX43L61BtrbaPeBfyO2vGw835\/S0eJALDMZgMKkILmkGB0Ugtwlsjxxq0WbXXMKOYSQSGAwm1wbfePuqXY6JHybtGIDDK0WmzRF94iRvHkJFOhBY06SDzSmHC2yv3SVdZ7TXMKA4gAsOBhNrgKwTG07ePFJUxAqP28WmlbZ4JziyJu9ZDgMdujiEjnQjZLR1kXilMeHvlPukqq72GGcUBpiEw6enpMHHiRFiyZAmkpqZClSpVYMyYMVCiRIlMvm7VqhWcPHnS5ucPHz5kdWvUqJFJXm3wY0Z3gJuhpzx+vBEVTLu5H5L3tIN89TdB1iD1b711N8Ep5KVA2+FQIHaYUcaz13q4s9vrDgzaANltUGA0Uovw1sixBm1W7TXMKGaahsAkJCTAzJkzYf78+RAcHAzjxo2DAwcOwKpVq9z6+pdffoFBgwbBDz\/8ADly5NCUwEzc\/CekvHUHAp4ARPcP9nj7SHmBWu3TRzwRmNQTO+DysHpM1Ax5LxSBET827\/ajkkSAFnJJgFJJTX\/FmwiMSgNIq2ZiY2MhJiYGOnbsyLpISUmBqlWrwrp16yA01HmEAiMvTZs2hVGjRkGtWrUcqqcm+GFfDYTOX7Rn\/Xj69hHWvbu3PTy6sU\/149PuCIz1g41BL8dBkfe+1wpSXdr11wmO7NZluOnWKeGtm+t16VjNNUwXA5x0apoIDJKV6dOnQ2RkpMXUBg0aQN++faFZs2ZOfT5t2jQ4ceIETJ482amMmuAr+S94++7s2Z96PBa0PIGESjma4KwjL4EV60HI8G0e62\/UijSxGxUZbfQivLXxq1Fb9Ve81VzDjIStaQhMWFgYLF68GCIiIiz+xchKXFwctGnTxqHP79+\/D\/Xq1YOlS5dC6dKlXRIY61927twZOnXKeMdItLx1MEOX535sCZ8OyYgWeVLyHX2ZVbtTaasn1d3WuXTpUub8oRlvQvq5fRDw5Xm39WUVcGi3rMYI6E12CzjLBKKEtwlAdGHCvHnzYO7cuTYS58+bb942DYHBCAxGUaKjoy2gRUVFweDBg6FJkyYOoUaQN2\/eDPhfV0Ut9nrx3l\/QfkMr1tV7i5ZBhxVlPPqKUs99D\/8e\/5zV9WUOjBlPHdkD4K9\/oZHdHn2K0lYivKWFziPF1VrDPOpcw0qmITDt2rUD3DLq2rUrc1dSUhLUrFkTNm3a5DS6glGUhg0buo2mqAX+7zcOQ49tXZh+3hyh\/vfIEEj9I0Ez8oL6OZrgiMBo+CXq3DQtaDoD4OPuCW8fO1zn7tRaw3Q2I1P3piEwy5cvh\/j4eHYKKSQkBEaOHAmJiYnsaDQWjLQUK1YMwsPDLU4oX748LFq0yGbbyRFAaoFvfQPvmD\/Xe3wCSev8F1cExmzHpikCk+EBWtCMNjVrqw\/hra1\/jda6WmuY0ewyDYFBxyKBWbBgATuBhPe54D0wRYsWZT7Hu18wQtOzZ09LhKZatWqwa9cuKF68uEtc1AL\/mREzoVT4NMjyJCvsfOOAR2NBOT6NlbXaPnK0oJn13hciMERgXOW\/efSRSlCJCIwEIKmoolprmIoqqdKUqQiMKh5x0Iga4D9MS4eyo76H96bWZD14eoRa6\/tfFPPtJzh\/2D6iSITzRHatvi2926WFXG8EfNu\/v+KtxhrmW6T4eiMCw+EnNcAfu\/EPWJv8OhQ6Wx5qxQ\/wmMD4YvvIfiH3l+gLERgiMBzTgSlE\/HUh91e71VjDjDjwicBwoKIG+AX7boPw2v1Yb41GfQVDf3uJo+fMIroSmDdGQIE2Qz3SW5ZK\/jrBkd2yjFB19CS81fGjLK2osYYZ0VYiMByoqAE+3sBbpMQm1tvsoG0Q+mpejp5tRVLOTIaUUxM1e8DRujfrCQ4jMGa9uM4eBJrYhYel1BUIb6nhE1beX\/FWYw0TdrYPKhCB4XCyt+BbR1+wu92xhzh6zSziq\/wX7Fn50NOunYe\/epaFoPpvQ5H3Z3mkt0yV\/HWCI7tlGqXe60p4e+9DmVrwdg0zqq1EYDiQ8RZ8tQiMr7aPrAmMP+W\/WNvNMSxMJUILmqngdGsM4e3WRaYS8HYNM6oziMBwIOMN+C2m\/ga7E5Ms+S81pvaFr7Z14Og1s4ivCUzuVQPh3t4lfrN9RASGkng9+jAlrEQERkLQvFDZmzXMi241r0oEhsPF3oCPp4\/mnZ0ABYL3sp68uYHXpwTm2z6Q\/lM807nMsnQOL5lDhCZ2c+DIawXhzespc8j5K97erGFGRp4IDAc63oBvv33kLYHJWTIWclf9kkNr70SUrSN\/Sd5VvOWvExzZ7d33Ilttwls2xLzT15s1zLueta1NBIbDv56C\/8\/dh1B+2B7L9hF29eNL+yFPcDaOXm1F0m7uh+Q97XxyAin1xA64PKye30Vf0GCa2IWHptQVCG+p4RNW3l\/x9nQNE3awjysQgeFwuKfg4\/bR2I0XLAQmdH1LmD37U44eM4v48gRS0rIRkLR0uF9tHVEE5oLTR089GrCSVPLXBY3slmSAqqSmp2uYSt1r1gwRGA7Xegr+B0tOw4L9l6FJxbnwZ96j4A2B8WX+yx+d8sKTlLtEYDjGhllEaEEzC5J8dhDefH4yi5Sna5jR7ScCw4GQp+Bj\/guWfgmF4FboaWj+2ktQ95Ngjh4ziyCBCciSDQq8dtaj+ryVlNwXyJ4LyixM4a1mGjma2E0DJZchhDeXm0wj5K94e7qGGR14IjAcCHkKPhKYAY1Lw+NOyayX6P7BEN3PcwITGNYHAsN6c2jsmci18a3h\/v6VrHLA299B6aZdPGtI4lr+OsGR3RIPWg9UJ7w9cJrEVTxdw4xuMhEYDoQ8Af\/gn8nQKP4QRNQeDw\/gqlfbR6giRmCyF64NQXUWcmgsLqLkvWDNYiO2w5XAkpQTIe5GaWvQgiYtdB4pTnh75DZpK3myhslgLBEYDpQ8Ab\/oxzvgYdoTmxNInj4hkHpuJvx7fJSmJ5DOx2YBgHTLpXU0wXEMDBOJEN4mApPDFMKbw0kmEvFkDZPBfCIwHCh5Ar6S\/6K8QF1uY3OYM3M4R2+ZRXyRwKvkviiX1tEE5xFU0lYivKWFziPFCW+P3CZtJU\/WMBmMJQLDgZIn4COBiSpXAG4VeYf1UHFHa5jxzWCO3nxPYBy9d0QTnEdQSVuJ8JYWOo8UJ7w9cpu0lTxZw2QwlggMB0qi4Cv3v6ztEQEDfm3AeuhS4T343\/OeJcVqGYGxzn0pMf53yFGqMtOXJjiOgWEiEcLbRGBymEJ4czjJRCKia5gsppuGwKSnp8PEiRNhyZIlkJqaClWqVIExY8ZAiRIlHGKxdetWGD16NFy7dg2eeeYZGDp0KERGRjqUFQVf2T66NbE+RC2rxtpcWHwnlKyTW3hcKBfYaXUCyX7rSFGQJjhhqKSuQHhLDZ+w8oS3sMukriC6hslirGkITEJCAsycORPmz58PwcHBMG7cODhw4ACsWrUqExbnz5+Hli1bwvTp0xlpWbZsGZNbuHAhBAQEZJIXBV\/ZPlrzflUYHXwE9vceC5\/k\/Qpq93xaeFxoGX1BZYjA2EJCE7vwEJW6AuEtNXzCyvsr3qJrmLBjdapgGgITGxsLMTEx0LFjR+bKlJQUqFq1Kqxbtw5CQ0Nt3IuRl+TkZPjiiy+43C4KvnL\/y4DGzzICg2XQtYytGdGiJYF5dPk0XOxdHgq0HQ4FYofZqOavHzrZLTpC5ZYnvOXGT1R7f8VbdA0T9ate8qYhMEhWlIiK4swGDRpA3759oVmzZjb+bdeuHVSuXBlOnz4N586dg+LFi8OgQYPYtpOjguBbl86dO0OnTp2cYlZt8h\/wbeuiMOlKBpnCJwSGDs34t2jJd\/RlVuVOpa2iVd3L\/xQP6T\/FQ8C7iwDK1LKRv3TpktPtN\/cNyytBdsuLnSeaE96eeE3eOv6C97x582Du3Lk2QOHOg9mKaQhMWFgYLF68GCIiIiwYNW3aFOLi4qBNmzY2uOHPHz16BLNnz4aQkBCYNWsWfPvtt7Bt2zbIkydPJoxF2KuSwLu5T3X4YH\/Gi85YPLkDRq\/8F9TXX\/9SIbvNNsW5tofwJrz9wQMia5hM\/jANgcEIzOTJkyE6Otri\/6ioKBg8eDA0adLEBpP27dtDrVq1oE+fPpafY\/Rl2rRpUKdOHa8IzJ5zt6H5lMMwvstZmHV8BmurxL5oWDxhkvC4uLu3PTy6sQ8KtrggXNddhSvD60PK8e1MTLn7xboOTezuPGiu3xPe5sLTnTWEtzsPmev3RGAMjiduC+GWUdeuXZmmSUlJULNmTdi0aVOmK\/GR1GTNmhVGjhxpQ2AwCoPExr6IgK9EYJQL7LCtpr1mwaCrjrenXLlVicAQgfHd4KOJ3Xe+NkJPhLcRUPCdDv6Kt8ga5js0vO\/JNBGY5cuXQ3x8PDuFhNtCSE4SExPZsWosmzdvhmLFikF4eDgcPXqUJfsuWLAAnn\/+eZgzZw7MmDGDbSEFBgaqT2B6zvYoiffWGsy9SdckAvNn16Lw+PY1h9EXdIC\/fuhkt\/eTikwtEN4yoeW9rv6KNxEY78eO5i0ggUFSgieQatSowe6BKVq0KOu3VatWLELTs2dP9v+R2OCW0927d6FcuXIwYsQIRmYcFRHw7Z8QeGHW+\/D1hozbeEVLxgOOkRBUZ5FoVbfyzo5PKxX99UMnu90OHVMJEN6mgtOtMf6Kt8ga5taJBhIwTQRGS5+KgG9PYF5a0RdGL+rgkXpaHqEmAuMYEn+d4Mhujz5RaSsR3tJC55HiImuYRx3oVIkIDIfjRcBvMfU32J2YZHmFevw\/P0JkjyIcvdiKKPkveV9cDNkKZc7LEW7QqsLloS9B6sldEFT3LSjSc57DpmiC88bD8tUlvOXDzBuNCW9vvCdfXZE1TCbriMBwoCUCPkZgXiiZF1KLd2Mt73r9EARk4ejETkTP6AuqQhOcOGYy1yC8ZUZPXHfCW9xnMtcQWcNkspMIDAdaIuAjgXm24jR4Ku851rIn979gPa0JjKPbd61dQRMcx8AwkQjhbSIwOUwhvDmcZCIRkTVMJrOJwHCgJQK+kgPTL6EQewNp9aAEjh5sRRTyomUCb7ER2yBX+H8X7dkrSROcMGxSVyC8pYZPWHnCW9hlUlcQWcNkMpQIDAdaIuAjgSn1VC6I\/S7j5WlP3kBK3tEM0u6c0OT4tJK8SxEYx8DTxM7xQZhIhPA2EZgcpvgr3iJrGIcbDSNCBIYDCl7wlehLTNJuCF0f4zGB0fP4tOIOf\/3QyW6OD8JEIoS3icDkMMVf8eZdwzhcaCgR3QnMihUroHHjxpA7d0bEAsvJkycBHZ4zZ05DOIsXfCQwRZ7ZBEVKbGJ6x1XsDnHhGcm8vEXZPkJ5tW\/gVZ4PcBd9wb799UMnu3lHqjnkCG9z4Mhrhb\/izbuG8frRKHK6Exh07M6dO21ePi5btiysX78e8IFGIxRe8JHAlHhuPuQt9DtTe8qjn6DKmwWFTLDkvxSpA0G1xfNnnHWWtGwEJC0dzn7t6O0j+3r++qGT3ULDVXphwlt6CIUM8Fe8edcwIWcaQJgIDAcIvOAjgbF+A2lh8Z1Qss5\/kSV3XWn5+rSS+xJYsR6EDN\/mThWKwLj1kLkE\/HViJ7vNNY7dWeOvePOuYe78Z7TfE4HhQIQXfGsCE3irMPzUfSNH6\/+JaHV0WiEvvNEXlPPXD53sFhqy0gsT3tJDKGSAv+LNu4YJOdMAwkRgOEDgBd\/6DpiGdzrC8C4fcrSuLYER3TpStPHXD53sFhqy0gsT3tJDKGSAv+LNu4YJOdMAwkRgOEDgBR8JTP81RWBf3BgodKYCzJozhKP1\/0SS1oVC+pM0VZN3lcRd3q0jIjAXoHTp0kK4mUHYXyd2stsMo5ffBn\/Fm3cN4\/ekMSQNQWBef\/11CAoKsnhk9uzZ7PXo\/PnzW342dOhQ3TzGA\/6ec7eh+ZTDgBfYYYnuHwzR\/YKFdFb7+PQfbwXBk9R7TAeexF1rZf31Qye7hYas9MKEt\/QQChngr3jzrGFCjjSIsO4Epnnz5lyuWLt2LZecFkI84H+z\/SIMXZMIvSochpxDGnpMYALD+kBgWG+vzLDOefGEvGAdf\/3QyW6vhp50lQlv6SDzSmF\/xZtnDfPKsTpV1p3A6GS3ULc84I\/d+AeM3XjBcgqpS8X34H\/hXbj7STkzGVJOTQRvX5+2znlROheNvhCBoS0k7oEruaC\/Lmhkt+QDV1B9njVMsElDiBuGwDx+\/BiOHDkCf\/75J7vADu+Cee655wzhJB7wG046BL\/+lWwhMJPqToPqT9fk1l+tE0iX+laEhxdPCG8Z2StKExw3dKYQJLxNASO3EYQ3t6tMIcizhsloqCEIzKFDh6Bv375w8eJFGx9WqVIFxo4dqzuR4QFfeUZAuQdmcr1vIaJINe4xwfJfCtWEoBeXcNdxJKhsH3kSdbFujyY4r2CQrjLhLR1kXilMeHvlPukq86xh0hkFALoTGCQtTZs2hdq1a8MHH3wA5cuXhwcPHsDvv\/8OX3\/9NZw9exYw\/6VYsWIu\/Zueng4TJ06EJUuWQGpqKiD5GTNmjM0Nv0oDc+fOhREjRkCOHDksbVaoUAFWrlzpsA8e8JHABASkQ4XI\/qyNHpU+hPblO3KPCSQw3ua\/eHpk2pGSNMFxQ2cKQcLbFDByG0F4c7vKFII8a5iMhupOYIYMGQLHjh1j5CEgIMDGh2lpafDGG28AkovPP\/\/cpX8TEhJg5syZMH\/+fAgODoZx48bBgQMHYNWqVZnqxcfHs60qJDw8hQd8JDC5852DUuHTWJO7Yw\/xNG2RUYPAKEemvY2+oFI0wQnBJ70w4S09hEIGEN5C7pJemGcNk9FI3QlM3bp1YcCAASwK46hs3rwZhg8fDrt373bp39jYWIiJiYGOHTOiHikpKVC1alVYt24dhIaG2tQdOXIkYM4NRmF4Cg\/4SGAin78AyUFTWJObXzkAufJn5Wke0m7uh+Q97byKwHhy264r5WiC44LONEKEt2mg5DKE8OZyk2mEeNYwGY3VncDglhFGXzDK4qj88ccf8Morr7CtJFcFycr06dMhMjLSItagQQOWW9OsWTObqh999BH89ddf8O+\/\/8L169dZjs2gQYMgPDzcYRc84Nu\/RC0SgVEjgVet3BfFATTByfg5e64z4e2572SsSXjLiJrnOvOsYZ63rl9N3QkMOhajK85yXC5fvgxRUVFw\/vx5l17Cl6sXL14MERERFjmM6sTFxUGbNm1s6s6bN48Rl86dO0PevHlhypQpsHDhQtiyZQv7\/\/YFdbQuWK9Tp042P6s2+Q8o8swmKFJiExQ6Wx4mtXe95WVdOd\/Rl9n\/vVNpq0cjIf3jDP0C3l0EUKaWR23YV7p06ZLD\/CFVGjdwI2S3gcHRQDXCWwOnGrhJf8Eb1zjM9bQu7tZQA8PmVDVDEJgff\/wRihYt6lDJq1evwquvvuqWwGAEZvLkyRAdHW1pB4nP4MGDoUmTJi6xwQRgrD9p0iSoX7++QwLjCvzk1DR4dtAum5eoRSMw3iTwqh19QQfQX2gyfs6e60x4e+47GWsS3jKi5rnOFIHx3Hcua9pHN5wJu2OP7dq1A9wy6tq1K2siKSkJatasCZs2bcr0ts3hw4dZdKFIkSJM9smTJ+zU0owZM6BOnTrCBEZ5RqBDzGI4\/M9BeOpWYdjE+RJ1yqkJkHLmG8j\/yh7IEuj6pJVT38QGQIG2w6FA7DDVUKIJTjVXStEQ4S0FTKopSXir5kopGiICoxFMmKTLUxo2bOhSbPny5YCni\/AUUkhICGCibmJiIjtWjQX7wW0qzHPp0qULZMmSBcaPHw+BgYGsHp5W2rhxI+TOnVuYwCi38CoEJq5id4gL78ZjFnib\/\/LvoXVw9YvmUGLcYcjxbFWuPnmEaILj8ZJ5ZAhv82DJYwnhzeMl88gQgZEASyQiCxYsYCeQatSowe6BUbam8HFIjND07NkTbty4wU4gYe4NEpnnn38e8Di3\/WklxWR34CsEZnWxcFi9aic8GLUZJtedweWxu3vbw6Mb+zx+gVq5+0WNo9PWCtMExwWfaYQIb9NAyWUI4c3lJtMIuVvDZDVU9xwYdNyGDRvg3LlzjFwoBbd1OnToAD169GBJvHoWd+Art\/AqL1FXf6cwNBrNtx3kbQRGi\/wX9DVNcHqOON\/3TXj73ud69kh46+l93\/ftbg3zvUbq9Kg7gdm3bx8jKu+\/\/z7g8Wal4CV2X375JdsSwu0dPGWkV3EHfoupv8HuxCSWxBu6viXEVewG0f2CudT15gK75B+\/gRvf92L9UASGy91uhWhid+siUwkQ3qaC060x\/oq3uzXMreMMKqA7gXn77bdZMi2SFUeld+\/egKeE8FkBvYor8O+mPoZSg3bCKxE34O9cXzAVx\/y53icERu3L66z9668fOtmt11emT7+Etz5+16tXf8WbCIxGI6569eowbdo0lrPiqGCEBknM\/v37NdLAfbOuwFfyX0a2fQKLL37MGvPVEWqtto\/QBn\/90Mlu99+DmSQIbzOh6d4Wf8WbCIz7seGRBN6Ci6d\/Spcu7bD+mTNnoHnz5nD69GmP2lejEg+B2TG4AAyZMQ5uhp7iJjBK\/kueF76CHCVaCqmqJO\/mb9EPCr41Tqguj7C\/fuhkN8\/oMI8M4W0eLHks8Ve8icDwjA4PZF588UV2IsjZMWl8iRofZty5c6cHratTxRX4Sv7L+C5nYfWqHazD1YMSuDr2JoFXy+gLRWAck2kuUCUV8teJneyWdMB6qLa\/4k0ExsMB467awIED2X0teJV\/9uzZbcSTk5Ph9ddfZ5fL8T686K4\/T37PE4FBAjPr+Az2jAARGE+8bIw6\/jrBkd3GGH++0oLw9pWnjdEPERiNcPj777\/htddeg0KFCrEj03gXC74UfezYMZg6dSo8fPgQ1q9fb7k1VyM1XDbLQ2C2DMwLvbZnXF7HmwPDTiCFvguBFQYImXWhXXZIf5wGWYMKQalZN4Tq8grTBMfrKXPIEd7mwJHXCsKb11PmkCMCoyGOeAcMXiRnnagbEBDA3iXCyEvx4sU17N19067AV+6AOfl2JHzf8CycbboaZs\/+1G2jaTf3Q\/KediD6BlLqiR1weVg91r7aR6etlaYJzi2EphIgvE0Fp1tjCG+3LjKVABEYH8CJN+RevHgRsmXLBiVLloR8+fL5oFf3XfAQmJbPrIZyX8RBqRfzQIcVtq9XO+oh5XQ8pJyeJHwDr9a5L4quNMG5HxdmkiC8zYSme1sIb\/c+MpMEERgzoSloC08Sb+uoBXDq8W\/sIjueCIynBObK8PqQcny7ptEXdA9NcIKDRHJxwltyAAXVJ7wFHSa5OBEYyQH0Rn13EZiSBQMhT1gPlsBb\/kAzGLvqTbfdeXoCCSMw2YPLwDPfnHPbhzcCNMF54z356hLe8mHmjcaEtzfek68uERj5MFNNY3cEBjvCZwSw9M\/xFcTEvOS2b28ITGDFehAyfJvbPrwRoAnOG+\/JV5fwlg8zbzQmvL3xnnx1icDIh5lqGosQmBEh30CDqNpu+\/aEwCj5LyFDNkJglUZu+\/BGgCY4b7wnX13CWz7MvNGY8PbGe\/LVJQIjH4rNDL4AACAASURBVGaqaeyOwHzYsBRsvB\/L+hM5Qo3yBVtccKvn7RWj4NaiIRY5LU8fKZ3QBOcWFlMJEN6mgtOtMYS3WxeZSoAIjKngFDPGGfh7zt2G5lMOw4DGpWFt8uus0Z2tD0GWrK7bV6IveV9cCtkKOX4DyroFLR9tdKYpTXBiY0R2acJbdgTF9Ce8xfwluzQRGNkR9EJ\/dxGYUa8XhITLcdwRGJHtI+XNI1\/kvVi7iCY4LwaMhFUJbwlB80JlwtsL50lYlQiMhKCppbIz8JWXqEVv4RUhML46Nm3vK5rg1Bo9crRDeMuBk1paEt5qeVKOdojAGByn9PR0mDhxIixZsgRSU1OhSpUqMGbMGChRooRLzfGxyN69e7N6NWo43s5xR2BW9wuAgXs+4orAKPe\/8N7A66uL64jAZHiAJnaDf+gqq0d4q+xQgzfnr3gTgTH4wExISICZM2fC\/PnzITg4mL1gfeDAAVi1apVTza9evQpt27YFfDTyu+++EyYwyjMCykOO2JG7JF7lCQGe5F2FvBRoOxwKxA7zKQL++qGT3T4dZrp3RnjrDoFPFfBXvInA+HSYiXcWGxsLMTEx0LFjR1Y5JSUFqlatCuvWrWMPRDoqnTt3hubNm8OkSZPgq6++8prA5LtYCn7ou8Kl8g8v\/wD3DvZ0e\/pIyX3Bxnxx6ogiMBSBKV26tPiHJ3kNf13QyG7JB66g+kRgBB3ma3EkK9OnT4fIyEhL1w0aNIC+fftCs2bNMqkzb9482LVrF4u8REVFeUVgfmtTExa0Osf1kOPdve3h0Y19bgnM5SF1IPX0z7qQF9pKoYXc19+vXv3RQq6X5\/Xp11\/xJgKjz3jj7jUsLAwWL14MERERljpNmzaFuLg4aNOmjU07OIjfeustWLlyJRQpUoSLwFg3gJGbig3aQrcVV6F7rfzQMlsQbHn\/XybSfl+QS53zHX2Z\/f5Opa0u5dI\/LgOQIxACPj\/O7QM1BS9duuQ2f0jN\/ozSFtltFCR8owfh7Rs\/G6UXf8Eb\/0CfO3eujdvPnz9vFBhU0yMgHbNfTVAwAjN58mSIjo62WIORlcGDB0OTJk0sP3v8+DHLe+nevTs0apRxm603EZiv3ygPiUcXQ47BDSC6fzBE9wt26U13J5DSHz2AC2\/mYm3okfuiKO+vf6mQ3SaYDARMILwFnGUCUX\/FmyIwBh+87dq1A9wy6tq1K9M0KSkJatasCZs2bQLrvf3Tp08zAhMU9F+k5MqVK1CwYEFWt1u3bpksdQT+d7v\/hgErzsCtifWh6eqXIfvxEIh4ujoM7\/KhU0+lP0qGpA1VIHvhSAiqs8ihnHJsGn+pR+4LEZgLNuPF4MNeNfX8dWInu1UbQlI05K94E4Ex+PBcvnw5xMfHs1NIISEhMHLkSEhMTGTHo7Fs3rwZihUrBuHh4Zks8SQCo9wBgwQmalk11ia+Rr16UIJTTylHqPO+uBiyFarlUO5C2yyAQTE9yQsq5q8fOtlt8A9dZfUIb5UdavDm\/BVvIjAGH5ioHhKYBQsWsBNIeKcL3gNTtGhRpnmrVq1YhKZnz56qEBjlCLU1gXnjuQ7Qq0pf5wTm1ERIOTPZZQIvHp329a27jhT21w+d7JbgQ1dRRcJbRWdK0JS\/4k0ERoLBqZWKjsB3RGDiKnaHuPDMW1CKXu7yX1COCIxWKPK1668THNnNNz7MIkV4mwVJPjuIwPD5yZRSvATmsycLoP4bFZz6wN0R6ocXDsOlj1+AYiO2Q67wurr6kiY4Xd3v884Jb5+7XNcOCW9d3e\/zzonA+NzlxunQGYHBV6gHNH7WkgOzsPhOKFknt1PFMQITkDUnFGh2yqGMcnmd3vkvqBxNcMYZf77QhPD2hZeN0wfhbRwsfKEJERhfeNmgfdiDv+fcbWg+5TAoBGZ08BF2id30+EGQM29WlwQGf+nsGQG93j1ypDBNcAYdjBqpRXhr5FiDNkt4GxQYjdQiAqORY2Vo1h78+w8ewzMDd7Ij1FiQwGAZdK2yS3MwAuPqCDURGP1HA03s+mPgSw0Ib196W\/++\/BVvIjD6jz3dNLAHP37rXzBi3TlY2yMCXiybn20hha5vCbNnf+qWwASGfQCBYY7viiECoxvElo79dYIju\/Ufe77UgPD2pbf174sIjP4Y6KaBPfjWd8DcTL0BMWsbM93cvUSNEZjAsD4QGNbboS1IYPS8fddaKZrgdBtuunRMeOvidt06Jbx1c70uHROB0cXtxujUFYGZdeJbmHV8hlsCo1xiRwTGGJg604ImdmPjo7Z2hLfaHjV2e\/6KNxEYY49LTbVzRWA+PzAMfvxznVsC4+4OGCOdQEJj\/PVDJ7s1\/ZQM1zjhbThINFXIX\/EmAqPpsDJ24\/bgN5n8K+y\/cIcl8SrPCLjbQnJHYIyU\/0IEprSxB6QG2vnrxE52azCYDNykv+JNBMbAg1Jr1ezBt76Ft9eO7nD4n4NcERhZto+IwBCB0fqbMkr7\/rqgkd1GGYG+0YMIjG\/8bMheXBGYaUe\/hoRTc6HhnY5OX6JOu7EXkvd2cJrAq0RfivSYDUH1\/mcIH9AEZwgYfKYE4e0zVxuiI8LbEDD4TAkiMD5ztfE6ckVgeF6iVhJ4ZbjATvE+TXDGG4daakR4a+ld47VNeBsPEy01IgKjpXcN3rYjAmP\/jEDJoFKw8NUVDi1xlf+SemIHXB5WD\/LUjoWn+y41jCdogjMMFD5RhPD2iZsN0wnhbRgofKIIERifuNmYnfAQGFcvUfMQmIIdxkD+lp8YxgE0wRkGCp8oQnj7xM2G6YTwNgwUPlGECIxP3GzMTqzBVy6xs76FF7X2lMAkb5wKN2b2ACM84GjtfZrgjDkWtdKK8NbKs8Zsl\/A2Ji5aaUUERivPStCuNfgNJx2CX\/9KtryDFDO6A9wMPQUJxXdCKScvUbu6gff61Di4u202ERiDjAOa2A0ChI\/UILx95GiDdOOveBOBMcgA1EMNa\/DxCHWNUvlgY+8XmCrKQ47R\/YMhul9wJvXuH\/4IHlxcAQVfOwOQJXum3xvt\/hdFQX\/90MluPb4w\/fokvPXzvR49+yveRGD0GG0G6dOewKBavC9Ry3aBHRGYC1C6NN0Do+an9+abb8K+ffvUbJLaIg\/4vQciIyNh4cKFXH4gAsPlJv2E0tPTYeLEibBkyRJITU2FKlWqwJgxY6BEiRKZlHrw4AH73Q8\/\/AAPHz6E8uXLw6effgrPP\/+8QwOcERjlHSRXL1Hf3dseHt3YBzIdoUYn+OtfKmS3+t+wWSdP9T1FLZIH+D0g8l2JyPJroL9kQDqu\/CYoCQkJMHPmTJg\/fz4EBwfDuHHj4MCBA7Bq1apM1o0ePRqOHj0K06dPhzx58sD48eNh7dq1sHv3bi4C06FWCEx+ozzXMwIYgQkIyAoFmic63T4yygvU1grSQm6Cj0LABC3xNuvkKeBeEiUPqO4Bke9KRFZ1RTVs0DQEJjY2FmJiYqBjx47MXSkpKVC1alVYt24dhIaG2rhwx44dLDJTtmxZ9vPTp09DkyZN4OTJk5AzZ85M7raPwESVKwBr3q8KyjMCOe7nga3\/2+EQJiQw2QtHQlCdRTa\/vzK8PqQc385+ZrQTSBSBoS0kNeccs06eavqI2iIPiHpA5LsSkRXVQ0950xAYJCsYUcF9QaU0aNAA+vbtC82aNXPq46SkJLaddOvWLRbBcVQQfKXcbvk9++ehXs9Cz9+7wJ1Ht6FCUEUYFDYiU9Vc\/8yDnNfmwIPgzpD6dGfb3894E9LP7YOAL8\/rib\/Tvi9duuRw+82QyqqoFNmtojP\/f1P4HZ4\/b8xxzmMt\/oEzatQoOHXqFDx+\/Jh9F\/369YPo6GhWHeecyZMnQ40aNQD\/kLpy5Qrs2rULAgICLM2vWbMG+vTpA\/PmzYOoqKhM3d64cQM++ugjVnfTpk08apGMn3sA16UtW7Y49AKOs7lz59r8TuZv0BnUpiEwYWFhsHjxYoiIiLDY2rRpU4iLi4M2bdo4tL9169bw22+\/sTpTpkyBokWLOiUwCvh4CumVCoVgSdfKli2kQmfLw+pBCZnqOntC4PLQaEg9mbFdZcToC+ql5ZaCkecdslt9dGT\/6+\/ll1+Gd955B9q3b89IyYYNGxiB2bNnDxQoUCATgbl8+TJMmjSJERqldO3aFY4cOQITJkzIRGDu3bsHrVq1AiR6W7duJQKj\/hA0ZYsi35WIrEzOMg2BwQgM\/hWk\/FWEIOBfOoMHD2bbQ84KTh5Lly6FGTNmwI8\/\/sgmJPtiv4Vk\/4xA24Mj4YOxmaM89w6+Dw8vb8iUwKscnX7m69OQPeQ5Q44XWsgNCYtmSmmJt8yT56NHjwD\/OMJTVE8\/\/bTF\/+ivkiVLQtasWTMRGNyazpEjB4wcOZLJJycnA\/4xhfLvv\/++QwJz\/fp1wCgMzlcUgdFsmJuqYZHvSkRWJieZhsC0a9eO\/QWDf+lgwa2hmjVrssnA\/ljsihUr2O+sTyhVrlyZEaC6detyE5ivB6yHpdU\/dXoLr7MTSEa9+8XacC0XNCN\/IGS3+ujIPnl269YNrl27Bm+\/\/TbUqVPHhsigt+y3kHAOGjJkCOzduxeyZcvGTkaeOHGC5dr17NnT4RYStvPLL78QgVF\/+Jm2RZHvSkRWJoeZhsAsX74c4uPj2SmkkJAQ9tdPYmIimzywbN68GYoVKwbh4eEsHJwlSxZ27BpPIa1cuRIGDRrEwrco4y4C823HitDmhadhROwqyL6zDHT4vzJQKipPpnqOCEzazUvw17vPQIHXB0OBdp8bdqzQQm5YaDRRTEu87SdP3IY1clHueFJ0xCgM3rexfv16+P333wHt6dGjhyW3zp7AfPzxxzB16lR46623ALef8L+Y\/4InI4nAGBl5uXQTISUisjJ5wTQEBp2OBGbBggXsBBLuP2NyrpLXouwx4wSC4drPPvsMfv75Z8A7YTC0i8m+ONk4Kgr4yjtIX7Z+DrpEFWc5MJj\/Mrnet1DSwTMCji6xS1o2ApKWDjds7otiv5YLmpE\/ELJbfXRkJzDWHsG5ZePGjewPHry6AfPnHBGYv\/\/+G7Zt28YiMW3btmX\/xigxERj1x5e\/tihCSkRkZfKnqQiMVo5XwO846xisP3bdcgsvEhgs3wSug6qvhWTq3tEbSDJsH6EhtJBrNZqM2a6WeMs8eSIROXPmDNSvX98GuM6dO0OjRo2gQ4cODgkMRnqxznvvvcf+YMKoDBEYY459WbUS+a5EZGXyBxEYDrTsIzBKiFkhMLtjD2VqJe3WQUjeHQsFGv8CATkLs98r5AX\/bdTTRxSBoacEOD4JIRGZJ088fYj3S+H2DxIWPIWEuSqYF7No0SKoUKGCQwKDEeBevXox2dmzZzM5IjBCw4aE3XhA5LsSkZXJ8URgONBSwG8x9TfYnZiUKQLjiMA4OkKtXF5ndPJCERi6yI7js+AWkX3yxDtd8Fj0uXPnGIEpVaoUO02EhAaLoy0kJDCYd\/fll19aThU5IzB4+hFzZPBSdHzaBC\/TRJ9hzg0V8oAzD4h8VyKyMnmcCAwHWgr4mHyYO2dWuDjmJVbLVQTGUf4LRmCyB5eBZ745x9GrviJabinoa5nr3slu9dEx6+SpvqeoRfIAvwdEvisRWX4N9JckAsOBgTWBQXGeLSR7AqNsHwVWrAchw419CoMiMBSB4fgsuEXMOnlyO4AEyQMaeEDkuxKR1UBVzZokAsPhWkcE5nTSSXhnc8a7S462kOwJjEzbR0RgiMBwfBbcImadPLkdQILkAQ08IPJdichqoKpmTRKB4XCtAn7Jgbvg3oM0FoE5fP0QDP9uItwMPZWJwCj5L4FhfSAwrDfr4cIbWSH9yRPDJ+8q7qCtFI6BYSIRLfE26+RpIvjJFAk9IPJdicjK5AoiMBxoWUdglJeod13eAQP39GX3wNi\/g+Qs\/0WW7SOKwFAEhuOz4BYx6+TJ7QASJA9o4AGR70pEVgNVNWuSCAyHax1tIc068S3MOj4Dsj3IBds77rFphQgMh1MNKqJlJMKgJmdECC9od3zcrJOnkfEk3czvAZHvSkRWJs8RgeFAy1EEJnpZdUiHdFbbPgfGnsAk\/zgFbnzfEwrEDoUCbUdw9Ki\/iJYLmv7WOdeA7FYfHdknT3zDaNSoUXDq1Cl4\/Pgxe0MNX6NWHo61P0Z95coVwKPXeORaKWvWrGFHpefNm+fwLSR8hw2fQcFj1Ngu3iKOz5xQIQ8484DIdyUiK5PHicBwoGVNYJSXqJUIjDMCk6tcN3gqfCBrXZbnA6xdQQs5x8AwkYiWeMs+eeITI\/h+Wvv27Rkp2bBhAyMwe\/bsYa\/X2xOYy5cvs3tj8C4YpeADj0eOHIEJEyZkIjDY3vjx49nTBEha8Pbe6tWrQ+\/eGflzVMgDjjwg8l2JyMrkbSIwHGg5IjC7xl+DXeOuQd6yAdBzbyWbVuyfEJDl+QAiMNpupXAMNd1EiMA4dj0+5BgWFgb79u2zeYUa\/YVvqGXNmjUTgSlbtizkyJGDPSiLJTk5GZo2bcrk8QK8qKgom87wgci0tDSoVi3jaZLvv\/8ejh8\/zh6bpUIecOYBEVIiIiuTx4nAcKBlTWDW9oiAF8vmh68HrId7c0pAqRfzQIcVZSytPPx7Ldw79AFkLxwJQXUWsZ8TgeFwskFEtFzIDWKiQzW0tNt+8sSIpJFLgdhhNurhswHXrl2Dt99+G+rUqWNDZFDQPgKD0RZ8xHHv3r2QLVs2WLJkCZw4cQJwK8rVY45Kp9hP48aN2dMDVMgDRGCcjwEiMBzfh6MIDN7CGxn\/CbRqXRde7Pu0pRVHTwgQgeFwskFEtFzIDWKi7gTG+k0wI\/rE\/qkPjMIsXLiQXe2P0RKcD3r06AHNmjVj6jt6SmDq1Knw1ltvsRfu8b+Y\/4LvKbkjMLj1dPDgQZYrkyVLFiO6h3QyiAdEoioisgYxj0sNIjAcbnJGYLDq5HrfQkSRjNAvFmcnkAq0HQ72f9lxdK2bCC3kurlel461xFv2CIw1ICkpKbBx40YYNGgQy1mJiIhwSGDwFett27axSEzbtm3Zv1095ojvIA0fPhz++OMPQPKTO3duXcYBdSqPB0RIiYisPB4AIALDgZYrAvNO+ffg7UpdiMBw+FEGES0XciPbr6XdMk+eSETOnDkD9evXt4Gvc+fO7DHHDh06OCQw4eHhrA4m5F6\/fh0+\/vhjlwQGTzldvXqV5b1kz57dyEOFdDOIB0S+KxFZg5jHpQYRGA43OcqBcfaQo30ERjmB9FT1FlB0wGqO3owhouWCZgwLHWtBdquPjsyT5\/nz5yEmJoZt\/yBhwVNIv\/zyC2BezKJFi6BChQpOX6Pu1asXk509ezaTcxaBwQThESNGwNq1a1nODBXyAI8HRL4rEVmevo0iQwSGAwlXERjrO2AeXdsCd\/d3AesnBGTMf0GX0ELOMTBMJKIl3rJPnninC+amnDt3jhGYUqVKsdNESGiwOMqBwSPUmzdvhi+\/\/BI2bdrE5JwRmI8++ghWrVplE3kpV64crFu3zkQjjExR2wMi35WIrNp6atmeaQgM7iFj+BUz\/lNTU6FKlSrsMii8dMq+4JFFnFhWr17NZPHY47Bhw1gdR8WawFwYFQ35ArOBowhM2s39kLynHRRsccHSDBEYLYev+m1ruZCrr616LWppt1knT\/W8Ty2RB8Q9IPJdiciKa6JfDdMQGEyomzlzJrvNMjg4mIV8Dxw4wP6ysS\/ffPMNO1EwZ84cKFy4MMTHx8Py5cvZxVTuCAw+5Hjg2j7ou7MHE7WOwJjhCQHFfi0XNP2Gu\/ueyW73PhKVMOvkKeoHkicPqOkBke9KRFZNHbVuyzQEJjY2lu1Vd+zYkfkMTwtUrVqVhWFDQ0Nt\/Lh7924ICgqyRFxw0cLjjnhXQ2BgYCafI\/hLt\/0GjeIPgXIPzOjgI3C26WqYPftTi\/zdve3h0Y19mSIwsp1AQoNoIdf60zNW+1ribdbJ01gIkjb+5gGR70pEViY\/mobAIFmZPn06249WSoMGDaBv376W+xqcAYP1tm7dCkuXLnUagek+bSuM3XgBMALTa0d3yDm4IRQ8GwaDrlW21MEITJachSF\/41\/Yz5QEXiIw8nwSWi7kRvaClnabdfI0Mp6km\/k9IPJdicjK5DnTEBi87nvx4sXsXgal4PXdcXFx0KZNG6eY4BbTF198wU4UlC5d2imBufdiP0grUgHyr3oHCncPghxlskGxGxVg7KufWerkO\/oy+\/edSlszfnZ+P6RPbw8BX56XaUwwXS9duuQwf0g6QwQVJrsFHcYhjn9I4GkeKuQB8oB6HkBSsmXLFocN4kWIc+fOtfmdGb9B0xAYjMDgi67KC7GIHL45MnjwYGjSpEkmkJ88ecIeUMMLpmbMmMHeKXFWcKBYR2A+OzAUNv75AxQ6Wx5WD0qwVLPPgfn30Fq4+kULsL\/ZU70hrF1LWv5Frp3W3rdMdnvvQ\/sWZP7rDxP+n3vuOZuXpRX7MO\/OOuKrhudwGxvvjtmxY4cazVEbJvaAyHclIiuTy0xDYPCIIv6lh++QYElKSoKaNWuyI4z2kRU8sYQXS924cYORHnfP1tsTGOUEEvZjn8RrfYRaxleolcFLC7lMn7H3umqJt8yTp0JgMG+uWLFi3jvaTQu+JjD4hxw9WaA5rJp0IPJdichqoqxGjZqGwOApIjxNhKeQQkJC2EuwiYmJ7Fg1FryTAScgvCET3zXB\/61cuZLr1ksEv+ag1bDpxA2WAxMzugPcDD0FTx+rCitGfM\/aV45Qm+EOGLRHywVNo7GsSrNktyputGlE5smTh8Dg1f8YjcmXLx+0atWKhe6R8EybNo1txeItu1is\/z++qYTPDNy5cwdy5szJLrLDhyJdEZgVK1bAlClT4OHDh1C8eHGYMGECO4yAd87gCUo8UYkF28qVKxcMGDCA\/YGmnMRUItL4UnalSpVYpAfz\/37++Wc2VzrSR9F7wYIFULBgQWjfvj176gDtw+KsffVHEbVo7wGR70pEViZPm4bAoNORwOCHhieQ8KPGe2CKFi3K8MCJBSM0+Jga5sacPXsWsmbNaoMVTkLKk\/bWv0Dwn++3AnYnJjEC8\/KKOhB0qjTEtKwLceHdLKK4hWSGO2CIwDjOhZLpwxbVVUviJvPk6Y7A4DyCOXb4BxISiN69e7MHH3ELyBWBee2111h+XuvWrWHNmjVs7sJ8BmcEBiPKuF31008\/se1ufIsJL9VDcoTPGmB7eBITCxIVJCaXL19m10kg8XnqqacYYalduzZ7VRu33PGNpoEDB7J2nOmD9mG72C8StO7du7N8JrQPo9vO2hcdfyQv7gGR70pEVlwT\/WqYisBo5UZnW0hx4d0hrmIGgUk5MxlSTk2EvC8uhmyFarGfyXqJHREYIjBqfkv2k6eSK6ZmH2q2Zf1HiEJgMGphXfCuKVzE8Y+e7du3w3fffcd+jT8bOnSoWwKDf2Rhm\/hH1D\/\/\/MNIB7655CoCc+\/ePct2N5IejDpjsiZGmZH8fPvtt3Ds2DH2Ujbq0b9\/f\/ZyNhIXLHjSEmWUww6zZs2yHHpwpg\/at3PnTpYniAWfO8DcQXftq4kHteXYAyKkRERWJn8TgeFAC8GvMXAV\/HTyJovAKDkwC4pvh2frBGUQmNPxkHJ6kiUCI\/MRaiIwRGA4PgtuETMQGGc5MHgpJr4gjYs6Foy+fPDBB24JDBIB3O5+9OgRIElC4oJPFTgjMJir8tVXX1m2bnDrCbfEMeKM0Zl69eqxiztRH8zx69evH7zzzjvw66+\/Wl62xjYKFSrESAie1sTIjJIf6EwfbA+3wfCkJpZDhw6xqymQwLhqn3twkKDHHhAhJSKyHiukQ0UiMBxOR\/Bvt8zIdbEmMMPuLYBX3q7Afm5\/id2dHybBzTkfSnkCiQgMERiOz4JbRObJ090WEhIIXMyVCAxuJX322WfsZxjtQHIzevRo5iskOUg2cBsbCYdyySZu9eDpSVcEBiMumHuCd1XlzZuXkQ\/8H\/aP5a233mJbQ\/hEChIdfDzyk08+YSeocKvKvlgTmCtXrjjVByM8+\/fvZ7k3WKwjMK7a5x4cJOixB0S+KxFZjxXSoSIRGA6nOyMwP0Tug3zPZM8gNmsyFj0l\/Czz9hERGCIwHJ8Ft4jMk6c7AnPy5En2SOOPP\/4IRYoUYds3p06dYgQGk2dxC2bZsmUsL+\/1119nkY9OnTrBm2++CXv37mWvT48dO5YRIIy+YC6So2PUmBiMj0ricykYfcF+7t+\/zw4iYMF+sE8kQcrdIEimMLcG77jCk5Z4cAG3rTBnx5rAnD592qk+uK2FkRbMgcE2kAz99ddfrC9X7XMPDhL02AMi35WIrMcK6VCRCAyH0xH8vHEJ8NetVNg6MC\/03J6R92J9hDrph3BIf5xCBIbDn0YW0TKZ1V\/tlnnydHUPzIcffsiiKXgaCPNKMDKC5AD\/jQs8kpZu3brBzZs32akhfGH69u3b7HABvkCNkQ1MjMW7qpBoYF8YvXFEYPDKB7wiAskL5t\/gNRDY9htvvMG2i\/D3mKD77rvvsraVgltASHLw5BLigGQJDzbYbyE50+f\/\/u\/\/WAQJH77FLSs8DDF79mx2fxYWZ+0beaybRTeR70pEVib\/EIHhQEuJwESVKwADW9+H\/rs+yERgMAKTvXAkBNVZxH6HERgZnxBQ3EELOcfAMJGIlnibdfJ0BL+v73HxxRC0vitm3759LB\/G0SO5vtCF+vjPAyLflYisTD4mAsOBlkJgBjQuDQMaPwv4kOP+3mMz3cJrfwcMERgO5xpMRMuF3GCm2qijpd1mnTz9gcBgZKd+\/fqAkRjMp8GTTbiVNGzYMCMPZ7\/QTeS7EpGVyXlEYDjQUghMdLkC8HXlcpDwesa7LvYPOWYvXBuC6iy0POJYbMR2yBVel6MH44louaAZz9r\/NCK71UfHrJOnPxAYPEUrfgAAIABJREFUtBFzZzCBGE83VaxYkW1DFShQQP2BQi0KeUDkuxKRFVJCZ2EiMBwAWEdgbv25APJ92sKGwKSemwn\/Hh8FSgTmfNssAOnp0p5AQuNoIecYGCYS0RJvs06eJoKfTJHQAyLflYisTK4gAsOBljWBuZRrDBz+5yDUPNEWJg4bwGorzwiY5QQSERg6hcTxWXCLmHXy5HYACZIHNPCAyHclIquBqpo1SQSGw7UKgVnarQr89WQ5zDo+AyKerg6T62bcTml9iV3SspGQtDRjf1jGV6gVd2j5FzmHy3UTIbvVd71ZJ0\/1PUUtkgf4PSDyXYnI8mugvyQRGA4MFAKztkcEfHEiFpJSb0Ghs+UtSbzWl9hdGV4fUo5vl5q8UASGIjAcnwW3iMyTZ5MmTeD999+H5s2bM3sfPHgAVapUYW8A2f9sw4YNlptt7Z2D97zgHS3KjbaOnIf3seC7RnjSx77gEWu8+K5ly5bcfkdBfC4A73Jx1a9QgyRsGA+IfFcisoYxkEMRIjAcTipZsynci+oPeAppbfLrlhrKPTAZl9gFQMEW56V+\/8jaFRSJ4BgYJhLREm+ZJ0+8A+Xu3bvs7hYsSC7w7pdGjRpZbtjFC+nwXhblhWZHwwLvhMFnA\/CuGGfFFYHBN46QNOGFdiLF1wTG+si1iJ4kK+4Bke9KRFZcE\/1qEIHh8L0SgVnUpTJ8drwRqxFXsbvlJWrlDpjcVafDH53zs9\/LvH2E+mu5oHG4XDcRslt918s8eeKFdEOGDGG34GLBa\/rxUriNGzeyxxGx4EV2eNwYSQ7+DIkGyjzzzDPsav+nn36a3bSrRGDwbSF8LwkvuKtRowakpqayq\/wrVarEbr3Fm33xMjwkA9hmtWrVoHHjxuwSO3xFGp8PwHGKL0lfv36dvTQ9fPhwJocRIjzqjG8ghYSEQHh4OPuZfQQGTxThpXl4wy7+G1+6Rhl8uBH1x2iSUpo2bQoDBgxg7eNDlb\/99ht7hPJ\/\/\/sfdOjQgYmh7ngBH76C\/fPPP0NiYiLzG+qcM2dOGDFiBNSpU4fpgmQP31TCV7Vr1qzJ3lrCZxbwsUpn7as\/KuVvUeS7EpGVyTNEYDjQKh1WEe40+4a9g9RrR3eWxGtPYLCZgAf\/g6Slw6UnL0RgaAuJ47PgFrGfPBNaZ1xDYNTSYUUZi2oYOcFba\/Ha\/BIlSkDbtm3Zzbl9+vRhx4uRJOATAXjFPi7GDRo0YE8HhIWFsbeQcKHGl5ytCUyvXr1YW0gKMGrTpUsXFs3BI8oxMTGMtOCNt9g+3r+C\/\/vhhx\/YO0hKBKZZs2bs\/SMkO\/iAJN7KiyQLZfDdJKyLumM71atXz0Rg0B48Do3bUlhQDgkIEiUkVfhzJGAXL15kW2W\/\/PIL0wtvEkbChv\/Fn6Nd+O4SEiv0DZKqgIAAeO2115hPWrduzfTBm4bxiQN88gD\/P\/4XSR\/eXIwvcSN5GjlypNP2jTpW9NRLhJSIyOppk2jfRGA4PFa80XuQ8nxbwByYHXX+gluhp+H1N+pB7V5F4Mn9P+H2lnqQNfezkLI\/EP49vIEIDIdPjSpCERj1kbGfPPEiSCMX6\/udUM+OHTuyxbpFixZsscVXn5HEIGHBBR8jExh1wOv18cp9hWRgRAHJD76NhFs5SgQGCQI+kogLPxYkPfi2ERIYXNCPHj3Kfo7vLOHzAUhyrAnM33\/\/Da+88grgtlKWLFmYLBIfJA9IDJBMYCQHC5IUfEDSUQQG31LCS+mwDBo0iD0VgNtj+Nr0888\/zwiINfHCBye\/\/vprZhMWJF0Y\/UEyhz9DG5XfIXnCd5cwUvPPP\/8wv2EuDpK3F154gT08iQX9+PjxY6afq\/aNPF700k2ElIjI6mWPJ\/0SgeHwmpIDgxEYZfLtuLIslKyT2\/KII94Bc2VIH9aa7NtHaAMt5BwDw0QiWuItcwQGIZ42bRojIbGxsewdoO+\/\/569BI35MBgJwe0PfKUZF3vcYipYsKBlZCQnJ7PIA169rxAYvNF2+\/btjDBgwccdMTkXCYx1Eq91Tow1gcGIC0Y2MPqjFCQMGMFYsmQJI1pIhLBgFOj8+fOZCAxGP5A0oE5YkBShHkhgNm3aBHPmzGFRHHxrCX+G5AK3ozCHBx+gxILbQUjscNvH\/m0l9Mf8+fNZ3g8mIOMTC9iXYivqjwXfUsItJNTFVfsm+tRUM0WElIjIqqagDxoiAsPh5GJNP4TU8i3YFlLUsmoQur4lzJ79KaupvEL94Ndn2ekjIjAcDjWwiJYLuYHN1pSwyj55YqQDH0nErSKMWGBU5PLly2wLByMwmEOC20FIavBVaiQN9sU6koERG3whGokMFusIDA+Bwb7xdBQSGfuCZAMjHBg9wYJREiRR9hEYjLhgng5GaDBK8sknn7BtLayPxKRWrVrsEUgkMMqr2XXr1mVkDomGfbEmMFeuXGE5PbgNFRoaynyFBAgJDEaasG0kMlisIzCu2jfyt6OXbiLflYisXvZ40i8RGA6v2RMYrIInkB7fPQt3tjVijzje23SVEZgS43+DHKWqcLRqbBFayI2Nj9raaYm37JMnEhTc9sFkXFzwMWEVC74RFBgYyJJVMUEVE2qRWGAOTOnSpRnBQFKDCazWBAa3T3Cbp3fv3izpF49pKzkwzggMRkWQPGB7So4J5r1gtAWjKRh9wTYwBwYTjDHRF09PIelCwmBPYLBP1AHbwK0q\/C\/mrSARw4JbPUhkChcubDlthUm\/mHD8+eefs6gKtomRI\/SHNYHByNGbb75pIT7oM7QfozAYlcFcHdxmu3btGquPfsS2XLWv9ng3Q3si35WIrEy+MQ2BwUlm4sSJLISKHxne1YBJZ\/hXhaOCkw3+FYLhXfyAMVPeWXFGYJQL7IJe+AEufliRVTfD9hHaoeWCZuQPhOxWHx0zTJ64oOMJHTzdgxELLBi1wK0hJCrK\/KGcQvr3338hd+7cjLwg+bEmMLjAY94IEoTatWuzHBHcisKcGGcEBucrTLRF4oBbV8opJIx2oD6Y84IngjCv5aOPPmLJwzj34ekirIvbXNYFf4+5LtmzZ2dEBo+F9+vXj82hDRs2ZKeQMFqCuTqYv4IFc3rwEceDBw+yvBWMHCF5wzbst5BQh\/3790O+fPlYlAWTeFF33JrCE1iYD4PRGbQZE4JxrnbVvvqjUv4WRb4rEVmZPGMaAoPJazNnzmQMPzg4mB0FxGQ7R8++Y2Y9ThS4D4sfrDsC83SnaZBWuLxlC0mJwCgX2N2e\/wfDXObXp+0HLS3kMn3G3uuqJd5mnTy98br1fSl4kggjIC+\/\/LI3TUpT19r2SZMmAZI9\/GOSipgHRL4rEVkxLfSVNg2BwQQ7zMTHEwNYMKkN\/7JQ9mGt3Xz16lV2xwLKYDa\/OwKjRGDmf3AXxvwygjWFW0hK\/otCYMwSfaEIDB2jVnNaMuvk6amPcKsHt09w8UbiiJEVPNZcpEgRT5uUph7aifZjUjJGcXDexq20V199VRobjKKoyHclImsU+3j0MA2BQbKClyhhyFQpGOLEMCmGZx0VTCoTITDFr\/aAfK0CWVPzqy+HfEcz\/mJCAhPw7iKAMrV4fC6FDJ4McLb9JoUBHipJdnvoOBfV8DvEkzBUMjyAWzo4L+H8g1tPuICLPhEgqy+RtOA2FG614RFwvCRPuTtGVpv00htJCaZAOCq49Wd\/a7MZv0HTEBi8OApvr1TuIUBQ8ePAbHzlSKE90LwEpmibz+FhyRehQ8xidokdlp8qd4KU05Mg7VoqS+Ats+wJe07ALEXLLQUj+4jsVh8ds\/71p76nqEXyAL8HRL4rEVl+DfSXNA2BwQjM5MmT2XE9pWDyGSaQ4ckAR4WXwBTsu+3\/tXcu0FaNXR+fQl6jy+t0lHr1NVwOKY2i3KN00VBRQoOQSyilSCofXSRC5ZrqCw1D7qmkq1LpgqTLF3ItdEG5fklSit5v\/OY71h6r3V57r33O3uesy3zGMDj2s9Z65n+utZ7\/ms985l8PdxOYN448Xvb8vEyjL7QoLR9hj03kZf9wluYI8unvqL48S9M\/di1DIBmBbJ6rbPqGCenIEBgS4QhVU6OBRvVJKmWy\/ZAtjSUhMNWuHid\/HV5b6p7ZV09zcrVT5P6DfzMCE6Y73edY8zmR+xxCmXTLp91hfnmaGvX+tyM7PNmlRB5LJoXtMrmZS+GiPC9sX2eHWS5acc6XzXOVTd9c2FNa54gMgZk8ebJu1WMXEhUqqYuAoBjbqmkkj1H50l2EyW8EpvCWOfLvgw5JEJjz\/+wsPSo\/ruclAnNY+35S5aoRpeWzUrlOPie0UjGgmBcxu4sJXJrDwvzyjLoaNUXuKNRHHRu\/zSEwBQUFGRW2\/Z7Tqx85M8629ZKeK5fHU42ZQoDoR+WiFed82TxX2fTNhT2ldY7IEBgAg8BQwIndRTBjagtUr15dsSTTnwgNlSYpCEVfasdQ6hrNDhpaHqkeZGcJyYnAIOR40ZcPuPJf\/l1a\/iq169hEXmpQB+JC+fR3mF+eUVWjdm46atIgzpiqerDTBwxQuybplmRjatpQcdgdgaE43bBhw7S2DVIDJOayLZxIuJOwzM5P6tU4OkgcQ0Iv56XwHaKXCFcS3WFcFL7jHNSnIT3AKYnhpAbw3k6nYF2\/fn09lij8hg0bVPKA5GHOTe0a3vdUVvZS9uZ3JCSwadOmTVrHho0iGzdu1OKDXJ9CgU7xPwcvL7uOPPLIlArglPtIPp+Xqrn7ZZHNc5VN30C8kHwOIlIExqfNWXdzCEz7re\/IujavqxI1BIZGBCZq+S\/Ylc8JLWsHlOIBZnfuwU5+eT7z6f6l9jNdtUvdrv\/5yEhzrJ8+XtdxH+v8N32jqkbtl8AQAeGjbsSIEUKpfz4QIQKOwKSzhEQhPAgMH46UpWAXDB+Q\/L9t27bp8ezw4yMSQsSW8caNG2tlY87LlvIxY8bob\/PmzVPCQtXhWrVqKQGhrhd\/Ix5J1IMCgBChdArWbOjo1q2bykCg60RRQYgXKQWdOnVS0oSWk5eyN3IPjIvrM17yKYk6QYogXnwcJ0dgwMvLLghZKgVwxuA+H7vUvFTNjcDs+wQbgcn05hQRh8D0fbFQ1rWZJqNu\/FTzX\/7askt+n88OJIvA+IAxFF2MwOTeTckEBj2xbBt1l2jpjvXTx+u67mOd\/3b6RlGNun\/\/\/io5QFSEyuUQAxpL8VTIdRoEhYKfju4S0Qiq5yYTGHShIAbs+jzqqKMSxxO1IMqN5AKNiRkyAxEgKu6cl0kbyQPOC4EhAuFsA+7Xr59wDzlkgd+IGLHrNJNCNikEaE5xXqJIpBLQiBAVFRVp\/RkvZW8iNRxHxImGkCfLbUSsvAgMW5W97CK1wUsB3H0+iJqXqrl7OS2bqEo2fbN9NsuyvxEYH+hDYM4uKpDCFTPkuNnt5aan2+tRRF+iVH3XDYVN5D5ujAh1yae\/w05goqhG7TcCg+QAdWqYyJ0GgaH+iHsJCQJCBIUIR6VKlTRKA7lYs2aNVjtnKQkNJ6Iz5CmybMTSEhEXp1EKY+HChUpgkGxgmZ\/GshN\/I81Ag3QVFhaqAngmhWz6UM8K+QWiQfxNI5oC0WKjh5eyN0m6H3zwgSqM0yBUzt9eBMaRaEhlF0tOXgrg7vOlUzV3FzvMhpRk0zdMrzUjMD68BYE5\/diDZXu1W+WMx\/9b7urfXw6t3Vu2DOxtBMYHfmHqks+JPMg45NPuMC8h4bMoqlH7JTDr1q3TOlpOpIScE3JLkiMw7nsbEoLW0+rVqzW6gUwCqtY0lovQZUIjiWUcJnxacgTGTRzQnCKK4ihsu6+VSSE7E4Fh+chL2dtNWPwSGDSevOwaPXq0pwJ4cgTGS9XcbXs2pCSbvkF+TyWPzQiMD29BYKr+15tSteab2nv6If+XSOD91z2L5B91m\/o4S7i65HNCCzISZnfuvRP2l2cU1aj9EhiWf8g3YdkEskB0gCgCiapOBIbJl3wSIjDs9ERrDmVrCEyjRo00HwbFapZGWMYhv4WK6UQ\/OB\/\/htSQxOvkwLgJDMs+RGPISSHplnwWohkQq0wK2ZkIDGU3GGsqZW8Sir0iMOTeUEU5OYGXJTYvu1Dx9lIAd58vnaq5EZh9309GYHy8ryEwzg4kh8BEtYCdA4dN5D5ujAh1yae\/w05gcHPU1KizuXVJYmWCpVFviyUgZ7eQk8RL7szYsWMT+S4sD5FfAnlh5w7EgygMytOTJk3S41kWIqm1cuXKqqRNbgyRB67nJg5cl+jF1KlTNYLB\/cRxJNFmUsj2Q2C8lL3TRWAgcOwcYpeUs9TlYMo1U9m1efNmTwVwyJD7fF6q5kZgjMBk8+xqXy8Cc+iJ50qNIf+p0hu1ls8JLchYmd25904UCEyuUYmzGrWDJZEtGrkx7N5hmYjlOifhN9eYl9b5SsuubJ6rbPqWFk65uI5FYHygCIE5pv4o+UeFTdJ4w9FyR+1VkU7gBRKbyH3cGBHqkk9\/R\/XlWVz3x1mN2o0Z24ed7cxEcKi7Mnv27OLCGpjjSsuubJ6rbPoGBkgfAzEC4wMkdwSm00E7pcPHW2TXR79Gcvu0A0c+JzQfkJdZF7M799BH9eVZXKTirEbtxowE3oEDB8r27dt1VxG7hOrVq1dcWANzXGnZlc1zlU3fwADpYyBGYHyABIGhBsz7tw6Xa+uuktM\/\/kF2fWgExgd0oetiBCb3LovqyzP3SNkZDQH\/CGTzXGXT1\/8Iyr6nERgfPnAIDF2bXDVO\/rXpfyK7fdoiMOs9xT993Cqh7ZJP4sYOFepwWDMEDIHcIcBOLnZk+WlGYPygFNE+zhLScbMvkoc7PKP5L1FO4MWN+ZzQgnybmN1B9k7ux2b+zj2mQT5jXP1tBCbId2Wex+bOgaEGTFT1j9wwxvVBN7vz\/DAF7PTm74A5JM\/Diau\/jcDk+cYK8umPGjZKah4\/QYc4\/cBf5IC\/r5OCjncHecglHltcH3Szu8S3TqhOYP4OlbtKPNi4+tsITIlvnfCeoPajd+5ThbdKu\/XhNcbnyOP6oJvdPm+QiHQzf0fEkT7NiKu\/jcD4vEGi2O2sV86Scgf+qaY99+r\/Rnr7tOO\/uD7oZncUn2Bvm8zf5u84IGAEJg5e9rDx7EmNEr9M332eVLnywcijYS\/2yLt4HwPN3+bvOCAQ1\/vcCEwc7u4MBOaU3\/+Q+4oGSsVzrow8Gvfcc48g0ha3ZnbHy+Pmb\/N3HBAwAhMhL6NV8cgjj8jEiRNl165d0qBBA3nggQekZs2aKa10IjAdPtkitw\/ZHCEkvE2J6g2fyXlmdyaEovW7+Tta\/sxkjfk7E0Lh+j2WhexefPFFGT9+vKqqHnHEETJy5EhZvny5vP7662kJzORvN0n1234Kl4eLOVp70IsJXEgPM3+H1HHFHLb5u5jAhfSwqPo7lgSmY8eO0r59e7nqqqv0dty5c6ecdNJJMnPmTDnuuOP2u0WdCMwMaRf57dOO8VG94TO9f8zuTAhF63fzd7T8mcka83cmhML1eywJDGRl3LhxQilmp7Vo0UL69Okjbdu2TUlgyu0tJ9\/e+Uu4vGujNQQMAUPAEIg9AtnIDoQJrFgSmNq1a8srr7wiJ598csJXbdq0kS5dusill14aJv\/ZWA0BQ8AQMAQMgVgiEEsCQwTmiSeekHPOOSfh9LPPPlsGDBggrVu3juWNYEYbAoaAIWAIGAJhQiCWBObyyy8XloxuvPFG9dXWrVvltNNOkzfffDOWSsRhumFtrIaAIWAIGAKGAAjEksBMnjxZHn\/8cd2FVKNGDRk6dKh8+eWXuq3amiFgCBgChoAhYAgEH4FYEhjcAoF54YUXdAfSqaeeqnVgqlevHnyP2QgNAUPAEDAEDAFDIJ4RGPO7IWAIGAKGgCFgCIQbgdhGYMLtNhu9IWAIGAKGgCEQbwSMwMTb\/2a9IWAIGAKGgCEQSgSMwHi4jZ1JgwYNkiVLlki5cuXk\/PPPF4TfDjnkkFA62j1otpGT+3PAAQck\/vfo0aOlZcuWuiMrnd2zZs2Shx9+WH766SdNgB48eLCwBT2o7cMPP5S+fftKxYoVZerUqYlhZtLD+uyzz3Rb\/bp166RChQq6Y+3666\/X4zMdGwQsvOyeMGGC3sfly5dPDLNOnToJbMJs948\/\/ihDhgyR999\/X20jt+3ee++VqlWrZvRZVO2Osr+\/\/fZbvZdXrlwpBx10kBYm5e8qVapE2t\/p7I6yv1O9V43AeMw2vXr1kj\/\/\/FNFH5mwmMDq168vd911VxDmp2KPAVuKiopk4cKFUqtWrf3Ok87utWvXykUXXSRPPfWUNG7cWObNm6fkYMGCBTpJBK298cYbWu+nYcOG8sknn+xDYNLpYe3evVuaNWumUhP4fePGjcLW++HDh0vz5s0lWy2t0sYlnd0kr2MP93VyC7vd+Atts\/vuu0\/+\/vtvufnmm6WwsFBtjbK\/09kdZX9feOGFcuaZZ0q\/fv30g6xHjx66EeOhhx6KtL\/T2R1lfxuB8TmT8DCgUE20wdFGWrp0qdx6662yYsUKn2cJZrfffvtNdZ9WrVolBQUF+wwyk91EXtavXy9Ea5x22WWXCQ+UoysVJKv5qkb7hG3z\/OOOwKTTw\/rhhx9UVoIveSdK5bY9Wy2t0sYknd2UDGBy50s1ub3zzjuhtvu1117TaGC1atXUNKptUxoBv0fZ3+nsjqq\/uYenTZsm5513nlSqVEn9TfQBQd4o+zuT3VH1t9c71CIwKZBhArjgggt0+YDlIxrhaUKUhCsJUYa1ffPNN9K0aVMlHdjCkli7du2kZ8+eam86u4k+nXDCCdK7d++E+fy\/gw8+OOWEGBSM+PpOJjDp9LB+\/vlnIYrBBOi06dOnK3Gj2GG2WlplhUMqu2+\/\/XbZtGmT\/PHHH7oMePzxx2tUsW7dujoBRMFu8CbSeMMNN6hd2BwHf6eyOw7+xtcbNmzQ9xJadl27do2Fv1PZHQd\/u9+nRmBSzC5EJ6644gr54osvEr9u375dozLkxNSsWbOs5qQSX3fz5s26rNKqVStp0qSJfPXVV5rb0alTJzn99NPT2s1ER8i2e\/fuiXHA+InqELYNaks1kafTw4KsEn1hQncaxAVbiVKERUsrld3PPfecEpdrrrlGKleuLGPGjJGXXnpJlwGpixQFu\/fu3au+Ig8IEgpJj4O\/U9kddX\/v2LFDNe327Nmjy7z4nQ+qqPvby+6o+zt5jjEC4xGBgcl\/\/vnniWRHEqeY8FMtvQR14vY7rmeeeUZmzJgh999\/v37BeNl955136pJM\/\/79E6d2EmRJngxq84rAeOlhQWAITxOadxoRnPHjx8ucOXP06y4MWlqp7E72EV9x2PPYY49pZCbsdkOmyeNiWWHkyJFy6KGHqsnpfBYFf3vZHXV\/O\/Z9\/\/33MmLECPnll1\/0wyPq\/vayOy7+duw0ApNi1iV5l4TdKVOmSL169bTH\/PnzNdS+fPnyoM7TvsbF1\/d3332nD7jTSMrlCxz2ns5uJjmSYZ9++unEsRCeK6+8UiM3QW2pJvJ0eljkwHTr1k3JKrsbaOxmAbtRo0bpl14YtLRS2b169WqNIDpJ13y1E1l88skndbk0zHZv27ZN70V8c9ttt+1zO0bZ3+nsjqq\/sXnmzJlCDp7zjH788ce6HE4KABFGr2c0zM93Jrs\/\/fTTyD7fqeYXIzAesy6RhV9\/\/VV3MEBo2I2CejVrjGFuhNV56IkmkPBI3st1112n9vHQp7ObBF5yZ1h24FjyQu6++25ZvHjxfgnBQcIo1USeTg\/rr7\/+0uTASy65RCd0lhI7d+6sURfsDouWViq7yQuBqLDkR3SCXQskPs6dO1eXWsJs9y233KL3YaoE5Sj7O53dUfU372RyEln+vummmzQxfdiwYbJs2TLNU4uqvzPZHVV\/e80nRmA8kCEkO3DgQFm0aJG+8Nk+zN8O2w\/SBJ3tWJiwSEglovDPf\/5T8194CbDjJpPdvBwefPBBIZeG5SQmC+ptBLFde+21+kLj5QYpcWr4EEU68MAD0+phQeyIuPFVR9I2W3LdUaYga2mls5s6P\/iMXB7uayKM3NfObruw2s2XKbkQ5D+46xthI1+ltHQ+i6rdJKRH0d\/4k48xNOxY8sbnRI\/5oOK9FFV\/Z7I7yv62CEwQZ1kbkyFgCBgChoAhYAhkjYBFYLKGzA4wBAwBQ8AQMAQMgbJGwAhMWXvArm8IGAKGgCFgCBgCWSNgBCZryOwAQ8AQMAQMAUPAEChrBIzAlLUH7PqGgCFgCBgChoAhkDUCRmCyhswOMAQMAUPAEDAEDIGyRsAITFl7wK5vCBgChoAhYAgYAlkjYAQma8jsAEPAEEhGgDo7CENSzZlif+kaNWiuvvpqWbt2bV7qKlHc7N1339W6Py+\/\/LIKOuaqUehvy5YtKoZJxVentlCuzm\/nMQQMAf8IGIHxj5X1NARCgQDVkinW5zSKL9aoUUOLMfbo0SMvky6aSghB1qlTR4sjpmsUnWPyRzzUXXQuV+BCYI499lgtRJiPRnl+qjQbgckHunZOQ8A\/AkZg\/GNlPQ2BUCAAgTnxxBOFEvO03bt3y5o1a2Tw4MHSoUMH\/XeqCEoUqkxjlxGYUNymNkhDoMQIGIEpMYR2AkMgWAhAYJB3SCYqaDk9\/\/zzKkj69ttvS9euXbUUO+XXURpH8JDy7GjKfPTRRyqhwHLQgAEDElEVoiyolqMPdfjhhyc0tJKXkJCpGDRokEZldu7cKUVFRXLHHXeonljyEhJ9hw4dqn0hW8gbcGzt2rWFyA7RFMY+ceJE2bhxo0pD9OnTRy6++OKUwKciMOhCjRs3TlCeRjaB6MxZZ50lS5cuVS0w9L0effRRQXUeSQIwQMATKQlkJ8CJ\/jSLwATrfrfRxBcBIzDx9b1ZHlEEvAgMquMoia9YsUJJDKKeLIWg3AxZgYQ0adJEunfvrsKe6GJxSM3GAAAEMUlEQVT169dPl5xQq2bph9\/RzerYsaNq0CACiuApuSHuHBhETx0xVEQjJ02aJCNHjlSSwvXdOTBEhQ477DAZMWJEQmDy1VdflSVLlihxgsBAahh7tWrVZMKECarHtXLlSqlQocJ+XkwmMJAU\/h8kBXLCWBjzW2+9pcrs4IB+FKQJLZnmzZsrHs8++6wcffTRqiXEuGfPnm0EJqLPjJkVTgSMwITTbzZqQ8ATgWQCQxQDskHEpXHjxjr5QyKYuKdMmaKTOo0EXMjBggULEucmAtGuXTtZtWqVzJgxQ6MYJMg6uSuQgMLCQl2ychMYCBDEZ+zYsYlE3b1796qApDsCw7g4\/5w5c\/R4GhEbxkTUA3IDgSFKBOmhQTqI5LiPcYORTGAQ4ixfvrxGWGjgAUEiuoQoKTjMnz8\/IQLoXBPFbhoCpr169dKok0Vg7MEzBIKDgBGY4PjCRmII5AQBCAyTrZPTwpIL5KF9+\/YyZMgQqVixYoLAQEwKCgr0uizjEHVI1aZOnSrTpk3T5GCWcpJb8hISCtBEZ1gSgmyce+650rp1a1WLdhOYuXPnSs+ePQU1aHcOTrNmzXS8vXv3VgJDBKhly5Z6WZacSABmTA0aNNhvLMkEpk2bNhohItKU3Bwix5IZuNBYSjvllFOkb9+++vfixYt1menrr782ApOTO9ROYgjkBgEjMLnB0c5iCAQGAQgM+SNEHmiQhurVq+9DEFJN3BAYcmCIyqRqLKVAYIheZCIw\/L5nzx5demEpaPr06ToGyI97CckhMGypZpxOa9q0qUZfckVgWrRoISxr+SUw5BA5\/Y3ABObWtoEYAvsgYATGbghDIGIIeOXAuM1MRWBYQiI3BNLh1DdhOef333+XqlWr6vLS6NGjZdmyZZrYSiMvhOUZIizuJSRySchrcaIqW7dulUaNGmldFoiNkwND5KVt27Yya9Ys3YJN27FjhzRs2FCGDx+uW79LGoEhZ4fG8pfTyAciMoRtLCElR2CMwETsoTBzIomAEZhIutWMijMCxSUw27dv10mdpRsn+kBUhuUo8l8gISTxdu7cWbp06aKF6CAH5NS0atUqQWDYrUM\/xkHdGZJ4yZUhIrRo0SJZv379Pkm8JBJXrlxZyRNkiGReiNHChQt1WaekBIYdVywrkQPD2IgGkV9Dro+TA2MEJs5PjNkeVgSMwITVczZuQ8ADgeISGE7HFmEmdyZ0ojBnnHGG5s1QCI\/Gjh62URM5ISoDkeGfVDkwbEMmF4aIyzHHHKN1aSA6yduoScpleeq9995TAlO\/fn3dAs4OIFpJCQznIHrELiYiQ5yPbeMk8XrlwFgExh4vQyD4CBiBCb6PbISGgCGQBQJWyC4LsKyrIRBiBIzAhNh5NnRDwBDYHwEjMHZXGALxQMAITDz8bFYaArFBwMQcY+NqMzTmCPw\/4YoCJ7i0L3wAAAAASUVORK5CYII=","height":337,"width":560}}
%---
%[output:7db090fb]
%   data: {"dataType":"text","outputData":{"text":"\nTest-set P90 results:\n","truncated":false}}
%---
%[output:1584fd48]
%   data: {"dataType":"text","outputData":{"text":"SIM 1          : 372.33 cm\n","truncated":false}}
%---
%[output:2832d795]
%   data: {"dataType":"text","outputData":{"text":"SIM 2          : 893.97 cm\n","truncated":false}}
%---
%[output:7b36806b]
%   data: {"dataType":"text","outputData":{"text":"Equal average  : 469.45 cm\n","truncated":false}}
%---
%[output:1455ce39]
%   data: {"dataType":"text","outputData":{"text":"Weighted average: 356.33 cm\n","truncated":false}}
%---
