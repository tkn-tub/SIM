Evaluation of DQN vs Brute Force Localization Precision (Episodic Navigation)
%% Evaluation of DQN vs Brute Force Localization Precision on Random Positions (CDF)
% Executed until Done/Terminate conditions are met
clc; clear all; close all;

fprintf('=== Initializing Random Deployment Environment ===\n');
% 1. Add required codebase folders to path
addingPathParentFolderByName('code');

% 2. Load Parameters
Parameters; 

% 3. Run Calibration (Requires Parameters to be in the workspace)
Calibration;

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
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Aligned.mat')


if isfile(agent_path)
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n');
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end


Training results
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
fig = figure('Name','Training quality (per episode)','Position',[100 100 1000 600]);

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
hRaw = plot(epIdx, epReward, '.', 'Color', rawColor, 'MarkerSize', 6); 
hold on;
% Plot the shaded deviation area
hShade = fill(xPoints, yPoints, shadeColor, ...
    'EdgeColor', 'none', 'FaceAlpha', alphaValue);
% Plot the crisp moving average line on top
hAvg = plot(epIdx, epAvgReward, 'Color', mainColor, 'LineWidth', 1.5);
% --- Aesthetics & Formatting ---
grid on;
% set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'Layer', 'top');
set(gca, 'Box', 'on', 'TickDir', 'out', 'LineWidth', 1);
set(gca, 'FontName', 'Helvetica', 'FontSize', font); % Fallback if 'font' variable isn't defined
xlabel('Episode', 'Interpreter', 'latex', 'FontSize', font);
ylabel('Reward', 'Interpreter', 'latex', 'FontSize', font);
% title('\textbf{DQN Agent Training Progress}', 'Interpreter', 'latex', 'FontSize', 16);
legend([hRaw, hAvg, hShade], {'Raw Episode Reward', 'Moving Average ($\mu$)', 'Deviation ($\mu \pm \sigma$)'}, ...
    'Location', 'best', 'Interpreter', 'latex', 'FontSize', font);


%% Numerical summary
nEp = numel(epReward);
[bestR, bestEp] = max(epReward);
fprintf('\n=== Training summary ===\n');
fprintf('Episodes completed   : %d\n', nEp);
fprintf('Total agent steps    : %d\n', totalAgentSteps(end));
fprintf('Final episode reward : %.3f\n', epReward(end));
fprintf('Final avg reward     : %.3f\n', epAvgReward(end));
fprintf('Best episode reward  : %.3f (episode %d)\n', bestR, bestEp);
fprintf('Mean steps/episode   : %.1f\n', mean(epSteps));
fprintf('========================\n\n');

DQN Agent and Brute Force
%% Evaluation Setup
N_eval = 1000; % Number of episodes (random positions) to evaluate

fprintf('Running %d evaluation episodes...\n', N_eval);
err_dqn = zeros(N_eval, 1);
err_bf  = zeros(N_eval, 1);
steps_taken_dqn = zeros(N_eval, 1);

for i = 1:N_eval
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval);
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
        G_Upsilon = EnvPars.G_CST * Upsilon;
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
end
Figures
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
h3 = boxplot([err_dqn, err_bf], g)
set(h3, 'LineWidth', 1.5);
grid on


Random Way Point mobility model
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
Including the channel coherence time
Description:In this block we include the number of transmittions along wich the channel is coherent.This is similar to assuming the user keeps the same position.
% pos_MU_traj = repelem(pos_MU_traj, 1, floor(N_packets_coh*delta_time/T_coh));
% pos_MU_traj = repelem(pos_MU_traj, 1, 5);
Ploting the mobility of the user
figure;
plot(pos_RWP(:,1), pos_RWP(:,2), 'b-', 'LineWidth', 1.5); hold on;
plot(pos_RWP(1,1), pos_RWP(1,2), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(pos_RWP(end,1), pos_RWP(end,2), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
axis([0 L_hall 0 W_hall]);
% xlim([0 L_hall]);
% ylim([0 W_hall]);
xlabel('$x$ [m]','Interpreter','latex');
ylabel('$y$ [m]','Interpreter','latex');
title('Random Waypoint Mobility with Random Pause');
grid on;
%axis equal;
legend('Trajectory','Start','End');

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
Agent and Brute force evaluations
N_steps=100;

R_max_per_frame = zeros(EnvPars.T_x, EnvPars.T_y, N_Steps);

[obs, LoggedSignals] = resetFunction_nav_CST_Reward_Mixed_deployment(EnvPars);

t_psi_x_dqn_frame = zeros(1,N_Steps);
t_psi_y_dqn_frame = zeros(1,N_Steps);

psi_x_dqn = zeros(1,N_Steps);
psi_y_dqn = zeros(1,N_Steps);

for i = 1:N_Steps
    if mod(i, 10) == 0
        fprintf('Evaluating step %d / %d...\n', i, N_Steps);
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
end




figure;
h_img = imagesc(1:EnvPars.T_x, 1:EnvPars.T_y, R_max_per_frame(:,:,1));
axis square; colorbar; hold on;

% Common color scale across frames
clim([min(R_max_per_frame(:)), max(R_max_per_frame(:))]);

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
h_trail_agent = plot(t_psi_x_dqn_frame(1), t_psi_y_dqn_frame(1), '-', ...
                     'Color', [0 0 0 0.4], 'LineWidth', 1.2);
% Past action markers (dim)
h_past_agent  = plot(t_psi_x_dqn_frame(1), t_psi_y_dqn_frame(1), 'ko', 'MarkerSize', 6, ...
                     'MarkerFaceColor',[0.6 0.6 0.6], 'MarkerEdgeColor','k');
%Current agent action (bright)
h_agent       = plot(t_psi_x_dqn_frame(1), t_psi_y_dqn_frame(1), 'ws', 'MarkerSize', 14, ...
                     'MarkerFaceColor', 'w', 'LineWidth', 1.5);

set(gca,'FontSize',font-5);

xlabel('$\Delta_{\psi_x}$','Interpreter','latex','FontSize',font);
ylabel('$\Delta_{\psi_y}$','Interpreter','latex','FontSize',font);
%h_title = title(sprintf('Peak 2D-DFT, $t=$ %.1fs / %.1fs', delta_time, time_simul), ...
               % 'Interpreter','latex');

xlim([0.5 EnvPars.T_x+0.5]); ylim([0.5 EnvPars.T_y+0.5]);

for k = 1:N_Steps-1
    set(h_img, 'CData', R_max_per_frame(:,:,k));
        
    % % Optimum: trail, past markers, current
    % set(h_trail_brute_force,   'XData', t_psi_x_max_frame(1:k),          'YData', t_psi_y_max_frame(1:k));
    % set(h_past_brute_force,    'XData', t_psi_x_max_frame(1:max(1,k-1)), 'YData', t_psi_y_max_frame(1:max(1,k-1)));
    % set(h_opt_brute_force,     'XData', t_psi_x_max_frame(k),            'YData', t_psi_y_max_frame(k));
    
    % Agent: trail, past markers, current
    set(h_trail_agent, 'XData', t_psi_x_dqn_frame(1:k),          'YData', t_psi_y_dqn_frame(1:k));
    set(h_past_agent,  'XData', t_psi_x_dqn_frame(1:max(1,k-1)), 'YData', t_psi_y_dqn_frame(1:max(1,k-1)));
    set(h_agent,       'XData', t_psi_x_dqn_frame(k),            'YData', t_psi_y_dqn_frame(k));


    % set(h_title, 'String', sprintf('Peak 2D-DFT, $t=$ %.1fs / %.1fs', k*delta_time, time_simul));
    drawnow;
    if k < Nsteps
        pause(0.05);
    end
end

Functions
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

