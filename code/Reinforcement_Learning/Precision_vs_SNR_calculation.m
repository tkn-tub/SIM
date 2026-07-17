%[text] ## Evaluation of DQN vs Brute Force Localization Precision (Episodic Navigation)
%% Evaluation of DQN vs Brute Force Localization Precision on Random Positions (CDF)
% Executed until Done/Terminate conditions are met
clc; clear all; close all;

fprintf('=== Initializing Random Deployment Environment ===\n'); %[output:1fe83716]
% 1. Add required codebase folders to path
addingPathParentFolderByName('code');

% 2. Load Parameters
Parameters;  %[output:5f1a3a2b] %[output:27a23586] %[output:6a55ab5c] %[output:27b098c8] %[output:5fcd4218] %[output:8da9d215] %[output:4625933f] %[output:820de4be] %[output:6e405bdd] %[output:14ac1867] %[output:9341c190] %[output:306a3f9f]

% 3. Run Calibration (Requires Parameters to be in the workspace)
Calibration; %[output:356a8dc7]
%%
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Tx_60_Aligned.mat') %[output:6e97e48a]


if isfile(agent_path) %[output:group:5ca79a17]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:89327129]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:5ca79a17]
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

% %% Figure 1 — Enhanced Training Quality vs Episode
% fig = figure('Name','Training quality (per episode)','Position',[100 100 1000 600]);
% 
% % --- Styling Configurations ---
% % Modern professional color palette (Deep Blue and Muted Cyan/Gray)
% mainColor = [0.00, 0.45, 0.74];       % Deep corporate blue for the average line
% shadeColor = [0.30, 0.75, 0.93];      % Light sky blue for the deviation band
% rawColor = [0.70, 0.70, 0.70];        % Soft gray for the raw background dots
% alphaValue = 0.3;                     % Transparency for the shaded area
% 
% % --- Calculate Deviation Band ---
% % Calculate the standard deviation using a moving window (e.g., 20 episodes)
% windowSize = 20; 
% movingStd = movstd(epReward, windowSize);
% 
% % Define upper and lower bounds for the shaded area
% upperBound = epAvgReward + movingStd;
% lowerBound = epAvgReward - movingStd;
% 
% % Clean up bounds to ensure they align cleanly as columns for the fill function
% xPoints = [epIdx; flipud(epIdx)];
% yPoints = [upperBound; flipud(lowerBound)];
% 
% % --- Plotting ---
% % Plot raw data as subtle background dots so it doesn't clutter the view
% hRaw = plot(epIdx, epReward, '.', 'Color', rawColor, 'MarkerSize', 6); 
% hold on;
% % Plot the shaded deviation area
% hShade = fill(xPoints, yPoints, shadeColor, ...
%     'EdgeColor', 'none', 'FaceAlpha', alphaValue);
% % Plot the crisp moving average line on top
% hAvg = plot(epIdx, epAvgReward, 'Color', mainColor, 'LineWidth', 1.5);
% % --- Aesthetics & Formatting ---
% grid on;
% % set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'Layer', 'top');
% set(gca, 'Box', 'on', 'TickDir', 'out', 'LineWidth', 1);
% set(gca, 'FontName', 'Helvetica', 'FontSize', font); % Fallback if 'font' variable isn't defined
% xlabel('Episode', 'Interpreter', 'latex', 'FontSize', font);
% ylabel('Reward', 'Interpreter', 'latex', 'FontSize', font);
% % title('\textbf{DQN Agent Training Progress}', 'Interpreter', 'latex', 'FontSize', 16);
% legend([hRaw, hAvg, hShade], {'Raw Episode Reward', 'Moving Average ($\mu$)', 'Deviation ($\mu \pm \sigma$)'}, ...
%     'Location', 'best', 'Interpreter', 'latex', 'FontSize', font);
% 
% 
% %% Numerical summary
% nEp = numel(epReward);
% [bestR, bestEp] = max(epReward);
% fprintf('\n=== Training summary ===\n');
% fprintf('Episodes completed   : %d\n', nEp);
% fprintf('Total agent steps    : %d\n', totalAgentSteps(end));
% fprintf('Final episode reward : %.3f\n', epReward(end));
% fprintf('Final avg reward     : %.3f\n', epAvgReward(end));
% fprintf('Best episode reward  : %.3f (episode %d)\n', bestR, bestEp);
% fprintf('Mean steps/episode   : %.1f\n', mean(epSteps));
% fprintf('========================\n\n'); %[output:56d44986] %[output:664e78eb]
%%
%[text] #### Precision vs SNR
%% ----------------- Start pool -----------------
delete(gcp('nocreate'));
parpool;


SNR_dB_vector = -10:1:30;

N_eval = 2000;    % Use 5000 or more for the final paper result
N_snr  = numel(SNR_dB_vector);

err_dqn_snr       = nan(N_eval, N_snr);
steps_dqn_snr     = nan(N_eval, N_snr);


% Reuse the same episode seeds at every SNR.
% This reduces Monte-Carlo variation between adjacent points.
episodeSeeds = 10000 + (1:N_eval);

t_start = tic;

for i_SNR=1:length(SNR_dB_vector)

    
    fprintf('Evaluating SNR %.1f out of %.1f...\n', SNR_dB_vector(i_SNR), SNR_dB_vector(end));
    
    EnvPars.SNR_dB = SNR_dB_vector(i_SNR);

    parfor i = 1:N_eval
        % if mod(i, 50) == 0
        %     fprintf('Evaluating episode %d / %d...\n', i, N_eval);
        % end
        
        % ---------------------------------------------------------
        % DQN AGENT EVALUATION (Episodic Loop)
        % ---------------------------------------------------------
        % Reset environment: automatically picks a random pos_cal position
        % [obs, LoggedSignals] = resetFunction_nav(EnvPars);
        % [obs, LoggedSignals] = resetFunction_nav_CST_Reward_Mixed(EnvPars);
        [obs, LoggedSignals] = resetFunction_nav_CST_Aligned(EnvPars);
    
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
            % [obs, reward, isDone, LoggedSignals] = stepFunction_nav_CST_Aligned(action, LoggedSignals, EnvPars);
            [obs, ~, isDone, LoggedSignals] = ...
                    stepFunction_nav_CST_Aligned( ...
                        action, LoggedSignals, EnvPars);
            step = step + 1;
        end
    
        % Record the number of steps it took to terminate
        steps_dqn_snr(i,i_SNR) = step;
    
        % Extract the final position estimate after the agent finishes
        pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x, LoggedSignals.psi_y, EnvPars, pos_MU_true);
        pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pos_MU_true);
        err_dqn_snr(i,i_SNR) = norm(pos_est_dqn(1:2) - pos_MU_true(1:2))*1e2; % cm
        
         
    end
end

fprintf('\nparfor done: %.1f s total.\n', toc(t_start));
delete(gcp('nocreate'));
fprintf('Parallel pool released.\n');

% -------------------------------------------------------------------------
% Save
% -------------------------------------------------------------------------
%Defining the file directory and name
datasetDir = fullfile('..', 'Dataset');
fileName = sprintf('Precision_vs_SNR_%d_%d_Nx_%d_L_%d_Mx_%d_CST_CDL.mat', ...
    SNR_dB_vector(1),SNR_dB_vector(end),N_x, L, round(M_x));
savepath = fullfile(datasetDir, fileName);
save(savepath, ...
    'steps_dqn_snr', 'err_dqn_snr','SNR_dB_vector','-v7.3');
fprintf('Results saved to %s\n', savepath);

fprintf('\nparfor done: %.1f s total.\n', toc(t_start));
delete(gcp('nocreate'));
fprintf('Parallel pool released.\n');


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
%[output:1fe83716]
%   data: {"dataType":"text","outputData":{"text":"=== Initializing Random Deployment Environment ===\n","truncated":false}}
%---
%[output:5f1a3a2b]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:27a23586]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:6a55ab5c]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"5"}}
%---
%[output:27b098c8]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"25"}}
%---
%[output:5fcd4218]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:8da9d215]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:4625933f]
%   data: {"dataType":"textualVariable","outputData":{"name":"zeta","value":"0.9800"}}
%---
%[output:820de4be]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:6e405bdd]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:14ac1867]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"50"}}
%---
%[output:9341c190]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:306a3f9f]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"              channelModel: 'rician_los'\n                    fc_GHz: 28\n                    V_hall: 1000\n                    S_hall: 600\n                   mu_lgDS: -7.5916\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 7\n                sigma_K_dB: 8\n                        KR: 2.2549\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 0\n                         N: 25\n                       N_x: 5\n                       N_y: 5\n                         T: 2500\n                       T_x: 50\n                       T_y: 50\n                    SNR_dB: 36.5005\n                 theta_min: 1.8485\n                 theta_max: 4.4347\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 23\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [5 5 4]\n                    pos_MU: [5.4688 9.5751 1.5000]\n                       n_y: [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5]\n                       n_x: [1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 … ] (1×2500 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 … ] (1×2500 double)\n                      h_MU: 1.5000\n                    L_hall: 10\n                    W_hall: 10\n                     N_cal: 100\n                 MU_margin: 0.5000\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 150\n                 tolerance: 0.0251\n         StopTrainingValue: 142.5000\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9500\n"}}
%---
%[output:356a8dc7]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\n","truncated":false}}
%---
%[output:6e97e48a]
%   data: {"dataType":"textualVariable","outputData":{"name":"agent_path","value":"'..\\Dataset\\dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Tx_60_Aligned_CDL.mat'"}}
%---
%[output:89327129]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:56d44986]
%   data: {"dataType":"text","outputData":{"text":"Running 100 evaluation episodes...\n","truncated":false}}
%---
%[output:664e78eb]
%   data: {"dataType":"text","outputData":{"text":"Evaluating step 10 \/ 31...\nEvaluating step 20 \/ 31...\nEvaluating step 30 \/ 31...\n","truncated":false}}
%---
