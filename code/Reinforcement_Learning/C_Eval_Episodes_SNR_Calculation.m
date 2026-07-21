%% Evaluation of DQN vs Brute Force Localization Precision on Random Positions (CDF)
% Executed until Done/Terminate conditions are met
clc;
clearvars;

fprintf('=== Initializing Random Deployment Environment ===\n');
% 1. Add required codebase folders to path
addingPathParentFolderByName('code');

% 2. Load Parameters
Parameters; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining the DFT as ideal or with CST
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %Recalculating the FFT
% %Analytic FFT
% % Kernel 2D-DFT == TO BE REPLACED BY SIM 1
% G_func = @(n,n_psi) exp(-1i*2*pi*(n_psi_x(n_psi)-1)/EnvPars.N_x.*(n_x(n)-1)).* ...
%     exp(-1i*2*pi*(n_psi_y(n_psi)-1)/EnvPars.N_y.*(n_y(n)-1));
% [n_psi_grid, n_s_grid] = ndgrid(1:EnvPars.N, 1:EnvPars.N);
% EnvPars.G = G_func(n_s_grid, n_psi_grid);
% 
% % Reclaculating the Upsilon matrix
% EnvPars.U_func = @(n_, t_n_) exp(1i * ( ...
%     -2*pi*(EnvPars.n_x(n_)-1) .* (EnvPars.t_x(t_n_)-1) / (EnvPars.N_x*EnvPars.T_x) ...
%     -2*pi*(EnvPars.n_y(n_)-1) .* (EnvPars.t_y(t_n_)-1) / (EnvPars.N_y*EnvPars.T_y) ));

% CST SIM1 front-end. Keep commented unless those fields exist.
EnvPars.G      = EnvPars.G_CST;
EnvPars.U_func = EnvPars.U_func_CST;

%% ----------------- Start pool -----------------
% delete(gcp('nocreate'));
% parpool;

% 3. Run Calibration (Requires Parameters to be in the workspace)
Calibration;
%Load the trained DQN agent
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
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Tx_40_Aligned.mat')
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Tx_50_Aligned.mat')
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Tx_50_Aligned.mat');
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_ideal.mat');
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_CST.mat');

if isfile(agent_path)
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n');
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end





SNR_dB_vector =45;%0:1:30;

N_eval = 3000;    % Use 5000 or more for the final paper result
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
% delete(gcp('nocreate'));
% fprintf('Parallel pool released.\n');


%Defining the file directory and name
datasetDir = fullfile('..', 'Dataset');

if ~exist(datasetDir, 'dir')
    mkdir(datasetDir);
end

datasetDir = fullfile('..', 'Dataset');

if ~exist(datasetDir, 'dir')
    mkdir(datasetDir);
end

fmtNum = @(x) strrep(regexprep(regexprep(sprintf('%.4f', x), '0+$', ''), '\.$', ''), '.', '_');

if isscalar(zeta)
    zetaStr = fmtNum(zeta);
else
    zetaStr = sprintf('%s_to_%s', fmtNum(min(zeta)), fmtNum(max(zeta)));
end

fileName = sprintf('Precision_vs_SNR_%d_%d_Nx_%d_L_%d_Mx_%d_Zeta_%s_CST_LoS.mat', ...
    SNR_dB_vector(1),SNR_dB_vector(end),N_x, L, round(M_x), zetaStr);

savepath = fullfile(datasetDir, fileName);

save(savepath, ...
    'steps_dqn_snr', 'err_dqn_snr','SNR_dB_vector','-v7.3');

fprintf('Results saved to %s\n', savepath);


p95_error_dqn = prctile(err_dqn_snr, 95, 1)
%%

function addingPathParentFolderByName(targetName)
    % Start from the current directory
    currFolder = pwd;
    found = false;
    
    % Continue searching until you reach the root folder
    while true
        % Get the parent folder
        [parentFolder, currentName] = fileparts(currFolder);
        
        % Check if the current folder's name is the target
        if strcmpi(currentName, targetName)
            found = true;
            break;
        end
        
        % If we've reached the root or no change, exit the loop
        if isempty(parentFolder) || strcmp(currFolder, parentFolder)
            break;
        end
        
        % Move one level up
        currFolder = parentFolder;
    end

    if found
        addpath(genpath(currFolder));
        fprintf('Adding matlab path to: %s\n', currFolder);
    else
        error('Folder named "%s" not found in any parent directory.', targetName);
    end
end

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