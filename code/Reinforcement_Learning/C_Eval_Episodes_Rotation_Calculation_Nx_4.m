%[text] ## Evaluation of DQN vs Brute Force Localization Precision (Episodic Navigation)
%% Evaluation of DQN vs Brute Force Localization Precision on Random Positions (CDF)
% Executed until Done/Terminate conditions are met
clc; clear all; close all;

fprintf('=== Initializing Random Deployment Environment ===\n');
% 1. Add required codebase folders to path
addingPathParentFolderByName('code');

% 2. Load Parameters
Parameters; 

%Interpolant functions for the real and imaginary parts
load T_jones.mat
t_yx_samples = t_yx;

t_yy_samples = t_yy;

phase_state_rad = mod(angle(t_yx_samples), 2*pi);

[phase_state_rad, idx] = sort(phase_state_rad);

t_yx_samples = t_yx_samples(idx);
t_yy_samples = t_yy_samples(idx);

F_tyx_real = griddedInterpolant(phase_state_rad, real(t_yx_samples), ...
    'pchip', 'nearest');

F_tyx_imag = griddedInterpolant(phase_state_rad, imag(t_yx_samples), ...
    'pchip', 'nearest');

F_tyy_real = griddedInterpolant(phase_state_rad, real(t_yy_samples), ...
    'pchip', 'nearest');

F_tyy_imag = griddedInterpolant(phase_state_rad, imag(t_yy_samples), ...
    'pchip', 'nearest');

EnvPars.F_tyx_real = F_tyx_real;
EnvPars.F_tyx_imag = F_tyx_imag;
EnvPars.F_tyy_real = F_tyy_real;
EnvPars.F_tyy_imag = F_tyy_imag;



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
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Tx_40_Aligned.mat')
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_4_Mx_5_Tx_50_Aligned.mat')
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Aligned.mat')

if isfile(agent_path)
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n');
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end

%% ----------------- Start pool -----------------
delete(gcp('nocreate'));
parpool;
%%
%[text] ### DQN Agent and Brute Force
%% Evaluation Setup
N_eval = 2000; % Number of episodes (random positions) to evaluate
angle_rot_deg=0:1:46;


fprintf('Running %d evaluation episodes...\n', N_eval);
err_bf  = zeros(N_eval, length(angle_rot_deg));
err_dqn = zeros(N_eval, length(angle_rot_deg));
steps_taken_dqn = zeros(N_eval, length(angle_rot_deg));

t_start = tic;

for k_rot=1:length(angle_rot_deg)

    fprintf('Iteration %.2f...\n', k_rot/length(angle_rot_deg)*100);

    EnvPars.angle_rot_deg = angle_rot_deg(k_rot);

    Calibration_rot;   % rotation-aware calibration

    parfor i = 1:N_eval
        % if mod(i, 50) == 0
        %     fprintf('Evaluating episode %d / %d...\n', i, N_eval);
        % end

        % ---------------------------------------------------------
        % DQN AGENT EVALUATION (Episodic Loop)
        % ---------------------------------------------------------
        % Reset environment: automatically picks a random pos_cal position
        % [obs, LoggedSignals] = resetFunction_nav(EnvPars);
        [obs, LoggedSignals] = resetFunction_nav_CST_Aligned_rot(EnvPars);

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
           [obs, reward, isDone, LoggedSignals] = stepFunction_nav_CST_Aligned_rot(action, LoggedSignals, EnvPars);
            step = step + 1;
        end

        % Record the number of steps it took to terminate
        steps_taken_dqn(i, k_rot) = step;

        % Extract the final position estimate after the agent finishes
        pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x, LoggedSignals.psi_y, EnvPars, pos_MU_true);
        pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pos_MU_true);
        err_dqn(i, k_rot) =norm(pos_est_dqn(1:2) - pos_MU_true(1:2))*1e2; %cm

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
            
            %Applying the rotation
            %Find the state of the atom
            xi_cmd = mod(angle(v0), 2*pi);
            %interpolate to find the transmission coefficient related to
            %that atom
            t_yx_cmd = F_tyx_real(xi_cmd) + 1i*F_tyx_imag(xi_cmd);
            t_yy_cmd = F_tyy_real(xi_cmd) + 1i*F_tyy_imag(xi_cmd);
            %find the rotation angle
            c_rot = t_yx_cmd*cosd(angle_rot_deg(k_rot)) + t_yy_cmd*sind(angle_rot_deg(k_rot));
            q_rot = c_rot ./ t_yx_cmd;

            %xy polarization


            r   = sqrt(db2pow(EnvPars.SNR_dB)) * G_Upsilon * (q_rot'.*a_psi_x_y);
            % r   = sqrt(db2pow(EnvPars.SNR_dB)) * G_Upsilon * (a_psi_x_y);

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
        err_bf(i,k_rot) = norm(pos_est_bf(1:2) - pos_MU_true(1:2))*1e2; % cm
    end
end

fprintf('\nparfor done: %.1f s total.\n', toc(t_start));
delete(gcp('nocreate'));
fprintf('Parallel pool released.\n');
%%
%[text] ### Saving results

save_path = fullfile('..', 'Dataset', 'Evaluation_Rotation_x_y_pol_0_45_Nx_5.mat');

save(save_path, 'err_dqn', 'steps_taken_dqn', 'err_bf', 'angle_rot_deg', 'EnvPars');

fprintf('Results saved to %s\n', save_path);
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
%   data: {"layout":"onright","rightPanelPercent":31.9}
%---
