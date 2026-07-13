%[text] ## Evaluation of DQN vs Brute Force Localization Precision (Episodic Navigation)
%% Evaluation of DQN vs Brute Force Localization Precision on Random Positions (CDF)
% Executed until Done/Terminate conditions are met
clc; clear all; close all;

fprintf('=== Initializing Random Deployment Environment ===\n'); %[output:297b57d4]
% 1. Add required codebase folders to path
addingPathParentFolderByName('code');

% 2. Load Parameters
Parameters;  %[output:2d332872] %[output:1a640bd8] %[output:7be24bd5] %[output:8057c546] %[output:785bff6e] %[output:8d64b965] %[output:1ab170d2] %[output:4531c99d] %[output:74de7a26] %[output:4ba67eca] %[output:62de2bef] %[output:7ea865ee]

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

%Definition of the FFT precision
N_x=5 %[output:613e5436]
N_y=N_x;
N=N_x*N_y;

EnvPars.N_x=N_x;
EnvPars.N_y=N_y;
EnvPars.N=N;

% Meta-atom indexing — store inside EnvPars so closures don't depend on workspace
n = 1:EnvPars.N;
EnvPars.n_y = ceil(n ./ EnvPars.N_x);
EnvPars.n_x = n - (EnvPars.n_y - 1) .* EnvPars.N_x;

% Environment variables
n = (1:EnvPars.N);
n_y = ceil(n ./ EnvPars.N_x);
n_x = n - (n_y - 1) .* EnvPars.N_x;
n_psi = n;
n_psi_y = ceil(n_psi ./ EnvPars.N_x);
n_psi_x = n_psi - (n_y - 1) .* EnvPars.N_x;

%Recalculating the FFT
%Analytic FFT
% Kernel 2D-DFT == TO BE REPLACED BY SIM 1
G_func = @(n,n_psi) exp(-1i*2*pi*(n_psi_x(n_psi)-1)/EnvPars.N_x.*(n_x(n)-1)).* ...
    exp(-1i*2*pi*(n_psi_y(n_psi)-1)/EnvPars.N_y.*(n_y(n)-1));
[n_psi_grid, n_s_grid] = ndgrid(1:EnvPars.N, 1:EnvPars.N);
EnvPars.G = G_func(n_s_grid, n_psi_grid);

% Reclaculating the Upsilon matrix
EnvPars.U_func = @(n_, t_n_) exp(1i * ( ...
    -2*pi*(EnvPars.n_x(n_)-1) .* (EnvPars.t_x(t_n_)-1) / (EnvPars.N_x*EnvPars.T_x) ...
    -2*pi*(EnvPars.n_y(n_)-1) .* (EnvPars.t_y(t_n_)-1) / (EnvPars.N_y*EnvPars.T_y) ));


% Optional CST SIM1 front-end. Keep commented unless those fields exist.
% EnvPars.G      = EnvPars.G_CST;
% EnvPars.U_func = EnvPars.U_func_CST;

% 3. Run Calibration (Requires Parameters to be in the workspace)
% Calibration_rot;

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
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Aligned.mat') %[output:028c75e6]

if isfile(agent_path) %[output:group:1085363b]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:3d3ef454]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:1085363b]

%% ----------------- Start pool -----------------
delete(gcp('nocreate'));
parpool; %[output:746fcd98]
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
            % G_Upsilon = EnvPars.G_CST * Upsilon;
            G_Upsilon = EnvPars.G * Upsilon;
            
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

save_path = fullfile('..', 'Dataset', 'Evaluation_Rotation_x_y_pol_0_45_Tx_50_Nx_5.mat');

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
%[output:297b57d4]
%   data: {"dataType":"text","outputData":{"text":"=== Initializing Random Deployment Environment ===\n","truncated":false}}
%---
%[output:2d332872]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:1a640bd8]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:7be24bd5]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"5"}}
%---
%[output:8057c546]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"25"}}
%---
%[output:785bff6e]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:8d64b965]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:1ab170d2]
%   data: {"dataType":"textualVariable","outputData":{"name":"zeta","value":"0.9800"}}
%---
%[output:4531c99d]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:74de7a26]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:4ba67eca]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"50"}}
%---
%[output:62de2bef]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:7ea865ee]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     N: 25\n                   N_x: 5\n                   N_y: 5\n                     T: 2500\n                   T_x: 50\n                   T_y: 50\n                SNR_dB: 36.5005\n             theta_min: 1.8485\n             theta_max: 4.4347\n                    fc: 2.8000e+10\n                lambda: 0.0107\n               Ptx_dBm: 23\n               Gtx_dBi: 14\n               Grx_dBi: 8\n               txArray: [1×1 struct]\n                   cdl: [1×1 nrCDLChannel]\n          var_noise_dB: -110.9794\n                     r: 0\n                   d_x: 0.0054\n               pos_SIM: [5 5 4]\n                pos_MU: [5.9369 3.7053 1.5000]\n                   n_y: [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5]\n                   n_x: [1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5]\n                   t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 … ] (1×2500 double)\n                   t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 … ] (1×2500 double)\n                  h_MU: 1.5000\n                L_hall: 10\n                W_hall: 10\n                 N_cal: 100\n             MU_margin: 0.5000\n           MaxEpisodes: 5000\n                 psi_x: 0\n                 psi_y: 0\n    MaxStepsPerEpisode: 150\n             tolerance: 0.0251\n     StopTrainingValue: 142.5000\n       episode_counter: 0\n           delta_moves: [9×2 double]\n             n_actions: 9\n        DiscountFactor: 0.9500\n"}}
%---
%[output:613e5436]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"5"}}
%---
%[output:028c75e6]
%   data: {"dataType":"textualVariable","outputData":{"name":"agent_path","value":"'..\\Dataset\\dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Aligned.mat'"}}
%---
%[output:3d3ef454]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:746fcd98]
%   data: {"dataType":"text","outputData":{"text":"Starting parallel pool (parpool) using the 'Processes' profile ...\n13-Jul-2026 11:07:54: Job Queued. Waiting for parallel pool job with ID 1 to start ...\n\n","truncated":false}}
%---
