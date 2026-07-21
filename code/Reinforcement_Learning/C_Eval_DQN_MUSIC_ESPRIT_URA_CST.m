%[text] ## Evaluation of Localization Precision (Episodic Navigation)
% =========================================================================
% Unified Baseline Benchmark: DQN vs. Brute Force vs. MUSIC vs. ESPRIT
% Runs on the exact same randomized episodic deployments.
% =========================================================================
clc; clear all; close all;

fprintf('=== Initializing Random Deployment Environment ===\n'); %[output:967af77d]
% Add required codebase folders to path
addingPathParentFolderByName('code');

% Load Parameters and Calibration
Parameters;  %[output:6f72c08c] %[output:06f68c98] %[output:6c58397d] %[output:1d92d124] %[output:66b1d22c] %[output:0146c4c8] %[output:480f87e8] %[output:22c12663] %[output:25087aa5] %[output:4b1d0ab1] %[output:0b2254f8] %[output:4f5fa235]
Calibration; %[output:00541869]

% Load the trained DQN agent
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Tx_50_Aligned.mat') %[output:9106ffda]

if isfile(agent_path) %[output:group:419431d0]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:066033fb]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:419431d0]
%%
%[text] ### DQN Agent and Brute Force
% Benchmark Configurations
N_eval = 500;              % Evaluation episodes (random positions)
K_snap = 16;               % Snapshots per point for digital estimators
Refine = true;             % 3-point parabolic peak refinement for MUSIC/DFT

% Preallocate error arrays (in cm)
err_dqn    = zeros(N_eval, 1);
err_bf     = zeros(N_eval, 1);
err_music  = zeros(N_eval, 1);
err_esprit = zeros(N_eval, 1);
steps_dqn  = zeros(N_eval, 1);

% Define physical constants matching Parameters / EnvPars
N_x  = EnvPars.N_x; N_y = EnvPars.N_y; N = EnvPars.N;
T_x  = EnvPars.T_x; T_y = EnvPars.T_y;
posS = EnvPars.pos_SIM(:).';
hgt  = EnvPars.pos_SIM(3) - EnvPars.h_MU;
dref = d_MU_SIM_max;

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

for i = 1:N_eval %[output:group:1fe0c4c2]
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval);
    end
    % -----------------------------------------------------------------
    % RESET ENV & GET TRUE UE POSITION
    % -----------------------------------------------------------------
    [obs, LoggedSignals] = resetFunction_nav_CST_Aligned(EnvPars);
    target_idx = LoggedSignals.pos_idx;
    pMU = EnvPars.pos_cal(target_idx, :); % True 3D coordinate [x, y, z]
    
    % -----------------------------------------------------------------
    % DQN EVALUATION
    % -----------------------------------------------------------------
    isDone = false;
    step = 0;
    while ~isDone
        action_out = getAction(agent, {obs}); %[output:697604e6]
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
    err_bf(i) = norm(pos_est_bf(1:2) - pos_MU_true(1:2))*1e2; % cm   %[output:group:1fe0c4c2]
%%
%[text] ### MUSIC and SPRIT
    % -----------------------------------------------------------------
    % GENERATE EQUIVALENT RAW SNR FIELD FOR TRADITIONAL ALGORITHMS
    % -----------------------------------------------------------------
    % Compute direction cosines
    dv = pMU - posS;
    d  = norm(dv);
    u_true  = dv(1)/d;  
    v_true  = dv(2)/d;
    
    % Path loss & element patterns scaling
    cosT   = hgt / d;
    ampRel = (dref/d) * cosT^2;
    snrLin = db2pow(EnvPars.SNR_dB) * ampRel^2;
    
    % Steer vector
    kappa  = 2*pi / EnvPars.lambda;
    a_true = c_cst .* exp(1i * kappa * (xe*u_true + ye*v_true));
    
    % Generate multi-snapshot digital covariance
    S = exp(1i*2*pi*rand(1, K_snap)); % Random phase symbols
    X = sqrt(snrLin) * a_true * S + (randn(N, K_snap) + 1i*randn(N, K_snap))/sqrt(2);
    R_cov = (X*X') / K_snap;
    
    % -----------------------------------------------------------------
    % 2-D MUSIC ESTIMATION
    % -----------------------------------------------------------------
    % Spatial scan grid
    u_ax = ((1:T_x) - (T_x+1)/2) * (2/T_x);
    v_ax = ((1:T_y) - (T_y+1)/2) * (2/T_y);
    [Ug, Vg] = ndgrid(u_ax, v_ax);
    vis = (Ug.^2 + Vg.^2) <= 0.98;
    A_vis = exp(1i * kappa * (xe*Ug(vis).' + ye*Vg(vis).'));
    A_est = A_vis.*c_cst;

    [E_noise, lam] = eig((R_cov + R_cov')/2, 'vector');
    [~, ix]  = sort(real(lam), 'descend');
    En = E_noise(:, ix(2:end)); % 1 Signal subspace, remaining are noise
    
    P_music = 1 ./ max(sum(abs(En'*A_est).^2, 1), 1e-18);
    [u_mus, v_mus] = pick_peak_local(P_music, Ug, Vg, vis, Refine);
    
    % Map back to position
    w_mus = sqrt(max(1 - u_mus^2 - v_mus^2, 1e-6));
    pos_est_music = [posS(1) + u_mus*hgt/w_mus, posS(2) + v_mus*hgt/w_mus];
    err_music(i) = norm(pos_est_music - pMU(1:2)) * 1e2;
    
    % -----------------------------------------------------------------
    % 2-D ESPRIT ESTIMATION
    % -----------------------------------------------------------------
    Es = E_noise(:, ix(1)); % Signal eigenvector
    
    J1x = find(nx_idx <= N_x-1); J2x = J1x + 1;
    J1y = find(ny_idx <= N_y-1); J2y = J1y + N_x;
    dxs = (xe(J2x(1)) - xe(J1x(1)));
    dys = (ye(J2y(1)) - ye(J1y(1)));
    
    Fx = pinv(Es(J1x)) * Es(J2x);
    Fy = pinv(Es(J1y)) * Es(J2y);
    
    u_esp = angle(Fx) / (kappa*dxs);
    v_esp = angle(Fy) / (kappa*dys);
    
    w_esp = sqrt(max(1 - u_esp^2 - v_esp^2, 1e-6));
    pos_est_esprit = [posS(1) + u_esp*hgt/w_esp, posS(2) + v_esp*hgt/w_esp];
    err_esprit(i) = norm(pos_est_esprit - pMU(1:2)) * 1e2;
end
%%
%[text] ### Figures
% -------------------------------------------------------------------------
% Plot Results
% -------------------------------------------------------------------------
figure; hold on; box on; grid on;
h_dqn = cdfplot(err_dqn);    set(h_dqn, 'LineWidth', 1.5);
h_bf  = cdfplot(err_bf);     set(h_bf, 'LineWidth', 1.5, 'LineStyle', '--');
h_mus = cdfplot(err_music);  set(h_mus, 'LineWidth', 1.5);
h_esp = cdfplot(err_esprit); set(h_esp, 'LineWidth', 1.5);

%patch
x_coords = [0    30   30 0];
y_coords = [0.95 0.95 1  1];
patch(x_coords, y_coords, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlim([0, 50]);
yticks([0,0.2,0.4,0.6,0.8,1])
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', 14);
legend({'DQN Agent', 'Brute Force', '2D MUSIC', '2D ESPRIT'}, ...
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12);
title('');

% Print Summary
fprintf('\n================ RESULTS SUMMARY ================\n');
fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n');
fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_dqn), median(err_dqn), prctile(err_dqn, 95), mean(steps_dqn));
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
%   data: {"layout":"onright","rightPanelPercent":41.2}
%---
%[output:967af77d]
%   data: {"dataType":"text","outputData":{"text":"=== Initializing Random Deployment Environment ===\n","truncated":false}}
%---
%[output:6f72c08c]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:06f68c98]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:6c58397d]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"4"}}
%---
%[output:1d92d124]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:66b1d22c]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:0146c4c8]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:480f87e8]
%   data: {"dataType":"textualVariable","outputData":{"name":"zeta","value":"0.9800"}}
%---
%[output:22c12663]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:25087aa5]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:4b1d0ab1]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"50"}}
%---
%[output:0b2254f8]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:4f5fa235]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"              channelModel: 'rician_los'\n                    fc_GHz: 28\n                    V_hall: 1000\n                    S_hall: 600\n                   mu_lgDS: -7.5916\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 7\n                sigma_K_dB: 8\n                        KR: 14.6330\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 0\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         T: 2500\n                       T_x: 50\n                       T_y: 50\n                    SNR_dB: 36.5005\n                 theta_min: 1.8485\n                 theta_max: 4.4347\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 23\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [5 5 4]\n                    pos_MU: [8.9462 5.0186 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 … ] (1×2500 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 … ] (1×2500 double)\n                      h_MU: 1.5000\n                    L_hall: 10\n                    W_hall: 10\n                     N_cal: 100\n                 MU_margin: 0.5000\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 150\n                 tolerance: 0.0314\n         StopTrainingValue: 142.5000\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9500\n"}}
%---
%[output:00541869]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:9106ffda]
%   data: {"dataType":"textualVariable","outputData":{"name":"agent_path","value":"'..\\Dataset\\dqn_agent_SIM2_BeamScanMAC_CST_1_layer_Nx_5_Mx_5_Tx_50_Aligned.mat'"}}
%---
%[output:066033fb]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:9e4a146d]
%   data: {"dataType":"text","outputData":{"text":"Running 500 joint evaluation episodes...\n","truncated":false}}
%---
%[output:697604e6]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"Error using <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('rl.internal.util.inferDataDimension', 'C:\\Program Files\\MATLAB\\R2026a\\toolbox\\rl\\rl\\+rl\\+internal\\+util\\inferDataDimension.m', 96)\" style=\"font-weight:bold\">rl.internal.util.inferDataDimension<\/a> (<a href=\"matlab: opentoline('C:\\Program Files\\MATLAB\\R2026a\\toolbox\\rl\\rl\\+rl\\+internal\\+util\\inferDataDimension.m',96,0)\">line 96<\/a>)\nData dimensions must match the dimensions specified in the corresponding specifications.\n\nError in <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('rl.policy.PolicyInterface\/getAction', 'C:\\Program Files\\MATLAB\\R2026a\\toolbox\\rl\\rl\\+rl\\+policy\\PolicyInterface.m', 30)\" style=\"font-weight:bold\">rl.policy.PolicyInterface\/getAction<\/a> (<a href=\"matlab: opentoline('C:\\Program Files\\MATLAB\\R2026a\\toolbox\\rl\\rl\\+rl\\+policy\\PolicyInterface.m',30,0)\">line 30<\/a>)\n            [batchSize,sequenceLength] = rl.internal.util.inferDataDimension(..."}}
%---
