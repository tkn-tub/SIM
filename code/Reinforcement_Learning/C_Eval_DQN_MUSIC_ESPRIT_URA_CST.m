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
Parameters; %[output:6f72c08c] %[output:06f68c98] %[output:6c58397d] %[output:1d92d124] %[output:66b1d22c] %[output:0146c4c8] %[output:480f87e8] %[output:22c12663] %[output:25087aa5] %[output:4b1d0ab1] %[output:0b2254f8] %[output:4f5fa235] %[output:790a4a29] %[output:00541869]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining the DFT as ideal or with CST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CST SIM1 front-end.
EnvPars.G      = EnvPars.G_CST;
EnvPars.U_func = EnvPars.U_func_CST;

% ---- CRITICAL: multipath must be ON, or K is inert -----------------------
% If nlosPowerScale = 0 there is no NLoS component and every K gives the same
% (pure-LoS) result. Force it on for the K sweep.
EnvPars.nlosPowerScale = 1;

Calibration; %[output:6350d298]

% Load the trained DQN agent
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_ideal.mat');
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_CST.mat');

if isfile(agent_path) %[output:group:419431d0]
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n'); %[output:066033fb]
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end %[output:group:419431d0]
%%
%[text] ### DQN Agent and Brute Force
% Benchmark Configurations
N_eval = 100;              % Evaluation episodes (random positions)
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

for i = 1:N_eval %[output:group:546e986d]
    if mod(i, 50) == 0
        fprintf('Evaluating episode %d / %d...\n', i, N_eval); %[output:0744fd13]
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
    % pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x, LoggedSignals.psi_y, EnvPars, pMU);
    pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pMU);
    err_dqn(i) = norm(pos_est_dqn(1:2) - pMU(1:2))*1e2; % cm
    
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
    pos_est_bf = estimatePosFromAngles(best_psi_x_est, best_psi_y_est, EnvPars, pMU);
    err_bf(i) = norm(pos_est_bf(1:2) - pMU(1:2))*1e2; % cm   %[output:group:546e986d]
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
figure; hold on; box on; grid on; %[output:8ca7c422]
h_dqn = cdfplot(err_dqn);    set(h_dqn, 'LineWidth', 1.5); %[output:8ca7c422]
h_bf  = cdfplot(err_bf);     set(h_bf, 'LineWidth', 1.5, 'LineStyle', '--'); %[output:8ca7c422]
h_mus = cdfplot(err_music);  set(h_mus, 'LineWidth', 1.5); %[output:8ca7c422]
h_esp = cdfplot(err_esprit); set(h_esp, 'LineWidth', 1.5); %[output:8ca7c422]

%patch
x_coords = [0    100   100 0];
y_coords = [0.9 0.9 1  1];
patch(x_coords, y_coords, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); %[output:8ca7c422]

xlim([0, 400]); %[output:8ca7c422]
yticks([0,0.2,0.4,0.6,0.8,1]) %[output:8ca7c422]
xlabel('Precision [cm]', 'Interpreter', 'latex', 'FontSize', 14); %[output:8ca7c422]
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', 14); %[output:8ca7c422]
legend({'DQN Agent', 'Brute Force', '2D MUSIC', '2D ESPRIT'}, ... %[output:8ca7c422]
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12); %[output:8ca7c422]
title(''); %[output:8ca7c422]

% Print Summary
fprintf('\n================ RESULTS SUMMARY ================\n'); %[output:3b24f5bc]
fprintf('Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n'); %[output:512f80f6]
fprintf('DQN Agent     | %13.2f | %11.2f | %20.2f (Steps: %.1f)\n', mean(err_dqn), median(err_dqn), prctile(err_dqn, 95), mean(steps_dqn)); %[output:6efe8e70]
fprintf('Brute Force   | %13.2f | %11.2f | %20.2f\n', mean(err_bf), median(err_bf), prctile(err_bf, 95)); %[output:2bc03e65]
fprintf('2-D MUSIC     | %13.2f | %11.2f | %20.2f\n', mean(err_music), median(err_music), prctile(err_music, 95)); %[output:7eaa459f]
fprintf('2-D ESPRIT    | %13.2f | %11.2f | %20.2f\n', mean(err_esprit), median(err_esprit), prctile(err_esprit, 95)); %[output:55a4ea6b]

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

% function pos_MU_est = estimatePosFromAngles(psi_x_est, psi_y_est, EnvPars, pos_MU_true)
%     % Shift phases back to principal domain [-pi, pi]
%     psi_x_est = mod(psi_x_est + pi, 2*pi) - pi;
%     psi_y_est = mod(psi_y_est + pi, 2*pi) - pi;
% 
%     phi_est = atan2(-psi_y_est, -psi_x_est);
%     % phi_est = atan(psi_y_est*EnvPars.d_x/(psi_x_est*EnvPars.d_x))
%     theta_est = asin(EnvPars.lambda/(2*pi)*sqrt(psi_x_est^2/EnvPars.d_x^2+psi_y_est^2/EnvPars.d_x^2));
% 
%     pos_MU_est(1) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*cos(phi_est)+EnvPars.pos_SIM(1);
%     pos_MU_est(2) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*sin(phi_est)+EnvPars.pos_SIM(2);
%     pos_MU_est(3) = pos_MU_true(3);
% end
function pos_MU_est = estimatePosFromAngles(psi_x_est, psi_y_est, EnvPars, pos_MU_true)
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
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"14"}}
%---
%[output:0146c4c8]
%   data: {"dataType":"textualVariable","outputData":{"name":"M","value":"196"}}
%---
%[output:480f87e8]
%   data: {"dataType":"textualVariable","outputData":{"name":"L","value":"12"}}
%---
%[output:22c12663]
%   data: {"dataType":"textualVariable","outputData":{"name":"zeta","value":"0.9800"}}
%---
%[output:25087aa5]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:4b1d0ab1]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:0b2254f8]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"60"}}
%---
%[output:4f5fa235]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"30.4577"}}
%---
%[output:790a4a29]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"              channelModel: 'rician_los_nlos'\n                    fc_GHz: 28\n                    V_hall: 72000\n                    S_hall: 18000\n                   mu_lgDS: -7.2781\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 7\n                sigma_K_dB: 8\n                        KR: 0.5858\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 1\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         M: 196\n                       M_x: 14\n                       M_y: 14\n                         T: 3600\n                       T_x: 60\n                       T_y: 60\n                    SNR_dB: 30.4577\n                 theta_min: 2.1410\n                 theta_max: 4.1422\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 24\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [5 5 10]\n                    pos_MU: [63.0611 48.0321 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 … ] (1×3600 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 … ] (1×3600 double)\n                      h_MU: 1.5000\n                    L_hall: 120\n                    W_hall: 60\n                     N_cal: 100\n                 MU_margin: 0\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 180\n                 tolerance: 0.0262\n         StopTrainingValue: 58\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9800\n"}}
%---
%[output:00541869]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 2.366%n","truncated":false}}
%---
%[output:6350d298]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:066033fb]
%   data: {"dataType":"text","outputData":{"text":"Trained agent successfully loaded.\n","truncated":false}}
%---
%[output:9e4a146d]
%   data: {"dataType":"text","outputData":{"text":"Running 100 joint evaluation episodes...\n","truncated":false}}
%---
%[output:0744fd13]
%   data: {"dataType":"text","outputData":{"text":"Evaluating episode 50 \/ 100...\nEvaluating episode 100 \/ 100...\n","truncated":false}}
%---
%[output:8ca7c422]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAA0gAAAH6CAYAAAA9emyMAAAQAElEQVR4AeydB3xUxdrG3yQEQigJSEswoSmWq6CAIkhTQREjqBTBrojyiYiCipUmV0EF8SqKFJV2QRAUiOi10oMFUK7XgrQkkNAEQgmEkPDlGTiHs5vdbN89Z\/fx5+yZM2fKO\/85CftkZt6Jrl69+ikGMuA7wHeA7wDfAb4DfAf4DvAd4DvAd4DvQPVT0cL\/SCBsCbBjJEACJEACJEACJEACJOAZAQokz3gxNwmQAAmYgwCtIAESIAESIAESCAgBCqSAYGWlJEACJEACJEAC3hJgORIgARIIJQEKpFDSZ9skQAIkQAIkQAIkQAKRRIB9tQABCiQLDBJNJAESIAESIAESIAESIAESCA4BCiRvObMcCZAACZAACZAACZAACZBA2BGgQAq7IWWHSMB3AqyBBEiABEiABEiABCKVAAVSpI48+00CJEACkUmAvSYBEiABEiCBMglQIJWJhw9JgARIgARIgARIwCoEaCcJkIA\/CFAg+YMi6yABEiABEiABEiABEiABEggcgSDWTIEURNhsigRIgARIgARIgARIgARIwNwEKJDMPT7haB37RAIkQAIkQAIkQAIkQAKmJUCBZNqhoWEkQALWI0CLSYAESIAESIAErE6AAsnqI0j7SYAESIAESCAYBNgGCZAACUQIAQqkCBlodpMESIAESIAESIAESMAxAaaSgJEABZKRBuMkQAIkQAIkQAIkQAIkQAIRTSDMBFJEjyU7TwIkQAIkQAIkQAIkQAIk4CMBCiQfAbI4CQSNABsiARIgARIgARIgARIIOAEKpIAjZgMkQAIkQAKuCPA5CZAACZAACZiFAAWSWUaCdpAACZAACZAACYQjAfaJBEjAYgQokCw2YDSXBEiABEiABEiABEiABMxBIDytoEAKz3Flr0iABEiABEiABEiABEiABLwgQIHkBbRwLMI+kQAJkAAJkAAJkAAJkAAJiFAg8S0gARIIdwLsHwmQAAmQAAmQAAm4TYACyW1UzEgCJEACJEACZiNAe0iABEiABPxNgALJ30RZHwmQAAmQAAmQAAmQgO8EWAMJhIgABVKIwLNZEiABEiABEiABEiABEiAB8xEIhkAyX69pEQmQAAmQAAmQAAmQAAmQAAk4IECB5ABKsJKaNWsmS5culbFjxwarSbbjdwKskARIgARIgARIgARIIJwIUCCFaDQTExNlwIABUq1atRBZwGZJgARIwAUBPiYBEiABEiCBCCRAgRSCQU9KSpJXXnlFLrroohC0ziZJgARIgARIgARIgARIgAScEaBAckYmQOlpaWkyadIkadq0qezbt0\/4HwmQAAmQAAmQAAmQAAn4kQCr8pEABZKPAD0p3qBBA7n\/\/vvVsrpvv\/1WvvnmG0+KMy8JkAAJkAAJkAAJkAAJkECACVAgBRiwffW7d++WV199VV544QUpLi62f2x7zzsSIAESIAESIAESIAESIIGgEqBACiLubdu2ySOPPCLp6elBbJVNkYA5CdAqEiABEiABEiABEjAjAQokM46KC5vg5MHdkFo3XtzNy3xJZJVEBvw54Dvgh3eAv0v4u4TvAN+BkL0DLr5G8rEbBCiQ3IBkpiz4h3vcfffJZ2PGyMwm02Xm5R+qsPiF11Ua0o3h87Ej5NeJF8iGt66UtRO6OsxjzM\/4GLcYfflUP1l7XRUVVt5+mVtlwo3tkrfekoULFzKQAd8BvgN8ByLqHeDvfbP\/2zdx4kQlzsz0\/dVqtlAgWWzEIJAuatRIFr6\/UE7lFcmp\/cXyd06OvP7GcBn70kulwqL580ROFEvsyQOy+dc1pZ47KhPuaYvmz5eowkKZ9u67XvNYNG+elMvbrcJfa5d7XY9VWWsMR48erc7zwpleDAM8ZjF16lT1G4gcPWenvW9k6D07MvSdncYQV76LvvMkQ\/8wbNasmfq3hR9eEDhThALpDAirXf744w8bk3HvKMTF7Nbz\/fK\/g+IoTySmAYqv\/UYdWvC1LiuWR9\/Xr18vDN4z2LBhAzCSoQ\/vERl6\/\/5pP7tk6DtDsCRH3zmSof8Yqn9c+OE1AQokr9GFtmD1uNq6AQcKzoogPfFMpHbNuDOxiLqwsyRAAiRAAiRAAiRAAiTgFQEKJK+whb5Qo4SmuhFb8n7R4\/aROrXOCqSf\/3fA\/jHvSYAELEeABpMACZAACZAACQSSAAVSIOmybhIgARIgARIgAfcJMCcJkAAJmIAABZIJBiGQJtQxLLHbted4IJuyTN2HDh2StWvXCq6WMdpkhoLdqlWrZP\/+\/SazzFrmgN+XX35Jjj4MGxn6AO9MUTI8A8LHCzn6CLCkeLgzLOki\/7cIAQqkEA7UW2+9Ja1atZKhQ4cGxArsP6pTq6Je9+69FEiAgS\/3EEiIM3hHAAxXr17tXWGW0gkcOHBAIJD0BEY8JkCGHiMrVYAMSyHxKoEcvcJmU4gMbXDwJoQEokPYtkWbNofZ1SqcddKw\/7hzJw2atbv2HNOivJIACZAACZAACZAACZAACTghQIHkBAyTSSAiCbDTJEACJEACJEACJBDhBCiQIvwFYPdJgARIIFIIsJ8kQAIkQAIk4A4BCiR3KFk0Tx2Di+9d3H9k0VGk2SRAAiRAAiTgkgAzkAAJ+JEABZIfYbKqyCHw330npcsnf6vwzKpDkdNx9pQESIAESIAESIAEgkog+I1RIAWfOVskARIgARIgARIgARIgARIwKQEKJJMOjCuzqsfV0bMcKNilx80coW0kQAIkQAIkQAIkQAIkYHYCFEhmHyEf7OMhsT7AY1ES8IwAc5MACZAACZAACYQJAQqkMBlIdoMESIAESIAEAkOAtZIACZBAZBGgQLLoeBsPirVoF2g2CZAACZAACZAACYSWAFsnAQcEKJAcQLFa0v7ju61msiXsrRUfLZfWKFcqWMJ4GkkCJEACJEACJEACJOAVgXARSF51PtwL1alVUe\/ibp6DpLNwJ3LnhRXlwxuqydi2CaWCO+WZhwRIgARIgARIgARIwJoEKJCsOW5SfKScbvmBAs4g6TD8FLnzong\/1eSPalgHCZAACZAACZAACZBAsAhQIAWLNNuxLIGNewvFGCzbERpOAmYkQJtIgARIgARIwGQEKJBMNiD+NKd2zTi9ul17julxRjwj8MyqQ2IMnpVmbhIgARIggUglwH6TAAlYkwAFkjXHzS2r69QyCCTuQXKLGTORAAmQAAmQAAmQAAm4JBDWGSiQwnp42TkSIAESIAESIAESIAESIAFPCFAgeUIrHPOyTyRAAiRAAiRAAiRAAiRAAjoBCiQdBSMkQALhRoD9IQESIAESIAESIAFPCVAgeUqM+UmABEiABEgg9ARoAQmQAAmQQIAIUCAFCGwgqy06HKNXv\/\/4Lj3OiP8IdPnkb9GC\/2plTSRAAiRAAiRAAq4JMAcJhJYABVJo+Qe09To2br6PB7QtVk4CJEACJEACJEACJEAC4UAgoAIpHACxD4EhUCs+WtwNnlrgqt7EcicFQcvnaf3MTwIkQAIkQAIkQAIkEL4EosO3a+yZGQlAlIxpU1U+vKGa28HTfriqe\/I1leTplFzBFXk9rf9Mfl5IgARIgARIgARIgATCkAAFUhgOqpm7VLtk5qhJzVgzm0jbSIAEhAhIgARIgARIIHIJUCBF7tiHvOe7jxaJO8FTQ13VeaAwRhC0fJ7Wz\/wkQAIkQAIWJkDTSYAESMAFAQokF4D4OHAEducXy\/1fHnQZPLXAVZ0PL8uX13YkC67I62n9zE8CJEACJEACJEACZiRAm\/xDgALJPxxZi5sE\/rvvpO4++5lVh9wsxWwkQAIkQAIkQAIkQAIkEBwCFEjB4exhK8xOAiRAAiRAAiRAAiRAAiQQCgIUSKGg7tc2o8qo7eyzs7EysvMRCQSDANsgARIgARIgARIgARMToEAy8eDQNBIgARIgAWsRoLUkQAIkQALWJ0CBZP0xZA9IgARIgARIgARIINAEWH+ICEQnJEnltOFS47kfXYaszu\/JzUvKhcjS8GmWAsmCY3nynNpSUDFJhfy4c2RXdLTDcKJ6nEjteBX2x5VzmMdZ2XBO3xMTI\/srVBBcw7mfgewb2B2K5XlWFvz1QZNJgARIgAQsRiAmIVnimqRZzGqrmWtrLwWSLQ9L3J2sWVNOlAgkhOxTJ2V3iUByFDRxhKuj55GatqdcOdkfF6cEY6Qy8LXfYHi4fHlL\/LzQSBIgARIgARIgARLwhAAFkie0TJK3KM+9qdPUuiWzR2dsztqZfyYW3hf2jgRIgARIgARIgATClUBh5jrZ9\/IVTkPqFw\/LkptPhmv3g9YvCqSgoWZDJEACJOATARYmARIgARKwOIH4tv1c7iOy32uUcNcki\/faeuZTIFlvzGgxCZAACZAACYQZAXaHBMKfQHRCksS3fSj8OxoGPaRACoNBtFIX5qVVs5K5tJUESIAESIAESIAEfCPgp9JFebmSv3KKn2pjNWURoEAqiw6fBYRAdr\/agtAqiZv8AwKYlZIACZAACZAACZiaAMROWXuJHD07MLGrFGatM3W\/wsU4CiT3R9JSOemgwVLDRWNJgARIgARIgATCjACW1CXcOUnfc1R9wOIw62H4docCKXzH1pQ9S6kco9uVfaRIjzMSagJsnwRIgARIgARIwJ8EYus1FwR\/1sm6gkOAAik4nNkKCZAACZBAqAiwXRIgARIIAYGYhCSHrWJ5Xf7KyQ6fMdEcBCiQzDEOHllRdDBWz7\/3WLYeZ4QESIAESIAESCCyCLC31iAAQaTtK8JeooKN6dYwPEKtpECK0IEPVbfPrXJ2id2Ow1xiF6pxYLskQAIkQAIkEOkEKjRJE+wRqjZgsQQi0KW3z29YyCqgQAoZejZMAiRAAiRAAiRAAiQQKgJxl6apPUJYCheIEKp+sV3fCVAg+c6QNbgicOY5Z4\/OgOCFBEiABEiABEgg5ASiE5ODYgP2HBVmrg9KW2zEPwQokPzDkbWQAAlEKAF2mwRIgARIwPoE8mb1l\/0TuwYkYM8Rzy+y1jtCgWSt8aK1JEACJEACJBAsAmyHBCxLIL5tP5f7irCsTutgUV6OFOflBiRobfBqHQIUSNYZK93SkwfL6fE9x7L0uDGSkhyv32bvzNfjjJAACZAACZAACZBAOBOITW0ucJAAAeQ8OHbBHc5c2Df3CVAguc+KOX0kkFL5rAe77CP0YOcjThYnARIgARIgARJwQCA60TPxc3xjupo5clAVkyKUgOUFUoSOmyW7nZF7QlKm7FahV\/oBS\/aBRpMACZAACZAACViHQGHmujL3FeFsoiPpI63TIVoaFAIUSEHBzEZIwCsCLEQCJEACJEACQSEQnZAU0DOB3DlnKPqumbK15TCJf3C+y\/1DZdVXJW24zqzIxb4iPSMjJGAgQIFkgMEoCZAACZBAsAiwHRIgATMRiGsS2DOBytoLZHxWGFddjPfexM3ElbZYkwAFkjXHjVaTAAmQAAmQAAmYlYAF7YpODbcBWQAAEABJREFUCM6ZQMFEg9mjgo3pwWySbYUJAQqkMBlIdoMESIAESIAESIAE\/EEgf+XkMvftBOq8oKJZd0vD70dJ\/rRefmmf5w\/5420oXUckpFAgRcIos48kQAIkQAIkQAKWI4CzfBLunOTTfpyy9uoYn2GJnQYIMy+BOhPIVb2xx\/dL0UH\/nEmk9YdXEvCUAAWSp8TCJj87QgIkQAIkQAIkYFYCcJqAs3xi6zX3eU9OTEKSyzqMHIoP5hpvGSeBiCNAgRRxQ84Ok0AEEGAXSYAESIAEvCKA2aPCrHVelWUhEggXAhRI4TKSdv1IrVtJT8nKOarHGSEBEiABErA2AVofeQQgWgK178e+XuzbiTzC7DEJ2BKgQLLlETZ3KXXj9b5k7czX44yQAAmQAAmQAAlYj4CrvTv+eh5iMmyeBExBgALJFMNAI0iABEiABEiABEiABEiABMxAIDACyQw9ow0kQAIkQAIkQAIkQAIkQAIk4CEBCiQPgVkle6phiV22SZbYtUoqL9n9aqswL62aVVCWspMJJEACJEACJEACJEAC4UuAAsmCY1uUV063es+xbD1uxsi5VWLkiWaVBILoieZnHUeY0VbaRAIkIERAAqYicNrVdT\/BWUBmD5I2VrKbPipxPd70i71V0oabaixoDAlEEgEKpDAd7dTks2Ika2fovNilVI6Rwc0rC2aPEMIUN7tFAiRAAiQQAAIxCcminQWE84B8C80l0OXzE8\/zaxsBQMoqSYAE3CBAgeQGJC1LuXLl5MEHH5SlS5fKmjVrZMWKFTJz5kxp27atlsXltXHjxvLuu+\/K8uXLJSMjQ11xj3SXhS2Y4aqkWIdWZ+SccJjORBIgARIgARLQCMTWa6ZFI\/pamMlziSL6BXCn88zjVwIUSB7gHDRokNxzzz1SVFQkX3zxhfz000+Smpoqzz77rHTo0MFlTRBBL7\/8sjRp0kS2b98uixYtUlfcIx3PXVZisQwpVWJ0i+dvOia90g+o8Mb60M1q6QYxQgIkQAIkYGoC0SUzSJqB+SsnS96s\/qYNsuRpSfnlbTn+8SC\/23gkfaSGgVcSIIEgEKBAchNymzZtpGPHjrJv3z4ZMmSIjBo1SgYPHiyTJ0+WSpUqSZ8+fQQzTGVVhzx169ZVs099+\/aVMWPGCK6YjUJ6nz59yipuyWfYg6QZnpFbKBm5J1TQ0nglARIgARIgAWcEYhKS9EeFmeulMGudicN6iT+4WTDb4287dQiMkAAJBIUABZKbmFu1aiUJCQnyww8\/yKZNm\/RS6enpsnPnTmnUqJG0a9dOT3cUwWzTiRMn5Mcff5STJ0+qLLjiHukNGzZUaeH6seNwUbh2jf1yiwAzkQAJkAAJkAAJkID5CVAguTlGEECFhYWSmZlpUyIvL0+ys7MlLi5OiSSbh3Y3mH2KjY2VOnXq2DypXr26xMTECOqyeRAGN3DSEAbdYBdIgARIoGwCfBoQAtGJyXq9RXk5epwREiABEggkAQokN+hi5qhq1aoCgZSVlVWqxO7du5XASU4++4u8VKaShJUrV8rRo0elS5cu0r17d7Uk79Zbb5WuXbuqGSU8L8nm8v+igwY33\/nmd\/OtdSj7CGeQNBa8kgAJkAAJkIBVCNBOEog0AhRIbow4ZngqVKggp06dUkLGvgiEE9Li4+NxcRqwHO+VV16R4uJiefLJJwWC6Omnn1b5X3vtNZk\/f76Ke\/KRlJQkWoCNWjDWEVcyu6WlB\/t63oyDooW9J8pJsNt31B5m8cqXL69m\/Rw9Z1oFl+NEhq4ZufMekaPvHMnQGgyjE5Ikvm0\/qZw23KMQU1JO+\/cslP+W8efZ9\/eMDP3PUPv+Z7xqPy+8uk3AYcZoh6lMDAgBeLqDMIJTh19++UU+\/vhjwRX3AwYMcMsTnr1hL774gixcuFCF5557Tlq0aKHC+Y2q6VmrVmuo0rRnkX5t1qyZXHDBBYJrpLPwtv9gR4anf9a8ZYhy5EiGeA9CHYLxHl7csXeJQHpI4pqkeRT0f8hKIpdccomp\/y0LBsdQvyuBbp8MPfud2L9\/f\/X9T\/seiOvEiRNLflr4v68EKJB8Jehmeah7vMiVK1eWxYsXC+Ljxo1T12nTpilPeEhDPjerVNleemm0QFwhfPDBB\/Lrr7+qcPx4gXqOj7\/++kulac\/C8nqm3+707Y8\/\/pCtW7fK\/\/73P3LxgJuRLRme\/jkzMvEmTo6+cyRDazDck9gU\/xx5HYrycuXPn1aY+nc230VrvIve\/K42axl878P3P2OYOnWq1z9nLHiWAAXSWRZOY\/v375eCggKJiopS+4bsM2KJB9IOHDiAi8PQvHlzqVWrluTm5srs2bNt8uA8JHjCgzhq3bq1zTNXN6hv\/fr1goCzleDoAcFY7tChPOUAAukMp1lgLxhZnGbhLQcy9I2fxt0djlpeXh0zJ0PHXDx5XwLNsDCuuv7PEs4zOpw+UjwJByZ2tcS\/Y4Hm6MmYWjUvGbr\/84zvffj+ZwwbNmzQf9YY8Z4ABZIb7PBL5tChQwIhhPOK7IvUrl1bHR77999\/2z\/S77E\/CZ7qjh8\/Lrt27dLTEUH98ISHOPLgykACJEACJEAC4UAA+4+0vUSYCcpfOUUKNqZrwa1rOHBgH0iABKxDgALJzbHasmWLEkj2ZxXBw11KSopA+CCPs+rg4htnHUEoYabImK9cuXLK0QLSioqKcGEgARIgARIggbAgEJNw1sNr8UG66g6LQWUn3CTAbFYlQIHk5shlZGSo6f0rr7xSGjdurJeCm26494Y4WrFihZ5uH1m3bp1aXofZpjvuuMPmcc+ePSU1NVU9X7Nmjc0z3pAACZAACZAACZAACZAACQSPAAWSG6yRZdWqVfL1119LjRo1ZMKECTJs2DAZP3683H\/\/\/ZKfny9z5szRXYA3aNBAFixYIMuXL5devXqhuBJX77\/\/vhw5ckS6desm06dPV66+J02apBw1YOYI3kewp0gVKOOjKC9Wf7r3mLnPQdINZYQESIAESIAESIAESIAELECAAsmDQXrzzTdl7ty5qkTnzp2Vu1EcHIuzjZYtW6bSy\/pAHuSFV7n69eurw2Ivuugi2bx5s4wYMULmzZtXVnFLPMvoXUOy+9VW4dwqMZawOcKNZPdJgARIwCcC0QlJUqFJmtPzjfDMpwZYmARIgASCTIACyQPgJ0+elHfeeUe6dOki8DbXrl07ufvuu9WBr8Zqtm3bpsRP+\/btS4keHA7bt29fwbNWrVqpK+6RbqyDcRIgARIgAV8JsHwwCMTWay5V0oaXeb5RMOxgGyRAAiTgLwIUSP4iabJ6UuvG6xZl7czX44yQAAmQAAmQgD8JxKY2d7u6wqx1budlRhcE+JgESCBgBCiQAoaWFZMACZAACZBAZBE4vjHd6flGebP6C1x8RxYR9pYESMAbAqEuQ4EU6hFg+yRAAiRAAiRQBgFtj098234SiBB9xT2yr15nibnyHq\/qxxI7zXzMEBWUiCRHAc+0fLySAAmQgJkJUCCZeXQsaJutYwYLdoAmkwAJkICJCEAcJdz1ntrjE9\/2oRIB4\/8AYfR3\/c5e1x2TkGQiYjSFBEiABHwnQIHkO0PW4ITAjsM89NYJGiaHAwH2gQSCRMBKAqT4YG6QqLAZEiABEggcAQqkwLENWc100BAy9GyYBEiABAJGoCgvV\/JXTvZ7KPphhpyz\/QvBVavfmyv2GHEZXcCGnxWTAAkEkQAFUhBhsykSIAESIAEScJcAltcZ9\/egXP7KKSUCyb+h+McZUiPzi5J6Ib68r5viCCPEYAECNJEEXBKgQHKJiBlIgARIgARIIPgEII5wvlDwW2aLJEACJBDZBKwrkCJ73Nh7EiABEiCBMCdgf75Q8cGcMO8xu0cCJEAC5iBAgWSOcaAVJGBDgDckQAIkYCSA84XyZvc3JjFOAiRAAiQQIAIUSAECG6nVpkzZLVqIVAbsNwmQQJkEIvIh9hNVaJImnpxjhCV2Gizu79FI8EoCJEACgSdAgRR4xkFvISU5Xm8ze2e+HmeEBEiABEgg+AQgjqoPWOzxWUZWcu8dfKpmbZF2kQAJhAMBCqRwGEX2gQRIgARIwLQE4kpmjnw1rpjnC\/mKkOVJgAR8JRBB5SmQwnCwr76ypt6rrJyjepwREiABEiCB4BMwOlsozFx3xp02XGq7Fw6njxQusQv+uLFFEiCByCVAgRTmY59VeoldmPeY3SMBEiAB8xLw5hyjgo3p5u0QLSMBEiCBMCRAgRSGg9rmirMzSKt\/2BuGPWSXSMAZAaaTgLkIYP+R5myhKC+XM0HmGh5aQwIkQAIOCVAgOcTCRBIgARIgARIwGQGaQwIkQAIkEBQCFEhBwRzcRlLqGrzY5dCLXXDpszUSIAESIAESIAFPCTA\/CZiJAAWSmUaDtpAACZAACQSUAJa84TyiYAVteV1AO8XKSYAESIAE\/ErAzwLJr7axMi8JpBpmkILhpGF8+6qS3a+2Cj0bV\/TSahYjARIggcATgGCpkjZcnUkUrGvge8UWSIAESIAE\/EmAAsmfNFlXeBNg70iABCxPIO7StJD1ofhgTsjaZsMkQAIkQALuE6BAcp8Vc5IACZBA2BKIxI4d35guwQr5KycLzjOKRM7sMwmQAAlYjQAFktVGjPaSAAmQAAl4TSA6MVkvC9FyJH2kBCPkr5wixXm5etuMBJUAGyMBEiABjwhQIHmEi5lJgARIgARIgARIgARIwCwEaEcgCFAgBYIq6yQBEiABEjAlgZiEJN0uzujoKBghARIgARIwEKBAMsAIZZRtkwAJkAAJkAAJkAAJkAAJhJ4ABVLox4AWkEC4E2D\/SMAtAt6cUSSNO0penSslrkmauHO2kVuGMBMJkAAJkEBEE6BACrPhD\/YZSGGGj90hARIIEQGIo4S73vP4fKLYTs\/IrgvuEFzdOdfI\/91jjSRAAiRAAuFGgAIp3EaU\/SEBEiABCxKISUgW4\/6gQHehiB7lAo2Y9YcDAfaBBCKUAAVShA48u00CJEACZiIQW6+Zbg7Ei7vnE8mmryRh1w8en2eUN+thvT1GSIAESIAEIo9AWT2mQCqLDp+RAAmQAAkEhUB0yQyS1lDBxiVun01U+NVYqfPnv6XwqzFul8G5R\/Rgp9HmlQRIgARIwJ4ABZI9EQvcFx0sp1u551i2Ho\/MCHtNAiQQDgRiDO63CzPXh0OX2AcSIAESIAGLEqBAsujAmcnswcsPScqU3SrM33TMTKbRFhKwNoEIsj46MVnvbVFejh5nhARIgARIgASCTYACKdjE2R4JkAAJkAAJkIAQAQmQAAmYlQAFkllHhnaRAAmQQBgRiE5IktjU5k5DGHWVXSEBEiABErA4AQokiw+gvfkpyfF6UvbOfD3OCAmQAAmEigDEEc44SrhrkjgLxj1IobKT7ZIACZAACZAACFAggYKzwHQSIAESIAGfCcQkJLt9xlFRXq7Qw5zPyFkBCZAACZCADx1kMl4AABAASURBVAQokHyAx6IkYGUCtJ0EQkEAAqgwc504C3DxHQq72CYJkAAJkAAJaAQokDQSvJIACZBAmBGIdrHvp6w9Qf58Fp2YpJMtPpgjebP7Ow35K6foeX2IsCgJkAAJkAAJeE2AAslrdCxIAiRAAuYlAHFUfcBip3t+nO0FCkR6lbTh5gVFy0jAcgRoMAmQQKAJUCAFmjDrJwESIIEQEIhrkhaCVtkkCZAACZAACfhAwCRFKZBMMhA0gwRIgAQCRcDVvh9n+4H8nX58Y7ocTh8ZqG6yXhIgARIgARLwCwEKJL9gjNxKsvvVFi0YKDBKAiTghACWvmmhMK66IMQkJouW5q+rsXk4Pihr30+wnh0pEUf0UGccGcZJgARIgATMSIACyYyjQptIgARMTMB70yB+sC9IC\/F958nWlsMEVy3NX9f4tg95byhLkgAJkAAJkEAEE6BAiuDBZ9dJgASCS4D7goLLm615QYBFSIAESIAEhAKJL4HXBM6tEqOX3XG4SI8zQgIk4JoA9gXFHt8vCIgHKmDfD11nux4P5iABEgh\/AuwhCbhLgALJXVLMRwIkQAIeEsCSOmMwFse+oPxpvaTh96Mkf2pPOTCxa0AC9v0Y22WcBEiABEiABEigbAIWFEhld4hPSYAESMAMBCo0SRP7\/UTcF2SGkaENJEACJEACJFA2AQqksvnwKQkElwBbCxsCcZeWfQ4RltSFTWfZERIgARIgARIIIwIUSGE0mOwKCZCAOQlADBkD9gUVbEw3p7EBtIpVkwAJkAAJkIAVCFAgWWGUaCMJkIClCRxZMtJmfxH3BVl6OGk8CTgiwDQSIIEwIkCBFEaDia6k1q2EiwpZOUfVNVAfKZXPerHLPkIvdoHizHqtR6By2nCJrdfceobTYhIgARIgARIoRSDyEiiQIm\/M2WMSIIEAEoDXOp53FEDArJoESIAESIAEAkyAAinAgM1UPW0hARIILgHsO8J+o8KsdcFtmK2RAAmQAAmQAAl4TYACyWt0LJiRe0JSpuxWoVf6AQIhgVASMG3b3G9k2qGhYSRAAiRAAiTgkAAFkkMs1k1MqRtvXeNpOQmYgEB8235S47kfvQ44+8gE3aAJYUWAnSEBEiABEggmAQqkYNIOcltZO\/OD3CKbIwHrE+BhrtYfQ\/aABEjAQgRoKgmYkAAFkgkHhSaRAAlYnwD2H+WvnGz9jrAHJEACJEACJBBhBPwlkCIMW+i6W3SwnN74nmPZepwREiAB\/xPY9\/IV4m04MLGrFGxM979RrJEESIAESIAESCCgBCiQAoo3+JWnGvYgZXOJnZ8GgNWEAwG438b5RK72F4VDX9kHEiABEiABEiAB7wlQIHnPjiVJgAQsRCAmIVk8OZ8IS+Qs1D3vTWVJEiABEiABEiABGwIUSDY4eEMCJEACIhBHBRuXEAUJkIDFCdB8EiABEvCGAAWSN9RYhgRIwNIECjPXlbm3CPuH8ldOsXQfaTwJkAAJkEBYE2DnAkiAAimAcENRdWpyJb3ZrJ1H9TgjJBCJBGJTm0vCnZPUmUYJd02KRATsMwmQAAmQAAmQgIcEKJA8BOb37KyQBEggYATi2\/aT2HrNA1Y\/KyYBEiABEiABEgg\/AhRIHoxpuXLl5MEHH5SlS5fKmjVrZMWKFTJz5kxp27at27XEx8fLkCFDbOr46KOPpHPnzm7XwYwkYBUCZrQT+4u4fM6MI0ObSIAESIAESMAcBCiQPBiHQYMGyT333CNFRUXyxRdfyE8\/\/SSpqany7LPPSocOHVzWBIE1fPhw6d69u+Tn58vixYvl559\/ljp16ijRdO2117qsw0wZ5qVVk+x+tVVolVTeTKbRFhIoRSBvVn\/Z9\/IVgv1FhVnrSj1nAgl4SIDZSYAESIAEwpQABZKbA9umTRvp2LGj7Nu3T4mZUaNGyeDBg2Xy5MlSqVIl6dOnj0AAlVXdXXfdJS1btpTff\/9dzUSNGTNGHnvsMVmyZIlUrFhRbr\/9dpd1lFU\/n5FAOBLAMjlXZxc5e87ldeH4RrBPJEACgSfAFkggsglQILk5\/q1atZKEhAT54YcfZNOmTXqp9PR02blzpzRq1EjatWunp9tHIJ7at28vJ0+elNmzZ8vBgwf1LJ999pns379fzjnnHGnYsKGe7ihy8mA5PXlPfrYeD0UkpXKM3mz2kSI9zggJ+IsAnCzEt33IX9WxHhIgARIgARIggUgn4Eb\/o93IwywlBCCACgsLJTMzs+Tu7P95eXmSnZ0tcXFxSiSdfWIbu\/TSS6V27dqyZ88eWbfOdnkPZpS6du0qPXr0sBFftjXwjgRIwFsC2HfEZXXe0mM5EiABEiABEogsAhRIbow3Zo6qVq0qEEhZWVmlSuzevVtiYmIkOTm51DMtoW7dukpE5eTkSLdu3WTRokWyevVqrxw9aHWG+npulbMzSDsOB30GKdTdZ\/tBJuDq7CLsL3IWsO8oyOayORIgARIgARIgAYsSoEByY+CqV68uFSpUkFOnTqklcvZFIJyQBg91uDoKeAYRdfHFF0vfvn0FM0+ffPKJ\/Pbbb8rRw7Bhw+TWW291VNRpWlJSkhgDbDRmxqwW0gIRjOJob2E5xScQ7QSiztjYWClfvrwSrIGoPxLq9JlhkzR1PlG1AYulrFD55uH6K12hZJY23Ng651jBUj9ToRwXMvT9XSFD3xniZ4AcfedIhp4xNH4H1OL6P5qM+EQg2qfSLOwxgcTERFm7dq088MAD8vrrr0v\/\/v2VNzv8coVAwmyVu5UmJdWRhQsX6uG5556TqglV9eLnnX++tGjRIjCheXO9HUQC1k4A7G\/WrJlccMEFgquV7DaTrWDnC8Pabe5U5xPFJCSJq4D3CyGu5I8UZmLgD1t85egPG6xeBxn6\/js+6AwD8HvdDO8xOfJdDPZ7iO+Qxu+BiE+cOBH\/ZDL4SIACyUeAnhaHc4b58+fbzET9+9\/\/FizTSyqZEWpuJzzKqn\/9+g0yYMAAPXzwwQdSUFCgF9n811\/y66+\/BiQc27X1bDv78gPSRqBs\/+OPP2Tr1q3yv\/\/9z1J2B4qHN\/X6yvBwVCX9\/XEngj1E+79+K+zGy1eO3oxduJUhQ99\/x5Oh7wzxc0WOvnMMB4Z4F4IV8L3P+D0Q8alTp7rzzyrzuCBAgeQCEB7DwxyER1RUlEM33JgSRr4DBw7gUmY4duyY\/P333zZ5cnNz1blIWPZVo0YNm2eubtavXy9a2L59uxQcL9CLHD9+XC3lw3I+f4dDeYf0dhDxd\/2Bru\/o0aMBYxNo281Svy8M8c5oAecT7Z\/YVcoK2EN0ZNOasBwzXzia5V0ItR1kmOfzzwYZ+s4QPwfk6DtHMnSfIb73ad8BteuGDRu0f1559YEABZJDeLaJ+KV36NAhgRCCswXbpyLwTldUVFRK+BjzwRW4tlfJmG7V+FVJsbrpGTkn9DgjkUOgMK66wAV3WXuInD3DsjqNVFFejhTn5ZYZtLy8kgAJkAAJkAAJkECgCVAguUl4y5YtSiDZn1OEPUMpKSmC2RrkcVYdnmHmCN7wGjRoYJMN95UrVxbMUsFluM1Dk96kGD3YHSk2qZU0yyEBPyVCIMVceY\/LPUQQQ\/bBTyawGhIgARIgARIgARLwOwEKJDeRZmRkqCUMV155pTRu3FgvBccKcO8NAbRixQo93T6ya9cuwbRnpUqV5M4775TExEQ9y2233SY1a9YU1PHjjz\/q6YyQgJkJQCD5ah9cd2P2yNd6WJ4ENAK8kgAJkAAJkICvBCiQ3CS4atUq+frrrwV7hCZMmCBwyz1+\/Hi5\/\/771f6hOXPm6I4XMCO0YMECWb58ufTq1UtvYdKkSYJDYS+66CLBJrpnnnlGkIZDYo8cOSIfffSRXodeiBESsAABCJ2y9hA5e5Y3u78FekcTSYAESMAUBGgECZBAkAhQIHkA+s0335S5c+eqEp07d1bus7OysuSVV16RZcuWqfSyPuDBbsiQIfLNN9+oGaRu3boJxNLmzZtl+PDhbtVRVv3BfDZ4+SFJmbJbhfmbjgWzabYVYgLxbfuJpI2VXRfcoVtS5GIPEWaJHAW9AkZIgARIgARIgAQimIC5uk6B5MF4nDx5Ut555x3p0qWLtG7dWtq1ayd33323rFy50qaWbdu2Sffu3aV9+\/Yyb948m2cQSS+++KJ07NhRWrVqpfL07dtXuLTOBhNvTEogOiFJOWaIrWd7DpZJzaVZJEACJEACJEACJOAxAQokj5GZu0BK3XjdwOycfD0erAjbiTwCmD0q2JgeeR1nj0mABEiABEiABMKSAAVSWA4rO0UCgSEQk5CsVxx\/cLPkT+slOKOoMGudnh7GEXaNBEiABEiABEggAghQIEXAILOLJOAvAtGJSXpVscf3S9HBHP2eERIgASsToO0kQAIkQAIaAQokjQSvJEACLgngPCMtU7kSgaTFeSUBEiABEiAB0xKgYSTgIQEKJA+BMTsJkAAJkAAJkAAJkAAJkED4EjC9QBo7dqzgkNaMjAx1rtDbb78tderUCd8R8bFnqQYnDVk76aTBR5wsTgIkQAIkQAIkQAIkEGEETC+QtPGAG+zevXvLo48+Kvv27ZOmTZtKy5YtSwWklytXTivGKwlYhIC5zYxW7r37SYUmN5vbUFpHAiRAAiRAAiRAAj4SsIRAOnLkiHz66aeSm5urulupUiXBIavPPvusvPHGGzJhwgQZP368PP3009KpUyfBc5WRHyRAAn4hgHOP4ts+JMY9SH6pmJVEBgH2kgRIgARIgAQsRMASAunYsWO6OALbvLw8GTVqlPTo0UNWr16NJFmzZo3cfvvt8vrrrwueq0R+kAAJ+IWAvTCCB7v4vM1+qZuVkAAJkICVCdB2EiCB8CNgCYFUWFgox48fL0X\/5MmTsn79ejl69Ki64r5UJiYEhEB2v9qihYA0wEpNS+D4xnR1\/hHOQTKtkTSMBEiABEiABEjAVwIRW94SAqms0SkqKpJTp04JrmXl4zMSIAHPCTjae1Scx7OPPCfJEiRAAiRAAiRAAlYhYHmBZBXQIbWTjZOAlwTimqQJ9x55CY\/FSIAESIAESIAELEkgYgQS3IUPHDjQkoNkNqPPrRKjm7TjcJEeZyT8CEQnJNt0qigvVwoz19ukhfqG7ZMACZAACZAACZCAPwlEhEBKSEiQlJQUf3JjXSQQcQTyV06WAxO7SmHWuojrOztMAiEiwGZJgARIgARCQMASAqlGjRry8MMPy5AhQ0qF5s2bS2xsrODq6Pkzzzwj77zzjtSrVy8EeIPbJA+JDS7vcGoNe40qpw0X+wD33lo\/MXukxXklARIgARIgAd8IsDQJmJeAJQRS+fLlpX379sqtN1x7G0O7du2kQoUKgqsxXYvjvKSGDRtKdLQlumreN4WWhTWBKiXiCPuN7ENMQlJY95udIwESIAESIAESIAF7Aj6rBvsKeU8CJGA9AtGJtnuN7HuA2aOCjen2ybzRQlV9AAAQAElEQVQnARIgARIgARIggbAjYAmBdOLECfnll1\/k448\/9jh8\/vnnsn37dikuLg67wQtVh1Iqn3XSkH0krJ00hApxSNvFXqPD6SPFGLD3KKRGsXESIAESIAESIAESCBIBSwgkiKP+\/fvLuHHjPA6jRo0SlM3MzAwSUjZDAqEh4Gwfkf2+Ikf3xqV0OAgWs0XGEJoesdXAEmDtJEACJEACJEACjgiYXiAdP35czQA5Mt7dtLy8PPnrr78EdblbhvlIwGoEnO0jst9X5Ojean2lvSRAAiRQJgE+JAESIAEfCJheIA0fPlzGjx\/vQxdPF0U9U6ZMOX1j4c9a8Wfdle89lm3hntB0fxMwepzztm7sNSrOy\/W2OMuRAAmQAAmQAAkEmACrDzwB0wskZwgaN26sls698cYbMm3aNBWwnK5Xr16SmJjorBjT\/UAgI\/eEpEzZrUKv9AN+qJFV+JuAcf+QJ3HuNfL3SLA+EiABEiABEiABqxGwnEBq27atzJ49Wz744AO599575aqrrpKLL75YhU6dOskTTzwhixYtkrfeeksgosw7ILSMBPxDAHuPKjRJs6nMuH\/Ik7hNJbwhARIgARIgARIggQgkYCmBNGDAABk9erS4OtcI5ya1aNFCJkyYIGlptl8cw3mMU5Lj9e5l78zX44yENwEsrcP+I1P1ksaQAAmQAAmQAAmQgEUJWEYg9enTR3r27CkQP2B96tQpOXDggPz++++yYsUKFRBHmubSu1q1avLII49Ihw4dUISBBMKSgNEDHTpYmLkOFwYSIIEAEWC1JEACJEAC4U3AEgKpSZMm0rt3b6lQoYJAGMEjHZbSdenSRR544AEZOnSoCogj7f7775eMjAwpLCwUiKR+\/fpxX1J4v8fs3RkCcNGdN7v\/mTteSIAESIAESMAjAsxMAiRQQiC6JJj+\/xtuuEFq1qypBM\/SpUuVKPr++++d2r1p0yYZPHiwzJw5UwoKCqRu3bpy0003Oc3PByQQLgSK83LCpSvsBwmQAAmQAAmQAAn4kYD7VZleICUkJMjll1+ueoRZoTFjxsjJkyfVvasPuPX+4osv1LI8OHdwld\/qz1PrVtK7kJVzVI8zQgIkQAIkQAIkQAIkQAIk4B4B0wskeKirUaOG5ObmyqRJk9wWR1r3Fy5cKPv27ZM6depIo0aNtGReLUyAppMACZAACZAACZAACZBAoAiYXiDVr19f7T368ccfZdu2bR5zwHK7X375RSpVqiT16tXzuDwLkAAJkEAQCbApEiABEiABEiCBEBMwvUDC7FFRUZH8+uuvXqPasmWLREVFCeryuhIWJAETEsAZSNEJySa0jCaRAAmQgD0B3pMACZCANQiYXiAlJyfLiRMnZNeuXV4TxRK76OhoqV27ttd1sOBpAhm9a0h2v9oqnFsl5nQiP0NCAOKo+oDFEmd3SGxIjGGjJEACJEACJBDJBNj3sCJgeoEEYXP06FH5+++\/vQa\/d+9ewSyU1xWwIAmYkECMg5mjwsz1JrSUJpEACZAACZAACZCAdQiYXiAFGSWbIwFLEijKy5W8Wf2lMGudJe2n0SRAAiRAAiRAAiRgFgKWEEjYO9SxY0dp2bKlV+Gaa65Rjh7MAt2qdrRKKi9cVhf60YtOSJL4tv2kgmFpXfHBHDfEUehtpwUkQAIkQAIkQAIkYHYClhBI5cuXV4fDTpgwQSZ4Ebp16yaxsbFmHwtT2wdxNC+tmqltjATjII6w7yi+7UPcexQJA84+uk+AOUmABEiABEjATwQsIZD81Newryalbrzex6yd+XrcH5GrkmwF5o7DRYLgj7pZh\/sEYus1d5gZS+wcPmAiCZAACZCA5QmwAyRAAsElYAmBdOrUKcnPz5edO3fKb7\/95lGAgwZ4wQsu1vBuLSP3hAxefii8O2mB3kEU5a+cLIfTR8qRkmABk2kiCZAACZAACZAACRgJmDJuCYG0bds26d69u\/To0UP69u3rUejatasMHDhQ9uzZY8oBsKJRGTknBCLJirZbwWYso8P+Iuwzsg9xl6bpXSjMXCf5K6dIwcZ0PY0REiABEiABEiABEiAB3whYQiBt2rRJDh486HVPN27cKDgLyesKWNA1AebwGwEso6uSNlywz8g+4JnfGmJFJEACJEACJEACJEACpQiYXiBNmjRJ3nvvvVKGe5owevRomT9\/vqfFmJ8Egk4gNtXxPiN7Q+jS255I4O5ZMwmQAAmQAAmQQOQQML1AwvK6Xbt26SMSHx8vzZo1k6ZNm0q5cuX0dEeRW2+9VXm\/wzP7epDGQAJmJ3B6Gd1kyV9pG7DviEvrzD56tI8ELEGARpIACZAACdgRMLVAqlOnjsyePVsyMjL08M0338irr74qd9xxh1SqVMmuO7a3eN6nTx8ZMmSI7QPeeUzgjfVHJWXKbhUQ97gCFtAJaHuMsM\/IUYhJSNLzHv9veok4mlIqUBzpiBghARIgARIgAScEmEwC3hEwtUDCzNG0adPkyJEjqnfwZPfBBx9I586dZejQoZKXl6fSnX3MmjVLIKhuuukm6dmzp7NslkqvVTFFt3fPsSw9zog1CEAcJdz1nmCPkbPAfUbWGEtaSQIkQAIkQAIkEJ4ETC2QgBxL6mJiYqSgoECmTp0qkydPlpMnT+KRWwF7mHJzc5WoSkhIcKsMM5FAoAjEJCRLjGGGyFU7xQdzXWXhcxIgARIgARIgARIgAT8SML1AatiwodprlJ6eLnPmzPG46\/B+t3LlSklNTZW2bdsK\/yMBsxDAOUbHN6bL8Y3pDgP2GdERg1lGi3aQAAmQAAmQAAlECgHTC6TzzjtPufhevHix12Py3\/\/+V6KioqRx48Ze18GCJOAPArH1munVFGxcog54PZI+0uGV+4x0VIxYlgANJwESIAESIAHrETC9QMKyuJycHMFZSN7iRXnsV6pdu7a3VbAcCZAACZAACZAACZwlwBgJkEDYEjC1QMKyOHiig7jxxwjAK54\/6mEdJOAtAeMZR4WZ672thuVIgARIgARIgARIIGAEIr1iUwskOGeIiooSzCL5MlDnnHOOS5fgvtTPsiRAAiRAAiRAAiRAAiRAAuFBwNQCCYe7wsV3zZo1pUGDBl4Tv+SSS6RixYqyd+9er+uwZkFabTYCsfWa6yYV5eXocUZIgARIgARIgARIgATMQcDUAgmIIGogkDp16oRbj0O5cuWkVatWUr58edm3b5\/H5VmABPxFINrg3rsoL1eKS4K\/6o7IethpEiABEiABEiABEggAAdMLpF9\/\/VVOnTolt9xyi3To0MFjBIMGDZKLLrpIjh07JqjL4wpYQOalVZPsfrVVaJVUnkRIgARIgAQCTIDVkwAJkAAJhI6A6QXSf\/7zH9mzZ49Uq1ZNhgwZImlpaW7RwszRY489Jl27dpXY2FjZsWOH4DwktwozEwkEgEBMQrJea\/FBLq\/TYTBCAiRAAiQQSQTYVxIwPQHTC6Tc3Fz57LPPpKCgQGrUqCHPPvusTJ8+Xbp37y6OvNLhrCMIqQULFkjv3r3V0jqUXbJkifjLG57pR5UGkgAJkAAJkAAJkAAJkAAJeEXAe4HkVXPeFZo1a5Z8\/fXXUlhYKNHR0QIR9OSTT8onn3wiy5cvl6+++kqF1atXK\/HUo0cPqVWrlkRFRUlxcbGsXbtW5fWudZYiAf8QiK3XTK8Ie5D0G0ZIgARIgARIgARIgARMQ8ASAunkyZMyZswYmT9\/vppJMtKD84XKlSsLAsST8dmJEyeUMHrhhRcEdRifMU4CZREI9LNierALNGLWTwIkQAIkQAIkQAJeEYj2qlQICkHgvPXWW\/LQQw\/JihUrJD8\/36kVWFL3008\/Sb9+\/eT111+PGHGUWjdeZ5K90zkfPRMjJEACkUiAfSYBEiABEiABEiiDgGUEktaHTZs2ydChQ+WGG26Q\/v37y9SpU+Xjjz9WAXuTBgwYIB07dpSBAwcK8gr\/I4EQEohOSJLY1OYqRBucNITQJDZNAiRAAmFMgF0jARIgAd8JWE4gaV3GjNIvv\/wi06ZNk3HjxqkwadIkWb9+fcTMGGkseDUnAYij6gMWS8Jdk1SIa5JmTkNpFQmQAAmQAAmQgPkJ0MKgEbCsQAoaITZEAl4SiCljxohOGryEymIkQAIkQAIkQAIkEGACFEgBBuygeiZFCIHoxCS9pxBEhZnrBCF\/5WQp2JiuP2OEBEiABEiABEiABEjAPAQokMwzFm5ZUqtiqp5vz7FsPc6I+QjEJJwVSBBGebP7C0L+yinmM9ZvFrEiEiABEiABEiABErA2AQoka49fUKx\/Y93RoLQTTo0k3DlJ4ts+FE5dYl9IgARIgARIgARIICIIUCB5MMzlypWTBx98UJYuXSpr1qxR7sZnzpwpbdu29aCWs1lR32uvvaYOu+3Vq9fZByaLZeSeMJlF5jYnumTmKLZecxsjee6RDQ7ekAAJkAAJmIwAzSEBEjhLgALpLAuXsUGDBsk999wjRUVF8sUXXwjOWkpNTZVnn31WOnTo4LK8fYbbb79drrzySvtkU95DJI1fd0RwNaWBJjUKe4+Ob0wXBJOaSLNIgARIgARIgARIIJwJeNw3CiQ3kbVp00adr7Rv3z4ZMmSIjBo1SgYPHiyTJ0+WSpUqSZ8+fQQzQm5WJ40bN5bu3btL+fLl3S0S0ny90g\/IG+u51K6sQcDMEYJ9niPpI6U4L9c+mfckQAIkQAIkQAIkQAImJECB5OagtGrVShISEuSHH36wOYA2PT1ddu7cKY0aNZJ27dq5VRuEVN++feWcc86RrKwst8owk4iYGAKEUcJd7wnOPUIwsak0jQRIgARIgARIgARIoAwCFEhlwDE+ggAqLCyUzMxMY7Lk5eVJdna2xMXFKZFk89DJTc+ePaVly5ZKbG3fvt1JLs+TU5Mr6YWydnK2R4cRhEhMQrLEJCSVaqn4YE6pNCaQgCMCTCMBEiABEiABEjAHAQokN8YBM0dVq1YVCCRHMz67d++WmJgYSU5OdllbgwYN1NI6CCs4eHBZgBksQSDa7swj7D1COJw+0hL200gSIAESCCABVk0CJEACliJAgeTGcFWvXl0qVKggp06dkpMnT5YqAeGExPj4eFzKDA8\/\/LDUrFlTPvnkE9m4cWOZefnQmgRw5tGBiV0FgXuPrDmGtJoESIAESIAE3CPAXOFIgAIpiKPatWtXueKKK+S3336TWbNm+dxyUlKSGIOxQiz5g6hjqKDErZFDbGysco7hT0bG5XWxx\/eXatPYfjjEA8EwHLh42gdyLP3zSYa+MyHD4DMEc\/48+86dDD1jaPwOqMWN3wUZ954ABdIZdoG+4MW94447pKCgQKZMmeJwJspTGx7s+6AsXLhQD5c3u1yv4rzzz5cWLVowOGDQrFkzueCCCwRXfzFKSUnV2deqVSvsuYOdvxn6ayysVA85+v47igzJ0Cw\/83wX+S4G+13s37+\/\/h1Q+z44ceJE\/fsIfpRdWAAAEABJREFUI94ToEByg93+\/fuVsImKinLoyht\/8UA1Bw4cwMVhwEsMkYTzk9avX+8wj6eJU6dNlQEDBuhhV26uXsXmv\/6SX3\/91atQ+e8\/Ze7Fm1Xwtg4zl\/vjjz9k69at8r\/\/\/c8rPo76tmfPHp094o7yhDDNb\/3U+hAIhlrdkXQlR+9+RxnfETIkQ+P7EMo430W+i8F+\/z744AP9O6D2fXDq1Kn69xFGvCdAgeQGOzhUOHTokEAI1a1bt1SJ2rVrq8Nj\/\/7771LPkADHDJdccola1oXzkjIyMkQLcA2Os5CeeOIJWb58ufTq1QtF3Aq5JYIIYksLubm79HLHjx9XHvZgu6fh4vhjej0PNDrpdT2ethvM\/EePHvWpX4clXoyhoOC4zgzxYPYlVG35yjBUdputXXLM8+lnMS8vT8iQDM3yc813ke9iMN9FeELWvgNq1w0bNujfRxjxngAFkpvstmzZogRSw4YNbUrAw11KSopAkCCPzcMzN5iBWrp0qXz88celwl8lMz0nTpyQFStWKMcN2J90phgvJiUQnZAkxjOPcO5RfNuHTGotzSIBEiABErAsARpOAiQQEgIUSG5ix4wP\/ipw5ZVXSuPGjfVSt956q3LvDXEEkaM\/MERQbtq0aTJu3LhSAbNAyLpu3TqZMGGCWgqF+1CGlCoxevM7jhTrcUZOE4hrkubwzKPTT\/lJAiRAAiRAAiRAAiTgioCZn1MguTk6q1atkq+\/\/lpq1KihhMywYcNk\/Pjxcv\/990t+fr7MmTNHd7yAJXULFizweMmcm6Ywm4kI4KwjYzi+MV3yV04xkYU0hQRIgARIgARIgARIwBMCFEge0HrzzTdl7ty5qkTnzp2VpzIcHPvKK6\/IsmXLVHo4fJxrnEE6XOSiS5HzGEvrEIw9Lti4RJ13hDOPEI6kjzQ+ZpwESIAESIAESIAESMBiBCiQPBgwHBL7zjvvSJcuXaR169YCBwt33323rFy50qaWbdu2Sffu3aV9+\/Yyb948m2f2N0OHDnUrn305R\/cpdc8eVJudk+8oC9O8JABhpO074n4jLyFasRhtJgESIAESIAESiDgCFEgRN+TssDcEIIqMh8F6UwfLkAAJkICZCNAWEiABEiABxwQokBxzYSoJ2BAwiiNtz1Fh5jruN7KhxBsSIAESIAESMAUBGkECPhGgQPIJHwtHIoEjS0aqfUd5s\/tHYvfZZxIgARIgARIgARIIawLmFkhhjd67ztWsmKIX3JOfrccZCSyB6MRkvYGivBw9zggJkAAJkAAJkAAJkEB4EaBACq\/xZG8CRMC4xK44L9cvrbASEiABEiABEiABEiAB8xGgQDLfmNAiEiABErA6AdpPAiRAAiRAApYlQIFk2aELnOG90g9IypTdKmTknghcQ6yZBEiABEiABCxHgAaTAAmEOwEKpDAa4VTDOUhZO\/PDqGeB7UqFJmlS47kfywyBtYC1kwAJkAAJkAAJkIAJCNAERYACSWHgRyQTiLs0ze3uF3H\/kdusmJEESIAESIAESIAErEiAAsmKo+baZubwgIDRQ11ZxSCOCjYuKSsLn5EACZAACZAACZAACVicAAWSxQeQ5vuXwP6JXWXfy1c4DAdKnuWvnOLfBlmbFwRYhARIgARIgARIgAQCR4ACKXBsWbPJCcSd2XtkdOFtcpNpHgmQQLgTYP9IgARIgARCToACKeRDQANCRSC20zOhaprtkgAJkAAJGAgkJSWJ1UO1atUEwer9CKT9ruoGPwRX+cL9ueFHg9EQEaBAChF4fzdLD3aeES2Mq25TAPuL8ldOlmI6YbDhwhsSIAESCDQBfNl94YUXZOHChZYOc+bMkeeff15wtXpfQmU\/2JHhQpk4caL6g0Ggf\/ZYv3MCXggk55XxCQlYkQDEEfcXWXHkaDMJkEA4EIBAatasmYwePVoGDBjAQAYR\/Q5MnTpV8PMQDj\/bVu4DBZKVR4+2e03AOINUfDDnbD2MkQAJkAAJhITA+vXrhYEMIv0d2LBhQ0h+\/tioLQEKJFsevIsQAkaBhBmkCOk2uxnhBNh9EiABEiABEiAB1wQokFwzskSOlOR43c7snfl63NPIE80qSXa\/2iog7ml55icBEiABEiCBEBBgkyRAAiTgNwIUSH5DGR4VpVSJ0Tuy40ixHmeEBEiABEiABEiABEggFATYZrAJUCAFm7jJ2zvXKJAOF5ncWs\/Ni01tLpI2VnZdcIfnhVmCBEiABEiABEiABEgg7AlQIAVxiNlU6AnEt+0nsfVKRFLoTaEFJEACJEACJEACJEACJiRAgWTCQaFJgSEQnZBUShzBQUPBxvTANBhZtbK3JEACJEACJEACJBAWBCiQwmIY2Ql3CMQkJOvZYo\/vl\/xpvQTnHxVmrdPTGSEBEiCB0gSYQgIkQAIkEEkEKJAiabQd9LVVUnmZl1ZNMnrXUAH3DrKFRVJsvWZ6PyCQinj+kc6DERIgARIgAfMSuPzyy+Wtt96Sr776SjIyMhyG5cuXy2effSZjx46VNm3auNWZ+vXrywsvvCBLly6VFStWqHrXrFkjX3\/9tbzzzjsO6xk2bJh8+umngnywZdmyZdKzZ0+H7SHvggULZPXq1apu2Dhz5kzp3r27w\/yuEp966ilZtGiRNG7c2FVWPicBnwhQIPmEz\/qFn2heSSCK4JwBwfo9ct6DaMMMUsWDm51n5BMSIAESIAESMBEBHB46cOBAJZIKCgqUZfv27ZP+\/ftLq1at5LrrrpOXXnpJcnJy1P2rr74qc+bMkbZt26q8jj4efPBBmTZtmtx4442SlZUlgwcPVmXvuOMOJZYuvvhiJbbGjRsniYmJehWjRo0S5IE4QmKFChXk9ttvdyhakBfPcPhrUVGRsunuu+8WiCaU9SQkJSXJlVdeKeecc45ce+21nhQNed5y5cpJampqyO2IRAO87XO0twVZLjQEasWn6A3vPZatx72NpFQ+69bbWMeOw0WSkXvCmBRW8diC\/WHVH3aGBEiABEgg\/Ans3r1bCgsLVUdPnDghhw4dUvH8\/Hw169OvXz+BOEI6ZodGjBghvXr1UnmMH0OGDJF77rlHKlasKKtWrZJHH31UfvrpJ5Vl+\/btAmHz5ptvyvHjx6V169ZiL5LQHvKpAiUfdevWlaFDh9oIqZJk9f\/Jkydl06ZNqq7\/\/ve\/Ks2bD4iimjVrSkxMjBJ+RtHmTX3BLNOlSxf55z\/\/KQ0aNAhms2zLBwIUSD7AC4eirebuk5Qpu1VA3BjM0z9aQgIkQAIkQAIk4A6B9PR0mT59umCmKT4+Xu677z7p0KGDXvT666+Xzp07S2xsrGzbtk1eeeUVgYjRM5yJfPLJJ2pJ36lTp+Siiy5Ss1VnHukXiDUINSQgz7PPPiuYLcG9fUA9jtqxz+foHnW2a9dOMFuF5xBkN910E6KmD1gOCDGKsTC9sTRQJ0CBpKNgBLNGWggHGjjzKOHOSaKFWLr3DodhDb8+sEckQAIk4GcCWF63fPlygSipVq2aWgIHkYGA\/T+VK1cWLHlbuXKlHDx40Gnr2IuE51FRUQKB0qxZM5u8EGELFy5UYiwqKkot0Rs0aJBNHn\/cYPaoUaNGsm7dOtUWhBLsQX\/8UX+g6sCyQMysQdAFqg3WGxgCFEiB4cpaTUCg8s3DlVtvCCOEmIQk3So4adBvGCEBEiABEggIAVYaOgIQSEePHlUG1KtXTy699FIVUlJSVBqWz7la8vbzzz8L9jqhQNWqVaV589LnCK5du1aWLFmilv5hVurmm2926rQB9XgTsE8KM1WzZ89W+6VQBwQTRBLizgJmb959911ZvHixcu4AJjNmzJC0tLRSRZAXjjDgdAL7q+BYYv78+Wq2Tctcp04dNZOGOuAw44EHHlD7wrQycHbRp08flR17pYYPHy7nnXeeusfywClTpiiHGLfeeqtK44d5CVAgmXdsaJkPBKJLxJBREBmrgjiKp5MGIxLGSYAESMCUBHC4d6SECk1Kf2n3ZVA2b94seXl5qopKlSrJ+eefrwLiSMQzOHVA3FnAkrjc3Fz1GHt\/tC\/7KsHwgf1KEBWYscLsDhwzQHAYsngdbdGihVxwwQVq9ghtwHseZr\/QD4gxZxWj\/Zdffllq164tTz75pNx7772ydetWxeD5559XXvUgclq2bKmWIE6YMEEtD8SSRDi9+M9\/\/iOYAXrmmWdEEz1wiPGPf\/xDiZ6KFSuqdCwzxDLFv\/76SzBbBycUmGn7+++\/lZiCgISNe\/fuFewRw34kLF9EGoN5CVAgmXdsPLIstW4lPX9Wzum\/GOkJERiJSTh75lFh5jrJm9VfDzj\/KAKRsMskQAIkYDkC8W0fkkgK\/hwgeKbTZpAgbhISEpRYKF++vGoGy+P273ftsAgOGVSBkg\/MoJRcSv0PIQWR8Pvvv6tnWFKGpWWJiYnq3pePa665Ru2X+u6771Q13377rWh2QzhBQKkHdh99+\/YV2AHhAicRWCoIYYJ+Yzbqvffek06dOskff\/whmAmC6Jo0aZJs375d0GfMHqEdCL5bbrlF0HeUx7JEiCLkh1t1eP+DmJo3b54cO3ZMINyaNGliZw1vrUaAAslqI0Z73SIQazjzqDBrnRiDWxUwEwmQAAmQAAmECQF8mceXemN3IGo0AWVMt49DJNinObqHAMEZTDt37lSPXTltUJlcfGAGB669MfODc5qQHWLnl19+QVQgwCCg1I3hAy61sQTPkKSiWFKImR2IRIgrJMIV+rnnnitYQvj6668rxxSYWcJyO9SPPNWrVxcsU0TcGIwzcJghAmfUXaVKFWM2xi1IgALJgoNGk10TgIMGLVdh5notGnFXdpgESIAErEwgf+VkiZRQsHGJX4cKMx5xcXF6nYcPH1aOGbAMDomY6dD2I+HeWYB40J5BVGlxR1eIl7ffflsOHDggUVFRPjttgHMGiBOII2PbOCxWa+Oqq64q5T4bM2ZRUVHKRLgyV5GSD3jtO3LkSElMlNtxROB6G3kwc4QZJS107NhROabAsjqkff\/998jOECEEKJAiZKDZTRIgARIIMwLsTgQQyF85pUQgRU7w55Bi9kWbycB+o99++005OIBzBrQDhwpGAYU0RwFL87T0PXv2aFGnVzgsmDlzpvI2hzaw56Z+\/fpO8zt7AA917dq1k\/j4eHVOE\/YfaQGzO9jvg7LYY3TDDTcgqgcIIZwZhQS0jf1IiKPOqKgoZRv2aCFNCxCMmHnS7nmNbAIUSJE9\/uw9CZAACZAACZCA6Qj4bhCcD2DZGGrasmWLYH8Q3GRjiRnS4Orb0TI0PNMCZqE0IYK9O7\/++qv2qMwr9u9gxgWzVRA4sKXMAg4eQhw1bNhQ4EYcszj2ATNVsAmzRa1btxajkEN1H3zwgWC5X40aNZTAwkzRXXfdpfYlwXHCRx99hGx6ACvsWdITDBFwQDAkMRrmBCiQwnyAHXXviWaVJLtfbRXmpVVzlMVSadEJSQIvR+CVTmoAABAASURBVNp5R7hGJ5510mCpztBYEiABEiABEvCRAGZMsEQM4gFL0T7++GNVIzzS\/fDDDwLhgmVll1xyiUp39gEHCHBPjee7du2Sb7\/9FlGXAcvhjE4bYIfLQnYZsLcI+6Y05wx2jwUOEiCAkI6Zn\/bt2yOqhx9\/\/FHGjRsn2CeEvr7wwgty4YUXClxvwzMdbERmPIfTBghG+zrwHPuQXnrpJYGjBtwz+IGABaqgQLLAIPnbxFbJpz3YoN75m47jYumgeTiKrddcP\/copkQ0WbpTNJ4ESIAESIAEvCAA0TNy5Eg1UwJvbB9++KHgTB+tqlmzZqmldrjHrIwzj2tYjnb99dcLxAVmarDvBwIL5bQQFRWlXGNr98arvdMG4zNXcQgznLn0559\/yk8\/\/eQwO+qHRzk4RoCnuZtuusnGlg4dOgg86cHzHNxrw6sdhNGqVats6oPbcCwdjIqKks6dOwtmmbQMEEcQVBBP6L+W7s0VIjE6ml+7vWEXijIcqVBQN1GbOw4XeWuNacqVJYaK8nKVBzvTGEtDSIAESIAESMBLAthvg309KA5vaVgWhjjEDJaxwYsclp7B4xqcDowYMULgfhp5tACRA3fWmFmqVauWDBo0SJ33oz3HFfVBTFx22WWC2RUctDpnzhw8UgHPIcSwhwkH0KpEBx9Gpw0OHjtMQt04RwlL++C9zmGmM4nwSqftqcIZTd26dVNPUAfOPYKDBzh6wDlIQ4YMEYgkcGratKkupsADy\/ggAiG0\/u\/\/\/k8WLFgg06dPF8y8wdsdXIIjHyrHkkFcESB6cEXAHiZcEbQxQhx7nSDiYEvPnj3VDNY777yj9lbhOYM5CVAgmXNcaJWXBA6nj9TPO8LZRwcmdvWyJhaLbALsPQmQAAmYh8Dll18ucEwwcOBAwZd4WIa9NRA6cFyAmZRXX31VIFo2bNggTz\/9tDrEFOnIax+WLVsmjz\/+uMBd9vnnny8zZswQiKu0tDSBkJg7d67AuQI8vkEojB8\/Xq9i2LBhSjhAGEEg3HHHHao8Zmz0TIYI2oLTBpwRZEh2GEUdaA97ipDhxhtvVP3G0jix+w92oA+aYMEVfLCsDmywdA72XXzxxdK9e3fp0aOHPPjggzJhwgQBN4igq6++WtUK8QcX3zg7CmImOTlZ6tevL9i79fzzzwv6gIz33XefqgviFAH1Iu3WW2+Vhx9+WDDThHxY3vh4CV\/si0pPTxfMhMEW9AdpWLKH2T3kZTAnAQokc44LrfKSQPHB0zNG2rlHXlbDYiRAAiQQvgTYM8sRgOjBl3+4m8ayOEcB+2ewzOyJJ54Q+2VkjjqM2Z3+\/fvLPffcI9nZ2cqlNcQAhAS+vD\/33HPStWtXef\/9922Kjxo1Sm655RaBiIEdcKaAOjQRYZP5zA0ECOqC44YzSQ4vqOPOO+8UCBfUDZGGfv\/xxx+l8sMOzMhodiB\/hw4dlMDDfimcZQSBA2cNmAmCIIKTCnjzQ98hVCCYMNuEyiFkMHOF\/qAu8IToAXs8R8ByxZtvvlm5L0cesEIalvGhLNIQYPeEEiEG74GYecLMFfqEOgcPHixIQ30M5iVAgWTesfHIspS68Xr+rJ35epwREiABEiABEiABEnBGYPv27TJmzBjl8U3LA6cHmFnSHBlo6Va4wkbM6EDgQQxNnjxZOWvADNGjjz6qltlh+R2WzWF\/lTtnQaFOhsgiQIEUWePN3pIACZAACZAACZCADQHMqGCJneYVDsv44KwAe3e0GRabAia+wcwQBBL6kJmZ6dRS7NPCMjycmeQ0Ex9ELAGTCqSIHQ92nARIgARIgARIgASCTgBusbFUDl7u4NUNjgZuu+025bAA+2kaNGgQdJt8aRB7hLA\/Cn1KTExUVeGKpX7Tpk0T7E3CXiT1gB8kYEeAAskOSLjfnlslRlIqx4R7N83dP1pHAiRAAiRAAiYkgL1HEydOFHiDw14gOIZYsWKF8nKHc4Rw78hhgpm6gn0\/sHPHjh1yzjnnCLzSff755wJnFrg+8sgjgqV32H+FmTMz2U5bzEOAAsk8YxEUS8a3ryoQScL\/SIAESCAABFglCZBAeBDA0jM4V4BXOJwnBEcDzhwmmK3HmA2DAweII3jFg5OGqVOnyv3336\/OOkKfIAbNZjftMQ8BCiTzjEVQLGmVdPaQWJyBlJF7IijtshESIAESIAESsDgBmm8xAnBjjmV0EERYVufIG57FukRzg0SAAilIoM3YTM\/PDpjRLNpEAiRAAiRAAiRAAiQQVAJszEiAAslIIwLiKVN2ixYwg2TVLsemNpfKacNViE5Mtmo3aDcJkAAJkAAJkAAJkIDJCFAgmWxAfDUnEspHJyRJwl2TJK5JmgoxJfeR0G\/2kQRIgARIgARIgARIIPAEKJACzzgoLaQaDooNSoMmbKQoL1cKs9aZ0DKa5CcCrIYESIAESIAESIAEAk6AAingiIPTQGpyJb2h1T\/s1ePhHoEoOpw+UhDyZj0c7t1l\/0iABMKWADtGAiRAAiRgFgIUSGYZCdqhE6jQJE3tLdL2GNlf49s+pOdFpGBjuiAUl8wg4Z6BBEiABEiABEjARARoCglYjAAFksUGzJm5V19ZQ3+UnZOvx60Wwf6iKmnD1d4ibY+Ro6vV+kV7SYAESIAESMBbApdcconAVfXSpUvVgacZGRmlrsuXL5fPPvtMxowZI40bN\/a2KVOUu\/baawVuuZctW1aqn476jjScd2QK42lEWBDwRCCFRYcjoRNZO60rkCCGPBmjwkzuOfKEF\/OSAAmQAAlYj8Cvv\/4qQ4YMkWeffVb27dunOnD06FF58sknpVWrVnLjjTcqQVFcXCzt27eXd999V3r16qXyBeMjNTVVypUr57emvv32W8HBtF988YVe5+rVq1Vf0V8ttG3bVkaNGiX79+\/X8zFCAv4gQIHkD4qsIyAEjm9MV3uLsL\/IWTiSPtJPbbMaEiABEiABEjA3gUOHDsmJEyeUkadOnZKTJ0+q+MGDB2XGjBlqlikvL0\/i4+PlvvvukyZNmqjngfxo1qyZvPrqq9K8eXO\/NwMRqFVaVFSkRfUr+v\/555\/LypUrJSoqyq8iTW+EkYgkQIEUIcN+bpUY6dm4ogpm7jLON9Lsw74iV0HLyysJkEAZBPiIBEggIghgSdr27dtVXytVqiQXXnihigfqIzExUQYMGCDnnHNOoJpwq9709HT5+uuvBX12qwAzkYALAhRILgCFy+Px7auKFsKlT+wHCZAACZAACZCALYEjR47YJgToDrNUzz\/\/vFx00UUBasG9arHfaseOHWoGDbNn7pViLhIomwAFUtl8LPHUeAaSs\/1HrZLKW6IvsfXOTtEX5eVYwmYaSQIkQAIkQAJmIJCUlCT16tVTpmB52tatW1UcHxA099xzj8yfP18++OAD6dOnj2CPz5o1a+Stt96SRx55RJYsWaKcInz11VfSsmVLSUhIUPucFi1apNLhDGHgwIGoToYPHy5YXhcVFaWW9I0ePVpQrl+\/fuo5PrBHCM4W4EACZVesWCHvv\/++XHHFFXjsc2jUqJHag3T99dfb1IU9Wejjxx9\/LNOnT1d2LV68WO3jAgctM+LOmLzxxhv6kj2IMDBC\/xCWL1+u9nyhf1pduKI+7BWDMw1wRZ\/BGGle7tFCtQwhIECBFALobNIxAXiw054U5eUK3XZrNHglARIgARIggbIJ1K9fXzlxqFu3rtqnBI92P\/30kyrUs2dPmTp1qnJ8cO655wry3H333QKnDlFRUQJhhWVq3333ncqvfWBG5vXXX1d7jCC4tHRchw4dKp9++imikp+fLy+88IJ06tRJpkyZotIgwIYNGyZ79uyRbt26yW233SY\/\/\/yzmnEaOXKkdOjQQeXz9gNipEePHlKzZk2bKgYPHiywDe327t1b7r33XrUMEHu3unfvLv\/6178ESwNdMUlJSZHk5GRBmbffflvt94KYQh8h9C6++GIZMWKEdO3aVbWPOiGi4CTjzTfflNatWwvYxcTEqDr++c9\/6oJLFeCHqQlQIAVjeALcRkpyvN5CtoU92OmdYIQESIAESIAESghk96stnoSSIm7\/70m9Wl63Kz+TUSvnzjWj99njOs4UL\/MCgaDN2nzzzTfy73\/\/WzlK+OWXX+Txxx+XiRMn6uUxa4Qv95mZmSotKipKJkyYoAQTZlomT54sWVlZUlhYqJ7bf8AZApxC2Kc7u8eMCwQSPO6NHTtW4EQiNzdXCSosAaxWrZpAvDgr7yi9Xbt2+iwWZmbQ51tuucVGdECs3HTTTUqwoc+wG3Vt2rRJ0EeIPAibJ554Qs2kuWJSo0YNxQhCEY4o0AfU9+OPP0pBQYGaOdNm7DBz1qhkRguzbf\/5z3+QTRYsWCC\/\/\/67ciCBWbPOnTurdH6YnwAFkvnHyKWFqXUr6Xmyco7qcatFYhKSdZOLD3J5nQ7D5BGaRwIkQAIkEHwCx44dk0mTJqmZG3x5h5MCzORcdtllguVhcAFuXNYFsaCJnJycHPnyyy8FX\/ghHBD3Zw8wi1K9enU1AwOhgmVpCHBTHh9\/+o+6mPlp0KCB281i1kZz743rddddpwQX+qVVcvXVVyvRAmGG2SotHdfvv\/9e9u7dq8TK5Zdfrs6KQtmymGDpHsQc3KyDFepBwBI6zBZBXGIZX506daRFixYSGxurhB\/6qoWmTZuiiFSsWFEaeNBfVYgfISMQHbKW2bBXBGpVTNHL7TmWrcfDIRKdmBQO3WAfSIAEwocAe0ICpiWAL\/ZwToAv\/pixwHI2zGL89ddfUqFCBbX0a9CgQSGx\/7zzzhMsLVu7dq1adodlaVqAiIHAwdK1bdu2eW0fxCDECUQPKoFIqV+\/PqICNhA\/6ubMB2aBtBm0ypUrK4F05pHTi+aAwv6cJdSN2SGISwgnzCJBEMImbamh1l+IRfQXAaLKaWN8YCoCFEimGo7INiYm4axAKsxaF9kw2HsSIAESIAFJmbLbo+AJMk\/rRn5P6kdelHE3tJp7+gBYlPM2bN++XXAuEPbbYDYDX86x3M1xfYFPxRK1QLYCwYMlbb\/99puaodFmzKKiHJ+JhJkz2APxps1k4d5RwGwPhBSegSWurkL58uUFe7xc5eNz8xOgQDL\/GPls4blVYnyuIxgVGM9AKsxcH4wm2QYJkAAJkAAJhBUBeK6DQEKnsKwLS8QQD0WA4wIIDUdtp6amCp47euZuGmZy5syZI1gCh1ke7AtC2apVqwpmdRB3FMAHy\/AcPdPSsIQR9eNem5lCvKwAIQUHGI7yQJBhj5KjZ0wLAAEfq6RA8hGgGYqn1D29nhe22Lv5hjjydOMn6glFoIvvUFBnmyRAAiRAAuFEAHt7MEOCPhUVFQkC4sEMmNnBMjccIItZLPu2MasFr25wvGD\/zNt7LKHLzj699QDC6x\/\/+EepquCVDokQU5h1QtxZ2LVrl+zevVs9hr2OvO7deeedMn78eLWXC+1HRUVdme7\/AAAQAElEQVQp9+hoXxU0fPzf\/\/2fcpmuzXIZHjFqQgIUSB4MCl7qBx98ULA5D\/7tsWFw5syZYu8Hv6wqO3bsKLNnzxb40IcXFlxnzJjhUR1l1W\/\/zP78o4zcE\/ZZTHl\/xsW3KW2jUSRAAiRAAiRgRgL4ngJBgpkj2Ic9PvbOCpDuLEBMQdhgJqR27dp6NuzFQZqeYBeJioqy8Sb3ww8\/CDzXYS9Ur169BM4OtCKYjcE+HXizw\/cpLd0fV5zjpLWLvU7GOnGmE2aV4Klv9erVAgFkfO4ojv1dmJVC2ccee8zmuxrE0e23367OkoIHQHirQx2YMcMBupgxwj0C9obBqQSWP2qzUkhnMC8BCiQPxgabHeESEr9AcPAXzhfAFDG8sjj6y4J91XB5iV8KWJ+KHyRsLsQVP7DwpY8Ni\/Zl\/HkPcTR4+SF\/Vsm6SIAEvCbAgiRAAiTgGQHMyGjiJzo62uYMoCZNmijvdTh\/B7XCecD777+vzu\/BPcRTVFQUosrbGr70qxvDh1HY9O\/fX4YPHy7Tpk1T5\/hAWCArxA6eoTwEGJaiQQzgnKOBAwcK\/uiLM4jghhtlsMQP332wFA5\/VP7www+Vpzl42nMlFmrVqoUmVcCsDPqgbpx8rFq1SnCeE5bQtWzZUvC9S8uKP3Bj+RuW48FGpKO+qCjnTD766COBSIJoxFlRY8aMEXinQ98gepYtW6a8AaIu9G3nzp3KS16bNm1k4cKFAv6LFy+Wu+66S1AGceRlMD+BaPObaA4L8bJj9gdrVocMGSKjRo0SHEYGDyaVKlVSP4T4QXNmLbyrwD9\/VFSU8sWPXy7jxo0TXOEmEtPhOEgNv3Cc1eFN+vxNx\/QNrr3SD8iOw0XeVBPQMtEJSVKhSVpA22DlJEACJEACQSTApvxK4JJLLpFXXnlFffeA4EDlECWYqcBqFIT33ntP4FIa31PguGDAgAGC83qQt127doLnmL3BPf5Qi6Vh+KMv7rWAP\/zii\/6BAwcEguTaa68ViI2XX35ZzQjhPCGIhrlz5wqWlK1cuVK1AaGDc37gYhwuxzGj8uabbyqBALGE7zhoGyJj3bp1aqkZ6tLatb\/iD8ZYbQO7tWdgMGvWLIG3Pi3N0RXnP0GQbdmyRX3HgjBDuPLKKwXftzAThFkm1O2KCfo1cuRIdZ4R+gtRCu6Iv\/7662p5nWYD+vPcc88JGGqzTvDmB5EIm\/CdT8vLq\/kJUCC5OUZwzwjxgr+u4IdAK4a\/VOAvBth4hx82Ld3+inWv8IaCX1xYnmd8jmV2f\/\/9t+AXx4UXXmh8FPZxOGaoPmCxVEkbHvZ9ZQdJgARIgARIwBsCmPXAapUuXboIvo84C\/gegj+2YqYDM0haW9gS0LdvX8FzlMUV99pMipYPV4gJtIOZKCzXw94ZCLCePXvKvffeKxApEBjIC6EwdOhQVa9WJ2zFM4gLzBbBHix3Q7v4QzP+yAxve8jjLMCFNpawoX2UQ0AdvXv3ViLRWTktHcIN\/UN5zCIhYDkc7IFdyOcuE7juhrjBIa+aHZgtw\/c\/1GMM+H6IWTSsKkJeMIHYmzdvnjEb4xYgQIHk5iBBAOGvAJmZmTYl8MsBmwLj4uIEeWweGm7Wr18vN998s5qmxpS04VFER2PrNSvV\/6K83FJpTCABEiABEiABEiABEiCBYBCIYIHkPl7MHMFlJAQSpo3tS8LLCaaPMUtk\/8yd+0svvVSwrvjQoUOCqWh3yoRLHswgaX0pzFwnxzemS96sh7UkXkmABEiABEiABEiABEggqAQokNzAjdOR4YkFm\/S0qVljMQgn3GNdKq6eBKzxvfXWWwWHi2FaOpJnl\/JXTpEj6SOFHuw8eYOc5GUyCZAACZAACZAACZCAVwQokLzC5p9CEFTYYHnBBRcIZqYmTZrkUcXYs4RgLBRXoYJAzFklRCcm6+bHFuwPiu1wVQpBimWRVuFkNjvJ0D8\/Z+ToHUfjzwMZWp+h\/o8AIyRAAjoB4+85Z3F8B7QPegWM+ESAAsknfN4XxswRDknDpsOcnBzllcW4odKdmhcuXKjcSD7Yt6+eHS4xW7RoIVYJMQlJuu2Xn5ccFLubNWsmEKW4WoWT2ewEOzL0\/eeMHMnQDD\/boX4P4Z1M+J9GgFcSUATwc+Hq9wM8IWvfBbUrPOapCvjhEwEKJDfw7d+\/X+CyMSrK9iA0rSj+gok43GLi6ipA7cMjCnz0wwPeiBEjBJ5PXJWzfw4Xngj\/\/WWZ\/uinn7cLlupZJeiGl0SCZfMff\/whW7dulf\/973+WYhUsPu60Q4a\/+uXdIUffOZKh9RliBUXJPwH8nwRIwEBg8+bNLv+dgdtyfA80hqlTpxpqMVvUOvZQILkxVvBUBwcKEEI4ZMy+SO3ataWoqEjgqtv+mf1948aNBb7zL774YsHL\/9RTT6kfAPt87tzDMx5CbNQhPftfW\/arswlgs9nDYYnX7UYkmPYePXrUMpyCycWTtsgwzy\/vEDn6zpEMrc0Q\/77i3wAGEiCBswTc+fcY7tLxPdAYNmzYcLYSxrwmQIHkJjocOAaB1LBhQ5sS8HCXkpIix48fF+SxeWh3A3GEZXUNGjSQ3377TXB2AF5uu2ymv\/WHgdEJSYLzj\/xRF+sgARIgARIgARIgARIgAX8RoEByk2RGRob6azFOYobQ0YrBAx3ce0Mc4dAxLd3+Wq5cOXn00UcFp1f\/\/vvvgoPSPN1zZF+ndp9S9+xMTHZOvpZs6mtckzQb+3j2kQ0O3oSOAFsmARIgARIgARKIcAIUSG6+AKtWrZKvv\/5aatSoIRMmTFBOFcaPHy\/333+\/5Ofny5w5c0RzAY4ZogULFsjy5culV69eqoVOnToJltVhKR6812EP0rRp08QY3nrrLbnwwgtVfn99PNGskr+q8ms9xvOPePaRX9GyMhIgARJwQoDJJEACJEAC7hCgQHKH0pk8b775psydO1fdde7cWXlcw+bSV155RZYtW6bSnX2cd955UqlSJcGBsvXr11diCYLJGCCOsGTPWR3epmf3qy0IZhVLBRvTefaRt4PLciRAAiRAAiRAAiJkQAJ+JECB5AFMzBC988470qVLF2ndurW0a9dO7r77blm5cqVNLdu2bZPu3btL+\/btZd68eeoZZodatWolZQXMMn3\/\/fcqfzh\/YP9RbL3mqotYWleYtU7F+UECJEACJEACJEACJEACoSZgNoEUah6WbD\/VsAcpa2e+JftAo0mABEiABEiABEiABEjADAQokMwwCj7YYEVxFJOQrPe4+GCOHg\/\/CHtIAiRAAiRAAr4RSEtLk48++khWrFghcCCF\/c7Yz9y2bVu94ssvv1ywcuWrr75SeZDPPqDcZ599JmPHjpU2bdroZV1F6tSpo8pgr\/Xq1av1+l966SVXRdVz5NNsWbNmjXz66aeqPmxXwH5urV\/IgzjSHnnkEVVW+8A9tjxo7aMvOBOoQ4cOWhYBjxkzZqj94KgLbX3zzTdqz\/jSpUtVm1rmYcOGqXS0h7wIiKNtPNPyaVc46xozZoygHs0GlIEd8+fPV464Ro0aJYg3aNBAK8arhQhQIFlosGgqCZAACZiWAA0jARIIOIH77rtPcH4iREpBQYGcOHFCypcvr\/Y1P\/vss9KhQwdlA87CGThwoBJJyIfEffv2Sf\/+\/dVS\/+uuu04gVHJyctT9q6++qgQCRAXylhV27dqljim5\/fbbbc5xbN68uTRr1qysogJh0bRpUz0PbILdOPYE1z59+sjPP\/+sP4fgQBq2N+iJJRHc33XXXYLzf8AAQghOs7T94OCA+lJTU9XecfTrjjvuUGIJnoerVaumnG6VVKX+h5hBOz\/88IO6xwfiSMMz3CPAIzG4Tp48WW2zwFlFONsSPLGF4vnnn1cej2+77TbBtono6GgUY7AgAY6cBQfNaHJKcrx+m83ldToLRkiABEiABEjAXwTMUA\/EBzzjQhT07t1bfQG\/8cYb5fPPP5fCwkLBl358oceXeM3e3bt3q2e4h5A4dOgQosr7Ljzz9uvXTyCOkA4HUiNGjNC976qMZXxgX\/bBgwf1HImJiQKhoCc4iHTr1s1GmMDu48eP2+Q8duyYfr9nzx49bh9B+5s2bVIi8ddff7V5jHbAY+PGjfLuu+8qL8Pbt28XiJ1JkyYJRGNsbKwYWaECCDZcETIzM3GxCYMGDZKePXsqUYr959iH\/sknnyieyAiPxxChSEPfqlatKrVq1cIjBosRoECy2IDRXBIgARIgARIggcgjAPGBY0X+9a9\/iXaOIu5ffvllwfmKIJKSkiKXXnopom6H9PR0mT59uhIN8fHxglkqzMC4WwGEF44wiYqKUsvaMEvkqGxSUpLgLEmIKggUR3n8kYZZI3BAXbANV2PAsjmIG3gW1vJpz48ePapFS10hjG6++WaBsIIzLiwJhEizz4i0CRMmiIWcbtl3gfclBKJLAv8nARIgARIgARIgARIwMQF8mf\/iiy8EX86NZuIL+Z9\/\/qmS8OUdS+7UjQcfEA3Lly+XU6dOqZkoLJ+zn11xVh0Ez86dO9Xj6tWry7XXXqvi9h9Ix2zK2rVr9Vkt+zz+vr\/kkkvEkdj77rvvVF8x6+VOmziCBeKoQoUKSkhi7xH67awsxgQzexCwzvIw3dwEKJDMPT4urUutW+l0npLPrJzSf\/lIqRJT8iQ0\/0cnJElsavNSIToxKTQGsVUSIAESIAESsCiBxx57TKZOnerQeiznwgPsidFml3DvSYBA0mZQ6tWr5\/ZMFMQAZktgA856hCDBbJGxbYgtHI0C+7BE0PjM33GcT5mdna2qhbDBvqAHH3zQZjndt99+Kz169BDs1VIZXXxgD9O5556rcmE5ojt9gJOHZ555Rv773\/+qcvywFgEKJGuNl8fW9mxc0eMy\/ihQoUmaVB+wWBLumlQqVEkb7o8mWEcEEWBXSYAEIpPA0wMukkgJfW6p5\/Uga1\/esdQOAsGbijZv3qwcDKAslp+df\/75iLoV4BwB+52QGQ4kMFuEuBZwfiTqgyDZu3evlhywK\/YdabNalStXlgceeEBmzpyplgB60yg80VWsePr7FGaO\/vrrL5fVQDhiPDiL5BKVKTNQIJlyWAJj1I4jxYGp2EGtcZemOUgtnYSDYkunMoUESIAEIoYAO1oGgaElAilSAoRgGSicPsJsTcOGDeXAgQOCJXhOM7p4AGGlzSBhJgizLy6K6I\/RNry+YYkelqFBIBnLX3PNNQJnDEuWLNHLBDIC5w0jRoyQrVu3qqV0UVFRUr9+fRk9erRy7w1mnrSvCVCUQR8hfhBnCF8CFEgWH9sUF4fEzt90TDJyT8j4dUcE8VB0tzBznTgKxzemS\/7KyaEwiW2SAAmQAAmQQFgQwN4Y7O3BGT84k8cfnYLTBSyZ86SuRYsWieYFrlGjRtK+fXsREWnRooVccMEFgn1SP\/30kydV+pQXnu3uvfdeee2110Tzhof9We3atVPuz6+44gqfC+Cg5wAAEABJREFU6mfh8CZAgWSh8a1VMUW3ds+x0+tr9QQnkcHLD0mv9APyxvrS+5OcFPF7cv7KKZI3u3+pcCR9pBTn5fq9PVZIAiRAAiQQHgTGTvxdIiXM+TTT40HDfp9bbrlF4PjgzTff9Li8sQCWxsXFxelJhw8f1uPuRDBr88svv6ismEWCC3LsPerYsaNyiw3HCOphED8w0wOX23B\/DqcJmle7unXrqrOcsHTOHXO05YPIC0cYxtkxpDGEHwG3BFL4dZs9CiSBhDsnSWy95oFsgnWTAAmQAAlEAIFXSwRSJAVPhrRx48by6KOPqpkZZy6nxYP\/sOysSpUqqgScKfz2228q7skHZpGw3A5lsOcI5xG1bt1atmzZIvD8hnRXQSvvKp8nz7EPCGcgTZkyRXmhQ1kcGIvZN8RdhR07dqjzlpAPe5pQFnGG8CVAgWTxsU01LLEzw0GxynOdtcSRxd8Amk8CJEACJBBpBOCeeujQoQKHASNHjlRXXxm0bNlScLAp6oGggYMBxD0JWEKHpXQoA0cPOEgVsy3w6IbZHKS7CvASh30+yIc6cHUWMJuDfVPa0j7kGz58uDocFrNXuDeGWbNmCc5AQlpUVJRgJglxV+HHH3\/Ulw+CEQ7tdVWGz61NgALJ2uMnqckGN987Q7eMTsMYk5CsRQUOGPJXTpbCrHV6GiMkQALBJMC2SIAEwo0AxNG4ceNUt5588slS4uiGG26Q+vXrq+fufmA2Ckvh4JwBMzgff\/yxu0VL5YMjBogWPKhdu7ba\/wO32rh3J8BZBBw6IG9ZMzUQQNjbhNkho0CClzzMXsFRBOqwD9p+JKSjr7i6Cjh7Cm7QsS8LywfhlQ\/j4KqcN2Phqk4+Dw4BCqTgcA5aK+eG8NwjdDK2XjNcVCjYuETyV05RcX6QAAmQAAmQgF8JRGBl+FL+8ssvq56\/8MILpcRR165dBY4JNJfUKqOLj0suuUQwC4XZFIiNDz\/8UHxx9oDZIsxAoVk4e4DzCE\/OZsIMD5a0oTyEG\/ZZIW4fevbsKfDeh\/1XWBKoPYdYwszSQw89JCivpWtXOLRAHCIOnvcQ10JZM1YzZswQ7LPC7Bb2LmGmCuOhlTVe4+Pjlce8QYMGeSxWjfUwHjoCFEihY+\/3lsd3SJCM3jX8Xq8nFeJgWC1\/YeZ6LcorCZAACZAACZCADwTwZXz48OHStGlT9aUbX9i\/+uorMQYsu4NDAW15HGZwIBbQLDy4YXkY4ph9admypXJ5\/fbbbwsOht2+fbuMGDFC5s2bhywuA+xJTU0VXI1CBEvpPvvsM7XXZ\/\/+\/fLtt9\/a1IW2YAsSIeTOOeccRPUAsfP+++8rt+VYnoeDXtEviBLN7vHjx0v\/\/v1l48aNAg564ZJIZmam2i8EwQcPdt27dxcIFoRHHnlEnYWEmSA4bbC3zThjhVmokur0\/7GcETN2EHAof9VVV6mzlYYMGaLGBBlh4+OPPy4fffSRXHjhhYLxWrZsGR55HFggtAQokELL3+fWjW6+6xYX+lyfrxVEJ55dYudrXSxPAiRAAiRAAiRwmsAzzzwjEDXR0dHqCz+cBdgHfHGHt7jLL79cubIeOHCgYEkYaqhRo4ZMmjRJMjIy1D6cV199VTB7hMNbn376aYGnN3z5R96yQp06dZSwwkwTxA6EB4QDhBZEAcpiORqWysGrHWZdkIaZoGnTpsnDDz+svNohrVq1agLnCWPHjhXUizQEiIq+ffvK4sWLldjBkrZ\/\/\/vfgnpxlhHKQfxAOEG4oIwWMIMEkYVlcVFRUTJgwADBLBbC7bffLrt27RL0XVumiHLDhg2TOXPmyGWXXYZbFZo3by5z584VPFMJJR9oC22CK2zBWGDWDlwx6zZ58mS59NJLlXDq3bu3YO9SSTH+b0ECFEgBHbTQVP5Es7P7kkJjwelWi\/JyTkf4SQIkQAIkQAIk4BMBCCR4hGvVqpU4CxAhEBUQPfgS36lTJ6d5cU7RTTfdJE888YSsWrXKbdsgMCASbrnlFtHsQTvwqPfHH3+oeiBQ7rnnHnnxxRfVPT400XPdddfZ2ATxg\/pQL\/JpAcvy4J0PNsJW9Pnqq68WtHX\/\/fdLenq6YLZKy69dsbyvR48ecscddwjEC\/YioSwC6oFwQVktP64QaRCIOCMJ+RDQFvLiGfIYA2auMB6ObIOwwyycI9uMdTBubgIUSOYeH6+sa5Vc3qtyLEQCHhFgZhIgARIgARIgARIIQwIUSGE4qOwSCZAACZCAbwRYmgRIgARIIHIJUCBF7tgHpOcxCUl6vcV5uXqcERIgARIgARIgAVMQoBEkQAIuCFAguQBk5sepdeMFATaeOlQgCIiHKkQbxFERxVGohoHtkgAJkAAJkAAJkECEEvBPtymQ\/MMxJLWkJMfr7RYfOq7HQxWJSaAHu1CxZ7skQAIkQAIkQAIkQAL+IUCB5B+OIanl6itr6u2u\/mGfHjdDpPigbx7szNAH2kACJEACJEACJEACJBB5BCiQLDzm2vI6dCF751FcQhqiE8\/uP+ISu5AOBRs3NwFaRwIkQAIkQAIkYGICFEgmHhxXpqUmnz3vKCsn31V2PicBEiABEiCBABNg9SRAAiRgfQIUSNYfQ\/aABEiABEiABEiABEgg0ARYf8QQoECKmKEOTEejE5Ik4c5JUuO5H6VK2vDANMJaSYAESIAESIAESIAESCBIBCJRIAUJbXCbycg5ISlTdqvQK\/1A0BqPSUiW2HrNS7VXnEcnDaWgMIEESIAESIAESIAESMD0BCiQTD9E5jbQ6JhBsxQOGo5vTNdueQ0qATZGAiRAAiRAAiRAAiTgCwEKJF\/ohbhsSt2z5yBlm8BJA0TRvpevkAMTu0oxD4oN8dvB5kkgDAmwSyRAApKWliYfffSRrFixQjIyMmT58uUybdo0adu2rU7n8ssvl7feeku++uorlQf57APKffbZZzJ27Fhp06aNXtZVJCEhQZ588kn59NNPZc2aNU7rN7aHtnr16mVTNeydMWOGsh95Udc333wjc+bMkaVLlyq7UADtDRs2rMz2UD\/q6tevn8THn\/5uNHDgQJk7d66sXr3aoY3gh3bGjBkjTZo0QVM24ZFHHpGZM2fq9sFG1LVgwQJl25tvvumwXuQrKwwfzu0INqBNekOBZNKBcccso5vvrJ30YucOM+YhARIgARIgATMScMem++67T5566impU6eOFBQUyIkTJ6R8+fJy8cUXy7PPPisdOnRQ1WzYsEEgECCSkA+J+\/btk\/79+0urVq3kuuuuk5deeklycnLU\/auvvqqECUQL8pYV8vLy5PXXX1ftoU7kPXr0qBJNqNsY+vTpIz\/++COy2ATYCXtTU1OViEG7d9xxhxIjycnJUq1aNalRo4Yqg\/ZGjRpl096RI0fk+eefV7bfe++9AmGFuh544AGBULriiiuUQLzrrrtky5Ytqp6ioiKZPn26KnPjjTfK22+\/LYWFhdK+fXuVd8CAASqf9vHOO+\/I3XffLYsWLdKSVF233367DB06VI4fP67Kf\/7554L60G\/0AUxRAFfcIx3PkQ\/taQIOeRjMSyDavKbRsrIIpCaf\/gsJ8lAcgQIDCZAACZAACYQvgWbNmglmYdavXy+9e\/eWTp06qS\/m2hdviAoIknLlyukQdu\/erb7EIwFi6tChQ4hKfn6+fP3114IZF4gjpNevX19GjBih2lCZXHxoAgHZTp06JSdPnkTUJmzfvl3effddgZAyCoNu3bopEbRx40b1HGWRF0Jo0qRJSvzFxsaKsS\/G9tAIRBmumzZtEpT78MMPVbm6desqcYiZJ9Sbm5uLbAKBtH\/\/fhU\/ePCgzJs3T1588UXZs2ePEpm33nqrXHvtteq58QNlwA5pqAt1Io4AmzGThPpw7yzg+ZQpUwTj4SxPmKdbrnsUSJYbMhpMAiRAAiRAAiQQaQQw6wNh869\/\/UvwRR39x\/3LL78sv\/\/+O24lJSVFLr30UhV39yM9PV3NrGCmCSIGs1QdOnRwt7jLfLBt4cKF8ueff6q8mOmBnbjRhAfiWsASu5UrV0qlSpVUf7R0V9fvv\/9eIPSQ79xzz7VZcog0RwEC7a+\/\/lKP0N5ll12m4sYPMDbea\/GYmBg1o4QZLi2trCvGbPPmzWVl4TMTEYg2kS00xUsC2cFaXuelfSxGAiRAAiRAAiTgGwGIii+++EK2bdtmUxFmNDTxgVkXLLmzyeDGDUTJ8uXLBTNBmInCMjLj7I0bVTjMAsF10UUXqT1T2Jdjn+mSSy4RR2Lsu+++U7YkJibaF3F6b5xhgnhB204zGx5gZkm7rVmzphZ1ecU+rJEjR7rMZ8yAZYVYnmdMY9ycBCiQzDkuDq2qWTFFT4+r\/rceD1YkOiFJKqcNV2ce4dwjBJ59FCz6bCfQBFg\/CZAACZiZwGOPPSZTp051aCL2tuABZjMwU4G4pwECSVu2Vq9ePY9nohy1B4cScLAAcac9z8rKkuzsbHWLZXDYS\/Tggw\/aLKf79ttvpUePHoK9VCqjGx9xcXECgYis4IFlfYiXFSDAMKOFPBBKW7duRZSBBIQCKQxegqyco0HpRUxCssQ1SXPaVjHPPnLKhg9IgARIIIQE2HSYE8CSMnQRy9kgQBD3NGD5FwQWymG52fnnn4+o16F+\/frSuXNntb\/HvhLsS9q5c6dKrly5ssC5AjzGwVmDSvTi46abbpLq1aurklg2By916sbJB2bIHn30UcGeJWTZtWuX2peFOAMJUCCFwTsQLCcNjs480vDh7KPCzPXaLa8kQAIkQAIk4DOBhV12S6SESdf85BWvpKQkadiwoRw4cECwBM+rSkoKQVhpM0hYoobZnZJkt\/6HyJkwYYKN22ss28PyOkcVwLHCiBEjBDM2WNYXFRUlEFSjR49WLrTRJ0flHKW1aNFCxo8fLzfffLPA7h07dsgbb7zh0GlE8+bNZciQIcoLH\/ZeQVSh\/Z9++kl54bNfvuioPaZFBgEKpDAY58HNKkl2v9oq9GxcMSg9Or4xXfa9fIUecPZRYda6oLTNRkiABEiABEiABE4TgDCoVauWcnWNc3pOp\/r2ieVmWKbmbi1wu\/34448rF9pwa40Aj3qY0XJWx6+\/\/ipw0f3aa68pT3LIh\/1T7dq1U2634aobaY6CUZDBlTmED2a\/Pv74Y1UnBJh9OYgnLKeDs4urr75aqlSpIl9++aXyBAiX6Nu3b7cvwvsIJmAagRTBY+Bz108dKvC5DlcVVGiSJvFtH3KVjc9JgARIgARIgASCRAAODm655RZZu3atwN20L83ibCXs49HqOHz4sBb16grBgRktR57qtArhYOKTTz4RiCm4K9fyYtkbnBk0aNBAy2pztRdkOMuoa9euMm7cOOXC3CbzmRuIPrQFt+aYbYuOjpbWrVt7dEjumap4iQACFEhhMMhZOfkB70XcpWkSk5AU8HbCtAF2iwRIgARIwAsCty2tLZES+n\/XwiNCjRs3FpLHdoMAABAASURBVOyhgQe7V155xeGSMk8qxLI2zKqgDGZjfvvtN0R9CnDXjaVsOEuorIrgShtnGeGsILgbR14cGIvZMcT9GZYtW6aWImKGDDNRDz30kIClP9tgXdYnQIFk\/TEMeA+iS4RRbL3mejvYb1SwMV2\/Z4QESCCSCbDvJEACwSYA72uYYTl48KDA1TSuvtrQsmVLqVq1qqpmy5Yt+tlKKsHLD3jUmz17tkBwYTYIYfjw4QInDXCSYF\/trFmzBKIK6VFRUboDBdz7M7zzzjtqvxT2H2mzVWDqzzZYl7UJUCBZe\/yU9YF20hCTkKzawQfEEfcbgQQDCZAACZBA2BMwYQfxRR5LyWAazuKxF0c33HCD1K9fH4\/dDphB6dixo3JygOVn2MvjdmE3MmJ2asyYMcqRwt69ewUe8q699lqHJffs2aOnwxb9xo8RLO3DrBtm31AtnEngjCJHog3PGSKPAAVSGIz5VZWKg9aL4oM5QWuLDZEACZAACZAACZwlAHH08ssvq4QXXnhB7MUR9uHA8UHFiu47bMJhrZiFwkwKlrp9+OGH4i9nD8rQkg\/YBBfccMyA84lwXpGzpW1wOFFSROBR74cffkDUpwDnDI4qALvp06cr739RUVFy1VVXCQ7Itc8bHx9vn+T0Hnu40DenGUz2gOY4J0CB5JyNJZ\/sOFzkd7tj6zXT66SnOh0FIyRAAiRAAiQQNAIQR1ie1rRpUzVDNGPGDPnqq69sApbd7d69W18eV7t2bf3wVHiI05bQYaYES+rGjh0rb7\/9tuBgWDhVGDFihMybN8+tPhnFAOo+77zzSpVr0eK0C+4uXboI9iFhX1NmZqbAGQMEGTzYde\/eXSBCEB555BHBWUjYHwSnDd9++61ep317sFl\/6CSCfmqCCzZq50Vp2ZctWyY4fwn7nvD8zjvvFDi+0J7jCucVeIY4Drwty\/05Zso0cYp82EeFcgzWI0CBZL0xs7H41KEC\/R7iKCP3hH7PCAmQAAmQAAmQQHgQeOaZZwSiBt7XICbgYMA+QFh89913cvnllytX2XBfXaFCBQWgRo0aMmnSJLX3Bvt84M0Ns0cbNmyQp59+WnmSQ7rKXMYHvvgPGzZMsEQNdSIrBAQcRmRkZKj6tStccMPlN2ZVsrOzBYexYgYJe5Jw5lBUVJQMGDBAuSj\/5ptv1AwO8sA2bRkhBAqEHNKM7fXr10\/ef\/99gcCCDfYBNn700UfSqFEj\/VG3bt2UIDKWwXlN8AKI\/UjVqlVTe7o++OADwZJAiKcbb7xRL4+li5h1gj2wS3sAUfXee+8JlulhTJCOw3ZffPFFQTqeI43BOgQokKwzVmVaOnj5IWk1d1+ZefiQBCxHgAaTAAmQAAkoAs+UCCS4pYbgcBbwRXzx4sUC0QNx1KlTJ5uziYzl4BobB6U+8cQTsmrVKtWGOx8QN\/A4d8sttyg32cY6y4pjvxTqhwOIHj16yB133CFYEoi9SFo52NS7d2+B5zvkRYBgwswYPNoZ+4++PfDAA7JgwQJkKxVgI4QQzjwy1n\/33XeXKmNkCxvuv\/9+QRry4l4rj\/bRb9gDu7RGMRP18MMPC2zS8uLauXNnQTqea3l5tQaBaGuYSStdEcDskas87j6PTW0uCXdOkmoDFqvA84\/cJcd8JEACJOA+AeYkARIgARIwJwEKJHOOi9tWFR867nZedzPGt+0nsfWaq3OPYhJ49pG73JiPBEiABEiABEhAEeAHCViaAAWSpYdPZG1OoaRM2S3+3HsUnXjWrbcRD1x8F2auNyYxTgIkQAIkQAIkQAIkQAJhRaBsgRRWXWVnvCGQN6u\/7J\/YVQWef+QNQZYhARIgARIgARIgARKwEgEKJCuNlgNbs3KOOkh1Pyk6IUmwpE7bb4SrcVldUV6OFOflquB+rdbISStJgARIgARIgARIgARIwJ4ABZI9ERPf14pP0a2Lq\/a3HvclEpOQXCKQHtL3GxnFkS\/1siwJkEBICbBxEiABEiABEiABLwlQIHkJzizFsnbm+2RKdKJzJwyFmes4c+QTXRYmARIgARLwPwHWSAIkQAKBJUCBFFi+lqodgkjbb4Rr3uz+lrKfxpIACZAACViXQFJSkjCQQaS\/A2LdH+GwspwCKayG07PORCckSdylaXqhojN7jbjnSEfCCAmQAAmQQIAJ5Obmyvr162XixImycOFCBjKI6HcAPwf4ecDPRYB\/9Fh9GQQokMqA48MjSxSNa5KmzjuyhLE0kgRIgARIICwJ4Ivg6NGjZcCAAZYOzz77rLz77rsyePBgS\/cjlONAhqd\/BvDzEJY\/7BbqFAWShQbLkanZPuxBik44e94RZo8KNqY7aoJpJGBHgLckQAIk4F8CEEn4q7mVw4YNG2TLli1qNszK\/Qil7WS4Xr0\/+Hnw708Ya\/OUAAWSp8TCNH\/+yslSmLUuTHvHbpEACZCAmwSYjQRIgARIIOIJUCBZ\/BXI2un5OUjYe4Szj2LrNbd472k+CZAACZAACZCAuwSYjwRIwD0CFEjucTJtrnlp1WVeWjWP7IMwim97+uwjjwoyMwmQAAmQAAmQAAmQAAmYj4BfLaJA8ivO0FS243CRRw3HJNiefcT9Rx7hY2YSIAESIAESIAESIIEwJkCBFMaD607Xjm9MlwMTu7qTNTh52AoJkAAJkAAJkAAJkAAJhJAABVII4QezaW3fUcKdk6RCk5v1povzcvQ4IyRAAoElwNpJgARIgARIgATMT4ACyfxj5BcLsecIAfuP7JfY+aUBVkICJEACJBDJBNh3EiABEggbAhRIQR7KpKQkGTt2rCxbtkwyMjLUFfdID6QpjkQR9h4VZq4PZLOsmwRIgARIgARIgAQsToDmRxoBCqQgjnhiYqKMHj1a2rRpI5mZmbJo0SLJysqStm3bqnQ8D4Y5h9NHSt6s\/iXh4Yg8+6hatWpy\/fXXS\/Xq1YOBOyzbIEP\/DCs5+s6RDMnQdwL+qYHvou8cydB3hqzBPwQiSiD5B5n3tTzwwANywQUXyJo1a6Rv374yZswYQdrq1atVeu\/evT2u\/NSh46XKVGiSJpXThtuE6MRkPV\/xwVwljIrzcvW0SIpAGEEgRVKf\/d1XMvQPUXL0nSMZkqHvBPxTA99F3zmSoe8MWYN\/CFAg+Yejy1oSEhLksssuk2PHjsnnn38uJ0+eVGVwxT3SW7duLcinHnj5AWcMVUrEUVyJSDKGGDvX3l5Wz2LmJUDLSIAESIAESIAESIAE\/ECAAskPEN2pomHDhlKjRg05evSoWl5nLLNt2zY5dOiQeo58xmeexmMSzs4UOSqr9h1lrXP0iGkkQAIkYFICNIsESIAESIAEgkeAAilIrLGuNjY2Vgkh7D8yNpudna2EU1xcnNStW9f4yKd4YeY6wX4jY+CZRz4hZWESIAESIAES8C8B1kYCJGA6AhRIQRoSzB6VL19eTp06pS+v05rGMjukx8TESHx8vJbs9hUe8LRgLIS06nvXiTEgLdKDxijSOfjSfzJMEl\/4aWXJ0XeOZEiG2s9TqK98F\/kuhvodRPvae2iWq1XtoECy0Mjtyc+WPceyJSbxpBSeSpDC4gQpqlxLLm\/WTBYuXKjCxx+8LbXiT6mAHxQtndfTfMBh4sSJatRxxT3DWTbusgA7QMTV3TLMV5oz+JFjaS6evCtk6Bs\/sCZD3xmSIxniHTBDwM\/z+vXrJTc3Mh1x4d9UfwQKJH9QDFId\/9u\/Rvp\/10KFJz4bIl3u+UVuXVIsd8z9UwYMGKDCyKcGSNzC\/irkze6v0rRn3l9P183y5MB3gO8A3wG+A3wH+A7wHTD3O4AjZYT\/+USAAsknfO4X3rdvn5w4cUKioqKkXLlyNgVxHxUVpZ4jn81DBzf4qwD+OsCwXsiADHx+B0r+0sY6+B7xHeA7wHeA70C4vAP4nujg6yOTPCBAgeQBLF+yHjhwQAoLC6VSpUqSkpJiUxXukY7nyGfzkDckQAIkQAIk4CUBFiMBEiABEvCcAAWS58y8KrF161bB7FDVqlWlQYMGNnXgHul4jnw2D3lDAiRAAiRAAiRAAiRgT4D3JBAwAhRIAUNrW3FeXp78\/PPPUrFiRbnxxhv1ZXZYXnfLLbeo9DVr1gjy2ZbkHQmQAAmQAAmQAAmQAAmQQLAIhF4gBaunJmjn\/ffflz\/\/\/FOuvvpqQfyZZ55R1xYtWqj0uXPnmsBKmkACJEACJEACJEACJEACkUuAAimIY3\/w4EF54YUXZO3atZKamirdunVT15UrV6p0PA+iOWwqCATYBAmQAAmQAAmQAAmQgLUIUCAFebzgWWTw4MHSoUMHadWqlboOHTqU\/uqDPA5sjgRIwGcCrIAESIAESIAEwpIABVJYDis7RQIkQAIkQAIk4D0BliQBEohkAhRIkTz67DsJkAAJkAAJkAAJkEBkEWBvXRKgQHKJiBlIgARIgARIgARIgARIgAQihQAFkkVGOi0tTRYuXCirV69WYeHChYI0i5gfVDP\/9a9\/SUZGhsMwffp0G1uSkpJk7NixsmzZMpUfV9wj3SZjhNzA7fy7774rCxYsKHVel4YA752776InebX6rX51xfDaa6+Vr776Sr1v9u\/p8uXLpVevXjYIIoVh\/fr1Zdy4cfL1118rNvhdt2TJEnnwwQf1YxGMYDzh4kleYxtWjHvCke+i4xHG7\/\/Ro0fr7yJ+LqdNmyZt27Z1WMCT98uTvA4bs1CiJxz5Lro3sHfeeafgfcT3FEclPHm\/PMnrqK1wT6NAssAI9+nTR5588kmpUqWKfPfddyogjjQ8s0AXgmZiQkKC1KhRQ06cOCGbNm2S3377zSZs2bJFtyUxMVHwj2CbNm0kMzNTFi1aJFlZWeofQaTjuZ45QiKDBg2Sf\/zjH057i\/cN7x3eP1fvoid5Szdo3RRXDOHBsnz58rJnzx6bdxPv6u+\/\/67Std5HCsPGjRvL66+\/rhzXwJsnfhbxfsXFxcn9998vI0eO1JCoqydcPMmrKrfwh6cc+S6WHmz83sfvf3xh197FdevWyXnnnSfPPvusdOjQwaaQJ++XJ3ltGrHgTeKZf1\/d5ch30fUg4+e7e\/fugn8\/HOX25P3yJK+jtiIhjQLJ5KPcoEED6dGjhxw\/flz++c9\/KnfgcBWOONJuu+02wV9pTN6NoJmXnJwslStXll27dgm8A\/bt21eMYdSoUbotDzzwgFxwwQWCA3qRZ8yYMYI0\/OUa6b1799bzhnsEsx6PPfaYdO3aVWJjYx1215N30ZO8DhuzYKI7DNGtOnXqSExMjKSnp9u8m3gH+\/fvL5jFRL5IYoifNfzs4mcPcfws4vfckCFDZN++fdKyZUvBFy1PuZiWIToSgAB27nJE83wXQcE2gCF+\/+OPFZi9xLsIz7OzZ89W\/7bgiyV+1lHKk\/fLk7yo2+rBE47oK99FUHAe8M7169fP6fc9T94vT\/I6tyj8n1AgmXyMW7duLeecc46aDdG+OMFkxDH50vXtAAAQAElEQVRDUrNmTf2LA9IjPTRq1Ej9I3bgwAGBSHLGAzNNl112mRw7dkw+\/\/xzOXnypMqKK+6RDvbIpx6E8Qf+KvX222\/L7bffrnhg9s1Rd8HD3XfRk7yO2rJamrsM0a\/69eurGU64\/Me9sxBJDPGlHu\/djz\/+qP8sgsvGjRvljz\/+kEqVKukzm55w8SQv2rN68IQj+sp3ERRsA5jg9z\/OJ8QMkvb0l19+kYKCArVCISUlRSV78n55kldVbuEPmO4JRy0\/fgfw9yJolA49e\/aUK664Qq12ASf7HJ68X57ktW8nku4pkEw+2pjWr1Chgmzfvr2UpX\/99ZfgGfKUehihCZhNw\/Szq1+yDRs2VP\/QHT16VP3CMeLatm2bHDp0SD1HPuOzcIxj5qJp06ZqeeE777yjvrw76ifeM7xv7ryLnuR11JbV0txliL+SVqtWTc0I79y5s8xuRhJDzJx16NBB5s2bVyYTPPSEiyd5UbfVgycc+S46Hu1nnnlGOnXqJB9++KFNhtq1a6vZdYik\/fv3q2eevF+e5FWVW\/zDE458F8sebPwBDkvr8N59++23DjN78n55ktdhYxGS6EQgRUjvLdBN\/MUefy1w9IUfS0\/wDKLAAl0Jion4C2pUVJQSN9jHgCU7K1askI8++kg6d+6s24AvqVhKBiGE\/Uf6g5JIdna2QDhh\/0PdunVLUsL7f\/T1008\/Vcu9du\/e7bSznryLnuR12qCFHrjLED+r2L9VWFgoDz\/8sCxbtkw5JIBjgpdeekmwbl\/rdqQx1PptvIIX\/kiB33Pau+kJF0\/yGtsNt7gjjkjju+h6pOPj4wVfTrG8KSoqSu0BzsvLUwU9eb88yasqD7OPsjjyXSx7sLH0H6uFFi9eLMZZTWMpT94vT\/Ia24i0OAWSyUdcW+JVVFRUylItrWLFiqWeRWoCNnpGR0dLkyZNBKLyk08+Ecy04S9U+IsW1o9LCRw4csBM06lTp2yW9JQ8UvdIxz4R\/FJHWjgH7MuCR5z8\/Pwyu+nJu+hJ3jIbtchDdxniyz6Wi9WqVUsgvvHXQHi0gwDo2LGj8uKmiaRIY+hoqOGxCV+e8LOMJU\/I4wkXT\/Ki7nANjjjyXXQ92vfdd59agg3HNHiX4L1zypQpekGk4Ub7txhxLWhp2r\/PnuTV6giXqyuOfBedjzSW1l111VWCJZ6zZs1ymtGT98uTvE4bjIAH0RHQR3YxQgice+65agkEvuhPmjRJsNwEnrGw\/Al\/eYFwgkDCdHWEIGE3TUZAEz\/wWHf33XcLhNWwYcPk8ccfFyy5u+iiiwR\/qfbF7HApe9ddd6mjDLCk6d\/\/\/rf6g0e49C2Y\/XDGke+i61HAKg3sSf3mm2\/U\/iO44IfTBmyYd12aOTQCrjjyXdRI2V7xxyHMXuI7DZZ8Yo+0bQ7eBZJAdCArZ90kEEwCO3bskHvuuUeuu+46mTNnjk3TU6dOFTzHX06aN29u84w3JBAsAjhLpX379mo5o3GpBByuwKsdlt5deumlDs\/9CZaNZmgHIhEexGALfnbxBw7EGTwjUBZHi76LngHwMTd+JiGI4FHx6aeflr\/\/\/lvw13w4tBH+5zYBVxz5LjpGiT\/yQiRhCfz69esdZ2JqwAhQIAUMrX8q1tY6Y7mXfY1ampbH\/jnvzxIAo5ycHHV+AJbX4S9aWNYUFRVV6sso\/joYFRWlnBUg39laIjsGhiCgvXeIa0FL0\/JoVy1dy4erlqblQRqDqKWgEEhYggcvWRofjZeRkZam5TE+s3IcP3tw7Y1ZDyxRwnIm+z92aH3WGBj7q6VpebSrll5WXuMzq8fd4VhWH7EsOdLfRXs+GzZsEJyHBEc1WMKN5568X57kRd3hGhxxLKuvkfUuniWBIzfatGmj\/l3AHuqzTxzHPHm\/PMnruLXISKVAMvk44y9W2CuDL\/X2piINz5DH\/lmk3mPPEFx9O+o\/WGFvEb54wQ248QuAMT++nOJLKp4jn\/FZJMfxnoEh3jt7DkjDM+TBM1xxj3TcGwPS8Ax5jOmREsc+OW1JibHP+OKF9xMB7yj4gBN4GfMhjjQ8Qx7ch0PAl\/rhw4dLt27dlLv5N954Q3D2jH3f0Gf0HQzsnyENz5AHz3DFPdJxbwxIwzPkMaZbPe4uR\/ST7yIouB\/gRcyYG+8O3iG8S8Z0xJGGZ8iDe1xxj3TcGwPS8Ax5jOnhGrfniH7yXQSFs+Hqq68WfJ\/5xz\/+IV988YVy5pORkSFPPPGE+kNvu3btVBr2D6MU3h28Q3iXcG8MSMMz5EE6rrhHOu6NAWl4hjzG9EiMUyD5edT9Xd3mzZvV2ufzzz+\/VNVIw\/p85Cn1MAIT0tLSBFP5r776quAgNCMCTFMjwNsYzlXZunWrYHaoatWqpfKiLNLxHPmM9URyHO8Z3je8d\/YckIZnyINnuOIe6bg3BqThGfIY0yMhjr1xc+fOVUtB7fuLvXHY0I2ZzqysLAEfcAIv+7xIwzPksX9mxXt8qR89erQ60+3IkSMyfvx49bPsqC\/oM\/oOBvbPkYZnyINnuOIe6bg3BqThGfIY060c94Qj38XSI419rDNmzJAvv\/xSvYv2OfAcadofzvDu4B3Cu4R0Y0AaniEP0nHFPdJxbwxIwzPkMaZbNQ5OnnDku1h6pOGU5uOPPxb7AK+8WP2CmTU8Qz6UxruDdwjvEu6NAWl4hjxIxxX3SMe9MSANz5DHmB6JcQokk4\/6mjVr1LpnfHnq0KGDbu31118v2NC9d+9egScs\/UEER7D8AX\/1gDvM2267zYaE5sUJXzyRD1PMP\/\/8s+AL6Y033qgvs8MXjFtuuUWlgz3y2VQUwTdr1qxx+10EO4wF31vbFwbvXHFxseCvf2CjPUUcP9OYOVq1apVKjiSGWFKHvR34ecMfOPAFVUFw8OEJF0\/yOmjKckmecOS7WHp4sU8Vxz7A\/bnx3wXkxL+\/OC8Oewe9+RmNpHfRU458F\/GG2Qb8sXfcuHHKs6nxiu8vyAnPnkhHPtx78n55khd1R2qgQDL5yOPQUvyVAGfyPP\/884K\/siI89dRTymMb3I7iB8Xk3QiKeeAAHvgCivW7+KsU3LNOnz5dbr31VnX468yZMwVfwmDQ+++\/L3\/++adgKhtxuAHHtUWLFiodf+lHPobTBDx5Fz3Je7r2yPjEO4V3Di6+8Y8bPNjh53nixImCgyixhALvMGhECkO44L\/hhhvUodfw0gTvfti0bR\/gzclTLp4xRO3WDZ5y5LvoeKzxbwQ8SuLfBTDCvwv4WR0xYoT6wxnc8msCyZP3y5O8ji2zVqonHMGZvxd9G19P3i9P8vpmlbVLUyBZYPywSRlfoODq8ZprrhGEw4cPC1xY45kFuhA0E8HjxRdfVMuTMMOGL1X16tUTeICBK+Vly5bptuAvgfBOtHbtWsH6Z+x9wBVT1kjHcz0zI4oA+Lr7LnqSV1UeAR94pwYOHKiWTcBxAA4vxs8zfrbhkADvHUSChiISGOLnE+ve0WfM\/l588cXiKOAQaORB8ISLJ3lRt1WDpxz5LjoeaXiUxB\/W8McK7BXEvwtXXHGF4KDi1157TS3\/NJb05P3yJK+xDb\/Eg1yJJxz5LvpncDx5vzzJ6x\/rrFcLBZJFxmz+\/Plq8zL+qoWAJWTa1KpFuhA0MyFwcPYR3Cm3atVKsDQCX0rxC9veCMw6DR48WOXR8g4dOjRiz1z5\/vvvpVOnTgJhib8y2fPCvSfvoid5UXc4BFcMIYbwF+kuXbpI69at1QwmvoQ5O+ci3BlqvPDzV1Z46623bF4PT7h4ktemEQvdeMOR76LjAd6+fbvAmyIOb8Y7iSWxcO3t7N9cT94vT\/I6ts46qZ5w5Lvo3rjOmzdP8N0G31MclfDk\/fIkr6O2zJQWCFsokAJBlXWSAAmQAAmQAAmQAAmQAAlYkgAFkiWHLRyNZp9IgARIgARIgARIgARIIPQEKJBCPwa0gARIINwJsH8kQAIkQAIkQAKWIUCBZJmhoqEkQAIkQAIkYD4CtIgESIAEwo0ABVK4jSj7QwIkQAIkQAIkQAIk4A8CrCNCCVAgRejAs9skQAIkQAIkQAIkQAIkQAKlCUSGQCrdb6aQAAmQAAmQAAmQAAmQAAmQQCkCFEilkDCBBKxFgNaSAAmQAAmQAAmQAAn4jwAFkv9YsiYSIAESIAH\/EmBtJEACJEACJBB0AhRIQUfOBkmABEiABEiABEiABEiABMxKgALJrCNDu0iABEiABEiABEiABEjAigQsbjMFksUHkOaTAAlYm8AjjzwiH3\/8saxYsUIyMjIchm+\/\/VY+++wzGTNmjDRp0sS0HU5ISJB\/\/etfsmrVKnnvvfcE994ae+utt8oXX3wh33zzjXTv3t3baoJSrlevXrJ8+XI1drD5\/fffl2nTpsn48eOlQYMGQbHBnUY6dOggkyZNUrbNnj1bt3nx4sVy0UUXuVMF85AACZBARBCgQIqIYfa6kyxIAiQQYALvvPOO9OjRQ7788ku9pQMHDsjzzz8vbdu2lQEDBsjnn38uMTEx0r59e3n33Xdl3LhxkpiYqOc3S+T888+Xxo0bK1sbNWokzZs399q0K6+8Ugms+Ph4adOmjdf1BLNgQUGBYDwfeOAB6du3rwwePFi2bdsWTBPKbGvZsmXSv39\/Zdudd94pX331VZn5+ZAESIAEIpUABVKkjjz7TQIRT8BcALZs2SInTpxQRp08eVJyc3MF1\/Xr18trr70mjz32mOzcuVOio6OldevWMnz4cJXXTB8\/\/\/yzrFmzRvbv3y\/LSr6Mw3Zv7UtPT5ft27fL1q1b1eyZt\/UEs1xhYaEcOXIkmE361FZeXp5P5VmYBEiABMKVAAVSuI4s+0UCJGApAkVFRWXau2nTJoFo0ETUJZdcItdee22ZZYL9EIJu1KhRctNNN8no0aPl4MGDXpuwevVq6dOnj2Cm4+uvv\/a6nogtyI6TAAmQAAl4TYACyWt0LEgCJEACwSXw+++\/67NMlStXln\/84x\/BNYCtkQAJkIAJCNAEEgg0AQqkQBNm\/SRAAiTgJwJ79uyRQ4cO6bVVqlRJjzNCAiRAAiRAAiTgHwIhFEj+6QBrIQESIIFIIVCrVi2pWrWq6i6W2mliCZ7tsCcJnu7uu+8+tTRt6dKlyqvajBkzbBw6pKWlCdI0r2u4wvEDnCuoiu0+4CThwQcflE8\/\/VRt6odHvbLK1K9fX4YMGSLwkgYnE8bqUBeeLVq0SObMmaO898FO2ANvfsa85cqVk86dOyunFFiuZ3ymxVEfbJs\/f76yDR7v4A0QtoID6tDy4grHFnfddZdMnz5d9Qee2+CBbuHChYIlfQhz584VLF9Efn8H2APvfGgDDhJgL\/Zqof\/2rGDrPffcIzNnzlS2YozRJ7CDndjrhfj111+vzMT4vfXWW4LliPCGuHz5cnn99ddtxl5l5AcJkAAJkIBLAhRILhExAwl4QYBFSCAABPDFvXz58qrmY8eOH5UNeAAAC+lJREFUScWKFQXiAJ7TICaw7A5f+uFBTZtdSkpKkmbNmgm+nL\/00kvy+OOPyy+\/\/CLdunVTnvJycnLksssuk4kTJ4r2ZVvO\/Icv3XALfdtttwlExI033ij4gr958+ZSZfDl\/aOPPlLiC175IOZiY2PP1HT6MnToUEEdH374oRJxyAcBkJqaKtWrV1eZ0MexY8cK3GVD9ME2TRSqDGc+YNuUKVMEggduxVHvddddJ2+88YbExcXJww8\/rFyNJyYmqhLYzzRhwgTp16+f7mkPcYTDhw9Ldna2ylevXj154YUXBParBD99wA7YBiH4ww8\/KP6w9\/vvvxd4\/8PYgCGa69q1q+oHxvG8884TjDk84kHM\/fXXX\/Ldd9\/J0aNHlY2PPvqoPPTQQ8qzIUQzuG3fvl3A\/uqrr1Ze61AnAwmQAAmQgPsEKJDcZ8WcJEACJBAyAhAEEAH4snzq1Cklct58803p2bOn+sIMw\/CsadOmAiEAcYEvyhAz8ASHL+aYpVi2bJn6Mg0HCpgNwkwO3FNDXME9tSYMIKieeOIJgXjBOU2Y5YATBng+W7t2rcCpBMqgfbS9cuVKGTRokC40kGYMEG6XX3655Ofnyx9\/\/KE\/QvsbNmxQX\/aRuGPHDiXWZs2aJbALafYBYgNiq0GDBur8KIg72IZ8n3zyiZp1gVi4+OKLBU4j0Be0AwECcYh8NWrUkJo1a8rTTz8t9957rxJa69atwyOBqOzQoYOK++sDLDELBOY4HwkcUDe4QexWqFBBObdA2ziX6P7771djjDzoL1y\/QzA9+eSTSsDB9TvGAH2AgMW5U5idw6wRxBQ8HqIsXK2jTsQZSIAESIAE3CNAgeQeJ+YiARIggZAQqFOnjpr1mTBhgtStW1eKi4tl48aNaumZJgqwNwnG4QszlljB2x2EEGZN\/u\/\/\/k8JDYgjfBFHOvJqAa659+3bp27xRVoTBvjSDYGBL9qYPVIZznygDaRDhMA9OZJx3bVrl0Dg4N4+YBYIM144PBZuyo3PIVrgIhtpEG5ZWVkChxRaGtKNAV7ycM4S+oPlZhoHLQ9msiAKcQ\/BiL4gjoAyuMId9wcffCAQZ7hHHXBLjj5BaEIYIt0fAec44Vwn9A3LII11QiBhVghjl5mZqQSk9lyzFbNFS5YsUa7ftWcYN+255uFQewYX8agL95hJxDuEOINbBJiJBEiABIQCiS8BCZAACZiMAGYF3n\/\/fbWHCDMimKXBkil8kcZyOiyr2r59eymr8SVbEzvGhxBHmBmCSMHMEva\/aGHatGlqJgX5jcIAMw+4h\/DBF3s81wIOP7399tsFB9eOGTNGSy7zii\/sOB8JdWJvzTPPPCPYQ4RCWHKH2Q\/E3QnoD2ZcIBwgpuzLQOxA+GCmDe2hL\/Z5HN1rszp4hlknXP0RsMQRwhDLGcHBWCdm5LAcECIKs0O4Nz53FkffnT1DOt4FXPHeVKtWDVEGEiCBiCdAAO4SoEBylxTzkQAJkECQCOzdu1ew3K1Vq1aCgL0knTp1EggLOD+AAPDElHPPPVftY8EMD\/aroC5jgNBBOwhYogVxgNkktGEUDbj3NqBtOE\/AsjmIFuyBwtI97GnypE7MhrjzhR8zSMePH1dVo4yKhOgDs1FRUVGCGTF3BVCITGWzJEACJEACJQQokEogWO1\/2ksCJEACnhCoXbu2yo7ZBDgwUDdlfKSkpAiWZiELxBKu\/gjYB4S9MppIgNB56qmnBLNimiBz1Q6W6Wk2xcTEKGcEjspAZBpnUTCD4yhfMNIwI4h2oqP5Ty44MJAACZCA2Qnwt7XZR4j2kUBkEWBvA0AAG\/xRLUQPZjMQLytAWGB5GvLAqxuu\/goQSXBAAHfc2O8TFRUlcN6AmavExESXzWCZHmahkBFizx1hpQkylAlF0Gb8IFTdsTcUNrJNEiABEiCBswQokM6yYIwESIAEwpLA33\/\/rbzOYc9Py5YtnfYRS\/ngihr7WzRRAU9xSHdUCIIGe4fcEV0XXnihwLsa6oETAXihgzttbS8VlgHecssteFxmgF27d+9WeSCQYJ+6sfvArA1mmCD0sLwP5eyyBO1Wc6JRq1Yt0Zxg2DeOWbGBAwcql+z2z3hPAiRAAiQQXAIUSMHlzdZIgARIIOgE4GhB248DBwdwCGBvBMQO9j0dPHhQIKjgihviArNOd955p+5QQSuHL\/RwFlG\/fn3RDqzVnjm6Yokb9jhhtkh7Dg9ur7zyikBAYF+Su3uFfvrpJ+WZD0vW4BEPtmh1alcILtQJ23BoqpYeiuuvv\/6q7IU9OOPI0SwSZtVatGghmCELhY1skwR8JsAKSCCMCFAghdFgsiskQALWJYDZjkBZj+Vs8ICH+nF20XPPPacObMU9As7nwZlKEDE4gwdp33zzjUAsIQ5Rg71DmrBCfsw0QZwsWLBAz4e8ZQWIsGuuucYmC1yWox0st9MOa7XJ4OAGbrxRDo+wBNDoxhtp6AdsQ3zNmjWCg2QRD1XA2UeYxUL7EJQ4fDctLU0d3ot7HBILr4DaIa\/Ix0ACJEACJBA6AvYCKXSWsGUSIAESiGACONcHMwxAAHfc559\/PqJuBczyICPKV69eHVGbgD0wEDTaGUVwjjBs2DDBzAqEEL6wo9zUqVP1c4FwJtDMmTPVzAcqO++88+S1115TrschlmDfv\/\/9bzGeq4SZHIgg5IctOIwVcWO48cYb5frrr9eTbrjhBsHeHCy1g5DQHiANTiVwD8GDqxbQH8w8\/fbbb8pJw3333afXiWWEcPwA4fTDDz\/IlClTBPm1slpdqBtiUUvHFTNYSEcc9eDqj4AlhWCs7QWDy\/Xnn39eMIOGPVmY1cM5R\/Pnz7dpDg4pkICZMs0u3CNoY444mIM94lrQyqKcfT+1PLySAAmQAAk4JkCB5JgLU8OSADtFAuYj8Mgjjwi+JBtFA874wd4eCBQ8d2b1tddeK9OnT7eZDYL7bHwZxwyFsdymTZvkscceE5x\/hC\/qcDmNL9VwePCf\/\/xH0B5mMIxlYNeLL74oECKY4cEz7E\/CGUM4T2nWrFlIUgEuyCGcsNcICRBI\/fr1EwgxLClDe\/Ash1miAQMGCMQA6h80aJB8\/\/33gvogJC655BJVpn\/\/\/gIOqAviDP00ugRHXpwfNHnyZIFN2M8Esbdo0SLBniikP\/3004J8qAP1wlveRRddhFtVN9ii3xCnsPPmm2+WqKgo9Ryuz8eOHSuwXSX4+AEh+fjjjwuWB8JeVAemWMqIdjCDpwk5jN3bb78tTZs2RTa1vBHLGcEE9qAe3GvCB\/aDPcq1a9dOcLaVVhYMMQ4og7KqQn6QAAmQAAmUSYACqUw8fEgCJEACgSWAL+19+vQRfLHFHh0t4Av63XffrVxgO7MAMy733nuvOrC11Zkzk1AOX6TT09NLFYNYgBDo0qWLag\/OFxAfOXKkQECVKlCSgFmOvn376m107NhRICwgakoe6\/\/PmDFDjPnQD5y1NGrUKCVSMCN1xx13qPOdIOJ69uwp6DfahzjCDBIqw34dlEE66kCAnejnJ598gix6gKD48MMPBXWB33XXXSdoE2LN\/rwo1Au7wQd1InTu3FnGjRsn2KOFNlEW6QjIB0cSYKY36GMEjOGIAQy1NrD36PPPP7epGWMHAQQbkA8B4g3CF\/ZMmDBBsKwQ6QjgA\/Yoh+WUiNuXRRmUtWmINyRAAiRAAg4JUCA5xMJEEiABEiABEiABsxCgHSRAAiQQTAIUSMGkzbZIgARIgARIgARIgARI4CwBxkxIgALJhINCk0iABEiABKxHAI4RmjdvrrzTmd16OKGABz2z20n7SIAESCAUBCiQ\/EWd9ZAACZAACUQkAZzj9Pvvv8uff\/4pcFIBhwlwlDB+\/Hhp0KCBaZh06NBBsI8Jtr311lsC73dwwLF582bRzskyjbE0hARIgARCSIACKYTw2TQJWIUA7SQBEnBOYNmyZQLHGHCOYAyDBw+Wbdu2OS8Y5CdWsTPIWNgcCZAACZQiQIFUCgkTSIAESIAEIogAu0oCJEACJEACNgQokGxw8IYESIAESIAESIAEwoUA+0ECJOANgf8HAAD\/\/zDp4p8AAAAGSURBVAMAoVwP8WsxHJ8AAAAASUVORK5CYII=","height":337,"width":560}}
%---
%[output:3b24f5bc]
%   data: {"dataType":"text","outputData":{"text":"\n================ RESULTS SUMMARY ================\n","truncated":false}}
%---
%[output:512f80f6]
%   data: {"dataType":"text","outputData":{"text":"Method        | Mean Err (cm) | Median (cm) | 95th Percentile (cm)\n","truncated":false}}
%---
%[output:6efe8e70]
%   data: {"dataType":"text","outputData":{"text":"DQN Agent     |        168.08 |       81.80 |               571.83 (Steps: 170.0)\n","truncated":false}}
%---
%[output:2bc03e65]
%   data: {"dataType":"text","outputData":{"text":"Brute Force   |         20.27 |       20.07 |                58.30\n","truncated":false}}
%---
%[output:7eaa459f]
%   data: {"dataType":"text","outputData":{"text":"2-D MUSIC     |          4.94 |        4.99 |                 7.42\n","truncated":false}}
%---
%[output:55a4ea6b]
%   data: {"dataType":"text","outputData":{"text":"2-D ESPRIT    |          0.75 |        0.70 |                 1.62\n","truncated":false}}
%---
