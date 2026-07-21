%% Evaluation of DQN Localization Precision vs Rician K-factor (floor(K) curve)
% Holds SNR on the multipath-limited plateau and sweeps the mean K-factor.
% sigma_K = 0 => each point is the DETERMINISTIC floor at that K.
% (This is the diagnostic curve, NOT the paper headline: the reportable
%  95th-percentile number needs the real distribution, sigma_K = 8 dB.)
clc;
clearvars;

fprintf('=== Initializing Random Deployment Environment ===\n');
% 1. Add required codebase folders to path
addingPathParentFolderByName('code');

% 2. Load Parameters
Parameters;

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

%% ----------------- Start pool -----------------
delete(gcp('nocreate'));
parpool;

% 3. Run Calibration ONCE (K-independent).
% Calibration builds the clean-LoS reference tables (pos_cal, best_tx_cal,
% global_max_cal). Those are the ground-truth targets and reward normalization
% and must NOT depend on K or SNR, so calibration stays outside the sweep.
Calibration;

% Load the trained DQN agent
agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_ideal.mat');
% agent_path = fullfile('..', 'Dataset', 'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_7_atoms_L_13_Nx_4_Mx_15_Tx_60_Aligned_CST.mat');

if isfile(agent_path)
    load(agent_path, 'agent');
    fprintf('Trained agent successfully loaded.\n');
else
    error('Agent file not found. Ensure the Dataset folder is positioned correctly relative to this script.');
end

%% ----------------- K-factor sweep (fixed, saturating SNR) -----------------
% Saturating SNR: the K=20 floor is essentially reached by 45-50 dB, and for
% lower K the saturation SNR is even lower, so 50 dB sits on the floor for the
% whole sweep while staying below the very-high-SNR regime where a fixed
% threshold/quantization artifact appeared (the 60 dB uptick).
SNR_dB      = 30;

% 9 dB ~ UMi/UMa LoS, 7 dB ~ InF/IOO LoS. 20 dB is your high-K anchor, but it
% may be partly bin-quantization-limited, so the amplitude-law fit should be
% anchored on the deeply multipath-limited points (K = 10 and K = 7).
K_dB_vector = 9;%[20 15 10 9 7];

N_eval = 3000;          % Use 5000+ for the final paper result
N_K    = numel(K_dB_vector);

err_dqn_K   = nan(N_eval, N_K);   % full per-episode error sample (cm) -> CDF + error bars
steps_dqn_K = nan(N_eval, N_K);
theta_K     = nan(N_eval, N_K);   % true off-nadir angle (deg), 0 = nadir
timeout_K   = nan(N_eval, N_K);   % 1 if episode never hit at_peak (timed out)

err_raw_K = nan(N_eval, N_K);
err_ref_K = nan(N_eval, N_K);
rej_K = nan(N_eval, N_K);
err_ab_K     = nan(N_eval, N_K);   % ablation: random start + v3, no DQN
probes_ab_K  = nan(N_eval, N_K);   % configs probed by the ablation
probes_ref_K = nan(N_eval, N_K);   % configs probed by v3 on the DQN side (for the latency table)


EnvPars.SNR_dB = SNR_dB;          % constant across the whole sweep


t_start = tic;

for i_K = 1:N_K

    fprintf('Evaluating K = %.1f dB (%d of %d) at SNR = %.1f dB...\n', ...
        K_dB_vector(i_K), i_K, N_K, SNR_dB);

    % --- set the K-factor the channel generator ACTUALLY consumes ----------
    % generateChannel reads EnvPars.KR (linear), not mu_K_dB. With sigma_K = 0
    % this is deterministic: KR = 10^(mu_K_dB/10).
    EnvPars.mu_K_dB    = K_dB_vector(i_K);
    EnvPars.sigma_K_dB = 0;                          % deterministic floor at this K
    EnvPars.KR         = 10^(EnvPars.mu_K_dB/10);    % <-- the line that makes the swap work

    parfor i = 1:N_eval

        % Reset: picks a random calibration position and builds the aligned obs.
        % The episode channel is (re)generated here from EnvPars.KR / nlosPowerScale.
        [obs, LoggedSignals] = resetFunction_nav_CST_Aligned(EnvPars);

        target_idx  = LoggedSignals.pos_idx;
        pos_MU_true = EnvPars.pos_cal(target_idx, :);

        isDone = false;
        step   = 0;

        while ~isDone
            action_out = getAction(agent, {obs});
            action = action_out{1};
            if iscategorical(action) || iscell(action)
                action = double(action);
            end
            [obs, ~, isDone, LoggedSignals] = ...
                stepFunction_nav_CST_Aligned(action, LoggedSignals, EnvPars);
            step = step + 1;
        end

        steps_dqn_K(i,i_K) = step;

        % Both true and estimated positions pass through the same angle->position
        % map, so the error isolates the angular (psi) estimation error.
        % pos_MU_true = estimatePosFromAngles(LoggedSignals.psi_x,     LoggedSignals.psi_y,     EnvPars, pos_MU_true);
        pos_est_dqn = estimatePosFromAngles(LoggedSignals.psi_x_est, LoggedSignals.psi_y_est, EnvPars, pos_MU_true);
        err_dqn_K(i,i_K) = norm(pos_est_dqn(1:2) - pos_MU_true(1:2))*1e2; % cm

        %Refinement
        pos_raw = estimatePosFromAngles(LoggedSignals.psi_x_est_best, LoggedSignals.psi_y_est_best, EnvPars, pos_MU_true);

        [psi_x_ref, psi_y_ref, dbg] = refineAngleParabolic3(LoggedSignals.h, ...
            LoggedSignals.best_t_x, LoggedSignals.best_t_y, EnvPars, ...
            struct('m',10,'n_avg',12));
        pos_ref = estimatePosFromAngles(psi_x_ref, psi_y_ref, EnvPars, pos_MU_true);

        rej_K(i,i_K) = double(dbg.rejected_x || dbg.rejected_y);   % preallocate rej_K like the others
        probes_ref_K(i,i_K) = dbg.n_configs;

        err_raw_K(i,i_K) = norm(pos_raw(1:2) - pos_MU_true(1:2))*1e2;   % cm
        err_ref_K(i,i_K) = norm(pos_ref(1:2) - pos_MU_true(1:2))*1e2;   % cm

        % --- ABLATION: no-DQN control — random start, v3 only, SAME episode channel ---
        t0x = randi(EnvPars.T_x);   t0y = randi(EnvPars.T_y);
        [psi_x_ab, psi_y_ab, dbg_ab] = refineAngleParabolic3(LoggedSignals.h, t0x, t0y, ...
            EnvPars, struct('m',12,'n_avg',8,'n_recenter',3));   % 3 hops: reach 48 > 30 = max torus distance
        pos_ab = estimatePosFromAngles(psi_x_ab, psi_y_ab, EnvPars, pos_MU_true);
        err_ab_K(i,i_K)    = norm(pos_ab(1:2) - pos_MU_true(1:2))*1e2;  % cm
        probes_ab_K(i,i_K) = dbg_ab.n_configs;

        % --- tail diagnostics (geometry + termination) ---
        dl               = EnvPars.pos_cal(target_idx,:).' - EnvPars.pos_SIM(:);
        theta_K(i,i_K)   = rad2deg(atan2(hypot(dl(1),dl(2)), -dl(3)));  % same convention as generateChannel
        timeout_K(i,i_K) = double(~LoggedSignals.at_peak);

    end
end

fprintf('\nparfor done: %.1f s total.\n', toc(t_start));
delete(gcp('nocreate'));

%% ----------------- Save -----------------
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

fileName = sprintf('Precision_vs_K_%d_to_%d_SNR_%d_Nx_%d_L_%d_Mx_%d_Zeta_%s_CST_LoS.mat', ...
    K_dB_vector(1), K_dB_vector(end), SNR_dB, N_x, L, round(M_x), zetaStr);

savepath = fullfile(datasetDir, fileName);

save(savepath, 'steps_dqn_K','err_dqn_K','theta_K','timeout_K', ...
     'err_raw_K','err_ref_K','rej_K', ...
     'err_ab_K','probes_ab_K','probes_ref_K', ...
     'K_dB_vector','SNR_dB','-v7.3');

fprintf('Results saved to %s\n', savepath);



%% ----------------- Summary: floor(K) with median + tail -----------------
% Median tracks the typical accuracy; p95 exposes the outlier tail. If p95 is
% far above the median, failures are wrong-bin blowups (peak-detection problem),
% not uniform smear (aperture/interpolation problem).
p50_raw = prctile(err_raw_K,50,1);  p95_raw = prctile(err_raw_K,90,1);
p50_ref = prctile(err_ref_K,50,1);  p95_ref = prctile(err_ref_K,90,1);
fprintf('\n K(dB) |  p50 raw->ref (cm)  |  p95 raw->ref (cm)  | reject%%\n');
for i_K = 1:N_K
    fprintf(' %5.1f | %7.1f -> %7.1f | %7.1f -> %7.1f | %5.1f\n', ...
        K_dB_vector(i_K), p50_raw(i_K), p50_ref(i_K), ...
        p95_raw(i_K), p95_ref(i_K), 100*mean(rej_K(:,i_K)));
end

%%
p50_ab = prctile(err_ab_K,50,1);  p95_ab = prctile(err_ab_K,95,1);
fprintf('\n=== Ablation: DQN+v3 vs random-start+v3 (same channels) ===\n');
fprintf(' K(dB) |  p50 DQN / abl (cm) |  p95 DQN / abl (cm) | probes: DQN-total / abl\n');
for i_K = 1:N_K
    fprintf(' %5.1f | %7.1f / %7.1f | %7.1f / %7.1f | %7.1f / %5.1f\n', ...
        K_dB_vector(i_K), p50_ref(i_K), p50_ab(i_K), ...
        p95_ref(i_K), p95_ab(i_K), ...
        mean(steps_dqn_K(:,i_K)) + mean(probes_ref_K(:,i_K)), ...
        mean(probes_ab_K(:,i_K)));
end
%% ---- Tail diagnostics: is the high-K tail oblique-angle or spurious-peak? ----
% For each K, split episodes into the top-5% error tail vs the bulk, and compare
% their off-nadir angle and timeout rate.
%   tail theta0 >> bulk  -> oblique-angle bin quantization  -> parabolic interp
%   tail TO%%   >> bulk   -> agent stops on a false peak     -> policy/termination
fprintf('\n=== Tail (top 5%% error) vs bulk ===\n');
fprintf('   K(dB) | med theta0 tail/bulk (deg) | timeout%% tail/bulk\n');
for i_K = 1:N_K
    e = err_ref_K(:,i_K); th = theta_K(:,i_K); to = timeout_K(:,i_K);
    tail = e >= prctile(e,95);
    fprintf('  %6.1f |   %6.1f / %6.1f          |  %5.1f / %5.1f\n', ...
        K_dB_vector(i_K), ...
        median(th(tail)), median(th(~tail)), ...
        100*mean(to(tail)), 100*mean(to(~tail)));
end

%% ================= helper functions =================
function [psi_x_est, psi_y_est, dbg] = refineAngleParabolic2(h, t_x0, t_y0, EnvPars, opts)
%REFINEANGLEPARABOLIC2  Sub-bin refinement of the SIM-2 steering estimate. (v2)
% v2 changes vs v1, same hardware, same conventions:
%   opts.n_avg : average n_avg packets (fresh noise draws) per probe config.
%                Physically: repeated packets at a frozen config within one
%                coherence interval (protocol budget N_packets_coh = 12).
%                Default 1 = single-shot (v1 behaviour).
%   opts.m     : default raised 3 -> 10 (contrast 0.8% -> 8.6%, see notes).
%   dbg        : now also reports the measured curvature contrast per axis.
% POST-PROCESSING ONLY: never touches the reward. Coarse SIM-1 bin read with
% the validated flip + mod 2*pi convention; only t is refined; |delta| <= m.

    if nargin < 5, opts = struct; end
    if ~isfield(opts,'m'),         opts.m = 10;            end
    if ~isfield(opts,'n_avg'),     opts.n_avg = 1;         end
    if ~isfield(opts,'noiseless'), opts.noiseless = false; end
    m  = opts.m;
    Tx = EnvPars.T_x;  Ty = EnvPars.T_y;
    snr_amp = sqrt(db2pow(EnvPars.SNR_dB));

    function Pvec = powerAt(tx, ty)   % per-port power, averaged over n_avg packets
        tx = mod(tx-1, Tx) + 1;       % toroidal wrap (physical config)
        ty = mod(ty-1, Ty) + 1;
        tpsi = (ty-1)*Tx + tx;
        v0   = EnvPars.U_func(1:EnvPars.N, tpsi);
        rsig = snr_amp * EnvPars.G * diag(v0') * h;
        if opts.noiseless
            Pvec = abs(rsig).^2;
        else
            Pvec = zeros(EnvPars.N,1);
            for p = 1:opts.n_avg
                u = (randn(EnvPars.N,1) + 1i*randn(EnvPars.N,1))/sqrt(2);
                Pvec = Pvec + abs(rsig + u).^2;
            end
            Pvec = Pvec / opts.n_avg;
        end
    end

    % centre: averaged power; coarse bin via the validated convention
    Pvec0 = powerAt(t_x0, t_y0);
    R = flipud(fliplr(reshape(Pvec0, [EnvPars.N_x, EnvPars.N_y])))';  % argmax over power == over abs
    [~, linear_idx] = max(R, [], 'all');
    n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
    n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;
    P_0 = max(Pvec0);

    % 3-point peak-power samples, separable
    P_xm = max(powerAt(t_x0 - m, t_y0));
    P_xp = max(powerAt(t_x0 + m, t_y0));
    P_ym = max(powerAt(t_x0, t_y0 - m));
    P_yp = max(powerAt(t_x0, t_y0 + m));

    [delta_x, rej_x] = parabolicVertex(P_xm, P_0, P_xp, m);
    [delta_y, rej_y] = parabolicVertex(P_ym, P_0, P_yp, m);

    psi_x_est = mod(2*pi * (n_psi_x_max + (t_x0-1 + delta_x)/Tx) / EnvPars.N_x, 2*pi);
    psi_y_est = mod(2*pi * (n_psi_y_max + (t_y0-1 + delta_y)/Ty) / EnvPars.N_y, 2*pi);

    dbg = struct('delta_x',delta_x,'delta_y',delta_y, ...
                 'rejected_x',rej_x,'rejected_y',rej_y, ...
                 'contrast_x',(P_0 - 0.5*(P_xm+P_xp))/max(P_0,eps), ...
                 'contrast_y',(P_0 - 0.5*(P_ym+P_yp))/max(P_0,eps));
end

function [delta, rejected] = parabolicVertex(y_m, y_0, y_p, spacing)
% Edge-guard: centre must be the (weak) max AND parabola concave; vertex in bracket.
    denom = (y_m - 2*y_0 + y_p);
    if (y_0 >= y_m) && (y_0 >= y_p) && (denom < 0)
        delta = 0.5 * spacing * (y_m - y_p) / denom;
        if abs(delta) > spacing, delta = 0; rejected = true;
        else, rejected = false; end
    else
        delta = 0; rejected = true;
    end
end

function addingPathParentFolderByName(targetName)
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
        fprintf('Adding matlab path to: %s\n', currFolder);
    else
        error('Folder named "%s" not found in any parent directory.', targetName);
    end
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
    delta = pos_MU(:) - EnvPars.pos_SIM(:);
    rng   = norm(delta);
    u_x = delta(1) / rng;
    u_y = delta(2) / rng;
    psi_x = mod(2*pi * EnvPars.d_x * u_x / EnvPars.lambda, 2*pi);
    psi_y = mod(2*pi * EnvPars.d_x * u_y / EnvPars.lambda, 2*pi);
end
