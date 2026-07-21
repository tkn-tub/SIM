%% Angle_consistency.m
% PART 1 : psi forward/inverse consistency (estimator vs computePsiFromPos)
% PART 2 : position round-trip acceptance test (estimatePosFromAngles)
%
% All noiseless, LoS-only, SIM steered to each position's calibration-optimal
% bin. Any nonzero error is a CONVENTION / MAPPING bug, not physics.
%
% PART 1 verdict (already established): the CURRENT estimator (flip + mod 2pi)
% is on the identity line -> psi_est == psi_true. Keep the step function as is.
%
% PART 2 tests the ONLY remaining suspect: estimatePosFromAngles (psi -> pos).
% The corrected inverse is defined at the bottom of this file.

clc; clearvars;

fprintf('=== angle + position consistency probe ===\n');
addingPathParentFolderByName('code');
Parameters;

EnvPars.G            = EnvPars.G_CST;
EnvPars.U_func       = EnvPars.U_func_CST;
EnvPars.channelModel = 'rician_los';
EnvPars.SNR_dB       = 50;

Calibration;

wrap = @(a) angle(exp(1i*a));

Nc = EnvPars.N_cal;
psi_true_x = zeros(Nc,1);  psi_true_y = zeros(Nc,1);
psi_curr_x = zeros(Nc,1);  psi_curr_y = zeros(Nc,1);
theta_pos  = zeros(Nc,1);

for pos = 1:Nc
    psi_true_x(pos) = EnvPars.psi_x_cal(pos);
    psi_true_y(pos) = EnvPars.psi_y_cal(pos);

    dl = EnvPars.pos_cal(pos,:).' - EnvPars.pos_SIM(:);
    theta_pos(pos) = atan2(hypot(dl(1),dl(2)), -dl(3));

    % LoS channel exactly as Calibration
    a_psi_x = exp(-1i * psi_true_x(pos) * (0:EnvPars.N_x-1)).';
    a_psi_y = exp(-1i * psi_true_y(pos) * (0:EnvPars.N_y-1)).';
    h = kron(a_psi_y, a_psi_x); h = h(:); h = h / sqrt(mean(abs(h).^2));

    % steer to calibration-optimal bin
    t_x = EnvPars.best_tx_cal(pos);
    t_y = EnvPars.best_ty_cal(pos);
    t_psi = (t_y - 1) * EnvPars.T_x + t_x;

    v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
    r  = EnvPars.G * diag(v0') * h;                 % noiseless

    t_psi_x_max = EnvPars.t_x(t_psi);
    t_psi_y_max = EnvPars.t_y(t_psi);

    % CURRENT estimator -- verbatim from stepFunction_nav_CST_Aligned
    R = flipud(fliplr(reshape(abs(r), [EnvPars.N_x, EnvPars.N_y])))';
    [~, linear_idx] = max(R, [], 'all');
    n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
    n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;
    psi_curr_x(pos) = mod(2*pi * (n_psi_x_max + (t_psi_x_max-1) / EnvPars.T_x) / EnvPars.N_x, 2*pi);
    psi_curr_y(pos) = mod(2*pi * (n_psi_y_max + (t_psi_y_max-1) / EnvPars.T_y) / EnvPars.N_y, 2*pi);
end

%% ---- PART 1: psi consistency (confirmation) ----
rms_ = @(v) sqrt(mean(v.^2));
fprintf('\n--- PART 1: psi (current estimator) ---\n');
fprintf('  psi_x : RMS_identity = %.3e rad | RMS_flip = %.3e rad\n', ...
    rms_(wrap(psi_curr_x - psi_true_x)), rms_(wrap(psi_curr_x + psi_true_x)));
fprintf('  psi_y : RMS_identity = %.3e rad | RMS_flip = %.3e rad\n', ...
    rms_(wrap(psi_curr_y - psi_true_y)), rms_(wrap(psi_curr_y + psi_true_y)));
fprintf('  (identity ~ 0 confirms the estimator convention is correct.)\n');


res_x = wrap(psi_curr_x - psi_true_x);   res_y = wrap(psi_curr_y - psi_true_y);
fprintf('bias x = %+.4f rad (%+.2f steps), scatter = %.4f rad (%.2f steps)\n', ...
    mean(res_x), mean(res_x)/0.0262, std(res_x), std(res_x)/0.0262);
fprintf('bias y = %+.4f rad (%+.2f steps), scatter = %.4f rad (%.2f steps)\n', ...
    mean(res_y), mean(res_y)/0.0262, std(res_y), std(res_y)/0.0262);

%% ==================== PART 2: POSITION ROUND-TRIP ====================
% Ground truth = the actual grid coordinates.
posX_true = EnvPars.pos_cal(:,1);
posY_true = EnvPars.pos_cal(:,2);

% (a) pure inverse: feed TRUE psi -> must reproduce pos_cal algebraically (~0).
% (b) full chain  : feed ESTIMATOR psi -> reproduces pos_cal up to bin quantization.
posX_fromTrue = zeros(Nc,1);  posY_fromTrue = zeros(Nc,1);
posX_est      = zeros(Nc,1);  posY_est      = zeros(Nc,1);
clamped       = false(Nc,1);

for pos = 1:Nc
    pref = EnvPars.pos_cal(pos,:);   % supplies the true height (drop) and output z

    p_a = estimatePosFromAngles(psi_true_x(pos), psi_true_y(pos), EnvPars, pref);
    posX_fromTrue(pos) = p_a(1);  posY_fromTrue(pos) = p_a(2);

    [p_b, clamped(pos)] = estimatePosFromAngles(psi_curr_x(pos), psi_curr_y(pos), EnvPars, pref);
    posX_est(pos) = p_b(1);  posY_est(pos) = p_b(2);
end

errX_inv   = posX_fromTrue - posX_true;      % inverse-only error (algebraic)
errY_inv   = posY_fromTrue - posY_true;
errX_chain = posX_est      - posX_true;      % full noiseless chain (quantization)
errY_chain = posY_est      - posY_true;
err_chain  = hypot(errX_chain, errY_chain);

fprintf('\n--- PART 2: position round-trip ---\n');
fprintf('  (a) inverse only  (from TRUE psi):   max|dx| = %.3e m  max|dy| = %.3e m\n', ...
    max(abs(errX_inv)), max(abs(errY_inv)));
fprintf('      -> must be ~1e-9 m. If not, the inverse is not a true inverse of the forward map.\n');
fprintf('  (b) full chain   (from EST  psi):    median = %.1f cm | p95 = %.1f cm | max = %.1f cm\n', ...
    median(err_chain)*1e2, prctile(err_chain,95)*1e2, max(err_chain)*1e2);
fprintf('      -> this is the noiseless bin-quantization floor of the position estimate.\n');
fprintf('  clamped (out-of-FoV) positions: %d of %d\n', sum(clamped), Nc);
if any(clamped)
    fprintf('      clamped theta0 range: %.1f - %.1f deg\n', ...
        rad2deg(min(theta_pos(clamped))), rad2deg(max(theta_pos(clamped))));
end

%% ---------------------------- Plots ----------------------------
doPlot = true;
if doPlot
    figure('Name','position round-trip','Color','w');

    subplot(1,2,1); hold on; grid on; axis equal;
    scatter(posX_true, posX_est, 18, rad2deg(theta_pos), 'filled');
    lim = [min(posX_true)-0.5, max(posX_true)+0.5];
    plot(lim, lim, 'k-'); xlim(lim); ylim(lim);
    xlabel('pos_X true [m]'); ylabel('pos_X estimated [m]');
    title('X: on-diagonal = correct'); c=colorbar; c.Label.String='\theta_0 [deg]';

    subplot(1,2,2); hold on; grid on; axis equal;
    scatter(posY_true, posY_est, 18, rad2deg(theta_pos), 'filled');
    lim = [min(posY_true)-0.5, max(posY_true)+0.5];
    plot(lim, lim, 'k-'); xlim(lim); ylim(lim);
    xlabel('pos_Y true [m]'); ylabel('pos_Y estimated [m]');
    title('Y: color = off-nadir angle'); colorbar;
end

%% =============================== helpers ===============================
function [pos_MU_est, was_clamped] = estimatePosFromAngles(psi_x_est, psi_y_est, EnvPars, pos_MU_true)
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

function addingPathParentFolderByName(targetName)
    currFolder = pwd; found = false;
    while true
        [parentFolder, currentName] = fileparts(currFolder);
        if strcmpi(currentName, targetName), found = true; break; end
        if isempty(parentFolder) || strcmp(currFolder, parentFolder), break; end
        currFolder = parentFolder;
    end
    if found, addpath(genpath(currFolder));
        fprintf('Adding matlab path to: %s\n', currFolder);
    else, error('Folder named "%s" not found in any parent directory.', targetName);
    end
end

function [psi_x, psi_y] = computePsiFromPos(pos_MU, EnvPars) %#ok<DEFNU>
    delta = pos_MU(:) - EnvPars.pos_SIM(:);
    rng   = norm(delta);
    u_x = delta(1) / rng;  u_y = delta(2) / rng;
    psi_x = mod(2*pi * EnvPars.d_x * u_x / EnvPars.lambda, 2*pi);
    psi_y = mod(2*pi * EnvPars.d_x * u_y / EnvPars.lambda, 2*pi);
end