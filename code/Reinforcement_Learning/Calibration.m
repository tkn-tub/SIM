%% Step 1. CALIBRATION ───────────────────────────────────────────────────────
fprintf('=== Calibration phase ===\n');

% Grid: 1 position per square metre over L_hall × W_hall
% Margin keeps MU away from walls

n_x_cal  = floor(sqrt(EnvPars.N_cal * L_hall / W_hall));
n_y_cal  = floor(EnvPars.N_cal / n_x_cal);
x_cal    = linspace(EnvPars.MU_margin, L_hall - EnvPars.MU_margin, n_x_cal);
y_cal    = linspace(EnvPars.MU_margin, W_hall - EnvPars.MU_margin, n_y_cal);
[X_cal, Y_cal] = meshgrid(x_cal, y_cal);
X_cal    = X_cal(:);
Y_cal    = Y_cal(:);
N_cal    = numel(X_cal);
fprintf('Grid: %d x %d = %d calibration positions\n', n_x_cal, n_y_cal, N_cal);

% Preallocate calibration tables
peak_map       = zeros(N_cal, EnvPars.T_x, EnvPars.T_y);
psi_x_cal      = zeros(N_cal, 1);
psi_y_cal      = zeros(N_cal, 1);
pos_cal        = zeros(N_cal, 3);
global_max_cal = zeros(N_cal, 1);
best_tx_cal    = zeros(N_cal, 1);
best_ty_cal    = zeros(N_cal, 1);

for pos = 1:N_cal
    pos_MU_k       = [X_cal(pos); Y_cal(pos); EnvPars.h_MU];
    pos_cal(pos,:) = pos_MU_k';

    [psi_x_k, psi_y_k] = computePsiFromPos(pos_MU_k, EnvPars);
    psi_x_cal(pos) = psi_x_k;
    psi_y_cal(pos) = psi_y_k;

    a_psi_x   = exp(1i * psi_x_k * ((1:EnvPars.N_x)-1))';
    a_psi_y   = exp(1i * psi_y_k * ((1:EnvPars.N_y)-1))';
    a_psi_x_y = kron(a_psi_y, a_psi_x);

    for t_psi = 1:EnvPars.T
        v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
        r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;
        peak_map(pos, EnvPars.t_x(t_psi), EnvPars.t_y(t_psi)) = max(abs(r).^2);
    end

    % Global maximum and best action for this position
    slice                 = squeeze(peak_map(pos,:,:));
    [global_max_cal(pos), idx] = max(slice(:));
    [best_tx_cal(pos), best_ty_cal(pos)] = ind2sub([EnvPars.T_x, EnvPars.T_y], idx);
end
fprintf('Calibration complete.\n\n');

% Store calibration in EnvPars — available to step/reset functions
EnvPars.N_cal          = N_cal;
EnvPars.peak_map       = peak_map;
EnvPars.psi_x_cal      = psi_x_cal;
EnvPars.psi_y_cal      = psi_y_cal;
EnvPars.pos_cal        = pos_cal;
EnvPars.global_max_cal = global_max_cal;
EnvPars.best_tx_cal    = best_tx_cal;
EnvPars.best_ty_cal    = best_ty_cal;

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

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
