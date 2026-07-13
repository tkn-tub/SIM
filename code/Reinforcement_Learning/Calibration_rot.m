%% Step 1. CALIBRATION 
%[output:106d88ec]

% Grid: 1 position per square metre over L_hall × W_hall
% Margin keeps MU away from walls

n_x_cal  = floor(sqrt(EnvPars.N_cal * L_hall / W_hall));
n_y_cal  = floor(EnvPars.N_cal / n_x_cal);
x_cal    = linspace(EnvPars.MU_margin, L_hall - EnvPars.MU_margin, n_x_cal);
y_cal    = linspace(EnvPars.MU_margin, W_hall - EnvPars.MU_margin, n_y_cal);
[X_cal, Y_cal] = meshgrid(x_cal, y_cal);
X_cal    = X_cal(:);
Y_cal    = Y_cal(:);
N_cal    = numel(X_cal); %[output:3da17f4a]

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

        v0 = EnvPars.U_func(1:EnvPars.N, t_psi);

        Upsilon   = diag(v0');
        % G_Upsilon = EnvPars.G_CST * Upsilon;
        G_Upsilon = EnvPars.G * Upsilon;

        % Commanded atom phase
        xi_cmd = mod(angle(v0), 2*pi);

        % State-dependent Jones coefficients
        t_yx_cmd = EnvPars.F_tyx_real(xi_cmd) + ...
            1i*F_tyx_imag(xi_cmd);

        t_yy_cmd = EnvPars.F_tyy_real(xi_cmd) + ...
            1i*F_tyy_imag(xi_cmd);

        % Force column-vector convention
        t_yx_cmd = t_yx_cmd(:);
        t_yy_cmd = t_yy_cmd(:);
        a_psi_x_y = a_psi_x_y(:);

        %% Dual-polarized emitter
        alpha_deg = angle_rot_deg(k_rot);  % Physical emitter rotation
        delta_deg = 0;                     % Relative phase between x and y feeds

        % Equal total-power allocation between x and y
        w_x = 1/sqrt(2);
        w_y = exp(1i*deg2rad(delta_deg))/sqrt(2);

        % Polarization components in the global x-y coordinates
        E_x = cosd(alpha_deg)*w_x - sind(alpha_deg)*w_y;
        E_y = sind(alpha_deg)*w_x + cosd(alpha_deg)*w_y;

        % Field coupled to the fixed y-polarized receiver
        c_xy = t_yx_cmd.*E_x + t_yy_cmd.*E_y;

        % Relative correction with respect to the nominal x-to-y response
        q_xy = c_xy ./ t_yx_cmd;

        % Received field
        r = sqrt(db2pow(EnvPars.SNR_dB)) * ...
            G_Upsilon * (q_xy .* a_psi_x_y);

        peak_map(pos, EnvPars.t_x(t_psi), EnvPars.t_y(t_psi)) = ...
            max(abs(r).^2);
    end

    % Global maximum and best action for this position
    slice                 = squeeze(peak_map(pos,:,:));
    [global_max_cal(pos), idx] = max(slice(:));
    [best_tx_cal(pos), best_ty_cal(pos)] = ind2sub([EnvPars.T_x, EnvPars.T_y], idx);
end

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
%[output:106d88ec]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\n","truncated":false}}
%---
%[output:3da17f4a]
%   data: {"dataType":"text","outputData":{"text":"Grid: 10 x 10 = 100 calibration positions\n","truncated":false}}
%---
