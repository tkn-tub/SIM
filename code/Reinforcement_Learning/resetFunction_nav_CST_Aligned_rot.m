function [observation, LoggedSignals] = resetFunction_nav_CST_Aligned(EnvPars)
% Matching reset for the aligned-observation variant.

%% Pick random calibration position
pos_idx = randi(EnvPars.N_cal);

LoggedSignals.pos_idx    = pos_idx;
LoggedSignals.psi_x      = EnvPars.psi_x_cal(pos_idx);
LoggedSignals.psi_y      = EnvPars.psi_y_cal(pos_idx);
LoggedSignals.global_max = EnvPars.global_max_cal(pos_idx);
LoggedSignals.stepCount  = 0;

%% Random starting (t_x, t_y)
LoggedSignals.t_x = randi(EnvPars.T_x);
LoggedSignals.t_y = randi(EnvPars.T_y);

%% Initial observation
t_psi = (LoggedSignals.t_y - 1) * EnvPars.T_x + LoggedSignals.t_x;

a_psi_x   = exp(1i * LoggedSignals.psi_x * ((1:EnvPars.N_x)-1))';
a_psi_y   = exp(1i * LoggedSignals.psi_y * ((1:EnvPars.N_y)-1))';
a_psi_x_y = kron(a_psi_y, a_psi_x);

% v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
%
% xi_cmd = mod(angle(v0), 2*pi);
%
% t_yx_cmd = EnvPars.F_tyx_real(xi_cmd) + ...
%            1i*EnvPars.F_tyx_imag(xi_cmd);
%
% t_yy_cmd = EnvPars.F_tyy_real(xi_cmd) + ...
%            1i*EnvPars.F_tyy_imag(xi_cmd);
%
% c_rot = t_yx_cmd*cosd(EnvPars.angle_rot_deg) + ...
%         t_yy_cmd*sind(EnvPars.angle_rot_deg);
%
% q_rot = c_rot ./ t_yx_cmd;
%
% r = sqrt(db2pow(EnvPars.SNR_dB)) * ...
%     EnvPars.G_CST * diag(v0') * ...
%     (q_rot' .* a_psi_x_y);

v0 = EnvPars.U_func(1:EnvPars.N, t_psi);

Upsilon   = diag(v0');
% G_Upsilon = EnvPars.G_CST * Upsilon;
G_Upsilon = EnvPars.G * Upsilon;

% Commanded atom phase
xi_cmd = mod(angle(v0), 2*pi);

% State-dependent Jones coefficients
t_yx_cmd = EnvPars.F_tyx_real(xi_cmd) + ...
    1i*EnvPars.F_tyx_imag(xi_cmd);

t_yy_cmd = EnvPars.F_tyy_real(xi_cmd) + ...
    1i*EnvPars.F_tyy_imag(xi_cmd);

% Force column-vector convention
t_yx_cmd = t_yx_cmd(:);
t_yy_cmd = t_yy_cmd(:);
a_psi_x_y = a_psi_x_y(:);

% Dual-polarized emitter
alpha_deg = EnvPars.angle_rot_deg;  % Physical emitter rotation
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

% ARGMAX-ALIGNED observation -- must match stepFunction_nav_CST_Aligned
Rc = reshape(r, [EnvPars.N_x, EnvPars.N_y]);
[~, mi] = max(abs(Rc(:)));
[mx, my] = ind2sub([EnvPars.N_x, EnvPars.N_y], mi);
Ral = circshift(Rc, [1-mx, 1-my]);
Ral = Ral * exp(-1i*angle(Ral(1,1)));
observation = [real(Ral(:)); imag(Ral(:))];

% kept for compatibility (unused by the current reward)
LoggedSignals.prev_peak = max(abs(r).^2) / LoggedSignals.global_max;
end
