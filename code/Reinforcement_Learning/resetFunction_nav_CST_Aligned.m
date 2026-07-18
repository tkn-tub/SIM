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

%% Channel

LoggedSignals.h      = EnvPars.h_cal(:, pos_idx);
LoggedSignals.SNR_dB = EnvPars.SNR_dB;

%% Initial observation
t_psi = (LoggedSignals.t_y - 1) * EnvPars.T_x + LoggedSignals.t_x;

% a_psi_x   = exp(1i * LoggedSignals.psi_x * ((1:EnvPars.N_x)-1))';
% a_psi_y   = exp(1i * LoggedSignals.psi_y * ((1:EnvPars.N_y)-1))';
% a_psi_x_y = kron(a_psi_y, a_psi_x);
% 
% v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
% r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

% v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
% r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * LoggedSignals.h;

% v0 = EnvPars.U_func(1:EnvPars.N, t_psi);
v0 = EnvPars.U_func_CST(1:EnvPars.N, t_psi);

% r_signal = sqrt(db2pow(LoggedSignals.SNR_dB)) * ...
%            EnvPars.G * diag(v0') * LoggedSignals.h;

r_signal = sqrt(db2pow(LoggedSignals.SNR_dB)) * ...
           EnvPars.G_CST * diag(v0') * LoggedSignals.h;

u = (randn(EnvPars.N,1) + ...
     1i*randn(EnvPars.N,1)) / sqrt(2);

r = r_signal + u;

% ARGMAX-ALIGNED observation -- must match stepFunction_nav_CST_Aligned
Rc = reshape(r, [EnvPars.N_x, EnvPars.N_y]);
[~, mi] = max(abs(Rc(:)));
[mx, my] = ind2sub([EnvPars.N_x, EnvPars.N_y], mi);
Ral = circshift(Rc, [1-mx, 1-my]);
Ral = Ral * exp(-1i*angle(Ral(1,1)));
observation = [real(Ral(:)); imag(Ral(:))];

% kept for compatibility (unused by the current reward)
% LoggedSignals.prev_peak = max(abs(r).^2) / LoggedSignals.global_max;
snr_linear = db2pow(LoggedSignals.SNR_dB);

reference_peak = ...
    snr_linear * LoggedSignals.global_max;

LoggedSignals.prev_peak = ...
    max(abs(r).^2) / max(reference_peak, eps);

LoggedSignals.prev_peak = ...
    min(LoggedSignals.prev_peak, 1);
end
