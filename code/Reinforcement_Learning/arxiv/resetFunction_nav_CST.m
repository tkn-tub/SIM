function [observation, LoggedSignals] = resetFunction_nav_CST_Reward_Mixed(EnvPars)

%% Pick random calibration position
pos_idx = randi(EnvPars.N_cal);

LoggedSignals.pos_idx    = pos_idx;
LoggedSignals.psi_x      = EnvPars.psi_x_cal(pos_idx);
LoggedSignals.psi_y      = EnvPars.psi_y_cal(pos_idx);
LoggedSignals.global_max = EnvPars.global_max_cal(pos_idx);
LoggedSignals.stepCount  = 0;

%% Random starting (t_x, t_y) — agent does not start at the peak
LoggedSignals.t_x = randi(EnvPars.T_x);
LoggedSignals.t_y = randi(EnvPars.T_y);

%% Initial observation
t_psi = (LoggedSignals.t_y - 1) * EnvPars.T_x + LoggedSignals.t_x;

a_psi_x   = exp(1i * LoggedSignals.psi_x * ((1:EnvPars.N_x)-1))';
a_psi_y   = exp(1i * LoggedSignals.psi_y * ((1:EnvPars.N_y)-1))';
a_psi_x_y = kron(a_psi_y, a_psi_x);

v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

% Observation: coherent field, [Re(r); Im(r)] stacked -- matches
% stepFunction_nav_CST.m exactly. Phase preserved; t_x/t_y stay in
% LoggedSignals for in-silico bookkeeping but are not observed.
observation = [real(r); imag(r)];
end

%[appendix]{"version":"1.0"}
%---
