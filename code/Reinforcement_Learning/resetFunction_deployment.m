function [observation, LoggedSignals] = resetFunction_deployment(EnvPars)

% Start at a random position in the (t_x, t_y) grid
LoggedSignals.t_x       = randi(EnvPars.T_x);
LoggedSignals.t_y       = randi(EnvPars.T_y);
LoggedSignals.stepCount = 0;

% Initial DOA from trajectory step 1
LoggedSignals.psi_x     = EnvPars.psi_x_traj(1);
LoggedSignals.psi_y     = EnvPars.psi_y_traj(1);

% Build initial observation
t_psi     = (LoggedSignals.t_y - 1) * EnvPars.T_x + LoggedSignals.t_x;
a_psi_x   = exp(1i * LoggedSignals.psi_x * ((1:EnvPars.N_x)-1))';
a_psi_y   = exp(1i * LoggedSignals.psi_y * ((1:EnvPars.N_y)-1))';
a_psi_x_y = kron(a_psi_y, a_psi_x);

v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

% ← must match training: [|r|², t_x, t_y] = 18D
observation = [abs(r).^2; LoggedSignals.t_x; LoggedSignals.t_y];
end

%[appendix]{"version":"1.0"}
%---
