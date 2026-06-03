function [observation, reward, LoggedSignals] = ...
         stepFunction_deployment(action, LoggedSignals, EnvPars)

LoggedSignals.stepCount = LoggedSignals.stepCount + 1;

% Update DOA from trajectory
k = LoggedSignals.stepCount;
LoggedSignals.psi_x     = EnvPars.psi_x_traj(k);
LoggedSignals.psi_y     = EnvPars.psi_y_traj(k);
LoggedSignals.psi_x_traj = EnvPars.psi_x_traj(k);
LoggedSignals.psi_y_traj = EnvPars.psi_y_traj(k);

% Apply navigation move
delta   = EnvPars.delta_moves(action, :);
t_x_new = mod(LoggedSignals.t_x - 1 + delta(1), EnvPars.T_x) + 1;
t_y_new = mod(LoggedSignals.t_y - 1 + delta(2), EnvPars.T_y) + 1;
LoggedSignals.t_x = t_x_new;
LoggedSignals.t_y = t_y_new;

% Compute received signal
t_psi     = (t_y_new - 1) * EnvPars.T_x + t_x_new;
a_psi_x   = exp(1i * LoggedSignals.psi_x * ((1:EnvPars.N_x)-1))';
a_psi_y   = exp(1i * LoggedSignals.psi_y * ((1:EnvPars.N_y)-1))';
a_psi_x_y = kron(a_psi_y, a_psi_x);

v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

power_vec    = abs(r).^2;
current_peak = max(power_vec);

% Reward — use nearest calibration position as reference for global_max
[~, nearest_pos] = min(vecnorm(EnvPars.pos_cal(:,1:2) - ...
                   [LoggedSignals.t_x, LoggedSignals.t_y], 2, 2));
reward = current_peak / EnvPars.global_max_cal(nearest_pos);

% DOA estimate from DFT peak
R_2D = reshape(abs(r), [EnvPars.N_x, EnvPars.N_y]);
[~, idx]           = max(R_2D, [], 'all');
[n_y_max, n_x_max] = ind2sub(size(R_2D), idx);
LoggedSignals.psi_x_est = mod(2*pi*((n_x_max-1) + ...
    (EnvPars.t_x(t_psi)-1)/EnvPars.T_x)/EnvPars.N_x, 2*pi);
LoggedSignals.psi_y_est = mod(2*pi*((n_y_max-1) + ...
    (EnvPars.t_y(t_psi)-1)/EnvPars.T_y)/EnvPars.N_y, 2*pi);
LoggedSignals.error_sum = 0.5*(abs(LoggedSignals.psi_x - LoggedSignals.psi_x_est) + ...
                                abs(LoggedSignals.psi_y - LoggedSignals.psi_y_est));

% ← must match training: [|r|², t_x, t_y] = 18D
observation = [power_vec; t_x_new; t_y_new];
end

%[appendix]{"version":"1.0"}
%---
