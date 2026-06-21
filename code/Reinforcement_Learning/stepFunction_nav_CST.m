function [observation, reward, IsDone, LoggedSignals] = ...
         stepFunction_nav_CST(action, LoggedSignals, EnvPars)

LoggedSignals.stepCount = LoggedSignals.stepCount + 1;

%% Apply navigation move — wrap around grid edges
delta   = EnvPars.delta_moves(action, :);
t_x_new = mod(LoggedSignals.t_x - 1 + delta(1), EnvPars.T_x) + 1;
t_y_new = mod(LoggedSignals.t_y - 1 + delta(2), EnvPars.T_y) + 1;
LoggedSignals.t_x = t_x_new;
LoggedSignals.t_y = t_y_new;

%% Received signal at new position
t_psi = (t_y_new - 1) * EnvPars.T_x + t_x_new;

a_psi_x   = exp(1i * LoggedSignals.psi_x * ((1:EnvPars.N_x)-1))';
a_psi_y   = exp(1i * LoggedSignals.psi_y * ((1:EnvPars.N_y)-1))';
a_psi_x_y = kron(a_psi_y, a_psi_x);

v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;

power_vec    = abs(r).^2;
current_peak = max(power_vec);

%Reward D
%% Reward — normalised peak amplitude
% global_max known to environment from calibration, not exposed to agent
reward = current_peak / LoggedSignals.global_max;   % ∈ (0, 1]

%Reward E
%% Reward — thresholded normalised peak amplitude
% Matches the R_map definition used in value iteration exactly:
% R = max(0, (normalised_peak - threshold) / (1 - threshold))

normalised_peak  = current_peak / LoggedSignals.global_max;
reward           = max(0, (normalised_peak - EnvPars.threshold) / (1 - EnvPars.threshold));

%% Termination
at_peak = (t_x_new == EnvPars.best_tx_cal(LoggedSignals.pos_idx)) && ...
          (t_y_new == EnvPars.best_ty_cal(LoggedSignals.pos_idx));
IsDone  = at_peak || (LoggedSignals.stepCount >= EnvPars.MaxStepsPerEpisode);

%% Next observation -- coherent field, [Re(r); Im(r)] stacked (2N real
% values encoding SIM1's true complex output). Phase is preserved through
% to SIM2 now, not discarded here -- detection happens once, at the
% diode readout at the far end of the network. CONVENTION: first N rows
% are Re(r), next N rows are Im(r) -- must match
% realImagToComplexLayer.m's predict() exactly.
%
% t_x_new/t_y_new remain available via LoggedSignals (in-silico tracking,
% used above for reward/termination and below for DOA logging) but are
% not part of the observation vector.
observation = [real(r); imag(r)];

%% Log DOA estimate at current position
R_2D = reshape(abs(r), [EnvPars.N_x, EnvPars.N_y]);
[~, idx]                   = max(R_2D, [], 'all');
[n_y_max, n_x_max]         = ind2sub(size(R_2D), idx);
LoggedSignals.psi_x_est    = mod(2*pi*((n_x_max-1) + ...
    (EnvPars.t_x(t_psi)-1)/EnvPars.T_x)/EnvPars.N_x, 2*pi);
LoggedSignals.psi_y_est    = mod(2*pi*((n_y_max-1) + ...
    (EnvPars.t_y(t_psi)-1)/EnvPars.T_y)/EnvPars.N_y, 2*pi);
LoggedSignals.error_sum    = 0.5*(abs(LoggedSignals.psi_x - LoggedSignals.psi_x_est) + ...
                                   abs(LoggedSignals.psi_y - LoggedSignals.psi_y_est));
LoggedSignals.at_peak      = at_peak;
end

%[appendix]{"version":"1.0"}
%---
