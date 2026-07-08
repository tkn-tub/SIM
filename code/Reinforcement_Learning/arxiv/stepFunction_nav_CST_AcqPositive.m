function [observation, reward, IsDone, LoggedSignals] = ...
         stepFunction_nav_CST_AcqPositive(action, LoggedSignals, EnvPars)
% ACQUISITION reward, NON-NEGATIVE variant.
%
% WHY THIS EXISTS: the previous acquisition reward (gradient shaping
% - step_cost + bonus) produced predominantly NEGATIVE returns, but the
% SIM-2 critic ends in a diode readout max(Re(field),0) whose output
% floor is exactly 0 -- it cannot represent negative Q-values, and its
% gradient is zero below the floor. Result: Q pinned at 0.00, no
% learning (observed empirically). Every reward here is >= 0, so every
% Q-value the critic must represent is >= 0.
%
%   reward = w_dense * p            (p = normalised peak, in [0,1] --
%                                    increases on approach, per the
%                                    original design intent)
%          + peak_bonus * at_peak   (the objective, paid once)
%
% NO step cost. Path length is penalized by DISCOUNTING alone: value of
% acquiring from distance d ~ gamma^d * peak_bonus, strictly decreasing
% in d -> shortest path optimal.
%
% Hover exploit closed by arithmetic (gamma = 0.95):
%   hover forever next to peak: w_dense * p / (1-gamma) <= 0.1/0.05 = 2
%   step onto peak:             peak_bonus = 40
% Undiscounted episode sums are also monotone with success:
%   timeout hover <= 0.1 * MaxSteps ~= 12   vs   success ~= 41.

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

%% Termination -- computed BEFORE the reward (terminal bonus needs it)
at_peak = (t_x_new == EnvPars.best_tx_cal(LoggedSignals.pos_idx)) && ...
          (t_y_new == EnvPars.best_ty_cal(LoggedSignals.pos_idx));
IsDone  = at_peak || (LoggedSignals.stepCount >= EnvPars.MaxStepsPerEpisode);

%% Reward -- non-negative acquisition
normalised_peak = current_peak / LoggedSignals.global_max;
reward = EnvPars.w_dense * normalised_peak ...
       + EnvPars.peak_bonus * double(at_peak);

%% Next observation -- coherent field, [Re(r); Im(r)] stacked
observation = [real(r); imag(r)];

%% Log DOA estimate at current position
R = flipud(fliplr(reshape(abs(r), [EnvPars.N_x, EnvPars.N_y])))';
[~, linear_idx] = max(R, [], 'all');
n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;

t_psi_x_max = EnvPars.t_x(t_psi);
t_psi_y_max = EnvPars.t_y(t_psi);
LoggedSignals.psi_x_est = mod(2*pi * (n_psi_x_max + (t_psi_x_max) / EnvPars.T_x) / EnvPars.N_x, 2*pi);
LoggedSignals.psi_y_est = mod(2*pi * (n_psi_y_max + (t_psi_y_max) / EnvPars.T_y) / EnvPars.N_y, 2*pi);

LoggedSignals.error_sum    = 0.5*(abs(LoggedSignals.psi_x - LoggedSignals.psi_x_est) + ...
                                   abs(LoggedSignals.psi_y - LoggedSignals.psi_y_est));
LoggedSignals.at_peak      = at_peak;
end
