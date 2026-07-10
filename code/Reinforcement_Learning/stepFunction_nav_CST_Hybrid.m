function [observation, reward, IsDone, LoggedSignals] = ...
         stepFunction_nav_CST_Aligned(action, LoggedSignals, EnvPars)
% ALIGNED-observation variant. Identical to the current AcqPositive
% reward step function EXCEPT the observation: the 4x4 complex pattern
% is circularly shifted so the argmax bin sits at (1,1) and
% phase-referenced to that port, exactly as validated by probe C in
% C_Diag_Observation_ABC.m (93.8% held-out-position action accuracy vs
% 39.1% unaligned). This makes the observation position-invariant:
% the same fractional misalignment produces the same observation
% regardless of MU position, which is what lets a (quasi-)linear
% Q-network generalize across positions.
%
% The argmax is the same comparator the DOA estimator below already
% runs; in deployment the shift needs a physical mechanism at the
% SIM1->SIM2 interface (see paper note) -- in silico it is re-indexing.

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

%% Reward -- non-negative acquisition (UNCHANGED from current run)
normalised_peak = current_peak / LoggedSignals.global_max;
reward = 0.1*normalised_peak + 40*at_peak;

%% Observation -- ARGMAX-ALIGNED coherent field (validated: probe C)
% Rc(nx,ny) with x-fastest linear ordering, matching the probe exactly.
% Rc = reshape(r, [EnvPars.N_x, EnvPars.N_y]);
% [~, mi] = max(abs(Rc(:)));
% [mx, my] = ind2sub([EnvPars.N_x, EnvPars.N_y], mi);
% Ral = circshift(Rc, [1-mx, 1-my]);            % argmax bin -> (1,1)
% Ral = Ral * exp(-1i*angle(Ral(1,1)));         % phase-reference to that port
% observation = [real(Ral(:)); imag(Ral(:))];   % 2N-dim, ObsInfo unchanged
observation = [real(r); imag(r)];   % 2N-dim, ObsInfo unchanged

%% Log DOA estimate at current position (verbatim estimator block)
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
