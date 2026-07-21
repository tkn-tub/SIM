%% Step 1. CALIBRATION ───────────────────────────────────────────────────────EnvPars.psi_x_cal
fprintf('=== Calibration phase ===\n'); %[output:106d88ec]

% Grid: 1 position per square metre over x_max × y_max
% Margin keeps MU away from walls

n_x_cal  = floor(sqrt(EnvPars.N_cal * x_max / y_max));
n_y_cal  = floor(EnvPars.N_cal / n_x_cal);
x_cal    = linspace(EnvPars.MU_margin, x_max - EnvPars.MU_margin, n_x_cal);
y_cal    = linspace(EnvPars.MU_margin, y_max - EnvPars.MU_margin, n_y_cal);
[X_cal, Y_cal] = meshgrid(x_cal, y_cal);
X_cal    = X_cal(:);
Y_cal    = Y_cal(:);
N_cal    = numel(X_cal);
fprintf('Grid: %d x %d = %d calibration positions\n', n_x_cal, n_y_cal, N_cal); %[output:3da17f4a]

% Preallocate calibration tables
peak_map       = zeros(N_cal, EnvPars.T_x, EnvPars.T_y);
psi_x_cal      = zeros(N_cal, 1);
psi_y_cal      = zeros(N_cal, 1);
pos_cal        = zeros(N_cal, 3);
global_max_cal = zeros(N_cal, 1);
best_tx_cal    = zeros(N_cal, 1);
best_ty_cal    = zeros(N_cal, 1);
% One N-element channel vector per calibration position
h_cal = complex(zeros(EnvPars.N, N_cal));

for pos = 1:N_cal
    pos_MU_k       = [X_cal(pos); Y_cal(pos); EnvPars.h_MU];
    pos_cal(pos,:) = pos_MU_k';

    [psi_x_k, psi_y_k] = computePsiFromPos(pos_MU_k, EnvPars);
    psi_x_cal(pos) = psi_x_k;
    psi_y_cal(pos) = psi_y_k;

    % a_psi_x   = exp(1i * psi_x_k * ((1:EnvPars.N_x)-1))';
    % a_psi_y   = exp(1i * psi_y_k * ((1:EnvPars.N_y)-1))';
    % a_psi_x_y = kron(a_psi_y, a_psi_x);

    %% Select the channel model

    % Deterministic LoS steering vector
    a_psi_x = exp( ...
        -1i * psi_x_k * ...
        (0:EnvPars.N_x-1)).';

    a_psi_y = exp( ...
        -1i * psi_y_k * ...
        (0:EnvPars.N_y-1)).';

    h = kron(a_psi_y, a_psi_x);
            
    % switch lower(EnvPars.channelModel)
    % 
    %     case 'rician_los'
    % 
    %         % Deterministic LoS steering vector
    %         a_psi_x = exp( ...
    %             -1i * psi_x_k * ...
    %             (0:EnvPars.N_x-1)).';
    % 
    %         a_psi_y = exp( ...
    %             -1i * psi_y_k * ...
    %             (0:EnvPars.N_y-1)).';
    % 
    %         h = kron(a_psi_y, a_psi_x);
    % 
    %     case 'rician_los_nlos'
    % 
    %         % LoS + NLoS channel generated
    %         h = generateChannel(pos_MU_k, EnvPars);
    % 
    %     otherwise
    % 
    %         error( ...
    %             'Unknown channel model "%s". Use "LoS" or "LoS_NLoS".', ...
    %             EnvPars.channelModel);
    % end

    % Guarantee an N-by-1 column vector
    h = h(:);

    assert(numel(h) == EnvPars.N, ...
        ['The selected channel returned %d coefficients, ' ...
        'but EnvPars.N = %d.'], ...
        numel(h), EnvPars.N);

    % Normalize to unit average element power.
    % This is appropriate for the controlled-SNR evaluation.
    channelPower = mean(abs(h).^2);

    if channelPower <= eps
        error('The generated channel has approximately zero power.');
    end

    h = h / sqrt(channelPower);

    % Store the exact channel used at this calibration position
    h_cal(:,pos) = h;

    for t_psi = 1:EnvPars.T
        v0  = EnvPars.U_func(1:EnvPars.N, t_psi);
        % v0  = EnvPars.U_func_CST(1:EnvPars.N, t_psi);
        % r   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * a_psi_x_y;
        % r = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * h;
        r = EnvPars.G * diag(v0') * h;
        peak_map(pos, EnvPars.t_x(t_psi), EnvPars.t_y(t_psi)) = max(abs(r).^2);
    end

    % Global maximum and best action for this position
    slice                 = squeeze(peak_map(pos,:,:));
    [global_max_cal(pos), idx] = max(slice(:));
    [best_tx_cal(pos), best_ty_cal(pos)] = ind2sub([EnvPars.T_x, EnvPars.T_y], idx);
end

% Guard: reward normalization must match a direct field measurement
pos    = 1;
t_best = (best_ty_cal(pos)-1)*EnvPars.T_x + best_tx_cal(pos);
v0     = EnvPars.U_func_CST(1:EnvPars.N, t_best);
p_direct = max(abs(sqrt(db2pow(EnvPars.SNR_dB)) * ...
           EnvPars.G_CST * diag(v0') * h_cal(:,pos)).^2);
p_ref    = db2pow(EnvPars.SNR_dB) * global_max_cal(pos);
mismatch_dB = 10*log10(p_direct / p_ref);
%assert(abs(mismatch_dB) < 0.5, ...
    % 'Reward normalization mismatch: %.1f dB — stale calibration tables?', mismatch_dB);

fprintf('Calibration complete.\n\n'); %[output:42000b26]

% Store calibration in EnvPars — available to step/reset functions
EnvPars.N_cal          = N_cal;
EnvPars.peak_map       = peak_map;
EnvPars.psi_x_cal      = psi_x_cal;
EnvPars.psi_y_cal      = psi_y_cal;
EnvPars.pos_cal        = pos_cal;
EnvPars.global_max_cal = global_max_cal;
EnvPars.best_tx_cal    = best_tx_cal;
EnvPars.best_ty_cal    = best_ty_cal;
EnvPars.h_cal = h_cal;

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
%   data: {"layout":"onright","rightPanelPercent":36.3}
%---
%[output:106d88ec]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\n","truncated":false}}
%---
%[output:3da17f4a]
%   data: {"dataType":"text","outputData":{"text":"Grid: 10 x 10 = 100 calibration positions\n","truncated":false}}
%---
%[output:42000b26]
%   data: {"dataType":"text","outputData":{"text":"Calibration complete.\n\n","truncated":false}}
%---
