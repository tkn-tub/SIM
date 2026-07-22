%% Step 1. CALIBRATION ───────────────────────────────────────────────────────EnvPars.psi_x_cal
fprintf('=== Calibration phase ===\n'); %[output:106d88ec]

% % Grid: 1 position per square metre over x_max × y_max
% % Margin keeps MU away from walls
% 
% n_x_cal  = floor(sqrt(EnvPars.N_cal * x_max / y_max));
% n_y_cal  = floor(EnvPars.N_cal / n_x_cal);
% x_cal    = linspace(EnvPars.MU_margin, x_max - EnvPars.MU_margin, n_x_cal);
% y_cal    = linspace(EnvPars.MU_margin, y_max - EnvPars.MU_margin, n_y_cal);
% [X_cal, Y_cal] = meshgrid(x_cal, y_cal);
% X_cal    = X_cal(:);
% Y_cal    = Y_cal(:);
% N_cal    = numel(X_cal);
% fprintf('Grid: %d x %d = %d calibration positions\n', n_x_cal, n_y_cal, N_cal); %[output:3da17f4a]

%% Generate calibration positions for one 120-degree sector

% Spatial separation between calibration points
calSpacing = EnvPars.calSpacing_m;

assert(calSpacing > 0, ...
    'EnvPars.calSpacing_m must be strictly positive.');

% Local D x D cell limits
x_min_cal = EnvPars.MU_margin;
x_max_cal = EnvPars.x_max - EnvPars.MU_margin;

y_min_cal = EnvPars.MU_margin;
y_max_cal = EnvPars.y_max - EnvPars.MU_margin;

assert(x_max_cal > x_min_cal && y_max_cal > y_min_cal, ...
    'The MU margin leaves no valid calibration area.');

% Use cell-centred calibration points.
%
% For D = 20 m and calSpacing = 1 m, this produces:
% x = 0.5, 1.5, ..., 19.5 m
% y = 0.5, 1.5, ..., 19.5 m
%
% Thus, there is approximately one calibration sample per square metre.
x_cal_full = ...
    (x_min_cal + calSpacing/2) : ...
    calSpacing : ...
    (x_max_cal - calSpacing/2);

y_cal_full = ...
    (y_min_cal + calSpacing/2) : ...
    calSpacing : ...
    (y_max_cal - calSpacing/2);

assert(~isempty(x_cal_full) && ~isempty(y_cal_full), ...
    'No calibration points were generated. Check the spacing and margin.');

% Complete D x D candidate grid
[X_cal_full, Y_cal_full] = meshgrid(x_cal_full, y_cal_full);

n_x_cal_full = numel(x_cal_full);
n_y_cal_full = numel(y_cal_full);
N_cal_full   = numel(X_cal_full);

%% Determine the selected sector

sectorBoresights_deg = EnvPars.sectorBoresights_deg;
selectedSector       = EnvPars.selectedSector;

assert(selectedSector >= 1 && ...
       selectedSector <= numel(sectorBoresights_deg), ...
    'EnvPars.selectedSector must index sectorBoresights_deg.');

sectorBoresight_deg = ...
    sectorBoresights_deg(selectedSector);

sectorBoresight = deg2rad(sectorBoresight_deg);
sectorHalfWidth = deg2rad(EnvPars.sectorHalfWidth_deg);

% Position relative to the SIM/gNB horizontal coordinates
delta_x = X_cal_full - EnvPars.pos_SIM(1);
delta_y = Y_cal_full - EnvPars.pos_SIM(2);

% MU azimuth relative to the SIM/gNB
phi_cal = atan2(delta_y, delta_x);

% Wrapped angular distance from the selected sector boresight
delta_phi = atan2( ...
    sin(phi_cal - sectorBoresight), ...
    cos(phi_cal - sectorBoresight));

% Retain only positions inside the selected 120-degree sector
sectorMask = abs(delta_phi) <= sectorHalfWidth + 10*eps;

X_cal = X_cal_full(sectorMask);
Y_cal = Y_cal_full(sectorMask);

% Convert explicitly to column vectors
X_cal = X_cal(:);
Y_cal = Y_cal(:);

% Actual number of calibration positions after sector filtering
N_cal = numel(X_cal);

assert(N_cal > 0, ...
    'The selected sector contains no calibration positions.');

fprintf( ...
    'Full local grid: %d x %d = %d candidate positions\n', ...
    n_x_cal_full, n_y_cal_full, N_cal_full);

fprintf( ...
    ['Selected sector %d: boresight = %.1f deg, ' ...
     'angular interval = [%.1f, %.1f] deg\n'], ...
    selectedSector, ...
    sectorBoresight_deg, ...
    sectorBoresight_deg - EnvPars.sectorHalfWidth_deg, ...
    sectorBoresight_deg + EnvPars.sectorHalfWidth_deg);

fprintf( ...
    'Calibration positions retained in sector: %d\n', ...
    N_cal);

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
